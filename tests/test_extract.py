import json
from types import SimpleNamespace

import pytest

import src.extract as ex
from src.extract import (ConfigError, ExtractionResult, LLMExtractor, MockExtractor, extract_all,
                         load_few_shot_examples, load_results, pdf_to_text, resolve_model, save_results,
                         select_few_shot)
from src.generate import build_records, generate_dataset, load_split
from src.schema import ALL_FIELDS


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    generate_dataset(d, seed=42)
    return d


@pytest.fixture(scope="module")
def records():
    return build_records(42)


@pytest.fixture(scope="module")
def truth(records):
    return records[0]["truth"]


def resp(text, stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="thinking", text=""),
                                                      SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=100, output_tokens=50))


class FakeClient:
    """Scripted stand-in for anthropic.Anthropic: each script item is a response or an exception to raise."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeTransient(Exception):
    pass


def make(tmp_path, script, strategy="zero_shot", model="m1", **kw):
    c = FakeClient(script)
    return LLMExtractor(model, strategy, client=c, cache_dir=tmp_path / "cache", sleep=lambda s: None, **kw), c


# ------------------------------------------------------------------ config
def test_resolve_model_precedence(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "env-model")
    assert resolve_model() == "env-model"
    assert resolve_model("cli-model") == "cli-model"
    monkeypatch.delenv("ANTHROPIC_MODEL")
    with pytest.raises(ConfigError):
        resolve_model()


def test_missing_api_key_raises_and_nothing_hardcoded(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        LLMExtractor("m", "zero_shot", cache_dir=tmp_path).client
    src_text = open(ex.__file__, encoding="utf-8").read()
    assert "sk-ant" not in src_text and "claude-" not in src_text


# ------------------------------------------------------------------ prompts / few-shot
def test_prompt_strategies_differ(tmp_path, truth):
    zs, _ = make(tmp_path, [])
    ru, _ = make(tmp_path, [], strategy="rules")
    fs, _ = make(tmp_path, [], strategy="few_shot",
                 examples=[{"doc_id": "dev_00", "text": "doc text", "truth": truth}])
    assert "NEGATIVE" not in zs.system and "NEGATIVE" in ru.system and "1.234,56" in ru.system
    assert "tax-inclusive" in ru.system
    assert len(zs.build_messages("x")) == 1 and len(fs.build_messages("x")) == 3
    assert json.loads(fs.build_messages("x")[1]["content"]) == truth
    assert len({zs.fingerprint, ru.fingerprint, fs.fingerprint}) == 3


def test_strategy_argument_validation(tmp_path, truth):
    with pytest.raises(ValueError):
        LLMExtractor("m", "bogus", cache_dir=tmp_path)
    with pytest.raises(ValueError):
        LLMExtractor("m", "few_shot", cache_dir=tmp_path)
    with pytest.raises(ValueError):
        LLMExtractor("m", "rules", cache_dir=tmp_path, examples=[{"doc_id": "d", "text": "t", "truth": truth}])


def test_few_shot_examples_only_from_dev(data_dir, records):
    shots = load_few_shot_examples(data_dir, k=3)
    assert len(shots) == 3 and all(s["doc_id"].startswith("dev_") for s in shots)
    assert [s["doc_id"] for s in shots] == [s["doc_id"] for s in load_few_shot_examples(data_dir, k=3)]
    test_vendors = {r["truth"]["vendor_name"] for r in records if r["split"] == "test"}
    blob = json.dumps(shots)
    assert not any(v in blob for v in test_vendors)
    with pytest.raises(ValueError):
        select_few_shot(load_split(data_dir, "test"))
    assert all(not r["meta"]["hard"] for r in load_split(data_dir, "dev") if r["doc_id"] in {s["doc_id"] for s in shots})


# ------------------------------------------------------------------ API behaviour
def test_successful_call_request_shape_and_cache(tmp_path, truth):
    e, c = make(tmp_path, [resp(json.dumps(truth))])
    r = e.extract("d1", "some text")
    assert r.ok and r.extraction == truth and not r.cached and r.attempts == 1
    assert r.usage == {"input_tokens": 100, "output_tokens": 50} and r.temperature_applied is True
    kw = c.calls[0]
    assert kw["model"] == "m1" and kw["extra_body"] == {"temperature": 0} and "temperature" not in kw
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert set(kw["output_config"]["format"]["schema"]["required"]) == set(ALL_FIELDS)
    assert "some text" in kw["messages"][-1]["content"]
    # second extractor, same cache dir: zero API calls
    e2, c2 = make(tmp_path, [])
    r2 = e2.extract("d1", "some text")
    assert r2.cached and r2.extraction == truth and c2.calls == []


def test_cache_key_depends_on_model_strategy_document_and_prompt(tmp_path, truth):
    base, _ = make(tmp_path, [])
    keys = {base.cache_key("t"),
            make(tmp_path, [], model="m2")[0].cache_key("t"),
            make(tmp_path, [], strategy="rules")[0].cache_key("t"),
            base.cache_key("t2")}
    assert len(keys) == 4
    assert base.cache_key("t") == make(tmp_path, [])[0].cache_key("t")


def test_prompt_edit_invalidates_cache(tmp_path, truth, monkeypatch):
    e, _ = make(tmp_path, [resp(json.dumps(truth))], strategy="rules")
    e.extract("d", "t")
    monkeypatch.setattr(ex, "RULES_SYSTEM", ex.RULES_SYSTEM + " extra rule")
    e2, c2 = make(tmp_path, [resp(json.dumps(truth))], strategy="rules")
    assert not e2.extract("d", "t").cached and len(c2.calls) == 1


def test_transient_errors_retry_with_growing_backoff(tmp_path, truth, monkeypatch):
    monkeypatch.setattr(ex, "RETRYABLE", (FakeTransient,))
    sleeps = []
    c = FakeClient([FakeTransient("429"), FakeTransient("529"), resp(json.dumps(truth))])
    e = LLMExtractor("m", "zero_shot", client=c, cache_dir=tmp_path / "c", sleep=sleeps.append)
    r = e.extract("d", "t")
    assert r.ok and r.attempts == 3 and len(sleeps) == 2
    assert 1.0 <= sleeps[0] <= 3.0 and 2.0 <= sleeps[1] <= 6.0  # 2^attempt * jitter in [0.5, 1.5)


def test_gives_up_after_max_attempts_and_does_not_cache_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "RETRYABLE", (FakeTransient,))
    e, c = make(tmp_path, [FakeTransient("x")] * 3, max_attempts=3)
    r = e.extract("d", "t")
    assert not r.ok and r.extraction is None and "gave up after 3" in r.error and len(c.calls) == 3
    assert list((tmp_path / "cache").glob("*.json")) == []


def test_malformed_outputs_are_retried(tmp_path, truth):
    bad_shape = json.dumps({k: v for k, v in truth.items() if k != "total"})
    e, c = make(tmp_path, [resp("not json {"), resp(bad_shape), resp(json.dumps(truth))])
    r = e.extract("d", "t")
    assert r.ok and r.attempts == 3 and len(c.calls) == 3


def test_max_tokens_truncation_doubles_budget(tmp_path, truth):
    e, c = make(tmp_path, [resp("{", stop="max_tokens"), resp(json.dumps(truth))], max_tokens=1000)
    assert e.extract("d", "t").ok
    assert [k["max_tokens"] for k in c.calls] == [1000, 2000]


def test_refusal_is_not_retried(tmp_path):
    e, c = make(tmp_path, [resp("", stop="refusal")])
    r = e.extract("d", "t")
    assert not r.ok and r.stop_reason == "refusal" and len(c.calls) == 1


def test_extra_fields_dropped_and_null_fields_allowed(tmp_path, truth):
    noisy = {**truth, "due_date": None, "tax_rate": None, "surprise": 1}
    e, _ = make(tmp_path, [resp(json.dumps(noisy))])
    r = e.extract("d", "t")
    assert r.ok and "surprise" not in r.extraction and r.extraction["due_date"] is None


def test_temperature_rejected_falls_back_and_is_recorded(tmp_path, truth):
    httpx2 = pytest.importorskip("httpx2")
    import anthropic
    err = anthropic.BadRequestError("`temperature` is not supported for this model",
                                    response=httpx2.Response(400, request=httpx2.Request("POST", "http://x")),
                                    body=None)
    e, c = make(tmp_path, [err, resp(json.dumps(truth))])
    r = e.extract("d", "t")
    assert r.ok and r.temperature_applied is False and r.attempts == 1
    assert "extra_body" in c.calls[0] and "extra_body" not in c.calls[1]


def test_other_bad_requests_propagate(tmp_path):
    httpx2 = pytest.importorskip("httpx2")
    import anthropic
    err = anthropic.BadRequestError("model: not found",
                                    response=httpx2.Response(400, request=httpx2.Request("POST", "http://x")),
                                    body=None)
    e, _ = make(tmp_path, [err])
    with pytest.raises(anthropic.BadRequestError):
        e.extract("d", "t")


# ------------------------------------------------------------------ mock, text, io
def test_mock_extractor_deterministic_and_error_rate(records):
    truths = {r["doc_id"]: r["truth"] for r in records}
    clean = MockExtractor(truths, error_rate=0.0)
    assert all(clean.extract(d).extraction == t for d, t in truths.items())
    noisy = MockExtractor(truths, error_rate=1.0, seed=3)
    again = MockExtractor(truths, error_rate=1.0, seed=3)
    outs = {d: noisy.extract(d).extraction for d in truths}
    assert outs == {d: again.extract(d).extraction for d in truths}
    assert sum(outs[d] != truths[d] for d in truths) >= 40
    assert truths["dev_00"] == records[0]["truth"]  # mock must not mutate the originals


def test_pdf_to_text_and_multipage_marker(data_dir):
    recs = {r["doc_id"]: r for r in load_split(data_dir, "dev")}
    t = pdf_to_text(recs["dev_00"]["pdf_path"])
    assert recs["dev_00"]["truth"]["invoice_number"] in t and "page 2" not in t
    multi = next(r for r in recs.values() if r["meta"]["hard_reason"] == "multi_page")
    assert "--- page 2 ---" in pdf_to_text(multi["pdf_path"])
    assert "    " not in t  # long space runs collapsed


def test_extract_all_and_results_roundtrip(tmp_path, records):
    truths = {r["doc_id"]: r["truth"] for r in records}
    docs = [{"doc_id": "dev_00", "text": ""}, {"doc_id": "dev_01", "text": ""}]
    res = extract_all(MockExtractor(truths, 0.0), docs, progress=False)
    p = tmp_path / "out" / "r.jsonl"
    save_results(res, p)
    back = load_results(p)
    assert set(back) == {"dev_00", "dev_01"} and isinstance(back["dev_00"], ExtractionResult)
    assert back["dev_00"].extraction == truths["dev_00"]
