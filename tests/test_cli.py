import json
import anthropic
import pytest
import src.extract as ex
from types import SimpleNamespace
from src.cli import changed_prompts, main, read_log
from src.cost import cost_usd, estimate_strategy, format_estimate, lookup_price
from src.extract import LLMExtractor, extract_all, format_user, load_few_shot_examples, pdf_to_text
from src.generate import generate_dataset, load_split


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli_data")
    generate_dataset(d, seed=42)
    return d


class TruthClient:
    """Fake Anthropic client that answers each document with its ground truth and counts calls."""

    def __init__(self, data_dir):
        self.map = {}
        for split in ("dev", "test"):
            for r in load_split(data_dir, split):
                self.map[format_user(pdf_to_text(r["pdf_path"]))] = json.dumps(r["truth"])
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls += 1
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text",
                                                                               text=self.map[kw["messages"][-1]["content"]])],
                               usage=SimpleNamespace(input_tokens=200, output_tokens=100))


@pytest.fixture(scope="module")
def truth_client_factory(data_dir):
    base = TruthClient(data_dir)

    def make():
        c = TruthClient.__new__(TruthClient)
        c.map, c.calls = base.map, 0
        c.messages = SimpleNamespace(create=c._create)
        return c
    return make


@pytest.fixture()
def env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "test-model-x")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def dirs(tmp_path, cache=True):
    """Common path flags. Only `run` takes --cache-dir."""
    out = ["--out-dir", str(tmp_path / "out"), "--results-dir", str(tmp_path / "res")]
    return out + (["--cache-dir", str(tmp_path / "cache")] if cache else [])


def no_confirm(_):
    raise AssertionError("confirmation should not be requested")


# ------------------------------------------------------------------ offline end-to-end
def test_offline_end_to_end_with_mock(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the API client must not be constructed in mock mode")
    monkeypatch.setattr(anthropic, "Anthropic", boom)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    data = str(tmp_path / "data")
    base = ["--data-dir", data] + dirs(tmp_path)
    rep = ["--data-dir", data] + dirs(tmp_path, cache=False)
    assert main(["generate", "--data-dir", data]) == 0
    assert len(list((tmp_path / "data" / "test").glob("*.pdf"))) == 40
    for split in ("dev", "test"):
        assert main(["run", "--mock", "--split", split, "--strategy", "all", "--yes", *base]) == 0
    assert main(["report", "--mock", "--split", "test", *rep]) == 0
    summary = (tmp_path / "out" / "mock_report" / "summary.md").read_text(encoding="utf-8")
    errors = (tmp_path / "out" / "mock_report" / "error_analysis.md").read_text(encoding="utf-8")
    assert "MOCK RUN" in summary and "MOCK RUN" in errors
    for s in ("zero_shot", "few_shot", "rules"):
        assert s in summary
    assert "| Strategy | All fields correct |" in summary and "(40 documents)" in summary
    assert not (tmp_path / "res" / "summary.md").exists()      # mock never writes into results/
    assert not (tmp_path / "res" / "run_log.json").exists()
    assert main(["report", "--mock", "--split", "dev", *rep]) == 0
    assert "DEV SPLIT" in (tmp_path / "out" / "mock_report" / "summary.md").read_text(encoding="utf-8")


# ------------------------------------------------------------------ live path with a fake client
def test_dry_run_decline_accept_and_cache(tmp_path, data_dir, truth_client_factory, env, capsys):
    c = truth_client_factory()
    base = ["--data-dir", str(data_dir), "--split", "dev", "--strategy", "rules", *dirs(tmp_path)]
    assert main(["run", "--dry-run", *base], client=c, confirm=no_confirm) == 0
    out = capsys.readouterr().out
    assert "Estimated cost" in out and "unknown" in out and "Dry run" in out and c.calls == 0
    assert main(["run", "--dry-run", "--price-in", "2", "--price-out", "10", *base], client=c) == 0
    assert "$" in capsys.readouterr().out
    # decline
    assert main(["run", *base], client=c, confirm=lambda prompt: "n") == 1
    assert c.calls == 0 and not (tmp_path / "out").exists()
    # accept: asks exactly once, mentions the call count
    prompts = []
    assert main(["run", *base], client=c, confirm=lambda p: prompts.append(p) or "y") == 0
    assert len(prompts) == 1 and "20 live API calls" in prompts[0] and "TEST" not in prompts[0]
    assert c.calls == 20 and (tmp_path / "out" / "test-model-x" / "dev" / "rules.jsonl").exists()
    # rerun: all cached, no prompt, no calls
    assert main(["run", *base], client=c, confirm=no_confirm) == 0
    assert c.calls == 20 and "already cached" in capsys.readouterr().out


def test_dev_report_is_labelled_not_held_out(tmp_path, data_dir, truth_client_factory, env):
    c = truth_client_factory()
    base = ["--data-dir", str(data_dir), *dirs(tmp_path)]
    assert main(["run", "--split", "dev", "--strategy", "zero_shot", "--yes", *base], client=c) == 0
    assert main(["report", "--split", "dev", "--data-dir", str(data_dir), *dirs(tmp_path, cache=False)]) == 0
    s = (tmp_path / "out" / "dev_report" / "summary.md").read_text(encoding="utf-8")
    assert "DEV SPLIT" in s and "MOCK" not in s and "20/20" in s
    assert not (tmp_path / "res" / "summary.md").exists()


def test_test_split_guard_log_and_report_notes(tmp_path, data_dir, truth_client_factory, env, monkeypatch, capsys):
    c = truth_client_factory()
    base = ["--data-dir", str(data_dir), "--split", "test", "--strategy", "rules", *dirs(tmp_path)]
    prompts = []
    assert main(["run", *base], client=c, confirm=lambda p: prompts.append(p) or "y") == 0
    assert "TEST split" in prompts[0] and c.calls == 40
    log = read_log(tmp_path / "res")
    assert len(log) == 1 and log[0]["strategy"] == "rules" and log[0]["live_calls"] == 40
    assert main(["report", "--split", "test", "--data-dir", str(data_dir), *dirs(tmp_path, cache=False)]) == 0
    summary = (tmp_path / "res" / "summary.md").read_text(encoding="utf-8")
    assert "one fixed prompt per strategy" in summary and "40/40" in summary and "WARNING" not in summary

    # same prompt again: allowed (served from cache), no new calls
    assert main(["run", *base], client=c, confirm=no_confirm) == 0 and c.calls == 40
    # change the prompt: refused
    monkeypatch.setattr(ex, "RULES_SYSTEM", ex.RULES_SYSTEM + " extra rule")
    assert main(["run", *base, "--yes"], client=c) == 3
    assert "REFUSED" in capsys.readouterr().out and c.calls == 40
    # explicit override: runs, and the report warns
    assert main(["run", *base, "--yes", "--allow-test-rerun"], client=c) == 0 and c.calls == 80
    assert changed_prompts(read_log(tmp_path / "res"), "test-model-x") == ["rules"]
    assert main(["report", "--split", "test", "--data-dir", str(data_dir), *dirs(tmp_path, cache=False)]) == 0
    assert "WARNING: the prompt for rules changed" in (tmp_path / "res" / "summary.md").read_text(encoding="utf-8")


def test_config_errors_exit_2(tmp_path, data_dir, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    base = ["--data-dir", str(data_dir), *dirs(tmp_path)]
    assert main(["run", "--split", "dev", *base]) == 2
    assert "No model given" in capsys.readouterr().err
    assert main(["run", "--split", "dev", "--model", "m", *base]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert main(["run", "--split", "dev", "--mock", "--data-dir", str(tmp_path / "nope"), *dirs(tmp_path)]) == 2
    assert main(["report", "--mock", "--data-dir", str(data_dir), *dirs(tmp_path, cache=False)]) == 2  # nothing run yet
    assert "No saved results" in capsys.readouterr().err


# ------------------------------------------------------------------ description-audit (post-hoc, no API calls)
def test_description_audit_reads_cache_only(tmp_path, data_dir, truth_client_factory, env, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("description-audit must not construct a live API client")
    monkeypatch.setattr(anthropic, "Anthropic", boom)
    c = truth_client_factory()
    base = ["--data-dir", str(data_dir), "--split", "test", *dirs(tmp_path)]
    assert main(["run", *base, "--strategy", "zero_shot", "--yes"], client=c) == 0
    assert c.calls == 40
    rep = ["--data-dir", str(data_dir), *dirs(tmp_path, cache=False)]
    assert main(["description-audit", "--split", "test", *rep], client=c) == 0
    assert c.calls == 40  # unchanged: the audit made no calls
    p = tmp_path / "res" / "description_audit.md"
    assert p.exists() and "post-hoc" in p.read_text(encoding="utf-8").lower()
    assert "zero_shot" in p.read_text(encoding="utf-8")


def test_description_audit_requires_saved_results(tmp_path, data_dir, env, capsys):
    rep = ["--data-dir", str(data_dir), *dirs(tmp_path, cache=False)]
    assert main(["description-audit", "--split", "test", *rep]) == 2
    assert "No saved results" in capsys.readouterr().err


# ------------------------------------------------------------------ stress-validator (post-hoc, no API calls)
def test_stress_validator_writes_report_without_api_or_cache(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("stress-validator must not construct a live API client")
    monkeypatch.setattr(anthropic, "Anthropic", boom)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    assert main(["stress-validator", *dirs(tmp_path, cache=False)]) == 0
    p = tmp_path / "res" / "validator_stress_test.md"
    assert p.exists()
    txt = p.read_text(encoding="utf-8")
    assert "post-hoc" in txt.lower() and "coordinated_price" in txt and "0 of 60" in txt


def test_model_flag_overrides_env(tmp_path, data_dir, truth_client_factory, env):
    c = truth_client_factory()
    base = ["--data-dir", str(data_dir), "--split", "dev", "--strategy", "rules", "--yes", *dirs(tmp_path)]
    assert main(["run", "--model", "flag-model", *base], client=c) == 0
    assert (tmp_path / "out" / "flag-model" / "dev" / "rules.jsonl").exists()
    assert not (tmp_path / "out" / "test-model-x").exists()


# ------------------------------------------------------------------ cost estimator
def test_lookup_price():
    assert lookup_price("claude-opus-5-5") == (4.0, 20.0)
    assert lookup_price("claude-sonnet-5-5-preview") == (2.0, 10.0)   # longest prefix wins
    assert lookup_price("some-other-model") is None
    assert lookup_price("some-other-model", 1.0, 3.0) == (1.0, 3.0)
    assert lookup_price("claude-opus-5-5", 9.0, 9.0) == (9.0, 9.0)    # override beats the table


def test_cost_arithmetic_and_formatting():
    est = {"strategy": "rules", "n_docs": 40, "n_to_call": 40, "input_tokens": 1_000_000, "output_tokens": 500_000}
    assert cost_usd(est, (2.0, 10.0)) == pytest.approx(2.0 + 5.0)
    assert cost_usd(est, None) is None
    txt = format_estimate([est], "m", (2.0, 10.0), 700)
    assert "$7.0000" in txt and "thinking" in txt and "lower bound" in txt
    assert "unknown" in format_estimate([est], "m", None, 700)


def test_estimate_excludes_cached_and_scales_with_strategy(tmp_path, data_dir, truth_client_factory):
    docs = [{"doc_id": r["doc_id"], "text": pdf_to_text(r["pdf_path"])} for r in load_split(data_dir, "dev")]
    shots = load_few_shot_examples(data_dir)
    zs = LLMExtractor("m", "zero_shot", client=truth_client_factory(), cache_dir=tmp_path / "c")
    fs = LLMExtractor("m", "few_shot", client=truth_client_factory(), cache_dir=tmp_path / "c", examples=shots)
    e_zs, e_fs = estimate_strategy(zs, docs), estimate_strategy(fs, docs)
    assert e_zs["n_to_call"] == 20 and e_fs["input_tokens"] > e_zs["input_tokens"] > 0
    assert e_zs["output_tokens"] == 20 * 700
    extract_all(zs, docs[:5], progress=False)
    after = estimate_strategy(zs, docs)
    assert after["n_to_call"] == 15 and after["input_tokens"] < e_zs["input_tokens"]
    extract_all(zs, docs, progress=False)
    assert estimate_strategy(zs, docs)["n_to_call"] == 0
