import pytest

from src.evaluate import score_run, wilson_interval
from src.extract import ExtractionResult, MockExtractor
from src.generate import build_records
from src.report import build_error_analysis, build_summary, evaluate_runs, md_table, rate_cell, write_reports


@pytest.fixture(scope="module")
def records():
    return [r for r in build_records(42) if r["split"] == "test"]


def mock_runs(records, rates=(0.5, 0.3, 0.1)):
    truths = {r["doc_id"]: r["truth"] for r in records}
    runs = {}
    for name, rate, seed in zip(("zero_shot", "few_shot", "rules"), rates, (1, 2, 3)):
        m = MockExtractor(truths, error_rate=rate, seed=seed)
        runs[name] = {d: m.extract(d) for d in truths}
    return runs


META = {"model": "mock", "mock": True, "split": "test", "seed": 42}


def test_md_table_escapes_pipes():
    t = md_table(["a", "b"], [["x|y", "line\nbreak"]])
    assert "x\\|y" in t and "line break" in t and t.count("\n") == 2


def test_rate_cell_matches_wilson():
    lo, hi = wilson_interval(36, 40)
    assert rate_cell(36, 40) == f"36/40 = 90.0% [{lo * 100:.1f}%–{hi * 100:.1f}%]"
    assert "n/a" in rate_cell(0, 0)


def test_reports_written_and_deterministic(records, tmp_path):
    runs = mock_runs(records)
    sp, ep = write_reports(tmp_path / "r", records, runs, META)
    s1, e1 = sp.read_text(encoding="utf-8"), ep.read_text(encoding="utf-8")
    sp2, ep2 = write_reports(tmp_path / "r2", records, runs, META)
    assert s1 == sp2.read_text(encoding="utf-8") and e1 == ep2.read_text(encoding="utf-8")
    for h in ("## 1. Strategy comparison", "## 2. Per-field accuracy", "### Layout template", "### Noise level",
              "### Hard cases", "## 4. Validation layer", "## 5. Usage", "McNemar"):
        assert h in s1
    for h in ("## Strategy: zero_shot", "### Taxonomy", "### Examples", "## What the validation layer missed"):
        assert h in e1
    assert "MOCK RUN" in s1 and "MOCK RUN" in e1
    assert "| credit_note |" in s1 and "| invoice |" in s1  # document-type breakdown has real group labels
    assert all(name in s1 for name in runs)
    assert "nan" not in s1.lower().replace("n/a", "") and "nan" not in e1.lower()


def test_summary_numbers_come_from_scores(records):
    runs = mock_runs(records)
    ev = evaluate_runs(records, runs)
    s = build_summary(records, ev, META)
    for name, results in runs.items():
        df = score_run(results, records)
        assert rate_cell(int(df["all_correct"].sum()), 40) in s
    assert len(ev["rules"]["errors"]) <= len(ev["zero_shot"]["errors"])


def test_real_run_has_no_mock_banner(records):
    ev = evaluate_runs(records, mock_runs(records))
    s = build_summary(records, ev, {"model": "some-model", "split": "test", "seed": 42})
    assert "MOCK RUN" not in s and "`some-model`" in s


def test_perfect_run_reports_no_errors_and_no_flags(records):
    runs = mock_runs(records, rates=(0.0, 0.0, 0.0))
    ev = evaluate_runs(records, runs)
    s, e = build_summary(records, ev, META), build_error_analysis(records, ev, META)
    assert "No errors." in e and "No document was flagged." in s
    assert "40/40 = 100.0%" in s and "None: every document" in e


def test_failed_extractions_reported(records):
    runs = mock_runs(records)
    runs["rules"]["test_03"] = ExtractionResult("test_03", None, False, "boom")
    del runs["rules"]["test_04"]
    ev = evaluate_runs(records, runs)
    e = build_error_analysis(records, ev, META)
    assert "extraction_failure" in e
    s = build_summary(records, ev, META)
    assert "| rules |" in s


def test_single_strategy_has_no_paired_section(records):
    runs = {"rules": mock_runs(records)["rules"]}
    s = build_summary(records, evaluate_runs(records, runs), META)
    assert "McNemar" not in s and "rules" in s


def test_evaluate_runs_rejects_dev_records():
    dev = [r for r in build_records(42) if r["split"] == "dev"]
    with pytest.raises(ValueError):
        evaluate_runs(dev, {"x": {}})
