import pytest

from src.generate import build_records
from src.stress_validator import (CORRUPTIONS, build_stress_report, run_clean_check, run_trials,
                                  write_stress_report)


@pytest.fixture(scope="module")
def records():
    return build_records(42)


def test_corruption_names_unique_and_declared_before_running():
    names = [c.name for c in CORRUPTIONS]
    assert len(names) == len(set(names))
    for c in CORRUPTIONS:
        assert isinstance(c.expected_detectable, bool) and c.reason


def test_every_applicable_corruption_actually_changes_the_record(records):
    """A trial is only valid if the corruption changed something; verify directly against apply()."""
    for c in CORRUPTIONS:
        for r in records:
            truth = r["truth"]
            if not c.applicable(truth):
                continue
            import random
            rng = random.Random(f"0-{c.name}-{r['doc_id']}")
            corrupted = c.apply(truth, rng)
            assert corrupted != truth, f"{c.name} on {r['doc_id']} did not change the record"


def test_run_trials_deterministic(records):
    a, b = run_trials(records, seed=0), run_trials(records, seed=0)
    assert a == b
    c = run_trials(records, seed=1)
    assert a != c  # different seed perturbs which digit/line/etc. is touched, not just outcomes


def test_run_trials_only_counts_applicable_and_changed_records(records):
    rows = run_trials(records, seed=0)
    by_name = {}
    for r in rows:
        by_name.setdefault(r["corruption"], 0)
        by_name[r["corruption"]] += 1
    # due_date corruptions need a due_date; not every one of the 60 has one
    n_with_due = sum(1 for r in records if r["truth"]["due_date"] is not None)
    assert by_name["due_date_earlier"] == n_with_due
    assert by_name["due_date_later"] == n_with_due
    n_with_rate = sum(1 for r in records if r["truth"]["tax_rate"] is not None)
    assert by_name["tax_rate_stale"] == n_with_rate
    assert by_name["vendor_name_char"] == 60  # applicable to every record


def test_clean_ground_truth_is_never_flagged(records):
    """The validator and generator must agree (this mirrors test_validate.py's own check, from the
    stress-test harness's point of view)."""
    assert run_clean_check(records) == []


def test_expected_detectable_matches_observed_for_every_corruption(records):
    """The core claim of Stage B: each corruption type's hand-reasoned expectation matches what the
    EXISTING validator actually does, on the real 60-record dataset."""
    rows = run_trials(records, seed=0)
    by_type = {}
    for r in rows:
        by_type.setdefault(r["corruption"], []).append(r)
    mismatches = []
    for c in CORRUPTIONS:
        trials = by_type.get(c.name, [])
        if not trials:
            continue
        detected = sum(t["detected"] for t in trials)
        if c.expected_detectable and detected != len(trials):
            mismatches.append(c.name)
        if not c.expected_detectable and detected != 0:
            mismatches.append(c.name)
    assert mismatches == [], f"expected_detectable mismatch for: {mismatches}"


def test_coordinated_price_is_the_documented_blind_spot(records):
    """coordinated_price is the one corruption explicitly designed to be invisible to every arithmetic
    check (unit_price changes but amount/subtotal/tax/total are recomputed consistently)."""
    rows = [r for r in run_trials(records, seed=0) if r["corruption"] == "coordinated_price"]
    assert len(rows) > 0 and all(not r["detected"] for r in rows)


def test_specific_checks_fire_for_the_corruptions_that_target_them(records):
    rows = run_trials(records, seed=0)
    by_type = {}
    for r in rows:
        by_type.setdefault(r["corruption"], []).append(r)
    assert all("due_before_invoice" in r["reason_codes"] for r in by_type["due_date_earlier"])
    assert all("invalid_currency" in r["reason_codes"] for r in by_type["currency_invalid"])
    assert all("line_amount_mismatch" in r["reason_codes"] for r in by_type["unit_price_only"])
    assert all("tax_rate_mismatch" in r["reason_codes"] for r in by_type["tax_rate_stale"])
    assert all("line_items_sum_mismatch" in r["reason_codes"] for r in by_type["line_dropped"])


def test_build_stress_report_lists_mismatches_first_and_separately():
    rows = [{"corruption": "x", "doc_id": "d1", "expected_detectable": True, "reason": "r",
             "detected": False, "reason_codes": []},
            {"corruption": "x", "doc_id": "d2", "expected_detectable": True, "reason": "r",
             "detected": True, "reason_codes": ["c"]}]
    txt = build_stress_report(rows, clean_rows=[], seed=0)
    assert "## Mismatches" in txt
    mismatch_section = txt.split("## Mismatches")[1].split("## Detection rate")[0]
    assert "x" in mismatch_section
    assert "post-hoc" in txt.lower() and "not a sample" in txt.lower()


def test_build_stress_report_reports_false_positives():
    txt = build_stress_report([], clean_rows=[{"doc_id": "dev_00", "reason_codes": ["total_mismatch"]}], seed=0)
    assert "1 of 60" in txt and "dev_00" in txt and "total_mismatch" in txt


def test_write_stress_report_file(tmp_path, records):
    p = write_stress_report(tmp_path / "out" / "stress.md", records, seed=0)
    assert p.exists()
    txt = p.read_text(encoding="utf-8")
    assert "coordinated_price" in txt and "vendor_name_char" in txt
    assert "0 of 60" in txt  # no false positives
