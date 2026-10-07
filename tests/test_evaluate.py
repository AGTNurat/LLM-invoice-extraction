import copy
import math
import pandas as pd
import pytest

from src.evaluate import (accuracy_table, breakdown, field_correct, line_item_correct, mcnemar_exact,
                          paired_comparison, score_document, score_run, strategy_field_table,
                          strategy_overview, wilson_interval)
from src.extract import ExtractionResult, MockExtractor
from src.generate import build_records
from src.schema import ALL_FIELDS


@pytest.fixture(scope="module")
def records():
    return build_records(42)


@pytest.fixture(scope="module")
def test_records(records):
    return [r for r in records if r["split"] == "test"]


@pytest.fixture()
def truth(records):
    return copy.deepcopy(next(r["truth"] for r in records if len(r["truth"]["line_items"]) >= 3))


#  field comparison
@pytest.mark.parametrize("field,pred,truth,expected", [
    ("vendor_name", "  ACME   corp. ", "Acme Corp", True),
    ("vendor_name", "Acme Corporation", "Acme Corp", False),
    ("invoice_number", "inv- 001", "INV-001", True),
    ("invoice_number", "INV-002", "INV-001", False),
    ("invoice_date", "2024-03-15", "2024-03-15", True),
    ("invoice_date", "15 Mar 2024", "2024-03-15", True),       # canonicalized
    ("invoice_date", "2024-15-03", "2024-03-15", False),
    ("invoice_date", "garbage", "2024-03-15", False),
    ("due_date", None, None, True),
    ("due_date", None, "2024-03-15", False),
    ("due_date", "2024-03-15", None, False),
    ("currency", "eur", "EUR", True),
    ("currency", "EURO", "EUR", False),
    ("document_type", "Credit Note", "credit_note", True),
    ("document_type", "invoice", "credit_note", False),
    ("total", 100.00, 100.01, True),
    ("total", 100.00, 100.02, False),
    ("total", "1.234,56", 1234.56, True),
    ("total", None, 0.0, False),
    ("tax_rate", None, None, True),
    ("tax_rate", 0, None, False),
    ("discount", 0, 0.0, True),
])
def test_field_correct(field, pred, truth, expected):
    assert field_correct(field, pred, truth) is expected


def test_line_items_order_insensitive_and_strict(truth):
    items = truth["line_items"]
    shuffled = copy.deepcopy(items)[::-1]
    assert line_item_correct(shuffled, items)
    ws = copy.deepcopy(items)
    ws[0]["description"] = "  " + ws[0]["description"].upper().replace(" ", "   ")
    assert line_item_correct(ws, items)
    off = copy.deepcopy(items)
    off[0]["amount"] += 0.02
    assert not line_item_correct(off, items)
    assert not line_item_correct(items[:-1], items)                 # dropped item
    assert not line_item_correct(items + [items[0]], items)         # extra item
    assert not line_item_correct(None, items) and not line_item_correct("x", items)
    qty = copy.deepcopy(items)
    qty[1]["quantity"] += 1
    assert not line_item_correct(qty, items)


def test_score_document_variants(truth):
    perfect = score_document(copy.deepcopy(truth), truth)
    assert all(perfect[f] for f in ALL_FIELDS) and perfect["all_correct"]
    none = score_document(None, truth)
    assert not any(none[f] for f in ALL_FIELDS) and not none["all_correct"]
    wrong = copy.deepcopy(truth)
    wrong["total"] += 50
    del wrong["vendor_name"]
    s = score_document(wrong, truth)
    assert not s["total"] and not s["vendor_name"] and not s["all_correct"]
    assert sum(s[f] for f in ALL_FIELDS) == len(ALL_FIELDS) - 2
    bad_type = copy.deepcopy(truth)
    bad_type["line_items"] = "oops"
    assert not score_document(bad_type, truth)["line_items"]


# statistics
def test_wilson_known_values():
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.7225, abs=1e-3)
    lo, hi = wilson_interval(36, 40)
    assert lo < 0.9 < hi and 0 < lo < hi < 1
    assert all(math.isnan(v) for v in wilson_interval(0, 0))


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 * 0.5 ** 10)
    assert mcnemar_exact(3, 1) == pytest.approx(0.625)


# run scoring and tables
def run(test_records, error_rate, seed=0):
    truths = {r["doc_id"]: r["truth"] for r in test_records}
    m = MockExtractor(truths, error_rate=error_rate, seed=seed)
    return score_run({d: m.extract(d) for d in truths}, test_records)


def test_score_run_perfect_and_columns(test_records):
    df = run(test_records, 0.0)
    assert len(df) == 40 and df["all_correct"].all() and not df["failed"].any()
    for c in ("doc_id", "template", "noise_level", "hard", "doc_type", "number_format", *ALL_FIELDS):
        assert c in df.columns
    # regression: the 'document_type' feature must not be overwritten by the scored boolean field
    assert set(df["doc_type"]) == {"invoice", "credit_note"} and df["document_type"].dtype == bool


def test_score_run_rejects_non_test_records(records):
    with pytest.raises(ValueError):
        score_run({}, [r for r in records if r["split"] == "dev"])
    df = score_run({}, [r for r in records if r["split"] == "dev"], split="dev")  # explicit opt-in
    assert len(df) == 20


def test_missing_and_failed_extractions_count_as_wrong(test_records):
    truths = {r["doc_id"]: r["truth"] for r in test_records}
    ids = sorted(truths)
    ext = {d: ExtractionResult(d, copy.deepcopy(truths[d]), True) for d in ids[1:]}
    ext[ids[0]] = ExtractionResult(ids[0], None, False, "boom")
    del ext[ids[1]]  # missing entirely
    df = score_run(ext, test_records)
    assert len(df) == 40 and df["failed"].sum() == 2 and df["all_correct"].sum() == 38
    tab = accuracy_table(df)
    assert tab.loc[tab.field == "ALL_FIELDS_CORRECT", "correct"].item() == 38


def test_accuracy_table_and_breakdown_consistency(test_records):
    df = run(test_records, 0.5, seed=1)
    tab = accuracy_table(df)
    assert list(tab.field) == list(ALL_FIELDS) + ["ALL_FIELDS_CORRECT"]
    assert (tab.ci_low <= tab.accuracy).all() and (tab.accuracy <= tab.ci_high).all()
    assert (tab.n == 40).all()
    for by in ("template", "noise_level", "hard", "doc_type"):
        b = breakdown(df, by)
        assert b["n"].sum() == 40 and b["correct"].sum() == df["all_correct"].sum()
        assert ((0 <= b.ci_low) & (b.ci_high <= 1)).all()
    assert set(breakdown(df, "template")["template"]) == set(df["template"])


def test_strategy_comparison_tables(test_records):
    runs = {"a": run(test_records, 0.1, 1), "b": run(test_records, 0.5, 2), "c": run(test_records, 0.9, 3)}
    ov = strategy_overview(runs)
    assert list(ov.strategy) == ["a", "b", "c"] and ov.all_correct_rate.is_monotonic_decreasing
    assert (ov.failed_extractions == 0).all()
    ft = strategy_field_table(runs)
    assert ft.shape == (len(ALL_FIELDS), 3)
    pc = paired_comparison(runs)
    assert len(pc) == 3
    for _, r in pc.iterrows():
        assert r.both_correct + r.only_a_correct + r.only_b_correct + r.both_wrong == 40
        assert 0 <= r.mcnemar_p <= 1
    row_ab = pc[(pc.a == "a") & (pc.b == "b")].iloc[0]
    assert row_ab.only_a_correct > row_ab.only_b_correct


def test_paired_comparison_requires_same_documents(test_records):
    d = run(test_records, 0.0)
    with pytest.raises(ValueError):
        paired_comparison({"x": d, "y": d.iloc[:30]})
    assert isinstance(d, pd.DataFrame)
