import copy
import math

import pandas as pd
import pytest

from src.evaluate import (CATEGORY_DESCRIPTIONS, classify_line_items_error, classify_scalar_error, error_records,
                          evaluate_validation, reason_code_table, score_document, score_run,
                          validation_recall_by_field)
from src.extract import ExtractionResult, MockExtractor
from src.generate import build_records
from src.schema import ALL_FIELDS
from src.validate import validate_batch


def scores_and_validation():
    """10 docs: d0-d3 wrong, d4-d9 correct. Flagged: d0,d1,d2 (wrong) and d4,d5 (correct)."""
    ids = [f"d{i}" for i in range(10)]
    wrong = {"d0", "d1", "d2", "d3"}
    scores = pd.DataFrame({"doc_id": ids, "all_correct": [d not in wrong for d in ids]})
    for f in ALL_FIELDS:
        scores[f] = scores["all_correct"]
    for f in ALL_FIELDS:                                           # d3: only the vendor is wrong, unflagged
        scores.loc[scores.doc_id == "d3", f] = f != "vendor_name"
    flagged = {"d0": ["total_mismatch"], "d1": ["total_mismatch", "tax_rate_mismatch"], "d2": ["invalid_date"],
               "d4": ["invalid_currency"], "d5": ["total_mismatch"]}
    val = pd.DataFrame({"doc_id": ids, "status": ["flag" if d in flagged else "pass" for d in ids],
                        "reason_codes": [flagged.get(d, []) for d in ids]})
    return scores, val


def test_validation_metrics_hand_computed():
    s, v = scores_and_validation()
    r = evaluate_validation(s, v)
    assert (r["n"], r["n_wrong"], r["n_flagged"]) == (10, 4, 5)
    assert (r["true_flags"], r["false_flags"], r["silent_errors"], r["straight_through"]) == (3, 2, 1, 4)
    assert r["recall"]["accuracy"] == pytest.approx(3 / 4) and r["recall"]["n"] == 4
    assert r["precision"]["accuracy"] == pytest.approx(3 / 5)
    assert r["straight_through_rate"]["accuracy"] == pytest.approx(4 / 10)
    assert r["false_flag_rate"]["accuracy"] == pytest.approx(2 / 6)
    assert r["error_rate_unflagged"]["accuracy"] == pytest.approx(1 / 5)
    lo, hi = r["recall"]["ci_low"], r["recall"]["ci_high"]
    assert lo < 0.75 < hi


def test_validation_metrics_degenerate_cases():
    s, v = scores_and_validation()
    s["all_correct"] = True
    r = evaluate_validation(s, v)
    assert r["n_wrong"] == 0 and math.isnan(r["recall"]["accuracy"])
    v["status"] = "pass"
    r = evaluate_validation(s, v)
    assert math.isnan(r["precision"]["accuracy"]) and r["straight_through_rate"]["accuracy"] == 1.0
    with pytest.raises(ValueError):
        evaluate_validation(s, v.iloc[:5])


def test_recall_by_field_and_reason_codes():
    s, v = scores_and_validation()
    rf = validation_recall_by_field(s, v).set_index("field")
    assert rf.loc["vendor_name", "n_wrong_docs"] == 4 and rf.loc["vendor_name", "flagged"] == 3
    assert rf.loc["total", "n_wrong_docs"] == 3
    rc = reason_code_table(s, v).set_index("reason_code")
    assert rc.loc["total_mismatch", "n_flagged"] == 3 and rc.loc["total_mismatch", "n_truly_wrong"] == 2
    assert rc.loc["total_mismatch", "precision"] == pytest.approx(2 / 3)
    assert rc.loc["invalid_currency", "precision"] == 0.0


# taxonomy
@pytest.mark.parametrize("field,pred,truth,feat,expected", [
    ("total", 50.0, -50.0, {}, "sign_error"),
    ("discount", -10.0, 10.0, {}, "sign_error"),
    ("subtotal", 120.0, 100.0, {"tax_inclusive": True}, "tax_inclusive_confusion"),
    ("subtotal", 120.0, 100.0, {"tax_inclusive": False}, "wrong_number"),
    ("total", 123456.0, 1234.56, {}, "magnitude_error"),
    ("total", 1.23456, 1234.56, {}, "magnitude_error"),
    ("total", 99.0, 100.0, {}, "wrong_number"),
    ("total", "abc", 100.0, {}, "unparseable_number"),
    ("tax_rate", None, 20.0, {}, "missing_value"),
    ("tax_rate", 20.0, None, {}, "spurious_value"),
    ("due_date", "2024-04-01", None, {}, "spurious_value"),
    ("due_date", None, "2024-04-01", {}, "missing_value"),
    ("invoice_date", "2024-05-03", "2024-03-05", {}, "day_month_swap"),
    ("invoice_date", "2024-15-03", "2024-03-15", {}, "unparseable_date"),
    ("invoice_date", "2024-04-15", "2024-03-15", {}, "wrong_date"),
    ("vendor_name", "Other Co", "Acme Ltd", {}, "vendor_name_mismatch"),
    ("invoice_number", "X1", "Y1", {}, "invoice_number_mismatch"),
    ("currency", "USD", "EUR", {}, "currency_mismatch"),
    ("document_type", "invoice", "credit_note", {}, "document_type_mismatch"),
    ("vendor_name", None, "Acme", {}, "missing_value"),
])
def test_classify_scalar_error(field, pred, truth, feat, expected):
    assert classify_scalar_error(field, pred, truth, feat) == expected
    assert expected in CATEGORY_DESCRIPTIONS


def test_classify_line_items_error():
    truth = [{"description": "Widget Pro Model X, blue", "quantity": 2, "unit_price": 10.0, "amount": 20.0},
             {"description": "Gadget", "quantity": 1, "unit_price": 5.0, "amount": 5.0}]
    c = lambda p: classify_line_items_error(p, truth)[0]  # noqa: E731
    assert c(truth[:1]) == "dropped_line_item"
    assert c(truth + truth[:1]) == "extra_line_item"
    assert c("nope") == "line_items_malformed" and c([1, 2]) == "line_items_malformed"
    trunc = copy.deepcopy(truth)
    trunc[0]["description"] = "Widget Pro"
    assert c(trunc) == "description_truncated"
    other = copy.deepcopy(truth)
    other[1]["description"] = "Gizmo"
    assert c(other) == "description_mismatch"
    val = copy.deepcopy(truth)
    val[1]["amount"] = 6.0
    cat, detail = classify_line_items_error(val, truth)
    assert cat == "line_item_value_error" and "amount" in detail and "Gadget" in detail


def test_error_records_cover_every_wrong_field_and_failures():
    recs = [r for r in build_records(42) if r["split"] == "test"]
    truths = {r["doc_id"]: r["truth"] for r in recs}
    m = MockExtractor(truths, error_rate=1.0, seed=5)
    ext = {d: m.extract(d) for d in truths}
    ext["test_00"] = ExtractionResult("test_00", None, False, "boom")
    val = validate_batch({d: r.extraction for d, r in ext.items()})
    errs = error_records("s", ext, recs, val)
    assert all(e["category"] in CATEGORY_DESCRIPTIONS for e in errs)
    assert [e for e in errs if e["doc_id"] == "test_00"][0]["category"] == "extraction_failure"
    df = score_run(ext, recs)
    expected_wrong_fields = int((~df[list(ALL_FIELDS)]).to_numpy().sum()) - len(ALL_FIELDS)  # failed doc: 1 row not 12
    assert len(errs) == expected_wrong_fields + 1
    assert {e["doc_id"] for e in errs} == set(df.loc[~df.all_correct, "doc_id"])
    flagged = {e["doc_id"] for e in errs if e["flagged"]}
    assert flagged == set(val.loc[val.status == "flag", "doc_id"]) & {e["doc_id"] for e in errs}


def test_error_records_empty_for_perfect_run():
    recs = [r for r in build_records(42) if r["split"] == "test"]
    ext = {r["doc_id"]: copy.deepcopy(r["truth"]) for r in recs}
    assert error_records("s", ext, recs) == []
    assert all(score_document(ext[r["doc_id"]], r["truth"])["all_correct"] for r in recs)
