import copy
import pytest

from src.generate import build_records
from src.validate import validate_batch, validate_invoice


@pytest.fixture(scope="module")
def records():
    return build_records(42)


@pytest.fixture(scope="module")
def truths(records):
    return {r["doc_id"]: r["truth"] for r in records}


def codes(x):
    return {r["code"] for r in validate_invoice(x)["reasons"]}


@pytest.fixture()
def base(records):
    # a plain tax-exclusive invoice with a discount, found by property rather than by id
    for r in records:
        f = r["meta"]["features"]
        if (f["document_type"] == "invoice" and not f["tax_inclusive"] and f["has_discount"]
                and r["truth"]["tax_rate"] and len(r["truth"]["line_items"]) >= 2):
            return copy.deepcopy(r["truth"])
    raise AssertionError("no suitable base document")


def test_all_generator_truths_pass(truths):
    """Validator and generator agree: every ground-truth record (incl. inclusive, credit, null-rate) passes."""
    for doc_id, t in truths.items():
        res = validate_invoice(t)
        assert res["status"] == "pass", (doc_id, res["reasons"])


def test_pass_has_no_reasons(base):
    assert validate_invoice(base) == {"status": "pass", "reasons": []}


def test_extraction_failed():
    assert codes(None) == {"extraction_failed"}


@pytest.mark.parametrize("field", ["vendor_name", "invoice_number", "invoice_date", "currency", "total", "line_items"])
def test_missing_required(base, field):
    base[field] = None
    assert "missing_required" in codes(base)


def test_nullable_fields_may_be_null(base):
    base["due_date"] = None
    base["tax_rate"] = None
    assert validate_invoice(base)["status"] == "pass"


def test_total_mismatch(base):
    base["total"] += 5
    assert codes(base) == {"total_mismatch"}


def test_line_items_sum_mismatch(base):
    base["line_items"][0]["amount"] += 3
    c = codes(base)
    assert "line_items_sum_mismatch" in c and "line_amount_mismatch" in c


def test_dropped_line_item_is_caught(base):
    base["line_items"].pop()
    assert "line_items_sum_mismatch" in codes(base)


def test_line_amount_mismatch_alone(base):
    # keep the sum intact by shifting amount between two lines: only per-line check should fire
    base["line_items"][0]["amount"] += 2
    base["line_items"][1]["amount"] -= 2
    assert codes(base) == {"line_amount_mismatch"}


def test_tax_rate_mismatch(base):
    base["tax_rate"] += 7
    assert codes(base) == {"tax_rate_mismatch"}


def test_tax_inclusive_basis_accepted():
    x = {"document_type": "invoice", "vendor_name": "V", "invoice_number": "1", "invoice_date": "2024-01-01",
         "due_date": None, "currency": "EUR", "tax_rate": 20.0, "discount": 0.0,
         "line_items": [{"description": "a", "quantity": 1, "unit_price": 120.0, "amount": 120.0}],
         "subtotal": 100.0, "tax_amount": 20.0, "total": 120.0}
    assert validate_invoice(x)["status"] == "pass"
    x["tax_amount"] = 25.0  # inconsistent both ways
    assert {"total_mismatch", "tax_rate_mismatch"} <= codes(x)


@pytest.mark.parametrize("d", ["15/03/2024", "2024-02-30", "March 5", "2024-3-5"])
def test_invalid_date(base, d):
    base["invoice_date"] = d
    assert "invalid_date" in codes(base)


def test_due_before_invoice(base):
    base["due_date"] = "2000-01-01"
    assert codes(base) == {"due_before_invoice"}


def test_currency_and_doc_type(base):
    base["currency"] = "XXX"
    assert "invalid_currency" in codes(base)
    base["currency"] = "USD"
    base["document_type"] = "receipt"
    assert "invalid_document_type" in codes(base)


def test_unparseable_number(base):
    base["total"] = "n/a"
    assert "unparseable_number" in codes(base)


def test_string_numbers_in_european_format_are_parsed(base):
    base["total"] = f"{base['total']:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    assert validate_invoice(base)["status"] == "pass"


def test_credit_note_sign_errors(truths):
    cn = next(copy.deepcopy(t) for t in truths.values() if t["document_type"] == "credit_note")
    assert validate_invoice(cn)["status"] == "pass"
    pos = copy.deepcopy(cn)
    for k in ("subtotal", "discount", "tax_amount", "total"):
        pos[k] = abs(pos[k])
    for it in pos["line_items"]:
        it["amount"] = abs(it["amount"])
    assert "sign_inconsistent" in codes(pos)
    asinv = copy.deepcopy(cn)
    asinv["document_type"] = "invoice"
    assert "sign_inconsistent" in codes(asinv)


def test_tolerance_boundary(base):
    base["total"] += 0.01
    assert validate_invoice(base)["status"] == "pass"
    base["total"] += 0.02
    assert "total_mismatch" in codes(base)


def test_validate_batch_dataframe_and_duplicates(base):
    bad = copy.deepcopy(base)
    bad["total"] += 10
    bad["invoice_number"] = "DIFFERENT-1"
    dup = copy.deepcopy(base)
    dup["invoice_number"] = " " + base["invoice_number"].lower()
    df = validate_batch({"a": base, "b": bad, "c": None, "d": dup})
    st = df.set_index("doc_id")
    assert st.loc["b", "reason_codes"] == ["total_mismatch"]
    assert st.loc["c", "reason_codes"] == ["extraction_failed"]
    assert st.loc["a", "reason_codes"] == ["duplicate_invoice_number"] == st.loc["d", "reason_codes"]
    assert list(df.columns) == ["doc_id", "status", "reason_codes", "reasons", "n_reasons"]
    assert (st["status"] == "flag").all()
    clean = validate_batch({"a": base})
    assert clean.loc[0, "status"] == "pass" and clean.loc[0, "n_reasons"] == 0
