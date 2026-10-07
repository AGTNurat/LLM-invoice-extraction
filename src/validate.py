"""Deterministic validation checks that flag suspect extractions.

These checks look only at the extraction itself, so they are usable in
production. Each failed check adds a reason code; an invoice is "pass" only if no check failed.

Reason codes:
  extraction_failed        no extraction object at all
  missing_required         a required field is missing/null/empty
  unparseable_number       a numeric field is not a number
  invalid_date             a date is not a real ISO date (YYYY-MM-DD)
  due_before_invoice       due_date earlier than invoice_date
  invalid_currency         currency not in the supported ISO set
  invalid_document_type    not "invoice" / "credit_note"
  sign_inconsistent        amounts' signs contradict the document type (schema sign convention)
  line_amount_mismatch     a line's amount != quantity * unit_price (sign-adjusted)
  line_items_sum_mismatch  line amounts do not sum to subtotal (or to total on a tax-inclusive basis)
  total_mismatch           subtotal - discount + tax_amount != total
  tax_rate_mismatch        tax_amount inconsistent with tax_rate (exclusive or inclusive basis)
  duplicate_invoice_number same vendor + invoice number appears more than once (batch check)
"""
from __future__ import annotations

import re
from datetime import date

import pandas as pd

from .normalize import parse_number
from .schema import CURRENCIES, DOCUMENT_TYPES, LINE_ITEM_FIELDS, REQUIRED_FIELDS

TOL = 0.01
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")
_MONEY = ("subtotal", "discount", "tax_amount", "total")


def _num(v):
    return parse_number(v)


def _iso_date(v) -> date | None:
    if not isinstance(v, str) or not _ISO.fullmatch(v.strip()):
        return None
    try:
        return date.fromisoformat(v.strip())
    except ValueError:
        return None


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol + 1e-9


def validate_invoice(x: dict | None, tol: float = TOL) -> dict:
    """Return {"status": "pass"|"flag", "reasons": [{"code", "message"}, ...]}."""
    reasons: list[dict] = []

    def fail(code: str, msg: str):
        reasons.append({"code": code, "message": msg})

    if not isinstance(x, dict):
        fail("extraction_failed", "no extraction available")
        return {"status": "flag", "reasons": reasons}

    missing = [f for f in REQUIRED_FIELDS
               if x.get(f) is None or (isinstance(x.get(f), (str, list)) and len(x.get(f)) == 0)]
    if missing:
        fail("missing_required", "missing: " + ", ".join(missing))

    # --- numbers
    nums = {k: _num(x.get(k)) for k in _MONEY}
    bad = [k for k, v in nums.items() if x.get(k) is not None and v is None]
    rate = _num(x.get("tax_rate")) if x.get("tax_rate") is not None else None
    if x.get("tax_rate") is not None and rate is None:
        bad.append("tax_rate")
    items = x.get("line_items") if isinstance(x.get("line_items"), list) else []
    parsed_items = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            bad.append(f"line_items[{i}]")
            continue
        p = {k: (_num(it.get(k)) if k != "description" else it.get(k)) for k in LINE_ITEM_FIELDS}
        if any(p[k] is None for k in ("quantity", "unit_price", "amount")):
            bad.append(f"line_items[{i}]")
        parsed_items.append(p)
    if bad:
        fail("unparseable_number", "not numeric: " + ", ".join(bad))

    # --- dates
    inv = _iso_date(x.get("invoice_date")) if x.get("invoice_date") is not None else None
    due = _iso_date(x.get("due_date")) if x.get("due_date") is not None else None
    bad_dates = [f for f in ("invoice_date", "due_date")
                 if x.get(f) is not None and (inv if f == "invoice_date" else due) is None]
    if bad_dates:
        fail("invalid_date", "not a valid ISO date: " + ", ".join(f"{f}={x.get(f)!r}" for f in bad_dates))
    if inv and due and due < inv:
        fail("due_before_invoice", f"due_date {due} is before invoice_date {inv}")

    # --- categorical
    cur = x.get("currency")
    if cur is not None and (not isinstance(cur, str) or cur.strip().upper() not in CURRENCIES):
        fail("invalid_currency", f"currency {cur!r} not in supported set")
    dt = x.get("document_type")
    if dt is not None and dt not in DOCUMENT_TYPES:
        fail("invalid_document_type", f"document_type {dt!r}")

    # --- sign convention (only checkable when the type is valid and numbers parsed)
    if dt in DOCUMENT_TYPES:
        sign = -1 if dt == "credit_note" else 1
        wrong = [k for k, v in nums.items() if v is not None and v * sign < -tol]
        if wrong:
            fail("sign_inconsistent", f"{dt} but wrong sign on: " + ", ".join(wrong))
    else:
        sign = None

    # --- line level and arithmetic
    if sign is not None:
        for i, p in enumerate(parsed_items):
            if all(p[k] is not None for k in ("quantity", "unit_price", "amount")):
                if not _close(p["amount"], sign * p["quantity"] * p["unit_price"], tol):
                    fail("line_amount_mismatch",
                         f"line {i + 1}: amount {p['amount']} != {sign} * {p['quantity']} * {p['unit_price']}")
    sub, disc, tax, total = (nums[k] for k in _MONEY)
    complete = [p for p in parsed_items if p["amount"] is not None]
    if complete and len(complete) == len(parsed_items) and sub is not None:
        s = sum(p["amount"] for p in complete)
        exclusive_ok = _close(s, sub, tol)
        inclusive_ok = total is not None and _close(s, total, tol) and (disc is None or _close(disc, 0.0, tol))
        if not (exclusive_ok or inclusive_ok):
            fail("line_items_sum_mismatch", f"sum of lines {s:.2f} vs subtotal {sub:.2f}"
                 + (f" / total {total:.2f}" if total is not None else ""))
    if None not in (sub, disc, tax, total) and not _close(sub - disc + tax, total, tol):
        fail("total_mismatch", f"{sub:.2f} - {disc:.2f} + {tax:.2f} = {sub - disc + tax:.2f} != total {total:.2f}")
    if rate is not None and tax is not None and sub is not None:
        excl = (sub - (disc or 0.0)) * rate / 100.0
        incl = total * rate / (100.0 + rate) if total is not None else None
        ok = _close(tax, excl, 2 * tol) or (incl is not None and _close(tax, incl, 2 * tol))
        if not ok:
            fail("tax_rate_mismatch", f"tax {tax:.2f} vs {rate:g}% (expected {excl:.2f}"
                 + (f" or {incl:.2f} if tax-inclusive)" if incl is not None else ")"))

    return {"status": "flag" if reasons else "pass", "reasons": reasons}


def validate_batch(extractions: dict[str, dict | None], tol: float = TOL) -> pd.DataFrame:
    """Validate many extractions (keyed by doc_id). Adds the cross-document duplicate check.

    Returns one row per doc: doc_id, status, reason_codes (list), reasons (messages), n_reasons.
    """
    rows = []
    for doc_id, x in extractions.items():
        res = validate_invoice(x, tol)
        rows.append({"doc_id": doc_id, "status": res["status"],
                     "reason_codes": [r["code"] for r in res["reasons"]],
                     "reasons": [r["message"] for r in res["reasons"]]})
    df = pd.DataFrame(rows, columns=["doc_id", "status", "reason_codes", "reasons"])
    keys = pd.Series({d: (re.sub(r"\s+", " ", str(x.get("vendor_name", ""))).strip().casefold(),
                          re.sub(r"\s+", "", str(x.get("invoice_number", ""))).upper())
                      for d, x in extractions.items() if isinstance(x, dict) and x.get("invoice_number")})
    dup_ids = set(keys[keys.duplicated(keep=False)].index) if len(keys) else set()
    for i in df.index[df["doc_id"].isin(dup_ids)]:
        df.at[i, "reason_codes"] = df.at[i, "reason_codes"] + ["duplicate_invoice_number"]
        df.at[i, "reasons"] = df.at[i, "reasons"] + ["same vendor and invoice number as another document"]
        df.at[i, "status"] = "flag"
    df["n_reasons"] = df["reason_codes"].map(len)
    return df
