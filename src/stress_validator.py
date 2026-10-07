"""POST-HOC offline fault-injection test of the validation layer (src/validate.py).

Ground rule for this module: it NEVER imports from or modifies src/validate.py's logic, and makes no
API calls. It takes the 60 seeded ground-truth records (unchanged), applies exactly one corruption to
a deep copy, and asks the EXISTING validator whether it noticed. If the validator has a gap, this
module reports it; it does not patch it.

Each corruption type declares `expected_detectable` BEFORE being run, reasoned from the checks
documented in validate.py's own docstring (arithmetic/date-order/currency-membership/sign checks).
Whether the observation matches that expectation is the headline of each row in the output report;
a mismatch means either a bug in validate.py or a wrong assumption in this module, and both are
reported, not silently resolved.

This is a test of ARTIFICIAL, hand-injected errors in a specific, declared mix (one corruption per
type below) — not a measurement of the model's natural error distribution. See the "natural error"
figures instead in results/error_analysis.md (validation_recall_by_field / reason_code_table), which
are computed from the model's actual mistakes.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from .evaluate import wilson_interval as _wilson_interval
from .report import md_table, pct
from .validate import validate_invoice

CORRUPTIBLE_CURRENCIES = ("USD", "EUR", "GBP", "CAD", "AUD")


@dataclass(frozen=True)
class Corruption:
    name: str
    expected_detectable: bool
    reason: str
    applicable: callable   # truth -> bool: can this corruption even be applied to this record?
    apply: callable        # (truth, rng) -> corrupted copy (deep copy; must differ from the input)


def _has_due_date(t):
    return t["due_date"] is not None


def _has_tax_rate(t):
    return t["tax_rate"] is not None


def _has_nonzero_field(field):
    return lambda t: abs(t[field]) > 0.01


def _has_multi_line_items(t):
    return len(t["line_items"]) >= 1


def _has_two_plus_line_items(t):
    return len(t["line_items"]) >= 2


def _shift_date(iso: str, days: int) -> str:
    return (date.fromisoformat(iso) + timedelta(days=days)).isoformat()


def _c_vendor_name(t, rng):
    c = copy.deepcopy(t)
    s = list(c["vendor_name"])
    idx = rng.randrange(len(s))
    s[idx] = "Z" if s[idx] != "Z" else "Y"
    c["vendor_name"] = "".join(s)
    return c


def _c_invoice_number(t, rng):
    c = copy.deepcopy(t)
    digits = [i for i, ch in enumerate(c["invoice_number"]) if ch.isdigit()]
    idx = rng.choice(digits) if digits else len(c["invoice_number"]) - 1
    s = list(c["invoice_number"])
    if idx < len(s) and s[idx].isdigit():
        s[idx] = str((int(s[idx]) + 1) % 10)
    else:
        s.append("9")
    c["invoice_number"] = "".join(s)
    return c


def _c_invoice_date_shift(t, rng):
    """Shift invoice_date by N days, nudging due_date to preserve due_date >= invoice_date."""
    c = copy.deepcopy(t)
    n = rng.choice([-5, -3, 3, 5, 10])
    c["invoice_date"] = _shift_date(t["invoice_date"], n)
    if c["due_date"] is not None and date.fromisoformat(c["due_date"]) < date.fromisoformat(c["invoice_date"]):
        c["due_date"] = _shift_date(c["invoice_date"], 14)
    return c


def _c_due_date_earlier(t, rng):
    c = copy.deepcopy(t)
    inv = date.fromisoformat(t["invoice_date"])
    c["due_date"] = (inv - timedelta(days=rng.choice([1, 5, 10]))).isoformat()
    return c


def _c_due_date_later(t, rng):
    c = copy.deepcopy(t)
    c["due_date"] = _shift_date(t["due_date"], rng.choice([5, 10, 20]))
    return c


def _c_currency_valid_swap(t, rng):
    c = copy.deepcopy(t)
    others = [x for x in CORRUPTIBLE_CURRENCIES if x != t["currency"]]
    c["currency"] = rng.choice(others) if others else "EUR"
    return c


def _c_currency_invalid(t, rng):
    c = copy.deepcopy(t)
    c["currency"] = "XXX"
    return c


def _c_unit_price_only(t, rng):
    c = copy.deepcopy(t)
    i = rng.randrange(len(c["line_items"]))
    c["line_items"][i]["unit_price"] = round(c["line_items"][i]["unit_price"] * 1.3 + 1, 2)
    return c  # amount deliberately left stale -> amount != quantity * unit_price


def _c_quantity_only(t, rng):
    c = copy.deepcopy(t)
    i = rng.randrange(len(c["line_items"]))
    sign = 1 if c["line_items"][i]["quantity"] >= 0 else -1
    c["line_items"][i]["quantity"] = sign * (abs(c["line_items"][i]["quantity"]) + 3)
    return c  # amount deliberately left stale


def _c_coordinated_price(t, rng):
    """Change ONE line's unit_price and recompute amount, subtotal, tax_amount and total consistently,
    i.e. a fully self-consistent wrong price: the kind of error no arithmetic check can see."""
    c = copy.deepcopy(t)
    i = rng.randrange(len(c["line_items"]))
    sign = 1 if c["line_items"][i]["amount"] >= 0 else -1
    old_amount = c["line_items"][i]["amount"]
    delta_unit = round(rng.uniform(2, 15), 2) * (1 if sign > 0 else -1)
    c["line_items"][i]["unit_price"] = round(c["line_items"][i]["unit_price"] + delta_unit, 2)
    new_amount = round(c["line_items"][i]["quantity"] * c["line_items"][i]["unit_price"] * sign, 2) \
        if sign < 0 else round(c["line_items"][i]["quantity"] * c["line_items"][i]["unit_price"], 2)
    delta = round(new_amount - old_amount, 2)
    c["line_items"][i]["amount"] = new_amount
    is_inclusive = abs(sum(it["amount"] for it in t["line_items"]) - t["total"]) < 0.02 and t["discount"] == 0
    if is_inclusive:
        c["total"] = round(c["total"] + delta, 2)
        if c["tax_rate"] is not None:
            c["tax_amount"] = round(c["total"] * c["tax_rate"] / (100.0 + c["tax_rate"]), 2)
        c["subtotal"] = round(c["total"] - c["tax_amount"], 2)
    else:
        c["subtotal"] = round(c["subtotal"] + delta, 2)
        if c["tax_rate"] is not None:
            c["tax_amount"] = round((c["subtotal"] - c["discount"]) * c["tax_rate"] / 100.0, 2)
        c["total"] = round(c["subtotal"] - c["discount"] + c["tax_amount"], 2)
    return c


def _c_total_transpose(t, rng):
    c = copy.deepcopy(t)
    s = f"{abs(t['total']):.2f}"
    digits = [i for i, ch in enumerate(s) if ch.isdigit() and ch != s[i - 1 if i > 0 else 0]]
    pos = next((i for i in range(len(s) - 1) if s[i].isdigit() and s[i + 1].isdigit() and s[i] != s[i + 1]), None)
    if pos is None:
        chars = list(s)
        chars[0] = str((int(chars[0]) + 1) % 10)
        s = "".join(chars)
    else:
        chars = list(s)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        s = "".join(chars)
    sign = -1 if t["total"] < 0 else 1
    c["total"] = sign * float(s)
    return c


def _c_sign_flip_single(t, rng):
    c = copy.deepcopy(t)
    field = rng.choice([f for f in ("subtotal", "discount", "tax_amount", "total") if abs(t[f]) > 0.01])
    c[field] = -c[field]
    return c


def _c_tax_rate_stale(t, rng):
    c = copy.deepcopy(t)
    c["tax_rate"] = round(c["tax_rate"] + rng.choice([-10, -5, 5, 10, 15]), 2)
    if c["tax_rate"] < 0:
        c["tax_rate"] = round(c["tax_rate"] + 20, 2)
    return c  # tax_amount deliberately left stale


def _c_line_dropped(t, rng):
    c = copy.deepcopy(t)
    c["line_items"].pop(rng.randrange(len(c["line_items"])))
    return c


def _c_line_duplicated(t, rng):
    c = copy.deepcopy(t)
    c["line_items"].append(copy.deepcopy(c["line_items"][rng.randrange(len(c["line_items"]))]))
    return c


def _c_description_changed(t, rng):
    c = copy.deepcopy(t)
    i = rng.randrange(len(c["line_items"]))
    c["line_items"][i]["description"] = c["line_items"][i]["description"] + " (substituted item)"
    return c


CORRUPTIONS = [
    Corruption("vendor_name_char", False, "no check reads vendor_name", lambda t: True, _c_vendor_name),
    Corruption("invoice_number_digit", False, "no check reads invoice_number on its own "
               "(only cross-document duplicates are checked)", lambda t: True, _c_invoice_number),
    Corruption("invoice_date_shift", False, "dates stay internally consistent (due_date >= invoice_date "
               "preserved); no check compares a date to an external reference", lambda t: True, _c_invoice_date_shift),
    Corruption("due_date_earlier", True, "due_before_invoice fires whenever due_date < invoice_date",
               _has_due_date, _c_due_date_earlier),
    Corruption("due_date_later", False, "still due_date >= invoice_date, which is all that check verifies",
               _has_due_date, _c_due_date_later),
    Corruption("currency_valid_swap", False, "the new code is still in the supported set; no check "
               "verifies currency against the vendor, amounts or anything else", lambda t: True, _c_currency_valid_swap),
    Corruption("currency_invalid", True, "invalid_currency fires for any code outside the supported set",
               lambda t: True, _c_currency_invalid),
    Corruption("unit_price_only", True, "line_amount_mismatch fires because amount != quantity * unit_price",
               _has_two_plus_line_items, _c_unit_price_only),
    Corruption("quantity_only", True, "line_amount_mismatch fires because amount != quantity * unit_price",
               _has_two_plus_line_items, _c_quantity_only),
    Corruption("coordinated_price", False, "unit_price, amount, subtotal, tax_amount and total are all "
               "recomputed consistently with each other, so every arithmetic check still balances; this is "
               "the textbook case the validator cannot see", _has_two_plus_line_items, _c_coordinated_price),
    Corruption("total_transpose", True, "total_mismatch fires because subtotal - discount + tax_amount no "
               "longer equals the transposed total", _has_nonzero_field("total"), _c_total_transpose),
    Corruption("sign_flip_single", True, "sign_inconsistent and/or total_mismatch fire: flipping one money "
               "field breaks both the document-type sign convention and the subtotal/tax/total identity",
               lambda t: any(abs(t[f]) > 0.01 for f in ("subtotal", "discount", "tax_amount", "total")),
               _c_sign_flip_single),
    Corruption("tax_rate_stale", True, "tax_rate_mismatch fires because tax_amount no longer matches the "
               "new rate (exclusive or inclusive basis)", _has_tax_rate, _c_tax_rate_stale),
    Corruption("line_dropped", True, "line_items_sum_mismatch fires because the remaining lines no longer "
               "sum to subtotal (or total)", _has_two_plus_line_items, _c_line_dropped),
    Corruption("line_duplicated", True, "line_items_sum_mismatch fires because the lines now over-sum",
               _has_multi_line_items, _c_line_duplicated),
    Corruption("description_changed", False, "no check reads line-item descriptions",
               _has_multi_line_items, _c_description_changed),
]


def run_trials(records: list[dict], seed: int = 0) -> list[dict]:
    """One trial per (corruption type, record) where the corruption is applicable and actually changes
    the record. Deterministic for a given seed and record list."""
    rows = []
    for c in CORRUPTIONS:
        for r in sorted(records, key=lambda r: r["doc_id"]):
            truth = r["truth"]
            if not c.applicable(truth):
                continue
            rng = random.Random(f"{seed}-{c.name}-{r['doc_id']}")
            corrupted = c.apply(truth, rng)
            if corrupted == truth:
                continue  # not a valid trial: the corruption had no effect on this record
            result = validate_invoice(corrupted)
            rows.append({"corruption": c.name, "doc_id": r["doc_id"], "expected_detectable": c.expected_detectable,
                        "reason": c.reason, "detected": result["status"] == "flag",
                        "reason_codes": [x["code"] for x in result["reasons"]]})
    return rows


def run_clean_check(records: list[dict]) -> list[dict]:
    """Validate every UNCORRUPTED ground-truth record; any flag here is a false positive."""
    rows = []
    for r in sorted(records, key=lambda r: r["doc_id"]):
        result = validate_invoice(r["truth"])
        if result["status"] == "flag":
            rows.append({"doc_id": r["doc_id"], "reason_codes": [x["code"] for x in result["reasons"]]})
    return rows


def build_stress_report(trial_rows: list[dict], clean_rows: list[dict], seed: int) -> str:
    """Render results/validator_stress_test.md. Mismatches (observed != expected_detectable) are listed
    first and separately so a validator bug or a wrong assumption here cannot hide in a pooled average.
    """
    by_type: dict[str, list[dict]] = {}
    for r in trial_rows:
        by_type.setdefault(r["corruption"], []).append(r)

    mismatches = []
    for name, rows in by_type.items():
        expected = rows[0]["expected_detectable"]
        observed_any = any(r["detected"] for r in rows)
        observed_all = all(r["detected"] for r in rows)
        if expected and not observed_all:
            mismatches.append((name, expected, f"expected every trial to be detected; "
                               f"{sum(r['detected'] for r in rows)}/{len(rows)} were"))
        elif not expected and observed_any:
            mismatches.append((name, expected, f"expected NONE to be detected; "
                               f"{sum(r['detected'] for r in rows)}/{len(rows)} were flagged"))

    L = ["# Validator fault-injection stress test (post-hoc)", "",
         "> **Post-hoc.** This file is generated by `src/stress_validator.py` against the EXISTING, "
         "unmodified `src/validate.py` (this stage made no changes to it). It injects artificial, "
         "hand-chosen corruptions into the 60 seeded ground-truth records and asks the validator whether "
         "it noticed. Made no API calls; the generator, seed and cached model responses are untouched.",
         "",
         f"Seed: {seed}. Only a trial where the corruption actually changed the record is counted "
         "(some corruptions are inapplicable to some records, e.g. a due-date shift needs a due date).",
         "",
         "**The corruption mix below is this module's own choice, not a sample of anything** — it says "
         "nothing about how often each error type occurs in practice. Per-corruption-type rows are what "
         "matter; do not average them into one headline detection rate. These are INJECTED errors, not "
         "the model's natural errors — see `results/error_analysis.md` and the "
         "`validation_recall_by_field` / reason-code tables in `results/summary.md` for the model's "
         "actual mistakes and how often the validator caught those.", ""]

    L += ["## False positives on clean (uncorrupted) ground truth", "",
          f"{len(clean_rows)} of 60 uncorrupted records were flagged (expected: 0).", ""]
    if clean_rows:
        L += [md_table(["doc_id", "reason codes fired"], [[r["doc_id"], ", ".join(r["reason_codes"])]
                                                           for r in clean_rows]), ""]

    L += ["## Mismatches between expected and observed detectability", ""]
    if mismatches:
        L += [md_table(["corruption", "expected_detectable", "what happened"], mismatches), ""]
    else:
        L += ["None: every corruption type's observed detection matched its `expected_detectable` "
              "assumption exactly (either 0/n or n/n detected, consistent with a deterministic check).", ""]

    L += ["## Detection rate by corruption type", "", "Not pooled into a single headline figure (see note above).",
          "", "| Corruption | Trials | Detected | Detection rate [95% Wilson CI] | expected_detectable | "
          "Reason | Matched? |", "|---|---|---|---|---|---|---|"]
    for c in CORRUPTIONS:
        rows = by_type.get(c.name, [])
        n, k = len(rows), sum(r["detected"] for r in rows)
        if n == 0:
            rate_s, matched = "n/a (0 applicable trials)", "n/a"
        else:
            lo, hi = _wilson_interval(k, n)
            rate_s = f"{k}/{n} = {pct(k / n)} [{pct(lo)}–{pct(hi)}]"
            matched = "yes" if (bool(k) == c.expected_detectable or (c.expected_detectable and k == n)
                                or (not c.expected_detectable and k == 0)) else "NO"
        cells = [c.name, n, k, rate_s, str(c.expected_detectable), c.reason, matched]
        L.append("| " + " | ".join(str(x).replace("|", "\\|") for x in cells) + " |")
    L.append("")
    return "\n".join(L).rstrip() + "\n"


def write_stress_report(out_path, records: list[dict], seed: int = 0):
    trial_rows = run_trials(records, seed)
    clean_rows = run_clean_check(records)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_stress_report(trial_rows, clean_rows, seed), encoding="utf-8")
    return out
