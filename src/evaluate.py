"""Evaluation metrics: normalized field comparison, accuracy tables with Wilson CIs, breakdowns,
and strategy comparison. Scores the TEST split by default and refuses to mix splits.

Scoring rules
  * Numbers match within 0.01 (None only matches None); dates are canonicalized to ISO; names are
    case-folded with whitespace collapsed; invoice numbers ignore case and whitespace; currency and
    document type are compared after normalization.
  * line_items is ONE composite field: correct only if the item count matches and every item's
    description, quantity, unit_price and amount match. Item order is ignored.
  * A failed or missing extraction counts as wrong on every field (never dropped from the denominator).
"""
from __future__ import annotations

import math
from itertools import combinations

import pandas as pd

from .normalize import (description_key, normalize_currency, normalize_date, normalize_doc_type,
                        normalize_invoice_number, normalize_name, numbers_match, parse_number)
from .schema import ALL_FIELDS, HEADER_NUM_FIELDS, LINE_ITEM_FIELDS

META_COLUMNS = ("template", "noise_level", "hard", "hard_reason")
FEATURE_COLUMNS = ("document_type", "number_format", "date_format", "tax_inclusive", "has_discount",
                   "multiline_descriptions")


# comparison
def _line_key(it: dict):
    return (normalize_name(it.get("description")) or "", parse_number(it.get("amount")) or 0.0,
            parse_number(it.get("quantity")) or 0.0, parse_number(it.get("unit_price")) or 0.0)


def line_item_correct(pred, truth) -> bool:
    if not isinstance(pred, list) or len(pred) != len(truth) or not all(isinstance(i, dict) for i in pred):
        return False
    for p, t in zip(sorted(pred, key=_line_key), sorted(truth, key=_line_key)):
        if normalize_name(p.get("description")) != normalize_name(t.get("description")):
            return False
        if not all(numbers_match(p.get(k), t.get(k)) for k in LINE_ITEM_FIELDS if k != "description"):
            return False
    return True


def field_correct(field: str, pred, truth) -> bool:
    if field == "line_items":
        return line_item_correct(pred, truth)
    if field in HEADER_NUM_FIELDS:
        return numbers_match(pred, truth)
    norm = {"invoice_date": normalize_date, "due_date": normalize_date, "vendor_name": normalize_name,
            "invoice_number": normalize_invoice_number, "currency": normalize_currency,
            "document_type": normalize_doc_type}[field]
    if truth is None or pred is None:
        return truth is None and pred is None
    n_pred, n_truth = norm(pred), norm(truth)
    return n_pred is not None and n_pred == n_truth


def score_document(pred: dict | None, truth: dict) -> dict[str, bool]:
    """Per-field correctness plus 'all_correct'. A missing prediction or field counts as wrong."""
    out = {}
    for f in ALL_FIELDS:
        if not isinstance(pred, dict) or f not in pred:
            out[f] = False
        else:
            try:
                out[f] = bool(field_correct(f, pred[f], truth[f]))
            except (TypeError, ValueError, AttributeError):
                out[f] = False
    out["all_correct"] = all(out[f] for f in ALL_FIELDS)
    return out


def _extraction_of(x):
    return getattr(x, "extraction", x)


def score_run(extractions: dict, records: list[dict], split: str = "test") -> pd.DataFrame:
    """One row per ground truth record: metadata, per-field booleans, all_correct, failed.

    ``extractions`` maps doc_id -> ExtractionResult | extraction dict | None. Records must all belong to
    ``split`` (default "test"); anything else raises, so dev documents cannot leak into reported numbers.
    """
    bad = [r["doc_id"] for r in records if r["split"] != split]
    if bad:
        raise ValueError(f"score_run is restricted to the {split!r} split; got {bad[:3]}...")
    rows = []
    for r in sorted(records, key=lambda r: r["doc_id"]):
        pred = _extraction_of(extractions.get(r["doc_id"]))
        sc = score_document(pred, r["truth"])
        m, f = r["meta"], r["meta"]["features"]
        # 'document_type' is both a feature and a scored field; the feature column is renamed to avoid a clash
        feats = {("doc_type" if c == "document_type" else c): f[c] for c in FEATURE_COLUMNS}
        rows.append({"doc_id": r["doc_id"], **{c: m[c] for c in META_COLUMNS}, **feats,
                     "failed": not isinstance(pred, dict), **sc})
    return pd.DataFrame(rows)


# statistics
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _prop_row(label, k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"label": label, "n": n, "correct": k, "accuracy": k / n if n else float("nan"),
            "ci_low": lo, "ci_high": hi}


# tables
def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-field accuracy with Wilson CIs, plus the invoice-level 'all fields correct' row."""
    rows = [_prop_row(f, int(df[f].sum()), len(df)) for f in ALL_FIELDS]
    rows.append(_prop_row("ALL_FIELDS_CORRECT", int(df["all_correct"].sum()), len(df)))
    return pd.DataFrame(rows).rename(columns={"label": "field"})


def breakdown(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Invoice level all-correct rate (Wilson CI) and pooled field accuracy per group of ``by``.

    pooled_field_accuracy is a point estimate over (documents x fields); no CI is given because fields
    within a document are not independent.
    """
    rows = []
    for key, g in df.groupby(by, sort=True):
        row = _prop_row(key, int(g["all_correct"].sum()), len(g))
        row["pooled_field_accuracy"] = float(g[list(ALL_FIELDS)].to_numpy().mean())
        rows.append(row)
    return pd.DataFrame(rows).rename(columns={"label": by})


def strategy_overview(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per strategy: invoice-level accuracy + CI, pooled field accuracy, failed extractions."""
    rows = []
    for name, df in runs.items():
        row = _prop_row(name, int(df["all_correct"].sum()), len(df))
        row["pooled_field_accuracy"] = float(df[list(ALL_FIELDS)].to_numpy().mean())
        row["failed_extractions"] = int(df["failed"].sum())
        rows.append(row)
    return pd.DataFrame(rows).rename(columns={"label": "strategy", "accuracy": "all_correct_rate"})


def strategy_field_table(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Field x strategy accuracy matrix (point estimates)."""
    return pd.DataFrame({name: {f: df[f].mean() for f in ALL_FIELDS} for name, df in runs.items()})


def paired_comparison(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """For each pair of strategies on the same documents: discordant counts and exact McNemar p-value
    on invoice-level correctness."""
    rows = []
    for a, b in combinations(runs, 2):
        da, db = runs[a].set_index("doc_id")["all_correct"], runs[b].set_index("doc_id")["all_correct"]
        if set(da.index) != set(db.index):
            raise ValueError("strategies were scored on different document sets")
        db = db.loc[da.index]
        only_a, only_b = int((da & ~db).sum()), int((~da & db).sum())
        rows.append({"a": a, "b": b, "both_correct": int((da & db).sum()), "only_a_correct": only_a,
                     "only_b_correct": only_b, "both_wrong": int((~da & ~db).sum()),
                     "mcnemar_p": mcnemar_exact(only_a, only_b)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- validation layer
def evaluate_validation(scores: pd.DataFrame, validation: pd.DataFrame) -> dict:
    """How well do the deterministic checks separate wrong extractions from correct ones?

    wrong = not all fields correct (failed extractions count as wrong); flagged = validator status "flag".
      recall               : share of wrong extractions that were flagged
      precision            : share of flagged extractions that were truly wrong
      straight_through     : share of ALL documents that are correct AND unflagged (no human touch needed)
      false_flag_rate      : share of correct extractions that were flagged (needless review)
      error_rate_unflagged : share of unflagged documents that are wrong (errors that would slip through)
    Each rate is a dict with n, correct (=numerator), accuracy (=rate) and a Wilson 95% interval.
    """
    m = scores[["doc_id", "all_correct"]].merge(validation[["doc_id", "status"]], on="doc_id", how="left",
                                                validate="one_to_one")
    if m["status"].isna().any():
        raise ValueError("validation is missing for some scored documents")
    wrong, flagged = ~m["all_correct"], m["status"] == "flag"
    n = len(m)
    c = {"n": n, "n_correct": int((~wrong).sum()), "n_wrong": int(wrong.sum()), "n_flagged": int(flagged.sum()),
         "true_flags": int((wrong & flagged).sum()), "false_flags": int((~wrong & flagged).sum()),
         "silent_errors": int((wrong & ~flagged).sum()), "straight_through": int((~wrong & ~flagged).sum())}
    return {**c,
            "recall": _prop_row("recall", c["true_flags"], c["n_wrong"]),
            "precision": _prop_row("precision", c["true_flags"], c["n_flagged"]),
            "straight_through_rate": _prop_row("straight_through_rate", c["straight_through"], n),
            "false_flag_rate": _prop_row("false_flag_rate", c["false_flags"], c["n_correct"]),
            "error_rate_unflagged": _prop_row("error_rate_unflagged", c["silent_errors"],
                                              c["silent_errors"] + c["straight_through"])}


def validation_recall_by_field(scores: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """For each field: among documents where that field is wrong, how many were flagged by ANY check.

    Shows which kinds of error the checks cannot see (e.g. a wrong vendor name is never flagged).
    """
    m = scores.merge(validation[["doc_id", "status"]], on="doc_id", validate="one_to_one")
    rows = []
    for f in ALL_FIELDS:
        w = m[~m[f]]
        rows.append(_prop_row(f, int((w["status"] == "flag").sum()), len(w)))
    return pd.DataFrame(rows).rename(columns={"label": "field", "n": "n_wrong_docs", "correct": "flagged",
                                              "accuracy": "recall"})


def reason_code_table(scores: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Per reason code: how often it fired and how often those documents were truly wrong."""
    m = scores[["doc_id", "all_correct"]].merge(validation, on="doc_id", validate="one_to_one")
    rows = []
    for code in sorted({c for codes in m["reason_codes"] for c in codes}):
        sel = m[m["reason_codes"].map(lambda cs: code in cs)]
        rows.append({"reason_code": code, "n_flagged": len(sel), "n_truly_wrong": int((~sel["all_correct"]).sum()),
                     "precision": float((~sel["all_correct"]).mean())})
    return pd.DataFrame(rows, columns=["reason_code", "n_flagged", "n_truly_wrong", "precision"])


# ----------------------------------------------------------------------------- error taxonomy
CATEGORY_DESCRIPTIONS = {
    "extraction_failure": "no usable extraction (API failure, refusal, or unparseable output after retries)",
    "missing_value": "field null/absent although the document has a value",
    "spurious_value": "value given for a field that is null in the document (e.g. computed rate, invented due date)",
    "sign_error": "right magnitude, wrong sign (typically credit-note amounts or the discount sign)",
    "tax_inclusive_confusion": "subtotal/total/tax wrong on a tax-inclusive document (net vs gross basis)",
    "magnitude_error": "off by a power of ten (typical of European vs US number-format misreads)",
    "wrong_number": "numeric field wrong for another reason",
    "unparseable_number": "numeric field is not a number",
    "day_month_swap": "date has day and month transposed",
    "unparseable_date": "date is not a valid date",
    "wrong_date": "date wrong for another reason",
    "vendor_name_mismatch": "vendor name differs after normalization",
    "invoice_number_mismatch": "invoice number differs after normalization",
    "currency_mismatch": "currency code wrong or unsupported",
    "document_type_mismatch": "invoice vs credit note misclassified",
    "dropped_line_item": "fewer line items than the document has",
    "extra_line_item": "more line items than the document has",
    "description_truncated": "line-item description cut short (e.g. second line lost)",
    "description_mismatch": "line-item description differs",
    "line_item_value_error": "a line item's quantity, unit price or amount is wrong",
    "line_items_malformed": "line_items is not a list of objects",
}


def _rel_close(a: float, b: float, rel: float = 0.02) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b))


def classify_scalar_error(field: str, pred, truth, features: dict) -> str:
    """Deterministic category for ONE wrong scalar field (call only when field_correct is False)."""
    if field in HEADER_NUM_FIELDS:
        p = parse_number(pred)
        if pred is None:
            return "missing_value"
        if truth is None:
            return "spurious_value"
        if p is None:
            return "unparseable_number"
        if abs(truth) > 0.01 and abs(p + truth) <= 0.01 + 1e-9:
            return "sign_error"
        if features.get("tax_inclusive") and field in ("subtotal", "total", "tax_amount"):
            return "tax_inclusive_confusion"
        if p != 0 and truth != 0 and any(_rel_close(abs(p / truth), k) or _rel_close(abs(truth / p), k)
                                         for k in (10, 100, 1000)):
            return "magnitude_error"
        return "wrong_number"
    if field in ("invoice_date", "due_date"):
        if pred is None:
            return "missing_value"
        if truth is None:
            return "spurious_value"
        n = normalize_date(pred)
        if n is None:
            return "unparseable_date"
        y, mth, d = truth.split("-")
        if int(d) <= 12 and n == f"{y}-{d}-{mth}":
            return "day_month_swap"
        return "wrong_date"
    if pred is None:
        return "missing_value"
    if truth is None:
        return "spurious_value"
    return {"vendor_name": "vendor_name_mismatch", "invoice_number": "invoice_number_mismatch",
            "currency": "currency_mismatch", "document_type": "document_type_mismatch"}[field]


def classify_line_items_error(pred, truth) -> tuple[str, str]:
    """(category, short detail) for a wrong line_items field."""
    if not isinstance(pred, list) or not all(isinstance(i, dict) for i in pred):
        return "line_items_malformed", "not a list of objects"
    if len(pred) < len(truth):
        return "dropped_line_item", f"{len(pred)} items vs {len(truth)} expected"
    if len(pred) > len(truth):
        return "extra_line_item", f"{len(pred)} items vs {len(truth)} expected"
    pairs = list(zip(sorted(pred, key=_line_key), sorted(truth, key=_line_key)))
    for p, t in pairs:
        pn, tn = normalize_name(p.get("description")) or "", normalize_name(t.get("description")) or ""
        if pn != tn:
            cat = "description_truncated" if pn and tn.startswith(pn) else "description_mismatch"
            return cat, f"{p.get('description')!r} vs {t.get('description')!r}"
    for p, t in pairs:
        for k in ("quantity", "unit_price", "amount"):
            if not numbers_match(p.get(k), t.get(k)):
                return "line_item_value_error", f"{t['description']!r}: {k} {p.get(k)} vs {t.get(k)}"
    return "line_item_value_error", "values differ"


def _short(v) -> str:
    s = repr(v)
    return s if len(s) <= 60 else s[:57] + "..."


# ----------------------------------------------------------------------------- POST-HOC: lenient-description metric
#
# Added after seeing test-split results (the zero-shot error taxonomy showed several line-item
# "description_mismatch" entries that were punctuation/whitespace-only, e.g. a missing comma). This
# section does not change field_correct, line_item_correct, score_document or score_run: it duplicates
# just enough logic under a new name so the primary metric is provably untouched (see
# tests/test_lenient_metric.py::test_primary_metric_unchanged). No edit distance, no reordering beyond
# the primary metric's own order-insensitive item matching, no synonym handling: only punctuation,
# whitespace and case are ignored in descriptions. A changed WORD still fails this metric too.
def lenient_line_item_correct(pred, truth) -> bool:
    """Same as line_item_correct, except descriptions are compared with punctuation/whitespace/case
    stripped instead of normalize_name's whitespace-collapse-only comparison."""
    if not isinstance(pred, list) or len(pred) != len(truth) or not all(isinstance(i, dict) for i in pred):
        return False
    for p, t in zip(sorted(pred, key=_line_key), sorted(truth, key=_line_key)):
        if description_key(p.get("description")) != description_key(t.get("description")):
            return False
        if not all(numbers_match(p.get(k), t.get(k)) for k in LINE_ITEM_FIELDS if k != "description"):
            return False
    return True


def lenient_score_document(pred: dict | None, truth: dict) -> dict[str, bool]:
    """Like score_document, but line_items uses lenient_line_item_correct. Every other field is scored
    identically to the primary metric (same function, same result)."""
    out = {f: score_document(pred, truth)[f] for f in ALL_FIELDS if f != "line_items"}
    out["line_items"] = (False if not isinstance(pred, dict) or "line_items" not in pred
                         else bool(lenient_line_item_correct(pred["line_items"], truth["line_items"])))
    out["all_correct"] = all(out[f] for f in ALL_FIELDS)
    return out


def lenient_score_run(extractions: dict, records: list[dict], split: str = "test") -> pd.DataFrame:
    """Like score_run, but using lenient_score_document. Same columns, same metadata, same split guard."""
    bad = [r["doc_id"] for r in records if r["split"] != split]
    if bad:
        raise ValueError(f"lenient_score_run is restricted to the {split!r} split; got {bad[:3]}...")
    rows = []
    for r in sorted(records, key=lambda r: r["doc_id"]):
        pred = _extraction_of(extractions.get(r["doc_id"]))
        sc = lenient_score_document(pred, r["truth"])
        m, f = r["meta"], r["meta"]["features"]
        feats = {("doc_type" if c == "document_type" else c): f[c] for c in FEATURE_COLUMNS}
        rows.append({"doc_id": r["doc_id"], **{c: m[c] for c in META_COLUMNS}, **feats,
                     "failed": not isinstance(pred, dict), **sc})
    return pd.DataFrame(rows)


def error_records(strategy: str, extractions: dict, records: list[dict],
                  validation: pd.DataFrame | None = None) -> list[dict]:
    """One row per wrong field with a taxonomy category (plus one 'extraction_failure' row per failed doc)."""
    vmap = {} if validation is None else validation.set_index("doc_id").to_dict("index")
    out = []
    for r in sorted(records, key=lambda r: r["doc_id"]):
        pred = _extraction_of(extractions.get(r["doc_id"]))
        v = vmap.get(r["doc_id"], {})
        base = {"strategy": strategy, "doc_id": r["doc_id"], "template": r["meta"]["template"],
                "noise_level": r["meta"]["noise_level"], "hard_reason": r["meta"]["hard_reason"],
                "flagged": v.get("status") == "flag", "validator_codes": ", ".join(v.get("reason_codes", []))}
        if not isinstance(pred, dict):
            out.append({**base, "field": "(all)", "category": "extraction_failure", "predicted": "none",
                        "expected": "-", "detail": ""})
            continue
        sc = score_document(pred, r["truth"])
        for f in ALL_FIELDS:
            if sc[f]:
                continue
            if f == "line_items":
                cat, detail = classify_line_items_error(pred.get(f), r["truth"][f])
                got = f"{len(pred[f])} items" if isinstance(pred.get(f), list) else _short(pred.get(f))
                out.append({**base, "field": f, "category": cat, "predicted": got,
                            "expected": f"{len(r['truth'][f])} items", "detail": detail})
            else:
                cat = classify_scalar_error(f, pred.get(f), r["truth"][f], r["meta"]["features"])
                out.append({**base, "field": f, "category": cat, "predicted": _short(pred.get(f)),
                            "expected": _short(r["truth"][f]), "detail": ""})
    return out
