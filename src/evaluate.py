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

from .normalize import (normalize_currency, normalize_date, normalize_doc_type, normalize_invoice_number,
                        normalize_name, numbers_match, parse_number)
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
        rows.append({"doc_id": r["doc_id"], **{c: m[c] for c in META_COLUMNS},
                     **{c: f[c] for c in FEATURE_COLUMNS}, "failed": not isinstance(pred, dict), **sc})
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
