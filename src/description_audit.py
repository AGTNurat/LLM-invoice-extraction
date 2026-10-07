"""POST-HOC audit of zero-shot/few-shot/rules line-item DESCRIPTION mismatches on the (already-scored)
test split, reading only cached extraction results already on disk. Makes no API calls.

Writes results/description_audit.md: every document/item where line_items was wrong wholly or partly
because of the description text, with a character-level diff and a class label:
  "punctuation/whitespace/case only"  vs  "word-level difference"

This module only reads and reports; it does not change the primary metric. See evaluate.lenient_*
for the post-hoc secondary metric that uses the same punctuation/whitespace-insensitive comparison.
"""
from __future__ import annotations

import difflib
from pathlib import Path

from .evaluate import _extraction_of, _line_key
from .normalize import description_key, normalize_name, numbers_match
from .schema import LINE_ITEM_FIELDS


def classify_description_diff(pred: str, truth: str) -> str:
    """'punctuation/whitespace/case only' if the two are identical once punctuation/whitespace/case are
    stripped, else 'word-level difference'."""
    return ("punctuation/whitespace/case only" if description_key(pred) == description_key(truth)
            else "word-level difference")


def char_diff(pred: str, truth: str) -> str:
    """Compact inline diff markup: [-removed-]{+added+}, built from truth -> pred."""
    sm = difflib.SequenceMatcher(None, truth or "", pred or "", autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.append(truth[i1:i2])
        else:
            if truth[i1:i2]:
                out.append(f"[-{truth[i1:i2]}-]")
            if pred[j1:j2]:
                out.append(f"{{+{pred[j1:j2]}+}}")
    return "".join(out)


def audit_strategy(strategy: str, extractions: dict, records: list[dict]) -> list[dict]:
    """One row per line item whose description differs (after the PRIMARY normalize_name comparison),
    for documents where line_items as a whole was scored wrong. Rows are sorted by doc_id then item index
    for a stable, reviewable table.
    """
    rows = []
    for r in sorted(records, key=lambda r: r["doc_id"]):
        pred = _extraction_of(extractions.get(r["doc_id"]))
        truth_items = r["truth"]["line_items"]
        if not isinstance(pred, dict):
            continue
        pred_items = pred.get("line_items")
        if not isinstance(pred_items, list) or len(pred_items) != len(truth_items):
            continue  # count mismatches are a different error category; not a description issue
        if not all(isinstance(i, dict) for i in pred_items):
            continue
        pairs = list(zip(sorted(pred_items, key=_line_key), sorted(truth_items, key=_line_key)))
        # only report documents where line_items as scored (primary metric) is actually wrong
        line_items_wrong = any(
            normalize_name(p.get("description")) != normalize_name(t.get("description"))
            or not all(numbers_match(p.get(k), t.get(k)) for k in LINE_ITEM_FIELDS if k != "description")
            for p, t in pairs)
        if not line_items_wrong:
            continue
        for idx, (p, t) in enumerate(pairs):
            pd_, td = p.get("description"), t.get("description")
            if normalize_name(pd_) == normalize_name(td):
                continue  # this item's description matched under the primary metric; not a desc issue
            rows.append({"strategy": strategy, "doc_id": r["doc_id"], "item_index": idx,
                        "predicted": pd_, "expected": td, "diff": char_diff(pd_ or "", td or ""),
                        "class": classify_description_diff(pd_, td)})
    return rows


def build_audit_report(rows_by_strategy: dict[str, list[dict]]) -> str:
    L = ["# Description mismatch audit (post-hoc)", "",
         "> **Post-hoc.** Written after seeing test-split results, from the cached extractions already "
         "scored in `results/summary.md`. No API calls were made and no prompt, generator, seed or cached "
         "response was changed to produce this file.", "",
         "Every line item whose description differs under the PRIMARY metric (`src/evaluate.py: "
         "normalize_name`, i.e. whitespace-collapsed, case-folded) on a document where `line_items` was "
         "scored wrong. `class` additionally strips all punctuation: "
         "\"punctuation/whitespace/case only\" means the only difference is punctuation, whitespace or "
         "case (e.g. a missing comma); \"word-level difference\" means the words themselves differ.", ""]
    for strategy, rows in rows_by_strategy.items():
        L += [f"## {strategy}", ""]
        if not rows:
            L += ["No description mismatches.", ""]
            continue
        n_punct = sum(1 for r in rows if r["class"] == "punctuation/whitespace/case only")
        L += [f"{len(rows)} mismatched line-item description(s): {n_punct} punctuation/whitespace/case only, "
              f"{len(rows) - n_punct} word-level.", "",
              "| doc | item | predicted | expected | diff (truth -> pred) | class |",
              "|---|---|---|---|---|---|"]
        for r in rows:
            cells = [r["doc_id"], r["item_index"], r["predicted"], r["expected"], r["diff"], r["class"]]
            L.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
        L.append("")
    return "\n".join(L).rstrip() + "\n"


def write_audit(out_path: str | Path, extractions_by_strategy: dict[str, dict], records: list[dict]) -> Path:
    rows_by_strategy = {s: audit_strategy(s, ext, records) for s, ext in extractions_by_strategy.items()}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_audit_report(rows_by_strategy), encoding="utf-8")
    return out
