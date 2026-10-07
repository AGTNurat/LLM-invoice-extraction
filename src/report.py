"""Generate results/summary.md and results/error_analysis.md from scored runs.

Every number in these files is computed here from extraction results and ground truth, nothing is
typed by hand. Output is deterministic (no timestamps) so identical inputs give identical files.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .evaluate import (CATEGORY_DESCRIPTIONS, _extraction_of, accuracy_table, breakdown, error_records,
                       evaluate_validation, lenient_score_run, paired_comparison, reason_code_table, score_run,
                       strategy_overview, validation_recall_by_field, wilson_interval)
from .schema import ALL_FIELDS
from .validate import validate_batch

NOISE_NAMES = {0: "none", 1: "low", 2: "medium"}
EXAMPLES_PER_CATEGORY = 3


#  formatting helpers
def pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.1f}%"


def rate_cell(k: int, n: int) -> str:
    if n == 0:
        return "n/a (0 cases)"
    lo, hi = wilson_interval(k, n)
    return f"{k}/{n} = {pct(k / n)} [{pct(lo)}–{pct(hi)}]"


def row_cell(row: dict) -> str:
    return rate_cell(int(row["correct"]), int(row["n"]))


def _cell(v) -> str:
    return str(v).replace("|", "\\|").replace("\n", " ")


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# evaluation bundle
def evaluate_runs(records: list[dict], runs: dict[str, dict], split: str = "test") -> dict:
    """Score every strategy on ``records`` (test split by default) and run validation + error classification."""
    out = {}
    for name, results in runs.items():
        extractions = {r["doc_id"]: _extraction_of(results.get(r["doc_id"])) for r in records}
        scores = score_run(results, records, split=split)
        validation = validate_batch(extractions)
        out[name] = {"scores": scores, "validation": validation, "results": results,
                     "errors": error_records(name, results, records, validation),
                     "lenient_scores": lenient_score_run(results, records, split=split)}
    return out


def _banner(meta: dict) -> str:
    out = ""
    if meta.get("mock"):
        out += ("> **MOCK RUN — NOT MODEL RESULTS.** These numbers come from the offline mock extractor, which "
                "perturbs ground truth with seeded errors. They only demonstrate the pipeline.\n")
    if meta.get("split", "test") != "test":
        out += (f"> **{meta['split'].upper()} SPLIT — used for prompt development, NOT a held-out result.** "
                "Do not report these numbers.\n")
    return out


def _usage_rows(ev: dict) -> list[list]:
    rows = []
    for name, e in ev.items():
        res = list(e["results"].values())
        rows.append([name, len(res), sum(1 for r in res if getattr(r, "cached", False)),
                     sum(getattr(r, "usage", {}).get("input_tokens", 0) for r in res),
                     sum(getattr(r, "usage", {}).get("output_tokens", 0) for r in res),
                     sum(1 for r in res if getattr(r, "attempts", 1) > 1),
                     sum(1 for r in res if getattr(r, "temperature_applied", None) is False)])
    return rows


#  summary.md
def build_summary(records: list[dict], ev: dict, meta: dict) -> str:
    names = list(ev)
    n = len(records)
    scores = {s: ev[s]["scores"] for s in names}
    L = ["# Evaluation summary", "", _banner(meta)]
    L += [f"- Model: `{meta.get('model', 'unknown')}`",
          f"- Split: **{meta.get('split', 'test')}** ({n} documents); strategies: {', '.join(names)}",
          f"- Dataset seed: {meta.get('seed', 'n/a')}",
          "- All data is synthetic and labels are exact by construction; see README limitations.",
          f"- Intervals are 95% Wilson score intervals; with n={n} they are wide, so small differences are "
          "not statistically meaningful."]
    L += [f"- {note}" for note in meta.get("notes", [])]
    L += [""]

    ov = strategy_overview(scores)
    L += ["## 1. Strategy comparison", "",
          md_table(["Strategy", "All fields correct", "Pooled field accuracy", "Failed extractions"],
                   [[r.strategy, rate_cell(r.correct, r.n), pct(r.pooled_field_accuracy), r.failed_extractions]
                    for r in ov.itertuples()]), ""]
    if len(names) > 1:
        pc = paired_comparison(scores)
        L += ["Paired comparison on the same documents (exact McNemar test on all-fields-correct; "
              "'only X' = documents that X got fully right and the other did not):", "",
              md_table(["A", "B", "both correct", "only A", "only B", "both wrong", "McNemar p"],
                       [[r.a, r.b, r.both_correct, r.only_a_correct, r.only_b_correct, r.both_wrong,
                         f"{r.mcnemar_p:.3f}"] for r in pc.itertuples()]), ""]

    tabs = {s: accuracy_table(scores[s]).set_index("field") for s in names}
    fields = list(ALL_FIELDS)
    L += ["## 2. Per-field accuracy", "",
          md_table(["Field"] + names, [[f] + [row_cell(tabs[s].loc[f]) for s in names] for f in fields]), ""]

    L += ["## 3. Accuracy by subgroup (all fields correct)", ""]
    specs = [("template", "Layout template", {}), ("noise_level", "Noise level", NOISE_NAMES),
             ("hard", "Hard cases", {True: "hard", False: "regular"}),
             ("doc_type", "Document type", {}), ("number_format", "Number format", {}),
             ("tax_inclusive", "Tax-inclusive", {True: "inclusive", False: "exclusive"})]
    for col, title, labels in specs:
        per = {s: breakdown(scores[s], col).set_index(col) for s in names}
        groups = sorted(set().union(*[set(p.index) for p in per.values()]), key=lambda g: str(g))
        L += [f"### {title}", "",
              md_table([title] + names, [[labels.get(g, g)] + [row_cell(per[s].loc[g]) if g in per[s].index else "n/a"
                                                                  for s in names] for g in groups]), ""]

    L += ["## 4. Validation layer", "",
          "'Wrong' = not all fields correct (failed extractions count as wrong); 'flagged' = at least one "
          "deterministic check failed. Straight-through = correct AND unflagged, as a share of all documents.", ""]
    vals = {s: evaluate_validation(scores[s], ev[s]["validation"]) for s in names}
    L += [md_table(["Strategy", "Wrong", "Flagged", "Recall (wrong that were flagged)",
                    "Precision (flags that were wrong)", "Straight-through", "Wrong among unflagged",
                    "Correct but flagged"],
                   [[s, v["n_wrong"], v["n_flagged"], row_cell(v["recall"]), row_cell(v["precision"]),
                     row_cell(v["straight_through_rate"]), row_cell(v["error_rate_unflagged"]),
                     row_cell(v["false_flag_rate"])] for s, v in vals.items()]), ""]
    for s in names:
        rf = validation_recall_by_field(scores[s], ev[s]["validation"])
        rc = reason_code_table(scores[s], ev[s]["validation"])
        L += [f"### Validation detail: {s}", ""]
        rows = [[r.field, r.n_wrong_docs, r.flagged, "n/a" if r.n_wrong_docs == 0 else pct(r.recall)]
                for r in rf.itertuples() if r.n_wrong_docs > 0]
        L += (["Recall by wrong field (a document counts as flagged if ANY check fired, "
               "not necessarily one about that field):", "",
               md_table(["Wrong field", "docs with this field wrong", "of which flagged", "recall"], rows), ""]
              if rows else ["No wrong fields for this strategy.", ""])
        L += ([md_table(["Reason code", "docs flagged", "of which truly wrong", "precision"],
                        [[r.reason_code, r.n_flagged, r.n_truly_wrong, pct(r.precision)] for r in rc.itertuples()]), ""]
              if len(rc) else ["No document was flagged.", ""])

    L += ["## 5. Usage", "",
          md_table(["Strategy", "documents", "served from cache", "input tokens", "output tokens",
                    "docs needing retries", "docs where temperature=0 was NOT applied"], _usage_rows(ev)), "",
          "Token counts are those recorded when each response was first fetched. If any document shows "
          "temperature=0 was not applied, the model rejected that setting and ran at its default.", ""]
    L += build_posthoc_lenient_section(ev, names)
    return "\n".join(L).rstrip() + "\n"


def build_posthoc_lenient_section(ev: dict, names: list[str]) -> list[str]:
    """POST-HOC: 'all fields correct (lenient descriptions)' secondary metric. Defined AFTER seeing the
    primary test-split results (the zero-shot error taxonomy showed punctuation/whitespace-only
    description mismatches; see results/description_audit.md). Primary tables above are unchanged."""
    lscores = {s: ev[s]["lenient_scores"] for s in names}
    L = ["## 6. Post-hoc secondary metric: lenient line-item descriptions", "",
         "> **Post-hoc.** This metric and the rule below were defined after seeing the primary test-split "
         "results, specifically after `results/description_audit.md` showed zero-shot line-item "
         "description mismatches that were only punctuation/whitespace/case (e.g. a missing comma). It "
         "is reported separately and does not change any number in sections 1-5 above.", "",
         "**Rule:** line-item descriptions are compared after casefolding and removing all punctuation "
         "and whitespace. Nothing fuzzier: no edit distance, no word reordering, no synonym handling. "
         "Every other field, and quantity/unit_price/amount, are compared exactly as in the primary "
         "metric. A document with a word-level description difference is still wrong under this metric.",
         "", "### Primary vs. lenient, all-fields-correct", "",
         md_table(["Strategy", "Primary (all fields correct)", "Lenient (all fields correct)"],
                  [[s, rate_cell(int(ev[s]["scores"]["all_correct"].sum()), len(ev[s]["scores"])),
                    rate_cell(int(lscores[s]["all_correct"].sum()), len(lscores[s]))]
                   for s in names]), ""]
    if len(names) > 1:
        pc = paired_comparison(lscores)
        L += ["Paired comparison under the LENIENT metric (exact McNemar test on all-fields-correct):", "",
              md_table(["A", "B", "both correct", "only A", "only B", "both wrong", "McNemar p"],
                       [[r.a, r.b, r.both_correct, r.only_a_correct, r.only_b_correct, r.both_wrong,
                         f"{r.mcnemar_p:.3f}"] for r in pc.itertuples()]), ""]
    L += ["### Validation layer under the lenient metric", "",
          "'Wrong' is redefined using the lenient all-fields-correct; validator behaviour (flags) is "
          "unchanged — only which documents count as wrong changes.", ""]
    lvals = {s: evaluate_validation(lscores[s], ev[s]["validation"]) for s in names}
    L += [md_table(["Strategy", "Wrong (lenient)", "Flagged", "Recall", "Precision", "Straight-through",
                    "Wrong among unflagged", "Correct but flagged"],
                   [[s, v["n_wrong"], v["n_flagged"], row_cell(v["recall"]), row_cell(v["precision"]),
                     row_cell(v["straight_through_rate"]), row_cell(v["error_rate_unflagged"]),
                     row_cell(v["false_flag_rate"])] for s, v in lvals.items()]), ""]
    return L


#  error_analysis.md
def build_error_analysis(records: list[dict], ev: dict, meta: dict) -> str:
    n = len(records)
    L = ["# Error analysis", "", _banner(meta),
         "Failures are classified deterministically from prediction vs ground truth (`src/evaluate.py`, "
         "`classify_*`). A document can contribute several wrong fields; a single root cause (e.g. a wrong "
         "document type) often cascades into several categories.", ""]
    for name, e in ev.items():
        errs = e["errors"]
        wrong_docs = {x["doc_id"] for x in errs}
        L += [f"## Strategy: {name}", "",
              f"{len(wrong_docs)} of {n} documents contain at least one wrong field ({len(errs)} wrong fields in total).",
              ""]
        if not errs:
            L += ["No errors.", ""]
            continue
        cats = Counter(x["category"] for x in errs)
        docs_by_cat = defaultdict(set)
        flagged_by_cat = defaultdict(set)
        for x in errs:
            docs_by_cat[x["category"]].add(x["doc_id"])
            if x["flagged"]:
                flagged_by_cat[x["category"]].add(x["doc_id"])
        order = sorted(cats, key=lambda c: (-cats[c], c))
        L += ["### Taxonomy", "",
              md_table(["Category", "Meaning", "wrong fields", "documents", "documents flagged by validator"],
                       [[c, CATEGORY_DESCRIPTIONS.get(c, ""), cats[c], len(docs_by_cat[c]),
                         len(flagged_by_cat[c])] for c in order]), ""]
        L += ["### Examples", ""]
        for c in order:
            ex = [x for x in errs if x["category"] == c][:EXAMPLES_PER_CATEGORY]
            L += [f"**{c}**", "",
                  md_table(["doc", "template", "noise", "hard case", "field", "predicted", "expected", "detail",
                            "validator"],
                           [[x["doc_id"], x["template"], NOISE_NAMES.get(x["noise_level"], x["noise_level"]),
                             x["hard_reason"] or "-", x["field"], x["predicted"], x["expected"], x["detail"] or "-",
                             x["validator_codes"] or "not flagged"] for x in ex]), ""]
        by_tpl = Counter(x["template"] for x in {(y["doc_id"], y["template"]): y for y in errs}.values())
        L += ["### Wrong documents by template", "",
              md_table(["template", "documents with an error"], [[t, k] for t, k in sorted(by_tpl.items())]), ""]

    if len(ev) > 1:
        wrong_sets = [{x["doc_id"] for x in e["errors"]} for e in ev.values()]
        always = sorted(set.intersection(*wrong_sets))
        L += ["## Documents wrong under every strategy", ""]
        if always:
            rec = {r["doc_id"]: r for r in records}
            first = next(iter(ev.values()))["errors"]
            L += [md_table(["doc", "template", "noise", "hard case", "categories (first strategy)"],
                           [[d, rec[d]["meta"]["template"], NOISE_NAMES.get(rec[d]["meta"]["noise_level"]),
                             rec[d]["meta"]["hard_reason"] or "-",
                             ", ".join(sorted({x["category"] for x in first if x["doc_id"] == d}))]
                            for d in always]), ""]
        else:
            L += ["None: every document was handled correctly by at least one strategy.", ""]

    L += ["## What the validation layer missed", "",
          "Wrong fields by category, split by whether the document was flagged (all strategies pooled). "
          "Categories that are mostly unflagged are errors the deterministic checks cannot see.", ""]
    pooled = [x for e in ev.values() for x in e["errors"]]
    if pooled:
        cnt = Counter((x["category"], x["flagged"]) for x in pooled)
        cats = sorted({c for c, _ in cnt}, key=lambda c: (-(cnt[(c, True)] + cnt[(c, False)]), c))
        L += [md_table(["Category", "in flagged documents", "in unflagged documents"],
                       [[c, cnt[(c, True)], cnt[(c, False)]] for c in cats]), ""]
    else:
        L += ["No errors to analyse.", ""]
    return "\n".join(L).rstrip() + "\n"


def write_reports(out_dir: str | Path, records: list[dict], runs: dict[str, dict], meta: dict) -> tuple[Path, Path]:
    """Evaluate and write results/summary.md and results/error_analysis.md. Returns the two paths."""
    ev = evaluate_runs(records, runs, split=meta.get("split", "test"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sp, ep = out / "summary.md", out / "error_analysis.md"
    sp.write_text(build_summary(records, ev, meta), encoding="utf-8")
    ep.write_text(build_error_analysis(records, ev, meta), encoding="utf-8")
    return sp, ep
