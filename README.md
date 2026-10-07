# LLM Based Invoice Extraction & Evaluation

LLM-based extraction of structured data from semi-structured vendor invoices, with deterministic
validation checks and a reproducible accuracy evaluation.

> **Status:** a live evaluation has been run (model `claude-haiku-4-5`, test split, all three strategies,
> scored once each — see [Results](#results)). Two post-hoc analyses were added afterwards, both computed
> from the cached results of that run with no new API calls (see [Post-hoc analyses](#post-hoc-analyses)).

## Problem

Finance operations teams receive invoices in many layouts. Keying vendor, dates, line items and totals by
hand is slow and error-prone. This project asks three questions:

1. How accurately can an LLM extract these fields, and how much does the prompting strategy matter?
2. Can cheap deterministic checks (do the numbers add up?) catch the model's mistakes?
3. What share of invoices could go straight through with no human review?

## Design

```
generate.py ──► PDFs + ground-truth JSON (dev 20 / test 40)
                       │ pdfplumber text
                       ▼
extract.py ──► LLM (structured output) ──► extraction JSON   (disk cache, retries)
                       │
          ┌────────────┴─────────────┐
          ▼                          ▼
   validate.py (pass/flag)     evaluate.py (vs ground truth, test split only)
          └────────────┬─────────────┘
                       ▼
        report.py ──► results/summary.md, results/error_analysis.md
```

Key decisions:

- **Exact labels by construction.** The generator decides what to print and what the label is from the same
  seeded plan; no LLM ever produces a label. A test extracts the text from every generated PDF and checks
  that each labelled value is actually findable in it.
- **Dev/test discipline.** Few-shot examples come only from the dev split (the code raises otherwise).
  Prompts are developed on dev; the test split is scored once per strategy. The CLI enforces this: each
  test run is logged with a fingerprint of the prompt in `results/run_log.json`, a re-run after a prompt
  change is refused unless `--allow-test-rerun` is passed, and the report then carries a warning.
- **Reproducibility.** Everything is seeded (default seed 42); generated PDFs are byte-identical across
  runs. Every API response is cached on disk keyed by `sha256(model, strategy, document text, prompt
  fingerprint)`, so reruns cost nothing and editing a prompt invalidates stale entries. Only successes
  are cached.
- **No hardcoded model or secrets.** The model comes from `--model` or `ANTHROPIC_MODEL`; the key is read
  by the SDK from `ANTHROPIC_API_KEY` only.
- **Honest scoring.** A failed or missing extraction counts as wrong on every field (it is never dropped
  from the denominator). Confidence intervals are Wilson 95% intervals. Strategies are compared on the same
  documents, so a paired exact McNemar test is reported alongside the intervals.

### Synthetic data (`src/generate.py`)

60 documents: 5 layout templates x 12 slots, 4 slots per template in dev (20 docs) and 8 in test (40 docs).

| Variation | Values |
|---|---|
| Layout | classic table, stacked meta, totals-first, freeform lines, sidebar |
| Number format | US `1,234.56`, European `1.234,56` |
| Dates | ISO, `15/03/2024`, `03/15/2024`, `15.03.2024`, `15 Mar 2024`, `March 15, 2024` |
| Totals | tax-exclusive, tax-inclusive (gross line prices) |
| Other | discounts, multi-line descriptions, reordered header fields, credit notes (negative amounts, sometimes in parentheses) |
| Noise level | 0 none; 1 low (distractor lines, boilerplate); 2 medium (adds OCR-style swaps in labels such as `Tota1`, scan artifacts, stamps) |

Noise touches labels and boilerplate only, never field values. Six clearly labelled **hard cases** are included
(two in dev, four in test): an ambiguous numeric date, a two-page invoice, a credit note with parenthesised
negatives in European format, fractional quantities with multi-line descriptions, a tax amount printed
without a rate, and lookalike amounts (previous balance, deposit, credit limit). Template, noise level, hard-case
label and features are recorded per document so accuracy can be broken down by them.

Labelling conventions (`src/schema.py`): dates are ISO; `tax_rate` is a percentage (null if not printed); on
credit notes `amount`, `subtotal`, `discount`, `tax_amount` and `total` are negative; on tax-inclusive documents
line amounts stay as printed (gross) and `subtotal = total - tax_amount`.

### Extraction (`src/extract.py`)

PDF text is extracted with pdfplumber and sent to the model with a JSON-schema structured-output format.
Three prompt strategies, all using the same schema:

- **zero_shot**: minimal instructions.
- **few_shot**: zero-shot plus three worked examples, chosen deterministically from non-hard dev documents.
- **rules**: zero-shot plus explicit rules on number formats, date conventions, credit-note signs,
  tax-inclusive totals, the discount sign, and distractors to ignore.

Transient API errors are retried with exponential backoff and jitter; malformed or truncated output is retried;
refusals and invalid-request errors are not. A `MockExtractor` (seeded perturbations of ground truth) lets the
whole pipeline and test suite run offline. Mock output is never a model result and reports built from it carry a
banner.

### Validation (`src/validate.py`)

Deterministic checks on the extraction alone (no ground truth): required fields present; numbers parse; dates are
valid ISO and `due_date >= invoice_date`; currency and document type valid; amount signs consistent with the
document type; each line `amount == quantity x unit_price`; line items sum to the subtotal (or to the total on a
tax-inclusive basis); `subtotal - discount + tax == total`; tax consistent with the rate (exclusive or inclusive
basis); duplicate vendor + invoice number across a batch. Each invoice gets `pass` or `flag` with reason codes.

### Evaluation (`src/evaluate.py`, `src/report.py`)

Per-field accuracy and the invoice-level "all fields correct" rate (both with Wilson intervals), breakdowns by
template, noise level, hard cases, document type, number format and tax basis, and a three-strategy comparison.
`line_items` is one composite field: correct only if the item count and every description, quantity, unit price
and amount match (order ignored). Validation-layer metrics: **recall** (share of wrong extractions that were
flagged), **precision** (share of flags that were truly wrong), **straight-through rate** (correct and unflagged,
as a share of all documents), plus the false-flag rate and the error rate among unflagged documents. Failures are
classified by deterministic rules into a taxonomy (sign errors, day/month swaps, dropped line items, and so on).

## How to run
Commands for Powershell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest                                   # offline: no API calls, no key needed
```

Offline pipeline check (mock extractor, never real results):

```powershell
python -m src.cli generate
python -m src.cli run --mock --split test --yes
python -m src.cli report --mock          # writes outputs/mock_report/, not results/
```

Live run. Set credentials in your shell only (never in a committed file; `.env` is git-ignored but the code does
not read it):

```powershell
$env:ANTHROPIC_API_KEY = "<your key>"
$env:ANTHROPIC_MODEL   = "<model id>"          # or pass --model on each command

python -m src.cli run --split dev --strategy all --dry-run   # cost estimate only
python -m src.cli run --split dev --strategy all             # prompts the cost estimate + confirmation
# ...inspect dev results, adjust prompts if needed (dev only)...
python -m src.cli run --split test --strategy all            # the reported run, once per strategy
python -m src.cli report --split test                        # writes results/summary.md and error_analysis.md

# Post-hoc (read the cache above; make no new API calls):
python -m src.cli description-audit --split test             # writes results/description_audit.md
python -m src.cli stress-validator                            # writes results/validator_stress_test.md
```

A live run prints an estimated token count and cost and asks for confirmation first (`--yes` skips the prompt).
The price table in `src/cost.py` is a dated reference and may be stale; override with `--price-in/--price-out`.
The estimate excludes thinking tokens, so treat it as a lower bound. Temperature 0 is requested through
`extra_body`; if the model rejects it, the call is retried without it and the summary's usage table reports how
many documents ran without it.

### Post-hoc analyses (`src/description_audit.py`, `src/stress_validator.py`)

Added after the live run above, reading only its cached results; neither makes an API call or changes a
primary number. `description_audit.py` compares every wrong line-item description under the primary metric
(`normalize_name`) against a stricter, punctuation/whitespace/case-stripped key and reports a character-level
diff; `evaluate.py`'s `lenient_*` functions reuse `score_document` for every field except `line_items`, so the
primary metric is provably unaffected (`tests/test_lenient_metric.py` asserts byte-identical primary output).
`stress_validator.py` applies one hand-declared corruption at a time to the ground-truth records and checks
the *existing* `validate.py` for a flag; it never imports validation logic to patch it, only to call it.

## Repository layout

```
src/schema.py              fields, conventions, JSON schema      src/validate.py    deterministic checks
src/normalize.py           number/date/name normalizers          src/evaluate.py    metrics, validation eval, taxonomy
src/generate.py            synthetic PDFs + ground truth         src/report.py      summary.md / error_analysis.md writers
src/extract.py             prompts, LLM call, cache, mock         src/cost.py        pre-run cost estimate
src/cli.py                 generate / run / report / ...          tests/             pytest suite (offline)
src/description_audit.py   post-hoc description-mismatch audit
src/stress_validator.py    post-hoc validator fault injection
```

## Results

Full tables: [results/summary.md](results/summary.md) · [results/error_analysis.md](results/error_analysis.md).
Every number below is copied from those files, which are themselves generated by `python -m src.cli report`
from the real extraction outputs in `outputs/claude-haiku-4-5/test/`. **The numbers below are a quoted excerpt,
not a separate claim** — `tests/test_readme_matches_results.py` parses `results/summary.md` and fails if this
section drifts from it.

**Run:** model `claude-haiku-4-5`, test split (40 documents), seed 42, each strategy scored once
(`results/run_log.json`). All confidence intervals below are 95% Wilson intervals; with n=40 they are wide
(roughly ±10-15 points), so small differences between strategies are not statistically meaningful — see the
paired McNemar column.

### Strategy comparison (all fields correct)

| Strategy | All fields correct | Pooled field accuracy | Failed extractions |
|---|---|---|---|
| zero_shot | 24/40 = 60.0% [44.6%–73.7%] | 91.7% | 0 |
| few_shot  | 30/40 = 75.0% [59.8%–85.8%] | 91.7% | 0 |
| rules     | 35/40 = 87.5% [73.9%–94.5%] | 99.0% | 0 |

Paired exact McNemar test (same 40 documents under each strategy): zero_shot vs. rules is the only pair that
reaches significance at the usual 0.05 threshold (p = 0.007, 13 documents only rules got right vs. 2 only
zero_shot did); zero_shot vs. few_shot (p = 0.146) and few_shot vs. rules (p = 0.125) are not significant at
this sample size.

### Where the strategies differed

Per-field accuracy ([results/summary.md §2](results/summary.md)) shows vendor_name, invoice_number,
invoice_date, due_date, currency and document_type at 100% for all three strategies — the model never missed
those. The gap between strategies is concentrated in `line_items` (62.5% zero_shot / 80.0% few_shot / 100%
rules) and the tax-inclusive-total arithmetic (`subtotal`, `tax_amount`, `total`), where the rules prompt's
explicit instructions on the tax-inclusive and credit-note conventions closed most of the gap.

### Validation layer

| Strategy | Wrong | Flagged | Recall | Precision | Straight-through |
|---|---|---|---|---|---|
| zero_shot | 16 | 8  | 8/16 = 50.0% [28.0%–72.0%]   | 8/8 = 100% [67.6%–100%]  | 24/40 = 60.0% [44.6%–73.7%] |
| few_shot  | 10 | 10 | 10/10 = 100% [72.2%–100%]    | 10/10 = 100% [72.2%–100%] | 30/40 = 75.0% [59.8%–85.8%] |
| rules     | 5  | 5  | 5/5 = 100% [56.6%–100%]      | 5/5 = 100% [56.6%–100%]  | 35/40 = 87.5% [73.9%–94.5%] |

For zero_shot, the validator caught exactly half of the wrong extractions (recall 50%) and never raised a
false alarm on a correct one (precision 100%, 0 correct-but-flagged in all three strategies). The 8 zero_shot
errors it missed were cases a deterministic arithmetic check cannot see by construction — see
[Post-hoc analyses](#post-hoc-analyses) below for a direct demonstration of that blind spot.

### Usage

| Strategy | input tokens | output tokens |
|---|---|---|
| zero_shot | 41,825 | 8,259 |
| few_shot  | 90,145 | 10,444 |
| rules     | 62,585 | 8,464 |

All 40 documents per strategy were live calls (0 served from cache on this run), 0 needed a retry, and
temperature=0 was accepted by the model on every call.

## Post-hoc analyses

Two analyses were added after the run above, from the already-cached results, with **no new API calls** and
no change to any prompt, the generator, the seed, or the primary numbers reported above. Each is clearly
labelled post-hoc in its own output file and in `results/summary.md`.

- **[results/description_audit.md](results/description_audit.md)** audits every zero_shot/few_shot/rules
  line-item description mismatch on the test split. Finding: **all 30 of zero_shot's description mismatches
  were punctuation-only** (it joined a multi-line description with `, ` or ` - ` where the generator used a
  plain space) — zero word-level mismatches anywhere. A labelled secondary metric,
  "all fields correct (lenient descriptions)" (results/summary.md §6), recomputes accuracy comparing
  descriptions with punctuation/whitespace/case stripped (no edit distance, no reordering, no synonyms — a
  changed word still fails). Under that metric zero_shot rises from 60.0% to 32/40 = 80.0%; few_shot and rules
  are unchanged (they had zero description mismatches to begin with).
- **[results/validator_stress_test.md](results/validator_stress_test.md)** fault-injects 16 corruption types
  (one character changed, one digit changed, a line dropped, a sign flipped, etc.) into the 60 seeded
  ground-truth records and checks whether the *existing, unmodified* validator notices. Every corruption
  type's observed detection matched its declared expectation exactly, with 0 false positives on uncorrupted
  records. The one deliberately designed to be invisible, `coordinated_price` (a unit price changed with
  amount/subtotal/tax/total all recomputed consistently), was undetected in 49/49 trials — demonstrating that
  a fully self-consistent wrong number is invisible to any check built purely on arithmetic identities.

## Limitations

- **The data is synthetic.** Documents come from five templates in a generator I wrote. Real invoices have
  scanned images, handwriting, stamps over text, unusual layouts and vendor-specific quirks that this does not
  capture. Accuracy here says little about accuracy on real invoices.
- **Labels are exact by construction.** That removes labelling noise, which is useful for measurement, but it
  also means there are no genuinely ambiguous or arguable labels as there would be in human-labelled data. Real
  invoices will be messier, and so will the ground truth.
- **Single model.** Results are for `claude-haiku-4-5` only. The strategy ranking, the validator's recall, and
  the description-formatting quirk in [Post-hoc analyses](#post-hoc-analyses) are all specific to this model
  and may not hold for a different one.
- **n=40.** The test split give wide confidence intervals (roughly ±10-15 points near 80-90%, see the Results
  tables). Subgroup cells of 8 documents or fewer are especially noisy. Only the zero_shot-vs-rules comparison
  reached significance under the paired McNemar test (p = 0.007); the others did not (p = 0.146, p = 0.125).
- **The rules prompt mirrors the generator's own variations** (negative credit-note amounts, `subtotal =
  total - tax` on tax-inclusive documents, positive discount amounts, US/European number formats). The
  zero-shot prompt is not told these conventions, so part of the 60.0% → 87.5% gap between zero_shot and rules
  measures whether the model happens to already follow my conventions, not extraction skill in general. The
  conventions are reasonable but they are choices, and the rules prompt was written knowing them in advance.
- **The test split has already been used** for the run reported above (`results/run_log.json`,
  `rerun_allowed: false` for all three strategies). The CLI refuses to re-score it under a changed prompt
  unless `--allow-test-rerun` is passed, and the report then carries a visible warning — but the number
  reported here cannot be treated as a true held-out estimate for any future change to these prompts.
- **Text input only.** The pipeline uses pdfplumber text, not images, so it does not test OCR or visual layout
  understanding. Some templates (notably the sidebar) interleave lines in extracted text.
- **Validation sees only internal consistency.** The stress test in [Post-hoc analyses](#post-hoc-analyses)
  demonstrates this directly: a price change recomputed consistently through subtotal/tax/total slipped past
  every check in 49/49 trials. On the real run, this shows up as zero_shot's validation recall of only 50% —
  half its wrong extractions were simply not the kind of error an arithmetic check can see.
- **Line-item scoring is strict** in the primary metric: one wrong digit, or one punctuation difference,
  anywhere makes the document's `line_items` wrong. The post-hoc lenient metric relaxes only the punctuation
  case and is reported separately, never blended into the primary numbers.
- **Taxonomy categories are heuristics.** For example, "magnitude error" means the ratio is near a power of ten,
  not proof of a number-format misread, and one root cause can appear as several wrong fields.
- **Temperature and thinking.** Temperature 0 was accepted by `claude-haiku-4-5` on every call this run (see
  the Usage table); a different model might reject it, in which case the summary's usage table would show it.
  Thinking behaviour is left at the model's default and is not controlled or reported per-call.
- **Cost estimate is approximate** and excludes thinking tokens; the Usage table above reports actual token
  counts from the real run, not the estimate.
- **The post-hoc stress test's corruption mix is my own choice**, not a sample of anything — it says nothing
  about how often each error type occurs in the model's actual output. `results/validator_stress_test.md`
  says this explicitly and points to the real per-field recall numbers in `results/summary.md` instead.

## What I would do next

- Run a second model and compare: is the punctuation-joining quirk and the 50% zero-shot validation recall
  specific to `claude-haiku-4-5`, or general?
- Evaluate on real invoices (with consent and redaction) with human-adjudicated labels, including scanned
  documents with an OCR step, and compare text-only against image input.
- Grow the test set and add a second seed so intervals tighten and results do not depend on one draw.
- Score line items per item and per cell, and add tolerance for vendor-name aliases beyond the lenient
  punctuation-only case already added post-hoc.
- Add validation checks that look beyond arithmetic (vendor master match, duplicate detection across periods, PO
  matching) and measure how much they raise recall — the stress test shows exactly which corruption types
  (vendor name, invoice number, coordinated price changes) today's checks cannot reach.
- Route flagged documents to a review queue and measure reviewer time against the straight-through savings.
- Calibrate: compare model self-reported confidence with correctness for risk-based sampling of unflagged documents.
