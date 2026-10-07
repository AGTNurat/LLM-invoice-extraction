# Evaluation summary


- Model: `claude-haiku-4-5`
- Split: **test** (40 documents); strategies: zero_shot, few_shot, rules
- Dataset seed: 42
- All data is synthetic and labels are exact by construction; see README limitations.
- Intervals are 95% Wilson score intervals; with n=40 they are wide, so small differences are not statistically meaningful.
- Test split was scored with one fixed prompt per strategy (see results/run_log.json).

## 1. Strategy comparison

| Strategy | All fields correct | Pooled field accuracy | Failed extractions |
|---|---|---|---|
| zero_shot | 24/40 = 60.0% [44.6%–73.7%] | 91.7% | 0 |
| few_shot | 30/40 = 75.0% [59.8%–85.8%] | 91.7% | 0 |
| rules | 35/40 = 87.5% [73.9%–94.5%] | 99.0% | 0 |

Paired comparison on the same documents (exact McNemar test on all-fields-correct; 'only X' = documents that X got fully right and the other did not):

| A | B | both correct | only A | only B | both wrong | McNemar p |
|---|---|---|---|---|---|---|
| zero_shot | few_shot | 21 | 3 | 9 | 7 | 0.146 |
| zero_shot | rules | 22 | 2 | 13 | 3 | 0.007 |
| few_shot | rules | 29 | 1 | 6 | 4 | 0.125 |

## 2. Per-field accuracy

| Field | zero_shot | few_shot | rules |
|---|---|---|---|
| vendor_name | 40/40 = 100.0% [91.2%–100.0%] | 40/40 = 100.0% [91.2%–100.0%] | 40/40 = 100.0% [91.2%–100.0%] |
| invoice_number | 40/40 = 100.0% [91.2%–100.0%] | 40/40 = 100.0% [91.2%–100.0%] | 40/40 = 100.0% [91.2%–100.0%] |
| invoice_date | 40/40 = 100.0% [91.2%–100.0%] | 40/40 = 100.0% [91.2%–100.0%] | 40/40 = 100.0% [91.2%–100.0%] |
| due_date | 40/40 = 100.0% [91.2%–100.0%] | 40/40 = 100.0% [91.2%–100.0%] | 40/40 = 100.0% [91.2%–100.0%] |
| currency | 40/40 = 100.0% [91.2%–100.0%] | 40/40 = 100.0% [91.2%–100.0%] | 40/40 = 100.0% [91.2%–100.0%] |
| document_type | 40/40 = 100.0% [91.2%–100.0%] | 40/40 = 100.0% [91.2%–100.0%] | 40/40 = 100.0% [91.2%–100.0%] |
| subtotal | 33/40 = 82.5% [68.0%–91.3%] | 31/40 = 77.5% [62.5%–87.7%] | 38/40 = 95.0% [83.5%–98.6%] |
| discount | 35/40 = 87.5% [73.9%–94.5%] | 34/40 = 85.0% [70.9%–92.9%] | 40/40 = 100.0% [91.2%–100.0%] |
| tax_rate | 39/40 = 97.5% [87.1%–99.6%] | 39/40 = 97.5% [87.1%–99.6%] | 39/40 = 97.5% [87.1%–99.6%] |
| tax_amount | 34/40 = 85.0% [70.9%–92.9%] | 32/40 = 80.0% [65.2%–89.5%] | 40/40 = 100.0% [91.2%–100.0%] |
| total | 34/40 = 85.0% [70.9%–92.9%] | 32/40 = 80.0% [65.2%–89.5%] | 38/40 = 95.0% [83.5%–98.6%] |
| line_items | 25/40 = 62.5% [47.0%–75.8%] | 32/40 = 80.0% [65.2%–89.5%] | 40/40 = 100.0% [91.2%–100.0%] |

## 3. Accuracy by subgroup (all fields correct)

### Layout template

| Layout template | zero_shot | few_shot | rules |
|---|---|---|---|
| classic_table | 6/8 = 75.0% [40.9%–92.9%] | 6/8 = 75.0% [40.9%–92.9%] | 7/8 = 87.5% [52.9%–97.8%] |
| freeform_lines | 4/8 = 50.0% [21.5%–78.5%] | 7/8 = 87.5% [52.9%–97.8%] | 8/8 = 100.0% [67.6%–100.0%] |
| sidebar | 4/8 = 50.0% [21.5%–78.5%] | 5/8 = 62.5% [30.6%–86.3%] | 5/8 = 62.5% [30.6%–86.3%] |
| stacked_meta | 6/8 = 75.0% [40.9%–92.9%] | 7/8 = 87.5% [52.9%–97.8%] | 8/8 = 100.0% [67.6%–100.0%] |
| totals_first | 4/8 = 50.0% [21.5%–78.5%] | 5/8 = 62.5% [30.6%–86.3%] | 7/8 = 87.5% [52.9%–97.8%] |

### Noise level

| Noise level | zero_shot | few_shot | rules |
|---|---|---|---|
| none | 11/17 = 64.7% [41.3%–82.7%] | 13/17 = 76.5% [52.7%–90.4%] | 14/17 = 82.4% [59.0%–93.8%] |
| low | 6/12 = 50.0% [25.4%–74.6%] | 7/12 = 58.3% [32.0%–80.7%] | 10/12 = 83.3% [55.2%–95.3%] |
| medium | 7/11 = 63.6% [35.4%–84.8%] | 10/11 = 90.9% [62.3%–98.4%] | 11/11 = 100.0% [74.1%–100.0%] |

### Hard cases

| Hard cases | zero_shot | few_shot | rules |
|---|---|---|---|
| regular | 23/36 = 63.9% [47.6%–77.5%] | 28/36 = 77.8% [61.9%–88.3%] | 32/36 = 88.9% [74.7%–95.6%] |
| hard | 1/4 = 25.0% [4.6%–69.9%] | 2/4 = 50.0% [15.0%–85.0%] | 3/4 = 75.0% [30.1%–95.4%] |

### Document type

| Document type | zero_shot | few_shot | rules |
|---|---|---|---|
| credit_note | 3/11 = 27.3% [9.7%–56.6%] | 3/11 = 27.3% [9.7%–56.6%] | 9/11 = 81.8% [52.3%–94.9%] |
| invoice | 21/29 = 72.4% [54.3%–85.3%] | 27/29 = 93.1% [78.0%–98.1%] | 26/29 = 89.7% [73.6%–96.4%] |

### Number format

| Number format | zero_shot | few_shot | rules |
|---|---|---|---|
| eu | 7/13 = 53.8% [29.1%–76.8%] | 9/13 = 69.2% [42.4%–87.3%] | 12/13 = 92.3% [66.7%–98.6%] |
| us | 17/27 = 63.0% [44.2%–78.5%] | 21/27 = 77.8% [59.2%–89.4%] | 23/27 = 85.2% [67.5%–94.1%] |

### Tax-inclusive

| Tax-inclusive | zero_shot | few_shot | rules |
|---|---|---|---|
| exclusive | 17/31 = 54.8% [37.8%–70.8%] | 22/31 = 71.0% [53.4%–83.9%] | 28/31 = 90.3% [75.1%–96.7%] |
| inclusive | 7/9 = 77.8% [45.3%–93.7%] | 8/9 = 88.9% [56.5%–98.0%] | 7/9 = 77.8% [45.3%–93.7%] |

## 4. Validation layer

'Wrong' = not all fields correct (failed extractions count as wrong); 'flagged' = at least one deterministic check failed. Straight-through = correct AND unflagged, as a share of all documents.

| Strategy | Wrong | Flagged | Recall (wrong that were flagged) | Precision (flags that were wrong) | Straight-through | Wrong among unflagged | Correct but flagged |
|---|---|---|---|---|---|---|---|
| zero_shot | 16 | 8 | 8/16 = 50.0% [28.0%–72.0%] | 8/8 = 100.0% [67.6%–100.0%] | 24/40 = 60.0% [44.6%–73.7%] | 8/32 = 25.0% [13.3%–42.1%] | 0/24 = 0.0% [0.0%–13.8%] |
| few_shot | 10 | 10 | 10/10 = 100.0% [72.2%–100.0%] | 10/10 = 100.0% [72.2%–100.0%] | 30/40 = 75.0% [59.8%–85.8%] | 0/30 = 0.0% [0.0%–11.4%] | 0/30 = 0.0% [0.0%–11.4%] |
| rules | 5 | 5 | 5/5 = 100.0% [56.6%–100.0%] | 5/5 = 100.0% [56.6%–100.0%] | 35/40 = 87.5% [73.9%–94.5%] | 0/35 = 0.0% [0.0%–9.9%] | 0/35 = 0.0% [0.0%–9.9%] |

### Validation detail: zero_shot

Recall by wrong field (a document counts as flagged if ANY check fired, not necessarily one about that field):

| Wrong field | docs with this field wrong | of which flagged | recall |
|---|---|---|---|
| subtotal | 7 | 7 | 100.0% |
| discount | 5 | 5 | 100.0% |
| tax_rate | 1 | 1 | 100.0% |
| tax_amount | 6 | 6 | 100.0% |
| total | 6 | 6 | 100.0% |
| line_items | 15 | 7 | 46.7% |

| Reason code | docs flagged | of which truly wrong | precision |
|---|---|---|---|
| line_amount_mismatch | 6 | 6 | 100.0% |
| sign_inconsistent | 6 | 6 | 100.0% |
| tax_rate_mismatch | 1 | 1 | 100.0% |
| total_mismatch | 1 | 1 | 100.0% |

### Validation detail: few_shot

Recall by wrong field (a document counts as flagged if ANY check fired, not necessarily one about that field):

| Wrong field | docs with this field wrong | of which flagged | recall |
|---|---|---|---|
| subtotal | 9 | 9 | 100.0% |
| discount | 6 | 6 | 100.0% |
| tax_rate | 1 | 1 | 100.0% |
| tax_amount | 8 | 8 | 100.0% |
| total | 8 | 8 | 100.0% |
| line_items | 8 | 8 | 100.0% |

| Reason code | docs flagged | of which truly wrong | precision |
|---|---|---|---|
| line_amount_mismatch | 8 | 8 | 100.0% |
| sign_inconsistent | 8 | 8 | 100.0% |
| tax_rate_mismatch | 1 | 1 | 100.0% |
| total_mismatch | 2 | 2 | 100.0% |

### Validation detail: rules

Recall by wrong field (a document counts as flagged if ANY check fired, not necessarily one about that field):

| Wrong field | docs with this field wrong | of which flagged | recall |
|---|---|---|---|
| subtotal | 2 | 2 | 100.0% |
| tax_rate | 1 | 1 | 100.0% |
| total | 2 | 2 | 100.0% |

| Reason code | docs flagged | of which truly wrong | precision |
|---|---|---|---|
| tax_rate_mismatch | 1 | 1 | 100.0% |
| total_mismatch | 4 | 4 | 100.0% |

## 5. Usage

| Strategy | documents | served from cache | input tokens | output tokens | docs needing retries | docs where temperature=0 was NOT applied |
|---|---|---|---|---|---|---|
| zero_shot | 40 | 0 | 41825 | 8259 | 0 | 0 |
| few_shot | 40 | 0 | 90145 | 10444 | 0 | 0 |
| rules | 40 | 0 | 62585 | 8464 | 0 | 0 |

Token counts are those recorded when each response was first fetched. If any document shows temperature=0 was not applied, the model rejected that setting and ran at its default.

## 6. Post-hoc secondary metric: lenient line-item descriptions

> **Post-hoc.** This metric and the rule below were defined after seeing the primary test-split results, specifically after `results/description_audit.md` showed zero-shot line-item description mismatches that were only punctuation/whitespace/case (e.g. a missing comma). It is reported separately and does not change any number in sections 1-5 above.

**Rule:** line-item descriptions are compared after casefolding and removing all punctuation and whitespace. Nothing fuzzier: no edit distance, no word reordering, no synonym handling. Every other field, and quantity/unit_price/amount, are compared exactly as in the primary metric. A document with a word-level description difference is still wrong under this metric.

### Primary vs. lenient, all-fields-correct

| Strategy | Primary (all fields correct) | Lenient (all fields correct) |
|---|---|---|
| zero_shot | 24/40 = 60.0% [44.6%–73.7%] | 32/40 = 80.0% [65.2%–89.5%] |
| few_shot | 30/40 = 75.0% [59.8%–85.8%] | 30/40 = 75.0% [59.8%–85.8%] |
| rules | 35/40 = 87.5% [73.9%–94.5%] | 35/40 = 87.5% [73.9%–94.5%] |

Paired comparison under the LENIENT metric (exact McNemar test on all-fields-correct):

| A | B | both correct | only A | only B | both wrong | McNemar p |
|---|---|---|---|---|---|---|
| zero_shot | few_shot | 29 | 3 | 1 | 7 | 0.625 |
| zero_shot | rules | 30 | 2 | 5 | 3 | 0.453 |
| few_shot | rules | 29 | 1 | 6 | 4 | 0.125 |

### Validation layer under the lenient metric

'Wrong' is redefined using the lenient all-fields-correct; validator behaviour (flags) is unchanged — only which documents count as wrong changes.

| Strategy | Wrong (lenient) | Flagged | Recall | Precision | Straight-through | Wrong among unflagged | Correct but flagged |
|---|---|---|---|---|---|---|---|
| zero_shot | 8 | 8 | 8/8 = 100.0% [67.6%–100.0%] | 8/8 = 100.0% [67.6%–100.0%] | 32/40 = 80.0% [65.2%–89.5%] | 0/32 = 0.0% [0.0%–10.7%] | 0/32 = 0.0% [0.0%–10.7%] |
| few_shot | 10 | 10 | 10/10 = 100.0% [72.2%–100.0%] | 10/10 = 100.0% [72.2%–100.0%] | 30/40 = 75.0% [59.8%–85.8%] | 0/30 = 0.0% [0.0%–11.4%] | 0/30 = 0.0% [0.0%–11.4%] |
| rules | 5 | 5 | 5/5 = 100.0% [56.6%–100.0%] | 5/5 = 100.0% [56.6%–100.0%] | 35/40 = 87.5% [73.9%–94.5%] | 0/35 = 0.0% [0.0%–9.9%] | 0/35 = 0.0% [0.0%–9.9%] |
