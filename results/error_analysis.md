# Error analysis


Failures are classified deterministically from prediction vs ground truth (`src/evaluate.py`, `classify_*`). A document can contribute several wrong fields; a single root cause (e.g. a wrong document type) often cascades into several categories.

## Strategy: zero_shot

16 of 40 documents contain at least one wrong field (40 wrong fields in total).

### Taxonomy

| Category | Meaning | wrong fields | documents | documents flagged by validator |
|---|---|---|---|---|
| sign_error | right magnitude, wrong sign (typically credit-note amounts or the discount sign) | 23 | 6 | 6 |
| description_mismatch | line-item description differs | 12 | 12 | 4 |
| line_item_value_error | a line item's quantity, unit price or amount is wrong | 3 | 3 | 3 |
| spurious_value | value given for a field that is null in the document (e.g. computed rate, invented due date) | 1 | 1 | 1 |
| tax_inclusive_confusion | subtotal/total/tax wrong on a tax-inclusive document (net vs gross basis) | 1 | 1 | 1 |

### Examples

**sign_error**

| doc | template | noise | hard case | field | predicted | expected | detail | validator |
|---|---|---|---|---|---|---|---|---|
| test_02 | totals_first | none | credit_paren_eu | subtotal | 1718.35 | -1718.35 | - | sign_inconsistent, line_amount_mismatch, line_amount_mismatch |
| test_02 | totals_first | none | credit_paren_eu | discount | 171.84 | -171.84 | - | sign_inconsistent, line_amount_mismatch, line_amount_mismatch |
| test_02 | totals_first | none | credit_paren_eu | tax_amount | 386.63 | -386.63 | - | sign_inconsistent, line_amount_mismatch, line_amount_mismatch |

**description_mismatch**

| doc | template | noise | hard case | field | predicted | expected | detail | validator |
|---|---|---|---|---|---|---|---|---|
| test_01 | stacked_meta | none | - | line_items | 2 items | 2 items | 'A4 copy paper (box of 5), 80gsm, FSC certified' vs 'A4 copy paper (box of 5) 80gsm, FSC certified' | not flagged |
| test_05 | classic_table | low | - | line_items | 6 items | 6 items | 'Cleaning services, Weekly, office floor 2' vs 'Cleaning services Weekly, office floor 2' | not flagged |
| test_08 | freeform_lines | medium | eu_fractional_multiline | line_items | 6 items | 6 items | 'A4 copy paper (box of 5), 80gsm, FSC certified' vs 'A4 copy paper (box of 5) 80gsm, FSC certified' | not flagged |

**line_item_value_error**

| doc | template | noise | hard case | field | predicted | expected | detail | validator |
|---|---|---|---|---|---|---|---|---|
| test_02 | totals_first | none | credit_paren_eu | line_items | 2 items | 2 items | 'Freight - pallet delivery': quantity 5 vs 20.0 | sign_inconsistent, line_amount_mismatch, line_amount_mismatch |
| test_37 | totals_first | none | - | line_items | 5 items | 5 items | 'Freight - pallet delivery': amount 854.56 vs -854.56 | sign_inconsistent, line_amount_mismatch, line_amount_mismatch, line_amount_mismatch, line_amount_mismatch, line_amount_mismatch |
| test_38 | freeform_lines | medium | - | line_items | 4 items | 4 items | 'LED desk lamp': amount 272.1 vs -272.1 | sign_inconsistent, line_amount_mismatch, line_amount_mismatch, line_amount_mismatch, line_amount_mismatch |

**spurious_value**

| doc | template | noise | hard case | field | predicted | expected | detail | validator |
|---|---|---|---|---|---|---|---|---|
| test_14 | sidebar | none | no_tax_rate | tax_rate | 20 | None | - | tax_rate_mismatch |

**tax_inclusive_confusion**

| doc | template | noise | hard case | field | predicted | expected | detail | validator |
|---|---|---|---|---|---|---|---|---|
| test_34 | sidebar | low | - | subtotal | 2760.18 | 2208.14 | - | total_mismatch |

### Wrong documents by template

| template | documents with an error |
|---|---|
| classic_table | 2 |
| freeform_lines | 4 |
| sidebar | 4 |
| stacked_meta | 2 |
| totals_first | 4 |

## Strategy: few_shot

10 of 40 documents contain at least one wrong field (40 wrong fields in total).

### Taxonomy

| Category | Meaning | wrong fields | documents | documents flagged by validator |
|---|---|---|---|---|
| sign_error | right magnitude, wrong sign (typically credit-note amounts or the discount sign) | 29 | 8 | 8 |
| line_item_value_error | a line item's quantity, unit price or amount is wrong | 8 | 8 | 8 |
| spurious_value | value given for a field that is null in the document (e.g. computed rate, invented due date) | 1 | 1 | 1 |
| tax_inclusive_confusion | subtotal/total/tax wrong on a tax-inclusive document (net vs gross basis) | 1 | 1 | 1 |
| wrong_number | numeric field wrong for another reason | 1 | 1 | 1 |

### Examples

**sign_error**

| doc | template | noise | hard case | field | predicted | expected | detail | validator |
|---|---|---|---|---|---|---|---|---|
| test_02 | totals_first | none | credit_paren_eu | subtotal | 1718.35 | -1718.35 | - | sign_inconsistent, line_amount_mismatch, line_amount_mismatch |
| test_02 | totals_first | none | credit_paren_eu | discount | 171.84 | -171.84 | - | sign_inconsistent, line_amount_mismatch, line_amount_mismatch |
| test_02 | totals_first | none | credit_paren_eu | tax_amount | 386.63 | -386.63 | - | sign_inconsistent, line_amount_mismatch, line_amount_mismatch |

**line_item_value_error**

| doc | template | noise | hard case | field | predicted | expected | detail | validator |
|---|---|---|---|---|---|---|---|---|
| test_02 | totals_first | none | credit_paren_eu | line_items | 2 items | 2 items | 'Freight - pallet delivery': quantity 5.0 vs 20.0 | sign_inconsistent, line_amount_mismatch, line_amount_mismatch |
| test_15 | classic_table | low | - | line_items | 7 items | 7 items | 'Cleaning services': quantity 1.5 vs 6.0 | sign_inconsistent, line_amount_mismatch, line_amount_mismatch, line_amount_mismatch, line_amount_mismatch, line_amount_mismatch, line_amount_mismatch, line_amount_mismatch |
| test_16 | stacked_meta | low | - | line_items | 7 items | 7 items | 'Cleaning services': amount 692.07 vs -692.07 | sign_inconsistent, line_amount_mismatch, line_amount_mismatch, line_amount_mismatch, line_amount_mismatch, line_amount_mismatch, line_amount_mismatch, line_amount_mismatch |

**spurious_value**

| doc | template | noise | hard case | field | predicted | expected | detail | validator |
|---|---|---|---|---|---|---|---|---|
| test_14 | sidebar | none | no_tax_rate | tax_rate | 20.0 | None | - | tax_rate_mismatch |

**tax_inclusive_confusion**

| doc | template | noise | hard case | field | predicted | expected | detail | validator |
|---|---|---|---|---|---|---|---|---|
| test_34 | sidebar | low | - | subtotal | 2760.18 | 2208.14 | - | total_mismatch |

**wrong_number**

| doc | template | noise | hard case | field | predicted | expected | detail | validator |
|---|---|---|---|---|---|---|---|---|
| test_17 | totals_first | low | - | total | 2517.85 | -2517.05 | - | sign_inconsistent, line_amount_mismatch, total_mismatch |

### Wrong documents by template

| template | documents with an error |
|---|---|
| classic_table | 2 |
| freeform_lines | 1 |
| sidebar | 3 |
| stacked_meta | 1 |
| totals_first | 3 |

## Strategy: rules

5 of 40 documents contain at least one wrong field (5 wrong fields in total).

### Taxonomy

| Category | Meaning | wrong fields | documents | documents flagged by validator |
|---|---|---|---|---|
| tax_inclusive_confusion | subtotal/total/tax wrong on a tax-inclusive document (net vs gross basis) | 2 | 2 | 2 |
| wrong_number | numeric field wrong for another reason | 2 | 2 | 2 |
| spurious_value | value given for a field that is null in the document (e.g. computed rate, invented due date) | 1 | 1 | 1 |

### Examples

**tax_inclusive_confusion**

| doc | template | noise | hard case | field | predicted | expected | detail | validator |
|---|---|---|---|---|---|---|---|---|
| test_09 | sidebar | none | - | subtotal | 13821.3 | 13821.24 | - | total_mismatch |
| test_34 | sidebar | low | - | subtotal | 2760.18 | 2208.14 | - | total_mismatch |

**wrong_number**

| doc | template | noise | hard case | field | predicted | expected | detail | validator |
|---|---|---|---|---|---|---|---|---|
| test_17 | totals_first | low | - | total | -2737.85 | -2517.05 | - | total_mismatch |
| test_35 | classic_table | none | - | total | -5998.71 | -5457.71 | - | total_mismatch |

**spurious_value**

| doc | template | noise | hard case | field | predicted | expected | detail | validator |
|---|---|---|---|---|---|---|---|---|
| test_14 | sidebar | none | no_tax_rate | tax_rate | 20 | None | - | tax_rate_mismatch |

### Wrong documents by template

| template | documents with an error |
|---|---|
| classic_table | 1 |
| sidebar | 3 |
| totals_first | 1 |

## Documents wrong under every strategy

| doc | template | noise | hard case | categories (first strategy) |
|---|---|---|---|---|
| test_14 | sidebar | none | no_tax_rate | spurious_value |
| test_17 | totals_first | low | - | description_mismatch, sign_error |
| test_34 | sidebar | low | - | description_mismatch, tax_inclusive_confusion |

## What the validation layer missed

Wrong fields by category, split by whether the document was flagged (all strategies pooled). Categories that are mostly unflagged are errors the deterministic checks cannot see.

| Category | in flagged documents | in unflagged documents |
|---|---|---|
| sign_error | 52 | 0 |
| description_mismatch | 4 | 8 |
| line_item_value_error | 11 | 0 |
| tax_inclusive_confusion | 4 | 0 |
| spurious_value | 3 | 0 |
| wrong_number | 3 | 0 |
