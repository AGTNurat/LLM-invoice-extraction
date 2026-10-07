# Description mismatch audit (post-hoc)

> **Post-hoc.** Written after seeing test-split results, from the cached extractions already scored in `results/summary.md`. No API calls were made and no prompt, generator, seed or cached response was changed to produce this file.

Every line item whose description differs under the PRIMARY metric (`src/evaluate.py: normalize_name`, i.e. whitespace-collapsed, case-folded) on a document where `line_items` was scored wrong. `class` additionally strips all punctuation: "punctuation/whitespace/case only" means the only difference is punctuation, whitespace or case (e.g. a missing comma); "word-level difference" means the words themselves differ.

## zero_shot

30 mismatched line-item description(s): 30 punctuation/whitespace/case only, 0 word-level.

| doc | item | predicted | expected | diff (truth -> pred) | class |
|---|---|---|---|---|---|
| test_01 | 0 | A4 copy paper (box of 5), 80gsm, FSC certified | A4 copy paper (box of 5) 80gsm, FSC certified | A4 copy paper (box of 5){+,+} 80gsm, FSC certified | punctuation/whitespace/case only |
| test_01 | 1 | Network switch 24-port, Managed, PoE+ | Network switch 24-port Managed, PoE+ | Network switch 24-port{+,+} Managed, PoE+ | punctuation/whitespace/case only |
| test_05 | 0 | Cleaning services, Weekly, office floor 2 | Cleaning services Weekly, office floor 2 | Cleaning services{+,+} Weekly, office floor 2 | punctuation/whitespace/case only |
| test_05 | 2 | Safety gloves (pack of 12), Size L, EN 388 | Safety gloves (pack of 12) Size L, EN 388 | Safety gloves (pack of 12){+,+} Size L, EN 388 | punctuation/whitespace/case only |
| test_05 | 3 | Safety gloves (pack of 12), Size L, EN 388 | Safety gloves (pack of 12) Size L, EN 388 | Safety gloves (pack of 12){+,+} Size L, EN 388 | punctuation/whitespace/case only |
| test_05 | 4 | Stainless steel bolts M8, Box of 200, A2 grade | Stainless steel bolts M8 Box of 200, A2 grade | Stainless steel bolts M8{+,+} Box of 200, A2 grade | punctuation/whitespace/case only |
| test_05 | 5 | Toner cartridge, Black, high yield | Toner cartridge Black, high yield | Toner cartridge{+,+} Black, high yield | punctuation/whitespace/case only |
| test_08 | 0 | A4 copy paper (box of 5), 80gsm, FSC certified | A4 copy paper (box of 5) 80gsm, FSC certified | A4 copy paper (box of 5){+,+} 80gsm, FSC certified | punctuation/whitespace/case only |
| test_08 | 1 | A4 copy paper (box of 5), 80gsm, FSC certified | A4 copy paper (box of 5) 80gsm, FSC certified | A4 copy paper (box of 5){+,+} 80gsm, FSC certified | punctuation/whitespace/case only |
| test_08 | 2 | Cloud hosting - standard, Monthly plan, 4 vCPU | Cloud hosting - standard Monthly plan, 4 vCPU | Cloud hosting - standard{+,+} Monthly plan, 4 vCPU | punctuation/whitespace/case only |
| test_08 | 3 | Laptop docking station, USB-C, dual display | Laptop docking station USB-C, dual display | Laptop docking station{+,+} USB-C, dual display | punctuation/whitespace/case only |
| test_08 | 4 | Packaging film roll, 500 mm x 300 m | Packaging film roll 500 mm x 300 m | Packaging film roll{+,+} 500 mm x 300 m | punctuation/whitespace/case only |
| test_17 | 0 | Software licence - Annual subscription, per seat | Software licence Annual subscription, per seat | Software licence{+ -+} Annual subscription, per seat | punctuation/whitespace/case only |
| test_18 | 3 | Packaging film roll - 500 mm x 300 m | Packaging film roll 500 mm x 300 m | Packaging film roll {+- +}500 mm x 300 m | punctuation/whitespace/case only |
| test_18 | 5 | Software licence - Annual subscription, per seat | Software licence Annual subscription, per seat | Software licence{+ -+} Annual subscription, per seat | punctuation/whitespace/case only |
| test_19 | 0 | Maintenance contract - Quarterly service visit | Maintenance contract Quarterly service visit | Maintenance contract{+ -+} Quarterly service visit | punctuation/whitespace/case only |
| test_19 | 3 | Software licence - Annual subscription, per seat | Software licence Annual subscription, per seat | Software licence{+ -+} Annual subscription, per seat | punctuation/whitespace/case only |
| test_19 | 4 | Software licence - Annual subscription, per seat | Software licence Annual subscription, per seat | Software licence{+ -+} Annual subscription, per seat | punctuation/whitespace/case only |
| test_19 | 5 | Toner cartridge - Black, high yield | Toner cartridge Black, high yield | Toner cartridge{+ -+} Black, high yield | punctuation/whitespace/case only |
| test_25 | 0 | Laptop docking station, USB-C, dual display | Laptop docking station USB-C, dual display | Laptop docking station{+,+} USB-C, dual display | punctuation/whitespace/case only |
| test_25 | 2 | Software licence, Annual subscription, per seat | Software licence Annual subscription, per seat | Software licence{+,+} Annual subscription, per seat | punctuation/whitespace/case only |
| test_28 | 0 | LED desk lamp, Warm white, dimmable | LED desk lamp Warm white, dimmable | LED desk lamp{+,+} Warm white, dimmable | punctuation/whitespace/case only |
| test_32 | 0 | Cleaning services - Weekly, office floor 2 | Cleaning services Weekly, office floor 2 | Cleaning services{+ -+} Weekly, office floor 2 | punctuation/whitespace/case only |
| test_32 | 1 | Maintenance contract - Quarterly service visit | Maintenance contract Quarterly service visit | Maintenance contract{+ -+} Quarterly service visit | punctuation/whitespace/case only |
| test_34 | 0 | Packaging film roll, 500 mm x 300 m | Packaging film roll 500 mm x 300 m | Packaging film roll{+,+} 500 mm x 300 m | punctuation/whitespace/case only |
| test_34 | 1 | Pallet racking bay, 3 levels, 2700 kg | Pallet racking bay 3 levels, 2700 kg | Pallet racking bay{+,+} 3 levels, 2700 kg | punctuation/whitespace/case only |
| test_34 | 2 | Safety gloves (pack of 12), Size L, EN 388 | Safety gloves (pack of 12) Size L, EN 388 | Safety gloves (pack of 12){+,+} Size L, EN 388 | punctuation/whitespace/case only |
| test_36 | 0 | Cleaning services - Weekly, office floor 2 | Cleaning services Weekly, office floor 2 | Cleaning services{+ -+} Weekly, office floor 2 | punctuation/whitespace/case only |
| test_39 | 0 | Cloud hosting - standard, Monthly plan, 4 vCPU | Cloud hosting - standard Monthly plan, 4 vCPU | Cloud hosting - standard{+,+} Monthly plan, 4 vCPU | punctuation/whitespace/case only |
| test_39 | 2 | LED desk lamp, Warm white, dimmable | LED desk lamp Warm white, dimmable | LED desk lamp{+,+} Warm white, dimmable | punctuation/whitespace/case only |

## few_shot

No description mismatches.

## rules

No description mismatches.
