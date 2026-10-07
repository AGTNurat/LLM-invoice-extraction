import copy

import pytest

from src.description_audit import audit_strategy, build_audit_report, char_diff, classify_description_diff
from src.extract import ExtractionResult
from src.generate import build_records


@pytest.fixture(scope="module")
def test_records():
    return [r for r in build_records(42) if r["split"] == "test"]


@pytest.fixture()
def truth(test_records):
    return copy.deepcopy(next(r["truth"] for r in test_records if len(r["truth"]["line_items"]) >= 2))


def wrap(d):
    return {k: ExtractionResult(k, v, True) for k, v in d.items()}


def test_classify_description_diff():
    assert classify_description_diff("A, B", "A B") == "punctuation/whitespace/case only"
    assert classify_description_diff("a b", "A  B") == "punctuation/whitespace/case only"
    assert classify_description_diff("A X", "A B") == "word-level difference"
    assert classify_description_diff(None, "A") == "word-level difference"


def test_char_diff_shows_insertions_and_deletions():
    d = char_diff("A4 copy paper (box of 5) 80gsm", "A4 copy paper (box of 5), 80gsm")
    assert "{+,+}" in d or "[-,-]" in d  # the comma shows up as an addition or removal depending on direction
    assert char_diff("same", "same") == "same"


def test_audit_strategy_finds_punctuation_mismatch_and_skips_correct_docs(test_records, truth):
    items = truth["line_items"]
    pred = copy.deepcopy(truth)
    original = pred["line_items"][0]["description"]
    pred["line_items"][0]["description"] = original.replace(" ", ", ", 1) if " " in original else original + ","
    if pred["line_items"][0]["description"] == original:
        pytest.skip("fixture item has no separable word")
    doc_id = next(r["doc_id"] for r in test_records if r["truth"] is truth or r["truth"]["invoice_number"] == truth["invoice_number"])
    extractions = {r["doc_id"]: r["truth"] for r in test_records}  # everything else perfect
    extractions[doc_id] = pred
    rows = audit_strategy("zero_shot", wrap(extractions), test_records)
    assert len(rows) == 1
    assert rows[0]["doc_id"] == doc_id and rows[0]["item_index"] == 0
    assert rows[0]["class"] == "punctuation/whitespace/case only"
    assert rows[0]["expected"] == original and rows[0]["predicted"] == pred["line_items"][0]["description"]


def test_audit_strategy_classifies_word_level_and_skips_count_mismatches(test_records, truth):
    pred = copy.deepcopy(truth)
    pred["line_items"][0]["description"] += " EXTRA WORD"
    doc_id = test_records[0]["doc_id"]
    test_records[0]["truth"]  # sanity
    extractions = {r["doc_id"]: r["truth"] for r in test_records}
    # pick a doc whose truth matches `truth` fixture content structurally by using truth directly
    target = next(r for r in test_records if r["truth"]["invoice_number"] == truth["invoice_number"])
    extractions[target["doc_id"]] = pred
    rows = audit_strategy("s", wrap(extractions), test_records)
    assert len(rows) == 1 and rows[0]["class"] == "word-level difference"

    dropped = copy.deepcopy(truth)
    dropped["line_items"].pop()
    extractions2 = {r["doc_id"]: r["truth"] for r in test_records}
    extractions2[target["doc_id"]] = dropped
    assert audit_strategy("s", wrap(extractions2), test_records) == []  # count mismatch: not a description issue


def test_audit_strategy_ignores_failed_and_perfect_extractions(test_records):
    extractions = {r["doc_id"]: r["truth"] for r in test_records}
    extractions[test_records[0]["doc_id"]] = None
    assert audit_strategy("s", wrap(extractions), test_records) == []


def test_build_audit_report_empty_and_nonempty():
    empty = build_audit_report({"zero_shot": [], "rules": []})
    assert "post-hoc" in empty.lower() and "No description mismatches." in empty
    rows = {"zero_shot": [{"strategy": "zero_shot", "doc_id": "test_00", "item_index": 0, "predicted": "a, b",
                           "expected": "a b", "diff": "a[-, -]{+ +}b", "class": "punctuation/whitespace/case only"}]}
    txt = build_audit_report(rows)
    assert "test_00" in txt and "| doc | item |" in txt and "punctuation/whitespace/case only" in txt


def test_report_table_escapes_pipes_in_descriptions():
    rows = {"s": [{"strategy": "s", "doc_id": "d", "item_index": 0, "predicted": "a | b", "expected": "a|b",
                  "diff": "a|b", "class": "word-level difference"}]}
    txt = build_audit_report(rows)
    assert "a \\| b" in txt
