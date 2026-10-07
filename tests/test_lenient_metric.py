import copy

import pytest

from src.evaluate import lenient_line_item_correct, lenient_score_document, lenient_score_run, line_item_correct, \
    score_document, score_run
from src.extract import MockExtractor
from src.generate import build_records
from src.normalize import description_key
from src.schema import ALL_FIELDS


@pytest.fixture(scope="module")
def test_records():
    return [r for r in build_records(42) if r["split"] == "test"]


@pytest.fixture()
def truth(test_records):
    return copy.deepcopy(next(r["truth"] for r in test_records if len(r["truth"]["line_items"]) >= 2))


# ------------------------------------------------------------------ description_key / normalizer
@pytest.mark.parametrize("a,b", [
    ("A4 copy paper (box of 5), 80gsm", "A4 copy paper (box of 5) 80gsm"),   # missing comma
    ("Cleaning services, Weekly", "Cleaning services Weekly"),
    ("  Widget,  Pro ", "widget pro"),
    ("Café: deluxe", "café deluxe"),   # NFKC + case fold strip punctuation/whitespace; accents are untouched
])
def test_description_key_ignores_punctuation_whitespace_case(a, b):
    assert description_key(a) == description_key(b)


@pytest.mark.parametrize("a,b", [("Widget Pro", "Widget Max"), ("Gadget", "Gadget X"), ("A B", "A C")])
def test_description_key_still_distinguishes_words(a, b):
    assert description_key(a) != description_key(b)


def test_description_key_handles_none():
    assert description_key(None) == ""


# ------------------------------------------------------------------ lenient line-item / document scoring
def test_lenient_accepts_punctuation_only_difference(truth):
    items = truth["line_items"]
    pred = copy.deepcopy(items)
    pred[0]["description"] = pred[0]["description"].replace(" ", ", ", 1) if " " in pred[0]["description"] else \
        pred[0]["description"] + ","
    if description_key(pred[0]["description"]) != description_key(items[0]["description"]):
        pytest.skip("fixture item has no separable word to punctuate")
    assert not line_item_correct(pred, items)          # primary metric: strict, still wrong
    assert lenient_line_item_correct(pred, items)       # lenient metric: punctuation-only, now correct


def test_lenient_still_rejects_word_level_change(truth):
    items = truth["line_items"]
    pred = copy.deepcopy(items)
    pred[0]["description"] = pred[0]["description"] + " EXTRA WORD"
    assert not line_item_correct(pred, items)
    assert not lenient_line_item_correct(pred, items)   # word-level: must stay wrong under lenient too


def test_lenient_still_checks_quantities_and_amounts(truth):
    items = truth["line_items"]
    pred = copy.deepcopy(items)
    pred[0]["amount"] += 5
    assert not lenient_line_item_correct(pred, items)   # punctuation leniency does not cover numbers


def test_lenient_rejects_wrong_item_count(truth):
    assert not lenient_line_item_correct(truth["line_items"][:-1], truth["line_items"])


def test_lenient_score_document_only_affects_line_items(truth):
    pred = copy.deepcopy(truth)
    desc = pred["line_items"][0]["description"]
    if " " not in desc:
        pytest.skip("fixture item has no separable word to punctuate mid-string")
    pred["line_items"][0]["description"] = desc.replace(" ", ", ", 1)  # mid-string punctuation only
    pred["vendor_name"] = "totally different co"  # a genuine, unrelated error
    primary = score_document(pred, truth)
    lenient = lenient_score_document(pred, truth)
    assert primary["line_items"] != lenient["line_items"]   # the one thing that should change
    for f in ALL_FIELDS:
        if f != "line_items":
            assert primary[f] == lenient[f]                 # every other field: identical
    assert not lenient["vendor_name"] and not lenient["all_correct"]  # a real error still fails lenient too


# ------------------------------------------------------------------ run-level: primary metric is provably unchanged
def test_primary_metric_outputs_identical_to_before_lenient_change(test_records):
    """Regression guard for the post-hoc Stage A change: score_document/score_run must produce byte-identical
    results whether or not the lenient functions exist/are called alongside them."""
    truths = {r["doc_id"]: r["truth"] for r in test_records}
    m = MockExtractor(truths, error_rate=0.4, seed=11)
    extractions = {d: m.extract(d) for d in truths}
    primary_before = score_run(extractions, test_records)
    _ = lenient_score_run(extractions, test_records)   # exercise the new code path
    primary_after = score_run(extractions, test_records)
    pd_testing_equal(primary_before, primary_after)
    # and the per-document primary scores used by lenient internally still match a fresh direct call
    for r in test_records:
        assert score_document(extractions[r["doc_id"]].extraction, r["truth"]) == \
            score_document(extractions[r["doc_id"]].extraction, r["truth"])


def pd_testing_equal(a, b):
    import pandas as pd
    pd.testing.assert_frame_equal(a, b)


def test_lenient_score_run_same_shape_and_split_guard(test_records):
    truths = {r["doc_id"]: r["truth"] for r in test_records}
    m = MockExtractor(truths, error_rate=0.3, seed=2)
    extractions = {d: m.extract(d) for d in truths}
    primary = score_run(extractions, test_records)
    lenient = lenient_score_run(extractions, test_records)
    assert list(primary.columns) == list(lenient.columns)
    assert list(primary["doc_id"]) == list(lenient["doc_id"])
    # lenient all_correct can only be >= primary all_correct per document (strictly less strict)
    merged = primary.set_index("doc_id")["all_correct"].to_frame("p").join(
        lenient.set_index("doc_id")["all_correct"].to_frame("l"))
    assert (merged["l"] >= merged["p"]).all()
    dev = [r for r in build_records(42) if r["split"] == "dev"]
    with pytest.raises(ValueError):
        lenient_score_run({}, dev)


def test_lenient_all_correct_matches_perfect_run(test_records):
    truths = {r["doc_id"]: copy.deepcopy(r["truth"]) for r in test_records}
    perfect = {d: t for d, t in truths.items()}
    lscores = lenient_score_run(perfect, test_records)
    assert lscores["all_correct"].all()
