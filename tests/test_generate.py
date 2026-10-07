import hashlib
import re
from datetime import date

import pdfplumber
import pytest

from src.generate import (HARD, TEMPLATES, build_records, generate_dataset, load_split)
from src.normalize import normalize_date
from src.schema import ALL_FIELDS


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("gen")
    return out, generate_dataset(out, seed=42)


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_build_records_deterministic_and_seed_sensitive():
    a, b, c = build_records(42), build_records(42), build_records(7)
    assert a == b
    assert [r["truth"] for r in a] != [r["truth"] for r in c]


def test_pdfs_and_json_byte_identical_across_runs(dataset, tmp_path):
    out1, _ = dataset
    generate_dataset(tmp_path, seed=42)
    for split in ("dev", "test"):
        for f in sorted((out1 / split).iterdir()):
            assert _sha(f) == _sha(tmp_path / split / f.name), f.name
    assert _sha(out1 / "manifest.csv") == _sha(tmp_path / "manifest.csv")


def test_split_sizes_ids_and_template_coverage(dataset):
    out, recs = dataset
    assert len(recs) == 60
    ids = [r["doc_id"] for r in recs]
    assert len(set(ids)) == 60
    dev, test = load_split(out, "dev"), load_split(out, "test")
    assert (len(dev), len(test)) == (20, 40)
    assert len(TEMPLATES) >= 4
    for split in (dev, test):
        assert {r["meta"]["template"] for r in split} == set(TEMPLATES)
    assert len({r["truth"]["vendor_name"] for r in recs}) == 60  # no vendor shared between dev and test


def test_metadata_recorded_and_hard_cases_labelled(dataset):
    _, recs = dataset
    assert {r["meta"]["noise_level"] for r in recs} == {0, 1, 2}
    hard = [r for r in recs if r["meta"]["hard"]]
    assert len(hard) == len(HARD) == 6
    assert all(r["meta"]["hard_reason"] and r["meta"]["hard_description"] for r in hard)
    assert all(not r["meta"]["hard_reason"] for r in recs if not r["meta"]["hard"])
    assert {r["split"] for r in hard} == {"dev", "test"}


def test_feature_coverage_in_both_splits(dataset):
    _, recs = dataset
    for split in ("dev", "test"):
        f = [r["meta"]["features"] for r in recs if r["split"] == split]
        assert any(x["document_type"] == "credit_note" for x in f)
        assert any(x["number_format"] == "eu" for x in f) and any(x["number_format"] == "us" for x in f)
        assert any(x["tax_inclusive"] for x in f) and any(not x["tax_inclusive"] for x in f)
        assert any(x["has_discount"] for x in f)
        assert any(x["multiline_descriptions"] for x in f)
        assert any(x["meta_shuffled"] for x in f)
        assert len({x["date_format"] for x in f}) >= 3


def test_ground_truth_internally_consistent(dataset):
    _, recs = dataset
    for r in recs:
        t, f = r["truth"], r["meta"]["features"]
        assert set(t) == set(ALL_FIELDS)
        sign = -1 if t["document_type"] == "credit_note" else 1
        for it in t["line_items"]:
            assert it["quantity"] > 0 and it["unit_price"] > 0
            assert it["amount"] == pytest.approx(sign * it["quantity"] * it["unit_price"], abs=0.01)
        assert t["subtotal"] - t["discount"] + t["tax_amount"] == pytest.approx(t["total"], abs=0.005)
        line_sum = sum(i["amount"] for i in t["line_items"])
        assert line_sum == pytest.approx(t["total"] if f["tax_inclusive"] else t["subtotal"], abs=0.005)
        for k in ("subtotal", "discount", "tax_amount", "total"):
            assert t[k] * sign >= 0, (r["doc_id"], k)
        if t["tax_rate"] is not None and not f["tax_inclusive"]:
            assert t["tax_amount"] == pytest.approx((t["subtotal"] - t["discount"]) * t["tax_rate"] / 100, abs=0.01)
        d = date.fromisoformat(t["invoice_date"])
        assert normalize_date(t["invoice_date"]) == t["invoice_date"]
        if t["due_date"]:
            assert date.fromisoformat(t["due_date"]) >= d
        if sign == -1:
            assert t["due_date"] is None


def test_hard_case_properties(dataset):
    _, recs = dataset
    by = {r["meta"]["hard_reason"]: r for r in recs if r["meta"]["hard"]}
    assert by["multi_page"]["meta"]["features"]["pages"] == 2
    assert by["no_tax_rate"]["truth"]["tax_rate"] is None
    assert by["credit_paren_eu"]["truth"]["document_type"] == "credit_note"
    assert by["credit_paren_eu"]["meta"]["features"]["negatives_in_parens"]
    amb = by["ambiguous_date"]
    assert amb["meta"]["features"]["date_format"] in ("dmy_slash", "mdy_slash")
    assert int(amb["truth"]["invoice_date"][8:]) <= 12 and int(amb["truth"]["due_date"][8:]) >= 13


def test_normal_slash_dates_are_unambiguous(dataset):
    _, recs = dataset
    for r in recs:
        if r["meta"]["features"]["date_format"].endswith("slash") and not r["meta"]["hard"]:
            assert int(r["truth"]["invoice_date"][8:]) >= 13


def test_pdf_text_contains_values_exactly_as_labelled(dataset):
    """Every labelled value must actually be findable in the PDF text (labels are exact by construction)."""
    out, recs = dataset
    squeeze = lambda s: re.sub(r"\s+", "", s)  # noqa: E731
    for r in recs:
        with pdfplumber.open(out / r["split"] / f"{r['doc_id']}.pdf") as pdf:
            text = squeeze("\n".join(p.extract_text() or "" for p in pdf.pages))
        t, m = r["truth"], r["meta"]
        assert squeeze(t["vendor_name"]) in text, r["doc_id"]
        assert squeeze(t["invoice_number"]) in text, r["doc_id"]
        assert squeeze(t["currency"]) in text, r["doc_id"]
        for k, v in m["printed"].items():
            if v is not None:
                assert squeeze(v) in text, (r["doc_id"], k, v)
        for it in r["plan"]["items"]:
            for s in (it["unit"], it["amount"]) + tuple(it["desc_lines"]):
                assert squeeze(s) in text, (r["doc_id"], s)
        gt_json = (out / r["split"] / f"{r['doc_id']}.json").read_text(encoding="utf-8")
        assert '"truth"' in gt_json and r["doc_id"] in gt_json
