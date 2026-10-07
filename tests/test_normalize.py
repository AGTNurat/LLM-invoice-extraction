import pytest

from src.normalize import (normalize_currency, normalize_date, normalize_doc_type, normalize_invoice_number,
                           normalize_name, numbers_match, parse_number)
from src.schema import ALL_FIELDS, REQUIRED_FIELDS, json_schema


@pytest.mark.parametrize("raw,expected", [
    ("1,234.56", 1234.56), ("1.234,56", 1234.56), ("1 234,56", 1234.56), ("1'234.56", 1234.56),
    ("$12.50", 12.5), ("EUR 1.234,56", 1234.56), ("-12.50", -12.5), ("(12.50)", -12.5), ("12.50-", -12.5),
    ("12,5", 12.5), ("1,23", 1.23), ("1,234", 1234.0), ("1.234.567,89", 1234567.89),
    ("1,234,567.89", 1234567.89), ("0,500", 0.5), ("0.5", 0.5), (7, 7.0), (7.25, 7.25), ("100", 100.0),
])
def test_parse_number(raw, expected):
    assert parse_number(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [None, "", "abc", "--", "1.2.3,4,5", True, [], "1,2,3.4.5"])
def test_parse_number_rejects(raw):
    assert parse_number(raw) is None


def test_numbers_match_tolerance_and_none():
    assert numbers_match(10.00, 10.01)
    assert not numbers_match(10.00, 10.02)
    assert numbers_match("1.234,56", 1234.56)
    assert numbers_match(None, None)
    assert not numbers_match(None, 0)
    assert not numbers_match(5, "x")


@pytest.mark.parametrize("raw,kw,expected", [
    ("2024-03-15", {}, "2024-03-15"), ("15 Mar 2024", {}, "2024-03-15"), ("March 15, 2024", {}, "2024-03-15"),
    ("15.03.2024", {}, "2024-03-15"), ("03/15/2024", {}, "2024-03-15"), ("15/03/2024", {}, "2024-03-15"),
    ("04/05/2024", {}, "2024-04-05"), ("04/05/2024", {"dayfirst": True}, "2024-05-04"),
    ("5th April 2024", {}, "2024-04-05"), ("Sept 3, 2024", {}, "2024-09-03"), ("  2024/3/5 ", {}, "2024-03-05"),
])
def test_normalize_date(raw, kw, expected):
    assert normalize_date(raw, **kw) == expected


@pytest.mark.parametrize("raw", [None, "", "31 Feb 2024", "2024-13-01", "tomorrow", 20240315])
def test_normalize_date_rejects(raw):
    assert normalize_date(raw) is None


def test_names_and_codes():
    assert normalize_name("  Acme   Corp.  ") == normalize_name("ACME CORP")
    assert normalize_name(None) is None
    assert normalize_invoice_number("inv- 001") == "INV-001"
    assert normalize_currency(" eur ") == "EUR" and normalize_currency("XXX") is None
    assert normalize_doc_type("Credit Note") == "credit_note"
    assert normalize_doc_type("credit-memo") == "credit_note"
    assert normalize_doc_type("Invoice") == "invoice" and normalize_doc_type("receipt") is None


def test_json_schema_consistent_with_field_lists():
    s = json_schema()
    assert set(s["properties"]) == set(ALL_FIELDS) == set(s["required"])
    assert s["additionalProperties"] is False
    assert set(REQUIRED_FIELDS) < set(ALL_FIELDS)
