"""Normalizers used to canonicalize values before comparison or validation.

All functions are total: unparseable input returns None rather than raising.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime

from .schema import CURRENCIES, DOCUMENT_TYPES

NUM_TOL = 0.01
_NON_NUMERIC = re.compile(r"[^\d,.\-+()\s']")  # strip symbols/letters like $, EUR, £


def parse_number(value) -> float | None:
    """Parse a number from int/float or a US/European formatted string.

    Handles "1,234.56", "1.234,56", "1 234,56", "1'234.56", "$12.50", "-12.50", "(12.50)" and
    trailing minus "12.50-". If both separators appear, the LAST one is the decimal mark.
    A single separator type is a thousands mark when it repeats or is followed by exactly three
    digits ("1,234" -> 1234, "1.234.567" -> 1234567), otherwise a decimal mark ("12,5" -> 12.5).
    "1.234" is therefore read as 1234: a heuristic, and the one known ambiguous case.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    s = _NON_NUMERIC.sub("", unicodedata.normalize("NFKC", value)).strip()
    if not s:
        return None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative, s = True, s[1:-1]
    s = s.replace(" ", "").replace("'", "")
    if s.endswith("-"):
        negative, s = True, s[:-1]
    if s.startswith("-"):
        negative, s = True, s[1:]
    elif s.startswith("+"):
        s = s[1:]
    if not s or not re.fullmatch(r"[\d,.]+", s) or not re.search(r"\d", s):
        return None
    has_c, has_d = "," in s, "." in s
    if has_c and has_d:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thou = "." if dec == "," else ","
        if s.count(dec) > 1:
            return None
        s = s.replace(thou, "").replace(dec, ".")
    elif has_c or has_d:
        sep = "," if has_c else "."
        parts = s.split(sep)
        if len(parts) > 2 or (len(parts[1]) == 3 and parts[0] not in ("0", "")):
            s = "".join(parts)  # thousands separator
        else:
            s = parts[0] + "." + parts[1]
    try:
        out = float(s)
    except ValueError:
        return None
    return -out if negative else out


def numbers_match(a, b, tol: float = NUM_TOL) -> bool:
    """True if both parse and differ by at most ``tol`` (None only matches None)."""
    na, nb = parse_number(a), parse_number(b)
    if na is None or nb is None:
        return na is None and nb is None
    return abs(na - nb) <= tol + 1e-9


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def normalize_date(value, dayfirst: bool = False) -> str | None:
    """Canonicalize a date to ISO YYYY-MM-DD, or None if unparseable/invalid.

    Accepts ISO, "15 Mar 2024", "March 15, 2024", "15.03.2024" (dotted is always day-first), and
    slashed/dashed numeric dates "03/15/2024". For ambiguous numeric dates (both parts <= 12) the
    ``dayfirst`` flag decides (default US month-first); if one part exceeds 12 it is unambiguous.
    """
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        return None
    s = re.sub(r"\s+", " ", value.strip().rstrip(".")).lower()
    if not s:
        return None
    try:
        m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
        if m:
            return date(int(m[1]), int(m[2]), int(m[3])).isoformat()
        m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", s)
        if m:
            return date(int(m[3]), int(m[2]), int(m[1])).isoformat()
        m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", s)
        if m:
            a, b, y = int(m[1]), int(m[2]), int(m[3])
            if a > 12:
                d, mo = a, b
            elif b > 12:
                mo, d = a, b
            else:
                d, mo = (a, b) if dayfirst else (b, a)
            return date(y, mo, d).isoformat()
        m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?[ -]([a-z]{3,9}),?[ -](\d{4})", s)
        if m and m[2][:3] in _MONTHS:
            return date(int(m[3]), _MONTHS[m[2][:3]], int(m[1])).isoformat()
        m = re.fullmatch(r"([a-z]{3,9}) (\d{1,2})(?:st|nd|rd|th)?,? (\d{4})", s)
        if m and m[1][:3] in _MONTHS:
            return date(int(m[3]), _MONTHS[m[1][:3]], int(m[2])).isoformat()
    except ValueError:  # e.g. 31 Feb
        return None
    return None


def normalize_name(value) -> str | None:
    """Case-fold, NFKC-normalize and collapse whitespace; trailing '.' / ',' are ignored."""
    if value is None:
        return None
    s = unicodedata.normalize("NFKC", str(value))
    s = re.sub(r"\s+", " ", s).strip().casefold()
    return s.rstrip(".,").strip()


def normalize_invoice_number(value) -> str | None:
    """Uppercase with all whitespace removed ("inv- 001" == "INV-001")."""
    if value is None:
        return None
    return re.sub(r"\s+", "", str(value)).upper()


def normalize_currency(value) -> str | None:
    """Uppercase 3-letter code if it is in the supported set, else None."""
    if not isinstance(value, str):
        return None
    c = value.strip().upper()
    return c if c in CURRENCIES else None


def normalize_doc_type(value) -> str | None:
    """Map 'credit note' / 'credit-memo' / 'Invoice' etc. to the schema enum, else None."""
    if not isinstance(value, str):
        return None
    s = re.sub(r"[\s\-_]+", "_", value.strip().lower())
    if s in ("credit_note", "credit_memo", "creditnote", "credit"):
        return "credit_note"
    if s in DOCUMENT_TYPES:
        return s
    return None
