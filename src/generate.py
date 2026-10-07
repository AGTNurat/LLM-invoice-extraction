"""Seeded synthetic invoice generator (PDF + exact ground-truth JSON).

 No LLM is involved anywhere in this module.

Layout of the dataset (60 docs): 5 templates x 12 slots; slots 0-3 -> dev (20 docs), slots 4-11 ->
test (40 docs), so every template appears in both splits. Six clearly labelled "hard" cases occupy
fixed slots (see HARD): two in dev, four in test.

Noise levels (applied to labels/boilerplate only, NEVER to field values):
  0 none    - clean document
  1 low     - distractor lines (PO number, customer id, phone, IBAN) and boilerplate footer
  2 medium  - level 1 plus OCR-style character swaps in some labels ("Tota1"), scan artifacts,
              and a stamp line that may contain a distractor date

Ground-truth conventions are documented in src/schema.py. Two choices specific to the generator:
  * Tax-inclusive documents: line amounts are printed gross; subtotal (net) = total - tax_amount;
    discount is 0. The net amount is printed on only some of these documents.
  * Slash dates (15/03/2024 vs 03/15/2024) are unambiguous in normal documents (day >= 13). The one
    "ambiguous_date" hard case has an ambiguous invoice date resolved only by its (unambiguous) due date.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

DEFAULT_SEED = 42
TEMPLATES = ("classic_table", "stacked_meta", "totals_first", "freeform_lines", "sidebar")
N_SLOTS, DEV_SLOTS = 12, 4  # 5 templates x 4 = 20 dev docs, 5 x 8 = 40 test docs
CREDIT_SLOTS = (3, 7, 11)
# (template index, slot) -> hard case name
HARD = {
    (0, 0): "ambiguous_date",
    (1, 1): "multi_page",
    (2, 4): "credit_paren_eu",
    (3, 5): "eu_fractional_multiline",
    (4, 6): "no_tax_rate",
    (0, 8): "distractor_amounts",
}
HARD_DESCRIPTIONS = {
    "ambiguous_date": "numeric slash date with day <= 12; convention only inferable from the due date",
    "multi_page": "44 line items; totals on page 2",
    "credit_paren_eu": "credit note, negatives in parentheses, European number format, references another invoice",
    "eu_fractional_multiline": "European format, fractional quantities, multi-line descriptions, medium noise, discount",
    "no_tax_rate": "tax amount printed without a tax rate (truth tax_rate is null)",
    "distractor_amounts": "extra amounts that look like totals (previous balance, deposit, credit limit), medium noise",
}

DATE_FORMATS = ("iso", "dmy_slash", "mdy_slash", "dmy_dot", "d_mon_y", "month_d_y")
SLASH_FORMATS = ("dmy_slash", "mdy_slash")
SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£"}

VENDOR_A = ["Northwind", "Bluepeak", "Harbor", "Crestline", "Ironwood", "Silverline", "Maple", "Quartz", "Redwood",
            "Summit", "Tidewater", "Urban", "Vertex", "Willow", "Zenith", "Alder", "Beacon", "Cobalt", "Delta", "Ember"]
VENDOR_B = ["Traders", "Logistics", "Supplies", "Systems", "Industries", "Consulting", "Foods", "Print", "Tools",
            "Works", "Partners", "Solutions", "Media", "Energy", "Textiles"]
VENDOR_SUFFIX = {"us": ["Inc.", "LLC", "Ltd", "Corp."], "eu": ["GmbH", "BV", "AB", "SARL", "AG"]}
STREETS = ["Market Street", "Industrial Way", "Station Road", "Harbour Lane", "Mill Road", "Elm Avenue", "King Street"]
CITIES = ["Springfield", "Rotterdam", "Leeds", "Hamburg", "Lyon", "Toronto", "Malmo", "Austin", "Denver", "Bristol"]
CUSTOMERS = ["Acme Retail Ltd", "Orion Health Group", "Pinecrest Schools", "Lakeside Hotels", "Fairway Motors",
             "Granite Bank", "Harvest Grocers", "Juniper Labs", "Keystone Builders", "Lumen Studios"]

# (description, detail line, min price, max price, kind)
CATALOG = [
    ("Ergonomic office chair", "Model EC-220, black mesh", 89, 240, "unit"),
    ("A4 copy paper (box of 5)", "80gsm, FSC certified", 14, 32, "unit"),
    ("Laptop docking station", "USB-C, dual display", 70, 190, "unit"),
    ("Consulting services", "Senior analyst, on-site", 60, 160, "hours"),
    ("Cloud hosting - standard", "Monthly plan, 4 vCPU", 40, 220, "unit"),
    ("Safety gloves (pack of 12)", "Size L, EN 388", 9, 28, "unit"),
    ("LED desk lamp", "Warm white, dimmable", 18, 55, "unit"),
    ("Freight - pallet delivery", "Zone 3, tail-lift", 45, 140, "unit"),
    ("Software licence", "Annual subscription, per seat", 90, 380, "unit"),
    ("Stainless steel bolts M8", "Box of 200, A2 grade", 11, 38, "unit"),
    ("Training workshop", "Half day, up to 12 attendees", 300, 900, "unit"),
    ("Maintenance contract", "Quarterly service visit", 120, 480, "unit"),
    ("Network switch 24-port", "Managed, PoE+", 160, 520, "unit"),
    ("Cleaning services", "Weekly, office floor 2", 35, 90, "hours"),
    ("Packaging film roll", "500 mm x 300 m", 12, 36, "unit"),
    ("Toner cartridge", "Black, high yield", 45, 130, "unit"),
    ("Translation services", "EN to DE, per 100 words", 8, 22, "unit"),
    ("Pallet racking bay", "3 levels, 2700 kg", 210, 640, "unit"),
]
LABELS = {
    "invoice_number": ["Invoice No.", "Invoice #", "Invoice Number", "Inv. No."],
    "credit_number": ["Credit Note No.", "Credit Memo #", "Credit Note Number"],
    "invoice_date": ["Invoice Date", "Date", "Date of issue", "Issued"],
    "due_date": ["Due Date", "Payment due", "Payable by", "Due"],
    "subtotal": ["Subtotal", "Net amount", "Total before tax", "Sub-total"],
    "discount": ["Discount", "Less discount", "Volume discount"],
    "tax": ["Tax", "VAT", "Sales Tax", "GST"],
    "total": ["Total", "Total Due", "Amount Due", "Grand Total"],
    "total_credit": ["Total Credit", "Credit Total", "Total credited"],
    "incl_total": ["Total (incl. tax)", "Total incl. VAT", "Gross total"],
    "incl_tax": ["of which tax", "Includes tax", "of which VAT"],
}
TITLES = {"invoice": ["INVOICE", "Tax Invoice", "Invoice"], "credit_note": ["CREDIT NOTE", "Credit Memo"]}
OCR = {"o": "0", "l": "1", "O": "0", "S": "5", "I": "l", "e": "c"}

D = Decimal


def q2(x: Decimal) -> Decimal:
    return x.quantize(D("0.01"), rounding=ROUND_HALF_UP)


# formatting
def fmt_money(x: float, nf: str, neg: str = "minus") -> str:
    s = f"{abs(x):,.2f}"
    if nf == "eu":
        s = s.replace(",", "\0").replace(".", ",").replace("\0", ".")
    if x < 0:
        return f"-{s}" if neg == "minus" else f"({s})"
    return s


def fmt_qty(q: float, nf: str) -> str:
    s = f"{q:g}"
    return s.replace(".", ",") if nf == "eu" else s


def fmt_date(d: date, fmt: str) -> str:
    return {
        "iso": d.isoformat(),
        "dmy_slash": d.strftime("%d/%m/%Y"),
        "mdy_slash": d.strftime("%m/%d/%Y"),
        "dmy_dot": d.strftime("%d.%m.%Y"),
        "d_mon_y": f"{d.day} {d.strftime('%b %Y')}",
        "month_d_y": f"{d.strftime('%B')} {d.day}, {d.year}",
    }[fmt]


def ocr_noise(s: str, rng: random.Random) -> str:
    if rng.random() < 0.4:
        idx = [i for i, ch in enumerate(s) if ch in OCR]
        if idx:
            i = rng.choice(idx)
            s = s[:i] + OCR[s[i]] + s[i + 1:]
    return s


# record building
def _make_vendors(rng: random.Random, n: int) -> list[tuple[str, str]]:
    combos = list(itertools.product(VENDOR_A, VENDOR_B))
    rng.shuffle(combos)
    out = []
    for a, b in combos[:n]:
        region = "eu" if rng.random() < 0.4 else "us"
        out.append((f"{a} {b} {rng.choice(VENDOR_SUFFIX[region])}", region))
    return out


def _pick_dates(rng: random.Random, fmt: str, hard: str | None) -> tuple[date, date]:
    while True:
        inv = date(2023, 1, 1) + timedelta(days=rng.randrange(0, 1000))
        due = inv + timedelta(days=rng.choice([14, 30, 30, 45, 60]))
        if hard == "ambiguous_date":
            ok = inv.day <= 12 and inv.day != inv.month and due.day >= 13
        elif fmt in SLASH_FORMATS:
            ok = inv.day >= 13
        else:
            ok = True
        if ok:
            return inv, due


def _make_items(rng, n, sign, fractional, multiline):
    items, plan = [], []
    for _ in range(n):
        name, detail, lo, hi, kind = rng.choice(CATALOG)
        if kind == "hours" or (fractional and rng.random() < 0.6):
            qty = D(rng.randint(1, 24)) / D(2)
        else:
            qty = D(rng.randint(1, 20))
        price = q2(D(str(round(rng.uniform(lo, hi), 2))))
        amount = q2(qty * price) * sign
        lines = [name] + ([detail] if multiline and rng.random() < 0.8 else [])
        items.append({"description": " ".join(lines), "quantity": float(qty), "unit_price": float(price),
                      "amount": float(amount)})
        plan.append(lines)
    return items, plan


def _totals(items, sign, discount_pct, tax_rate, inclusive):
    amounts = [q2(D(str(i["amount"]))) for i in items]
    s = sum(amounts, D(0))
    if inclusive:
        tax = q2(s * D(str(tax_rate)) / (D(100) + D(str(tax_rate))))
        return s - tax, D(0), tax, s
    disc = q2(s * D(str(discount_pct)) / D(100))
    tax = q2((s - disc) * D(str(tax_rate)) / D(100))
    return s, disc, tax, s - disc + tax


def _build_one(seed, g, slot, t_idx, split, vendor, region):
    template, hard = TEMPLATES[t_idx], HARD.get((t_idx, slot))
    rng = random.Random(f"{seed}-{g}")
    credit = (slot in CREDIT_SLOTS and hard is None) or hard == "credit_paren_eu"
    sign = -1 if credit else 1
    eu = region == "eu" if rng.random() < 0.7 else rng.random() < 0.35
    if hard in ("credit_paren_eu", "eu_fractional_multiline"):
        eu = True
    nf = "eu" if eu else "us"
    currency = (rng.choices(["EUR", "CHF", "SEK", "DKK"], [70, 10, 10, 10])[0] if eu
                else rng.choices(["USD", "GBP", "CAD", "AUD"], [55, 20, 15, 10])[0])
    date_fmt = rng.choice(["dmy_slash", "dmy_dot", "d_mon_y", "iso"] if eu else
                          ["mdy_slash", "month_d_y", "iso", "d_mon_y"])
    if hard == "ambiguous_date":
        date_fmt = rng.choice(SLASH_FORMATS)
    inv_d, due_d = _pick_dates(rng, date_fmt, hard)
    no_due = credit or (rng.random() < 0.1 and hard != "ambiguous_date")
    inclusive = rng.random() < 0.25 and hard not in ("eu_fractional_multiline", "no_tax_rate", "multi_page")
    discount_pct = 0 if inclusive else rng.choice([0, 0, 0, 5, 10, 15])
    if hard in ("eu_fractional_multiline", "credit_paren_eu"):
        discount_pct = 10
    multiline = rng.random() < 0.35 or hard == "eu_fractional_multiline"
    if hard == "multi_page":
        multiline = False
    noise = rng.choices([0, 1, 2], [0.4, 0.35, 0.25])[0]
    if hard in ("eu_fractional_multiline", "distractor_amounts"):
        noise = 2
    tax_rate = rng.choice([0, 5, 7, 8.25, 10, 19, 20, 21, 25])
    tax_rate_shown = hard != "no_tax_rate"
    n_items = 44 if hard == "multi_page" else rng.randint(1, 7)
    items, item_lines = _make_items(rng, n_items, sign, fractional=hard == "eu_fractional_multiline",
                                    multiline=multiline)
    sub, disc, tax, total = _totals(items, sign, discount_pct, tax_rate, inclusive)
    neg = "paren" if (credit and (hard == "credit_paren_eu" or rng.random() < 0.3)) else "minus"
    disc_neg_printed = (not credit) and rng.random() < 0.5
    net_printed = inclusive and rng.random() < 0.5
    sym_totals = currency in SYMBOLS and rng.random() < 0.4
    num_n = rng.randint(1, 9999)
    num_style = rng.randrange(4)
    yy, yyyy = inv_d.strftime("%y"), inv_d.year
    if credit:
        inv_no = [f"CN-{yy}-{num_n:04d}", f"CR{num_n:05d}", f"CN/{yyyy}/{num_n:04d}", f"C{num_n:04d}-{yy}"][num_style]
    else:
        inv_no = [f"INV-{yy}-{num_n:04d}", f"{yyyy}/{num_n:05d}", f"SI{num_n:06d}", f"A{num_n:04d}-{yy}"][num_style]

    truth = {
        "document_type": "credit_note" if credit else "invoice",
        "vendor_name": vendor,
        "invoice_number": inv_no,
        "invoice_date": inv_d.isoformat(),
        "due_date": None if no_due else due_d.isoformat(),
        "currency": currency,
        "line_items": items,
        "subtotal": float(sub),
        "discount": float(disc),
        "tax_rate": float(tax_rate) if tax_rate_shown else None,
        "tax_amount": float(tax),
        "total": float(total),
    }

    # rendering plan (strings exactly as printed)
    r = random.Random(f"{seed}-{g}-render")
    lab = {k: r.choice(v) for k, v in LABELS.items()}
    if noise == 2:
        lab = {k: ocr_noise(v, r) for k, v in lab.items()}
    money = lambda x: fmt_money(x, nf, neg)  # noqa: E731
    rate_s = f"{tax_rate:g}".replace(".", ",") if eu else f"{tax_rate:g}"
    cur_hdr = f" ({currency})"
    tot_val = money(truth["total"])
    if sym_totals:
        tot_val = f"{SYMBOLS[currency]} {tot_val}" if not eu else f"{tot_val} {SYMBOLS[currency]}"
    tax_label = lab["tax"] + (f" {rate_s}%" if tax_rate_shown else "")
    rows = []
    if inclusive:
        if net_printed:
            rows.append(["subtotal", lab["subtotal"], money(truth["subtotal"])])
        rows.append(["total", lab["total_credit"] if credit else lab["incl_total"], tot_val])
        rows.append(["tax_amount", lab["incl_tax"] + (f" {rate_s}%" if tax_rate_shown else ""),
                     money(truth["tax_amount"])])
    else:
        rows.append(["subtotal", lab["subtotal"], money(truth["subtotal"])])
        if discount_pct:
            dv = truth["discount"] if credit else (-truth["discount"] if disc_neg_printed else truth["discount"])
            rows.append(["discount", f"{lab['discount']} ({discount_pct}%)", money(dv)])
        rows.append(["tax_amount", tax_label, money(truth["tax_amount"])])
        rows.append(["total", lab["total_credit"] if credit else lab["total"], tot_val])
    meta = [[lab["credit_number" if credit else "invoice_number"], inv_no],
            [lab["invoice_date"], fmt_date(inv_d, date_fmt)]]
    if not no_due:
        meta.append([lab["due_date"], fmt_date(due_d, date_fmt)])
    meta.append(["Currency", currency])
    if r.random() < 0.5:
        r.shuffle(meta)
    base_order = [lab["credit_number" if credit else "invoice_number"], lab["invoice_date"]] + \
                 ([lab["due_date"]] if not no_due else []) + ["Currency"]
    meta_shuffled = [m[0] for m in meta] != base_order

    distractors: list[str] = []
    if noise >= 1:
        pool = [f"PO Number: {r.randint(4500000000, 4599999999)}", f"Customer ID: C-{r.randint(10000, 99999)}",
                f"Tel: +{r.choice([44, 49, 1, 31])} {r.randint(20, 99)} {r.randint(1000, 9999)} {r.randint(1000, 9999)}",
                f"Order No: {r.randint(10000, 99999)}", f"Ref: {yyyy}-{r.randint(1000, 9999)}-{r.randint(10, 99)}",
                "IBAN: DE89 3704 0044 0532 0130 00"]
        r.shuffle(pool)
        distractors = pool[: 2 if noise == 1 else 4]
    if hard == "distractor_amounts":
        distractors += [f"Previous balance: {money(round(r.uniform(300, 4000), 2))}",
                        f"Deposit received: {money(round(r.uniform(50, 500), 2))}",
                        f"Credit limit: {money(r.choice([5000, 10000, 25000]))}"]
        r.shuffle(distractors)
    if noise == 2:
        stamp = r.choice(["PAID", "COPY", "DRAFT", f"RECEIVED {fmt_date(inv_d + timedelta(days=r.randint(1, 6)), date_fmt)}"])
        distractors.append(f"[{stamp}]")
    footer = []
    if noise >= 1:
        footer.append("Thank you for your business. Payment by bank transfer only.")
    if noise == 2:
        footer.append("~~~~~ scanned copy ~~~~~   page 1 of 1")
    ref_lines = []
    if credit and (hard == "credit_paren_eu" or r.random() < 0.6):
        ref_no = f"INV-{yy}-{(num_n + 137) % 9999 + 1:04d}"
        ref_lines.append(f"Original invoice: {ref_no}")

    item_rows = [{"desc_lines": item_lines[i], "qty": fmt_qty(it["quantity"], nf),
                  "unit": money(it["unit_price"]), "amount": money(it["amount"])} for i, it in enumerate(items)]
    street = f"{r.randint(1, 240)} {r.choice(STREETS)}"
    plan = {
        "vendor": {"name": vendor, "addr": [street, f"{r.choice(CITIES)} {r.randint(10000, 99999)}"]},
        "bill_to": ["Bill To:", r.choice(CUSTOMERS), f"{r.randint(1, 99)} {r.choice(STREETS)}"],
        "title": r.choice(TITLES[truth["document_type"]]),
        "meta_rows": meta, "items": item_rows, "totals_rows": rows, "distractors": distractors,
        "ref_lines": ref_lines, "footer": footer, "currency": currency, "cur_hdr": cur_hdr,
        "row_labels": {"qty": r.choice(["Qty", "Quantity"]), "unit": "Unit Price", "amount": r.choice(["Amount", "Total"]),
                       "desc": r.choice(["Description", "Item"])},
    }
    printed = {"invoice_date": fmt_date(inv_d, date_fmt), "due_date": None if no_due else fmt_date(due_d, date_fmt)}
    for key, _label, val in rows:
        printed[key] = val
    printed.setdefault("subtotal", None)
    printed["discount"] = next((v for k, _l, v in rows if k == "discount"), None)
    printed["total"] = tot_val
    meta_out = {
        "template": template, "noise_level": noise, "hard": hard is not None, "hard_reason": hard,
        "hard_description": HARD_DESCRIPTIONS.get(hard),
        "features": {
            "document_type": truth["document_type"], "number_format": nf, "date_format": date_fmt,
            "tax_inclusive": inclusive, "has_discount": bool(discount_pct), "multiline_descriptions": multiline,
            "meta_shuffled": meta_shuffled, "negatives_in_parens": neg == "paren", "n_line_items": n_items,
            "has_due_date": not no_due, "tax_rate_shown": tax_rate_shown,
        },
        "printed": printed,
    }
    return {"split": split, "meta": meta_out, "truth": truth, "plan": plan}


def build_records(seed: int = DEFAULT_SEED) -> list[dict]:
    """Pure function of the seed: returns 60 records (no I/O)."""
    vendors = _make_vendors(random.Random(f"{seed}-vendors"), N_SLOTS * len(TEMPLATES))
    records, counters = [], {"dev": 0, "test": 0}
    for slot in range(N_SLOTS):
        for t_idx in range(len(TEMPLATES)):
            g = slot * len(TEMPLATES) + t_idx
            split = "dev" if slot < DEV_SLOTS else "test"
            rec = _build_one(seed, g, slot, t_idx, split, *vendors[g])
            rec["doc_id"] = f"{split}_{counters[split]:02d}"
            counters[split] += 1
            records.append(rec)
    return records


# PDF rendering
class Writer:
    top, bottom = 792, 56

    def __init__(self, path: Path):
        self.c = Canvas(str(path), pagesize=A4, invariant=1)  # invariant=1 -> byte-reproducible PDFs
        self.w, self.y, self.pages = A4[0], self.top, 1

    def _font(self, size, bold):
        self.c.setFont("Helvetica-Bold" if bold else "Helvetica", size)

    def text(self, x, s, size=9, bold=False, y=None):
        self._font(size, bold)
        self.c.drawString(x, self.y if y is None else y, s)

    def rtext(self, xr, s, size=9, bold=False, y=None):
        self._font(size, bold)
        self.c.drawRightString(xr, self.y if y is None else y, s)

    def ctext(self, xc, s, size=9, bold=False, y=None):
        self._font(size, bold)
        self.c.drawCentredString(xc, self.y if y is None else y, s)

    def hline(self, x1, x2, y=None):
        self.c.line(x1, self.y if y is None else y, x2, self.y if y is None else y)

    def ensure(self, need) -> bool:
        if self.y - need < self.bottom:
            self.c.showPage()
            self.y, self.pages = self.top, self.pages + 1
            return True
        return False

    def finish(self, footer_lines):
        for i, ln in enumerate(footer_lines):
            self.text(40, ln, 8, y=40 - 10 * i)
        self.c.save()


def _block(w, x, y, lines, size=9, bold=False, step=11):
    for ln in lines:
        w.text(x, ln, size, bold, y=y)
        y -= step
    return y


def _items_table(w, p, x_desc, xr_qty, xr_unit, xr_amt, row_h=13, pos=False):
    L, cur = p["row_labels"], p["cur_hdr"]
    heads = [L["desc"], L["qty"], L["unit"] + cur, L["amount"] + cur]

    def header():
        if pos:
            w.text(40, "Pos", 9, True)
        w.text(x_desc, heads[0], 9, True)
        w.rtext(xr_qty, heads[1], 9, True)
        w.rtext(xr_unit, heads[2], 9, True)
        w.rtext(xr_amt, heads[3], 9, True)
        w.y -= 4
        w.hline(40, xr_amt)
        w.y -= row_h

    w.ensure(60)
    header()
    for i, it in enumerate(p["items"], 1):
        if w.ensure(row_h * (len(it["desc_lines"]) + 1)):
            header()
        if pos:
            w.text(40, str(i), 9)
        w.text(x_desc, it["desc_lines"][0], 9)
        w.rtext(xr_qty, it["qty"])
        w.rtext(xr_unit, it["unit"])
        w.rtext(xr_amt, it["amount"])
        for extra in it["desc_lines"][1:]:
            w.y -= row_h - 3
            w.text(x_desc + 8, extra, 8)
        w.y -= row_h
    w.hline(40, xr_amt, w.y + row_h - 4)
    w.y -= 4


def _totals_right(w, p, xl, xr):
    w.ensure(len(p["totals_rows"]) * 15 + 10)
    for key, label, val in p["totals_rows"]:
        bold = key == "total"
        w.text(xl, label, 9, bold)
        w.rtext(xr, val, 9, bold)
        w.y -= 15


def _t_classic(w, p):
    v = p["vendor"]
    w.text(40, v["name"], 16, True, y=792)
    w.rtext(555, p["title"], 20, True, y=792)
    y_left = _block(w, 40, 776, v["addr"])
    ym = 770
    for lab, val in p["meta_rows"]:
        w.text(380, lab, 9, True, y=ym)
        w.text(470, val, 9, y=ym)
        ym -= 13
    ym = _block(w, 380, ym - 2, p["ref_lines"] + p["distractors"], 8)
    y = _block(w, 40, min(y_left, ym) - 14, p["bill_to"])
    w.y = y - 10
    _items_table(w, p, 40, 330, 430, 555)
    _totals_right(w, p, 380, 555)


def _t_stacked(w, p):
    w.ctext(w.w / 2, p["title"], 20, True, y=792)
    y = 765
    for lab, val in p["meta_rows"]:
        w.text(40, f"{lab}: {val}", 10, y=y)
        y -= 14
    y = _block(w, 40, y, p["ref_lines"], 9)
    y = _block(w, 40, y - 8, ["From:", p["vendor"]["name"]] + p["vendor"]["addr"])
    y = _block(w, 40, y - 6, p["bill_to"])
    y = _block(w, 40, y - 6, p["distractors"], 8)
    w.y = y - 10
    _items_table(w, p, 40, 300, 410, 555)
    w.ensure(len(p["totals_rows"]) * 15 + 10)
    for key, label, val in p["totals_rows"]:
        w.text(40, f"{label}: {val}", 10, key == "total")
        w.y -= 15


def _t_totals_first(w, p):
    v = p["vendor"]
    w.text(40, v["name"], 14, True, y=792)
    w.rtext(555, p["title"], 16, True, y=792)
    y = _block(w, 40, 778, v["addr"])
    y -= 8
    for lab, val in p["meta_rows"]:
        w.text(40, lab, 9, True, y=y)
        w.text(200, val, 9, y=y)
        y -= 13
    y = _block(w, 40, y - 2, p["ref_lines"], 9)
    w.text(40, "Summary", 11, True, y=y - 10)
    w.y = y - 26
    _totals_right(w, p, 40, 300)
    y = _block(w, 40, w.y - 4, p["bill_to"] + p["distractors"], 8)
    w.y = y - 10
    _items_table(w, p, 70, 330, 430, 555, pos=True)


def _t_freeform(w, p):
    v = p["vendor"]
    w.rtext(555, v["name"], 13, True, y=792)
    y = 779
    for ln in v["addr"]:
        w.rtext(555, ln, 9, y=y)
        y -= 11
    w.text(40, p["title"], 16, True, y=792)
    w.text(40, "   |   ".join(f"{lab}: {val}" for lab, val in p["meta_rows"]), 9, y=745)
    y = _block(w, 40, 731, p["ref_lines"] + p["bill_to"] + p["distractors"], 8)
    w.y = min(y, 700) - 10
    w.text(40, f"Items (amounts in {p['currency']})", 10, True)
    w.y -= 16
    for it in p["items"]:
        w.ensure(30)
        w.text(48, f"{it['qty']} x {it['desc_lines'][0]} @ {it['unit']} = {it['amount']}", 9)
        for extra in it["desc_lines"][1:]:
            w.y -= 10
            w.text(60, extra, 8)
        w.y -= 14
    w.y -= 6
    w.ensure(len(p["totals_rows"]) * 15 + 10)
    for key, label, val in p["totals_rows"]:
        bold = key == "total"
        font = "Helvetica-Bold" if bold else "Helvetica"
        gap = 330 - 40 - stringWidth(label, font, 9) - stringWidth(val, font, 9) - 12
        dots = "." * max(3, int(gap / stringWidth(".", font, 9)))
        w.text(40, f"{label} {dots}", 9, bold)
        w.rtext(330, val, 9, bold)
        w.y -= 15


def _t_sidebar(w, p):
    v = p["vendor"]
    w.text(40, v["name"], 14, True, y=792)
    y_left = _block(w, 40, 778, v["addr"])
    y_left = _block(w, 40, y_left - 8, p["bill_to"])
    w.c.rect(375, 560, 185, 245)
    w.text(385, p["title"], 14, True, y=786)
    y = 768
    for lab, val in p["meta_rows"]:
        w.text(385, lab, 8, y=y)
        w.text(385, val, 9, True, y=y - 10)
        y -= 25
    y = _block(w, 385, y, p["ref_lines"], 8)
    w.y = y - 6
    for key, label, val in p["totals_rows"]:
        bold = key == "total"
        w.text(385, label, 8, bold)
        w.rtext(552, val, 9, bold)
        w.y -= 13
    _block(w, 385, w.y - 4, p["distractors"], 7)
    w.y = min(y_left, 555) - 14  # items start below the sidebar box, so they can use the full width
    _items_table(w, p, 40, 330, 430, 555)


RENDERERS = {"classic_table": _t_classic, "stacked_meta": _t_stacked, "totals_first": _t_totals_first,
             "freeform_lines": _t_freeform, "sidebar": _t_sidebar}


def render_pdf(rec: dict, path: Path) -> int:
    w = Writer(path)
    RENDERERS[rec["meta"]["template"]](w, rec["plan"])
    w.finish(rec["plan"]["footer"])
    return w.pages


#  orchestration / IO
def generate_dataset(out_dir: Path, seed: int = DEFAULT_SEED) -> list[dict]:
    """Write PDFs, ground-truth JSON and manifest.csv under out_dir/{dev,test}/."""
    out_dir = Path(out_dir)
    records = build_records(seed)
    rows = []
    for rec in records:
        d = out_dir / rec["split"]
        d.mkdir(parents=True, exist_ok=True)
        pages = render_pdf(rec, d / f"{rec['doc_id']}.pdf")
        rec["meta"]["features"]["pages"] = pages
        gt = {"doc_id": rec["doc_id"], "split": rec["split"], "meta": rec["meta"], "truth": rec["truth"]}
        (d / f"{rec['doc_id']}.json").write_text(json.dumps(gt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        m = rec["meta"]
        rows.append({"doc_id": rec["doc_id"], "split": rec["split"], "template": m["template"],
                     "noise_level": m["noise_level"], "hard": m["hard"], "hard_reason": m["hard_reason"],
                     **m["features"], "currency": rec["truth"]["currency"]})
    pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)
    return records


def load_split(data_dir: Path, split: str) -> list[dict]:
    """Load ground-truth records for a split; each gets a 'pdf_path'. Sorted by doc_id."""
    out = []
    for jp in sorted((Path(data_dir) / split).glob("*.json")):
        rec = json.loads(jp.read_text(encoding="utf-8"))
        rec["pdf_path"] = str(jp.with_suffix(".pdf"))
        out.append(rec)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate the synthetic invoice dataset.")
    ap.add_argument("--out", default="data/generated")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = ap.parse_args(argv)
    recs = generate_dataset(Path(a.out), a.seed)
    n = {s: sum(r["split"] == s for r in recs) for s in ("dev", "test")}
    print(f"Wrote {len(recs)} documents to {a.out} (seed={a.seed}): {n}")


if __name__ == "__main__":
    main()
