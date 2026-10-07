"""
Conventions (shared by generator, extractor prompts, validator and evaluator):

* Dates are (YYYY-MM-DD) strings.
* ``tax_rate`` is a percentage (20.0 means 20%), or null when the document shows no tax rate.
* ``discount`` is an amount, 0 when absent. ``due_date`` is null when absent.
* ``document_type`` is "invoice" or "credit_note".
* Sign convention for credit notes: ``amount``, ``subtotal``, ``discount``, ``tax_amount`` and
  ``total`` are negative, ``quantity`` and ``unit_price`` stay positive. This keeps
  ``subtotal - discount + tax_amount == total`` true for both document types.
"""
from __future__ import annotations

DOCUMENT_TYPES = ("invoice", "credit_note")
CURRENCIES = ("USD", "EUR", "GBP", "CAD", "AUD", "CHF", "JPY", "INR", "SEK", "NOK", "DKK")

HEADER_STR_FIELDS = ("vendor_name", "invoice_number", "invoice_date", "due_date", "currency", "document_type")
HEADER_NUM_FIELDS = ("subtotal", "discount", "tax_rate", "tax_amount", "total")
# Header fields scored one by one, line_items is scored as a single composite field.
SCALAR_FIELDS = HEADER_STR_FIELDS + HEADER_NUM_FIELDS
ALL_FIELDS = SCALAR_FIELDS + ("line_items",)
LINE_ITEM_FIELDS = ("description", "quantity", "unit_price", "amount")
DATE_FIELDS = ("invoice_date", "due_date")
# Fields that may legitimately be null.
NULLABLE_FIELDS = ("due_date", "tax_rate")
REQUIRED_FIELDS = tuple(f for f in ALL_FIELDS if f not in NULLABLE_FIELDS)


def _nullable(t: str) -> dict:
    return {"anyOf": [{"type": t}, {"type": "null"}]}


def json_schema() -> dict:
    """JSON Schema passed to the model as the structured-output format."""
    return {
        "type": "object",
        "properties": {
            "document_type": {"type": "string", "enum": list(DOCUMENT_TYPES)},
            "vendor_name": {"type": "string"},
            "invoice_number": {"type": "string"},
            "invoice_date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
            "due_date": {**_nullable("string"), "description": "ISO date YYYY-MM-DD, or null if absent"},
            "currency": {"type": "string", "description": "ISO 4217 code, e.g. USD"},
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "number"},
                        "unit_price": {"type": "number"},
                        "amount": {"type": "number"},
                    },
                    "required": list(LINE_ITEM_FIELDS),
                    "additionalProperties": False,
                },
            },
            "subtotal": {"type": "number"},
            "discount": {"type": "number", "description": "discount amount, 0 if none"},
            "tax_rate": {**_nullable("number"), "description": "percentage, e.g. 20 for 20%; null if not shown"},
            "tax_amount": {"type": "number"},
            "total": {"type": "number"},
        },
        "required": list(ALL_FIELDS),
        "additionalProperties": False,
    }
