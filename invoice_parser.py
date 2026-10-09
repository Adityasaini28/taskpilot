"""Deterministic invoice text parser. Only reads known 'Label: value' lines,
so free text (e.g. notes containing instructions) can never become a field."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Optional

LABELS = {
    "supplier": ("supplier", "vendor", "supplier name"),
    "invoice_number": ("invoice number", "invoice no", "invoice #", "invoice id"),
    "invoice_date": ("invoice date", "date issued"),
    "due_date": ("due date", "payment due"),
    "total": ("total amount", "total", "amount due", "grand total", "total due"),
    "currency": ("currency",),
    "payment_terms": ("payment terms", "terms"),
}
LOOKUP = {a: k for k, aliases in LABELS.items() for a in aliases}
LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z #.]{0,30}?)\s*:\s*(.+?)\s*$")
DATE_FORMATS = ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%B %d, %Y", "%b %d, %Y")
NUMBER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_/]{1,39}")
AMOUNT_RE = re.compile(r"(?:([A-Za-z]{3})\s*)?[$€£₹]?\s*([\d,]+(?:\.\d{1,2})?)\s*(?:([A-Za-z]{3}))?")
STOP_WORDS = {"inc", "llc", "ltd", "co", "corp", "the", "and", "company", "pvt", "limited"}


def normalize_name(name: str) -> str:
    """Lowercase, '&'->'and', strip punctuation, collapse whitespace."""
    s = name.lower().replace("&", " and ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def name_tokens(name: str) -> set:
    return {t for t in normalize_name(name).split() if t not in STOP_WORDS}


def supplier_matches(query: str, supplier: str) -> bool:
    q, s = name_tokens(query), name_tokens(supplier)
    return bool(q) and (q <= s or s <= q)


def parse_date(value: str) -> Optional[str]:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def format_cents(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"


@dataclass
class ParsedInvoice:
    fields: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.missing and not self.errors


def parse_invoice_text(text: str) -> ParsedInvoice:
    found: dict[str, list[str]] = {}
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        label = " ".join(m.group(1).lower().replace(".", "").split())
        key = LOOKUP.get(label)
        if key:
            found.setdefault(key, []).append(m.group(2).strip())

    res = ParsedInvoice()
    raw: dict[str, str] = {}
    for key, values in found.items():
        if len(set(values)) > 1:
            res.errors.append(f"conflicting values for '{key}'")
        else:
            raw[key] = values[0]

    for key in ("supplier", "invoice_number", "invoice_date", "due_date", "total"):
        if key not in raw and not any(key in e for e in res.errors):
            res.missing.append("total_amount" if key == "total" else key)

    f = res.fields
    if "supplier" in raw:
        if len(raw["supplier"]) > 100:
            res.errors.append("supplier name too long")
        else:
            f["supplier"] = raw["supplier"]
    if "invoice_number" in raw:
        if NUMBER_RE.fullmatch(raw["invoice_number"]):
            f["invoice_number"] = raw["invoice_number"]
        else:
            res.errors.append(f"invalid invoice number '{raw['invoice_number'][:40]}'")
    for key in ("invoice_date", "due_date"):
        if key in raw:
            iso = parse_date(raw[key])
            if iso:
                f[key] = iso
            else:
                res.errors.append(f"unparseable {key} '{raw[key][:30]}'")
    code = raw.get("currency", "").upper() or None
    if "total" in raw:
        m = AMOUNT_RE.fullmatch(raw["total"])
        try:
            if not m:
                raise InvalidOperation
            cents = int(Decimal(m.group(2).replace(",", "")) * 100)
            if cents <= 0:
                raise InvalidOperation
            f["amount_cents"] = cents
            f["amount"] = format_cents(cents)
            code = code or (m.group(1) or m.group(3) or "").upper() or None
        except InvalidOperation:
            res.errors.append(f"invalid total amount '{raw['total'][:30]}'")
    if code and re.fullmatch(r"[A-Z]{3}", code):
        f["currency"] = code
    elif "total" in raw and "amount_cents" in f:
        res.missing.append("currency")
    elif code:
        res.errors.append(f"invalid currency '{code[:10]}'")
    if "payment_terms" in raw:
        f["payment_terms"] = raw["payment_terms"][:60]
    if f.get("due_date") and f.get("invoice_date") and f["due_date"] < f["invoice_date"]:
        res.errors.append("due date is before invoice date")
    return res
