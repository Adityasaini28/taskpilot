"""Safety helpers: path confinement, prompt-injection flags, task policy."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from invoice_parser import normalize_name, supplier_matches

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|rules)",
    r"disregard .{0,40}(instructions|policy|rules)",
    r"^\s*(system|assistant)\s*:",
    r"reveal .{0,40}(api key|secret|password|system prompt)",
    r"mark (all|every) .{0,30}(paid|approved)",
    r"delete (all|every|the) .{0,30}(record|invoice|table|database)",
    r"drop table",
    r"new instructions",
]
WRITE_RE = re.compile(r"\b(enter|register|process|save[sd]?|record|add|post|create|book|input|store|log|submit|write|insert)\b", re.I)
UNSUP_WORDS = r"(?:pay|payment|payments|paid|wire|transfer|delete|remove|refund|cancel)"
UNSUPPORTED_RE = re.compile(rf"\b{UNSUP_WORDS}\b", re.I)
# Negation: "do not / don't / never / without ... <write verb>" (active or passive form).
NEG_LEAD = r"(?:do\s*not|don[\u2019']?t|never|without|no\s+need\s+to|not\s+to|should\s*n[\u2019']?t|must\s+not)"
NEG_VERBS = (r"(?:enter(?:ed)?|regist(?:er|ered)|process(?:ed)?|sav(?:e|ed|es)|record(?:ed)?|add(?:ed)?|post(?:ed)?|"
             r"creat(?:e|ed)|book(?:ed)?|input|stor(?:e|ed)|log(?:ged)?|submit(?:ted)?|writ(?:e|ten)|insert(?:ed)?)")
NEG_RE = re.compile(rf"\b{NEG_LEAD}\s+(?:\w+\s+){{0,4}}?{NEG_VERBS}\b(?:\s+(?:it|this|that|anything))?", re.I)
NEG_UNSUP_RE = re.compile(rf"\b{NEG_LEAD}\s+(?:\w+\s+){{0,2}}?{UNSUP_WORDS}\b", re.I)
# Explicit read-only phrasing: "only extract", "just tell me", "read-only", "no writing", "without saving"...
ONLY_RE = re.compile(r"\b(?:(?:only|just|simply)\s+(?:(?:want|need)\s+(?:you\s+)?to\s+|to\s+)?"
                     r"(?:extract|tell|read|show|report|give|look|find|list|know|see|check|get|answer)"
                     r"|read[- ]?only|extract(?:ion)?[- ]only|information\s+only|no\s+writ\w*"
                     r"|without\s+(?:writ|sav|regist|record|enter)\w*)\b", re.I)
NO_WRITE_RE = re.compile(r"\bno\s+(?:\w+\s+){0,2}(?:writ\w*|insert\w*|saves?|saving|records?|registration)\b", re.I)
LATEST_RE = re.compile(r"\b(latest|newest|most recent|last)\b", re.I)
NAME_RE = re.compile(r"\b(?:from|by|supplier|vendor)\s+((?:[A-Z][\w.'-]*)(?:\s+(?:&|[A-Z][\w.'-]*))*)")


class SafetyError(Exception):
    pass


def safe_invoice_path(invoices_dir: Path, filename: str) -> Path:
    """Resolve a bare filename inside the invoice folder, or raise SafetyError."""
    if not isinstance(filename, str) or not filename or "/" in filename or "\\" in filename or ".." in filename:
        raise SafetyError("filename must be a bare file name inside the invoice folder")
    base = Path(invoices_dir).resolve()
    path = (base / filename).resolve()
    if path.parent != base or path.suffix.lower() != ".txt":
        raise SafetyError("only .txt files directly inside the invoice folder are permitted")
    if not path.is_file():
        raise SafetyError(f"file not found: {filename}")
    return path


def detect_injection(text: str) -> list[str]:
    hits = []
    for pat in INJECTION_PATTERNS:
        if re.search(pat, text, re.I | re.M):
            hits.append(pat)
    return hits


@dataclass
class TaskPolicy:
    intent: str                      # "register" | "extract" | "ambiguous" | "unsupported"
    unsupported: Optional[str] = None
    clarification: Optional[str] = None


def classify_task(task: str) -> TaskPolicy:
    """Resolve intent. Explicit negation / read-only wording beats generic write keywords;
    if both a write instruction and a read-only instruction remain, the intent is 'ambiguous'
    (the caller must ask, never write)."""
    cleaned = re.sub(r"payment terms?", "", task, flags=re.I)
    m = UNSUPPORTED_RE.search(NEG_UNSUP_RE.sub(" ", cleaned))
    if m:
        return TaskPolicy("unsupported", f"'{m.group(0)}' is outside this prototype's authorized capabilities "
                          "(it can only read invoices and create records in the local AP database).")
    stripped = NO_WRITE_RE.sub(" ", NEG_RE.sub(" ", cleaned))   # drop negated write instructions
    negated = stripped != cleaned
    read_only = bool(ONLY_RE.search(cleaned))
    stripped = ONLY_RE.sub(" ", stripped)            # "read-only", "without saving" are not write requests
    affirmative = bool(WRITE_RE.search(stripped))
    if (negated or read_only) and affirmative:
        return TaskPolicy("ambiguous", clarification=(
            "Your request mixes a write instruction with a read-only instruction. Should I only report the "
            "invoice details, or also record the invoice in the AP system? I have not written anything."))
    if negated or read_only:
        return TaskPolicy("extract")
    return TaskPolicy("register" if affirmative else "extract")


@dataclass
class SupplierResolution:
    supplier: Optional[str]
    problem: Optional[str] = None            # "missing" | "unknown" | "ambiguous"
    candidates: tuple = ()


def resolve_supplier(task: str, known: list) -> SupplierResolution:
    """Resolve the requested supplier from the user's task text alone (planner-independent)."""
    nt = normalize_name(task)
    named = sorted({n for n in known if normalize_name(n) in nt})
    if len(named) == 1:
        return SupplierResolution(named[0])
    if len(named) > 1:
        return SupplierResolution(None, "ambiguous", tuple(named))
    m = NAME_RE.search(task)
    if not m:
        return SupplierResolution(None, "missing")
    hits = sorted({n for n in known if supplier_matches(m.group(1), n)})
    if len(hits) == 1:
        return SupplierResolution(hits[0])
    return SupplierResolution(None, "ambiguous" if hits else "unknown", tuple(hits))


def sanitize_args(args: dict) -> dict:
    out = {}
    for k, v in (args or {}).items():
        if re.search(r"key|secret|token|password", str(k), re.I):
            out[k] = "[redacted]"
        elif isinstance(v, str):
            out[k] = v[:200]
        else:
            out[k] = v
    return out
