"""OFFLINE DEMONSTRATION planner: a deterministic state machine for the invoice workflow.
It is NOT an LLM and does not reason; it reacts to tool observations with fixed rules."""
from __future__ import annotations

import re

from invoice_parser import normalize_name
from models import RunState, Step, ToolResult

NAME_RE = re.compile(r"\b(?:from|by|supplier|vendor)\s+((?:[A-Z][\w.'-]*)(?:\s+(?:&|[A-Z][\w.'-]*))*)")


class OfflinePlanner:
    mode = "offline"

    def initial_plan(self, state: RunState) -> list:
        plan = ["List invoice documents", "Identify supplier from the request (ask if missing)",
                "Search documents for the supplier and pick the newest by invoice date",
                "Extract and validate invoice fields"]
        if state.intent == "register":
            plan += ["Check AP system for an existing record", "Create the AP record (retry transient errors)",
                     "Read the record back and verify against the source invoice"]
        return plan

    def observe(self, step: Step, result: ToolResult) -> None:
        return None  # stateless: decisions are derived from RunState.calls

    @staticmethod
    def _last(state: RunState, tool: str, filename: str | None = None):
        for c in reversed(state.calls):
            if c["tool"] == tool and (filename is None or c["args"].get("filename") == filename):
                return c["result"]
        return None

    @staticmethod
    def _find_supplier(task: str, listing: ToolResult) -> str | None:
        nt = normalize_name(task)
        names = sorted({d["supplier"] for d in listing.data.get("documents", []) if d.get("supplier")}, key=len, reverse=True)
        for n in names:
            if normalize_name(n) in nt:
                return n
        m = NAME_RE.search(task)
        return m.group(1) if m else None

    @staticmethod
    def _ask(reason: str, missing: list) -> Step:
        return Step("request_human_input", {"reason": reason, "missing": missing}, "Cannot proceed safely without input")

    def next_step(self, state: RunState) -> Step:
        listing = self._last(state, "list_invoice_documents")
        if listing is None:
            return Step("list_invoice_documents", {}, "Discover available invoice documents")
        known = sorted({d["supplier"] for d in listing.data["documents"] if d.get("supplier")})
        # The orchestrator is the source of truth for task constraints; do not
        # independently re-parse the supplier differently in the offline planner.
        supplier = state.required_supplier or self._find_supplier(state.task, listing)
        if not supplier:
            return self._ask("Which supplier do you mean? The request does not name one. Known suppliers: "
                             + ", ".join(known), ["supplier"])
        search = self._last(state, "search_invoices")
        if search is None:
            return Step("search_invoices", {"supplier": supplier}, f"Find invoices for '{supplier}'")
        d = search.data
        if d.get("ambiguous"):
            return self._ask(f"'{supplier}' matches several suppliers: {', '.join(d['suppliers'])}. Which one?", ["supplier"])
        dated = [m for m in d.get("matches", []) if m["invoice_date"]]
        if not dated:
            return self._ask(f"No readable invoice found for '{supplier}'. Known suppliers: " + ", ".join(known),
                             ["valid supplier name"])
        if state.requested_invoice_number:
            explicit = [m for m in dated if m.get("invoice_number", "").casefold() == state.requested_invoice_number.casefold()]
            if not explicit:
                return self._ask(f"Invoice {state.requested_invoice_number!r} was not found for '{supplier}'.", ["invoice number"])
            target = explicit[0]["filename"]
        else:
            target = dated[0]["filename"]   # tool sorts by parsed invoice date, newest first
        ex = self._last(state, "extract_invoice_fields", target)
        if ex is None:
            return Step("extract_invoice_fields", {"filename": target}, "Newest invoice by invoice date; extract fields")
        if not ex.ok:
            return self._ask(f"Invoice {target} is incomplete or invalid ({ex.error}). I did not write anything.",
                             ex.data.get("missing", []) + ex.data.get("errors", []))
        f = ex.data["fields"]
        if state.intent != "register":
            return Step(None, message=f"Invoice {f['invoice_number']} from {f['supplier']}: amount {f['amount']} "
                                      f"{f['currency']}, due {f['due_date']} (source {target}).")
        got = self._last(state, "get_ap_record")
        if got is None:
            return Step("get_ap_record", {"supplier": f["supplier"], "invoice_number": f["invoice_number"]},
                        "Check for an existing AP record before writing")
        verify_args = {"supplier": f["supplier"], "invoice_number": f["invoice_number"], "filename": target}
        if got.data.get("found"):
            if self._last(state, "verify_ap_record") is None:
                return Step("verify_ap_record", verify_args, "Record already exists; verify it matches the invoice")
            return Step(None, message="Record already existed; no duplicate created.")
        create = self._last(state, "create_ap_record")
        if create is None:
            return Step("create_ap_record", {"filename": target}, "No record exists; create it")
        if not create.ok and create.error_type != "duplicate":
            return Step(None, message=f"Write failed: {create.error}")
        if self._last(state, "verify_ap_record") is None:
            return Step("verify_ap_record", verify_args, "Read the record back and compare")
        return Step(None, message="Write attempted and verification performed.")
