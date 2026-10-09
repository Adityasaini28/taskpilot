"""Orchestrator: GOAL -> UNDERSTAND -> PLAN -> EXECUTE -> OBSERVE -> ADAPT -> VERIFY -> COMPLETE.
The orchestrator (not the planner) owns state, limits, retries, audit and final status."""
from __future__ import annotations

import re
import uuid
from dataclasses import asdict
from typing import Protocol

import config
from ap_repository import APRepository, now
from invoice_parser import normalize_name, supplier_matches
from models import (AWAITING_APPROVAL, COMPLETED, FAILED, NEEDS_CLARIFICATION, RUNNING, RunState, Step, ToolResult)
from safety import LATEST_RE, classify_task, resolve_supplier, sanitize_args
from tools import ToolRegistry


class PlannerError(Exception):
    """Raised by a planner when it cannot produce a next step (e.g. API failure)."""


class Planner(Protocol):
    mode: str
    def initial_plan(self, state: RunState) -> list: ...
    def next_step(self, state: RunState) -> Step: ...
    def observe(self, step: Step, result: ToolResult) -> None: ...


class Agent:
    def __init__(self, registry: ToolRegistry, planner: Planner, repo: APRepository,
                 max_tool_calls: int = config.MAX_TOOL_CALLS, max_retries: int = config.MAX_RETRIES):
        self.registry, self.planner, self.repo = registry, planner, repo
        self.max_tool_calls, self.max_retries = max_tool_calls, max_retries

    # ---------- trace ----------
    def _event(self, s: RunState, etype: str, tool=None, args=None, outcome=None, error=None, retry=None) -> None:
        s.step += 1
        ev = {"run_id": s.run_id, "step": s.step, "timestamp": now(), "event_type": etype, "tool": tool,
              "args": sanitize_args(args) if args else None, "outcome": outcome, "error": error, "retry": retry}
        s.events.append(ev)
        self.repo.log_event(ev)

    # ---------- main loop ----------
    def run(self, task: str, inject_failure: bool = False, approved_tools: set | None = None) -> RunState:
        s = RunState(run_id=uuid.uuid4().hex[:8], task=task.strip(), mode=self.planner.mode,
                     approved_tools=set(approved_tools or ()))
        self.registry.run_id = s.run_id
        self._event(s, "goal", outcome={"task": s.task[:300], "mode": s.mode, "failure_injection": inject_failure})
        policy = classify_task(s.task)
        s.intent = policy.intent
        # Inject only for an authorized write task. If no checkbox injection was
        # requested, preserve explicitly armed repository failures used by tests.
        # _end clears any unused injection before the next run can start.
        if inject_failure:
            self.repo.arm_transient_failure(1 if s.intent == "register" else 0)
        if policy.unsupported:
            return self._end(s, FAILED, f"Request declined: {policy.unsupported}", "unsupported_action")
        if policy.intent == "ambiguous":
            s.human_request = {"needs_input": True, "reason": policy.clarification, "missing": ["write_intent"]}
            return self._end(s, NEEDS_CLARIFICATION, policy.clarification, "ambiguous_intent")
        self._event(s, "understand", outcome={"intent": s.intent})
        ended = self._resolve_objective(s)
        if ended:
            return ended
        s.plan = self.planner.initial_plan(s)
        self._event(s, "plan", outcome={"plan": s.plan})

        for _ in range(self.max_tool_calls * 3):
            if s.tool_calls >= self.max_tool_calls:
                return self._end(s, FAILED, f"Stopped: tool-call limit ({self.max_tool_calls}) reached.", "budget_exhausted")
            try:
                step = self.planner.next_step(s)
            except PlannerError as e:
                return self._end(s, FAILED, f"Planner error ({s.mode} mode): {e}", "planner_error")
            if step.tool is None:
                self._event(s, "final_answer", outcome={"agent_message": step.message[:500]})
                s.summary = step.message
                break
            self._event(s, "decision", tool=step.tool, args=step.args, outcome={"thought": step.thought[:300]})
            spec = self.registry.tools.get(step.tool)
            if spec and spec.requires_approval and step.tool not in s.approved_tools:
                self._event(s, "approval_required", tool=step.tool, args=step.args)
                return self._end(s, AWAITING_APPROVAL, f"Approval required before running '{step.tool}'.", "approval")
            result = self._execute(s, step)
            self.planner.observe(step, result)
            self._update(s, step, result)
            if result.error_type == "budget_exhausted":
                return self._end(s, FAILED, f"Stopped: tool-call limit ({self.max_tool_calls}) reached.", "budget_exhausted")
            if step.tool == "request_human_input" and result.ok:
                s.human_request = result.data
                return self._end(s, NEEDS_CLARIFICATION, result.data["reason"], "clarification")
        else:
            return self._end(s, FAILED, "Stopped: iteration limit reached.", "budget_exhausted")
        return self._finalize(s)

    # ---------- task-level objective (independent of the planner) ----------
    def _resolve_objective(self, s: RunState):
        """Fix the requested supplier from the task text. Returns an ended RunState if clarification is needed."""
        r = self.registry.execute("list_invoice_documents", {})
        known = sorted({d["supplier"] for d in r.data.get("documents", []) if d.get("supplier")}) if r.ok else []
        res = resolve_supplier(s.task, known)
        self._event(s, "objective", outcome={"intent": s.intent, "supplier": res.supplier, "problem": res.problem,
                                             "known_suppliers": known})
        if res.supplier is None:
            reason = {"missing": "Which supplier do you mean? The request does not name one.",
                      "unknown": f"No local invoice matches supplier '{res.candidates[0] if res.candidates else 'the name given'}'.",
                      "ambiguous": f"The request matches several suppliers ({', '.join(res.candidates)}). Which one?"}[res.problem]
            reason += " Known suppliers: " + ", ".join(known)
            s.human_request = {"needs_input": True, "reason": reason, "missing": ["supplier"]}
            return self._end(s, NEEDS_CLARIFICATION, reason, "clarification")
        s.required_supplier, s.require_latest = res.supplier, bool(LATEST_RE.search(s.task))
        s.requested_invoice_number = _explicit_invoice_number(s.task)
        candidates = self.registry.invoice_candidates(s.required_supplier)
        if s.requested_invoice_number and not any(
                m.get("invoice_number", "").casefold() == s.requested_invoice_number.casefold()
                for m in candidates):
            reason = (f"I could not find invoice {s.requested_invoice_number!r} for "
                      f"{s.required_supplier!r}. No records were changed.")
            s.human_request = {"needs_input": True, "reason": reason, "missing": ["valid invoice number"]}
            return self._end(s, NEEDS_CLARIFICATION, reason, "invoice_not_found")
        if s.require_latest:
            latest = self.registry.latest_file(s.required_supplier)
            if latest is None:
                reason = (f"I cannot safely determine the latest invoice for {s.required_supplier!r} "
                          "because at least one matching document has no valid invoice date. "
                          "Please correct the invoice date or specify an invoice number.")
                s.human_request = {"needs_input": True, "reason": reason,
                                   "missing": ["valid invoice date or explicit invoice number"]}
                return self._end(s, NEEDS_CLARIFICATION, reason, "latest_invoice_ambiguous")
            latest_fields = self.registry.peek_fields(latest) or {}
            if s.requested_invoice_number and latest_fields.get("invoice_number", "").casefold() != s.requested_invoice_number.casefold():
                reason = (f"Your request names invoice {s.requested_invoice_number!r} but also asks for the latest invoice, "
                          f"which is {latest_fields.get('invoice_number')!r}. Please clarify which invoice to use.")
                s.human_request = {"needs_input": True, "reason": reason,
                                   "missing": ["invoice selection"]}
                return self._end(s, NEEDS_CLARIFICATION, reason, "conflicting_invoice_objective")
        return None

    def _guard(self, s: RunState, step: Step):
        """Enforce the user's objective before every tool call, independent of planner behavior."""
        a = step.args if isinstance(step.args, dict) else {}
        req = s.required_supplier

        def block(kind: str, msg: str) -> ToolResult:
            return ToolResult(False, {"required_supplier": req}, kind, msg)

        # Writes are only authorized for an unambiguous registration task.
        if step.tool == "create_ap_record":
            if s.intent != "register":
                return block("intent_violation", "writes are not permitted: the request is read-only")
            # Force the agent to inspect/validate the source before any side effect.
            if not s.fields or not s.selected_file:
                return block("objective_violation", "cannot write before extracting and validating the requested invoice")

        sup = a.get("supplier")
        if step.tool in ("search_invoices", "get_ap_record", "verify_ap_record"):
            if not isinstance(sup, str) or not supplier_matches(req or "", sup):
                return block("objective_violation", f"supplier '{str(sup)[:60]}' is not the requested supplier '{req}'")
            # Database keys should use the canonical supplier resolved from the user's task.
            if step.tool in ("get_ap_record", "verify_ap_record") and normalize_name(sup) != normalize_name(req or ""):
                return block("objective_violation", "AP lookup/verification must use the canonical requested supplier name")

        # The invoice number is part of the objective, not a free planner choice.
        if step.tool in ("get_ap_record", "verify_ap_record"):
            requested_number = ((s.fields or {}).get("invoice_number")
                                or s.requested_invoice_number)
            if not requested_number and s.require_latest and req:
                latest = self.registry.latest_file(req)
                latest_fields = self.registry.peek_fields(latest) if latest else None
                requested_number = (latest_fields or {}).get("invoice_number")
            asked_number = a.get("invoice_number")
            if requested_number and asked_number != requested_number:
                return block("objective_violation", f"invoice '{asked_number}' is not the requested invoice '{requested_number}'")

        fn = a.get("filename")
        file_tools = ("read_invoice_document", "extract_invoice_fields", "create_ap_record", "verify_ap_record")
        if isinstance(fn, str) and step.tool in file_tools:
            fields = self.registry.peek_fields(fn)
            if fields is not None and normalize_name(fields.get("supplier", "")) != normalize_name(req or ""):
                return block("objective_violation", f"file '{fn[:60]}' is not an invoice from the requested supplier '{req}'")
            if (fields is not None and s.requested_invoice_number
                    and fields.get("invoice_number", "").casefold() != s.requested_invoice_number.casefold()):
                return block("objective_violation", f"file '{fn[:60]}' does not contain explicitly requested invoice {s.requested_invoice_number!r}")
            if fields is not None and s.fields and step.tool in ("create_ap_record", "verify_ap_record"):
                mismatches = _field_mismatches(s.fields, fields)
                if mismatches:
                    return block("objective_violation", "file does not match the invoice already extracted: " + "; ".join(mismatches))
            if step.tool == "create_ap_record":
                # A write must use precisely the source that was extracted and validated.
                if fn != s.selected_file:
                    return block("objective_violation", "write source differs from the extracted invoice")
                mismatches = _field_mismatches(s.fields or {}, fields or {})
                if mismatches:
                    return block("objective_violation", "write source fields changed after extraction: " + "; ".join(mismatches))
            if s.require_latest and step.tool in file_tools:
                latest = self.registry.latest_file(req or "") if req else None
                if latest is None:
                    return block("objective_violation", f"cannot establish the latest invoice for '{req}' safely")
                if fn != latest:
                    return block("objective_violation", f"'{fn[:60]}' is not the newest invoice (by invoice date) from '{req}'")
        return None

    def _wrong_records(self, s: RunState):
        """Message if this run created any record for a supplier other than the requested one."""
        for r in self.repo.list_records():
            if r.get("run_id") == s.run_id and normalize_name(r["supplier"]) != normalize_name(s.required_supplier or ""):
                return f"a record for a different supplier ({r['supplier']}) was created during this run"
        return None

    def _objective_failure(self, s: RunState):
        """Independently verify that the result matches the entire original objective and source invoice."""
        req = s.required_supplier
        if not req:
            return "requested supplier was not resolved"
        wrong = self._wrong_records(s)
        if wrong:
            return wrong

        if not s.selected_file:
            return "no source invoice was selected"
        source_fields = self.registry.peek_fields(s.selected_file)
        if not source_fields:
            return f"selected source file {s.selected_file!r} could not be independently parsed"
        if normalize_name(source_fields.get("supplier", "")) != normalize_name(req):
            return f"selected source file {s.selected_file!r} is not an invoice from the requested supplier {req!r}"
        field_mismatches = _field_mismatches(s.fields or {}, source_fields)
        if field_mismatches:
            return "extracted fields no longer match the selected source: " + "; ".join(field_mismatches)
        if s.require_latest:
            latest = self.registry.latest_file(req)
            if latest is None or s.selected_file != latest:
                return f"selected source {s.selected_file!r} is not verified as the latest invoice for {req!r}"

        if s.intent == "register":
            expected = s.fields or {}
            rec = s.record or {}
            v = s.verification or {}
            vrec = v.get("record") or {}
            rid = rec.get("id")
            fresh = self.repo.get_by_id(rid) if rid is not None else None
            if not v.get("passed"):
                return "independent verification did not pass"
            if not rid or not fresh:
                return "the expected AP record does not exist in the database"
            if vrec.get("id") != rid or fresh.get("id") != rid:
                return "the saved record and independently verified record have different IDs"
            for label, actual in (("agent record", rec), ("verification record", vrec), ("fresh database readback", fresh)):
                mismatches = _field_mismatches(expected, actual)
                if mismatches:
                    return f"{label} does not match the requested source invoice: " + "; ".join(mismatches)
                if normalize_name(actual.get("supplier", "")) != normalize_name(req):
                    return f"{label} belongs to a different supplier than {req!r}"
            # If the task named an invoice explicitly, a different invoice number must never satisfy it.
            requested_number = _explicit_invoice_number(s.task)
            if requested_number and expected.get("invoice_number", "").casefold() != requested_number.casefold():
                return f"processed invoice {expected.get('invoice_number')!r} does not match explicitly requested invoice {requested_number!r}"
        elif normalize_name((s.fields or {}).get("supplier", "")) != normalize_name(req):
            return f"extracted supplier does not match the requested supplier {req!r}"
        else:
            requested_number = _explicit_invoice_number(s.task)
            if requested_number and (s.fields or {}).get("invoice_number", "").casefold() != requested_number.casefold():
                return f"extracted invoice {s.fields.get('invoice_number')!r} does not match explicitly requested invoice {requested_number!r}"
        return None

    def _execute(self, s: RunState, step: Step) -> ToolResult:
        blocked = self._guard(s, step)
        if blocked:
            s.tool_calls += 1                      # blocked attempts still consume budget (bounded loop)
            s.violations += 1
            self._event(s, "objective_violation", tool=step.tool, args=step.args,
                        outcome={"ok": False, "blocked": True}, error=f"{blocked.error_type}: {blocked.error}")
            s.calls.append({"tool": step.tool, "args": step.args, "result": blocked})
            return blocked
        attempt = 0
        while True:
            if s.tool_calls >= self.max_tool_calls:
                return ToolResult(False, error_type="budget_exhausted", error="tool-call limit reached")
            s.tool_calls += 1
            result = self.registry.execute(step.tool, step.args)
            self._event(s, "tool_result", tool=step.tool, args=step.args, retry=attempt or None,
                        outcome={"ok": result.ok, "data": _brief(result.data)},
                        error=f"{result.error_type}: {result.error}" if not result.ok else None)
            if not result.ok and result.retryable and attempt < self.max_retries:
                attempt += 1
                s.retries += 1
                self._event(s, "retry", tool=step.tool, retry=attempt,
                            outcome={"classification": "retryable transient error", "next_attempt": attempt + 1,
                                     "max_retries": self.max_retries})
                continue
            if not result.ok and result.retryable:
                self._event(s, "retries_exhausted", tool=step.tool, retry=attempt)
            if step.tool == "search_invoices" and result.ok:      # only the requested supplier is eligible
                ms = [m for m in result.data["matches"] if normalize_name(m["supplier"]) == normalize_name(s.required_supplier)]
                result.data = {**result.data, "matches": ms, "suppliers": sorted({m["supplier"] for m in ms}), "ambiguous": False}
            s.calls.append({"tool": step.tool, "args": step.args, "result": result})
            return result

    def _update(self, s: RunState, step: Step, r: ToolResult) -> None:
        if r.error_type in ("objective_violation", "intent_violation"):
            return                                  # blocked calls must not change task state
        a, d = step.args or {}, r.data or {}
        if step.tool == "search_invoices" and r.ok:
            s.supplier = a.get("supplier")
            s.candidates = d.get("matches", [])
        if step.tool in ("extract_invoice_fields", "create_ap_record") and isinstance(a.get("filename"), str):
            s.selected_file = a["filename"]
        if step.tool == "extract_invoice_fields" and r.ok:
            s.fields = d["fields"]
        if step.tool == "create_ap_record":
            if r.ok:
                s.record = d["record"]
            elif r.error_type == "duplicate":
                s.record, s.already_existed = d.get("existing_record"), True
        if step.tool == "get_ap_record" and d.get("found"):
            s.record, s.already_existed = d["record"], True
        if step.tool == "verify_ap_record" and r.ok:
            s.verification = d
        if d.get("suspicious_content"):
            self._event(s, "security_warning", tool=step.tool,
                        outcome={"note": "Document contains instruction-like text; treated as data and ignored."})

    # ---------- completion ----------
    def _finalize(self, s: RunState) -> RunState:
        v = s.verification
        wrong = self._wrong_records(s)
        if wrong:
            return self._end(s, FAILED, "Objective mismatch: " + wrong, "objective_mismatch")
        if s.intent == "register":
            if v and v.get("passed"):
                bad = self._objective_failure(s)
                if bad:
                    return self._end(s, FAILED, "Objective mismatch: " + bad, "objective_mismatch")
                return self._end(s, COMPLETED, "Invoice recorded and independently verified for the requested supplier.")
            if v:
                return self._end(s, FAILED, "Verification FAILED: " + "; ".join(v.get("mismatches", [])), "verification_failed")
            return self._end(s, FAILED, "Not completed: the write was not independently verified.", "unverified")
        if s.fields:
            bad = self._objective_failure(s)
            if bad:
                return self._end(s, FAILED, "Objective mismatch: " + bad, "objective_mismatch")
            return self._end(s, COMPLETED, "Invoice details extracted (no AP record requested).")
        return self._end(s, FAILED, "Could not extract invoice details.", "no_result")

    def _end(self, s: RunState, status: str, message: str, error_type: str | None = None) -> RunState:
        # Demo fault injection is scoped to one task run and must never leak forward.
        self.repo.arm_transient_failure(0)
        s.status, s.error_type = status, error_type
        s.summary = message + _evidence_text(s)
        self._event(s, "final_status", outcome={"status": status, "message": message})
        self.repo.save_run(s.run_id, s.task, s.mode, status, s.summary)
        return s



PERSISTED_FIELDS = ("supplier", "invoice_number", "invoice_date", "due_date", "amount_cents", "currency")


def _field_mismatches(expected: dict, actual: dict) -> list[str]:
    """Compare all fields that define invoice identity and persisted business values."""
    if not expected or not actual:
        return ["missing expected or actual invoice fields"]
    return [f"{key}: expected {expected.get(key)!r}, got {actual.get(key)!r}"
            for key in PERSISTED_FIELDS if expected.get(key) != actual.get(key)]


def _explicit_invoice_number(task: str) -> str | None:
    """Extract an explicit invoice identifier when users provide one in the task."""
    patterns = (r"\binvoice\s*(?:number|no\.?|#|id)\s*[:#=-]?\s*([A-Z0-9][A-Z0-9-]{2,})\b",
                r"\b(?:invoice|number)\s+([A-Z]{2,}[A-Z0-9-]*\d[A-Z0-9-]*)\b")
    for pattern in patterns:
        match = re.search(pattern, task, re.I)
        if match:
            return match.group(1).strip().rstrip(".,;")
    return None


def _brief(data: dict) -> dict:
    out = {}
    for k, v in (data or {}).items():
        out[k] = (v[:300] + "…") if isinstance(v, str) and len(v) > 300 else v
    return out


def evidence(s: RunState) -> dict:
    f, rec = s.fields or {}, s.record or {}
    return {"mode": s.mode, "source_file": s.selected_file, "supplier": rec.get("supplier") or f.get("supplier"),
            "invoice_number": rec.get("invoice_number") or f.get("invoice_number"),
            "invoice_date": f.get("invoice_date"), "due_date": f.get("due_date"),
            "amount": f.get("amount"), "currency": f.get("currency"),
            "ap_record_id": rec.get("id"), "already_existed": s.already_existed,
            "verification": None if not s.verification else ("PASSED" if s.verification["passed"] else "FAILED"),
            "retries": s.retries, "tool_calls": s.tool_calls}


def _evidence_text(s: RunState) -> str:
    e = {k: v for k, v in evidence(s).items() if v not in (None, False)}
    if s.status in (NEEDS_CLARIFICATION, AWAITING_APPROVAL) or len(e) <= 3:
        return ""
    return "\nEvidence: " + ", ".join(f"{k}={v}" for k, v in e.items())
