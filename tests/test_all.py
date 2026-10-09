import shutil
import sqlite3
from pathlib import Path

import pytest

import config
from agent import Agent
from ap_repository import APRepository
from invoice_parser import parse_invoice_text
from llm_planner import LLMPlanner
from models import COMPLETED, FAILED, NEEDS_CLARIFICATION
from offline_planner import OfflinePlanner
from tools import ToolRegistry

NS = ("Find the latest invoice from Northstar Components, extract the invoice number, amount and due date, "
      "enter it into our AP system, and confirm that it has been saved.")
CS = "Register the newest invoice from Cedar & Stone Logistics in the AP system."


@pytest.fixture
def env(tmp_path):
    inv = tmp_path / "invoices"
    shutil.copytree(config.BASE_DIR / "sample_data" / "invoices", inv)
    repo = APRepository(tmp_path / "t.db")
    reg = ToolRegistry(inv, repo)
    return inv, repo, reg


def agent(env, **kw):
    _, repo, reg = env
    return Agent(reg, OfflinePlanner(), repo, **kw)


def events(s, t):
    return [e for e in s.events if e["event_type"] == t]


def test_latest_invoice_selected_by_date_not_filename(env):
    r = env[2].execute("search_invoices", {"supplier": "Northstar Components"})
    names = [m["filename"] for m in r.data["matches"]]
    assert names[0] == "northstar_invoice_m.txt"          # newest by date
    assert names != sorted(names, reverse=True)           # filename order would differ
    assert sorted(names)[-1] == "northstar_invoice_z.txt"


def test_extraction_fields():
    p = parse_invoice_text((config.BASE_DIR / "sample_data/invoices/northstar_invoice_m.txt").read_text())
    assert p.valid and p.fields["invoice_number"] == "NS-2026-0587"
    assert p.fields["amount"] == "18742.50" and p.fields["currency"] == "USD"
    assert p.fields["due_date"] == "2026-10-22"


def test_end_to_end_creates_and_verifies(env):
    s = agent(env).run(NS)
    assert s.status == COMPLETED and s.verification["passed"]
    rows = env[1].list_records()
    assert len(rows) == 1
    r = rows[0]
    assert (r["invoice_number"], r["amount_cents"], r["due_date"], r["source_file"]) == \
           ("NS-2026-0587", 1874250, "2026-10-22", "northstar_invoice_m.txt")
    assert s.record["id"] == r["id"]
    assert [e["tool"] for e in events(s, "decision")][:3] == ["list_invoice_documents", "search_invoices", "extract_invoice_fields"]


def test_independent_readback_via_raw_sqlite(env):
    s = agent(env).run(NS)
    con = sqlite3.connect(env[1].db_path)
    row = con.execute("SELECT invoice_number, amount_cents FROM ap_invoices WHERE id=?", (s.record["id"],)).fetchone()
    assert row == ("NS-2026-0587", 1874250)
    assert any(e["tool"] == "verify_ap_record" for e in events(s, "tool_result"))


def test_retry_after_injected_transient_failure(env):
    s = agent(env).run(CS, inject_failure=True)
    creates = [e for e in events(s, "tool_result") if e["tool"] == "create_ap_record"]
    assert len(creates) == 2
    assert creates[0]["outcome"]["ok"] is False and "transient_error" in creates[0]["error"]
    assert creates[1]["retry"] == 1 and creates[1]["outcome"]["ok"] is True
    assert len(events(s, "retry")) == 1
    assert env[1].count() == 1 and s.status == COMPLETED and s.retries == 1
    assert s.record["invoice_number"] == "CS-9102" and s.record["currency"] == "EUR"


def test_duplicate_prevention(env):
    agent(env).run(NS)
    s2 = agent(env).run(NS)
    assert env[1].count() == 1
    assert s2.already_existed and s2.status == COMPLETED
    assert not [e for e in events(s2, "decision") if e["tool"] == "create_ap_record"]
    r = env[2].execute("create_ap_record", {"filename": "northstar_invoice_m.txt"})
    assert not r.ok and r.error_type == "duplicate" and env[1].count() == 1


def test_missing_supplier_asks_for_clarification(env):
    s = agent(env).run("Process the latest invoice and register it.")
    assert s.status == NEEDS_CLARIFICATION and "supplier" in s.human_request["missing"]
    assert env[1].count() == 0


def test_unknown_supplier(env):
    s = agent(env).run("Process the newest invoice from Zenith Traders")
    assert s.status == NEEDS_CLARIFICATION and env[1].count() == 0


def test_incomplete_invoice_not_written(env):
    s = agent(env).run("Register the latest invoice from Meridian Supplies in the accounts-payable system.")
    assert s.status == NEEDS_CLARIFICATION and env[1].count() == 0
    assert "due_date" in s.human_request["missing"] and "total_amount" in s.human_request["missing"]
    r = env[2].execute("create_ap_record", {"filename": "meridian_supplies_invoice.txt"})
    assert not r.ok and r.error_type == "invalid_invoice" and env[1].count() == 0


def test_verification_mismatch_detected(env):
    inv, repo, reg = env
    orig = repo.create_invoice

    def tampered(*a, **k):
        rec = orig(*a, **k)
        con = sqlite3.connect(repo.db_path)
        con.execute("UPDATE ap_invoices SET amount_cents=1 WHERE id=?", (rec["id"],))
        con.commit(); con.close()
        return repo.get_by_id(rec["id"])
    repo.create_invoice = tampered
    s = agent(env).run(NS)
    assert s.status == FAILED and s.error_type == "verification_failed"
    assert s.verification["passed"] is False and "amount_cents" in s.summary


def test_prompt_injection_in_invoice_is_ignored(env):
    agent(env).run(NS)
    s = agent(env).run("Process the newest invoice from Atlas Office Systems")
    assert s.status == COMPLETED
    assert env[1].count() == 2                       # nothing deleted, nothing extra
    assert s.record["amount"] == "84500.00" and s.record["supplier"] == "Atlas Office Systems"
    assert events(s, "security_warning")
    assert {d["tool"] for d in events(s, "decision")} <= set(env[2].tools)


def test_tool_call_limit_enforced(env):
    s = agent(env, max_tool_calls=3).run(NS)
    assert s.status == FAILED and s.error_type == "budget_exhausted" and s.tool_calls <= 3
    assert env[1].count() == 0


def test_bounded_retries_then_fail_without_record(env):
    env[1].arm_transient_failure(10)
    s = agent(env, max_retries=2).run(CS)
    creates = [e for e in events(s, "tool_result") if e["tool"] == "create_ap_record"]
    assert len(creates) == 3 and s.retries == 2
    assert s.status == FAILED and env[1].count() == 0


def test_extract_only_does_not_write(env):
    s = agent(env).run("Find the latest invoice from Northstar Components and tell me the amount and due date")
    assert s.status == COMPLETED and env[1].count() == 0
    assert "18742.50" in s.summary and "2026-10-22" in s.summary


def test_unsupported_request_declined(env):
    s = agent(env).run("Pay the latest invoice from Northstar Components")
    assert s.status == FAILED and s.tool_calls == 0


def test_path_traversal_and_bad_args_rejected(env):
    reg = env[2]
    assert reg.execute("read_invoice_document", {"filename": "../config.py"}).error_type == "safety_violation"
    assert reg.execute("read_invoice_document", {"filename": "/etc/passwd"}).error_type == "safety_violation"
    assert reg.execute("run_sql", {"q": "DROP TABLE ap_invoices"}).error_type == "unknown_tool"
    assert reg.execute("search_invoices", {"supplier": "x", "extra": 1}).error_type == "invalid_arguments"


def test_llm_loop_with_fake_client(env):
    """LLM-path wiring test using a scripted fake client (NOT a real API call)."""
    from types import SimpleNamespace as NS_
    import json
    script = [("list_invoice_documents", {}), ("search_invoices", {"supplier": "Cedar & Stone Logistics"}),
              ("extract_invoice_fields", {"filename": "cedar_stone_invoice_1.txt"}),
              ("get_ap_record", {"supplier": "Cedar & Stone Logistics", "invoice_number": "CS-9102"}),
              ("create_ap_record", {"filename": "cedar_stone_invoice_1.txt"}),
              ("verify_ap_record", {"supplier": "Cedar & Stone Logistics", "invoice_number": "CS-9102",
                                    "filename": "cedar_stone_invoice_1.txt"}), None]
    it = iter(enumerate(script))

    def create(**kw):
        i, item = next(it)
        if item is None:
            msg = NS_(content="Done.", tool_calls=None)
        else:
            tc = NS_(id=f"c{i}", function=NS_(name=item[0], arguments=json.dumps(item[1])))
            msg = NS_(content="plan" if i == 0 else None, tool_calls=[tc])
        return NS_(choices=[NS_(message=msg)])
    client = NS_(chat=NS_(completions=NS_(create=create)))
    inv, repo, reg = env
    s = Agent(reg, LLMPlanner(reg, "k", client=client), repo).run(CS)
    assert s.mode == "llm" and s.status == COMPLETED and repo.count() == 1


def test_llm_api_failure_reported_honestly(env):
    from types import SimpleNamespace as NS_
    def boom(**kw): raise TimeoutError("timed out")
    client = NS_(chat=NS_(completions=NS_(create=boom)))
    inv, repo, reg = env
    s = Agent(reg, LLMPlanner(reg, "k", client=client), repo).run(CS)
    assert s.status == FAILED and s.error_type == "planner_error" and "llm" in s.summary


# ======================= regression tests: intent + objective constraint =======================
from models import Step, ToolResult  # noqa: E402
from safety import classify_task  # noqa: E402

NS_ONLY = "Find the latest invoice from Northstar Components"
READ_ONLY_TASKS = [
    NS_ONLY + ", but do NOT register it; just tell me the amount and due date.",
    NS_ONLY + ". Do not save it, just give me the amount and due date.",
    "Only extract the invoice number, amount and due date of the newest invoice from Northstar Components.",
    NS_ONLY + " and tell me the amount. Please don't enter it into the AP system.",
    NS_ONLY + ", do not save it, only extract the amount.",
    "Read the newest invoice from Northstar Components without registering it and report the due date.",
    NS_ONLY + "; read-only, no database write.",
]


class ScriptedPlanner:
    """Deliberately scripted (possibly wrong) planner used to attack the orchestrator."""
    mode = "scripted"

    def __init__(self, steps):
        self.steps = list(steps)

    def initial_plan(self, state):
        return ["scripted"]

    def next_step(self, state):
        if self.steps:
            tool, args = self.steps.pop(0)
            return Step(tool, args, "scripted")
        return Step(None, message="All done, completed successfully.")

    def observe(self, step, result):
        pass


def seed_one(env):
    """Put one unrelated record in the DB so 'count unchanged' is a meaningful check."""
    s = agent(env).run(CS)
    assert s.status == COMPLETED and env[1].count() == 1


@pytest.mark.parametrize("task", READ_ONLY_TASKS)
def test_negated_or_only_extract_requests_never_write(env, task):
    assert classify_task(task).intent == "extract"
    seed_one(env)
    before = env[1].list_records()
    s = agent(env).run(task)
    assert s.status == COMPLETED and s.intent == "extract"
    assert env[1].count() == 1 and env[1].list_records() == before        # DB untouched
    assert not [e for e in s.events if e.get("tool") == "create_ap_record"]
    assert "18742.50" in s.summary and "2026-10-22" in s.summary


def test_do_not_save_variants_and_positive_intents_still_work():
    assert classify_task("Process the newest invoice from Cedar & Stone Logistics").intent == "register"
    assert classify_task(NS).intent == "register"
    assert classify_task("Find the invoice from Northstar Components, don't save it").intent == "extract"
    assert classify_task("Pay it, but do not delete anything").intent == "unsupported"
    assert classify_task("Don't pay it, just tell me the amount").intent == "extract"


def test_ambiguous_write_intent_does_not_write(env):
    task = "Do not register the newest invoice from Northstar Components, but save it in the AP system."
    assert classify_task(task).intent == "ambiguous"
    seed_one(env)
    s = agent(env).run(task)
    assert s.status == NEEDS_CLARIFICATION and s.tool_calls == 0 and env[1].count() == 1


def test_correct_extraction_only_request_never_writes_record(env):
    s = agent(env).run("Find the newest invoice from Cedar & Stone Logistics and tell me the amount and due date.")
    assert s.status == COMPLETED and s.fields["invoice_number"] == "CS-9102"
    assert env[1].count() == 0
    assert not [r for r in env[1].list_records() if r["run_id"] == s.run_id]
    assert "create_ap_record" not in {c["tool"] for c in s.calls}


def test_wrong_supplier_planner_cannot_complete_or_write(env):
    """Planner ignores the request (Northstar) and goes after Atlas: must fail safely, no Atlas row."""
    task = "Register the latest invoice from Northstar Components in the AP system."
    atlas = "atlas_office_systems_invoice.txt"
    steps = [("list_invoice_documents", {}), ("search_invoices", {"supplier": "Atlas Office Systems"}),
             ("extract_invoice_fields", {"filename": atlas}),
             ("get_ap_record", {"supplier": "Atlas Office Systems", "invoice_number": "AOS-7781"}),
             ("create_ap_record", {"filename": atlas}),
             ("verify_ap_record", {"supplier": "Atlas Office Systems", "invoice_number": "AOS-7781", "filename": atlas})]
    _, repo, reg = env
    s = Agent(reg, ScriptedPlanner(steps), repo).run(task)
    assert s.status in (FAILED, NEEDS_CLARIFICATION) and s.status != COMPLETED
    assert repo.count() == 0 and repo.search(supplier="Atlas") == []
    assert s.violations >= 4 and s.record is None
    assert s.required_supplier == "Northstar Components"


def test_final_verification_catches_wrong_supplier_even_if_guard_bypassed(env):
    """Defence in depth: with the pre-execution guard disabled, a verified Atlas record must NOT count as success."""
    task = "Register the latest invoice from Northstar Components in the AP system."
    atlas = "atlas_office_systems_invoice.txt"
    steps = [("extract_invoice_fields", {"filename": atlas}), ("create_ap_record", {"filename": atlas}),
             ("verify_ap_record", {"supplier": "Atlas Office Systems", "invoice_number": "AOS-7781", "filename": atlas})]
    _, repo, reg = env
    ag = Agent(reg, ScriptedPlanner(steps), repo)
    ag._guard = lambda s, step: None                       # simulate a guard bypass
    s = ag.run(task)
    assert s.verification["passed"] is True                # valid vs the Atlas file...
    assert s.status == FAILED and s.error_type == "objective_mismatch"   # ...but not the requested supplier


def test_scripted_planner_cannot_write_or_pick_old_invoice(env):
    _, repo, reg = env
    s = Agent(reg, ScriptedPlanner([("create_ap_record", {"filename": "northstar_invoice_m.txt"})]), repo).run(
        "Only extract the invoice details of the newest invoice from Northstar Components.")
    assert repo.count() == 0 and s.status == FAILED and s.violations == 1
    s = Agent(reg, ScriptedPlanner([("extract_invoice_fields", {"filename": "northstar_invoice_z.txt"}),
                                    ("create_ap_record", {"filename": "northstar_invoice_z.txt"})]), repo).run(
        "Register the latest invoice from Northstar Components in the AP system.")
    assert repo.count() == 0 and s.status == FAILED and s.violations == 2   # older invoice not eligible


def test_supplier_resolved_before_planner_runs(env):
    s = agent(env).run("Register the latest invoice from Northstar Components in the AP system.")
    assert s.required_supplier == "Northstar Components"
    assert events(s, "objective")[0]["outcome"]["supplier"] == "Northstar Components"
    before = env[1].count()
    s = Agent(env[2], ScriptedPlanner([("create_ap_record", {"filename": "atlas_office_systems_invoice.txt"})]),
              env[1]).run("Process the latest invoice and register it.")
    assert s.status == NEEDS_CLARIFICATION and env[1].count() == before and s.tool_calls == 0
