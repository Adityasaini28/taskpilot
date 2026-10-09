"""TaskPilot - Autonomous Invoice Operations Worker (Streamlit UI)."""
import pandas as pd
import streamlit as st

import config
from agent import Agent, evidence
from ap_repository import APRepository
from llm_planner import LLMPlanner
from offline_planner import OfflinePlanner
from tools import ToolRegistry

st.set_page_config(page_title="TaskPilot", page_icon="🧭", layout="wide")
st.markdown("<style>.block-container{padding-top:1.5rem}</style>", unsafe_allow_html=True)

repo = APRepository(config.DB_PATH)
registry = ToolRegistry(config.INVOICES_DIR, repo)
api_key = config.openai_api_key()

SAMPLES = {
    "1 · Normal execution": "Find the latest invoice from Northstar Components, extract the invoice number, amount and due date, enter it into our AP system, and confirm that it has been saved.",
    "2 · Failure recovery (enable injection)": "Register the newest invoice from Cedar & Stone Logistics in the AP system.",
    "3 · Safety: missing supplier": "Process the latest invoice and register it.",
    "Extract only (no write)": "Find the latest invoice from Northstar Components and tell me the amount and due date.",
    "Read-only (do NOT register)": "Find the latest invoice from Northstar Components, but do NOT register it; just tell me the amount and due date.",
    "Incomplete invoice": "Register the latest invoice from Meridian Supplies in the accounts-payable system.",
    "Unknown supplier": "Process the newest invoice from Zenith Traders",
    "Embedded instruction in invoice": "Process the newest invoice from Atlas Office Systems",
    "Unsupported: payment": "Pay the latest invoice from Northstar Components",
}
st.session_state.setdefault("task", list(SAMPLES.values())[0])

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Controls")
    use_llm = st.checkbox("Use LLM mode", value=bool(api_key), disabled=not api_key,
                          help="Requires OPENAI_API_KEY in the environment.")
    inject = st.checkbox("Inject one transient write failure", help="First create_ap_record attempt fails for real; the agent must retry.")
    st.caption(f"Limits: {config.MAX_TOOL_CALLS} tool calls, {config.MAX_RETRIES} retries")
    st.divider()
    st.subheader("Sample tasks")
    for label, prompt in SAMPLES.items():
        st.button(label, key=f"s_{label}", width="stretch", on_click=lambda p=prompt: st.session_state.update(task=p))
    st.divider()
    st.subheader("Reset demo data")
    confirm = st.checkbox("I understand this clears AP records and audit history")
    if st.button("Reset database", disabled=not confirm):
        repo.reset()
        st.session_state.pop("last", None)
        st.rerun()

# ---------------- header ----------------
st.title("🧭 TaskPilot")
st.caption("Autonomous Invoice Operations Worker · synthetic data · local SQLite · no real payments")
if use_llm and api_key:
    st.success(f"Mode: LLM tool-calling ({config.openai_model()}). The model chooses tools at runtime.")
else:
    st.warning("Mode: OFFLINE DEMONSTRATION. A deterministic rule-based planner (not an LLM) drives the real tools. "
               "Set OPENAI_API_KEY for LLM mode.")

task = st.text_area("Task", key="task", height=90)
c1, c2 = st.columns([1, 5])
run = c1.button("▶ Run task", type="primary", width="stretch")


def execute(force_offline: bool = False):
    llm = use_llm and api_key and not force_offline
    planner = LLMPlanner(registry, api_key) if llm else OfflinePlanner()
    with st.spinner("Worker running…"):
        st.session_state["last"] = Agent(registry, planner, repo).run(st.session_state["task"], inject_failure=inject)


if run and task.strip():
    execute()

s = st.session_state.get("last")
if s:
    banner = {"Completed": st.success, "Failed": st.error}.get(s.status, st.warning)
    banner(f"**{s.status}** — {s.summary}")
    if s.error_type == "planner_error":
        st.button("Run offline demonstration workflow instead", on_click=execute, kwargs={"force_offline": True})
    if s.human_request:
        st.info(f"Clarification needed: {s.human_request['reason']}  \nMissing: {', '.join(s.human_request['missing']) or '-'}")

t_time, t_plan, t_ver, t_ap, t_docs, t_hist = st.tabs(
    ["Execution timeline", "Plan & state", "Verification", "AP records", "Invoice documents", "Audit history"])

ICON = {"goal": "🎯", "understand": "🧠", "plan": "🗺️", "decision": "🤔", "tool_result": "🔧", "retry": "🔁",
        "retries_exhausted": "⛔", "security_warning": "🛡️", "final_answer": "💬", "final_status": "🏁", "approval_required": "✋"}

def render_events(events):
    for e in events:
        head = f"{ICON.get(e['event_type'], '•')} #{e['step']} {e['event_type']}" + (f" · {e['tool']}" if e.get("tool") else "")
        if e.get("retry"):
            head += f" · retry {e['retry']}"
        failed = bool(e.get("error"))
        with st.expander(("❌ " if failed else "") + head, expanded=e["event_type"] in ("retry", "final_status")):
            st.write({k: v for k, v in e.items() if k in ("timestamp", "args", "outcome", "error", "retry") and v})

with t_time:
    if s: render_events(s.events)
    else: st.write("Run a task to see planning, tool calls, observations, retries and verification.")
with t_plan:
    if s:
        st.write("**Objectives**"); [st.write(f"- {p}") for p in s.plan]
        st.write("**Worker state**")
        st.json({"intent": s.intent, "requested_supplier": s.required_supplier, "supplier": s.supplier, "selected_file": s.selected_file, "fields": s.fields,
                 "record_id": (s.record or {}).get("id"), "tool_calls": s.tool_calls, "retries": s.retries,
                 "candidates": [f"{c['filename']} ({c['invoice_date']})" for c in s.candidates]})
with t_ver:
    if s and s.verification:
        v = s.verification
        (st.success if v["passed"] else st.error)("Independent SQLite readback: " + ("PASSED" if v["passed"] else "FAILED"))
        st.json({"checked_fields": v.get("checked_fields"), "mismatches": v["mismatches"], "database_record": v["record"]})
    elif s: st.write("No verification performed in this run.")
    if s: st.write("**Evidence**"); st.json(evidence(s))
with t_ap:
    st.dataframe(pd.DataFrame(repo.list_records()), width="stretch", hide_index=True)
    st.caption(f"{repo.count()} record(s) in {config.DB_PATH}")
with t_docs:
    r = registry.execute("list_invoice_documents", {})
    st.dataframe(pd.DataFrame(r.data["documents"]), width="stretch", hide_index=True)
    st.caption("Documents are untrusted data. 'valid=False' means required fields are missing/invalid.")
with t_hist:
    runs = repo.list_runs()
    if runs:
        st.dataframe(pd.DataFrame(runs)[["run_id", "created_at", "mode", "status", "task"]], width="stretch", hide_index=True)
        pick = st.selectbox("Inspect run", [r["run_id"] for r in runs])
        render_events([{**e, "event_type": e["event_type"], "tool": e["tool"], "retry": e["retry"] or None,
                        "args": e["args"] if e["args"] != "null" else None,
                        "outcome": e["outcome"] if e["outcome"] != "null" else None} for e in repo.get_events(pick)])
    else:
        st.write("No runs yet.")
