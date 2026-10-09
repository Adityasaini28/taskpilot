# Submission Notes — TaskPilot

**Summary:** Orchestrator (`agent.py`) + swappable planners + allowlisted tools + SQLite AP system. 32 pytest tests on real files/DB.

## Design decisions (and why)
- **Small allowlisted toolset:** limits what a model (or injected text) can do; easy to validate and audit.
- **Writes take only a filename:** `create_ap_record`/`verify_ap_record` re-parse the file, so model- or injection-supplied values can never be saved.
- **SQLite + unique constraint:** real persistence, duplicates blocked at DB level as well as by a pre-check.
- **Bounded retries/limits:** 10 tool calls, 2 retries — no infinite loops.
- **Independent verification:** fresh DB read compared to the source document; status `Completed` requires it.
- **Task objective enforced outside the planner:** requested supplier/intent are resolved from the task text; wrong-supplier or write-on-read-only calls are blocked and final success re-checks the supplier.
- **Orchestrator owns final status:** the model cannot declare success.

## Criteria → where to see it
| Criterion | Code | Demo |
|---|---|---|
| Autonomy | `llm_planner.py`, `offline_planner.py` | Timeline "decision" events; extract-only task skips DB write |
| Execution | `tools.py`, `ap_repository.py` | Files read; AP table row appears |
| Reliability | `ap_repository.arm_transient_failure`, `Agent._execute` | Scenario 2 |
| Verification | `ToolRegistry._verify`, `Agent._finalize` | Verification tab |
| Generalization | same loop for all suppliers/phrasings | Sample tasks for 4 suppliers |
| Engineering | modules, `tests/test_all.py` | `python -m pytest -q` |
| Product thinking | clarification, duplicates, refusals, evidence | Scenario 3, rerun scenario 1 |

## Model/API
OpenAI Chat Completions with tools; `OPENAI_MODEL` default `gpt-4o-mini`; 30 s timeout. **Live API not tested**; the LLM loop is tested with a scripted fake client.

## Limitations
Offline planner is rules, not reasoning. Text invoices only. No upload. Local single-user.
