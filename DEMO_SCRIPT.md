# Demo script (~90 s)
1. (10 s) Open app; point out the **OFFLINE DEMONSTRATION** / LLM mode banner and the invoice documents tab (note filenames are not in date order).
2. (25 s) Click sample **1**, Run. Show timeline: list → search → extract → get → create → verify; Verification tab; AP records row.
3. (30 s) Tick *Inject transient failure*, click sample **2**, Run. Show failed `create_ap_record`, `retry` event, success on retry, verification PASSED; AP table has one new row.
4. (15 s) Click sample **3**, Run → "Needs clarification", no new row.
5. (10 s) Re-run sample 1 → existing record detected, no duplicate. Mention `python -m pytest -q`.


### Extra integrity check (optional, if time permits)
Seed the AP system with an older Northstar invoice, then use a scripted planner test to attempt to satisfy a request for the latest invoice using the old record. The agent must fail safely and must not report Completed. Automated regression: `test_preexisting_older_record_cannot_satisfy_latest_invoice_task`.
