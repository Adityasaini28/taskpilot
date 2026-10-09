# Demo script (~90 s)
1. (10 s) Open app; point out the **OFFLINE DEMONSTRATION** / LLM mode banner and the invoice documents tab (note filenames are not in date order).
2. (25 s) Click sample **1**, Run. Show timeline: list → search → extract → get → create → verify; Verification tab; AP records row.
3. (30 s) Tick *Inject transient failure*, click sample **2**, Run. Show failed `create_ap_record`, `retry` event, success on retry, verification PASSED; AP table has one new row.
4. (15 s) Click sample **3**, Run → "Needs clarification", no new row.
5. (10 s) Re-run sample 1 → existing record detected, no duplicate. Mention `python -m pytest -q`.
6. (Optional, 15 s) Click *Read-only (do NOT register)* → details extracted, AP table unchanged. Mention the regression tests in `tests/test_all.py` (wrong-supplier planner, read-only planner write attempt).
