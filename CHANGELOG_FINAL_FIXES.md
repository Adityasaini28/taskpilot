# Final integrity fixes

- Fixed cross-record verification: successful completion now requires exact agreement between the selected source invoice, extracted fields, agent record, verifier readback, and a fresh database retrieval.
- Fixed stale-record success: a pre-existing older invoice cannot satisfy a task explicitly asking for the latest invoice.
- Tightened pre-write requirements: source must have been extracted and validated, and a write must use the same selected source.
- Strengthened low-level `verify_ap_record` to reject invoice-number/supplier arguments that do not match the source file.
- Cleared failure injection on ordinary/read-only runs and when resetting demo data.
- Added regression tests covering stale existing records, finalizer defense in depth, wrong lookup arguments, explicit invoice identifiers, supplier resolution, and failure-injection lifecycle.
- Improved UI state inspection with the requested supplier, latest constraint, already-existing flag, and blocked objective violations.
- Added exact invoice-number constraints and clarification for contradictory old-invoice/latest requests.
- Latest selection now fails safely if a matching supplier document has no parseable invoice date.
