"""SQLite-backed simulated accounts-payable system + audit trail."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from invoice_parser import format_cents, normalize_name

SCHEMA = """
CREATE TABLE IF NOT EXISTS ap_invoices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    supplier TEXT NOT NULL,
    supplier_key TEXT NOT NULL,
    invoice_number TEXT NOT NULL,
    invoice_date TEXT NOT NULL,
    due_date TEXT NOT NULL,
    amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),
    currency TEXT NOT NULL,
    source_file TEXT NOT NULL,
    run_id TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (supplier_key, invoice_number)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, task TEXT, mode TEXT, status TEXT, summary TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, step INTEGER, ts TEXT,
    event_type TEXT, tool TEXT, args TEXT, outcome TEXT, error TEXT, retry INTEGER
);
"""


class TransientWriteError(Exception):
    """Simulated retryable write failure (injected for the demo)."""


class DuplicateRecordError(Exception):
    def __init__(self, existing: dict):
        super().__init__("invoice already exists")
        self.existing = existing


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class APRepository:
    def __init__(self, db_path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._fail_next = 0
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _rec(row) -> Optional[dict]:
        if row is None:
            return None
        d = dict(row)
        d["amount"] = format_cents(d["amount_cents"])
        return d

    # --- failure injection (consumed inside the real write path) ---
    def arm_transient_failure(self, times: int = 1) -> None:
        self._fail_next = times

    # --- operations ---
    def get_by_key(self, supplier: str, invoice_number: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM ap_invoices WHERE supplier_key=? AND invoice_number=?",
                            (normalize_name(supplier), invoice_number)).fetchone()
        return self._rec(row)

    def get_by_id(self, record_id: int) -> Optional[dict]:
        with self._conn() as c:
            return self._rec(c.execute("SELECT * FROM ap_invoices WHERE id=?", (record_id,)).fetchone())

    def exists(self, supplier: str, invoice_number: str) -> bool:
        return self.get_by_key(supplier, invoice_number) is not None

    def list_records(self) -> list[dict]:
        with self._conn() as c:
            return [self._rec(r) for r in c.execute("SELECT * FROM ap_invoices ORDER BY id")]

    def search(self, supplier: str = "", invoice_number: str = "") -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM ap_invoices WHERE supplier_key LIKE ? AND invoice_number LIKE ? ORDER BY id",
                             (f"%{normalize_name(supplier)}%", f"%{invoice_number}%")).fetchall()
        return [self._rec(r) for r in rows]

    def count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM ap_invoices").fetchone()[0]

    def create_invoice(self, fields: dict, source_file: str, run_id: str | None = None) -> dict:
        if self._fail_next > 0:                       # fails BEFORE any insert
            self._fail_next -= 1
            raise TransientWriteError("database is locked (simulated transient failure)")
        existing = self.get_by_key(fields["supplier"], fields["invoice_number"])
        if existing:
            raise DuplicateRecordError(existing)
        try:
            with self._conn() as c:
                cur = c.execute(
                    "INSERT INTO ap_invoices (supplier, supplier_key, invoice_number, invoice_date, due_date,"
                    " amount_cents, currency, source_file, run_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (fields["supplier"], normalize_name(fields["supplier"]), fields["invoice_number"],
                     fields["invoice_date"], fields["due_date"], fields["amount_cents"], fields["currency"],
                     source_file, run_id, now()))
                new_id = cur.lastrowid
        except sqlite3.IntegrityError:
            raise DuplicateRecordError(self.get_by_key(fields["supplier"], fields["invoice_number"]) or {})
        return self.get_by_id(new_id)

    # --- audit ---
    def log_event(self, ev: dict) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO audit_events (run_id, step, ts, event_type, tool, args, outcome, error, retry)"
                      " VALUES (?,?,?,?,?,?,?,?,?)",
                      (ev["run_id"], ev["step"], ev["timestamp"], ev["event_type"], ev.get("tool"),
                       json.dumps(ev.get("args"), default=str), json.dumps(ev.get("outcome"), default=str),
                       ev.get("error"), ev.get("retry")))

    def save_run(self, run_id, task, mode, status, summary) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?)", (run_id, task, mode, status, summary, now()))

    def list_runs(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM runs ORDER BY created_at DESC, rowid DESC LIMIT 50")]

    def get_events(self, run_id: str) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM audit_events WHERE run_id=? ORDER BY id", (run_id,))]

    def reset(self) -> None:
        """Clear demo tables and cancel any pending transient-failure injection."""
        self._fail_next = 0
        with self._conn() as c:
            c.executescript("DELETE FROM ap_invoices; DELETE FROM runs; DELETE FROM audit_events;"
                            "DELETE FROM sqlite_sequence;")
