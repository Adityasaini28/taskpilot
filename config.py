"""Central configuration. Secrets come only from environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INVOICES_DIR = Path(os.environ.get("TASKPILOT_INVOICES", BASE_DIR / "sample_data" / "invoices"))
DB_PATH = Path(os.environ.get("TASKPILOT_DB", BASE_DIR / "data" / "taskpilot.db"))
MAX_TOOL_CALLS = 10   # hard cap on tool executions (retries count)
MAX_RETRIES = 2       # retries for a retryable (transient) error
LLM_TIMEOUT_S = 30


def openai_api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY") or None


def openai_model() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
