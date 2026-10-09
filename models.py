"""Shared data models for the agent, tools and UI."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

COMPLETED = "Completed"
NEEDS_CLARIFICATION = "Needs clarification"
AWAITING_APPROVAL = "Awaiting approval"
FAILED = "Failed"
RUNNING = "Running"


@dataclass
class ToolResult:
    """Structured result returned by every tool."""
    ok: bool
    data: dict = field(default_factory=dict)
    error_type: Optional[str] = None
    error: Optional[str] = None
    retryable: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Step:
    """A planner decision: call a tool, or (tool=None) finish with a message."""
    tool: Optional[str]
    args: dict = field(default_factory=dict)
    thought: str = ""
    message: str = ""
    call_id: Optional[str] = None


@dataclass
class RunState:
    """Everything the worker remembers during one task."""
    run_id: str
    task: str
    mode: str
    intent: str = "unknown"            # "register" | "extract"
    supplier: Optional[str] = None
    required_supplier: Optional[str] = None   # resolved from the task, independent of the planner
    require_latest: bool = False
    violations: int = 0
    plan: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    selected_file: Optional[str] = None
    fields: Optional[dict] = None
    record: Optional[dict] = None
    already_existed: bool = False
    verification: Optional[dict] = None
    human_request: Optional[dict] = None
    calls: list = field(default_factory=list)   # {"tool","args","result": ToolResult}
    tool_calls: int = 0
    retries: int = 0
    status: str = RUNNING
    summary: str = ""
    error_type: Optional[str] = None
    events: list = field(default_factory=list)
    approved_tools: set = field(default_factory=set)
    step: int = 0
