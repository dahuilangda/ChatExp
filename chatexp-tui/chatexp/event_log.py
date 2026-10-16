"""Append-only audit event log for the terminal app (pi-agent event log parity).

Events are JSON lines persisted at ``<task_dir>/event-log-v1.jsonl`` with the
schema ``{v: 1, ts, kind, ...}``. The in-file sequence is the source of truth:
every append is a single small write (immediately consistent for readers), a
corrupt line never blocks reading, and logging failures never break the
conversation. Recovery replays mirror the plugin contract: only
``plan_executed`` entries with ``status == "pending"`` count as in-flight
writes; executed plans are never replayed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

EVENT_LOG_VERSION = 1
EVENT_LOG_FILENAME = "event-log-v1.jsonl"
MAX_READ_EVENTS = 2000
# Write-action types used by recovery replay to describe in-flight work.
WRITE_ACTION_TYPES = frozenset(
    {
        "create_project",
        "define_factors",
        "define_objective",
        "update_objective",
        "update_factor",
        "initialize_design",
        "append_results",
        "update_plan_experiment",
        "import_experiment_sheet",
        "generate_next_design",
        "skip_pending_designs",
        "refresh_scores",
    }
)


def event_log_path(task_dir: str | Path) -> Path:
    return Path(task_dir).expanduser() / EVENT_LOG_FILENAME


def append_event(task_dir: str | Path, kind: str, **fields: Any) -> dict[str, Any] | None:
    """Append one event. Best-effort: returns the entry, or None when the log
    is unavailable — logging must never break the caller's work."""
    try:
        path = event_log_path(task_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"v": EVENT_LOG_VERSION, "ts": int(time.time() * 1000), "kind": kind}
        entry.update(fields)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return entry
    except Exception:
        return None


def read_events(task_dir: str | Path, limit: int = MAX_READ_EVENTS) -> list[dict[str, Any]]:
    """Read events oldest-first, skipping corrupt lines."""
    path = event_log_path(task_dir)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events[-max(int(limit), 0):] if limit else events


def clear_event_log(task_dir: str | Path) -> None:
    try:
        event_log_path(task_dir).unlink(missing_ok=True)
    except OSError:
        pass


def replay_for_recovery(task_dir: str | Path) -> dict[str, Any] | None:
    """Summarize the log for session recovery. Returns None when there is no
    user ask to recover (fresh workspace)."""
    events = read_events(task_dir)
    if not events:
        return None
    user_asks = [
        str(event.get("text") or "")[:150]
        for event in events
        if event.get("kind") == "user_message"
    ][-10:]
    if not user_asks:
        return None
    key_findings = [
        str(event.get("fact") or "")[:200]
        for event in events
        if event.get("kind") == "observation"
    ][-8:]
    errors = [
        str(event.get("message") or "")[:150]
        for event in events
        if event.get("kind") == "error"
    ][-3:]
    pending_writes: list[str] = []
    for event in events:
        if event.get("kind") != "plan_executed" or event.get("status") != "pending":
            continue
        for action in event.get("actions") or []:
            action_type = str(action.get("type") or "")
            if action_type in WRITE_ACTION_TYPES:
                target = action.get("target")
                pending_writes.append(f"{action_type}:{target}" if target else action_type)
    pending_writes = pending_writes[-5:]
    last_budget = next(
        (event for event in reversed(events) if event.get("kind") == "budget"),
        None,
    )
    return {
        "recovered_at": int(time.time() * 1000),
        "user_asks": user_asks,
        "key_findings": key_findings,
        "errors": errors,
        "pending_writes": pending_writes,
        "last_budget_used": (
            {
                "input": last_budget.get("input"),
                "output": last_budget.get("output"),
                "used": last_budget.get("used"),
            }
            if last_budget
            else None
        ),
        "total_events": len(events),
    }


def summarize_usage(usage: dict[str, Any] | None) -> dict[str, int] | None:
    """Normalize an OpenAI-style usage block into {input, output, total}.

    Accepts snake_case and camelCase aliases; returns None when no usable
    numbers are present so callers keep their previous value instead of
    overwriting it with garbage.
    """
    if not isinstance(usage, dict):
        return None

    def _number(*keys: str) -> int | None:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        return None

    input_tokens = _number("prompt_tokens", "input_tokens", "input")
    output_tokens = _number("completion_tokens", "output_tokens", "output")
    total = _number("total_tokens", "total")
    if input_tokens is None and output_tokens is None and total is None:
        return None
    used = total if total is not None else (input_tokens or 0) + (output_tokens or 0)
    return {"input": input_tokens or 0, "output": output_tokens or 0, "total": used}


__all__ = [
    "EVENT_LOG_FILENAME",
    "EVENT_LOG_VERSION",
    "WRITE_ACTION_TYPES",
    "append_event",
    "clear_event_log",
    "event_log_path",
    "read_events",
    "replay_for_recovery",
    "summarize_usage",
]
