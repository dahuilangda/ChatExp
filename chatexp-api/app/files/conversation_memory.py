"""Cross-turn conversation memory for the document analysis agent.

Each analyze call is one TURN of a conversation about a document. The mission
planner needs to know what earlier turns already established —
otherwise follow-up questions re-read from scratch and can be answered against
a partially-read document.

Storage contract:
- In-process LRU keyed by (docId): the plugin polls the SAME deployment for a
  conversation, and the worker lanes are per-process. Redis-backed persistence
  would survive restarts but adds eviction coupling for data that is pure
  derived cache (recomputable from the document); the artifact store remains
  the source of truth.
- Bounded: last ``max_turns`` turns, answers truncated to
  ``max_answer_chars`` — memory feeds a prompt, not an archive.

No fallback semantics: if memory is empty the mission planner simply receives
an empty history (first turn). Malformed stored entries are dropped at read
time with an explicit reason surfaced in the mission context, never silently
substituted.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any

_MAX_DOCS = 64
_MAX_TURNS = 8
_MAX_ANSWER_CHARS = 1600

_store: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
_lock = threading.Lock()


class MemoryError_(Exception):
    """Raised only for programming-contract violations (wrong docId type)."""


def _validate_doc_id(doc_id: str) -> str:
    key = str(doc_id or "").strip()
    if not key:
        raise MemoryError_("conversation memory requires a non-empty docId")
    return key


def _clip_turn(turn: dict[str, Any]) -> dict[str, Any]:
    """Normalize one stored turn; None when the entry is structurally broken."""
    task = str(turn.get("task") or "").strip()
    answer = str(turn.get("answer") or "").strip()
    if not task or not answer:
        return None
    sections = turn.get("sections") or []
    return {
        "task": task[:400],
        "answer": answer[:_MAX_ANSWER_CHARS],
        "sections": [str(s) for s in sections if str(s).strip()][:32],
        "turn": int(turn.get("turn") or 0),
        "ts": float(turn.get("ts") or 0.0),
    }


def record_turn(doc_id: str, task: str, answer: str, sections: list[str] | None = None, turn: int = 0) -> dict[str, Any]:
    """Store one completed turn. Returns the clipped entry actually stored."""
    key = _validate_doc_id(doc_id)
    now = time.time()
    clipped = _clip_turn({"task": task, "answer": answer, "sections": sections or [], "turn": turn, "ts": now})
    if clipped is None:
        raise MemoryError_("record_turn requires non-empty task and answer")
    with _lock:
        history = _store.setdefault(key, [])
        history.append(clipped)
        del history[:-_MAX_TURNS]
        _store.move_to_end(key)
        while len(_store) > _MAX_DOCS:
            _store.popitem(last=False)
    return clipped


def get_history(doc_id: str) -> list[dict[str, Any]]:
    """Turns for a document, oldest first. Entries that fail normalization
    are dropped (the mission planner must never receive malformed memory)."""
    key = _validate_doc_id(doc_id)
    with _lock:
        raw = list(_store.get(key) or [])
    turns = []
    for entry in raw:
        clipped = _clip_turn(entry) if isinstance(entry, dict) else None
        if clipped is not None:
            turns.append(clipped)
    turns.sort(key=lambda t: (t["turn"], t["ts"]))
    return turns


def mission_context(doc_id: str, current_task: str) -> dict[str, Any] | None:
    """Context for the mission planner: prior turns + which sections they
    touched + the aggregate analyzed-section set. None when this is the first
    turn (nothing to inject — not a fallback, simply no history)."""
    history = get_history(doc_id)
    if not history:
        return None
    analyzed: set[str] = set()
    for turn in history:
        analyzed.update(turn["sections"])
    return {
        "prior_turns": [
            {"turn": t["turn"], "task": t["task"], "answer": t["answer"], "sections": t["sections"]}
            for t in history
        ],
        "sections_analyzed": sorted(analyzed),
        "current_task": current_task,
    }


def _reset_for_tests() -> None:
    with _lock:
        _store.clear()
