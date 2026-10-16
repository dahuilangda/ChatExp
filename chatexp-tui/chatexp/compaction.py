"""History compaction with chained summarization (pi-agent mediaPipeline parity).

The planner payload keeps the historical keys ``history`` (verbatim window) and
``history_digest`` (compressed earlier turns), but the digest is upgraded to a
chained summary: a prior summary is always carried forward as the base, older
turns are distilled into it (user asks verbatim-short, assistant findings only
when they carry numbers or conclusion keywords), and the saturation guard
truncates the PRIOR summary text rather than the new findings.

Modes:
  * ``full``            — verbatim window 10 turns x 2000 chars
  * ``token_shrunk``    — 4 turns x 800 chars (last input exceeded the token budget)
  * ``overflow_minimal``— 2 turns x 500 chars (context-overflow retry)
"""

from __future__ import annotations

import re
from typing import Any

WINDOW_TURNS = {"full": 10, "token_shrunk": 4, "overflow_minimal": 2}
WINDOW_CHARS = {"full": 2000, "token_shrunk": 800, "overflow_minimal": 500}
SUMMARY_TURN_THRESHOLD = 4
SUMMARY_CAP_CHARS = 1200
PRIOR_SUMMARY_BUDGET_CHARS = SUMMARY_CAP_CHARS // 2
USER_ASK_SLICE = 120
USER_ASK_LIMIT = 5
FINDING_SLICE = 200
FINDING_LIMIT = 3
FINDING_PATTERN = re.compile(r"[0-9]|结论|总结|建议|conclusion|recommend|result", re.IGNORECASE)
TRUNCATION_SUFFIX = "..."


def _truncate(text: str, limit: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + TRUNCATION_SUFFIX


def distill_summary_entry(history: list[dict[str, Any]], prior_summary: str | None) -> str | None:
    """Distill turns beyond the verbatim window into one chained summary.

    Returns None when there is nothing worth summarizing (no prior summary and
    fewer than SUMMARY_TURN_THRESHOLD older turns).
    """
    older = history  # caller passes the already-sliced turns beyond the window
    if prior_summary is None and len(older) < SUMMARY_TURN_THRESHOLD:
        return None

    user_asks: list[str] = []
    findings: list[str] = []
    for item in older:
        role = str(item.get("role") or "")
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            user_asks.append(_truncate(content, USER_ASK_SLICE))
        elif FINDING_PATTERN.search(content):
            findings.append(_truncate(content, FINDING_SLICE))

    parts: list[str] = []
    if prior_summary:
        # Saturation guard: the prior summary is budgeted to half the cap; when
        # space runs out the PRIOR text is truncated, never the new findings.
        budget = PRIOR_SUMMARY_BUDGET_CHARS
        prior = " ".join(str(prior_summary).split())
        if len(prior) > budget:
            prior = prior[:budget] + TRUNCATION_SUFFIX
        parts.append(f"[继承摘要] {prior}" if _contains_cjk(prior) else f"[carry-over] {prior}")
    if user_asks:
        parts.append(
            ("最近用户诉求: " if _contains_cjk(" ".join(user_asks)) else "recent user asks: ")
            + " | ".join(user_asks[-USER_ASK_LIMIT:])
        )
    if findings:
        parts.append(
            ("已确认的结论: " if _contains_cjk(" ".join(findings)) else "confirmed findings: ")
            + " | ".join(findings[-FINDING_LIMIT:])
        )
    if not parts:
        return None
    summary = "\n".join(parts)
    if len(summary) > SUMMARY_CAP_CHARS:
        summary = summary[:SUMMARY_CAP_CHARS] + TRUNCATION_SUFFIX
    return summary


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def build_history_window(
    history: list[dict[str, Any]],
    mode: str = "full",
    prior_summary: str | None = None,
) -> tuple[list[dict[str, str]], str | None]:
    """Split conversation history into (verbatim_window, chained_summary).

    The returned window items keep the {role, content} shape the planner
    payload already uses. ``mode`` selects the window size; ``prior_summary``
    is always chained forward as the base of the new summary.
    """
    if mode not in WINDOW_TURNS:
        mode = "full"
    turns = WINDOW_TURNS[mode]
    chars = WINDOW_CHARS[mode]
    if not history:
        return [], prior_summary

    recent = history[-turns:]
    older = history[:-turns] if len(history) > turns else []
    window = [
        {
            "role": str(item.get("role") or ""),
            "content": _truncate(str(item.get("content") or ""), chars),
        }
        for item in recent
    ]
    summary = distill_summary_entry(older, prior_summary)
    return window, summary


__all__ = [
    "build_history_window",
    "distill_summary_entry",
]
