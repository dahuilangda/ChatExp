"""Planner-side agent harness (pi-agent parity for the terminal app).

Ports the seatable plugin's planner/orchestration/transport harness behaviors
to Python:

* JSON salvage ladder — fenced JSON, bare JSON, prose-wrapped first-{..last-};
  truncated output is never salvageable (returns None, never throws).
* Context-overflow classification with a rate-limit exclusion checked FIRST
  (a throttling message that mentions tokens must not burn the degraded retry).
* Quota-exhaustion fail-fast (phrase-scoped patterns, never retried).
* Retry-After handling: ``retry-after-ms`` header wins over ``Retry-After``
  (seconds or HTTP-date); requested waits are clamped to 30s; a server-asked
  delay beyond 60s fails fast.
* Per-request planner call budget (16 calls) counted at the single entry.
"""

from __future__ import annotations

import json
import re
import time
from email.utils import parsedate_to_datetime
from typing import Any

PLANNER_CALL_LIMIT = 16
MAX_RETRY_AFTER_SECONDS = 30.0
SERVER_DELAY_FAIL_FAST_SECONDS = 60.0

# Overflow signatures; the throttle exclusion is checked BEFORE these.
_RATE_LIMIT_PATTERN = re.compile(r"rate[ _-]?limit|too many requests|throttling|\b429\b", re.IGNORECASE)
_CONTEXT_OVERFLOW_PATTERN = re.compile(
    r"request_too_large"
    r"|context[_ ]?length"
    r"|input.{0,80}exceeds.{0,60}limit"
    r"|prompt.{0,40}too.{0,20}long"
    r"|maximum.{0,40}context"
    r"|too many tokens"
    r"|range of input length"
    r"|exceeds the context window"
    r"|上下文.{0,12}(超|过长|超出)",
    re.IGNORECASE,
)

# Phrase-scoped quota exhaustion: bare words like "billing" inside ordinary
# body text must not trigger fail-fast.
_QUOTA_PATTERN = re.compile(
    r"insufficient_quota"
    r"|quota[ _-]?exceed\w*"
    r"|exceeded[ _-]?your[ _-]?current[ _-]?quota"
    r"|billing[ _-]?(?:issue|hard[ _-]?limit)"
    r"|monthly[ _-]?usage[ _-]?limit[ _-]?reached"
    r"|available[ _-]?balance",
    re.IGNORECASE,
)


# ── JSON salvage ──────────────────────────────────────────────────────────────

def extract_json(text: str | None) -> dict[str, Any] | None:
    """Salvage a JSON object from an LLM reply.

    Order: strip a leading/trailing code fence → parse bare text → slice the
    first ``{`` to the last ``}`` (prose-wrapped JSON) → parse. Returns None
    when nothing parses — a truncated object has no closing brace and is
    therefore NOT salvageable here; callers treat None as repairable.
    """
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    candidates: list[str] = []
    fenced = _strip_code_fence(cleaned)
    if fenced:
        candidates.append(fenced)
    candidates.append(cleaned)
    braced = _slice_outermost_braces(cleaned)
    if braced:
        candidates.append(braced)
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _strip_code_fence(text: str) -> str | None:
    match = re.match(r"^```[a-zA-Z0-9]*\s*\n(.*)\n?```\s*$", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _slice_outermost_braces(text: str) -> str | None:
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    return text[first : last + 1]


# ── Error classification ─────────────────────────────────────────────────────

def is_context_overflow_error(error: BaseException | str) -> bool:
    message = error if isinstance(error, str) else str(error)
    if _RATE_LIMIT_PATTERN.search(message):
        return False
    return bool(_CONTEXT_OVERFLOW_PATTERN.search(message))


def is_quota_exhaustion_text(text: str | None) -> bool:
    if not text:
        return False
    return bool(_QUOTA_PATTERN.search(text))


class QuotaExhaustionError(RuntimeError):
    """Deterministic quota/billing exhaustion — the request was NOT retried."""

    def __init__(self, message: str):
        super().__init__(message)
        self.no_retry = True


class LLMTruncatedError(RuntimeError):
    """finish_reason == 'length' (or empty content) — the reply must never be
    executed; carries the tail of the cut text for the compact re-emit."""

    def __init__(self, message: str, truncated_tail: str = ""):
        super().__init__(message)
        self.truncated_tail = truncated_tail[-400:]


class ServerDelayFailFastError(RuntimeError):
    """Server asked for a wait longer than the fail-fast threshold."""

    def __init__(self, message: str):
        super().__init__(message)
        self.no_retry = True


# ── Transport helpers ────────────────────────────────────────────────────────

def parse_retry_after_headers(headers: Any) -> float | None:
    """Return the requested wait in seconds from retry-after-ms / Retry-After.

    ``retry-after-ms`` (milliseconds) takes precedence; ``Retry-After`` accepts
    delay-seconds or an HTTP-date. Raises ServerDelayFailFastError when the
    server asks for more than SERVER_DELAY_FAIL_FAST_SECONDS.
    """
    if headers is None:
        return None
    try:
        keys = {key.lower(): value for key, value in dict(headers).items()}
    except (TypeError, ValueError):
        return None
    ms = keys.get("retry-after-ms")
    if ms is not None:
        try:
            return _clamp_wait(float(str(ms).strip()) / 1000.0)
        except (TypeError, ValueError):
            pass
    raw = keys.get("retry-after")
    if raw is None:
        return None
    raw = str(raw).strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", raw):
        return _clamp_wait(float(raw))
    try:
        target = parsedate_to_datetime(raw)
        wait = target.timestamp() - time.time()
        return _clamp_wait(max(wait, 0.0))
    except (TypeError, ValueError, OverflowError):
        return None


def _clamp_wait(seconds: float) -> float:
    if seconds > SERVER_DELAY_FAIL_FAST_SECONDS:
        raise ServerDelayFailFastError(
            f"server requested a retry delay of {seconds:.0f}s (>{SERVER_DELAY_FAIL_FAST_SECONDS:.0f}s) — failing fast."
        )
    return min(max(seconds, 0.0), MAX_RETRY_AFTER_SECONDS)


# ── Planner call budget ──────────────────────────────────────────────────────

class BudgetExceededError(RuntimeError):
    def __init__(self, calls: int, limit: int):
        super().__init__(
            f"planner call budget exhausted: {calls} LLM calls for one user request (limit {limit}). "
            "Stopping to prevent a runaway loop; simplify the request or raise the limit."
        )
        self.calls = calls
        self.limit = limit


class PlannerCallBudget:
    """Counts every LLM call issued while handling one user request."""

    def __init__(self, limit: int = PLANNER_CALL_LIMIT):
        self.limit = int(limit)
        self.calls: list[str] = []

    def register(self, stage: str) -> int:
        if len(self.calls) >= self.limit:
            raise BudgetExceededError(len(self.calls), self.limit)
        self.calls.append(stage)
        return len(self.calls)

    @property
    def used(self) -> int:
        return len(self.calls)


__all__ = [
    "BudgetExceededError",
    "LLMTruncatedError",
    "MAX_RETRY_AFTER_SECONDS",
    "PLANNER_CALL_LIMIT",
    "PlannerCallBudget",
    "QuotaExhaustionError",
    "SERVER_DELAY_FAIL_FAST_SECONDS",
    "ServerDelayFailFastError",
    "extract_json",
    "is_context_overflow_error",
    "is_quota_exhaustion_text",
    "parse_retry_after_headers",
]
