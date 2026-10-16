"""Skill execution functions for the planner+skills+harness file analysis system.

Each skill runner is a pure function that takes parameters and state, and returns
a result. Data-only skills (read_manifest, read_section) extract data from the
parsed document without LLM calls. LLM skills (analyze_text, synthesize, answer)
call the configured LLM.

The LLM client functions (_call_llm, _llm_settings, _normalize_endpoint) are
shared with the planner module.
"""
from __future__ import annotations

import logging
import os
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from ..security import normalize_chat_completions_endpoint
from . import artifacts
from .skills import SINGLE_PASS_MAX_CHARS

logger = logging.getLogger(__name__)


class AnalyzeError(Exception):
    """Raised for any unrecoverable analysis failure."""


class ResponseTruncatedError(AnalyzeError):
    """finish_reason=length: the response body was cut at the output token
    limit. Carries the partial text so planner-level callers can re-enter once
    with a compact re-emit instruction; skill-level callers treat it as an
    ordinary error observation (the harness routes it back to the planner)."""

    def __init__(self, partial_text: str = "") -> None:
        super().__init__("model response truncated (finish_reason=length)")
        self.partial_text = str(partial_text or "")


# ---------------------------------------------------------------------------
# Shared HTTP client with connection pooling
# ---------------------------------------------------------------------------

_llm_client: httpx.Client | None = None


def _get_llm_client() -> httpx.Client:
    """Get or create a shared httpx.Client with connection pooling.

    Reusing a single client avoids TCP handshake + TLS negotiation on every
    LLM call, which matters when an analysis may involve 10-50 sequential calls.
    The per-request timeout is passed to client.post() explicitly.

    trust_env=False: the model endpoint is an allowlisted direct target;
    daemon-injected HTTP_PROXY env would otherwise route agent LLM calls
    through a corporate proxy that intermittently returns empty 502s.
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = httpx.Client(timeout=httpx.Timeout(120.0), trust_env=False)
    return _llm_client


# ---------------------------------------------------------------------------
# LLM client (shared with planner)
# ---------------------------------------------------------------------------


def _llm_settings() -> dict[str, str]:
    """Read LLM endpoint/key/model from the same env the proxy uses."""
    return {
        "endpoint": os.getenv("CHATEXP_LLM_ENDPOINT", "").strip(),
        "apiKey": os.getenv("CHATEXP_LLM_API_KEY", "").strip(),
        "model": os.getenv("CHATEXP_LLM_MODEL", "").strip(),
    }


def _normalize_endpoint(endpoint: str) -> str:
    try:
        return normalize_chat_completions_endpoint(endpoint)
    except ValueError as exc:
        raise AnalyzeError(f"LLM endpoint is not configured ({exc})") from exc


def _json_response_format_enabled() -> bool:
    """Whether JSON-contract calls may send response_format json_object.

    Defaults to ON; only an explicit falsey value of CHATEXP_LLM_RESPONSE_FORMAT
    disables it (mirrors the plugin's `!== false` pattern) because some
    OpenAI-compatible servers reject response_format with a 400. The
    non-JSON repair path in the planners stays in place either way."""
    value = str(os.getenv("CHATEXP_LLM_RESPONSE_FORMAT", "")).strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    return True


_RETRY_AFTER_CAP_SECONDS = 30.0


def _retry_delay_seconds(attempt: int, response: httpx.Response | None = None) -> float:
    """Exponential backoff with ~10% jitter: base * 2**(attempt-1) scaled by a
    uniform (0.9, 1.1) factor, so concurrent workers retrying the same
    upstream do not stampede in lockstep. When the server sent a Retry-After
    header — numeric seconds OR the HTTP-date form (RFC 7231) — it is honored
    instead, capped at 30s (never trust a server that asks for an hour). A
    date already in the past means "retry now" (0s)."""
    if response is not None:
        header = response.headers.get("Retry-After") if response.headers is not None else None
        if header:
            retry_after: float | None = None
            try:
                retry_after = float(header)
            except ValueError:
                retry_after = _retry_after_date_seconds(header)
            if retry_after is not None:
                return min(_RETRY_AFTER_CAP_SECONDS, max(0.0, retry_after))
    return 1.0 * (2 ** (attempt - 1)) * (0.9 + 0.2 * random.random())


def _retry_after_date_seconds(header: str) -> float | None:
    """Seconds until an HTTP-date Retry-After (e.g. "Wed, 21 Oct 2015 07:28:00
    GMT"), None for a date we cannot parse. Naive datetimes are read as UTC
    (the HTTP date format mandates GMT); a past date clamps to 0 at the
    caller."""
    try:
        when = parsedate_to_datetime(header)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (when - datetime.now(timezone.utc)).total_seconds()


def _call_llm(
    system_prompt: str,
    user_content: str,
    settings: dict[str, str] | None = None,
    max_tokens: int = 4096,
    timeout: float = 120.0,
    retries: int = 1,
    json_mode: bool = False,
) -> str:
    """Call the OpenAI-compatible chat completions endpoint.

    Transport tier: retries up to ``retries`` times with exponential backoff
    (plus jitter) on transient failures only — network errors, timeouts, 5xx,
    and 429 (rate limit; a numeric Retry-After header is honored, capped at
    30s). Other 4xx are semantic errors and fail immediately.

    ``json_mode=True`` marks a JSON-contract call (planners) and requests
    ``response_format: {"type": "json_object"}`` when enabled via env.

    Returns the assistant message content. Raises AnalyzeError on any failure.
    """
    s = settings or _llm_settings()
    if not s.get("endpoint") or not s.get("model"):
        raise AnalyzeError(
            "LLM is not configured for server-side analysis "
            "(need CHATEXP_LLM_ENDPOINT + CHATEXP_LLM_MODEL)."
        )
    endpoint = _normalize_endpoint(s["endpoint"])
    headers = {"Content-Type": "application/json"}
    if s.get("apiKey"):
        headers["Authorization"] = f"Bearer {s['apiKey']}"

    body: dict[str, Any] = {
        "model": s["model"],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    if json_mode and _json_response_format_enabled():
        body["response_format"] = {"type": "json_object"}

    last_exc: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            client = _get_llm_client()
            # client is a shared module-level instance: `with client:` would
            # close it after one request, so pass the timeout per request.
            response = client.post(endpoint, json=body, headers=headers, timeout=timeout)
            if response.status_code >= 500 or response.status_code == 429:
                # Transient upstream conditions: 5xx overload/failure (502/
                # 503/504) and 429 rate limiting. Retry with backoff + jitter
                # (honoring Retry-After) — other 4xx are semantic errors and
                # are NOT retried (they would fail identically).
                detail = response.text[:500]
                last_exc = AnalyzeError(
                    f"LLM returned HTTP {response.status_code}: {detail}"
                )
                logger.warning(
                    "LLM returned HTTP %d (attempt %d/%d): %s",
                    response.status_code,
                    attempt,
                    retries,
                    detail[:200],
                )
                if attempt < retries:
                    time.sleep(_retry_delay_seconds(attempt, response))
                    continue
                raise last_exc
            if response.status_code >= 400:
                detail = response.text[:500]
                raise AnalyzeError(f"LLM returned HTTP {response.status_code}: {detail}")
            try:
                data = response.json()
            except ValueError as exc:
                raise AnalyzeError("LLM returned a non-JSON response.") from exc
            choices = data.get("choices")
            if not choices or not isinstance(choices, list) or len(choices) == 0:
                # Check for known error response formats
                error_info = data.get("error", {})
                if error_info:
                    raise AnalyzeError(
                        f"LLM returned error: {error_info.get('message', str(error_info))[:500]}"
                    )
                raise AnalyzeError(
                    f"LLM response has no 'choices' array. Keys: {list(data.keys())}"
                )
            choice = choices[0] if isinstance(choices[0], dict) else {}
            message = choice.get("message") or {}
            if not isinstance(message, dict):
                raise AnalyzeError(
                    f"LLM response 'message' field is not a dict: {type(message).__name__}"
                )
            finish_reason = choice.get("finish_reason")
            content = message.get("content", "")
            if finish_reason == "length":
                # A length-cut body can still parse into a structurally valid
                # but silently incomplete result. Fail as an explicit
                # transport-tier error (no retry) so the errors-as-observations
                # loop repairs it once; missing finish_reason / "stop" is fine.
                # The partial text rides on the typed exception so the
                # planners can re-enter once with a compact re-emit.
                raise ResponseTruncatedError(str(content) if content else "")
            if content is None:
                # Some models return content: null for refusal
                refusal = message.get("refusal", "")
                if refusal:
                    raise AnalyzeError(f"LLM refused to respond: {refusal[:500]}")
                raise AnalyzeError("LLM returned content: null (possibly refused).")
            if not str(content).strip():
                raise AnalyzeError("LLM returned an empty response.")
            return str(content)
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            last_exc = exc
            logger.warning(
                "LLM request failed (attempt %d/%d): %s",
                attempt,
                retries,
                exc,
            )
            if attempt < retries:
                time.sleep(_retry_delay_seconds(attempt))
            else:
                raise AnalyzeError(f"LLM request failed after {retries} attempts: {exc}") from exc
        except AnalyzeError:
            raise  # Re-raise semantic failures immediately (no retry for bad responses)
    raise AnalyzeError(f"LLM request failed after {retries} attempts: {last_exc}")


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------


def _read_document(doc_id: str) -> dict[str, Any]:
    """Read a parsed document from the artifact store."""
    document = artifacts.read(doc_id)
    if not document:
        raise AnalyzeError(
            f"Document {doc_id} not found or evicted. Re-parse the file first."
        )
    return document


def _manifest_summary(document: dict[str, Any]) -> dict[str, Any]:
    """Build a compact summary of the document manifest."""
    manifest = document.get("manifest") or {}
    sections = manifest.get("sections") or []
    return {
        "fileName": manifest.get("fileName", "document"),
        "totalPages": manifest.get("totalPages", 0),
        "totalChars": manifest.get("totalChars", 0),
        "estTokens": manifest.get("estTokens", 0),
        "sectionCount": len(sections),
        "sections": [
            {
                "sectionId": s.get("sectionId", ""),
                "heading": s.get("heading") or "",
                "label": s.get("label", ""),
                "page": s.get("page"),
                "charCount": s.get("charCount", 0),
            }
            for s in sections[:50]
        ],
        "tables": len(manifest.get("tables") or []),
    }


def _section_text(document: dict[str, Any], section_id: str) -> str:
    """Extract text for a specific section from the document."""
    blocks = document.get("blocks") or []
    parts: list[str] = []
    for block in blocks:
        if block.get("sectionId") != section_id:
            continue
        text = block.get("text") or ""
        if text.strip():
            parts.append(text)
    return "\n".join(parts)


def _resolve_section(document: dict[str, Any], section_id: str) -> tuple[str, str, str]:
    """Resolve a section reference to (sectionId, heading, text).

    The planner may pass a sectionId ("s1") OR a heading/label ("Financial
    Results"); both resolve to the same section. Raises AnalyzeError when the
    reference matches nothing or the section has no text — the caller routes
    that failure back to the planner as an observation.
    """
    resolved_id = section_id
    if not _section_text(document, section_id).strip():
        manifest = document.get("manifest", {})
        for sect in (manifest.get("sections") or []):
            heading = sect.get("heading") or ""
            label = sect.get("label") or ""
            if section_id.strip().lower() in (heading.strip().lower(), label.strip().lower()):
                resolved_id = sect.get("sectionId", "")
                break
        if resolved_id != section_id:
            logger.info("Resolved section %r to sectionId %r", section_id, resolved_id)

    text = _section_text(document, resolved_id)
    if not text.strip():
        # The observation must be actionable: list the real section ids so the
        # planner can correct the reference in ONE step.
        available = [
            sect.get("sectionId", "")
            for sect in ((document.get("manifest", {}) or {}).get("sections") or [])
            if _section_text(document, sect.get("sectionId", "")).strip()
        ][:15]
        raise AnalyzeError(
            f"Section '{section_id}' does not exist or has no text content. "
            f"Available text sections: {', '.join(available) or 'none'}. "
            f"Reference one of these section ids; do not repeat '{section_id}'."
        )
    heading = ""
    for sect in ((document.get("manifest", {}) or {}).get("sections") or []):
        if sect.get("sectionId") == resolved_id:
            heading = sect.get("heading") or sect.get("label", "")
            break
    return resolved_id, heading, text


# ---------------------------------------------------------------------------
# Skill runners
# ---------------------------------------------------------------------------

_RUNNERS: dict[str, Any] = {}


def _register(skill_name: str):
    """Decorator to register a skill runner."""

    def decorator(func):
        _RUNNERS[skill_name] = func
        return func

    return decorator


def run(
    skill_name: str,
    params: dict[str, Any],
    state: dict[str, Any],
    llm_settings: dict[str, str] | None = None,
    report_progress: Any = None,
) -> Any:
    """Execute a skill by name with the given parameters.

    Args:
        skill_name: The skill to execute.
        params: Parameters for the skill.
        state: Current execution state (document, images, results, etc.).
        llm_settings: LLM settings for LLM-based skills.
        report_progress: Optional progress callback.

    Returns:
        Skill execution result.

    Raises:
        AnalyzeError: If the skill is unknown or execution fails.
    """
    runner = _RUNNERS.get(skill_name)
    if not runner:
        raise AnalyzeError(f"Unknown skill: {skill_name}")
    return runner(params, state, llm_settings, report_progress)


# --- read_manifest ---


@_register("read_manifest")
def _run_read_manifest(
    params: dict[str, Any],
    state: dict[str, Any],
    llm_settings: dict[str, str] | None = None,
    report_progress: Any = None,
) -> dict[str, Any]:
    """Read the document manifest from state.

    Prefers the document from ``state["document"]`` (already loaded by the
    harness) to avoid a redundant artifact-store read. Falls back to reading
    from the artifact store when the document is not in state.
    """
    doc_id = state.get("doc_id", "")
    if not doc_id:
        raise AnalyzeError("No doc_id in state for read_manifest.")

    if report_progress:
        report_progress("reading", "Reading document structure...", 10.0)

    document = state.get("document") or _read_document(doc_id)
    summary = _manifest_summary(document)
    return {
        "skill": "read_manifest",
        "manifest": summary,
        "sections": summary.get("sections", []),
        "total_chars": summary.get("totalChars", 0),
        "total_pages": summary.get("totalPages", 0),
    }


# --- read_section ---


@_register("read_section")
def _run_read_section(
    params: dict[str, Any],
    state: dict[str, Any],
    llm_settings: dict[str, str] | None = None,
    report_progress: Any = None,
) -> dict[str, Any]:
    """Read a specific section's text from the document.

    Prefers the document from ``state["document"]`` (already loaded by the
    harness) to avoid a redundant artifact-store read. Falls back to reading
    from the artifact store when the document is not in state.
    """
    section_id = params.get("section_id", "")
    if not section_id:
        raise AnalyzeError("read_section requires section_id parameter.")

    raw_max_chars = params.get("max_chars", 20000)
    try:
        max_chars = int(raw_max_chars)
    except (TypeError, ValueError) as exc:
        raise AnalyzeError(
            f"read_section max_chars must be an integer, got {raw_max_chars!r}."
        ) from exc
    doc_id = state.get("doc_id", "")
    if not doc_id:
        raise AnalyzeError("No doc_id in state for read_section.")

    if report_progress:
        section_label = params.get("label", section_id)
        report_progress("reading", f"Reading section: {section_label}...", 30.0)

    document = state.get("document") or _read_document(doc_id)
    resolved_id, heading, text = _resolve_section(document, section_id)

    # Truncate to max_chars
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...(truncated)"

    return {
        "skill": "read_section",
        "section_id": resolved_id,
        "requested_section_id": section_id,
        "text": text,
        "heading": heading,
        "char_count": len(text),
    }


# --- analyze_text ---


@_register("analyze_text")
def _run_analyze_text(
    params: dict[str, Any],
    state: dict[str, Any],
    llm_settings: dict[str, str] | None = None,
    report_progress: Any = None,
) -> dict[str, Any]:
    """Analyze a section's real text content with the LLM.

    Address-based (pi thesis): the planner passes a section_id — never the
    content. Content flowing through model params gets truncated or fabricated;
    only section references are accepted.
    """
    from .skills import get_skill

    skill_def = get_skill("analyze_text")
    section_id = str(params.get("section_id", "")).strip()
    if not section_id:
        raise AnalyzeError(
            "analyze_text requires section_id. Reference the section from the "
            "manifest — never paste or invent the content yourself."
        )
    doc_id = state.get("doc_id", "")
    document = state.get("document") or (doc_id and _read_document(doc_id)) or None
    if not document:
        raise AnalyzeError("analyze_text has no document in the analysis state.")

    resolved_id, heading, text = _resolve_section(document, section_id)
    question = params.get("question", state.get("task", "Summarize this document."))
    label = params.get("label", heading or resolved_id)

    settings = llm_settings or _llm_settings()

    if report_progress:
        report_progress("analyzing", f"Analyzing: {label}...", 60.0)

    user_content = (
        f"Text label: {label}\n\n"
        f"Text content:\n{text}\n\n"
        f"---\n\nQuestion: {question}"
    )

    result = _call_llm(
        skill_def["prompt"],
        user_content,
        settings,
        max_tokens=4096,
        timeout=skill_def["timeout"],
    )

    return {
        "skill": "analyze_text",
        "analysis": result,
        "label": label,
        "section_id": resolved_id,
    }


# --- synthesize ---


@_register("synthesize")
def _run_synthesize(
    params: dict[str, Any],
    state: dict[str, Any],
    llm_settings: dict[str, str] | None = None,
    report_progress: Any = None,
) -> dict[str, Any]:
    """Combine multiple analyses into a coherent synthesis."""
    from .skills import get_skill

    skill_def = get_skill("synthesize")
    # Address-based: the harness gathers every collected analysis from the
    # execution state — the planner never restates content (it cannot copy
    # long analyses faithfully).
    analyses = [
        result["analysis"]
        for result in state.get("results", [])
        if isinstance(result, dict) and result.get("skill") == "analyze_text" and result.get("analysis")
    ]
    question = params.get("question", state.get("task", "Summarize the findings."))

    if not analyses:
        raise AnalyzeError(
            "synthesize was called but no analyze_text results are in the "
            "execution state. Collect analyses first, then synthesize."
        )

    settings = llm_settings or _llm_settings()

    if report_progress:
        report_progress(
            "synthesizing",
            f"Synthesizing {len(analyses)} analyses...",
            30.0,
        )

    analyses_text = "\n\n---\n\n".join(
        f"Analysis {i + 1}:\n{a}" for i, a in enumerate(analyses)
    )
    user_content = f"Analyses:\n\n{analyses_text}\n\n---\n\nQuestion: {question}"

    result = _call_llm(
        skill_def["prompt"],
        user_content,
        settings,
        max_tokens=4096,
        timeout=skill_def["timeout"],
    )

    return {
        "skill": "synthesize",
        "synthesis": result,
        "analysis_count": len(analyses),
    }


# --- answer ---


@_register("answer")
def _run_answer(
    params: dict[str, Any],
    state: dict[str, Any],
    llm_settings: dict[str, str] | None = None,
    report_progress: Any = None,
) -> dict[str, Any]:
    """Generate the final answer from all collected evidence."""
    from .skills import get_skill

    skill_def = get_skill("answer")
    all_evidence = params.get("all_evidence", "")
    question = params.get("question", state.get("task", ""))

    if not all_evidence:
        # Build evidence from state
        evidence_parts = []
        for result in state.get("results", []):
            if isinstance(result, dict):
                for key in ("analysis", "synthesis", "text"):
                    val = result.get(key)
                    if val:
                        label = result.get("label") or result.get("section_id", "")
                        evidence_parts.append(f"[{label}]\n{val}" if label else val)
        all_evidence = "\n\n".join(evidence_parts)

    if not all_evidence.strip():
        raise AnalyzeError(
            "answer was called but no evidence is available. "
            "The planner must collect analysis results before requesting the final answer. "
            "Ensure analyze_text or synthesize skills have been executed "
            "and their results are stored in the analysis state."
        )

    settings = llm_settings or _llm_settings()

    if report_progress:
        report_progress("answering", "Generating final answer...", 30.0)

    user_content = (
        f"Evidence collected:\n\n{all_evidence}\n\n"
        f"---\n\nQuestion: {question}"
    )

    result = _call_llm(
        skill_def["prompt"],
        user_content,
        settings,
        max_tokens=4096,
        timeout=skill_def["timeout"],
    )

    return {
        "skill": "answer",
        "answer": result,
    }


# --- check_file_size ---


@_register("check_file_size")
def _run_check_file_size(
    params: dict[str, Any],
    state: dict[str, Any],
    llm_settings: dict[str, str] | None = None,
    report_progress: Any = None,
) -> dict[str, Any]:
    """Check if the parsed document is within size limits for analysis.

    Extracts the total character count from the document manifest and compares
    it against the configured limit. This is a data-only skill — no LLM call.

    The limit is derived from:
    1. ``params.max_chars`` (if provided by the planner)
    2. ``state.get("max_chars")`` (set by the harness)
    3. The default ``SINGLE_PASS_MAX_CHARS`` (300K)

    Returns:
        Dict with total_chars, max_chars, within_limit, and a message.
    """
    document = state.get("document")
    if not document:
        raise AnalyzeError(
            "check_file_size has no document in the analysis state. This skill "
            "only applies to document analyses (images arrive as OCR-parsed "
            "documents, which always carry a manifest)."
        )

    manifest = document.get("manifest", {})
    total_chars = manifest.get("totalChars", 0) or 0
    raw_max_chars = params.get("max_chars") or state.get("max_chars") or SINGLE_PASS_MAX_CHARS
    try:
        max_chars = int(raw_max_chars)
    except (TypeError, ValueError) as exc:
        raise AnalyzeError(
            f"check_file_size max_chars must be an integer, got {raw_max_chars!r}."
        ) from exc

    within_limit = total_chars <= max_chars

    if within_limit:
        message = f"File size ({total_chars:,} chars) is within the analysis limit ({max_chars:,} chars)."
    else:
        mb = total_chars / (1024 * 1024)
        limit_mb = max_chars / (1024 * 1024)
        message = (
            f"File is too large for direct analysis: {total_chars:,} chars "
            f"(~{mb:.1f} MB) exceeds the {max_chars:,} char limit (~{limit_mb:.0f} MB). "
            f"The planner should use map_reduce strategy to analyze sections "
            f"individually, or inform the user that the file exceeds processing limits."
        )

    return {
        "skill": "check_file_size",
        "total_chars": total_chars,
        "max_chars": max_chars,
        "within_limit": within_limit,
        "message": message,
    }


# --- report_size_limit ---


@_register("report_size_limit")
def _run_report_size_limit(
    params: dict[str, Any],
    state: dict[str, Any],
    llm_settings: dict[str, str] | None = None,
    report_progress: Any = None,
) -> dict[str, Any]:
    """Generate a structured report about the file size exceeding limits.

    This skill is invoked by the planner when ``check_file_size`` returns
    ``within_limit: False``. It produces a clear, actionable message for the
    user about the size constraint.

    Returns:
        Dict with a user-facing message about the size limit.
    """
    total_chars = params.get("total_chars", 0)
    max_chars = params.get("max_chars", SINGLE_PASS_MAX_CHARS)
    total_mb = params.get("total_mb", total_chars / (1024 * 1024))
    max_mb = params.get("max_mb", max_chars / (1024 * 1024))

    message = (
        f"The file contains {total_chars:,} characters (~{total_mb:.1f} MB of text), "
        f"which exceeds the analysis limit of {max_chars:,} characters (~{max_mb:.0f} MB). "
        f"The document will be analyzed section by section using the map_reduce strategy, "
        f"which may take longer but will process all content. "
        f"If you continue to experience issues, consider splitting the file into smaller parts."
    )

    return {
        "skill": "report_size_limit",
        "message": message,
    }