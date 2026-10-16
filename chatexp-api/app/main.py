from __future__ import annotations

import logging
import os
import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, ProxyHandler, build_opener

from celery.result import AsyncResult
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from .tasks import analyze_project_task, celery_app, suggest_project_task
from .files.router import router as files_router
from .files.admin import router as files_admin_router
from .security import require_api_key, require_api_key_for, validate_llm_endpoint
from .security import normalize_chat_completions_endpoint

logger = logging.getLogger(__name__)


CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CHATEXP_CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

app = FastAPI(
    title="ChatExp BO API",
    version="0.1.0",
    description="Bayesian optimization API for the ChatExp SeaTable plugin.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Wire compression (server occupancy): analysis results, BO payloads, and
# document text are large JSON — gzip cuts them ~5-10x on the wire. Added
# BEFORE CORSMiddleware so CORS stays the outermost layer. Responses already
# compressed upstream (Caddy encode) are untouched; bodies under 1KB skip
# compression where the overhead outweighs the win.
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.include_router(files_router)
app.include_router(files_admin_router)


class BOJobRequest(BaseModel):
    project_name: str = "seatable"
    table: str | None = None
    view: str | None = None
    rows: list[dict[str, Any]] = Field(default_factory=list)
    factors: list[dict[str, Any]] = Field(default_factory=list)
    objectives: list[dict[str, Any]] = Field(default_factory=list)
    n_points: int = Field(default=5, ge=1, le=50)
    design_strategy: str | None = None
    designStrategy: str | None = None
    runtime: dict[str, Any] = Field(default_factory=dict)
    language: str = "zh"


class LLMSettings(BaseModel):
    endpoint: str | None = None
    apiKey: str | None = None
    model: str | None = None
    temperature: float | None = 0
    maxTokens: int | None = 8192
    responseFormat: bool | None = True


class LLMProxyRequest(BaseModel):
    message: str = ""
    history: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    llm: LLMSettings | None = None
    systemPrompt: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/llm/chat", dependencies=[Depends(require_api_key_for("/v1/llm/chat"))])
def proxy_llm_chat(payload: LLMProxyRequest) -> dict[str, Any]:
    # Size guard: uvicorn imposes no default body limit, so an oversized
    # planner payload would be parsed and relayed wholesale. 4 MB comfortably
    # fits any legitimate planner call (the plugin's own degraded contexts
    # stay far below this) and fails fast with 413 instead.
    payload_bytes = len(payload.message.encode("utf-8")) + sum(
        len(str(item).encode("utf-8")) for item in payload.history
    ) + len(json.dumps(payload.context, ensure_ascii=False).encode("utf-8"))
    if payload_bytes > _LLM_PROXY_MAX_PAYLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="LLM proxy payload exceeds the size limit.",
        )
    llm = _resolve_llm_settings(payload.llm)
    endpoint = _normalize_chat_completions_endpoint(llm.endpoint or "")
    validate_llm_endpoint(endpoint)
    request_body: dict[str, Any] = {
        "model": llm.model,
        "temperature": llm.temperature if llm.temperature is not None else 0,
        "messages": [
            {"role": "system", "content": payload.systemPrompt or _default_llm_system_prompt()},
            {"role": "user", "content": _build_llm_user_content(payload)},
        ],
    }
    if llm.maxTokens and llm.maxTokens > 0:
        request_body["max_tokens"] = llm.maxTokens
    if llm.responseFormat is not False:
        request_body["response_format"] = {"type": "json_object"}
    raw_response = _post_json(endpoint, request_body, llm.apiKey or "")
    choices = raw_response.get("choices") or []
    if not isinstance(choices, list) or not choices:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Model returned an empty response.")
    choice = choices[0] if isinstance(choices[0], dict) else {}
    content = (choice.get("message") or {}).get("content", "")
    body: dict[str, Any] = {"plan": _parse_llm_plan(content)}
    # finishReason passthrough: the plugin refuses to execute truncated plans
    # (payload.finishReason === "length") — a length-cut body can still parse
    # into a structurally valid but silently incomplete plan. Omitted when the
    # upstream sends none (missing / "stop" need no signal).
    finish_reason = choice.get("finish_reason")
    if finish_reason:
        body["finishReason"] = str(finish_reason)
    # usage passthrough, normalized to a stable {input, output, total} contract
    # (input=prompt tokens, output=completion tokens, total=total tokens) so
    # the plugin can track token usage without re-mapping OpenAI field names.
    usage = _normalize_usage(raw_response.get("usage"))
    if usage is not None:
        body["usage"] = usage
    return body


@app.post("/v1/bo/analyze", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_api_key)])
def create_analysis_job(payload: BOJobRequest, request: Request) -> dict[str, str]:
    task = analyze_project_task.delay(payload.model_dump())
    return _job_response(task.id, request)


@app.post("/v1/bo/suggest", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_api_key)])
def create_suggestion_job(payload: BOJobRequest, request: Request) -> dict[str, str]:
    task = suggest_project_task.delay(payload.model_dump())
    return _job_response(task.id, request)


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def get_job(job_id: str) -> dict[str, Any]:
    result = AsyncResult(job_id, app=celery_app)
    state = result.state.lower()
    if result.successful():
        return {"job_id": job_id, "status": "succeeded", "result": result.result}
    if result.failed():
        return {"job_id": job_id, "status": "failed", "error": str(result.result)}
    if state == "revoked":
        return {"job_id": job_id, "status": "canceled"}
    return {"job_id": job_id, "status": "running" if state == "started" else "queued"}


@app.delete("/v1/jobs/{job_id}/cancel", dependencies=[Depends(require_api_key)])
@app.post("/v1/jobs/{job_id}/cancel", dependencies=[Depends(require_api_key)])
def cancel_job(job_id: str) -> dict[str, str]:
    result = AsyncResult(job_id, app=celery_app)
    result.revoke(terminate=True, signal="SIGTERM")
    return {"job_id": job_id, "status": "canceled"}


def _job_response(job_id: str, request: Request) -> dict[str, str]:
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": str(request.url_for("get_job", job_id=job_id)),
    }


def _resolve_llm_settings(settings: LLMSettings | None) -> LLMSettings:
    endpoint = _first_text(
        settings.endpoint if settings else "",
        os.getenv("CHATEXP_LLM_ENDPOINT", ""),
        os.getenv("CHATEXP_OPENAI_BASE_URL", ""),
        os.getenv("OPENAI_BASE_URL", ""),
    )
    api_key = _first_text(
        settings.apiKey if settings else "",
        os.getenv("CHATEXP_LLM_API_KEY", ""),
        os.getenv("CHATEXP_OPENAI_API_KEY", ""),
        os.getenv("OPENAI_API_KEY", ""),
    )
    model = _first_text(
        settings.model if settings else "",
        os.getenv("CHATEXP_LLM_MODEL", ""),
        os.getenv("CHATEXP_OPENAI_MODEL", ""),
        os.getenv("OPENAI_MODEL", ""),
    )
    if not endpoint or not model:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Server-managed LLM is not configured. Set CHATEXP_LLM_ENDPOINT, CHATEXP_LLM_API_KEY, and CHATEXP_LLM_MODEL on chatexp-api, or send an llm payload.")
    return LLMSettings(
        endpoint=endpoint,
        apiKey=api_key,
        model=model,
        temperature=settings.temperature if settings and settings.temperature is not None else _read_float_env("CHATEXP_LLM_TEMPERATURE", 0),
        maxTokens=settings.maxTokens if settings and settings.maxTokens is not None else _read_int_env("CHATEXP_LLM_MAX_TOKENS", 8192),
        responseFormat=settings.responseFormat if settings and settings.responseFormat is not None else _read_bool_env("CHATEXP_LLM_RESPONSE_FORMAT", True),
    )


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _read_float_env(name: str, fallback: float) -> float:
    try:
        return float(os.getenv(name, fallback))
    except (TypeError, ValueError):
        return fallback


def _read_int_env(name: str, fallback: int) -> int:
    try:
        return int(os.getenv(name, fallback))
    except (TypeError, ValueError):
        return fallback


def _read_bool_env(name: str, fallback: bool) -> bool:
    value = str(os.getenv(name, "")).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return fallback


def _normalize_chat_completions_endpoint(endpoint: str) -> str:
    try:
        return normalize_chat_completions_endpoint(endpoint)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


def _default_llm_system_prompt() -> str:
    return (
        "You are ChatExp inside a SeaTable plugin. Return only one raw JSON object. "
        "Use the user-provided context to plan SeaTable actions. If no table action is needed, "
        'return {"message":"...","reason":"...","confidence":0.0,"skill":"respond","requiresConfirmation":false}.'
    )


def _build_llm_user_content(payload: LLMProxyRequest) -> Any:
    # Key order is load-bearing for provider prefix caching: the stable
    # context (schema, catalog) leads, volatile message/history follow. The
    # plugin-side buildPlannerUserPayload emits the same order.
    user_payload = {
        "context": payload.context,
        "message": payload.message,
        "history": payload.history,
    }
    image_references = payload.context.get("imageColumnReferences") or []
    images = [
        item
        for item in image_references
        if isinstance(item, dict) and item.get("url")
    ]
    if not images:
        # Compact JSON: the model never sees the whitespace.
        return json.dumps(user_payload, ensure_ascii=False)
    return [
        {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)},
        *[
            {"type": "image_url", "image_url": {"url": item["url"]}}
            for item in images
        ],
    ]


def _llm_proxy_timeout_seconds() -> int:
    # Default 180s: matches the plugin's planner-side abort timeout.
    return max(1, int(os.getenv("CHATEXP_LLM_PROXY_TIMEOUT_SECONDS", "180")))


# The model endpoint is an explicitly allowlisted direct target
# (CHATEXP_LLM_PROXY_ALLOWED_BASE_URLS). Docker-daemon-injected HTTP_PROXY
# env would otherwise route these calls through a corporate proxy that
# intermittently fails with empty 502s — the opener pins every model call
# to a direct connection regardless of environment proxies.
_DIRECT_OPENER = build_opener(ProxyHandler({}))

# /v1/llm/chat payload ceiling (see proxy_llm_chat).
_LLM_PROXY_MAX_PAYLOAD_BYTES = 4 * 1024 * 1024

# Classify upstream errors WITHOUT echoing their bodies to the client: an
# upstream body may quote request content. Only a fixed, content-free verdict
# travels back — enough for the plugin's context-overflow classifier
# (isContextOverflowError in chatexp-seatable-plugin
# src/lib/llm/orchestrationHarness.js) to run its one degraded-context retry.
# Keep the verdict wording in lockstep with that classifier; the exact string
# is pinned on both sides (tests/test_main.py, tests/piGaps.spec.js).
_UPSTREAM_OVERFLOW_PATTERN = re.compile(
    r"request_too_large|context[_ -]?length|input.{0,80}exceed.{0,60}limit"
    r"|prompt.{0,40}too.{0,20}long|maximum.{0,40}context|too many tokens"
    r"|range of input length|exceeds the context window",
    re.IGNORECASE,
)


def _classify_upstream_http_error(status_code: int, body: str) -> str | None:
    # 429 stays unclassified: a token-throttle whose body happens to say
    # "too many tokens" is throttling, not context overflow — the plugin's
    # classifier excludes 429 anyway, and a wrong verdict would misdiagnose
    # the failure the user sees.
    if status_code == 429:
        return None
    if _UPSTREAM_OVERFLOW_PATTERN.search(body or ""):
        return (
            f"LLM upstream returned HTTP {status_code}: the planner prompt "
            "exceeded the model context length (context overflow)."
        )
    return None


def _post_json(endpoint: str, body: dict[str, Any], api_key: str) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    token = str(api_key or "").strip()
    if token:
        headers["Authorization"] = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    request = UrlRequest(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    timeout = _llm_proxy_timeout_seconds()
    try:
        with _DIRECT_OPENER.open(request, timeout=timeout) as response:
            data = response.read().decode("utf-8")
    except HTTPError as exc:
        # The upstream error body may echo request content or internal detail;
        # log it server-side and return a classified, content-free message to
        # the client (context overflow is the one verdict the plugin retries
        # against, so it must survive the proxy boundary).
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        logger.warning("LLM upstream returned HTTP %s: %s", exc.code, detail)
        classified = _classify_upstream_http_error(exc.code, detail)
        raise HTTPException(
            status_code=exc.code,
            detail=classified or f"LLM request failed with HTTP {exc.code}.",
        ) from exc
    except URLError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc.reason)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="LLM request timed out.") from exc
    except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, ConnectionError) as exc:
        # The LLM peer reset the connection mid-request (e.g. the model server
        # is overloaded or restarted). Surface as a clean 502 so the response
        # flows through the CORS middleware. An unhandled socket exception would
        # bubble to uvicorn's ServerErrorMiddleware, which returns a 500 WITHOUT
        # CORS headers — the browser then reports a confusing CORS error instead
        # of the real upstream failure.
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"LLM connection was reset: {exc}") from exc
    except OSError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"LLM connection error: {exc}") from exc
    try:
        parsed = json.loads(data or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="LLM returned a non-JSON HTTP response.") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="LLM returned an invalid response payload.")
    return parsed


def _normalize_usage(usage: Any) -> dict[str, int | None] | None:
    """Normalize the upstream usage object to {input, output, total}.

    Stable passthrough contract: input = prompt_tokens, output =
    completion_tokens, total = total_tokens. Returns None when the upstream
    sent no usage object (the field is then omitted from the response);
    unknown extra fields are dropped, non-numeric counts become None."""
    if not isinstance(usage, dict):
        return None

    def _count(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return {
        "input": _count(usage.get("prompt_tokens")),
        "output": _count(usage.get("completion_tokens")),
        "total": _count(usage.get("total_tokens")),
    }


def _parse_llm_plan(content: str) -> dict[str, Any]:
    raw = str(content or "").strip()
    if not raw:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="LLM returned empty content.")
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Log the raw content server-side only; do not echo it to the client,
        # since it may contain echoed system-prompt content or request data.
        logger.warning("LLM returned non-JSON content. Starts with: %s", raw[:500])
        # 422, not 502: a non-JSON body is a deterministic semantic failure of
        # THIS request — 502 sits in the plugin transport's retryable set and
        # would trigger blind re-rolls that fail identically.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="LLM returned non-JSON content. Check that the model returns a raw JSON object (no markdown fences or prose).",
        ) from exc
    if not isinstance(plan, dict):
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="LLM plan must be a JSON object.")
    return plan

