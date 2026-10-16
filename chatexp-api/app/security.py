"""Shared security primitives for chatexp-api.

API-key guards: each key is read once at import and compared with
hmac.compare_digest. `require_api_key_strict` and `require_api_key_for` refuse
with 503 when no key is configured, so management and LLM-cost endpoints can
never run open against the docker-compose empty default.
"""
from __future__ import annotations

import hmac
import os
import re
from urllib.parse import urlparse

from fastapi import Header, HTTPException, status

_BO_API_KEY = os.getenv("CHATEXP_BO_API_KEY", "").strip()
# Separate key for the document-management routes. When unset, management
# falls back to the BO key so existing single-key deployments keep working.
_ADMIN_API_KEY = os.getenv("CHATEXP_ADMIN_API_KEY", "").strip()

# docId values are sha256 prefixes (build_doc_id returns [:32]); enforce that at
# every trust boundary to defuse path-traversal via the {doc_id} route param.
_DOC_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def _check_bearer(authorization: str | None, key: str, detail: str) -> None:
    provided = authorization or ""
    if not (hmac.compare_digest(provided, f"Bearer {key}") or hmac.compare_digest(provided, key)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    """Gate BO + files endpoints. No-op when no key is configured (the
    docker-compose default). Use require_api_key_strict for management endpoints
    that must never be open."""
    if not _BO_API_KEY:
        return
    _check_bearer(authorization, _BO_API_KEY, "Invalid BO API key.")


def require_api_key_strict(authorization: str | None = Header(default=None)) -> None:
    """Gate management endpoints. ALWAYS requires a configured key — refuses the
    request with 503 when neither CHATEXP_ADMIN_API_KEY nor CHATEXP_BO_API_KEY
    is configured, so the insecure empty-default can never expose document
    listing/deletion. Prefer the dedicated admin key when set."""
    expected_key = _ADMIN_API_KEY or _BO_API_KEY
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Management API disabled: no API key is configured.",
        )
    _check_bearer(authorization, expected_key, "Invalid admin API key.")


def require_api_key_for(route: str):
    """Dependency factory for EXPENSIVE endpoints (LLM-cost-incurring relays,
    ~200MB-body uploads). Unlike require_api_key — a no-op when the key is
    unset (the docker-compose default) — this gate refuses the request with
    503 until CHATEXP_BO_API_KEY is configured, so those routes can never run
    unauthenticated. The 503 names the refused route."""
    def dependency(authorization: str | None = Header(default=None)) -> None:
        if not _BO_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"CHATEXP_BO_API_KEY is not configured; refusing unauthenticated access to {route}.",
            )
        _check_bearer(authorization, _BO_API_KEY, "Invalid BO API key.")
    return dependency


def validate_doc_id(doc_id: str) -> str:
    """Return doc_id if it matches the canonical 32-hex shape, else raise 400.
    Defense-in-depth against path traversal through the {doc_id} route param."""
    if not _DOC_ID_PATTERN.match(doc_id or ""):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document id.")
    return doc_id


def normalize_chat_completions_endpoint(endpoint: str) -> str:
    """Return the OpenAI-compatible chat-completions URL for an endpoint,
    appending the /chat/completions suffix when absent."""
    raw = str(endpoint or "").strip().rstrip("/")
    if not raw:
        raise ValueError("LLM endpoint must not be empty.")
    if raw.lower().endswith("/chat/completions"):
        return raw
    return f"{raw}/chat/completions"


def validate_llm_endpoint(endpoint: str) -> None:
    """Allowlist gate for client-supplied LLM endpoints.

    Shared by /v1/llm/chat and the /v1/files/analyze path: both forward
    request content to the endpoint the client names, so the endpoint must be
    an HTTP(S) URL without credentials and, when CHATEXP_LLM_PROXY_ALLOWED_
    BASE_URLS is configured, must match one of its scheme://host[:port]
    entries. Default ports are normalized away on BOTH sides (https:443 and
    http:80 — "https://host:443" matches an allowlist entry "https://host").
    Raises 400 for a malformed endpoint (bad scheme, embedded credentials, or
    an invalid/out-of-range port), 403 for one outside the allowlist. An empty
    allowlist allows any endpoint (deployment choice).
    """
    target = _normalized_endpoint_origin(endpoint)
    allowed = [
        item.strip().rstrip("/")
        for item in os.getenv("CHATEXP_LLM_PROXY_ALLOWED_BASE_URLS", "").split(",")
        if item.strip()
    ]
    # Compare scheme://host[:port] exactly, not a string prefix — a prefix check
    # lets "http://llm.internal" also match "http://llm.internal.evil.com".
    allowed_hosts = []
    for item in allowed:
        try:
            allowed_hosts.append(_normalized_endpoint_origin(item))
        except HTTPException:
            # Allowlist entries are server config, not client input: a
            # malformed entry never matches (skip it) instead of
            # failing every request.
            continue
    if allowed_hosts and target not in allowed_hosts:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="LLM endpoint is not allowed by this proxy.")


_DEFAULT_PORTS = {"https": 443, "http": 80}


def _normalized_endpoint_origin(url: str) -> str:
    """Normalize a URL to scheme://host[:port], dropping the scheme's default
    port so an explicit default (https://host:443) compares equal to an
    implicit one (https://host).

    Raises the 400 validation error for a malformed URL, embedded credentials,
    or an invalid port — urlparse raises a bare ValueError from ``.port`` for
    out-of-range ports (e.g. host:99999), which must surface as the explicit
    400/422 validation path, never an unhandled 500."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="LLM endpoint must be an HTTP(S) URL.")
    if parsed.username or parsed.password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="LLM endpoint must not contain credentials.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="LLM endpoint port is invalid (must be 1-65535).",
        ) from exc
    hostname = parsed.hostname or ""
    if not hostname:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="LLM endpoint must be an HTTP(S) URL.")
    default_port = _DEFAULT_PORTS.get(parsed.scheme)
    if port is None or port == default_port:
        return f"{parsed.scheme}://{hostname}"
    return f"{parsed.scheme}://{hostname}:{port}"
