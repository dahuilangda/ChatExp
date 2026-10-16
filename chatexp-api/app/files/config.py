"""Configuration for file parsing, read from the environment.

Mirrors the existing chatexp-api convention of os.getenv with CHATEXP_* prefix
(there is no pydantic-settings BaseSettings in this service).
"""
from __future__ import annotations

import os


def docling_base_url() -> str:
    return os.getenv("CHATEXP_DOCLING_BASE_URL", "").strip().rstrip("/")


def docling_timeout() -> float:
    return _read_float("CHATEXP_DOCLING_TIMEOUT_SECONDS", 300.0)


def file_max_bytes() -> int:
    return _read_int("CHATEXP_FILE_MAX_BYTES", 50 * 1024 * 1024)


def file_cache_ttl() -> int:
    return _read_int("CHATEXP_FILE_CACHE_TTL", 86400)


def seatable_timeout() -> float:
    return _read_float("CHATEXP_SEATABLE_TIMEOUT_SECONDS", 60.0)


def seatable_api_token() -> str:
    """Optional server-side SeaTable API-Token used by the worker to resolve
    app-download-link when the plugin cannot supply one (integrated plugins do
    not always receive dtablePluginConfig.APIToken). Generate it in the SeaTable
    UI for the target base (read permission is enough)."""
    return os.getenv("CHATEXP_SEATABLE_API_TOKEN", "").strip()


def seatable_server() -> str:
    """Optional SeaTable server base URL paired with CHATEXP_SEATABLE_API_TOKEN
    (e.g. https://oa.vicrobx.com). Used as a fallback when the plugin payload
    does not carry a server."""
    return os.getenv("CHATEXP_SEATABLE_SERVER", "").strip().rstrip("/")


def parsed_files_dir() -> str:
    """Shared volume mount where parsed document artifacts (blocks/markdown)
    are stored. Both the api and worker containers mount this path so the
    worker can write and the api can serve fetches without round-tripping
    multi-MB payloads through redis."""
    return os.getenv("CHATEXP_PARSED_FILES_DIR", "/workspace/parsed-files").strip() or "/workspace/parsed-files"


def parsed_files_max_bytes() -> int:
    """High-water mark for the parsed-files volume. The LRU cleanup task evicts
    the least-recently-accessed artifacts until total bytes fall below this."""
    return _read_int("CHATEXP_PARSED_FILES_MAX_BYTES", 2 * 1024 * 1024 * 1024)


def parse_lock_ttl() -> int:
    """How long a parse/dedup lock is held (seconds). Covers download + docling
    for one document; auto-releases on crash so a peer can retry."""
    return _read_int("CHATEXP_PARSE_LOCK_TTL_SECONDS", 600)


def redis_url() -> str:
    # Reuse the same redis the celery app uses.
    return os.getenv("CHATEXP_REDIS_URL", "redis://redis:6379/0")


def _read_float(name: str, fallback: float) -> float:
    try:
        return float(os.getenv(name, fallback))
    except (TypeError, ValueError):
        return fallback


def _read_int(name: str, fallback: int) -> int:
    try:
        return int(os.getenv(name, fallback))
    except (TypeError, ValueError):
        return fallback
