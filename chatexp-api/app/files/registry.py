"""redis-backed registry of parsed documents (lightweight metadata only).

The registry is the source of truth for "what has been parsed, by whom, how
big, and when was it last used". It holds NO artifact payload (those live on the
parsed-files volume via artifacts.py) — only KB-scale metadata, so redis stays
bounded even with tens of thousands of documents.

Four redis structures back it:
  - chatexp:doc:{docId}        hash/string -> metadata JSON
  - chatexp:doc:index          ZSET, score=lastAccessedAt, member=docId
                               (drives time-ordered listing + LRU eviction)
  - chatexp:doc:lock:{key}     SETNX parse/dedup lock, TTL = parse_lock_ttl
  - chatexp:doc:stats          hash of counters (parse_attempts/cache_hits/...)
"""
from __future__ import annotations

import json
import time
from typing import Any

import redis

from . import config

_DOC_KEY = "chatexp:doc:{doc_id}"
_INDEX_KEY = "chatexp:doc:index"
_LOCK_KEY = "chatexp:doc:lock:{key}"
_STATS_KEY = "chatexp:doc:stats"

_SORT_KEYS = {"recent", "oldest", "size", "name"}

# Module-level singleton: redis.Redis is thread-safe and pools connections,
# so one shared client is reused across all requests.
_CLIENT: redis.Redis | None = None


def _client() -> redis.Redis:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = redis.Redis.from_url(config.redis_url(), decode_responses=True)
    return _CLIENT


# --- metadata ---------------------------------------------------------------


def register(meta: dict[str, Any]) -> None:
    """Insert/replace a document's metadata and refresh the access index."""
    doc_id = meta["docId"]
    now = int(time.time())
    meta.setdefault("createdAt", now)
    meta["lastAccessedAt"] = meta.get("lastAccessedAt") or now
    meta.setdefault("accessCount", 0)
    pipe = _client().pipeline()
    pipe.set(_DOC_KEY.format(doc_id=doc_id), json.dumps(meta, ensure_ascii=False), ex=config.file_cache_ttl())
    pipe.zadd(_INDEX_KEY, {doc_id: meta["lastAccessedAt"]})
    pipe.execute()


def get(doc_id: str) -> dict[str, Any] | None:
    raw = _client().get(_DOC_KEY.format(doc_id=doc_id))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def exists(doc_id: str) -> bool:
    return _client().exists(_DOC_KEY.format(doc_id=doc_id)) > 0


def touch(doc_id: str) -> None:
    """Record an access: bump accessCount and move the index score forward.

    Preserves the remaining TTL (does NOT reset it to the full window) so hot
    documents still expire on their natural schedule and the registry cannot
    grow unbounded from effectively-immortal entries."""
    client = _client()
    key = _DOC_KEY.format(doc_id=doc_id)
    raw = client.get(key)
    if not raw:
        return
    try:
        meta = json.loads(raw)
    except (TypeError, ValueError):
        return
    now = int(time.time())
    meta["lastAccessedAt"] = now
    meta["accessCount"] = int(meta.get("accessCount", 0)) + 1
    # Read remaining TTL and re-set with the SAME remaining seconds (rounded up),
    # so a fetch does not extend the document's lifetime.
    remaining = client.ttl(key)
    if remaining == -2:
        # The key expired between get() and ttl() (redis race). Writing the
        # metadata back would RESURRECT it without its TTL — an immortal entry
        # that LRU/eviction then has to clean up. Treat the access as a miss.
        # Only -1 means the key legitimately carries no TTL.
        return
    pipe = client.pipeline()
    if remaining and remaining > 0:
        pipe.set(key, json.dumps(meta, ensure_ascii=False), ex=remaining)
    elif remaining == -1:
        # No TTL configured — preserve that, do not invent one.
        pipe.set(key, json.dumps(meta, ensure_ascii=False))
    pipe.zadd(_INDEX_KEY, {doc_id: now})
    pipe.execute()


def remove(doc_id: str) -> bool:
    pipe = _client().pipeline()
    pipe.delete(_DOC_KEY.format(doc_id=doc_id))
    pipe.zrem(_INDEX_KEY, doc_id)
    results = pipe.execute()
    return bool(results[0])


def reconcile_index(max_scan: int = 5000) -> int:
    """Remove ZSET members whose metadata key has expired (TTL passed) but whose
    index entry lingered (ZSETs carry no per-member expiry). Returns the count
    of orphans removed.

    Without this, the ZSET grows monotonically and list_docs' total/pagination
    drifts away from reality. Called lazily from list_docs and at startup."""
    client = _client()
    members = client.zrange(_INDEX_KEY, 0, max_scan - 1)
    orphans = [m for m in members if not client.exists(_DOC_KEY.format(doc_id=m))]
    if orphans:
        client.zrem(_INDEX_KEY, *orphans)
    return len(orphans)


def list_docs(
    offset: int = 0,
    limit: int = 50,
    sort: str = "recent",
    query: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Return (docs, total_matching) for the admin list view.

    - sort=recent/oldest: by lastAccessedAt via the ZSET (cheap).
    - sort=size/name: fetch all matching metadata and sort in Python (the
      registry is metadata-only and small, so this is fine at production scale).
    - query: case-insensitive substring filter on fileName.
    """
    sort = sort if sort in _SORT_KEYS else "recent"
    client = _client()
    query_lower = ""
    if query:
        query_lower = query.strip().lower()
        if not query_lower:
            query = None

    if not query and sort in ("recent", "oldest"):
        # Fast path: range over the ZSET directly. Drop orphan members first so
        # the total reflects live documents, not expired-but-lingering entries.
        reconcile_index()
        total = client.zcard(_INDEX_KEY)
        if sort == "recent":
            doc_ids = client.zrevrange(_INDEX_KEY, offset, offset + limit - 1)
        else:
            doc_ids = client.zrange(_INDEX_KEY, offset, offset + limit - 1)
        docs = [meta for doc_id in doc_ids if (meta := get(doc_id)) is not None]
        return docs, total

    # Slow path: materialize everything to filter/sort in Python.
    all_ids = client.zrange(_INDEX_KEY, 0, -1)
    metas: list[dict[str, Any]] = []
    for doc_id in all_ids:
        meta = get(doc_id)
        if not meta:
            continue
        if query and query_lower not in str(meta.get("fileName", "")).lower():
            continue
        metas.append(meta)
    total = len(metas)
    reverse = sort in ("recent", "size")
    sort_key = {
        "recent": lambda m: m.get("lastAccessedAt", 0),
        "oldest": lambda m: m.get("lastAccessedAt", 0),
        "size": lambda m: m.get("byteSize", 0),
        "name": lambda m: str(m.get("fileName", "")).lower(),
    }[sort]
    metas.sort(key=sort_key, reverse=reverse)
    return metas[offset : offset + limit], total


def all_doc_ids() -> list[str]:
    """Every docId in the index, oldest-first (for LRU cleanup)."""
    return list(_client().zrange(_INDEX_KEY, 0, -1))


def count() -> int:
    return int(_client().zcard(_INDEX_KEY))


# --- LRU eviction -----------------------------------------------------------


def evict_until(max_bytes: int) -> list[str]:
    """Remove least-recently-used documents (metadata + their artifacts) until
    the on-disk artifact total fits under max_bytes. Returns evicted docIds.

    Iterates oldest-first via the ZSET. Stops as soon as the budget is met, so
    a cache far under budget is a no-op.
    """
    from . import artifacts

    evicted: list[str] = []
    if max_bytes <= 0:
        return evicted
    used = artifacts.total_bytes()
    if used <= max_bytes:
        return evicted
    for doc_id in all_doc_ids():  # oldest first
        if used <= max_bytes:
            break
        meta = get(doc_id) or {}
        artifact_bytes = meta.get("byteSize") or artifacts.size(doc_id)
        artifacts.delete(doc_id)
        remove(doc_id)
        used -= artifact_bytes
        evicted.append(doc_id)
    return evicted


# --- concurrency / dedup locks ----------------------------------------------


def acquire_parse_lock(key: str) -> bool:
    """Try to become the sole parser for `key`. Returns True if this caller now
    holds the lock. The lock auto-expires so a crashed worker cannot stall peers
    forever."""
    return _client().set(_LOCK_KEY.format(key=key), "1", nx=True, ex=config.parse_lock_ttl())


def release_parse_lock(key: str) -> None:
    _client().delete(_LOCK_KEY.format(key=key))


# --- stats ------------------------------------------------------------------


def record_stat(field: str, increment: int = 1) -> None:
    """Best-effort stat increment. Stats are pure telemetry — a transient redis
    error must never fail a parse, so we swallow redis errors here."""
    try:
        _client().hincrby(_STATS_KEY, field, increment)
    except redis.RedisError:
        pass


def get_stats() -> dict[str, Any]:
    raw = _client().hgetall(_STATS_KEY) or {}
    stats = {key: int(value) for key, value in raw.items()}
    stats.setdefault("parse_attempts", 0)
    stats.setdefault("cache_hits", 0)
    stats.setdefault("cleanup_runs", 0)
    return stats


def reset_stats() -> dict[str, Any]:
    _client().delete(_STATS_KEY)
    return get_stats()


# --- helpers ----------------------------------------------------------------


def build_meta(doc_id: str, manifest: dict[str, Any], byte_size: int, artifacts_path: str) -> dict[str, Any]:
    """Construct the lightweight metadata record stored in redis from a manifest
    (which itself stays part of the artifact payload on disk)."""
    now = int(time.time())
    return {
        "docId": doc_id,
        "fileName": manifest.get("fileName", ""),
        "mimeType": manifest.get("mimeType"),
        "totalChars": manifest.get("totalChars", 0),
        "estTokens": manifest.get("estTokens", 0),
        "totalPages": manifest.get("totalPages"),
        "parserEngine": manifest.get("parserEngine", "docling"),
        "sectionsCount": len(manifest.get("sections", [])),
        "tablesCount": len(manifest.get("tables", [])),
        "lowConfidencePages": manifest.get("lowConfidencePages", []),
        "byteSize": byte_size,
        "artifactsPath": artifacts_path,
        "createdAt": now,
        "lastAccessedAt": now,
        "accessCount": 0,
    }


def get_analysis_memo(analysis_key: str) -> str | None:
    """Cached answer for an identical (document, question, model) analysis.

    Occupancy guard: a repeated first-turn question re-runs the full LLM
    loop (minutes of planner calls) for a byte-identical answer. Served
    only for first-turn questions (see harness.analyze) so conversation
    context can never make a cached answer wrong. Redis failures degrade
    to a cache miss, never an error."""
    try:
        client = _client()
        if client is None:
            return None
        return client.get(f"chatexp:analysis:{analysis_key}")
    except redis.RedisError:
        return None


def set_analysis_memo(analysis_key: str, answer: str, ttl: int = 86400) -> None:
    try:
        client = _client()
        if client is None or not answer:
            return
        client.setex(f"chatexp:analysis:{analysis_key}", ttl, answer)
    except redis.RedisError:
        pass


def delete_analysis_memo(analysis_key: str) -> None:
    try:
        client = _client()
        if client is None:
            return
        client.delete(f"chatexp:analysis:{analysis_key}")
    except redis.RedisError:
        pass
