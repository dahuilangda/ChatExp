"""Admin / management endpoints for the parsed-document registry.

These are operational endpoints (list, inspect, delete, evict, stats) for
operators running chatexp-api. They are gated by the same BO API key as the
parse endpoints and never expose document bytes (only metadata), so they are
safe to surface in a management UI.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.security import require_api_key_strict, validate_doc_id

from . import artifacts, config, registry


router = APIRouter()


class DocSummary(BaseModel):
    docId: str
    fileName: str
    mimeType: str | None = None
    totalChars: int = 0
    estTokens: int = 0
    totalPages: int | None = None
    parserEngine: str = "docling"
    sectionsCount: int = 0
    tablesCount: int = 0
    lowConfidencePages: list[int] = []
    byteSize: int = 0
    createdAt: int = 0
    lastAccessedAt: int = 0
    accessCount: int = 0


class DocListResponse(BaseModel):
    docs: list[DocSummary]
    total: int
    offset: int
    limit: int
    sort: str


class DocDetailResponse(BaseModel):
    meta: DocSummary
    manifest: dict[str, Any] | None = None
    artifactPresent: bool


class StatsResponse(BaseModel):
    documents: int
    totalBytes: int
    parseAttempts: int
    cacheHits: int
    parseFailures: int
    lockContentions: int
    cleanupRuns: int
    hitRate: float


class CleanupResponse(BaseModel):
    evicted: list[str]
    totalBytesBefore: int
    totalBytesAfter: int
    maxBytes: int


@router.get("/v1/files/docs", response_model=DocListResponse, tags=["Files Admin"], dependencies=[Depends(require_api_key_strict)])
def list_docs(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    sort: str = Query("recent", pattern="^(recent|oldest|size|name)$"),
    q: str | None = Query(None, description="Case-insensitive substring filter on file name"),
) -> DocListResponse:
    docs, total = registry.list_docs(offset=offset, limit=limit, sort=sort, query=q)
    return DocListResponse(
        docs=[DocSummary(**_coerce_meta(doc)) for doc in docs],
        total=total,
        offset=offset,
        limit=limit,
        sort=sort,
    )


@router.get("/v1/files/docs/{doc_id}", response_model=DocDetailResponse, tags=["Files Admin"], dependencies=[Depends(require_api_key_strict)])
def get_doc(doc_id: str) -> DocDetailResponse:
    validate_doc_id(doc_id)
    meta = registry.get(doc_id)
    if not meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found in registry.")
    document = artifacts.read(doc_id)
    return DocDetailResponse(
        meta=DocSummary(**_coerce_meta(meta)),
        manifest=document.get("manifest") if document else None,
        artifactPresent=bool(document),
    )


@router.delete("/v1/files/docs/{doc_id}", tags=["Files Admin"], dependencies=[Depends(require_api_key_strict)])
def delete_doc(doc_id: str) -> dict[str, Any]:
    validate_doc_id(doc_id)
    # Capture pre-state so we report 404 correctly even when only one of the
    # two tiers (registry vs artifact) holds the document.
    in_registry = registry.exists(doc_id)
    on_disk = artifacts.exists(doc_id)
    if not in_registry and not on_disk:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    artifacts.delete(doc_id)
    registry.remove(doc_id)
    return {"docId": doc_id, "deleted": True}


@router.post("/v1/files/docs/cleanup", response_model=CleanupResponse, tags=["Files Admin"], dependencies=[Depends(require_api_key_strict)])
def cleanup_docs() -> CleanupResponse:
    max_bytes = config.parsed_files_max_bytes()
    before = artifacts.total_bytes()
    evicted = registry.evict_until(max_bytes)
    after = artifacts.total_bytes()
    registry.record_stat("cleanup_runs")
    return CleanupResponse(evicted=evicted, totalBytesBefore=before, totalBytesAfter=after, maxBytes=max_bytes)


@router.get("/v1/files/stats", response_model=StatsResponse, tags=["Files Admin"], dependencies=[Depends(require_api_key_strict)])
def get_stats() -> StatsResponse:
    stats = registry.get_stats()
    attempts = stats.get("parse_attempts", 0)
    hits = stats.get("cache_hits", 0)
    return StatsResponse(
        documents=registry.count(),
        totalBytes=artifacts.total_bytes(),
        parseAttempts=attempts,
        cacheHits=hits,
        parseFailures=stats.get("parse_failures", 0),
        lockContentions=stats.get("lock_contentions", 0),
        cleanupRuns=stats.get("cleanup_runs", 0),
        hitRate=(hits / attempts) if attempts else 0.0,
    )


@router.post("/v1/files/stats/reset", tags=["Files Admin"], dependencies=[Depends(require_api_key_strict)])
def reset_stats() -> dict[str, Any]:
    return {"stats": registry.reset_stats()}


def _coerce_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Tolerant coercion: registry entries written by older versions may miss
    fields; fill DocSummary defaults so the response never 500s on a partial
    record."""
    fields = DocSummary.model_fields
    return {key: meta.get(key) for key in fields}
