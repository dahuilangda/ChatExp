"""HTTP endpoints for server-side file parsing.

- POST   /v1/files/parse            enqueue a parse job, returns {job_id, status_url}
- GET    /v1/files/jobs/{job_id}    poll job status + manifest when done
- POST   /v1/files/jobs/{job_id}/fetch   pull section text / page images

Mirrors the existing /v1/bo/* + /v1/jobs/{job_id} pattern in main.py: celery
.enqueue via .delay, AsyncResult polling, status_url via request.url_for, and
the same BO API-key gate applied to every endpoint.
"""
from __future__ import annotations

import logging

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, status

from app.security import require_api_key, require_api_key_for, validate_doc_id, validate_llm_endpoint
from app.tasks import celery_app

logger = logging.getLogger(__name__)

from . import artifacts, fetch as fetch_module, registry
from .models import (
    AnalyzeProgress,
    AnalyzeRequest,
    AnalyzeResult,
    EditJobResponse,
    EditJobResult,
    EditOperation,
    EditProgress,
    EditRequest,
    FetchRequest,
    FetchResponse,
    JobProgress,
    JobResult,
    Manifest,
    ParseJobResponse,
    ParseRequest,
)
from .tasks import analyze_document_task, edit_file_task, parse_file_task
router = APIRouter()


@router.post(
    "/v1/files/parse",
    response_model=ParseJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Files"],
    dependencies=[Depends(require_api_key)],
)
def parse_file(payload: ParseRequest) -> ParseJobResponse:
    # Validation is about INPUT SHAPE only: the plugin must supply either
    # inline bytes (contentBase64), a SeaTable asset path, or a direct url.
    # Credential completeness is checked in the worker, which knows whether
    # CHATEXP_SEATABLE_API_TOKEN is configured and fails with a precise reason
    # when the plugin supplied no token. Checking credentials here would either
    # reject valid server-configured requests or (worse) accept broken ones and
    # hide the real error.
    if not (payload.contentBase64 or payload.path or payload.url):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide inline file bytes, a SeaTable asset path, or a direct url.")
    logger.info(
        "parse request: name=%r path=%r server=%r apiToken=%s inline=%s url=%r",
        payload.name,
        payload.path,
        payload.server,
        f"<{len(payload.apiToken)} chars>" if payload.apiToken else "<empty>",
        f"<{len(payload.contentBase64)} chars>" if payload.contentBase64 else "no",
        payload.url,
    )
    task = parse_file_task.delay(payload.model_dump())
    return ParseJobResponse(
        job_id=task.id,
        status="queued",
        # Relative path: the api sits behind proxies/tunnels whose scheme/host
        # differs from what request.url_for() would infer (it sees the internal
        # http endpoint). Returning a relative path lets every client resolve it
        # against the base URL it already knows (the HTTPS proxy endpoint).
        status_url=f"/v1/files/jobs/{task.id}",
    )


@router.get(
    "/v1/files/jobs/{job_id}",
    response_model=JobResult,
    tags=["Files"],
    dependencies=[Depends(require_api_key)],
)
def get_file_job(job_id: str) -> JobResult:
    result: AsyncResult = AsyncResult(job_id, app=celery_app)
    state = result.state.lower()
    if result.successful():
        outcome = result.result
        if isinstance(outcome, dict) and outcome.get("error"):
            return JobResult(job_id=job_id, status="failed", error=str(outcome["error"]))
        manifest_dict = (outcome or {}).get("manifest") if isinstance(outcome, dict) else None
        if not manifest_dict:
            # A parse that "succeeded" without a manifest lost the document
            # (e.g. a corrupt cached artifact) — never report success with
            # nothing.
            return JobResult(job_id=job_id, status="failed", error="Parse finished but produced no manifest. The cached artifact may be corrupt; re-parse the file.")
        return JobResult(job_id=job_id, status="succeeded", manifest=Manifest.model_validate(manifest_dict))
    if result.failed():
        return JobResult(job_id=job_id, status="failed", error=_failure_reason(result))
    if state == "revoked":
        return JobResult(job_id=job_id, status="canceled")
    # Surface docling progress (queue position + elapsed/estimated time) when
    # the celery task is in PROGRESS state. The worker calls update_state with
    # meta={queuePosition, elapsedSeconds, estimatedTotalSeconds} during poll.
    progress: JobProgress | None = None
    if state == "progress" and isinstance(result.info, dict):
        progress = JobProgress(
            queuePosition=result.info.get("queuePosition"),
            elapsedSeconds=float(result.info.get("elapsedSeconds") or 0),
            estimatedTotalSeconds=float(result.info.get("estimatedTotalSeconds") or 0),
        )
    return JobResult(
        job_id=job_id,
        status="running" if state in ("started", "progress") else "queued",
        progress=progress,
    )


def _fetch_parsed_document(doc_id: str, request: FetchRequest, not_found_detail: str) -> FetchResponse:
    """Read the parsed artifact, refresh its access stamp, and return the
    requested sections."""
    validate_doc_id(doc_id)
    document = artifacts.read(doc_id)
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=not_found_detail)
    registry.touch(doc_id)
    try:
        return fetch_module.fetch(
            document,
            section_ids=request.sectionIds,
            page_images=request.pageImages,
            max_chars=request.maxChars,
        )
    except fetch_module.PageImagesNotSupported as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post(
    "/v1/files/jobs/{job_id}/fetch",
    response_model=FetchResponse,
    tags=["Files"],
    dependencies=[Depends(require_api_key)],
)
def fetch_file(job_id: str, request: FetchRequest) -> FetchResponse:
    result: AsyncResult = AsyncResult(job_id, app=celery_app)
    if not result.successful():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Job is {result.state.lower()}, not succeeded.")
    outcome = result.result
    if isinstance(outcome, dict) and outcome.get("error"):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(outcome["error"]))
    doc_id = (outcome or {}).get("docId") if isinstance(outcome, dict) else None
    if not doc_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job has no associated document.")
    # Artifacts live on the shared parsed-files volume, not redis, so a fetch
    # after the registry metadata expired can still succeed if the artifact is
    # on disk; refresh the access stamp when it does.
    return _fetch_parsed_document(
        doc_id,
        request,
        not_found_detail="Parsed document artifact is unavailable. Re-parse the file.",
    )


def _failure_reason(result: AsyncResult) -> str:
    try:
        reason = str(result.result)
        if reason.strip():
            return reason
        return f"Job failed with {type(result.result).__name__}."
    except Exception as exc:  # noqa: BLE001 — the reason itself is unavailable
        logger.warning("Could not read failure reason for job %s: %s", getattr(result, "id", "?"), exc)
        return f"Job failed with {type(exc).__name__}."


# --- Fast path: retrieve an already-parsed document by doc_id ----------------
# The plugin caches url→docId in sessionStorage and calls this endpoint to get
# sections without re-downloading or re-running docling. 404 means the artifact
# was evicted → the plugin re-parses.


@router.post(
    "/v1/files/docs/{doc_id}/fetch",
    response_model=FetchResponse,
    tags=["Files"],
    dependencies=[Depends(require_api_key)],
)
def fetch_document(doc_id: str, request: FetchRequest) -> FetchResponse:
    """Fetch parsed sections/page images directly by doc_id, bypassing the
    celery job layer. Used from the plugin's sessionStorage url→docId cache so
    a previously-parsed file is retrieved in milliseconds without re-downloading
    or re-parsing."""
    return _fetch_parsed_document(
        doc_id,
        request,
        not_found_detail="Document not found. It may have been evicted; re-parse the file.",
    )


# --- Document analysis agent ------------------------------------------------


@router.post(
    "/v1/files/analyze",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Files"],
    # Strict gate (same rationale as /v1/files/edit): the analysis agent
    # forwards content to a client-named LLM endpoint and burns real tokens
    # — it must never run unauthenticated. Refuses with 503 until
    # CHATEXP_BO_API_KEY is configured; the client sends the same key.
    dependencies=[Depends(require_api_key_for("/v1/files/analyze"))],
)
def analyze_file(payload: AnalyzeRequest) -> dict[str, str]:
    """Enqueue the server-side analysis agent for a previously-parsed document
    (docId). Inline images are not analyzed — parse them via /v1/files/parse so
    docling OCR runs at parse time. Progress is polled via
    GET /v1/files/analyze/{job_id}."""
    # Only validate docId when it is actually provided — image-only requests
    # have docId="" and must NOT be rejected by the 32-hex pattern check.
    if payload.docId:
        validate_doc_id(payload.docId)
    # A client-supplied llm.endpoint receives document content in the worker;
    # gate it against the same allowlist /v1/llm/chat uses — fail fast here,
    # not after the job is queued.
    endpoint = str((payload.llm or {}).get("endpoint") or "").strip()
    if endpoint:
        try:
            validate_llm_endpoint(endpoint)
        except HTTPException as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.detail) from exc
    task = analyze_document_task.delay(payload.model_dump())
    return {"job_id": task.id, "status": "queued"}


@router.get(
    "/v1/files/analyze/{job_id}",
    response_model=AnalyzeResult,
    tags=["Files"],
    dependencies=[Depends(require_api_key)],
)
def get_analyze_job(job_id: str) -> AnalyzeResult:
    """Poll the analysis agent job."""
    result: AsyncResult = AsyncResult(job_id, app=celery_app)
    state = result.state.lower()
    if result.successful():
        outcome = result.result
        if isinstance(outcome, dict) and outcome.get("error"):
            return AnalyzeResult(job_id=job_id, status="failed", error=str(outcome["error"]))
        answer = (outcome or {}).get("answer") if isinstance(outcome, dict) else None
        if not answer or not str(answer).strip():
            return AnalyzeResult(job_id=job_id, status="failed", error="Analysis completed but produced an empty answer.")
        return AnalyzeResult(job_id=job_id, status="succeeded", answer=str(answer))
    if result.failed():
        return AnalyzeResult(job_id=job_id, status="failed", error=_failure_reason(result))
    if state == "revoked":
        return AnalyzeResult(job_id=job_id, status="canceled")
    progress: AnalyzeProgress | None = None
    if state == "progress" and isinstance(result.info, dict):
        progress = AnalyzeProgress(
            phase=result.info.get("phase", "analyzing"),
            detail=result.info.get("detail", ""),
            elapsedSeconds=float(result.info.get("elapsedSeconds") or 0),
            estimatedTotalSeconds=float(result.info.get("estimatedTotalSeconds") or 30),
            queuePosition=result.info.get("queuePosition"),
        )
    return AnalyzeResult(
        job_id=job_id,
        status="running" if state in ("started", "progress") else "queued",
        progress=progress,
    )


# --- Word track-changes editing ----------------------------------------------


@router.post(
    "/v1/files/edit",
    response_model=EditJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Files"],
    # Strict gate (unlike the lenient parse/analyze routes): this route takes
    # ~200MB inline bodies and incurs LLM cost planning the edits — it must
    # not be reachable unauthenticated when CHATEXP_BO_API_KEY is unset.
    dependencies=[Depends(require_api_key_for("/v1/files/edit"))],
)
def edit_file(payload: EditRequest) -> EditJobResponse:
    """Enqueue a tracked-changes Word edit. The plugin (fileEditHarness.js)
    downloads the attachment bytes in the browser and posts them inline; the
    worker plans the edits with the LLM and applies them as native Word
    revisions. Poll via GET /v1/files/edit/{job_id}.

    Input-shape validation only (same policy as /v1/files/parse): inline bytes
    + instruction must be present; everything else fails in the worker with a
    precise reason."""
    if not payload.contentBase64.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide the document bytes inline (contentBase64).")
    if not payload.instruction.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide the editing instruction.")
    logger.info(
        "edit request: name=%r instruction=%s inline=%s",
        payload.name,
        f"<{len(payload.instruction)} chars>",
        f"<{len(payload.contentBase64)} chars>",
    )
    task = edit_file_task.delay(payload.model_dump())
    return EditJobResponse(
        job_id=task.id,
        status="queued",
        # Relative path for the same proxy/tunnel reasons as the parse jobs.
        status_url=f"/v1/files/edit/{task.id}",
    )


@router.get(
    "/v1/files/edit/{job_id}",
    response_model=EditJobResult,
    tags=["Files"],
    # The result carries fileBase64 of the edited document (and the enqueue
    # side is strict) — polling must not be reachable through the lenient gate.
    dependencies=[Depends(require_api_key_for("/v1/files/edit"))],
)
def get_edit_job(job_id: str) -> EditJobResult:
    """Poll an edit job. Response fields match the plugin's fileEditHarness
    contract: status (queued/running/succeeded/failed/canceled), progress
    while running, and on success fileBase64/fileName/editsApplied/
    editsFailed/operations (convertedFromLegacy stays false — legacy formats
    are refused, never converted)."""
    try:
        return _edit_job_status(job_id)
    except Exception as exc:  # noqa: BLE001 — unknown/expired ids whose result backend raises must degrade to a status shape, never a 500
        logger.warning("Edit job %s status could not be read: %s", job_id, exc)
        return EditJobResult(
            job_id=job_id,
            status="failed",
            error=f"Job status could not be read: {type(exc).__name__}: {exc}",
        )


def _edit_job_status(job_id: str) -> EditJobResult:
    result: AsyncResult = AsyncResult(job_id, app=celery_app)
    state = result.state.lower()
    if result.successful():
        outcome = result.result
        if isinstance(outcome, dict) and outcome.get("error"):
            return EditJobResult(job_id=job_id, status="failed", error=str(outcome["error"]))
        if not isinstance(outcome, dict) or not outcome.get("fileBase64"):
            # An edit that "succeeded" without a file lost the document —
            # never report success with nothing.
            return EditJobResult(job_id=job_id, status="failed", error="Edit finished but produced no document.")
        return EditJobResult(
            job_id=job_id,
            status="succeeded",
            fileBase64=str(outcome.get("fileBase64") or ""),
            fileName=str(outcome.get("fileName") or ""),
            convertedFromLegacy=bool(outcome.get("convertedFromLegacy")),
            editsApplied=int(outcome.get("editsApplied") or 0),
            editsFailed=int(outcome.get("editsFailed") or 0),
            operations=[EditOperation.model_validate(op) for op in (outcome.get("operations") or []) if isinstance(op, dict)],
        )
    if result.failed():
        return EditJobResult(job_id=job_id, status="failed", error=_failure_reason(result))
    if state == "revoked":
        return EditJobResult(job_id=job_id, status="canceled")
    progress: EditProgress | None = None
    if state == "progress" and isinstance(result.info, dict):
        progress = EditProgress(
            phase=result.info.get("phase", "editing"),
            detail=result.info.get("detail", ""),
            progressPercent=float(result.info.get("progressPercent") or 0),
            elapsedSeconds=float(result.info.get("elapsedSeconds") or 0),
            estimatedTotalSeconds=float(result.info.get("estimatedTotalSeconds") or 120),
        )
    return EditJobResult(
        job_id=job_id,
        status="running" if state in ("started", "progress") else "queued",
        progress=progress,
    )
