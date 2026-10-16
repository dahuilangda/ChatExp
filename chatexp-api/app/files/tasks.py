"""Celery tasks: download + parse a SeaTable attachment, run the document
analysis agent, and edit Word documents with tracked changes.

Registered on the same celery_app the BO tasks use (app.tasks.celery_app); the
deployment runs queue-partitioned workers off that one app (see
docker-compose.yml), each consuming only the queues task_routes assigns.

Concurrency model (parse):
  Two identical requests racing on the same file must NOT run docling twice.
  The content registry (keyed by sha256(bytes,opts)) is the primary dedup:
  identical bytes reuse one artifact across rows/users. A per-source lock
  guards the download+parse critical section so a concurrent duplicate waits
  out and is resolved by the cache on the caller's natural retry. There is no
  silent wait-and-reuse path and no retry-on-timeout: if the lock cannot be
  acquired the request fails with an explicit conflict so the client can retry
  deliberately.

  Edits take no lock and no dedup: every edit job carries its own inline
  bytes + instruction, so two jobs never contend on shared state.

Failures are captured as a structured result rather than re-raised, so
the job-status endpoints report a clean error message without stack traces.
"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import time
from typing import Any

from app.tasks import celery_app

logger = logging.getLogger(__name__)

from . import agent, artifacts, config, docx_editor, downloader, edit_planner, manifest, parser, registry
from .skill_runners import AnalyzeError


class FileParseError(Exception):
    pass


@celery_app.task(name="chatexp.files.parse", bind=True)
def parse_file_task(self, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return _run(payload, celery_task=self)
    except (FileParseError, downloader.DownloadError, parser.ParseError) as exc:
        registry.record_stat("parse_failures")
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 — surface any worker crash cleanly
        registry.record_stat("parse_failures")
        logger.exception("Parse worker crashed")
        return {"error": f"Unexpected parsing failure: {type(exc).__name__}: {exc}"}


def _source_lock_key(payload: dict[str, Any]) -> str:
    """Stable hash of the SeaTable source locator + parse options. Computable
    before download, so two requests for the same row's attachment race on one
    lock instead of double-downloading. When the plugin supplies the file bytes
    inline (contentBase64), the content itself is the locator: the key hashes
    the payload's length + first 4096 chars, so two different files that share
    a name never contend on each other's lock."""
    inline = str(payload.get("contentBase64") or "")
    if inline:
        return hashlib.sha1(f"inline:{len(inline)}:{inline[:4096]}".encode("utf-8")).hexdigest()[:24]
    parts = [
        str(payload.get("path") or ""),
        str(payload.get("name") or ""),
        str(payload.get("server") or ""),
        repr(sorted((payload.get("docOptions") or {}).items())),
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:24]


def _content_doc_id(data: bytes, doc_options: dict[str, Any]) -> str:
    return manifest.build_doc_id(data, doc_options)


def _run(payload: dict[str, Any], celery_task=None) -> dict[str, Any]:
    if "maxBytes" not in payload or payload.get("maxBytes") is None:
        payload = {**payload, "maxBytes": config.file_max_bytes()}
    doc_options = payload.get("docOptions") if isinstance(payload.get("docOptions"), dict) else {}

    registry.record_stat("parse_attempts")
    source_key = _source_lock_key(payload)

    # Acquire the per-source lock or fail explicitly. No silent waiting: a
    # concurrent caller that lost the race gets a clear conflict and retries on
    # its own schedule, by which point the holder's result is in the cache.
    if not registry.acquire_parse_lock(source_key):
        registry.record_stat("lock_contentions")
        raise FileParseError(
            "Another parse of this file is in progress. Retry in a moment to pick up its cached result."
        )
    try:
        inline_b64 = str(payload.get("contentBase64") or "").strip()
        if inline_b64:
            # The plugin downloaded the bytes in the browser (only it has the
            # SeaTable session) and shipped them inline. Decode and parse; no
            # SeaTable credentials are needed on the worker.
            import base64
            try:
                data = base64.b64decode(inline_b64, validate=True)
            except Exception as exc:
                raise FileParseError(f"Invalid inline contentBase64: {exc}") from exc
            if not data:
                raise FileParseError("Inline contentBase64 decoded to an empty file.")
            # The inline path bypasses downloader.download, so the size guard
            # is enforced here: the effective limit is the stricter of the
            # client-declared maxBytes and the server-configured cap.
            limit = min(int(payload.get("maxBytes") or config.file_max_bytes()), config.file_max_bytes())
            if len(data) > limit:
                raise FileParseError(
                    f"Inline file is {len(data)} bytes, exceeding the {limit}-byte limit."
                )
            content_type = str(payload.get("contentType") or "") or None
            return _serve_from_content(data, doc_options, payload, content_type=content_type, celery_task=celery_task)
        data, content_type = downloader.download(payload)
        return _serve_from_content(data, doc_options, payload, content_type=content_type, celery_task=celery_task)
    finally:
        registry.release_parse_lock(source_key)


def _serve_from_content(
    data: bytes,
    doc_options: dict[str, Any],
    payload: dict[str, Any],
    content_type: str | None = None,
    celery_task=None,
) -> dict[str, Any]:
    """Cache-first serve by content hash. Parse only on miss, then persist to
    the artifact volume + registry so subsequent requests (from any source with
    identical bytes) hit."""
    doc_id = _content_doc_id(data, doc_options)

    meta = registry.get(doc_id)
    if meta and artifacts.exists(doc_id):
        registry.record_stat("cache_hits")
        registry.touch(doc_id)
        cached_manifest = _read_manifest(doc_id)
        if not cached_manifest:
            # Registry says cached but the artifact does not read back: the
            # file is corrupt. Serving {"manifest": None} here surfaced as a
            # "succeeded" job with no content — fail loudly instead.
            raise FileParseError(
                f"Cached artifact for {doc_id} is unreadable. Re-parse the file."
            )
        return {"docId": doc_id, "manifest": cached_manifest, "cached": True}

    # Cache miss: parse and persist.
    name = str(payload.get("name") or "document")
    mime = content_type or downloader.guess_mime(name)
    result = parser.parse(data, name, mime, doc_options, celery_task=celery_task)
    stored = manifest.build(result, payload, data, mime, doc_options)

    document = {
        "docId": stored.doc_id,
        "fileName": stored.file_name,
        "mimeType": stored.mime_type,
        "manifest": stored.manifest,
        "blocks": stored.blocks,
        "tables": stored.tables,
        "markdown": result.markdown,
    }
    artifacts_path = artifacts.write(stored.doc_id, document)
    byte_size = artifacts.size(stored.doc_id)
    meta = registry.build_meta(stored.doc_id, stored.manifest, byte_size, artifacts_path)
    registry.register(meta)
    return {"docId": stored.doc_id, "manifest": stored.manifest, "cached": False}


def _read_manifest(doc_id: str) -> dict[str, Any] | None:
    document = artifacts.read(doc_id)
    if not document:
        return None
    return document.get("manifest")


# --- Document analysis agent task -------------------------------------------


@celery_app.task(name="chatexp.files.analyze", bind=True)
def analyze_document_task(self, payload: dict[str, Any]) -> dict[str, Any]:
    """Run the server-side analysis agent on a previously-parsed document
    (docId). A request carrying inline images fails explicitly — images are
    OCR'd by docling at parse time, so parse first and analyze the docId.
    Progress is reported via update_state."""
    try:
        doc_id = str(payload.get("docId") or "").strip()
        task_text = str(payload.get("task") or "").strip()
        llm_settings = payload.get("llm") if isinstance(payload.get("llm"), dict) else None
        images = payload.get("images") or []
        if not doc_id and not images:
            return {"error": "Either docId or images must be provided for analysis."}
        answer = agent.analyze_document(
            doc_id=doc_id,
            task=task_text,
            images=images,
            celery_task=self,
            llm_settings=llm_settings,
        )
        return {"answer": answer}
    except agent.AnalyzeError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Analysis worker crashed")
        return {"error": f"Unexpected analysis failure: {type(exc).__name__}: {exc}"}


# --- Word track-changes edit task --------------------------------------------


class FileEditError(Exception):
    """Raised for any recoverable edit-pipeline failure (bad input, wrong file
    type, unusable edit plan). Captured as {"error": ...} by the task wrapper."""


@celery_app.task(name="chatexp.files.edit", bind=True)
def edit_file_task(self, payload: dict[str, Any]) -> dict[str, Any]:
    """Edit a .docx with native Word revisions (w:ins/w:del).

    The bytes arrive inline (contentBase64) — the plugin downloads them in the
    browser, the only place holding the SeaTable session. The LLM plans the
    edits (edit_planner); docx_editor applies them deterministically. Progress
    is reported via update_state; the final result (base64 file + audit) rides
    the celery result backend exactly like the parse manifest does.
    """
    try:
        return _run_edit(payload, celery_task=self)
    except (
        FileEditError,
        docx_editor.EditError,
        AnalyzeError,
    ) as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 — surface any worker crash cleanly
        logger.exception("Edit worker crashed")
        return {"error": f"Unexpected editing failure: {type(exc).__name__}: {exc}"}


def _run_edit(payload: dict[str, Any], celery_task=None) -> dict[str, Any]:
    start = time.monotonic()

    def report(phase: str, detail: str, percent: float, est_total: float = 120.0) -> None:
        if celery_task is None:
            return
        celery_task.update_state(
            state="PROGRESS",
            meta={
                "phase": phase,
                "detail": detail,
                "progressPercent": percent,
                "elapsedSeconds": round(time.monotonic() - start, 1),
                "estimatedTotalSeconds": est_total,
            },
        )

    report("receiving", "Decoding the uploaded document...", 5.0)
    name = str(payload.get("name") or "document.docx")
    instruction = str(payload.get("instruction") or "").strip()
    inline_b64 = str(payload.get("contentBase64") or "").strip()
    if not inline_b64:
        raise FileEditError("No document bytes: contentBase64 is required.")
    if not instruction:
        raise FileEditError("No editing instruction was provided.")
    try:
        data = base64.b64decode(inline_b64, validate=True)
    except Exception as exc:
        raise FileEditError(f"Invalid inline contentBase64: {exc}") from exc
    if not data:
        raise FileEditError("Inline contentBase64 decoded to an empty file.")
    # Same inline-size guard the parse path enforces: the stricter of the
    # client-declared maxBytes and the server-configured cap.
    limit = min(int(payload.get("maxBytes") or config.file_max_bytes()), config.file_max_bytes())
    if len(data) > limit:
        raise FileEditError(
            f"Inline file is {len(data)} bytes, exceeding the {limit}-byte limit."
        )

    # Word-only contract, no conversion layer: legacy .doc/.rtf/.dot and every
    # non-.docx suffix fail here with an explicit reason.
    data, name = docx_editor.ensure_docx(data, name)

    report("planning", "Indexing paragraphs and planning edits...", 20.0)
    try:
        doc = docx_editor.Document(io.BytesIO(data))
    except Exception as exc:
        raise FileEditError(f"Cannot open the document as .docx: {exc}") from exc
    views = docx_editor.index_paragraphs(doc)
    if not views:
        raise FileEditError("The document has no editable text paragraphs.")
    paragraph_view = docx_editor.render_paragraph_view(views)

    report("planning", "The model is planning the edits...", 40.0)
    edits = edit_planner.plan_edits(paragraph_view, instruction)

    report("applying", f"Applying {len(edits)} edit(s) as tracked changes...", 70.0)
    edited, audit = docx_editor.apply_edits(data, edits)
    if audit.all_failed:
        details = "; ".join(f"{outcome.op.pid}: {outcome.detail}" for outcome in audit.failed[:5])
        raise FileEditError(f"No edits could be applied. {details}")
    summary = audit.summary()

    report("validating", "Validating the edited document...", 90.0)
    return {
        "fileBase64": base64.b64encode(edited).decode("ascii"),
        "fileName": name,
        # ensure_docx never converts (legacy formats are refused outright), so
        # this stays a stable False for the plugin's card flag.
        "convertedFromLegacy": False,
        "editsApplied": summary["applied"],
        "editsFailed": summary["failed"],
        "operations": summary["operations"],
    }


# --- Registry maintenance (celery beat) -------------------------------------


@celery_app.task(name="chatexp.enforce_parsed_files_budget")
def enforce_parsed_files_budget_task() -> dict[str, object]:
    """Enforce the parsed-files disk high-water mark (CHATEXP_PARSED_FILES_MAX_BYTES).
    Previously invoked only from the manual admin cleanup route — scheduled
    hourly so the budget is actually kept without human intervention."""
    max_bytes = config.parsed_files_max_bytes()
    evicted = registry.evict_until(max_bytes)
    if evicted:
        logger.info("Parsed-files budget enforced: evicted %d document(s)", len(evicted))
    return {"evicted": len(evicted), "max_bytes": max_bytes}


@celery_app.task(name="chatexp.files.reconcile")
def reconcile_registry_task() -> dict[str, int]:
    """Prune chatexp:doc:index members whose metadata key has expired. The
    ZSET carries no per-member TTL and redis runs volatile-lru (which never
    evicts it), so beat schedules this hourly to keep the index honest."""
    removed = registry.reconcile_index()
    if removed:
        logger.info("Reconciled document index: removed %d orphaned members", removed)
    return {"removed": removed}
