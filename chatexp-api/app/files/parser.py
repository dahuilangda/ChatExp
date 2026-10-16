"""docling-serve client — the sole document parsing backend.

Every supported format (PDF, DOCX, PPTX, XLSX, HTML, CSV, JSON, Markdown, plain
text, images, ...) is parsed by docling-serve's /v1/convert/source endpoint.
There is no fallback parser: if docling is unreachable or rejects the input,
the request fails with a clear error rather than silently degrading. docling
returns both markdown and the structured DoclingDocument (Doctags/JSON), which
we consume to build a section-aware manifest.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from . import config


class ParseError(Exception):
    pass


@dataclass
class TextItem:
    text: str
    label: str = "paragraph"          # docling label: title/section_header/paragraph/list_item/...
    page: int | None = None
    heading_level: int | None = None  # 1 for title, 2 for section_header depth, etc.


@dataclass
class TableItem:
    rows: list[list[list[Any]]]       # row -> cells -> cell paragraphs (each a list of {text, label})
    page: int | None = None
    n_rows: int = 0
    n_cols: int = 0


@dataclass
class ParseResult:
    markdown: str = ""
    texts: list[TextItem] = field(default_factory=list)
    tables: list[TableItem] = field(default_factory=list)
    total_pages: int | None = None
    parser_engine: str = "docling"
    low_confidence_pages: list[int] = field(default_factory=list)


def parse(data: bytes, name: str, mime: str, doc_options: dict[str, Any], celery_task=None) -> ParseResult:
    """Parse via docling-serve. There is no fallback: a docling failure is a
    hard failure. Returns a unified ParseResult built from docling's structured
    DoclingDocument output.

    OCR is adaptive: docling's do_ocr significantly increases processing time
    (docling docs: "OCR significantly increases processing time"), so text-type
    documents are converted WITHOUT OCR first (fast); only when the extracted
    text is too sparse (a scanned/image-based document) do we retry WITH OCR.

    When ``celery_task`` is provided (a bound Celery task), progress is reported
    via ``self.update_state(state='PROGRESS', meta=...)`` so the frontend can
    show queue position and estimated time.
    """
    try:
        result = _parse_via_docling(data, name, mime, doc_options, ocr=False, celery_task=celery_task)
    except ParseError as first_error:
        if not _ocr_requested(doc_options):
            raise
        # ONLY a "no structured content" failure (scanned/image document) is a
        # legitimate reason to retry with OCR. Transport/timeout/docling errors
        # must surface as-is — retrying them with OCR would double the wait and
        # mask the real cause.
        if "no structured content" not in str(first_error):
            raise
        # The fast no-OCR pass produced nothing usable — retry WITH OCR.
        return _parse_via_docling(data, name, mime, doc_options, ocr=True, celery_task=celery_task)
    if _text_too_sparse(result) and _ocr_requested(doc_options):
        # Fast pass succeeded but extracted almost no text — scanned document.
        # Retry WITH OCR (slow but required).
        result = _parse_via_docling(data, name, mime, doc_options, ocr=True, celery_task=celery_task)
        if _text_too_sparse(result):
            # OCR ran and STILL extracted (almost) nothing. Reporting success
            # with an empty manifest would tell the caller the document has no
            # content — fail loudly so the user knows OCR failed.
            raise ParseError(
                "OCR extracted no usable text from this document. It may be a "
                "pure-image or severely degraded scan; no analysis is possible."
            )
    return result

def _ocr_requested(doc_options: dict[str, Any]) -> bool:
    return bool(doc_options.get("ocr", True))


def _text_too_sparse(result: ParseResult) -> bool:
    """Heuristic for "scanned document": almost no text extracted from a
    non-empty document. A real text PDF yields far more than this."""
    if result is None:
        return True
    total = sum(len((item.text or "").strip()) for item in getattr(result, "texts", []) or [])
    return total < 40


# Empirical average parse time for a typical multi-page PDF on this CPU-only
# docling-serve deployment (measured from a 120-page PDF at ~560s without OCR).
# Used for time-based progress estimation since docling-serve's local
# orchestrator does not report page-level progress.
_AVERAGE_PARSE_SECONDS = 560.0
# Byte size of the reference document the average above was measured on
# (~50KB/page for text PDFs). The estimate scales linearly with input size and
# is clamped so tiny documents do not promise instant completion while huge
# ones do not promise hours.
_REFERENCE_BYTES = 120 * 50 * 1024
_MIN_ESTIMATED_SECONDS = 5.0


def _estimated_parse_seconds(num_bytes: int, ocr: bool) -> float:
    scaled = _AVERAGE_PARSE_SECONDS * (max(num_bytes, 1) / _REFERENCE_BYTES)
    return max(_MIN_ESTIMATED_SECONDS, scaled) * (2.0 if ocr else 1.0)


def _parse_via_docling(data: bytes, name: str, mime: str, doc_options: dict[str, Any], ocr: bool, celery_task=None) -> ParseResult:
    base_url = config.docling_base_url()
    if not base_url:
        raise ParseError(
            "Server-side document parsing is not configured (CHATEXP_DOCLING_BASE_URL is empty)."
        )
    timeout = config.docling_timeout()
    b64 = base64.b64encode(data).decode("ascii")
    payload = {
        "options": {
            "to_formats": ["md", "json"],
            "do_ocr": ocr,
            "do_table_structure": bool(doc_options.get("table_structure", True)),
        },
        "sources": [
            {
                # FileSourceRequest: kind="file" with base64_string + filename.
                "kind": "file",
                "base64_string": b64,
                "filename": name or "document",
            }
        ],
    }
    # Use the ASYNC conversion API: the sync endpoint caps at
    # DOCLING_SERVE_MAX_SYNC_WAIT (120s by default), which large/OCR-heavy
    # documents exceed. The async flow is: submit -> poll status -> fetch result.
    submit_endpoint = f"{base_url}/v1/convert/source/async"
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(submit_endpoint, json=payload)
            if response.status_code >= 400:
                raise ParseError(f"docling-serve submit returned HTTP {response.status_code}: {response.text[:500]}")
            try:
                task = response.json()
            except ValueError as exc:
                raise ParseError("docling-serve returned a non-JSON submit response.") from exc
            task_id = task.get("task_id")
            if not task_id:
                raise ParseError(f"docling-serve submit returned no task_id: {str(task)[:300]}")
            # Poll until completion (no sync timeout; bounded by our own timeout).
            # The docling-serve `wait` param does NOT reliably block on the
            # local orchestrator (observed: immediate 200 while running), so we
            # enforce our own backoff between polls to avoid hammering the API.
            poll_endpoint = f"{base_url}/v1/status/poll/{task_id}"
            deadline = time.monotonic() + timeout
            poll_interval = 2.0
            start_time = time.monotonic()
            # When OCR is enabled, the parse takes ~2x longer than no-OCR.
            estimated_total = _estimated_parse_seconds(len(data), ocr)
            while True:
                if time.monotonic() > deadline:
                    raise ParseError(f"docling-serve conversion timed out after {timeout}s (task {task_id}).")
                status = client.get(poll_endpoint, params={"wait": 30.0})
                if status.status_code >= 400:
                    raise ParseError(f"docling-serve status returned HTTP {status.status_code}: {status.text[:500]}")
                try:
                    status_body = status.json()
                except ValueError as exc:
                    raise ParseError("docling-serve returned a non-JSON status response.") from exc
                task_status = str(status_body.get("task_status") or status_body.get("status") or "").lower()
                # docling-serve 1.29 TaskStatus enum: success/pending/started/failure.
                if task_status in ("success", "completed"):
                    break
                if task_status in ("failure", "failed", "canceled", "cancelled", "error", "revoked"):
                    raise ParseError(
                        f"docling-serve task {task_id} {task_status}: "
                        f"{status_body.get('error_message') or status_body.get('failure') or 'unknown error'}"
                    )
                if task_status not in ("pending", "started", "queued", "running", "processing"):
                    # Unknown status string: surface it instead of polling into
                    # the timeout — an unexpected enum value is a real problem.
                    raise ParseError(f"docling-serve task {task_id} returned unknown status {task_status!r}.")
                # Report progress to celery so the frontend can show it.
                if celery_task is not None:
                    queue_position = status_body.get("task_position")
                    elapsed = time.monotonic() - start_time
                    celery_task.update_state(
                        state="PROGRESS",
                        meta={
                            "queuePosition": queue_position if isinstance(queue_position, int) else None,
                            "elapsedSeconds": round(elapsed, 1),
                            "estimatedTotalSeconds": estimated_total,
                        },
                    )
                # Task still running/queued: back off before the next poll.
                # Start at 2s and cap at 30s so long conversions (minutes) do
                # not hammer docling-serve.
                time.sleep(poll_interval)
                poll_interval = min(30.0, poll_interval * 1.5)
            # Fetch the converted result.
            result_endpoint = f"{base_url}/v1/result/{task_id}"
            result = client.get(result_endpoint)
            if result.status_code >= 400:
                raise ParseError(f"docling-serve result returned HTTP {result.status_code}: {result.text[:500]}")
            try:
                envelope = result.json()
            except ValueError as exc:
                raise ParseError("docling-serve returned a non-JSON result.") from exc
    except ParseError:
        raise
    except httpx.RequestError as exc:
        raise ParseError(f"docling-serve unreachable: {exc}") from exc
    return _from_docling(envelope)


def _from_docling(envelope: Any) -> ParseResult:
    """Normalize a docling-serve /v1/convert/source response envelope.

    docling-serve returns a list of per-document results. Each item carries the
    markdown in ``document.md_content`` and the structured DoclingDocument in
    ``document.json_content`` (a dict). We read those exact field names.
    """
    docs = envelope if isinstance(envelope, list) else [envelope]
    result = ParseResult()
    merged_md: list[str] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        body = doc.get("document") if isinstance(doc.get("document"), dict) else doc
        md = body.get("md_content") or ""
        if md:
            merged_md.append(md)
        doc_json = body.get("json_content")
        if isinstance(doc_json, dict):
            _consume_docling_json(doc_json, result)
        # Pages / OCR provenance for low-confidence detection.
        _collect_provenance(doc, body, result)
    result.markdown = "\n\n".join(merged_md).strip()
    if not result.texts:
        # docling returned no structured text items. This means the conversion
        # produced nothing usable (empty document, unreadable bytes, or an
        # unexpected response shape) — fail explicitly rather than synthesizing
        # a fake structure or returning an empty manifest.
        raise ParseError(
            "docling returned no structured content for this document. "
            "The file may be empty, corrupt, or in an unsupported variant."
        )
    return result


def _consume_docling_json(doc_json: dict[str, Any], result: ParseResult) -> None:
    texts = doc_json.get("texts")
    if isinstance(texts, list):
        for entry in texts:
            if not isinstance(entry, dict):
                continue
            text = (entry.get("text") or "").strip()
            if not text:
                continue
            label = str(entry.get("label") or "paragraph")
            prov = entry.get("prov") or entry.get("provenance")
            page = _first_page(prov)
            level = entry.get("level") if isinstance(entry.get("level"), int) else None
            result.texts.append(TextItem(text=text, label=label, page=page, heading_level=level))
    tables = doc_json.get("tables")
    if isinstance(tables, list):
        for index, entry in enumerate(tables):
            if not isinstance(entry, dict):
                continue
            prov = entry.get("prov") or entry.get("provenance")
            page = _first_page(prov)
            # docling nests row/col counts and the cell grid under `data`:
            #   data.num_rows, data.num_cols, data.grid (2-D array of cells)
            # Each cell carries `text`, plus bbox/spans. We render the grid to
            # a markdown-style table string so the table content reaches the
            # manifest text blocks the LLM actually reads.
            data_obj = entry.get("data") if isinstance(entry.get("data"), dict) else {}
            n_rows = _as_int(data_obj.get("num_rows"), entry.get("num_rows"), entry.get("n_rows"))
            n_cols = _as_int(data_obj.get("num_cols"), entry.get("num_cols"), entry.get("n_cols"))
            grid = data_obj.get("grid") or entry.get("grid") or []
            table_text = _render_table_grid(grid, n_rows, n_cols)
            result.tables.append(TableItem(rows=grid, page=page, n_rows=n_rows, n_cols=n_cols))
            if table_text:
                result.texts.append(TextItem(text=table_text, label="table", page=page))
    pages = doc_json.get("pages")
    if isinstance(pages, dict):
        # pages is { "1": {...}, "2": {...} }
        try:
            result.total_pages = max(int(key) for key in pages.keys() if str(key).isdigit())
        except ValueError:
            pass
    elif isinstance(pages, list):
        result.total_pages = len(pages)


def _as_int(*values: Any, default: int = 0) -> int:
    """Tolerant int coercion across several candidate fields. docling has
    changed numeric field shapes before (int vs numeric string), so we parse
    defensively and fall back to `default` rather than letting a ValueError
    escape and fail the whole document."""
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return default


def _render_table_grid(grid: Any, n_rows: int, n_cols: int) -> str:
    """Render docling's table grid (2-D array of cells, each with a `text`
    field) into a compact markdown-style table string so the table content is
    readable as plain text in the manifest, not just metadata.

    docling cells also carry row_span/col_span/column_header flags; we render
    the cell text only (merging is rare in tabular data we ingest) and keep the
    first row as a header separator for readability.
    """
    if not isinstance(grid, list) or not grid:
        return ""
    rows_out: list[str] = []
    header_done = False
    for row in grid:
        if not isinstance(row, list):
            continue
        cells = []
        for cell in row:
            if isinstance(cell, dict):
                cells.append(str(cell.get("text") or "").strip())
            else:
                cells.append(str(cell or "").strip())
        if not cells:
            continue
        rows_out.append("| " + " | ".join(cells) + " |")
        if not header_done:
            rows_out.append("| " + " | ".join(["---"] * len(cells)) + " |")
            header_done = True
    return "\n".join(rows_out)


def _collect_provenance(doc: dict[str, Any], body: dict[str, Any], result: ParseResult) -> None:
    """Identify pages whose text was recovered via OCR / low-confidence layout.

    docling marks OCR-sourced items in each text item's provenance
    (``ocr: true`` or a ``bbox`` from an OCR pipeline). Pages that are heavily
    OCR-dependent are the best visual-fallback candidates for the VLM.
    """
    doc_json = body.get("json_content") if isinstance(body, dict) else None
    if not isinstance(doc_json, dict):
        return
    ocr_pages: set[int] = set()
    for entry in doc_json.get("texts", []) or []:
        if not isinstance(entry, dict):
            continue
        prov = entry.get("prov") or entry.get("provenance")
        page = _first_page(prov)
        is_ocr = bool(entry.get("ocr")) or _prov_is_ocr(prov)
        if is_ocr and page is not None:
            ocr_pages.add(page)
    if ocr_pages and not result.low_confidence_pages:
        result.low_confidence_pages = sorted(ocr_pages)[:8]


def _first_page(prov: Any) -> int | None:
    """Extract a 1-based page number from docling provenance, which may be a
    dict, a list of dicts, or absent."""
    if isinstance(prov, dict):
        for key in ("page", "page_no", "pageno"):
            value = prov.get(key)
            if isinstance(value, int):
                return value
    if isinstance(prov, list) and prov:
        return _first_page(prov[0])
    return None


def _prov_is_ocr(prov: Any) -> bool:
    items = prov if isinstance(prov, list) else [prov]
    for item in items:
        if isinstance(item, dict) and (item.get("ocr") or item.get("pipeline") == "ocr"):
            return True
    return False


def _suffix(name: str) -> str:
    dot = name.rfind(".")
    return name[dot:].lower() if dot >= 0 else ""
