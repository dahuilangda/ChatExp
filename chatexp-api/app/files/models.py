"""Pydantic models for the /v1/files parsing endpoints.

Field names are chosen to line up with both the plugin's fetchViaMediaProxy
contract (url/path/name/server/apiToken) and the plugin's existing
fileContentChunks chunk object shape, so the plugin adapter layer stays thin.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

# Hard bound on client-supplied inline uploads: 200 MiB of decoded bytes.
# contentBase64 carries the same payload base64-encoded (4 chars per 3 bytes).
_MAX_INLINE_BYTES = 200 * 1024 * 1024
_MAX_CONTENT_BASE64_CHARS = 4 * ((_MAX_INLINE_BYTES + 2) // 3)


# --- Requests ---------------------------------------------------------------


class ParseRequest(BaseModel):
    """A SeaTable attachment to parse. The server downloads it on the plugin's
    behalf using the same app-download-link API the browser would have used, so
    file bytes never traverse the browser and never leave the host.

    Either ``url`` or ``path`` must be present. ``path`` is preferred because it
    lets the server resolve a short-lived download link via the SeaTable API;
    ``url`` is a direct download URL used as a fallback.
    """

    url: str | None = None
    path: str = ""
    name: str = ""
    server: str = ""
    apiToken: str = ""
    # Inline file bytes (base64) downloaded by the plugin in the browser — the
    # only client that has the SeaTable session. When present, the worker skips
    # SeaTable download entirely and parses these bytes.
    contentBase64: str = Field("", max_length=_MAX_CONTENT_BASE64_CHARS)
    contentType: str = ""
    maxBytes: int | None = Field(None, ge=1, le=_MAX_INLINE_BYTES)
    docOptions: dict[str, Any] = Field(default_factory=dict)


class FetchRequest(BaseModel):
    """Pull section text out of a parsed document.

    - ``sectionIds`` selects specific sections (agentic on-demand reading).
      When None, all sections up to ``maxChars`` are returned.
    - ``pageImages`` is not implemented: a request carrying page numbers is
      refused with 422. Fetch section text instead — images are OCR'd by
      docling at parse time, so their text is already in the sections.
    - ``maxChars`` caps the total text returned in this call.
    """

    sectionIds: list[str] | None = None
    pageImages: list[int] | None = None
    maxChars: int | None = Field(default=None, ge=1)


# --- Manifest ---------------------------------------------------------------


class SectionInfo(BaseModel):
    sectionId: str
    label: str
    heading: str | None = None
    page: int | None = None
    charCount: int
    estTokens: int
    truncated: bool = False


class TableInfo(BaseModel):
    tableId: str
    page: int | None = None
    rows: int = 0
    cols: int = 0
    sectionId: str | None = None


class Manifest(BaseModel):
    docId: str
    fileName: str
    mimeType: str | None = None
    totalPages: int | None = None
    totalChars: int
    estTokens: int
    sections: list[SectionInfo] = Field(default_factory=list)
    tables: list[TableInfo] = Field(default_factory=list)
    lowConfidencePages: list[int] = Field(default_factory=list)
    parserEngine: str = "docling"


# --- Job responses ----------------------------------------------------------


class ParseJobResponse(BaseModel):
    job_id: str
    status: str = "queued"
    status_url: str


class JobProgress(BaseModel):
    """Progress info surfaced while a parse job is running/queued."""
    queuePosition: int | None = None      # 1-based; None = already dequeued/processing
    elapsedSeconds: float = 0.0
    estimatedTotalSeconds: float = 0.0    # 0 = unknown


class JobResult(BaseModel):
    job_id: str
    status: str
    manifest: Manifest | None = None
    error: str | None = None
    progress: JobProgress | None = None


class FetchChunk(BaseModel):
    """Mirrors the plugin's fileContentChunks chunk object so the plugin can
    adapt it with minimal reshaping (sectionId <-> chunkIndex, page, text)."""

    sectionId: str
    chunkIndex: int
    totalChunks: int
    page: int | None
    label: str
    heading: str | None = None
    text: str
    includedChars: int
    truncated: bool = False


class PageImage(BaseModel):
    page: int
    dataUrl: str


class FetchResponse(BaseModel):
    chunks: list[FetchChunk] = Field(default_factory=list)
    pageImages: list[PageImage] = Field(default_factory=list)
    truncated: bool = False
    nextCursor: str | None = None


# --- Analyze (server-side document analysis agent) --------------------------


class AnalyzeRequest(BaseModel):
    """Request the server-side analysis agent for a previously-parsed document.

    ``docId`` points at a document parsed via /v1/files/parse. Inline images
    are not analyzed — no skill ingests them; images are OCR'd by docling at
    parse time, so a request carrying ``images`` fails explicitly in the
    worker (parse the image first, then analyze its docId).

    The agent runs entirely in the worker, calls the configured LLM, and
    returns a structured answer with progress reported via celery state.
    """

    docId: str = ""
    task: str = Field("", max_length=20_000)
    llm: dict[str, Any] | None = Field(default_factory=dict)
    # Kept only so requests carrying inline images fail explicitly in the
    # worker instead of being silently dropped. Bounded to the pipeline cap.
    images: list[dict[str, str]] = Field(default_factory=list, max_length=64)

    @field_validator("images")
    @classmethod
    def _bound_image_entries(cls, value):
        # Fail image payloads at the GATE, not after enqueue+worker pickup:
        # uvicorn/Caddy read the whole body first, so unbounded dataUrl
        # strings were a memory-amplification surface (64 entries x unbounded).
        for item in value or []:
            for text in (item or {}).values():
                if isinstance(text, str) and len(text) > 1_400_000:
                    raise ValueError(
                        "image dataUrl exceeds the 1MB inline cap; images are not "
                        "analyzed inline anyway — parse them via /v1/files/parse."
                    )
        return value


class AnalyzeProgress(BaseModel):
    phase: str = "planning"  # planning → reading → analyzing → synthesizing
    detail: str = ""
    elapsedSeconds: float = 0.0
    estimatedTotalSeconds: float = 30.0
    queuePosition: int | None = None  # 1-based; None = already dequeued/processing


class AnalyzeResult(BaseModel):
    job_id: str
    status: str
    answer: str | None = None
    progress: AnalyzeProgress | None = None
    error: str | None = None


# --- Word track-changes editing (/v1/files/edit) -----------------------------


class EditRequest(BaseModel):
    """Edit a Word document with tracked changes on the backend.

    The plugin downloads the attachment bytes in the browser (the only place
    with the SeaTable session) and ships them inline — one round trip, no
    server-side SeaTable credentials involved. There is no docId
    form: parse artifacts hold extracted blocks/markdown, not the original
    .docx bytes, and track-changes surgery needs the real file.
    """

    contentBase64: str = Field("", max_length=_MAX_CONTENT_BASE64_CHARS)
    name: str = "document.docx"
    instruction: str = Field("", max_length=20_000)
    maxBytes: int | None = Field(None, ge=1, le=_MAX_INLINE_BYTES)


class EditJobResponse(BaseModel):
    job_id: str
    status: str = "queued"
    status_url: str


class EditProgress(BaseModel):
    """Progress while an edit job runs. ``progressPercent`` is 0-100 (the
    plugin divides by 100 for its UI)."""

    phase: str = "editing"  # receiving → planning → applying → validating
    detail: str = ""
    progressPercent: float = 0.0
    elapsedSeconds: float = 0.0
    estimatedTotalSeconds: float = 120.0


class EditOperation(BaseModel):
    """One planned edit and whether it applied (mirrors EditResult.summary)."""

    op: str
    pid: str
    reason: str = ""
    ok: bool = True
    detail: str = ""


class EditJobResult(BaseModel):
    """Poll result for GET /v1/files/edit/{job_id}. Field names match the
    plugin's fileEditHarness contract exactly (status/fileBase64/fileName/
    convertedFromLegacy/editsApplied/editsFailed/operations/error/progress)."""

    job_id: str
    status: str
    fileBase64: str | None = None
    fileName: str | None = None
    convertedFromLegacy: bool = False
    editsApplied: int = 0
    editsFailed: int = 0
    operations: list[EditOperation] = Field(default_factory=list)
    progress: EditProgress | None = None
    error: str | None = None
