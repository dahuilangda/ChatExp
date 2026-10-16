"""File evidence: the LLM autonomously studies user-specified files.

Ports the seatable plugin's evidence pipeline to the terminal app. The flow:

  1. The planner emits ``evidence_requests`` (files it needs to see) instead of
     guessing their contents.
  2. This module resolves the requests against the user's filesystem, parses
     each file into text chunks, and returns a selection
     ({chunks, failures, read_plan}).
  3. The coordinator re-plans ONCE with the chunks attached to the planner
     context; the planner then acts on real content (import rows, answer
     questions) and must acknowledge failures instead of inventing contents.

Parsing order per file: the chatexp-api document service (``POST /v1/files/parse``
→ poll → fetch, docling-based, shared with the seatable plugin) when the BO API
is configured; otherwise local Python readers (openpyxl / python-docx / pypdf /
plain text). Both paths produce the SAME chunk shape
``{name, path, chunk_index, total_chunks, label, heading, text}`` plus
structured failures with stable codes — so the planner side never cares which
parser ran. Local xlsx reading additionally emits a sheet-summary section
(sheets, dimensions, headers, first rows), which even the backend lacks.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd

from .bo_client import normalize_bo_api_endpoint
from .settings import BOApiSettings

logger = logging.getLogger(__name__)

# Evidence budget (plugin-parity magnitudes, slightly widened for local files).
MAX_FILES = 3
MAX_CHUNKS = 6
CHUNK_SIZE_CHARS = 8192
MAX_FILE_BYTES = 50 * 1024 * 1024
PARSE_TIMEOUT_SECONDS = 600
POLL_INITIAL_SECONDS = 1.2
POLL_MAX_SECONDS = 15.0

# Structured failure codes (stable, plugin-aligned).
CODE_FILE_NOT_FOUND = "file_not_found"
CODE_TOO_LARGE = "file_too_large"
CODE_UNSUPPORTED = "unsupported_type"
CODE_REMOTE_PARSE_FAILED = "remote_parse_failed"
CODE_REMOTE_PARSE_UNAVAILABLE = "remote_parse_unavailable"
CODE_LOCAL_PARSE_FAILED = "local_parse_failed"
CODE_MISSING_DEPENDENCY = "missing_dependency"
CODE_BUDGET_EXHAUSTED = "budget_exhausted"

TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".log"}


@dataclass
class FileReference:
    """One user-specified file location."""

    name: str
    path: str
    reason: str = ""

    @property
    def suffix(self) -> str:
        return Path(self.path).suffix.lower()


@dataclass
class EvidenceChunk:
    name: str
    path: str
    chunk_index: int
    total_chunks: int
    label: str
    heading: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "label": self.label,
            "heading": self.heading,
            "text": self.text,
        }


@dataclass
class EvidenceFailure:
    name: str
    path: str
    code: str
    error: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path, "code": self.code, "error": self.error[:300]}


@dataclass
class EvidenceSelection:
    chunks: list[EvidenceChunk] = field(default_factory=list)
    failures: list[EvidenceFailure] = field(default_factory=list)
    read_plan: dict[str, Any] = field(default_factory=dict)

    @property
    def has_failures(self) -> bool:
        return bool(self.failures)

    def context_block(self) -> dict[str, Any]:
        """The evidence payload merged into the planner context (2nd pass)."""
        return {
            "file_content_chunks": [chunk.to_dict() for chunk in self.chunks],
            "file_read_failures": [f.to_dict() for f in self.failures],
            "file_read_plan": self.read_plan,
        }


# ═════════════════════════════════════════════════════════════════
# Request resolution
# ═════════════════════════════════════════════════════════════════


def resolve_evidence_requests(
    requests: list[dict[str, Any]],
    base_dir: str | Path | None = None,
) -> tuple[list[FileReference], list[FileReference]]:
    """Resolve planner requests into concrete file references.

    Accepted selectors per request: ``path`` (absolute or relative to the
    workspace), ``name`` (matched against the user-mentioned files), or both.
    Unknown names become references that will fail with file_not_found — the
    failure is reported honestly instead of silently dropped. Returns
    ``(references, overflow)``; ``overflow`` lists requests beyond MAX_FILES
    so the read plan can report them as budget_exhausted.
    """
    references: list[FileReference] = []
    overflow: list[FileReference] = []
    seen: set[str] = set()
    for request in requests or []:
        if not isinstance(request, dict):
            continue
        raw_path = str(request.get("path") or "").strip()
        raw_name = str(request.get("name") or request.get("file") or "").strip()
        reason = str(request.get("reason") or "").strip()
        candidate_path = raw_path or raw_name
        if not candidate_path:
            continue
        path = Path(candidate_path).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if len(references) >= MAX_FILES:
            # Beyond-budget requests are recorded as structured failures, not
            # silently dropped — the read plan reports them.
            overflow.append(FileReference(name=raw_name or path.name, path=str(path), reason=reason))
            continue
        references.append(FileReference(name=raw_name or path.name, path=str(path), reason=reason))
    return references, overflow


# ═════════════════════════════════════════════════════════════════
# Gathering: remote (chatexp-api document service) → local readers
# ═════════════════════════════════════════════════════════════════


def gather_evidence(
    references: list[FileReference],
    settings: BOApiSettings | None = None,
    max_chunks: int = MAX_CHUNKS,
    overflow: list[FileReference] | None = None,
) -> EvidenceSelection:
    """Parse each referenced file into chunks. Remote document service first
    (shared with the seatable plugin), local Python readers otherwise. Both
    paths produce the same chunk shape; failures are structured, never silent
    and never fabricated content."""
    selection = EvidenceSelection()
    remaining_chunks = max_chunks
    parsed_via: dict[str, str] = {}
    for reference in overflow or []:
        selection.failures.append(
            EvidenceFailure(reference.name, reference.path, CODE_BUDGET_EXHAUSTED, "beyond the per-request file budget")
        )
    for reference in references:
        if remaining_chunks <= 0:
            selection.failures.append(
                EvidenceFailure(reference.name, reference.path, CODE_BUDGET_EXHAUSTED, "evidence chunk budget exhausted")
            )
            continue
        file_bytes, load_failure = _load_bytes(reference)
        if load_failure is not None:
            selection.failures.append(load_failure)
            continue
        assert file_bytes is not None
        chunks, failure = _parse_one(reference, file_bytes, settings, remaining_chunks)
        if failure is not None:
            selection.failures.append(failure)
            continue
        chunks = chunks[:remaining_chunks]
        selection.chunks.extend(chunks)
        remaining_chunks -= len(chunks)
        parsed_via[reference.path] = "remote" if settings is not None and settings.configured else "local"
    selection.read_plan = {
        "unit_operation": "read_file_evidence",
        "policy": {"max_files": MAX_FILES, "max_chunks": MAX_CHUNKS, "chunk_size_chars": CHUNK_SIZE_CHARS},
        "files": [
            {"name": r.name, "path": r.path, "parser": parsed_via.get(r.path, "failed"), "reason": r.reason}
            for r in references
        ],
        "included_chunks": len(selection.chunks),
        "failed": len(selection.failures),
    }
    return selection


def _load_bytes(reference: FileReference) -> tuple[bytes | None, EvidenceFailure | None]:
    path = Path(reference.path)
    if not path.exists():
        return None, EvidenceFailure(reference.name, reference.path, CODE_FILE_NOT_FOUND, "file does not exist")
    if path.is_dir():
        return None, EvidenceFailure(reference.name, reference.path, CODE_UNSUPPORTED, "path is a directory")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        return None, EvidenceFailure(reference.name, reference.path, CODE_TOO_LARGE, f"{size} bytes exceeds the 50MiB limit")
    try:
        return path.read_bytes(), None
    except OSError as exc:
        return None, EvidenceFailure(reference.name, reference.path, CODE_LOCAL_PARSE_FAILED, str(exc))


def _parse_one(
    reference: FileReference,
    file_bytes: bytes,
    settings: BOApiSettings | None,
    max_chunks: int,
) -> tuple[list[EvidenceChunk], EvidenceFailure | None]:
    if settings is not None and settings.configured:
        chunks, failure = _parse_via_remote(reference, file_bytes, settings, max_chunks)
        if failure is None or failure.code not in (CODE_REMOTE_PARSE_UNAVAILABLE,):
            return chunks, failure
        # The document service is configured but unreachable — that outcome is
        # reported as a failure (never silently swapped for a local parser).
        return [], failure
    return _parse_locally(reference, file_bytes, max_chunks)


# ── remote: chatexp-api document service (plugin-aligned contract) ──────────


def _doc_id(payload_bytes: bytes) -> str:
    return hashlib.sha256(payload_bytes).hexdigest()[:32]


def _parse_via_remote(
    reference: FileReference,
    file_bytes: bytes,
    settings: BOApiSettings,
    max_chunks: int,
) -> tuple[list[EvidenceChunk], EvidenceFailure | None]:
    import requests

    base = normalize_bo_api_endpoint(settings.endpoint or "").rstrip("/") + "/"
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    payload = {
        "contentBase64": base64.b64encode(file_bytes).decode("ascii"),
        "contentType": _guess_mime(reference),
        "name": reference.name,
        "docOptions": {"ocr": True, "table_structure": True},
    }
    try:
        response = requests.post(urljoin(base, "files/parse"), headers=headers, json=payload, timeout=60)
        if response.status_code >= 400:
            return [], EvidenceFailure(reference.name, reference.path, CODE_REMOTE_PARSE_FAILED, f"HTTP {response.status_code}: {response.text[:200]}")
        job_id = response.json().get("job_id")
        if not job_id:
            return [], EvidenceFailure(reference.name, reference.path, CODE_REMOTE_PARSE_FAILED, "no job_id in parse response")
        deadline = time.monotonic() + PARSE_TIMEOUT_SECONDS
        delay = POLL_INITIAL_SECONDS
        while True:
            status_response = requests.get(urljoin(base, f"jobs/{job_id}"), headers=headers, timeout=60)
            body = status_response.json() if status_response.status_code < 500 else {}
            status = str(body.get("status") or "")
            if status == "succeeded":
                break
            if status in {"failed", "canceled"}:
                return [], EvidenceFailure(reference.name, reference.path, CODE_REMOTE_PARSE_FAILED, str(body.get("error") or status)[:300])
            if time.monotonic() >= deadline:
                return [], EvidenceFailure(reference.name, reference.path, CODE_REMOTE_PARSE_FAILED, "parse job timed out")
            time.sleep(delay)
            delay = min(delay * 1.5, POLL_MAX_SECONDS)
        fetch = requests.post(
            urljoin(base, f"files/jobs/{job_id}/fetch"),
            headers=headers,
            json={"maxChars": max_chunks * CHUNK_SIZE_CHARS},
            timeout=120,
        )
        if fetch.status_code >= 400:
            return [], EvidenceFailure(reference.name, reference.path, CODE_REMOTE_PARSE_FAILED, f"fetch HTTP {fetch.status_code}")
        body = fetch.json()
        sections = []
        for chunk in body.get("chunks") or []:
            sections.append(
                EvidenceChunk(
                    name=reference.name,
                    path=reference.path,
                    chunk_index=int(chunk.get("chunk_index") or chunk.get("chunkIndex") or 0),
                    total_chunks=int(chunk.get("total_chunks") or chunk.get("totalChunks") or 1),
                    label=str(chunk.get("label") or ""),
                    heading=str(chunk.get("heading") or ""),
                    text=str(chunk.get("text") or ""),
                )
            )
        return sections[:max_chunks], None
    except requests.RequestException as exc:
        # configured-but-unreachable: structured failure, NOT a local fallback
        return [], EvidenceFailure(reference.name, reference.path, CODE_REMOTE_PARSE_UNAVAILABLE, f"{type(exc).__name__}: {exc}")


def _guess_mime(reference: FileReference) -> str:
    return {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".csv": "text/csv",
        ".md": "text/markdown",
        ".txt": "text/plain",
    }.get(reference.suffix, "application/octet-stream")


# ── local readers (real parsers, same chunk shape) ──────────────────────────


def _split_chunks(text: str, reference: FileReference, label: str) -> list[EvidenceChunk]:
    text = text.strip()
    if not text:
        return []
    total = max(1, (len(text) + CHUNK_SIZE_CHARS - 1) // CHUNK_SIZE_CHARS)
    return [
        EvidenceChunk(
            name=reference.name,
            path=reference.path,
            chunk_index=index,
            total_chunks=total,
            label=label,
            heading=label if index == 0 else "",
            text=text[index * CHUNK_SIZE_CHARS : (index + 1) * CHUNK_SIZE_CHARS],
        )
        for index in range(total)
    ]


def _parse_locally(
    reference: FileReference,
    file_bytes: bytes,
    max_chunks: int,
) -> tuple[list[EvidenceChunk], EvidenceFailure | None]:
    try:
        if reference.suffix == ".xlsx":
            return _read_xlsx(reference, max_chunks), None
        if reference.suffix == ".docx":
            return _read_docx(reference, max_chunks), None
        if reference.suffix == ".pdf":
            return _read_pdf(reference, file_bytes, max_chunks)
        if reference.suffix in TEXT_SUFFIXES:
            text = _decode_text(file_bytes)
            if reference.suffix == ".csv":
                text = _csv_to_markdown(text)
            return _split_chunks(text, reference, "text")[:max_chunks], None
        return [], EvidenceFailure(reference.name, reference.path, CODE_UNSUPPORTED, f"unsupported type {reference.suffix or 'unknown'}")
    except ImportError as exc:
        return [], EvidenceFailure(reference.name, reference.path, CODE_MISSING_DEPENDENCY, str(exc))
    except Exception as exc:
        return [], EvidenceFailure(reference.name, reference.path, CODE_LOCAL_PARSE_FAILED, f"{type(exc).__name__}: {exc}")


def _decode_text(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "gb18030", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except Exception:
        # no tabulate dependency: render a plain pipe table
        header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
        separator = "|" + "|".join("---" for _ in frame.columns) + "|"
        rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
        return "\n".join([header, separator, *rows])


def _csv_to_markdown(text: str) -> str:
    try:
        frame = pd.read_csv(pd.io.common.StringIO(text[:200_000]))
        return _frame_to_markdown(frame)
    except Exception:
        return text


def _read_xlsx(reference: FileReference, max_chunks: int) -> list[EvidenceChunk]:
    """xlsx: per-sheet markdown PLUS a synthetic sheet-summary section — the
    one thing neither the plugin nor the backend document service provides."""
    frames = pd.read_excel(reference.path, sheet_name=None, header=None)
    chunks: list[EvidenceChunk] = []
    summaries = []
    for sheet_name, frame in frames.items():
        summaries.append(f"- {sheet_name}: {frame.shape[0]} rows x {frame.shape[1]} cols")
    header_lines = ["# Sheets", *summaries]
    chunks.extend(_split_chunks("\n".join(header_lines), reference, "sheet_summary"))

    for sheet_name in frames:
        sniffed = pd.read_excel(reference.path, sheet_name=sheet_name)
        columns = ", ".join(str(c) for c in sniffed.columns)
        preview = _frame_to_markdown(sniffed.head(12))
        body = f"# Sheet {sheet_name}\n\nColumns: {columns}\n\n{preview}"
        chunks.extend(_split_chunks(body, reference, f"sheet:{sheet_name}"))
    return chunks[:max_chunks]


def _read_docx(reference: FileReference, max_chunks: int) -> list[EvidenceChunk]:
    """docx: paragraphs walked in order; Heading 1-3 become section headings;
    tables render as markdown pipe tables."""
    import docx  # python-docx

    document = docx.Document(reference.path)
    sections: list[tuple[str, list[str]]] = [("preamble", [])]
    for element in document.element.body:
        tag = element.tag.split("}")[-1]
        if tag == "tbl":
            from docx.table import Table

            table = Table(element, document)
            rows = []
            for row in table.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                rows.append("| " + " | ".join(cells) + " |")
            if rows:
                sections[-1][1].extend(rows)
                sections[-1][1].append("")
        elif tag == "p":
            from docx.text.paragraph import Paragraph

            paragraph = Paragraph(element, document)
            style = (paragraph.style.name or "").lower()
            text = paragraph.text.strip()
            if not text:
                continue
            if style.startswith(("heading 1", "heading 2", "heading 3", "标题 1", "标题 2", "标题 3")):
                sections.append((text, []))
            else:
                sections[-1][1].append(text)
    chunks: list[EvidenceChunk] = []
    for index, (heading, lines) in enumerate(sections):
        body = (f"# {heading}\n\n" if heading != "preamble" else "") + "\n".join(lines)
        if body.strip():
            chunks.extend(_split_chunks(body, reference, f"section:{heading}"))
    return chunks[:max_chunks]


def _read_pdf(
    reference: FileReference,
    file_bytes: bytes,
    max_chunks: int,
) -> tuple[list[EvidenceChunk], EvidenceFailure | None]:
    try:
        import io

        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore[no-redef]
        except ImportError:
            return [], EvidenceFailure(
                reference.name,
                reference.path,
                CODE_MISSING_DEPENDENCY,
                "pdf parsing requires the 'pypdf' package (pip install pypdf)",
            )
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(f"# Page {index}\n\n{text}")
        return _split_chunks("\n\n".join(pages), reference, "pages")[:max_chunks], None
    except Exception as exc:
        return [], EvidenceFailure(reference.name, reference.path, CODE_LOCAL_PARSE_FAILED, f"{type(exc).__name__}: {exc}")


# ═════════════════════════════════════════════════════════════════
# Honesty check: no invented content for failed files
# ═════════════════════════════════════════════════════════════════

_ADMITS_FAILURE = re.compile(
    r"(could not|cannot|couldn't|unavailable|failed( to (parse|read|extract))?|not available"
    r"|parse error|binary|unreadable|无法读取|无法解析|解析失败|读取失败|不可用|不存在|打不开|未能|提取失败)",
    re.IGNORECASE,
)


def claimed_content_despite_failures(message: str, selection: EvidenceSelection) -> bool:
    """Plugin-parity 'pretended to read' check: when files failed to parse and
    the planner's reply describes THOSE files' results WITHOUT admitting the
    failure, the reply is treated as invented content. Correlation is by
    failed file name/path stem, so unrelated failures in the same batch do
    not force a rejection of an honestly-sourced answer."""
    if not selection.has_failures:
        return False
    message_text = message or ""
    names = {failure.name for failure in selection.failures}
    stems = {Path(failure.path).stem for failure in selection.failures if failure.path}
    mentioned = any(name and name in message_text for name in names) or any(
        stem and stem in message_text for stem in stems
    )
    if not mentioned:
        # The reply never touches the failed files — nothing to police.
        return False
    return not bool(_ADMITS_FAILURE.search(message_text))


__all__ = [
    "EvidenceChunk",
    "EvidenceSelection",
    "FileReference",
    "build_evidence_context",
    "claimed_content_despite_failures",
    "gather_evidence",
    "resolve_evidence_requests",
]


def build_evidence_context(requests: list[dict[str, Any]], settings: BOApiSettings | None, base_dir: str | Path | None = None) -> tuple[EvidenceSelection, dict[str, Any]]:
    """One-call helper: resolve + gather + build the planner context block."""
    references, overflow = resolve_evidence_requests(requests, base_dir=base_dir)
    selection = gather_evidence(references, settings=settings, overflow=overflow)
    return selection, selection.context_block()
