"""Build a section-aware manifest from a ParseResult.

Builds structure-preserving sections (one per heading or top-level block)
instead of fixed-size character slicing. Each section is independently
addressable so the plugin can fetch just the sections the model asks for,
and oversized sections are split into chunked text blocks with a cursor for
continuation.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .models import Manifest, SectionInfo, TableInfo
from .parser import ParseResult, TextItem

# Heading labels recognised by docling, mapped to a depth. Anything not in here
# is treated as body content attached to the most recent heading.
_HEADING_DEPTH = {
    "title": 1,
    "page_header": 1,
    "section_header": 2,
    "caption": 3,
    "formula": 3,
}

# Soft cap: sections larger than this get split into multiple text blocks.
SECTION_CHUNK_CHARS = 8000


@dataclass
class StoredDocument:
    doc_id: str
    file_name: str
    mime_type: str | None
    manifest: dict[str, Any]
    blocks: list[dict[str, Any]] = field(default_factory=list)  # {sectionId, chunkIndex, totalChunks, page, label, heading, text}
    tables: list[dict[str, Any]] = field(default_factory=list)


def build_doc_id(data: bytes, doc_options: dict[str, Any]) -> str:
    """Stable content hash. We hash bytes (not url/path) so the same file under
    different SeaTable rows reuses the cache, but we fold docOptions in so that
    flipping OCR changes the cache key."""
    h = hashlib.sha256()
    h.update(data)
    h.update(b"\x00")
    h.update(repr(sorted((doc_options or {}).items())).encode("utf-8"))
    return h.hexdigest()[:32]


def build(result: ParseResult, payload: dict[str, Any], data: bytes, mime: str, doc_options: dict[str, Any] | None = None) -> StoredDocument:
    doc_id = build_doc_id(data, doc_options or {})
    sections, blocks, section_tables = _sectionize(result)
    tables = _build_tables(result, section_tables)
    total_chars = sum(len(block["text"]) for block in blocks)
    manifest = Manifest(
        docId=doc_id,
        fileName=str(payload.get("name") or "document"),
        mimeType=mime or None,
        totalPages=result.total_pages,
        totalChars=total_chars,
        estTokens=_est_tokens(total_chars),
        sections=sections,
        tables=tables,
        lowConfidencePages=result.low_confidence_pages,
        parserEngine=result.parser_engine,
    )
    return StoredDocument(
        doc_id=doc_id,
        file_name=manifest.fileName,
        mime_type=manifest.mimeType,
        manifest=manifest.model_dump(),
        blocks=blocks,
        tables=[t.model_dump() for t in tables],
    )


def _sectionize(result: ParseResult) -> tuple[list[SectionInfo], list[dict[str, Any]], dict[str, str]]:
    """Walk texts in reading order, opening a new section at each heading.

    Returns (section_infos, text_blocks, table_section_lookup).
    - text_blocks: flat list of {sectionId, chunkIndex, totalChunks, page, label,
      heading, text} where chunkIndex indexes within a section.
    - table_section_lookup: maps docling table index -> the sectionId it sits in.
    """
    sections: list[SectionInfo] = []
    blocks: list[dict[str, Any]] = []
    table_section: dict[int, str] = {}

    current_id: str | None = None
    current_heading: str | None = None
    current_label = "paragraph"
    current_page: int | None = None
    pending_text: list[str] = []
    table_index = 0

    def flush() -> None:
        nonlocal current_id
        if current_id is None:
            return
        text = "\n".join(pending_text).strip()
        if not text:
            return
        chunks = _split_chunk(text, SECTION_CHUNK_CHARS)
        for index, chunk in enumerate(chunks):
            blocks.append(
                {
                    "sectionId": current_id,
                    "chunkIndex": index,
                    "totalChunks": len(chunks),
                    "page": current_page,
                    "label": current_label,
                    "heading": current_heading,
                    "text": chunk,
                }
            )

    # Insert tables at the position their page suggests; for simplicity, track
    # them separately and assign each to the section whose page matches.
    section_seq = 0
    for item in result.texts:
        depth = _heading_depth(item)
        if depth is not None and depth <= 3:
            # Start a new section.
            flush()
            pending_text.clear()
            section_seq += 1
            current_id = f"s{section_seq}"
            current_heading = item.text
            current_label = item.label
            current_page = item.page
            sections.append(
                SectionInfo(
                    sectionId=current_id,
                    label=item.label,
                    heading=current_heading,
                    page=item.page,
                    charCount=0,  # filled after flush
                    estTokens=0,
                )
            )
        else:
            if current_id is None:
                # Content before any heading: synthesize a preamble section.
                section_seq += 1
                current_id = f"s{section_seq}"
                current_heading = None
                current_label = "paragraph"
                current_page = item.page
                sections.append(
                    SectionInfo(
                        sectionId=current_id,
                        label="paragraph",
                        heading=None,
                        page=item.page,
                        charCount=0,
                        estTokens=0,
                    )
                )
            pending_text.append(item.text)
            # Attribute any table on this page to the current section.
            while table_index < len(result.tables) and _matches_page(result.tables[table_index], item.page):
                table_section[table_index] = current_id
                table_index += 1

    # Drain remaining tables to the last section.
    while table_index < len(result.tables):
        table_section[table_index] = current_id or "s1"
        table_index += 1

    flush()
    # Backfill per-section char counts from the blocks we emitted.
    char_by_section: dict[str, int] = {}
    for block in blocks:
        char_by_section[block["sectionId"]] = char_by_section.get(block["sectionId"], 0) + len(block["text"])
    for section in sections:
        chars = char_by_section.get(section.sectionId, 0)
        section.charCount = chars
        section.estTokens = _est_tokens(chars)
        section.truncated = chars > SECTION_CHUNK_CHARS
    # Drop sections that ended up empty (heading with no body).
    sections = [s for s in sections if s.charCount > 0]
    return sections, blocks, table_section


def _build_tables(result: ParseResult, table_section: dict[int, str]) -> list[TableInfo]:
    tables: list[TableInfo] = []
    for index, raw in enumerate(result.tables):
        tables.append(
            TableInfo(
                tableId=f"t{index + 1}",
                page=raw.page,
                rows=raw.n_rows,
                cols=raw.n_cols,
                sectionId=table_section.get(index),
            )
        )
    return tables


def _heading_depth(item: TextItem) -> int | None:
    if item.heading_level is not None:
        return item.heading_level
    return _HEADING_DEPTH.get(item.label)


def _matches_page(table, page: int | None) -> bool:
    return page is not None and table.page is not None and table.page == page


def _split_chunk(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        # Try to break on a newline within the last 20% of the window.
        if end < len(text):
            boundary = text.rfind("\n", start + int(size * 0.8), end)
            if boundary > start:
                end = boundary + 1
        chunks.append(text[start:end])
        start = end
    return chunks


def _est_tokens(chars: int) -> int:
    # Local heuristic: ~4 characters per token (roughly one English word).
    # Simple and deterministic: the estimate only sizes prompts
    # and budgets, it never gates correctness.
    return max(1, chars // 4)
