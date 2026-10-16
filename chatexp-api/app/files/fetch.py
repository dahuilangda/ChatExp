"""Slice a cached StoredDocument into the chunks/page-images a fetch request
asks for. This is the read-side of the manifest-addressable contract: the
plugin sends either a list of sectionIds (phase-2 agentic) or leaves it open to
get everything under a char budget."""
from __future__ import annotations

from typing import Any

from .models import FetchChunk, FetchResponse


def fetch(document: dict[str, Any], section_ids: list[str] | None, page_images: list[int] | None, max_chars: int | None) -> FetchResponse:
    blocks = document.get("blocks") or []
    if section_ids:
        wanted = set(section_ids)
        selected = [b for b in blocks if b.get("sectionId") in wanted]
    else:
        selected = list(blocks)
    chunks: list[FetchChunk] = []
    used_chars = 0
    truncated = False
    next_cursor = None
    for index, block in enumerate(selected):
        text = block.get("text") or ""
        remaining = max_chars - used_chars if max_chars else len(text)
        if remaining <= 0:
            truncated = True
            next_cursor = block.get("sectionId")
            break
        piece = text if len(text) <= remaining else text[:remaining]
        truncated = truncated or len(piece) < len(text)
        chunks.append(
            FetchChunk(
                sectionId=block["sectionId"],
                chunkIndex=int(block.get("chunkIndex", 0)),
                totalChunks=int(block.get("totalChunks", 1)),
                page=block.get("page"),
                label=block.get("label", "paragraph"),
                heading=block.get("heading"),
                text=piece,
                includedChars=len(piece),
                truncated=len(piece) < len(text),
            )
        )
        used_chars += len(piece)
        if max_chars and used_chars >= max_chars:
            # Budget exhausted with blocks remaining: the response is
            # truncated, so expose the next cursor instead of implying the
            # document ended.
            if index + 1 < len(selected):
                truncated = True
                next_cursor = selected[index + 1].get("sectionId")
            break
    # Page-image rendering is not implemented. Serving a placeholder image
    # would substitute fake content for real pages — refuse loudly instead.
    if page_images:
        raise PageImagesNotSupported("Page image rendering is not implemented. Fetch section text instead.")
    return FetchResponse(chunks=chunks, pageImages=[], truncated=truncated, nextCursor=next_cursor)


class PageImagesNotSupported(Exception):
    """Raised when a fetch request asks for rendered page images."""
