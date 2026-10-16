"""LLM edit planner for track-changes document editing.

The LLM sees the paragraph-indexed document view and the user's instruction,
and produces a strict edit-plan JSON. The plan is executed deterministically
by docx_editor — the LLM never touches the file bytes.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from . import docx_editor
from .skill_runners import AnalyzeError, _call_llm, _llm_settings

logger = logging.getLogger(__name__)

_EDIT_PLANNER_SYSTEM = """You are an expert document editor that plans tracked-changes revisions to Word documents. You receive a paragraph-indexed view of a document (each line: p<index>: text) and the user's editing instruction.

Return ONLY a raw JSON object (no markdown, no prose):
{"edits": [{"op": "...", "pid": "...", "find": "...", "text": "...", "reason": "..."}]}

Operation contract:
- "replace": replace the exact text in "find" with "text" inside paragraph "pid". "find" MUST be an exact substring of that paragraph, copied character for character. Every occurrence within the paragraph is replaced. Prefer short, unique anchors (10-80 characters) centered on the text being changed.
- "delete_paragraph": mark the whole paragraph "pid" as deleted. "find" must be an exact substring of that paragraph confirming it is the right one.
- "insert_paragraph_after": insert a new tracked paragraph with "text" after paragraph "pid". "find" must be an exact substring of that paragraph.

Hard rules:
- Copy anchors EXACTLY from the document view — including punctuation, spacing, casing, and numerals. A paraphrased anchor fails and the edit is skipped with an explicit error.
- Never invent content that contradicts the document. Insertions must follow the user's instruction.
- Choose the minimal set of edits that satisfies the instruction. Do not restructure the document unless asked.
- If the instruction cannot be satisfied with these operations (e.g. reordering sections, editing tables), return edits for what IS possible and explain the rest in "reason" fields.
- Use each pid only as given in the view; do not renumber.
- "reason" is a short human-readable explanation of why the edit was made (it appears in the report to the user).
- Output ONLY the JSON object."""


def plan_edits(
    paragraph_view: str,
    instruction: str,
    llm_settings: dict[str, str] | None = None,
) -> list[docx_editor.EditOp]:
    """Ask the LLM to plan the edits. One call; a JSON failure gets one
    strict retry (mirrors planner.py's contract)."""
    settings = llm_settings or _llm_settings()
    user_content = (
        f"Document view (p<index>: text):\n\n{paragraph_view}\n\n---\n\n"
        f"User's editing instruction: {instruction}"
    )

    last_error: Exception | None = None
    for attempt in range(2):
        reminder = ""
        if attempt == 1:
            reminder = (
                "\n\nYour previous response was not a valid edit plan JSON. "
                "Return ONLY the raw JSON object: {\"edits\": [...]}"
            )
        raw = _call_llm(
            _EDIT_PLANNER_SYSTEM,
            user_content + reminder,
            settings,
            max_tokens=8192,
            timeout=240.0,
            retries=3,
            json_mode=True,
        )
        try:
            plan = _extract_json(raw)
            return docx_editor.parse_edit_plan(plan)
        except (docx_editor.EditError, ValueError) as exc:
            last_error = exc
            logger.warning("Edit plan parse failed (attempt %d): %s", attempt + 1, exc)
    raise AnalyzeError(f"The edit plan could not be parsed: {last_error}")


def _extract_json(raw: str) -> Any:
    """Tolerant JSON extraction: direct parse, then first balanced braces."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
    raise ValueError("No JSON object found in the response.")
