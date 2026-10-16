"""Atomic skill definitions for the planner+skills+harness file analysis system.

Each skill is a pure-English-prompted atomic unit operation. No examples.
The full catalog is embedded in the step-planner prompt; the harness audits
results against each skill's declared output_required contract.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Skill catalog
# ---------------------------------------------------------------------------

# Maximum chars for a single-pass analysis. Owned here so the harness, the
# skill runners, and the planner prompt share ONE constant (no drift).
SINGLE_PASS_MAX_CHARS = 300_000

SKILLS_CATALOG: dict[str, dict[str, Any]] = {
    "read_manifest": {
        "name": "read_manifest",
        "description": "Read the document structure manifest including sections, tables, and page count. Data-only, no LLM call.",
        "prompt": "",
        "input_schema": {"type": "object", "properties": {}, "description": "No parameters needed. The manifest is already available in the analysis state."},
        "output_required": ["manifest"],
        "data_only": True,
        "timeout": 5.0,
    },
    "read_section": {
        "name": "read_section",
        "description": "Read a specific section's text content from the parsed document. Data-only, no LLM call. Pass the sectionId or the section heading.",
        "prompt": "",
        "input_schema": {"type": "object", "properties": {"section_id": {"type": "string", "description": "The section identifier to read"}, "max_chars": {"type": "integer", "description": "Maximum characters to return"}}, "required": ["section_id"]},
        "output_required": ["text"],
        "data_only": True,
        "timeout": 5.0,
    },
    "analyze_text": {
        "name": "analyze_text",
        "description": "Analyze the actual text content of a document section with the LLM. The harness reads the section from the parsed document — pass the section_id (from the manifest or read_manifest), NEVER paste, summarize, or invent the content yourself.",
        "prompt": (
            "You are a text analysis assistant. Examine the provided text content "
            "and respond to the user's question accurately and comprehensively. "
            "Base your answer ONLY on the text provided. If the text does not contain "
            "enough information to answer, say so explicitly. "
            "Preserve key facts, numbers, names, dates, and conclusions. "
            "Do not invent information not present in the text."
        ),
        "input_schema": {"type": "object", "properties": {"section_id": {"type": "string", "description": "The section to analyze (sectionId from the manifest, or its heading)"}, "question": {"type": "string", "description": "The user's question or analysis task (optional; defaults to the overall analysis task)"}, "label": {"type": "string", "description": "A label for this text segment (e.g., section heading)"}}, "required": ["section_id"]},
        "output_required": ["analysis"],
        "data_only": False,
        "timeout": 240.0,
    },
    "synthesize": {
        "name": "synthesize",
        "description": "Combine the analyses collected so far into a coherent synthesis. The harness gathers every analyze_text result from the execution state — do not restate or paste analysis content yourself.",
        "prompt": (
            "You are a synthesis assistant. Combine the following individual analyses "
            "into a coherent synthesis that answers the user's question. "
            "Preserve key facts, numbers, names, dates, and conclusions from each analysis. "
            "Resolve any contradictions between analyses by noting them explicitly. "
            "Do not invent information not present in the provided analyses."
        ),
        "input_schema": {"type": "object", "properties": {"question": {"type": "string", "description": "The user's question (optional; defaults to the overall analysis task)"}}, "required": []},
        "output_required": ["synthesis"],
        "data_only": False,
        "timeout": 240.0,
    },
    "answer": {
        "name": "answer",
        "description": "Generate the final answer from all collected evidence. This is the last skill executed in the analysis loop. The all_evidence parameter is optional; when omitted, the runner builds it from the execution state.",
        "prompt": (
            "You are a document analysis assistant. Based on all the evidence "
            "collected from the document and images, answer the user's question "
            "comprehensively and accurately. Synthesize information from all sources "
            "into a clear, well-structured answer. Reference specific sections, "
            "tables, or images when relevant. If the evidence is insufficient to "
            "answer fully, state what is missing."
        ),
        "input_schema": {"type": "object", "properties": {"all_evidence": {"type": "string", "description": "All collected evidence from previous analysis steps (optional; built from state when omitted)"}, "question": {"type": "string", "description": "The user's original question (optional; defaults to the overall analysis task)"}}, "required": []},
        "output_required": ["answer"],
        "data_only": False,
        "timeout": 240.0,
    },
    "check_file_size": {
        "name": "check_file_size",
        "description": "Check if the parsed document or image content is within the size limits for LLM processing. Data-only, no LLM call. Returns the file size, the limit, and whether the file is within bounds.",
        "prompt": "",
        "input_schema": {"type": "object", "properties": {"max_chars": {"type": "integer", "description": "Maximum allowed characters for analysis (optional override)"}}, "description": "Optional: pass max_chars to check against a custom character limit. When omitted, the limit comes from the analysis state (default 300,000 characters); the size itself is always read from the document manifest."},
        "output_required": ["total_chars", "max_chars", "within_limit", "message"],
        "data_only": True,
        "timeout": 5.0,
    },
    "report_size_limit": {
        "name": "report_size_limit",
        "description": "Report that the file exceeds size limits and provide a clear, actionable message to the user. This is a data-only skill that generates a structured response about the size constraint.",
        "prompt": "",
        "input_schema": {"type": "object", "properties": {"total_chars": {"type": "integer", "description": "Actual file size in characters"}, "max_chars": {"type": "integer", "description": "Maximum allowed characters"}, "total_mb": {"type": "number", "description": "Actual file size in MB"}, "max_mb": {"type": "number", "description": "Maximum allowed size in MB"}}, "required": ["total_chars", "max_chars", "total_mb", "max_mb"]},
        "output_required": ["message"],
        "data_only": True,
        "timeout": 5.0,
    },
}

# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_skill(skill_name: str) -> dict[str, Any]:
    """Look up a skill definition by name. Raises KeyError if not found."""
    if skill_name not in SKILLS_CATALOG:
        raise KeyError(f"Unknown skill: {skill_name}")
    return SKILLS_CATALOG[skill_name]


def list_available_skills() -> list[str]:
    """Return the list of all available skill names."""
    return list(SKILLS_CATALOG.keys())
