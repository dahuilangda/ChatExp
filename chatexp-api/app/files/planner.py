"""LLM-based mission planner and step planner for the file analysis system.

Follows the planner+skills+harness pattern:
1. **plan_mission**: First, the big picture (mission) is established by analyzing
   the document structure, user question, and available content types.
2. **plan_next_step**: Then, progressively plan based on execution results. The
   planner does not plan everything upfront — it plans the next atomic step.

The planner is itself an LLM call with structured prompts. The harness
audits each step for direction consistency and handles failures.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from .skill_runners import ResponseTruncatedError, _call_llm, _llm_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Planner system prompts (pure English, no examples)
# ---------------------------------------------------------------------------

_PLANNER_MISSION_PROMPT = (
    "You are a document analysis mission planner. Your task is to analyze the "
    "document structure and user question, then produce a mission plan that "
    "defines the overall analysis strategy. The mission guides every later step.\n\n"
    "Consider the following factors when planning:\n"
    "- Document size: small documents (<300K chars) can be analyzed in one pass; "
    "larger documents need per-section map-reduce.\n"
    "- Section structure: documents with clear section headings are good candidates "
    "for map-reduce; unstructured text may need a single pass.\n"
    "- Table presence: tables may need special handling for numerical data.\n"
    "- Image presence: images are OCR'd to text before analysis.\n"
    "- Question complexity: simple factual questions need less analysis than "
    "comparative or analytical questions.\n"
    "- Multi-image questions: images arrive as OCR'd text sections at parse "
    "time, so comparing or contrasting multiple images means analyzing each "
    "image's OCR section, then synthesizing.\n"
    "- File size limit: the analysis limit is 300K characters by default. "
    "When the file exceeds this limit, use map_reduce strategy to analyze sections "
    "individually.\n\n"
    "Respond with a JSON object containing these fields:\n"
    "- intent: a concise one-sentence description of what the analysis should achieve\n"
    "- content_types: array listing the types of content available, "
    "one or more of [\"text\", \"table\", \"image\"]\n"
    "- strategy: one of \"single_pass\" (small doc, analyze all at once), "
    "\"map_reduce\" (large doc with sections, analyze sections separately then synthesize), "
    "\"vision_only\" (only images, no document text), "
    "\"mixed\" (document text plus images)\n"
    "- key_sections: array of section labels or headings most relevant to the question, "
    "or empty array if not applicable\n"
    "- complexity: one of \"low\", \"medium\", \"high\"\n"
    "- estimated_steps: estimated number of analysis steps needed (1-80)\n"
    "- scope: object describing what the analysis should cover, with fields:\n"
    "  - sections: number of sections to analyze (0 for all, N for specific count)\n"
    "  - images: number of images to analyze (0 if none)\n"
    "  - require_table_analysis: boolean, whether tables need special handling\n"
    "- completion_condition: one-sentence description of when the analysis is complete\n\n"
    "Return ONLY valid JSON. No markdown formatting, no code fences, no explanation."
)

def _skills_catalog_lines() -> str:
    """Generate skill list from SKILLS_CATALOG (single source of truth).

    Required parameters (per the skill's input_schema, cross-checked against
    the runner's own enforcement) are marked with a trailing '*' — the harness
    rejects a plan whose params omit a required parameter, so the planner must
    know which ones are mandatory."""
    from .skills import SKILLS_CATALOG
    lines = []
    for _name, sk in SKILLS_CATALOG.items():
        desc = sk.get("description") or sk.get("prompt") or ""
        schema = sk.get("input_schema") or {}
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        params = ", ".join(
            f"{name}*" if name in required else name
            for name in (props.keys() if isinstance(props, dict) else [])
        )
        # Full descriptions — they ARE the skill contract. A hard
        # truncation silently drops parameter contracts mid-sentence.
        lines.append(
            f"- {sk.get('name', _name)}: {desc}"
            + (f" Parameters: {params} (a * marks a required parameter)." if params else "")
        )
    return "\n".join(lines)

_PLANNER_SKILLS_BLOCK = "Available skills:\n" + _skills_catalog_lines() + "\n\n"

_PLANNER_STEP_PROMPT = (
    "You are a document analysis step planner. Based on the mission, the execution "
    "history (including any FAILED skills), and the current state, plan the next "
    "atomic step. Stay within the mission's intent and strategy.\n\n"
    "Failure handling (errors are observations):\n"
    "- History entries marked FAILED report real execution errors. Do not repeat "
    "the same skill call that just failed with the same parameters; route around "
    "the failure (different section, different approach) or move toward answering "
    "with the evidence already collected.\n"
    "- If a step plan was rejected by harness validation, the rejection reason is "
    "in the history — fix that specific contract violation.\n\n"
    + _PLANNER_SKILLS_BLOCK +
    "Strategy guidelines:\n"
    "- Phase progression: read -> analyze -> synthesize -> answer\n"
    "- For single_pass strategy: read all text at once, then analyze\n"
    "- For map_reduce strategy: read sections one by one, analyze each, synthesize all\n"
    "- For vision_only strategy: the document contains OCR text from images. "
    "Read the manifest, analyze the OCR text, synthesize, answer.\n"
    "- For mixed strategy: analyze text, analyze OCR text from images, synthesize, answer\n"
    "- Do not repeat skills that have already been executed successfully\n"
    "- Each step must produce a meaningful increment toward the final answer\n"
    "- If the mission's scope limits sections/images, respect those limits\n"
    "- When all evidence is collected, move to synthesize phase, then answer phase\n\n"
    "Completion guidelines:\n"
    "- The mission defines the completion_condition. Only enter the answer phase when "
    "all required evidence has been collected.\n"
    "- If the mission scope is fully covered, transition to answer phase even if some "
    "non-critical sections remain unread.\n\n"
    "Respond with a JSON object containing these fields:\n"
    "- phase: one of \"read\", \"analyze\", \"synthesize\", \"answer\"\n"
    "- reason: concise explanation of why this step is needed, referencing the mission\n"
    "- skills: array of skill objects — at least one skill per step — each with:\n"
    "  - skill: skill name (must be one of the available skills listed above)\n"
    "  - params: object with skill-specific parameters\n"
    "  - reason: why this specific skill invocation is needed, referencing the mission\n"
    "- confidence: number between 0.0 and 1.0 indicating confidence in this plan\n\n"
    "Return ONLY valid JSON. No markdown formatting, no code fences, no explanation."
)

# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _parse_json_response(response_text: str) -> dict[str, Any]:
    """Parse a JSON response, handling markdown-wrapped JSON (```json ... ```)."""
    text = response_text.strip()
    # Try to extract JSON from markdown code fences first
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    # Try parsing as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise


_TRUNCATION_REPAIR_INSTRUCTION = (
    "The previous response hit the output token limit and was cut off. "
    "Re-emit the COMPLETE JSON, more compactly — shorten fields, keep every "
    "required key/action."
)


def _truncation_repair_content(base_user_content: str, truncated_text: str) -> str:
    """Repair context for a finish_reason=length planner response (pi parity):
    the model re-enters once with instructions to re-emit the complete JSON
    compactly. The tail of the truncated response is included so the model
    sees exactly where it was cut."""
    tail = (truncated_text or "")[-400:]
    return (
        f"{base_user_content}\n\n"
        f"{_TRUNCATION_REPAIR_INSTRUCTION}\n"
        f"The truncated response ended with:\n{tail}"
    )


# ---------------------------------------------------------------------------
# Mission planning
# ---------------------------------------------------------------------------


def _mission_history_block(conversation_history: dict[str, Any] | None) -> str:
    """Render the cross-turn memory block for the mission prompt.

    Empty string on the first turn — the prompt is unchanged, no fallback
    semantics, no history to inject.
    """
    if not conversation_history:
        return ""
    turns = conversation_history.get("prior_turns", [])
    if not turns:
        return ""
    qa_lines = "\n".join(
        f"- Turn {t.get('turn', '?')}: Q: {t.get('task', '')}\n  A: {t.get('answer', '')}"
        for t in turns
    )
    sections = ", ".join(conversation_history.get("sections_analyzed", [])) or "none"
    return (
        "\n\nConversation so far on this document (the current question is a "
        "FOLLOW-UP unless it changes the goal — plan from what earlier turns "
        "already established):\n"
        f"{qa_lines}\n"
        f"Sections already analyzed across all turns: {sections}\n"
    )


def plan_mission(
    task: str,
    manifest_summary: dict[str, Any],
    image_count: int,
    llm_settings: dict[str, str] | None = None,
    report_progress: Callable | None = None,
    conversation_history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan the analysis mission: the big picture strategy.

    Args:
        task: The user's question or analysis task.
        manifest_summary: Document manifest summary with sections, tables, etc.
        image_count: Number of inline images provided.
        llm_settings: LLM endpoint/key/model settings.
        report_progress: Optional progress callback.
        conversation_history: Cross-turn memory (conversation_memory.mission_context)
            — prior Q/A turns for this document; None on the first turn.

    Returns:
        Mission dict with intent, strategy, content_types, etc.

    Raises:
        AnalyzeError: If the planner fails to produce a valid mission.
    """
    from .skill_runners import AnalyzeError

    settings = llm_settings or _llm_settings()

    if report_progress:
        report_progress("planning", "Planning analysis strategy...", 10.0)

    system_prompt = _PLANNER_MISSION_PROMPT + _mission_history_block(conversation_history)

    user_content = (
        f"Document structure:\n"
        f"- File name: {manifest_summary.get('fileName', 'unknown')}\n"
        f"- Total characters: {manifest_summary.get('totalChars', 0):,}\n"
        f"- Total pages: {manifest_summary.get('totalPages', 0)}\n"
        f"- Section count: {manifest_summary.get('sectionCount', 0)}\n"
        f"- Table count: {manifest_summary.get('tables', 0)}\n"
        f"- Estimated tokens: {manifest_summary.get('estTokens', 0):,}\n"
        f"\n"
        f"Sections:\n"
    )
    sections = manifest_summary.get("sections", [])
    for s in sections[:30]:
        heading = s.get("heading") or s.get("label", "untitled")
        chars = s.get("charCount", 0)
        user_content += f"  - [{s.get('sectionId', '?')}] {heading} ({chars:,} chars)\n"
    if len(sections) > 30:
        user_content += f"  ... and {len(sections) - 30} more sections\n"

    user_content += (
        f"\nImages provided: {image_count}\n"
        f"\nUser question: {task}\n"
    )

    try:
        try:
            response = _call_llm(
                system_prompt,
                user_content,
                settings,
                max_tokens=1024,
                timeout=240.0,
                retries=3,
                json_mode=True,
            )
        except ResponseTruncatedError as exc:
            # Planner-level truncation repair (pi parity): a finish_reason=
            # length cut here would kill the WHOLE analysis (skill-level
            # truncations already route through the error-observation loop).
            # One repair re-entry with a compact-instruction context; a second
            # truncation is a hard failure.
            logger.warning(
                "Mission planner response truncated (finish_reason=length); "
                "re-entering once with compact instructions."
            )
            if report_progress:
                report_progress(
                    "repairing",
                    "Mission plan was cut off by the output token limit — "
                    "requesting a compact re-emit...",
                    10.0,
                )
            try:
                response = _call_llm(
                    system_prompt,
                    _truncation_repair_content(user_content, exc.partial_text),
                    settings,
                    max_tokens=1024,
                    timeout=240.0,
                    retries=3,
                    json_mode=True,
                )
            except ResponseTruncatedError as exc2:
                raise AnalyzeError(
                    "mission planner response was truncated twice "
                    "(finish_reason=length) even after the compact re-emit"
                ) from exc2
        try:
            mission = _parse_json_response(response)
        except json.JSONDecodeError as json_exc:
            # One repair: the parse error is the observation.
            logger.warning("Mission planner returned invalid JSON, retrying once: %s", json_exc)
            retry_content = (
                f"{user_content}\n"
                f"YOUR PREVIOUS RESPONSE WAS NOT VALID JSON: {json_exc}\n"
                f"Return ONLY one valid JSON object matching the required fields. "
                f"No markdown fences, no commentary."
            )
            response = _call_llm(
                system_prompt,
                retry_content,
                settings,
                max_tokens=1024,
                timeout=240.0,
                retries=3,
                json_mode=True,
            )
            mission = _parse_json_response(response)
    except Exception as exc:
        raise AnalyzeError(f"Mission planning failed: {exc}") from exc

    # Validate required fields. content_types is derivable from the manifest
    # and image count, so an omitted value is normalized (not fabricated):
    # the strategy and complexity must come from the model.
    if "content_types" not in mission:
        derived = ["text"]
        if manifest_summary.get("tables", 0):
            derived.append("table")
        if image_count:
            derived.append("image")
        mission["content_types"] = derived
    required = ["intent", "strategy", "content_types", "complexity"]
    for field in required:
        if field not in mission:
            raise AnalyzeError(
                f"Mission planner response missing required field '{field}'. "
                f"Got: {list(mission.keys())}"
            )

    valid_strategies = {"single_pass", "map_reduce", "vision_only", "mixed"}
    if mission.get("strategy") not in valid_strategies:
        raise AnalyzeError(
            f"Mission planner returned unknown strategy '{mission.get('strategy')}'. "
            f"Valid strategies: {sorted(valid_strategies)}"
        )

    mission["key_sections"] = mission.get("key_sections", [])
    if "estimated_steps" in mission:
        # A non-numeric value (e.g. "about 7") must fail like any other
        # malformed mission field — the harness multiplies it into the
        # iteration cap, so a silent default would corrupt the loop bound.
        try:
            mission["estimated_steps"] = int(mission["estimated_steps"])
        except (TypeError, ValueError) as exc:
            raise AnalyzeError(
                f"Mission planner returned non-numeric estimated_steps "
                f"{mission['estimated_steps']!r} (expected an integer 1-80)."
            ) from exc
    else:
        mission["estimated_steps"] = 5
    mission["image_count"] = image_count
    # New mission contract fields (optional but tracked)
    mission["scope"] = mission.get("scope", {
        "sections": manifest_summary.get("sectionCount", len(manifest_summary.get("sections", []))),
        "images": image_count,
        "require_table_analysis": bool(manifest_summary.get("tables", 0)),
    })
    mission["completion_condition"] = mission.get(
        "completion_condition",
        "All relevant content has been analyzed and a final answer has been generated.",
    )

    logger.info(
        "Mission: intent=%r strategy=%s complexity=%s sections=%d steps=%d",
        mission["intent"][:80],
        mission["strategy"],
        mission.get("complexity"),
        len(mission.get("key_sections", [])),
        mission["estimated_steps"],
    )
    return mission


# ---------------------------------------------------------------------------
# Step planning
# ---------------------------------------------------------------------------


def plan_next_step(
    mission: dict[str, Any],
    state: dict[str, Any],
    execution_history: list[dict[str, Any]],
    llm_settings: dict[str, str] | None = None,
    report_progress: Callable | None = None,
    resume_note: str | None = None,
) -> dict[str, Any]:
    """Plan the next atomic step based on current state.

    Args:
        mission: The mission plan from plan_mission().
        state: Current execution state including results so far.
        execution_history: List of all executed skill results.
        llm_settings: LLM endpoint/key/model settings.
        report_progress: Optional progress callback.
        resume_note: One-line note the harness adds to the FIRST planning call
            of a resumed analysis (history rebuilt from a checkpoint).

    Returns:
        Step plan dict with phase, reason, skills array.

    Raises:
        AnalyzeError: If the planner fails to produce a valid step plan.
    """
    from .skill_runners import AnalyzeError

    settings = llm_settings or _llm_settings()

    # Build a compact state summary for the planner prompt
    results_so_far = state.get("results", [])
    completed_skills = [
        r.get("skill") for r in execution_history if r.get("status", "ok") == "ok" and r.get("skill")
    ]
    completed_skills_str = ", ".join(completed_skills) if completed_skills else "none"

    # Last 5 execution history entries. FAILURES are rendered as first-class
    # observations — the model must see WHAT failed and WHY to route
    # around it; successes are truncated previews.
    history_summary = ""
    for entry in execution_history[-5:]:
        skill = entry.get("skill", "?")
        if entry.get("status") == "error":
            error_text = str(entry.get("error", ""))[:300]
            params_text = json.dumps(entry.get("params", {}), ensure_ascii=False)[:200]
            history_summary += f"  - {skill} FAILED: {error_text} (params: {params_text})\n"
            continue
        result = entry.get("result", {})
        if isinstance(result, dict):
            # Truncate long results
            text = str(result.get("analysis") or result.get("synthesis") or result.get("answer") or result.get("text", ""))
            if len(text) > 100:
                text = text[:100] + "..."
        else:
            text = str(result)[:100]
        history_summary += f"  - {skill}: {text}\n"

    if report_progress:
        report_progress("planning", "Planning next analysis step...", 30.0)

    user_content = (
        f"Mission:\n"
        f"- Intent: {mission.get('intent', 'unknown')}\n"
        f"- Strategy: {mission.get('strategy', 'unknown')}\n"
        f"- Content types: {mission.get('content_types', [])}\n"
        f"- Key sections: {mission.get('key_sections', [])}\n"
        f"- Complexity: {mission.get('complexity', 'unknown')}\n"
        f"- Estimated steps: {mission.get('estimated_steps', 5)}\n"
        f"- Scope: sections={mission.get('scope', {}).get('sections', 'all')}, "
        f"images={mission.get('scope', {}).get('images', 0)}\n"
        f"- Completion condition: {mission.get('completion_condition', 'unknown')}\n"
        f"\n"
        f"Current state:\n"
        f"- Results collected: {len(results_so_far)}\n"
        f"- Completed skills: {completed_skills_str}\n"
        f"- Phase: {state.get('phase', 'not_started')}\n"
        f"\n"
        f"Execution history (last 5; FAILED entries are errors you must route around):\n"
        f"{history_summary or '  (no history yet)'}\n"
        f"\n"
        f"User question: {state.get('task', '')}\n"
    )

    if resume_note:
        # Resumed analysis: the history above was replayed from a checkpoint,
        # not produced by this process — the planner must know so it plans
        # forward from the recorded results instead of re-deriving them.
        user_content += (
            f"\nNOTE: {resume_note} — results listed above were already "
            f"computed and recorded; do not re-execute skills for them.\n"
        )

    # One P2 repair for invalid JSON: the parse error is the observation the
    # model needs (a blind re-roll cannot fix anything). Second failure raises.
    try:
        response = _call_llm(
            _PLANNER_STEP_PROMPT,
            user_content,
            settings,
            max_tokens=2048,
            timeout=240.0,
            retries=3,
            json_mode=True,
        )
    except ResponseTruncatedError as exc:
        # Planner-level truncation repair (pi parity): mirror of the mission
        # planner — one compact re-emit, second truncation raises.
        logger.warning(
            "Step planner response truncated (finish_reason=length); "
            "re-entering once with compact instructions."
        )
        if report_progress:
            report_progress(
                "repairing",
                "Step plan was cut off by the output token limit — "
                "requesting a compact re-emit...",
                30.0,
            )
        try:
            response = _call_llm(
                _PLANNER_STEP_PROMPT,
                _truncation_repair_content(user_content, exc.partial_text),
                settings,
                max_tokens=2048,
                timeout=240.0,
                retries=3,
                json_mode=True,
            )
        except ResponseTruncatedError as exc2:
            raise AnalyzeError(
                "step planner response was truncated twice "
                "(finish_reason=length) even after the compact re-emit"
            ) from exc2
    try:
        step_plan = _parse_json_response(response)
    except json.JSONDecodeError as exc:
        logger.warning("Step planner returned invalid JSON, retrying once with the parse error as observation: %s", exc)
        if report_progress:
            report_progress("repairing", "Step plan was not valid JSON — asking the planner to fix it...", 30.0)
        retry_content = (
            f"{user_content}\n"
            f"YOUR PREVIOUS RESPONSE WAS NOT VALID JSON: {exc}\n"
            f"Return ONLY one valid JSON object matching the required fields. "
            f"No markdown fences, no commentary."
        )
        response = _call_llm(
            _PLANNER_STEP_PROMPT,
            retry_content,
            settings,
            max_tokens=2048,
            timeout=240.0,
            retries=3,
            json_mode=True,
        )
        try:
            step_plan = _parse_json_response(response)
        except json.JSONDecodeError as exc2:
            raise AnalyzeError(f"Step planning failed: planner returned invalid JSON twice: {exc2}") from exc2

    return step_plan


def list_available_skills() -> list[str]:
    """Return available skill names for planner validation."""
    from .skills import list_available_skills as _list

    return _list()