"""Harness for the planner+skills+harness file analysis system.

The harness is the coordinator (pi-agent pattern):
1. Read the document manifest, establish the mission via the planner
2. One flat loop: plan → validate plan shape → execute skills → audit results
3. Failures become OBSERVATIONS for the planner (errors-as-observations) —
   the harness never silently retries a model-level failure and never
   fabricates content
4. Deterministic gates only: plan-shape validation, parameter contracts,
   repetition guard, iteration cap, answer extraction
5. Transport-level retries (network/5xx) live inside _call_llm, not here

Direction is model judgment, enforced through the mission in the planner
prompt; there is no mechanical mission-lock blocking.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable

from . import artifacts, planner, registry, skills, skill_runners
from .skill_runners import AnalyzeError

logger = logging.getLogger(__name__)

# Maximum iterations of the plan→execute→audit loop. This prevents infinite
# loops when the planner repeatedly plans the same step. For large documents
# (map_reduce) the planner may need 50+ iterations. The actual limit is
# adjusted dynamically: min(MAX_ITERATIONS_HARD, estimated_steps * 2).
MAX_ITERATIONS_HARD = 80
MAX_ITERATIONS_DEFAULT = 30

# A skill (name + params) may fail at most this many times before the harness
# concludes the planner cannot route around the failure and raises. The first
# failure is an observation; an identical second failure is a dead end.
MAX_IDENTICAL_FAILURES = 2

# Maximum chars for a single-pass analysis. skills.py owns the constant so the
# skill runners and the planner prompt can share it without circular imports.
SINGLE_PASS_MAX_CHARS = skills.SINGLE_PASS_MAX_CHARS

_VALID_PHASES = {"read", "analyze", "synthesize", "answer"}


# ---------------------------------------------------------------------------
# Audit result
# ---------------------------------------------------------------------------


class AuditResult:
    """Result of an audit check on a skill execution or step plan."""

    def __init__(self, ok: bool = True, reason: str = ""):
        self.ok = ok
        self.reason = reason

    @classmethod
    def success(cls) -> "AuditResult":
        return cls(ok=True)

    @classmethod
    def failure(cls, reason: str) -> "AuditResult":
        return cls(ok=False, reason=reason)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def analyze(
    task: str,
    doc_id: str,
    document: dict[str, Any] | None,
    images: list[dict[str, str]] | None = None,
    llm_settings: dict[str, str] | None = None,
    report_progress: Callable | None = None,
) -> str:
    """Run the planner+skills+harness analysis loop.

    Args:
        task: The user's question or analysis task.
        doc_id: The document ID (empty if no document).
        document: The parsed document dict (None if no document).
        images: List of inline image data (empty if no images).
        llm_settings: LLM endpoint/key/model settings.
        report_progress: Optional progress callback.

    Returns:
        The final analysis answer string.

    Raises:
        AnalyzeError: On any unrecoverable failure.
    """
    settings = llm_settings if llm_settings is not None else skill_runners._llm_settings()
    images = images or []
    has_doc = bool(doc_id and document)
    has_images = bool(images)
    task_text = task or "Summarize this document."

    if not has_doc and not has_images:
        raise AnalyzeError("Either a document or images must be provided for analysis.")

    from . import conversation_memory

    # --- Checkpoint identity: same doc + task + model resumes the same run ---
    checkpoint = artifacts.checkpoint_path(
        artifacts.analysis_id(doc_id, task_text, settings.get("model") or "")
    )
    # A checkpoint from a previous crashed run may already carry the
    # {"e":"recorded"} marker; honoring it is what keeps record_turn
    # idempotent across restarts.
    recorded_already = any(
        isinstance(event, dict) and event.get("e") == "recorded"
        for event in artifacts.read_checkpoint(checkpoint)
    )
    # Analysis answer memo (server occupancy): an identical first-turn
    # question over the same document with the same model re-runs the full
    # LLM loop for a byte-identical answer. Served ONLY when this worker has
    # no conversation history for the document and no inline images —
    # conversational follow-ups must never receive a pre-context answer.
    memo_key: str | None = None
    if doc_id and not images:
        from . import conversation_memory as _memo_mod
        try:
            if not _memo_mod.get_history(doc_id):
                memo_key = artifacts.analysis_id(doc_id, task_text, settings.get("model") or "")
        except Exception:  # noqa: BLE001 - memory unavailable => no memo
            memo_key = None
    if memo_key:
        memo_hit = registry.get_analysis_memo(memo_key)
        if memo_hit:
            logger.info("Analysis memo hit for %s", doc_id)
            return memo_hit

    try:
        # --- Phase 1: Build manifest summary ---
        manifest_summary = {}
        if has_doc:
            if report_progress:
                report_progress("reading", "Reading document structure...", 10.0)
            manifest_summary = skill_runners._manifest_summary(document)

        # --- Phase 2: Check file size (deterministic preflight before planning) ---
        size_check_result = None
        if has_doc:
            if report_progress:
                report_progress("reading", "Checking file size limits...", 5.0)
            try:
                size_check_result = skill_runners.run(
                    "check_file_size",
                    {"max_chars": SINGLE_PASS_MAX_CHARS},
                    {"document": document},
                )
                logger.info(
                    "Size check: %s chars (limit %s) — %s",
                    size_check_result.get("total_chars", "?"),
                    size_check_result.get("max_chars", "?"),
                    "within limit" if size_check_result.get("within_limit") else "EXCEEDS LIMIT",
                )
            except Exception as exc:
                logger.error("Size check failed: %s", exc)
                raise AnalyzeError(
                    f"File size verification failed before analysis could start: {exc}. "
                    f"The file could not be checked against size limits. "
                    f"Ensure the file is a valid document and try again."
                ) from exc

        # --- Phase 3: Plan the mission (with cross-turn memory when present) ---
        conversation_history = conversation_memory.mission_context(doc_id, task_text) if doc_id else None
        mission = planner.plan_mission(
            task=task_text,
            manifest_summary=manifest_summary,
            image_count=len(images),
            llm_settings=settings,
            report_progress=report_progress,
            conversation_history=conversation_history,
        )

        if size_check_result:
            mission["size_check"] = size_check_result

        # --- Phase 4: Execute the plan loop (checkpointed, resumable) ---
        state: dict[str, Any] = {
            "task": task_text,
            "doc_id": doc_id,
            "document": document,
            "results": [],
            "phase": "not_started",
            "mission": mission,
            "max_chars": SINGLE_PASS_MAX_CHARS,
        }

        answer = _execute_plan_loop(mission, state, settings, report_progress, checkpoint=checkpoint)
    except AnalyzeError:
        # Terminal analysis failure: the checkpoint must not outlive the run.
        _delete_checkpoint(checkpoint)
        raise

    # --- Phase 5: Record the completed turn for future follow-ups ---
    if doc_id and not recorded_already:
        # The {"e":"recorded"} marker is written BEFORE recording so a crash
        # between the two can skip a turn but never double-record one.
        _checkpoint_append(checkpoint, {"e": "recorded"})
        sections_analyzed = [
            result["section_id"]
            for result in state.get("results", [])
            if isinstance(result, dict) and result.get("section_id")
        ]
        try:
            conversation_memory.record_turn(
                doc_id, task_text, answer, sections_analyzed,
                turn=len(conversation_memory.get_history(doc_id)) + 1,
            )
        except conversation_memory.MemoryError_:
            # Contentless turns are not memory-worthy; the analysis result
            # itself was already returned to the user.
            logger.warning("Could not record conversation turn for %s", doc_id)

    if memo_key and answer:
        registry.set_analysis_memo(memo_key, answer)
        # The memo must outlive the guard's ability to refuse it: once a turn
        # is RECORDED, the first-turn-only guard can no longer be trusted
        # (in-process history is capped at 64 docs and dies on restart, while
        # the memo lives 24h in Redis) — drop the memo so a conversational
        # repeat re-runs the analysis instead of serving a stale first answer.
        try:
            registry.delete_analysis_memo(memo_key)
        except Exception:  # noqa: BLE001
            pass
    _delete_checkpoint(checkpoint)
    return answer


# ---------------------------------------------------------------------------
# Plan loop — one flat loop, errors as observations
# ---------------------------------------------------------------------------


def _validate_step_plan(step_plan: Any) -> AuditResult:
    """Deterministic plan-shape validation (contract, not model judgment).

    Checks the structural contract the step prompt declares: phase is one of
    the valid phases, skills is a non-empty array, every skill exists, and
    required params per the skill's input_schema are present and non-empty.
    """
    if not isinstance(step_plan, dict):
        return AuditResult.failure(f"Step plan is not a JSON object (got {type(step_plan).__name__}).")

    phase = step_plan.get("phase", "")
    if phase not in _VALID_PHASES:
        return AuditResult.failure(
            f"Invalid phase '{phase}'. Valid phases: {', '.join(sorted(_VALID_PHASES))}."
        )

    skills_list = step_plan.get("skills")
    if not isinstance(skills_list, list) or not skills_list:
        return AuditResult.failure(
            f"Step plan (phase '{phase}') has no skills. The contract requires at "
            f"least one skill per step; for the answer phase include the 'answer' skill."
        )

    for skill_spec in skills_list:
        if not isinstance(skill_spec, dict):
            return AuditResult.failure(f"Skill spec is not an object: {skill_spec!r}")
        skill_name = skill_spec.get("skill", "")
        if not skill_name:
            return AuditResult.failure(f"Skill spec has no 'skill' field: {skill_spec!r}")
        try:
            skill_def = skills.get_skill(skill_name)
        except KeyError:
            return AuditResult.failure(
                f"Unknown skill '{skill_name}'. Available: {planner.list_available_skills()}"
            )
        params = skill_spec.get("params", {})
        if not isinstance(params, dict):
            return AuditResult.failure(f"Skill '{skill_name}' params is not an object.")
        for param_name in skill_def.get("input_schema", {}).get("required", []):
            value = params.get(param_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                return AuditResult.failure(
                    f"Skill '{skill_name}' missing required parameter '{param_name}'. "
                    f"Got params: {list(params.keys())}"
                )

    return AuditResult.success()


def _execute_plan_loop(
    mission: dict[str, Any],
    state: dict[str, Any],
    llm_settings: dict[str, str],
    report_progress: Callable | None = None,
    checkpoint: str | None = None,
) -> str:
    """Execute the plan→execute→audit loop until the answer phase completes.

    Each iteration:
    1. Planner produces the next step plan (sees SUCCESS and FAILURE history)
    2. Harness validates the plan shape (deterministic contract gate)
       — an invalid plan becomes an observation; the planner fixes it next
         iteration. No auto-injection, no silent repair.
    3. Harness executes each skill ONCE and audits the result
       — success: appended to history; failure: appended as an error
         observation (the planner re-plans around it)
    4. Answer phase with a successful answer skill: return the answer

    Checkpointing (pi session persistence): with ``checkpoint`` set, one JSONL
    event is appended per step ({"e":"plan"/"skill_ok"/"skill_err"/...}). On
    entry, a checkpoint without a done marker is replayed to rebuild
    execution_history / state results — every skill in the loop is a read or
    an LLM call (no external writes), so resume is safe and recorded skills
    are NOT re-executed. A done marker means a previous run finished but
    crashed before cleanup: the recorded answer is returned directly.

    Raises AnalyzeError when: the planner itself fails, an identical skill
    failure repeats MAX_IDENTICAL_FAILURES times, or the iteration cap is hit.
    """
    execution_history: list[dict[str, Any]] = []
    failure_counts: dict[str, int] = {}
    start_time = time.monotonic()
    resumed = False

    if checkpoint:
        events = [
            event for event in artifacts.read_checkpoint(checkpoint)
            if isinstance(event, dict)
        ]
        if any(event.get("e") == "done" for event in events):
            # A previous run appended the terminal done marker but crashed
            # before cleanup — the analysis is complete; do not re-run skills.
            for event in reversed(events):
                if event.get("e") == "answer" and event.get("text"):
                    logger.info("Returning completed analysis from checkpoint %s", checkpoint)
                    return str(event["text"])
            raise AnalyzeError(
                "Analysis checkpoint has a done marker but no answer text; "
                "the checkpoint is corrupt and the analysis must be re-run."
            )
        if events:
            execution_history, failure_counts = _replay_checkpoint_events(events, state)
            resumed = True
            logger.info(
                "Resuming analysis from checkpoint %s: %d history entries replayed",
                checkpoint,
                len(execution_history),
            )

    estimated_steps = mission.get("estimated_steps", 10)
    max_iterations = min(MAX_ITERATIONS_HARD, max(MAX_ITERATIONS_DEFAULT, estimated_steps * 2))

    for iteration in range(1, max_iterations + 1):
        logger.info(
            "Harness iteration %d/%d, phase=%s",
            iteration, max_iterations, state.get("phase", "?"),
        )

        # --- Step 1: Plan next step (planner sees full history incl. failures) ---
        try:
            step_plan = planner.plan_next_step(
                mission=mission,
                state=state,
                execution_history=execution_history,
                llm_settings=llm_settings,
                report_progress=report_progress,
                resume_note=(
                    "resumed after restart; history is replayed"
                    if resumed and iteration == 1
                    else None
                ),
            )
        except AnalyzeError as exc:
            results_count = len(state.get("results", []))
            if results_count > 0:
                raise AnalyzeError(
                    f"Analysis stopped at iteration {iteration} after {results_count} "
                    f"successful steps. The next step could not be planned: {exc}. "
                    f"Partial results ({results_count} steps) are available but cannot "
                    f"be synthesized into a final answer because the planner failed."
                ) from exc
            raise
        _checkpoint_append(checkpoint, {"e": "plan", "iter": iteration, "plan": step_plan})
        resumed = False  # the resume note applies to the first planner call only

        # --- Step 2: Deterministic plan-shape gate → observation on failure ---
        validation = _validate_step_plan(step_plan)
        if not validation.ok:
            logger.warning("Plan rejected at iteration %d: %s", iteration, validation.reason)
            execution_history.append({
                "skill": "(plan)",
                "status": "error",
                "error": f"Step plan rejected by harness validation: {validation.reason}",
            })
            if report_progress:
                report_progress("repairing", "Re-planning after invalid step plan...", 30.0)
            continue

        # --- Step 3: Execute each skill once; failures become observations ---
        answer_result = None
        for skill_spec in step_plan["skills"]:
            skill_name = skill_spec["skill"]
            params = skill_spec.get("params", {})
            logger.info("Executing skill: %s (iteration %d)", skill_name, iteration)
            state["_step_phase"] = step_plan.get("phase", "analyze")

            try:
                result = skill_runners.run(
                    skill_name=skill_name,
                    params=params,
                    state=state,
                    llm_settings=llm_settings,
                    report_progress=report_progress,
                )
                audit = _audit_skill_result(skill_name, result, params)
                if not audit.ok:
                    raise AnalyzeError(audit.reason)
            except AnalyzeError as exc:
                # The failure is the observation. No blind retry — the
                # planner sees this entry and re-plans around the failure.
                failure_key = json.dumps(
                    {"skill": skill_name, "params": _stable_params(params)},
                    sort_keys=True, ensure_ascii=False, default=str,
                )
                failure_counts[failure_key] = failure_counts.get(failure_key, 0) + 1
                logger.warning(
                    "Skill %s failed (occurrence %d): %s",
                    skill_name, failure_counts[failure_key], exc,
                )
                execution_history.append({
                    "skill": skill_name,
                    "status": "error",
                    "error": str(exc),
                    "params": _stable_params(params),
                })
                _checkpoint_append(checkpoint, {
                    "e": "skill_err",
                    "iter": iteration,
                    "skill": skill_name,
                    "error": str(exc),
                    "params": _stable_params(params),
                })
                if failure_counts[failure_key] >= MAX_IDENTICAL_FAILURES:
                    raise AnalyzeError(
                        f"Skill '{skill_name}' failed {failure_counts[failure_key]} times "
                        f"with the same parameters and the planner did not route around it. "
                        f"Last error: {exc}"
                    )
                if report_progress:
                    report_progress("repairing", f"{skill_name} failed — re-planning...", 40.0)
                continue

            state.setdefault("results", []).append(result)
            execution_history.append({"skill": skill_name, "status": "ok", "result": result})
            _checkpoint_append(checkpoint, {
                "e": "skill_ok",
                "iter": iteration,
                "skill": skill_name,
                "result": result,
            })
            if skill_name == "answer":
                answer_result = result
            state["phase"] = step_plan.get("phase", "analyze")

        # --- Step 4: Answer phase complete? ---
        if step_plan.get("phase") == "answer":
            answer = _extract_answer(answer_result)
            if answer:
                elapsed = time.monotonic() - start_time
                logger.info(
                    "Analysis complete in %d iterations, %.1f seconds",
                    iteration,
                    elapsed,
                )
                _checkpoint_append(checkpoint, {"e": "answer", "text": answer})
                _checkpoint_append(checkpoint, {"e": "done"})
                return answer
            # answer phase but the answer skill failed/missing — already an
            # observation in history; loop so the planner reacts.
            logger.warning("Answer phase reached without a usable answer result.")

    # --- Loop exhausted ---
    results_count = len(state.get("results", []))
    raise AnalyzeError(
        f"Analysis did not complete within {max_iterations} iterations. "
        f"{results_count} steps were completed successfully, but the planner "
        f"did not reach the answer phase. "
        f"The document may be too large or too complex for the current "
        f"analysis strategy. Try a more specific question or split the "
        f"document into smaller sections."
    )


def _replay_checkpoint_events(
    events: list[dict[str, Any]],
    state: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Rebuild execution_history, state["results"], and failure_counts from
    checkpoint events. Pure state reconstruction — no skill is executed, no
    LLM is called; the loop continues from the last recorded step. The
    failure-count keys are computed exactly like the live path so the
    MAX_IDENTICAL_FAILURES dead-end detection survives a restart."""
    execution_history: list[dict[str, Any]] = []
    failure_counts: dict[str, int] = {}
    for event in events:
        kind = event.get("e")
        if kind == "skill_ok":
            result = event.get("result")
            execution_history.append({
                "skill": str(event.get("skill") or ""),
                "status": "ok",
                "result": result,
            })
            if isinstance(result, dict):
                state.setdefault("results", []).append(result)
        elif kind == "skill_err":
            params = event.get("params") if isinstance(event.get("params"), dict) else {}
            execution_history.append({
                "skill": str(event.get("skill") or ""),
                "status": "error",
                "error": str(event.get("error") or ""),
                "params": params,
            })
            failure_key = json.dumps(
                {"skill": str(event.get("skill") or ""), "params": params},
                sort_keys=True, ensure_ascii=False, default=str,
            )
            failure_counts[failure_key] = failure_counts.get(failure_key, 0) + 1
    return execution_history, failure_counts


def _checkpoint_append(checkpoint: str | None, event: dict[str, Any]) -> None:
    """Append one checkpoint event; a None checkpoint (un-checkpointed run)
    is an explicit no-op."""
    if not checkpoint:
        return
    artifacts.append_checkpoint_line(checkpoint, event)


def _delete_checkpoint(checkpoint: str | None) -> None:
    """Remove a checkpoint file (cleanup on success and on terminal
    AnalyzeError). Missing files are fine; other removal failures are logged
    so an undeletable checkpoint is visible, never silently ignored."""
    if not checkpoint:
        return
    try:
        os.remove(checkpoint)
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("Could not remove analysis checkpoint %s", checkpoint, exc_info=True)


def _stable_params(params: dict[str, Any]) -> dict[str, Any]:
    """Params safe to echo back into the planner prompt (bounded)."""
    return {
        key: (value if len(str(value)) <= 300 else str(value)[:300] + "...")
        for key, value in (params or {}).items()
    }


# ---------------------------------------------------------------------------
# Skill result audit
# ---------------------------------------------------------------------------


def _audit_skill_result(
    skill_name: str,
    result: Any,
    params: dict[str, Any],
) -> AuditResult:
    """Audit a skill execution result against its declared output contract.

    Checks: result is a dict and every declared output_required field is
    present and non-empty. The contract lives in skills.py — this audit reads
    it dynamically, so new skills are covered automatically.
    """
    if result is None:
        return AuditResult.failure(f"Skill '{skill_name}' returned None.")

    if not isinstance(result, dict):
        return AuditResult.failure(
            f"Skill '{skill_name}' returned non-dict result: {type(result).__name__}"
        )

    try:
        skill_def = skills.get_skill(skill_name)
    except KeyError:
        return AuditResult.failure(f"Unknown skill '{skill_name}' — cannot audit.")

    required_fields = skill_def.get("output_required", [])
    for field in required_fields:
        if field not in result:
            return AuditResult.failure(
                f"Skill '{skill_name}' missing required field '{field}'."
            )
        value = result.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            return AuditResult.failure(
                f"Skill '{skill_name}' returned empty '{field}'."
            )

    return AuditResult.success()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_answer(result: Any) -> str | None:
    """Extract the answer string from the answer skill's result.

    Checks ``answer`` first, then ``synthesis``. Uses explicit ``is not None``
    checks so an empty string in ``answer`` is not silently replaced by
    ``synthesis`` — the audit layer already caught the empty content.
    """
    if not result:
        return None
    if isinstance(result, dict):
        answer = result.get("answer")
        if answer is not None:
            return str(answer) if answer else None
        synthesis = result.get("synthesis")
        if synthesis is not None:
            return str(synthesis) if synthesis else None
        return None
    if isinstance(result, str):
        return result if result.strip() else None
    return None
