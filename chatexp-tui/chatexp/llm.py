from __future__ import annotations

import ast
import json
import random
import re
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

from .agent_harness import (
    BudgetExceededError,
    LLMTruncatedError,
    PlannerCallBudget,
    QuotaExhaustionError,
    ServerDelayFailFastError,
    extract_json as salvage_extract_json,
    is_context_overflow_error,
    is_quota_exhaustion_text,
)
from .compaction import build_history_window
from .event_log import append_event, summarize_usage
from .models import ChatPending, IntentPlan, ProjectState
from .provenance import write_llm_provenance
from .settings import load_openai_settings
from .skills import build_skill_context


class HarnessCompletion:
    """One completed chat-completion call: text plus transport evidence."""

    __slots__ = ("text", "finish_reason", "usage")

    def __init__(self, text: str | None, finish_reason: str | None, usage: dict[str, Any] | None):
        self.text = text
        self.finish_reason = finish_reason
        self.usage = usage


class PlannerContractError(RuntimeError):
    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors or []


class LLMAuthenticationError(RuntimeError):
    pass


PROMPT_ASSET_DIR = Path(__file__).resolve().with_name("prompt_assets")
RECENT_HISTORY_LIMIT = 4
HISTORY_DIGEST_LIMIT = 6
HISTORY_SNIPPET_LIMIT = 72
# When the previous planner call's input exceeded this many tokens, the next
# call's verbatim history window shrinks (token-aware compaction trigger).
TOKEN_COMPACT_INPUT_THRESHOLD = 24000
PROMPT_BUNDLES: dict[str, list[tuple[str, str]]] = {
    "planner": [
        ("角色", "planner_role.txt"),
        ("场景范围", "planner_scope.txt"),
        ("技能路由", "planner_skills.txt"),
        ("工作流", "planner_workflow.txt"),
        ("动作目录", "planner_actions.txt"),
        ("输出契约", "planner_output_contract.txt"),
        ("决策规则", "planner_rules.txt"),
    ],
    "repair": [
        ("修复器", "repair_prompt.txt"),
    ],
    "import_mapper": [
        ("角色", "import_mapper_role.txt"),
        ("工作流", "import_mapper_workflow.txt"),
        ("输出契约", "import_mapper_contract.txt"),
        ("字段语义", "import_mapper_examples.txt"),
    ],
    "import_interpreter": [
        ("角色", "import_interpreter_role.txt"),
        ("工作流", "import_interpreter_workflow.txt"),
        ("输出契约", "import_interpreter_contract.txt"),
        ("字段语义", "import_interpreter_examples.txt"),
    ],
    "analysis_narrator": [
        ("角色", "analysis_narrator_role.txt"),
        ("输出契约", "analysis_narrator_contract.txt"),
    ],
    "strategist": [
        ("角色", "strategist.txt"),
    ],
}
@lru_cache(maxsize=None)
def _load_prompt_asset(name: str) -> str:
    return (PROMPT_ASSET_DIR / name).read_text(encoding="utf-8").strip()


def _bundle_system_messages(bundle_name: str) -> list[dict[str, str]]:
    sections: list[str] = []
    for title, asset_name in PROMPT_BUNDLES[bundle_name]:
        sections.append(f"[{title}]\n{_load_prompt_asset(asset_name)}")
    return [{"role": "system", "content": "\n\n".join(sections)}]


def _build_plan_messages(user_content: str, language: str = "zh") -> list[dict[str, str]]:
    language_guidance = {
        "zh": "",
        "en": "\n\n[LANGUAGE INSTRUCTION: The user's input is in English. ALL your output (including workbook_name, objective names, factor names, and messages) MUST be in English.]"
    }
    messages = _bundle_system_messages("planner")
    if language == "en":
        messages[0]["content"] += (
            "\n\n[LANGUAGE]\nCRITICAL: The user's input is in English. ALL your output must be in English, including:\n"
            "- workbook_name: use English names\n"
            "- objective names: use English names\n"
            "- factor names: use English names\n"
            "- When action='respond': respond in English only\n"
            "Even if other context is in Chinese, output everything in English."
        )
    messages.append({"role": "user", "content": (language_guidance.get(language, "") + user_content)})
    return messages


def _detect_language(text: str) -> str:
    """Detect if text is primarily English or Chinese."""
    if not text:
        return "zh"
    # Count Chinese characters (Unicode range for common Chinese characters)
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    english_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
    # Simple heuristic: if more than 30% Chinese characters, treat as Chinese
    total_chars = chinese_chars + english_chars
    if total_chars == 0:
        return "zh"
    return "zh" if chinese_chars / total_chars > 0.3 else "en"


def _build_repair_messages(user_content: str) -> list[dict[str, str]]:
    return _bundle_system_messages("repair") + [{"role": "user", "content": user_content}]


def _build_import_mapper_messages(user_content: str) -> list[dict[str, str]]:
    return _bundle_system_messages("import_mapper") + [{"role": "user", "content": user_content}]


def _build_import_interpreter_messages(user_content: str) -> list[dict[str, str]]:
    return _bundle_system_messages("import_interpreter") + [{"role": "user", "content": user_content}]


def _build_analysis_narrator_messages(user_content: str) -> list[dict[str, str]]:
    return _bundle_system_messages("analysis_narrator") + [{"role": "user", "content": user_content}]


class OpenAIIntentPlanner:
    def __init__(self, project_root: str | Path | None = None):
        self.settings = load_openai_settings(project_root)
        self._client = _ChatCompletionClient(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            timeout_seconds=self.settings.timeout_seconds,
            max_tokens=self.settings.max_tokens,
        ) if self.settings.configured else None
        # Harness state (chained compaction + usage accounting + provenance).
        self._history_summary: str | None = None
        self._skip_next_token_compaction = False
        self._last_input_tokens = 0
        self._request_usage = {"input": 0, "output": 0, "total": 0, "calls": 0}
        self.last_request_usage: dict[str, int] | None = None
        self._provenance_written: set[str] = set()

    @property
    def available(self) -> bool:
        return self._client is not None and self.settings.configured

    def plan(
        self,
        message: str,
        project: ProjectState | None,
        pending: ChatPending | None = None,
        history: list[dict[str, str]] | None = None,
        ui_state: dict[str, Any] | None = None,
        trace_dir: str | Path | None = None,
        task_id: str | None = None,
        language: str = "zh",
        extra_context: dict[str, Any] | None = None,
    ) -> IntentPlan | None:
        """Plan one user turn into an IntentPlan. Retries up to 3 attempts
        with one degraded-context retry on overflow and one JSON repair
        round; raises PlannerContractError when the contract cannot be met."""
        if not self.available:
            return None
        budget = PlannerCallBudget()
        event_dir = self._event_dir_for(trace_dir)
        self._ensure_provenance(event_dir)
        self._reset_request_usage()
        history_payload = self._history_payload(history)
        normalized_message = _normalize_message(message, pending, history_payload)
        context_mode = "full"
        errors: list[str] = []
        truncation_retried = False
        truncation_tail = ""
        json_repair_done = False
        overflow_retried = False
        attempt = 0
        max_attempts = 3

        while True:
            attempt_payload = self._build_plan_payload(
                message=message,
                normalized_message=normalized_message,
                project=project,
                pending=pending,
                history=history_payload,
                ui_state=ui_state,
                language=language,
                context_mode=context_mode,
                extra_context=extra_context,
            )
            if errors:
                attempt_payload["previous_validation_errors"] = errors[-2:]
                if truncation_tail:
                    attempt_payload["retry_instruction"] = (
                        "Your previous plan JSON was cut off mid-output (finish_reason=length). "
                        "Re-emit the COMPLETE plan, more compactly: shorten message/reason fields, "
                        "do not drop actions.\nTruncated tail:\n" + truncation_tail
                    )
                else:
                    attempt_payload["retry_instruction"] = (
                        "Retry by re-reading the original user message and selected skills. "
                        "Do not repair by rules; use the LLM planner itself to emit a valid action JSON. "
                        "If the selected skill is conversation, return action=respond with a concise natural message. "
                        "For pasted initial data with no project, append_results must include rows, factors, and objectives."
                    )
            messages = _build_plan_messages(json.dumps(attempt_payload, ensure_ascii=False), language=language)
            attempt += 1
            budget.register(f"planner_attempt_{attempt}")
            content, request_error, finish_reason, usage = self._request_stage(
                stage=f"planner_attempt_{attempt}",
                messages=messages,
                trace_dir=trace_dir,
                task_id=task_id,
                request_payload=attempt_payload,
            )
            self._account_usage(usage, event_dir)

            if request_error is not None:
                self._write_trace(
                    trace_dir=trace_dir,
                    stage=f"planner_attempt_{attempt}",
                    task_id=task_id,
                    request_payload=attempt_payload,
                    raw_response=content,
                    parsed_plan=None,
                    error=request_error,
                )
                if is_context_overflow_error(request_error) and not overflow_retried:
                    # Exactly ONE degraded-context retry; second overflow is loud.
                    overflow_retried = True
                    context_mode = "overflow_minimal"
                    self._skip_next_token_compaction = True
                    append_event(
                        event_dir,
                        "observation",
                        fact="LLM context overflow detected; retrying once with overflow_minimal context",
                    )
                    continue
                errors.append(request_error)
                if attempt >= max_attempts:
                    raise PlannerContractError(
                        f"planner failed after {attempt} attempts: {errors[-1]}", errors
                    )
                continue

            if finish_reason == "length":
                truncation_tail = (content or "")[-400:]
                self._write_trace(
                    trace_dir=trace_dir,
                    stage=f"planner_attempt_{attempt}",
                    task_id=task_id,
                    request_payload=attempt_payload,
                    raw_response=content,
                    parsed_plan=None,
                    error="response truncated (finish_reason=length)",
                )
                if not truncation_retried:
                    truncation_retried = True
                    append_event(
                        event_dir,
                        "observation",
                        fact="planner response truncated (finish_reason=length); one compact re-emit",
                    )
                    errors.append("previous response was truncated mid-JSON (finish_reason=length)")
                    continue
                raise PlannerContractError(
                    "planner response truncated twice (finish_reason=length). "
                    "Increase CHATEXP_OPENAI_MAX_TOKENS or simplify the request.",
                    errors,
                )

            plan, validation_error = self._coerce_intent_plan_with_error(content or "", attempt_payload)
            if (
                plan is None
                and content
                and not json_repair_done
                and validation_error == "planner did not return a JSON object"
            ):
                # Exactly ONE JSON repair round quoting the parse failure; a
                # failing repair is loud.
                json_repair_done = True
                budget.register("planner_json_repair")
                plan, validation_error = self._repair_plan(
                    raw_text=content,
                    trace_dir=trace_dir,
                    task_id=task_id,
                    event_dir=event_dir,
                    parse_error=validation_error,
                )

            self._write_trace(
                trace_dir=trace_dir,
                stage=f"planner_attempt_{attempt}",
                task_id=task_id,
                request_payload=attempt_payload,
                raw_response=content,
                parsed_plan=plan,
                error=validation_error,
            )
            if plan is not None and validation_error is None:
                self.last_request_usage = dict(self._request_usage)
                self.last_request_usage["llm_calls"] = budget.used
                return plan
            errors.append(validation_error or f"attempt {attempt}: planner returned no valid plan")
            if attempt >= max_attempts:
                raise PlannerContractError(
                    f"planner failed skill/action contract after {attempt} attempts: {errors[-1]}", errors
                )

    def _coerce_intent_plan(self, text: str) -> IntentPlan | None:
        data = _extract_plan_json(text)
        if data is None:
            return None
        try:
            return IntentPlan.model_validate(data)
        except Exception:
            return None

    def _coerce_intent_plan_with_error(
        self,
        text: str,
        original_payload: dict[str, Any],
    ) -> tuple[IntentPlan | None, str | None]:
        data = _extract_plan_json(text)
        if data is None:
            return None, "planner did not return a JSON object"
        try:
            plan = IntentPlan.model_validate(data)
        except Exception as exc:
            return None, f"planner JSON failed IntentPlan validation: {type(exc).__name__}: {exc}"
        validation_error = _validate_planner_plan(plan, original_payload)
        if validation_error:
            return plan, validation_error
        return plan, None

    def _repair_plan(
        self,
        raw_text: str,
        trace_dir: str | Path | None = None,
        task_id: str | None = None,
        event_dir: str | Path | None = None,
        parse_error: str | None = None,
    ) -> tuple[IntentPlan | None, str | None]:
        """One JSON repair round: hand the raw (unparseable) reply back to the
        model with the repair contract and the exact parse failure."""
        if not raw_text.strip():
            return None, "repair skipped: empty raw text"
        repair_instruction = (
            "The following text was expected to be a single JSON plan object but failed to parse"
            + (f' ({parse_error})' if parse_error else "")
            + ". Return ONE valid JSON object that matches the planner output contract — "
            "no markdown fences, no commentary, no trailing text.\n\nRaw text:\n" + raw_text
        )
        stage = "planner_json_repair"
        self._last_stage_usage = None
        try:
            repaired_text, request_error, finish_reason, usage = self._request_stage(
                stage=stage,
                messages=_build_repair_messages(repair_instruction),
                trace_dir=trace_dir,
                task_id=task_id,
                request_payload={"raw_text": raw_text[:4000], "parse_error": parse_error},
            )
        except PlannerContractError:
            raise
        self._last_stage_usage = usage
        self._account_usage(usage, event_dir)
        plan = self._coerce_intent_plan(repaired_text or "")
        error = request_error or (None if plan is not None else "repair round did not return a valid JSON plan")
        self._write_trace(
            trace_dir=trace_dir,
            stage=stage,
            task_id=task_id,
            request_payload={"raw_text": raw_text[:4000], "parse_error": parse_error},
            raw_response=repaired_text,
            parsed_plan=plan,
            error=error,
        )
        if plan is None and not request_error:
            append_event(event_dir, "error", message=f"planner JSON repair round failed: {error}")
        return plan, error

    def _request_stage(
        self,
        stage: str,
        messages: list[dict[str, str]],
        trace_dir: str | Path | None,
        task_id: str | None,
        request_payload: dict[str, Any],
    ) -> tuple[str | None, str | None, str | None, dict[str, Any] | None]:
        """One LLM call. Returns (content, request_error, finish_reason, usage).

        Auth and quota failures propagate loudly; every other transport failure
        is returned as an error string so the orchestration loop can classify
        it (context overflow, truncation, contract failure).
        """
        try:
            completion = self._client.create_chat_completion_full(
                model=str(self.settings.model),
                temperature=0,
                response_format={"type": "json_object"},
                messages=messages,
            )
            return completion.text, None, completion.finish_reason, completion.usage
        except LLMAuthenticationError:
            raise
        except QuotaExhaustionError:
            raise
        except BudgetExceededError:
            raise
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}", None, None

    # ── harness state helpers ────────────────────────────────────────────

    def _event_dir_for(self, trace_dir: str | Path | None) -> Path | None:
        # trace_dir is <task_dir>/llm_traces; the audit event log lives one
        # level up, next to session.json.
        if trace_dir is None:
            return None
        try:
            return Path(trace_dir).expanduser().resolve().parent
        except (OSError, ValueError):
            return None

    def _ensure_provenance(self, event_dir: str | Path | None) -> None:
        if event_dir is None:
            return
        key = str(event_dir)
        if key in self._provenance_written:
            return
        if write_llm_provenance(event_dir, self.settings) is not None:
            self._provenance_written.add(key)

    def _history_payload(self, history: list[dict[str, str]] | None) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for item in history or []:
            if hasattr(item, "model_dump"):
                payload.append(item.model_dump())
            else:
                payload.append(item)
        return payload

    def _reset_request_usage(self) -> None:
        self._request_usage = {"input": 0, "output": 0, "total": 0, "calls": 0}

    def _account_usage(self, usage: dict[str, Any] | None, event_dir: str | Path | None) -> None:
        normalized = summarize_usage(usage)
        if normalized is None:
            return
        self._request_usage["input"] += normalized["input"]
        self._request_usage["output"] += normalized["output"]
        self._request_usage["total"] += normalized["total"]
        self._request_usage["calls"] += 1
        self._last_input_tokens = normalized["input"]
        append_event(
            event_dir,
            "budget",
            input=normalized["input"],
            output=normalized["output"],
            used=normalized["total"],
        )

    def _build_plan_payload(
        self,
        *,
        message: str,
        normalized_message: str,
        project: ProjectState | None,
        pending: ChatPending | None,
        history: list[dict[str, Any]],
        ui_state: dict[str, Any] | None,
        language: str,
        context_mode: str,
        extra_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if context_mode == "overflow_minimal":
            window, summary = build_history_window(
                history, "overflow_minimal", prior_summary=self._history_summary
            )
            skill_context = build_skill_context(
                message=message, project=project, pending=pending, ui_state=ui_state, language=language
            )
            return {
                "message": normalized_message,
                "raw_message": message,
                "context": {
                    "project": project.model_dump() if project is not None else None,
                    "pending": {"action": pending.action} if pending is not None and pending.action else None,
                    "history": window,
                    "history_digest": self._digest_lines(history, window, summary),
                    "planner_state": None,
                    "skills": {k: v for k, v in skill_context.items() if k != "catalog"},
                    "ui_state": {},
                    "guidance": None,
                    "context_mode": "overflow_minimal",
                    **(extra_context or {}),
                },
            }

        # Full context with token-aware shrink: when the previous call's input
        # exceeded the compaction threshold, the next window shrinks. The flag
        # from an overflow retry is consumed exactly once.
        mode = "full"
        if self._skip_next_token_compaction:
            self._skip_next_token_compaction = False
        elif self._last_input_tokens and self._last_input_tokens > TOKEN_COMPACT_INPUT_THRESHOLD:
            mode = "token_shrunk"
        window, summary = build_history_window(history, mode, prior_summary=self._history_summary)
        self._history_summary = summary or self._history_summary
        history_recent = window
        history_digest = self._digest_lines(history, window, summary)
        planner_state = _build_planner_state(
            message=message,
            project=project,
            pending=pending,
            history=history,
            ui_state=ui_state,
            language=language,
        )
        skill_context = build_skill_context(
            message=message,
            project=project,
            pending=pending,
            ui_state=ui_state,
            language=language,
        )
        guidance = _build_guidance_for_skill_context(
            message=message,
            project=project,
            pending=pending,
            planner_state=planner_state,
            skill_context=skill_context,
        )
        return {
            "message": normalized_message,
            "raw_message": message,
            "context": {
                "project": project.model_dump() if project is not None else None,
                "pending": pending.model_dump() if pending is not None else None,
                "history": history_recent,
                "history_digest": history_digest,
                "planner_state": planner_state,
                "skills": {k: v for k, v in skill_context.items() if k != "catalog"},
                "ui_state": ui_state or {},
                "guidance": guidance,
                "context_mode": mode,
                **(extra_context or {}),
            },
        }

    def _digest_lines(
        self,
        history: list[dict[str, Any]],
        window: list[dict[str, str]],
        summary: str | None,
    ) -> list[str]:
        """Digest of turns beyond the verbatim window; the chained summary
        (when present) rides as the first line."""
        digest: list[str] = []
        if summary:
            digest.append(f"[对话摘要] {summary}")
        window_ids = {id(item) for item in window}
        older = [item for item in history if id(item) not in window_ids]
        digest.extend([_history_digest_line(item) for item in older if _history_digest_line(item)])
        return digest[-HISTORY_DIGEST_LIMIT:]

    def _write_trace(
        self,
        trace_dir: str | Path | None,
        stage: str,
        task_id: str | None,
        request_payload: dict[str, Any],
        raw_response: str | None,
        parsed_plan: IntentPlan | None,
        error: str | None,
    ) -> None:
        if trace_dir is None:
            return
        trace_root = Path(trace_dir).expanduser().resolve()
        trace_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        path = trace_root / f"{stamp}_{stage}.json"
        # Surface the most recent token-usage reported by the API (if any) so
        # the trace records real consumption instead of just the request.
        usage: dict[str, Any] | None = None
        client = getattr(self, "_client", None)
        if client is not None and getattr(client, "last_usage", None) is not None:
            usage = dict(client.last_usage)
        payload = {
            "task_id": task_id,
            "stage": stage,
            "model": self.settings.model,
            "base_url": self.settings.base_url,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "request": request_payload,
            "raw_response": raw_response,
            "parsed_plan": parsed_plan.model_dump() if parsed_plan is not None else None,
            "error": error,
            "usage": usage,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class OpenAIImportMapper:
    def __init__(self, project_root: str | Path | None = None):
        self.settings = load_openai_settings(project_root)
        self._client = _ChatCompletionClient(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            timeout_seconds=self.settings.timeout_seconds,
            max_tokens=self.settings.max_tokens,
        ) if self.settings.configured else None

    @property
    def available(self) -> bool:
        return self._client is not None and self.settings.configured

    def map_instruction(
        self,
        *,
        instruction: str,
        source_columns: list[str],
        target_candidates: list[str],
    ) -> dict[str, str]:
        """Map a natural-language instruction to a {source: target} column
        mapping; retries up to 3 times and raises on persistent failure."""
        if not self.available:
            raise RuntimeError("列映射需要可用的 LLM 配置。")
        payload = {
            "instruction": instruction,
            "source_columns": source_columns,
            "target_candidates": target_candidates,
            "task": "Map the user's natural-language column mapping instruction to source->target JSON.",
        }
        return self._request_mapping(payload)

    def _request_mapping(self, payload: dict[str, Any]) -> dict[str, str]:
        errors: list[str] = []
        data: dict[str, Any] | None = None
        for attempt in range(1, 4):
            retry_note = ""
            if errors:
                retry_note = (
                    "\n\n上一次列映射输出未通过校验："
                    + errors[-1]
                    + "\n请重新判断用户原话，只输出 JSON：{\"column_mapping\":{\"原列名\":\"目标列名\"}}。"
                )
            try:
                text = self._client.create_chat_completion(
                    model=str(self.settings.model),
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=_build_import_mapper_messages(
                        json.dumps(payload, ensure_ascii=False)
                        + retry_note
                        + '\n请只输出 JSON：{"column_mapping":{"原列名":"规范列名"}}'
                    ),
                )
            except LLMAuthenticationError:
                raise
            except Exception as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                continue
            data = _extract_json(text or "")
            if isinstance(data, dict) and isinstance(data.get("column_mapping"), dict):
                break
            errors.append(f"attempt {attempt}: 列映射模型未返回 column_mapping JSON。")
            data = None
        if not isinstance(data, dict):
            raise ValueError("列映射模型 3 次尝试后仍未返回合法 JSON：" + " | ".join(errors))
        mapping = data.get("column_mapping")
        if not isinstance(mapping, dict):
            raise ValueError("列映射模型没有返回 column_mapping。")
        cleaned: dict[str, str] = {}
        for key, value in mapping.items():
            source = str(key or "").strip()
            target = str(value or "").strip()
            if source and target:
                cleaned[source] = target
        if not cleaned:
            raise ValueError("列映射模型返回了空映射。")
        return cleaned


class OpenAIImportInterpreter:
    def __init__(self, project_root: str | Path | None = None):
        self.settings = load_openai_settings(project_root)
        self._client = _ChatCompletionClient(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            timeout_seconds=self.settings.timeout_seconds,
            max_tokens=self.settings.max_tokens,
        ) if self.settings.configured else None

    @property
    def available(self) -> bool:
        return self._client is not None and self.settings.configured

    def interpret_sheet(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Interpret a raw-sheet payload into a schema dict (sheet type,
        header row, column mapping, factors, objectives, records). Retries
        up to 3 times; requires the LLM to return non-empty ``records``."""
        if not self.available:
            raise RuntimeError("实验导入需要可用的 LLM 配置。")
        base_content = (
            json.dumps(payload, ensure_ascii=False)
            + '\n请只输出 JSON，字段必须符合契约，尤其要返回 records 数组。'
        )
        errors: list[str] = []
        data: dict[str, Any] | None = None
        for attempt in range(1, 4):
            retry_note = ""
            if errors:
                retry_note = (
                    "\n\n上一次输出未通过结构化校验："
                    + errors[-1]
                    + "\n请重新识别原始数据，只输出符合契约的 JSON，不要省略 records。"
                )
            try:
                text = self._client.create_chat_completion(
                    model=str(self.settings.model),
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=_build_import_interpreter_messages(base_content + retry_note),
                )
            except LLMAuthenticationError:
                raise
            except Exception as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                continue
            data = _extract_json(text or "")
            if isinstance(data, dict) and isinstance(data.get("records"), list) and data.get("records"):
                break
            errors.append(f"attempt {attempt}: 导入解释模型未返回合法 records JSON。")
            data = None
        if not isinstance(data, dict):
            raise ValueError("导入解释模型 3 次尝试后仍未返回合法 JSON：" + " | ".join(errors))
        cleaned_records: list[dict[str, Any]] = []
        for row in data.get("records") or []:
            if not isinstance(row, dict):
                continue
            cleaned_row = {str(key).strip(): value for key, value in row.items() if str(key).strip()}
            if cleaned_row:
                cleaned_records.append(cleaned_row)
        if not cleaned_records:
            raise ValueError("导入解释模型 3 次尝试后没有返回可导入记录。")
        return {
            "sheet_type": str(data.get("sheet_type") or "unknown").strip() or "unknown",
            "header_row": self._coerce_optional_int(data.get("header_row")),
            "notes": str(data.get("notes") or "").strip(),
            "workbook_name": str(data.get("workbook_name") or "").strip(),
            "ignored_columns": self._clean_str_list(data.get("ignored_columns")),
            "factor_candidates": self._clean_str_list(data.get("factor_candidates")),
            "objective_candidates": self._clean_str_list(data.get("objective_candidates")),
            "column_mapping": self._clean_mapping(data.get("column_mapping")),
            "factors": self._clean_factor_specs(data.get("factors")),
            "objectives": self._clean_objective_specs(data.get("objectives")),
            "records": cleaned_records,
        }

    def _clean_str_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned

    def _clean_mapping(self, value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        cleaned: dict[str, str] = {}
        for key, item in value.items():
            source = str(key or "").strip()
            target = str(item or "").strip()
            if source and target:
                cleaned[source] = target
        return cleaned

    def _clean_factor_specs(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        cleaned: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            factor = {
                "name": name,
                "kind": str(item.get("kind") or item.get("type") or "continuous").strip(),
                "unit": str(item.get("unit")).strip() if item.get("unit") not in (None, "") else None,
                "min_value": item.get("min_value"),
                "max_value": item.get("max_value"),
                "step": item.get("step"),
                "categories": item.get("categories") if isinstance(item.get("categories"), list) else [],
            }
            cleaned.append(factor)
        return cleaned

    def _clean_objective_specs(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        cleaned: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            goal = str(item.get("goal") or "maximize").strip().lower()
            if goal not in {"maximize", "minimize", "target"}:
                goal = "maximize"
            objective = {
                "name": name,
                "goal": goal,
                "target_value": item.get("target_value"),
                "weight": item.get("weight", 1.0),
                "unit": str(item.get("unit")).strip() if item.get("unit") not in (None, "") else None,
                "desirability": str(item.get("desirability")).strip() if item.get("desirability") not in (None, "") else None,
            }
            cleaned.append(objective)
        return cleaned

    def _coerce_optional_int(self, value: Any) -> int | None:
        try:
            if value in (None, ""):
                return None
            return int(value)
        except Exception:
            return None


class OpenAIExperimentAnalyst:
    """LLM-only narrative layer for data diagnosis and terminal summaries."""

    def __init__(self, project_root: str | Path | None = None):
        self.settings = load_openai_settings(project_root)
        self._client = _ChatCompletionClient(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            timeout_seconds=self.settings.timeout_seconds,
            max_tokens=self.settings.max_tokens,
        ) if self.settings.configured else None

    @property
    def available(self) -> bool:
        return self._client is not None and self.settings.configured

    def summarize(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Narrate an analysis payload as {message, highlights, next_actions,
        display_hints}; retries up to 3 times and requires a non-empty
        ``message``."""
        if not self.available:
            raise RuntimeError("数据分析和结果展示需要可用的 LLM 配置。")
        errors: list[str] = []
        data: dict[str, Any] | None = None
        base_content = (
            json.dumps(payload, ensure_ascii=False)
            + '\n请只输出 JSON：{"message":"...","highlights":["..."],"next_actions":["..."],"display_hints":["..."]}'
        )
        for attempt in range(1, 4):
            retry_note = ""
            if errors:
                retry_note = (
                    "\n\n上一次分析输出未通过校验："
                    + errors[-1]
                    + "\n请重新读取原始 payload，输出完整 JSON，不要退回模板化或规则化结论。"
                )
            try:
                text = self._client.create_chat_completion(
                    model=str(self.settings.model),
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    messages=_build_analysis_narrator_messages(base_content + retry_note),
                )
            except LLMAuthenticationError:
                raise
            except Exception as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                continue
            parsed = _extract_json(text or "")
            if isinstance(parsed, dict):
                message = str(parsed.get("message") or "").strip()
                if message:
                    data = parsed
                    break
            errors.append(f"attempt {attempt}: analysis narrator did not return a JSON object with message.")
        if not isinstance(data, dict):
            raise ValueError("LLM 数据分析 3 次尝试后仍未返回合法 JSON：" + " | ".join(errors))
        return {
            "message": str(data.get("message") or "").strip(),
            "highlights": self._clean_str_list(data.get("highlights")),
            "next_actions": self._clean_str_list(data.get("next_actions")),
            "display_hints": self._clean_str_list(data.get("display_hints")),
        }

    def _clean_str_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                cleaned.append(text)
        return cleaned


def _normalize_message(message: str, pending: ChatPending | None, history: list[dict[str, Any]]) -> str:
    """Whitespace normalization only. Intent steering (pasted data, starter
    requests, continuations) is intentionally NOT done here: the planner
    routes semantically from the skill catalog and full context — no keyword
    pre-processing, no worked examples, no domain word lists.""" 
    return " ".join(message.split())


def _validate_planner_plan(plan: IntentPlan, payload: dict[str, Any]) -> str | None:
    if plan.actions:
        for child in plan.actions:
            error = _validate_planner_plan(child, payload)
            if error:
                return error
        return None
    if plan.action == "append_results":
        rows = plan.arguments.get("rows")
        if rows is not None and (not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows)):
            return "append_results.rows must be a list of objects"
    return None


def _build_planner_state(
    message: str,
    project: ProjectState | None,
    pending: ChatPending | None,
    history: list[dict[str, Any]],
    ui_state: dict[str, Any] | None = None,
    language: str = "zh",
) -> dict[str, Any]:
    factor_state = _factor_state(project=project, pending=pending)
    objective = _objective_state(project=project, pending=pending)
    missing_slots = _detect_missing_slots(
        project=project,
        pending=pending,
        objective=objective,
        factor_state=factor_state,
    )
    stage = _planner_stage(project=project, pending=pending)
    recent_focus = _recent_user_focus(history)
    workspace_focus = _workspace_focus_state(ui_state)
    reply_options = _reply_options_state(ui_state)
    return {
        "stage": stage,
        "summary": _planner_summary(stage=stage, objective=objective, factor_state=factor_state, missing_slots=missing_slots),
        "objective": objective,
        "factor_state": factor_state,
        "missing_slots": missing_slots,
        "open_questions": _build_open_questions(
            missing_slots,
            factor_state=factor_state,
            objective=objective,
            stage=stage,
            language=language,
        ),
        "recent_user_focus": recent_focus,
        "workspace_focus": workspace_focus,
        "reply_options": reply_options,
        "ready_for_draft": bool(objective.get("name")) and bool(factor_state),
    }


def _planner_stage(project: ProjectState | None, pending: ChatPending | None) -> str:
    if pending is not None and pending.action == "confirm_setup":
        return "awaiting_setup_confirmation"
    if pending is not None and pending.action == "confirm_init_design":
        return "awaiting_initialize_confirmation"
    if pending is not None and pending.action == "confirm_generate_design":
        return "awaiting_generation_confirmation"
    if project is None:
        return "scoping"
    if not project.factors or not project.objectives:
        return "project_setup"
    return "active_project"


def _objective_state(project: ProjectState | None, pending: ChatPending | None) -> dict[str, Any]:
    objective_payload: dict[str, Any] | None = None
    if project is not None and project.objectives:
        objective_payload = project.objectives[0].model_dump()
    elif pending is not None and pending.data.get("objective"):
        value = pending.data.get("objective")
        if isinstance(value, dict):
            objective_payload = value
    if objective_payload is None:
        return {
            "name": None,
            "goal": None,
            "known": False,
        }
    return {
        "name": str(objective_payload.get("name") or ""),
        "goal": objective_payload.get("goal"),
        "target_value": objective_payload.get("target_value"),
        "known": True,
    }


def _factor_state(project: ProjectState | None, pending: ChatPending | None) -> list[dict[str, Any]]:
    factor_payloads: list[dict[str, Any]] = []
    if project is not None and project.factors:
        factor_payloads = [factor.model_dump() for factor in project.factors]
    elif pending is not None and pending.data.get("factors"):
        raw = pending.data.get("factors")
        if isinstance(raw, list):
            factor_payloads = [item for item in raw if isinstance(item, dict)]
    state: list[dict[str, Any]] = []
    for item in factor_payloads:
        kind = str(item.get("kind") or "continuous")
        categories = item.get("categories") or []
        range_known = (
            bool(categories) if kind == "categorical"
            else item.get("min_value") is not None and item.get("max_value") is not None
        )
        state.append(
            {
                "name": str(item.get("name") or ""),
                "kind": kind,
                "range_known": range_known,
                "unit": item.get("unit"),
            }
        )
    return state


def _detect_missing_slots(
    project: ProjectState | None,
    pending: ChatPending | None,
    objective: dict[str, Any],
    factor_state: list[dict[str, Any]],
) -> list[str]:
    missing_slots: list[str] = []
    if not objective.get("name"):
        missing_slots.append("主要优化目标")
    if not factor_state:
        missing_slots.append("可控因素")
    if not factor_state and project is None and pending is None:
        missing_slots.append("因素范围或约束")
    for factor in factor_state:
        if factor.get("name") and not factor.get("range_known"):
            missing_slots.append(f"{factor['name']}的范围或类别")
    return missing_slots


def _build_open_questions(
    missing_slots: list[str],
    factor_state: list[dict[str, Any]],
    objective: dict[str, Any],
    stage: str,
    language: str = "zh",
) -> list[str]:
    # Clarify questions are derived purely from state (missing slots); they
    # never enumerate domain-specific example words.
    questions: list[str] = []

    if language == "en":
        if "主要优化目标" in missing_slots:
            questions.append("Which metric should be optimized, and in which direction (higher or lower)?")
        if "可控因素" in missing_slots:
            questions.append("Which controllable factors should be considered, and what are their ranges or candidate categories?")
        incomplete = [factor["name"] for factor in factor_state if factor.get("name") and not factor.get("range_known")]
        if incomplete:
            joined = ", ".join(incomplete[:3])
            questions.append(f"What are the possible ranges or candidate categories for {joined}?")
        if stage == "active_project" and objective.get("name") and not questions:
            questions.append(f"To continue optimizing {objective['name']}, which factors should change or which results should be added?")
    else:
        if "主要优化目标" in missing_slots:
            questions.append("要优化的指标是什么？优化方向是越高越好还是越低越好？")
        if "可控因素" in missing_slots:
            questions.append("有哪些可控因素？请一并给出各因素的范围或候选类别。")
        incomplete = [factor["name"] for factor in factor_state if factor.get("name") and not factor.get("range_known")]
        if incomplete:
            joined = "、".join(incomplete[:3])
            questions.append(f"{joined} 的可选范围或候选类别是什么？")
        if stage == "active_project" and objective.get("name") and not questions:
            questions.append(f"如果你要继续优化 {objective['name']}，优先告诉我这轮想改哪些因素或追加哪些实验结果。")
    return questions[:3]


def _planner_summary(
    stage: str,
    objective: dict[str, Any],
    factor_state: list[dict[str, Any]],
    missing_slots: list[str],
) -> str:
    objective_name = objective.get("name") or "目标未明确"
    factor_names = "、".join(factor["name"] for factor in factor_state[:4] if factor.get("name")) or "暂无因素"
    missing = "、".join(missing_slots[:4]) or "当前关键信息完整"
    return f"阶段={stage}；目标={objective_name}；因素={factor_names}；待补充={missing}"


def _recent_user_focus(history: list[dict[str, Any]]) -> list[str]:
    focus: list[str] = []
    for item in history:
        if str(item.get("role") or "") != "user":
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        focus.append(_truncate_text(" ".join(content.split()), HISTORY_SNIPPET_LIMIT))
    return focus[-3:]


def _workspace_focus_state(ui_state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(ui_state, dict):
        return None
    raw = ui_state.get("workspace_focus")
    if not isinstance(raw, dict) or not raw:
        return None
    conditions: dict[str, Any] = {}
    for key, value in raw.items():
        if key in {
            "kind",
            "Trial",
            "candidate_id",
            "experiment_id",
            "status",
            "COMMENT",
            "source",
            "score",
            "FuzzyScore",
        }:
            continue
        if value in (None, ""):
            continue
        conditions[str(key)] = value
    return {
        "kind": str(raw.get("kind") or ""),
        "experiment_id": raw.get("experiment_id"),
        "trial": raw.get("Trial"),
        "candidate_id": raw.get("candidate_id"),
        "status": raw.get("status"),
        "conditions": conditions,
        "is_pending_plan": str(raw.get("kind") or "") == "plan" and str(raw.get("status") or "").lower() not in {"completed", "skipped"},
    }


def _reply_options_state(ui_state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(ui_state, dict):
        return None
    raw = ui_state.get("reply_options")
    if not isinstance(raw, dict):
        return None
    items = raw.get("items")
    if not isinstance(items, list):
        return None
    normalized_items: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        option_id = str(item.get("id") or item.get("label") or "").strip()
        text = str(item.get("text") or "").strip()
        if option_id and text:
            normalized_items.append({"id": option_id, "text": text})
    if not normalized_items:
        return None
    return {
        "type": "reply_options",
        "version": raw.get("version") or 1,
        "items": normalized_items,
        "schema_hint": "Use id/text semantics when the user sends a short option reference.",
    }


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 1)].rstrip() + "…"


def _build_guidance_context(
    message: str,
    project: ProjectState | None,
    pending: ChatPending | None,
    planner_state: dict[str, Any],
) -> dict[str, Any]:
    """State-derived guidance only: no message keyword inference, no domain
    word lists, no factor type hints. Deictic reference resolution (e.g. 'this
    run' pointing at the focused workspace row) is a generic harness mechanism."""
    known_factor_count = len(planner_state.get("factor_state") or [])
    objective_payload = planner_state.get("objective") or {}
    objective_known = bool(objective_payload.get("name"))
    missing_fields = list(planner_state.get("missing_slots") or [])
    workspace_focus = planner_state.get("workspace_focus") or {}
    stage = planner_state.get("stage")
    if _looks_like_plan_edit_request(message, workspace_focus):
        recommended_next_action = "update_plan_experiment"
    elif stage in ("scoping", "project_setup") and not project:
        recommended_next_action = "draft_setup"
    elif missing_fields:
        recommended_next_action = "clarify"
    else:
        recommended_next_action = None

    return {
        "assistant_role": "工艺优化助手",
        "known_factor_count": known_factor_count,
        "objective_known": objective_known,
        "missing_fields": missing_fields,
        "planner_stage": stage,
        "open_questions": planner_state.get("open_questions") or [],
        "workspace_focus": workspace_focus,
        "reply_options": planner_state.get("reply_options"),
        "recommended_next_action": recommended_next_action,
    }


def _build_guidance_for_skill_context(
    *,
    message: str,
    project: ProjectState | None,
    pending: ChatPending | None,
    planner_state: dict[str, Any],
    skill_context: dict[str, Any],
) -> dict[str, Any]:
    guidance = _build_guidance_context(
        message,
        project=project,
        pending=pending,
        planner_state=planner_state,
    )
    if skill_context.get("active_layer") == "conversational interface":
        guidance["recommended_next_action"] = "respond"
        guidance["missing_fields"] = []
        guidance["open_questions"] = []
    return guidance


# Generic deictic reference tokens: "this run / that entry" style pointers
# resolved against the focused workspace row (harness-level, domain-agnostic).
_DEICTIC_TOKENS = ("这个实验", "这条", "这一条", "该实验", "它", "改", "调整", "换成", "改成", "需要把", "改到", "this run", "this entry", "that row")


def _looks_like_plan_edit_request(message: str, workspace_focus: dict[str, Any] | None) -> bool:
    """Resolve a deictic plan reference against the focused workspace row."""
    if not isinstance(workspace_focus, dict) or not workspace_focus.get("is_pending_plan"):
        return False
    lowered = message.lower()
    if not any(token in lowered for token in _DEICTIC_TOKENS):
        return False
    conditions = workspace_focus.get("conditions") or {}
    return any(str(name).lower() in lowered for name in conditions)


def _extract_plan_json(text: str) -> dict[str, Any] | None:
    """Hybrid JSON salvage: a strict ladder (fences → bare → brace-slice,
    truncated output is never salvageable) first, then tolerant parsers
    (quote normalization, python literal) for messy real-world output."""
    data = salvage_extract_json(text)
    if data is not None:
        return data
    return _extract_json(text)


def _extract_json(text: str) -> dict[str, Any] | None:
    candidates = _json_candidates(text)
    for candidate in candidates:
        parsed = _parse_candidate(candidate)
        if parsed is not None:
            return parsed
    return None


def _json_candidates(text: str) -> list[str]:
    cleaned = re.sub(r"<[^>]+>", "", text, flags=re.IGNORECASE).strip()
    candidates: list[str] = []
    if cleaned:
        candidates.append(cleaned)
    first_object = _first_braced_object(cleaned)
    if first_object and first_object not in candidates:
        candidates.append(first_object)
    return candidates


def _first_braced_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _parse_candidate(text: str) -> dict[str, Any] | None:
    for parser in (_parse_json, _parse_json_after_quote_normalization, _parse_python_literal):
        parsed = parser(text)
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_json(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _parse_json_after_quote_normalization(text: str) -> dict[str, Any] | None:
    normalized = text.replace("'", '"')
    normalized = re.sub(r',\s*([}\]])', r"\1", normalized)
    try:
        data = json.loads(normalized)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _parse_python_literal(text: str) -> dict[str, Any] | None:
    normalized = text.replace("null", "None").replace("true", "True").replace("false", "False")
    try:
        data = ast.literal_eval(normalized)
    except (ValueError, SyntaxError):
        return None
    return data if isinstance(data, dict) else None


class _ChatCompletionClient:
    def __init__(
        self,
        api_key: str | None,
        base_url: str | None,
        timeout_seconds: int = 60,
        max_tokens: int | None = None,
    ):
        self.api_key = api_key or ""
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/"
        self.timeout_seconds = max(int(timeout_seconds or 60), 1)
        self.max_tokens = int(max_tokens or 0)
        # Most recent token-usage payload reported by the API, exposed so the
        # trace writer can record it. None when no successful response has
        # been observed yet (or the server returned no usage block).
        self.last_usage: dict[str, Any] | None = None

    def create_chat_completion(
        self,
        model: str,
        temperature: float,
        response_format: dict[str, Any] | None,
        messages: list[dict[str, str]],
    ) -> str:
        """Completion returning text only; raises LLMTruncatedError on
        finish_reason=length and RuntimeError on empty content."""
        completion = self.create_chat_completion_full(model, temperature, response_format, messages)
        if completion.finish_reason == "length":
            raise LLMTruncatedError(
                "LLM response was cut off (finish_reason=length); the content is incomplete.",
                truncated_tail=completion.text or "",
            )
        if not (completion.text or "").strip():
            raise RuntimeError("LLM returned empty content.")
        return completion.text or ""

    def create_chat_completion_full(
        self,
        model: str,
        temperature: float,
        response_format: dict[str, Any] | None,
        messages: list[dict[str, str]],
    ) -> HarnessCompletion:
        """Completion with finish_reason and usage; truncation is rejected."""
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
        }
        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        errors: list[str] = []
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(
                    urljoin(self.base_url, "chat/completions"),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                try:
                    response.raise_for_status()
                except requests.HTTPError as exc:
                    body = response.text[:1000] if response is not None else ""
                    status = response.status_code
                    # 401/403 are fatal — do not retry; surface as auth error.
                    if status in {401, 403} or _looks_like_auth_error(body):
                        raise LLMAuthenticationError(
                            f"LLM authentication failed ({status}). Check API token/key and base URL. Response body: {body}"
                        ) from exc
                    # Deterministic quota/billing exhaustion is never retried.
                    if is_quota_exhaustion_text(body):
                        raise QuotaExhaustionError(
                            f"LLM quota exhausted (HTTP {status}); request NOT retried. Response body: {body}"
                        ) from exc
                    # 429 (and other 5xx retriable) — honor Retry-After, then
                    # back off exponentially with jitter before retrying.
                    if status == 429 or 500 <= status < 600:
                        retry_after = self._retry_wait_seconds(response)
                        if attempt < max_attempts:
                            self._sleep_for_retry(attempt, retry_after)
                            errors.append(f"attempt {attempt}: HTTP {status}: {body}")
                            continue
                        raise RuntimeError(f"{exc}; response body: {body}") from exc
                    raise RuntimeError(f"{exc}; response body: {body}") from exc
                data = response.json()
                # Record usage in the trace channel so callers can persist it.
                usage = data.get("usage") if isinstance(data, dict) else None
                self.last_usage = usage
                choices = data.get("choices") or []
                finish_reason = str((choices[0] or {}).get("finish_reason") or "") if choices else ""
                text = self._extract_message_text(data)
                return HarnessCompletion(text=text, finish_reason=finish_reason or None, usage=usage)
            except (LLMAuthenticationError, QuotaExhaustionError, ServerDelayFailFastError):
                raise
            except Exception as exc:
                # Connection / timeout errors: back off and retry.
                if attempt < max_attempts:
                    self._sleep_for_retry(attempt, None)
                    errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                    continue
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
        raise RuntimeError("LLM request failed after 3 attempts: " + " | ".join(errors))

    def _retry_wait_seconds(self, response: requests.Response) -> float | None:
        """retry-after-ms wins over Retry-After (seconds or HTTP-date); a
        server-asked wait beyond the fail-fast threshold aborts the request."""
        from .agent_harness import parse_retry_after_headers

        try:
            return parse_retry_after_headers(response.headers)
        except ServerDelayFailFastError:
            raise

    def _sleep_for_retry(self, attempt: int, retry_after: float | None) -> None:
        # Server-provided wait is already clamped by the harness (max 30s);
        # otherwise exponential backoff with full jitter: base 0.5s × 2^(attempt-1).
        if retry_after is not None:
            time.sleep(retry_after)
            return
        base = 0.5 * (2 ** (attempt - 1))
        jitter = random.uniform(0, base)
        delay = jitter
        # Cap the wait so a flaky server cannot stall the UI indefinitely.
        time.sleep(min(max(delay, 0.0), 30.0))

    def _extract_message_text(self, data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM 返回为空: {data}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        text = ""
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text") or ""))
            text = "".join(text_parts)
        elif isinstance(content, str):
            text = content
        return text


def _looks_like_auth_error(text: str) -> bool:
    return bool(re.search(r"(invalid|incorrect|unauthorized|forbidden|authentication|api[\s_-]?key|api[\s_-]?token|bearer token)", text or "", re.IGNORECASE))
