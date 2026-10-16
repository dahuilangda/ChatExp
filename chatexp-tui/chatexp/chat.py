from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .llm import LLMAuthenticationError, OpenAIIntentPlanner, PlannerContractError
from .event_log import append_event, replay_for_recovery
from .models import (
    ChatMessage,
    ChatPending,
    FactorSpec,
    IntentPlan,
    ObjectiveSpec,
    ProjectState,
    normalize_factor_kind,
)
from .provenance import write_llm_provenance
from .service import ChatExpService
from .settings import Language
from .storage import create_or_update_workbook, display_task_id, load_session, save_session, task_llm_trace_dir
from .i18n import get_translator
from .terminal_harness import (
    CONFIRMATION_APPLY_ALL,
    CONFIRMATION_CANCEL,
    CONFIRMATION_CONFIRM,
    TerminalAction,
    TerminalWorkflowHarness,
)


ACTION_PROGRESS: dict[str, dict[Language, str]] = {
    "respond": {Language.ZH: "正在整理回复...", Language.EN: "Preparing response..."},
    "clarify": {Language.ZH: "正在整理需要补充的信息...", Language.EN: "Organizing additional information needed..."},
    "draft_setup": {Language.ZH: "正在整理推荐方案...", Language.EN: "Preparing recommended plan..."},
    "create_project": {Language.ZH: "正在创建工作簿...", Language.EN: "Creating workbook..."},
    "load_project": {Language.ZH: "正在读取已有项目...", Language.EN: "Loading existing project..."},
    "define_factors": {Language.ZH: "正在更新因素定义...", Language.EN: "Updating factor definitions..."},
    "define_objective": {Language.ZH: "正在更新目标定义...", Language.EN: "Updating objective definitions..."},
    "update_objective": {Language.ZH: "正在更新目标定义...", Language.EN: "Updating objective definitions..."},
    "update_factor": {Language.ZH: "正在更新因素配置...", Language.EN: "Updating factor configuration..."},
    "initialize_design": {Language.ZH: "正在准备初始化设计...", Language.EN: "Preparing initial design..."},
    "append_results": {Language.ZH: "正在写入实验结果...", Language.EN: "Writing experiment results..."},
    "update_plan_experiment": {Language.ZH: "正在修改实验计划...", Language.EN: "Modifying experiment plan..."},
    "analyze_data": {Language.ZH: "正在分析数据与建议...", Language.EN: "Analyzing data and recommendations..."},
    "generate_next_design": {Language.ZH: "正在准备生成新设计...", Language.EN: "Preparing new design generation..."},
    "import_experiment_sheet": {Language.ZH: "正在导入实验数据...", Language.EN: "Importing experiment data..."},
    "export": {Language.ZH: "正在导出项目摘要...", Language.EN: "Exporting project summary..."},
    "refresh_scores": {Language.ZH: "正在刷新评分...", Language.EN: "Refreshing scores..."},
    "inspect_workbook": {Language.ZH: "正在检查工作簿结构...", Language.EN: "Checking workbook structure..."},
    "status": {Language.ZH: "正在整理当前项目状态...", Language.EN: "Organizing current project status..."},
    "view_experiments": {Language.ZH: "正在更新实验视图...", Language.EN: "Updating experiment view..."},
    "skip_pending_designs": {Language.ZH: "正在跳过待做实验计划...", Language.EN: "Skipping pending experiment designs..."},
}


def _detect_input_language(text: str) -> str:
    """Detect if input text is primarily English or Chinese."""
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


def _detect_language_with_context(text: str, history: list | None, default_language: str = "zh") -> str:
    """Detect language with context from chat history.

    Uses the most recent message with actual text content to determine language preference.
    This handles cases where user replies with just numbers ("1", "2", etc.).
    """
    detected = _detect_input_language(text)
    if detected != "zh":
        return detected

    # If current text is ambiguous (no language content), check history
    if history and len(history) > 0:
        # Look for recent user messages with actual text content
        for msg in reversed(history[-5:]):  # Check last 5 messages
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            elif hasattr(msg, "role") and hasattr(msg, "content"):
                role = getattr(msg, "role", "")
                content = str(getattr(msg, "content", ""))
            else:
                continue

            if role == "user" and content and len(content.strip()) > 2:
                # Check if this message has language indicators
                hist_lang = _detect_input_language(content)
                if hist_lang != "zh":
                    return hist_lang

    # Default to the provided default_language
    return default_language


APPLY_ALL_WORDS = {"全部应用", "全部执行", "应用全部", "全部确认", "全部继续", "apply all", "apply_all", "confirm all", "run all"}
HISTORY_LIMIT = 12

# Language-specific word sets for i18n support
AFFIRMATIVE_WORDS_ZH = {"确认", "可以", "好的", "好", "行", "就这样", "继续", "下一步"}
AFFIRMATIVE_WORDS_EN = {"ok", "yes", "y", "confirm", "continue", "next"}
NEGATIVE_WORDS_ZH = {"取消", "不", "不要", "算了"}
NEGATIVE_WORDS_EN = {"no", "n", "cancel", "don't", "dont"}
REPLY_OPTIONS_UI_KEY = "reply_options"
MAX_REPLY_OPTIONS = 9


class ChatCoordinator:
    def __init__(self, workspace: str | Path, session_path: str | Path, language: Language = Language.ZH):
        workspace_path = Path(workspace).expanduser().resolve()
        session_file = Path(session_path).expanduser().resolve()
        self.service = ChatExpService(workspace_path)
        self.intent_planner = OpenAIIntentPlanner(project_root=Path(__file__).resolve().parents[1])
        self.workflow = TerminalWorkflowHarness(self)
        self._progress_callback: Callable[[str], None] | None = None
        self.session_path = session_file
        self.session = load_session(self.session_path)
        self.session.state_path = str(self.session_path)
        self.session.task_dir = str(workspace_path)
        if self.session.task_id is None:
            self.session.task_id = workspace_path.name
        if self.session.message_count < len(self.session.history):
            self.session.message_count = len(self.session.history)

        # Audit event log + LLM provenance live next to session.json; the BO
        # service writes its design events into the same log.
        self.event_dir = self.session_path.parent
        self.service.event_dir = self.event_dir
        self.last_recovery = replay_for_recovery(self.event_dir)
        if self.last_recovery and self.last_recovery.get("pending_writes"):
            append_event(
                self.event_dir,
                "observation",
                fact="recovery replay found pending writes: " + ", ".join(self.last_recovery["pending_writes"]),
            )
        write_llm_provenance(self.event_dir, self.intent_planner.settings)
        # Steering: user input submitted while a turn is active is queued
        # (cap 1 — a second message REPLACES the first) and auto-submitted as
        # a fresh turn when the active turn completes.
        self._steering_queue: list[str] = []
        self.busy = False

        self.language = language
        self._ = get_translator(language)
        self.affirmative_words = (
            AFFIRMATIVE_WORDS_ZH | AFFIRMATIVE_WORDS_EN
            if language == Language.ZH
            else AFFIRMATIVE_WORDS_EN
        )
        self.negative_words = (
            NEGATIVE_WORDS_ZH | NEGATIVE_WORDS_EN
            if language == Language.ZH
            else NEGATIVE_WORDS_EN
        )
        self.generic_title_words = {
            self._("generic_help"), self._("generic_example"),
            self._("generic_exit"), self._("generic_continue"),
            self._("generic_continue_alt"), self._("generic_next"),
            self._("generic_next_alt"), self._("generic_then"),
            self._("generic_step"), self._("generic_start"),
            self._("generic_ok"), self._("generic_yes"),
            self._("generic_yes_short"),
            "help", "example", "exit", "quit", "continue", "next", "start", "ok", "yes", "good",
        }

    def handle(self, message: str, progress_callback: Callable[[str], None] | None = None) -> str:
        """Process one user turn (routing, planning, execution, confirmation
        flows) and return the assistant reply; always persists the session."""
        self._progress_callback = progress_callback
        self.busy = True
        text = message.strip()
        try:
            if not text:
                return self._("enter_requirement")

            self._progress(self._("understanding_input"))
            self._ensure_task_title(text)
            self._remember("user", text)
            append_event(self.event_dir, "user_message", text=text[:2000])
            reply = self._route(text)
            self._progress(self._("saving_session"))
            self._commit(reply)
            return reply
        finally:
            self.busy = False
            self._progress_callback = None

    def queue_steering(self, text: str) -> bool:
        """Queue input submitted while a turn is active. Cap 1: a second
        message replaces the first (course correction, not backlog)."""
        normalized = (text or "").strip()
        if not normalized:
            return False
        self._steering_queue.clear()
        self._steering_queue.append(normalized)
        append_event(self.event_dir, "steering_queued", text=normalized[:500])
        return True

    def drain_steering(self) -> str | None:
        """Pop and return the queued steering message, if any."""
        if not self._steering_queue:
            return None
        text = self._steering_queue.pop(0)
        append_event(self.event_dir, "steering_drained", text=text[:500])
        return text

    @property
    def steering_queued_text(self) -> str | None:
        """The queued (not yet drained) steering message, if any."""
        return self._steering_queue[0] if self._steering_queue else None

    def _progress(self, message: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(message)

    def _route(self, text: str) -> str:
        lowered = text.lower()
        if any(keyword in lowered for keyword in {"help", "帮助", "示例"}):
            return self._("help_text")
        if any(keyword in lowered for keyword in {"退出", "exit", "quit"}):
            return self._("chat_ended")

        try:
            if self.session.pending.action:
                self._progress(self._("processing_pending"))
                return self._handle_pending(text)
            if not self.intent_planner.available:
                return self._planner_required_reply()
            self._progress(self._("calling_model"))
            reply = self._handle_llm_plan(text)
            if reply is not None:
                return reply
            return self._handle_no_action_reply(text)
        except PlannerContractError as exc:
            append_event(self.event_dir, "error", message=f"planner contract error: {exc}"[:500])
            if self.language == Language.ZH:
                return f"LLM planner 连续 3 次未能按当前 skill/action contract 返回可执行 JSON：{exc}。请检查 llm_traces 中的 planner_attempt 记录。"
            return f"The LLM planner did not return executable JSON matching the current skill/action contract after 3 attempts: {exc}. Check the planner_attempt records in llm_traces."
        except LLMAuthenticationError as exc:
            append_event(self.event_dir, "error", message=f"LLM authentication failed: {exc}"[:500])
            if self.language == Language.ZH:
                return f"LLM 认证失败：请检查 API token/key、Base URL 和模型配置是否正确。原始错误：{exc}"
            return f"LLM authentication failed: check the API token/key, base URL, and model settings. Original error: {exc}"
        except Exception as exc:
            append_event(self.event_dir, "error", message=f"{type(exc).__name__}: {exc}"[:500])
            return f"执行时出错：{exc}" if self.language == Language.ZH else f"Error during execution: {exc}"

    def _commit(self, reply: str) -> None:
        self._remember("assistant", reply)
        self.session.last_reply = reply
        save_session(self.session)

    def _remember(self, role: str, content: str) -> None:
        self.session.history.append(ChatMessage(role=role, content=content))
        self.session.message_count += 1
        self.session.history = self.session.history[-HISTORY_LIMIT:]

    def _set_reply_options(self, options: list[str] | list[dict[str, str]]) -> None:
        normalized: list[dict[str, str]] = []
        for index, item in enumerate(options[:MAX_REPLY_OPTIONS], start=1):
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
            else:
                text = str(item or "").strip()
            if text:
                normalized.append(
                    {
                        "id": str(index),
                        "label": str(index),
                        "text": text,
                        "source": "clarify.examples",
                    }
                )
        if normalized:
            self.session.ui_state[REPLY_OPTIONS_UI_KEY] = {
                "type": "reply_options",
                "version": 1,
                "selection_instruction": (
                    "If the user's next message is a short reference such as a number, "
                    "interpret it as selecting the option with the matching id, then plan from that option text."
                ),
                "items": normalized,
            }
        else:
            self._clear_reply_options()

    def _clear_reply_options(self) -> None:
        self.session.ui_state.pop(REPLY_OPTIONS_UI_KEY, None)

    def _ensure_task_title(self, text: str) -> None:
        normalized = " ".join(text.split())
        lowered = normalized.lower()
        if lowered in self.generic_title_words or len(normalized) < 4:
            return
        if self._has_meaningful_title(self.session.title):
            return
        self.session.title = normalized[:48] or self._("unnamed_task")

    def _has_meaningful_title(self, title: str | None) -> bool:
        if not title:
            return False
        normalized = " ".join(title.split()).lower()
        return bool(normalized and normalized not in self.generic_title_words and len(normalized) >= 4)

    def _refresh_task_title_from_project(self, project: ProjectState) -> None:
        if self._has_meaningful_title(self.session.title):
            return
        if project.objectives:
            objective = project.objectives[0].name
            suffix = self._("optimization_suffix")
            self.session.title = f"{objective}{suffix}"
            return
        self.session.title = project.name

    def _handle_pending(self, text: str) -> str:
        action = self.session.pending.action
        if action == "confirm_setup":
            return self._handle_pending_setup(text)
        if action == "confirm_init_design":
            return self._handle_pending_design(
                text=text,
                expected_action="initialize_design",
                confirm_label="初始化",
                cancel_reply=self._("design_cancelled"),
            )
        if action == "confirm_generate_design":
            return self._handle_pending_design(
                text=text,
                expected_action="generate_next_design",
                confirm_label="生成新设计",
                cancel_reply=self._("generating_cancelled"),
            )
        if action == "confirm_import_experiment_sheet":
            return self._handle_pending_import(text)
        if action == "confirm_result_report":
            return self._handle_pending_result_report(text)
        if action == "confirm_terminal_action":
            return self._handle_pending_terminal_action(text)
        self.session.pending = ChatPending()
        return self._("pending_cleared")

    def _handle_pending_terminal_action(self, text: str) -> str:
        decision = self._confirmation_decision(text)
        if decision == CONFIRMATION_CANCEL:
            self.session.pending = ChatPending()
            return "已取消待执行操作。" if self.language == Language.ZH else "Pending action cancelled."
        if decision is None:
            return (
                "如果确认执行，请回复 `确认`；如果要执行本轮剩余低/中风险步骤，请回复 `全部应用`；如果要取消，请回复 `取消`。"
                if self.language == Language.ZH
                else "Reply `confirm` to execute, `apply all` to execute the remaining low/medium-risk steps, or `cancel` to abort."
            )
        consumed_pending = self.session.pending.model_dump()
        result = self.workflow.execute_pending(
            self.session.pending,
            self._executor_for_terminal_action,
            decision=decision,
        )
        if result.blocked:
            if result.executed_actions:
                executed_types = [item.type for item in result.executed_actions]
                self.session.pending = ChatPending()
                if self.language == Language.EN:
                    note = (
                        f"Steps already applied and not repeatable: {', '.join(executed_types)}. "
                        "The remaining steps were blocked, so the pending plan has been cleared; "
                        "re-issue the follow-up request after fixing the blocker above."
                    )
                else:
                    note = (
                        f"已执行且不会重复执行：{', '.join(executed_types)}。"
                        "后续步骤被阻止，待执行计划已清除；请先解决上面的问题，再重新发起后续操作。"
                    )
                return "\n\n".join([part for part in [result.content, note] if part])
            return result.content or ("审核未通过。" if self.language == Language.ZH else "Audit failed.")
        if result.pending is not None:
            self.session.pending = result.pending
        if self.session.pending.action == "confirm_terminal_action" and self.session.pending.model_dump() == consumed_pending:
            self.session.pending = ChatPending()
        return result.content or ("已完成。" if self.language == Language.ZH else "Done.")

    def _handle_pending_setup(self, text: str) -> str:
        if self._is_negative(text):
            self.session.pending = ChatPending()
            return self._("setup_cancelled")
        if self._is_affirmative(text):
            self._progress(self._("creating_project"))
            workbook_name, factors, objectives = self._pending_setup_payload()
            action = TerminalAction(
                type="create_project",
                arguments={
                    "workbook_name": workbook_name,
                    "factors": [factor.model_dump() for factor in factors],
                    "objectives": [objective.model_dump() for objective in objectives],
                },
                source="pending",
            )
            result = self.workflow.execute_workflow(action, self._executor_for_terminal_action(action), force=True)
            if result.blocked:
                return result.content or ("审核未通过。" if self.language == Language.ZH else "Audit failed.")
            self.session.pending = ChatPending()
            return result.content or ("已完成。" if self.language == Language.ZH else "Done.")
        self._progress("正在根据你的修改更新草案..." if self.language == Language.ZH else "Updating draft based on your modifications...")
        reply = self._apply_pending_setup_revision(text)
        if reply is not None:
            return reply
        return (
            "如果这个方案对，请回复 `确认` / `可以`；如果要调整，请直接描述你想改的内容。"
            if self.language == Language.ZH
            else "If this plan looks good, reply `confirm` / `yes`; if you want to adjust, describe what you'd like to change."
        )

    def _handle_pending_design(
        self,
        text: str,
        expected_action: str,
        confirm_label: str,
        cancel_reply: str,
    ) -> str:
        if self._is_negative(text):
            self.session.pending = ChatPending()
            return cancel_reply
        if self._is_affirmative(text):
            count = int(self.session.pending.data.get("count") or 0)
            arguments: dict[str, Any] = {"count": count}
            strategy_dump = self.session.pending.data.get("strategy")
            if isinstance(strategy_dump, dict):
                # Carry the confirmed directive so execution uses exactly the
                # strategy the operator approved.
                arguments["strategy"] = strategy_dump
            self._progress(f"正在执行{confirm_label}..." if self.language == Language.ZH else f"Executing {confirm_label}...")
            action = TerminalAction(type=expected_action, arguments=arguments, source="pending")
            executor = self._executor_for_terminal_action(action)
            result = self.workflow.execute_workflow(action, executor, force=True)
            if result.blocked:
                return result.content or ("审核未通过。" if self.language == Language.ZH else "Audit failed.")
            self.session.pending = ChatPending()
            return result.content or ("已完成。" if self.language == Language.ZH else "Done.")
        self._progress("正在更新执行数量..." if self.language == Language.ZH else "Updating execution count...")
        reply = self._handle_llm_pending_count(text, expected_action=expected_action, confirm_label=confirm_label)
        if reply is not None:
            return reply
        return (
            "如果确认执行，请回复 `确认` / `可以`；如果要改数量，请直接说明新的组数。"
            if self.language == Language.ZH
            else "If confirmed, reply `confirm` / `yes`; to change count, specify the new number of runs."
        )

    def _handle_pending_import(self, text: str) -> str:
        if self._is_negative(text):
            self.session.pending = ChatPending()
            return self._("import_cancelled")
        revised = self._apply_pending_import_mapping(text)
        if revised is not None:
            return revised
        if self._is_affirmative(text):
            import_summary = self.session.pending.data.get("import_summary") or {}
            if isinstance(import_summary, dict) and import_summary.get("needs_mapping_clarification"):
                missing = import_summary.get("missing_critical_targets") or []
                if missing:
                    if self.language == Language.EN:
                        return f"Missing key result columns: {', '.join(str(item) for item in missing)}. Please tell me which column corresponds to each."
                    else:
                        return f"当前还缺少关键结果列：{', '.join(str(item) for item in missing)}。请直接告诉我哪一列对应它。"
                if self.language == Language.EN:
                    return "The import preview is missing key mappings. Please add column correspondences first."
                else:
                    return "当前导入预览里还缺少关键映射，请先补充列对应关系。"
            self._progress("Writing import data..." if self.language == Language.EN else "正在写入导入数据...")
            pending_data = self.session.pending.data
            action = TerminalAction(
                type="import_experiment_sheet",
                arguments={
                    "source_excel": pending_data.get("source_excel"),
                    "objective_names": pending_data.get("objective_names"),
                    "manual_column_map": pending_data.get("manual_column_map"),
                },
                source="pending",
            )
            result = self.workflow.execute_workflow(action, self._executor_for_terminal_action(action), force=True)
            if result.blocked:
                return result.content or ("审核未通过。" if self.language == Language.ZH else "Audit failed.")
            self.session.pending = ChatPending()
            return result.content or ("已完成。" if self.language == Language.ZH else "Done.")
        if self.language == Language.EN:
            return 'If preview looks good, reply `confirm` / `yes`; if column mapping is wrong, say e.g. "Yield% column is the yield"; to cancel, reply `cancel`.'
        else:
            return '如果预览没问题，请回复 `确认` / `可以`；如果列对应有问题，直接说例如"收率% 这一列就是得率"；如果要取消，回复 `取消`。'

    def _handle_pending_result_report(self, text: str) -> str:
        """Handle user's response to result report confirmation."""
        lowered = text.lower()

        # Check for cancel: only standalone negative tokens cancel; a longer
        # reply that merely contains 不 (e.g. "不对，第3组是85") is a
        # correction, not a cancellation.
        if self._is_negative(text) or self._is_standalone_cancel(text):
            self.session.pending = ChatPending()
            return "Operation cancelled." if self.language == Language.EN else "操作已取消。"

        pending_data = self.session.pending.data
        action = TerminalAction(
            type="append_results",
            arguments={
                "rows": pending_data.get("rows") or [],
                "analysis": pending_data.get("analysis") or {},
                "new_objectives": pending_data.get("new_objectives") or [],
                "next_design_count": pending_data.get("next_design_count"),
                "update_plan": any(word in lowered for word in ("update", "更新", "update plan", "更新计划")),
            },
            source="pending",
        )
        result = self.workflow.execute_workflow(action, self._executor_for_terminal_action(action), force=True)
        if result.blocked:
            return result.content or ("审核未通过。" if self.language == Language.ZH else "Audit failed.")
        if self.session.pending.action == "confirm_result_report":
            self.session.pending = ChatPending()
        return result.content or ("已完成。" if self.language == Language.ZH else "Done.")

    def _handle_llm_plan(self, text: str) -> str | None:
        return self._handle_llm_plan_with_evidence(text, evidence_round_used=False)

    def _handle_llm_plan_with_evidence(self, text: str, evidence_round_used: bool = False, extra_context: dict[str, Any] | None = None) -> str | None:
        plan = self._plan(text, extra_context=extra_context)
        if plan is None:
            return None
        # Honesty gate, PRE-execution: in the evidence re-plan round, a plan
        # whose message/rows describe failed files without acknowledging the
        # failure is refused BEFORE anything can be staged.
        if evidence_round_used and getattr(self, "_active_evidence_selection", None) is not None:
            from .file_evidence import claimed_content_despite_failures

            selection = self._active_evidence_selection
            # Deterministic hard gate: when the round read NOTHING successfully
            # (every file failed), any re-planned write carrying data payloads
            # is refused outright — a pure-data fabrication cannot mention the
            # file it invented, so the textual check below cannot see it.
            if selection.has_failures and not selection.chunks and plan.arguments.get("rows"):
                append_event(
                    self.event_dir,
                    "error",
                    message="evidence hard gate: all requested files failed to parse; data-carrying write refused",
                )
                return (
                    "所请求的文件全部解析失败，没有任何真实文件内容可供写入。"
                    "基于不存在内容的实验数据写入已被拒绝；请提供可读的文件或直接粘贴数据。"
                    if self.language == Language.ZH
                    else "Every requested file failed to parse, so there is no real file content to write. "
                    "The data-carrying write has been rejected; provide a readable file or paste the data directly."
                )
            plan_text = " ".join([
                str(plan.arguments.get("message") or ""),
                json.dumps(plan.arguments.get("rows") or "", ensure_ascii=False, default=str)[:2000],
                json.dumps(plan.arguments.get("updates") or "", ensure_ascii=False, default=str)[:1000],
            ])
            if claimed_content_despite_failures(plan_text, selection):
                append_event(
                    self.event_dir,
                    "error",
                    message="evidence honesty rejection: re-planned write describes failed files without acknowledging the failure",
                )
                return (
                    "部分文件解析失败，无法确认文件内容；基于失败文件的写入已被拒绝。"
                    "请检查 file_read_failures，或直接粘贴关键数据。"
                    if self.language == Language.ZH
                    else "Some files failed to parse, so their contents cannot be verified; the write based on "
                    "failed files has been rejected. Check file_read_failures or paste the key data directly."
                )
        # File-evidence protocol: the planner requested files instead of
        # guessing their contents. This runs BEFORE the action-emptiness gate
        # (a first-stage plan is exactly {evidence_requests} with no action).
        # Budget: one evidence round per user message — a second request in
        # the re-plan round is explicitly refused, never executed ungrounded.
        if plan.evidence_requests:
            if evidence_round_used:
                return (
                    "本轮文件解析次数已用完，无法继续读取更多文件。请基于已提供的 file_content_chunks 继续，"
                    "或在下一轮消息中再指定其他文件。"
                    if self.language == Language.ZH
                    else "The file-parsing budget for this turn is exhausted. Continue from the provided "
                    "file_content_chunks, or specify other files in your next message."
                )
            return self._run_evidence_round(plan, text)
        if not self.workflow.actions_from_plan(plan):
            return None
        progress_msg = "Executing understood operation..." if self.language == Language.EN else "正在执行理解后的操作..."
        self._progress(progress_msg)
        result = self._execute_plan(plan)
        return result

    def _run_evidence_round(self, plan: IntentPlan, original_text: str) -> str | None:
        from .file_evidence import build_evidence_context, claimed_content_despite_failures
        from .settings import load_bo_api_settings

        self._progress("正在解析你指定的文件..." if self.language == Language.ZH else "Parsing the files you specified...")
        selection, evidence_context = build_evidence_context(
            plan.evidence_requests,
            settings=load_bo_api_settings(Path(__file__).resolve().parents[1]),
            base_dir=self.service.workspace,
        )
        append_event(
            self.event_dir,
            "observation",
            fact=(
                f"file evidence: {len(evidence_context.get('file_content_chunks', []))} chunks, "
                f"{len(evidence_context.get('file_read_failures', []))} failures"
            ),
        )
        followup_text = (
            "已根据你的请求解析文件，文件内容以 file_content_chunks 提供（这是唯一可信来源）。"
            "file_read_failures 中的文件解析失败——对它们的任何内容描述都必须明确承认失败，不得编造。"
            "现在基于真实文件内容输出最终 action。"
            if self.language == Language.ZH
            else "The files have been parsed and attached as file_content_chunks (the only trustworthy source). "
            "For entries in file_read_failures you must explicitly acknowledge the failure — never invent contents. "
            "Now emit the final action based on the real file contents."
        )
        # Drain a queued steering message at this fresh planner call (never
        # into repair rounds).
        if self._steering_queue:
            steering = self._steering_queue.pop(0)
            followup_text += (
                f"\n\n[用户在文件解析期间补充] {steering}"
                if self.language == Language.ZH
                else f"\n\n[User addendum while parsing] {steering}"
            )
            append_event(self.event_dir, "steering_drained", text=steering[:500])
        self._active_evidence_selection = selection
        try:
            followup = self._handle_llm_plan_with_evidence(
                followup_text, evidence_round_used=True, extra_context=evidence_context
            )
        finally:
            self._active_evidence_selection = None
        if followup is None:
            return None
        if claimed_content_despite_failures(followup, selection):
            # Second layer: the executor's reply text admits no failure while
            # failed files were involved — roll back anything staged.
            staged = self.session.pending.action
            if staged:
                self.session.pending = ChatPending()
                append_event(
                    self.event_dir,
                    "error",
                    message=f"evidence honesty rejection: staged '{staged}' rolled back (files failed to parse)",
                )
            return (
                "部分文件解析失败，无法确认你描述的文件内容，已撤销待执行的写入。"
                "请检查 file_read_failures，或直接粘贴关键数据。"
                if self.language == Language.ZH
                else "Some files failed to parse, so the described file contents cannot be verified; "
                "the staged write has been rolled back. Check file_read_failures or paste the key data directly."
            )
        return followup

    def _handle_no_action_reply(self, text: str) -> str:
        if self._is_affirmative(text) or self._is_short_continuation(text):
            return self._clarify_after_unbound_confirmation()
        if self.language == Language.EN:
            return (
                "I could not turn that into a concrete experiment-planning step yet. "
                "Please provide the optimization objective and the controllable factors, or paste existing experiment records."
            )
        return (
            "我还不能把这句话变成明确的实验优化步骤。"
            "请直接补充优化目标和可控因素，或粘贴已有实验记录。"
        )

    def _is_short_continuation(self, text: str) -> bool:
        normalized = " ".join(text.strip().lower().split())
        return normalized in {"继续", "继续吧", "下一步", "接着", "接着来", "然后呢", "continue", "next", "go on"}

    def _clarify_after_unbound_confirmation(self) -> str:
        project = self.session.project
        if project is not None:
            if self.language == Language.EN:
                return "Please tell me the specific next step: report results, generate the next batch, analyze data, or revise factors/objectives."
            return "请直接告诉我下一步要做什么：回填结果、生成下一批、分析数据，或修改因素/目标。"
        if self.language == Language.EN:
            return (
                "I do not have a confirmed setup to proceed with yet. Please provide:\n\n"
                "- The primary objective to optimize.\n"
                "- The controllable factors and their ranges or levels."
            )
        return (
            "当前还没有可继续执行的已确认方案。请补充：\n\n"
            "- 要优化的主要目标。\n"
            "- 可调因素及其范围或水平。"
        )

    def _apply_pending_setup_revision(self, text: str) -> str | None:
        plan = self._plan(text)
        if plan is None or plan.action == "none":
            return None
        workbook_name, factors, objectives = self._pending_setup_payload()

        if plan.action == "draft_setup":
            prefix = "I've updated the draft based on context:\n\n" if self.language == Language.EN else "我已经按上下文更新草案：\n\n"
            return self._stage_draft_setup(plan.arguments, prefix=prefix)

        if plan.action == "create_project":
            updated_name = self._workbook_name_from(plan.arguments)
            if not updated_name:
                return None
            self._set_pending_setup(updated_name, factors, objectives)
            prefix = "I've updated the workbook name:\n\n" if self.language == Language.EN else "我已经更新工作簿名称：\n\n"
            return prefix + self._render_setup_confirmation(updated_name, factors, objectives)

        if plan.action == "define_factors":
            factor_payload = plan.arguments.get("factors") or []
            if not factor_payload:
                return None
            factors = [FactorSpec.model_validate(item) for item in factor_payload]
            self._set_pending_setup(workbook_name, factors, objectives)
            prefix = "I've updated the factors:\n\n" if self.language == Language.EN else "我已经更新因素方案：\n\n"
            return prefix + self._render_setup_confirmation(workbook_name, factors, objectives)

        if plan.action in {"define_objective", "update_objective"}:
            objective_payload = plan.arguments.get("objective")
            if not objective_payload:
                return None
            # Replace the objectives list with the updated single objective
            updated_objective = ObjectiveSpec.model_validate(objective_payload)
            self._set_pending_setup(workbook_name, factors, [updated_objective])
            prefix = "I've updated the objective definition:\n\n" if self.language == Language.EN else "我已经更新目标定义：\n\n"
            return prefix + self._render_setup_confirmation(workbook_name, factors, [updated_objective])

        if plan.action == "update_factor":
            factor_name = str(plan.arguments.get("factor_name") or "")
            if not factor_name:
                return None
            factors = self._apply_factor_update_to_list(factors, factor_name, plan.arguments)
            self._set_pending_setup(workbook_name, factors, objectives)
            prefix = "I've updated the factor configuration:\n\n" if self.language == Language.EN else "我已经更新因素配置：\n\n"
            return prefix + self._render_setup_confirmation(workbook_name, factors, objectives)

        return None

    def _handle_llm_pending_count(self, text: str, expected_action: str, confirm_label: str) -> str | None:
        plan = self._plan(text)
        if plan is None or plan.action != expected_action:
            return None
        count = self._coerce_int(plan.arguments.get("count"))
        if count is None or count <= 0:
            return None
        self.session.pending.data["count"] = count
        if self.language == Language.EN:
            return f"Count updated to {count} runs. Reply `confirm` / `yes` to execute this {confirm_label}."
        else:
            return f"已将数量更新为 {count} 组。回复 `确认` / `可以` 执行这次{confirm_label}。"

    def _execute_plan(self, plan: IntentPlan) -> str | None:
        default_msg = "Processing request..." if self.language == Language.EN else "正在处理请求..."
        progress = ACTION_PROGRESS.get(plan.action, {}).get(self.language, default_msg)
        self._progress(progress)
        actions = self.workflow.actions_from_plan(plan)
        if not actions:
            return None
        if plan.action != "clarify":
            self._clear_reply_options()
        result = self.workflow.execute_actions(actions, self._executor_for_terminal_action)
        append_event(
            self.event_dir,
            "plan_executed",
            actions=[{"type": action.type} for action in actions],
            action_count=len(actions),
            status="pending" if result.pending is not None else "executed",
        )
        if result.pending is not None:
            self.session.pending = result.pending
        return result.content

    def _executor_for_terminal_action(self, action: TerminalAction) -> Callable[[dict[str, Any]], str | None] | None:
        method_name = f"_execute_{action.type}"
        handler = getattr(self, method_name, None)
        return handler if callable(handler) else None

    def executor_for_terminal_action(self, action: TerminalAction) -> Callable[[dict[str, Any]], str | None] | None:
        """Public accessor for external callers (e.g., the MCP server)."""
        return self._executor_for_terminal_action(action)

    def _execute_respond(self, arguments: dict[str, Any]) -> str | None:
        message = str(arguments.get("message") or "").strip()
        return message or None

    def _execute_clarify(self, arguments: dict[str, Any]) -> str | None:
        summary = str(arguments.get("summary") or "").strip()
        questions_raw = arguments.get("questions") or []
        examples_raw = arguments.get("examples") or []
        if isinstance(questions_raw, list):
            questions = [str(item).strip() for item in questions_raw if str(item).strip()]
        else:
            questions = []
        if isinstance(examples_raw, list):
            examples = [str(item).strip() for item in examples_raw if str(item).strip()]
        else:
            examples = []
        if not summary and not questions and not examples:
            return None

        lang = self.language
        # Quick heuristic: if summary contains English words, treat as English
        if summary:
            english_ratio = sum(1 for c in summary if c.isalpha() and ord(c) < 128) / max(len(summary), 1)
            if english_ratio > 0.5:
                lang = Language.EN

        parts = []
        if summary:
            parts.append(summary)
        if questions:
            if lang == Language.EN:
                parts.append("Please clarify:\n\n" + self._render_bulleted_list(questions))
            else:
                parts.append("请补充：\n\n" + self._render_bulleted_list(questions))
        if examples:
            self._set_reply_options(examples)
            if lang == Language.EN:
                parts.append("Quick choices (reply with a number, or copy/edit the text):\n\n" + self._render_numbered_list(examples))
            else:
                parts.append("快捷选项（可回复编号，或复制后修改）：\n\n" + self._render_numbered_list(examples))
        else:
            self._clear_reply_options()
        return "\n\n".join(parts)

    def _execute_draft_setup(self, arguments: dict[str, Any]) -> str | None:
        return self._stage_draft_setup(arguments)

    def _execute_create_project(self, arguments: dict[str, Any]) -> str | None:
        if arguments.get("_confirmed_by_harness") and arguments.get("factors") and arguments.get("objectives"):
            factors = [FactorSpec.model_validate(item) for item in arguments.get("factors") or []]
            objectives = [ObjectiveSpec.model_validate(item) for item in arguments.get("objectives") or []]
            workbook_name = self._workbook_name_from(arguments) or self._default_workbook_name(objectives)
            project = self.service.create_project(
                workbook_name=workbook_name,
                factors=factors,
                objectives=objectives,
                initial_batch_size=self._coerce_int(arguments.get("initial_batch_size")) or 8,
            )
            self.session.project = project
            self._refresh_task_title_from_project(project)
            success_msg = "Project created according to confirmed plan.\n\n" if self.language == Language.EN else "已按确认方案创建项目。\n\n"
            return success_msg + self._render_project_summary(project)
        workbook_name = self._workbook_name_from(arguments)
        if not workbook_name:
            return "我需要一个工作簿名，比如 `反应优化.xlsx`。"
        project = self.service.create_project(workbook_name=workbook_name, factors=[], objectives=[])
        self.session.project = project
        if not self._has_meaningful_title(self.session.title):
            self.session.title = project.name
        self.session.pending = ChatPending()
        return (
            f"已创建工作簿 `{project.workbook.name}`。\n\n"
            "继续补充因素和目标，或直接让我给起步方案。"
        )

    def _execute_load_project(self, arguments: dict[str, Any]) -> str | None:
        workbook_name = self._workbook_name_from(arguments)
        if not workbook_name:
            workbooks = ", ".join(self.service.list_workbooks()) or ("None" if self.language == Language.EN else "暂无")
            if self.language == Language.EN:
                return f"Please tell me which workbook to open. Available workbooks: {workbooks}."
            else:
                return f"请告诉我要打开哪个工作簿。当前可用工作簿：{workbooks}。"
        project = self.service.load_project(workbook_name)
        self.session.project = project
        self.session.pending = ChatPending()
        return self._render_project_summary(project)

    def _execute_define_factors(self, arguments: dict[str, Any]) -> str | None:
        project = self._require_project()
        if project is None:
            return "当前还没有项目。先描述需求，或直接让我给起步方案。"
        factor_payload = arguments.get("factors") or []
        if not factor_payload:
            return None
        project.factors = [FactorSpec.model_validate(item) for item in factor_payload]
        create_or_update_workbook(project)
        self.session.project = project
        self.session.pending = ChatPending()
        return self._render_project_summary(project)

    def _execute_define_objective(self, arguments: dict[str, Any]) -> str | None:
        project = self._require_project()
        if project is None:
            return "当前还没有项目。先描述需求，或直接让我给起步方案。"
        objectives = self._objective_specs_from_optional_payload(arguments)
        if not objectives:
            return None
        project.objectives = objectives
        create_or_update_workbook(project)
        self.session.project = project
        self._refresh_task_title_from_project(project)
        self.session.pending = ChatPending()
        return self._render_project_summary(project)

    def _execute_update_objective(self, arguments: dict[str, Any]) -> str | None:
        project = self._require_project()
        if project is None:
            return "当前还没有项目。先描述需求，或直接让我给起步方案。"
        objectives = self._objective_specs_from_optional_payload(arguments)
        if not objectives:
            return None
        for objective in objectives:
            project = self.service.update_objective(
                project,
                objective_name=objective.name,
                goal=objective.goal,
                target_value=objective.target_value,
                weight=objective.weight,
                desirability=objective.desirability,
            )
        self.session.project = project
        self._refresh_task_title_from_project(project)
        self.session.pending = ChatPending()
        return self._render_project_summary(project)

    def _execute_update_factor(self, arguments: dict[str, Any]) -> str | None:
        project = self._require_project()
        if project is None:
            return "当前还没有项目。先描述需求，或直接让我给起步方案。"
        factor_name = str(arguments.get("factor_name") or "")
        if not factor_name:
            return None
        project = self.service.update_factor(
            project,
            factor_name=factor_name,
            kind=arguments.get("kind"),
            min_value=self._coerce_float(arguments.get("min_value")),
            max_value=self._coerce_float(arguments.get("max_value")),
            step=self._coerce_float(arguments.get("step")),
            categories=self._coerce_categories(arguments.get("categories")),
            unit=str(arguments.get("unit")) if arguments.get("unit") is not None else None,
        )
        self.session.project = project
        self.session.pending = ChatPending()
        return self._render_project_summary(project)

    def _execute_initialize_design(self, arguments: dict[str, Any]) -> str | None:
        project = self._require_project()
        if project is None:
            return "当前还没有项目。请先确认实验方案。"
        count = self._coerce_int(arguments.get("count")) or project.initial_batch_size
        if arguments.get("_confirmed_by_harness"):
            score_profile = self.service.recommend_score_profile(project, self.language)
            project, changed_count = self.service.apply_score_profile(project, score_profile)
            project, frame = self.service.initialize_design(project, n_points=count)
            self.session.project = project
            score_notice = self._render_score_profile_result(score_profile, changed_count)
            parts = [
                f"{score_notice}",
                f"Generated {count} initialization runs. You can view batch {project.latest_batch_id} on the left." if self.language == Language.EN
                else f"已生成 {count} 组初始化实验计划，左侧可直接查看批次 {project.latest_batch_id}。",
            ]
            if len(project.objectives) > 1 and not frame.empty and "FuzzyScore" in frame.columns:
                parts.append(self._explain_multi_objective_scoring(project, frame))
            parts.append(
                "You can tell me the result of each experiment, skip some, complete with modified conditions, or generate more runs." if self.language == Language.EN
                else "你可以直接告诉我每个实验的结果，也可以说某个实验先跳过、按修改后的条件完成，或者继续再生成几组。"
            )
            return "\n\n".join(parts)
        score_profile = self.service.recommend_score_profile(project, self.language)
        self.session.pending = ChatPending(
            action="confirm_init_design",
            workbook=project.workbook.name,
            data={"count": count, "score_profile": score_profile},
        )
        mode = "初始化设计" if self.language == Language.ZH else "Initialize design"
        prefix = f"准备初始化 {count} 组设计：\n\n" if self.language == Language.ZH else f"Preparing to initialize {count} runs:\n\n"
        confirm_msg = "回复 `确认` 执行，或直接改组数。" if self.language == Language.ZH else "Reply `confirm` to execute, or change the number of runs."
        return (
            prefix +
            f"{self._render_execution_table(project, count, mode=mode)}\n\n" +
            f"{self._render_score_profile_confirmation(score_profile)}\n\n" +
            confirm_msg
        )

    def _execute_append_results(self, arguments: dict[str, Any]) -> str | None:
        project = self._require_project()
        rows = arguments.get("rows") or []
        confirmed_by_harness = bool(arguments.get("_confirmed_by_harness"))
        if rows and project is None:
            if not confirmed_by_harness:
                action = TerminalAction(type="append_results", arguments=dict(arguments), source="planner")
                result = self.workflow.execute_workflow(action, self._executor_for_terminal_action(action))
                if result.pending is not None:
                    self.session.pending = result.pending
                return result.content
            factors = self._factor_specs_from_optional_payload(arguments.get("factors"))
            objectives = self._objective_specs_from_optional_payload(arguments)
            workbook_name = self._workbook_name_from(arguments) or None
            project, frame, import_summary = self.service.bootstrap_project_from_rows(
                rows=rows,
                workbook_name=workbook_name,
                factors=factors,
                objectives=objectives,
                objective_names=self._objective_names_from_arguments(arguments),
            )
            self.session.project = project
            self._refresh_task_title_from_project(project)
            analysis = self.service.analyze_project(project, include_bo_diagnostics=False, language=self.language)
            quality = self.service.data_quality_report(project, language=self.language)
            if self.language == Language.EN:
                intro = (
                    f"Created `{project.workbook.name}` from pasted initial data and imported "
                    f"{import_summary.get('imported_rows') or len(frame)} records."
                )
            else:
                intro = (
                    f"已根据粘贴的初始数据创建 `{project.workbook.name}`，"
                    f"并导入 {import_summary.get('imported_rows') or len(frame)} 条记录。"
                )
            response = intro + "\n\n" + self._render_data_analysis(project, analysis, quality, scenario="pasted_initial_data")
            next_count = self._coerce_int(arguments.get("next_design_count") or arguments.get("count"))
            if next_count is not None and next_count > 0:
                score_profile = self.service.recommend_score_profile(project, self.language)
                self.session.pending = ChatPending(
                    action="confirm_generate_design",
                    workbook=project.workbook.name,
                    data={"count": next_count, "score_profile": score_profile},
                )
                if self.language == Language.EN:
                    response += (
                        f"\n\nI also prepared the next batch request for {next_count} runs. "
                        "Reply `confirm` to generate them, or tell me a different count."
                    )
                else:
                    response += (
                        f"\n\n我也已准备好下一批 {next_count} 组实验设计。"
                        "回复 `确认` 生成，或直接告诉我改成几组。"
                    )
            return response
        if confirmed_by_harness and rows:
            analysis = arguments.get("analysis") if isinstance(arguments.get("analysis"), dict) else self._analyze_result_report(project, rows)
            declared_objectives = self._objective_specs_from_optional_payload(
                {"objectives": arguments.get("new_objectives") or []}
            )
            if arguments.get("update_plan"):
                for item in analysis.get("modified", []):
                    trial = item.get("trial")
                    candidate_id = item.get("candidate_id")
                    reported = item.get("reported", {})
                    if trial is not None and candidate_id is not None and reported:
                        self.service.update_plan_experiment(
                            project,
                            trial=trial,
                            candidate_id=candidate_id,
                            updates=reported,
                        )
            rows_to_record = self._resolved_result_rows_from_analysis(rows, analysis)
            incomplete_new_rows = self._incomplete_unplanned_result_rows(project, rows_to_record, analysis)
            if incomplete_new_rows:
                if self.language == Language.EN:
                    return (
                        "I cannot record these as new experiments because the factor conditions are missing. "
                        "Please report them with a plan reference such as batch + candidate ID, or include the factor conditions."
                    )
                return (
                    "这些结果缺少因素条件，不能作为新增实验记录。"
                    "请用批次+计划编号/第几组来指向待做计划，或补充完整因素条件。"
                )
            project, frame = self.service.append_results(
                project,
                rows=rows_to_record,
                new_objectives=declared_objectives,
            )
            self.session.project = project
            completed = sum(1 for row in rows if str(row.get("status") or "completed").lower() != "skipped")
            skipped = sum(1 for row in rows if str(row.get("status") or "completed").lower() == "skipped")
            parts = []
            if arguments.get("update_plan"):
                modified_count = len(analysis.get("modified", []))
                if modified_count > 0:
                    parts.append(f"Updated {modified_count} plan(s) to match reported conditions" if self.language == Language.EN else f"已更新 {modified_count} 条计划为汇报条件")
            new_count = len(analysis.get("new_experiments", []))
            if new_count > 0:
                parts.append(f"Recorded {new_count} new experiment(s)" if self.language == Language.EN else f"已记录 {new_count} 条新增实验")
            if completed:
                matched_count = len(analysis.get("matched", []))
                if matched_count > 0:
                    parts.append(f"Recorded {matched_count} result(s) matching plans" if self.language == Language.EN else f"已记录 {matched_count} 条计划匹配的结果")
            if skipped:
                parts.append(f"Skipped {skipped} planned experiments" if self.language == Language.EN else f"已跳过 {skipped} 条计划实验")
            missing_count = len(analysis.get("missing", []))
            if missing_count > 0:
                parts.append(f"{missing_count} plan(s) still waiting for results" if self.language == Language.EN else f"还有 {missing_count} 条计划等待结果")
            if len(project.objectives) > 1:
                parts.append(self._explain_multi_objective_scoring(project, frame))
            next_count = self._coerce_int(arguments.get("next_design_count"))
            if next_count is not None and next_count > 0:
                score_profile = self.service.recommend_score_profile(project, self.language)
                self.session.pending = ChatPending(
                    action="confirm_generate_design",
                    workbook=project.workbook.name,
                    data={"count": next_count, "score_profile": score_profile},
                )
                parts.append(
                    f"Prepared the next batch request for {next_count} runs; reply `confirm` to generate them."
                    if self.language == Language.EN
                    else f"已准备好下一批 {next_count} 组实验设计；回复 `确认` 生成。"
                )
            if self.language == Language.EN:
                return ", ".join(parts) + f", {len(frame)} records in `EXPERIMENT`."
            return "，".join(parts) + f"，`EXPERIMENT` 当前共 {len(frame)} 条。"
        if rows:
            # Analyze the result report against pending plans before writing
            # anything: rows may match plans, modify conditions, or be new.
            declared_objectives = self._objective_specs_from_optional_payload(arguments)
            analysis = self._analyze_result_report(project, rows)

            if analysis["missing"] or analysis["modified"] or analysis["new_experiments"]:
                self.session.pending = ChatPending(
                    action="confirm_result_report",
                    workbook=project.workbook.name,
                    data={
                        "rows": rows,
                        "analysis": analysis,
                        "new_objectives": [objective.model_dump() for objective in declared_objectives or []],
                        "next_design_count": self._coerce_int(arguments.get("next_design_count") or arguments.get("count")),
                    }
                )
                return self._render_result_report_confirmation(analysis)

            action = TerminalAction(
                type="append_results",
                arguments={
                    "rows": rows,
                    "analysis": analysis,
                    "new_objectives": [objective.model_dump() for objective in declared_objectives or []],
                    "next_design_count": self._coerce_int(arguments.get("next_design_count") or arguments.get("count")),
                },
                source="planner",
            )
            result = self.workflow.execute_workflow(action, self._executor_for_terminal_action(action))
            if result.pending is not None:
                self.session.pending = result.pending
            return result.content
        source_excel = str(arguments.get("source_excel") or arguments.get("workbook_name") or "")
        if not source_excel:
            if project is None:
                return "当前还没有项目。可以先导入 Excel，或直接粘贴历史实验数据让我整理。"
            return None
        if not confirmed_by_harness:
            action = TerminalAction(type="append_results", arguments=dict(arguments), source="planner")
            result = self.workflow.execute_workflow(action, self._executor_for_terminal_action(action))
            if result.pending is not None:
                self.session.pending = result.pending
            return result.content
        project, frame = self.service.append_results(project, source_excel=source_excel)
        self.session.project = project
        if self.language == Language.EN:
            return f"Imported {len(frame)} records from `{source_excel}` to `EXPERIMENT`."
        else:
            return f"已从 `{source_excel}` 导入到 `EXPERIMENT`，共 {len(frame)} 条。"

    def _explain_multi_objective_scoring(self, project: ProjectState, frame: pd.DataFrame) -> str:
        """Generate a brief explanation of how FuzzyScore combines multiple objectives.

        Scoring pipeline:
        1. 每个目标通过 desirability 函数压缩到 [0,1]
        2. 应用自适应权重（基于数据分布方差和目标表现）
        3. 多个目标通过加权几何平均合并：FuzzyScore = (∏d_i^w_i)^(1/∑w_i)
        4. 权重通过 bounded = d + (1-d)×(1-weight) 应用
        """
        from .desirability import analyze_weight_distribution

        if len(project.objectives) <= 1:
            return ""

        objective_names = [obj.name for obj in project.objectives]

        # Get recent completed experiments with scores
        if "FuzzyScore" not in frame.columns:
            if self.language == Language.EN:
                return f"Multi-objective score combines {len(project.objectives)} objectives: {', '.join(objective_names)}"
            else:
                return f"多目标综合评分合并 {len(project.objectives)} 个目标：{', '.join(objective_names)}"

        scored = frame[pd.to_numeric(frame["FuzzyScore"], errors="coerce").notna()].copy()
        if scored.empty:
            if self.language == Language.EN:
                return f"Multi-objective score combines {len(project.objectives)} objectives: {', '.join(objective_names)}"
            else:
                return f"多目标综合评分合并 {len(project.objectives)} 个目标：{', '.join(objective_names)}"

        parts = []

        # Step 1: Explain the scoring method (operator-weighted geometric mean)
        if self.language == Language.EN:
            parts.append(f"Operator-weighted geometric aggregation of {len(project.objectives)} objectives")
        else:
            parts.append(f"按已确认目标权重几何聚合 {len(project.objectives)} 个目标")

        # Step 2: Get and display adaptive weights
        weight_analysis = analyze_weight_distribution(scored, project.objectives, self.language)
        adaptive_weights = weight_analysis.get("weights", {})
        weight_details = weight_analysis.get("details", [])

        if weight_details:
            weight_parts = []
            for detail in weight_details:
                name = detail["name"]
                weight = detail["weight"]

                if self.language == Language.EN:
                    weight_str = f"{name}(w={weight:.2f})"
                else:
                    weight_str = f"{name}(权={weight:.2f})"

                weight_parts.append(weight_str)

            if self.language == Language.EN:
                parts.append(f"Adaptive weights: {', '.join(weight_parts)}")
            else:
                parts.append(f"自适应权重：{', '.join(weight_parts)}")

        # Step 3: Show current batch statistics
        recent_scores = pd.to_numeric(scored["FuzzyScore"], errors="coerce")
        best_score = recent_scores.max()
        avg_score = recent_scores.mean()
        worst_score = recent_scores.min()

        if self.language == Language.EN:
            parts.append(f"Current batch: best {best_score:.3f} / avg {avg_score:.3f} / worst {worst_score:.3f}")
        else:
            parts.append(f"当前批次：最佳 {best_score:.3f} / 平均 {avg_score:.3f} / 最低 {worst_score:.3f}")

        # Step 4: Show best experiment details
        # idxmax() returns an index LABEL (not a position), so use .loc; using
        # .iloc here crashes when the scored rows do not start at position 0
        # (e.g. only the latest appended rows are scored).
        best_idx = recent_scores.idxmax()
        best_row = scored.loc[best_idx]

        obj_details = []
        for obj in project.objectives:
            if obj.name in best_row.index and pd.notna(best_row.get(obj.name)):
                val = best_row.get(obj.name)
                goal_symbol = "↑" if obj.goal == "maximize" else "↓"
                obj_details.append(f"{obj.name}{goal_symbol}={val}")

        if obj_details:
            if self.language == Language.EN:
                parts.append(f"Current best: {', '.join(obj_details)}")
            else:
                parts.append(f"当前最佳：{', '.join(obj_details)}")

        # Step 5: LLM智能分析和优化建议 — routed through
        # recommend_score_profile so auth/runtime errors surface as a visible
        # note (llm_weight_error) instead of propagating and breaking the
        # whole render.
        score_profile = self.service.recommend_score_profile(project, self.language)
        llm_analysis = score_profile.get("llm_weight_recommendation") or {}
        llm_weight_error = str(score_profile.get("llm_weight_error") or "").strip()
        if llm_weight_error:
            if self.language == Language.EN:
                parts.append(f"LLM weight analysis unavailable: {llm_weight_error}")
            else:
                parts.append(f"LLM权重分析不可用：{llm_weight_error}")

        llm_summary = llm_analysis.get("analysis", "")
        if llm_summary:
            parts.append(llm_summary)

        weight_adjustments = llm_analysis.get("weight_adjustments", [])
        if weight_adjustments:
            adj_parts = []
            for adj in weight_adjustments[:3]:  # 最多显示3个
                name = adj.get("objective", "")
                curr = adj.get("current_weight", 1.0)
                rec = adj.get("recommended_weight", 1.0)
                if self.language == Language.EN:
                    adj_str = f"{name}: {curr:.2f}→{rec:.2f}"
                else:
                    adj_str = f"{name}: {curr:.2f}→{rec:.2f}"
                adj_parts.append(adj_str)

            if self.language == Language.EN:
                parts.append(f"LLM weight suggestions: {', '.join(adj_parts)}")
            else:
                parts.append(f"LLM权重建议：{', '.join(adj_parts)}")

        optimization_focus = llm_analysis.get("optimization_focus", "")
        if optimization_focus:
            if self.language == Language.EN:
                parts.append(f"Next round focus: {optimization_focus}")
            else:
                parts.append(f"下轮优化重点：{optimization_focus}")

        if self.language == Language.EN:
            return "; ".join(parts)
        else:
            return "；".join(parts)

    def _execute_update_plan_experiment(self, arguments: dict[str, Any]) -> str | None:
        project = self._require_project()
        if project is None:
            return "No project yet. Please confirm the experiment plan first." if self.language == Language.EN else "当前还没有项目。请先确认实验方案。"
        focus = self.session.ui_state.get("workspace_focus") or {}
        if not isinstance(focus, dict):
            focus = {}
        experiment_id = self._coerce_int(arguments.get("experiment_id"))
        trial = self._coerce_int(arguments.get("Trial"))
        candidate_id = self._coerce_int(arguments.get("candidate_id"))
        if experiment_id is None:
            experiment_id = self._coerce_int(focus.get("experiment_id"))
        if trial is None:
            trial = self._coerce_int(focus.get("Trial"))
        if candidate_id is None:
            candidate_id = self._coerce_int(focus.get("candidate_id"))
        if experiment_id is None and (trial is None or candidate_id is None):
            return "请先在左侧聚焦一条待做计划，或者直接告诉我批次和编号。"
        updates = {
            factor.name: arguments.get(factor.name)
            for factor in project.factors
            if arguments.get(factor.name) not in (None, "")
        }
        if not updates:
            return "请直接告诉我要把这条计划的哪个条件改成什么。"
        comment = str(arguments.get("COMMENT") or "").strip() or "计划条件已调整"
        project = self.service.update_plan_experiment(
            project,
            experiment_id=experiment_id,
            trial=trial,
            candidate_id=candidate_id,
            updates=updates,
            comment=comment,
        )
        self.session.project = project
        self.session.pending = ChatPending()
        label = f"编号 {experiment_id}" if experiment_id is not None else f"批次 {trial}，编号 {candidate_id}"
        changed = "，".join(f"{key}={value}" for key, value in updates.items())
        return f"已更新{label}的实验计划：{changed}。做完后直接把结果发给我即可。"

    def _execute_generate_next_design(self, arguments: dict[str, Any]) -> str | None:
        project = self._require_project()
        if project is None:
            return "No project yet. Please confirm the experiment plan first." if self.language == Language.EN else "当前还没有项目。请先确认实验方案。"
        quality = self.service.data_quality_report(project, language=self.language)
        if not quality.get("model_ready", True):
            # Data cannot support modeling yet (confounded factors, missing
            # responses, or too few runs): defer BO and return a supplementary-
            # test plan instead of generating candidates.
            self.session.ui_state["workspace_mode"] = "bi"
            return self._render_data_quality_blocker(project, quality)
        count = self._coerce_int(arguments.get("count")) or project.initial_batch_size
        # Campaign strategist: per-batch objective-state directive. The
        # directive rides into the confirmation message, so the operator
        # approves the batch KNOWING the strategy that shaped it; a stop
        # recommendation is advisory text, never a silent halt.
        # Confirmation-identity: on the confirmed execution pass, reuse the
        # EXACT directive the operator approved (from the staged pending) —
        # the strategist is only consulted when no approved directive exists
        # (e.g. MCP direct calls). This makes "the operator approves the batch
        # knowing the strategy that shaped it" a guarantee, not best-effort,
        # and avoids a second strategist LLM call per confirmed batch.
        if arguments.get("_confirmed_by_harness") and isinstance(arguments.get("strategy"), dict):
            from .strategist import StrategyDirective, log_strategy_decision

            try:
                strategy = StrategyDirective.model_validate(arguments["strategy"])
                log_strategy_decision(self.service.event_dir, strategy)
            except Exception:
                strategy = None
        else:
            strategy = self.service.derive_campaign_strategy(project)
        strategy_dump = strategy.model_dump() if strategy is not None else None
        if arguments.get("_confirmed_by_harness"):
            if strategy_dump is None and isinstance(arguments.get("strategy"), dict):
                from .strategist import StrategyDirective, log_strategy_decision

                try:
                    strategy = StrategyDirective.model_validate(arguments["strategy"])
                    log_strategy_decision(self.service.event_dir, strategy)
                except Exception:
                    strategy = None
            score_profile = self.service.recommend_score_profile(project, self.language)
            project, changed_count = self.service.apply_score_profile(project, score_profile)
            if strategy is not None and strategy.range_extensions:
                project, ext_warnings = self.service.apply_range_extensions(project, strategy.range_extensions)
                for warning in ext_warnings[:3]:
                    append_event(self.event_dir, "strategy_warnings", warnings=[warning[:200]])
            project, frame = self.service.generate_next_design(project, n_points=count, strategy=strategy)
            self.session.project = project
            score_notice = self._render_score_profile_result(score_profile, changed_count)
            parts = [
                f"{score_notice}",
                (f"Generated {count} new experiment plans. You can view batch {project.latest_batch_id} on the left." if self.language == Language.EN
                 else f"已生成 {count} 组新实验计划，左侧可直接查看批次 {project.latest_batch_id}。")
            ]
            if strategy is not None:
                from .strategist import strategy_summary, validate_and_apply_weights

                adjusted_objectives, strategy_warnings = validate_and_apply_weights(strategy, project.objectives)
                parts.append(strategy_summary(strategy, adjusted_objectives))
                for warning in strategy_warnings[:4]:
                    parts.append(f"[strategy] {warning}")
                if strategy.stop_recommendation:
                    parts.append(
                        "Strategist recommends STOPPING after this batch (see rationale above). The decision is yours: "
                        "reply `skip pending designs` to stop, or keep reporting results to continue."
                        if self.language == Language.EN
                        else "策略器建议在本批之后停止优化（理由见上）。是否停止由你决定：回复 `跳过待做设计` 可止步，继续回填结果则继续优化。"
                    )
            if len(project.objectives) > 1 and not frame.empty and "FuzzyScore" in frame.columns:
                parts.append(self._explain_multi_objective_scoring(project, frame))
            parts.append(
                "You can report results one by one, mention modifications, or tell me which to skip." if self.language == Language.EN
                else "你可以逐条回填实验结果、说明某个实验有改动，或者告诉我先跳过哪一个。"
            )
            return "\n\n".join(parts)
        score_profile = self.service.recommend_score_profile(project, self.language)
        self.session.pending = ChatPending(
            action="confirm_generate_design",
            workbook=project.workbook.name,
            data={"count": count, "score_profile": score_profile, "strategy": strategy_dump},
        )
        mode = "生成新设计" if self.language == Language.ZH else "Generate Design"
        prefix = f"准备生成 {count} 组新设计：\n\n" if self.language == Language.ZH else f"Preparing to generate {count} new designs:\n\n"
        confirm_msg = "回复 `确认` 执行，或直接改组数。" if self.language == Language.ZH else "Reply `confirm` to execute, or change the number of runs."
        body = ""
        if strategy is not None:
            from .strategist import strategy_summary, validate_and_apply_weights, validate_range_extensions

            adjusted_objectives, strategy_warnings = validate_and_apply_weights(strategy, project.objectives)
            body = strategy_summary(strategy, adjusted_objectives) + "\n\n"
            for warning in strategy_warnings[:4]:
                body += f"[strategy] {warning}\n"
            if strategy.range_extensions:
                clamped_ext, ext_warnings = validate_range_extensions(
                    strategy, project.factors,
                    original_bounds=getattr(self.service, "_original_factor_bounds", None),
                )
                if clamped_ext:
                    ext_bits = [f"{name}: {box[0]:g}~{box[1]:g}" for name, box in clamped_ext.items()]
                    body += f"[strategy] 实际应用的范围（已钳制）: " + ", ".join(ext_bits) + "\n"
                else:
                    body += "[strategy] 范围扩展被拒绝（仅允许扩展且受累计上限约束）\n"
                for warning in ext_warnings[:3]:
                    body += f"[strategy] {warning}\n"
            if body.endswith("\n"):
                body += "\n"
        return (
            prefix +
            f"{self._render_execution_table(project, count, mode=mode)}\n\n" +
            f"{self._render_score_profile_confirmation(score_profile)}\n\n" +
            body +
            confirm_msg
        )

    def _execute_import_experiment_sheet(self, arguments: dict[str, Any]) -> str | None:
        source_excel = str(arguments.get("source_excel") or arguments.get("workbook_name") or "")
        if not source_excel:
            return "请告诉我要导入哪个 Excel 文件。"
        objective_names = arguments.get("objective_names")
        if objective_names is not None and not isinstance(objective_names, list):
            objective_names = None
        manual_column_map = arguments.get("manual_column_map")
        if manual_column_map is not None and not isinstance(manual_column_map, dict):
            manual_column_map = None
        if arguments.get("_confirmed_by_harness"):
            project = self._require_project()
            if project is None:
                return "当前还没有项目。请先创建或导入项目。"
            project, import_summary = self.service.import_experiment_sheet(
                project,
                source_excel=source_excel,
                objective_names=objective_names,
                manual_column_map=manual_column_map,
            )
            self.session.project = project
            mapped_columns = import_summary.get("mapped_columns") or []
            llm_mapped_columns = import_summary.get("llm_mapped_columns") or []
            if self.language == Language.EN:
                parts = [
                    f"Imported {import_summary.get('imported_rows') or 0} experimental data from `{source_excel}`.",
                    f"Auto-detected header at row {int(import_summary.get('header_row') or 0)}.",
                ]
                if mapped_columns:
                    parts.append(f"Mapped columns: {', '.join(str(item) for item in mapped_columns[:6])}")
                if llm_mapped_columns:
                    parts.append(f"LLM mapped: {', '.join(str(item) for item in llm_mapped_columns[:4])}")
            else:
                parts = [
                    f"已从 `{source_excel}` 导入 {import_summary.get('imported_rows') or 0} 条实验数据。",
                    f"自动识别表头第 {int(import_summary.get('header_row') or 0)} 行。",
                ]
                if mapped_columns:
                    parts.append(f"已匹配列：{', '.join(str(item) for item in mapped_columns[:6])}")
                if llm_mapped_columns:
                    parts.append(f"LLM 补充匹配：{', '.join(str(item) for item in llm_mapped_columns[:4])}")
            return " ".join(parts)
        project = self._require_project()
        if project is None:
            project, _frame, import_summary = self.service.bootstrap_project_from_import(
                source_excel,
                objective_names=objective_names,
            )
            self.session.project = project
            self._refresh_task_title_from_project(project)
        else:
            _frame, import_summary = self.service.preview_import_sheet(
                project,
                source_excel,
                objective_names=objective_names,
                manual_column_map=manual_column_map,
            )
        self.session.pending = ChatPending(
            action="confirm_import_experiment_sheet",
            workbook=project.workbook.name,
            data={
                "source_excel": source_excel,
                "objective_names": list(import_summary.get("objective_names") or objective_names or []),
                "import_summary": import_summary,
                "manual_column_map": {},
            },
        )
        mapped_columns = import_summary.get("mapped_columns") or []
        llm_mapped_columns = import_summary.get("llm_mapped_columns") or []
        ignored_columns = import_summary.get("ignored_columns") or []
        parts = [
            f"已为 `{source_excel}` 生成导入预览。",
            f"自动识别表头第 {int(import_summary.get('header_row') or 0)} 行。",
            f"检测到 {import_summary.get('detected_rows') or 0} 条候选数据。",
        ]
        if import_summary.get("bootstrapped_project"):
            parts.insert(0, f"已先根据导入表创建临时项目 `{project.workbook.name}`。")
        objective_labels = import_summary.get("objective_names") or []
        if objective_labels:
            parts.append(f"推断目标列：{', '.join(str(item) for item in objective_labels[:4])}")
        factor_labels = import_summary.get("inferred_factor_names") or []
        if factor_labels:
            parts.append(f"推断因素：{', '.join(str(item) for item in factor_labels[:6])}")
        if mapped_columns:
            parts.append(f"已匹配列：{', '.join(str(item) for item in mapped_columns[:6])}")
        if llm_mapped_columns:
            parts.append(f"LLM 补充匹配：{', '.join(str(item) for item in llm_mapped_columns[:4])}")
        if ignored_columns:
            parts.append(f"忽略列：{', '.join(str(item) for item in ignored_columns[:4])}")
        if not objective_labels:
            return (
                " ".join(parts)
                + '\n\n左侧已显示导入预览。还没有自动识别出目标结果列，请直接告诉我列对应关系，例如"收率% 这一列就是得率"。'
            )
        if import_summary.get("needs_mapping_clarification"):
            missing = import_summary.get("missing_critical_targets") or []
            missing_text = f"还缺关键列：{', '.join(str(item) for item in missing)}。" if missing else "还有关键列未识别。"
            return (
                " ".join(parts)
                + f'\n\n左侧已显示导入预览。{missing_text} 请直接告诉我列对应关系，例如"收率% 这一列就是得率"。'
            )
        return " ".join(parts) + "\n\n左侧已显示导入预览。回复 `确认` 正式写入，或回复 `取消` 放弃这次导入。"

    def _apply_pending_import_mapping(self, text: str) -> str | None:
        pending_data = self.session.pending.data
        import_summary = pending_data.get("import_summary") or {}
        if not isinstance(import_summary, dict):
            return None
        source_columns = [str(item) for item in import_summary.get("source_columns") or []]
        target_candidates = [
            str(item) for item in (import_summary.get("missing_critical_targets") or import_summary.get("critical_targets") or [])
        ]
        objective_names = pending_data.get("objective_names")
        if objective_names is not None and not isinstance(objective_names, list):
            objective_names = None
        if not source_columns:
            return None
        manual_column_map = dict(pending_data.get("manual_column_map") or {})
        try:
            updated_map = self.service.import_mapper.map_instruction(
                instruction=text,
                source_columns=source_columns,
                target_candidates=target_candidates,
            )
        except Exception as exc:
            if self.language == Language.EN:
                return f"The LLM could not turn that into a column mapping after 3 attempts: {exc}"
            return f"LLM 连续 3 次未能把这句话整理成列映射：{exc}"
        if not objective_names:
            inferred_targets = list(dict.fromkeys(str(target).strip() for target in updated_map.values() if str(target).strip()))
            if inferred_targets:
                pending_data["objective_names"] = inferred_targets
                objective_names = inferred_targets
        manual_column_map.update(updated_map)
        project = self._ensure_project()
        source_excel = str(pending_data.get("source_excel") or "")
        _frame, updated_summary = self.service.preview_import_sheet(
            project,
            source_excel=source_excel,
            objective_names=objective_names,
            manual_column_map=manual_column_map,
        )
        self.session.pending.data["manual_column_map"] = manual_column_map
        self.session.pending.data["import_summary"] = updated_summary
        mapped = ", ".join(f"{source}->{target}" for source, target in updated_map.items())
        if updated_summary.get("needs_mapping_clarification"):
            missing = updated_summary.get("missing_critical_targets") or []
            if self.language == Language.EN:
                missing_text = f"Missing: {', '.join(str(item) for item in missing)}." if missing else "Key columns not recognized."
                return f"Import mapping updated: {mapped}. Preview refreshed. {missing_text}"
            else:
                missing_text = f"还缺：{', '.join(str(item) for item in missing)}。" if missing else "还有关键列未识别。"
                return f"已更新导入映射：{mapped}。左侧预览已刷新。{missing_text}"
        if self.language == Language.EN:
            return f"Import mapping updated: {mapped}. Preview refreshed, reply `confirm` to finalize import."
        else:
            return f"已更新导入映射：{mapped}。左侧预览已刷新，回复 `确认` 即可正式导入。"

    def _execute_export(self, arguments: dict[str, Any]) -> str | None:
        project = self._require_project()
        if project is None:
            return "No project yet. Please confirm the experiment plan first." if self.language == Language.EN else "当前还没有项目。请先确认实验方案。"
        output_name = arguments.get("output_name")
        output = self.service.export_summary(project, output_name=str(output_name) if output_name else None)
        if self.language == Language.EN:
            return f"Summary exported: `{output}`."
        else:
            return f"已导出摘要：`{output}`。"

    def _execute_refresh_scores(self, arguments: dict[str, Any]) -> str | None:
        project = self._require_project()
        if project is None:
            return "No project yet. Please confirm the experiment plan first." if self.language == Language.EN else "当前还没有项目。请先确认实验方案。"
        frame = self.service.refresh_scores(project)
        if self.language == Language.EN:
            return f"Refreshed `FuzzyScore`, total {len(frame)} records."
        else:
            return f"已刷新 `FuzzyScore`，当前共 {len(frame)} 条记录。"

    def _execute_inspect_workbook(self, arguments: dict[str, Any]) -> str | None:
        project = self._require_project()
        if project is None:
            return "当前还没有项目。请先确认实验方案。"
        info = self.service.inspect_workbook(project)
        rows = [
            ["META", "1"],
            ["PARAMETERS", str(len(project.factors))],
            ["CATEGORY", str(sum(len(factor.categories) for factor in project.factors if factor.kind == "categorical"))],
            ["MPO", str(len(project.objectives))],
            ["EXPERIMENT", str(info["experiment_rows"])],
            ["DESIGNS", str(info["design_rows"])],
        ]
        return (
            f"工作簿 `{project.workbook.name}` 结构如下：\n\n"
            f"{self._render_key_value_rows(rows)}\n\n"
            f"EXPERIMENT 列：{', '.join(info['experiment_columns'])}"
        )

    def _execute_status(self, arguments: dict[str, Any]) -> str | None:
        project = self._require_project()
        if project is None:
            return "当前还没有项目。先描述需求，或直接让我给起步方案。"
        return self._render_status_dashboard(project)

    def _execute_analyze_data(self, arguments: dict[str, Any]) -> str | None:
        project = self._require_project()
        if project is None:
            return "当前还没有项目。可以先导入 Excel，或描述实验目标和因素让我建立项目。"
        self.session.ui_state["workspace_mode"] = "bi"
        analysis = self.service.analyze_project(project, include_bo_diagnostics=False, language=self.language)
        quality = self.service.data_quality_report(project, language=self.language)
        return self._render_data_analysis(project, analysis, quality)

    def _execute_view_experiments(self, arguments: dict[str, Any]) -> str | None:
        project = self._require_project()
        if project is None:
            return "当前还没有项目。先确认方案或导入实验数据。"
        view = self._experiment_view_state()
        scope = str(arguments.get("scope") or "").strip().lower()
        if scope in {"recent", "completed", "pending", "best", "all"}:
            view["scope"] = scope
            view["page"] = 1
        sort = str(arguments.get("sort") or "").strip().lower()
        if sort in {"recent", "score_desc", "trial_asc"}:
            view["sort"] = sort
            view["page"] = 1
        if arguments.get("clear_batch"):
            view["batch_id"] = None
            view["page"] = 1
        batch_id = self._coerce_int(arguments.get("batch_id"))
        if batch_id is not None:
            view["batch_id"] = batch_id
            view["page"] = 1
        page = self._coerce_int(arguments.get("page"))
        if page is not None and page > 0:
            view["page"] = page
        page_delta = self._coerce_int(arguments.get("page_delta"))
        if page_delta is not None:
            view["page"] = max(1, int(view.get("page") or 1) + page_delta)
        self.session.ui_state["experiment_view"] = view
        if self.language == Language.EN:
            return f"Updated experiment view: {self._experiment_view_label(view)}."
        else:
            return f"已更新左侧实验视图：{self._experiment_view_label(view)}。"

    def _execute_skip_pending_designs(self, arguments: dict[str, Any]) -> str | None:
        """Skip pending designs for specified experiment IDs or all in a batch."""
        project = self._require_project()
        if project is None:
            return "No project found." if self.language == Language.EN else "未找到项目。"

        from .storage import read_sheet, SHEET_DESIGNS

        candidate_ids = arguments.get("candidate_ids") or []

        batch_id = self._coerce_int(arguments.get("batch_id"))
        if batch_id is None:
            batch_id = project.latest_batch_id

        if batch_id is None:
            return "No batch specified and no current batch found." if self.language == Language.EN else "未指定批次且未找到当前批次。"

        design_df = read_sheet(project, SHEET_DESIGNS)
        if design_df.empty:
            return "No pending designs found." if self.language == Language.EN else "没有找到待做设计。"

        pending_mask = (
            pd.to_numeric(design_df["Trial"], errors="coerce").eq(batch_id) &
            design_df["status"].isin(["proposed", "initialized", "Pending", "待做", "pending"])
        )

        if candidate_ids:
            id_mask = pd.to_numeric(design_df["candidate_id"], errors="coerce").isin(candidate_ids)
            pending_mask = pending_mask & id_mask

        pending_count = pending_mask.sum()
        if pending_count == 0:
            if candidate_ids:
                if self.language == Language.EN:
                    return f"No pending designs found with IDs {candidate_ids} in batch {batch_id}."
                else:
                    return f"批次 {batch_id} 中未找到编号为 {candidate_ids} 的待做设计。"
            else:
                if self.language == Language.EN:
                    return f"No pending designs found in batch {batch_id}."
                else:
                    return f"批次 {batch_id} 中没有待做设计。"

        skip_rows = []
        skipped_ids = []
        for idx in design_df[pending_mask].index:
            row = design_df.loc[idx].to_dict()
            cid = row.get("candidate_id")
            skipped_ids.append(cid)
            skip_rows.append({
                "Trial": row.get("Trial"),
                "candidate_id": cid,
                "experiment_id": row.get("experiment_id"),
                "status": "skipped",
                "COMMENT": arguments.get("comment") or ("Skipped by user" if self.language == Language.EN else "用户跳过"),
            })

        project, frame = self.service.append_results(project, rows=skip_rows)
        self.session.project = project

        if self.language == Language.EN:
            if candidate_ids:
                return f"Skipped experiment(s) {skipped_ids} in batch {batch_id}."
            else:
                return f"Skipped {pending_count} pending design(s) in batch {batch_id}."
        else:
            if candidate_ids:
                return f"已跳过批次 {batch_id} 中的实验 {skipped_ids}。"
            else:
                return f"已跳过批次 {batch_id} 中的 {pending_count} 条待做设计。"

    def _stage_draft_setup(self, arguments: dict[str, Any], prefix: str = "") -> str | None:
        factor_payload = arguments.get("factors") or []
        objective_payload = arguments.get("objective")
        objectives_payload = arguments.get("objectives")

        if not factor_payload:
            return None
        if not objective_payload and not objectives_payload:
            return None

        factors = [FactorSpec.model_validate(item) for item in factor_payload]

        # Handle both single objective and multiple objectives
        if objectives_payload:
            objectives = [ObjectiveSpec.model_validate(item) for item in objectives_payload]
        else:
            objectives = [ObjectiveSpec.model_validate(objective_payload)]

        workbook_name = self._workbook_name_from(arguments) or self._default_workbook_name(objectives)
        self._set_pending_setup(workbook_name, factors, objectives)
        if not self._has_meaningful_title(self.session.title):
            self.session.title = self._default_title(objectives)
        return prefix + self._render_setup_confirmation(workbook_name, factors, objectives)

    def _set_pending_setup(self, workbook_name: str, factors: list[FactorSpec], objectives: list[ObjectiveSpec]) -> None:
        self.session.pending = ChatPending(
            action="confirm_setup",
            workbook=workbook_name,
            data={
                "workbook_name": workbook_name,
                "factors": [factor.model_dump() for factor in factors],
                "objectives": [objective.model_dump() for objective in objectives],
            },
        )

    def _pending_setup_payload(self) -> tuple[str, list[FactorSpec], list[ObjectiveSpec]]:
        data = self.session.pending.data
        workbook_name = str(data.get("workbook_name") or "实验设计.xlsx")
        factors = [FactorSpec.model_validate(item) for item in data.get("factors", [])]

        # Pending data may store either a single ``objective`` or an
        # ``objectives`` list; normalize to the list form.
        objectives_data = data.get("objectives")
        if objectives_data:
            objectives = [ObjectiveSpec.model_validate(item) for item in objectives_data]
        else:
            objective = ObjectiveSpec.model_validate(data.get("objective") or {})
            objectives = [objective]

        return workbook_name, factors, objectives

    def _factor_specs_from_optional_payload(self, payload: object) -> list[FactorSpec] | None:
        if not isinstance(payload, list):
            return None
        factors: list[FactorSpec] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                factors.append(FactorSpec.model_validate(item))
            except Exception:
                continue
        return factors or None

    def _objective_specs_from_optional_payload(self, arguments: dict[str, Any]) -> list[ObjectiveSpec] | None:
        payload = arguments.get("objectives")
        if isinstance(payload, list):
            objectives: list[ObjectiveSpec] = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                try:
                    objectives.append(ObjectiveSpec.model_validate(item))
                except Exception:
                    continue
            if objectives:
                return objectives
        single = arguments.get("objective")
        if isinstance(single, dict):
            try:
                return [ObjectiveSpec.model_validate(single)]
            except Exception:
                return None
        if isinstance(arguments.get("name"), str) and arguments.get("goal") is not None:
            try:
                return [ObjectiveSpec.model_validate(arguments)]
            except Exception:
                return None
        return None

    def _objective_names_from_arguments(self, arguments: dict[str, Any]) -> list[str] | None:
        raw = arguments.get("objective_names")
        if isinstance(raw, list):
            names = [str(item).strip() for item in raw if str(item).strip()]
            if names:
                return names
        objectives = self._objective_specs_from_optional_payload(arguments)
        if objectives:
            return [objective.name for objective in objectives]
        return None

    def _apply_factor_update_to_list(
        self,
        factors: list[FactorSpec],
        factor_name: str,
        arguments: dict[str, Any],
    ) -> list[FactorSpec]:
        updated_factors: list[FactorSpec] = []
        matched = False
        for factor in factors:
            if factor.name != factor_name:
                updated_factors.append(factor)
                continue
            matched = True
            updated_factors.append(self._updated_factor(factor, factor_name, arguments))
        if not matched:
            updated_factors.append(self._updated_factor(None, factor_name, arguments))
        return updated_factors

    def _updated_factor(
        self,
        factor: FactorSpec | None,
        factor_name: str,
        arguments: dict[str, Any],
    ) -> FactorSpec:
        updated = factor.model_copy(deep=True) if factor is not None else FactorSpec(name=factor_name, kind="continuous")
        if arguments.get("kind") is not None:
            updated.kind = normalize_factor_kind(arguments["kind"], categories=arguments.get("categories"))
        if arguments.get("min_value") is not None:
            updated.min_value = self._coerce_float(arguments["min_value"])
        if arguments.get("max_value") is not None:
            updated.max_value = self._coerce_float(arguments["max_value"])
        if arguments.get("step") is not None:
            updated.step = self._coerce_float(arguments["step"])
        if arguments.get("categories") is not None:
            updated.categories = self._coerce_categories(arguments["categories"]) or []
        if arguments.get("unit") is not None:
            updated.unit = str(arguments["unit"])
        if updated.kind == "categorical":
            updated.min_value = None
            updated.max_value = None
            updated.step = None
        return updated

    def _plan(self, text: str, extra_context: dict[str, Any] | None = None) -> IntentPlan | None:
        if not self.intent_planner.available:
            return None
        prior_history = self.session.history[:-1] if self.session.history and self.session.history[-1].role == "user" else self.session.history
        detected_lang = _detect_language_with_context(text, prior_history, default_language="zh" if self.language == Language.ZH else "en")
        return self.intent_planner.plan(
            message=text,
            project=self.session.project,
            pending=self.session.pending,
            history=prior_history,
            ui_state=self.session.ui_state,
            trace_dir=task_llm_trace_dir(self.session.task_dir) if self.session.task_dir else None,
            task_id=self.session.task_id,
            language=detected_lang,
            extra_context=extra_context,
        )

    def _experiment_view_state(self) -> dict[str, Any]:
        state = dict(self.session.ui_state.get("experiment_view") or {})
        state["scope"] = str(state.get("scope") or "recent").lower()
        state["sort"] = str(state.get("sort") or "recent").lower()
        state["page"] = max(self._coerce_int(state.get("page")) or 1, 1)
        state["batch_id"] = self._coerce_int(state.get("batch_id"))
        return state

    def _experiment_view_label(self, view: dict[str, Any]) -> str:
        if self.language == Language.EN:
            scope_map = {
                "recent": "Recently completed",
                "completed": "Completed",
                "pending": "Pending",
                "best": "Best first",
                "all": "All experiments",
            }
            sort_map = {
                "recent": "Sort by recent",
                "score_desc": "Sort by score",
                "trial_asc": "Sort by batch",
            }
            default_scope = "Experiment records"
            default_sort = "Sort by recent"
            batch_prefix = "Batch "
        else:
            scope_map = {
                "recent": "最近完成",
                "completed": "已完成",
                "pending": "待执行",
                "best": "最佳优先",
                "all": "全部实验",
            }
            sort_map = {
                "recent": "按最近排序",
                "score_desc": "按评分排序",
                "trial_asc": "按批次正序",
            }
            default_scope = "实验记录"
            default_sort = "按最近排序"
            batch_prefix = "批次 "
        parts = [scope_map.get(str(view.get("scope") or ""), default_scope)]
        if view.get("batch_id") is not None:
            parts.append(f"{batch_prefix}{view['batch_id']}")
        parts.append(sort_map.get(str(view.get("sort") or ""), default_sort))
        return "，".join(parts)

    def _ensure_project(self) -> ProjectState:
        if self.session.project is None:
            raise ValueError("当前还没有项目。")
        return self.session.project

    def _require_project(self) -> ProjectState | None:
        return self.session.project

    def _planner_required_reply(self) -> str:
        if self.language == Language.ZH:
            return (
                "当前启用了基于上下文的自然语言规划，但还没有可用的模型配置。\n"
                "请在 `.env` 中设置 `API_TOKEN`、`BASE_URL`、`MODEL_NAME` 后再使用聊天规划。"
            )
        return (
            "Context-based natural language planning is enabled, but no model configuration is available.\n"
            "Please set `API_TOKEN`, `BASE_URL`, and `MODEL_NAME` in `.env` before using chat planning."
        )

    def _is_affirmative(self, text: str) -> bool:
        normalized = text.strip().lower()
        return normalized in self.affirmative_words or "按你写的" in text

    def _is_negative(self, text: str) -> bool:
        return text.strip().lower() in self.negative_words

    def _is_standalone_cancel(self, text: str) -> bool:
        normalized = text.strip().lower()
        return bool(
            re.fullmatch(r"(?:不(?:要|用)?|取消|算了|cancel|no|n)[\s。.！!~？?了吧]*", normalized)
        )

    def _confirmation_decision(self, text: str) -> str | None:
        normalized = " ".join(text.strip().lower().split())
        if normalized in APPLY_ALL_WORDS:
            return CONFIRMATION_APPLY_ALL
        if self._is_negative(text):
            return CONFIRMATION_CANCEL
        if self._is_affirmative(text):
            return CONFIRMATION_CONFIRM
        return None

    def _workbook_name_from(self, arguments: dict[str, Any]) -> str | None:
        value = arguments.get("workbook_name") or arguments.get("path")
        if value in (None, ""):
            return None
        workbook_name = str(value).strip()
        return workbook_name if workbook_name.endswith(".xlsx") else f"{workbook_name}.xlsx"

    def _default_workbook_name(self, objectives: list[ObjectiveSpec]) -> str:
        objective_names = " ".join(obj.name for obj in objectives)
        if self.language == Language.EN:
            if any(token in objective_names for token in {"yield", "production", "reaction", "selectivity", "purity", "fastness", "uniformity"}):
                return "Chemical_experiment.xlsx"
            return "Experimental_design.xlsx"
        else:
            if any(token in objective_names for token in {"产率", "收率", "反应", "选择性", "纯度", "得率"}):
                return "化学实验.xlsx"
            return "实验设计.xlsx"

    def _default_title(self, objectives: list[ObjectiveSpec]) -> str:
        if self.language == Language.EN:
            if len(objectives) == 1:
                return f"{objectives[0].name} optimization"
            return "Multi-objective optimization: " + "+".join(obj.name for obj in objectives)
        else:
            if len(objectives) == 1:
                return f"{objectives[0].name}优化"
            return "多目标优化：" + "+".join(obj.name for obj in objectives)

    def _render_setup_confirmation(
        self,
        workbook_name: str,
        factors: list[FactorSpec],
        objectives: list[ObjectiveSpec],
    ) -> str:
        if self.language == Language.EN:
            return (
                "Pending plan:\n\n"
                f"{self._render_setup_table(workbook_name, factors, objectives)}\n\n"
                "Reply `confirm` to execute, or describe what you want to change."
            )
        return (
            "待确认方案：\n\n"
            f"{self._render_setup_table(workbook_name, factors, objectives)}\n\n"
            "回复 `确认` 执行，或直接说要改哪一项。"
        )

    def _render_project_summary(self, project: ProjectState) -> str:
        objectives = project.objectives if project.objectives else [ObjectiveSpec(name="Undefined" if self.language == Language.EN else "未定义", goal="maximize")]
        if self.language == Language.EN:
            header = (
                f"Task ID: `{display_task_id(self.session.task_id)}`\n"
                f"Current project: `{project.name}`\n"
                f"Workbook: `{project.workbook.name}`\n"
            )
        else:
            header = (
                f"任务 ID：`{display_task_id(self.session.task_id)}`\n"
                f"当前项目：`{project.name}`\n"
                f"工作簿：`{project.workbook.name}`\n"
            )
        return header + "\n" + self._render_setup_table(project.workbook.name, project.factors, objectives)

    def _render_status_dashboard(self, project: ProjectState) -> str:
        analysis = self.service.analyze_project(project)
        objective_payload = analysis.get("primary_objective") or {}
        objective_name = str(objective_payload.get("name") or "目标")
        goal_text = self._goal_cn(str(objective_payload.get("goal") or "maximize"), objective_payload.get("target_value"))

        if len(project.objectives) > 1:
            target_desc = f"{len(project.objectives)} 个目标：" + "+".join(obj.name for obj in project.objectives)
        else:
            target_desc = f"{objective_name} | {goal_text}"

        summary_rows = [
            ("任务 ID", display_task_id(self.session.task_id)),
            ("当前项目", str(analysis.get("project_name") or project.name)),
            ("工作簿", str(analysis.get("workbook_name") or project.workbook.name)),
            ("目标", target_desc),
            ("已完成实验", str(analysis.get("completed_count") or 0)),
            ("待测建议", str(analysis.get("suggested_count") or 0)),
            ("设计总数", str(analysis.get("design_count") or 0)),
            ("当前批次", str(analysis.get("latest_batch_id") or 0)),
        ]

        best_section = self._render_best_experiment(analysis.get("best_experiment"), project, objective_name)
        trend_lines = []
        batch_counts = analysis.get("batch_counts") or []
        if batch_counts:
            trend_lines.append(f"批次进度：{self._sparkline(batch_counts)}  {', '.join(str(v) for v in batch_counts)}")
        score_trend = analysis.get("score_trend") or []
        if score_trend:
            trend_lines.append(f"最佳评分：{self._sparkline(score_trend)}  {self._trend_endpoints(score_trend)}")
        objective_trend = analysis.get("objective_trend") or []
        if objective_trend:
            trend_lines.append(f"{objective_name}趋势：{self._sparkline(objective_trend)}  {self._trend_endpoints(objective_trend)}")

        factor_ranges = analysis.get("factor_ranges") or []
        factor_lines = [
            f"{index}. {item.get('name')}：{item.get('summary')}"
            for index, item in enumerate(factor_ranges[:6], start=1)
        ]

        parts = [
            self._render_key_value_rows(summary_rows),
            "当前最优工艺：\n" + best_section,
        ]
        if trend_lines:
            parts.append("趋势：\n" + "\n".join(f"- {line}" for line in trend_lines))
        if factor_lines:
            parts.append("已探索范围：\n" + "\n".join(factor_lines))

        if len(project.objectives) > 1:
            experiments = self.service.refresh_scores(project)
            if not experiments.empty and "FuzzyScore" in experiments.columns:
                parts.append(self._explain_multi_objective_scoring(project, experiments))

        return "\n\n".join(parts)

    def _render_data_quality_blocker(self, project: ProjectState, quality: dict[str, Any]) -> str:
        """Render the deferral message returned when data cannot yet support BO
        (confounded factors / missing responses / too few runs): a structured
        diagnosis plus a minimal supplementary-test plan, instead of candidates."""
        is_en = self.language == Language.EN
        parts: list[str] = []
        parts.append(
            "Supplementary tests needed before model-driven optimization"
            if is_en
            else "数据不足以直接建模，先补测再生成下一批"
        )
        parts.append(
            "The current data cannot yet support Bayesian-optimization candidates. Resolve the issues below, then retry the next batch."
            if is_en
            else "当前数据尚不能支持贝叶斯优化候选的生成。请先按下列诊断补测，补齐后再重新生成下一批。"
        )
        plan: list[str] = []
        for blocker in quality.get("blockers") or []:
            kind = blocker.get("kind")
            if kind == "confounded_factors":
                for group in blocker.get("groups") or []:
                    factors = "、".join(str(f) for f in group.get("factors") or [])
                    pivot = str(group.get("pivot") or (group.get("factors") or ["factor"])[0])
                    patterns = []
                    for entry in group.get("pattern") or []:
                        locks = "、".join(f"{k}={'/'.join(v)}" for k, v in (entry.get("locked") or {}).items())
                        patterns.append(f"{entry.get('value')} → {locks}")
                    joined = "; ".join(patterns)
                    if is_en:
                        plan.append(f"Break the confounding of {factors}: each {pivot} currently locks the others ({joined}); cross these factors in supplementary runs, and add untreated / treatment controls.")
                    else:
                        plan.append(f"解除因素混杂 {factors}：当前每取一种 {pivot} 只对应（{joined}）；补测时让这些因素交叉取值，并补 untreated / treatment 对照。")
            elif kind == "missing_responses":
                objectives = "、".join(str(o) for o in blocker.get("objectives") or [])
                if is_en:
                    plan.append(f"Measure the missing responses: {objectives} (add replicates for the key responses).")
                else:
                    plan.append(f"补测缺失响应：{objectives}（建议同时增加关键响应的重复）。")
            elif kind == "too_few_completed":
                if is_en:
                    plan.append(f"Collect at least {blocker.get('need')} completed runs (now {blocker.get('have')}).")
                else:
                    plan.append(f"补齐至少 {blocker.get('need')} 条完成实验（当前 {blocker.get('have')} 条）。")
        if plan:
            parts.append(("Supplementary plan:\n" if is_en else "补测计划：\n") + "\n".join(f"- {item}" for item in plan))
        parts.append(
            "You can review the data-quality analysis first, or tell me to retry generation after the supplementary tests."
            if is_en
            else "可以先查看数据质量分析，或补测后让我重新生成下一批。"
        )
        return "\n\n".join(parts)

    def _render_data_analysis(
        self,
        project: ProjectState,
        analysis: dict[str, Any],
        quality: dict[str, Any],
        scenario: str = "analysis",
    ) -> str:
        is_en = self.language == Language.EN
        llm_summary = self.service.llm_data_analysis_summary(
            project,
            analysis=analysis,
            quality=quality,
            language=self.language,
            scenario=scenario,
        )
        objective_payload = analysis.get("primary_objective") or {}
        objective_name = str(objective_payload.get("name") or ("objective" if is_en else "目标"))
        best_row = analysis.get("best_experiment") or {}
        overview_rows = [
            ("Project" if is_en else "项目", str(analysis.get("project_name") or project.name)),
            ("Workbook" if is_en else "工作簿", str(analysis.get("workbook_name") or project.workbook.name)),
            ("Completed" if is_en else "已完成", str(quality.get("completed_count") or 0)),
            ("Pending" if is_en else "待做", str(quality.get("pending_count") or 0)),
            ("Total records" if is_en else "总记录", str(quality.get("row_count") or 0)),
            ("Design rows" if is_en else "设计记录", str(quality.get("design_count") or 0)),
        ]
        if best_row:
            best_label = (
                f"Batch {self._format_metric(best_row.get('Trial'))}, ID {self._format_metric(best_row.get('candidate_id'))}"
                if is_en
                else f"批次 {self._format_metric(best_row.get('Trial'))}，编号 {self._format_metric(best_row.get('candidate_id'))}"
            )
            overview_rows.append(("Best record" if is_en else "当前最优", best_label))
            overview_rows.append(("Best score" if is_en else "最佳评分", self._format_metric(best_row.get("FuzzyScore"))))
            if objective_name in best_row:
                overview_rows.append((f"Best {objective_name}" if is_en else f"最佳{objective_name}", self._format_metric(best_row.get(objective_name))))

        objective_lines = []
        for item in quality.get("objective_coverage") or []:
            name = str(item.get("name") or "-")
            count = self._format_metric(item.get("count"))
            missing = self._format_metric(item.get("missing"))
            summary = str(item.get("summary") or "-")
            label = "valid/missing" if is_en else "有效/缺失"
            objective_lines.append(f"- {name}: {label} {count}/{missing}; {summary}")

        factor_lines = []
        for item in (quality.get("factor_coverage") or [])[:6]:
            name = str(item.get("name") or "-")
            summary = str(item.get("summary") or "-")
            ratio = item.get("coverage_ratio")
            if isinstance(ratio, float):
                summary = f"{summary}; coverage {ratio:.0%}" if is_en else f"{summary}；覆盖 {ratio:.0%}"
            factor_lines.append(f"- {name}: {summary}")

        missing_lines = []
        for item in (quality.get("missing_columns") or [])[:5]:
            missing_lines.append(f"- {item.get('name')}: {item.get('missing')} ({float(item.get('ratio') or 0):.0%})")

        trend_lines = []
        score_trend = analysis.get("score_trend") or []
        if score_trend:
            trend_lines.append(
                f"- {'Best score' if is_en else '最佳评分'}: {self._sparkline(score_trend)}  {self._trend_endpoints(score_trend)}"
            )
        objective_trend = analysis.get("objective_trend") or []
        if objective_trend:
            trend_lines.append(
                f"- {objective_name}: {self._sparkline(objective_trend)}  {self._trend_endpoints(objective_trend)}"
            )

        recommendation_lines = [f"- {item}" for item in (quality.get("recommendations") or [])[:4]]

        parts = [
            "Data analysis" if is_en else "数据分析",
            str(llm_summary.get("message") or "").strip(),
            self._render_key_value_rows(overview_rows),
        ]
        highlights = [str(item) for item in llm_summary.get("highlights") or [] if str(item).strip()]
        if highlights:
            parts.append(("LLM highlights:\n" if is_en else "LLM 关键发现：\n") + "\n".join(f"- {item}" for item in highlights))
        if objective_lines:
            parts.append(("Objective coverage:\n" if is_en else "目标覆盖：\n") + "\n".join(objective_lines))
        if trend_lines:
            parts.append(("Trends:\n" if is_en else "趋势：\n") + "\n".join(trend_lines))
        if factor_lines:
            parts.append(("Factor coverage:\n" if is_en else "因素探索：\n") + "\n".join(factor_lines))
        if missing_lines:
            parts.append(("Missing values:\n" if is_en else "缺失值：\n") + "\n".join(missing_lines))
        if int(quality.get("duplicate_plan_keys") or 0) > 0:
            duplicate_text = (
                f"Duplicate Trial/candidate_id rows: {quality.get('duplicate_plan_keys')}"
                if is_en
                else f"发现重复 Trial/candidate_id 记录：{quality.get('duplicate_plan_keys')}"
            )
            parts.append(duplicate_text)
        if recommendation_lines:
            parts.append(("Recommendations:\n" if is_en else "下一步建议：\n") + "\n".join(recommendation_lines))
        llm_next_actions = [str(item) for item in llm_summary.get("next_actions") or [] if str(item).strip()]
        if llm_next_actions:
            parts.append(("LLM next actions:\n" if is_en else "LLM 下一步动作：\n") + "\n".join(f"- {item}" for item in llm_next_actions))
        display_hints = [str(item) for item in llm_summary.get("display_hints") or [] if str(item).strip()]
        if display_hints:
            parts.append(("Display hints: " if is_en else "展示建议：") + ", ".join(display_hints[:4]))
        parts.append(
            "The left workspace has been switched to the statistics panel."
            if is_en
            else "左侧工作区已切到统计面板。"
        )
        return "\n\n".join(parts)

    def _render_setup_table(
        self,
        workbook_name: str,
        factors: list[FactorSpec],
        objectives: list[ObjectiveSpec],
    ) -> str:
        is_en = self.language == Language.EN
        basic_rows = [
            ("Workbook" if is_en else "工作簿", workbook_name),
            ("Factors" if is_en else "因素数量", str(len(factors))),
            ("Objectives" if is_en else "目标数量", str(len(objectives))),
        ]
        factor_lines = [
            f"{index}. {factor.name} | {self._factor_kind_text(factor.kind)} | {self._factor_range_text(factor)}"
            + (f" | {factor.unit}" if factor.unit else "")
            for index, factor in enumerate(factors, start=1)
        ]
        objective_lines = [
            f"{index}. {obj.name} | {self._goal_text(obj)} | {self._target_label(obj)} | {self._weight_label(obj)}"
            for index, obj in enumerate(objectives, start=1)
        ]
        sections = [
            self._render_key_value_rows(basic_rows),
            ("Factors:\n" if is_en else "因素：\n") + ("\n".join(factor_lines) if factor_lines else "1. -"),
            ("Objectives:\n" if is_en else "目标：\n") + "\n".join(objective_lines),
        ]
        return "\n\n".join(sections)

    def _render_execution_table(self, project: ProjectState, count: int, mode: str) -> str:
        if self.language == Language.EN:
            rows = [
                ("Execution type", mode),
                ("Workbook", project.workbook.name),
                ("Runs", str(count)),
                ("Factors", str(len(project.factors))),
                ("Objectives", str(len(project.objectives))),
            ]
        else:
            rows = [
                ("执行类型", mode),
                ("工作簿", project.workbook.name),
                ("实验点数", str(count)),
                ("因素数量", str(len(project.factors))),
                ("目标数量", str(len(project.objectives))),
            ]
        return self._render_key_value_rows(rows)

    def _render_score_profile_confirmation(self, recommendation: dict[str, Any] | None) -> str:
        if not recommendation:
            return "Scoring function: No adjustable items." if self.language == Language.EN else "评分函数：当前没有可调整项。"
        primary = recommendation.get("primary")
        if not isinstance(primary, dict):
            return "Scoring function: No adjustable items." if self.language == Language.EN else "评分函数：当前没有可调整项。"
        label = str(primary.get("name") or ("目标" if self.language == Language.ZH else "objective"))
        summary = primary.get("summary") or ("Using current scoring logic." if self.language == Language.EN else "当前沿用现有评分逻辑。")
        error_note = str(recommendation.get("llm_weight_error") or "").strip()
        if primary.get("changed"):
            if self.language == Language.EN:
                message = f"Suggest adjusting scoring function: {label}.\n- Reason: {summary}\n- FuzzyScore will be recalculated before generating new combinations."
            else:
                message = f"建议同步调整评分函数：{label}。\n- 原因：{summary}\n- 执行时会先重算 FuzzyScore，再生成新组合。"
        else:
            weight_recommendation = recommendation.get("llm_weight_recommendation")
            if isinstance(weight_recommendation, dict) and weight_recommendation.get("weight_adjustments"):
                focus = str(weight_recommendation.get("optimization_focus") or "")
                if self.language == Language.EN:
                    message = f"LLM suggests objective-weight refinement before BO.\n- Focus: {focus}"
                else:
                    message = f"LLM 建议在 BO 前微调目标权重。\n- 下一轮重点：{focus}"
            elif self.language == Language.EN:
                message = f"Scoring function unchanged: {label}.\n- Reason: {summary}"
            else:
                message = f"评分函数保持不变：{label}。\n- 原因：{summary}"
        if error_note:
            message += f"\n- {error_note}"
        return message

    def _analyze_result_report(self, project: ProjectState, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze the user's result report against pending plans.

        Returns a dict with:
        - matched: List of results matching planned experiments
        - missing: List of planned experiments without results
        - new_experiments: List of results for experiments not in plan
        - modified: List of results with different conditions than plan
        - summary: Brief summary of the analysis
        """
        from .storage import read_sheet, SHEET_DESIGNS

        result = {
            "matched": [],  # Results that exactly match plans
            "missing": [],  # Plans without results
            "new_experiments": [],  # Results not matching any plan
            "modified": [],  # Results matching plans but with different conditions
            "summary": "",
        }

        factor_names = {f.name for f in project.factors}

        design_df = read_sheet(project, SHEET_DESIGNS)
        if design_df.empty:
            result["summary"] = "No pending plans found." if self.language == Language.EN else "没有待确认的计划。"
            result["new_experiments"] = rows
            return result

        pending_mask = design_df["status"].isin(["proposed", "initialized", "Pending", "待做", "pending"])
        pending_plans = design_df[pending_mask].copy()

        if pending_plans.empty:
            result["summary"] = "No pending plans found." if self.language == Language.EN else "没有待做计划。"
            result["new_experiments"] = rows
            return result

        latest_pending_trial = self._latest_pending_trial(pending_plans)

        matched_plan_indices = set()

        for row in rows:
            trial = self._coerce_int(row.get("Trial"))
            candidate_id = self._coerce_int(row.get("candidate_id"))
            experiment_id = self._coerce_int(row.get("experiment_id"))

            # Try to find matching plan by explicit key. Candidate IDs alone are
            # ambiguous across batches unless the latest pending batch has a
            # unique match.
            match_mask = pd.Series([False] * len(pending_plans))

            if experiment_id is not None and "experiment_id" in row and "experiment_id" in pending_plans.columns:
                match_mask = pd.to_numeric(pending_plans["experiment_id"], errors="coerce").eq(experiment_id)
            elif trial is not None and candidate_id is not None and "Trial" in pending_plans.columns and "candidate_id" in pending_plans.columns:
                match_mask = (
                    pd.to_numeric(pending_plans["Trial"], errors="coerce").eq(trial)
                    & pd.to_numeric(pending_plans["candidate_id"], errors="coerce").eq(candidate_id)
                )
                if not match_mask.any():
                    match_mask = self._pending_plan_position_mask(pending_plans, trial, candidate_id)
            elif candidate_id is not None and latest_pending_trial is not None and "Trial" in pending_plans.columns and "candidate_id" in pending_plans.columns:
                latest_mask = (
                    pd.to_numeric(pending_plans["Trial"], errors="coerce").eq(latest_pending_trial)
                    & pd.to_numeric(pending_plans["candidate_id"], errors="coerce").eq(candidate_id)
                )
                if int(latest_mask.sum()) == 1:
                    match_mask = latest_mask
                elif not latest_mask.any():
                    match_mask = self._pending_plan_position_mask(pending_plans, latest_pending_trial, candidate_id)

            if not match_mask.any():
                # No matching plan - this is a new experiment
                result["new_experiments"].append(row)
                continue

            match_idx = match_mask[match_mask].index[0]
            matched_plan_indices.add(match_idx)
            plan_row = pending_plans.loc[match_idx].to_dict()
            resolved_row = dict(row)
            if "Trial" in plan_row:
                resolved_row["Trial"] = self._coerce_int(plan_row.get("Trial"))
            if "candidate_id" in plan_row:
                resolved_row["candidate_id"] = self._coerce_int(plan_row.get("candidate_id"))
            if "experiment_id" in plan_row and self._coerce_int(plan_row.get("experiment_id")) is not None:
                resolved_row["experiment_id"] = self._coerce_int(plan_row.get("experiment_id"))
            for factor_name in factor_names:
                if factor_name in plan_row and self._is_blank_value(resolved_row.get(factor_name)):
                    resolved_row[factor_name] = plan_row.get(factor_name)

            differences = []
            planned_values = {}
            reported_values = {}

            for factor_name in factor_names:
                if factor_name not in plan_row or factor_name not in row:
                    continue

                plan_val = plan_row[factor_name]
                report_val = row[factor_name]

                if plan_val is None or report_val is None:
                    continue
                if pd.isna(plan_val) or pd.isna(report_val):
                    continue

                try:
                    plan_numeric = float(plan_val) if str(plan_val).strip() != "" else None
                    report_numeric = float(report_val) if str(report_val).strip() != "" else None

                    if plan_numeric is not None and report_numeric is not None:
                        if abs(plan_numeric - report_numeric) > 0.001:
                            differences.append(factor_name)
                            planned_values[factor_name] = plan_val
                            reported_values[factor_name] = report_val
                    elif str(plan_val).strip() != str(report_val).strip():
                        differences.append(factor_name)
                        planned_values[factor_name] = plan_val
                        reported_values[factor_name] = report_val
                except (ValueError, TypeError):
                    if str(plan_val).strip() != str(report_val).strip():
                        differences.append(factor_name)
                        planned_values[factor_name] = plan_val
                        reported_values[factor_name] = report_val

            match_info = {
                "trial": self._coerce_int(resolved_row.get("Trial")),
                "candidate_id": self._coerce_int(resolved_row.get("candidate_id")),
                "experiment_id": self._coerce_int(resolved_row.get("experiment_id")),
                "row": resolved_row,
            }

            if differences:
                match_info["planned"] = planned_values
                match_info["reported"] = reported_values
                match_info["differences"] = differences
                result["modified"].append(match_info)
            else:
                result["matched"].append(match_info)

        for idx in pending_plans.index:
            if idx not in matched_plan_indices:
                plan_row = pending_plans.loc[idx].to_dict()
                result["missing"].append({
                    "trial": int(plan_row.get("Trial")) if pd.notna(plan_row.get("Trial")) else None,
                    "candidate_id": int(plan_row.get("candidate_id")) if pd.notna(plan_row.get("candidate_id")) else None,
                    "experiment_id": int(plan_row.get("experiment_id")) if pd.notna(plan_row.get("experiment_id")) else None,
                })

        summary_parts = []
        if result["matched"]:
            count = len(result["matched"])
            summary_parts.append(f"{count} plan(s) with exact match" if self.language == Language.EN else f"{count} 条计划完全匹配")
        if result["modified"]:
            count = len(result["modified"])
            summary_parts.append(f"{count} plan(s) with modified conditions" if self.language == Language.EN else f"{count} 条计划条件有修改")
        if result["new_experiments"]:
            count = len(result["new_experiments"])
            summary_parts.append(f"{count} new experiment(s) not in plan" if self.language == Language.EN else f"{count} 条新增实验（不在计划中）")
        if result["missing"]:
            count = len(result["missing"])
            summary_parts.append(f"{count} plan(s) still missing results" if self.language == Language.EN else f"{count} 条计划还缺结果")

        if summary_parts:
            result["summary"] = "; ".join(summary_parts) + "."
        else:
            result["summary"] = "No results reported." if self.language == Language.EN else "未汇报结果。"

        return result

    def _latest_pending_trial(self, pending_plans: pd.DataFrame) -> int | None:
        if pending_plans.empty or "Trial" not in pending_plans.columns:
            return None
        trials = pd.to_numeric(pending_plans["Trial"], errors="coerce").dropna()
        if trials.empty:
            return None
        return int(trials.max())

    def _pending_plan_position_mask(
        self,
        pending_plans: pd.DataFrame,
        trial: int,
        position: int,
    ) -> pd.Series:
        empty = pd.Series([False] * len(pending_plans), index=pending_plans.index)
        if position <= 0 or "Trial" not in pending_plans.columns:
            return empty
        trial_mask = pd.to_numeric(pending_plans["Trial"], errors="coerce").eq(trial)
        trial_plans = pending_plans.loc[trial_mask].copy()
        if trial_plans.empty or position > len(trial_plans):
            return empty
        trial_plans["_row_order"] = range(len(trial_plans))
        sort_columns = ["_row_order"]
        ascending = [True]
        if "candidate_id" in trial_plans.columns:
            trial_plans["_candidate_sort"] = pd.to_numeric(trial_plans["candidate_id"], errors="coerce")
            sort_columns.insert(0, "_candidate_sort")
            ascending.insert(0, True)
        ordered = trial_plans.sort_values(sort_columns, ascending=ascending, kind="stable", na_position="last")
        target_index = ordered.index[position - 1]
        mask = empty.copy()
        mask.loc[target_index] = True
        return mask

    def _is_blank_value(self, value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    def _resolved_result_rows_from_analysis(
        self,
        original_rows: list[dict[str, Any]],
        analysis: dict[str, Any],
    ) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        for section in ("matched", "modified"):
            for item in analysis.get(section, []) or []:
                row = item.get("row")
                if isinstance(row, dict):
                    resolved.append(row)
        for row in analysis.get("new_experiments", []) or []:
            if isinstance(row, dict):
                resolved.append(row)
        if resolved:
            return resolved
        return original_rows

    def _incomplete_unplanned_result_rows(
        self,
        project: ProjectState,
        rows_to_record: list[dict[str, Any]],
        analysis: dict[str, Any],
    ) -> list[dict[str, Any]]:
        new_rows = [row for row in analysis.get("new_experiments", []) or [] if isinstance(row, dict)]
        if not new_rows:
            return []
        factor_names = [factor.name for factor in project.factors]
        incomplete = []
        for row in new_rows:
            if any(not self._is_blank_value(row.get(name)) for name in factor_names):
                continue
            incomplete.append(row)
        return incomplete

    def _render_result_report_confirmation(self, analysis: dict[str, Any]) -> str:
        """Render a comprehensive confirmation message for result reporting.

        Shows:
        - Which plans have exact match
        - Which plans have modified conditions
        - Which plans are missing results
        - Which experiments are new (not in plan)
        """
        if self.language == Language.EN:
            lines = ["Here's what I found from your result report:"]
        else:
            lines = ["根据你汇报的结果，我分析如下："]

        # Summary
        summary = analysis.get("summary", "")
        if summary:
            lines.append(f"\n{summary}")

        # Modified conditions
        modified = analysis.get("modified", [])
        if modified:
            if self.language == Language.EN:
                lines.append("\n**Modified conditions:**")
            else:
                lines.append("\n**条件有修改的实验：**")

            for item in modified:
                cid = item.get("candidate_id")
                trial = item.get("trial")
                differences = item.get("differences", [])
                planned = item.get("planned", {})
                reported = item.get("reported", {})

                if self.language == Language.EN:
                    lines.append(f"- Experiment #{cid} (Batch {trial}):")
                    for factor in differences:
                        lines.append(f"  - {factor}: {planned.get(factor)} → {reported.get(factor)}")
                else:
                    lines.append(f"- 实验 #{cid}（批次 {trial}）：")
                    for factor in differences:
                        lines.append(f"  - {factor}：{planned.get(factor)} → {reported.get(factor)}")

        # Missing results
        missing = analysis.get("missing", [])
        if missing:
            if self.language == Language.EN:
                lines.append("\n**Still missing results:**")
            else:
                lines.append("\n**还缺结果的实验：**")

            missing_list = []
            for item in missing:
                cid = item.get("candidate_id")
                trial = item.get("trial")
                if self.language == Language.EN:
                    missing_list.append(f"#{cid} (Batch {trial})")
                else:
                    missing_list.append(f"#{cid}（批次 {trial}）")
            lines.append(", ".join(missing_list))

        # New experiments
        new_experiments = analysis.get("new_experiments", [])
        if new_experiments:
            if self.language == Language.EN:
                lines.append("\n**New experiments (not in plan):**")
            else:
                lines.append("\n**新增实验（不在计划中）：**")

            for item in new_experiments:
                cid = item.get("candidate_id")
                trial = item.get("Trial")
                if self.language == Language.EN:
                    lines.append(f"- Experiment #{cid} (Batch {trial})")
                else:
                    lines.append(f"- 实验 #{cid}（批次 {trial}）")

        # Options
        if self.language == Language.EN:
            lines.append("\nHow would you like to proceed?")
            lines.append("1. Reply `confirm` or `yes` to record all results as-is")
            lines.append("2. Reply `update plan` to update plans with modified conditions, then record")
            lines.append("3. Reply `cancel` to cancel and re-enter the data")
        else:
            lines.append("\n请问你想如何处理？")
            lines.append("1. 回复 `确认` 或 `可以` 按当前汇报条件记录所有结果")
            lines.append("2. 回复 `更新计划` 将计划修改为汇报的条件，然后记录")
            lines.append("3. 回复 `取消` 取消本次操作，重新输入数据")

        return "\n".join(lines)

    def _render_score_profile_result(self, recommendation: dict[str, Any] | None, changed_count: int) -> str:
        if not recommendation:
            return "Scoring function unchanged." if self.language == Language.EN else "评分函数保持不变。"
        primary = recommendation.get("primary")
        if not isinstance(primary, dict):
            return "Scoring function unchanged." if self.language == Language.EN else "评分函数保持不变。"
        label = str(primary.get("name") or ("目标" if self.language == Language.ZH else "objective"))
        summary = str(primary.get("summary") or "")
        error_note = str(recommendation.get("llm_weight_error") or "").strip()
        if changed_count > 0:
            weight_recommendation = recommendation.get("llm_weight_recommendation")
            weight_note = ""
            if isinstance(weight_recommendation, dict) and weight_recommendation.get("weight_adjustments"):
                focus = str(weight_recommendation.get("optimization_focus") or "")
                weight_note = f"；LLM 权重建议已纳入：{focus}" if self.language == Language.ZH else f"; LLM weight focus applied: {focus}"
            if self.language == Language.EN:
                message = f"Adjusted scoring/weights for {changed_count} items. Primary {label}: {summary}{weight_note}"
            else:
                message = f"已同步调整 {changed_count} 个评分/权重项。主目标 {label}：{summary}{weight_note}"
        else:
            if self.language == Language.EN:
                message = f"No scoring adjustments this round. Primary {label}: {summary}"
            else:
                message = f"本轮未调整评分函数。主目标 {label}：{summary}"
        if error_note:
            message += f"\n{error_note}"
        return message

    def _render_key_value_rows(self, rows: list[tuple[str, str]] | list[list[str]]) -> str:
        normalized = [(str(row[0]), str(row[1])) for row in rows]
        separator = ": " if self.language == Language.EN else "："
        return "\n".join(f"- {label}{separator}{value}" for label, value in normalized)

    def _render_numbered_list(self, items: list[str]) -> str:
        return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))

    def _render_bulleted_list(self, items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items)

    def _render_best_experiment(
        self,
        best_row: dict[str, Any] | None,
        project: ProjectState,
        objective_name: str,
    ) -> str:
        if not isinstance(best_row, dict) or not best_row:
            return "暂无已完成实验。"
        lines = []
        if best_row.get("Trial") not in (None, ""):
            lines.append(f"- 批次：{best_row.get('Trial')}")
        if best_row.get("FuzzyScore") not in (None, ""):
            lines.append(f"- 最佳评分：{self._format_metric(best_row.get('FuzzyScore'))}")
        if objective_name in best_row and best_row.get(objective_name) not in (None, ""):
            lines.append(f"- {objective_name}：{self._format_metric(best_row.get(objective_name))}")
        factor_lines = []
        for factor in project.factors:
            value = best_row.get(factor.name)
            if value in (None, ""):
                continue
            factor_lines.append(f"{factor.name}={self._format_metric(value)}")
        if factor_lines:
            lines.append(f"- 条件：{', '.join(factor_lines)}")
        return "\n".join(lines) if lines else "暂无已完成实验。"

    def _sparkline(self, values: list[float | int]) -> str:
        ticks = "▁▂▃▄▅▆▇█"
        cleaned = [float(value) for value in values if value is not None]
        if not cleaned:
            return "-"
        if len(cleaned) == 1:
            return ticks[-1]
        low = min(cleaned)
        high = max(cleaned)
        if high - low < 1e-12:
            return ticks[3] * len(cleaned)
        chars = []
        for value in cleaned:
            ratio = (value - low) / (high - low)
            index = min(int(ratio * (len(ticks) - 1)), len(ticks) - 1)
            chars.append(ticks[index])
        return "".join(chars)

    def _trend_endpoints(self, values: list[float | int]) -> str:
        if not values:
            return "-"
        start = self._format_metric(values[0])
        end = self._format_metric(values[-1])
        return f"{start} -> {end}"

    def _format_metric(self, value: object) -> str:
        if value in (None, ""):
            return "-"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    def _factor_kind_text(self, kind: str) -> str:
        if self.language == Language.EN:
            mapping = {"continuous": "continuous", "ordinal": "ordinal", "categorical": "categorical"}
        else:
            mapping = {"continuous": "连续", "ordinal": "离散", "categorical": "分类"}
        return mapping.get(kind, kind)

    def _factor_range_text(self, factor: FactorSpec) -> str:
        if factor.kind == "categorical":
            return ", ".join(factor.categories) or "-"
        if self.language == Language.EN:
            if factor.kind == "ordinal":
                return f"{factor.min_value} to {factor.max_value}, step {factor.step}"
            return f"{factor.min_value} to {factor.max_value}"
        else:
            if factor.kind == "ordinal":
                return f"{factor.min_value} 到 {factor.max_value}，步长 {factor.step}"
            return f"{factor.min_value} 到 {factor.max_value}"

    def _goal_text(self, objective: ObjectiveSpec) -> str:
        if self.language == Language.EN:
            if objective.goal == "maximize":
                return "maximize"
            if objective.goal == "minimize":
                return "minimize"
            return f"target {objective.target_value}"
        else:
            if objective.goal == "maximize":
                return "最大化"
            if objective.goal == "minimize":
                return "最小化"
            return f"贴近 {objective.target_value}"

    def _goal_cn(self, goal: str, target_value: object | None) -> str:
        if goal == "maximize":
            return "最大化"
        if goal == "minimize":
            return "最小化"
        return f"贴近 {target_value}"

    def _target_label(self, objective: ObjectiveSpec) -> str:
        if self.language == Language.EN:
            return f"Target: {objective.target_value}" if objective.goal == "target" else "Target: -"
        return f"目标值 {objective.target_value}" if objective.goal == "target" else "目标值 -"

    def _weight_label(self, objective: ObjectiveSpec) -> str:
        if self.language == Language.EN:
            return f"Weight: {objective.weight}"
        return f"权重 {objective.weight}"

    def _coerce_float(self, value: object) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _coerce_int(self, value: object) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _coerce_categories(self, value: object) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]


def project_snapshot(project: ProjectState | None) -> str:
    """Serialize a project state (or ``{"project": null}``) as pretty JSON."""
    if project is None:
        return json.dumps({"project": None}, ensure_ascii=False, indent=2)
    return json.dumps(project.model_dump(), ensure_ascii=False, indent=2)
