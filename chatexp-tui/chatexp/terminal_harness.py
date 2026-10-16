from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .action_contract import ACTION_CONTRACTS, validate_action_contracts
from .models import ChatPending, IntentPlan
from .storage import SHEET_DESIGNS, SHEET_EXPERIMENTS, read_sheet


CONFIRMATION_CONFIRM = "confirm"
CONFIRMATION_APPLY_ALL = "apply_all"
CONFIRMATION_CANCEL = "cancel"
CONFIRMATION_DECISIONS = {CONFIRMATION_CONFIRM, CONFIRMATION_APPLY_ALL, CONFIRMATION_CANCEL}
RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass(frozen=True)
class TerminalAction:
    type: str
    arguments: dict[str, Any] = field(default_factory=dict)
    skill: str = ""
    source: str = "planner"

    @classmethod
    def from_plan(cls, plan: IntentPlan) -> "TerminalAction":
        return cls(type=plan.action, arguments=dict(plan.arguments or {}), source="planner")

    def model_dump(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "arguments": dict(self.arguments),
            "skill": self.skill,
            "source": self.source,
        }


@dataclass(frozen=True)
class ActionRisk:
    level: str
    label_zh: str
    label_en: str


@dataclass
class PreflightResult:
    blocked: bool = False
    message: str = ""
    action: TerminalAction | None = None
    risk: ActionRisk | None = None
    permission: str = ""
    requires_confirmation: bool = False
    reason: str = ""


@dataclass
class WorkflowResult:
    content: str | None = None
    blocked: bool = False
    needs_confirmation: bool = False
    pending: ChatPending | None = None
    action: TerminalAction | None = None
    preflight: PreflightResult | None = None
    preflights: list[PreflightResult] = field(default_factory=list)
    postflight: "PostflightResult | None" = None
    executed_actions: list[TerminalAction] = field(default_factory=list)


@dataclass
class PostflightResult:
    ok: bool = True
    message: str = ""
    reason: str = ""
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)


class TerminalWorkflowHarness:
    """Plan/skills/harness layer for terminal-side primitive actions.

    The LLM only chooses structured actions. The harness owns action
    normalization, permission/risk grading, preflight, confirmation, and the
    final call into the existing service-backed executor.
    """

    def __init__(self, coordinator: Any):
        self.coordinator = coordinator

    def action_from_plan(self, plan: IntentPlan) -> TerminalAction:
        return TerminalAction.from_plan(plan)

    def actions_from_plan(self, plan: IntentPlan) -> list[TerminalAction]:
        if plan.actions:
            return [TerminalAction.from_plan(child) for child in plan.actions if child.action != "none"]
        if plan.action == "none":
            return []
        return [TerminalAction.from_plan(plan)]

    def action_permission(self, action: TerminalAction) -> str:
        contract = ACTION_CONTRACTS.get(action.type)
        return contract.permission if contract else "unknown"

    def action_risk(self, action: TerminalAction) -> ActionRisk:
        contract = ACTION_CONTRACTS.get(action.type)
        if not contract:
            return ActionRisk("medium", "未知终端操作", "Unknown terminal action")
        if action.type == "skip_pending_designs":
            ids = action.arguments.get("candidate_ids")
            if not ids:
                return ActionRisk("high", "高风险数据变更", "High-risk data change")
        return ActionRisk(contract.risk_level, contract.risk_zh, contract.risk_en)

    def preflight_action(self, action: TerminalAction) -> PreflightResult:
        language = self.coordinator.language
        zh = str(language.value if hasattr(language, "value") else language).lower().startswith("zh")
        risk = self.action_risk(action)
        permission = self.action_permission(action)
        contract_errors = validate_action_contracts()
        if contract_errors:
            return PreflightResult(
                blocked=True,
                message="; ".join(contract_errors),
                action=action,
                risk=risk,
                permission=permission,
                reason="invalid_action_contract",
            )
        contract = ACTION_CONTRACTS.get(action.type)
        if contract is None:
            return PreflightResult(
                blocked=True,
                message=f"Unsupported action: {action.type}",
                action=action,
                risk=risk,
                permission=permission,
                reason="unsupported_action",
            )

        if not self._permission_enabled(contract.permission):
            return PreflightResult(
                blocked=True,
                message=(
                    f"权限未开启，不能执行 {action.type}：{contract.permission}。"
                    if zh
                    else f"Permission is disabled for {action.type}: {contract.permission}."
                ),
                action=action,
                risk=risk,
                permission=permission,
                reason="permission_disabled",
            )

        project_required = {
            "define_factors",
            "define_objective",
            "update_objective",
            "update_factor",
            "initialize_design",
            "append_results",
            "update_plan_experiment",
            "analyze_data",
            "generate_next_design",
            "import_experiment_sheet",
            "export",
            "refresh_scores",
            "inspect_workbook",
            "status",
            "view_experiments",
            "skip_pending_designs",
        }
        project = self.coordinator.session.project
        if action.type in project_required and project is None:
            if action.type == "append_results" and action.arguments.get("rows"):
                pass
            elif action.type == "import_experiment_sheet":
                pass
            else:
                return PreflightResult(
                    blocked=True,
                    message="当前还没有项目。请先创建或导入项目。" if zh else "No project is loaded. Create or import a project first.",
                    action=action,
                    risk=risk,
                    permission=permission,
                    reason="project_required",
                )

        # generate_next_design data-readiness is intentionally NOT checked
        # here: earlier actions in the same plan (e.g. append_results) may
        # make the data model-ready, so the gate is evaluated inside the
        # executor at execution time, after those actions have mutated data.

        missing = self._missing_required_fields(action)
        if missing:
            return PreflightResult(
                blocked=True,
                message=(
                    f"动作 {action.type} 缺少必要字段：{', '.join(missing)}。"
                    if zh
                    else f"Action {action.type} is missing required fields: {', '.join(missing)}."
                ),
                action=action,
                risk=risk,
                permission=permission,
                reason="missing_required_fields",
            )

        requires_confirmation = contract.requires_confirmation and not contract.delegated_confirmation
        return PreflightResult(
            blocked=False,
            message=self._confirmation_message(action, risk, permission, zh),
            action=action,
            risk=risk,
            permission=permission,
            requires_confirmation=requires_confirmation,
        )

    def execute_workflow(
        self,
        action: TerminalAction,
        executor: Callable[[dict[str, Any]], str | None] | None,
        *,
        force: bool = False,
        continuation_actions: list[TerminalAction] | None = None,
    ) -> WorkflowResult:
        preflight = self.preflight_action(action)
        if preflight.blocked:
            return WorkflowResult(content=preflight.message, blocked=True, action=action, preflight=preflight)
        if preflight.requires_confirmation and not force:
            confirmation = self._confirmation_payload([action], continuation_actions or [])
            pending = ChatPending(
                action="confirm_terminal_action",
                workbook=self._pending_workbook_name(action),
                data={
                    "action": action.model_dump(),
                    "audit": self._audit_payload(preflight),
                    "confirmation": confirmation,
                    "display_actions": [item.model_dump() for item in [action, *(continuation_actions or [])]],
                },
            )
            if continuation_actions:
                pending.data["continuation_actions"] = [item.model_dump() for item in continuation_actions]
            return WorkflowResult(
                content=self.render_confirmation(preflight, confirmation=confirmation),
                needs_confirmation=True,
                pending=pending,
                action=action,
                preflight=preflight,
            )
        if executor is None:
            return WorkflowResult(
                content=f"No executor registered for action: {action.type}",
                blocked=True,
                action=action,
                preflight=preflight,
            )
        before = self._state_snapshot()
        execution_arguments = {**action.arguments, "_confirmed_by_harness": True} if force else action.arguments
        try:
            content = executor(execution_arguments)
        except Exception as exc:
            # An executor exception mid-apply-all must not orphan the actions
            # already executed in this batch. Surface a blocked WorkflowResult
            # carrying the actions executed so far so callers can resume or
            # reconcile the pending state instead of losing them.
            return WorkflowResult(
                content=f"Executor raised {type(exc).__name__}: {exc}",
                blocked=True,
                action=action,
                preflight=preflight,
                postflight=PostflightResult(
                    ok=False,
                    message=f"Executor raised {type(exc).__name__}: {exc}",
                    reason="executor_exception",
                    before=before,
                    after=self._state_snapshot(),
                ),
            )
        postflight = self.postflight_action(action, before, self._state_snapshot(), content)
        if not postflight.ok:
            return WorkflowResult(
                content=postflight.message,
                blocked=True,
                action=action,
                preflight=preflight,
                postflight=postflight,
            )
        return WorkflowResult(content=content, action=action, preflight=preflight, postflight=postflight)

    def execute_actions(
        self,
        actions: list[TerminalAction],
        executor_for_action: Callable[[TerminalAction], Callable[[dict[str, Any]], str | None] | None],
        *,
        force_first: bool = False,
    ) -> WorkflowResult:
        remaining = [action for action in actions if action and action.type]
        preflights = self.preflight_actions(remaining)
        blocked = next((preflight for preflight in preflights if preflight.blocked), None)
        if blocked is not None:
            return WorkflowResult(
                content=blocked.message,
                blocked=True,
                action=blocked.action,
                preflight=blocked,
                preflights=preflights,
            )
        contents: list[str] = []
        executed: list[TerminalAction] = []
        while remaining:
            action = remaining.pop(0)
            executor = executor_for_action(action)
            result = self.execute_workflow(action, executor, force=force_first, continuation_actions=remaining)
            force_first = False
            if result.needs_confirmation and result.pending is not None:
                result.content = "\n\n".join([part for part in ["\n\n".join(contents), result.content or ""] if part])
                result.preflights = preflights
                result.executed_actions = executed
                return result
            if result.blocked:
                result.content = "\n\n".join([part for part in ["\n\n".join(contents), result.content or ""] if part])
                result.preflights = preflights
                result.executed_actions = executed
                return result
            if result.content:
                contents.append(result.content)
            executed.append(action)
        return WorkflowResult(
            content="\n\n".join(contents),
            action=actions[-1] if actions else None,
            preflights=preflights,
            executed_actions=executed,
        )

    def preflight_actions(self, actions: list[TerminalAction]) -> list[PreflightResult]:
        return [self.preflight_action(action) for action in actions if action and action.type]

    def render_confirmation(self, preflight: PreflightResult, confirmation: dict[str, Any] | None = None) -> str:
        language = self.coordinator.language
        zh = str(language.value if hasattr(language, "value") else language).lower().startswith("zh")
        action = preflight.action
        if not action:
            return preflight.message
        decisions = confirmation.get("decisions") if isinstance(confirmation, dict) else []
        can_apply_all = CONFIRMATION_APPLY_ALL in decisions
        if zh:
            choices = "回复 `确认` 执行，回复 `全部应用` 执行本轮剩余低/中风险步骤，或回复 `取消` 放弃。"
            if not can_apply_all:
                choices = "回复 `确认` 执行，或回复 `取消` 放弃。"
            return "\n".join([
                preflight.message,
                f"审核：权限 {preflight.permission}；风险 {preflight.risk.label_zh if preflight.risk else '未知'}。",
                choices,
            ])
        choices = "Reply `confirm` to execute, `apply all` to execute the remaining low/medium-risk steps, or `cancel` to abort."
        if not can_apply_all:
            choices = "Reply `confirm` to execute, or `cancel` to abort."
        return "\n".join([
            preflight.message,
            f"Audit: permission {preflight.permission}; risk {preflight.risk.label_en if preflight.risk else 'unknown'}.",
            choices,
        ])

    def execute_pending(
        self,
        pending: ChatPending,
        executor_for_action: Callable[[TerminalAction], Callable[[dict[str, Any]], str | None] | None],
        *,
        decision: str = CONFIRMATION_CONFIRM,
    ) -> WorkflowResult:
        if decision not in CONFIRMATION_DECISIONS:
            return WorkflowResult(content=f"Unsupported confirmation decision: {decision}", blocked=True)
        allowed = self._pending_confirmation_decisions(pending)
        if decision not in allowed:
            return WorkflowResult(content=f"Confirmation decision is not allowed for this pending action: {decision}", blocked=True)
        if decision == CONFIRMATION_CANCEL:
            return WorkflowResult(content="Pending action cancelled.")
        payload = pending.data.get("action") if isinstance(pending.data, dict) else None
        if not isinstance(payload, dict):
            return WorkflowResult(content="No audited pending action.", blocked=True)
        action = TerminalAction(
            type=str(payload.get("type") or ""),
            arguments=dict(payload.get("arguments") or {}),
            skill=str(payload.get("skill") or ""),
            source=str(payload.get("source") or "pending"),
        )
        executor = executor_for_action(action)
        if executor is None:
            return WorkflowResult(content=f"Unsupported pending action: {action.type}", blocked=True, action=action)
        confirmed_action = TerminalAction(
            type=action.type,
            arguments={**action.arguments, "_confirmed_by_harness": True},
            skill=action.skill,
            source=action.source,
        )
        continuation = []
        raw_continuation = pending.data.get("continuation_actions") if isinstance(pending.data, dict) else None
        if isinstance(raw_continuation, list):
            for item in raw_continuation:
                if isinstance(item, dict):
                    continuation.append(TerminalAction(
                        type=str(item.get("type") or ""),
                        arguments=dict(item.get("arguments") or {}),
                        skill=str(item.get("skill") or ""),
                        source=str(item.get("source") or "pending"),
                    ))
        if decision == CONFIRMATION_APPLY_ALL:
            return self._execute_with_apply_all(confirmed_action, continuation, executor_for_action)
        return self.execute_actions([confirmed_action, *continuation], executor_for_action, force_first=True)

    def _audit_payload(self, preflight: PreflightResult) -> dict[str, Any]:
        return {
            "permission": preflight.permission,
            "risk": preflight.risk.__dict__ if preflight.risk else {},
            "message": preflight.message,
            "requires_confirmation": preflight.requires_confirmation,
            "reason": preflight.reason,
        }

    def _pending_workbook_name(self, action: TerminalAction) -> str | None:
        """Workbook a pending confirmation belongs to.

        Actions that name their target workbook (create/load project) record
        that name; every other project-bound action records the workbook of
        the project currently loaded in the session.
        """
        workbook_name = action.arguments.get("workbook_name") or action.arguments.get("path")
        if workbook_name:
            return str(workbook_name)
        project = getattr(self.coordinator.session, "project", None)
        if project is not None:
            return project.workbook.name
        return None

    def _confirmation_payload(
        self,
        actions: list[TerminalAction],
        continuation_actions: list[TerminalAction],
    ) -> dict[str, Any]:
        decisions = [CONFIRMATION_CONFIRM, CONFIRMATION_CANCEL]
        if continuation_actions and not self.continuation_requires_confirmation_boundary([*actions, *continuation_actions]):
            decisions.insert(1, CONFIRMATION_APPLY_ALL)
        return {
            "kind": "standard",
            "decisions": decisions,
            "action_count": len([action for action in actions if action and action.type]),
            "continuation_count": len([action for action in continuation_actions if action and action.type]),
        }

    def continuation_requires_confirmation_boundary(self, actions: list[TerminalAction]) -> bool:
        return any(self.requires_explicit_confirmation_boundary(action) for action in actions)

    def requires_explicit_confirmation_boundary(self, action: TerminalAction) -> bool:
        risk = self.action_risk(action)
        return RISK_ORDER.get(risk.level, 1) >= RISK_ORDER["high"]

    def _execute_with_apply_all(
        self,
        first_action: TerminalAction,
        continuation: list[TerminalAction],
        executor_for_action: Callable[[TerminalAction], Callable[[dict[str, Any]], str | None] | None],
    ) -> WorkflowResult:
        contents: list[str] = []
        executed: list[TerminalAction] = []
        actions = [first_action, *continuation]
        while actions:
            action = actions.pop(0)
            executor = executor_for_action(action)
            result = self.execute_workflow(action, executor, force=True, continuation_actions=actions)
            if result.blocked or result.needs_confirmation:
                result.content = "\n\n".join([part for part in ["\n\n".join(contents), result.content or ""] if part])
                result.executed_actions = executed
                return result
            if result.content:
                contents.append(result.content)
            executed.append(action)
            if actions and self.requires_explicit_confirmation_boundary(actions[0]):
                boundary_result = self.execute_actions(actions, executor_for_action)
                boundary_result.content = "\n\n".join([part for part in ["\n\n".join(contents), boundary_result.content or ""] if part])
                boundary_result.executed_actions = executed + boundary_result.executed_actions
                return boundary_result
        return WorkflowResult(
            content="\n\n".join(contents),
            action=continuation[-1] if continuation else first_action,
            executed_actions=executed,
        )

    def _pending_confirmation_decisions(self, pending: ChatPending) -> set[str]:
        data = pending.data if isinstance(pending.data, dict) else {}
        confirmation = data.get("confirmation")
        if isinstance(confirmation, dict) and isinstance(confirmation.get("decisions"), list):
            decisions = {str(item) for item in confirmation.get("decisions") if str(item) in CONFIRMATION_DECISIONS}
            if decisions:
                return decisions
        return {CONFIRMATION_CONFIRM, CONFIRMATION_CANCEL}

    def postflight_action(
        self,
        action: TerminalAction,
        before: dict[str, Any],
        after: dict[str, Any],
        content: str | None,
    ) -> PostflightResult:
        contract = ACTION_CONTRACTS.get(action.type)
        if not contract or not contract.mutation:
            return PostflightResult(ok=True, before=before, after=after)
        if self._content_indicates_failure(content):
            return PostflightResult(
                ok=False,
                message=str(content or "Action failed verification."),
                reason="executor_reported_failure",
                before=before,
                after=after,
            )

        if action.type == "draft_setup":
            pending = getattr(self.coordinator.session, "pending", None)
            ok = bool(pending and pending.action == "confirm_setup")
            return self._postflight_result(ok, "draft_setup_pending_not_created", before, after)

        if action.type in {"create_project", "load_project"}:
            ok = bool(after.get("has_project") and after.get("workbook_exists"))
            return self._postflight_result(ok, "project_not_loaded_or_workbook_missing", before, after)

        if action.type in {"define_factors", "define_objective", "update_objective", "update_factor"}:
            ok = bool(after.get("has_project") and after.get("workbook_exists"))
            if action.type in {"define_factors", "update_factor"}:
                ok = ok and after.get("factor_count", 0) > 0
            if action.type in {"define_objective", "update_objective"}:
                ok = ok and after.get("objective_count", 0) > 0
            return self._postflight_result(ok, "project_structure_not_verified", before, after)

        if action.type in {"initialize_design", "generate_next_design"}:
            ok = (
                after.get("has_project")
                and after.get("workbook_exists")
                and after.get("design_rows", 0) > before.get("design_rows", 0)
                and after.get("latest_batch_id", 0) >= before.get("latest_batch_id", 0)
            )
            return self._postflight_result(ok, "design_write_not_verified", before, after)

        if action.type in {"append_results", "import_experiment_sheet", "skip_pending_designs"}:
            ok = (
                after.get("has_project")
                and after.get("workbook_exists")
                and (
                    after.get("experiment_rows", 0) > before.get("experiment_rows", 0)
                    or after.get("workbook_mtime_ns") != before.get("workbook_mtime_ns")
                )
            )
            return self._postflight_result(ok, "experiment_write_not_verified", before, after)

        if action.type == "update_plan_experiment":
            ok = (
                after.get("has_project")
                and after.get("workbook_exists")
                and after.get("workbook_mtime_ns") != before.get("workbook_mtime_ns")
            )
            return self._postflight_result(ok, "plan_update_not_verified", before, after)

        if action.type == "refresh_scores":
            ok = bool(after.get("has_project") and after.get("workbook_exists"))
            return self._postflight_result(ok, "score_refresh_not_verified", before, after)

        if action.type == "export":
            ok = self._export_verified(action, content)
            return self._postflight_result(ok, "export_file_not_verified", before, after)

        return PostflightResult(ok=True, before=before, after=after)

    def _postflight_result(self, ok: bool, reason: str, before: dict[str, Any], after: dict[str, Any]) -> PostflightResult:
        if ok:
            return PostflightResult(ok=True, before=before, after=after)
        return PostflightResult(
            ok=False,
            message=f"Action executed but postflight verification failed: {reason}.",
            reason=reason,
            before=before,
            after=after,
        )

    def _state_snapshot(self) -> dict[str, Any]:
        project = getattr(self.coordinator.session, "project", None)
        snapshot: dict[str, Any] = {
            "has_project": project is not None,
            "workbook_path": "",
            "workbook_exists": False,
            "workbook_mtime_ns": None,
            "factor_count": 0,
            "objective_count": 0,
            "latest_batch_id": 0,
            "experiment_rows": 0,
            "design_rows": 0,
        }
        if project is None:
            return snapshot
        workbook = project.workbook
        snapshot.update({
            "workbook_path": str(workbook),
            "workbook_exists": workbook.exists(),
            "factor_count": len(project.factors or []),
            "objective_count": len(project.objectives or []),
            "latest_batch_id": int(project.latest_batch_id or 0),
        })
        if workbook.exists():
            snapshot["workbook_mtime_ns"] = workbook.stat().st_mtime_ns
            try:
                snapshot["experiment_rows"] = len(read_sheet(project, SHEET_EXPERIMENTS))
            except Exception:
                snapshot["experiment_rows"] = -1
            try:
                snapshot["design_rows"] = len(read_sheet(project, SHEET_DESIGNS))
            except Exception:
                snapshot["design_rows"] = -1
        return snapshot

    def _export_verified(self, action: TerminalAction, content: str | None) -> bool:
        candidates = []
        output_name = action.arguments.get("output_name")
        if output_name:
            candidates.append(Path(str(output_name)).expanduser())
        text = str(content or "")
        for token in text.replace("`", " ").split():
            if token.endswith(".json") or token.endswith(".xlsx") or token.endswith(".csv"):
                candidates.append(Path(token).expanduser())
        workspace = getattr(self.coordinator.service, "workspace", None)
        for candidate in candidates:
            paths = [candidate]
            if workspace is not None and not candidate.is_absolute():
                paths.append(Path(workspace) / candidate)
            if any(path.exists() for path in paths):
                return True
        return "exported" in text.lower() or "已导出" in text

    def _content_indicates_failure(self, content: str | None) -> bool:
        """Opt-in failure detection. Only content carrying the ``[FAILED]``
        sentinel prefix counts as failure (an exception counts too — the
        harness wraps the executor and records the exception as a blocked
        WorkflowResult). Absent that explicit signal, postflight defaults to
        success and relies on the structural state-snapshot checks."""
        if content is None:
            return False
        text = str(content)
        # Explicit failure sentinel: only this prefix is treated as failure.
        return text.startswith("[FAILED]")

    def _missing_required_fields(self, action: TerminalAction) -> list[str]:
        args = action.arguments
        missing: list[str] = []
        if action.type in {"create_project", "load_project"} and not (args.get("workbook_name") or args.get("path")):
            missing.append("workbook_name")
        if action.type == "draft_setup":
            if not args.get("factors"):
                missing.append("factors")
            if not (args.get("objective") or args.get("objectives")):
                missing.append("objective/objectives")
        if action.type in {"define_objective", "update_objective"}:
            if not (args.get("objective") or args.get("objectives") or (args.get("name") and args.get("goal"))):
                missing.append("objective/objectives")
        if action.type == "define_factors" and not args.get("factors"):
            missing.append("factors")
        if action.type == "update_factor" and not (args.get("factor_name") or args.get("name")):
            missing.append("factor_name")
        if action.type == "initialize_design":
            return []
        if action.type == "generate_next_design":
            return []
        if action.type == "append_results" and not (args.get("rows") or args.get("source_excel") or args.get("workbook_name")):
            missing.append("rows/source_excel")
        if action.type == "import_experiment_sheet" and not (args.get("source_excel") or args.get("workbook_name")):
            missing.append("source_excel")
        if action.type == "update_plan_experiment":
            editable = {key for key, value in args.items() if key not in {"experiment_id", "Trial", "candidate_id", "COMMENT"} and value not in (None, "")}
            if not editable and not args.get("COMMENT"):
                missing.append("factor updates")
        return missing

    def _confirmation_message(self, action: TerminalAction, risk: ActionRisk, permission: str, zh: bool) -> str:
        args = action.arguments
        count = args.get("count") or args.get("n_points")
        if action.type == "initialize_design":
            return f"准备初始化 {count or '默认'} 组实验设计。" if zh else f"Prepared to initialize {count or 'default'} design runs."
        if action.type == "generate_next_design":
            return f"准备生成 {count or '默认'} 组新实验设计。" if zh else f"Prepared to generate {count or 'default'} new design runs."
        if action.type == "append_results":
            rows = args.get("rows") if isinstance(args.get("rows"), list) else []
            return f"准备写入 {len(rows)} 条实验结果。" if zh else f"Prepared to write {len(rows)} experiment result rows."
        if action.type == "draft_setup":
            return "准备创建/更新实验优化项目配置。" if zh else "Prepared to create or update the optimization project setup."
        if action.type == "skip_pending_designs":
            ids = args.get("candidate_ids")
            return f"准备跳过待做实验：{ids or '当前批次全部待做项'}。" if zh else f"Prepared to skip pending designs: {ids or 'all pending in the current batch'}."
        if action.type == "import_experiment_sheet":
            source = args.get("source_excel") or args.get("workbook_name")
            return f"准备从 `{source}` 导入实验数据。" if zh else f"Prepared to import experiment data from `{source}`."
        if permission == "project_setup":
            return f"准备执行项目结构变更：{action.type}。" if zh else f"Prepared project structure action: {action.type}."
        if permission in {"write_workbook", "write_files"}:
            return f"准备执行文件/工作簿操作：{action.type}。" if zh else f"Prepared workbook/file action: {action.type}."
        return f"准备执行：{action.type}。" if zh else f"Prepared action: {action.type}."

    def _permission_enabled(self, permission: str) -> bool:
        ui_state = getattr(self.coordinator.session, "ui_state", {}) or {}
        permissions = ui_state.get("operation_permissions")
        if not isinstance(permissions, dict):
            return True
        if permission in permissions:
            return bool(permissions[permission])
        if "all" in permissions:
            return bool(permissions["all"])
        return True
