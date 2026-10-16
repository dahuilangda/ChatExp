from __future__ import annotations

import os
from pathlib import Path

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:
    class FastMCP:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

        def tool(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

        def run(self, *args, **kwargs):
            raise RuntimeError("The `mcp` package is required to run chatexp-mcp.")

from .chat import ChatCoordinator, project_snapshot
from .models import ChatPending
from .service import ChatExpService, parse_factor_text, parse_objective_text
from .storage import save_session
from .terminal_harness import TerminalAction


DEFAULT_WORKSPACE = Path(__file__).resolve().parents[2] / "data" / "workspace"

mcp = FastMCP(
    name="ChatExp",
    instructions=(
        "用于聊天式实验设计。可以创建 Excel、定义因素和目标、初始化设计、追加实验结果、"
        "生成下一批设计，并允许用户通过自然语言修改目标和输入。"
    ),
)


def _service() -> ChatExpService:
    return ChatExpService(_workspace())


def _chat() -> ChatCoordinator:
    workspace = _workspace()
    return ChatCoordinator(workspace=workspace, session_path=workspace / "session.json")


def _workspace() -> Path:
    return Path(os.environ.get("CHATEXP_MCP_WORKSPACE") or DEFAULT_WORKSPACE).expanduser().resolve()


def _load_project_for_action(coordinator: ChatCoordinator, workbook_name: str) -> None:
    coordinator.session.project = coordinator.service.load_project(workbook_name)


def _run_audited_action(action_type: str, arguments: dict, workbook_name: str | None = None) -> dict:
    coordinator = _chat()
    if workbook_name:
        _load_project_for_action(coordinator, workbook_name)
    action = TerminalAction(type=action_type, arguments=arguments, source="mcp")
    result = coordinator.workflow.execute_workflow(action, coordinator.executor_for_terminal_action(action))
    if result.pending is not None:
        coordinator.session.pending = result.pending
    save_session(coordinator.session)
    return {
        "blocked": result.blocked,
        "needs_confirmation": result.needs_confirmation,
        "message": result.content,
        "pending_action": coordinator.session.pending.model_dump() if coordinator.session.pending.action else None,
    }


@mcp.tool(description="通过自然语言和 ChatExp 交互，返回系统回复。")
def chat(message: str) -> str:
    return _chat().handle(message)


@mcp.tool(description="确认当前待审核动作。")
def confirm_pending_action() -> str:
    coordinator = _chat()
    pending = coordinator.session.pending
    project = coordinator.session.project
    if (
        pending.action
        and pending.workbook
        and project is not None
        and _workbook_label(pending.workbook) != _workbook_label(project.workbook.name)
    ):
        pending_workbook = pending.workbook
        loaded_workbook = project.workbook.name
        coordinator.session.pending = ChatPending()
        save_session(coordinator.session)
        return (
            f"Refused: pending confirmation was for workbook {pending_workbook}, not {loaded_workbook}. "
            "The stale pending action has been cleared; re-issue the request for the workbook you intended."
        )
    return coordinator.handle("确认")


def _workbook_label(name: str) -> str:
    cleaned = str(name).strip()
    return cleaned if cleaned.lower().endswith(".xlsx") else f"{cleaned}.xlsx"


@mcp.tool(description="创建实验 Excel 项目。")
def create_project(
    workbook_name: str,
    factors: list[str],
    objectives: list[str],
    initial_batch_size: int = 8,
) -> dict:
    return _run_audited_action(
        "create_project",
        {
            "workbook_name": workbook_name,
            "factors": [parse_factor_text(item).model_dump() for item in factors],
            "objectives": [parse_objective_text(item).model_dump() for item in objectives],
            "initial_batch_size": initial_batch_size,
        },
    )


@mcp.tool(description="初始化一批实验设计。")
def initialize_design(workbook_name: str, n_points: int = 8) -> dict:
    return _run_audited_action("initialize_design", {"count": n_points}, workbook_name=workbook_name)


@mcp.tool(description="追加实验结果。rows 传入字段到值的字典列表。")
def append_results(workbook_name: str, rows: list[dict]) -> dict:
    return _run_audited_action("append_results", {"rows": rows}, workbook_name=workbook_name)


@mcp.tool(description="从外部 Excel 导入实验数据，并同步 PARAMETERS/CATEGORY/MPO/EXPERIMENT。")
def import_experiment_sheet(
    workbook_name: str,
    source_excel: str,
    objective_names: list[str] | None = None,
) -> dict:
    return _run_audited_action(
        "import_experiment_sheet",
        {"source_excel": source_excel, "objective_names": objective_names or []},
        workbook_name=workbook_name,
    )


@mcp.tool(description="根据已有实验结果生成下一批设计。")
def generate_next_design(workbook_name: str, n_points: int = 5) -> dict:
    return _run_audited_action("generate_next_design", {"count": n_points}, workbook_name=workbook_name)


@mcp.tool(description="修改目标。")
def update_objective(
    workbook_name: str,
    objective_name: str,
    goal: str,
    target_value: float | str | None = None,
    weight: float | None = None,
    desirability: str | None = None,
) -> dict:
    return _run_audited_action(
        "update_objective",
        {
            "objective": {
                "name": objective_name,
                "goal": goal,
                "target_value": target_value,
                "weight": weight if weight is not None else 1.0,
                "desirability": desirability,
            }
        },
        workbook_name=workbook_name,
    )


@mcp.tool(description="修改因素定义。")
def update_factor(
    workbook_name: str,
    factor_name: str,
    kind: str | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    step: float | None = None,
    categories: list[str] | None = None,
    unit: str | None = None,
) -> dict:
    return _run_audited_action(
        "update_factor",
        {
            "factor_name": factor_name,
            "kind": kind,
            "min_value": min_value,
            "max_value": max_value,
            "step": step,
            "categories": categories,
            "unit": unit,
        },
        workbook_name=workbook_name,
    )


@mcp.tool(description="刷新 EXPERIMENT 中的 FuzzyScore。")
def refresh_scores(workbook_name: str) -> dict:
    return _run_audited_action("refresh_scores", {}, workbook_name=workbook_name)


@mcp.tool(description="检查当前工作簿的 sheet、列和记录数。")
def inspect_workbook(workbook_name: str) -> dict:
    return _run_audited_action("inspect_workbook", {}, workbook_name=workbook_name)


@mcp.tool(description="查看当前项目快照。")
def project_state(workbook_name: str) -> str:
    service = _service()
    project = service.load_project(workbook_name)
    return project_snapshot(project)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
