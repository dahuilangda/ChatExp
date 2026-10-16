from __future__ import annotations

from dataclasses import dataclass

from .skills import load_skills


@dataclass(frozen=True)
class ActionContract:
    name: str
    permission: str
    risk_level: str
    risk_zh: str
    risk_en: str
    mutation: bool
    requires_confirmation: bool
    delegated_confirmation: bool = False


ACTION_CONTRACTS: dict[str, ActionContract] = {
    "respond": ActionContract("respond", "chat", "low", "低风险", "Low risk", False, False),
    "clarify": ActionContract("clarify", "chat", "low", "低风险", "Low risk", False, False),
    "analyze_data": ActionContract("analyze_data", "read_workbook", "low", "低风险", "Low risk", False, False),
    "inspect_workbook": ActionContract("inspect_workbook", "read_workbook", "low", "低风险", "Low risk", False, False),
    "status": ActionContract("status", "read_workbook", "low", "低风险", "Low risk", False, False),
    "view_experiments": ActionContract("view_experiments", "view_state", "low", "低风险", "Low risk", False, False),
    "draft_setup": ActionContract("draft_setup", "project_setup", "medium", "项目结构变更", "Project structure change", True, False, True),
    "create_project": ActionContract("create_project", "write_workbook", "medium", "文件写入", "File write", True, True),
    "load_project": ActionContract("load_project", "read_workbook", "low", "读取项目", "Project read", True, False),
    "define_factors": ActionContract("define_factors", "project_setup", "medium", "项目结构变更", "Project structure change", True, True),
    "define_objective": ActionContract("define_objective", "project_setup", "medium", "项目结构变更", "Project structure change", True, True),
    "update_objective": ActionContract("update_objective", "project_setup", "medium", "项目结构变更", "Project structure change", True, True),
    "update_factor": ActionContract("update_factor", "project_setup", "medium", "项目结构变更", "Project structure change", True, True),
    "initialize_design": ActionContract("initialize_design", "write_workbook", "medium", "设计批次写入", "Design batch write", True, True),
    "append_results": ActionContract("append_results", "write_experiment_rows", "medium", "实验数据变更", "Experiment data change", True, True),
    "update_plan_experiment": ActionContract("update_plan_experiment", "write_experiment_rows", "medium", "实验数据变更", "Experiment data change", True, True),
    "generate_next_design": ActionContract("generate_next_design", "write_workbook", "medium", "设计批次写入", "Design batch write", True, True),
    "import_experiment_sheet": ActionContract("import_experiment_sheet", "write_experiment_rows", "medium", "实验数据变更", "Experiment data change", True, True),
    "export": ActionContract("export", "write_files", "medium", "文件写入", "File write", True, True),
    "refresh_scores": ActionContract("refresh_scores", "write_experiment_rows", "medium", "实验数据变更", "Experiment data change", True, True),
    "skip_pending_designs": ActionContract("skip_pending_designs", "write_experiment_rows", "high", "高风险数据变更", "High-risk data change", True, True),
}


def skill_action_names() -> set[str]:
    names: set[str] = set()
    for skill in load_skills():
        names.update(skill.actions)
    names.discard("none")
    return names


def validate_action_contracts() -> list[str]:
    errors: list[str] = []
    skill_actions = skill_action_names()
    missing = sorted(skill_actions - set(ACTION_CONTRACTS))
    if missing:
        errors.append(f"missing action contracts: {', '.join(missing)}")
    return errors
