from __future__ import annotations

import json
import os
import secrets
import string
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from .desirability import auto_desirability_points, format_desirability
from .models import ChatMessage, ChatSession, FactorSpec, ObjectiveSpec, ProjectState


SHEET_META = "META"
SHEET_PARAMETERS = "PARAMETERS"
SHEET_CATEGORY = "CATEGORY"
SHEET_MPO = "MPO"
SHEET_EXPERIMENTS = "EXPERIMENT"
SHEET_DESIGNS = "DESIGNS"

LEGACY_SHEET_FACTORS = "FACTORS"
LEGACY_SHEET_OBJECTIVES = "OBJECTIVES"
LEGACY_SHEET_EXPERIMENTS = "EXPERIMENTS"
STORAGE_SCHEMA_VERSION = 2


def ensure_workspace(workspace: str | Path) -> Path:
    path = Path(workspace).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def tasks_root(workspace: str | Path) -> Path:
    return ensure_workspace(workspace) / "tasks"


def display_task_id(task_id: str | None) -> str:
    if not task_id:
        return "-"
    return task_id


def short_task_id(task_id: str | None) -> str:
    if not task_id:
        return "-"
    return task_id if len(task_id) <= 16 else task_id[:16]


def _generate_task_id(root: Path) -> str:
    alphabet = string.ascii_lowercase + string.digits
    while True:
        token = "".join(secrets.choice(alphabet) for _ in range(10))
        candidate = f"task_{token}"
        if not (root / candidate).exists():
            return candidate


def ensure_task_dir(workspace: str | Path, task_id: str | None = None) -> tuple[str, Path]:
    root = tasks_root(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    resolved_task_id = task_id or _generate_task_id(root)
    if "/" in resolved_task_id or "\\" in resolved_task_id or ".." in resolved_task_id:
        raise ValueError(f"非法任务 ID: {task_id}")
    task_dir = (root / resolved_task_id).resolve()
    if not task_dir.is_relative_to(root):
        raise ValueError(f"非法任务 ID: {task_id}")
    task_dir.mkdir(parents=True, exist_ok=True)
    return resolved_task_id, task_dir


def task_session_file(workspace: str | Path, task_id: str | None = None) -> tuple[str, Path, Path]:
    resolved_task_id, task_dir = ensure_task_dir(workspace, task_id=task_id)
    return resolved_task_id, task_dir, task_dir / "session.json"


def resolve_task_dir(workspace: str | Path, task_id: str) -> Path:
    root = tasks_root(workspace).resolve()
    resolved_id = str(task_id).strip()
    if not resolved_id or "/" in resolved_id or "\\" in resolved_id or ".." in resolved_id:
        raise ValueError(f"非法任务 ID: {task_id}")
    task_dir = (root / resolved_id).resolve()
    if not task_dir.is_relative_to(root):
        raise ValueError(f"非法任务 ID: {task_id}")
    if task_dir.exists():
        return task_dir
    matches = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith(resolved_id))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(f"任务 ID 前缀不唯一: {task_id}")
    raise FileNotFoundError(f"任务不存在: {task_id}")


def task_metadata_file(task_dir: str | Path) -> Path:
    return Path(task_dir) / "task.json"


def task_chat_history_file(task_dir: str | Path) -> Path:
    return Path(task_dir) / "chat_history.jsonl"


def task_llm_trace_dir(task_dir: str | Path) -> Path:
    path = Path(task_dir) / "llm_traces"
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_llm_traces(task_dir: str | Path) -> int:
    trace_dir = Path(task_dir) / "llm_traces"
    if not trace_dir.exists():
        return 0
    return sum(1 for path in trace_dir.iterdir() if path.is_file() and path.suffix == ".json")


def list_llm_traces(task_dir: str | Path) -> list[dict[str, object]]:
    trace_dir = Path(task_dir) / "llm_traces"
    if not trace_dir.exists():
        return []
    traces: list[dict[str, object]] = []
    for path in sorted(trace_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["trace_file"] = str(path)
        traces.append(payload)
    return traces


def list_task_ids(workspace: str | Path) -> list[str]:
    root = tasks_root(workspace)
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def load_session(session_path: str | Path) -> ChatSession:
    session_file = Path(session_path)
    if not session_file.exists():
        return ChatSession(state_path=str(session_file.resolve()))
    session = ChatSession.model_validate_json(session_file.read_text(encoding="utf-8"))
    if session.message_count < len(session.history):
        session.message_count = len(session.history)
    if session.persisted_message_count == 0 and session.message_count > 0:
        session.persisted_message_count = session.message_count
    # Crash-window recovery: save_session now writes session.json with the
    # NEW cursors BEFORE appending chat_history.jsonl. A crash between those
    # two writes leaves cursors ahead of the on-disk line count (new-lines
    # missing), so detect that gap here and re-derive the next-index from the
    # actual file line count — the next save will then re-append the missing
    # lines with monotonic, gap-free indexes instead of reusing stale cursors.
    if session.task_dir and session.history_next_index:
        history_file = task_chat_history_file(session.task_dir)
        file_lines = _count_history_lines(history_file)
        if session.history_next_index > file_lines + 1:
            session.history_next_index = file_lines + 1
        # persisted_message_count may also be ahead of the on-disk line count
        # in the same crash window; snap it back so the next save re-appends.
        if session.persisted_message_count > file_lines:
            session.persisted_message_count = file_lines
    return session


def save_session(session: ChatSession) -> None:
    session_file = Path(session.state_path)
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session.updated_at = datetime.now().isoformat(timespec="seconds")
    # Compute the next-index cursor WITHOUT mutating session state, so we can
    # persist session.json with the NEW cursors BEFORE appending chat_history.
    # A crash between the session.json write and the jsonl append then leaves
    # cursors ahead of the on-disk line count; load_session detects that gap
    # and re-derives the cursor from the file line count, so the next save
    # re-appends the missing lines with monotonic indexes (no duplicates).
    next_index = session.history_next_index
    if next_index <= 0 and session.task_dir:
        next_index = _count_history_lines(task_chat_history_file(session.task_dir)) + 1
    delta = max(session.message_count - session.persisted_message_count, 0)
    pending_lines = []
    if delta > 0:
        new_messages = session.history[-delta:]
        if new_messages:
            pending_lines = [
                json.dumps(_message_record(session, message, index), ensure_ascii=False)
                for index, message in enumerate(new_messages, start=next_index)
            ]
    new_next_index = next_index + len(pending_lines)
    new_persisted = session.message_count if pending_lines else session.persisted_message_count
    # Step 1: persist session.json with the NEW cursors first.
    session.history_next_index = new_next_index
    session.persisted_message_count = new_persisted
    _save_task_metadata(session)
    payload = session.model_dump()
    payload["storage_schema_version"] = STORAGE_SCHEMA_VERSION
    _atomic_write_text(session_file, json.dumps(payload, ensure_ascii=False, indent=2))
    # Step 2: append the chat-history lines. A crash here leaves the cursors
    # ahead of the file; load_session will re-derive and the next save re-appends.
    if pending_lines:
        history_file = task_chat_history_file(session.task_dir) if session.task_dir else None
        if history_file is not None:
            with history_file.open("a", encoding="utf-8") as handle:
                for line in pending_lines:
                    handle.write(line + "\n")


def list_task_summaries(workspace: str | Path) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    for task_id in list_task_ids(workspace):
        task_dir = tasks_root(workspace) / task_id
        metadata_path = task_metadata_file(task_dir)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            metadata = {"task_id": task_id, "task_dir": str(task_dir)}
        summaries.append(
            {
                "task_id": str(metadata.get("task_id") or task_id),
                "title": str(metadata.get("title") or ""),
                "project_name": str(metadata.get("project_name") or ""),
                "status": str(metadata.get("status") or ""),
                "message_count": str(metadata.get("message_count") or ""),
                "trace_count": str(count_llm_traces(task_dir)),
                "updated_at": str(metadata.get("updated_at") or ""),
                "task_dir": str(metadata.get("task_dir") or task_dir),
            }
        )
    return summaries


def load_task_metadata(task_dir: str | Path) -> dict[str, object]:
    metadata_path = task_metadata_file(task_dir)
    if not metadata_path.exists():
        raise FileNotFoundError(f"任务元数据不存在: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def read_chat_history(task_dir: str | Path) -> list[dict[str, object]]:
    history_path = task_chat_history_file(task_dir)
    if not history_path.exists():
        return []
    records: list[dict[str, object]] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def export_chat_history(task_dir: str | Path, output_path: str | Path | None = None) -> Path:
    task_path = Path(task_dir)
    output = Path(output_path).expanduser().resolve() if output_path else task_path / "chat_history.md"
    metadata = load_task_metadata(task_path)
    history = read_chat_history(task_path)
    lines = [
        f"# ChatExp Task {metadata.get('task_id', '')}",
        "",
        f"- 标题: {metadata.get('title', '') or '-'}",
        f"- 状态: {metadata.get('status', '') or '-'}",
        f"- 项目: {metadata.get('project_name', '') or '-'}",
        f"- 更新时间: {metadata.get('updated_at', '') or '-'}",
        "",
        "## 聊天记录",
        "",
    ]
    for item in history:
        lines.append(f"### {item.get('index', '-')}. {item.get('role', '-')}")
        lines.append("")
        lines.append(str(item.get("content", "")))
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _save_task_metadata(session: ChatSession) -> None:
    if not session.task_dir:
        return
    task_dir = Path(session.task_dir)
    metadata = {
        "storage_schema_version": STORAGE_SCHEMA_VERSION,
        "task_id": session.task_id,
        "task_dir": str(task_dir),
        "title": session.title or "",
        "status": _task_status(session),
        "project_name": session.project.name if session.project is not None else "",
        "workbook_path": session.project.workbook_path if session.project is not None else "",
        "message_count": session.message_count,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }
    _atomic_write_text(
        task_metadata_file(task_dir),
        json.dumps(metadata, ensure_ascii=False, indent=2),
    )


def _count_history_lines(history_file: Path) -> int:
    if not history_file.exists():
        return 0
    with history_file.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _message_record(session: ChatSession, message: ChatMessage, index: int) -> dict[str, str | int]:
    return {
        "index": index,
        "task_id": str(session.task_id or ""),
        "title": session.title or "",
        "role": message.role,
        "content": message.content,
        "updated_at": session.updated_at,
    }


def _task_status(session: ChatSession) -> str:
    if session.pending.action:
        return "awaiting_confirmation"
    if session.project is not None:
        return "active"
    return "draft"


def _project_to_frames(project: ProjectState) -> dict[str, pd.DataFrame]:
    meta_df = pd.DataFrame(
        [
            {
                "storage_schema_version": STORAGE_SCHEMA_VERSION,
                "name": project.name,
                "workspace_dir": project.workspace_dir,
                "workbook_path": project.workbook_path,
                "initial_batch_size": project.initial_batch_size,
                "latest_batch_id": project.latest_batch_id,
                "created_at": project.created_at,
                "updated_at": project.updated_at,
            }
        ]
    )

    parameter_rows = []
    category_rows = []
    for factor in project.factors:
        parameter_rows.append(
            {
                "Name": factor.name,
                "Type": factor.kind,
                "Use": True,
                "Unit": factor.unit or "",
                "Start": factor.min_value if factor.min_value is not None else "",
                "End": factor.max_value if factor.max_value is not None else "",
                "Step": factor.step if factor.step is not None else "",
            }
        )
        if factor.kind == "categorical":
            for category in factor.categories:
                category_rows.append({"Parameter": factor.name, "Name": category})

    mpo_rows = []
    for objective in project.objectives:
        desirability = objective.desirability
        if not desirability:
            auto_points = auto_desirability_points(pd.Series(dtype=float), objective)
            desirability = format_desirability(auto_points) if auto_points else ""
        mpo_rows.append(
            {
                "PROPERTIES": objective.name,
                "UNIT": objective.unit or "",
                "TYPES": "regression",
                "USE": True,
                "DESIRABILITY": desirability or "",
                "IMPORTANCE": objective.weight,
                "GOAL": objective.goal,
                "TARGET_VALUE": objective.target_value if objective.target_value is not None else "",
            }
        )

    parameters_df = pd.DataFrame(parameter_rows)
    category_df = pd.DataFrame(category_rows)
    mpo_df = pd.DataFrame(mpo_rows)
    if parameters_df.empty:
        parameters_df = pd.DataFrame(columns=["Name", "Type", "Use", "Unit", "Start", "End", "Step"])
    if category_df.empty:
        category_df = pd.DataFrame(columns=["Parameter", "Name"])
    if mpo_df.empty:
        mpo_df = pd.DataFrame(
            columns=["PROPERTIES", "UNIT", "TYPES", "USE", "DESIRABILITY", "IMPORTANCE", "GOAL", "TARGET_VALUE"]
        )

    return {
        SHEET_META: meta_df,
        SHEET_PARAMETERS: parameters_df,
        SHEET_CATEGORY: category_df,
        SHEET_MPO: mpo_df,
    }


def create_or_update_workbook(project: ProjectState) -> None:
    project.updated_at = datetime.now().isoformat(timespec="seconds")
    workbook = project.workbook
    workbook.parent.mkdir(parents=True, exist_ok=True)
    frames = _project_to_frames(project)

    if workbook.exists():
        experiment_frame = _read_excel_sheet(
            workbook,
            SHEET_EXPERIMENTS,
            fallback_sheet_names=(LEGACY_SHEET_EXPERIMENTS,),
            default=_empty_experiments_frame(project),
        )
        design_frame = _read_excel_sheet(
            workbook,
            SHEET_DESIGNS,
            default=_empty_designs_frame(project),
        )
        frames[SHEET_EXPERIMENTS] = _align_frame_columns(experiment_frame, _empty_experiments_frame(project).columns.tolist())
        frames[SHEET_DESIGNS] = _align_frame_columns(design_frame, _empty_designs_frame(project).columns.tolist())
    else:
        frames[SHEET_EXPERIMENTS] = _empty_experiments_frame(project)
        frames[SHEET_DESIGNS] = _empty_designs_frame(project)

    _write_workbook_atomic(workbook, frames)


def load_project_from_workbook(workbook_path: str | Path) -> ProjectState:
    workbook = Path(workbook_path).expanduser().resolve()
    sheets = pd.read_excel(workbook, sheet_name=None)
    meta = sheets[SHEET_META].iloc[0].to_dict()
    factors = _load_factors_from_sheets(sheets)
    objectives = _load_objectives_from_sheets(sheets)
    return ProjectState(
        name=str(meta.get("name") or workbook.stem),
        workspace_dir=str(meta.get("workspace_dir") or workbook.parent),
        workbook_path=str(meta.get("workbook_path") or workbook),
        factors=factors,
        objectives=objectives,
        initial_batch_size=int(meta.get("initial_batch_size") or 8),
        latest_batch_id=int(meta.get("latest_batch_id") or 0),
        created_at=str(meta.get("created_at") or datetime.now().isoformat(timespec="seconds")),
        updated_at=str(meta.get("updated_at") or datetime.now().isoformat(timespec="seconds")),
    )


def _load_factors_from_sheets(sheets: dict[str, pd.DataFrame]) -> list[FactorSpec]:
    if SHEET_PARAMETERS in sheets:
        parameters = sheets.get(SHEET_PARAMETERS, pd.DataFrame()).fillna("")
        categories_df = sheets.get(SHEET_CATEGORY, pd.DataFrame()).fillna("")
        category_map: dict[str, list[str]] = {}
        for row in categories_df.to_dict(orient="records"):
            category_map.setdefault(str(row.get("Parameter")), []).append(str(row.get("Name")))

        factors = []
        for row in parameters.to_dict(orient="records"):
            kind = str(row.get("Type") or "")
            factors.append(
                FactorSpec(
                    name=str(row["Name"]),
                    kind=kind,
                    unit=str(row.get("Unit") or "") or None,
                    min_value=_float_or_none(row.get("Start")),
                    max_value=_float_or_none(row.get("End")),
                    step=_float_or_none(row.get("Step")),
                    categories=category_map.get(str(row["Name"]), []),
                )
            )
        return factors

    legacy = sheets.get(LEGACY_SHEET_FACTORS, pd.DataFrame()).fillna("")
    factors = []
    for row in legacy.to_dict(orient="records"):
        categories = row.get("categories", "")
        factors.append(
            FactorSpec(
                name=row["name"],
                kind=row["kind"],
                unit=row.get("unit") or None,
                min_value=_float_or_none(row.get("min_value")),
                max_value=_float_or_none(row.get("max_value")),
                step=_float_or_none(row.get("step")),
                categories=[item for item in str(categories).split("|") if item],
            )
        )
    return factors


def _load_objectives_from_sheets(sheets: dict[str, pd.DataFrame]) -> list[ObjectiveSpec]:
    if SHEET_MPO in sheets:
        mpo = sheets.get(SHEET_MPO, pd.DataFrame()).fillna("")
        objectives = []
        for row in mpo.to_dict(orient="records"):
            objectives.append(
                ObjectiveSpec(
                    name=str(row["PROPERTIES"]),
                    goal=str(row.get("GOAL") or "maximize"),
                    target_value=row.get("TARGET_VALUE") or None,
                    weight=float(row.get("IMPORTANCE") or 1.0),
                    unit=str(row.get("UNIT") or "") or None,
                    desirability=str(row.get("DESIRABILITY") or "") or None,
                )
            )
        return objectives

    legacy = sheets.get(LEGACY_SHEET_OBJECTIVES, pd.DataFrame()).fillna("")
    objectives = []
    for row in legacy.to_dict(orient="records"):
        objectives.append(
            ObjectiveSpec(
                name=row["name"],
                goal=row.get("goal") or "maximize",
                target_value=row.get("target_value") or None,
                weight=float(row.get("weight") or 1.0),
                unit=row.get("unit") or None,
                desirability=row.get("desirability") or None,
            )
        )
    return objectives


def _empty_experiments_frame(project: ProjectState) -> pd.DataFrame:
    factor_names = [factor.name for factor in project.factors]
    objective_names = [objective.name for objective in project.objectives]
    columns = ["experiment_id", "Trial", "candidate_id", "source", "status"] + factor_names + objective_names + ["FuzzyScore", "COMMENT"]
    return pd.DataFrame(columns=columns)


def _empty_designs_frame(project: ProjectState) -> pd.DataFrame:
    factor_names = [factor.name for factor in project.factors]
    columns = ["experiment_id", "Trial", "candidate_id", "score", "status"] + factor_names
    return pd.DataFrame(columns=columns)


def read_sheet(project: ProjectState, sheet_name: str) -> pd.DataFrame:
    workbook = project.workbook
    if not workbook.exists():
        raise FileNotFoundError(f"Workbook does not exist: {workbook}")
    return _read_excel_sheet(workbook, sheet_name, default=pd.DataFrame())


def write_sheet(project: ProjectState, sheet_name: str, frame: pd.DataFrame) -> None:
    workbook = project.workbook
    sheets = pd.read_excel(workbook, sheet_name=None) if workbook.exists() else {}
    sheets[sheet_name] = frame
    project.updated_at = datetime.now().isoformat(timespec="seconds")
    sheets.update(_project_to_frames(project))
    frames = {
        current_sheet: current_frame
        for current_sheet, current_frame in sheets.items()
        if current_sheet not in {LEGACY_SHEET_FACTORS, LEGACY_SHEET_OBJECTIVES, LEGACY_SHEET_EXPERIMENTS}
    }
    _write_workbook_atomic(workbook, frames)


def concat_rows(current: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """Append row dicts to a frame without pandas' all-NA concat FutureWarning.

    pandas >= 2.1 warns when a concat operand has empty or all-NA entries
    (e.g. a row whose value for a typed column is None, or a column one side
    does not have) because those entries no longer stay out of result-dtype
    determination. Giving the addition the current frame's columns and casting
    its all-NA columns to the matching dtype keeps result dtypes stable and
    silences the deprecation.
    """
    if not rows:
        return current
    addition = pd.DataFrame(rows)
    ordered_columns = list(current.columns) + [column for column in addition.columns if column not in current.columns]
    if len(current) == 0:
        # An empty frame is exactly the "empty entry" the deprecation warns
        # about; per pandas' guidance, exclude it and keep its column order.
        return addition.reindex(columns=ordered_columns).reset_index(drop=True)
    for column in current.columns:
        if column not in addition.columns:
            # Integer dtypes cannot hold NA; concat upcast them to float
            # before, so seed the missing column as float directly.
            dtype = "float64" if current[column].dtype.kind in "iub" else current[column].dtype
            addition[column] = pd.Series([pd.NA] * len(addition), dtype=dtype)
    for column in addition.columns:
        if column in current.columns and addition[column].isna().all():
            try:
                addition[column] = addition[column].astype(current[column].dtype)
            except (ValueError, TypeError):
                try:
                    addition[column] = addition[column].astype("float64")
                except (ValueError, TypeError):
                    # Keep the all-NA column as-is; NA values stay NA.
                    continue
    return pd.concat([current, addition], ignore_index=True)


def append_rows(project: ProjectState, sheet_name: str, rows: list[dict]) -> pd.DataFrame:
    current = read_sheet(project, sheet_name)
    updated = concat_rows(current, rows)
    write_sheet(project, sheet_name, updated)
    return updated


def export_project_json(project: ProjectState, output_path: str | Path) -> Path:
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = project.model_dump()
    payload["storage_schema_version"] = STORAGE_SCHEMA_VERSION
    _atomic_write_text(output_file, json.dumps(payload, ensure_ascii=False, indent=2))
    return output_file


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _write_workbook_atomic(workbook: Path, frames: dict[str, pd.DataFrame]) -> None:
    workbook.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=workbook.parent,
        prefix=f".{workbook.name}.",
        suffix=".tmp.xlsx",
        delete=False,
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        with pd.ExcelWriter(temp_path, engine="openpyxl", mode="w") as writer:
            for sheet_name, frame in frames.items():
                frame.to_excel(writer, sheet_name=sheet_name, index=False)
        os.replace(temp_path, workbook)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _read_excel_sheet(
    workbook: Path,
    sheet_name: str,
    *,
    fallback_sheet_names: tuple[str, ...] = (),
    default: pd.DataFrame | None = None,
) -> pd.DataFrame:
    for candidate in (sheet_name, *fallback_sheet_names):
        try:
            return pd.read_excel(workbook, sheet_name=candidate)
        except ValueError:
            continue
    return default.copy() if isinstance(default, pd.DataFrame) else pd.DataFrame()


def _float_or_none(value: object) -> float | None:
    if value in (None, "", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _align_frame_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    aligned = frame.copy()
    for column in columns:
        if column not in aligned.columns:
            aligned[column] = None
    return aligned[columns]
