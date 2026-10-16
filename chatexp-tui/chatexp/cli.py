from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .i18n import get_translator
from .settings import Language, load_openai_settings
from .storage import (
    count_llm_traces,
    export_chat_history,
    list_llm_traces,
    list_task_summaries,
    load_task_metadata,
    read_chat_history,
    resolve_task_dir,
    short_task_id,
)


app = typer.Typer(help="ChatExp CLI")
console = Console()


def _default_workspace() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "workspace"


def _get_language(language: str | None = None) -> Language:
    """Get language from CLI argument or environment variable."""
    if language:
        try:
            return Language(language.lower())
        except ValueError:
            console.print(f"Invalid language '{language}', using 'zh' (Chinese)")
            return Language.ZH
    # Load from settings (which checks env vars)
    settings = load_openai_settings()
    return settings.language


def _t(language: Language):
    """Get translator function for the given language."""
    return get_translator(language)


@app.command()
def chat(
    workspace: Path = typer.Option(_default_workspace(), help="ChatExp 工作目录"),
    task_id: str | None = typer.Option(None, "--task-id", "-t", help="任务 ID；不传则新建一个任务"),
    language: str | None = typer.Option(None, "--language", "-l", help="界面语言 (zh: 中文, en: English)"),
) -> None:
    """启动 Textual 聊天仪表盘。"""
    lang = _get_language(language)
    _ = _t(lang)

    try:
        from .tui import run_chat_dashboard
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        console.print(_("no_textbook_dependency"))
        raise typer.Exit(code=1) from exc
    run_chat_dashboard(workspace=workspace, task_id=task_id, language=lang)


@app.command()
def paths(
    language: str | None = typer.Option(None, "--language", "-l", help="界面语言 (zh: 中文, en: English)"),
) -> None:
    """输出默认工作目录。"""
    lang = _get_language(language)
    _ = _t(lang)
    workspace = _default_workspace()
    console.print(f"{_('paths_workspace')}: {workspace}")
    console.print(f"{_('paths_tasks')}: {workspace / 'tasks'}")


@app.command("tasks")
def tasks_cmd(
    workspace: Path = typer.Option(_default_workspace(), help="ChatExp 工作目录"),
    language: str | None = typer.Option(None, "--language", "-l", help="界面语言 (zh: 中文, en: English)"),
) -> None:
    """列出已有任务 ID。"""
    lang = _get_language(language)
    _ = _t(lang)

    summaries = list_task_summaries(workspace)
    if not summaries:
        console.print(_("no_tasks"))
        return
    table = Table(title=_("tasks_table_title"))
    table.add_column(_("col_task_id"), no_wrap=True)
    table.add_column(_("col_prefix"), no_wrap=True)
    table.add_column(_("col_title"))
    table.add_column(_("col_status"), no_wrap=True)
    table.add_column(_("col_project"))
    table.add_column(_("col_message_count"), justify="right", no_wrap=True)
    table.add_column(_("col_trace_count"), justify="right", no_wrap=True)
    table.add_column(_("col_updated_at"), no_wrap=True)
    for item in summaries:
        table.add_row(
            item["task_id"],
            short_task_id(item["task_id"]),
            item["title"] or "-",
            item["status"] or "-",
            item["project_name"] or "-",
            item["message_count"] or "0",
            item["trace_count"] or "0",
            item["updated_at"] or "-",
        )
    console.print(table)


@app.command("show-task")
def show_task_cmd(
    task_id: str = typer.Option(..., "--task-id", "-t", help="任务 ID，支持前缀"),
    workspace: Path = typer.Option(_default_workspace(), help="ChatExp 工作目录"),
    language: str | None = typer.Option(None, "--language", "-l", help="界面语言 (zh: 中文, en: English)"),
) -> None:
    """显示指定任务的元数据和最近聊天记录。"""
    lang = _get_language(language)
    _ = _t(lang)

    task_dir = resolve_task_dir(workspace, task_id)
    metadata = load_task_metadata(task_dir)
    history = read_chat_history(task_dir)

    info = Table(title=_("task_detail_title"))
    info.add_column(_("col_field"), no_wrap=True)
    info.add_column(_("col_value"))
    for key in ["task_id", "title", "status", "project_name", "workbook_path", "message_count", "created_at", "updated_at", "task_dir"]:
        info.add_row(key, str(metadata.get(key) or "-"))
    info.add_row("trace_count", str(count_llm_traces(task_dir)))
    console.print(info)

    recent = Table(title=_("recent_messages_title"))
    recent.add_column(_("col_index"), no_wrap=True)
    recent.add_column(_("col_role"), no_wrap=True)
    recent.add_column(_("col_content"))
    for item in history[-6:]:
        recent.add_row(str(item.get("index", "-")), str(item.get("role", "-")), str(item.get("content", "")))
    console.print(recent)


@app.command("export-chat")
def export_chat_cmd(
    task_id: str = typer.Option(..., "--task-id", "-t", help="任务 ID，支持前缀"),
    workspace: Path = typer.Option(_default_workspace(), help="ChatExp 工作目录"),
    output: Path | None = typer.Option(None, "--output", "-o", help="导出路径，默认写入任务目录中的 chat_history.md"),
    language: str | None = typer.Option(None, "--language", "-l", help="界面语言 (zh: 中文, en: English)"),
) -> None:
    """导出指定任务的聊天记录。"""
    lang = _get_language(language)
    _ = _t(lang)

    task_dir = resolve_task_dir(workspace, task_id)
    exported = export_chat_history(task_dir, output_path=output)
    console.print(_("chat_exported", exported))


@app.command("show-traces")
def show_traces_cmd(
    task_id: str = typer.Option(..., "--task-id", "-t", help="任务 ID，支持前缀"),
    workspace: Path = typer.Option(_default_workspace(), help="ChatExp 工作目录"),
    limit: int = typer.Option(6, "--limit", "-n", help="显示最近多少条 trace"),
    language: str | None = typer.Option(None, "--language", "-l", help="界面语言 (zh: 中文, en: English)"),
) -> None:
    """显示指定任务最近的 LLM trace。"""
    lang = _get_language(language)
    _ = _t(lang)

    task_dir = resolve_task_dir(workspace, task_id)
    traces = list_llm_traces(task_dir)
    if not traces:
        console.print(_("no_traces"))
        return
    table = Table(title=_("traces_table_title"))
    table.add_column(_("col_time"), no_wrap=True)
    table.add_column(_("col_stage"), no_wrap=True)
    table.add_column(_("col_result"), no_wrap=True)
    table.add_column(_("col_action"), no_wrap=True)
    table.add_column(_("col_error"))
    for item in traces[-limit:]:
        parsed = item.get("parsed_plan") or {}
        table.add_row(
            str(item.get("created_at") or "-"),
            str(item.get("stage") or "-"),
            "ok" if parsed else "empty",
            str(parsed.get("action") or "-") if isinstance(parsed, dict) else "-",
            str(item.get("error") or "-"),
        )
    console.print(table)


if __name__ == "__main__":
    app()
