from __future__ import annotations

import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.style import Style
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.css.query import NoMatches
from textual.containers import Horizontal, HorizontalScroll, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, Static, TextArea

from .chat import ChatCoordinator
from .i18n import get_translator
from .settings import Language, load_openai_settings
from .storage import display_task_id, save_session, task_session_file


VISIBLE_MESSAGE_LIMIT = 80


class SubmitTextArea(TextArea):
    """A TextArea that submits on Enter (instead of Ctrl+Enter)."""

    def __init__(self, *args, **kwargs):
        self.on_submit_callback = kwargs.pop("on_submit_callback", None)
        self.on_history_callback = kwargs.pop("on_history_callback", None)
        super().__init__(*args, **kwargs)

    def on_key(self, event: events.Key) -> None:
        if event.key == "ctrl+e":
            event.prevent_default()
            event.stop()
            action = getattr(self.app, "action_toggle_composer_expand", None)
            if callable(action):
                action()
            return
        if event.key in {"shift+enter", "ctrl+j"}:
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        if event.key in {"up", "down"} and self.on_history_callback and not self.text.strip():
            replacement = self.on_history_callback(-1 if event.key == "up" else 1, self.text)
            if replacement is not None:
                event.prevent_default()
                event.stop()
                self.text = replacement
            return
        if event.key == "enter":
            if self.on_submit_callback:
                event.prevent_default()
                event.stop()
                self.on_submit_callback(self.text)
        # Let other keys pass through to default handler


class PaneResizeHandle(Widget):
    DEFAULT_CSS = """
    PaneResizeHandle {
        width: 1;
        height: 100%;
        background: #ededed;
        color: #d6d6d6;
    }

    PaneResizeHandle:hover {
        color: #0a84ff;
    }

    PaneResizeHandle.-active {
        color: #0a84ff;
        background: #ededed;
    }
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.dragging = False

    def render(self) -> Text:
        height = max(int(self.size.height or 1), 1)
        style = "#0a84ff" if self.has_class("-active") else "#d6d6d6"
        marker = "⋮"
        if height <= 1:
            return Text(marker, style=style)
        top = max((height - 1) // 2, 0)
        bottom = max(height - 1 - top, 0)
        return Text("\n" * top + marker + "\n" * bottom, style=style)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self.dragging = True
        self.capture_mouse()
        self.add_class("-active")
        self.app.adjust_chat_width(event.screen_x if event.screen_x is not None else event.x)
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if not self.dragging:
            return
        self.app.adjust_chat_width(event.screen_x if event.screen_x is not None else event.x)
        event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if not self.dragging:
            return
        self.dragging = False
        self.capture_mouse(False)
        self.remove_class("-active")
        self.app.adjust_chat_width(event.screen_x if event.screen_x is not None else event.x)
        event.stop()


class ComposerResizeHandle(Widget):
    """A thin horizontal drag handle for resizing the bottom composer."""

    DEFAULT_CSS = """
    ComposerResizeHandle {
        width: 100%;
        height: 1;
        background: #f7f7f7;
        color: #d6d6d6;
    }

    ComposerResizeHandle:hover {
        color: #0a84ff;
    }

    ComposerResizeHandle.-active {
        color: #0a84ff;
        background: #f7f7f7;
    }
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.dragging = False
        self._last_y: int | None = None

    def render(self) -> Text:
        width = max(int(self.size.width or 1), 1)
        marker = "⋯"
        style = "#0a84ff" if self.has_class("-active") else "#d6d6d6"
        if width <= len(marker):
            return Text(marker[:width], style=style)
        left = max((width - len(marker)) // 2, 0)
        return Text(" " * left + marker, style=style)

    def _event_y(self, event: events.MouseDown | events.MouseMove | events.MouseUp) -> int:
        return int(event.screen_y if event.screen_y is not None else event.y)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self.dragging = True
        self._last_y = self._event_y(event)
        self.capture_mouse()
        self.add_class("-active")
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if not self.dragging:
            return
        current_y = self._event_y(event)
        if self._last_y is None:
            self._last_y = current_y
            return
        delta = self._last_y - current_y
        self._last_y = current_y
        if delta:
            self.app.adjust_composer_height_by(delta)
        event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if not self.dragging:
            return
        self.dragging = False
        self._last_y = None
        self.capture_mouse(False)
        self.remove_class("-active")
        event.stop()


class ChatExpDashboardApp(App[None]):
    CSS = """
    Screen {
        width: 100%;
        height: 100%;
        background: #ededed;
        color: #0f172a;
    }

    #app-shell {
        width: 100%;
        height: 100%;
        min-width: 0;
        min-height: 0;
        layout: vertical;
        background: #ededed;
    }

    #root {
        layout: horizontal;
        width: 100%;
        height: 1fr;
        min-width: 0;
        min-height: 0;
        padding: 0 1;
        background: #ededed;
    }

    #dashboard-pane {
        width: auto;
        height: 100%;
        min-width: 0;
        min-height: 0;
        padding: 0;
    }

    #resize-handle {
        width: 1;
        min-width: 1;
        height: 100%;
    }

    #chat-pane {
        width: auto;
        height: 100%;
        min-width: 0;
        min-height: 0;
        padding: 0;
    }

    .panel {
        border: round #e3e3e3;
        background: #ffffff;
        padding: 0 1;
        margin-bottom: 0;
    }

    #workspace-shell {
        width: 100%;
        height: 100%;
        min-width: 0;
        min-height: 0;
        padding: 1;
        background: #ffffff;
    }

    #chat-shell {
        width: 100%;
        height: 100%;
        min-width: 0;
        min-height: 0;
        margin-bottom: 0;
        padding: 1;
        background: #f2f2f7;
    }

    #chat-head {
        height: 1;
        min-height: 1;
        background: #ededed;
        border: none;
        padding: 0 1;
        color: #475569;
    }

    #best-card {
        height: auto;
        margin-top: 0;
    }

    #plot-card {
        width: 100%;
        height: 1fr;
        min-height: 0;
        margin-top: 0;
        layout: vertical;
    }

    #recent-card {
        height: auto;
        margin-top: 1;
    }

    #experiment-toolbar {
        height: auto;
        margin-top: 0;
        padding: 0 1;
        background: #ffffff;
        border: none;
        dock: bottom;
    }

    .toolbar-row {
        width: 100%;
        height: auto;
        align: left middle;
        background: #ffffff;
    }

    .toolbar-group {
        width: auto;
        height: auto;
        align: left middle;
        background: #ffffff;
    }

    .toolbar-spacer {
        width: 1fr;
    }

    .toolbar-btn {
        min-width: 5;
        width: auto;
        height: 2;
        padding: 0 1;
        margin-right: 0;
        background: #ffffff;
        color: #475569;
        border: none;
        text-style: none;
    }

    .toolbar-btn:hover {
        color: #0a84ff;
        background: #eaf3ff;
        text-style: bold;
    }

    .toolbar-btn:focus {
        color: #0a84ff;
        background: #eaf3ff;
        text-style: bold;
    }

    .toolbar-btn.-active {
        color: #0a84ff;
        background: #eaf3ff;
        text-style: bold;
    }

    #page-indicator {
        width: auto;
        min-width: 7;
        padding: 0;
        color: #475569;
        margin: 0 1;
        content-align: center middle;
    }

    #page-jump {
        width: 6;
        height: 2;
        margin: 0 1 0 0;
        padding: 0 1;
        border: none;
        background: #ededed;
        color: #0f172a;
    }

    #page-jump > .input--placeholder {
        color: #a0a0a0;
    }

    #workspace-main {
        width: 100%;
        height: 100%;
        min-height: 0;
        margin-bottom: 0;
    }

    #plot-card-body {
        width: 100%;
        height: auto;
        min-height: 0;
        margin-bottom: 0;
        background: #ffffff;
        border: none;
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 1;
        scrollbar-background: #ffffff;
        scrollbar-background-hover: #ffffff;
        scrollbar-background-active: #ffffff;
        scrollbar-color: #eeeeee;
        scrollbar-color-hover: #d9d9d9;
        scrollbar-color-active: #a0a0a0;
        scrollbar-corner-color: #ffffff;
    }

    #plot-card-content {
        width: 100%;
        height: auto;
        padding: 0 1;
        margin-bottom: 0;
        background: #ffffff;
    }

    #workspace-table {
        width: 100%;
        height: 1fr;
        min-height: 0;
        background: #ffffff;
        color: #0f172a;
        border: none;
        padding: 0 1;
        margin-top: 0;
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 1;
        scrollbar-background: #ffffff;
        scrollbar-background-hover: #ffffff;
        scrollbar-background-active: #ffffff;
        scrollbar-color: #eeeeee;
        scrollbar-color-hover: #d9d9d9;
        scrollbar-color-active: #a0a0a0;
        scrollbar-corner-color: #ffffff;
    }

    .card-title {
        color: #0f172a;
        text-style: bold;
        margin: 0;
    }

    #chat-scroll {
        width: 100%;
        height: 1fr;
        min-height: 0;
        background: #f2f2f7;
        padding: 0 1;
        border: none;
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 1;
        scrollbar-background: #f2f2f7;
        scrollbar-background-hover: #f2f2f7;
        scrollbar-background-active: #f2f2f7;
        scrollbar-color: #eeeeee;
        scrollbar-color-hover: #d9d9d9;
        scrollbar-color-active: #a0a0a0;
        scrollbar-corner-color: #f2f2f7;
    }

    #chat-view {
        width: 100%;
        height: auto;
        background: #f2f2f7;
        border: none;
    }

    #composer {
        width: 100%;
        border-top: none;
        background: #f7f7f7;
        padding: 0 1 0 1;
        margin-top: 0;
        margin-bottom: 0;
        height: auto;
    }

    #composer.-expanded {
        border-top: none;
    }

    #quick-row {
        width: 100%;
        height: 1;
        min-height: 1;
        align: left middle;
        padding: 0;
        background: #f7f7f7;
    }

    #quick-actions {
        width: 1fr;
        height: 1;
        min-height: 1;
        align: left middle;
        padding: 0;
        background: #f7f7f7;
        scrollbar-size-horizontal: 1;
        scrollbar-background: #f7f7f7;
        scrollbar-background-hover: #f7f7f7;
        scrollbar-background-active: #f7f7f7;
        scrollbar-color: #eeeeee;
        scrollbar-color-hover: #d9d9d9;
        scrollbar-color-active: #a0a0a0;
        scrollbar-corner-color: #f7f7f7;
    }

    .quick-btn {
        min-width: 5;
        width: auto;
        height: 1;
        min-height: 1;
        padding: 0 1;
        margin-right: 1;
        background: #f7f7f7;
        color: #475569;
        border: none;
        text-style: none;
    }

    .quick-btn:hover {
        color: #0f172a;
        background: #eaf3ff;
        border: none;
        text-style: bold;
    }

    .quick-btn:focus {
        color: #0f172a;
        background: #eaf3ff;
        border: none;
        text-style: bold;
    }

    .quick-btn.-primary {
        color: #0a84ff;
        text-style: bold;
    }

    .quick-btn.-danger {
        color: #d86b63;
    }

    .quick-btn.-draft {
        color: #5d8f6a;
    }

    .quick-btn:disabled {
        color: #d9d9d9;
        background: #f7f7f7;
        border: none;
    }

    #composer-hint {
        width: auto;
        min-width: 18;
        height: 1;
        color: #a0a0a0;
        text-align: right;
        content-align: right middle;
        padding-left: 1;
        background: #f7f7f7;
    }

    #input-frame {
        width: 100%;
        height: auto;
        background: #ededed;
        border: round #d9d9d9;
        padding: 0;
        margin: 0;
    }

    #input-frame.-expanded {
        border: round #0a84ff;
    }

    #chat-input {
        width: 100%;
        height: 2;
        max-height: 12;
        background: #ffffff;
        border: none;
        color: #0f172a;
        margin: 0;
        padding: 0 1;
        scrollbar-size-vertical: 1;
        scrollbar-background: #ffffff;
        scrollbar-background-hover: #ffffff;
        scrollbar-background-active: #ffffff;
        scrollbar-color: #eeeeee;
        scrollbar-color-hover: #d9d9d9;
        scrollbar-color-active: #a0a0a0;
        scrollbar-corner-color: #ffffff;
    }

    #chat-input:focus {
        border: none;
        background: #ffffff;
    }

    #chat-input .text-area--caret {
        background: #0a84ff;
        color: #ffffff;
    }

    #chat-input:focus .text-area--caret {
        background: #0a84ff;
    }

    #chat-input .textarea--placeholder {
        color: #a0a0a0;
    }

    """

    BINDINGS = [
        ("ctrl+r", "refresh_dashboard", "刷新统计/Refresh"),
        ("ctrl+l", "clear_chat_view", "清屏/Clear"),
        ("ctrl+i", "focus_input", "输入/Input"),
        ("ctrl+e", "toggle_composer_expand", "展开输入/Expand"),
        ("ctrl+t", "toggle_selection_mode", "精确选择/Sel"),
        ("ctrl+shift+c,super+shift+c", "copy_latest_reply", "复制回复/Copy"),
        ("enter", "focus_workspace_selection", "聚焦记录/Focus"),
        ("ctrl+d", "quit", "退出/Exit"),
        ("ctrl+c", "quit", "退出/Exit"),
    ]

    def __init__(
        self,
        workspace: Path,
        task_id: str | None = None,
        language: Language = Language.ZH,
    ):
        super().__init__()
        resolved_task_id, task_dir, session_file = task_session_file(workspace, task_id=task_id)
        self.workspace = workspace
        self.task_id = resolved_task_id
        self.task_dir = task_dir
        self.session_file = session_file
        self.language = language
        self._ = get_translator(language)
        self.model_name = str(load_openai_settings(Path(__file__).resolve().parents[1]).model or "-")
        self.coordinator = ChatCoordinator(workspace=task_dir, session_path=session_file, language=language)
        self.processing = False
        self.visible_messages: list[tuple[str, str]] = []
        self.chat_width: int | None = None
        self.chat_width_ratio: float | None = None
        self.context_message: str | None = self._build_idle_context_message()
        self.last_assistant_message = ""
        self._history_loaded = False
        self.selection_mode = False
        self.status_text = self._("status_ready")
        self._scroll_to_latest_pending = True
        self.pending_reply = ""
        self.workspace_hint = ""
        self.plan_focus_rows: list[dict[str, Any]] = []
        self.current_total_pages = 1
        self.bi_loading = False
        self.bi_request_token = 0
        self._last_viewport_size: tuple[int, int] | None = None
        self.input_history: list[str] = []
        self._input_history_index: int | None = None
        self._input_history_draft = ""
        self.composer_expanded = False
        self._composer_user_pinned: bool | None = None
        self.input_height_override: int | None = None
        self._last_input_density: tuple[int, int, bool] = (0, 0, False)
        self._quick_button_actions: dict[str, dict[str, str]] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="app-shell"):
            with Horizontal(id="root"):
                with Vertical(id="dashboard-pane"):
                    with Vertical(id="workspace-shell", classes="panel"):
                        with Vertical(id="workspace-main"):
                            yield Static(id="best-card")
                            with Vertical(id="plot-card"):
                                with VerticalScroll(id="plot-card-body"):
                                    yield Static(id="plot-card-content")
                                with Horizontal(id="paging-toolbar", classes="toolbar-row"):
                                    yield Static("", classes="toolbar-spacer")
                                    with Horizontal(classes="toolbar-group"):
                                        yield Button(self._("page_first"), id="page-first", classes="toolbar-btn")
                                        yield Button(self._("page_prev"), id="page-prev", classes="toolbar-btn")
                                        yield Static("1 / 1", id="page-indicator")
                                        yield Button(self._("page_next"), id="page-next", classes="toolbar-btn")
                                        yield Button(self._("page_last"), id="page-last", classes="toolbar-btn")
                                        yield Input(placeholder=self._("page_jump_placeholder"), id="page-jump")
                                        yield Button(self._("page_jump_button"), id="page-apply", classes="toolbar-btn")
                                yield DataTable(id="workspace-table")
                                with Vertical(id="experiment-toolbar"):
                                    with Horizontal(id="footer-toolbar", classes="toolbar-row"):
                                        with Horizontal(classes="toolbar-group"):
                                            yield Button(self._("btn_table") if self.language == Language.EN else "记录", id="mode-table", classes="toolbar-btn")
                                            yield Button(self._("btn_stats"), id="mode-bi", classes="toolbar-btn")
                                            yield Button(self._("btn_recent"), id="scope-recent", classes="toolbar-btn")
                                            yield Button(self._("btn_completed"), id="scope-completed", classes="toolbar-btn")
                                            yield Button(self._("btn_pending"), id="scope-pending", classes="toolbar-btn")
                                            yield Button(self._("btn_best"), id="scope-best", classes="toolbar-btn")
                                            yield Button(self._("btn_all"), id="scope-all", classes="toolbar-btn")
                                            yield Button(self._("btn_clear_batch"), id="clear-batch", classes="toolbar-btn")
                                            yield Button(self._("label_best_score") if self.language == Language.EN else "评分", id="sort-score", classes="toolbar-btn")
                            with Vertical(id="recent-card"):
                                yield Static(id="recent-card-body")
                yield PaneResizeHandle(id="resize-handle")
                with Vertical(id="chat-pane"):
                    with Vertical(id="chat-shell", classes="panel"):
                        yield Static(id="chat-head")
                        with VerticalScroll(id="chat-scroll"):
                            yield Static(id="chat-view")
            with Vertical(id="composer"):
                yield ComposerResizeHandle(id="composer-resize")
                with Horizontal(id="quick-row"):
                    with HorizontalScroll(id="quick-actions"):
                        yield Button("", id="quick-0", classes="quick-btn")
                        yield Button("", id="quick-1", classes="quick-btn")
                        yield Button("", id="quick-2", classes="quick-btn")
                        yield Button("", id="quick-3", classes="quick-btn")
                        yield Button("", id="quick-4", classes="quick-btn")
                    yield Static(self._composer_hint(), id="composer-hint")
                with Vertical(id="input-frame"):
                    yield SubmitTextArea(id="chat-input")

    def on_mount(self) -> None:
        input_widget = self.query_one("#chat-input", SubmitTextArea)
        input_widget.on_submit_callback = self._handle_chat_input_submit
        input_widget.on_history_callback = self._navigate_input_history
        self._sync_composer_layout()
        self.set_focus(input_widget)
        input_widget.focus()
        workspace_table = self.query_one("#workspace-table", DataTable)
        if hasattr(workspace_table, "cursor_type"):
            workspace_table.cursor_type = "row"
        if hasattr(workspace_table, "zebra_stripes"):
            workspace_table.zebra_stripes = False
        page_jump = self.query_one("#page-jump", Input)
        page_jump.cursor_blink = False
        self._sync_input_density(input_widget.text)
        self.set_interval(0.12, self._sync_viewport_size)
        self.call_after_refresh(self._bootstrap_view)

    def on_resize(self, event: events.Resize) -> None:
        self._sync_viewport_size(force=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.startswith("quick-"):
            self._handle_quick_action(button_id)
            return
        if button_id == "mode-table":
            self._update_workspace_mode("table")
            return
        if button_id == "mode-bi":
            self._update_workspace_mode("bi")
            return
        if button_id.startswith("scope-"):
            self._update_experiment_view(scope=button_id.removeprefix("scope-"), page=1)
            return
        if button_id == "clear-batch":
            self._update_experiment_view(clear_batch=True, page=1)
            return
        if button_id == "sort-score":
            current = self._experiment_view_state()
            next_sort = "recent" if current.get("sort") == "score_desc" else "score_desc"
            self._update_experiment_view(sort=next_sort, page=1)
            return
        if button_id == "page-first":
            self._update_experiment_view(page=1)
            return
        if button_id == "page-prev":
            current_page = int(self._experiment_view_state().get("page") or 1)
            self._update_experiment_view(page=max(1, current_page - 1))
            return
        if button_id == "page-next":
            current_page = int(self._experiment_view_state().get("page") or 1)
            self._update_experiment_view(page=current_page + 1)
            return
        if button_id == "page-last":
            self._update_experiment_view(page=max(int(self.current_total_pages or 1), 1))
            return
        if button_id == "page-apply":
            jump_input = self.query_one("#page-jump", Input)
            self._apply_page_jump(jump_input.value)
            return
    def on_data_table_cell_selected(self, event) -> None:
        data_table = getattr(event, "data_table", None)
        if getattr(data_table, "id", "") != "workspace-table":
            return
        coordinate = getattr(event, "coordinate", None)
        row_index = getattr(coordinate, "row", None)
        if isinstance(row_index, int):
            self._focus_workspace_row(kind="plan", index=row_index)

    def on_data_table_row_selected(self, event) -> None:
        data_table = getattr(event, "data_table", None)
        if getattr(data_table, "id", "") != "workspace-table":
            return
        row_index = getattr(event, "cursor_row", None)
        if not isinstance(row_index, int):
            coordinate = getattr(event, "coordinate", None)
            row_index = getattr(coordinate, "row", None)
        if isinstance(row_index, int):
            self._focus_workspace_row(kind="plan", index=row_index)

    def action_refresh_dashboard(self) -> None:
        self._refresh_dashboard()
        self._set_status(self._("status_refreshed"))

    def action_clear_chat_view(self) -> None:
        self.visible_messages.clear()
        self.workspace_hint = ""
        self.pending_reply = ""
        self._scroll_to_latest_pending = True
        self._render_chat()
        self._set_status("已清屏" if self.language == Language.ZH else "Screen cleared")
        self._schedule_input_focus(delay=0.02)

    def action_focus_input(self) -> None:
        if self.selection_mode:
            self.selection_mode = False
            self._set_terminal_mouse_reporting(True)
            self._render_chat()
        self._focus_input_after_layout()

    def action_toggle_composer_expand(self) -> None:
        self.input_height_override = None
        self.composer_expanded = not self.composer_expanded
        self._composer_user_pinned = self.composer_expanded
        self._sync_composer_layout()
        if self.composer_expanded:
            message = "输入框已展开（再次 Ctrl+E 收起）" if self.language == Language.ZH else "Composer expanded (Ctrl+E again to collapse)"
        else:
            message = "输入框已收起" if self.language == Language.ZH else "Composer collapsed"
        self._set_status(message)
        self._focus_input_after_layout()

    def action_toggle_selection_mode(self) -> None:
        self.selection_mode = not self.selection_mode
        self._set_terminal_mouse_reporting(not self.selection_mode)
        if self.selection_mode:
            self._set_status(self._("status_selection_mode"))
        else:
            self._set_status(self._("status_ready"))
        self._render_chat()

    def action_copy_latest_reply(self) -> None:
        if not self.last_assistant_message:
            self._set_status(self._("status_ready"))
            return
        if self._copy_text(self.last_assistant_message):
            self._set_status(self._("status_completed"))
        else:
            self._set_status(self._("status_call_failed"))

    def action_focus_workspace_selection(self) -> None:
        try:
            focused = self.focused
            table = self.query_one("#workspace-table", DataTable)
        except Exception:
            return
        if focused is not table:
            return
        row_index = getattr(table, "cursor_row", None)
        if isinstance(row_index, int):
            self._focus_workspace_row(kind="plan", index=row_index)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "page-jump":
            self._apply_page_jump(event.value)
            return

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "chat-input":
            return
        self._sync_input_density(event.text_area.text)

    def _handle_chat_input_submit(self, value: str) -> None:
        message = value.strip()
        if not message or self.processing:
            return
        try:
            input_widget = self.query_one("#chat-input", SubmitTextArea)
        except NoMatches:
            return
        input_widget.text = ""
        self._composer_user_pinned = None
        self.input_height_override = None
        self._sync_input_density("")
        self._remember_input(message)
        self._submit_message(message)

    def _remember_input(self, message: str) -> None:
        if not message:
            return
        if not self.input_history or self.input_history[-1] != message:
            self.input_history.append(message)
            self.input_history = self.input_history[-80:]
        self._input_history_index = None
        self._input_history_draft = ""

    def _navigate_input_history(self, direction: int, draft: str) -> str | None:
        if not self.input_history:
            return None
        if self._input_history_index is None:
            self._input_history_draft = draft
            if direction < 0:
                self._input_history_index = len(self.input_history) - 1
                return self.input_history[self._input_history_index]
            return None
        next_index = self._input_history_index + direction
        if next_index < 0:
            next_index = 0
        if next_index >= len(self.input_history):
            self._input_history_index = None
            return self._input_history_draft
        self._input_history_index = next_index
        return self.input_history[self._input_history_index]

    def _apply_page_jump(self, value: str) -> None:
        jump_input = self.query_one("#page-jump", Input)
        text = value.strip()
        if not text:
            jump_input.value = ""
            return
        try:
            page = int(text)
        except ValueError:
            jump_input.value = ""
            return
        jump_input.value = ""
        if page > 0:
            self._update_experiment_view(page=page)

    def _composer_hint(self) -> str:
        chars, rows, is_long = self._last_input_density
        narrow = int(self.size.width or 100) < 100
        expanded = self.composer_expanded or (self.input_height_override or 0) > self._input_height_bounds()[0]
        expand_hint = (
            ("Ctrl+E收起" if expanded else "Ctrl+E展开")
            if self.language == Language.ZH
            else ("Ctrl+E collapse" if expanded else "Ctrl+E expand")
        )
        if chars > 0:
            if self.language == Language.ZH:
                draft = f"{chars}字" if rows <= 1 else f"{chars}字/{rows}行"
                if is_long or expanded:
                    return f"{draft} · Enter发送 · {expand_hint}"
                return f"{draft} · Enter发送"
            draft = f"{chars} chars" if rows <= 1 else f"{chars} chars/{rows} lines"
            if is_long or expanded:
                return f"{draft} · Enter send · {expand_hint}"
            return f"{draft} · Enter send"
        if self._reply_option_items():
            return "编号选择 · Enter发送" if self.language == Language.ZH else "Type # · Enter send"
        if narrow:
            return f"Enter发送 · {expand_hint}" if self.language == Language.ZH else f"Enter send · {expand_hint}"
        return f"Enter发送 · ⇧换行 · {expand_hint}" if self.language == Language.ZH else f"Enter send · ⇧ newline · {expand_hint}"

    def _estimate_input_rows(self, text: str) -> int:
        if not text:
            return 1
        width = max(24, int(self.size.width or 80) - 6)
        rows = 0
        for line in text.splitlines() or [""]:
            rows += max(1, len(self._wrap_display_line(line, width)))
        return max(rows, 1)

    def _sync_input_density(self, text: str) -> None:
        chars = len(text or "")
        rows = self._estimate_input_rows(text or "") if chars else 1
        newline_count = (text or "").count("\n")
        is_long = chars >= 160 or rows >= 3 or newline_count >= 1
        self._last_input_density = (chars, rows, is_long)
        if self._composer_user_pinned is None:
            auto_expanded = chars >= 220 or rows >= 4 or newline_count >= 2
            if auto_expanded != self.composer_expanded:
                self.composer_expanded = auto_expanded
        self._sync_composer_layout()

    def _sync_composer_layout(self) -> None:
        if not self.is_mounted:
            return
        try:
            composer = self.query_one("#composer", Vertical)
            input_frame = self.query_one("#input-frame", Vertical)
            input_widget = self.query_one("#chat-input", SubmitTextArea)
        except Exception:
            return
        _, rows, _ = self._last_input_density
        min_height, max_height = self._input_height_bounds()
        override_height = self.input_height_override
        if override_height is not None:
            height = max(min_height, min(max_height, int(override_height)))
            input_widget.styles.height = height
            input_widget.styles.max_height = max_height
            if height > min_height:
                composer.add_class("-expanded")
                input_frame.add_class("-expanded")
            else:
                composer.remove_class("-expanded")
                input_frame.remove_class("-expanded")
        elif self.composer_expanded:
            composer.add_class("-expanded")
            input_frame.add_class("-expanded")
            input_widget.styles.height = max(6, min(max_height, rows + 2))
            input_widget.styles.max_height = max_height
        else:
            composer.remove_class("-expanded")
            input_frame.remove_class("-expanded")
            input_widget.styles.height = min_height
            input_widget.styles.max_height = max_height
        try:
            self.query_one("#composer-hint", Static).update(self._composer_hint())
        except Exception:
            pass

    def _handle_quick_action(self, button_id: str) -> None:
        if self.processing:
            return
        action = self._quick_button_actions.get(button_id)
        if not action:
            return
        message = str(action.get("message") or "").strip()
        if not message:
            return
        mode = str(action.get("mode") or "submit")
        if mode == "draft":
            self._insert_input_draft(message)
            return
        self._remember_input(message)
        self._submit_message(message)

    def _insert_input_draft(self, text: str) -> None:
        try:
            input_widget = self.query_one("#chat-input", SubmitTextArea)
        except NoMatches:
            return
        current = input_widget.text or ""
        if current.strip():
            next_text = current.rstrip() + "\n" + text
        else:
            next_text = text
        input_widget.text = next_text
        self._composer_user_pinned = None
        self._sync_input_density(next_text)
        self._focus_input_after_layout()

    def _reply_option_items(self) -> list[dict[str, object]]:
        raw = self.coordinator.session.ui_state.get("reply_options")
        if not isinstance(raw, dict):
            return []
        items = raw.get("items")
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

    def _quick_action_items(self) -> list[dict[str, str]]:
        pending_items = self._pending_quick_action_items()
        if pending_items:
            return pending_items
        reply_items = self._reply_option_quick_items()
        if reply_items:
            return reply_items
        return self._project_quick_action_items()

    def _reply_option_quick_items(self) -> list[dict[str, str]]:
        options = self._reply_option_items()
        if not options:
            return []
        label_width = self._quick_option_label_width(len(options))
        items: list[dict[str, str]] = []
        for index, item in enumerate(options[:5], start=1):
            label = str(item.get("label") or item.get("id") or index).strip()
            text = str(item.get("text") or "").strip()
            if not text and not label:
                continue
            display = text or label
            items.append(
                {
                    "label": f"{label}. {self._truncate_display_text(display, label_width)}",
                    "message": str(item.get("id") or label or index),
                    "mode": "submit",
                    "tooltip": text or label,
                    "tone": "primary" if index == 1 else "",
                    "source": "reply_options",
                }
            )
        return items

    def _pending_quick_action_items(self) -> list[dict[str, str]]:
        pending = self.coordinator.session.pending
        action = str(pending.action or "")
        if not action:
            return []
        data = pending.data if isinstance(pending.data, dict) else {}
        if action == "confirm_setup":
            return [
                self._quick_item("确认创建", "确认", tone="primary") if self.language == Language.ZH else self._quick_item("Create", "confirm", tone="primary"),
                self._quick_item("调整方案…", "我想调整：", mode="draft", tone="draft") if self.language == Language.ZH else self._quick_item("Revise…", "I want to revise: ", mode="draft", tone="draft"),
                self._quick_item("取消", "取消", tone="danger") if self.language == Language.ZH else self._quick_item("Cancel", "cancel", tone="danger"),
            ]
        if action in {"confirm_init_design", "confirm_generate_design"}:
            count = self._fmt(data.get("count") or "")
            if action == "confirm_init_design":
                label = f"确认初始化{count}组" if self.language == Language.ZH else f"Init {count} runs"
                draft = f"改成 {count} 组" if self.language == Language.ZH else f"Change to {count} runs"
            else:
                label = f"确认生成{count}组" if self.language == Language.ZH else f"Generate {count} runs"
                draft = f"改成 {count} 组" if self.language == Language.ZH else f"Change to {count} runs"
            return [
                self._quick_item(label, "确认" if self.language == Language.ZH else "confirm", tone="primary"),
                self._quick_item("改组数…" if self.language == Language.ZH else "Change count…", draft, mode="draft", tone="draft"),
                self._quick_item("取消" if self.language == Language.ZH else "Cancel", "取消" if self.language == Language.ZH else "cancel", tone="danger"),
            ]
        if action == "confirm_import_experiment_sheet":
            import_summary = data.get("import_summary") or {}
            needs_mapping = bool(isinstance(import_summary, dict) and import_summary.get("needs_mapping_clarification"))
            objective_names = import_summary.get("objective_names") if isinstance(import_summary, dict) else []
            has_objective = bool(objective_names or data.get("objective_names"))
            mapping_draft = "列映射：" if self.language == Language.ZH else "Column mapping: "
            items = []
            if has_objective and not needs_mapping:
                items.append(self._quick_item("确认导入" if self.language == Language.ZH else "Import", "确认" if self.language == Language.ZH else "confirm", tone="primary"))
            items.append(self._quick_item("补列映射…" if self.language == Language.ZH else "Map columns…", mapping_draft, mode="draft", tone="draft"))
            items.append(self._quick_item("取消导入" if self.language == Language.ZH else "Cancel", "取消" if self.language == Language.ZH else "cancel", tone="danger"))
            return items
        if action == "confirm_result_report":
            analysis = data.get("analysis") or {}
            modified_count = len(analysis.get("modified") or []) if isinstance(analysis, dict) else 0
            items = [
                self._quick_item("确认记录" if self.language == Language.ZH else "Record", "确认" if self.language == Language.ZH else "confirm", tone="primary")
            ]
            if modified_count:
                items.append(
                    self._quick_item(
                        f"更新计划并记录({modified_count})" if self.language == Language.ZH else f"Update plan + record ({modified_count})",
                        "更新计划并记录" if self.language == Language.ZH else "update plan and record",
                        tone="primary",
                    )
                )
            items.append(self._quick_item("取消" if self.language == Language.ZH else "Cancel", "取消" if self.language == Language.ZH else "cancel", tone="danger"))
            return items
        if action == "confirm_terminal_action":
            payload = data.get("action") if isinstance(data, dict) else {}
            action_type = str(payload.get("type") or "") if isinstance(payload, dict) else ""
            continuation = data.get("continuation_actions") if isinstance(data, dict) else []
            continuation_count = len(continuation) if isinstance(continuation, list) else 0
            confirmation = data.get("confirmation") if isinstance(data, dict) else {}
            decisions = confirmation.get("decisions") if isinstance(confirmation, dict) else []
            if not isinstance(decisions, list) or not decisions:
                decisions = ["confirm", "cancel"]
            items: list[dict[str, str]] = []
            for decision in decisions:
                decision_text = str(decision)
                if decision_text == "confirm":
                    label = f"确认执行 {action_type}" if self.language == Language.ZH else f"Run {action_type}"
                    items.append(self._quick_item(label, "确认" if self.language == Language.ZH else "confirm", tone="primary"))
                elif decision_text == "apply_all":
                    tooltip = (
                        f"执行当前动作和后续 {continuation_count} 个低/中风险动作"
                        if self.language == Language.ZH
                        else f"Run the current action and {continuation_count} remaining low/medium-risk action(s)"
                    )
                    items.append(self._quick_item("全部应用" if self.language == Language.ZH else "Apply all", "全部应用" if self.language == Language.ZH else "apply all", tooltip=tooltip))
                elif decision_text == "cancel":
                    items.append(self._quick_item("取消" if self.language == Language.ZH else "Cancel", "取消" if self.language == Language.ZH else "cancel", tone="danger"))
            return items
        return []

    def _project_quick_action_items(self) -> list[dict[str, str]]:
        project = self.coordinator.session.project
        if project is None:
            if self.language == Language.EN:
                return [
                    self._quick_item("Starter plan", "Help me draft a starter optimization plan.", tone="primary"),
                    self._quick_item("Import data…", "Import experimental data from file.xlsx, objective column is ", mode="draft", tone="draft"),
                    self._quick_item("Paste data…", "I will paste experimental data:\n", mode="draft", tone="draft"),
                    self._quick_item("Help", "help"),
                ]
            return [
                self._quick_item("起步方案", "帮我整理一个起步优化方案。", tone="primary"),
                self._quick_item("导入数据…", "从 文件名.xlsx 导入实验数据，目标列为 ", mode="draft", tone="draft"),
                self._quick_item("粘贴数据…", "我要粘贴一批实验数据：\n", mode="draft", tone="draft"),
                self._quick_item("帮助", "帮助"),
            ]
        try:
            analysis = self.coordinator.service.analyze_project(project, include_bo_diagnostics=False, language=self.language)
            pending_view = self.coordinator.service.experiment_view(project, scope="pending", page=1, page_size=1)
            pending_count = int(pending_view.get("total_rows") or 0)
        except Exception as exc:
            error_text = str(exc)
            if self.language == Language.EN:
                return [
                    self._quick_item("Status unavailable", f"Dashboard refresh failed: {error_text}", tone="danger"),
                    self._quick_item("Retry status", "Show current project status."),
                ]
            return [
                self._quick_item("状态读取失败", f"仪表盘刷新失败：{error_text}", tone="danger"),
                self._quick_item("重试状态", "查看当前项目状态。"),
            ]
        completed_count = int(analysis.get("completed_count") or 0)
        design_count = int(analysis.get("design_count") or 0)
        experiment_count = int(analysis.get("experiment_count") or 0)
        init_count = int(project.initial_batch_size or 8)
        next_count = 5
        if experiment_count == 0 and design_count == 0:
            if self.language == Language.EN:
                return [
                    self._quick_item(f"Initialize {init_count}", f"Initialize {init_count} experimental runs.", tone="primary"),
                    self._quick_item("Import existing…", "Import experimental data from file.xlsx, objective column is ", mode="draft", tone="draft"),
                    self._quick_item("Revise setup…", "I want to revise: ", mode="draft", tone="draft"),
                    self._quick_item("Status", "Show current project status."),
                ]
            return [
                self._quick_item(f"初始化{init_count}组", f"初始化设计 {init_count} 组。", tone="primary"),
                self._quick_item("导入已有数据…", "从 文件名.xlsx 导入实验数据，目标列为 ", mode="draft", tone="draft"),
                self._quick_item("修改方案…", "我想调整：", mode="draft", tone="draft"),
                self._quick_item("查看状态", "查看当前项目状态。"),
            ]
        if pending_count > 0 and completed_count == 0:
            if self.language == Language.EN:
                return [
                    self._quick_item("Report results…", "Results for experiment ", mode="draft", tone="draft"),
                    self._quick_item("Import results…", "Import experimental data from file.xlsx, objective column is ", mode="draft", tone="draft"),
                    self._quick_item("Skip…", "Skip experiment ", mode="draft"),
                    self._quick_item("Status", "Show current project status."),
                ]
            return [
                self._quick_item("回填结果…", "实验  的结果是：", mode="draft", tone="draft"),
                self._quick_item("导入结果…", "从 文件名.xlsx 导入实验数据，目标列为 ", mode="draft", tone="draft"),
                self._quick_item("跳过实验…", "跳过实验 ", mode="draft"),
                self._quick_item("查看状态", "查看当前项目状态。"),
            ]
        if completed_count > 0:
            if self.language == Language.EN:
                return [
                    self._quick_item("Analyze", "Analyze the current data and recommend the next step.", tone="primary"),
                    self._quick_item(f"Next {next_count}", f"Generate {next_count} new experimental runs.", tone="primary"),
                    self._quick_item("Report results…", "Results for experiment ", mode="draft", tone="draft"),
                    self._quick_item("Import data…", "Import experimental data from file.xlsx, objective column is ", mode="draft", tone="draft"),
                    self._quick_item("Refresh scores", "Refresh scores."),
                ]
            return [
                self._quick_item("分析数据", "分析当前数据并给下一步建议。", tone="primary"),
                self._quick_item(f"下一批{next_count}组", f"生成下一批 {next_count} 组实验。", tone="primary"),
                self._quick_item("回填结果…", "实验  的结果是：", mode="draft", tone="draft"),
                self._quick_item("导入数据…", "从 文件名.xlsx 导入实验数据，目标列为 ", mode="draft", tone="draft"),
                self._quick_item("刷新评分", "刷新评分。"),
            ]
        if self.language == Language.EN:
            return [
                self._quick_item("Analyze status", "Analyze current status and tell me what is missing.", tone="primary"),
                self._quick_item(f"Next {next_count}", f"Generate {next_count} new experimental runs.", tone="primary"),
                self._quick_item("Report results…", "Results for experiment ", mode="draft", tone="draft"),
                self._quick_item("Import data…", "Import experimental data from file.xlsx, objective column is ", mode="draft", tone="draft"),
            ]
        return [
            self._quick_item("分析状态", "分析当前状态，告诉我还缺什么。", tone="primary"),
            self._quick_item(f"下一批{next_count}组", f"生成下一批 {next_count} 组实验。", tone="primary"),
            self._quick_item("回填结果…", "实验  的结果是：", mode="draft", tone="draft"),
            self._quick_item("导入数据…", "从 文件名.xlsx 导入实验数据，目标列为 ", mode="draft", tone="draft"),
        ]

    def _quick_item(
        self,
        label: str,
        message: str,
        *,
        mode: str = "submit",
        tone: str = "",
        tooltip: str | None = None,
    ) -> dict[str, str]:
        return {
            "label": label,
            "message": message,
            "mode": mode,
            "tone": tone,
            "tooltip": tooltip or message,
        }

    def _quick_option_label_width(self, option_count: int) -> int:
        available = max(int(self.size.width or 100) - 28, 44)
        per_option = available // max(min(option_count, 5), 1) - 5
        return max(10, min(24, per_option))

    def _quick_button_ids(self) -> list[str]:
        return [f"quick-{index}" for index in range(5)]

    def _set_quick_button_tone(self, button: Button, tone: str) -> None:
        for class_name in ("-primary", "-danger", "-draft"):
            try:
                button.remove_class(class_name)
            except Exception:
                pass
        if tone in {"primary", "danger", "draft"}:
            try:
                button.add_class(f"-{tone}")
            except Exception:
                pass

    def _sync_quick_actions(self) -> None:
        if not self.is_mounted:
            return
        button_ids = self._quick_button_ids()
        items = self._quick_action_items()[: len(button_ids)]
        self._quick_button_actions = {}
        for index, button_id in enumerate(button_ids):
            try:
                button = self.query_one(f"#{button_id}", Button)
            except Exception:
                continue
            if index < len(items):
                item = items[index]
                label_width = max(8, min(24, (int(self.size.width or 100) - 30) // max(len(items), 1)))
                label = str(item.get("label") or "")
                button.label = self._truncate_display_text(label, label_width)
                button.tooltip = str(item.get("tooltip") or item.get("message") or label)
                button.disabled = False
                button.styles.display = "block"
                self._set_quick_button_tone(button, str(item.get("tone") or ""))
                self._quick_button_actions[button_id] = item
            else:
                button.label = ""
                button.tooltip = None
                button.disabled = True
                button.styles.display = "none"
                self._set_quick_button_tone(button, "")
        try:
            self.query_one("#composer-hint", Static).update(self._composer_hint())
        except Exception:
            pass

    def _submit_message(self, message: str) -> None:
        if self.selection_mode:
            self.selection_mode = False
            self._set_terminal_mouse_reporting(True)
            self._render_chat()
        if self.workspace_hint:
            self.visible_messages.append(("ChatExp", self.workspace_hint))
            self.visible_messages = self.visible_messages[-VISIBLE_MESSAGE_LIMIT:]
            self.workspace_hint = ""
            self.coordinator.session.ui_state.pop("workspace_hint", None)
        # Steering: input submitted while a response is active is queued
        # (cap 1, a new message replaces the queued one) instead of being
        # dropped; it auto-submits as a fresh turn when the active one ends.
        if self.processing or getattr(self.coordinator, "busy", False):
            if self.coordinator.queue_steering(message):
                queued = self.coordinator.steering_queued_text or ""
                self._set_status(("已排队，本轮结束后应用: " if self.language == Language.ZH else "queued for after this turn: ") + queued[:48])
            return
        self._append_message("你" if self.language == Language.ZH else "You", message)
        self.processing = True
        self._set_interaction_enabled(False)
        self._start_pending_reply(self._("calling_model"))
        self._set_status(self._("calling_model"))
        self._process_message(message)

    @work(thread=True, exclusive=True)
    def _process_message(self, message: str) -> None:
        reply = self._process_message_body(message)
        self.call_from_thread(self._complete_message, message, reply)

    def _process_message_body(self, message: str) -> str:
        """Synchronous body of the message worker. Returns the assistant
        reply string, or an error reply if the coordinator raises. Wrapping
        the worker body here guarantees `processing`/`input-enabled` state is
        always restored by `_complete_message`, and lets the freeze-resistance
        be unit-tested without mounting the Textual app."""
        try:
            return self.coordinator.handle(
                message,
                progress_callback=lambda text: self.call_from_thread(self._update_pending_reply, text),
            )
        except Exception as exc:  # surface the error and always re-enable interaction
            return self._format_worker_error(self._("status_call_failed"), exc)

    def _complete_message(self, message: str, reply: str) -> None:
        self._finish_pending_reply(reply)
        self.processing = False
        self._set_interaction_enabled(True)
        self._refresh_dashboard()
        self._set_status("就绪")
        self._schedule_input_focus(delay=0.04)
        if message.strip().lower() in {"exit", "quit"} or message.strip() == "退出":
            self.exit()
            return
        # followUp: queued steering auto-submits as a fresh turn once the
        # active turn has fully completed (deferred one tick so state settles).
        queued = self.coordinator.drain_steering() if hasattr(self.coordinator, "drain_steering") else None
        if queued:
            self.call_later(self._submit_message, queued)

    def _format_worker_error(self, prefix: str, exc: BaseException) -> str:
        reason = f"{type(exc).__name__}: {exc}"
        return f"{prefix}: {reason}"

    def _append_message(self, role: str, content: str) -> None:
        if role == "ChatExp":
            self.last_assistant_message = content or ""
        self.visible_messages.append((role, content or ""))
        self.visible_messages = self.visible_messages[-VISIBLE_MESSAGE_LIMIT:]
        self._scroll_to_latest_pending = True
        self._render_chat()

    def _render_chat(self) -> None:
        self._sync_quick_actions()
        self._render_chat_head()
        chat_view = self.query_one("#chat-view", Static)
        message_rows = list(self.visible_messages)
        if self.workspace_hint:
            message_rows.append(("ChatExp", self.workspace_hint))
        if self.processing and self.pending_reply:
            message_rows.append(("ChatExp", self.pending_reply))

        if not message_rows and not self.context_message:
            chat_view.update(Text(""))
            self._maybe_scroll_chat_to_latest()
            return

        if self.selection_mode:
            chat_view.update(self._render_selection_text())
            self._maybe_scroll_chat_to_latest()
            return

        blocks: list[object] = []
        for role, content in message_rows:
            blocks.extend(self._render_chat_bubble(role, content))
            blocks.append(Text(""))
        chat_view.update(Group(*blocks[:-1] if blocks else blocks))
        self._maybe_scroll_chat_to_latest()

    def _render_chat_bubble(self, role: str, content: str) -> list[object]:
        is_user = self._is_user_role(role)
        max_chars = self._message_wrap_width(role="user" if is_user else "assistant")
        bubble_color = "#0a84ff" if is_user else "#ffffff"
        text_style = "#ffffff" if is_user else "#111827"
        label_text = self._message_role_label(role, is_user)
        label = Text(label_text, style="#94a3b8")
        bubble = Panel(
            self._render_message_body(content, max_chars, style=text_style),
            box=box.ROUNDED,
            border_style=Style(color=bubble_color),
            style=Style(bgcolor=bubble_color, color=text_style),
            padding=(0, 1),
            width=max_chars + 4,
        )
        if is_user:
            return [Align.right(label), Align.right(bubble)]
        return [Align.left(label), Align.left(bubble)]

    def _is_user_role(self, role: str) -> bool:
        return role in {"你", "You", "user", "User"}

    def _message_role_label(self, role: str, is_user: bool) -> str:
        if is_user:
            return "You" if self.language == Language.EN else "你"
        if role in {"ChatExp", "assistant", "Assistant"}:
            return "ChatExp"
        return self._fmt(role) or "ChatExp"

    def _render_chat_head(self) -> None:
        chat_head = self.query_one("#chat-head", Static)
        project = self.coordinator.session.project
        if project is None:
            chat_head.update("")
            chat_head.styles.display = "none"
            return

        chat_head.styles.display = "block"
        summary = self.context_message or self._build_idle_context_message()
        pane_width = max(int(self.query_one("#chat-pane", Vertical).size.width or 48) - 4, 24)
        label = "ChatExp"
        summary_width = max(pane_width - self._display_width(label) - 6, 12)
        line = Text()
        line.append("● ", style="#0a84ff")
        line.append(label, style="bold #0a84ff")
        line.append(" · ", style="#d9d9d9")
        line.append(self._truncate_display_text(summary, summary_width), style="#475569")
        chat_head.update(line)

    def _render_selection_text(self) -> Group:
        width = max(self._message_wrap_width(role="context"), 24)
        message_rows = list(self.visible_messages)
        if self.workspace_hint:
            message_rows.append(("ChatExp", self.workspace_hint))
        if self.processing and self.pending_reply:
            message_rows.append(("ChatExp", self.pending_reply))
        transcript_parts: list[str] = ["ChatExp", self._("precise_selection_mode")]
        workspace_transcript = self._selection_workspace_transcript()
        if workspace_transcript:
            transcript_parts.append("")
            transcript_parts.append(f"[{self._('left_workspace')}]")
            transcript_parts.append(workspace_transcript)
        if self.context_message:
            transcript_parts.append("")
            transcript_parts.append(f"[{self._('right_chat_head')}]")
            transcript_parts.append(self._wrap_message_content(self.context_message, width))
        if message_rows:
            transcript_parts.append("")
            transcript_parts.append(f"[{self._('right_chat_history')}]")
        for index, (role, content) in enumerate(message_rows, start=1):
            label = "你" if role == "你" else "ChatExp"
            transcript_parts.append("")
            transcript_parts.append(f"[{label} #{index}]")
            transcript_parts.append(self._wrap_message_content(content, width))
        return Group(Text("\n".join(transcript_parts).rstrip(), style="#0f172a"))

    def _selection_workspace_transcript(self) -> str:
        project = self.coordinator.session.project
        if project is None:
            return self._("no_project_yet")
        pending = self.coordinator.session.pending
        if pending.action == "confirm_import_experiment_sheet":
            return self._("import_preview")
        if pending.action in {"confirm_init_design", "confirm_generate_design"}:
            count = "-"
            if isinstance(pending.data, dict):
                count = self._fmt(pending.data.get("count"))
            mode = self._("init_design") if pending.action == "confirm_init_design" else self._("generate_design")
            return f"{self._('current_display')}{self._('pending_confirm')}\n{self._('execute')}\t{mode}\n{self._('group_count')}\t{count}"
        if pending.action == "confirm_terminal_action":
            data = pending.data if isinstance(pending.data, dict) else {}
            action_payload = data.get("action") if isinstance(data.get("action"), dict) else {}
            audit = data.get("audit") if isinstance(data.get("audit"), dict) else {}
            risk = audit.get("risk") if isinstance(audit.get("risk"), dict) else {}
            continuation = data.get("continuation_actions") if isinstance(data.get("continuation_actions"), list) else []
            display_actions = data.get("display_actions") if isinstance(data.get("display_actions"), list) else []
            action_count = len(display_actions) if display_actions else 1 + len(continuation)
            if self.language == Language.EN:
                return (
                    f"{self._('current_display')}Waiting for confirmation\n"
                    f"Action\t{action_payload.get('type') or '-'}\n"
                    f"Permission\t{audit.get('permission') or '-'}\n"
                    f"Risk\t{risk.get('label_en') or risk.get('level') or '-'}\n"
                    f"Workflow actions\t{action_count}\n"
                    f"Remaining\t{len(continuation)}"
                )
            return (
                f"{self._('current_display')}等待确认\n"
                f"动作\t{action_payload.get('type') or '-'}\n"
                f"权限\t{audit.get('permission') or '-'}\n"
                f"风险\t{risk.get('label_zh') or risk.get('level') or '-'}\n"
                f"动作数\t{action_count}\n"
                f"后续\t{len(continuation)}"
            )
        mode = self._workspace_mode_state()
        if mode == "bi":
            return self._("current_display") + self._("stats_display")
        analysis = self.coordinator.service.analyze_project(project, include_bo_diagnostics=False, language=self.language)
        experiment_view = self.coordinator.service.experiment_view(
            project,
            **self._effective_experiment_view_state(analysis),
        )
        rows = [dict(row) for row in (experiment_view.get("rows") or [])]
        if not rows:
            return self._("current_display") + self._("confirmed_plan")
        specs = [spec for spec in self._workspace_column_specs(project, analysis, rows) if str(spec["key"]) != "marker"]
        headers = [str(spec["title"]) for spec in specs]
        lines = [self._experiment_panel_header(experiment_view), "\t".join(headers)]
        for row in rows:
            values = self._workspace_table_row_values(project, analysis, row)
            lines.append("\t".join(str(values.get(str(spec["key"]), "")) for spec in specs))
        return "\n".join(lines)

    def _start_pending_reply(self, message: str) -> None:
        self.pending_reply = message
        self._scroll_to_latest_pending = True
        self._render_chat()

    def _update_pending_reply(self, message: str) -> None:
        self.pending_reply = message
        self._set_status(message)
        self._scroll_to_latest_pending = True
        self._render_chat()

    def _finish_pending_reply(self, reply: str) -> None:
        self.pending_reply = ""
        self._append_message("ChatExp", reply)

    def _bootstrap_view(self) -> None:
        self._sync_viewport_size(force=True)
        if not self._history_loaded:
            self._load_history()
            self._history_loaded = True
        self._restore_workspace_hint()
        self._scroll_to_latest_pending = True
        self._render_chat()
        self._set_status("就绪")
        self._schedule_input_focus(delay=0.18)

    def adjust_chat_width(self, screen_x: int) -> None:
        total_width = self._layout_total_width()
        requested = total_width - screen_x - 1
        chat_width = self._clamp_chat_width(requested, total_width)
        self.chat_width = chat_width
        self.chat_width_ratio = chat_width / total_width if total_width > 0 else None
        self._apply_layout_widths()
        self._refresh_dashboard()

    def adjust_composer_height_by(self, delta: int) -> None:
        if not self.is_mounted:
            return
        try:
            input_widget = self.query_one("#chat-input", SubmitTextArea)
        except NoMatches:
            return
        min_height, max_height = self._input_height_bounds()
        current = self.input_height_override
        if current is None:
            current = int(input_widget.size.height or (6 if self.composer_expanded else min_height))
        next_height = max(min_height, min(max_height, current + int(delta)))
        self.input_height_override = next_height
        self.composer_expanded = next_height > min_height
        self._composer_user_pinned = self.composer_expanded
        self._sync_composer_layout()
        if self.language == Language.ZH:
            self._set_status(f"输入框高度 {next_height} 行")
        else:
            self._set_status(f"Composer height {next_height} rows")
        self._focus_input_after_layout()

    def _input_height_bounds(self) -> tuple[int, int]:
        min_height = 2
        max_height = max(6, min(16, int((self.size.height or 36) * 0.42)))
        return min_height, max(max_height, min_height)

    def _apply_layout_widths(self) -> None:
        total_width = self._layout_total_width()
        min_chat, max_chat = self._chat_width_bounds(total_width)
        default_chat = self._clamp_chat_width(int(total_width * 0.38), total_width)
        if self.chat_width_ratio is not None:
            preferred_chat = int(round(total_width * self.chat_width_ratio))
        elif self.chat_width is not None:
            preferred_chat = self.chat_width
        else:
            preferred_chat = default_chat
        chat_width = max(min_chat, min(preferred_chat, max_chat))
        dashboard_width = max(1, total_width - chat_width - 1)
        self.chat_width = chat_width
        self.query_one("#chat-pane", Vertical).styles.width = chat_width
        self.query_one("#dashboard-pane", Vertical).styles.width = dashboard_width

    def _layout_total_width(self) -> int:
        return max(int(self.size.width or 0) - 2, 1)

    def _chat_width_bounds(self, total_width: int) -> tuple[int, int]:
        min_chat = min(32, max(18, total_width // 3))
        preferred_max_chat = max(min_chat, int(total_width * 0.52))
        max_chat = total_width - min(48, max(24, total_width // 2))
        if max_chat < 18:
            max_chat = max(18, total_width - 12)
        max_chat = min(max_chat, preferred_max_chat, total_width - 1)
        if max_chat < min_chat:
            min_chat = max(1, max_chat)
        return min_chat, max_chat

    def _clamp_chat_width(self, requested: int, total_width: int) -> int:
        min_chat, max_chat = self._chat_width_bounds(total_width)
        return max(min_chat, min(requested, max_chat))

    def _sync_viewport_size(self, force: bool = False) -> None:
        if not self.is_mounted:
            return
        viewport_size = (int(self.size.width or 0), int(self.size.height or 0))
        if not force and viewport_size == self._last_viewport_size:
            return
        self._last_viewport_size = viewport_size
        self._apply_layout_widths()
        try:
            self._sync_input_density(self.query_one("#chat-input", SubmitTextArea).text)
        except Exception:
            pass
        self._refresh_dashboard()

    def _load_history(self) -> None:
        history = self.coordinator.session.history[-12:]
        if not history:
            self._render_chat()
            return
        for item in history:
            role = "你" if item.role == "user" else "ChatExp"
            self._append_message(role, item.content)

    def _refresh_dashboard(self) -> None:
        try:
            self._refresh_dashboard_impl()
        except Exception as exc:  # never let the timer die — surface the error
            # Every UI helper below may itself fail when the app is not yet
            # mounted (or a widget was removed mid-render). Guard each one so
            # a secondary UI failure cannot re-raise and escape the timer.
            try:
                self._set_status(self._("status_panel_error"))
            except Exception:
                pass
            try:
                self._set_workspace_panel_body(
                    title=self._("status_panel_error"),
                    body=self._format_worker_error(self._("status_panel_error"), exc),
                )
            except Exception:
                pass
            try:
                self._render_chat()
            except Exception:
                pass

    def _refresh_dashboard_impl(self) -> None:
        project = self.coordinator.session.project
        pending = self.coordinator.session.pending
        workspace_mode = self._workspace_mode_state()
        if project is None:
            self.context_message = self._build_idle_context_message()
            self._configure_workspace_layout(mode="empty", show_table=False)
            self._set_section_visible("#best-card", False)
            self._set_section_visible("#experiment-toolbar", False)
            self._set_section_visible("#recent-card", False)
            self._set_section_visible("#plot-card", True)
            self._set_section_visible("#workspace-table", False)
            self._clear_workspace_table()
            self._set_workspace_panel_body(title=self._("panel_plan_title"), empty_text=self._("panel_plan_empty"))
            self._sync_experiment_toolbar(total_pages=1, show_paging=False)
            self._render_chat()
            return

        analysis = self.coordinator.service.analyze_project(
            project,
            include_bo_diagnostics=False,
        )
        experiment_view_state = self._effective_experiment_view_state(analysis)
        experiment_view = self.coordinator.service.experiment_view(project, **experiment_view_state)
        self.context_message = self._build_context_message(analysis, experiment_view)
        if pending.action == "confirm_import_experiment_sheet":
            self._configure_workspace_layout(mode="import", show_table=False)
            self._set_section_visible("#best-card", False)
            self._set_section_visible("#plot-card", True)
            self._set_section_visible("#experiment-toolbar", False)
            self._set_section_visible("#recent-card", False)
            self._set_section_visible("#workspace-table", False)
            self._clear_workspace_table()
            self._set_workspace_panel_body(title=self._("title_import_preview"), body=self._render_import_preview_panel())
            self._sync_experiment_toolbar(total_pages=1, show_paging=False)
            self._render_chat()
            return
        if pending.action in {"confirm_init_design", "confirm_generate_design"}:
            pending_rows = [dict(row) for row in (experiment_view.get("rows") or [])]
            self._set_section_visible("#best-card", False)
            self._set_section_visible("#plot-card", True)
            self._set_section_visible("#recent-card", False)
            self._set_workspace_panel_body(title=self._("title_pending_confirm"), body=self._render_pending_design_panel(project))
            if pending_rows:
                self._configure_workspace_layout(mode="table", show_table=True)
                self._set_section_visible("#experiment-toolbar", True)
                self._set_section_visible("#workspace-table", True)
                self.plan_focus_rows = pending_rows
                self._populate_workspace_table(project, analysis, pending_rows)
                self._sync_experiment_toolbar(
                    total_pages=int(experiment_view.get("total_pages") or 1),
                    show_paging=bool(self.plan_focus_rows),
                )
            else:
                self._configure_workspace_layout(mode="pending", show_table=False)
                self._set_section_visible("#experiment-toolbar", False)
                self._set_section_visible("#workspace-table", False)
                self.plan_focus_rows = []
                self._clear_workspace_table()
                self._sync_experiment_toolbar(total_pages=1, show_paging=False)
            self._render_chat()
            return
        self._set_section_visible("#best-card", False)
        self._set_section_visible("#plot-card", True)
        self._set_section_visible("#experiment-toolbar", True)
        self._set_section_visible("#recent-card", False)
        if workspace_mode == "bi":
            self._configure_workspace_layout(mode="bi", show_table=False)
            self.bi_loading = True
            self.plan_focus_rows = []
            self._set_section_visible("#workspace-table", False)
            self._clear_workspace_table()
            self._set_workspace_panel_body(title=self._("section_statistics"), body=self._render_bi_loading_panel(analysis, project))
            self.bi_request_token += 1
            self._load_bi_panel(project, self.bi_request_token)
        else:
            self.bi_loading = False
            self._configure_workspace_layout(mode="table", show_table=True)
            self._set_section_visible("#workspace-table", True)
            self._render_workspace_table(project, analysis, experiment_view)
        self._sync_experiment_toolbar(
            total_pages=int(experiment_view.get("total_pages") or 1),
            show_paging=workspace_mode == "table" and bool(self.plan_focus_rows),
        )
        self._render_chat()

    def _build_context_message(self, analysis: dict[str, Any], experiment_view: dict[str, Any]) -> str:
        best_row = analysis.get("best_experiment") or {}
        best_score = self._fmt(best_row.get("FuzzyScore"))
        return (
            f"{analysis.get('project_name') or '-'}  ·  "
            f"{self._('label_completed')} {analysis.get('completed_count') or 0}  ·  "
            f"{self._('label_pending')} {analysis.get('suggested_count') or 0}  ·  "
            f"{self._('label_best_score')} {best_score}  ·  "
            f"{self._('section_trial')} {analysis.get('latest_batch_id') or 0}  ·  "
            f"{self._experiment_view_summary(experiment_view)}"
        )

    def _build_idle_context_message(self) -> str:
        return (
            f"{display_task_id(self.task_id)}  ·  {self.model_name}  ·  "
            f"{self._('no_project_data')}"
        )

    def _workspace_mode_state(self) -> str:
        mode = str(self.coordinator.session.ui_state.get("workspace_mode") or "table").lower()
        return mode if mode in {"table", "bi"} else "table"

    def _update_workspace_mode(self, mode: str) -> None:
        self.coordinator.session.ui_state["workspace_mode"] = mode
        save_session(self.coordinator.session)
        self._refresh_dashboard()

    def _render_bi_panel(self, analysis: dict[str, Any], project) -> Group:
        objective = analysis.get("primary_objective") or {}
        objective_name = str(objective.get("name") or self._("label_objective"))
        best_row = analysis.get("best_experiment") or {}
        score_trend = analysis.get("score_trend") or []
        objective_trend = analysis.get("objective_trend") or []
        batch_counts = analysis.get("batch_counts") or []
        diagnostics = analysis.get("bo_diagnostics") or {}
        summary = self._bi_kpi_grid(
            [
                (self._("label_completed"), str(analysis.get("completed_count") or 0)),
                (self._("label_pending"), str(analysis.get("suggested_count") or 0)),
                (self._("label_total_experiments"), str(analysis.get("experiment_count") or 0)),
                (self._("label_total_designs"), str(analysis.get("design_count") or 0)),
                (self._("label_best_score"), self._fmt(best_row.get("FuzzyScore"))),
                (self._("label_current_batch"), str(analysis.get("latest_batch_id") or 0)),
                (f"Best {objective_name}" if self.language == Language.EN else f"最佳{objective_name}", self._fmt(best_row.get(objective_name))),
                (self._("label_iteration_logic"), self._("label_second_round")),
            ]
        )

        trend = Table.grid(padding=(0, 1), expand=True)
        trend.add_column(style="bold #0f172a", no_wrap=True)
        trend.add_column()
        trend.add_row(
            self._("label_best_score") if self.language == Language.EN else "评分趋势",
            (
                f"{self._sparkline(score_trend)}  {self._trend_endpoints(score_trend)}"
                f"  ({self._trend_delta_text(score_trend)})"
            )
            if score_trend
            else "-",
        )
        trend.add_row(
            f"{objective_name}{' trend' if self.language == Language.EN else '趋势'}",
            (
                f"{self._sparkline(objective_trend)}  {self._trend_endpoints(objective_trend)}"
                f"  ({self._trend_delta_text(objective_trend)})"
            )
            if objective_trend
            else "-",
        )
        trend.add_row(
            self._("label_batch_distribution") if self.language == Language.EN else "批次分布",
            (
                f"{self._sparkline(batch_counts)}  {', '.join(str(v) for v in batch_counts[-8:])}"
                f"  ({len(batch_counts)}{' batches' if self.language == Language.EN else '批'})"
            )
            if batch_counts
            else "-",
        )

        best = Table.grid(padding=(0, 1), expand=True)
        best.add_column(style="bold #0f172a", no_wrap=True)
        best.add_column()
        if isinstance(best_row, dict) and best_row:
            if self.language == Language.EN:
                best.add_row(self._("label_best_record"), f"Batch {self._fmt(best_row.get('Trial'))} · ID {self._fmt(best_row.get('candidate_id'))}")
            else:
                best.add_row(self._("label_best_record"), f"批次 {self._fmt(best_row.get('Trial'))} · 编号 {self._fmt(best_row.get('candidate_id'))}")
            condition_text = []
            for factor in project.factors[:4]:
                value = best_row.get(factor.name)
                if value in (None, ""):
                    continue
                condition_text.append(f"{factor.name}={self._fmt(value)}")
            best.add_row(self._("label_best_conditions"), " | ".join(condition_text) or "-")
        else:
            best.add_row(self._("label_best_record"), self._("section_no_experiments"))

        windows = Table.grid(padding=(0, 1), expand=True)
        windows.add_column(style="bold #0f172a", no_wrap=True)
        windows.add_column()
        factor_ranges = analysis.get("factor_ranges") or []
        if factor_ranges:
            for item in factor_ranges[:6]:
                windows.add_row(str(item.get("name") or "-"), str(item.get("summary") or "-"))
        else:
            windows.add_row(self._("section_window"), self._("label_no_window"))

        recent = self._render_recent_completed_bi(analysis, objective_name)
        model_diag = self._render_bo_diagnostics_bi(diagnostics)

        return Group(
            self._bi_section(self._("section_summary"), summary, "Summary"),
            Rule(style="#d9d9d9"),
            self._bi_section(self._("section_trend"), trend, "Trend"),
            Rule(style="#d9d9d9"),
            self._bi_section(self._("section_best_conditions"), best, "Best Conditions"),
            Rule(style="#d9d9d9"),
            self._bi_section(self._("section_bo_stats"), model_diag, "BO Statistics"),
            Rule(style="#d9d9d9"),
            self._bi_section(self._("label_recent_completed"), recent, "Recently Completed"),
            Rule(style="#d9d9d9"),
            self._bi_section(self._("section_factor_windows"), windows, "Factor Ranges"),
        )

    def _render_bi_loading_panel(self, analysis: dict[str, Any], project) -> Group:
        objective = analysis.get("primary_objective") or {}
        objective_name = str(objective.get("name") or ("objective" if self.language == Language.EN else "目标"))
        best_row = analysis.get("best_experiment") or {}
        summary = self._bi_kpi_grid(
            [
                (self._("label_completed"), str(analysis.get("completed_count") or 0)),
                (self._("label_pending"), str(analysis.get("suggested_count") or 0)),
                (self._("label_total_experiments"), str(analysis.get("experiment_count") or 0)),
                (self._("label_total_designs"), str(analysis.get("design_count") or 0)),
                (self._("label_best_score"), self._fmt(best_row.get("FuzzyScore"))),
                (f"Best {objective_name}" if self.language == Language.EN else f"最佳{objective_name}", self._fmt(best_row.get(objective_name))),
            ]
        )
        loading = Table.grid(padding=(0, 1), expand=True)
        loading.add_column(style="bold #0f172a", no_wrap=True)
        loading.add_column()
        loading.add_row(self._("label_status"), self._("label_state_computing"))
        loading.add_row(self._("label_content_bo"), self._("label_content_bo"))
        return Group(
            self._bi_section(self._("section_summary"), summary),
            Rule(style="#d9d9d9"),
            self._bi_section(self._("section_statistics"), loading),
        )

    @work(thread=True, exclusive=True)
    def _load_bi_panel(self, project, token: int) -> None:
        analysis = self._load_bi_panel_body(project)
        self.call_from_thread(self._complete_bi_panel, project.workbook_path, token, analysis)

    def _load_bi_panel_body(self, project) -> dict[str, Any]:
        """Synchronous body of the bi-panel worker. Returns the analysis
        dict, or a fallback `analysis={}` payload carrying the error note so
        `bi_loading` is always cleared and the error is surfaced. Extracted
        so the freeze-resistance can be unit-tested without the app running."""
        try:
            return self.coordinator.service.analyze_project(
                project,
                include_bo_diagnostics=True,
                language=self.language,
            )
        except Exception as exc:  # never leave bi_loading=True on failure
            return {
                "bo_diagnostics": {},
                "llm_weight_error": self._format_worker_error(self._("status_call_failed"), exc),
            }

    def _complete_bi_panel(self, workbook_path: str, token: int, analysis: dict[str, Any]) -> None:
        project = self.coordinator.session.project
        if project is None:
            return
        if self._workspace_mode_state() != "bi":
            return
        if token != self.bi_request_token:
            return
        if project.workbook_path != workbook_path:
            return
        self.bi_loading = False
        self.context_message = self._build_context_message(
            analysis,
            self.coordinator.service.experiment_view(project, **self._effective_experiment_view_state(analysis)),
        )
        self._set_workspace_panel_body(title=self._("section_statistics"), body=self._render_bi_panel(analysis, project))
        self._sync_experiment_toolbar(total_pages=1, show_paging=False)
        self._render_chat()

    def _render_workspace_table(self, project, analysis: dict[str, Any], experiment_view: dict[str, Any]) -> None:
        rows = [dict(row) for row in (experiment_view.get("rows") or [])]
        if not rows:
            self.plan_focus_rows = []
            self._clear_workspace_table()
            self._set_workspace_panel_body(title=self._("confirmed_plan"), body=self._render_confirmed_setup_panel(analysis, project))
            return
        self.plan_focus_rows = rows
        header = self._experiment_panel_header(experiment_view)
        body = Group(
            Text(header, style="bold #0a84ff"),
            Rule(style="#d9d9d9"),
        )
        self._populate_workspace_table(project, analysis, rows)
        self._set_workspace_panel_body(title=header, body=body)

    def _render_import_preview_panel(self) -> Group:
        import_summary = self.coordinator.session.pending.data.get("import_summary") or {}
        if not isinstance(import_summary, dict):
            return self._empty_section(self._("import_preview"), self._("panel_plan_empty"))
        rows = Table.grid(padding=(0, 0), expand=True)
        rows.add_column(style="bold #0f172a", no_wrap=True)
        rows.add_column()
        if self.language == Language.EN:
            header_label = "Header"
            row_label = "Row"
            data_label = "Data rows"
            mapped_label = "Mapped"
            llm_label = "LLM added"
            manual_label = "Manual"
            ignore_label = "Ignored"
            missing_label = "Missing"
            preview_title = "Import Preview"
        else:
            header_label = "表头"
            row_label = "行"
            data_label = "候选数据"
            mapped_label = "匹配列"
            llm_label = "LLM补充"
            manual_label = "手动指定"
            ignore_label = "忽略列"
            missing_label = "待补充"
            preview_title = "导入预览"
        rows.add_row(header_label, f"{row_label} {int(import_summary.get('header_row') or 0) + 1}")
        rows.add_row(data_label, str(import_summary.get("detected_rows") or 0))
        mapped_columns = import_summary.get("mapped_columns") or []
        llm_mapped_columns = import_summary.get("llm_mapped_columns") or []
        ignored_columns = import_summary.get("ignored_columns") or []
        if mapped_columns:
            rows.add_row(mapped_label, " | ".join(str(item) for item in mapped_columns[:4]))
        if llm_mapped_columns:
            rows.add_row(llm_label, " | ".join(str(item) for item in llm_mapped_columns[:3]))
        manual_mapped_columns = import_summary.get("manual_mapped_columns") or []
        if manual_mapped_columns:
            rows.add_row(manual_label, " | ".join(str(item) for item in manual_mapped_columns[:3]))
        if ignored_columns:
            rows.add_row(ignore_label, " | ".join(str(item) for item in ignored_columns[:3]))
        missing_targets = import_summary.get("missing_critical_targets") or []
        if missing_targets:
            rows.add_row(missing_label, " | ".join(str(item) for item in missing_targets[:3]))
        preview_rows = import_summary.get("preview_rows") or []
        preview_columns = [str(column) for column in (import_summary.get("preview_columns") or [])][:4]
        if not preview_rows or not preview_columns:
            return Group(Text(preview_title, style="bold #0a84ff"), rows)
        table = Table(box=None, expand=True, pad_edge=False)
        for column in preview_columns:
            table.add_column(column)
        for row in preview_rows[:4]:
            values = [self._fmt(row.get(column)) for column in preview_columns]
            table.add_row(*values)
        return Group(
            Text(self._("title_import_preview_confirm") if not missing_targets else self._("title_import_preview_missing"), style="bold #0a84ff"),
            rows,
            Text(self._("title_sample_data"), style="#475569"),
            table,
        )

    def _render_confirmed_setup_panel(self, analysis: dict[str, Any], project) -> Group:
        objective = analysis.get("primary_objective") or {}
        objective_name = str(objective.get("name") or ("objective" if self.language == Language.EN else "目标"))
        goal = str(objective.get("goal") or "maximize")
        rows = Table.grid(padding=(0, 0), expand=True)
        rows.add_column(style="bold #0f172a", no_wrap=True)
        rows.add_column()
        if self.language == Language.EN:
            obj_label = "Objective"
            factor_label = "Factors"
            next_label = "Next"
            next_msg = 'Send "initialize design 8 runs" or "generate 5 new designs"'
            title = self._("confirmed_plan")
        else:
            obj_label = "目标"
            factor_label = "因素"
            next_label = "下一步"
            next_msg = '发送"初始化设计 8 组"或"生成 5 组新设计"'
            title = "已确认方案"
        rows.add_row(obj_label, f"{objective_name} · {self._goal_cn(goal, objective.get('target_value'))}")
        factor_summaries = []
        for factor in project.factors[:4]:
            factor_summaries.append(f"{factor.name}: {self._factor_range_text(factor)}")
        rows.add_row(factor_label, " | ".join(factor_summaries) or "-")
        rows.add_row(next_label, next_msg)
        return Group(
            Text(title, style="bold #0a84ff"),
            rows,
        )

    def _render_pending_design_panel(self, project) -> Group:
        pending = self.coordinator.session.pending
        count = 0
        if isinstance(pending.data, dict):
            try:
                count = int(pending.data.get("count") or 0)
            except (TypeError, ValueError):
                count = 0
        mode = self._("init_design") if pending.action == "confirm_init_design" else self._("generate_design")
        rows = Table.grid(padding=(0, 0), expand=True)
        rows.add_column(style="bold #0f172a", no_wrap=True)
        rows.add_column()
        rows.add_row(self._("execute"), mode)
        rows.add_row(self._("group_count"), self._fmt(count))
        rows.add_row(self._("label_project"), self._fmt(getattr(project, "name", "")))
        workbook = getattr(project, "workbook", None)
        rows.add_row(self._("label_workbook"), self._fmt(workbook.name if workbook is not None else ""))
        rows.add_row(self._("label_status"), self._("label_status_waiting"))
        return Group(
            Text(self._("pending_confirm"), style="bold #0a84ff"),
            rows,
            Text(self._("confirm_plan_notice"), style="#475569"),
        )

    def _fmt(self, value: object) -> str:
        if value in (None, ""):
            return "-"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

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
        return f"{self._fmt(values[0])} -> {self._fmt(values[-1])}"

    def _factor_range_text(self, factor) -> str:
        if factor.kind == "categorical":
            return ", ".join(factor.categories) or "-"
        if self.language == Language.EN:
            if factor.kind == "ordinal":
                return f"{self._fmt(factor.min_value)} to {self._fmt(factor.max_value)}, step {self._fmt(factor.step)}"
            return f"{self._fmt(factor.min_value)} to {self._fmt(factor.max_value)}"
        else:
            if factor.kind == "ordinal":
                return f"{self._fmt(factor.min_value)} 到 {self._fmt(factor.max_value)}，步长 {self._fmt(factor.step)}"
            return f"{self._fmt(factor.min_value)} 到 {self._fmt(factor.max_value)}"

    def _goal_cn(self, goal: str, target_value: object | None) -> str:
        if self.language == Language.EN:
            if goal == "maximize":
                return "maximize"
            if goal == "minimize":
                return "minimize"
            return f"target {self._fmt(target_value)}"
        else:
            if goal == "maximize":
                return "最大化"
            if goal == "minimize":
                return "最小化"
            return f"贴近 {self._fmt(target_value)}"

    def _plan_status_text(self, status: object) -> str:
        value = str(status or "").strip().lower()
        if self.language == Language.EN:
            mapping = {
                "initialized": "Pending",
                "suggested": "Pending",
                "proposed": "Suggested",
                "completed": "Done",
                "skipped": "Skipped",
            }
        else:
            mapping = {
                "initialized": "待做",
                "suggested": "待做",
                "proposed": "建议",
                "completed": "完成",
                "skipped": "跳过",
            }
        return mapping.get(value, self._fmt(status))

    def _has_positive_score(self, value: object) -> bool:
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            return False

    def _empty_section(self, title: str, body: str, hint: str | None = None) -> Group:
        parts: list[Text] = [Text(body, style="#334155")]
        if hint:
            parts.append(Text(hint, style="#475569"))
        return Group(Text(title, style="bold #0a84ff"), *parts)

    def _bi_section(self, title: str, body: object, en_title: str | None = None) -> Group:
        # If en_title is provided, use it for English, otherwise use title
        if self.language == Language.EN and en_title:
            display_title = en_title
        else:
            display_title = title
        return Group(
            Text(display_title, style="bold #0a84ff"),
            body,
        )

    def _bi_kpi_grid(self, items: list[tuple[str, str]]) -> Table:
        grid = Table.grid(padding=(0, 2), expand=True)
        for _ in range(4):
            grid.add_column()
        for index in range(0, len(items), 2):
            left = items[index]
            right = items[index + 1] if index + 1 < len(items) else ("", "")
            grid.add_row(
                Text(left[0], style="#475569"),
                Text(left[1], style="bold #0f172a"),
                Text(right[0], style="#475569"),
                Text(right[1], style="bold #0f172a"),
            )
        return grid

    def _trend_delta_text(self, values: list[float | int], *, suffix: str = "") -> str:
        cleaned = [float(value) for value in values if value is not None]
        if len(cleaned) < 2:
            return "-"
        delta = cleaned[-1] - cleaned[0]
        sign = "+" if delta >= 0 else ""
        return f"{sign}{self._fmt(delta)}{suffix}"

    def _render_recent_completed_bi(self, analysis: dict[str, Any], objective_name: str) -> Table:
        recent = analysis.get("recent_completed") or []
        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column(style="bold #0f172a", no_wrap=True)
        table.add_column(no_wrap=True)
        table.add_column()
        if not recent:
            table.add_row(self._("label_recent_completed"), "-", self._("section_no_experiments"))
            return table
        for row in recent[-4:]:
            label = f"{self._fmt(row.get('Trial'))}-{self._fmt(row.get('candidate_id'))}"
            score = self._fmt(row.get("FuzzyScore"))
            objective_value = self._fmt(row.get(objective_name)) if objective_name else "-"
            table.add_row(label, score, objective_value)
        return table

    def _render_bo_diagnostics_bi(self, diagnostics: dict[str, Any]) -> Group | Table:
        if not diagnostics:
            return self._empty_section(self._("section_bo_stats"), self._("label_no_model_stats"))
        if not diagnostics.get("model_ready"):
            return self._empty_section(self._("section_bo_stats"), str(diagnostics.get("reason") or self._("label_model_not_ready")))
        fit_summary = diagnostics.get("fit_summary") or {}
        summary = Table.grid(padding=(0, 1), expand=True)
        summary.add_column(style="bold #0f172a", no_wrap=True)
        summary.add_column()
        summary.add_row(self._("label_candidate_pool"), self._fmt(diagnostics.get("candidate_count")))
        summary.add_row(self._("label_high_value_region"), self._fmt(diagnostics.get("top_candidate_count")))
        summary.add_row(self._("label_anchor_count"), self._fmt(diagnostics.get("anchor_count")))
        summary.add_row(self._("label_surrogate_model"), self._fmt(diagnostics.get("surrogate")))
        summary.add_row(self._("label_acquisition_function"), self._fmt(diagnostics.get("acquisition")))
        summary.add_row(self._("label_batch_policy"), self._fmt(diagnostics.get("batch_policy")))
        summary.add_row(self._("label_best_score"), self._fmt(diagnostics.get("best_score")))
        summary.add_row(self._("label_expected_upper"), self._fmt(diagnostics.get("top_expected_score")))
        summary.add_row(self._("label_avg_uncertainty"), self._fmt(diagnostics.get("top_uncertainty")))
        summary.add_row(self._("label_val_rmse"), self._fmt(fit_summary.get("mean_validation_rmse")))
        summary.add_row(self._("label_avg_epochs"), self._fmt(fit_summary.get("mean_epochs")))
        summary.add_row(self._("label_ensemble"), self._fmt(fit_summary.get("effective_ensemble_size")))
        summary.add_row(self._("label_calibration_coefficient"), self._fmt(fit_summary.get("uncertainty_scale")))
        summary.add_row(self._("label_training_loss"), self._fmt(fit_summary.get("training_loss")))

        factor_table = Table(box=None, expand=True, pad_edge=False)
        factor_table.add_column(self._("col_factor"), no_wrap=True, style="bold #0f172a")
        factor_table.add_column(self._("col_sensitivity"), no_wrap=True)
        factor_table.add_column(self._("col_uncertainty"), no_wrap=True)
        factor_table.add_column(self._("col_current_best"), no_wrap=True)
        factor_table.add_column(self._("col_suggested_range"))
        factor_effects = diagnostics.get("factor_effects") or []
        if not factor_effects:
            factor_table.add_row("-", "-", "-", "-", self._("label_no_available_stats"))
        else:
            for item in factor_effects[:8]:
                factor_table.add_row(
                    str(item.get("name") or "-"),
                    self._ratio_text(item.get("importance")),
                    self._fmt(item.get("uncertainty")),
                    self._fmt(item.get("current_best")),
                    self._factor_range_summary(item),
                )
        return Group(summary, Text(self._("section_factor_sensitivity"), style="#475569"), factor_table)

    def _ratio_text(self, value: object) -> str:
        try:
            return f"{float(value) * 100:.0f}%"
        except (TypeError, ValueError):
            return "-"

    def _factor_range_summary(self, item: dict[str, Any]) -> str:
        recommended = item.get("recommended_range")
        if isinstance(recommended, list):
            if not recommended:
                return "-"
            if item.get("kind") == "categorical":
                return " / ".join(str(value) for value in recommended)
            if len(recommended) == 1:
                return self._fmt(recommended[0])
            return f"{self._fmt(recommended[0])} - {self._fmt(recommended[-1])}"
        return self._fmt(recommended)

    def _set_workspace_panel_body(
        self,
        *,
        title: str,
        empty_text: str | None = None,
        body: object | None = None,
    ) -> None:
        body_widget = self.query_one("#plot-card-content", Static)
        if body is not None:
            body_widget.update(body)
        else:
            body_widget.update(self._empty_section(title, empty_text or "暂无数据。"))

    def _configure_workspace_layout(self, *, mode: str, show_table: bool) -> None:
        body_widget = self.query_one("#plot-card-body", VerticalScroll)
        table_widget = self.query_one("#workspace-table", DataTable)
        if mode == "bi" or not show_table:
            body_widget.styles.height = "1fr"
            table_widget.styles.height = 0
        else:
            body_widget.styles.height = "auto"
            table_widget.styles.height = "1fr"

    def _workspace_column_specs(
        self,
        project,
        analysis: dict[str, Any],
        rows: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        row_values = [self._workspace_table_row_values(project, analysis, row) for row in (rows or [])[:12]]
        # Use wider columns for English
        if self.language == Language.EN:
            base_specs: list[dict[str, Any]] = [
                {"key": "marker", "title": "", "min_width": 1, "max_width": 1, "align": "left"},
                {"key": "kind", "title": self._("col_type"), "min_width": 5, "max_width": 5, "align": "left"},
                {"key": "trial", "title": self._("col_trial"), "min_width": 5, "max_width": 5, "align": "right"},
                {"key": "candidate", "title": self._("col_candidate"), "min_width": 3, "max_width": 3, "align": "right"},
                {"key": "status", "title": self._("col_status"), "min_width": 7, "max_width": 7, "align": "left"},
            ]
            obj_min_width = 12
            obj_max_width = 18
            score_min_width = 6
            score_max_width = 10
            factor_min_width = 12
            factor_max_width_cat = 20
            factor_max_width_ord = 18
        else:
            base_specs: list[dict[str, Any]] = [
                {"key": "marker", "title": "", "min_width": 1, "max_width": 1, "align": "left"},
                {"key": "kind", "title": self._("col_type"), "min_width": 4, "max_width": 4, "align": "left"},
                {"key": "trial", "title": self._("col_trial"), "min_width": 4, "max_width": 4, "align": "right"},
                {"key": "candidate", "title": self._("col_candidate"), "min_width": 4, "max_width": 4, "align": "right"},
                {"key": "status", "title": self._("col_status"), "min_width": 4, "max_width": 4, "align": "left"},
            ]
            obj_min_width = 6
            obj_max_width = 12
            score_min_width = 6
            score_max_width = 10
            factor_min_width = 6
            factor_max_width_cat = 12
            factor_max_width_ord = 10
        # Create columns for each objective instead of generic "目标" column
        objective_specs: list[dict[str, Any]] = []
        for objective in project.objectives:
            objective_specs.append(
                {"key": f"objective:{objective.name}", "title": objective.name, "min_width": obj_min_width, "max_width": obj_max_width, "align": "right"}
            )
        # Add FuzzyScore column for multi-objective or single-objective projects
        score_label = "Score" if self.language == Language.EN else "评分"
        score_specs: list[dict[str, Any]] = [
            {"key": "score", "title": score_label, "min_width": score_min_width, "max_width": score_max_width, "align": "right"},
        ]
        specs = list(base_specs)
        for factor in project.factors:
            specs.append(
                {
                    "key": f"factor:{factor.name}",
                    "title": factor.name,
                    "min_width": factor_min_width,
                    "max_width": factor_max_width_cat if factor.kind == "categorical" else factor_max_width_ord,
                    "align": "left" if factor.kind == "categorical" else "right",
                }
            )
        specs.extend(objective_specs)
        specs.extend(score_specs)
        for spec in specs:
            if str(spec["key"]) == "marker":
                spec["width"] = 1
                continue
            header_width = self._display_width(str(spec["title"]))
            content_width = max(
                [header_width]
                + [self._display_width(str(values.get(str(spec["key"]), ""))) for values in row_values]
            )
            min_width = int(spec["min_width"])
            max_width = int(spec["max_width"])
            spec["width"] = min(max(content_width, min_width), max_width)
        return specs

    def _clear_workspace_table(self) -> None:
        table = self.query_one("#workspace-table", DataTable)
        try:
            table.clear(columns=True)
        except TypeError:
            table.clear()

    def _populate_workspace_table(self, project, analysis: dict[str, Any], rows: list[dict[str, Any]]) -> None:
        table = self.query_one("#workspace-table", DataTable)
        specs = self._workspace_column_specs(project, analysis, rows)
        visible_specs = [spec for spec in specs if str(spec["key"]) != "marker"]
        self._clear_workspace_table()
        for spec in visible_specs:
            header = str(spec["title"])
            # Don't set fixed width - let DataTable auto-size columns
            table.add_column(header, key=str(spec["key"]))
        for row in rows:
            values = self._workspace_table_row_values(project, analysis, row)
            cells = [
                self._truncate_display_text(values.get(str(spec["key"]), ""), int(spec["width"]))
                for spec in visible_specs
            ]
            try:
                table.add_row(*cells, key=f"{self._fmt(row.get('Trial'))}-{self._fmt(row.get('candidate_id'))}")
            except TypeError:
                table.add_row(*cells)
        self._sync_workspace_table_cursor()

    def _workspace_table_row_values(self, project, analysis: dict[str, Any], row: dict[str, Any]) -> dict[str, str]:
        status = self._plan_status_text(row.get("status"))
        if self.language == Language.EN:
            row_type = "Plan" if status in {"Pending", "Suggested"} else "Exp"
        else:
            row_type = "计划" if status in {"待做", "建议"} else "实验"
        score_value = self._fmt(row.get("FuzzyScore"))
        if score_value == "-" and row.get("score") not in (None, ""):
            score_value = self._fmt(row.get("score"))
        experiment_id = row.get("experiment_id")
        display_id = self._fmt(experiment_id) if experiment_id not in (None, "") else self._fmt(row.get("candidate_id"))
        values: dict[str, str] = {
            "marker": self._workspace_row_marker(row),
            "kind": row_type,
            "trial": self._fmt(row.get("Trial")),
            "candidate": display_id,
            "status": status,
            "score": score_value,
        }
        # Add values for each objective (instead of generic "objective" column)
        for objective in project.objectives:
            values[f"objective:{objective.name}"] = self._fmt(row.get(objective.name))
        for factor in project.factors:
            values[f"factor:{factor.name}"] = self._fmt(row.get(factor.name))
        return values


    def _truncate_display_text(self, text: str, width: int) -> str:
        if self._display_width(text) <= width:
            return text
        if width <= 1:
            return text[:1]
        current: list[str] = []
        current_width = 0
        for char in text:
            char_width = self._display_width(char)
            if current and current_width + char_width > width - 1:
                break
            current.append(char)
            current_width += char_width
        return "".join(current).rstrip() + "…"

    def _workspace_row_marker(self, row: dict[str, Any]) -> str:
        focused = self.coordinator.session.ui_state.get("workspace_focus") or {}
        if not isinstance(focused, dict) or not focused:
            return " "
        focused_trial = self._fmt(focused.get("Trial"))
        focused_candidate = self._fmt(focused.get("candidate_id"))
        row_trial = self._fmt(row.get("Trial"))
        row_candidate = self._fmt(row.get("candidate_id"))
        return "›" if row_trial == focused_trial and row_candidate == focused_candidate else " "

    def _focus_workspace_row(self, *, kind: str, index: int) -> None:
        rows = self.plan_focus_rows
        if index < 0 or index >= len(rows):
            return
        payload = dict(rows[index])
        status = self._plan_status_text(payload.get("status"))
        payload["kind"] = "plan" if status in {"待做", "建议"} else "experiment"
        self.coordinator.session.ui_state["workspace_focus"] = payload
        self.coordinator.session.ui_state["workspace_hint"] = self._workspace_focus_message(payload)
        save_session(self.coordinator.session)
        self._sync_workspace_table_cursor(index if kind == "plan" else None)
        self.workspace_hint = str(self.coordinator.session.ui_state.get("workspace_hint") or "")
        self._scroll_to_latest_pending = True
        self._refresh_dashboard()

    def _restore_workspace_hint(self) -> None:
        hint = str(self.coordinator.session.ui_state.get("workspace_hint") or "").strip()
        if hint:
            self.workspace_hint = hint
            return
        focused = self.coordinator.session.ui_state.get("workspace_focus") or {}
        if not isinstance(focused, dict) or not focused:
            return
        self.workspace_hint = self._workspace_focus_message(focused)
        self.coordinator.session.ui_state["workspace_hint"] = self.workspace_hint

    def _sync_workspace_table_cursor(self, row_index: int | None = None) -> None:
        try:
            table = self.query_one("#workspace-table", DataTable)
        except Exception:
            return
        if row_index is None:
            focused = self.coordinator.session.ui_state.get("workspace_focus") or {}
            if not isinstance(focused, dict) or not focused:
                return
            focused_trial = self._fmt(focused.get("Trial"))
            focused_candidate = self._fmt(focused.get("candidate_id"))
            row_index = None
            for index, row in enumerate(self.plan_focus_rows):
                if (
                    self._fmt(row.get("Trial")) == focused_trial
                    and self._fmt(row.get("candidate_id")) == focused_candidate
                ):
                    row_index = index
                    break
        if row_index is None:
            return
        try:
            table.move_cursor(row=row_index, column=0)
        except Exception:
            pass

    def _workspace_focus_message(self, payload: dict[str, Any]) -> str:
        if self.language == Language.EN:
            kind = "experiment plan" if str(payload.get("kind") or "") == "plan" else "experiment record"
            left_prefix = "left"
            id_label = "ID"
            batch_label = "Trial"
            status_prefix = "Current status:"
        else:
            kind = self._("type_plan") if str(payload.get("kind") or "") == "plan" else self._("focus_experiment_record")
            left_prefix = "左侧"
            id_label = "编号"
            batch_label = "批次"
            status_prefix = self._("label_current_status")
        experiment_id = payload.get("experiment_id")
        if experiment_id not in (None, ""):
            head = f"Focused on {left_prefix}{kind}: {id_label} {self._fmt(experiment_id)}." if self.language == Language.EN else f"已聚焦{left_prefix}{kind}：{id_label} {self._fmt(experiment_id)}。"
        else:
            head = f"Focused on {left_prefix}{kind}: {batch_label} {self._fmt(payload.get('Trial'))}, {id_label} {self._fmt(payload.get('candidate_id'))}." if self.language == Language.EN else f"已聚焦{left_prefix}{kind}：{batch_label} {self._fmt(payload.get('Trial'))}，{id_label} {self._fmt(payload.get('candidate_id'))}。"
        parts = [
            head
        ]
        if payload.get("status") not in (None, ""):
            parts.append(f"{status_prefix}{self._plan_status_text(payload.get('status'))}." if self.language == Language.EN else f"{status_prefix}{self._plan_status_text(payload.get('status'))}。")
        conditions = []
        for key, value in payload.items():
            if key in {"Trial", "candidate_id", "experiment_id", "status", "score", "FuzzyScore", "kind", "COMMENT", "source"}:
                continue
            if value in (None, ""):
                continue
            conditions.append(f"{key}={self._fmt(value)}")
        if conditions:
            if self.language == Language.EN:
                parts.append(f"Conditions: {', '.join(conditions[:3])}.")
            else:
                parts.append(f"条件：{', '.join(conditions[:3])}。")
        if str(payload.get("kind") or "") == "plan":
            if self.language == Language.EN:
                parts.append('You can say "change catalyst to A" or "set temperature to 75", or report results after completion.')
            else:
                parts.append('你可以直接说"把催化剂改成 A""温度改成 75"，或者做完后直接回填结果。')
        else:
            if self.language == Language.EN:
                parts.append("You can add results, describe actual changes, or let me continue designing based on it.")
            else:
                parts.append("你可以继续补这条实验结果、说明实际改动，或者让我基于它继续往下设计。")
        return " ".join(parts)

    def _set_section_visible(self, selector: str, visible: bool) -> None:
        widget = self.query_one(selector)
        widget.styles.display = "block" if visible else "none"

    def _experiment_view_state(self) -> dict[str, Any]:
        raw_state = dict(self.coordinator.session.ui_state.get("experiment_view") or {})
        state = dict(raw_state)
        scope = str(state.get("scope") or "recent").lower()
        sort = str(state.get("sort") or "recent").lower()
        page = state.get("page")
        try:
            normalized_page = max(int(page), 1)
        except (TypeError, ValueError):
            normalized_page = 1
        batch_id = state.get("batch_id")
        try:
            normalized_batch = int(batch_id) if batch_id not in (None, "", False) else None
        except (TypeError, ValueError):
            normalized_batch = None
        return {
            "scope": scope,
            "sort": sort,
            "page": normalized_page,
            "page_size": self._auto_page_size(),
            "batch_id": normalized_batch,
            "_scope_explicit": "scope" in raw_state,
        }

    def _effective_experiment_view_state(self, analysis: dict[str, Any]) -> dict[str, Any]:
        view = self._experiment_view_state()
        if view.get("_scope_explicit"):
            view.pop("_scope_explicit", None)
            return view
        completed_count = int(analysis.get("completed_count") or 0)
        suggested_count = int(analysis.get("suggested_count") or 0)
        if completed_count > 0 and suggested_count > 0 and view.get("scope") in {"recent", "pending"}:
            view["scope"] = "all"
        if completed_count == 0 and suggested_count > 0 and view.get("scope") == "recent":
            view["scope"] = "pending"
        if suggested_count == 0 and view.get("scope") == "pending":
            view["scope"] = "completed" if completed_count > 0 else "recent"
        view.pop("_scope_explicit", None)
        return view

    def _update_experiment_view(
        self,
        *,
        scope: str | None = None,
        sort: str | None = None,
        page: int | None = None,
        batch_id: int | None = None,
        clear_batch: bool = False,
    ) -> None:
        view = self._experiment_view_state()
        if scope is not None:
            view["scope"] = scope
        if sort is not None:
            view["sort"] = sort
        if page is not None:
            view["page"] = max(page, 1)
        if clear_batch:
            view["batch_id"] = None
        elif batch_id is not None:
            view["batch_id"] = batch_id
        self.coordinator.session.ui_state["experiment_view"] = view
        save_session(self.coordinator.session)
        self._refresh_dashboard()

    def _auto_page_size(self) -> int:
        try:
            table = self.query_one("#workspace-table", DataTable)
            height = int(table.size.height or 0)
            if height > 0:
                return max(height - 4, 4)
        except Exception:
            pass
        try:
            plot_card = self.query_one("#plot-card", Vertical)
            height = int(plot_card.size.height or 0)
            if height > 0:
                return max(height - 8, 4)
        except Exception:
            pass
        return 8

    def _sync_experiment_toolbar(self, total_pages: int, *, show_paging: bool) -> None:
        state = self._experiment_view_state()
        mode = self._workspace_mode_state()
        self.current_total_pages = max(total_pages, 1)
        mode_table = self.query_one("#mode-table", Button)
        mode_bi = self.query_one("#mode-bi", Button)
        if mode == "table":
            mode_table.add_class("-active")
            mode_bi.remove_class("-active")
        else:
            mode_bi.add_class("-active")
            mode_table.remove_class("-active")
        active_scope = str(state.get("scope") or "recent")
        for button_id, scope in {
            "scope-recent": "recent",
            "scope-completed": "completed",
            "scope-pending": "pending",
            "scope-best": "best",
            "scope-all": "all",
        }.items():
            button = self.query_one(f"#{button_id}", Button)
            if active_scope == scope:
                button.add_class("-active")
            else:
                button.remove_class("-active")
            button.disabled = mode != "table"
        sort_button = self.query_one("#sort-score", Button)
        if state.get("sort") == "score_desc":
            sort_button.add_class("-active")
        else:
            sort_button.remove_class("-active")
        sort_button.disabled = mode != "table"
        self.query_one("#clear-batch", Button).disabled = mode != "table" or state.get("batch_id") is None
        current_page = int(state.get("page") or 1)
        self.query_one("#page-first", Button).disabled = mode != "table" or current_page <= 1
        self.query_one("#page-prev", Button).disabled = mode != "table" or current_page <= 1
        self.query_one("#page-next", Button).disabled = mode != "table" or current_page >= max(total_pages, 1)
        self.query_one("#page-last", Button).disabled = mode != "table" or current_page >= max(total_pages, 1)
        self.query_one("#page-indicator", Static).update(f"{current_page} / {max(total_pages, 1)}")
        self.query_one("#page-jump", Input).disabled = mode != "table"
        self.query_one("#page-apply", Button).disabled = mode != "table"
        self._set_section_visible("#paging-toolbar", mode == "table" and show_paging)
        self._set_section_visible("#footer-toolbar", True)

    def _experiment_panel_header(self, experiment_view: dict[str, Any]) -> str:
        if self.language == Language.EN:
            scope_map = {
                "recent": "Records Recent",
                "completed": "Records Completed",
                "pending": "Records Pending",
                "best": "Records Best",
                "all": "Records All",
            }
            default_header = "Records"
            item_label = "items"
            batch_prefix = "Trial"
        else:
            scope_map = {
                "recent": "记录 最近",
                "completed": "记录 完成",
                "pending": "记录 待做",
                "best": "记录 最佳",
                "all": "记录 全部",
            }
            default_header = "记录"
            item_label = "条"
            batch_prefix = "批次"
        header = scope_map.get(str(experiment_view.get("scope") or ""), default_header)
        batch_id = experiment_view.get("batch_id")
        total_rows = int(experiment_view.get("total_rows") or 0)
        parts = [header, f"{total_rows}{item_label}"]
        if batch_id is not None:
            parts.insert(1, f"{batch_prefix}{batch_id}")
        return " | ".join(parts)

    def _experiment_view_summary(self, experiment_view: dict[str, Any]) -> str:
        if self.language == Language.EN:
            scope_map = {
                "recent": "View Recent",
                "completed": "View Completed",
                "pending": "View Pending",
                "best": "View Best",
                "all": "View All",
            }
            default_summary = "View Recent"
            batch_prefix = "Trial"
            score_label = "Score"
            page_suffix = "p"
        else:
            scope_map = {
                "recent": "视图 最近",
                "completed": "视图 完成",
                "pending": "视图 待做",
                "best": "视图 最佳",
                "all": "视图 全部",
            }
            default_summary = "视图 最近"
            batch_prefix = "批次"
            score_label = "评分"
            page_suffix = "页"
        summary = scope_map.get(str(experiment_view.get("scope") or ""), default_summary)
        if experiment_view.get("batch_id") is not None:
            summary += f"·{batch_prefix}{experiment_view['batch_id']}"
        if str(experiment_view.get("sort") or "") == "score_desc":
            summary += f"·{score_label}"
        page = int(experiment_view.get("page") or 1)
        total_pages = int(experiment_view.get("total_pages") or 1)
        summary += f"·{page}/{total_pages}{page_suffix}"
        return summary

    def _render_message_body(self, content: str, width: int, style: str) -> Group | Text:
        blocks = self._parse_message_blocks(content)
        renderables: list[object] = []
        for kind, payload in blocks:
            if kind == "table":
                table = self._build_rich_table(payload)  # type: ignore[arg-type]
                if table is not None:
                    renderables.append(table)
                    continue
            text = str(payload)
            renderables.append(Text(self._wrap_message_content(text, width), style=style))
        if not renderables:
            return Text("", style=style)
        if len(renderables) == 1 and isinstance(renderables[0], Text):
            return renderables[0]
        return Group(*renderables)

    def _parse_message_blocks(self, content: str) -> list[tuple[str, object]]:
        lines = (content or "").splitlines()
        if not lines:
            return [("text", "")]
        blocks: list[tuple[str, object]] = []
        text_buffer: list[str] = []
        table_buffer: list[str] = []

        def flush_text() -> None:
            if text_buffer:
                blocks.append(("text", "\n".join(text_buffer).strip("\n")))
                text_buffer.clear()

        def flush_table() -> None:
            if table_buffer:
                blocks.append(("table", list(table_buffer)))
                table_buffer.clear()

        for line in lines:
            if self._looks_like_table_line(line):
                flush_text()
                table_buffer.append(line)
                continue
            flush_table()
            text_buffer.append(line)
        flush_table()
        flush_text()
        return blocks or [("text", content)]

    def _looks_like_table_line(self, line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2

    def _build_rich_table(self, lines: list[str]) -> Table | None:
        rows = [self._split_markdown_row(line) for line in lines if self._looks_like_table_line(line)]
        if len(rows) < 2:
            return None
        header = rows[0]
        divider = rows[1]
        if not self._is_markdown_divider_row(divider, expected_columns=len(header)):
            return None
        body_rows = [row for row in rows[2:] if len(row) == len(header)]
        table = Table(box=box.SIMPLE, expand=False, show_header=True, pad_edge=False)
        for column in header:
            table.add_column(column or " ", overflow="fold")
        for row in body_rows:
            table.add_row(*row)
        return table

    def _split_markdown_row(self, line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    def _is_markdown_divider_row(self, row: list[str], expected_columns: int) -> bool:
        if len(row) != expected_columns:
            return False
        return all(cell and set(cell.replace(":", "")) <= {"-"} for cell in row)

    def _maybe_scroll_chat_to_latest(self) -> None:
        if not self.is_mounted or not self._scroll_to_latest_pending:
            return
        self._scroll_to_latest_pending = False
        scroll_view = self.query_one("#chat-scroll", VerticalScroll)
        self.call_after_refresh(lambda: scroll_view.scroll_end(animate=False))

    def _message_wrap_width(self, role: str = "assistant") -> int:
        pane_width = self.query_one("#chat-pane", Vertical).size.width or 48
        ratios = {
            "context": 0.78,
            "assistant": 0.72,
            "user": 0.62,
        }
        return max(18, int(pane_width * ratios.get(role, 0.62)))

    def _wrap_message_content(self, content: str, width: int) -> str:
        wrapped_lines: list[str] = []
        for line in (content or "").splitlines() or [""]:
            wrapped_lines.extend(self._wrap_display_line(line, width))
        return "\n".join(wrapped_lines)

    def _wrap_display_line(self, line: str, width: int) -> list[str]:
        if not line:
            return [""]
        rows: list[str] = []
        current: list[str] = []
        current_width = 0
        for char in line:
            char_width = self._display_width(char)
            if current and current_width + char_width > width:
                rows.append("".join(current).rstrip())
                current = [char]
                current_width = char_width
                continue
            current.append(char)
            current_width += char_width
        if current:
            rows.append("".join(current).rstrip())
        return rows or [""]

    def _display_width(self, text: str) -> int:
        width = 0
        for char in text:
            if unicodedata.combining(char):
                continue
            width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        return width

    def _copy_text(self, text: str) -> bool:
        try:
            self.copy_to_clipboard(text)
            return True
        except Exception:
            pass
        if sys.platform == "darwin":
            try:
                subprocess.run(["pbcopy"], input=text, text=True, check=True)
                return True
            except Exception:
                return False
        return False

    def _set_status(self, message: str) -> None:
        self.status_text = message
        self._render_chat_head()

    def _set_interaction_enabled(self, enabled: bool) -> None:
        input_widget = self.query_one("#chat-input", SubmitTextArea)
        input_widget.disabled = not enabled

    def _set_terminal_mouse_reporting(self, enabled: bool) -> None:
        driver = getattr(self, "_driver", None)
        if driver is None:
            return
        if enabled:
            method = getattr(driver, "_enable_mouse_support", None)
        else:
            self.capture_mouse(None)
            method = getattr(driver, "_disable_mouse_support", None)
        if callable(method):
            try:
                method()
            except Exception:
                pass

    def _schedule_input_focus(self, delay: float = 0.02) -> None:
        self.set_timer(delay, self._focus_input_after_layout)

    def _focus_input_after_layout(self) -> None:
        if not self.is_mounted:
            return
        try:
            input_widget = self.query_one("#chat-input", SubmitTextArea)
        except NoMatches:
            return
        if self.screen.focused is input_widget:
            return
        self.set_focus(input_widget)
        input_widget.focus()


def run_chat_dashboard(
    workspace: Path,
    task_id: str | None = None,
    language: Language = Language.ZH,
) -> None:
    app = ChatExpDashboardApp(workspace=workspace, task_id=task_id, language=language)
    app.run()
