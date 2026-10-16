from datetime import datetime
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


FactorType = Literal["continuous", "ordinal", "categorical"]
GoalType = Literal["maximize", "minimize", "target"]
IntentAction = Literal[
    "respond",
    "clarify",
    "draft_setup",
    "create_project",
    "load_project",
    "define_factors",
    "define_objective",
    "initialize_design",
    "append_results",
    "update_plan_experiment",
    "analyze_data",
    "generate_next_design",
    "update_objective",
    "update_factor",
    "import_experiment_sheet",
    "export",
    "refresh_scores",
    "inspect_workbook",
    "status",
    "view_experiments",
    "skip_pending_designs",
    "none",
]


class FactorSpec(BaseModel):
    name: str
    kind: FactorType
    unit: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    categories: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_factor_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        raw_kind = data.get("kind", data.get("type", data.get("input_type")))
        normalized_kind = normalize_factor_kind(raw_kind, categories=data.get("categories"))
        if normalized_kind is not None:
            data["kind"] = normalized_kind
        if data.get("categories") is None:
            data["categories"] = []
        if data.get("kind") == "categorical":
            data["min_value"] = None
            data["max_value"] = None
            data["step"] = None
        return data


def normalize_factor_kind(raw_kind: Any, categories: Any = None) -> FactorType:
    if categories:
        return "categorical"
    text = str(raw_kind or "").strip().lower()
    mapping = {
        "continuous": "continuous",
        "numeric": "continuous",
        "number": "continuous",
        "float": "continuous",
        "range": "continuous",
        "ordinal": "ordinal",
        "discrete": "ordinal",
        "integer": "ordinal",
        "int": "ordinal",
        "stepped": "ordinal",
        "categorical": "categorical",
        "category": "categorical",
        "classification": "categorical",
        "class": "categorical",
        "enum": "categorical",
    }
    return mapping.get(text, "continuous")


class ObjectiveSpec(BaseModel):
    name: str
    goal: GoalType = "maximize"
    target_value: Optional[Union[float, str]] = None
    target_min: Optional[Union[float, str]] = None
    target_max: Optional[Union[float, str]] = None
    weight: float = 1.0
    unit: Optional[str] = None
    desirability: Optional[str] = None


class ProjectState(BaseModel):
    name: str = "default"
    workspace_dir: str
    workbook_path: str
    factors: List[FactorSpec] = Field(default_factory=list)
    objectives: List[ObjectiveSpec] = Field(default_factory=list)
    initial_batch_size: int = 8
    latest_batch_id: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def workbook(self) -> Path:
        return Path(self.workbook_path)

    @property
    def workspace(self) -> Path:
        return Path(self.workspace_dir)


class ChatPending(BaseModel):
    action: Optional[str] = None
    # Workbook the pending action was staged against. The MCP confirm path
    # uses it to refuse confirmations that would execute against a different
    # workbook than the one currently loaded in the session.
    workbook: Optional[str] = None
    missing_fields: List[str] = Field(default_factory=list)
    data: dict = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class IntentPlan(BaseModel):
    action: IntentAction = "none"
    arguments: dict = Field(default_factory=dict)
    actions: List["IntentPlan"] = Field(default_factory=list)
    # File-evidence protocol: when the user references external files
    # (word/excel/pdf/csv/...), the planner may request them INSTEAD of
    # guessing their contents. The harness parses the files and re-plans once
    # with the real text attached. Each request: {path|name, reason}.
    evidence_requests: Optional[List[dict]] = None


IntentPlan.model_rebuild()


class ChatSession(BaseModel):
    state_path: str
    task_id: Optional[str] = None
    task_dir: Optional[str] = None
    title: Optional[str] = None
    project: Optional[ProjectState] = None
    pending: ChatPending = Field(default_factory=ChatPending)
    ui_state: dict = Field(default_factory=dict)
    history: List[ChatMessage] = Field(default_factory=list)
    message_count: int = 0
    persisted_message_count: int = 0
    # 1-based index of the NEXT chat-history line to write (0 = unknown, must
    # be resolved from the history file once). Keeping the cursor on the
    # session makes save_session incremental instead of re-reading
    # the whole chat_history.jsonl on every save (O(n^2) over long sessions).
    history_next_index: int = 0
    last_reply: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
