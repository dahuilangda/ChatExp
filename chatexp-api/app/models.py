from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


FactorType = Literal["continuous", "ordinal", "categorical"]
GoalType = Literal["maximize", "minimize", "target"]
Language = Literal["zh", "en"]


class FactorSpec(BaseModel):
    name: str
    kind: FactorType = "continuous"
    unit: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    categories: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        raw_kind = data.get("kind", data.get("type", data.get("input_type")))
        if data.get("categories"):
            data["kind"] = "categorical"
        else:
            text = str(raw_kind or "continuous").strip().lower()
            data["kind"] = {
                "numeric": "continuous",
                "number": "continuous",
                "float": "continuous",
                "range": "continuous",
                "integer": "ordinal",
                "int": "ordinal",
                "discrete": "ordinal",
                "category": "categorical",
                "enum": "categorical",
            }.get(text, text if text in {"continuous", "ordinal", "categorical"} else "continuous")
        if data.get("kind") == "categorical":
            data["min_value"] = None
            data["max_value"] = None
            data["step"] = None
        data["categories"] = data.get("categories") or []
        return data


class ObjectiveSpec(BaseModel):
    name: str
    goal: GoalType = "maximize"
    target_value: float | str | None = None
    target_min: float | None = None
    target_max: float | None = None
    weight: float = 1.0
    unit: str | None = None
    desirability: str | list[dict[str, float | str]] | None = None


class ProjectSpec(BaseModel):
    name: str = "seatable"
    factors: list[FactorSpec] = Field(default_factory=list)
    objectives: list[ObjectiveSpec] = Field(default_factory=list)
