"""Harness-side automatic objective inference (自动目标推断).

When the user configures NO objectives (neither single- nor multi-objective
desirability), the harness infers them instead of silently degrading to
initial-design forever (the old behavior) or guessing without disclosure.

HARD CONSTRAINT — bounded sampling: every statistic (name hints are metadata,
but completeness, variance, factor correlation, quantile curves) is computed
on at most ``max_sample_rows`` rows (head + tail + seeded uniform fill).
Tables can be very large; inference never scans the full frame. The sample
basis (sampled/total rows) is part of the report so the user knows exactly
what the inference saw.

Inference is EXPLICIT, not a fallback: the report carries per-column
rationale (zh/en), the scores that drove the decision, and the constructed
desirability curves. The UI surfaces it; configuring objectives manually
always overrides it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .models import ObjectiveSpec
from .scoring import auto_desirability_points, coerce_numeric_series, is_missing_value


class InferenceError(Exception):
    """No objective column could be inferred — explicit, never silent."""


# ---------------------------------------------------------------------------
# Bounded sampling
# ---------------------------------------------------------------------------


def sample_for_inference(
    frame: pd.DataFrame,
    *,
    max_rows: int = 200,
    head_rows: int = 100,
    tail_rows: int = 50,
    seed: int = 2026,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Deterministically sample at most ``max_rows`` rows: head + tail + a
    seeded uniform fill from the middle. Returns (sample, basis) where basis
    discloses sampled/total counts."""
    total = int(len(frame))
    if total <= max_rows or total <= head_rows + tail_rows:
        return frame, {"sampled_rows": total, "total_rows": total}

    # Document order (head → middle → tail): the sample preserves row order,
    # which matters for recency-sensitive statistics.
    parts = [frame.head(head_rows)]
    remaining = max(max_rows - head_rows - tail_rows, 0)
    if remaining > 0:
        rng = np.random.default_rng(seed)
        middle_span = total - head_rows - tail_rows
        middle_idx = np.sort(rng.choice(middle_span, size=min(remaining, middle_span), replace=False))
        parts.append(frame.iloc[head_rows + middle_idx])
    parts.append(frame.tail(tail_rows))
    sampled = pd.concat(parts)
    return sampled, {"sampled_rows": int(len(sampled)), "total_rows": total}


# ---------------------------------------------------------------------------
# Direction hints (bilingual name semantics)
# ---------------------------------------------------------------------------

_MAXIMIZE_PATTERNS = [
    r"yield", r"产率", r"得率", r"收率", r"转化率", r"selectivity", r"选择性",
    r"purity", r"纯度", r"score", r"得分", r"评分", r"quality", r"质量",
    r"efficiency", r"效率", r"output", r"产出", r"产量", r"性能", r"strength",
    r"强度", r"capacity", r"容量", r"power", r"功率", r"enrichment", r"富集",
]
_MINIMIZE_PATTERNS = [
    r"cost", r"成本", r"费用", r"开销", r"消耗", r"time", r"时长", r"耗时",
    r"duration", r"waste", r"废品", r"废物", r"defect", r"缺陷", r"不良",
    r"error", r"误差", r"偏差", r"deviation", r"impurity", r"杂质", r"loss",
    r"损耗", r"emission", r"排放", r"腐蚀", r"corrosion", r"磨损", r"wear",
    r"残[留余]", r"residual", r"fouling", r"结垢", r"毒性", r"toxicity",
]
# Columns that are never objectives regardless of name.
_NON_OBJECTIVE_PATTERNS = [
    r"^_+", r"id$", r"^id", r"uuid", r"编号", r"序号", r"序数", r"^no\.?$",
    r"index", r"^row", r"名称", r"name$", r"备注", r"note", r"comment",
    r"备注", r"标签", r"label", r"tag", r"日期", r"date", r"时间戳", r"timestamp",
    r"创建", r"修改", r"creator", r"状态", r"status", r"版本", r"version",
    r"fuzzyscore", r"score_$", r"_score$", r"_bo_target",
]


def _name_matches(name: str, patterns: list[str]) -> bool:
    lowered = str(name).lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def _direction_hint(column: str) -> tuple[str | None, float]:
    """(goal, confidence) from the column name; (None, 0.0) when neutral."""
    if _name_matches(column, _MAXIMIZE_PATTERNS):
        return "maximize", 1.0
    if _name_matches(column, _MINIMIZE_PATTERNS):
        return "minimize", 1.0
    return None, 0.0


# ---------------------------------------------------------------------------
# Candidate evaluation on the SAMPLE
# ---------------------------------------------------------------------------


@dataclass
class CandidateReport:
    column: str
    score: float = 0.0
    completeness: float = 0.0
    signal: float = 0.0
    hint_goal: str | None = None
    hint_confidence: float = 0.0
    selected: bool = False
    rejected_reason: str = ""
    rationale_zh: str = ""
    rationale_en: str = ""


def _numeric_series(sample: pd.DataFrame, column: str) -> pd.Series:
    present = sample[column][~sample[column].map(is_missing_value)]
    return coerce_numeric_series(present)


def _factor_signal(
    sample: pd.DataFrame,
    column: str,
    numeric: pd.Series,
    factor_names: list[str],
) -> float:
    """How strongly the candidate co-varies with any factor — a column that
    moves with the factors carries learnable signal for the next design
    round; a pure-noise or constant-vs-factors column does not. Spearman on
    the sample only."""
    if not factor_names:
        return 0.25  # no factors to correlate against: neutral default
    best = 0.0
    aligned = numeric.dropna()
    for factor in factor_names:
        if factor not in sample.columns:
            continue
        factor_values = _numeric_series(sample, factor).reindex(aligned.index)
        pair = pd.concat([aligned, factor_values], axis=1).dropna()
        if len(pair) < 4:
            continue
        try:
            corr = pair.corr(method="spearman").iloc[0, 1]
        except Exception:
            continue
        if pd.notna(corr):
            best = max(best, abs(float(corr)))
    return best


def infer_objectives(
    frame: pd.DataFrame,
    factor_names: list[str],
    *,
    max_sample_rows: int = 200,
    seed: int = 2026,
) -> dict[str, Any]:
    """Infer objectives from a bounded sample. Returns a report dict with the
    constructed ObjectiveSpec-compatible entries + full disclosure. Raises
    InferenceError when nothing plausible exists (explicit failure — the
    caller surfaces it, never a silent fallback)."""
    sample, basis = sample_for_inference(frame, max_rows=max_sample_rows, seed=seed)
    factor_set = {str(name) for name in factor_names}

    candidates: list[CandidateReport] = []
    for column in sample.columns:
        name = str(column)
        if name in factor_set:
            candidates.append(CandidateReport(column=name, rejected_reason="配置为因子列"))
            continue
        if _name_matches(name, _NON_OBJECTIVE_PATTERNS):
            candidates.append(CandidateReport(column=name, rejected_reason="标识/元数据类列名"))
            continue
        numeric = _numeric_series(sample, column)
        non_null = int(numeric.notna().sum())
        completeness = non_null / max(len(sample), 1)
        if completeness < 0.5:
            candidates.append(CandidateReport(column=name, completeness=completeness,
                                              rejected_reason=f"非空比例仅 {completeness:.0%}（采样）"))
            continue
        distinct = int(numeric.nunique())
        if distinct < 3:
            candidates.append(CandidateReport(column=name, completeness=completeness,
                                              rejected_reason=f"数值仅 {distinct} 个不同取值，无优化空间"))
            continue

        hint_goal, hint_confidence = _direction_hint(name)
        signal = _factor_signal(sample, name, numeric, sorted(factor_set))
        goal = hint_goal or "maximize"
        # Score: name semantics carry the most weight; factor signal decides
        # "most suitable for the next design round"; completeness rewards
        # usable columns.
        score = 0.45 * hint_confidence + 0.35 * min(signal, 1.0) + 0.20 * completeness
        candidates.append(CandidateReport(
            column=name, score=score, completeness=completeness, signal=signal,
            hint_goal=hint_goal, hint_confidence=hint_confidence,
        ))

    viable = [c for c in candidates if not c.rejected_reason and c.score > 0]
    if not viable:
        raise InferenceError(
            "未能从采样数据中推断出目标列：没有满足条件的数值列（排除因子列、"
            "标识列、低非空或无方差的列）。请在插件中手动配置目标（objectives）。"
        )
    viable.sort(key=lambda c: c.score, reverse=True)

    top = viable[0]
    selected = [top]
    # Multi-objective when a second candidate is nearly as informative.
    if len(viable) > 1 and viable[1].score >= 0.85 * top.score and viable[1].signal >= 0.15:
        selected.append(viable[1])
    weights = [1.0] if len(selected) == 1 else [0.6, 0.4]

    objectives: list[dict[str, Any]] = []
    for candidate, weight in zip(selected, weights):
        goal = candidate.hint_goal or "maximize"
        spec = ObjectiveSpec(name=candidate.column, goal=goal, weight=weight)  # type: ignore[arg-type]
        curve = auto_desirability_points(_numeric_series(sample, candidate.column), spec)
        hint_note_zh = (
            f"列名语义指向「{ '越大越好' if goal == 'maximize' else '越小越好' }」"
            if candidate.hint_goal else "列名无方向提示，默认按「越大越好」处理，请确认"
        )
        rationale_zh = (
            f"{hint_note_zh}；与因子相关性 |ρ|={candidate.signal:.2f}，"
            f"非空率 {candidate.completeness:.0%}（基于 {basis['sampled_rows']}/{basis['total_rows']} 行采样）"
        )
        rationale_en = (
            f"name hint → {goal}" if candidate.hint_goal else "no name hint — defaulted to maximize, please confirm"
        ) + (
            f"; factor signal |ρ|={candidate.signal:.2f}, completeness {candidate.completeness:.0%}"
            f" (sampled {basis['sampled_rows']}/{basis['total_rows']} rows)"
        )
        candidate.selected = True
        objectives.append({
            "name": candidate.column,
            "goal": goal,
            "weight": weight,
            "desirability": curve,
            "rationale_zh": rationale_zh,
            "rationale_en": rationale_en,
        })

    return {
        "autoInferred": True,
        "sampledRows": basis["sampled_rows"],
        "totalRows": basis["total_rows"],
        "objectives": objectives,
        "considered": [
            {
                "column": c.column,
                "score": round(c.score, 4),
                "signal": round(c.signal, 4),
                "completeness": round(c.completeness, 4),
                "hintGoal": c.hint_goal,
                "selected": c.selected,
                "rejectedReason": c.rejected_reason,
            }
            for c in candidates
        ],
    }


def inferred_specs(report: dict[str, Any]) -> list[ObjectiveSpec]:
    """Materialize the report's objectives as ObjectiveSpec list."""
    return [
        ObjectiveSpec(
            name=item["name"],
            goal=item.get("goal", "maximize"),  # type: ignore[arg-type]
            weight=float(item.get("weight", 1.0)),
            desirability=item.get("desirability"),
        )
        for item in report.get("objectives", [])
    ]
