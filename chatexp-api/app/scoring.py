from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .models import ObjectiveSpec


EPS = 1e-9
NUMERIC_TEXT_PATTERN = r"-?\d+(?:\.\d+)?"


def score_components(df: pd.DataFrame, objectives: list[ObjectiveSpec]) -> tuple[pd.DataFrame, dict[str, list[dict[str, float | str]]]]:
    result = df.copy()
    updates: dict[str, list[dict[str, float | str]]] = {}
    component_columns: list[str] = []
    for objective in objectives:
        if objective.name not in result.columns:
            continue
        column = f"_desirability_{objective.name}"
        component_columns.append(column)
        configured = parse_desirability(objective.desirability)
        # Configured brackets are the operator's contract and are NEVER
        # rewritten from the data: a history entirely below the lower
        # breakpoint is the honest desirability floor (the informative
        # regime on floor-dominated surfaces), not a stale-scale artifact
        # to re-base. rebase_disjoint_configured_points stays available for
        # explicit operator-invoked repair.
        points = configured if configured else auto_desirability_points(result[objective.name], objective)
        if points and not configured:
            updates[objective.name] = points
        result[column] = [desirability_score(value, points) for value in result[objective.name].tolist()]

    scores = []
    for _, row in result.iterrows():
        values = []
        weights = []
        for objective in objectives:
            column = f"_desirability_{objective.name}"
            if column not in result.columns or pd.isna(row[column]):
                continue
            values.append(max(EPS, float(row[column])))
            weights.append(max(0.0, float(objective.weight or 1.0)))
        if not values or sum(weights) <= 0:
            scores.append(np.nan)
            continue
        weights_array = np.asarray(weights, dtype=float)
        weights_array = weights_array / weights_array.sum()
        scores.append(float(np.exp(np.sum(np.log(values) * weights_array))))
    if component_columns:
        result["FuzzyScore"] = scores
    return result, updates


def parse_desirability(raw: Any) -> list[dict[str, float | str]]:
    if not raw:
        return []
    if isinstance(raw, list):
        points = []
        for item in raw:
            if not isinstance(item, dict) or item.get("x") is None or not _is_number(item.get("y")):
                continue
            numeric_x = parse_numeric_value(item.get("x"))
            points.append({
                "x": numeric_x if numeric_x is not None else item.get("x"),
                "y": float(parse_numeric_value(item.get("y"))),
            })
        return points
    points = []
    text = str(raw).replace("|", "\n")
    for line in text.splitlines():
        if "," in line:
            left, right = [part.strip(" ()") for part in line.split(",", 1)]
        elif ":" in line:
            left, right = [part.strip(" ()") for part in line.split(":", 1)]
        else:
            continue
        if not _is_number(right):
            continue
        numeric_left = parse_numeric_value(left)
        points.append({"x": numeric_left if numeric_left is not None else left, "y": float(parse_numeric_value(right))})
    return points


def format_desirability(points: list[dict[str, float | str]]) -> str:
    return "\n".join(f"{point.get('x')}, {point.get('y')}" for point in points)


def auto_desirability_points(series: pd.Series, objective: ObjectiveSpec) -> list[dict[str, float | str]]:
    present = series[~series.map(is_missing_value)]
    numeric = coerce_numeric_series(present)
    if len(present) and numeric.notna().sum() == len(present):
        values = numeric.dropna().astype(float)
        low = float(values.min())
        high = float(values.max())
        if abs(high - low) < 1e-12:
            high = low + 1.0
        q10, q35, q50, q65, q90 = [float(values.quantile(q)) for q in (0.10, 0.35, 0.50, 0.65, 0.90)]
        if objective.goal == "minimize":
            return _dedupe_numeric_points([
                {"x": low, "y": 1.0},
                {"x": min(q10, q35), "y": 0.82},
                {"x": q35, "y": 0.45},
                {"x": q65, "y": 0.16},
                {"x": high, "y": 0.02},
            ], prefer="low")
        if objective.goal == "target":
            if objective.target_min is not None and objective.target_max is not None:
                left = min(float(objective.target_min), float(objective.target_max))
                right = max(float(objective.target_min), float(objective.target_max))
                return _dedupe_numeric_points([
                    {"x": min(low, left), "y": 0.02},
                    {"x": left, "y": 1.0},
                    {"x": right, "y": 1.0},
                    {"x": max(high, right), "y": 0.02},
                ], prefer="center")
            target = parse_numeric_value(objective.target_value)
            if target is None:
                target = q50
            spread = max((q90 - q10) * 0.5, (high - low) / 6.0, 1e-6)
            return _dedupe_numeric_points([
                {"x": min(low, target - spread), "y": 0.02},
                {"x": target - spread, "y": 0.55},
                {"x": target, "y": 1.0},
                {"x": target + spread, "y": 0.55},
                {"x": max(high, target + spread), "y": 0.02},
            ], prefer="center")
        ceiling = high + max((high - low) * 0.2, 0.5)
        return _dedupe_numeric_points([
            {"x": low, "y": 0.02},
            {"x": q35, "y": 0.16},
            {"x": q65, "y": 0.45},
            {"x": max(q90, q65), "y": 0.82},
            {"x": ceiling, "y": 1.0},
        ], prefer="high")
    categories = [str(item) for item in present.dropna().unique().tolist()]
    if not categories:
        return []
    if objective.goal == "target" and objective.target_value is not None:
        return [{"x": category, "y": 1.0 if category == str(objective.target_value) else 0.2} for category in categories]
    dominant = present.dropna().astype(str).mode().iloc[0]
    good = 1.0 if objective.goal == "maximize" else 0.2
    other = 0.5 if objective.goal == "maximize" else 1.0
    return [{"x": category, "y": good if category == dominant else other} for category in categories]


def _dedupe_numeric_points(points: list[dict[str, float | str]], *, prefer: str) -> list[dict[str, float | str]]:
    ordered = sorted(
        ({"x": float(point["x"]), "y": float(point["y"])} for point in points),
        key=lambda item: item["x"],
    )
    deduped: list[dict[str, float | str]] = []
    for point in ordered:
        if deduped and abs(float(deduped[-1]["x"]) - float(point["x"])) <= 1e-9:
            previous = float(deduped[-1]["y"])
            current = float(point["y"])
            deduped[-1]["y"] = min(previous, current) if prefer == "low" else max(previous, current)
            continue
        deduped.append(point)
    return deduped


def rebase_disjoint_configured_points(points: list[dict[str, float | str]], series: pd.Series) -> list[dict[str, float | str]]:
    """A configured curve whose numeric x-support lies entirely on one side of
    the observed data says nothing about any row: every value clamps to the
    same end anchor (y=0.02) and FuzzyScore collapses to a constant — this
    happens when the curve outlives its data (settings kept from a previous
    table where the column lived on another scale). Re-base the curve's
    x-domain onto the observed range: shape and direction are preserved and
    the score discriminates again. Curves overlapping the data (even
    partially) are kept verbatim — only the fully disjoint degenerate case is
    repaired."""
    if not points:
        return points
    xs = [parse_numeric_value(point.get("x")) for point in points]
    if any(item is None for item in xs):
        return points
    numeric = coerce_numeric_series(series[~series.map(is_missing_value)]).dropna()
    if len(numeric) < 2:
        return points
    support_low, support_high = float(min(xs)), float(max(xs))
    observed_low, observed_high = float(numeric.min()), float(numeric.max())
    if not (observed_high < support_low or observed_low > support_high):
        return points
    support_span = max(EPS, support_high - support_low)
    observed_span = observed_high - observed_low
    return [
        {**point, "x": observed_low + (float(x) - support_low) * (observed_span / support_span)}
        for point, x in zip(points, xs)
    ]


def desirability_score(value: Any, points: list[dict[str, float | str]]) -> float | None:
    if is_missing_value(value) or not points:
        return None
    x_values = [point["x"] for point in points]
    y_values = [max(EPS, min(1.0, float(point["y"]))) for point in points]
    numeric_x_values = [parse_numeric_value(x) for x in x_values]
    numeric_value = parse_numeric_value(value)
    if all(item is not None for item in numeric_x_values) and numeric_value is not None:
        xs = np.asarray([float(x) for x in numeric_x_values], dtype=float)
        ys = np.asarray(y_values, dtype=float)
        order = np.argsort(xs)
        return float(np.interp(float(numeric_value), xs[order], ys[order], left=ys[order][0], right=ys[order][-1]))
    mapping = {str(x): y for x, y in zip(x_values, y_values)}
    return float(mapping.get(str(value), min(y_values)))


def _is_number(value: Any) -> bool:
    return parse_numeric_value(value) is not None


def parse_numeric_value(value: Any) -> float | None:
    if is_missing_value(value):
        return None
    if isinstance(value, (int, float, np.number)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    text = str(value).strip()
    normalized = text.replace(",", "").replace("%", "").replace("℃", "").replace("°C", "").replace("°", "")
    try:
        numeric = float(normalized)
        return numeric if np.isfinite(numeric) else None
    except (TypeError, ValueError):
        pass
    match = re.search(NUMERIC_TEXT_PATTERN, normalized)
    if not match:
        return None
    numeric = float(match.group(0))
    return numeric if np.isfinite(numeric) else None


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    return pd.Series([parse_numeric_value(value) for value in series.tolist()], index=series.index, dtype="float64")


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def json_value(value: Any) -> Any:
    """Convert numpy/pandas scalars and containers into JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_value(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def json_ready(record: dict[str, Any]) -> dict[str, Any]:
    return {str(key): json_value(value) for key, value in record.items()}
