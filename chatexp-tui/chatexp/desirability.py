from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from .models import ObjectiveSpec
from .settings import Language


EPS = 1e-9
NUMERIC_TEXT_PATTERN = r"-?\d+(?:\.\d+)?"


class FuzzyScore:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._x_dict: dict[object, float] = {}

    def score_dataframe(self, objectives: list[ObjectiveSpec]) -> tuple[pd.DataFrame, dict[str, list[dict[str, float | str]]]]:
        return score_components(self.df, objectives)

    def score_value(
        self,
        value: object,
        objective: ObjectiveSpec,
        series: pd.Series,
    ) -> float | None:
        if is_missing_value(value):
            return None
        # The configured desirability profile is the scoring contract approved by
        # the operator. Data-driven profiles are recommendations; they should not
        # silently replace the approved profile during scoring.
        desirability = parse_desirability_text(objective.desirability)
        if not desirability:
            desirability = auto_desirability_points(series, objective)
        if not desirability:
            return None
        return desirability_score(value, desirability, objective.weight)


def desirability_score(value: object, desirability: list[dict[str, float | str]], importance: float) -> float:
    if is_missing_value(value) or not desirability:
        return None
    x_points = [point["x"] for point in desirability]
    y_points = [float(np.clip(float(point["y"]), EPS, 1.0)) for point in desirability]
    numeric_x = [parse_numeric_value(point) for point in x_points]
    numeric_value = parse_numeric_value(value)
    if all(point is not None for point in numeric_x) and numeric_value is not None:
        xs = np.asarray([float(point) for point in numeric_x], dtype=float)
        ys = np.asarray(y_points, dtype=float)
        order = np.argsort(xs)
        interpolated = np.interp(float(numeric_value), xs[order], ys[order], left=ys[order][0], right=ys[order][-1])
        return float(np.clip(interpolated, EPS, 1.0))
    mapping = {str(x_point): y_point for x_point, y_point in zip(x_points, y_points)}
    return float(mapping.get(str(value), min(y_points)))


def score_components(df: pd.DataFrame, objectives: list[ObjectiveSpec]) -> tuple[pd.DataFrame, dict[str, list[dict[str, float | str]]]]:
    """Score experiments using operator-approved fuzzy desirability profiles.

    The scoring pipeline:
    1. Use the configured desirability profile for each objective when present
    2. Auto-propose a profile only for objectives without an approved profile
    3. Aggregate objective desirabilities with the weighted geometric mean
    4. Return component scores, final FuzzyScore, and newly proposed profiles

    Returns:
        tuple: (scored_dataframe, desirability_updates)
            - scored_dataframe: Original dataframe with added desirability and FuzzyScore columns
            - desirability_updates: Auto-calculated profiles for objectives lacking a configured profile
    """
    result = df.copy()
    objective_map = {objective.name: objective for objective in objectives if objective.name in result.columns}
    component_columns: list[str] = []
    desirability_updates: dict[str, list[dict[str, float | str]]] = {}

    # Step 1: Calculate desirability for each objective based on current data
    for name, objective in objective_map.items():
        column_name = f"_desirability_{name}"
        component_columns.append(column_name)
        values = []
        series = result[name]

        configured_points = parse_desirability_text(objective.desirability)
        if configured_points:
            desirability_points = configured_points
        else:
            desirability_points = auto_desirability_points(series, objective)
        if desirability_points and not configured_points:
            desirability_updates[name] = desirability_points

        scorer = FuzzyScore(result)
        for value in series.tolist():
            values.append(scorer.score_value(value, objective, series))
        result[column_name] = values

    # Step 2: Calculate weighted FuzzyScore
    fuzzy_scores = []
    for _, row in result.iterrows():
        values = []
        weights = []

        for i, (name, objective) in enumerate(objective_map.items()):
            column_name = f"_desirability_{name}"
            if column_name in result.columns and pd.notna(row[column_name]):
                desirability_value = float(row[column_name])

                values.append(desirability_value)
                weights.append(float(objective.weight or 1.0))

        if values:
            weighted_product = 1.0
            weight_sum = 0.0
            for v, w in zip(values, weights):
                weighted_product *= (v ** w)
                weight_sum += w

            if weight_sum > 0:
                fuzzy_score = weighted_product ** (1.0 / weight_sum)
            else:
                fuzzy_score = np.prod(values) ** (1.0 / len(values))

            fuzzy_scores.append(float(np.clip(fuzzy_score, 1e-9, 1.0)))
        else:
            fuzzy_scores.append(np.nan)

    result["FuzzyScore"] = fuzzy_scores

    return result, desirability_updates


def _calculate_adaptive_weights(
    df: pd.DataFrame,
    objectives: list[ObjectiveSpec],
    component_columns: list[str]
) -> dict[str, float]:
    """Calculate adaptive weights based on data distribution characteristics.

    The adaptive weight considers:
    1. Optimization potential (variance) - objectives with more spread get higher weights
    2. Goal type - maximize/minimize/target goals are treated differently
    3. Data availability - objectives with more complete data get higher weights

    Returns a dictionary mapping objective name to its adaptive weight.
    """
    weights = {}

    for objective in objectives:
        name = objective.name
        column_name = f"_desirability_{name}"

        if column_name not in df.columns:
            weights[name] = 1.0
            continue

        desirability_values = df[column_name].dropna()

        if len(desirability_values) == 0:
            weights[name] = 1.0
            continue

        # Single-value series (or any series with fewer than 2 points) has
        # no measurable spread: var(ddof=1) is NaN, which would surface a NaN
        # weight to the operator. Keep a neutral weight 1.0; too little spread does not establish optimization.
        if len(desirability_values) < 2:
            weights[name] = 1.0
            continue

        # Calculate variance-based weight
        # Objectives with higher variance have more optimization potential.
        # ddof=0 keeps the variance finite even for tiny samples.
        variance = float(desirability_values.var(ddof=0))

        # Normalize variance to [0.5, 2.0] range
        # - Variance 0 → weight 0.5 (already optimized)
        # - Variance 0.1+ → weight 2.0 (high optimization potential)
        import math
        variance_weight = 0.5 + min(variance * 10, 1.5)

        # Adjust for goal type
        # - For maximize: high desirability is good, low values need focus
        # - For minimize: low desirability is good, high values need focus
        mean_desirability = desirability_values.mean()

        if objective.goal == "maximize":
            # If mean is already high (>0.7), reduce weight (already doing well)
            # If mean is low (<0.3), increase weight (needs more attention)
            focus_factor = 1.0
            if mean_desirability > 0.7:
                focus_factor = 0.7
            elif mean_desirability < 0.3:
                focus_factor = 1.3
        elif objective.goal == "minimize":
            # If mean is already low (>0.7 as desirability), reduce weight
            # If mean is high (>0.3 as actual value), needs attention
            # Minimize profiles also assign high desirability to good performance
            focus_factor = 1.0
            if mean_desirability > 0.7:
                focus_factor = 0.7
            elif mean_desirability < 0.3:
                focus_factor = 1.3
        else:  # target
            # For target goals, weight by distance from target
            focus_factor = 1.0

        # Combine factors
        adaptive_weight = variance_weight * focus_factor

        # Normalize to [0.5, 2.0] range
        adaptive_weight = max(0.5, min(2.0, adaptive_weight))

        weights[name] = adaptive_weight

    return weights


def parse_desirability_text(text: Any | None) -> list[dict[str, float | str]]:
    if not text:
        return []
    if isinstance(text, list):
        desirability = []
        for item in text:
            if not isinstance(item, dict) or item.get("x") is None:
                continue
            y_value = parse_numeric_value(item.get("y"))
            if y_value is None:
                continue
            numeric_x = parse_numeric_value(item.get("x"))
            desirability.append({"x": numeric_x if numeric_x is not None else item.get("x"), "y": y_value})
        return desirability
    desirability = []
    raw_text = str(text).replace("|", "\n")
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            left, right = [item.strip(" ()") for item in line.split(",", 1)]
        elif ":" in line:
            left, right = [item.strip(" ()") for item in line.split(":", 1)]
        else:
            continue
        y_value = parse_numeric_value(right)
        if y_value is None:
            continue
        numeric_left = parse_numeric_value(left)
        x_value: float | str = numeric_left if numeric_left is not None else left
        desirability.append({"x": x_value, "y": y_value})
    return desirability


def auto_desirability_points(series: pd.Series, objective: ObjectiveSpec) -> list[dict[str, float | str]]:
    present = series[~series.map(is_missing_value)]
    numeric = coerce_numeric_series(present)
    if len(present) and numeric.notna().sum() == len(present):
        numeric = numeric.dropna().astype(float)
        if numeric.empty:
            return []
        return _auto_numeric_desirability(numeric, objective)

    categories = [str(item) for item in present.dropna().unique().tolist()]
    if not categories:
        return []
    if objective.goal == "target" and objective.target_value is not None:
        return [{"x": category, "y": 1.0 if category == str(objective.target_value) else 0.2} for category in categories]
    dominant = present.dropna().astype(str).mode().iloc[0]
    if objective.goal == "maximize":
        return [{"x": category, "y": 1.0 if category == dominant else 0.5} for category in categories]
    return [{"x": category, "y": 0.2 if category == dominant else 1.0} for category in categories]


def format_desirability(points: list[dict[str, float | str]]) -> str:
    return "\n".join(f"({point['x']}, {point['y']})" for point in points)


def describe_desirability(points: list[dict[str, float | str]], objective: ObjectiveSpec, language: Language = Language.ZH) -> str:
    if not points:
        return "Not enough results yet, keeping current scoring function." if language == Language.EN else "当前没有足够结果，暂不调整评分函数。"
    if objective.goal == "maximize":
        if language == Language.EN:
            return f"Compress low values, spread high values; key inflection: {_point_summary(points)}"
        else:
            return f"低值区压缩，高值区拉开；关键拐点：{_point_summary(points)}"
    if objective.goal == "minimize":
        if language == Language.EN:
            return f"Compress high values, spread low values; key inflection: {_point_summary(points)}"
        else:
            return f"高值区压缩，低值区拉开；关键拐点：{_point_summary(points)}"
    if language == Language.EN:
        return f"Strengthen middle range around target; key inflection: {_point_summary(points)}"
    else:
        return f"围绕目标值强化中间区间；关键拐点：{_point_summary(points)}"


def normalize_desirability_text(text: str | None) -> str:
    points = parse_desirability_text(text)
    if not points:
        return ""
    normalized_lines = []
    for point in points:
        x_value = point["x"]
        if isinstance(x_value, float):
            x_repr = f"{x_value:.8g}"
        else:
            x_repr = str(x_value)
        normalized_lines.append(f"{x_repr}:{float(point['y']):.8g}")
    return "|".join(normalized_lines)


def _auto_numeric_desirability(series: pd.Series, objective: ObjectiveSpec) -> list[dict[str, float | str]]:
    minimum = float(series.min())
    maximum = float(series.max())
    if math.isclose(minimum, maximum):
        maximum = minimum + 1.0
    q10, q35, q65, q90 = [float(series.quantile(q)) for q in (0.10, 0.35, 0.65, 0.90)]
    if objective.goal == "maximize":
        target = _numeric_target(objective)
        # Extend ceiling beyond current maximum to preserve optimization potential
        # This ensures current best experiments don't get perfect scores
        range_size = maximum - minimum
        # Extend ceiling by 20% of range or 0.5, whichever is larger
        extended_ceiling = maximum + max(range_size * 0.2, 0.5)
        ceiling = max(extended_ceiling, target) if target is not None else extended_ceiling

        points = [
            {"x": minimum, "y": 0.02},
            {"x": q35, "y": 0.16},
            {"x": q65, "y": 0.45},
            {"x": max(q90, q65), "y": 0.82},
            {"x": ceiling, "y": 1.0},
        ]
        return _dedupe_numeric_points(points, prefer="high")
    if objective.goal == "minimize":
        target = _numeric_target(objective)
        floor = min(minimum, target) if target is not None else minimum
        points = [
            {"x": floor, "y": 1.0},
            {"x": min(q10, q35), "y": 0.82},
            {"x": q35, "y": 0.45},
            {"x": q65, "y": 0.16},
            {"x": maximum, "y": 0.02},
        ]
        return _dedupe_numeric_points(points, prefer="low")
    target_min = parse_numeric_value(objective.target_min)
    target_max = parse_numeric_value(objective.target_max)
    if target_min is not None and target_max is not None:
        left = min(float(target_min), float(target_max))
        right = max(float(target_min), float(target_max))
        lower_bound = min(minimum, left)
        upper_bound = max(maximum, right)
        points = [
            {"x": lower_bound, "y": 0.02},
            {"x": left, "y": 1.0},
            {"x": right, "y": 1.0},
            {"x": upper_bound, "y": 0.02},
        ]
        return _dedupe_numeric_points(points, prefer="center")
    target = _numeric_target(objective)
    center = target if target is not None else float(series.median())
    spread = max((q90 - q10) * 0.5, (maximum - minimum) / 6.0, 1e-6)
    left = min(center - spread, center)
    right = max(center + spread, center)
    lower_bound = min(minimum, left)
    upper_bound = max(maximum, right)
    points = [
        {"x": lower_bound, "y": 0.02},
        {"x": left, "y": 0.55},
        {"x": center, "y": 1.0},
        {"x": right, "y": 0.55},
        {"x": upper_bound, "y": 0.02},
    ]
    return _dedupe_numeric_points(points, prefer="center")


def _numeric_target(objective: ObjectiveSpec) -> float | None:
    return parse_numeric_value(objective.target_value)


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


def _dedupe_numeric_points(
    points: list[dict[str, float | str]],
    *,
    prefer: str,
) -> list[dict[str, float | str]]:
    ordered = sorted(
        ({"x": float(point["x"]), "y": float(point["y"])} for point in points),
        key=lambda item: item["x"],
    )
    deduped: list[dict[str, float | str]] = []
    for point in ordered:
        if deduped and math.isclose(float(deduped[-1]["x"]), float(point["x"]), rel_tol=1e-9, abs_tol=1e-9):
            previous_y = float(deduped[-1]["y"])
            current_y = float(point["y"])
            if prefer == "high":
                deduped[-1]["y"] = max(previous_y, current_y)
            elif prefer == "low":
                deduped[-1]["y"] = min(previous_y, current_y)
            else:
                deduped[-1]["y"] = max(previous_y, current_y)
            continue
        deduped.append(point)
    return deduped


def _point_summary(points: list[dict[str, float | str]]) -> str:
    preview = []
    for point in points[:4]:
        x_value = point["x"]
        x_text = f"{float(x_value):.4g}" if isinstance(x_value, (int, float)) else str(x_value)
        preview.append(f"{x_text}->{float(point['y']):.2f}")
    return " | ".join(preview)


def _interp_with_edge_hold(x_eval: float, x_points: list[float], y_points: list[float]) -> float:
    if not x_points:
        return 1.0
    if x_eval <= x_points[0]:
        return y_points[0]
    if x_eval >= x_points[-1]:
        return y_points[-1]
    return float(np.interp(x_eval, x_points, y_points))


def analyze_weight_distribution(
    df: pd.DataFrame,
    objectives: list[ObjectiveSpec],
    language: Language = Language.ZH
) -> dict[str, Any]:
    """Analyze the weight distribution for multi-objective scoring.

    Returns a dictionary with:
    - weights: dict of adaptive weights for each objective
    - analysis: human-readable analysis of why weights were assigned
    - recommendations: suggestions for improvement
    """
    from .storage import SHEET_EXPERIMENTS

    component_columns = [f"_desirability_{obj.name}" for obj in objectives]
    available_columns = [col for col in component_columns if col in df.columns]

    if not available_columns:
        return {
            "weights": {},
            "analysis": "No data available for weight analysis." if language == Language.EN else "暂无数据可用于权重分析。",
            "recommendations": []
        }

    adaptive_weights = _calculate_adaptive_weights(df, objectives, available_columns)

    # Build analysis
    weight_details = []
    recommendations = []

    for objective in objectives:
        name = objective.name
        weight = adaptive_weights.get(name, 1.0)
        column_name = f"_desirability_{name}"

        if column_name not in df.columns:
            continue

        desirability_values = df[column_name].dropna()

        if len(desirability_values) == 0:
            continue

        variance = desirability_values.var()
        mean_val = desirability_values.mean()
        min_val = desirability_values.min()
        max_val = desirability_values.max()

        detail = {
            "name": name,
            "weight": weight,
            "variance": variance,
            "mean": mean_val,
            "min": min_val,
            "max": max_val,
            "goal": objective.goal,
        }

        # Generate analysis text
        if language == Language.EN:
            if weight > 1.2:
                detail["reason"] = f"High optimization potential (variance={variance:.3f})"
            elif weight < 0.8:
                detail["reason"] = f"Well optimized (variance={variance:.3f})"
            else:
                detail["reason"] = f"Normal optimization potential"

            if objective.goal == "maximize":
                if mean_val > 0.7:
                    detail["status"] = f"Performing well (avg desirability={mean_val:.2f})"
                elif mean_val < 0.3:
                    detail["status"] = f"Needs attention (avg desirability={mean_val:.2f})"
                else:
                    detail["status"] = f"Moderate performance (avg desirability={mean_val:.2f})"
        else:
            if weight > 1.2:
                detail["reason"] = f"优化空间大 (方差={variance:.3f})"
            elif weight < 0.8:
                detail["reason"] = f"已优化较好 (方差={variance:.3f})"
            else:
                detail["reason"] = f"正常优化潜力"

            if objective.goal == "maximize":
                if mean_val > 0.7:
                    detail["status"] = f"表现良好 (平均满意度={mean_val:.2f})"
                elif mean_val < 0.3:
                    detail["status"] = f"需要关注 (平均满意度={mean_val:.2f})"
                else:
                    detail["status"] = f"表现中等 (平均满意度={mean_val:.2f})"

        weight_details.append(detail)

        # Generate recommendations
        if variance < 0.01 and objective.weight > 0.8:
            if language == Language.EN:
                recommendations.append(f"{name}: Already optimized and highly weighted. Consider adjusting scoring focus.")
            else:
                recommendations.append(f"{name}：已优化且权重高，可考虑调整评分重点。")
        elif variance > 0.1 and objective.weight < 0.3:
            if language == Language.EN:
                recommendations.append(f"{name}: High optimization potential but low weight. Consider increasing weight.")
            else:
                recommendations.append(f"{name}：优化空间大但权重低，可考虑增加权重。")

    return {
        "weights": adaptive_weights,
        "details": weight_details,
        "analysis": weight_details,
        "recommendations": recommendations,
    }
