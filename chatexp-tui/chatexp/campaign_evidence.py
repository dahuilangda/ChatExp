"""Observed-data evidence for the strategist; no policy or oracle access.

Approved desirability curves remain fixed. These summaries describe the
adaptively sampled observations, not the distribution of the whole domain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .desirability import desirability_score, parse_desirability_text, parse_numeric_value


def _numeric(series):
    # Match the scorer's accepted numeric notation, including percentage
    # text, rather than treating otherwise valid observations as missing.
    return pd.to_numeric(series.map(parse_numeric_value), errors="coerce").replace([np.inf, -np.inf], np.nan)


def _summary(values):
    values = values.dropna()
    if values.empty:
        return {"n": 0}
    q = values.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(len(values)),
        "quantiles": {str(p): float(v) for p, v in q.items()},
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
        "n_unique": int(values.nunique()),
    }


def fixed_utility_scores(project, frame):
    """Local complete-case scores; None means the contract is unspecified."""
    profiles = [parse_desirability_text(o.desirability) for o in project.objectives]
    if not profiles or not all(profiles):
        return None
    parts = {}
    for objective, profile in zip(project.objectives, profiles):
        raw = _numeric(frame[objective.name]) if objective.name in frame else pd.Series(np.nan, index=frame.index)
        parts[objective.name] = raw.map(lambda v: desirability_score(v, profile, 1.0) if pd.notna(v) else np.nan)
    matrix = pd.DataFrame(parts)
    weights = np.asarray([max(float(o.weight), 1e-9) for o in project.objectives])
    weights /= weights.sum()
    score = np.exp(np.log(matrix.clip(lower=1e-9, upper=1.0)).mul(weights, axis=1).sum(axis=1))
    return score.where(matrix.notna().all(axis=1))


def build_campaign_evidence(project, frame):
    """Summarize responses and the FIXED utility, using complete cases only
    for composite comparisons. Missing outcomes never improve a row's score.
    No synthetic data, surrogate predictions or task descriptions are used.
    """
    frame = pd.DataFrame() if frame is None else frame.copy()
    if "status" in frame:
        frame = frame[frame["status"].astype(str).str.strip().str.lower() == "completed"]
    objectives = []
    components = {}
    weights = {}
    for objective in project.objectives:
        raw = _numeric(frame[objective.name]) if objective.name in frame else pd.Series(np.nan, index=frame.index, dtype=float)
        profile = parse_desirability_text(objective.desirability)
        entry = {
            "name": objective.name, "goal": objective.goal,
            "operator_weight": float(objective.weight),
            "response_distribution": _summary(raw),
            "missing_or_nonfinite_fraction": float(raw.isna().mean()) if len(raw) else None,
            "approved_desirability_profile": profile or [],
        }
        if profile:
            d = raw.map(lambda v: desirability_score(v, profile, 1.0) if pd.notna(v) else np.nan)
            lower = max(1e-9, min(float(p["y"]) for p in profile))
            upper = min(1.0, max(float(p["y"]) for p in profile))
            valid = d.dropna()
            entry.update({
                "desirability_distribution": _summary(d),
                "profile_floor": lower, "profile_ceiling": upper,
                "floor_fraction": float((valid <= lower + 1e-9).mean()) if len(valid) else None,
                "ceiling_fraction": float((valid >= upper - 1e-9).mean()) if len(valid) else None,
                "profile_plateau_segments": [
                    [a["x"], b["x"]] for a, b in zip(profile, profile[1:])
                    if abs(float(a["y"]) - float(b["y"])) <= 1e-12
                ],
            })
            components[objective.name] = d
            weights[objective.name] = max(float(objective.weight), 1e-9)
        else:
            entry["note"] = "No approved profile: fixed utility cannot be assessed; do not invent a changing scoring contract."
        objectives.append(entry)

    result = {
        "version": "observed-evidence-v2",
        "n_completed": len(frame), "objectives": objectives,
        "fixed_utility": {"available": False}, "objective_associations": [],
        "interpretation": [
            "Summaries condition on adaptively selected observations, not the whole search space.",
            "Variance alone does not measure recoverability or distinguish noise from signal.",
            "Objective associations are descriptive, not causal or proof of unavoidable trade-offs.",
            "Profiles and operator weights define the fixed evaluation contract; search emphasis is separate.",
        ],
    }
    if components:
        matrix = pd.DataFrame(components)
        names = list(matrix)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pair = matrix[[a, b]].dropna()
                if len(pair) >= 4 and pair[a].nunique() > 1 and pair[b].nunique() > 1:
                    result["objective_associations"].append({
                        "objectives": [a, b], "n_pairs": len(pair),
                        "spearman_desirability": float(pair[a].rank().corr(pair[b].rank())),
                    })
        if len(components) == len(project.objectives):
            complete = matrix.dropna()
            if len(complete):
                w = pd.Series(weights); w = w / w.sum()
                losses = -np.log(complete.clip(lower=1e-9, upper=1.0)).mul(w)
                total = losses.sum(axis=1)
                # iloc handles duplicate external row indices deterministically.
                best = int(np.argmin(total.to_numpy()))
                row = losses.iloc[best]
                result["fixed_utility"] = {
                    "available": True, "n_complete_cases": len(complete),
                    "best_observed_score": float(np.exp(-total.iloc[best])),
                    "incumbent_desirabilities": {k: float(v) for k, v in complete.iloc[best].items()},
                    "incumbent_weighted_log_loss": {k: float(v) for k, v in row.items()},
                    "incumbent_loss_share": {k: float(v / row.sum()) if row.sum() > 0 else 0.0 for k, v in row.items()},
                }
    result["joint_observations"] = []
    scores = fixed_utility_scores(project, frame)
    if scores is not None:
        # Preserve actual joint factor/response evidence, bounded for prompt size.
        # Recent failures and best rows are selected explicitly, never fabricated.
        valid = np.flatnonzero(scores.notna().to_numpy())
        chosen = list(valid[-12:])
        chosen += list(valid[np.argsort(scores.iloc[valid].to_numpy())[-3:]])
        for pos in sorted(set(chosen)):
            row = frame.iloc[int(pos)]
            def scalar(v):
                if isinstance(v, np.generic):
                    v = v.item()
                return v if isinstance(v, (str, int, float, bool)) and (not isinstance(v, float) or np.isfinite(v)) else None
            result["joint_observations"].append({"row_position": int(pos),
                "factors": {f.name: scalar(row.get(f.name)) for f in project.factors},
                "responses": {o.name: float(_numeric(pd.Series([row.get(o.name)])).iloc[0]) for o in project.objectives},
                "fixed_score": float(scores.iloc[int(pos)])})
    result["joint_observation_selection"] = "union of last 12 complete rows and best 3 complete rows; original order; at most 15"
    result["factor_evidence"] = []
    scores = fixed_utility_scores(project, frame)
    for factor in project.factors:
        if factor.name not in frame:
            continue
        entry = {"name": factor.name, "kind": factor.kind}
        if factor.kind == "categorical":
            entry["observed_counts"] = {str(k): int(v) for k, v in frame[factor.name].dropna().value_counts().items()}
            if scores is not None:
                groups = pd.DataFrame({"level": frame[factor.name], "score": scores}).dropna()
                entry["level_fixed_utility"] = [
                    {"level": str(k), "n": len(v), "mean": float(v.score.mean())}
                    for k, v in groups.groupby("level", sort=False)
                ]
        else:
            values = _numeric(frame[factor.name])
            entry["observed_distribution"] = _summary(values)
            entry["declared_range"] = [factor.min_value, factor.max_value]
            if scores is not None:
                pair = pd.DataFrame({"factor": values, "score": scores}).dropna()
                if len(pair) >= 4 and pair.factor.nunique() > 1 and pair.score.nunique() > 1:
                    entry["spearman_fixed_utility"] = float(pair.factor.rank().corr(pair.score.rank()))
                if len(pair):
                    # Descriptive support only: tied scores retain all rows.
                    top = pair[pair.score >= pair.score.quantile(0.75)]
                    entry["top_quartile_observed_support"] = {"n": len(top), "min": float(top.factor.min()), "max": float(top.factor.max())}
        result["factor_evidence"].append(entry)
    result["interpretation"].append("Factor associations and top-quartile support reflect selection bias; they neither establish causal effects nor justify shrinking the domain.")
    return result
