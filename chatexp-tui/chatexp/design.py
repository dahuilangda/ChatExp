from __future__ import annotations

import os

import copy
import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .desirability import coerce_numeric_series, is_missing_value, parse_numeric_value, score_components
from .settings import Language
from .models import FactorSpec, GoalType, ObjectiveSpec, ProjectState

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - runtime guard for environments not yet updated
    torch = None
    nn = None


MAX_TRAINING_ROWS = 1200
MAX_CANDIDATE_POOL_SIZE = 12000
MAX_DIAGNOSTIC_CANDIDATE_POOL_SIZE = 16000


@dataclass
class DesignSuggestion:
    batch_id: int
    candidate_id: int
    score: float
    values: dict[str, object]
    experiment_id: int | None = None
    # Selection provenance (api optimizer parity): "greedy_noisy_ei" engine
    # picks, "diagnostic_point" operator-approved probes. Defaulted so
    # existing constructors keep the engine label.
    strategy: str = "greedy_noisy_ei"


@dataclass
class BoundedTargetTransform:
    lower: float = 1e-4
    upper: float = 1.0 - 1e-4
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None

    def fit(self, y: torch.Tensor) -> torch.Tensor:
        latent = self._logit(y)
        self.mean = latent.mean()
        self.std = latent.std(unbiased=False).clamp_min(1e-6)
        return (latent - self.mean) / self.std

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise TorchBOError("The target transform has not been fitted.")
        latent = self._logit(y)
        return (latent - self.mean) / self.std

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise TorchBOError("The target transform has not been fitted.")
        latent = z * self.std + self.mean
        return torch.sigmoid(latent)

    def _logit(self, y: torch.Tensor) -> torch.Tensor:
        clipped = y.clamp(self.lower, self.upper)
        return torch.logit(clipped)


class TorchBOError(RuntimeError):
    """Raised when torch-based BO cannot run."""


def generate_initial_design(project: ProjectState, n_points: int, starting_candidate_id: int = 1) -> list[DesignSuggestion]:
    """Generate a space-filling initial design for a fresh campaign.

    Deterministic in the project's latest batch id; suggestions carry score
    0.0 and the engine strategy label."""
    _require_torch()
    _validate_factor_domains(project.factors)
    candidates_df = _initial_design_frame(
        factors=project.factors,
        observed_rows=pd.DataFrame(),
        n_points=n_points,
        seed=42 + project.latest_batch_id,
        strategy="balanced",
    )
    batch_id = project.latest_batch_id + 1
    return [
        DesignSuggestion(batch_id=batch_id, candidate_id=starting_candidate_id + index, score=0.0, values=row)
        for index, row in enumerate(candidates_df.to_dict(orient="records"))
    ]


def suggest_next_design(
    project: ProjectState,
    experiments_df: pd.DataFrame,
    n_points: int,
    candidate_pool_size: int = 2048,
    ensemble_size: int = 5,
    training_steps: int = 300,
    starting_candidate_id: int = 1,
    strategy_phase: str | None = None,
    soft_constraints: list[dict] | None = None,
    llm_candidates: list[dict] | None = None,
    prior_pseudo: list[dict] | None = None,
    diagnostic_point: dict | None = None,
) -> list[DesignSuggestion]:
    """Suggest the next batch of experiments for a project.

    Fits the surrogate ensemble on completed rows and selects the batch by
    greedy noisy-EI; returns a space-filling initial design when there are
    too few completed rows to train. Directive inputs (strategy phase, prior
    hints, LLM candidates, soft constraints) are ignored whenever the
    observed incumbent is healthy (best composite >= 0.35). Shares with the
    chatexp-api optimizer the incumbent health clamp, the data-fingerprint
    Monte-Carlo seed, the LLM justify gate, and the soft-constraint penalty;
    the API additionally applies its own trust-weight schedule, handles
    target-band objectives, and gates the batch regime."""
    _require_torch()
    _validate_factor_domains(project.factors)
    n_points = max(int(n_points), 1)
    clean = _drop_incomplete_rows(project, experiments_df)
    if clean.shape[0] < max(4, len(project.factors) + 1):
        return _generate_initial_design_from_rows(project, experiments_df, n_points, starting_candidate_id)

    scored, _ = score_components(clean.copy(), project.objectives)
    component_columns = [
        f"_desirability_{objective.name}"
        for objective in project.objectives
        if f"_desirability_{objective.name}" in scored.columns
    ]
    if not component_columns:
        scored["_desirability_scalar"] = score_rows(scored, project.objectives)
        component_columns = ["_desirability_scalar"]
    scored = scored.dropna(subset=component_columns + ["FuzzyScore"])
    target_columns, objective_weights = _learning_target_columns(scored, project.objectives, component_columns)
    scored = scored.dropna(subset=target_columns)
    if scored.shape[0] < max(4, len(project.factors) + 1):
        return _generate_initial_design_from_rows(project, experiments_df, n_points, starting_candidate_id)
    # Deterministic health clamp (api parity): with a decent incumbent the
    # engine works -- ignore phase/hints/candidates from any source.
    if "FuzzyScore" in scored.columns and len(scored):
        incumbent = float(pd.to_numeric(scored["FuzzyScore"], errors="coerce").max())
        if incumbent >= 0.35:
            # strategist meta-control inert on healthy campaigns (api
            # parity); soft constraints exempt -- operator-stated
            # infeasibility approved in the confirmation summary applies
            # regardless of campaign health
            strategy_phase = None
            prior_pseudo = None
            llm_candidates = None
    if prior_pseudo and _prior_gate_open(scored):
        # LGBO-style confidence gating: hints engage only under floor
        # pathology (see api optimizer note); healthy campaigns stay clean.
        observed_mean = {
            column: float(scored[column].mean())
            for column in target_columns
            if column in scored.columns and len(scored)
        }
        pseudo = _validated_pseudo_rows(project, prior_pseudo, target_columns, observed_mean)
        if pseudo is not None and len(pseudo):
            scored = pd.concat([scored, pseudo], ignore_index=True)
    scored = _active_training_set(
        scored,
        project.factors,
        target_columns,
        max_rows=MAX_TRAINING_ROWS,
        seed=79 + project.latest_batch_id,
    )
    runtime = _bo_runtime_config(
        n_factors=len(project.factors),
        n_rows=int(scored.shape[0]),
        candidate_pool_size=candidate_pool_size,
        ensemble_size=ensemble_size,
        training_steps=training_steps,
    )

    encoder = TorchFactorEncoder(project.factors)
    feature_columns = [factor.name for factor in project.factors]
    X = encoder.encode_frame(scored[feature_columns])
    Y = torch.tensor(scored[target_columns].to_numpy(dtype=np.float32), dtype=torch.float32)

    model = TorchObjectiveEnsemble(
        input_dim=encoder.input_dim,
        n_objectives=Y.shape[1],
        objective_weights=objective_weights,
        ensemble_size=int(runtime["ensemble_size"]),
        training_steps=int(runtime["training_steps"]),
    )
    model.fit(X, Y)

    observed_scores_tensor = torch.tensor(scored["FuzzyScore"].to_numpy(dtype=np.float32), dtype=torch.float32)
    candidates_df = _propose_candidate_frame(
        factors=project.factors,
        encoder=encoder,
        model=model,
        observed_df=scored[feature_columns],
        observed_scores=observed_scores_tensor,
        n_points=n_points,
        raw_samples=int(runtime["candidate_pool_size"]),
        seed=project.latest_batch_id + 101,
    )
    candidates_df = _remove_existing_points(candidates_df, scored[feature_columns], project.factors)
    llm_rows = _validated_llm_candidate_rows(project, llm_candidates)
    if llm_rows is not None and len(llm_rows):
        candidates_df = pd.concat([candidates_df, llm_rows], ignore_index=True)
    n_llm = len(llm_rows) if llm_rows is not None else 0
    if len(candidates_df) < n_points:
        fill = _initial_design_frame(
            factors=project.factors,
            observed_rows=experiments_df,
            n_points=n_points - len(candidates_df),
            seed=project.latest_batch_id + 157,
            scored_rows=scored,
            strategy="coverage",
        )
        candidates_df = (
            pd.concat([candidates_df, fill], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        candidates_df = _remove_existing_points(candidates_df, _observed_factor_frame(experiments_df, project.factors), project.factors)
    if len(candidates_df) < n_points:
        raise TorchBOError(f"Only {len(candidates_df)} untested candidates are available in the configured design space.")

    X_candidate = encoder.encode_frame(candidates_df[feature_columns])
    X_candidate_unit = encoder.to_unit_frame(candidates_df[feature_columns])
    X_baseline_unit = encoder.to_unit_frame(scored[feature_columns])
    recent_gain = _recent_incumbent_gain(scored, n_points)
    floor_pathology = False
    if recent_gain is not None and "FuzzyScore" in getattr(scored, "columns", []) and len(scored):
        try:
            floor_pathology = float(pd.to_numeric(scored["FuzzyScore"], errors="coerce").max()) < 0.35
        except Exception:
            floor_pathology = False
    # Deterministic acquisition sampling: derive the MC seed from the data
    # fingerprint (API-engine parity) so auxiliary fits cannot shift batches.
    data_seed = int(observed_scores_tensor.sum().item() * 1e6) % (2**31)
    neis = torch.Generator(device="cpu")
    neis.manual_seed(101 + data_seed % (2**30))
    selected_indices, selected_scores = _select_batch_by_noisy_ei(
        factors=project.factors,
        model=model,
        X_candidate=X_candidate,
        X_baseline=X,
        X_candidate_unit=X_candidate_unit,
        X_baseline_unit=X_baseline_unit,
        observed_scores=observed_scores_tensor,
        n_points=n_points + n_llm,
        strategy_phase=strategy_phase,
        soft_constraints=soft_constraints,
        recent_gain=recent_gain,
        floor_pathology=floor_pathology,
        generator=neis,
    )

    ranked = candidates_df.iloc[selected_indices].copy().reset_index(drop=True)
    ranked["_score"] = np.array(selected_scores, dtype=float)
    if n_llm:
        # LLINBO-style justify gate: an LLM proposal displaces an engine point
        # only when its acquisition score is within tolerance of the best
        # engine-ranked candidate; otherwise the engine ranking stands.
        is_llm = ranked["_llm"].fillna(0).astype(bool).tolist()
        engine_order = [k for k in range(len(ranked)) if not is_llm[k]]
        llm_order = [k for k in range(len(ranked)) if is_llm[k]]
        best_engine = max((ranked.iloc[k]["_score"] for k in engine_order), default=0.0)
        # LLINBO-style decaying tolerance (see api optimizer): permissive
        # early, strict as data accumulates.
        bo_rounds = max(1, (len(scored) - n_points) // max(1, n_points) + 1)
        tolerance = max(0.02, (0.35 / bo_rounds) * abs(best_engine))
        for k in sorted(llm_order, key=lambda k: -ranked.iloc[k]["_score"]):
            weakest = min(engine_order, key=lambda k: ranked.iloc[k]["_score"], default=None)
            if weakest is None:
                break
            if ranked.iloc[k]["_score"] >= best_engine - tolerance:
                engine_order.remove(weakest)
                engine_order.append(k)
            else:
                break
        final_order = sorted(engine_order, key=lambda k: -ranked.iloc[k]["_score"])[:n_points]
        ranked = ranked.iloc[final_order].copy().reset_index(drop=True)

    batch_id = project.latest_batch_id + 1
    suggestions = []
    for index, row in enumerate(ranked.to_dict(orient="records")):
        score = float(row.pop("_score"))
        row.pop("_llm", None)
        suggestions.append(
            DesignSuggestion(batch_id=batch_id, candidate_id=starting_candidate_id + index, score=score, values=row)
        )
    return _apply_diagnostic_point(
        suggestions, project, diagnostic_point,
        _observed_factor_frame(experiments_df, project.factors))


def _apply_diagnostic_point(suggestions, project, diagnostic_point, observed):
    """Replace the weakest engine suggestion with an operator-approved
    diagnostic point: a real evaluation of the agent's best hypothesis.

    Insertion happens after selection, so engine-chosen rows are unchanged;
    the point is domain-clamped and skipped when it duplicates an observed
    row. Exempt from the health clamp so a locked-in basin stays
    challengeable. Selection mirrors the API engine's guarded blend."""
    if not isinstance(diagnostic_point, dict) or not diagnostic_point:
        return suggestions
    engine = [s for s in suggestions if s.strategy == "greedy_noisy_ei"]
    if not engine:
        return suggestions
    rows = _validated_llm_candidate_rows(project, [diagnostic_point])
    if rows is None or not len(rows):
        return suggestions
    values = rows.iloc[0].to_dict()
    values.pop("_llm", None)
    if observed is not None and len(observed):
        # categorical-aware duplicate check: float() on a category label
        # would raise (api parity)
        for record in observed.to_dict(orient="records"):
            same = True
            for f in project.factors:
                candidate = values.get(f.name)
                existing = record.get(f.name)
                if f.kind == "categorical":
                    if str(candidate) != str(existing):
                        same = False
                        break
                    continue
                try:
                    if abs(float(candidate) - float(existing)) >= 1e-9:
                        same = False
                        break
                except (TypeError, ValueError):
                    same = False
                    break
            if same:
                return suggestions
    weakest = min(engine, key=lambda s: s.score)
    replacement = DesignSuggestion(batch_id=weakest.batch_id, candidate_id=weakest.candidate_id, score=0.0, values=values, strategy="diagnostic_point")
    return [replacement if s is weakest else s for s in suggestions]


def bo_diagnostics(
    project: ProjectState,
    experiments_df: pd.DataFrame,
    *,
    candidate_pool_size: int = 2048,
    ensemble_size: int = 5,
    training_steps: int = 300,
    language: Language = Language.ZH,
) -> dict[str, object]:
    """Return surrogate diagnostics for a project's experiment history.

    Fits the ensemble on completed rows and reports per-factor effects
    (importance, recommended range, uncertainty) plus fit and runtime
    summaries. Returns {"model_ready": False, "reason": ...} when there are
    too few completed rows to train; `language` selects the reason string.
    Never mutates campaign state."""
    _require_torch()
    _validate_factor_domains(project.factors)
    clean = _drop_incomplete_rows(project, experiments_df)
    minimum_rows = max(4, len(project.factors) + 1)
    if clean.shape[0] < minimum_rows:
        if language == Language.EN:
            return {
                "model_ready": False,
                "reason": f"Need at least {minimum_rows} completed experiments.",
                "factor_effects": [],
            }
        else:
            return {
                "model_ready": False,
                "reason": f"已完成实验不足，至少需要 {minimum_rows} 条。",
                "factor_effects": [],
            }

    scored, _ = score_components(clean.copy(), project.objectives)
    component_columns = [
        f"_desirability_{objective.name}"
        for objective in project.objectives
        if f"_desirability_{objective.name}" in scored.columns
    ]
    if not component_columns:
        scored["_desirability_scalar"] = score_rows(scored, project.objectives)
        component_columns = ["_desirability_scalar"]
    scored = scored.dropna(subset=component_columns + ["FuzzyScore"])
    target_columns, objective_weights = _learning_target_columns(scored, project.objectives, component_columns)
    scored = scored.dropna(subset=target_columns)
    if scored.shape[0] < minimum_rows:
        if language == Language.EN:
            return {
                "model_ready": False,
                "reason": f"Need at least {minimum_rows} scored samples.",
                "factor_effects": [],
            }
        else:
            return {
                "model_ready": False,
                "reason": f"有效评分样本不足，至少需要 {minimum_rows} 条。",
                "factor_effects": [],
            }
    scored = _active_training_set(
        scored,
        project.factors,
        target_columns,
        max_rows=MAX_TRAINING_ROWS,
        seed=83 + project.latest_batch_id,
    )
    runtime = _bo_runtime_config(
        n_factors=len(project.factors),
        n_rows=int(scored.shape[0]),
        candidate_pool_size=candidate_pool_size,
        ensemble_size=ensemble_size,
        training_steps=training_steps,
    )

    feature_columns = [factor.name for factor in project.factors]
    encoder = TorchFactorEncoder(project.factors)
    X = encoder.encode_frame(scored[feature_columns])
    Y = torch.tensor(scored[target_columns].to_numpy(dtype=np.float32), dtype=torch.float32)

    model = TorchObjectiveEnsemble(
        input_dim=encoder.input_dim,
        n_objectives=Y.shape[1],
        objective_weights=objective_weights,
        ensemble_size=int(runtime["ensemble_size"]),
        training_steps=int(runtime["training_steps"]),
    )
    model.fit(X, Y)
    observed_scores = torch.tensor(scored["FuzzyScore"].to_numpy(dtype=np.float32), dtype=torch.float32)
    candidate_df = _propose_candidate_frame(
        factors=project.factors,
        encoder=encoder,
        model=model,
        observed_df=scored[feature_columns],
        observed_scores=observed_scores,
        n_points=max(96, len(project.factors) * 10),
        raw_samples=min(MAX_DIAGNOSTIC_CANDIDATE_POOL_SIZE, max(int(runtime["candidate_pool_size"]), len(project.factors) * 512)),
        seed=project.latest_batch_id + 211,
    )
    candidate_df = _remove_existing_points(candidate_df, scored[feature_columns], project.factors)
    observed_df = scored[feature_columns].drop_duplicates().reset_index(drop=True)
    analysis_df = (
        pd.concat([candidate_df, observed_df], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if analysis_df.empty:
        analysis_df = observed_df.copy()

    X_candidate = encoder.encode_frame(analysis_df[feature_columns])
    score_samples = model.sample_scores(X_candidate, n_samples=160)
    score_mean = score_samples.mean(dim=0)
    score_std = score_samples.std(dim=0, unbiased=False)
    acquisition = model.noisy_expected_improvement(
        X=X_candidate,
        X_baseline=X,
        n_samples=160,
    )

    ranked_indices = torch.argsort(acquisition, descending=True).detach().cpu().tolist()
    top_k = min(max(32, len(project.factors) * 8), len(analysis_df))
    top_indices = ranked_indices[:top_k]
    top_candidates = (
        analysis_df.iloc[top_indices].reset_index(drop=True)
        if top_indices
        else analysis_df.iloc[:0].copy()
    )
    anchor_k = min(max(12, len(project.factors) * 4), len(top_candidates))
    anchor_df = (
        top_candidates.iloc[:anchor_k].reset_index(drop=True)
        if anchor_k > 0
        else analysis_df.head(min(12, len(analysis_df))).reset_index(drop=True)
    )

    score_series = coerce_numeric_series(scored["FuzzyScore"])
    best_idx = score_series.fillna(-1).idxmax()
    best_observed = scored.loc[best_idx].to_dict()

    total_importance = 0.0
    factor_effects: list[dict[str, object]] = []
    for factor in project.factors:
        support_values = _diagnostic_support_values(factor, analysis_df[factor.name] if factor.name in analysis_df.columns else pd.Series(dtype=object))
        if not support_values:
            continue
        if anchor_df.empty:
            continue
        eval_records: list[dict[str, object]] = []
        for anchor in anchor_df.to_dict(orient="records"):
            for support_value in support_values:
                record = dict(anchor)
                record[factor.name] = support_value
                eval_records.append(record)
        eval_df = pd.DataFrame(eval_records, columns=feature_columns)
        eval_X = encoder.encode_frame(eval_df[feature_columns])
        eval_samples = model.sample_scores(eval_X, n_samples=96)
        eval_mean = eval_samples.mean(dim=0).detach().cpu().numpy()
        eval_std = eval_samples.std(dim=0, unbiased=False).detach().cpu().numpy()
        n_support = len(support_values)
        n_anchor = len(anchor_df)
        mean_grid = eval_mean.reshape(n_anchor, n_support)
        std_grid = eval_std.reshape(n_anchor, n_support)
        if n_support >= 3:
            low_curve = np.quantile(mean_grid, 0.1, axis=1)
            high_curve = np.quantile(mean_grid, 0.9, axis=1)
            importance_raw = float(np.mean(high_curve - low_curve))
        else:
            importance_raw = float(np.mean(mean_grid.max(axis=1) - mean_grid.min(axis=1)))
        total_importance += importance_raw

        top_values = top_candidates[factor.name] if factor.name in top_candidates.columns else pd.Series(dtype=object)
        recommended_range = _diagnostic_recommended_range(factor, top_values)
        factor_effects.append(
            {
                "name": factor.name,
                "kind": factor.kind,
                "current_best": best_observed.get(factor.name),
                "recommended_range": recommended_range,
                "importance_raw": importance_raw,
                "uncertainty": float(np.mean(std_grid)),
                "support_size": int(n_support),
            }
        )

    if total_importance > 1e-12:
        for item in factor_effects:
            item["importance"] = float(item["importance_raw"]) / total_importance
    else:
        for item in factor_effects:
            item["importance"] = 0.0

    factor_effects.sort(
        key=lambda item: (
            float(item.get("importance") or 0.0),
            float(item.get("uncertainty") or 0.0),
        ),
        reverse=True,
    )
    return {
        "model_ready": True,
        "reason": "",
        "surrogate": "weighted_calibrated_bootstrap_ensemble",
        "acquisition": "noisy_expected_improvement",
        "batch_policy": "greedy_noisy_ei",
        "factor_effects": factor_effects,
        "candidate_count": int(len(analysis_df)),
        "top_candidate_count": int(len(top_candidates)),
        "anchor_count": int(len(anchor_df)),
        "best_score": float(score_series.fillna(-1).max()),
        "top_expected_score": float(score_mean[top_indices].mean().item()) if top_indices else None,
        "top_uncertainty": float(score_std[top_indices].mean().item()) if top_indices else None,
        "fit_summary": model.fit_summary,
        "runtime": runtime,
    }


def _diagnostic_support_values(
    factor: FactorSpec,
    series: pd.Series,
    *,
    max_points: int = 9,
) -> list[object]:
    if factor.kind == "categorical":
        values = [str(value) for value in factor.categories if str(value).strip()]
        if values:
            return values
        observed = [str(value) for value in series.dropna().astype(str).tolist() if str(value).strip()]
        return list(dict.fromkeys(observed))
    if factor.kind == "ordinal":
        min_value = _factor_min(factor)
        max_value = _factor_max(factor)
        step = _factor_step(factor)
        levels = _ordinal_level_count(factor)
        if levels <= max_points:
            return [round(min_value + idx * step, 6) for idx in range(max(levels, 1))]
        grid = np.linspace(min_value, max_value, num=max_points)
        snapped = sorted({round(min_value + round((value - min_value) / step) * step, 6) for value in grid})
        return snapped
    numeric = coerce_numeric_series(series).dropna()
    min_value = _factor_min(factor)
    max_value = _factor_max(factor)
    if max_value - min_value < 1e-12:
        return [float(min_value)]
    quantiles = np.linspace(0.05, 0.95, num=max_points)
    if numeric.nunique() >= max_points:
        return sorted({round(float(numeric.quantile(q)), 6) for q in quantiles})
    return [round(float(value), 6) for value in np.linspace(min_value, max_value, num=max_points)]


def _diagnostic_recommended_range(factor: FactorSpec, series: pd.Series) -> object:
    if factor.kind == "categorical":
        values = [str(value) for value in series.dropna().astype(str).tolist() if str(value).strip()]
        ordered = list(dict.fromkeys(values))
        return ordered[:3]
    numeric = coerce_numeric_series(series).dropna()
    if numeric.empty:
        return []
    if factor.kind == "ordinal":
        step = _factor_step(factor)
        base = _factor_min(factor)
        low = numeric.quantile(0.15)
        high = numeric.quantile(0.85)
        low = round((float(low) - base) / step) * step + base
        high = round((float(high) - base) / step) * step + base
        return [round(float(low), 6), round(float(high), 6)]
    return [round(float(numeric.quantile(0.15)), 6), round(float(numeric.quantile(0.85)), 6)]


def _bo_runtime_config(
    *,
    n_factors: int,
    n_rows: int,
    candidate_pool_size: int,
    ensemble_size: int,
    training_steps: int,
) -> dict[str, int]:
    adaptive_pool = min(
        MAX_CANDIDATE_POOL_SIZE,
        max(candidate_pool_size, n_factors * 512, int(math.sqrt(max(n_rows, 1))) * 256, 2048),
    )
    adaptive_ensemble = max(4, min(8, ensemble_size + (1 if n_rows >= 24 else 0)))
    adaptive_steps = max(training_steps, min(900, 220 + n_factors * 24 + n_rows * 4))
    return {
        "candidate_pool_size": int(adaptive_pool),
        "ensemble_size": int(adaptive_ensemble),
        "training_steps": int(adaptive_steps),
    }


def _learning_target_columns(
    scored: pd.DataFrame,
    objectives: list[ObjectiveSpec],
    component_columns: list[str],
) -> tuple[list[str], list[float]]:
    """Build BO training targets that keep response gradients in low-score plateaus."""

    target_columns: list[str] = []
    weights: list[float] = []
    for objective in objectives:
        desirability_column = f"_desirability_{objective.name}"
        if desirability_column not in component_columns or objective.name not in scored.columns:
            continue
        formal = coerce_numeric_series(scored[desirability_column]).astype(float)
        raw = coerce_numeric_series(scored[objective.name])
        progress = _objective_progress(raw, objective)
        if progress is None:
            learning = formal
        else:
            progress_utility = 0.06 + 0.58 * progress
            learning = np.maximum(formal.to_numpy(dtype=float), progress_utility)
        column = f"_bo_target_{objective.name}"
        scored[column] = np.clip(np.asarray(learning, dtype=float), 1e-6, 1.0)
        target_columns.append(column)
        weights.append(float(objective.weight or 1.0))
    if target_columns:
        if len(target_columns) <= 2:
            return target_columns, weights
        matrix = scored[target_columns].to_numpy(dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.sum() <= 0:
            weight_array = np.ones_like(weight_array)
        weight_array = weight_array / weight_array.sum()
        scored["_bo_target_score"] = np.exp(
            np.sum(np.log(np.clip(matrix, 1e-6, 1.0)) * weight_array, axis=1)
        )
        return ["_bo_target_score"], [1.0]
    return component_columns, [1.0] * len(component_columns)


def _objective_progress(series: pd.Series, objective: ObjectiveSpec) -> np.ndarray | None:
    numeric = coerce_numeric_series(series).astype(float)
    if numeric.notna().sum() < 2:
        return None
    values = numeric.to_numpy(dtype=float)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return None
    observed = values[valid]
    low = float(np.min(observed))
    high = float(np.max(observed))
    span = high - low
    if span < 1e-12:
        return None
    if objective.goal == "minimize":
        progress = (high - values) / span
    elif objective.goal == "target" and objective.target_value not in (None, ""):
        target = parse_numeric_value(objective.target_value)
        if target is None:
            target = (low + high) / 2.0
        radius = max(abs(high - target), abs(low - target), span * 0.5, 1e-12)
        progress = 1.0 - np.abs(values - target) / radius
    else:
        progress = (values - low) / span
    progress = np.where(valid, progress, np.nan)
    return np.clip(progress, 0.0, 1.0)


def score_rows(df: pd.DataFrame, objectives: list[ObjectiveSpec]) -> np.ndarray:
    available_objectives = [objective for objective in objectives if objective.name in df.columns]
    if not available_objectives:
        return np.ones(len(df))
    component_scores = []
    for objective in available_objectives:
        series = df[objective.name]
        component_scores.append(_score_series(series, objective))
    score_matrix = np.vstack(component_scores).T
    safe_matrix = np.clip(score_matrix, 1e-6, 1.0)
    weights = np.array([objective.weight for objective in available_objectives], dtype=float)
    weights = weights / weights.sum()
    return np.exp(np.sum(np.log(safe_matrix) * weights, axis=1))


def _score_series(series: pd.Series, objective: ObjectiveSpec) -> np.ndarray:
    goal: GoalType = objective.goal
    present = series[~series.map(is_missing_value)]
    numeric_all = coerce_numeric_series(series)
    numeric = numeric_all.loc[present.index]
    if len(present) and numeric.notna().sum() == len(present):
        values = numeric_all.to_numpy(dtype=float)
        observed = numeric.dropna().astype(float).to_numpy()
        lower = np.nanmin(observed)
        upper = np.nanmax(observed)
        span = max(upper - lower, 1e-9)
        if goal == "maximize":
            return np.clip((values - lower) / span, 0.0, 1.0)
        if goal == "minimize":
            return np.clip((upper - values) / span, 0.0, 1.0)
        target = parse_numeric_value(objective.target_value)
        if target is None:
            target = float(np.nanmedian(values))
        return np.clip(1.0 - np.abs(values - target) / span, 0.0, 1.0)

    labels = series.astype(str).to_numpy()
    if goal == "target" and objective.target_value is not None:
        target_label = str(objective.target_value)
        return np.where(labels == target_label, 1.0, 0.2)
    most_common = pd.Series(labels).mode().iloc[0]
    if goal == "maximize":
        return np.where(labels == most_common, 1.0, 0.5)
    return np.where(labels == most_common, 0.2, 1.0)


class TorchFactorEncoder:
    def __init__(self, factors: list[FactorSpec]):
        _require_torch()
        self.factors = factors
        self.input_dim = 0
        self._category_maps: dict[str, dict[str, int]] = {}
        for factor in self.factors:
            if factor.kind == "categorical":
                self._category_maps[factor.name] = {
                    value: index for index, value in enumerate(factor.categories)
                }
                self.input_dim += max(len(factor.categories), 1)
            else:
                self.input_dim += 1

    def encode_frame(self, df: pd.DataFrame) -> torch.Tensor:
        rows = [self.encode_row(record) for record in df.to_dict(orient="records")]
        if not rows:
            return torch.empty((0, self.input_dim), dtype=torch.float32)
        return torch.stack(rows, dim=0)

    def to_unit_frame(self, df: pd.DataFrame) -> torch.Tensor:
        rows = [self.to_unit_row(record) for record in df.to_dict(orient="records")]
        if not rows:
            return torch.empty((0, max(len(self.factors), 1)), dtype=torch.float32)
        return torch.stack(rows, dim=0)

    def encode_row(self, row: dict[str, object]) -> torch.Tensor:
        return self.encode_unit(self.to_unit_row(row), relaxed=False).squeeze(0)

    def encode_unit(self, X_unit: torch.Tensor, relaxed: bool = False) -> torch.Tensor:
        if X_unit.ndim == 1:
            X_unit = X_unit.unsqueeze(0)
        encoded_parts = []
        for idx, factor in enumerate(self.factors):
            values = X_unit[:, idx].clamp(0.0, 1.0)
            if factor.kind == "categorical":
                n_categories = max(len(factor.categories), 1)
                if relaxed:
                    centers = (
                        torch.arange(n_categories, device=X_unit.device, dtype=X_unit.dtype) + 0.5
                    ) / n_categories
                    logits = -24.0 * (values.unsqueeze(-1) - centers).pow(2)
                    encoded_parts.append(torch.softmax(logits, dim=-1))
                else:
                    indices = torch.clamp((values * n_categories).floor().long(), max=n_categories - 1)
                    encoded_parts.append(
                        torch.nn.functional.one_hot(indices, num_classes=n_categories).to(
                            dtype=X_unit.dtype
                        )
                    )
            else:
                encoded_parts.append(values.unsqueeze(-1))
        return torch.cat(encoded_parts, dim=-1)

    def to_unit_row(self, row: dict[str, object]) -> torch.Tensor:
        values = []
        for factor in self.factors:
            raw = row.get(factor.name)
            if factor.kind == "categorical":
                categories = factor.categories or ["A", "B"]
                category_map = self._category_maps.get(factor.name, {})
                index = category_map.get(str(raw), 0)
                values.append((float(index) + 0.5) / float(len(categories)))
                continue
            numeric = parse_numeric_value(raw)
            if numeric is None:
                raise TorchBOError(f"Factor {factor.name} has non-numeric value {raw!r}.")
            min_value = _factor_min(factor)
            max_value = _factor_max(factor)
            if factor.kind == "ordinal":
                levels = _ordinal_level_count(factor)
                if levels <= 1:
                    values.append(0.0)
                else:
                    index = round((numeric - min_value) / _factor_step(factor))
                    values.append(float(max(0, min(index, levels - 1))) / float(levels - 1))
            else:
                span = max(max_value - min_value, 1e-9)
                values.append((numeric - min_value) / span)
        return torch.tensor(values or [0.5], dtype=torch.float32)


class TorchEnsembleRegressor:
    def __init__(
        self,
        input_dim: int,
        ensemble_size: int = 5,
        hidden_dim: int = 64,
        training_steps: int = 300,
        learning_rate: float = 1e-3,
    ):
        _require_torch()
        self.input_dim = input_dim
        self.ensemble_size = ensemble_size
        self.hidden_dim = hidden_dim
        self.training_steps = training_steps
        self.learning_rate = learning_rate
        self.models: list[nn.Module] = []
        self.target_transform = BoundedTargetTransform()
        self.fit_summary: dict[str, float | int | None] = {}
        self.uncertainty_scale = 1.0
        self.uncertainty_floor = 0.0
        self.member_weights = torch.empty((0,), dtype=torch.float32)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        if X.ndim != 2:
            raise TorchBOError("X must be a 2D tensor.")
        if y.ndim != 1:
            y = y.view(-1)
        y_normalized = self.target_transform.fit(y)

        self.models = []
        sample_count = X.shape[0]
        validation_size = 0
        if sample_count >= 10:
            validation_size = min(max(sample_count // 5, 2), max(sample_count - 4, 0))
        if validation_size > 0:
            permutation = torch.randperm(sample_count)
            val_indices = permutation[:validation_size]
            train_indices = permutation[validation_size:]
            if train_indices.numel() == 0:
                train_indices = permutation
                val_indices = permutation[:0]
        else:
            train_indices = torch.arange(sample_count)
            val_indices = torch.arange(0)
        X_train = X[train_indices]
        y_train = y_normalized[train_indices]
        X_val = X[val_indices] if val_indices.numel() else None
        y_val = y[val_indices] if val_indices.numel() else None
        validation_losses: list[float] = []
        training_losses: list[float] = []
        epochs_ran: list[int] = []
        for model_index in range(self.ensemble_size):
            torch.manual_seed(2026 + model_index)
            model = _Regressor(self.input_dim, self.hidden_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            loss_fn = nn.SmoothL1Loss(beta=0.1)

            bootstrap_size = max(int(X_train.shape[0]), 1)
            bootstrap_indices = torch.randint(low=0, high=bootstrap_size, size=(bootstrap_size,))
            X_boot = X_train[bootstrap_indices]
            y_boot = y_train[bootstrap_indices].unsqueeze(-1)

            model.train()
            best_state = None
            best_metric = float("inf")
            best_epoch = 0
            patience = max(20, min(80, self.training_steps // 5))
            for step in range(self.training_steps):
                optimizer.zero_grad()
                prediction = model(X_boot)
                loss = loss_fn(prediction, y_boot)
                loss.backward()
                optimizer.step()
                metric = float(loss.item())
                if X_val is not None and y_val is not None and X_val.shape[0] > 0:
                    model.eval()
                    with torch.no_grad():
                        val_prediction = model(X_val).squeeze(-1)
                        val_prob = self.target_transform.inverse(val_prediction)
                        metric = float(torch.sqrt(torch.mean((val_prob - y_val) ** 2)).item())
                    model.train()
                if metric + 1e-6 < best_metric:
                    best_metric = metric
                    best_epoch = step
                    best_state = copy.deepcopy(model.state_dict())
                elif step - best_epoch >= patience:
                    break
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            self.models.append(model)
            epochs_ran.append(best_epoch + 1)
            training_losses.append(float(loss.item()))
            validation_losses.append(best_metric if best_metric < float("inf") else float(loss.item()))
        self.member_weights = self._build_member_weights(validation_losses, device=X.device, dtype=X.dtype)
        self.fit_summary = {
            "train_size": int(X_train.shape[0]),
            "validation_size": int(X_val.shape[0]) if X_val is not None else 0,
            "mean_train_loss": float(np.mean(training_losses)) if training_losses else None,
            "mean_validation_rmse": float(np.mean(validation_losses)) if validation_losses else None,
            "mean_epochs": int(round(float(np.mean(epochs_ran)))) if epochs_ran else 0,
            "effective_ensemble_size": self._effective_member_count(),
            "training_loss": "smooth_l1",
        }
        if X_val is not None and y_val is not None and X_val.shape[0] > 0:
            y_val_normalized = self.target_transform.transform(y_val)
            mean_pred, std_pred = self.predict_normalized(X_val)
            residual = (y_val_normalized - mean_pred).abs()
            scale = float(torch.sqrt(torch.mean(residual.pow(2))).item() / max(float(std_pred.mean().item()), 1e-6))
            self.uncertainty_scale = float(np.clip(scale, 0.5, 3.0))
            self.uncertainty_floor = float(max(0.04, 0.18 / math.sqrt(max(sample_count, 1))))
            self.fit_summary["uncertainty_scale"] = self.uncertainty_scale
        else:
            self.uncertainty_scale = 2.0
            self.uncertainty_floor = float(max(0.12, 0.45 / math.sqrt(max(sample_count, 1))))
            self.fit_summary["uncertainty_scale"] = self.uncertainty_scale
        self.fit_summary["uncertainty_floor"] = self.uncertainty_floor

    def _predict_members(self, X: torch.Tensor) -> torch.Tensor:
        if not self.models:
            raise TorchBOError("The ensemble model has not been fitted.")
        return torch.stack([model(X).squeeze(-1) for model in self.models], dim=0)

    def _member_weight_tensor(self, device: torch.device, dtype: torch.dtype, count: int) -> torch.Tensor:
        if self.member_weights.numel() != count:
            return torch.full((count,), 1.0 / max(count, 1), dtype=dtype, device=device)
        weights = self.member_weights.to(device=device, dtype=dtype)
        return weights / weights.sum().clamp_min(1e-6)

    def _weighted_moments(self, predictions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self._member_weight_tensor(
            device=predictions.device,
            dtype=predictions.dtype,
            count=predictions.shape[0],
        )
        mean_pred = (weights.unsqueeze(-1) * predictions).sum(dim=0)
        variance = (weights.unsqueeze(-1) * (predictions - mean_pred.unsqueeze(0)).pow(2)).sum(dim=0)
        std_pred = variance.clamp_min(1e-6).sqrt() * float(self.uncertainty_scale)
        std_pred = std_pred.clamp_min(float(self.uncertainty_floor))
        return mean_pred, std_pred

    def _build_member_weights(
        self,
        validation_losses: list[float],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not validation_losses:
            return torch.full((self.ensemble_size,), 1.0 / max(self.ensemble_size, 1), dtype=dtype, device=device)
        values = np.asarray(validation_losses, dtype=float)
        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            return torch.full((len(values),), 1.0 / max(len(values), 1), dtype=dtype, device=device)
        safe_values = np.where(finite_mask, values, np.nanmax(values[finite_mask]) + 1.0)
        scale = max(float(np.median(safe_values[finite_mask])), 1e-6)
        logits = -safe_values / scale
        logits = logits - float(np.max(logits))
        weights = np.exp(logits)
        weights = weights / max(float(np.sum(weights)), 1e-9)
        return torch.tensor(weights, dtype=dtype, device=device)

    def _effective_member_count(self) -> float:
        if self.member_weights.numel() == 0:
            return float(self.ensemble_size)
        weights = self.member_weights.detach().cpu().numpy()
        denom = float(np.sum(np.square(weights)))
        if denom <= 1e-9:
            return float(self.ensemble_size)
        return float(1.0 / denom)

    def predict_normalized(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            predictions = self._predict_members(X)
        return self._weighted_moments(predictions)

    def predict_normalized_grad(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        predictions = self._predict_members(X)
        return self._weighted_moments(predictions)

    def sample_prob(
        self,
        X: torch.Tensor,
        n_samples: int = 128,
        base_samples: torch.Tensor | None = None,
        with_grad: bool = False,
    ) -> torch.Tensor:
        if base_samples is None:
            base_samples = torch.randn(
                (n_samples, X.shape[0]),
                dtype=X.dtype,
                device=X.device,
            )
        mean_pred, std_pred = (
            self.predict_normalized_grad(X) if with_grad else self.predict_normalized(X)
        )
        latent = mean_pred.unsqueeze(0) + std_pred.unsqueeze(0) * base_samples
        return self.target_transform.inverse(latent)


class TorchObjectiveEnsemble:
    def __init__(
        self,
        input_dim: int,
        n_objectives: int,
        objective_weights: list[float] | None = None,
        ensemble_size: int = 5,
        hidden_dim: int = 64,
        training_steps: int = 300,
        learning_rate: float = 1e-3,
    ):
        if objective_weights and len(objective_weights) == n_objectives:
            weights = torch.tensor(objective_weights, dtype=torch.float32)
        else:
            weights = torch.ones(n_objectives, dtype=torch.float32)
        self.objective_weights = weights / weights.sum().clamp_min(1e-6)
        self.models = [
            TorchEnsembleRegressor(
                input_dim=input_dim,
                ensemble_size=ensemble_size,
                hidden_dim=hidden_dim,
                training_steps=training_steps,
                learning_rate=learning_rate,
            )
            for _ in range(n_objectives)
        ]
        self.fit_summary: dict[str, object] = {}

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        if Y.ndim == 1:
            Y = Y.unsqueeze(-1)
        if Y.shape[1] != len(self.models):
            raise TorchBOError("The number of targets does not match the number of objective models.")
        self.objective_weights = self.objective_weights.to(device=Y.device, dtype=Y.dtype)
        for idx, model in enumerate(self.models):
            model.fit(X, Y[:, idx])
        validation_values = [
            float(model.fit_summary["mean_validation_rmse"])
            for model in self.models
            if model.fit_summary.get("mean_validation_rmse") is not None
        ]
        calibration_values = [
            float(model.fit_summary["uncertainty_scale"])
            for model in self.models
            if model.fit_summary.get("uncertainty_scale") is not None
        ]
        self.fit_summary = {
            "objective_count": int(len(self.models)),
            "train_size": int(self.models[0].fit_summary.get("train_size") or 0) if self.models else 0,
            "validation_size": int(self.models[0].fit_summary.get("validation_size") or 0) if self.models else 0,
            "mean_validation_rmse": float(np.mean(validation_values)) if validation_values else None,
            "uncertainty_scale": float(np.mean(calibration_values)) if calibration_values else None,
            "effective_ensemble_size": float(
                np.mean(
                    [
                        float(model.fit_summary.get("effective_ensemble_size") or model.ensemble_size)
                        for model in self.models
                    ]
                )
            ) if self.models else 0.0,
            "mean_epochs": int(
                round(
                    float(
                        np.mean(
                            [
                                float(model.fit_summary.get("mean_epochs") or 0)
                                for model in self.models
                            ]
                        )
                    )
                )
            ) if self.models else 0,
            "training_loss": "smooth_l1",
        }

    def noisy_expected_improvement(
        self,
        X: torch.Tensor,
        X_baseline: torch.Tensor,
        n_samples: int = 128,
        candidate_base_samples: torch.Tensor | None = None,
        baseline_base_samples: torch.Tensor | None = None,
        with_grad: bool = False,
    ) -> torch.Tensor:
        candidate_samples = self.sample_scores(
            X=X,
            n_samples=n_samples,
            base_samples=candidate_base_samples,
            with_grad=with_grad,
        )
        baseline_samples = self.sample_scores(
            X=X_baseline,
            n_samples=n_samples,
            base_samples=baseline_base_samples,
            with_grad=False,
        )
        baseline_best = baseline_samples.max(dim=1).values.unsqueeze(-1)
        return (candidate_samples - baseline_best).clamp_min(0.0).mean(dim=0)

    def sample_scores(
        self,
        X: torch.Tensor,
        n_samples: int = 128,
        base_samples: torch.Tensor | None = None,
        with_grad: bool = False,
    ) -> torch.Tensor:
        if base_samples is None:
            base_samples = torch.randn(
                (n_samples, X.shape[0], len(self.models)),
                dtype=X.dtype,
                device=X.device,
            )
        parts = []
        for idx, model in enumerate(self.models):
            parts.append(
                model.sample_prob(
                    X,
                    n_samples=n_samples,
                    base_samples=base_samples[:, :, idx],
                    with_grad=with_grad,
                ).clamp_min(1e-6)
            )
        return _aggregate_score_tensor(torch.stack(parts, dim=-1), self.objective_weights)


if nn is not None:
    class _Regressor(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)
else:  # pragma: no cover - import compatibility before torch is installed
    class _Regressor:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()


def _draw_unit_samples(n_points: int, n_dim: int, seed: int) -> torch.Tensor:
    sobol = torch.quasirandom.SobolEngine(dimension=n_dim, scramble=True, seed=seed)
    return sobol.draw(n_points).to(dtype=torch.float32)


def _active_training_set(
    frame: pd.DataFrame,
    factors: list[FactorSpec],
    target_columns: list[str],
    *,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    selected: set[object] = set()
    score = coerce_numeric_series(frame.get("FuzzyScore", pd.Series(dtype=float))).fillna(-1.0)
    top_indices = score.sort_values(ascending=False).head(max(120, max_rows // 4)).index.tolist()
    selected.update(top_indices)
    selected.update(score.sort_values(ascending=True).head(max(16, max_rows // 40)).index.tolist())
    if top_indices:
        feature_columns = [factor.name for factor in factors]
        candidate_frame = frame.dropna(subset=feature_columns)
        if not candidate_frame.empty:
            try:
                encoder = TorchFactorEncoder(factors)
                units = encoder.to_unit_frame(candidate_frame[feature_columns])
                anchor_units = encoder.to_unit_frame(frame.loc[top_indices, feature_columns].dropna())
                if anchor_units.numel():
                    distances = torch.cdist(units, anchor_units).min(dim=1).values.detach().cpu().numpy()
                    neighbor_order = np.argsort(distances)[: max(160, max_rows // 3)]
                    selected.update(candidate_frame.iloc[neighbor_order].index.tolist())
            except (TorchBOError, ValueError, RuntimeError):
                pass
    for target in target_columns:
        values = coerce_numeric_series(frame.get(target, pd.Series(dtype=float)))
        valid = values.dropna()
        if valid.empty:
            continue
        selected.update(valid.sort_values(ascending=False).head(24).index.tolist())
        selected.update(valid.sort_values(ascending=True).head(24).index.tolist())
    for factor in factors:
        if factor.name not in frame.columns:
            continue
        series = frame[factor.name]
        if factor.kind == "categorical":
            groups = series.astype(str)
            for _, group in frame.groupby(groups, sort=False, dropna=False):
                selected.update(group.index[: max(2, max_rows // 240)].tolist())
            continue
        numeric = coerce_numeric_series(series)
        valid = numeric.dropna()
        if valid.empty:
            continue
        selected.update(valid.sort_values(ascending=True).head(32).index.tolist())
        selected.update(valid.sort_values(ascending=False).head(32).index.tolist())
        bins = pd.qcut(valid.rank(method="first"), q=min(12, max(2, valid.nunique())), duplicates="drop")
        for _, group in frame.loc[valid.index].groupby(bins, sort=False, observed=True):
            selected.update(group.index[: max(2, max_rows // 240)].tolist())
    selected_indices = _order_active_set_indices(frame, selected, score)
    if len(selected_indices) < max_rows:
        remaining = frame.drop(index=selected_indices, errors="ignore")
        sample_size = min(max_rows - len(selected_indices), len(remaining))
        if sample_size > 0:
            selected_indices.extend(remaining.sample(n=sample_size, random_state=seed).index.tolist())
    if len(selected_indices) > max_rows:
        selected_indices = selected_indices[:max_rows]
    return frame.loc[selected_indices].reset_index(drop=True)


def _order_active_set_indices(frame: pd.DataFrame, selected: set[object], score: pd.Series) -> list[object]:
    if not selected:
        return []
    selected_set = set(selected)
    ordered = [idx for idx in score.sort_values(ascending=False).index if idx in selected_set]
    ordered.extend(idx for idx in frame.index if idx in selected_set and idx not in set(ordered))
    return ordered


def _generate_initial_design_from_rows(
    project: ProjectState,
    experiments_df: pd.DataFrame,
    n_points: int,
    starting_candidate_id: int,
) -> list[DesignSuggestion]:
    candidates_df = _initial_design_frame(
        factors=project.factors,
        observed_rows=experiments_df,
        n_points=n_points,
        seed=42 + project.latest_batch_id,
        scored_rows=experiments_df,
        strategy="balanced",
    )
    batch_id = project.latest_batch_id + 1
    return [
        DesignSuggestion(batch_id=batch_id, candidate_id=starting_candidate_id + index, score=0.0, values=row)
        for index, row in enumerate(candidates_df.to_dict(orient="records"))
    ]


def _initial_design_frame(
    factors: list[FactorSpec],
    observed_rows: pd.DataFrame,
    n_points: int,
    seed: int,
    scored_rows: pd.DataFrame | None = None,
    strategy: str = "balanced",
) -> pd.DataFrame:
    n_points = max(int(n_points), 1)
    exhaustive = _finite_design_space(factors)
    observed = _observed_factor_frame(observed_rows, factors)
    anchor = _best_observed_factor_frame(scored_rows if scored_rows is not None else observed_rows, factors)
    encoder = TorchFactorEncoder(factors)
    if exhaustive is not None:
        candidates = _remove_existing_points(exhaustive, observed, factors)
        if len(candidates) < n_points:
            raise TorchBOError(f"Only {len(candidates)} untested candidates are available in the configured design space.")
        units = encoder.to_unit_frame(candidates)
        baseline = _baseline_unit_frame(encoder, observed, anchor, factors)
        order = _rank_initial_candidates(factors, units, baseline, n_points, strategy)
        return candidates.iloc[order[:n_points]].reset_index(drop=True)

    sobol = _draw_unit_samples(n_points=max(512, n_points * 128), n_dim=max(len(factors), 1), seed=seed)
    local = _draw_initial_local_samples(factors, anchor, max(128, n_points * 32), seed + 19)
    units = _snap_unit_samples(factors, torch.cat([sobol, local], dim=0) if local.numel() else sobol)
    candidates = _unit_candidates_to_frame(factors, units)
    candidates = _remove_existing_points(candidates, observed, factors)
    if len(candidates) < n_points:
        raise TorchBOError(f"Only {len(candidates)} untested candidates are available in the configured design space.")
    candidate_units = encoder.to_unit_frame(candidates)
    baseline = _baseline_unit_frame(encoder, observed, anchor, factors)
    order = _rank_initial_candidates(factors, candidate_units, baseline, n_points, strategy)
    return candidates.iloc[order[:n_points]].reset_index(drop=True)


def _baseline_unit_frame(
    encoder: "TorchFactorEncoder",
    observed: pd.DataFrame,
    anchor: pd.DataFrame,
    factors: list[FactorSpec],
) -> torch.Tensor:
    if observed.empty and anchor.empty:
        return torch.empty((0, max(len(factors), 1)), dtype=torch.float32)
    return encoder.to_unit_frame(pd.concat([observed, anchor], ignore_index=True))


def _best_observed_factor_frame(rows: pd.DataFrame | None, factors: list[FactorSpec]) -> pd.DataFrame:
    columns = [factor.name for factor in factors]
    if rows is None or rows.empty or "FuzzyScore" not in rows.columns or any(column not in rows.columns for column in columns):
        return pd.DataFrame(columns=columns)
    scores = coerce_numeric_series(rows["FuzzyScore"])
    if scores.dropna().empty:
        return pd.DataFrame(columns=columns)
    row = rows.loc[scores.idxmax(), columns]
    if row.map(is_missing_value).any():
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([row.to_dict()], columns=columns)


def _rank_initial_candidates(
    factors: list[FactorSpec],
    units: torch.Tensor,
    baseline: torch.Tensor,
    n_points: int,
    strategy: str = "balanced",
) -> list[int]:
    if int(units.shape[0]) == 0:
        return []
    strategy = str(strategy or "balanced").lower()
    center = torch.full((1, units.shape[1]), 0.5, dtype=units.dtype, device=units.device)
    selected: list[int] = []
    remaining = list(range(int(units.shape[0])))
    current = baseline.clone() if baseline.numel() else center.clone()
    # Center-anchored start (classic DOE center point): the discrete-entropy
    # gain term (0.36) systematically outranks the center score (0.25), so
    # the max-min greedy never picks a center-proximal candidate — the design
    # starts corner-first and the comfort prior never has a home observation.
    # Pick the nearest-to-center candidate FIRST unless observations already
    # anchor the region (a local/observed campaign re-centers itself).
    if not baseline.numel() and len(factors) <= 4 and strategy in {"balanced", "coverage", "space_filling"}:
        remaining_units = units[remaining]
        center_dist = torch.cdist(remaining_units, center).squeeze(-1)
        first_position = int(torch.argmin(center_dist).item())
        index = remaining.pop(first_position)
        selected.append(index)
        current = torch.cat([current, units[index : index + 1]], dim=0)
    while remaining and len(selected) < n_points:
        remaining_units = units[remaining]
        min_dist = torch.cdist(remaining_units, current).min(dim=1).values
        center_score = 1.0 - torch.cdist(remaining_units, center).squeeze(-1).clamp(0.0, 1.0)
        if strategy in {"local", "local_best", "exploit"} and baseline.numel():
            local_score = 1.0 - torch.cdist(remaining_units, baseline[-1:]).squeeze(-1).clamp(0.0, 1.0)
            score = 0.62 * local_score + 0.28 * _normalize_selection_signal(min_dist) + 0.10 * center_score
        elif strategy in {"coverage", "space_filling", "explore"}:
            score = 0.82 * _normalize_selection_signal(min_dist) + 0.18 * center_score
        else:
            local_score = torch.zeros_like(center_score)
            if baseline.numel():
                local_score = 1.0 - torch.cdist(remaining_units, baseline[-1:]).squeeze(-1).clamp(0.0, 1.0)
            score = 0.55 * _normalize_selection_signal(min_dist) + 0.25 * center_score + 0.20 * local_score
        selected_units = (
            units[selected]
            if selected
            else torch.empty((0, units.shape[1]), dtype=units.dtype, device=units.device)
        )
        discrete_gain = _discrete_level_entropy_gain(factors, remaining_units, baseline, selected_units)
        if strategy in {"local", "local_best", "exploit"} and baseline.numel():
            score = score + 0.18 * discrete_gain
        else:
            score = score + 0.36 * discrete_gain
        position = int(torch.argmax(score).item())
        index = remaining.pop(position)
        selected.append(index)
        current = torch.cat([current, units[index : index + 1]], dim=0)
    return selected


def _draw_initial_local_samples(factors: list[FactorSpec], anchor: pd.DataFrame, n_points: int, seed: int) -> torch.Tensor:
    if anchor is None or anchor.empty:
        return torch.empty((0, max(len(factors), 1)), dtype=torch.float32)
    encoder = TorchFactorEncoder(factors)
    anchor_units = encoder.to_unit_frame(anchor)
    scores = torch.ones(anchor_units.shape[0], dtype=torch.float32)
    trust = torch.full((max(len(factors), 1),), 0.24, dtype=torch.float32)
    return _draw_local_mixed_samples(factors, anchor_units, scores, n_points, seed, trust)


def _finite_design_space(factors: list[FactorSpec], limit: int = 20000) -> pd.DataFrame | None:
    values: list[list[object]] = []
    for factor in factors:
        if factor.kind == "categorical":
            levels = [str(value) for value in factor.categories if str(value).strip()]
            if not levels:
                return None
            values.append(levels)
        elif factor.kind == "ordinal":
            min_value = _factor_min(factor)
            step = _factor_step(factor)
            levels = _ordinal_level_count(factor)
            if levels <= 0:
                return None
            values.append([round(min_value + idx * step, 6) for idx in range(levels)])
        else:
            return None
        size = math.prod(len(item) for item in values)
        if size > limit:
            return None
    records = [{factor.name: value for factor, value in zip(factors, combo)} for combo in itertools.product(*values)]
    return pd.DataFrame(records, columns=[factor.name for factor in factors]).drop_duplicates().reset_index(drop=True)


def _aggregate_score_tensor(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if values.ndim == 1:
        return values.clamp_min(1e-6)
    safe = values.clamp_min(1e-6)
    if weights is None or weights.numel() != safe.shape[-1]:
        return torch.exp(torch.log(safe).mean(dim=-1))
    normalized = weights.to(device=safe.device, dtype=safe.dtype)
    normalized = normalized / normalized.sum().clamp_min(1e-6)
    return torch.exp(torch.sum(torch.log(safe) * normalized, dim=-1))


def _propose_candidate_frame(
    factors: list[FactorSpec],
    encoder: TorchFactorEncoder,
    model: TorchObjectiveEnsemble,
    observed_df: pd.DataFrame,
    observed_scores: torch.Tensor,
    n_points: int,
    raw_samples: int,
    seed: int,
) -> pd.DataFrame:
    n_dim = max(len(factors), 1)
    unit_samples = _draw_unit_samples(n_points=max(raw_samples, 64), n_dim=n_dim, seed=seed)
    observed_units = encoder.to_unit_frame(observed_df)
    trust_region = _estimate_trust_region(
        observed_units=observed_units,
        observed_scores=observed_scores,
        n_factors=len(factors),
    )
    local_samples = _draw_local_mixed_samples(
        factors=factors,
        observed_units=observed_units,
        observed_scores=observed_scores,
        n_points=max(raw_samples // 2, n_points * 16),
        seed=seed + 17,
        trust_region=trust_region,
    )
    comfort_samples = _draw_comfort_zone_samples(
        n_points=max(raw_samples // 4, n_points * 12),
        n_dim=n_dim,
        seed=seed + 29,
    )
    baseline_X = encoder.encode_frame(observed_df)
    optimized = _optimize_acquisition(
        unit_samples=torch.cat([unit_samples, local_samples, comfort_samples], dim=0),
        encoder=encoder,
        model=model,
        baseline_X=baseline_X,
        n_candidates=n_points,
    )
    candidate_units = torch.cat([unit_samples, local_samples, comfort_samples, optimized], dim=0)
    snapped_units = _snap_unit_samples(factors, candidate_units)
    return _unit_candidates_to_frame(factors, snapped_units)


def _draw_comfort_zone_samples(n_points: int, n_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    count = max(int(n_points), 1)
    center = torch.full((1, max(n_dim, 1)), 0.5, dtype=torch.float32)
    if count == 1:
        return center
    jitter = torch.randn(count - 1, max(n_dim, 1), generator=generator, dtype=torch.float32) * 0.18
    return torch.cat([center, (0.5 + jitter).clamp(0.08, 0.92)], dim=0)


def _optimize_acquisition(
    unit_samples: torch.Tensor,
    encoder: TorchFactorEncoder,
    model: TorchObjectiveEnsemble,
    baseline_X: torch.Tensor,
    n_candidates: int,
    n_restarts: int | None = None,
    steps: int = 64,
    learning_rate: float = 0.05,
    mc_samples: int = 64,
) -> torch.Tensor:
    n_restarts = min(n_restarts or max(10, n_candidates * 8), unit_samples.shape[0])
    with torch.no_grad():
        baseline_features = encoder.encode_unit(unit_samples, relaxed=False)
        baseline_ei = model.noisy_expected_improvement(
            X=baseline_features,
            X_baseline=baseline_X,
            n_samples=mc_samples,
        )
        start_indices = torch.topk(baseline_ei, k=n_restarts, dim=0).indices

    candidates = unit_samples[start_indices].clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([candidates], lr=learning_rate)
    candidate_base_samples = torch.randn(
        (mc_samples, n_restarts, len(model.models)),
        dtype=candidates.dtype,
        device=candidates.device,
    )
    baseline_base_samples = torch.randn(
        (mc_samples, baseline_X.shape[0], len(model.models)),
        dtype=baseline_X.dtype,
        device=baseline_X.device,
    )

    for _ in range(steps):
        optimizer.zero_grad()
        features = encoder.encode_unit(candidates, relaxed=True)
        scores = model.noisy_expected_improvement(
            X=features,
            X_baseline=baseline_X,
            n_samples=mc_samples,
            candidate_base_samples=candidate_base_samples,
            baseline_base_samples=baseline_base_samples,
            with_grad=True,
        )
        (-scores.sum()).backward()
        optimizer.step()
        with torch.no_grad():
            candidates.clamp_(0.0, 1.0)

    return candidates.detach()


def _snap_unit_samples(factors: list[FactorSpec], unit_points: torch.Tensor) -> torch.Tensor:
    snapped = unit_points.clone().clamp(0.0, 1.0)
    for idx, factor in enumerate(factors):
        column = snapped[:, idx]
        if factor.kind == "ordinal":
            levels = _ordinal_level_count(factor)
            if levels > 1:
                snapped[:, idx] = torch.round(column * (levels - 1)) / (levels - 1)
            else:
                snapped[:, idx] = 0.0
        elif factor.kind == "categorical":
            n_categories = max(len(factor.categories), 1)
            indices = torch.clamp((column * n_categories).floor(), max=n_categories - 1)
            snapped[:, idx] = (indices + 0.5) / n_categories
    return snapped


def _draw_local_mixed_samples(
    factors: list[FactorSpec],
    observed_units: torch.Tensor,
    observed_scores: torch.Tensor,
    n_points: int,
    seed: int,
    trust_region: torch.Tensor,
) -> torch.Tensor:
    if observed_units.numel() == 0:
        return torch.empty((0, max(len(factors), 1)), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    weights = observed_scores.clamp_min(1e-6)
    if weights.shape[0] != observed_units.shape[0]:
        weights = torch.ones(observed_units.shape[0], dtype=observed_units.dtype)
    weights = weights / weights.sum()
    indices = torch.multinomial(weights, num_samples=n_points, replacement=True, generator=generator)
    samples = observed_units[indices].clone()
    if observed_units.shape[0] > 1:
        partner_indices = torch.multinomial(weights, num_samples=n_points, replacement=True, generator=generator)
        partners = observed_units[partner_indices]
    else:
        partners = samples
    for idx, factor in enumerate(factors):
        column = samples[:, idx]
        radius = float(trust_region[idx].item()) if trust_region.numel() > idx else 0.18
        if factor.kind == "continuous":
            noise = torch.randn(n_points, generator=generator, dtype=samples.dtype) * max(radius * 0.45, 0.03)
            mix = torch.rand(n_points, generator=generator, dtype=samples.dtype)
            drift = (partners[:, idx] - column) * min(max(radius * 1.2, 0.12), 0.4) * mix
            samples[:, idx] = (column + noise + drift).clamp(0.0, 1.0)
        elif factor.kind == "ordinal":
            levels = _ordinal_level_count(factor)
            if levels > 1:
                max_jump = 1 if radius <= 0.18 else 2
                delta = torch.randint(-max_jump, max_jump + 1, size=(n_points,), generator=generator)
                base = torch.round(column * (levels - 1)).long()
                samples[:, idx] = (base.add(delta).clamp(0, levels - 1).to(samples.dtype)) / (levels - 1)
            else:
                samples[:, idx] = 0.0
        else:
            n_categories = max(len(factor.categories), 1)
            choice = torch.randint(0, n_categories, size=(n_points,), generator=generator, dtype=torch.long)
            partner_choice = torch.clamp((partners[:, idx] * n_categories).floor().long(), max=n_categories - 1)
            keep_probability = 0.75 if radius <= 0.18 else 0.6
            keep_mask = torch.rand(n_points, generator=generator) < keep_probability
            partner_mask = torch.rand(n_points, generator=generator) < 0.5
            current = torch.clamp((column * n_categories).floor().long(), max=n_categories - 1)
            switched = torch.where(partner_mask, partner_choice, choice)
            samples[:, idx] = (torch.where(keep_mask, current, switched).to(samples.dtype) + 0.5) / n_categories
    return samples


def _estimate_trust_region(
    observed_units: torch.Tensor,
    observed_scores: torch.Tensor,
    n_factors: int,
) -> torch.Tensor:
    if observed_units.numel() == 0:
        return torch.full((max(n_factors, 1),), 0.2, dtype=torch.float32)
    candidate_count = observed_units.shape[0]
    top_k = min(max(4, candidate_count // 3), candidate_count)
    top_indices = torch.topk(observed_scores.clamp_min(1e-6), k=top_k, dim=0).indices
    anchors = observed_units[top_indices]
    if anchors.shape[0] <= 1:
        return torch.full((anchors.shape[1],), 0.22, dtype=anchors.dtype, device=anchors.device)
    low = torch.quantile(anchors, 0.1, dim=0)
    high = torch.quantile(anchors, 0.9, dim=0)
    spread = (high - low).clamp_min(0.02)
    sample_scale = 1.15 if candidate_count < 16 else 1.0
    radii = (spread * 0.75 * sample_scale).clamp(0.08, 0.35)
    return radii


def _unit_candidates_to_frame(factors: list[FactorSpec], unit_points: torch.Tensor) -> pd.DataFrame:
    if unit_points.numel() == 0:
        return pd.DataFrame(columns=[factor.name for factor in factors])
    records = [_decode_sample_row(factors, row) for row in unit_points]
    return pd.DataFrame(records).drop_duplicates().reset_index(drop=True)


def _coerce_candidate_value(factor, raw):
    if factor.kind == "categorical":
        categories = [str(value) for value in factor.categories]
        text = str(raw).strip()
        for value in categories:
            if value.lower() == text.lower():
                return value
        return None
    try:
        numeric = float(raw)
    except (TypeError, ValueError):
        return None
    numeric = min(max(numeric, float(factor.min_value)), float(factor.max_value))
    if factor.kind == "ordinal" and factor.step:
        steps = round((numeric - float(factor.min_value)) / factor.step)
        numeric = float(factor.min_value) + steps * factor.step
    return numeric


def _validated_llm_candidate_rows(project, llm_candidates):
    if not llm_candidates:
        return None
    rows: list[dict] = []
    for cand in list(llm_candidates)[:3]:
        if not isinstance(cand, dict):
            continue
        row: dict = {"_llm": 1}
        ok = True
        for factor in project.factors:
            value = _coerce_candidate_value(factor, cand.get(factor.name))
            if value is None:
                ok = False
                break
            row[factor.name] = value
        if ok:
            rows.append(row)
    return pd.DataFrame(rows) if rows else None


def _prior_gate_open(scored) -> bool:
    """Prior hints engage only under floor pathology: best score < 0.35 and
    at least half the scores on the desirability floor (< 0.15), with at
    least 3 scored rows. Mirrors the api optimizer gate."""
    if "FuzzyScore" not in scored.columns or not len(scored):
        return False
    scores = pd.to_numeric(scored["FuzzyScore"], errors="coerce").dropna()
    if len(scores) < 3:
        return False
    floor_mass = float((scores < 0.15).mean())
    return float(scores.max()) < 0.35 and floor_mass >= 0.5


def _validated_pseudo_rows(project, prior_pseudo, target_columns, observed_mean=None):
    """Convert prior hints into damped pseudo-observation rows at the
    surrogate's target-column level (LGBO-style). With a single composite
    target (>= 3 objectives) the value is the weighted geometric mean of
    all claimed desirabilities; each claim is shrunk 50% toward the
    campaign's observed mean target first. Mirrors the api optimizer."""
    if not prior_pseudo or not target_columns:
        return None
    composite = len(target_columns) == 1 and len(project.objectives) > 1
    rows: list[dict] = []
    for hint in list(prior_pseudo)[:2]:
        if not isinstance(hint, dict):
            continue
        factors_hint = hint.get("factors") or {}
        desirability_hint = hint.get("desirability") or {}
        row: dict = {"_llm": 0}
        ok = True
        for factor in project.factors:
            value = _coerce_candidate_value(factor, factors_hint.get(factor.name))
            if value is None:
                ok = False
                break
            row[factor.name] = value
        if not ok:
            continue
        weight_sum = 0.0
        weighted_log = 0.0
        for objective in project.objectives:
            value = _to_valid_desirability(desirability_hint.get(objective.name))
            if value is None:
                ok = False
                break
            shrink_column = target_columns[0] if composite else f"_bo_target_{objective.name}"
            if shrink_column not in target_columns:
                shrink_column = target_columns[0]
            shrink_target = (observed_mean or {}).get(shrink_column)
            if shrink_target is not None:
                value = 0.5 * value + 0.5 * float(shrink_target)
            weight = max(float(objective.weight or 1.0), 1e-6)
            weight_sum += weight
            weighted_log += weight * float(np.log(max(value, 1e-6)))
            if not composite and f"_bo_target_{objective.name}" in target_columns:
                row[f"_bo_target_{objective.name}"] = value
        if not ok or weight_sum <= 0:
            continue
        if composite:
            row[target_columns[0]] = float(np.exp(weighted_log / weight_sum))
        row["FuzzyScore"] = float(np.exp(weighted_log / weight_sum))
        rows.append(row)
    return pd.DataFrame(rows) if rows else None


def _to_valid_desirability(raw):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value:
        return None
    return min(max(value, 0.02), 1.0)


def _soft_constraint_penalty(candidate_unit, soft_constraints, factors):
    """Bounded acquisition penalty for LLM-proposed soft region constraints
    (mirrors app/optimizer.py in chatexp-api). Unit-space: weight x violated
    fraction, clipped to [0, 0.5]."""
    if not soft_constraints:
        return torch.zeros(candidate_unit.shape[0])
    index_of = {factor.name: i for i, factor in enumerate(factors)}
    specs = []
    for constraint in (soft_constraints or [])[:2]:
        if not isinstance(constraint, dict):
            continue
        boxes = constraint.get("factors") or constraint.get("boxes") or {}
        try:
            weight = min(max(float(constraint.get("weight", 0.5)), 0.0), 1.0)
            if weight != weight or weight in (float("inf"), float("-inf")):
                # NaN survives min/max; it would turn the whole selection
                # score NaN (argmax degenerates to first index).
                weight = 0.5
        except (TypeError, ValueError):
            continue
        for name, box in (boxes or {}).items():
            index = index_of.get(str(name))
            if index is None or index >= candidate_unit.shape[1]:
                continue
            if not (isinstance(box, (list, tuple)) and len(box) == 2):
                continue
            try:
                lo, hi = float(box[0]), float(box[1])
            except (TypeError, ValueError):
                continue
            lo, hi = min(lo, hi), max(lo, hi)
            specs.append((index, lo, hi, weight))
    if not specs:
        return torch.zeros(candidate_unit.shape[0])
    penalty = torch.zeros(candidate_unit.shape[0])
    for index, lo, hi, weight in specs:
        factor = factors[index]
        if factor.kind == "categorical":
            continue
        lo_f = float(factor.min_value if factor.min_value is not None else 0.0)
        hi_f = float(factor.max_value if factor.max_value is not None else 1.0)
        span = max(hi_f - lo_f, 1e-9)
        lo_u = (lo - lo_f) / span
        hi_u = (hi - lo_f) / span
        values = candidate_unit[:, index]
        # The box is the region to AVOID (operator-declared infeasible zone):
        # penalty grows with depth INTO the box (API-engine parity).
        inside = ((values >= lo_u) & (values <= hi_u)).float()
        span_u = max(hi_u - lo_u, 1e-9)
        depth_frac = inside * torch.clamp(
            torch.minimum(values - lo_u, hi_u - values) / (0.5 * span_u), 0.0, 1.0
        )
        violation = torch.maximum(inside * 0.5 + 0.5 * depth_frac, torch.zeros_like(values))
        penalty = penalty + weight * violation
    # nan_to_num as defense in depth: upstream validators already reject
    # non-finite weights, but a NaN penalty must never reach the selection
    # score (NaN argmax picks the first candidate, silently arbitrary).
    return torch.nan_to_num(penalty / max(len(specs), 1)).clamp(0.0, 0.5)


def _select_batch_by_noisy_ei(
    factors: list[FactorSpec],
    model: TorchObjectiveEnsemble,
    X_candidate: torch.Tensor,
    X_baseline: torch.Tensor,
    X_candidate_unit: torch.Tensor,
    X_baseline_unit: torch.Tensor,
    observed_scores: torch.Tensor,
    n_points: int,
    strategy_phase: str | None = None,
    soft_constraints: list[dict] | None = None,
    generator: torch.Generator | None = None,
    recent_gain: "float | None" = None,
    floor_pathology: bool = False,
) -> tuple[list[int], list[float]]:
    candidate_count = int(X_candidate.shape[0])
    if candidate_count == 0:
        return [], []
    if candidate_count <= n_points:
        n_models_s = len(model.models)
        sc = torch.randn((192, candidate_count, n_models_s), generator=generator, dtype=X_candidate.dtype) if generator is not None else None
        sb = torch.randn((192, X_baseline.shape[0], n_models_s), generator=generator, dtype=X_baseline.dtype) if generator is not None else None
        acquisition = model.noisy_expected_improvement(
            X=X_candidate,
            X_baseline=X_baseline,
            n_samples=192,
            **({"candidate_base_samples": sc, "baseline_base_samples": sb} if generator is not None else {}),
        )
        return list(range(candidate_count)), acquisition.detach().cpu().tolist()

    remaining = list(range(candidate_count))
    selected: list[int] = []
    selected_scores: list[float] = []
    current_baseline = X_baseline
    current_baseline_unit = X_baseline_unit
    sample_count = int(X_baseline.shape[0])
    incumbent = float(observed_scores.max().item()) if observed_scores.numel() else 0.0
    while remaining and len(selected) < n_points:
        remaining_X = X_candidate[remaining]
        remaining_unit = X_candidate_unit[remaining]
        n_models_g = len(model.models)
        cand_base = torch.randn((192, remaining_X.shape[0], n_models_g), generator=generator, dtype=remaining_X.dtype) if generator is not None else None
        base_base = torch.randn((192, current_baseline.shape[0], n_models_g), generator=generator, dtype=current_baseline.dtype) if generator is not None else None
        acquisition = model.noisy_expected_improvement(
            X=remaining_X,
            X_baseline=current_baseline,
            n_samples=192,
            **({"candidate_base_samples": cand_base, "baseline_base_samples": base_base} if generator is not None else {}),
        )
        min_dist = torch.cdist(remaining_unit, current_baseline_unit).min(dim=1).values
        comfort = 1.0 - (remaining_unit - 0.5).abs().mul(2.0).mean(dim=1)
        acq_norm = _normalize_selection_signal(acquisition)
        dist_norm = _normalize_selection_signal(min_dist)
        comfort_norm = comfort.clamp(0.0, 1.0)
        acquisition_span = float((acquisition.max() - acquisition.min()).item())
        trust_weight = _surrogate_trust_weight(
            sample_count=sample_count,
            incumbent=incumbent,
            acquisition_span=acquisition_span,
            recent_gain=recent_gain,
        )
        exploration_weight = _exploration_guard_weight(
            sample_count=sample_count,
            incumbent=incumbent,
            batch_position=len(selected),
            acquisition_span=acquisition_span,
        )
        comfort_weight = _comfort_guard_weight(sample_count=sample_count, incumbent=incumbent)
        if _TRUST_REKEY_ENABLED and floor_pathology:
            # Interior-first exploration under diagnosed floor pathology
            # (API-engine parity): the comfort term takes the exploration
            # weight; distance demotes to a batch-spread tie-breaker.
            comfort_weight = max(comfort_weight, exploration_weight)
            exploration_weight = 0.30
        # Campaign-strategy phase bias (bounded), mirroring the API engine:
        # the LLM strategist steers the SELECTION regime only.
        if strategy_phase == "explore":
            exploration_weight = min(exploration_weight + 0.25, 1.5)
            trust_weight *= 0.5
        # "exploit" adds no extra bias: the trust-gated blend
        # already concentrates selection on the learned best region when
        # trust is high, and an additional exploit bias double-counts and
        # over-converges small batches (mirrors the API engine).
        selection_score = trust_weight * acq_norm + exploration_weight * dist_norm + comfort_weight * comfort_norm
        if soft_constraints:
            selection_score = selection_score - _soft_constraint_penalty(remaining_unit, soft_constraints, factors)
        selected_unit = X_candidate_unit[selected] if selected else torch.empty((0, X_candidate_unit.shape[1]), dtype=X_candidate_unit.dtype, device=X_candidate_unit.device)
        batch_gain = _batch_marginal_utility_signal(factors, remaining_unit, X_baseline_unit, selected_unit)
        selection_score = selection_score + _batch_utility_weight(sample_count, incumbent, acquisition_span) * batch_gain
        best_position = int(torch.argmax(selection_score).item())
        best_index = remaining.pop(best_position)
        selected.append(best_index)
        selected_scores.append(float(acquisition[best_position].item()))
        current_baseline = torch.cat([current_baseline, X_candidate[best_index : best_index + 1]], dim=0)
        current_baseline_unit = torch.cat(
            [current_baseline_unit, X_candidate_unit[best_index : best_index + 1]], dim=0
        )
    return selected, selected_scores


def _normalize_selection_signal(values: torch.Tensor) -> torch.Tensor:
    span = values.max() - values.min()
    if float(span.item()) < 1e-12:
        return torch.zeros_like(values)
    return (values - values.min()) / span.clamp_min(1e-12)


def _batch_marginal_utility_signal(
    factors: list[FactorSpec],
    candidate_units: torch.Tensor,
    baseline_units: torch.Tensor,
    selected_units: torch.Tensor,
) -> torch.Tensor:
    if candidate_units.numel() == 0:
        return torch.empty((0,), dtype=candidate_units.dtype, device=candidate_units.device)
    reference = torch.cat([baseline_units, selected_units], dim=0) if selected_units.numel() else baseline_units
    if reference.numel():
        distance = torch.cdist(candidate_units, reference).min(dim=1).values
        continuous_gain = _normalize_selection_signal(distance)
    else:
        center = torch.full((1, candidate_units.shape[1]), 0.5, dtype=candidate_units.dtype, device=candidate_units.device)
        continuous_gain = 1.0 - torch.cdist(candidate_units, center).squeeze(-1).clamp(0.0, 1.0)
    level_gain = _discrete_level_entropy_gain(factors, candidate_units, baseline_units, selected_units)
    return (0.55 * continuous_gain + 0.45 * level_gain).clamp(0.0, 1.0)


def _discrete_level_entropy_gain(
    factors: list[FactorSpec],
    candidate_units: torch.Tensor,
    baseline_units: torch.Tensor,
    selected_units: torch.Tensor,
) -> torch.Tensor:
    gains = []
    existing_units = torch.cat([baseline_units, selected_units], dim=0) if selected_units.numel() else baseline_units
    for idx, factor in enumerate(factors):
        if factor.kind == "continuous" or idx >= candidate_units.shape[1]:
            continue
        n_levels = _discrete_level_count(factor)
        if n_levels <= 1:
            continue
        candidate_codes = _discrete_level_codes(factor, candidate_units[:, idx])
        existing_codes = _discrete_level_codes(factor, existing_units[:, idx]) if existing_units.numel() else torch.empty((0,), dtype=torch.long, device=candidate_units.device)
        counts = torch.bincount(existing_codes, minlength=n_levels).to(dtype=candidate_units.dtype, device=candidate_units.device)
        total = counts.sum()
        current_entropy = _level_entropy(counts)
        level_gains = []
        for code in candidate_codes:
            updated = counts.clone()
            updated[int(code.item())] += 1.0
            entropy_gain = (_level_entropy(updated) - current_entropy).clamp_min(0.0)
            rarity_gain = 1.0 / (1.0 + counts[int(code.item())])
            level_gains.append((0.65 * entropy_gain / math.log(max(n_levels, 2)) + 0.35 * rarity_gain).clamp(0.0, 1.0))
        gain = torch.stack(level_gains) if level_gains else torch.zeros((candidate_units.shape[0],), dtype=candidate_units.dtype, device=candidate_units.device)
        if total < n_levels:
            gain = gain.clamp_min((counts[candidate_codes] == 0).to(dtype=candidate_units.dtype))
        gains.append(gain)
    if not gains:
        return torch.zeros((candidate_units.shape[0],), dtype=candidate_units.dtype, device=candidate_units.device)
    return torch.stack(gains, dim=0).mean(dim=0).clamp(0.0, 1.0)


def _level_entropy(counts: torch.Tensor) -> torch.Tensor:
    total = counts.sum().clamp_min(1e-9)
    probs = counts / total
    return -(probs * probs.clamp_min(1e-9).log()).sum()


def _discrete_level_count(factor: FactorSpec) -> int:
    if factor.kind == "categorical":
        return max(len(factor.categories), 1)
    return _ordinal_level_count(factor)


def _discrete_level_codes(factor: FactorSpec, values: torch.Tensor) -> torch.Tensor:
    if factor.kind == "categorical":
        n_categories = max(len(factor.categories), 1)
        return torch.clamp((values.clamp(0.0, 1.0) * n_categories).floor().long(), max=n_categories - 1)
    levels = _discrete_level_count(factor)
    return torch.round(values.clamp(0.0, 1.0) * (levels - 1)).long()


def _batch_utility_weight(sample_count: int, incumbent: float, acquisition_span: float) -> float:
    if acquisition_span > 0.12 and incumbent >= 0.9:
        return 0.18
    if sample_count < 12:
        return 0.42
    if sample_count < 24:
        return 0.32
    return 0.24


def _exploration_guard_weight(
    sample_count: int,
    incumbent: float,
    batch_position: int,
    acquisition_span: float,
) -> float:
    """Balance exploitation with coverage while the surrogate is weakly calibrated."""

    sample_term = max(0.0, (18.0 - float(sample_count)) / 18.0)
    quality_term = max(0.0, (0.72 - float(incumbent)) / 0.72)
    tie_term = 1.0 if acquisition_span < 1e-5 else 0.0
    batch_term = min(max(float(batch_position), 0.0), 3.0) * 0.12
    guarded = 0.18 + 0.38 * sample_term + 0.32 * quality_term + 0.35 * tie_term + batch_term
    return float(min(max(guarded, 0.18), 1.25))


# Realized-informativeness trust gating, mirrored from the chatexp-api engine
# (TuRBO/FuRBO lineage: exploitation is keyed to realized incumbent
# improvement, not sample count). Kill switch: CHATEXP_TRUST_REKEY=0.
_TRUST_INFO_GAIN_SCALE = 0.15
_TRUST_REKEY_ENABLED = os.environ.get("CHATEXP_TRUST_REKEY", "1") != "0"


def _recent_incumbent_gain(clean: "pd.DataFrame", n_points: int) -> "float | None":
    """Realized improvement of the incumbent over the pre-batch best
    (batch/Trial column when present, otherwise the last n_points rows)."""
    if clean is None or "FuzzyScore" not in getattr(clean, "columns", []) or not len(clean):
        return None
    scores = pd.to_numeric(clean["FuzzyScore"], errors="coerce").dropna()
    if len(scores) < 2:
        return None
    # The floor-pathology response engages only when the observations are
    # floor-dominated (same regime as the prior gate), so a low-ceiling but
    # genuinely informative campaign keeps its learned gradient driving
    # selection — on every path, with or without a batch column.
    if float((scores < 0.15).mean()) < 0.5:
        return None
    if "Trial" in clean.columns:
        try:
            trials = pd.to_numeric(clean["Trial"], errors="coerce")
            valid = clean.assign(_score=scores, _trial=trials).dropna(subset=["_score", "_trial"])
            if len(valid) >= 2:
                last_trial = float(valid["_trial"].max())
                prior = valid[valid["_trial"] < last_trial]["_score"]
                if len(prior) == 0:
                    return None
                return float(scores.max()) - float(prior.max())
        except Exception:
            pass
    prior = scores.iloc[:-max(int(n_points), 1)]
    if len(prior) == 0:
        return None
    return float(scores.max()) - float(prior.max())


def _surrogate_trust_weight(
    sample_count: int,
    incumbent: float,
    acquisition_span: float,
    recent_gain: "float | None" = None,
) -> float:
    # NOTE: the chatexp-api engine (which ran all reported benchmarks) uses a
    # wider trust schedule, 0.10 + 0.45*sample + 0.85*quality clamped to
    # [0.10, 1.60]; this offline mirror keeps the tighter [0.12, 1.0] schedule
    # it shipped with. The two implementations are documented as distinct.

    if acquisition_span < 1e-5:
        return 0.0
    sample_term = min(max((float(sample_count) - 4.0) / 24.0, 0.0), 1.0)
    quality_term = min(max(float(incumbent) / 0.65, 0.0), 1.0)
    if (
        _TRUST_REKEY_ENABLED
        and recent_gain is not None
        and incumbent < 0.35
    ):
        info_term = min(max(float(recent_gain) / _TRUST_INFO_GAIN_SCALE, 0.0), 1.0)
        quality_term = min(quality_term, info_term)
    return float(min(max(0.12 + 0.55 * sample_term + 0.33 * quality_term, 0.12), 1.0))


def _comfort_guard_weight(sample_count: int, incumbent: float) -> float:
    sample_term = max(0.0, (20.0 - float(sample_count)) / 20.0)
    low_score_term = max(0.0, (0.45 - float(incumbent)) / 0.45)
    return float(min(0.10 + 0.45 * sample_term + 0.55 * low_score_term, 0.95))


def _decode_sample_row(factors: list[FactorSpec], row: torch.Tensor) -> dict[str, object]:
    values: dict[str, object] = {}
    for idx, factor in enumerate(factors):
        x = float(row[idx].item()) if idx < len(row) else 0.5
        if factor.kind == "continuous":
            min_value = _factor_min(factor)
            max_value = _factor_max(factor)
            values[factor.name] = round(min_value + x * (max_value - min_value), 6)
        elif factor.kind == "ordinal":
            min_value = _factor_min(factor)
            step = _factor_step(factor)
            levels = _ordinal_level_count(factor)
            index = min(int(math.floor(x * levels)), max(levels - 1, 0))
            values[factor.name] = round(min_value + index * step, 6)
        else:
            categories = factor.categories or ["A", "B"]
            index = min(int(math.floor(x * len(categories))), len(categories) - 1)
            values[factor.name] = categories[index]
    return values


def _drop_incomplete_rows(project: ProjectState, df: pd.DataFrame) -> pd.DataFrame:
    required = [factor.name for factor in project.factors] + [objective.name for objective in project.objectives]
    if df.empty:
        return df
    missing = [column for column in required if column not in df.columns]
    if missing:
        return pd.DataFrame(columns=df.columns)
    mask = df[required].map(lambda value: not is_missing_value(value)).all(axis=1)
    return df.loc[mask].copy()


def _remove_existing_points(
    candidates_df: pd.DataFrame,
    existing_df: pd.DataFrame,
    factors: list[FactorSpec],
) -> pd.DataFrame:
    if existing_df.empty:
        return candidates_df
    normalized_existing = existing_df.copy()
    normalized_candidates = candidates_df.copy()
    for factor in factors:
        if factor.kind != "categorical":
            normalized_existing[factor.name] = coerce_numeric_series(normalized_existing[factor.name]).round(6)
            normalized_candidates[factor.name] = coerce_numeric_series(normalized_candidates[factor.name]).round(6)
        else:
            normalized_existing[factor.name] = normalized_existing[factor.name].astype(str)
            normalized_candidates[factor.name] = normalized_candidates[factor.name].astype(str)
    existing_keys = {
        tuple(record[factor.name] for factor in factors)
        for record in normalized_existing.to_dict(orient="records")
    }
    keep_mask = []
    for record in normalized_candidates.to_dict(orient="records"):
        key = tuple(record[factor.name] for factor in factors)
        keep_mask.append(key not in existing_keys)
    return candidates_df.loc[keep_mask].reset_index(drop=True)


def _observed_factor_frame(rows: pd.DataFrame, factors: list[FactorSpec]) -> pd.DataFrame:
    columns = [factor.name for factor in factors]
    if rows.empty or any(column not in rows.columns for column in columns):
        return pd.DataFrame(columns=columns)
    frame = rows[columns].copy()
    mask = frame.map(lambda value: not is_missing_value(value)).all(axis=1)
    return frame.loc[mask].drop_duplicates().reset_index(drop=True)


def _validate_factor_domains(factors: list[FactorSpec]) -> None:
    if not factors:
        raise TorchBOError("No input factors are configured.")
    problems = []
    for factor in factors:
        if factor.kind == "categorical":
            categories = [str(value).strip() for value in factor.categories if str(value).strip()]
            if not categories:
                problems.append(f"{factor.name}: categorical factor has no levels")
            continue
        min_value = parse_numeric_value(factor.min_value)
        max_value = parse_numeric_value(factor.max_value)
        if min_value is None or max_value is None or max_value <= min_value:
            problems.append(f"{factor.name}: numeric factor needs a valid min/max range")
            continue
        if factor.kind == "ordinal":
            step = parse_numeric_value(factor.step)
            if step is None or step <= 0:
                problems.append(f"{factor.name}: ordinal factor needs a positive step")
            elif _ordinal_level_count(factor) < 2:
                problems.append(f"{factor.name}: ordinal factor range has fewer than two levels")
    if problems:
        raise TorchBOError("Invalid design space: " + "; ".join(problems))


def _factor_min(factor: FactorSpec) -> float:
    value = parse_numeric_value(factor.min_value)
    if value is None:
        raise TorchBOError(f"Factor {factor.name} is missing min_value.")
    return float(value)


def _factor_max(factor: FactorSpec) -> float:
    value = parse_numeric_value(factor.max_value)
    if value is None:
        raise TorchBOError(f"Factor {factor.name} is missing max_value.")
    return float(value)


def _factor_step(factor: FactorSpec) -> float:
    value = parse_numeric_value(factor.step)
    if value is None or value <= 0:
        raise TorchBOError(f"Factor {factor.name} is missing a positive step.")
    return float(value)


def _ordinal_level_count(factor: FactorSpec) -> int:
    min_value = _factor_min(factor)
    max_value = _factor_max(factor)
    step = _factor_step(factor)
    return max(1, int(math.floor((max_value - min_value) / step + 1e-9)) + 1)


def _require_torch() -> None:
    if torch is None or nn is None:
        raise TorchBOError(
            "ChatExp 的 BO 已切换到 PyTorch 实现，但当前环境没有安装 torch。"
            "请先更新 ChatExp 环境后再运行。"
        )
