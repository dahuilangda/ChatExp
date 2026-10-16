from __future__ import annotations

import copy
import itertools
import logging
import math
import os
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_INTEROP_THREADS", "1")

import numpy as np
import pandas as pd

from .models import FactorSpec, ObjectiveSpec, ProjectSpec
from .scoring import coerce_numeric_series, is_missing_value, parse_numeric_value, score_components
from .scoring import json_ready, json_value

logger = logging.getLogger(__name__)

MAX_TRAINING_ROWS = 1200
MAX_CANDIDATE_POOL_SIZE = 12000
MAX_DIAGNOSTIC_CANDIDATE_POOL_SIZE = 16000

try:
    import torch
    from torch import nn
    torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))
    torch.set_num_interop_threads(max(1, int(os.getenv("TORCH_NUM_INTEROP_THREADS", "1"))))
except ImportError:  # pragma: no cover
    torch = None
    nn = None


class BOError(RuntimeError):
    pass


@dataclass
class DesignSuggestion:
    candidate_id: int
    score: float
    values: dict[str, Any]
    strategy: str = "greedy_noisy_ei"


@dataclass
class BoundedTargetTransform:
    lower: float = 1e-4
    upper: float = 1.0 - 1e-4
    mean: "torch.Tensor | None" = None
    std: "torch.Tensor | None" = None

    def fit(self, y: "torch.Tensor") -> "torch.Tensor":
        latent = self._logit(y)
        self.mean = latent.mean()
        self.std = latent.std(unbiased=False).clamp_min(1e-6)
        return (latent - self.mean) / self.std

    def transform(self, y: "torch.Tensor") -> "torch.Tensor":
        if self.mean is None or self.std is None:
            raise BOError("The target transform has not been fitted.")
        latent = self._logit(y)
        return (latent - self.mean) / self.std

    def inverse(self, z: "torch.Tensor") -> "torch.Tensor":
        if self.mean is None or self.std is None:
            raise BOError("The target transform has not been fitted.")
        return torch.sigmoid(z * self.std + self.mean)

    def _logit(self, y: "torch.Tensor") -> "torch.Tensor":
        return torch.logit(y.clamp(self.lower, self.upper))


def _coerce_candidate_value(factor, raw):
    """Coerce an LLM-proposed factor value into the factor's legal domain."""
    if factor.kind == "categorical":
        categories = [str(value) for value in factor.categories]
        text = str(raw).strip()
        for value in categories:
            if value.lower() == text.lower():
                return value
        return None
    numeric = parse_numeric_value(raw)
    if numeric is None:
        return None
    min_value = _factor_min(factor)
    max_value = _factor_max(factor)
    numeric = min(max(float(numeric), min_value), max_value)
    if factor.kind == "ordinal":
        step = _factor_step(factor)
        index = round((numeric - min_value) / step)
        levels = _ordinal_level_count(factor)
        numeric = min_value + min(max(index, 0), levels - 1) * step
    return numeric


def _validated_llm_candidate_rows(project: ProjectSpec, llm_candidates: list[dict] | None):
    """Feasibility-clamp LLM-proposed candidate points into the factor domains.

    Values outside a factor domain are clamped (continuous/ordinal) or matched
    case-insensitively (categorical); candidates missing any factor are dropped.
    A maximum of three proposals is accepted per batch."""
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
    if not rows:
        return None
    return pd.DataFrame(rows)


def _prior_gate_open(clean: pd.DataFrame) -> bool:
    """Prior hints engage only under floor pathology.

    True when at least 3 scored rows exist, the best score is below 0.35 (no
    incumbent worth exploiting), and at least half the scores sit on the
    desirability floor (< 0.15) -- the regime where the literature (LGBO,
    ChemBOMAS) shows injected priors help. Healthy campaigns keep the
    surrogate untouched."""
    if "FuzzyScore" not in clean.columns or not len(clean):
        return False
    scores = pd.to_numeric(clean["FuzzyScore"], errors="coerce").dropna()
    if len(scores) < 3:
        return False
    floor_mass = float((scores < 0.15).mean())
    return float(scores.max()) < 0.35 and floor_mass >= 0.5


def _validated_pseudo_rows(project: ProjectSpec, prior_pseudo: list[dict] | None,
                           target_columns: list[str] | None = None,
                           observed_mean: dict[str, float] | None = None):
    """Convert LLM prior hints into damped pseudo-observation rows.

    Rows are built at the surrogate's target-column level: one column per
    objective, or -- when the objectives collapse to a single composite
    target (>= 3 objectives) -- the weighted geometric mean of ALL claimed
    desirabilities, so every objective's claim contributes. Each claim is
    first shrunk 50% toward the campaign's observed mean target (LGBO's
    calibrated lift). Returns None when no hint yields a complete row."""
    if not prior_pseudo or not target_columns:
        return None
    objectives = project.objectives
    if not objectives:
        return None
    composite = len(target_columns) == 1 and len(objectives) > 1

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
        per_objective: dict[str, float] = {}
        for objective in objectives:
            value = _to_valid_desirability(desirability_hint.get(objective.name))
            if value is None:
                ok = False
                break
            shrink_column = target_columns[0] if composite else f"_bo_target_{objective.name}"
            shrink_target = (observed_mean or {}).get(shrink_column)
            if shrink_target is not None:
                value = 0.5 * value + 0.5 * float(shrink_target)
            per_objective[objective.name] = value
            weight = max(float(objective.weight or 1.0), 1e-6)
            weight_sum += weight
            weighted_log += weight * float(np.log(max(value, 1e-6)))
        if not ok or weight_sum <= 0:
            continue

        if composite:
            row[target_columns[0]] = float(np.exp(weighted_log / weight_sum))
        else:
            for objective in objectives:
                column = f"_bo_target_{objective.name}"
                if column in target_columns:
                    row[column] = per_objective[objective.name]
        row["FuzzyScore"] = float(np.exp(weighted_log / weight_sum))
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows)


def _to_valid_desirability(raw):
    numeric = parse_numeric_value(raw)
    if numeric is None:
        return None
    return min(max(float(numeric), 0.02), 1.0)


def suggest_next_design(
    project: ProjectSpec,
    rows: pd.DataFrame,
    n_points: int,
    *,
    candidate_pool_size: int = 2048,
    ensemble_size: int = 5,
    training_steps: int = 300,
    initial_strategy: str = "balanced",
    strategy_phase: str | None = None,
    soft_constraints: list[dict] | None = None,
    llm_candidates: list[dict] | None = None,
    prior_pseudo: list[dict] | None = None,
    diagnostic_point: dict | None = None,
    floor_mode: bool = False,
) -> list[DesignSuggestion]:
    """Suggest the next batch of experiments for a project.

    Fits the surrogate ensemble on completed rows and selects the batch by
    greedy noisy-EI; returns a space-filling initial design when there are
    too few completed rows to train. Directive inputs (strategy phase, prior
    hints, LLM candidates, soft constraints) are ignored whenever the
    observed incumbent is healthy (best composite >= 0.35). Batches are
    deterministic in the observation history."""
    _require_torch()
    _validate_factor_domains(project.factors)
    n_points = max(int(n_points), 1)

    scored, _ = score_components(rows.copy(), project.objectives)
    target_columns, objective_weights = _training_targets(scored, project.objectives)
    clean = _scored_complete_rows(scored, project)
    # Deterministic health clamp: once the incumbent reaches a decent region
    # (best observed composite >= 0.35), every directive channel -- phase
    # steering, prior hints, LLM candidates -- is ignored regardless of
    # source. Directive-driven meta-control engages only under diagnosed
    # pathology (low incumbent), mirroring the strategist's campaign_health.
    recent_gain = _recent_incumbent_gain(clean, n_points)
    floor_pathology = False
    if recent_gain is not None and clean.shape[0] and "FuzzyScore" in clean.columns:
        try:
            floor_pathology = float(pd.to_numeric(clean["FuzzyScore"], errors="coerce").max()) < 0.35
        except Exception:
            floor_pathology = False
    if clean.shape[0] and "FuzzyScore" in clean.columns:
        incumbent = float(pd.to_numeric(clean["FuzzyScore"], errors="coerce").max())
        if incumbent >= 0.35:
            # Healthy campaign: strategist META-CONTROL is inert (phase
            # steering, prior hints, LLM candidates). Soft constraints are
            # exempt: they encode operator-stated infeasibility, which the
            # operator approves in the design-confirmation summary -- the
            # operator's contract applies regardless of campaign health.
            strategy_phase = None
            prior_pseudo = None
            llm_candidates = None
    needed = _minimum_training_rows(project)
    if clean.shape[0] < needed or not target_columns:
        candidates = _initial_design_frame(project.factors, rows, n_points, seed=41, scored_rows=scored, strategy=initial_strategy)
        return [
            DesignSuggestion(index + 1, 0.0, _json_ready(record), "space_filling_initial_design")
            for index, record in enumerate(candidates.to_dict(orient="records"))
        ]

    clean = clean.dropna(subset=target_columns + ["FuzzyScore"])
    if clean.shape[0] < needed:
        candidates = _initial_design_frame(project.factors, rows, n_points, seed=43, scored_rows=scored, strategy=initial_strategy)
        return [
            DesignSuggestion(index + 1, 0.0, _json_ready(record), "space_filling_initial_design")
            for index, record in enumerate(candidates.to_dict(orient="records"))
        ]
    clean = _active_training_set(clean, project.factors, target_columns, max_rows=MAX_TRAINING_ROWS, seed=79)

    runtime = _bo_runtime_config(
        n_factors=len(project.factors),
        n_rows=int(clean.shape[0]),
        candidate_pool_size=candidate_pool_size,
        ensemble_size=ensemble_size,
        training_steps=training_steps,
    )
    feature_columns = [factor.name for factor in project.factors]
    if prior_pseudo and _prior_gate_open(clean):
        # LGBO-style confidence gating: prior hints are applied only when the
        # campaign shows floor pathology (mass of observations on the
        # desirability floor with a low incumbent). On healthy campaigns any
        # hint -- oracle or LLM -- is pure bias, so the gate stays shut for
        # ALL sources: no hint, real or sham, may help a healthy campaign.
        observed_mean = {
            column: float(clean[column].mean())
            for column in target_columns
            if column in clean.columns and len(clean)
        }
        pseudo = _validated_pseudo_rows(project, prior_pseudo, target_columns, observed_mean)
        if pseudo is not None and len(pseudo):
            clean = pd.concat([clean, pseudo], ignore_index=True)
    encoder = FactorEncoder(project.factors)
    X = encoder.encode_frame(clean[feature_columns])
    Y = torch.tensor(clean[target_columns].to_numpy(dtype=np.float32), dtype=torch.float32)
    model = TorchObjectiveEnsemble(
        input_dim=encoder.input_dim,
        n_objectives=Y.shape[1],
        objective_weights=objective_weights,
        ensemble_size=int(runtime["ensemble_size"]),
        training_steps=int(runtime["training_steps"]),
    )
    model.fit(X, Y)

    observed = _observed_factor_frame(rows, project.factors)
    candidates = _propose_candidate_frame(
        factors=project.factors,
        encoder=encoder,
        model=model,
        observed_df=clean[feature_columns],
        observed_scores=torch.tensor(clean["FuzzyScore"].to_numpy(dtype=np.float32), dtype=torch.float32),
        n_points=n_points,
        raw_samples=int(runtime["candidate_pool_size"]),
        seed=101,
    )
    candidates = _remove_existing(candidates, observed, project.factors)
    llm_rows = _validated_llm_candidate_rows(project, llm_candidates)
    if llm_rows is not None and len(llm_rows):
        candidates = pd.concat([candidates, llm_rows], ignore_index=True)
    n_llm = int(len(llm_rows)) if llm_rows is not None else 0
    if len(candidates) < n_points:
        fill = _initial_design_frame(project.factors, rows, n_points - len(candidates), seed=157, scored_rows=scored, strategy="coverage")
        candidates = pd.concat([candidates, fill], ignore_index=True).drop_duplicates().reset_index(drop=True)
        candidates = _remove_existing(candidates, observed, project.factors)
    if len(candidates) < n_points:
        raise BOError(f"Only {len(candidates)} untested candidates are available in the configured design space.")

    X_candidate = encoder.encode_frame(candidates[feature_columns])
    X_candidate_unit = encoder.to_unit_frame(candidates[feature_columns])
    X_baseline_unit = encoder.to_unit_frame(clean[feature_columns])
    # Floor mode (Thompson-style posterior draws replacing NEI) is opt-in via
    # the floor_mode argument for explicit experiments: with a floor-only
    # history the post-init posterior is too weak for the draws to beat
    # coverage selection, so it is never triggered automatically.
    observed_scores_tensor = torch.tensor(clean["FuzzyScore"].to_numpy(dtype=np.float32), dtype=torch.float32)
    # Deterministic acquisition sampling: derive the MC seed from the
    # observed data fingerprint (not the global RNG), so the same observation
    # history always yields the same batch no matter how many auxiliary model
    # fits (diagnostics, strategist refits) ran in between.
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
        floor_mode=floor_mode,
        soft_constraints=soft_constraints,
        factor_encoder=encoder,
        generator=neis,
        recent_gain=recent_gain,
        floor_pathology=floor_pathology,
    )
    ranked = candidates.iloc[selected_indices].reset_index(drop=True)
    records = ranked.to_dict(orient="records")
    is_llm = [float(record.get("_llm") or 0) == 1.0 for record in records]
    engine_order = [k for k in range(len(selected_scores)) if not is_llm[k]]
    llm_order = [k for k in range(len(selected_scores)) if is_llm[k]]
    if not llm_order:
        # No LLM row reached the selected set: return exactly n_points engine
        # suggestions (the selection over-sampled by n_llm to give proposals a
        # chance; the surplus must NOT become extra evaluations).
        top = sorted(zip(selected_scores, records), key=lambda pair: -float(pair[0]))[:n_points]
        engine_only = []
        for index, (score, record) in enumerate(top):
            record = dict(record)
            record.pop("_llm", None)  # internal marker must not leak into values
            engine_only.append(DesignSuggestion(index + 1, float(score), _json_ready(record), "greedy_noisy_ei"))
        return _apply_diagnostic_point(engine_only, project, diagnostic_point, observed)
    # LLINBO-style "justify" gate: an LLM-proposed point may displace an engine
    # point only when the surrogate scores it within a relative tolerance of
    # the best engine-ranked candidate; otherwise the engine ranking stands.
    best_engine = max((selected_scores[k] for k in engine_order), default=0.0)
    # LLINBO "Justify" with a decaying tolerance (sigma(x_LLM)/t in the paper,
    # approximated relative to the pool-best score): an LLM proposal may
    # displace an engine point when within ~35% of the best early, tightening
    # to ~7% by round 5, so engagement front-loads and the engine dominates
    # as data accumulates.
    bo_rounds = max(1, (len(rows) - int(n_points)) // max(1, int(n_points)) + 1)
    tolerance = max(0.02, (0.35 / bo_rounds) * abs(best_engine))
    for k in sorted(llm_order, key=lambda k: -selected_scores[k]):
        weakest_engine = min(engine_order, key=lambda k: selected_scores[k], default=None)
        if weakest_engine is None:
            break
        if selected_scores[k] >= best_engine - tolerance:
            engine_order.remove(weakest_engine)
            engine_order.append(k)
        else:
            break
    final_order = sorted(engine_order, key=lambda k: -selected_scores[k])[:n_points]
    suggestions = []
    for index, k in enumerate(final_order):
        strategy = "llm_candidate" if is_llm[k] else "greedy_noisy_ei"
        record = dict(records[k])
        record.pop("_llm", None)
        suggestions.append(DesignSuggestion(index + 1, float(selected_scores[k]), _json_ready(record), strategy))
    return _apply_diagnostic_point(suggestions, project, diagnostic_point, observed)


def _apply_diagnostic_point(suggestions, project: ProjectSpec, diagnostic_point: dict | None, observed):
    """Replace the weakest engine suggestion with an operator-approved
    diagnostic point: a real evaluation of the agent's single best hypothesis.

    A real evaluation is used rather than a pseudo-observation because pseudo
    rows enter the NEI baseline and suppress exploration exactly where the
    hypothesis points. Insertion happens after selection, so every
    engine-chosen row is unchanged; the point is domain-clamped and skipped
    when it duplicates an observed row. Exempt from the health
    clamp so a locked-in mediocre basin remains challengeable."""
    if not isinstance(diagnostic_point, dict) or not diagnostic_point:
        return suggestions
    engine = [s for s in suggestions if s.strategy == "greedy_noisy_ei"]
    if not engine:
        return suggestions
    row = _validated_llm_candidate_rows(project, [diagnostic_point])
    if row is None or not len(row):
        return suggestions
    values = row.iloc[0].to_dict()
    values.pop("_llm", None)
    if _row_exists(values, observed, project.factors):
        return suggestions
    weakest = min(engine, key=lambda s: s.score)
    replacement = DesignSuggestion(
        weakest.candidate_id, 0.0, _json_ready(values), "diagnostic_point")
    return [replacement if s is weakest else s for s in suggestions]


def _row_exists(values: dict, observed, factors) -> bool:
    """Exact-row duplicate check, categorical-aware: numeric factors compare
    within 1e-9, categorical factors by canonical string -- calling float()
    on a category label would raise."""
    if observed is None or not len(observed):
        return False
    for record in observed.to_dict(orient="records"):
        same = True
        for factor in factors:
            candidate = values.get(factor.name)
            existing = record.get(factor.name)
            if factor.kind == "categorical":
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
            return True
    return False


def bo_diagnostics(
    project: ProjectSpec,
    rows: pd.DataFrame,
    *,
    candidate_pool_size: int = 2048,
    ensemble_size: int = 5,
    training_steps: int = 300,
    language: str = "zh",
) -> dict[str, Any]:
    """Return surrogate diagnostics for a project's experiment history.

    Fits the ensemble on completed rows and reports per-factor effects
    (importance, recommended range, uncertainty) plus fit and runtime
    summaries. Returns {"model_ready": False, "reason": ...} when there are
    too few completed rows to train; `language` ("en"/"zh") selects the
    reason string. Never mutates campaign state."""
    _require_torch()
    _validate_factor_domains(project.factors)
    scored, _ = score_components(rows.copy(), project.objectives)
    target_columns, objective_weights = _training_targets(scored, project.objectives)
    clean = _scored_complete_rows(scored, project)
    minimum_rows = _minimum_training_rows(project)
    if clean.shape[0] < minimum_rows or not target_columns:
        reason = (
            f"Need at least {minimum_rows} completed experiments before surrogate training."
            if language == "en"
            else f"已完成实验不足，至少需要 {minimum_rows} 条后才训练代理模型。"
        )
        return {
            "model_ready": False,
            "reason": reason,
            "strategy": "space_filling_initial_design",
            "completed_rows": int(clean.shape[0]),
            "minimum_rows": int(minimum_rows),
            "factor_effects": [],
        }

    clean = clean.dropna(subset=target_columns + ["FuzzyScore"])
    if clean.shape[0] < minimum_rows:
        reason = (
            f"Need at least {minimum_rows} scored samples before surrogate training."
            if language == "en"
            else f"有效评分样本不足，至少需要 {minimum_rows} 条后才训练代理模型。"
        )
        return {
            "model_ready": False,
            "reason": reason,
            "strategy": "space_filling_initial_design",
            "completed_rows": int(clean.shape[0]),
            "minimum_rows": int(minimum_rows),
            "factor_effects": [],
        }
    clean = _active_training_set(clean, project.factors, target_columns, max_rows=MAX_TRAINING_ROWS, seed=83)

    runtime = _bo_runtime_config(
        n_factors=len(project.factors),
        n_rows=int(clean.shape[0]),
        candidate_pool_size=candidate_pool_size,
        ensemble_size=ensemble_size,
        training_steps=training_steps,
    )
    feature_columns = [factor.name for factor in project.factors]
    encoder = FactorEncoder(project.factors)
    X = encoder.encode_frame(clean[feature_columns])
    Y = torch.tensor(clean[target_columns].to_numpy(dtype=np.float32), dtype=torch.float32)
    model = TorchObjectiveEnsemble(
        input_dim=encoder.input_dim,
        n_objectives=Y.shape[1],
        objective_weights=objective_weights,
        ensemble_size=int(runtime["ensemble_size"]),
        training_steps=int(runtime["training_steps"]),
    )
    model.fit(X, Y)
    observed_scores = torch.tensor(clean["FuzzyScore"].to_numpy(dtype=np.float32), dtype=torch.float32)
    candidate_df = _propose_candidate_frame(
        factors=project.factors,
        encoder=encoder,
        model=model,
        observed_df=clean[feature_columns],
        observed_scores=observed_scores,
        n_points=max(96, len(project.factors) * 10),
        raw_samples=min(MAX_DIAGNOSTIC_CANDIDATE_POOL_SIZE, max(int(runtime["candidate_pool_size"]), len(project.factors) * 512)),
        seed=211,
    )
    candidate_df = _remove_existing(candidate_df, _observed_factor_frame(rows, project.factors), project.factors)
    observed_df = clean[feature_columns].drop_duplicates().reset_index(drop=True)
    analysis_df = pd.concat([candidate_df, observed_df], ignore_index=True).drop_duplicates().reset_index(drop=True)
    if analysis_df.empty:
        analysis_df = observed_df.copy()

    X_candidate = encoder.encode_frame(analysis_df[feature_columns])
    score_samples = model.sample_scores(X_candidate, n_samples=160)
    score_mean = score_samples.mean(dim=0)
    score_std = score_samples.std(dim=0, unbiased=False)
    acquisition = model.noisy_expected_improvement(X_candidate, X, n_samples=160)
    ranked_indices = torch.argsort(acquisition, descending=True).detach().cpu().tolist()
    top_k = min(max(32, len(project.factors) * 8), len(analysis_df))
    top_indices = ranked_indices[:top_k]
    top_candidates = analysis_df.iloc[top_indices].reset_index(drop=True) if top_indices else analysis_df.iloc[:0].copy()
    anchor_k = min(max(12, len(project.factors) * 4), len(top_candidates))
    anchor_df = top_candidates.iloc[:anchor_k].reset_index(drop=True) if anchor_k > 0 else analysis_df.head(min(12, len(analysis_df))).reset_index(drop=True)

    score_series = pd.to_numeric(clean["FuzzyScore"], errors="coerce")
    best_observed = clean.loc[score_series.fillna(-1).idxmax()].to_dict()
    factor_effects: list[dict[str, Any]] = []
    total_importance = 0.0
    for factor in project.factors:
        support = _diagnostic_support_values(factor, analysis_df[factor.name] if factor.name in analysis_df.columns else pd.Series(dtype=object))
        if not support or anchor_df.empty:
            continue
        eval_records = []
        for anchor in anchor_df.to_dict(orient="records"):
            for support_value in support:
                record = dict(anchor)
                record[factor.name] = support_value
                eval_records.append(record)
        eval_df = pd.DataFrame(eval_records, columns=feature_columns)
        eval_samples = model.sample_scores(encoder.encode_frame(eval_df[feature_columns]), n_samples=96)
        eval_mean = eval_samples.mean(dim=0).detach().cpu().numpy()
        eval_std = eval_samples.std(dim=0, unbiased=False).detach().cpu().numpy()
        mean_grid = eval_mean.reshape(len(anchor_df), len(support))
        std_grid = eval_std.reshape(len(anchor_df), len(support))
        if len(support) >= 3:
            importance_raw = float(np.mean(np.quantile(mean_grid, 0.9, axis=1) - np.quantile(mean_grid, 0.1, axis=1)))
        else:
            importance_raw = float(np.mean(mean_grid.max(axis=1) - mean_grid.min(axis=1)))
        total_importance += importance_raw
        factor_effects.append(
            {
                "name": factor.name,
                "kind": factor.kind,
                "current_best": _json_value(best_observed.get(factor.name)),
                "recommended_range": _diagnostic_recommended_range_from_response(factor, support, mean_grid, std_grid),
                "importance_raw": importance_raw,
                "uncertainty": float(np.mean(std_grid)),
                "support_size": int(len(support)),
            }
        )
    # Boundary-saturation feedback (BO audit Finding 5): when the surrogate's
    # recommended range hugs a CONFIGURED bracket edge, the optimizer is
    # pinned at that bound — indistinguishable from progress without this
    # flag, and cycle N+1 piles at the bound again once the boundary
    # experiment lands. A tolerance of 2% of the bracket span (or the
    # ordinal step) separates "near the edge" from interior optima.
    for item in factor_effects:
        item["importance"] = float(item["importance_raw"] / total_importance) if total_importance > 1e-12 else 0.0
        factor = next((f for f in project.factors if f.name == item.get("name")), None)
        if factor is None or factor.kind == "categorical":
            continue
        try:
            lower = float(_factor_min(factor))
            upper = float(_factor_max(factor))
        except (TypeError, ValueError):
            continue
        span = upper - lower
        if not np.isfinite(span) or span <= 0:
            continue
        # _factor_step RAISES for continuous factors without a step — read
        # the raw value tolerantly instead.
        step_value = parse_numeric_value(factor.step)
        tolerance = max(0.02 * span, float(step_value) if (step_value is not None and step_value > 0) else 0.0)
        recommended = item.get("recommended_range") or []
        numeric = [float(value) for value in recommended if isinstance(value, (int, float))]
        if not numeric:
            continue
        at_lower = abs(min(numeric) - lower) <= tolerance
        at_upper = abs(max(numeric) - upper) <= tolerance
        if at_lower or at_upper:
            edge = "lower" if at_lower else "upper"
            item["boundary_saturation"] = {
                "edge": edge,
                "bound": lower if at_lower else upper,
                "note": (
                    f"Optimizer pinned at the configured {edge} bound of {item.get('name')} "
                    f"({('min' if at_lower else 'max')}={lower if at_lower else upper}). "
                    "Widen the bracket in Parameters if the true optimum may lie beyond it."
                ),
            }
    factor_effects.sort(key=lambda item: (float(item.get("importance") or 0.0), float(item.get("uncertainty") or 0.0)), reverse=True)
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


class FactorEncoder:
    def __init__(self, factors: list[FactorSpec]):
        _require_torch()
        self.factors = factors
        self.input_dim = 0
        self.category_maps: dict[str, dict[str, int]] = {}
        for factor in self.factors:
            if factor.kind == "categorical":
                categories = [str(value) for value in factor.categories]
                self.category_maps[factor.name] = {value: index for index, value in enumerate(categories)}
                self.input_dim += max(len(categories), 1)
            else:
                self.input_dim += 1
        self.input_dim = max(self.input_dim, 1)

    def encode_frame(self, frame: pd.DataFrame) -> "torch.Tensor":
        rows = [self.encode_row(record) for record in frame.to_dict(orient="records")]
        return torch.stack(rows, dim=0) if rows else torch.empty((0, self.input_dim), dtype=torch.float32)

    def to_unit_frame(self, frame: pd.DataFrame) -> "torch.Tensor":
        rows = [self.to_unit_row(record) for record in frame.to_dict(orient="records")]
        return torch.stack(rows, dim=0) if rows else torch.empty((0, max(len(self.factors), 1)), dtype=torch.float32)

    def encode_row(self, row: dict[str, Any]) -> "torch.Tensor":
        return self.encode_unit(self.to_unit_row(row), relaxed=False).squeeze(0)

    def encode_unit(self, X_unit: "torch.Tensor", relaxed: bool = False) -> "torch.Tensor":
        if X_unit.ndim == 1:
            X_unit = X_unit.unsqueeze(0)
        encoded_parts = []
        for idx, factor in enumerate(self.factors):
            values = X_unit[:, idx].clamp(0.0, 1.0)
            if factor.kind == "categorical":
                n_categories = max(len(factor.categories), 1)
                if relaxed:
                    centers = (torch.arange(n_categories, device=X_unit.device, dtype=X_unit.dtype) + 0.5) / n_categories
                    logits = -24.0 * (values.unsqueeze(-1) - centers).pow(2)
                    encoded_parts.append(torch.softmax(logits, dim=-1))
                else:
                    indices = torch.clamp((values * n_categories).floor().long(), max=n_categories - 1)
                    encoded_parts.append(torch.nn.functional.one_hot(indices, num_classes=n_categories).to(dtype=X_unit.dtype))
            else:
                encoded_parts.append(values.unsqueeze(-1))
        return torch.cat(encoded_parts, dim=-1) if encoded_parts else X_unit.new_full((X_unit.shape[0], 1), 0.5)

    def to_unit_row(self, row: dict[str, Any]) -> "torch.Tensor":
        values = []
        for factor in self.factors:
            raw = row.get(factor.name)
            if factor.kind == "categorical":
                categories = [str(value) for value in factor.categories]
                index = self.category_maps.get(factor.name, {}).get(str(raw), 0)
                values.append((float(index) + 0.5) / float(len(categories)))
                continue
            numeric = parse_numeric_value(raw)
            if numeric is None:
                raise BOError(f"Factor {factor.name} has non-numeric value {raw!r}.")
            min_value = _factor_min(factor)
            max_value = _factor_max(factor)
            if factor.kind == "ordinal":
                step = _factor_step(factor)
                levels = _ordinal_level_count(factor)
                index = round((numeric - min_value) / step)
                values.append(float(max(0, min(index, levels - 1))) / float(levels - 1))
            else:
                values.append((numeric - min_value) / max(max_value - min_value, 1e-9))
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
        self.ensemble_size = max(2, int(ensemble_size))
        self.hidden_dim = int(hidden_dim)
        self.training_steps = max(50, int(training_steps))
        self.learning_rate = float(learning_rate)
        self.models: list[nn.Module] = []
        self.target_transform = BoundedTargetTransform()
        self.fit_summary: dict[str, Any] = {}
        self.uncertainty_scale = 1.0
        self.uncertainty_floor = 0.0
        self.member_weights = torch.empty((0,), dtype=torch.float32)

    def fit(self, X: "torch.Tensor", y: "torch.Tensor") -> None:
        if X.ndim != 2:
            raise BOError("X must be a 2D tensor.")
        if y.ndim != 1:
            y = y.view(-1)
        y_normalized = self.target_transform.fit(y)
        sample_count = int(X.shape[0])
        validation_size = min(max(sample_count // 5, 2), max(sample_count - 4, 0)) if sample_count >= 10 else 0
        if validation_size > 0:
            # Data-derived split seed: the train/val permutation must not
            # depend on the global RNG entry state (auxiliary fits elsewhere
            # would silently change this model).
            split_seed = 977 + int(y.detach().abs().sum().item() * 1e4) % (2**30)
            split_gen = torch.Generator()
            split_gen.manual_seed(split_seed)
            permutation = torch.randperm(sample_count, generator=split_gen)
            val_indices = permutation[:validation_size]
            train_indices = permutation[validation_size:]
        else:
            train_indices = torch.arange(sample_count)
            val_indices = torch.arange(0)
        X_train = X[train_indices]
        y_train = y_normalized[train_indices]
        X_val = X[val_indices] if val_indices.numel() else None
        y_val = y[val_indices] if val_indices.numel() else None
        self.models = []
        validation_losses: list[float] = []
        training_losses: list[float] = []
        epochs_ran: list[int] = []
        for model_index in range(self.ensemble_size):
            torch.manual_seed(2026 + model_index)
            model = _Regressor(self.input_dim, self.hidden_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            loss_fn = nn.SmoothL1Loss(beta=0.1)
            bootstrap_indices = torch.randint(low=0, high=max(int(X_train.shape[0]), 1), size=(max(int(X_train.shape[0]), 1),))
            X_boot = X_train[bootstrap_indices]
            y_boot = y_train[bootstrap_indices].unsqueeze(-1)
            best_state = None
            best_metric = float("inf")
            best_epoch = 0
            patience = max(20, min(80, self.training_steps // 5))
            loss = torch.tensor(0.0)
            for step in range(self.training_steps):
                optimizer.zero_grad()
                loss = loss_fn(model(X_boot), y_boot)
                loss.backward()
                optimizer.step()
                metric = float(loss.item())
                if X_val is not None and y_val is not None and X_val.shape[0] > 0:
                    model.eval()
                    with torch.no_grad():
                        val_prob = self.target_transform.inverse(model(X_val).squeeze(-1))
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
            validation_losses.append(best_metric if best_metric < float("inf") else float(loss.item()))
            training_losses.append(float(loss.item()))
            epochs_ran.append(best_epoch + 1)
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
        else:
            self.uncertainty_scale = 2.0
            self.uncertainty_floor = float(max(0.12, 0.45 / math.sqrt(max(sample_count, 1))))
        self.fit_summary["uncertainty_scale"] = self.uncertainty_scale
        self.fit_summary["uncertainty_floor"] = self.uncertainty_floor

    def _predict_members(self, X: "torch.Tensor") -> "torch.Tensor":
        if not self.models:
            raise BOError("The ensemble model has not been fitted.")
        return torch.stack([model(X).squeeze(-1) for model in self.models], dim=0)

    def _member_weight_tensor(self, device: "torch.device", dtype: "torch.dtype", count: int) -> "torch.Tensor":
        if self.member_weights.numel() != count:
            return torch.full((count,), 1.0 / max(count, 1), dtype=dtype, device=device)
        weights = self.member_weights.to(device=device, dtype=dtype)
        return weights / weights.sum().clamp_min(1e-6)

    def _weighted_moments(self, predictions: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        weights = self._member_weight_tensor(predictions.device, predictions.dtype, predictions.shape[0])
        mean_pred = (weights.unsqueeze(-1) * predictions).sum(dim=0)
        variance = (weights.unsqueeze(-1) * (predictions - mean_pred.unsqueeze(0)).pow(2)).sum(dim=0)
        std_pred = variance.clamp_min(1e-6).sqrt() * float(self.uncertainty_scale)
        return mean_pred, std_pred.clamp_min(float(self.uncertainty_floor))

    def _build_member_weights(self, validation_losses: list[float], *, device: "torch.device", dtype: "torch.dtype") -> "torch.Tensor":
        if not validation_losses:
            return torch.full((self.ensemble_size,), 1.0 / max(self.ensemble_size, 1), dtype=dtype, device=device)
        values = np.asarray(validation_losses, dtype=float)
        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            return torch.full((len(values),), 1.0 / max(len(values), 1), dtype=dtype, device=device)
        safe = np.where(finite_mask, values, np.nanmax(values[finite_mask]) + 1.0)
        scale = max(float(np.median(safe[finite_mask])), 1e-6)
        logits = -safe / scale
        weights = np.exp(logits - float(np.max(logits)))
        weights = weights / max(float(np.sum(weights)), 1e-9)
        return torch.tensor(weights, dtype=dtype, device=device)

    def _effective_member_count(self) -> float:
        if self.member_weights.numel() == 0:
            return float(self.ensemble_size)
        weights = self.member_weights.detach().cpu().numpy()
        denom = float(np.sum(np.square(weights)))
        return float(1.0 / denom) if denom > 1e-9 else float(self.ensemble_size)

    def predict_normalized(self, X: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        with torch.no_grad():
            predictions = self._predict_members(X)
        return self._weighted_moments(predictions)

    def predict_normalized_grad(self, X: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        return self._weighted_moments(self._predict_members(X))

    def sample_prob(
        self,
        X: "torch.Tensor",
        n_samples: int = 128,
        base_samples: "torch.Tensor | None" = None,
        with_grad: bool = False,
    ) -> "torch.Tensor":
        if base_samples is None:
            base_samples = torch.randn((n_samples, X.shape[0]), dtype=X.dtype, device=X.device)
        mean_pred, std_pred = self.predict_normalized_grad(X) if with_grad else self.predict_normalized(X)
        return self.target_transform.inverse(mean_pred.unsqueeze(0) + std_pred.unsqueeze(0) * base_samples)


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
        weights = torch.tensor(objective_weights, dtype=torch.float32) if objective_weights and len(objective_weights) == n_objectives else torch.ones(n_objectives, dtype=torch.float32)
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
        self.fit_summary: dict[str, Any] = {}

    def fit(self, X: "torch.Tensor", Y: "torch.Tensor") -> None:
        if Y.ndim == 1:
            Y = Y.unsqueeze(-1)
        if Y.shape[1] != len(self.models):
            raise BOError("The number of targets does not match the number of objective models.")
        self.objective_weights = self.objective_weights.to(device=Y.device, dtype=Y.dtype)
        for idx, model in enumerate(self.models):
            model.fit(X, Y[:, idx])
        validation_values = [float(model.fit_summary["mean_validation_rmse"]) for model in self.models if model.fit_summary.get("mean_validation_rmse") is not None]
        calibration_values = [float(model.fit_summary["uncertainty_scale"]) for model in self.models if model.fit_summary.get("uncertainty_scale") is not None]
        self.fit_summary = {
            "objective_count": int(len(self.models)),
            "train_size": int(self.models[0].fit_summary.get("train_size") or 0) if self.models else 0,
            "validation_size": int(self.models[0].fit_summary.get("validation_size") or 0) if self.models else 0,
            "mean_validation_rmse": float(np.mean(validation_values)) if validation_values else None,
            "uncertainty_scale": float(np.mean(calibration_values)) if calibration_values else None,
            "effective_ensemble_size": float(np.mean([float(model.fit_summary.get("effective_ensemble_size") or model.ensemble_size) for model in self.models])) if self.models else 0.0,
            "mean_epochs": int(round(float(np.mean([float(model.fit_summary.get("mean_epochs") or 0) for model in self.models])))) if self.models else 0,
            "training_loss": "smooth_l1",
        }

    def noisy_expected_improvement(
        self,
        X: "torch.Tensor",
        X_baseline: "torch.Tensor",
        n_samples: int = 128,
        candidate_base_samples: "torch.Tensor | None" = None,
        baseline_base_samples: "torch.Tensor | None" = None,
        with_grad: bool = False,
    ) -> "torch.Tensor":
        candidate_samples = self.sample_scores(X, n_samples=n_samples, base_samples=candidate_base_samples, with_grad=with_grad)
        baseline_samples = self.sample_scores(X_baseline, n_samples=n_samples, base_samples=baseline_base_samples, with_grad=False)
        return (candidate_samples - baseline_samples.max(dim=1).values.unsqueeze(-1)).clamp_min(0.0).mean(dim=0)

    def sample_scores(
        self,
        X: "torch.Tensor",
        n_samples: int = 128,
        base_samples: "torch.Tensor | None" = None,
        with_grad: bool = False,
    ) -> "torch.Tensor":
        if base_samples is None:
            base_samples = torch.randn((n_samples, X.shape[0], len(self.models)), dtype=X.dtype, device=X.device)
        parts = [
            model.sample_prob(X, n_samples=n_samples, base_samples=base_samples[:, :, idx], with_grad=with_grad).clamp_min(1e-6)
            for idx, model in enumerate(self.models)
        ]
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

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.network(x)
else:  # pragma: no cover
    class _Regressor:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            _require_torch()


def _training_targets(scored: pd.DataFrame, objectives: list[ObjectiveSpec]) -> tuple[list[str], list[float]]:
    component_columns = [f"_desirability_{objective.name}" for objective in objectives if f"_desirability_{objective.name}" in scored.columns]
    if not component_columns:
        return [], []
    return _learning_target_columns(scored, objectives, component_columns)


def _learning_target_columns(scored: pd.DataFrame, objectives: list[ObjectiveSpec], component_columns: list[str]) -> tuple[list[str], list[float]]:
    target_columns: list[str] = []
    weights: list[float] = []
    for objective in objectives:
        desirability_column = f"_desirability_{objective.name}"
        if desirability_column not in component_columns or objective.name not in scored.columns:
            continue
        formal = pd.to_numeric(scored[desirability_column], errors="coerce").astype(float)
        raw = coerce_numeric_series(scored[objective.name])
        progress = _objective_progress(raw, objective)
        learning = formal if progress is None else np.maximum(formal.to_numpy(dtype=float), 0.06 + 0.58 * progress)
        column = f"_bo_target_{objective.name}"
        scored[column] = np.clip(np.asarray(learning, dtype=float), 1e-6, 1.0)
        target_columns.append(column)
        weights.append(float(objective.weight or 1.0))
    if not target_columns:
        return component_columns, [1.0] * len(component_columns)
    if len(target_columns) <= 2:
        return target_columns, weights
    matrix = scored[target_columns].to_numpy(dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.sum() <= 0:
        weight_array = np.ones_like(weight_array)
    weight_array = weight_array / weight_array.sum()
    scored["_bo_target_score"] = np.exp(np.sum(np.log(np.clip(matrix, 1e-6, 1.0)) * weight_array, axis=1))
    return ["_bo_target_score"], [1.0]


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
    selected: set[Any] = set()
    score = coerce_numeric_series(frame.get("FuzzyScore", pd.Series(dtype=float))).fillna(-1.0)
    top_indices = score.sort_values(ascending=False).head(max(120, max_rows // 4)).index.tolist()
    selected.update(top_indices)
    selected.update(score.sort_values(ascending=True).head(max(16, max_rows // 40)).index.tolist())
    if top_indices:
        encoder = FactorEncoder(factors)
        feature_columns = [factor.name for factor in factors]
        candidate_frame = frame.dropna(subset=feature_columns)
        if not candidate_frame.empty:
            try:
                units = encoder.to_unit_frame(candidate_frame[feature_columns])
                anchor_units = encoder.to_unit_frame(frame.loc[top_indices, feature_columns].dropna())
                if anchor_units.numel():
                    distances = torch.cdist(units, anchor_units).min(dim=1).values.detach().cpu().numpy()
                    neighbor_order = np.argsort(distances)[: max(160, max_rows // 3)]
                    selected.update(candidate_frame.iloc[neighbor_order].index.tolist())
            except (BOError, ValueError, RuntimeError) as exc:
                logger.debug("Active-set diversity sampling skipped: %s", exc)
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


def _order_active_set_indices(frame: pd.DataFrame, selected: set[Any], score: pd.Series) -> list[Any]:
    if not selected:
        return []
    selected_set = set(selected)
    ordered = [idx for idx in score.sort_values(ascending=False).index if idx in selected_set]
    ordered.extend(idx for idx in frame.index if idx in selected_set and idx not in set(ordered))
    return ordered


def _objective_progress(series: pd.Series, objective: ObjectiveSpec) -> np.ndarray | None:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
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
    elif objective.goal == "target":
        if objective.target_min is not None and objective.target_max is not None:
            left = min(float(objective.target_min), float(objective.target_max))
            right = max(float(objective.target_min), float(objective.target_max))
            center = (left + right) / 2.0
            radius = max(abs(high - center), abs(low - center), span * 0.5, 1e-12)
            distance = np.where(values < left, left - values, np.where(values > right, values - right, 0.0))
            progress = 1.0 - distance / radius
        else:
            target = parse_numeric_value(objective.target_value)
            if target is None:
                target = (low + high) / 2.0
            radius = max(abs(high - target), abs(low - target), span * 0.5, 1e-12)
            progress = 1.0 - np.abs(values - target) / radius
    else:
        progress = (values - low) / span
    return np.clip(np.where(valid, progress, np.nan), 0.0, 1.0)


def _propose_candidate_frame(
    factors: list[FactorSpec],
    encoder: FactorEncoder,
    model: TorchObjectiveEnsemble,
    observed_df: pd.DataFrame,
    observed_scores: "torch.Tensor",
    n_points: int,
    raw_samples: int,
    seed: int,
) -> pd.DataFrame:
    n_dim = max(len(factors), 1)
    unit_samples = _draw_unit_samples(max(raw_samples, 64), n_dim, seed)
    observed_units = encoder.to_unit_frame(observed_df)
    trust_region = _estimate_trust_region(observed_units, observed_scores, len(factors))
    local_samples = _draw_local_mixed_samples(factors, observed_units, observed_scores, max(raw_samples // 2, n_points * 16), seed + 17, trust_region)
    comfort_samples = _draw_comfort_zone_samples(max(raw_samples // 4, n_points * 12), n_dim, seed + 29)
    optimized = _optimize_acquisition(
        unit_samples=torch.cat([unit_samples, local_samples, comfort_samples], dim=0),
        encoder=encoder,
        model=model,
        baseline_X=encoder.encode_frame(observed_df),
        n_candidates=n_points,
    )
    snapped = _snap_unit_samples(factors, torch.cat([unit_samples, local_samples, comfort_samples, optimized], dim=0))
    return _unit_candidates_to_frame(factors, snapped)


def _optimize_acquisition(
    unit_samples: "torch.Tensor",
    encoder: FactorEncoder,
    model: TorchObjectiveEnsemble,
    baseline_X: "torch.Tensor",
    n_candidates: int,
    steps: int = 64,
    learning_rate: float = 0.05,
    mc_samples: int = 64,
) -> "torch.Tensor":
    n_restarts = min(max(10, n_candidates * 8), unit_samples.shape[0])
    with torch.no_grad():
        baseline_features = encoder.encode_unit(unit_samples, relaxed=False)
        baseline_ei = model.noisy_expected_improvement(baseline_features, baseline_X, n_samples=mc_samples)
        start_indices = torch.topk(baseline_ei, k=n_restarts, dim=0).indices
    candidates = unit_samples[start_indices].clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([candidates], lr=learning_rate)
    candidate_base_samples = torch.randn((mc_samples, n_restarts, len(model.models)), dtype=candidates.dtype, device=candidates.device)
    baseline_base_samples = torch.randn((mc_samples, baseline_X.shape[0], len(model.models)), dtype=baseline_X.dtype, device=baseline_X.device)
    for _ in range(steps):
        optimizer.zero_grad()
        features = encoder.encode_unit(candidates, relaxed=True)
        scores = model.noisy_expected_improvement(features, baseline_X, n_samples=mc_samples, candidate_base_samples=candidate_base_samples, baseline_base_samples=baseline_base_samples, with_grad=True)
        (-scores.sum()).backward()
        optimizer.step()
        with torch.no_grad():
            candidates.clamp_(0.0, 1.0)
    return candidates.detach()


def _soft_constraint_penalty(
    candidate_unit: "torch.Tensor",
    soft_constraints: list[dict] | None,
    factors: list[FactorSpec],
) -> "torch.Tensor":
    """Bounded acquisition penalty for LLM-proposed soft region constraints.

    Each constraint: {"factors": {"<name>": [lo, hi]}, "weight": 0..1} in
    REAL factor units. The penalty is weight x (mean violated fraction across
    the constrained coordinates), clipped to [0, 0.5] - constraints bias the
    selection, they can never truncate the search space.
    """
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
        # the penalty grows with depth INTO the box. Penalizing points
        # outside it instead would steer batches into forbidden regions.
        inside = ((values >= lo_u) & (values <= hi_u)).float()
        # full penalty for any presence inside; smooth ramp by relative depth
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
    X_candidate: "torch.Tensor",
    X_baseline: "torch.Tensor",
    X_candidate_unit: "torch.Tensor",
    X_baseline_unit: "torch.Tensor",
    observed_scores: "torch.Tensor",
    n_points: int,
    strategy_phase: str | None = None,
    floor_mode: bool = False,
    soft_constraints: list[dict] | None = None,
    factor_encoder: "FactorEncoder | None" = None,
    generator: "torch.Generator | None" = None,
    recent_gain: float | None = None,
    floor_pathology: bool = False,
) -> tuple[list[int], list[float]]:
    if int(X_candidate.shape[0]) == 0:
        return [], []
    candidate_count = int(X_candidate.shape[0])
    if candidate_count <= n_points:
        small_cand = torch.randn((192, candidate_count, len(model.models)), generator=generator, dtype=X_candidate.dtype) if generator is not None else None
        small_base = torch.randn((192, X_baseline.shape[0], len(model.models)), generator=generator, dtype=X_baseline.dtype) if generator is not None else None
        acquisition = model.noisy_expected_improvement(
            X=X_candidate,
            X_baseline=X_baseline,
            n_samples=192,
            **({"candidate_base_samples": small_cand, "baseline_base_samples": small_base} if generator is not None else {}),
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
        n_models = len(model.models)
        cand_base = torch.randn((192, remaining_X.shape[0], n_models), generator=generator, dtype=remaining_X.dtype)
        base_base = torch.randn((192, current_baseline.shape[0], n_models), generator=generator, dtype=current_baseline.dtype)
        acquisition = model.noisy_expected_improvement(
            remaining_X, current_baseline, n_samples=192,
            candidate_base_samples=cand_base, baseline_base_samples=base_base,
        )
        min_dist = torch.cdist(remaining_unit, current_baseline_unit).min(dim=1).values
        comfort = 1.0 - (remaining_unit - 0.5).abs().mul(2.0).mean(dim=1)
        acquisition_span = float((acquisition.max() - acquisition.min()).item())
        acq_norm = _normalize_selection_signal(acquisition)
        dist_norm = _normalize_selection_signal(min_dist)
        comfort_norm = comfort.clamp(0.0, 1.0)
        # Active recovery (floor mode): the NEI surface is structurally flat
        # when every observation sits at the desirability floor, so selection
        # switches to posterior-sampled (Thompson-style) draws -- one draw of
        # the aggregated posterior per candidate; its argmax is the pick.
        thompson_norm = None
        if floor_mode:
            # sample_scores -> [n_samples, n_candidates]; a SINGLE posterior
            # draw per greedy step is a true Thompson pick (mean over draws
            # would cancel the exploration noise).
            draws = model.sample_scores(remaining_X, n_samples=4)
            thompson = draws[0] if draws.dim() == 2 else draws
            thompson_norm = _normalize_selection_signal(thompson)
        # Trust-gated blend: the acquisition signal drives selection only as
        # far as the surrogate is trusted. In low-signal regimes (flat
        # desirability plateaus, tiny histories) the ensemble's NEI surface is
        # mostly bootstrap noise; weighting it at full strength makes the batch
        # chase that noise into corners. The trust gate hands selection over to
        # coverage (distance) and comfort regularization until real signal
        # accumulates.
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
            # Interior-first exploration under diagnosed floor pathology: with
            # every observation at the floor, max-min distance rewards the
            # BOUNDARY (the empirically failed region) while the interior is
            # unsampled — exactly where the chemistry-informed comfort prior
            # says good regions live. The comfort term takes the exploration
            # weight; distance demotes to a batch-spread tie-breaker until
            # real signal returns (recent incumbent improvement).
            comfort_weight = max(comfort_weight, exploration_weight)
            exploration_weight = 0.30
        # Campaign-strategy phase bias (bounded): the LLM strategist may steer
        # the SELECTION regime only — never the data, the model, or the
        # operator's objective contract. explore widens coverage and halves
        # surrogate trust; exploit deepens trust and center preference.
        if strategy_phase == "explore":
            exploration_weight = min(exploration_weight + 0.25, 1.5)
            trust_weight *= 0.5
        # "exploit" adds no extra bias: the trust-gated blend
        # already concentrates selection on the learned best region when
        # trust is high, and an additional exploit bias double-counts and
        # over-converges small batches.
        selection_score = trust_weight * acq_norm + exploration_weight * dist_norm + comfort_weight * comfort_norm
        if thompson_norm is not None:
            selection_score = selection_score + 0.8 * thompson_norm
        if soft_constraints:
            selection_score = selection_score - _soft_constraint_penalty(
                remaining_unit, soft_constraints, factors
            )
        selected_unit = X_candidate_unit[selected] if selected else torch.empty((0, X_candidate_unit.shape[1]), dtype=X_candidate_unit.dtype, device=X_candidate_unit.device)
        batch_gain = _batch_marginal_utility_signal(factors, remaining_unit, X_baseline_unit, selected_unit)
        batch_weight = _batch_utility_weight(sample_count, incumbent, acquisition_span)
        if trust_weight >= 0.5:
            # Exploitation regime: discrete-diversity gain is a tie-breaker, not
            # a driver — it must not pull the batch away from the learned best
            # region (e.g. toward ordinal levels only seen in failed rows).
            batch_weight = min(batch_weight, 0.15)
        selection_score = selection_score + batch_weight * batch_gain
        force_mask = _discrete_coverage_force_mask(
            factors=factors,
            candidate_units=remaining_unit,
            selected_units=selected_unit,
            batch_size=n_points,
            remaining_slots=n_points - len(selected),
            observed_units=X_baseline_unit,
            observed_scores=observed_scores,
        )
        if force_mask.any():
            selection_score = torch.where(
                force_mask,
                selection_score,
                torch.full_like(selection_score, -float("inf")),
            )
        best_position = int(torch.argmax(selection_score).item())
        best_index = remaining.pop(best_position)
        selected.append(best_index)
        selected_scores.append(float(acquisition[best_position].item()))
        current_baseline = torch.cat([current_baseline, X_candidate[best_index : best_index + 1]], dim=0)
        current_baseline_unit = torch.cat([current_baseline_unit, X_candidate_unit[best_index : best_index + 1]], dim=0)
    return selected, selected_scores


def _discrete_coverage_force_mask(
    *,
    factors: list[FactorSpec],
    candidate_units: "torch.Tensor",
    selected_units: "torch.Tensor",
    batch_size: int,
    remaining_slots: int,
    observed_units: "torch.Tensor | None" = None,
    observed_scores: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    if candidate_units.numel() == 0 or remaining_slots <= 0:
        return torch.zeros((int(candidate_units.shape[0]),), dtype=torch.bool, device=candidate_units.device)
    categorical_masks: list["torch.Tensor"] = []
    ordinal_masks: list["torch.Tensor"] = []
    categorical_missing_total = 0
    ordinal_missing_total = 0
    competitive_codes = _competitive_discrete_codes(factors, observed_units, observed_scores)
    for idx, factor in enumerate(factors):
        if factor.kind == "continuous" or idx >= candidate_units.shape[1]:
            continue
        n_levels = _discrete_level_count(factor)
        eligible_codes = competitive_codes.get(idx) if competitive_codes else None
        target_levels = len(eligible_codes) if eligible_codes is not None else n_levels
        if factor.kind == "ordinal":
            target_levels = min(target_levels, max(2, int(batch_size)))
        target = min(max(target_levels, 1), max(int(batch_size), 1))
        candidate_codes = _discrete_level_codes(factor, candidate_units[:, idx])
        selected_codes = _discrete_level_codes(factor, selected_units[:, idx]) if selected_units.numel() else torch.empty((0,), dtype=torch.long, device=candidate_units.device)
        selected_levels = {int(code.item()) for code in selected_codes}
        missing_codes = [
            code for code in range(n_levels)
            if code not in selected_levels and (eligible_codes is None or code in eligible_codes)
        ]
        missing_count = max(0, target - len(selected_levels))
        if missing_count <= 0 or not missing_codes:
            continue
        mask = torch.zeros((int(candidate_units.shape[0]),), dtype=torch.bool, device=candidate_units.device)
        for code in missing_codes:
            mask = mask | (candidate_codes == int(code))
        if mask.any():
            if factor.kind == "categorical":
                categorical_masks.append(mask)
                categorical_missing_total += min(missing_count, len(missing_codes))
            else:
                ordinal_masks.append(mask)
                ordinal_missing_total += min(missing_count, len(missing_codes))
    masks = categorical_masks or ordinal_masks
    missing_total = categorical_missing_total if categorical_masks else ordinal_missing_total
    if not masks or remaining_slots > missing_total:
        return torch.zeros((int(candidate_units.shape[0]),), dtype=torch.bool, device=candidate_units.device)
    forced = masks[0].clone()
    for mask in masks[1:]:
        forced = forced | mask
    return forced


def _competitive_discrete_codes(
    factors: list[FactorSpec],
    observed_units: "torch.Tensor | None",
    observed_scores: "torch.Tensor | None",
) -> dict[int, set[int]]:
    if observed_units is None or observed_scores is None or observed_units.numel() == 0 or observed_scores.numel() == 0:
        return {}
    count = min(int(observed_units.shape[0]), int(observed_scores.shape[0]))
    if count <= 0:
        return {}
    scores = observed_scores[:count].to(device=observed_units.device, dtype=observed_units.dtype)
    finite = torch.isfinite(scores)
    if not finite.any():
        return {}
    valid_scores = scores[finite]
    incumbent = float(valid_scores.max().item())
    competitive_floor = float(torch.quantile(valid_scores, 0.50).item())
    tolerance = _competitive_score_tolerance(valid_scores)
    result: dict[int, set[int]] = {}
    units = observed_units[:count]
    for idx, factor in enumerate(factors):
        if factor.kind == "continuous" or idx >= units.shape[1]:
            continue
        codes = _discrete_level_codes(factor, units[:, idx])
        eligible: set[int] = set()
        for code in range(_discrete_level_count(factor)):
            mask = (codes == code) & finite
            if not mask.any():
                continue
            best = float(scores[mask].max().item())
            if best >= competitive_floor or best >= incumbent - tolerance:
                eligible.add(code)
        if eligible:
            result[idx] = eligible
    return result


def _competitive_score_tolerance(scores: "torch.Tensor") -> float:
    if scores.numel() <= 1:
        return 0.20
    q75 = torch.quantile(scores, 0.75)
    q25 = torch.quantile(scores, 0.25)
    iqr = max(float((q75 - q25).item()), 0.0)
    return float(min(max(0.20, 0.35 * iqr), 0.30))


def _batch_marginal_utility_signal(
    factors: list[FactorSpec],
    candidate_units: "torch.Tensor",
    baseline_units: "torch.Tensor",
    selected_units: "torch.Tensor",
) -> "torch.Tensor":
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
    candidate_units: "torch.Tensor",
    baseline_units: "torch.Tensor",
    selected_units: "torch.Tensor",
) -> "torch.Tensor":
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


def _level_entropy(counts: "torch.Tensor") -> "torch.Tensor":
    total = counts.sum().clamp_min(1e-9)
    probs = counts / total
    return -(probs * probs.clamp_min(1e-9).log()).sum()


def _discrete_level_count(factor: FactorSpec) -> int:
    if factor.kind == "categorical":
        return max(len(factor.categories), 1)
    return _ordinal_level_count(factor)


def _discrete_level_codes(factor: FactorSpec, values: "torch.Tensor") -> "torch.Tensor":
    if factor.kind == "categorical":
        n_categories = max(len(factor.categories), 1)
        return torch.clamp((values.clamp(0.0, 1.0) * n_categories).floor().long(), max=n_categories - 1)
    levels = _ordinal_level_count(factor)
    return torch.round(values.clamp(0.0, 1.0) * (levels - 1)).long()


def _batch_utility_weight(sample_count: int, incumbent: float, acquisition_span: float) -> float:
    if acquisition_span > 0.12 and incumbent >= 0.9:
        return 0.18
    if sample_count < 12:
        return 0.42
    if sample_count < 24:
        return 0.32
    return 0.24


def _initial_design_frame(
    factors: list[FactorSpec],
    observed_rows: pd.DataFrame,
    n_points: int,
    seed: int,
    scored_rows: pd.DataFrame | None = None,
    strategy: str = "balanced",
) -> pd.DataFrame:
    exhaustive = _finite_design_space(factors)
    observed = _observed_factor_frame(observed_rows, factors)
    anchor = _best_observed_factor_frame(scored_rows if scored_rows is not None else observed_rows, factors)
    if exhaustive is not None:
        candidates = _remove_existing(exhaustive, observed, factors)
        if len(candidates) < n_points:
            raise BOError(f"Only {len(candidates)} untested candidates are available in the configured design space.")
        units = FactorEncoder(factors).to_unit_frame(candidates)
        baseline = FactorEncoder(factors).to_unit_frame(pd.concat([observed, anchor], ignore_index=True)) if not observed.empty or not anchor.empty else torch.empty((0, max(len(factors), 1)))
        order = _rank_initial_candidates(factors, units, baseline, n_points, strategy)
        return candidates.iloc[order[:n_points]].reset_index(drop=True)
    sobol = _draw_unit_samples(max(512, n_points * 128), max(len(factors), 1), seed)
    local = _draw_initial_local_samples(factors, anchor, max(128, n_points * 32), seed + 19)
    units = _snap_unit_samples(factors, torch.cat([sobol, local], dim=0) if local.numel() else sobol)
    candidates = _unit_candidates_to_frame(factors, units)
    candidates = _remove_existing(candidates, observed, factors)
    if len(candidates) < n_points:
        raise BOError(f"Only {len(candidates)} untested candidates are available in the configured design space.")
    candidate_units = FactorEncoder(factors).to_unit_frame(candidates)
    baseline = FactorEncoder(factors).to_unit_frame(pd.concat([observed, anchor], ignore_index=True)) if not observed.empty or not anchor.empty else torch.empty((0, max(len(factors), 1)))
    order = _rank_initial_candidates(factors, candidate_units, baseline, n_points, strategy)
    return candidates.iloc[order[:n_points]].reset_index(drop=True)


def _best_observed_factor_frame(rows: pd.DataFrame, factors: list[FactorSpec]) -> pd.DataFrame:
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


def _rank_initial_candidates(factors: list[FactorSpec], units: "torch.Tensor", baseline: "torch.Tensor", n_points: int, strategy: str = "balanced") -> list[int]:
    if int(units.shape[0]) == 0:
        return []
    strategy = str(strategy or "balanced").lower()
    center = torch.full((1, units.shape[1]), 0.5, dtype=units.dtype, device=units.device)
    selected: list[int] = []
    remaining = list(range(int(units.shape[0])))
    current = baseline.clone() if baseline.numel() else center.clone()
    # Center-anchored start (classic DOE center point): in the max-min
    # greedy the coverage term (0.55) dominates the center score (0.25) on
    # the first pick, so cold-start designs begin corner-first and the
    # comfort prior never has a home observation. Pick the nearest-to-center
    # candidate FIRST for cold-start campaigns.
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
            anchor = baseline[-1:]
            local_score = 1.0 - torch.cdist(remaining_units, anchor).squeeze(-1).clamp(0.0, 1.0)
            score = 0.62 * local_score + 0.28 * _normalize_selection_signal(min_dist) + 0.10 * center_score
        elif strategy in {"coverage", "space_filling", "explore"}:
            score = 0.82 * _normalize_selection_signal(min_dist) + 0.18 * center_score
        else:
            local_score = torch.zeros_like(center_score)
            if baseline.numel():
                local_score = 1.0 - torch.cdist(remaining_units, baseline[-1:]).squeeze(-1).clamp(0.0, 1.0)
            score = 0.55 * _normalize_selection_signal(min_dist) + 0.25 * center_score + 0.20 * local_score
        selected_units = units[selected] if selected else torch.empty((0, units.shape[1]), dtype=units.dtype, device=units.device)
        force_mask = _discrete_coverage_force_mask(
            factors=factors,
            candidate_units=remaining_units,
            selected_units=selected_units,
            batch_size=n_points,
            remaining_slots=n_points - len(selected),
        )
        if force_mask.any():
            score = torch.where(force_mask, score, torch.full_like(score, -float("inf")))
        position = int(torch.argmax(score).item())
        index = remaining.pop(position)
        selected.append(index)
        current = torch.cat([current, units[index : index + 1]], dim=0)
    return selected


def _draw_initial_local_samples(factors: list[FactorSpec], anchor: pd.DataFrame, n_points: int, seed: int) -> "torch.Tensor":
    if anchor is None or anchor.empty:
        return torch.empty((0, max(len(factors), 1)), dtype=torch.float32)
    encoder = FactorEncoder(factors)
    anchor_units = encoder.to_unit_frame(anchor)
    scores = torch.ones(anchor_units.shape[0], dtype=torch.float32)
    trust = torch.full((max(len(factors), 1),), 0.24, dtype=torch.float32)
    return _draw_local_mixed_samples(factors, anchor_units, scores, n_points, seed, trust)


def _finite_design_space(factors: list[FactorSpec], limit: int = 20000) -> pd.DataFrame | None:
    values: list[list[Any]] = []
    for factor in factors:
        if factor.kind == "categorical":
            values.append([str(value) for value in factor.categories])
        elif factor.kind == "ordinal":
            min_value = _factor_min(factor)
            max_value = _factor_max(factor)
            step = _factor_step(factor)
            levels = _ordinal_level_count(factor)
            values.append([round(min_value + idx * step, 6) for idx in range(levels)])
        else:
            return None
        size = math.prod(len(item) for item in values)
        if size > limit:
            return None
    records = [{factor.name: value for factor, value in zip(factors, combo)} for combo in itertools.product(*values)]
    return pd.DataFrame(records, columns=[factor.name for factor in factors]).drop_duplicates().reset_index(drop=True)


def _unit_candidates_to_frame(factors: list[FactorSpec], unit_points: "torch.Tensor") -> pd.DataFrame:
    if unit_points.numel() == 0:
        return pd.DataFrame(columns=[factor.name for factor in factors])
    return pd.DataFrame([_decode_sample_row(factors, row) for row in unit_points]).drop_duplicates().reset_index(drop=True)


def _decode_sample_row(factors: list[FactorSpec], row: "torch.Tensor") -> dict[str, Any]:
    values: dict[str, Any] = {}
    for idx, factor in enumerate(factors):
        x = float(row[idx].item()) if idx < len(row) else 0.5
        if factor.kind == "categorical":
            categories = [str(value) for value in factor.categories]
            index = min(int(math.floor(x * len(categories))), len(categories) - 1)
            values[factor.name] = categories[index]
        elif factor.kind == "ordinal":
            min_value = _factor_min(factor)
            step = _factor_step(factor)
            levels = _ordinal_level_count(factor)
            index = min(int(math.floor(x * levels)), max(levels - 1, 0))
            values[factor.name] = round(min_value + index * step, 6)
        else:
            min_value = _factor_min(factor)
            values[factor.name] = round(min_value + x * (_factor_max(factor) - min_value), 6)
    return values


def _draw_unit_samples(n_points: int, n_dim: int, seed: int) -> "torch.Tensor":
    return torch.quasirandom.SobolEngine(dimension=max(n_dim, 1), scramble=True, seed=seed).draw(max(int(n_points), 1)).to(dtype=torch.float32)


def _draw_comfort_zone_samples(n_points: int, n_dim: int, seed: int) -> "torch.Tensor":
    generator = torch.Generator().manual_seed(seed)
    count = max(int(n_points), 1)
    center = torch.full((1, max(n_dim, 1)), 0.5, dtype=torch.float32)
    if count == 1:
        return center
    jitter = torch.randn(count - 1, max(n_dim, 1), generator=generator, dtype=torch.float32) * 0.18
    return torch.cat([center, (0.5 + jitter).clamp(0.08, 0.92)], dim=0)


def _draw_local_mixed_samples(
    factors: list[FactorSpec],
    observed_units: "torch.Tensor",
    observed_scores: "torch.Tensor",
    n_points: int,
    seed: int,
    trust_region: "torch.Tensor",
) -> "torch.Tensor":
    if observed_units.numel() == 0:
        return torch.empty((0, max(len(factors), 1)), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    weights = observed_scores.clamp_min(1e-6)
    if weights.shape[0] != observed_units.shape[0]:
        weights = torch.ones(observed_units.shape[0], dtype=observed_units.dtype)
    weights = weights / weights.sum()
    indices = torch.multinomial(weights, num_samples=n_points, replacement=True, generator=generator)
    samples = observed_units[indices].clone()
    partner_indices = torch.multinomial(weights, num_samples=n_points, replacement=True, generator=generator) if observed_units.shape[0] > 1 else indices
    partners = observed_units[partner_indices]
    for idx, factor in enumerate(factors):
        column = samples[:, idx]
        radius = float(trust_region[idx].item()) if trust_region.numel() > idx else 0.18
        if factor.kind == "continuous":
            noise = torch.randn(n_points, generator=generator, dtype=samples.dtype) * max(radius * 0.45, 0.03)
            mix = torch.rand(n_points, generator=generator, dtype=samples.dtype)
            drift = (partners[:, idx] - column) * min(max(radius * 1.2, 0.12), 0.4) * mix
            samples[:, idx] = (column + noise + drift).clamp(0.0, 1.0)
        elif factor.kind == "ordinal":
            step = _factor_step(factor)
            levels = _ordinal_level_count(factor)
            max_jump = 1 if radius <= 0.18 else 2
            delta = torch.randint(-max_jump, max_jump + 1, size=(n_points,), generator=generator)
            base = torch.round(column * (levels - 1)).long()
            samples[:, idx] = (base.add(delta).clamp(0, levels - 1).to(samples.dtype)) / (levels - 1)
        else:
            n_categories = max(len(factor.categories), 1)
            choice = torch.randint(0, n_categories, size=(n_points,), generator=generator, dtype=torch.long)
            partner_choice = torch.clamp((partners[:, idx] * n_categories).floor().long(), max=n_categories - 1)
            keep_mask = torch.rand(n_points, generator=generator) < (0.75 if radius <= 0.18 else 0.6)
            partner_mask = torch.rand(n_points, generator=generator) < 0.5
            current = torch.clamp((column * n_categories).floor().long(), max=n_categories - 1)
            switched = torch.where(partner_mask, partner_choice, choice)
            samples[:, idx] = (torch.where(keep_mask, current, switched).to(samples.dtype) + 0.5) / n_categories
    return samples


def _snap_unit_samples(factors: list[FactorSpec], unit_points: "torch.Tensor") -> "torch.Tensor":
    snapped = unit_points.clone().clamp(0.0, 1.0)
    for idx, factor in enumerate(factors):
        column = snapped[:, idx]
        if factor.kind == "ordinal":
            levels = _ordinal_level_count(factor)
            snapped[:, idx] = torch.round(column * (levels - 1)) / (levels - 1)
        elif factor.kind == "categorical":
            n_categories = max(len(factor.categories), 1)
            indices = torch.clamp((column * n_categories).floor(), max=n_categories - 1)
            snapped[:, idx] = (indices + 0.5) / n_categories
    return snapped


def _estimate_trust_region(observed_units: "torch.Tensor", observed_scores: "torch.Tensor", n_factors: int) -> "torch.Tensor":
    if observed_units.numel() == 0:
        return torch.full((max(n_factors, 1),), 0.2, dtype=torch.float32)
    top_k = min(max(4, observed_units.shape[0] // 3), observed_units.shape[0])
    anchors = observed_units[torch.topk(observed_scores.clamp_min(1e-6), k=top_k, dim=0).indices]
    if anchors.shape[0] <= 1:
        return torch.full((anchors.shape[1],), 0.22, dtype=anchors.dtype)
    spread = (torch.quantile(anchors, 0.9, dim=0) - torch.quantile(anchors, 0.1, dim=0)).clamp_min(0.02)
    sample_scale = 1.15 if observed_units.shape[0] < 16 else 1.0
    return (spread * 0.75 * sample_scale).clamp(0.08, 0.35)


def _diagnostic_support_values(factor: FactorSpec, series: pd.Series, *, max_points: int = 9) -> list[Any]:
    if factor.kind == "categorical":
        return [str(value) for value in factor.categories if str(value).strip()]
    if factor.kind == "ordinal":
        min_value = _factor_min(factor)
        max_value = _factor_max(factor)
        step = _factor_step(factor)
        levels = _ordinal_level_count(factor)
        if levels <= max_points:
            return [round(min_value + idx * step, 6) for idx in range(levels)]
        grid = np.linspace(min_value, max_value, num=max_points)
        return sorted({round(min_value + round((value - min_value) / step) * step, 6) for value in grid})
    numeric = coerce_numeric_series(series).dropna()
    min_value = _factor_min(factor)
    max_value = _factor_max(factor)
    if numeric.nunique() >= max_points:
        return sorted({round(float(numeric.quantile(q)), 6) for q in np.linspace(0.05, 0.95, num=max_points)})
    return [round(float(value), 6) for value in np.linspace(min_value, max_value, num=max_points)]



def _diagnostic_recommended_range_from_response(
    factor: FactorSpec,
    support: list[Any],
    mean_grid: np.ndarray,
    std_grid: np.ndarray,
) -> list[Any]:
    if not support or mean_grid.size == 0:
        return []
    profile = np.nanmean(mean_grid, axis=0)
    uncertainty = np.nanmean(std_grid, axis=0) if std_grid.size else np.zeros_like(profile)
    finite = np.isfinite(profile)
    if not finite.any():
        return []
    safe_profile = np.where(finite, profile, -np.inf)
    best = float(np.max(safe_profile))
    spread = float(np.nanmax(profile[finite]) - np.nanmin(profile[finite])) if finite.any() else 0.0
    tolerance = float(min(max(0.05, 0.25 * spread), 0.16))
    stable_profile = safe_profile - 0.15 * np.where(np.isfinite(uncertainty), uncertainty, 0.0)
    stable_best = float(np.max(stable_profile[finite]))
    mask = finite & ((safe_profile >= best - tolerance) | (stable_profile >= stable_best - tolerance))
    if not mask.any():
        mask[int(np.argmax(safe_profile))] = True
    if factor.kind == "categorical":
        ranked = sorted(
            [idx for idx, keep in enumerate(mask.tolist()) if keep],
            key=lambda idx: (safe_profile[idx], stable_profile[idx]),
            reverse=True,
        )
        return [str(support[idx]) for idx in ranked[:3]]
    selected = [float(support[idx]) for idx, keep in enumerate(mask.tolist()) if keep]
    if not selected:
        selected = [float(support[int(np.argmax(safe_profile))])]
    if factor.kind == "ordinal":
        step = _factor_step(factor)
        base = _factor_min(factor)
        low = round((min(selected) - base) / step) * step + base
        high = round((max(selected) - base) / step) * step + base
        return [round(float(low), 6), round(float(high), 6)]
    return [round(float(min(selected)), 6), round(float(max(selected)), 6)]


def _aggregate_score_tensor(values: "torch.Tensor", weights: "torch.Tensor | None" = None) -> "torch.Tensor":
    if values.ndim == 1:
        return values.clamp_min(1e-6)
    safe = values.clamp_min(1e-6)
    if weights is None or weights.numel() != safe.shape[-1]:
        return torch.exp(torch.log(safe).mean(dim=-1))
    normalized = weights.to(device=safe.device, dtype=safe.dtype)
    normalized = normalized / normalized.sum().clamp_min(1e-6)
    return torch.exp(torch.sum(torch.log(safe) * normalized, dim=-1))


def _normalize_selection_signal(values: "torch.Tensor") -> "torch.Tensor":
    span = values.max() - values.min()
    if float(span.item()) < 1e-12:
        return torch.zeros_like(values)
    return (values - values.min()) / span.clamp_min(1e-12)


def _exploration_guard_weight(sample_count: int, incumbent: float, batch_position: int, acquisition_span: float) -> float:
    sample_term = max(0.0, (18.0 - float(sample_count)) / 18.0)
    quality_term = max(0.0, (0.72 - float(incumbent)) / 0.72)
    tie_term = 1.0 if acquisition_span < 1e-5 else 0.0
    batch_term = min(max(float(batch_position), 0.0), 3.0) * 0.12
    return float(min(max(0.18 + 0.38 * sample_term + 0.32 * quality_term + 0.35 * tie_term + batch_term, 0.18), 1.25))


# Realized-informativeness trust gating (TuRBO/FuRBO lineage): trust in the
# acquisition may rise with sample count only while batches keep improving the
# incumbent. A batch that lifts the best composite by this much desirability
# counts as fully informative; a flat batch counts as uninformative. Generic
# desirability scale, not tuned to any task. Kill switch: CHATEXP_TRUST_REKEY=0.
_TRUST_INFO_GAIN_SCALE = 0.15
_TRUST_REKEY_ENABLED = os.environ.get("CHATEXP_TRUST_REKEY", "1") != "0"


def _recent_incumbent_gain(clean: pd.DataFrame, n_points: int) -> float | None:
    """Realized improvement of the incumbent over the pre-batch best.

    Uses the batch (Trial) column when present; otherwise assumes the last
    ``n_points`` rows are the most recent batch (campaigns run a fixed batch
    size). Returns None when there is no earlier batch to compare against.
    """
    if "FuzzyScore" not in clean.columns or not len(clean):
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
    if "Trial" in clean.columns and len(clean):
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
    recent_gain: float | None = None,
) -> float:
    if acquisition_span < 1e-5:
        return 0.0
    sample_term = min(max((float(sample_count) - 4.0) / 24.0, 0.0), 1.0)
    quality_term = min(max(float(incumbent) / 0.65, 0.0), 1.0)
    # The incumbent term dominates: a well-scored incumbent
    # means the surrogate has real response signal worth exploiting; a
    # near-floor incumbent means every acquisition structure is extrapolation
    # from noise, and coverage/comfort should lead until the space reveals
    # something worth exploiting.
    if (
        _TRUST_REKEY_ENABLED
        and recent_gain is not None
        and incumbent < 0.35  # the engine's own health line; inert on healthy campaigns
    ):
        # On a floor-dominated history the NEI surface correlates with the
        # posterior spread (rho ~ 0.76-0.83 measured), so sample count alone
        # must NOT hand selection to the acquisition: trust also needs the
        # recent batch to have actually improved the incumbent (realized
        # informativeness, as trust-region BO keys expansion to realized
        # successes). A flat batch keeps trust at the coverage floor.
        info_term = min(max(float(recent_gain) / _TRUST_INFO_GAIN_SCALE, 0.0), 1.0)
        quality_term = min(quality_term, info_term)
    return float(min(max(0.10 + 0.45 * sample_term + 0.85 * quality_term, 0.10), 1.60))


def _comfort_guard_weight(sample_count: int, incumbent: float) -> float:
    sample_term = max(0.0, (20.0 - float(sample_count)) / 20.0)
    low_score_term = max(0.0, (0.45 - float(incumbent)) / 0.45)
    return float(min(0.10 + 0.45 * sample_term + 0.55 * low_score_term, 0.95))


def _bo_runtime_config(*, n_factors: int, n_rows: int, candidate_pool_size: int, ensemble_size: int, training_steps: int) -> dict[str, int]:
    return {
        "candidate_pool_size": int(min(MAX_CANDIDATE_POOL_SIZE, max(candidate_pool_size, n_factors * 512, int(math.sqrt(max(n_rows, 1))) * 256, 2048))),
        "ensemble_size": int(max(4, min(8, ensemble_size + (1 if n_rows >= 24 else 0)))),
        "training_steps": int(max(training_steps, min(900, 220 + n_factors * 24 + n_rows * 4))),
    }


def _minimum_training_rows(project: ProjectSpec) -> int:
    return max(4, len(project.factors) + 1)


def _scored_complete_rows(frame: pd.DataFrame, project: ProjectSpec) -> pd.DataFrame:
    required = [factor.name for factor in project.factors] + [objective.name for objective in project.objectives]
    if "FuzzyScore" in frame.columns:
        required.append("FuzzyScore")
    missing = [column for column in required if column not in frame.columns]
    if frame.empty or missing:
        return pd.DataFrame(columns=frame.columns)
    mask = frame[required].map(lambda value: not is_missing_value(value)).all(axis=1)
    return frame.loc[mask].copy()


def _observed_factor_frame(rows: pd.DataFrame, factors: list[FactorSpec]) -> pd.DataFrame:
    columns = [factor.name for factor in factors]
    if rows.empty or any(column not in rows.columns for column in columns):
        return pd.DataFrame(columns=columns)
    frame = rows[columns].copy()
    mask = frame.map(lambda value: not is_missing_value(value)).all(axis=1)
    return frame.loc[mask].drop_duplicates().reset_index(drop=True)


def _remove_existing(pool: pd.DataFrame, observed: pd.DataFrame, factors: list[FactorSpec]) -> pd.DataFrame:
    if pool.empty or observed.empty:
        return pool.reset_index(drop=True)
    keys = [factor.name for factor in factors]
    normalized_pool = pool.copy()
    normalized_observed = observed.copy()
    for factor in factors:
        if factor.kind == "categorical":
            normalized_pool[factor.name] = normalized_pool[factor.name].astype(str)
            normalized_observed[factor.name] = normalized_observed[factor.name].astype(str)
        else:
            normalized_pool[factor.name] = coerce_numeric_series(normalized_pool[factor.name]).round(6)
            normalized_observed[factor.name] = coerce_numeric_series(normalized_observed[factor.name]).round(6)
    existing = {tuple(record.get(key) for key in keys) for record in normalized_observed.to_dict(orient="records")}
    keep = [tuple(record.get(key) for key in keys) not in existing for record in normalized_pool.to_dict(orient="records")]
    return pool.loc[keep].reset_index(drop=True)


def _validate_factor_domains(factors: list[FactorSpec]) -> None:
    if not factors:
        raise BOError("No input factors are configured.")
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
        raise BOError("Invalid design space: " + "; ".join(problems))


def _factor_min(factor: FactorSpec) -> float:
    value = parse_numeric_value(factor.min_value)
    if value is None:
        raise BOError(f"Factor {factor.name} is missing min_value.")
    return float(value)


def _factor_max(factor: FactorSpec) -> float:
    value = parse_numeric_value(factor.max_value)
    if value is None:
        raise BOError(f"Factor {factor.name} is missing max_value.")
    return float(value)


def _factor_step(factor: FactorSpec) -> float:
    value = parse_numeric_value(factor.step)
    if value is None or value <= 0:
        raise BOError(f"Factor {factor.name} is missing a positive step.")
    return float(value)


def _ordinal_level_count(factor: FactorSpec) -> int:
    min_value = _factor_min(factor)
    max_value = _factor_max(factor)
    step = _factor_step(factor)
    return max(1, int(math.floor((max_value - min_value) / step + 1e-9)) + 1)


def _json_ready(record: dict[str, Any]) -> dict[str, Any]:
    return json_ready(record)


def _json_value(value: Any) -> Any:
    return json_value(value)


def _require_torch() -> None:
    if torch is None or nn is None:
        raise BOError("PyTorch is not installed in the BO API worker.")
