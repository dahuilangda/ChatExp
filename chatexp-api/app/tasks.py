from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_INTEROP_THREADS", "1")

import numpy as np
import pandas as pd
from celery import Celery
from celery.schedules import crontab

from .models import FactorSpec, ObjectiveSpec, ProjectSpec
from .objective_inference import infer_objectives, inferred_specs
from .optimizer import BOError, bo_diagnostics, suggest_next_design
from .scoring import coerce_numeric_series, format_desirability, is_missing_value, json_value as _json_value, parse_desirability, parse_numeric_value, score_components


REDIS_URL = os.getenv("CHATEXP_REDIS_URL", "redis://redis:6379/0")

celery_app = Celery(
    "chatexp_bo",
    broker=REDIS_URL,
    backend=REDIS_URL,
    # Explicitly list every module whose @celery_app.task decorators must run
    # for the worker to discover tasks. The worker is started as
    # `celery -A app.tasks.celery_app worker`, which only imports app.tasks —
    # without this include, app.files.tasks.parse_file_task would never register.
    include=["app.tasks", "app.files.tasks"],
)
celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Clients poll promptly; 24h retention of up-to-1MB results (BO scored
    # rows, base64 edited docs) needlessly crowded the shared 256MB->1GB
    # Redis under volatile-lru and risked evicting in-flight job metadata.
    result_expires=7200,
    # Compress broker + result payloads: BO rows (~0.4MB) and results
    # (~1MB) shrink 5-10x in Redis for ~5ms gzip CPU per message.
    task_compression="gzip",
    # Explicit routing: every task name maps to the queue its dedicated
    # worker listens on (-Q files / analysis,bo). Without this, tasks land
    # on the default "celery" queue that no worker consumes.
    task_routes={
        "chatexp.files.parse": {"queue": "files"},
        "chatexp.files.analyze": {"queue": "analysis"},
        "chatexp.files.edit": {"queue": "files"},
        "chatexp.files.reconcile": {"queue": "files"},
        "chatexp.bo.analyze": {"queue": "bo"},
        "chatexp.bo.suggest": {"queue": "bo"},
    },
    # chatexp:doc:index (ZSET) carries no per-member TTL, so members whose
    # metadata key expired linger until reconcile_index prunes them.
    beat_schedule={
        "chatexp-enforce-parsed-files-budget": {
          # The 2GB parsed-files high-water mark was previously enforced only
          # by the manual admin route — nothing scheduled it, so disk fill
          # was bounded only by upload rate.
          "task": "chatexp.enforce_parsed_files_budget",
          "schedule": 3600.0,
        },
        "chatexp-reconcile-doc-index": {
            "task": "chatexp.files.reconcile",
            "schedule": crontab(minute=0),
        },
    },
)


def _attach_diagnostics(
    analysis: dict[str, Any],
    project: ProjectSpec,
    frame: "pd.DataFrame",
    runtime: dict[str, Any],
    language: str | None,
) -> None:
    """Attach engine diagnostics to an analysis payload; on engine failure the
    payload reports an explicit not-ready reason instead of raising."""
    try:
        analysis["boDiagnostics"] = bo_diagnostics(
            project,
            frame,
            candidate_pool_size=int(runtime.get("candidate_pool_size", 2048)),
            ensemble_size=int(runtime.get("ensemble_size", 5)),
            training_steps=int(runtime.get("training_steps", 300)),
            language=language,
        )
    except (BOError, RuntimeError, ValueError) as error:
        analysis["boDiagnostics"] = {
            "model_ready": False,
            "reason": str(error),
            "factor_effects": [],
        }


@celery_app.task(name="chatexp.bo.analyze")
def analyze_project_task(payload: dict[str, Any]) -> dict[str, Any]:
    project, frame, runtime, language = _build_job(payload)
    analysis = _build_analysis(project, frame)
    _attach_diagnostics(analysis, project, frame, runtime, language)
    return {"analysis": analysis}


@celery_app.task(name="chatexp.bo.suggest")
def suggest_project_task(payload: dict[str, Any]) -> dict[str, Any]:
    project, frame, runtime, language = _build_job(payload)
    inference_report = _maybe_infer_objectives(project, frame)
    analysis = _build_analysis(project, frame)
    if inference_report is not None:
        analysis["autoInferredObjectives"] = inference_report
    n_points = int(payload.get("n_points") or 5)
    strategy_phase = str(runtime.get("strategy_phase") or "").strip().lower() or None
    soft_constraints = runtime.get("soft_constraints") or None
    llm_candidates = runtime.get("llm_candidates") or None
    prior_pseudo = runtime.get("prior_pseudo") or None
    diagnostic_point = runtime.get("diagnostic_point") or None
    suggestions = suggest_next_design(
        project,
        frame,
        n_points=n_points,
        candidate_pool_size=int(runtime.get("candidate_pool_size", 2048)),
        ensemble_size=int(runtime.get("ensemble_size", 5)),
        training_steps=int(runtime.get("training_steps", 300)),
        initial_strategy=str(payload.get("design_strategy") or payload.get("designStrategy") or "balanced"),
        strategy_phase=strategy_phase if strategy_phase in {"explore", "exploit"} else None,
        soft_constraints=soft_constraints if isinstance(soft_constraints, list) else None,
        llm_candidates=llm_candidates if isinstance(llm_candidates, list) else None,
        prior_pseudo=prior_pseudo if isinstance(prior_pseudo, list) else None,
        diagnostic_point=diagnostic_point if isinstance(diagnostic_point, dict) else None,
    )
    analysis["suggestions"] = [
        {
            "factor": "BO",
            "reason": (
                "Space-filling initial design"
                if suggestion.strategy == "space_filling_initial_design"
                else f"Expected improvement score {suggestion.score:.4f}"
            ),
            "score": suggestion.score,
            "strategy": suggestion.strategy,
            "values": suggestion.values,
            "candidate_id": suggestion.candidate_id,
        }
        for suggestion in suggestions
    ]
    _attach_diagnostics(analysis, project, frame, runtime, language)
    return {"analysis": analysis, "suggestions": analysis.get("suggestions", [])}


def _build_job(payload: dict[str, Any]) -> tuple[ProjectSpec, pd.DataFrame, dict[str, Any], str]:
    factors = [FactorSpec.model_validate(_normalize_factor(item)) for item in payload.get("factors", [])]
    objectives = [ObjectiveSpec.model_validate(_normalize_objective(item)) for item in payload.get("objectives", [])]
    frame = pd.DataFrame(payload.get("rows") or [])
    _hydrate_factor_domains(factors, frame)
    project = ProjectSpec(
        name=str(payload.get("project_name") or "seatable"),
        factors=factors,
        objectives=objectives,
    )
    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    language = "en" if str(payload.get("language") or "").lower().startswith("en") else "zh"
    return project, frame, runtime, language


def _normalize_objective(item: Any) -> dict[str, Any]:
    data = dict(item or {})
    target_value = data.get("target_value")
    if isinstance(target_value, dict) and any(key in target_value for key in ("min", "max", "target_min", "target_max")):
        if "target_min" not in data:
            data["target_min"] = target_value.get("min", target_value.get("target_min"))
        if "target_max" not in data:
            data["target_max"] = target_value.get("max", target_value.get("target_max"))
        data["target_value"] = None
    desirability = data.get("desirability")
    if isinstance(desirability, list):
        data["desirability"] = format_desirability(desirability)
    for key in ("target_min", "target_max"):
        if data.get(key) is not None:
            numeric = parse_numeric_value(data.get(key))
            if numeric is not None:
                data[key] = numeric
    if data.get("target_value") is not None:
        numeric = parse_numeric_value(data.get("target_value"))
        if numeric is not None:
            data["target_value"] = numeric
    return data


def _normalize_factor(item: Any) -> dict[str, Any]:
    data = dict(item or {})
    for key in ("min_value", "max_value", "step"):
        if data.get(key) is not None:
            numeric = parse_numeric_value(data.get(key))
            if numeric is not None:
                data[key] = numeric
    return data


def _hydrate_factor_domains(factors: list[FactorSpec], frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    for factor in factors:
        if factor.name not in frame.columns:
            continue
        series = frame[factor.name]
        present = series[~series.map(is_missing_value)]
        if factor.kind == "categorical":
            if not factor.categories:
                factor.categories = [str(value) for value in present.astype(str).drop_duplicates().head(32).tolist()]
            continue
        numeric = coerce_numeric_series(present).dropna()
        if numeric.empty:
            continue
        if factor.min_value is None:
            factor.min_value = float(numeric.min())
        if factor.max_value is None:
            factor.max_value = float(numeric.max())
        if factor.kind == "ordinal" and factor.step is None:
            values = sorted(set(float(value) for value in numeric.tolist()))
            diffs = [right - left for left, right in zip(values, values[1:]) if right > left]
            if diffs:
                factor.step = float(np.median(diffs))


def _maybe_infer_objectives(project: ProjectSpec, frame: pd.DataFrame) -> dict[str, Any] | None:
    """When the request carries no objectives, infer them from a bounded
    sample and install the inferred specs on the project so scoring and
    suggestions run against real targets. Returns the disclosure report to
    embed as ``autoInferredObjectives``, or None when objectives were
    configured explicitly (manual configuration always wins). Raises
    InferenceError when nothing plausible exists — never a silent fallback."""
    if project.objectives:
        return None
    report = infer_objectives(frame, [factor.name for factor in project.factors])
    project.objectives = inferred_specs(report)
    return report


def _build_analysis(project: ProjectSpec, frame: pd.DataFrame) -> dict[str, Any]:
    scored, desirability_updates = score_components(frame.copy(), project.objectives) if project.objectives else (frame.copy(), {})
    if project.objectives and "FuzzyScore" not in scored.columns:
        scored["FuzzyScore"] = np.nan
    best_row = _best_row(scored)
    score_values = coerce_numeric_series(scored.get("FuzzyScore", pd.Series(dtype=float))).dropna()
    return {
        "ready": bool(len(score_values)),
        "rowCount": int(len(frame)),
        "completedCount": int(_completed_count(frame, project.objectives)),
        "scoreCount": int(len(score_values)),
        "bestRow": _json_record(best_row) if best_row is not None else None,
        "bestRecipe": _recipe(best_row, project.factors) if best_row is not None else {},
        "scoreSummary": {
            "min": _finite_or_none(score_values.min() if len(score_values) else None),
            "max": _finite_or_none(score_values.max() if len(score_values) else None),
            "mean": _finite_or_none(score_values.mean() if len(score_values) else None),
        },
        "objectiveBounds": _objective_bounds(scored, project.objectives),
        "factorBounds": _factor_bounds(scored, project.factors),
        "scoredRows": [_json_record(row) for row in scored.to_dict(orient="records")],
        "desirabilityUpdates": desirability_updates,
    }


def _completed_count(frame: pd.DataFrame, objectives: list[ObjectiveSpec]) -> int:
    if frame.empty or not objectives:
        return 0
    columns = [objective.name for objective in objectives if objective.name in frame.columns]
    if not columns:
        return 0
    completed = frame[columns].map(lambda value: not is_missing_value(value)).any(axis=1)
    return int(completed.sum())


def _best_row(scored: pd.DataFrame) -> dict[str, Any] | None:
    if scored.empty or "FuzzyScore" not in scored.columns:
        return None
    scores = coerce_numeric_series(scored["FuzzyScore"])
    if scores.dropna().empty:
        return None
    return scored.loc[scores.idxmax()].to_dict()


def _recipe(row: dict[str, Any], factors: list[FactorSpec]) -> dict[str, Any]:
    return {factor.name: _json_value(row.get(factor.name)) for factor in factors}


def _objective_bounds(frame: pd.DataFrame, objectives: list[ObjectiveSpec]) -> list[dict[str, Any]]:
    bounds = []
    for objective in objectives:
        values = coerce_numeric_series(frame.get(objective.name, pd.Series(dtype=float))).dropna()
        desirability_points = objective.desirability if isinstance(objective.desirability, list) else parse_desirability(objective.desirability)
        desirability_values = [
            parse_numeric_value(point.get("x"))
            for point in desirability_points
            if isinstance(point, dict)
        ]
        desirability_values = [value for value in desirability_values if value is not None]
        bounds.append(
            {
                "name": objective.name,
                "goal": objective.goal,
                "weight": float(objective.weight or 1.0),
                "min": _finite_or_none(values.min() if len(values) else None),
                "max": _finite_or_none(values.max() if len(values) else None),
                "best": _finite_or_none((values.min() if objective.goal == "minimize" else values.max()) if len(values) else None),
                "target": _json_value(objective.target_value),
                "target_min": _finite_or_none(objective.target_min),
                "target_max": _finite_or_none(objective.target_max),
                "desirabilityRange": {
                    "min": _finite_or_none(min(desirability_values)),
                    "max": _finite_or_none(max(desirability_values)),
                } if desirability_values else None,
            }
        )
    return bounds


def _factor_bounds(frame: pd.DataFrame, factors: list[FactorSpec]) -> list[dict[str, Any]]:
    bounds = []
    for factor in factors:
        raw_series = frame.get(factor.name, pd.Series(dtype=object))
        series = raw_series[~raw_series.map(is_missing_value)]
        numeric = coerce_numeric_series(series).dropna()
        configured_levels = [str(value) for value in factor.categories if str(value).strip()]
        levels = [] if len(numeric) else (configured_levels or [str(value) for value in series.astype(str).drop_duplicates().head(12).tolist()])
        configured_lower = _finite_or_none(factor.min_value)
        configured_upper = _finite_or_none(factor.max_value)
        observed_lower = _finite_or_none(numeric.min() if len(numeric) else None)
        observed_upper = _finite_or_none(numeric.max() if len(numeric) else None)
        coverage_ratio = None
        if factor.kind == "categorical":
            observed_levels = {str(value) for value in series.astype(str).drop_duplicates().tolist()}
            coverage_ratio = len([level for level in configured_levels if level in observed_levels]) / len(configured_levels) if configured_levels else None
        elif configured_lower is not None and configured_upper is not None and configured_upper > configured_lower and observed_lower is not None and observed_upper is not None:
            coverage_ratio = (observed_upper - observed_lower) / (configured_upper - configured_lower)
        bounds.append(
            {
                "name": factor.name,
                "kind": factor.kind,
                "lower": observed_lower,
                "upper": observed_upper,
                "configuredLower": configured_lower,
                "configuredUpper": configured_upper,
                "configuredStep": _finite_or_none(factor.step),
                "levels": levels,
                "observed": int(series.astype(str).nunique()) if len(series) else 0,
                "configuredLevels": configured_levels,
                "coverageRatio": _finite_or_none(max(0.0, min(1.0, coverage_ratio)) if coverage_ratio is not None else None),
            }
        )
    return bounds


def _json_record(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _json_value(value) for key, value in dict(row or {}).items()}


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None
