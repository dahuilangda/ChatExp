"""Campaign strategist: per-batch LLM management of the BO objective state.

Role separation (the human-machine contract):
  * LLM strategist — reads per-batch diagnostics (factor effects, fit summary,
    incumbent/uncertainty, score history) and emits a STRUCTURED directive:
    ``{phase: explore|exploit|stop, weight_adjustments, stop_recommendation,
    rationale}``. This is the "desire" of the campaign: what to want now, and
    when wanting more is no longer worth an experiment.
  * BO engine — owns the math (which exact points to run next).
  * Operator — owns approval. Design generation already stages a confirmation;
    the strategist directive rides inside that confirmation, and a stop
    recommendation is ALWAYS advisory text surfaced to the operator, never a
    silent halt.

Safety envelope (audited design choices):
  * Weight adjustments are BOUNDED: each objective's effective weight is
    clamped to ``[0.5x, 2x]`` of the operator-configured weight and to
    ``[WEIGHT_FLOOR, WEIGHT_CEILING]`` absolutely. The strategist cannot
    silently rewrite the objective; the operator's contract dominates.
  * The directive is validated and clamped deterministically before use; any
    out-of-contract emission is recorded as a warning, not executed.
  * No fallback: the strategist is LLM-only. When the LLM is unavailable or
    its output cannot be validated, the campaign runs without a
    strategy this batch — a missing strategy is surfaced, never substituted.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

from .event_log import append_event
from .llm import _bundle_system_messages, _extract_plan_json, _ChatCompletionClient
from .models import ObjectiveSpec, ProjectState


@contextmanager
def preserve_torch_rng():
    """Save/restore the global torch RNG state around auxiliary model fits
    (e.g., bo_diagnostics for the strategist payload). Without this, a
    diagnostics refit consumes global entropy and silently shifts the NEXT
    batch's Monte Carlo sampling -- the campaign trajectory becomes dependent
    on whether a strategist read the data, which breaks reproducibility and
    contaminates A/B comparisons."""
    try:
        import torch

        state = torch.get_rng_state()
    except ImportError:
        yield
        return
    try:
        yield
    finally:
        torch.set_rng_state(state)

logger = logging.getLogger(__name__)

WEIGHT_FLOOR = 0.05
WEIGHT_CEILING = 5.0
# Bounded-adjustment envelope relative to the operator-configured weight:
# the strategist may re-balance emphasis but never rewrite the contract.
RELATIVE_WEIGHT_FLOOR = 0.5
RELATIVE_WEIGHT_CEILING = 2.0
class StrategyDirective(BaseModel):
    """Validated per-batch objective-state directive."""

    phase: Literal["explore", "exploit", "stop"] = "exploit"
    weight_adjustments: dict[str, float] = Field(default_factory=dict)
    stop_recommendation: bool = False
    rationale: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: Literal["llm"] = "llm"
    warnings: list[str] = Field(default_factory=list)
    # LLM-proposed soft region constraints: at most 2 boxes over known
    # factors, real units, weight 0-1. Bounded at parse time and clamped at
    # apply time; the engine treats them as acquisition penalties, never as
    # truncations.
    soft_constraints: list[dict] = Field(default_factory=list)
    # Guarded LLM candidate injection (LLINBO-style): up to three concrete
    # candidate points the strategist believes are promising. The engine
    # admits them only through the acquisition "justify" gate, so a wrong
    # proposal costs at most one displaced engine candidate.
    candidate_proposals: list[dict] = Field(default_factory=list)
    # Prior hints (LGBO/ChemBOMAS-style pseudo-observations): factor point +
    # per-objective expected desirabilities, injected once into the surrogate
    # training frame with damping. Requires task semantics to be useful.
    prior_hint: list[dict] = Field(default_factory=list)
    # LLM-proposed factor-range extensions (boundary-pin response): at most 2
    # factors, {name: [new_min, new_max]} in real units. Clamped so each new
    # bound may EXTEND the operator-declared bound by at most 2x the current
    # span and never contracts it. Applied only with operator confirmation.
    range_extensions: dict[str, list[float]] = Field(default_factory=dict)

    @field_validator("candidate_proposals", "prior_hint", mode="before")
    @classmethod
    def _clean_point_lists(cls, value: Any) -> list[dict]:
        """Keep only dict entries; cap the list length (3 proposals / 2 hints)."""
        if not isinstance(value, list):
            return []
        # downstream engines re-validate and further cap (3 proposals / 2 hints)
        cleaned = [entry for entry in value if isinstance(entry, dict)]
        return cleaned[:3]

    @field_validator("range_extensions", mode="before")
    @classmethod
    def _clean_extensions(cls, value: Any) -> dict[str, list[float]]:
        # clean ALL entries first, then cap at 2 (invalid entries must not
        # consume slots) -- same principle as soft_constraints.
        if not isinstance(value, dict):
            return {}
        clean: dict[str, list[float]] = {}
        for name, box in value.items():
            if isinstance(box, (list, tuple)) and len(box) == 2:
                try:
                    lo, hi = float(box[0]), float(box[1])
                    if lo == lo and hi == hi and abs(lo) != float("inf") and abs(hi) != float("inf"):
                        clean[str(name)] = [min(lo, hi), max(lo, hi)]
                except (TypeError, ValueError):
                    continue
        return dict(list(clean.items())[:2])

    @field_validator("soft_constraints", mode="before")
    @classmethod
    def _clamp_constraints(cls, value: Any) -> list[dict]:
        # clean ALL entries first, then cap at 2 -- invalid entries must not
        # consume slots that valid constraints could use.
        if not isinstance(value, list):
            return []
        clean: list[dict] = []
        for constraint in value:
            if not isinstance(constraint, dict):
                continue
            boxes = constraint.get("factors") or {}
            cleaned_boxes: dict[str, list[float]] = {}
            for name, box in (boxes or {}).items():
                if isinstance(box, (list, tuple)) and len(box) == 2:
                    try:
                        lo, hi = float(box[0]), float(box[1])
                        if lo == lo and hi == hi:
                            cleaned_boxes[str(name)] = [min(lo, hi), max(lo, hi)]
                    except (TypeError, ValueError):
                        continue
            if cleaned_boxes:
                try:
                    weight = min(max(float(constraint.get("weight", 0.5) or 0.5), 0.0), 1.0)
                except (TypeError, ValueError):
                    weight = 0.5
                if weight != weight or weight in (float("inf"), float("-inf")):
                    # NaN survives min/max clamping and would poison the whole
                    # selection score; a non-finite weight falls back to 0.5.
                    weight = 0.5
                clean.append({"factors": cleaned_boxes, "weight": weight})
        return clean[:2]

    @field_validator("weight_adjustments", mode="before")
    @classmethod
    def _drop_non_finite(cls, value: Any) -> dict[str, float]:
        if not isinstance(value, dict):
            return {}
        clean: dict[str, float] = {}
        for name, weight in value.items():
            try:
                numeric = float(weight)
            except (TypeError, ValueError):
                continue
            if numeric != numeric or numeric in (float("inf"), float("-inf")):
                continue
            clean[str(name)] = numeric
        return clean


# ═════════════════════════════════════════════════════════════════
# Diagnostics → compact strategist payload
# ═════════════════════════════════════════════════════════════════


def build_strategist_payload(
    project: ProjectState,
    experiments_df: "Any",
    diagnostics: dict[str, Any] | None = None,
    score_history: list[float] | None = None,
    past_directives: list[dict] | None = None,
) -> dict[str, Any]:
    """Compose the per-batch evidence the strategist conditions on.

    Bounded measured joint rows, response quantiles and approved profiles
    leave this path. Evidence is descriptive and is not scale-free.
    """
    objective_state = []
    frame = experiments_df
    if frame is not None and "status" in getattr(frame, "columns", []):
        frame = frame[frame["status"].astype(str).str.strip().str.lower() == "completed"]
    if frame is not None:
        from .campaign_evidence import fixed_utility_scores
        fixed_scores = fixed_utility_scores(project, frame)
        if fixed_scores is not None:
            frame = frame.copy()
            frame["FuzzyScore"] = fixed_scores
    for objective in project.objectives:
        entry: dict[str, Any] = {
            "name": objective.name,
            "goal": objective.goal,
            "operator_weight": float(objective.weight or 1.0),
        }
        if frame is not None and objective.name in getattr(frame, "columns", []):
            values = (
                frame[objective.name]
                .map(_to_number)
                .dropna()
            )
            if len(values) >= 2:
                entry["observed_min"] = float(values.min())
                entry["observed_max"] = float(values.max())
                entry["observed_span"] = float(values.max() - values.min())
                entry["n_observed"] = int(len(values))
        # Where the composite is bottlenecked: the weighted geometric mean is
        # influenced by weighted negative-log desirabilities. Reporting
        # each objective's desirability at the incumbent row (and its floor
        # mass) tells the strategist whether re-weighting can help at all --
        # a floored bottleneck is a feasibility problem (probe it), a
        # mid-range bottleneck with span to recover may deserve weight.
        if frame is not None and score_history:
            try:
                from .desirability import desirability_score, parse_desirability_text

                profile = parse_desirability_text(objective.desirability)
                if profile:
                    sub = frame.dropna(subset=[objective.name])
                    if len(sub) and "FuzzyScore" in sub.columns:
                        incumbent_row = sub.loc[pd.to_numeric(sub["FuzzyScore"], errors="coerce").idxmax()]
                        incumbent_value = _to_number(incumbent_row.get(objective.name))
                        if incumbent_value is not None:
                            entry["desirability_at_incumbent"] = round(
                                float(desirability_score(incumbent_value, profile, 1.0)), 4)
                        d_values = [float(desirability_score(v, profile, 1.0))
                                    for v in sub[objective.name].map(_to_number).dropna()]
                        if d_values:
                            entry["desirability_mean"] = round(float(np.mean(d_values)), 4)
                            entry["at_floor_fraction"] = round(
                                float(sum(d <= 0.02 + 1e-9 for d in d_values) / len(d_values)), 4)
            except Exception:  # diagnostics are evidence, never a gate
                pass
        objective_state.append(entry)

    # Geometry of the evaluated set in the declared design space: on
    # floor-dominated surfaces the interior can stay unsampled while every
    # observation sits near the boundary. min_distance_to_center is the
    # cheapest indicator that a center-region hypothesis is untested.
    geometry: dict[str, Any] = {}
    if frame is not None and len(frame):
        numeric_factors = [
            f for f in project.factors
            if f.kind != "categorical" and f.min_value is not None and f.max_value is not None
            and f.name in getattr(frame, "columns", [])
        ]
        if numeric_factors:
            try:
                dists = []
                for _, row in frame.iterrows():
                    units = []
                    for f in numeric_factors:
                        v = _to_number(row.get(f.name))
                        if v is None:
                            units = []
                            break
                        span = max(float(f.max_value) - float(f.min_value), 1e-9)
                        units.append((v - float(f.min_value)) / span)
                    if units:
                        dists.append(float(np.linalg.norm([u - 0.5 for u in units])))
                if dists:
                    geometry = {
                        "n_rows": len(dists),
                        "min_distance_to_center": round(min(dists), 4),
                        "mean_distance_to_center": round(float(np.mean(dists)), 4),
                        "note": "unit-space L2 distance of evaluated points from the center of the declared design space",
                    }
            except Exception:
                geometry = {}

    factor_effects = []
    for effect in (diagnostics or {}).get("factor_effects") or []:
        if isinstance(effect, dict) and effect.get("name"):
            factor_effects.append(
                {
                    "name": effect["name"],
                    "importance": effect.get("importance"),
                }
            )

    best_score = None
    if score_history:
        best_score = max(score_history)

    # Data-interpretation indicators: campaign pathology the directive should
    # respond to (or explicitly not, when the engine is healthy).
    floor_mass = None
    if frame is not None and "FuzzyScore" in getattr(frame, "columns", []):
        scores = pd.to_numeric(frame["FuzzyScore"], errors="coerce").dropna()
        if len(scores) >= 3:
            floor_mass = float((scores < 0.15).mean())
    trajectory_slope = None
    if score_history and len(score_history) >= 3:
        recent = score_history[-6:]
        trajectory_slope = float(recent[-1] - recent[0]) / max(len(recent) - 1, 1)

    boundary_pins = _boundary_pin_state(project, frame)
    factor_semantics = [
        {
            "name": factor.name,
            "unit": factor.unit or "",
            "kind": factor.kind,
            "domain": (
                [factor.min_value, factor.max_value]
                if factor.kind != "categorical"
                else list(factor.categories)
            ),
        }
        for factor in project.factors
        if factor.name
    ]

    from .campaign_evidence import build_campaign_evidence
    data_evidence = build_campaign_evidence(project, frame)
    fixed = data_evidence["fixed_utility"]
    approved = data_evidence["objectives"] and all(
        entry["approved_desirability_profile"] for entry in data_evidence["objectives"])
    if approved:
        # An incomplete row can have a partial legacy FuzzyScore. It must not
        # become the strategist's incumbent or make a stalled campaign look
        # healthy. Use the common complete-case incumbent for every objective.
        best_score = fixed.get("best_observed_score")
        for entry in objective_state:
            entry.pop("desirability_at_incumbent", None)
            if entry["name"] in fixed.get("incumbent_desirabilities", {}):
                entry["desirability_at_incumbent"] = fixed["incumbent_desirabilities"][entry["name"]]

    payload = {
        "data_evidence": data_evidence,
        "n_completed": int(len(frame)) if frame is not None else 0,
        "n_factors": len(project.factors),
        "boundary_pins": boundary_pins,
        "objectives": objective_state,
        "factor_importance": factor_effects[:8],
        "factor_semantics": factor_semantics,
        "past_directives": [d for d in (past_directives or []) if isinstance(d, dict)][:3],
        "best_composite_score": best_score,
        "campaign_geometry": geometry,
        "campaign_health": {
            "floor_mass": floor_mass,
            "trajectory_slope": trajectory_slope,
            # The incumbent finding a decent region is itself evidence the
            # engine works; a high floor_mass with a decent incumbent is NOT
            # pathology (intervening there is what harms campaigns).
            "healthy": bool(best_score is not None and best_score >= 0.35),
        },
        "recent_best_trajectory": [round(float(v), 6) for v in (score_history or [])[-6:]],
        "model_ready": bool((diagnostics or {}).get("model_ready", False)),
        "surrogate_fit": {
            "top_expected_score": (diagnostics or {}).get("top_expected_score"),
            "top_uncertainty": (diagnostics or {}).get("top_uncertainty"),
        },
    }
    return payload


def _boundary_pin_state(project, frame) -> list[dict[str, Any]]:
    """For each numeric factor: does the best observation sit within 5% of a
    declared bound? This is the operator-visible signal that the optimum may
    lie OUTSIDE the declared range and the range should be extended."""
    if frame is None or getattr(frame, "empty", True):
        return []
    score_col = "FuzzyScore" if "FuzzyScore" in getattr(frame, "columns", []) else None
    pins = []
    for factor in project.factors:
        if factor.kind == "categorical" or factor.min_value is None or factor.max_value is None:
            continue
        if factor.name not in getattr(frame, "columns", []):
            continue
        sub = frame.dropna(subset=[factor.name])
        if sub.empty:
            continue
        if score_col is not None and score_col in sub.columns:
            sub = sub.dropna(subset=[score_col])
            if sub.empty:
                continue
            best = sub.loc[sub[score_col].idxmax()]
        else:
            # Fallback: rank by the first objective column present in the
            # frame (max for maximize, min for minimize) so boundary pins are
            # still detected when the harness has not computed FuzzyScore.
            ranked = None
            for objective in project.objectives:
                if objective.name in sub.columns:
                    col = pd.to_numeric(sub[objective.name], errors="coerce")
                    valid = col.dropna()
                    if len(valid) >= 2:
                        ranked = sub.loc[valid.idxmax() if objective.goal != "minimize" else valid.idxmin()]
                        break
            if ranked is None:
                ranked = sub.iloc[-1]
            best = ranked
        value = float(best[factor.name])
        lo, hi = float(factor.min_value), float(factor.max_value)
        span = max(hi - lo, 1e-9)
        if value >= hi - 0.05 * span:
            pins.append({"factor": factor.name, "pin": "max", "best_value": value,
                         "declared_range": [lo, hi]})
        elif value <= lo + 0.05 * span:
            pins.append({"factor": factor.name, "pin": "min", "best_value": value,
                         "declared_range": [lo, hi]})
    return pins[:4]


def _to_number(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and value != value):
            return None
        from .desirability import parse_numeric_value
        numeric = parse_numeric_value(value)
        return numeric if numeric is not None and np.isfinite(numeric) else None
    except (TypeError, ValueError):
        return None


# ═════════════════════════════════════════════════════════════════
# Derivation: LLM path
# ═════════════════════════════════════════════════════════════════


def derive_strategy(
    payload: dict[str, Any],
    client: _ChatCompletionClient,
    model: str,
    *,
    trace_path: str | Path | None = None,
) -> StrategyDirective | None:
    """Derive the directive from the LLM strategist. This is LLM-helps-BO:
    there is NO rule-based fallback — when the LLM is unavailable or its
    output cannot be validated, the result is None and the campaign runs
    without a strategy this batch (surfaced, never substituted)."""
    return _derive_via_llm(payload, client, model, trace_path=trace_path)


_STRATEGIST_MAX_ATTEMPTS = 2


def _derive_via_llm(payload: dict[str, Any], client: _ChatCompletionClient, model: str, *, trace_path: str | Path | None = None) -> StrategyDirective | None:
    """Ask the LLM strategist. Returns None on any failure — never substitutes
    rules for the model's judgment."""
    import json as _json
    import uuid
    import time

    call_id = uuid.uuid4().hex

    def trace(kind, **fields):
        if trace_path is None:
            return
        # Explicit benchmark traces are mandatory once requested: a write
        # failure aborts the run instead of silently losing its provenance.
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"schema": "strategist-trace-v1", "call_id": call_id, "kind": kind,
                 "timestamp_unix": time.time(), **fields}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(_json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")

    messages = _bundle_system_messages("strategist") + [
        {"role": "user", "content": _json.dumps(payload, ensure_ascii=False)}
    ]
    for attempt in range(1, _STRATEGIST_MAX_ATTEMPTS + 1):
        trace("request", attempt=attempt, model=model, temperature=0.0,
              response_format={"type": "json_object"}, messages=messages)
        started = time.perf_counter()
        try:
            completion = client.create_chat_completion_full(
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=messages,
            )
        except Exception as exc:
            trace("call_error", attempt=attempt, error_type=type(exc).__name__)
            logger.warning("strategist LLM call failed: %s", exc)
            return None
        trace("response", attempt=attempt, text=completion.text,
              finish_reason=completion.finish_reason, usage=getattr(completion, "usage", None),
              duration_seconds=time.perf_counter() - started)
        if completion.finish_reason == "length":
            messages = messages[:1] + [
                {"role": "user", "content": _json.dumps(payload, ensure_ascii=False)
                 + "\n\nRespond with the complete JSON object, more compactly."}
            ]
            continue
        data = _extract_plan_json(completion.text or "")
        if data is None:
            trace("rejected", attempt=attempt, reason="invalid_json")
            continue
        try:
            directive = StrategyDirective.model_validate(
                {
                    # Single-field slips (a missing phase, confidence outside
                    # [0,1]) must not discard valid weights/extensions with
                    # them: defaults are lenient, structure stays strict.
                    "phase": data.get("phase") or "exploit",
                    "weight_adjustments": data.get("weight_adjustments") or {},
                    "stop_recommendation": bool(data.get("stop_recommendation")),
                    "rationale": str(data.get("rationale") or "")[:500],
                    "confidence": min(max(float(data.get("confidence", 0.5) or 0.5), 0.0), 1.0),
                    "soft_constraints": data.get("soft_constraints") or [],
                    "range_extensions": data.get("range_extensions") or {},
                    "candidate_proposals": data.get("candidate_proposals") or [],
                    "prior_hint": data.get("prior_hint") or [],
                    "source": "llm",
                }
            )
        except (ValidationError, ValueError, TypeError):
            trace("rejected", attempt=attempt, reason="invalid_directive")
            continue
        trace("validated_directive", attempt=attempt, directive=directive.model_dump(mode="json"))
        return directive
    trace("exhausted", attempts=_STRATEGIST_MAX_ATTEMPTS)
    return None


# ═════════════════════════════════════════════════════════════════
# Validation / application
# ═════════════════════════════════════════════════════════════════


def validate_and_apply_weights(
    directive: StrategyDirective,
    objectives: list[ObjectiveSpec],
) -> tuple[list[ObjectiveSpec], list[str]]:
    """Clamp the directive onto the operator contract.

    Returns ``(adjusted_objectives, warnings)``. Unknown objective names are
    dropped with a warning; every adjusted weight is clamped both absolutely
    and relative to the operator-configured weight.
    """
    warnings: list[str] = list(directive.warnings)
    known = {objective.name: objective for objective in objectives}
    adjusted: list[ObjectiveSpec] = []
    for objective in objectives:
        clone = objective.model_copy()
        requested = directive.weight_adjustments.get(objective.name)
        if requested is not None:
            operator_weight = float(objective.weight or 1.0)
            envelope_lo = max(WEIGHT_FLOOR, operator_weight * RELATIVE_WEIGHT_FLOOR)
            envelope_hi = min(WEIGHT_CEILING, operator_weight * RELATIVE_WEIGHT_CEILING)
            if envelope_lo > envelope_hi:
                # Extreme operator weights make the absolute and relative
                # envelopes disjoint; the operator's contract wins unchanged.
                warnings.append(
                    f"weight for {objective.name} left at operator value {operator_weight:.3f} "
                    "(adjustment envelope empty)"
                )
                adjusted.append(clone)
                continue
            clamped = min(max(float(requested), envelope_lo), envelope_hi)
            if abs(clamped - float(requested)) > 1e-9:
                warnings.append(
                    f"weight for {objective.name} clamped from {requested:.3f} to {clamped:.3f} "
                    f"(operator envelope [{operator_weight * RELATIVE_WEIGHT_FLOOR:.3f}, "
                    f"{operator_weight * RELATIVE_WEIGHT_CEILING:.3f}])"
                )
            clone.weight = clamped
        adjusted.append(clone)
    for name in directive.weight_adjustments:
        if name not in known:
            warnings.append(f"ignored weight adjustment for unknown objective {name!r}")
    return adjusted, warnings


def strategy_summary(directive: StrategyDirective, objectives: list[ObjectiveSpec]) -> str:
    """Human-readable summary that rides inside the design-confirmation flow.
    Every channel that shaped the batch must appear here: the operator
    approves the strategy, so nothing directive-driven may stay invisible."""
    weight_bits = []
    for objective in objectives:
        requested = directive.weight_adjustments.get(objective.name)
        if requested is not None:
            weight_bits.append(f"{objective.name}→{objective.weight:.2f}")
    bits = [f"phase={directive.phase}"]
    if weight_bits:
        bits.append("weights: " + ", ".join(weight_bits))
    if directive.soft_constraints:
        zone_bits = []
        for constraint in directive.soft_constraints[:2]:
            for name, box in (constraint.get("factors") or {}).items():
                zone_bits.append(f"{name} {box[0]:g}~{box[1]:g}")
        if zone_bits:
            bits.append("avoid zones: " + ", ".join(zone_bits[:4]))
    if directive.range_extensions:
        ext_bits = [f"{name}: {box[0]:g}~{box[1]:g}" for name, box in directive.range_extensions.items()]
        bits.append("range→ " + ", ".join(ext_bits))
    if directive.stop_recommendation:
        bits.append("STOP recommended")
    if directive.candidate_proposals:
        probe = directive.candidate_proposals[0]
        rendered = ", ".join(f"{k}={v}" for k, v in list(probe.items())[:6])
        bits.append(f"hypothesis probe ready: {rendered}")
    if directive.prior_hint:
        hint = directive.prior_hint[0]
        point = hint.get("factors") or hint
        rendered = ", ".join(f"{k}={v}" for k, v in list(point.items())[:4])
        bits.append(f"prior hint: {rendered}")
    if directive.rationale:
        bits.append(f"({directive.rationale[:160]})")
    return "[strategy] " + " | ".join(bits)


def log_strategy_decision(
    task_dir: str | Path,
    directive: StrategyDirective,
    best_so_far: float | None = None,
) -> None:
    # best_so_far anchors the outcome_delta computed for the NEXT directive's
    # past_directives feedback (realized change in best composite).
    append_event(
        task_dir,
        "strategy_decision",
        directive=directive.model_dump(mode="json"),
        phase=directive.phase,
        weights=dict(directive.weight_adjustments),
        stop=directive.stop_recommendation,
        source=directive.source,
        proposals=len(directive.candidate_proposals),
        point=dict(directive.candidate_proposals[0]) if directive.candidate_proposals else None,
        rationale=directive.rationale[:300],
        best_so_far=best_so_far,
    )


MAX_CUMULATIVE_EXTENSION_SPANS = 4.0  # global cap: original span x 4 per side


def validate_range_extensions(
    directive: StrategyDirective,
    factors: list,
    original_bounds: dict[str, tuple[float, float]] | None = None,
) -> tuple[dict[str, list[float]], list[str]]:
    """Clamp range extensions onto the operator-declared bounds.

    Rules: extension only (a new bound may never move inward); each side may
    extend by at most 2x the current span PER ROUND; cumulatively a bound may
    never exceed 4x the ORIGINAL span from the campaign's originally declared
    bounds (when provided) -- closing the geometric-growth hole; unknown
    factor names dropped with a warning; at most 2 factors per round;
    malformed factor specs (min >= max) skipped.
    """
    warnings: list[str] = list(directive.warnings)
    known = {f.name: f for f in factors}
    result: dict[str, list[float]] = {}
    for name, box in list(directive.range_extensions.items())[:2]:
        factor = known.get(str(name))
        if factor is None or getattr(factor, "kind", "") == "categorical" or factor.min_value is None:
            warnings.append(f"ignored range extension for unknown/non-numeric factor {name!r}")
            continue
        lo, hi = float(factor.min_value), float(factor.max_value)
        if hi <= lo:
            warnings.append(f"ignored range extension for malformed factor {name!r} (min >= max)")
            continue
        span = max(hi - lo, 1e-9)
        new_lo = min(max(float(box[0]), lo - 2 * span), lo)  # extend only
        new_hi = max(min(float(box[1]), hi + 2 * span), hi)  # extend only
        # Global cumulative cap against the ORIGINAL campaign bounds.
        if original_bounds and str(name) in original_bounds:
            o_lo, o_hi = original_bounds[str(name)]
            o_span = max(o_hi - o_lo, 1e-9)
            cap_lo = o_lo - MAX_CUMULATIVE_EXTENSION_SPANS * o_span
            cap_hi = o_hi + MAX_CUMULATIVE_EXTENSION_SPANS * o_span
            if new_lo < cap_lo or new_hi > cap_hi:
                new_lo = max(new_lo, cap_lo)
                new_hi = min(new_hi, cap_hi)
                if new_hi <= new_lo:
                    warnings.append(
                        f"range extension for {name} refused: cumulative cap "
                        f"[{cap_lo:g}, {cap_hi:g}] (4x original span) already reached"
                    )
                    continue
                warnings.append(
                    f"range extension for {name} capped to the cumulative limit "
                    f"[{new_lo:g}, {new_hi:g}] (4x original span)"
                )
        if abs(new_lo - float(box[0])) > 1e-9 or abs(new_hi - float(box[1])) > 1e-9:
            warnings.append(
                f"range extension for {name} clamped from {box} to [{new_lo:g}, {new_hi:g}] "
                "(extension-only, at most 2x span per side)"
            )
        if new_lo < lo or new_hi > hi:
            result[str(name)] = [new_lo, new_hi]
    return result, warnings


__all__ = [
    "validate_range_extensions",
    "RELATIVE_WEIGHT_CEILING",
    "RELATIVE_WEIGHT_FLOOR",
    "StrategyDirective",
    "build_strategist_payload",
    "derive_strategy",
    "log_strategy_decision",
    "strategy_summary",
    "validate_and_apply_weights",
]
