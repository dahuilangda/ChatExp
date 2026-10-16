"""HTTP client for the chatexp-api BO backend.

The TUI's local torch engine (``chatexp.design``) and the chatexp-api service
(``app/optimizer.py``) share the incumbent health clamp, the data-fingerprint
Monte-Carlo seed, the LLM justify gate, and the soft-constraint penalty; the
API runs a stronger runtime profile (larger candidate pool / ensemble /
training budget) and additionally applies its own trust-weight schedule,
handles target-band objectives, and gates the batch regime.
When ``CHATEXP_BO_API_ENDPOINT`` is configured, design suggestions are produced
by the API service and fall back to the local engine on any failure, mirroring
the seatable plugin's boHarness contract (202 + poll + cancel, bearer auth).
"""

from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests

from .design import DesignSuggestion
from .models import FactorSpec, ObjectiveSpec, ProjectState
from .settings import BOApiSettings

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
POLL_INITIAL_DELAY_SECONDS = 1.0
POLL_BACKOFF_FACTOR = 1.5
POLL_MAX_DELAY_SECONDS = 15.0
TERMINAL_JOB_STATES = {"succeeded", "failed", "canceled"}


class BOApiError(RuntimeError):
    """Raised when the BO API cannot complete a suggest/analyze request."""


def normalize_bo_api_endpoint(endpoint: str) -> str:
    """Mirror the plugin contract: the endpoint may be given with or without /v1."""
    normalized = (endpoint or "").strip().rstrip("/")
    if normalized and not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


class BOApiClient:
    def __init__(self, settings: BOApiSettings, session: requests.Session | None = None):
        self.settings = settings
        self.endpoint = normalize_bo_api_endpoint(settings.endpoint or "")
        self._session = session or requests.Session()

    @property
    def configured(self) -> bool:
        return bool(self.endpoint)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def _post_job(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST a BO job and return either the inline result or a queued envelope."""
        url = urljoin(self.endpoint + "/", f"bo/{kind}")
        response = self._session.post(url, headers=self._headers(), json=payload, timeout=self.settings.timeout_seconds)
        if response.status_code in RETRYABLE_STATUS:
            # One retry for transient gateway failures, then surface loudly.
            time.sleep(POLL_INITIAL_DELAY_SECONDS)
            response = self._session.post(url, headers=self._headers(), json=payload, timeout=self.settings.timeout_seconds)
        if response.status_code >= 400:
            raise BOApiError(f"BO API {kind} failed with HTTP {response.status_code}: {response.text[:300]}")
        try:
            body = response.json()
        except ValueError as exc:
            raise BOApiError(f"BO API {kind} returned non-JSON body: {response.text[:300]}") from exc
        if not isinstance(body, dict):
            raise BOApiError(f"BO API {kind} returned an unexpected body type: {type(body).__name__}")
        return body

    def _poll_job(self, job_id: str, deadline_seconds: float) -> dict[str, Any]:
        url = urljoin(self.endpoint + "/", f"jobs/{job_id}")
        deadline = time.monotonic() + max(deadline_seconds, 1.0)
        delay = POLL_INITIAL_DELAY_SECONDS
        while True:
            response = self._session.get(url, headers=self._headers(), timeout=self.settings.timeout_seconds)
            if response.status_code >= 400:
                raise BOApiError(f"BO API job poll failed with HTTP {response.status_code}: {response.text[:300]}")
            try:
                body = response.json()
            except ValueError as exc:
                raise BOApiError(f"BO API job poll returned non-JSON body: {response.text[:300]}") from exc
            status = str(body.get("status") or "")
            if status == "succeeded":
                result = body.get("result")
                if not isinstance(result, dict):
                    raise BOApiError("BO API job succeeded without a result payload.")
                return result
            if status in TERMINAL_JOB_STATES:
                raise BOApiError(f"BO API job ended with status '{status}': {str(body.get('error') or '')[:300]}")
            if time.monotonic() >= deadline:
                self.cancel_job(job_id)
                raise BOApiError(f"BO API job {job_id} timed out after {deadline_seconds:.0f}s (canceled).")
            time.sleep(delay)
            delay = min(delay * POLL_BACKOFF_FACTOR, POLL_MAX_DELAY_SECONDS)

    def cancel_job(self, job_id: str) -> None:
        try:
            url = urljoin(self.endpoint + "/", f"jobs/{job_id}/cancel")
            self._session.post(url, headers=self._headers(), timeout=max(self.settings.timeout_seconds, 30))
        except Exception:  # pragma: no cover - best-effort cancel must never mask the original error
            logger.debug("BO API cancel for job %s failed (ignored)", job_id, exc_info=True)

    def run_job(self, kind: str, payload: dict[str, Any], timeout_seconds: float | None = None) -> dict[str, Any]:
        body = self._post_job(kind, payload)
        if isinstance(body.get("result"), dict):
            return body["result"]
        job_id = body.get("job_id")
        if not job_id:
            raise BOApiError(f"BO API {kind} returned neither a result nor a job_id.")
        return self._poll_job(str(job_id), timeout_seconds or self.settings.timeout_seconds)


def build_bo_suggest_payload(
    project: ProjectState,
    experiments_df: pd.DataFrame,
    n_points: int,
    settings: BOApiSettings,
    strategy_phase: str | None = None,
    soft_constraints: list[dict] | None = None,
    llm_candidates: list[dict] | None = None,
    prior_pseudo: list[dict] | None = None,
    diagnostic_point: dict | None = None,
) -> dict[str, Any]:
    """Build the /v1/bo/suggest request body from the TUI project state."""
    factor_names = [factor.name for factor in project.factors]
    objective_names = [objective.name for objective in project.objectives]
    columns = [name for name in dict.fromkeys([*factor_names, *objective_names, "FuzzyScore"]) if name]
    rows_df = experiments_df.reindex(columns=columns) if not experiments_df.empty else pd.DataFrame(columns=columns)
    rows = _json_safe_records(rows_df)
    return {
        "project_name": project.name,
        "table": project.name,
        "rows": rows,
        "factors": [_factor_payload(factor) for factor in project.factors],
        "objectives": [_objective_payload(objective) for objective in project.objectives],
        "n_points": int(n_points),
        "runtime": {
            "candidate_pool_size": int(settings.candidate_pool_size),
            "ensemble_size": int(settings.ensemble_size),
            "training_steps": int(settings.training_steps),
            **({"strategy_phase": strategy_phase} if strategy_phase in {"explore", "exploit"} else {}),
            **({"soft_constraints": soft_constraints} if soft_constraints else {}),
            **({"llm_candidates": llm_candidates} if llm_candidates else {}),
            **({"prior_pseudo": prior_pseudo} if prior_pseudo else {}),
            **({"diagnostic_point": diagnostic_point} if diagnostic_point else {}),
        },
        "language": "zh",
    }


def _factor_payload(factor: FactorSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": factor.name, "kind": factor.kind}
    if factor.unit:
        payload["unit"] = factor.unit
    if factor.kind != "categorical":
        payload["min_value"] = factor.min_value
        payload["max_value"] = factor.max_value
        if factor.step is not None:
            payload["step"] = factor.step
    if factor.categories:
        payload["categories"] = list(factor.categories)
    return payload


def _objective_payload(objective: ObjectiveSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": objective.name,
        "goal": objective.goal,
        "weight": float(objective.weight or 1.0),
    }
    if objective.unit:
        payload["unit"] = objective.unit
    if objective.target_value is not None:
        payload["target_value"] = objective.target_value
    if objective.target_min is not None:
        payload["target_min"] = objective.target_min
    if objective.target_max is not None:
        payload["target_max"] = objective.target_max
    if objective.desirability:
        payload["desirability"] = objective.desirability
    return payload


def _json_safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    records: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in record.items():
            if value is None or (isinstance(value, float) and pd.isna(value)) or (value is pd.NA):
                clean[key] = None
            elif hasattr(value, "item"):
                clean[key] = value.item()
            else:
                clean[key] = value
        records.append(clean)
    return records


def suggest_design_via_api(
    project: ProjectState,
    experiments_df: pd.DataFrame,
    n_points: int,
    starting_candidate_id: int,
    settings: BOApiSettings,
    client: BOApiClient | None = None,
    strategy_phase: str | None = None,
    soft_constraints: list[dict] | None = None,
    llm_candidates: list[dict] | None = None,
    prior_pseudo: list[dict] | None = None,
    diagnostic_point: dict | None = None,
) -> list[DesignSuggestion]:
    """Request the next design batch from chatexp-api and map to DesignSuggestion."""
    client = client or BOApiClient(settings)
    if not client.configured:
        raise BOApiError("BO API endpoint is not configured.")
    payload = build_bo_suggest_payload(
        project, experiments_df, n_points, settings,
        strategy_phase=strategy_phase, soft_constraints=soft_constraints,
        llm_candidates=llm_candidates, prior_pseudo=prior_pseudo,
        diagnostic_point=diagnostic_point,
    )
    result = client.run_job("suggest", payload)
    raw_suggestions = result.get("suggestions")
    if not isinstance(raw_suggestions, list) or not raw_suggestions:
        raise BOApiError("BO API suggest returned no suggestions.")
    batch_id = project.latest_batch_id + 1
    suggestions: list[DesignSuggestion] = []
    known_factors = {factor.name for factor in project.factors}
    for index, raw in enumerate(raw_suggestions):
        if not isinstance(raw, dict) or not isinstance(raw.get("values"), dict):
            raise BOApiError(f"BO API suggestion #{index + 1} has no values payload.")
        values = {key: value for key, value in raw["values"].items() if key in known_factors}
        missing = known_factors - set(values)
        if missing:
            raise BOApiError(f"BO API suggestion #{index + 1} is missing factor values: {sorted(missing)}")
        suggestions.append(
            DesignSuggestion(
                batch_id=batch_id,
                candidate_id=int(starting_candidate_id) + index,
                score=float(raw.get("score") or 0.0),
                values=values,
                strategy=str(raw.get("strategy") or "greedy_noisy_ei"),
            )
        )
    return suggestions


def suggest_with_fallback(
    project: ProjectState,
    experiments_df: pd.DataFrame,
    n_points: int,
    starting_candidate_id: int,
    settings: BOApiSettings,
    local_suggest,
    client: BOApiClient | None = None,
    on_fallback=None,
    strategy_phase: str | None = None,
    soft_constraints: list[dict] | None = None,
    llm_candidates: list[dict] | None = None,
    prior_pseudo: list[dict] | None = None,
    diagnostic_point: dict | None = None,
) -> tuple[list[DesignSuggestion], str]:
    """Remote-first suggest with local-engine fallback.

    Returns ``(suggestions, source)`` where source is ``"bo_api"`` or
    ``"local_engine"``. ``on_fallback`` receives the failure when the remote
    path degrades so callers can surface an audit event.
    """
    if settings.configured:
        try:
            return (
                suggest_design_via_api(
                    project, experiments_df, n_points, starting_candidate_id, settings, client,
                    strategy_phase=strategy_phase, soft_constraints=soft_constraints,
                    llm_candidates=llm_candidates, prior_pseudo=prior_pseudo,
                    diagnostic_point=diagnostic_point,
                ),
                "bo_api",
            )
        except Exception as exc:
            logger.warning("BO API suggest failed; falling back to the local engine: %s", exc)
            if on_fallback is not None:
                on_fallback(exc)
    return local_suggest(), "local_engine"


__all__ = [
    "BOApiError",
    "BOApiClient",
    "build_bo_suggest_payload",
    "normalize_bo_api_endpoint",
    "suggest_design_via_api",
    "suggest_with_fallback",
]
