from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .strategist import StrategyDirective

import pandas as pd

from .desirability import (
    FuzzyScore,
    auto_desirability_points,
    describe_desirability,
    format_desirability,
    normalize_desirability_text,
)
from .bo_client import suggest_with_fallback
from .design import (
    DesignSuggestion,
    TorchBOError,
    bo_diagnostics,
    generate_initial_design,
    suggest_next_design,
)
from .excel_data import ExcelDataHandler
from .llm import OpenAIExperimentAnalyst, OpenAIImportInterpreter, OpenAIImportMapper
from .models import FactorSpec, GoalType, ObjectiveSpec, ProjectState
from .settings import Language
from .storage import (
    SHEET_DESIGNS,
    SHEET_EXPERIMENTS,
    append_rows,
    concat_rows,
    create_or_update_workbook,
    ensure_workspace,
    export_project_json,
    load_project_from_workbook,
    read_sheet,
    write_sheet,
)


# Windows reserves these base names with any extension (CON.xlsx is device
# CON); they cannot be stored portably and are rejected up front.
WINDOWS_RESERVED_STEM_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)
# ext4/APFS NAME_MAX is 255 bytes; longer names fail deep inside openpyxl, so
# reject them with a clear message instead.
WORKBOOK_NAME_MAX_BYTES = 255


def validate_workbook_name(workbook_name: str) -> str:
    """Reject workbook names that could escape the workspace directory or that
    no supported platform can store reliably.

    Rejected: path separators, traversal, absolute paths, drive letters, NUL
    and other control characters, Windows reserved device names (with any
    extension), trailing dots/spaces, and names over 255 UTF-8 bytes.
    Fullwidth separators (e.g. "／") are NOT path separators on any target
    platform and are accepted as ordinary filename characters.
    """
    name = str(workbook_name).strip()
    if not name:
        raise ValueError("工作簿名不能为空。")
    if (
        Path(name).is_absolute()
        or re.match(r"^[A-Za-z]:", name)
        or "/" in name
        or "\\" in name
        or ".." in name
    ):
        raise ValueError(
            f"非法工作簿名：{workbook_name}。只能使用当前工作区内不含路径分隔符的文件名。"
        )
    if "\x00" in name or any(ord(char) < 32 for char in name):
        raise ValueError(f"非法工作簿名：{workbook_name}。文件名不能包含控制字符。")
    stem = name[:-5] if name.lower().endswith(".xlsx") else name
    if stem.upper() in WINDOWS_RESERVED_STEM_NAMES:
        raise ValueError(
            f"非法工作簿名：{workbook_name}。Windows 保留设备名（CON/NUL/COM1 等）不能用作文件名。"
        )
    if name.endswith(".") or name.endswith(" "):
        raise ValueError(f"非法工作簿名：{workbook_name}。文件名不能以点号或空格结尾。")
    if len(name.encode("utf-8")) > WORKBOOK_NAME_MAX_BYTES:
        raise ValueError(
            f"非法工作簿名：{workbook_name[:32]}...。文件名超过 {WORKBOOK_NAME_MAX_BYTES} 字节上限。"
        )
    return name


logger = logging.getLogger(__name__)


class ChatExpService:
    def __init__(self, workspace: str | Path):
        self.workspace = ensure_workspace(workspace)
        self.project_root = Path(__file__).resolve().parents[1]
        # Audit events go to the coordinator's task dir when one is attached;
        # standalone service use logs at the workspace root.
        self.event_dir: Path = self.workspace
        self.data_handler = ExcelDataHandler()
        self.import_interpreter = OpenAIImportInterpreter(project_root=self.project_root)
        self.import_mapper = OpenAIImportMapper(project_root=self.project_root)
        self.analyst = OpenAIExperimentAnalyst(project_root=self.project_root)

    def create_project(
        self,
        workbook_name: str,
        factors: list[FactorSpec],
        objectives: list[ObjectiveSpec],
        initial_batch_size: int = 8,
    ) -> ProjectState:
        """Create a new project workbook in the workspace and return its state."""
        workbook_name = validate_workbook_name(workbook_name)
        workbook_name = workbook_name if workbook_name.endswith(".xlsx") else f"{workbook_name}.xlsx"
        workbook_path = self.workspace / workbook_name
        project = ProjectState(
            name=Path(workbook_name).stem,
            workspace_dir=str(self.workspace),
            workbook_path=str(workbook_path),
            factors=factors,
            objectives=objectives,
            initial_batch_size=initial_batch_size,
        )
        create_or_update_workbook(project)
        return project

    def load_project(self, workbook_name_or_path: str) -> ProjectState:
        """Load a project state from a workbook in the workspace (the .xlsx
        suffix is optional)."""
        workbook_name = validate_workbook_name(workbook_name_or_path)
        workbook = self.workspace / workbook_name
        if workbook.suffix.lower() != ".xlsx":
            workbook = workbook.with_suffix(".xlsx")
        return load_project_from_workbook(workbook)

    def update_objective(
        self,
        project: ProjectState,
        objective_name: str,
        goal: GoalType | None = None,
        target_value: float | str | None = None,
        weight: float | None = None,
        desirability: str | None = None,
    ) -> ProjectState:
        """Update an existing objective in place (only the fields given) or
        append it when the name is unknown; persists to the workbook."""
        matched = False
        for objective in project.objectives:
            if objective.name == objective_name:
                if goal is not None:
                    objective.goal = goal
                if target_value is not None:
                    objective.target_value = target_value
                if weight is not None:
                    objective.weight = weight
                if desirability is not None:
                    objective.desirability = desirability
                matched = True
        if not matched:
            project.objectives.append(
                ObjectiveSpec(
                    name=objective_name,
                    goal=goal or "maximize",
                    target_value=target_value,
                    weight=weight or 1.0,
                    desirability=desirability,
                )
            )
        create_or_update_workbook(project)
        return project

    def update_factor(
        self,
        project: ProjectState,
        factor_name: str,
        kind: str | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
        step: float | None = None,
        categories: list[str] | None = None,
        unit: str | None = None,
    ) -> ProjectState:
        """Update an existing factor in place (only the fields given) or
        append it when the name is unknown; persists and realigns the
        experiment schema."""
        matched = False
        for factor in project.factors:
            if factor.name == factor_name:
                matched = True
                if kind is not None:
                    factor.kind = kind
                if min_value is not None:
                    factor.min_value = min_value
                if max_value is not None:
                    factor.max_value = max_value
                if step is not None:
                    factor.step = step
                if categories is not None:
                    factor.categories = categories
                if unit is not None:
                    factor.unit = unit
        if not matched:
            project.factors.append(
                FactorSpec(
                    name=factor_name,
                    kind=kind or "continuous",
                    min_value=min_value,
                    max_value=max_value,
                    step=step,
                    categories=categories or [],
                    unit=unit,
                )
            )
        create_or_update_workbook(project)
        self._refresh_experiment_schema(project)
        return project

    def initialize_design(self, project: ProjectState, n_points: int | None = None) -> tuple[ProjectState, pd.DataFrame]:
        """Generate the initial (space-filling) design batch, write it to the
        DESIGNS sheet, and stage the rows in EXPERIMENT as ``initialized``."""
        size = n_points or project.initial_batch_size
        starting_candidate_id = self._next_candidate_id(project)
        suggestions = generate_initial_design(project, size, starting_candidate_id)
        frame = self._save_design_suggestions(project, suggestions, status="initialized")
        self._append_suggestions_to_experiment(project, suggestions)
        return project, frame

    def append_results(
        self,
        project: ProjectState,
        rows: list[dict] | None = None,
        source_excel: str | None = None,
        new_objectives: list[ObjectiveSpec] | None = None,
    ) -> tuple[ProjectState, pd.DataFrame]:
        """Append (or update, when a row matches an existing experiment) result
        rows and return the refreshed experiment frame. Rows come from the
        caller, an Excel file, or an LLM import that must declare objectives."""
        if source_excel:
            source_path = Path(source_excel)
            if not source_path.is_absolute():
                source_path = self.workspace / source_excel
            source_df = pd.read_excel(source_path)
            rows = source_df.to_dict(orient="records")
        if not rows:
            raise ValueError("No experiment rows provided.")

        if new_objectives:
            existing_objectives = {objective.name for objective in project.objectives}
            changed = False
            for objective in new_objectives:
                if objective.name not in existing_objectives:
                    project.objectives.append(objective)
                    existing_objectives.add(objective.name)
                    changed = True
            if changed:
                create_or_update_workbook(project)

        self._validate_llm_declared_result_columns(project, rows)

        cleaned_rows = []
        for row in rows:
            normalized = {"source": "manual", "status": "completed"}
            normalized.update(row)
            # NOTE: do NOT default Trial here. _normalize_experiment_row (called
            # inside _upsert_experiment_rows) handles the default, and
            # _match_experiment_row inspects the original row dict to decide
            # whether the caller actually provided a Trial. Pre-filling Trial
            # here would mask that signal and turn a candidate_id-only result
            # row into a duplicate (matching by the defaulted Trial misses the
            # existing row, so the append branch creates a duplicate).
            if normalized.get("status") in (None, ""):
                normalized["status"] = "completed"
            cleaned_rows.append(normalized)
        resolved_rows = self._resolve_result_row_references(project, cleaned_rows)
        frame = self._upsert_experiment_rows(project, resolved_rows)
        # Recompute desirability over the complete dataset, not just the appended rows.
        frame = self._refresh_fuzzy_scores(project, frame, force_refresh_all=True)
        self._sync_design_status(project, resolved_rows)
        return project, frame

    def generate_next_design(
        self,
        project: ProjectState,
        n_points: int,
        strategy: "StrategyDirective | None" = None,
    ) -> tuple[ProjectState, pd.DataFrame]:
        """Propose the next batch. ``strategy`` (when given) temporarily reweights
        the objectives for THIS batch's acquisition only — the operator's
        configured weights in the workbook are never rewritten."""
        experiments_df = read_sheet(project, SHEET_EXPERIMENTS)
        experiments_df = self._refresh_fuzzy_scores(project, experiments_df)
        starting_candidate_id = self._next_candidate_id(project, experiment_df=experiments_df)
        suggestions, source = self._next_design_suggestions(
            project, experiments_df, n_points, starting_candidate_id, strategy=strategy
        )
        # Audit trail for diagnostic probes: record the staged hypothesis,
        # whether it executed, and which candidate it replaced. Without this
        # a production probe is unauditable.
        try:
            from .event_log import append_event

            staged = (
                self._pick_novel_probe(strategy.candidate_proposals, experiments_df, project.factors)
                if strategy is not None and strategy.candidate_proposals and self.campaign_stalled(experiments_df)
                else None
            )
            executed = [s for s in suggestions if getattr(s, "strategy", "") == "diagnostic_point"]
            if staged is not None or executed:
                if executed:
                    dropped = None
                elif staged is not None:
                    total_budget = len(experiments_df) + n_points
                    cap = max(1, int(0.20 * total_budget))
                    dropped = (
                        "budget-cap"
                        if self._count_past_probes() >= cap
                        else "hypothesis-rejected (dedup-or-invalid)"
                    )
                else:
                    dropped = "not-staged"
                append_event(
                    self.event_dir,
                    "diagnostic_probe",
                    staged=staged,
                    bin=self._hypothesis_bin(staged, project.factors) if isinstance(staged, dict) else None,
                    executed=bool(executed),
                    candidate_id=executed[0].candidate_id if executed else None,
                    dropped=dropped,
                )
        except Exception:  # audit events must never break generation
            pass
        frame = self._save_design_suggestions(project, suggestions, status="proposed")
        self._append_suggestions_to_experiment(project, suggestions)
        self._record_design_source(project, source, len(suggestions))
        return project, frame

    def derive_campaign_strategy(self, project: ProjectState) -> "StrategyDirective | None":
        """Derive the per-batch objective-state directive (campaign strategist).

        Gated by CHATEXP_STRATEGIST in .env: ``off`` disables the role;
        ``llm`` asks the LLM strategist. There is NO rule fallback — when the
        LLM is unconfigured or fails, this returns None and the batch runs on
        operator weights alone (surfaced in the audit log)."""
        from .settings import load_strategist_mode

        mode = load_strategist_mode(self.project_root)
        if mode != "llm":
            return None
        from .settings import load_openai_settings
        from .strategist import build_strategist_payload, derive_strategy
        from .llm import _ChatCompletionClient

        experiments_df = self._refresh_fuzzy_scores(project, read_sheet(project, SHEET_EXPERIMENTS))
        from .campaign_evidence import fixed_utility_scores
        fixed_scores = fixed_utility_scores(project, experiments_df)
        if fixed_scores is not None:
            experiments_df = experiments_df.copy()
            experiments_df['FuzzyScore'] = fixed_scores
        if 'status' in experiments_df:
            experiments_df = experiments_df[experiments_df['status'].astype(str).str.strip().str.lower() == 'completed']
        score_series = (
            pd.to_numeric(experiments_df.get("FuzzyScore"), errors="coerce").dropna()
            if experiments_df is not None and "FuzzyScore" in getattr(experiments_df, "columns", [])
            else pd.Series(dtype=float)
        )
        # Cheap, truthful surrogate-state signal: the engine's own readiness
        # rule (max(4, n_factors+1) completed rows) without a refit.
        completed = experiments_df
        if "status" in getattr(experiments_df, "columns", []):
            completed = experiments_df[experiments_df["status"].astype(str).str.lower() == "completed"]
        n_completed = int(len(completed))
        model_ready = n_completed >= max(4, len(project.factors) + 1)
        diagnostics: dict = {"model_ready": model_ready}
        if model_ready:
            try:
                # Real engine evidence for the strategist: factor importance,
                # top-candidate expectation and uncertainty (one refit per
                # batch). The torch RNG state is preserved so the diagnostics
                # refit cannot perturb the next batch's MC sampling.
                from .strategist import preserve_torch_rng

                with preserve_torch_rng():
                    diagnostics.update(bo_diagnostics(project, experiments_df) or {})
                diagnostics["model_ready"] = bool(diagnostics.get("model_ready", model_ready))
            except Exception as exc:  # diagnostics are evidence, never a gate
                logger.debug("bo_diagnostics unavailable for strategist: %s", exc)
        # Feed the strategist its own recent directives so it can avoid
        # repeating what did not work. outcome_delta is the realized change
        # in best composite since each directive round; scheduler_block
        # records a probe the engine refused (dedup/invalid or budget cap).
        past_directives: list[dict] = []
        current_best = float(score_series.max()) if len(score_series) else None
        try:
            from .event_log import read_events

            events = list(read_events(self.event_dir))
            decisions = [
                {"at": i, "event": e}
                for i, e in enumerate(events)
                if e.get("kind") == "strategy_decision"
            ]
            for pos in range(max(0, len(decisions)-3), len(decisions)):
                decision = decisions[pos]
                event = decision["event"]
                entry = {
                    "phase": event.get("phase"),
                    "weights": event.get("weights"),
                    "stop": bool(event.get("stop")),
                    "rationale": str(event.get("rationale", ""))[:160],
                }
                if isinstance(event.get("point"), dict) and event["point"]:
                    entry["proposed_point"] = {
                        str(k): v for k, v in list(event["point"].items())[:8]
                    }
                best_then = event.get("best_so_far")
                next_best = decisions[pos+1]["event"].get("best_so_far") if pos+1 < len(decisions) else current_best
                if isinstance(next_best, (int, float)) and isinstance(best_then, (int, float)):
                    entry["outcome_delta"] = round(float(next_best) - float(best_then), 6)
                next_decision_at = decisions[pos + 1]["at"] if pos + 1 < len(decisions) else len(events)
                for later in events[decision["at"] + 1 : next_decision_at]:
                    if later.get("kind") == "diagnostic_probe" and not later.get("executed"):
                        dropped = later.get("dropped")
                        if dropped in {"budget-cap", "hypothesis-rejected (dedup-or-invalid)"}:
                            entry["scheduler_block"] = dropped
                            break
                past_directives.append(entry)
        except Exception as exc:  # memory is evidence, never a gate
            logger.debug("strategist memory unavailable: %s", exc)
        payload = build_strategist_payload(
            project,
            experiments_df,
            diagnostics=diagnostics,
            score_history=score_series.cummax().tolist() if len(score_series) else None,
            past_directives=past_directives,
        )
        payload["recent_best_trajectory"] = [
            float(v) for v in score_series.cummax().tolist()[-6:]
        ]
        # build_strategist_payload owns the complete-case fixed incumbent.

        settings = load_openai_settings(self.project_root)
        if not settings.configured:
            try:
                from .event_log import append_event

                append_event(self.event_dir, "observation", fact="strategist unavailable this batch; continuing with operator weights")
            except Exception:
                logger.debug("Failed to log strategist unavailability", exc_info=True)
            return None
        client = _ChatCompletionClient(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout_seconds=settings.timeout_seconds,
            max_tokens=settings.max_tokens,
        )
        model = str(settings.model)
        directive = derive_strategy(payload, client=client, model=model,
                                    trace_path=self.event_dir / "strategist_trace.jsonl")
        if directive is None:
            # LLM-helps-BO with no substitute: a missing strategy is surfaced,
            # never replaced by rules.
            try:
                from .event_log import append_event

                append_event(self.event_dir, "observation", fact="strategist unavailable this batch; continuing with operator weights")
            except Exception:
                logger.debug("Failed to log strategist unavailability", exc_info=True)
            return None
        from .strategist import log_strategy_decision

        log_strategy_decision(self.event_dir, directive, best_so_far=current_best)
        return directive

    def strategy_adjusted_project(
        self,
        project: ProjectState,
        strategy: "StrategyDirective | None",
    ) -> tuple[ProjectState, list[str]]:
        """Return a per-batch clone of the project with strategist-adjusted
        objective weights. The caller's project object is never mutated.
        Safety-envelope interventions (clamps, unknown-name drops) are logged
        to the audit event log so they can never be silent."""
        if strategy is None:
            return project, []
        from .strategist import validate_and_apply_weights

        adjusted_objectives, warnings = validate_and_apply_weights(strategy, project.objectives)
        adjusted = project.model_copy(deep=True)
        adjusted.objectives = adjusted_objectives
        if warnings:
            try:
                from .event_log import append_event

                append_event(self.event_dir, "strategy_warnings", warnings=[w[:200] for w in warnings])
            except Exception:  # pragma: no cover - audit must never break design
                logger.debug("Failed to append strategy warnings", exc_info=True)
        return adjusted, warnings

    def _count_past_probes(self) -> int:
        """Diagnostic probes executed so far in this campaign (event log)."""
        try:
            from .event_log import read_events
            return sum(1 for e in read_events(self.event_dir)
                       if e.get("kind") == "diagnostic_probe" and e.get("executed"))
        except Exception:
            return 0

    @staticmethod
    def _hypothesis_bin(point: dict, factors) -> list:
        """Coarse unit-space bin (0.1 per factor) of a probe point — the
        hypothesis's region identity, independent of factor scales."""
        bins = []
        for factor in factors:
            raw = point.get(factor.name)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                bins.append(str(raw))
                continue
            lo = float(factor.min_value or 0.0)
            hi = float(factor.max_value or 1.0)
            span = max(hi - lo, 1e-9)
            bins.append(min(int(((value - lo) / span) * 10), 9))
        return bins

    def _tried_probe_bins(self, factors) -> set:
        """Hypothesis bins already probed (executed or staged) this campaign."""
        tried: set = set()
        try:
            from .event_log import read_events

            for event in read_events(self.event_dir):
                if event.get("kind") != "diagnostic_probe":
                    continue
                staged = event.get("staged")
                if isinstance(staged, dict) and event.get("bin"):
                    tried.add(tuple(event["bin"]))
                elif isinstance(staged, dict):
                    tried.add(tuple(self._hypothesis_bin(staged, factors)))
        except Exception:  # memory is evidence, never a gate
            pass
        return tried

    def _pick_novel_probe(self, proposals: list[dict], experiments_df, factors):
        """First proposal whose hypothesis bin is untested: neither probed
        before (event log) nor already evaluated (existing rows). The common
        case — a novel first proposal — returns it unchanged, so behavior is
        identical to the single-probe protocol except after a repeat."""
        tried = self._tried_probe_bins(factors)
        if experiments_df is not None and len(experiments_df):
            for _, row in experiments_df.iterrows():
                tried.add(tuple(self._hypothesis_bin(row.to_dict(), factors)))
        for proposal in proposals[:3]:
            if not isinstance(proposal, dict):
                continue
            if tuple(self._hypothesis_bin(proposal, factors)) not in tried:
                return proposal
        return None

    @staticmethod
    def campaign_stalled(experiments_df: pd.DataFrame) -> bool:
        """Diagnostic-agent trigger: the incumbent is below the health
        threshold AND the latest completed batch added no new best.
        Deterministic, engine-progress-only evidence."""
        if experiments_df is None or experiments_df.empty:
            return False
        if "FuzzyScore" not in experiments_df.columns:
            return False
        completed = experiments_df
        if "status" in completed.columns:
            completed = completed[completed["status"].astype(str).str.lower() == "completed"]
        scores = pd.to_numeric(completed["FuzzyScore"], errors="coerce").dropna()
        if len(scores) < 2:
            return False
        best = float(scores.max())
        if best >= 0.35:
            return False
        batch_col = "Trial" if "Trial" in completed.columns else None
        if batch_col is None:
            return abs(float(scores.iloc[-1]) - best) < 1e-9
        batch_best = completed.groupby(batch_col)["FuzzyScore"].apply(
            lambda x: pd.to_numeric(x, errors="coerce").max()
        ).dropna()
        if len(batch_best) < 2:
            return False
        overall = float(batch_best.cummax().iloc[-1])
        last_batch_best = float(batch_best.iloc[-1])
        prior_best = float(batch_best.cummax().iloc[-2])
        return last_batch_best <= prior_best + 1e-9 and overall < 0.35

    def _next_design_suggestions(
        self,
        project: ProjectState,
        experiments_df: pd.DataFrame,
        n_points: int,
        starting_candidate_id: int,
        strategy: "StrategyDirective | None" = None,
    ) -> tuple[list[DesignSuggestion], str]:
        """Remote-first BO suggestions: the chatexp-api service runs the same
        engine with the stronger runtime profile; the local torch engine is the
        fallback when the API is unconfigured or unreachable. A campaign
        strategy (when given) reweights objectives on a per-batch clone."""
        from .settings import load_bo_api_settings

        bo_settings = load_bo_api_settings(self.project_root)
        suggest_project, _ = self.strategy_adjusted_project(project, strategy)
        strategy_phase = strategy.phase if strategy is not None and strategy.phase in {"explore", "exploit"} else None
        soft_constraints = [c for c in (strategy.soft_constraints if strategy is not None else []) if c]
        llm_candidates = [c for c in (strategy.candidate_proposals if strategy is not None else []) if c]
        prior_pseudo = [p for p in (strategy.prior_hint if strategy is not None else []) if p]

        # Diagnostic-agent protocol: under stall the strategist's best NOVEL
        # hypothesis becomes a real evaluation probe; the justify-gated
        # candidate channel stays off that batch.
        diagnostic_point = None
        if strategy is not None and strategy.candidate_proposals and self.campaign_stalled(experiments_df):
            # Cumulative probe spend cap: probes may use at most 20% of the
            # evaluated budget.
            total_budget = len(experiments_df) + n_points
            probes_spent = self._count_past_probes()
            if probes_spent < max(1, int(0.20 * total_budget)):
                diagnostic_point = self._pick_novel_probe(
                    strategy.candidate_proposals, experiments_df, project.factors
                )
                llm_candidates = None

        def _local() -> list[DesignSuggestion]:
            return suggest_next_design(
                suggest_project,
                experiments_df,
                n_points=n_points,
                starting_candidate_id=starting_candidate_id,
                strategy_phase=strategy_phase,
                soft_constraints=soft_constraints or None,
                llm_candidates=llm_candidates or None,
                prior_pseudo=prior_pseudo or None,
                diagnostic_point=diagnostic_point,
            )

        return suggest_with_fallback(
            suggest_project,
            experiments_df,
            n_points,
            starting_candidate_id,
            bo_settings,
            _local,
            on_fallback=self._record_bo_api_failure,
            strategy_phase=strategy_phase,
            soft_constraints=soft_constraints or None,
            llm_candidates=llm_candidates or None,
            prior_pseudo=prior_pseudo or None,
            diagnostic_point=diagnostic_point,
        )

    def apply_range_extensions(
        self,
        project: ProjectState,
        extensions: dict[str, list[float]],
    ) -> tuple[ProjectState, list[str]]:
        """Apply operator-confirmed range extensions to the project (persisted
        to the workbook PARAMETERS sheet) and audit the change. Extensions are
        pre-clamped by chatexp.strategist.validate_range_extensions; this
        method re-validates defensively and refuses contractions."""
        if not extensions:
            return project, []
        from .strategist import StrategyDirective, validate_range_extensions

        # Track the campaign's ORIGINAL bounds (first call wins) so the
        # cumulative 4x-span cap always references the operator's initial
        # declaration, not the already-extended bounds.
        originals = getattr(self, "_original_factor_bounds", None)
        if originals is None:
            originals = {
                factor.name: (float(factor.min_value), float(factor.max_value))
                for factor in project.factors
                if factor.kind != "categorical" and factor.min_value is not None and factor.max_value is not None
            }
            self._original_factor_bounds = originals
        directive = StrategyDirective(phase="exploit", range_extensions=dict(extensions))
        clamped, warnings = validate_range_extensions(directive, project.factors, original_bounds=originals)
        if not clamped:
            return project, warnings or ["range extension rejected: no valid extension after clamping"]
        changed = False
        for factor in project.factors:
            box = clamped.get(factor.name)
            if box is None:
                continue
            factor.min_value = box[0]
            factor.max_value = box[1]
            changed = True
        if changed:
            create_or_update_workbook(project)
            try:
                from .event_log import append_event

                append_event(
                    self.event_dir,
                    "range_extended",
                    extensions={k: list(v) for k, v in clamped.items()},
                )
            except Exception:
                logger.debug("Failed to log range extension", exc_info=True)
        return project, warnings

    def _record_bo_api_failure(self, error: Exception) -> None:
        try:
            from .event_log import append_event

            append_event(
                self.event_dir,
                kind="observation",
                fact=f"BO API unavailable, local engine used ({type(error).__name__}).",
            )
        except Exception:  # pragma: no cover - audit logging must never break design
            logger.debug("Failed to append BO fallback event", exc_info=True)

    def _record_design_source(self, project: ProjectState, source: str, count: int) -> None:
        """Append which engine produced the latest batch to the audit log."""
        try:
            from .event_log import append_event

            append_event(
                self.event_dir,
                kind="design_requested",
                fact=f"batch={project.latest_batch_id} source={source} points={count}",
            )
        except Exception:  # pragma: no cover
            logger.debug("Failed to append design source event", exc_info=True)

    def update_plan_experiment(
        self,
        project: ProjectState,
        *,
        experiment_id: int | None = None,
        trial: int | None = None,
        candidate_id: int | None = None,
        updates: dict[str, object] | None = None,
        comment: str | None = None,
    ) -> ProjectState:
        """Edit planned factor values and/or the COMMENT of experiment and
        design rows selected by experiment_id or (Trial, candidate_id)."""
        updates = dict(updates or {})
        design_df = read_sheet(project, SHEET_DESIGNS)
        experiment_df = self._align_experiment_columns(project, read_sheet(project, SHEET_EXPERIMENTS).copy())
        design_mask = self._plan_row_mask(design_df, experiment_id=experiment_id, trial=trial, candidate_id=candidate_id)
        experiment_mask = self._plan_row_mask(experiment_df, experiment_id=experiment_id, trial=trial, candidate_id=candidate_id)
        if not design_mask.any() and not experiment_mask.any():
            raise ValueError("没有找到要修改的实验计划。")
        allowed_factor_names = {factor.name for factor in project.factors}
        cleaned_updates = {key: value for key, value in updates.items() if key in allowed_factor_names and value not in (None, "")}
        if not cleaned_updates and not comment:
            return project
        if design_mask.any():
            for key, value in cleaned_updates.items():
                if key in design_df.columns:
                    design_df = self._prepare_column_for_assignment(project, design_df, key, value)
                    design_df.loc[design_mask, key] = value
            write_sheet(project, SHEET_DESIGNS, design_df)
        if experiment_mask.any():
            for key, value in cleaned_updates.items():
                if key in experiment_df.columns:
                    experiment_df = self._prepare_column_for_assignment(project, experiment_df, key, value)
                    experiment_df.loc[experiment_mask, key] = value
            if comment:
                experiment_df = self._prepare_column_for_assignment(project, experiment_df, "COMMENT", comment)
                experiment_df.loc[experiment_mask, "COMMENT"] = comment
            write_sheet(project, SHEET_EXPERIMENTS, experiment_df)
        return project

    def refresh_scores(self, project: ProjectState) -> pd.DataFrame:
        """Recompute FuzzyScore for all experiments and return the scored frame."""
        experiments = read_sheet(project, SHEET_EXPERIMENTS)
        return self._refresh_fuzzy_scores(project, experiments)

    def analyze_weight_distribution(self, project: ProjectState, language: Language = Language.ZH) -> dict[str, object]:
        """Analyze the adaptive weight distribution for multi-objective scoring.

        Returns an analysis of how weights are distributed across objectives
        and recommendations for optimization.
        """
        from .desirability import analyze_weight_distribution, score_components

        experiments = read_sheet(project, SHEET_EXPERIMENTS)
        if experiments.empty:
            return {
                "weights": {},
                "analysis": "No experiment data yet." if language == Language.EN else "暂无实验数据。",
                "recommendations": []
            }

        scored, _ = score_components(experiments, project.objectives)
        return analyze_weight_distribution(scored, project.objectives, language)

    def llm_analyze_and_optimize_weights(
        self,
        project: ProjectState,
        language: Language = Language.ZH
    ) -> dict[str, object]:
        """Use LLM to analyze experiment data and recommend intelligent weight adjustments.

        This method leverages the LLM's reasoning capabilities to:
        1. Analyze the optimization status of each objective
        2. Recommend adaptive weight adjustments
        3. Identify trade-offs between objectives
        4. Suggest optimization focus for next round
        """
        if len(project.objectives) <= 1:
            return {
                "analysis": "Single objective optimization - no weight adjustment needed." if language == Language.EN else "单目标优化，无需权重调整。",
                "weight_adjustments": [],
                "optimization_focus": "Focus on the single objective." if language == Language.EN else "专注单一目标优化。",
                "scoring_insights": ""
            }

        experiments = read_sheet(project, SHEET_EXPERIMENTS)
        if experiments.empty:
            return {
                "analysis": "No experiment data yet for LLM analysis." if language == Language.EN else "暂无实验数据可供LLM分析。",
                "weight_adjustments": [],
                "optimization_focus": "Start by collecting initial experimental data." if language == Language.EN else "先收集初始实验数据。",
                "scoring_insights": ""
            }

        weight_analysis = self.analyze_weight_distribution(project, language)
        objectives_summary = []
        for obj in project.objectives:
            if obj.name in experiments.columns:
                series = pd.to_numeric(experiments[obj.name], errors="coerce").dropna()
                if not series.empty:
                    objectives_summary.append({
                        "name": obj.name,
                        "goal": obj.goal,
                        "weight": obj.weight,
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "mean": float(series.mean()),
                        "variance": float(series.var()),
                    })

        prompt_data = {
            "experiments": experiments.head(20)[[obj.name for obj in project.objectives if obj.name in experiments.columns]].to_dict("records"),
            "objectives": [obj.model_dump() for obj in project.objectives],
            "weight_analysis": weight_analysis,
            "language": "zh" if language == Language.ZH else "en",
            "objectives_summary": objectives_summary,
        }

        prompt_dir = Path(__file__).resolve().with_name("prompt_assets")
        optimizer_prompt = (prompt_dir / "weight_optimizer.txt").read_text(encoding="utf-8")

        from .llm import _extract_json

        errors: list[str] = []
        retry_note = ""
        for attempt in range(1, 4):
            text = self.import_interpreter._client.create_chat_completion(
                model=str(self.import_interpreter.settings.model),
                temperature=0.3,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": optimizer_prompt},
                    {"role": "user", "content": str(prompt_data) + retry_note + "\n请只输出JSON。"}
                ],
            )
            data = _extract_json(text)
            if data and isinstance(data, dict):
                llm_analysis = data.get("analysis", "")
                llm_adjustments = data.get("weight_adjustments", [])
                llm_focus = data.get("optimization_focus", "")
                if llm_analysis and llm_adjustments and llm_focus:
                    return {
                        "analysis": llm_analysis,
                        "weight_adjustments": llm_adjustments,
                        "optimization_focus": llm_focus,
                        "scoring_insights": data.get("scoring_insights", "")
                    }
            errors.append(f"attempt {attempt}: invalid optimizer JSON")
            retry_note = (
                "\n\n上一次输出没有包含完整的 analysis、weight_adjustments、optimization_focus。"
                "请重新输出完整 JSON，不要省略任何字段。"
            )

        raise RuntimeError("LLM optimizer failed after 3 attempts: " + " | ".join(errors))

    def recommend_score_profile(self, project: ProjectState, language: Language = Language.ZH) -> dict[str, object]:
        """Recommend a desirability profile per objective (plus an optional
        LLM weight recommendation); nothing is applied until
        apply_score_profile is called with the returned dict."""
        experiments = read_sheet(project, SHEET_EXPERIMENTS)
        items: list[dict[str, object]] = []
        changed = False

        # Weight distribution analysis only applies to multi-objective projects.
        weight_analysis = None
        if len(project.objectives) > 1:
            weight_analysis = self.analyze_weight_distribution(project, language)

        for objective in project.objectives:
            series = experiments[objective.name] if objective.name in experiments.columns else pd.Series(dtype=float)
            points = auto_desirability_points(series, objective)
            if not points:
                items.append(
                    {
                        "name": objective.name,
                        "goal": objective.goal,
                        "recommended": "",
                        "changed": False,
                        "summary": "Not enough results yet, using existing scoring logic." if language == Language.EN else "当前没有足够结果，先沿用现有评分逻辑。",
                    }
                )
                continue
            recommended = format_desirability(points)
            current = normalize_desirability_text(objective.desirability)
            suggested = normalize_desirability_text(recommended)
            item_changed = current != suggested
            changed = changed or item_changed

            adaptive_weight = None
            if weight_analysis and "weights" in weight_analysis:
                adaptive_weight = weight_analysis["weights"].get(objective.name)

            items.append(
                {
                    "name": objective.name,
                    "goal": objective.goal,
                    "recommended": recommended,
                    "changed": item_changed,
                    "summary": describe_desirability(points, objective, language),
                    "adaptive_weight": adaptive_weight,
                }
            )
        llm_weight_recommendation: dict[str, object] | None = None
        llm_weight_error = ""
        if len(project.objectives) > 1 and self.import_interpreter.settings.configured:
            try:
                llm_weight_recommendation = self.llm_analyze_and_optimize_weights(project, language)
            except Exception as exc:
                logger.exception("LLM weight analysis failed")
                llm_weight_recommendation = None
                llm_weight_error = f"权重分析失败: {exc}"
        return {
            "changed": changed,
            "items": items,
            "primary": items[0] if items else None,
            "weight_analysis": weight_analysis,
            "llm_weight_recommendation": llm_weight_recommendation,
            "llm_weight_error": llm_weight_error,
        }

    def apply_score_profile(self, project: ProjectState, recommendation: dict[str, object] | None) -> tuple[ProjectState, int]:
        """Apply a recommend_score_profile result: set recommended
        desirability texts and clamp LLM weight adjustments to
        [0.5x, 2x] of the operator's weight. Returns the project and the
        number of changed items."""
        if not recommendation:
            return project, 0
        changed_items = 0
        item_map = {
            str(item.get("name") or ""): item
            for item in recommendation.get("items") or []
            if isinstance(item, dict)
        }
        for objective in project.objectives:
            item = item_map.get(objective.name)
            if not item or not item.get("changed"):
                continue
            recommended = str(item.get("recommended") or "").strip()
            if not recommended:
                continue
            objective.desirability = recommended
            changed_items += 1
        weight_recommendation = recommendation.get("llm_weight_recommendation")
        if isinstance(weight_recommendation, dict):
            adjustments = weight_recommendation.get("weight_adjustments") or []
            adjustment_map = {
                str(item.get("objective") or ""): item
                for item in adjustments
                if isinstance(item, dict)
            }
            for objective in project.objectives:
                item = adjustment_map.get(objective.name)
                if not item:
                    continue
                try:
                    recommended_weight = float(item.get("recommended_weight"))
                except (TypeError, ValueError):
                    continue
                operator_w = float(objective.weight or 1.0)
                recommended_weight = max(
                    max(0.05, 0.5 * operator_w),
                    min(min(5.0, 2.0 * operator_w), recommended_weight)
                )
                if abs(float(objective.weight or 1.0) - recommended_weight) > 1e-9:
                    objective.weight = recommended_weight
                    changed_items += 1
        if changed_items:
            create_or_update_workbook(project)
            self._refresh_fuzzy_scores(project)
        return project, changed_items

    def inspect_workbook(self, project: ProjectState) -> dict:
        """Return workbook diagnostics: factors, objectives, row counts and
        column lists of the EXPERIMENT and DESIGNS sheets."""
        experiment_df = read_sheet(project, SHEET_EXPERIMENTS)
        design_df = read_sheet(project, SHEET_DESIGNS)
        return {
            "project": project.name,
            "workbook": str(project.workbook),
            "factors": [factor.model_dump() for factor in project.factors],
            "objectives": [objective.model_dump() for objective in project.objectives],
            "experiment_rows": int(len(experiment_df)),
            "design_rows": int(len(design_df)),
            "experiment_columns": list(experiment_df.columns),
            "design_columns": list(design_df.columns),
        }

    def analyze_project(self, project: ProjectState, *, include_bo_diagnostics: bool = False, language: Language = Language.ZH) -> dict[str, object]:
        """Dashboard payload: counts, best experiment, batch/score/objective
        trends, factor ranges, latest design batch, and optional BO
        diagnostics (failures degrade to a model_ready=False entry)."""
        experiments_df = read_sheet(project, SHEET_EXPERIMENTS)
        designs_df = read_sheet(project, SHEET_DESIGNS)
        scored_df = experiments_df.copy()
        if not scored_df.empty and project.objectives:
            if "FuzzyScore" not in scored_df.columns or pd.to_numeric(scored_df.get("FuzzyScore"), errors="coerce").notna().sum() == 0:
                scored_df, _ = FuzzyScore(scored_df).score_dataframe(project.objectives)

        completed_df = self._completed_experiments(project, scored_df)
        suggested_df = self._filter_status_rows(scored_df, status="suggested")
        primary_objective = project.objectives[0] if project.objectives else None
        best_row = self._best_experiment_row(completed_df, primary_objective)

        factor_ranges: list[dict[str, object]] = []
        if not completed_df.empty:
            for factor in project.factors:
                if factor.name not in completed_df.columns:
                    continue
                series = completed_df[factor.name]
                factor_ranges.append(
                    {
                        "name": factor.name,
                        "kind": factor.kind,
                        "summary": self._factor_series_summary(series, factor.kind),
                    }
                )

        batch_counts = self._batch_counts(completed_df)
        score_trend = self._score_trend(completed_df)
        objective_trend = self._objective_trend(completed_df, primary_objective)
        latest_design_batch_id = self._latest_design_batch_id(designs_df)
        latest_design_rows = self._latest_design_rows(designs_df, latest_design_batch_id)
        diagnostics: dict[str, object] | None = None
        if include_bo_diagnostics:
            try:
                diagnostics = bo_diagnostics(project, scored_df, language=language)
            except TorchBOError as exc:
                diagnostics = {
                    "model_ready": False,
                    "reason": str(exc),
                    "factor_effects": [],
                }
            except Exception as exc:
                # A ValueError/KeyError/etc. from bo_diagnostics must not
                # propagate and break the dashboard (analyze_project is called
                # from the bi-panel worker and the dashboard refresh timer).
                logger.exception("bo_diagnostics failed with non-torch error")
                diagnostics = {
                    "model_ready": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "factor_effects": [],
                }

        return {
            "project_name": project.name,
            "workbook_name": project.workbook.name,
            "design_count": int(len(designs_df)),
            "experiment_count": int(len(experiments_df)),
            "completed_count": int(len(completed_df)),
            "suggested_count": int(len(suggested_df)),
            "latest_batch_id": int(project.latest_batch_id or 0),
            "primary_objective": primary_objective.model_dump() if primary_objective is not None else None,
            "best_experiment": best_row,
            "batch_counts": batch_counts,
            "score_trend": score_trend,
            "objective_trend": objective_trend,
            "factor_ranges": factor_ranges,
            "recent_completed": completed_df.tail(6).fillna("").to_dict(orient="records") if not completed_df.empty else [],
            "latest_design_batch_id": latest_design_batch_id,
            "latest_design_rows": latest_design_rows,
            "bo_diagnostics": diagnostics,
        }

    def _factor_pair_determines(self, frame: pd.DataFrame, src: str, dst: str) -> bool:
        """True if every observed value of `src` maps to exactly one value of `dst`
        (pairwise, ignoring rows where either is missing) — i.e. `src` perfectly
        determines `dst`, so the two factors are aliased over the observed data.
        A `src` whose values are all distinct carries no aliasing information:
        it trivially "determines" every other column, so require at least one
        repeated `src` value before claiming determination — UNLESS `dst` is also
        all-distinct over the same rows AND their rank permutation matches (or
        anti-matches), which is the canonical 1:1 confounded design
        (e.g. temp=[60,70,80], catalyst=[A,B,C]). Latin-hypercube designs have
        independent orderings, so their ranks do not match and they are not
        flagged as aliased."""
        pair = frame[[src, dst]].dropna()
        if pair.shape[0] < 2:
            return False
        src_unique = bool(pair[src].is_unique)
        dst_unique = bool(pair[dst].is_unique)
        if src_unique and dst_unique:
            # Structural-aliasing test: identical (or reversed) rank permutation
            # across both columns ⇒ the two columns are 1:1 confounded.
            src_rank = pair[src].rank(method="dense").to_numpy()
            dst_rank = pair[dst].rank(method="dense").to_numpy()
            if len(src_rank) >= 2 and (
                bool((src_rank == dst_rank).all())
                or bool((src_rank == dst_rank[::-1]).all())
            ):
                return True
            return False
        if src_unique:
            return False
        return bool((pair.groupby(src)[dst].nunique() <= 1).all())

    def _confounded_factor_groups(self, project: ProjectState, completed: pd.DataFrame) -> list[dict[str, object]]:
        """Detect perfectly aliased (confounded) factor groups among completed
        rows. Two factors are confounded when one deterministically determines
        the other over the observed data, so their effects cannot be separated.
        Returns one entry per connected group (≥2 factors), with the locked
        co-occurrence pattern that makes the aliasing visible to the user."""
        import itertools

        factor_names = [f.name for f in project.factors if f.name in completed.columns]
        if len(factor_names) < 2 or completed.empty:
            return []
        frame = completed[factor_names]
        # Only factors that actually vary (>=2 distinct values) can be aliased.
        active = [c for c in factor_names if frame[c].dropna().astype(str).nunique() >= 2]
        if len(active) < 2:
            return []
        adjacency: dict[str, set[str]] = {c: set() for c in active}
        for a, b in itertools.combinations(active, 2):
            if self._factor_pair_determines(frame, a, b) or self._factor_pair_determines(frame, b, a):
                adjacency[a].add(b)
                adjacency[b].add(a)
        seen: set[str] = set()
        groups: list[dict[str, object]] = []
        for start in active:
            if start in seen:
                continue
            stack = [start]
            component: list[str] = []
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                component.append(node)
                stack.extend(adjacency[node] - seen)
            if len(component) < 2:
                continue
            pivot = min(component, key=lambda c: frame[c].dropna().astype(str).nunique())
            others = [c for c in component if c != pivot]
            pattern: list[dict[str, object]] = []
            for value, grp in frame.groupby(pivot, dropna=False):
                if pd.isna(value):
                    continue
                locked = {
                    other: sorted(str(v) for v in grp[other].dropna().astype(str).unique())
                    for other in others
                }
                pattern.append({"value": str(value), "locked": locked})
            groups.append({"factors": sorted(component), "pivot": pivot, "pattern": pattern})
        return groups

    def data_quality_report(self, project: ProjectState, language: Language = Language.ZH) -> dict[str, object]:
        """Coverage/quality audit of the experiment table. ``blockers``
        (confounded factors, missing responses) mark generation-blocking
        states; everything else is advisory ``recommendations``."""
        experiments_df = read_sheet(project, SHEET_EXPERIMENTS)
        designs_df = read_sheet(project, SHEET_DESIGNS)
        if experiments_df.empty:
            return {
                "row_count": 0,
                "completed_count": 0,
                "pending_count": 0,
                "objective_coverage": [],
                "factor_coverage": [],
                "missing_columns": [],
                "duplicate_plan_keys": 0,
                "recommendations": [
                    "先导入历史实验数据或初始化一批实验设计。"
                    if language == Language.ZH
                    else "Import historical experimental data or initialize the first design batch."
                ],
            }

        aligned = self._align_experiment_columns(project, experiments_df.copy())
        completed = self._completed_experiments(project, aligned)
        pending_status = aligned.get("status", pd.Series(dtype=object)).astype(str).str.lower()
        pending = aligned.loc[pending_status.isin({"suggested", "initialized", "proposed", "pending"})].copy()

        objective_coverage: list[dict[str, object]] = []
        for objective in project.objectives:
            if objective.name not in aligned.columns:
                objective_coverage.append({"name": objective.name, "count": 0, "missing": int(len(aligned)), "summary": "-"})
                continue
            series = pd.to_numeric(completed[objective.name], errors="coerce") if objective.name in completed.columns else pd.Series(dtype=float)
            valid = series.dropna()
            if valid.empty:
                summary = "暂无有效结果" if language == Language.ZH else "no valid results"
                objective_coverage.append({"name": objective.name, "count": 0, "missing": int(len(completed)), "summary": summary})
                continue
            best_value = valid.min() if objective.goal == "minimize" else valid.max()
            objective_coverage.append(
                {
                    "name": objective.name,
                    "goal": objective.goal,
                    "count": int(valid.shape[0]),
                    "missing": int(max(len(completed) - valid.shape[0], 0)),
                    "min": float(valid.min()),
                    "max": float(valid.max()),
                    "mean": float(valid.mean()),
                    "best": float(best_value),
                    "summary": f"{valid.min():.4g} ~ {valid.max():.4g}, mean {valid.mean():.4g}",
                }
            )

        factor_coverage: list[dict[str, object]] = []
        for factor in project.factors:
            if factor.name not in aligned.columns:
                continue
            series = completed[factor.name] if factor.name in completed.columns else pd.Series(dtype=object)
            if factor.kind == "categorical":
                counts = series.dropna().astype(str).value_counts()
                factor_coverage.append(
                    {
                        "name": factor.name,
                        "kind": factor.kind,
                        "observed_levels": counts.index[:8].tolist(),
                        "unique_count": int(counts.shape[0]),
                        "summary": ", ".join(f"{key}({value})" for key, value in counts.head(4).items()) or "-",
                    }
                )
                continue
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                factor_coverage.append({"name": factor.name, "kind": factor.kind, "summary": "-"})
                continue
            configured_span = None
            observed_span = float(numeric.max() - numeric.min())
            if factor.min_value is not None and factor.max_value is not None:
                configured_span = max(float(factor.max_value) - float(factor.min_value), 0.0)
            coverage_ratio = None
            if configured_span and configured_span > 0:
                coverage_ratio = max(0.0, min(1.0, observed_span / configured_span))
            factor_coverage.append(
                {
                    "name": factor.name,
                    "kind": factor.kind,
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                    "unique_count": int(numeric.nunique()),
                    "coverage_ratio": coverage_ratio,
                    "summary": f"{numeric.min():.4g} ~ {numeric.max():.4g}",
                }
            )

        missing_columns = []
        for column in aligned.columns:
            missing = int(aligned[column].isna().sum())
            if missing:
                missing_columns.append({"name": str(column), "missing": missing, "ratio": float(missing / max(len(aligned), 1))})
        missing_columns = sorted(missing_columns, key=lambda item: item["missing"], reverse=True)[:8]

        duplicate_plan_keys = 0
        key_columns = [column for column in ("Trial", "candidate_id") if column in aligned.columns]
        if len(key_columns) == 2:
            duplicate_plan_keys = int(aligned.duplicated(subset=key_columns, keep=False).sum())

        recommendations: list[str] = []
        objective_gaps = [
            item for item in objective_coverage
            if int(item.get("count") or 0) == 0
        ]
        if objective_gaps:
            names = "、".join(str(item.get("name") or "-") for item in objective_gaps[:3])
            recommendations.append(
                f"当前只有工艺条件，尚未形成 {names} 等目标结果；先补测这些响应，再进行模型驱动优化。"
                if language == Language.ZH
                else f"The table contains process conditions but no measured responses for {names}; measure these objectives before model-driven optimization."
            )
        min_bo_rows = max(4, len(project.factors) + 1)
        if len(completed) < min_bo_rows:
            recommendations.append(
                f"先补齐至少 {min_bo_rows} 条完成实验，再让 BO 模型接管下一批建议。"
                if language == Language.ZH
                else f"Collect at least {min_bo_rows} completed runs before relying on BO for the next batch."
            )
        if pending.shape[0] > 0:
            recommendations.append(
                f"当前还有 {pending.shape[0]} 条待做计划，优先回填结果或标记跳过。"
                if language == Language.ZH
                else f"There are {pending.shape[0]} pending plans; record results or mark them skipped first."
            )
        low_coverage = [
            item for item in factor_coverage
            if isinstance(item.get("coverage_ratio"), float) and float(item["coverage_ratio"]) < 0.35
        ]
        if low_coverage:
            names = "、".join(str(item["name"]) for item in low_coverage[:3])
            recommendations.append(
                f"{names} 的探索范围偏窄，下一批可主动覆盖边界或未探索区间。"
                if language == Language.ZH
                else f"{names} has narrow coverage; include boundary or unexplored values in the next batch."
            )
        confounded_groups = self._confounded_factor_groups(project, completed)
        if confounded_groups:
            group = confounded_groups[0]
            factors_str = "、".join(str(f) for f in group["factors"])
            pivot = str(group.get("pivot") or group["factors"][0])
            recommendations.append(
                f"因素 {factors_str} 完全混杂：每取一种 {pivot} 只对应其它因素的单值，无法独立估计各自效应；需补测解除混杂后再建模。"
                if language == Language.ZH
                else f"Factors {factors_str} are perfectly confounded: each {pivot} locks the others to a single value, so their effects cannot be separated; run supplementary tests to break the aliasing before modeling."
            )
        if not recommendations:
            recommendations.append(
                "数据结构基本完整，可以查看统计面板或继续生成下一批设计。"
                if language == Language.ZH
                else "The data structure is mostly complete; inspect the statistics panel or generate the next batch."
            )

        blockers: list[dict[str, object]] = []
        if confounded_groups:
            blockers.append({
                "kind": "confounded_factors",
                "groups": confounded_groups,
            })
        if objective_gaps:
            blockers.append({
                "kind": "missing_responses",
                "objectives": [str(item.get("name")) for item in objective_gaps],
            })
        # A low completed-run count stays a recommendation (min_bo_rows above):
        # the model can still fit on fewer rows, so it must not hard-block
        # generation the way confounded factors or missing responses do.

        return {
            "row_count": int(len(aligned)),
            "completed_count": int(len(completed)),
            "pending_count": int(len(pending)),
            "design_count": int(len(designs_df)),
            "objective_coverage": objective_coverage,
            "factor_coverage": factor_coverage,
            "missing_columns": missing_columns,
            "duplicate_plan_keys": duplicate_plan_keys,
            "confounded_factor_groups": confounded_groups,
            "blockers": blockers,
            "model_ready": len(blockers) == 0,
            "recommendations": recommendations,
        }

    def llm_data_analysis_summary(
        self,
        project: ProjectState,
        *,
        analysis: dict[str, object],
        quality: dict[str, object],
        language: Language = Language.ZH,
        scenario: str = "analysis",
    ) -> dict[str, object]:
        """Ask the experiment analyst LLM for a natural-language summary of
        the given analysis and quality payloads."""
        payload = {
            "language": "zh" if language == Language.ZH else "en",
            "scenario": scenario,
            "project": project.model_dump(),
            "analysis": analysis,
            "quality": quality,
            "skill_policy": {
                "navigation": "domain intent -> data state -> objective progress -> factor coverage -> next action",
                "no_rule_fallback": True,
                "use_llm_reasoning": True,
            },
        }
        return self.analyst.summarize(payload)

    def experiment_view(
        self,
        project: ProjectState,
        *,
        scope: str = "recent",
        sort: str = "recent",
        batch_id: int | None = None,
        page: int = 1,
        page_size: int = 8,
    ) -> dict[str, object]:
        """Return one paginated page of experiment rows for a scope
        (pending/completed/best/all) with server-side sorting; the returned
        ``page`` is clamped into [1, total_pages]."""
        experiments_df = read_sheet(project, SHEET_EXPERIMENTS)
        if experiments_df.empty:
            return {
                "scope": scope,
                "sort": sort,
                "batch_id": batch_id,
                "page": 1,
                "page_size": page_size,
                "total_rows": 0,
                "total_pages": 1,
                "rows": [],
            }
        scored_df = self._refresh_fuzzy_scores(project, experiments_df)
        frame = self._view_frame_for_scope(project, scored_df, scope)
        if batch_id is not None and "Trial" in frame.columns:
            frame = frame.loc[pd.to_numeric(frame["Trial"], errors="coerce").eq(batch_id)].copy()
        frame = self._sort_experiment_view(frame, sort=sort)
        total_rows = int(len(frame))
        total_pages = max((total_rows - 1) // max(page_size, 1) + 1, 1)
        normalized_page = min(max(page, 1), total_pages)
        start = (normalized_page - 1) * page_size
        end = start + page_size
        rows = frame.iloc[start:end].fillna("").to_dict(orient="records")
        return {
            "scope": scope,
            "sort": sort,
            "batch_id": batch_id,
            "page": normalized_page,
            "page_size": page_size,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "rows": rows,
        }

    def import_experiment_sheet(
        self,
        project: ProjectState,
        source_excel: str,
        sheet_name: str = 0,
        objective_names: list[str] | None = None,
        manual_column_map: dict[str, str] | None = None,
    ) -> tuple[ProjectState, dict[str, object]]:
        """Import an Excel sheet's rows into an existing project. The LLM
        supplies only the schema (factors, objectives, column mapping);
        Python applies it to the full sheet."""
        df, import_summary = self.preview_import_sheet(
            project,
            source_excel=source_excel,
            sheet_name=sheet_name,
            objective_names=objective_names,
            manual_column_map=manual_column_map,
        )
        objective_names = list(dict.fromkeys(
            [
                str(name)
                for name in (import_summary.get("objective_names") or objective_names or [objective.name for objective in project.objectives])
                if str(name).strip() and str(name) in df.columns
            ]
        ))
        if not objective_names:
            raise ValueError("LLM 导入解释没有返回可用目标列；请补充目标列说明后重试。")
        if not project.factors:
            llm_factors = self._factor_specs_from_llm(import_summary.get("factors"))
            if not llm_factors:
                raise ValueError("LLM 导入解释没有返回 factors；请补充数据列语义后重试。")
            project.factors = llm_factors
        if not project.objectives and objective_names:
            llm_objectives = self._objective_specs_from_llm(import_summary.get("objectives"))
            if not llm_objectives:
                raise ValueError("LLM 导入解释没有返回 objectives；请补充目标列、优化方向后重试。")
            project.objectives = llm_objectives
        elif project.objectives:
            existing = {objective.name for objective in project.objectives}
            for objective in self._objective_specs_from_llm(import_summary.get("objectives")):
                if objective.name not in existing and objective.name in df.columns:
                    project.objectives.append(objective)
                    existing.add(objective.name)
        create_or_update_workbook(project)
        frame = append_rows(project, SHEET_EXPERIMENTS, df.to_dict(orient="records"))
        # Recompute desirability over the complete dataset, not just the imported rows.
        self._refresh_fuzzy_scores(project, frame, force_refresh_all=True)
        import_summary["imported_rows"] = int(len(df))
        import_summary["objective_names"] = list(objective_names or [])
        return project, import_summary

    def bootstrap_project_from_import(
        self,
        source_excel: str,
        *,
        sheet_name: str = 0,
        objective_names: list[str] | None = None,
        manual_column_map: dict[str, str] | None = None,
    ) -> tuple[ProjectState, pd.DataFrame, dict[str, object]]:
        """Bootstrap a brand-new project from an imported sheet: the LLM
        derives the workbook name, factors and objectives, then the sheet is
        re-imported against that schema."""
        source_path = Path(source_excel)
        if not source_path.is_absolute():
            source_path = self.workspace / source_excel
        workbook_name = validate_workbook_name(self._suggest_import_workbook_name(source_path))
        project = ProjectState(
            name=Path(workbook_name).stem,
            workspace_dir=str(self.workspace),
            workbook_path=str(self.workspace / workbook_name),
            factors=[],
            objectives=[],
        )
        df, import_summary = self.preview_import_sheet(
            project,
            source_excel=str(source_path),
            sheet_name=sheet_name,
            objective_names=objective_names,
            manual_column_map=manual_column_map,
        )
        llm_workbook_name = str(import_summary.get("workbook_name") or "").strip()
        if llm_workbook_name:
            llm_workbook_name = validate_workbook_name(llm_workbook_name)
            if not llm_workbook_name.endswith(".xlsx"):
                llm_workbook_name = f"{llm_workbook_name}.xlsx"
            project.name = Path(llm_workbook_name).stem
            project.workbook_path = str(self.workspace / llm_workbook_name)
        inferred_objectives = list(dict.fromkeys(
            [
                str(name)
                for name in (import_summary.get("objective_names") or objective_names or [])
                if str(name).strip() and str(name) in df.columns
            ]
        ))
        if not inferred_objectives:
            raise ValueError("LLM 导入解释没有返回可用目标列；请补充目标列说明后重试。")
        project.objectives = self._objective_specs_from_llm(import_summary.get("objectives"))
        project.factors = self._factor_specs_from_llm(import_summary.get("factors"))
        if not project.objectives:
            raise ValueError("LLM 导入解释没有返回 objectives；请补充目标列、优化方向后重试。")
        if not project.factors:
            raise ValueError("LLM 导入解释没有返回 factors；请补充因素列语义后重试。")
        create_or_update_workbook(project)
        df, import_summary = self.preview_import_sheet(
            project,
            source_excel=str(source_path),
            sheet_name=sheet_name,
            objective_names=inferred_objectives,
            manual_column_map=manual_column_map,
        )
        import_summary["bootstrapped_project"] = True
        import_summary["objective_names"] = list(inferred_objectives)
        import_summary["inferred_factor_names"] = [factor.name for factor in project.factors]
        return project, df, import_summary

    def bootstrap_project_from_rows(
        self,
        rows: list[dict[str, object]],
        *,
        workbook_name: str | None = None,
        factors: list[FactorSpec] | None = None,
        objectives: list[ObjectiveSpec] | None = None,
        objective_names: list[str] | None = None,
    ) -> tuple[ProjectState, pd.DataFrame, dict[str, object]]:
        """Bootstrap a project from pasted rows plus LLM-declared factors and
        objectives; both declarations are mandatory (no rule guessing)."""
        if not rows:
            raise ValueError("No pasted experiment rows provided.")
        frame = self.data_handler.clean_dataframe(pd.DataFrame(rows))
        frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
        frame = self.data_handler._coalesce_duplicate_columns(frame)
        if frame.empty:
            raise ValueError("Pasted experiment rows were empty after cleaning.")

        inferred_objective_names = [objective.name for objective in objectives or []]
        if not inferred_objective_names:
            raise ValueError("LLM 已返回 rows，但没有返回 objectives；请重新让 LLM 识别目标列、优化方向和权重。")
        if not factors:
            raise ValueError("LLM 已返回 rows，但没有返回 factors；请重新让 LLM 识别因素列。")
        inferred_objectives = objectives
        inferred_factors = factors
        resolved_workbook_name = workbook_name or self._suggest_pasted_workbook_name()
        if not resolved_workbook_name.endswith(".xlsx"):
            resolved_workbook_name = f"{resolved_workbook_name}.xlsx"
        project = self.create_project(
            workbook_name=resolved_workbook_name,
            factors=inferred_factors,
            objectives=inferred_objectives,
            initial_batch_size=max(4, min(12, len(frame))),
        )
        normalized_rows = frame.to_dict(orient="records")
        project, experiment_frame = self.append_results(project, rows=normalized_rows)
        summary = {
            "bootstrapped_project": True,
            "source": "pasted_rows",
            "imported_rows": int(len(frame)),
            "inferred_factor_names": [factor.name for factor in project.factors],
            "objective_names": [objective.name for objective in project.objectives],
            "workbook_name": project.workbook.name,
        }
        return project, experiment_frame, summary

    def _python_frame_from_llm_schema(
        self,
        *,
        raw_sheet: pd.DataFrame,
        interpreted: dict[str, object],
        manual_column_map: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Use LLM schema decisions, then let Python import the full sheet.

        The LLM sees only a compact sheet window.  For file-based imports we
        never ask the model to emit every row of a large workbook; instead it
        identifies the table type, header row, semantic column mapping, factors,
        and objectives.  pandas then applies that schema to the complete raw
        sheet.
        """
        sheet_type = str(interpreted.get("sheet_type") or "unknown").strip().lower()
        header_row = self._coerce_int_like(interpreted.get("header_row"))
        merged_column_map = self._merged_column_map(interpreted, manual_column_map)

        if sheet_type not in {"flat_table", "unknown"}:
            raise ValueError(
                f"LLM 将该表识别为 `{sheet_type}`。为避免只导入样本，文件导入必须由 Python 全量执行；"
                "请先把该 sheet 另存为一行一个实验的平面表，或指定可全量解析的 sheet。"
            )
        if header_row is None or header_row <= 0:
            raise ValueError("LLM 没有返回可用于 Python 全量导入的 header_row；请补充表头行说明后重试。")
        header_index = header_row - 1
        if header_index >= len(raw_sheet):
            raise ValueError(f"LLM 返回的 header_row={header_row} 超出 sheet 范围。")

        header_values = [
            self.data_handler._header_cell_text(value, index)
            for index, value in enumerate(raw_sheet.iloc[header_index].tolist(), start=1)
        ]
        frame = raw_sheet.iloc[header_index + 1 :].reset_index(drop=True).copy()
        frame.columns = header_values
        frame = self.data_handler.clean_dataframe(frame)
        frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if frame.empty:
            return frame

        if merged_column_map:
            normalized_map = self._resolve_llm_column_map(frame.columns, merged_column_map)
            if normalized_map:
                frame = self.data_handler.clean_dataframe(frame.rename(columns=normalized_map))
                frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
        return self.data_handler._coalesce_duplicate_columns(frame)

    def _merged_column_map(
        self,
        interpreted: dict[str, object],
        manual_column_map: dict[str, str] | None = None,
    ) -> dict[str, str]:
        merged_column_map = dict(interpreted.get("column_mapping") or {})
        for source, target in (manual_column_map or {}).items():
            if str(source).strip() and str(target).strip():
                merged_column_map[str(source)] = str(target)
        return merged_column_map

    def _resolve_llm_column_map(self, columns: list[object] | pd.Index, mapping: dict[str, str]) -> dict[str, str]:
        source_columns = [str(column) for column in columns]
        normalized_sources = {self.data_handler.normalize_header(column): column for column in source_columns}
        resolved: dict[str, str] = {}
        for source, target in mapping.items():
            source_text = str(source or "").strip()
            target_text = str(target or "").strip()
            if not source_text or not target_text:
                continue
            if source_text in source_columns:
                resolved[source_text] = target_text
                continue
            normalized = self.data_handler.normalize_header(source_text)
            matched = normalized_sources.get(normalized)
            if matched is not None:
                resolved[matched] = target_text
        return resolved

    def _factor_specs_from_llm(self, payload: object) -> list[FactorSpec]:
        if not isinstance(payload, list):
            return []
        factors: list[FactorSpec] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                factors.append(FactorSpec.model_validate(item))
            except Exception:
                continue
        return factors

    def _objective_specs_from_llm(self, payload: object) -> list[ObjectiveSpec]:
        if not isinstance(payload, list):
            return []
        objectives: list[ObjectiveSpec] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                objectives.append(ObjectiveSpec.model_validate(item))
            except Exception:
                continue
        return objectives

    def preview_import_sheet(
        self,
        project: ProjectState,
        source_excel: str,
        sheet_name: str = 0,
        objective_names: list[str] | None = None,
        manual_column_map: dict[str, str] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        """Interpret a source sheet without writing anything: returns the
        full parsed frame and an import summary (schema decisions, matched
        targets, mapping diagnostics)."""
        source_path = Path(source_excel)
        if not source_path.is_absolute():
            source_path = self.workspace / source_excel
        known_factor_names = [factor.name for factor in project.factors]
        known_objective_names = [objective.name for objective in project.objectives]
        raw_sheet = self.data_handler.read_raw_sheet(source_path, sheet_name=sheet_name)
        llm_payload = self.data_handler.build_llm_sheet_payload(
            raw_sheet,
            known_factor_names=known_factor_names,
            known_objective_names=known_objective_names,
            objective_names=objective_names,
        )
        interpreted = self.import_interpreter.interpret_sheet(llm_payload)
        df = self._python_frame_from_llm_schema(
            raw_sheet=raw_sheet,
            interpreted=interpreted,
            manual_column_map=manual_column_map,
        )
        if df.empty:
            raise ValueError("没有识别出可导入的实验记录。请检查表头、sheet 名或补充列对应关系。")
        merged_column_map = self._merged_column_map(interpreted, manual_column_map)
        matched_targets = list(dict.fromkeys(str(column) for column in df.columns if str(column).strip()))
        llm_objective_candidates = [
            str(name)
            for name in (interpreted.get("objective_candidates") or [])
            if str(name) in df.columns
        ]
        llm_objective_spec_names = [
            str(item.get("name") or "")
            for item in (interpreted.get("objectives") or [])
            if isinstance(item, dict) and str(item.get("name") or "") in df.columns
        ]
        effective_objective_names = list(dict.fromkeys(
            llm_objective_candidates + llm_objective_spec_names + list(objective_names or known_objective_names)
        ))
        critical_targets = [name for name in effective_objective_names if name]
        missing_critical_targets = [name for name in critical_targets if name not in matched_targets]
        matched_factor_targets = [name for name in known_factor_names if name in matched_targets]
        import_summary = {
            "header_row": interpreted.get("header_row") or 0,
            "source_sheet_type": interpreted.get("sheet_type") or "unknown",
            "interpreter_notes": interpreted.get("notes") or "",
            "interpreter": "llm_schema_python_full_import",
            "llm_payload_rows": len(llm_payload.get("rows") or []),
            "python_full_rows": int(len(df)),
            "workbook_name": interpreted.get("workbook_name") or "",
            "source_columns": list(dict.fromkeys(str(column) for column in df.columns if str(column).strip())),
            "mapped_columns": [f"{source}->{target}" for source, target in merged_column_map.items()],
            "ignored_columns": interpreted.get("ignored_columns") or [],
            "preview_rows": df.head(5).fillna("").to_dict(orient="records"),
            "preview_columns": list(df.columns),
            "detected_rows": int(len(df)),
            "llm_mapped_columns": [f"{source}->{target}" for source, target in (interpreted.get("column_mapping") or {}).items()],
            "manual_mapped_columns": [f"{source}->{target}" for source, target in (manual_column_map or {}).items()],
            "matched_targets": matched_targets,
            "critical_targets": critical_targets,
            "objective_candidates": interpreted.get("objective_candidates") or [],
            "factor_candidates": [name for name in (interpreted.get("factor_candidates") or []) if name in df.columns],
            "factors": interpreted.get("factors") or [],
            "objectives": interpreted.get("objectives") or [],
            "objective_names": list(effective_objective_names),
            "missing_critical_targets": missing_critical_targets,
            "matched_factor_count": len(matched_factor_targets),
            "needs_mapping_clarification": bool(missing_critical_targets) or (
                bool(known_factor_names) and not matched_factor_targets
            ),
        }
        return df, import_summary

    def _suggest_import_workbook_name(self, source_path: Path) -> str:
        base_name = source_path.stem.strip() or "导入实验"
        candidate = f"{base_name}.xlsx"
        if not (self.workspace / candidate).exists():
            return candidate
        for index in range(2, 1000):
            numbered = f"{base_name}_{index}.xlsx"
            if not (self.workspace / numbered).exists():
                return numbered
        return f"{base_name}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.xlsx"

    def _suggest_pasted_workbook_name(self) -> str:
        base_name = "粘贴初始数据"
        candidate = f"{base_name}.xlsx"
        if not (self.workspace / candidate).exists():
            return candidate
        for index in range(2, 1000):
            numbered = f"{base_name}_{index}.xlsx"
            if not (self.workspace / numbered).exists():
                return numbered
        return f"{base_name}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.xlsx"

    def export_summary(self, project: ProjectState, output_name: str | None = None) -> Path:
        """Export the project state as JSON into the workspace and return the path."""
        output = self.workspace / (output_name or f"{project.name}_summary.json")
        return export_project_json(project, output)

    def list_workbooks(self) -> list[str]:
        """Return the names of all workbook files in the workspace, sorted."""
        return sorted(path.name for path in self.workspace.glob("*.xlsx"))

    def _save_design_suggestions(
        self,
        project: ProjectState,
        suggestions: list[DesignSuggestion],
        status: str,
    ) -> pd.DataFrame:
        next_experiment_id = self._next_experiment_id(project)
        rows = []
        for item in suggestions:
            if item.experiment_id is None:
                item.experiment_id = next_experiment_id
                next_experiment_id += 1
            row = {
                "experiment_id": item.experiment_id,
                "Trial": item.batch_id,
                "candidate_id": item.candidate_id,
                "score": round(item.score, 6),
                "status": status,
            }
            row.update(item.values)
            rows.append(row)
        project.latest_batch_id = suggestions[0].batch_id if suggestions else project.latest_batch_id
        frame = append_rows(project, SHEET_DESIGNS, rows)
        return frame

    def _append_suggestions_to_experiment(self, project: ProjectState, suggestions: list[DesignSuggestion]) -> None:
        rows = []
        for item in suggestions:
            row = {
                "experiment_id": item.experiment_id,
                "Trial": item.batch_id,
                "candidate_id": item.candidate_id,
                "source": "design",
                "status": "suggested",
            }
            row.update(item.values)
            rows.append(row)
        if rows:
            frame = self._upsert_experiment_rows(project, rows)
            self._refresh_fuzzy_scores(project, frame)

    def _refresh_fuzzy_scores(self, project: ProjectState, frame: pd.DataFrame | None = None, force_refresh_all: bool = False) -> pd.DataFrame:
        # When refreshing all, read the sheet so desirability is computed over
        # the complete dataset rather than the caller's partial frame.
        if force_refresh_all or frame is None:
            experiments = read_sheet(project, SHEET_EXPERIMENTS)
        else:
            experiments = frame
        if experiments.empty:
            return experiments
        scored, desirability_updates = FuzzyScore(experiments).score_dataframe(project.objectives)

        # Sync updated desirability functions to MPO sheet for transparency
        if desirability_updates:
            project = self._sync_desirability_to_mpo(project, desirability_updates)

        # Avoid write-on-read: experiment_view calls this on every dashboard
        # tick. Only persist when the scored frame actually differs from the
        # one we read in (e.g. FuzzyScore changed or a new column appeared).
        if not _frames_equivalent(experiments, scored):
            write_sheet(project, SHEET_EXPERIMENTS, scored)
        return scored

    def _refresh_experiment_schema(self, project: ProjectState) -> None:
        current = read_sheet(project, SHEET_EXPERIMENTS)
        required_columns = ["experiment_id", "Trial", "candidate_id", "source", "status"]
        required_columns.extend([factor.name for factor in project.factors])
        required_columns.extend([objective.name for objective in project.objectives])
        required_columns.extend(["FuzzyScore", "COMMENT"])
        for column in required_columns:
            if column not in current.columns:
                current[column] = None
        current = current[required_columns]
        write_sheet(project, SHEET_EXPERIMENTS, current)

    def _filter_status_rows(self, frame: pd.DataFrame, status: str) -> pd.DataFrame:
        if frame.empty or "status" not in frame.columns:
            return frame.iloc[0:0].copy()
        mask = frame["status"].astype(str).str.lower() == status.lower()
        return frame.loc[mask].copy()

    def _completed_experiments(self, project: ProjectState, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        if "status" in frame.columns:
            completed = self._filter_status_rows(frame, status="completed")
            if not completed.empty:
                return completed
        objective_columns = [objective.name for objective in project.objectives if objective.name in frame.columns]
        if objective_columns:
            mask = frame[objective_columns].notna().any(axis=1)
            return frame.loc[mask].copy()
        return frame.iloc[0:0].copy()

    def _best_experiment_row(
        self,
        completed_df: pd.DataFrame,
        objective: ObjectiveSpec | None,
    ) -> dict[str, object] | None:
        if completed_df.empty:
            return None
        if "FuzzyScore" in completed_df.columns:
            fuzzy = pd.to_numeric(completed_df["FuzzyScore"], errors="coerce")
            if fuzzy.notna().any():
                row = completed_df.loc[fuzzy.idxmax()]
                return row.to_dict()
        if objective is None or objective.name not in completed_df.columns:
            return completed_df.iloc[0].to_dict()
        values = pd.to_numeric(completed_df[objective.name], errors="coerce")
        valid = completed_df.loc[values.notna()].copy()
        if valid.empty:
            return completed_df.iloc[0].to_dict()
        metric = pd.to_numeric(valid[objective.name], errors="coerce")
        if objective.goal == "minimize":
            row = valid.loc[metric.idxmin()]
            return row.to_dict()
        if objective.goal == "target" and objective.target_value is not None:
            target = float(objective.target_value)
            row = valid.loc[(metric - target).abs().idxmin()]
            return row.to_dict()
        row = valid.loc[metric.idxmax()]
        return row.to_dict()

    def _batch_counts(self, completed_df: pd.DataFrame) -> list[int]:
        if completed_df.empty or "Trial" not in completed_df.columns:
            return []
        trial_values = pd.to_numeric(completed_df["Trial"], errors="coerce").dropna().astype(int)
        if trial_values.empty:
            return []
        grouped = trial_values.value_counts().sort_index()
        return [int(value) for value in grouped.tolist()]

    def _score_trend(self, completed_df: pd.DataFrame) -> list[float]:
        if completed_df.empty or "FuzzyScore" not in completed_df.columns:
            return []
        scores = pd.to_numeric(completed_df["FuzzyScore"], errors="coerce").dropna()
        if scores.empty:
            return []
        cumulative_best = scores.cummax()
        return [float(value) for value in cumulative_best.tolist()]

    def _objective_trend(self, completed_df: pd.DataFrame, objective: ObjectiveSpec | None) -> list[float]:
        if completed_df.empty or objective is None or objective.name not in completed_df.columns:
            return []
        values = pd.to_numeric(completed_df[objective.name], errors="coerce").dropna()
        if values.empty:
            return []
        if objective.goal == "minimize":
            trend = values.cummin()
        elif objective.goal == "target" and objective.target_value is not None:
            target = float(objective.target_value)
            trend = (values - target).abs().iloc[::-1].cummin().iloc[::-1]
        else:
            trend = values.cummax()
        return [float(value) for value in trend.tolist()]

    def _factor_series_summary(self, series: pd.Series, kind: str) -> str:
        if kind == "categorical":
            counts = series.dropna().astype(str).value_counts().head(3)
            return ", ".join(f"{label}({count})" for label, count in counts.items()) or "-"
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return "-"
        return f"{numeric.min():g} ~ {numeric.max():g}"

    def _view_frame_for_scope(self, project: ProjectState, frame: pd.DataFrame, scope: str) -> pd.DataFrame:
        normalized_scope = str(scope or "recent").strip().lower()
        if normalized_scope == "pending":
            statuses = {"suggested", "initialized", "proposed"}
            if "status" not in frame.columns:
                return frame.iloc[0:0].copy()
            mask = frame["status"].astype(str).str.lower().isin(statuses)
            return frame.loc[mask].copy()
        if normalized_scope in {"best", "completed", "recent"}:
            return self._completed_experiments(project, frame)
        if normalized_scope == "all":
            return frame.copy()
        return self._completed_experiments(project, frame)

    def _sort_experiment_view(self, frame: pd.DataFrame, *, sort: str) -> pd.DataFrame:
        if frame.empty:
            return frame
        ordered = frame.copy()
        ordered = ordered.assign(_row_order=range(len(ordered)))
        normalized_sort = str(sort or "recent").strip().lower()
        if normalized_sort == "score_desc" and "FuzzyScore" in ordered.columns:
            ordered = ordered.assign(_score_sort=pd.to_numeric(ordered["FuzzyScore"], errors="coerce"))
            sort_by = ["_score_sort", "_row_order"]
            ascending = [False, False]
            if "Trial" in ordered.columns:
                ordered = ordered.assign(_trial_sort=pd.to_numeric(ordered["Trial"], errors="coerce"))
                sort_by.insert(1, "_trial_sort")
                ascending.insert(1, False)
            return ordered.sort_values(by=sort_by, ascending=ascending, kind="stable", na_position="last").drop(
                columns=[column for column in ["_score_sort", "_trial_sort", "_row_order"] if column in ordered.columns]
            )
        if normalized_sort == "trial_asc" and "Trial" in ordered.columns:
            ordered = ordered.assign(_trial_sort=pd.to_numeric(ordered["Trial"], errors="coerce"))
            sort_by = ["_trial_sort", "_row_order"]
            ascending = [True, True]
            if "candidate_id" in ordered.columns:
                ordered = ordered.assign(_candidate_sort=pd.to_numeric(ordered["candidate_id"], errors="coerce"))
                sort_by.insert(1, "_candidate_sort")
                ascending.insert(1, True)
            return ordered.sort_values(by=sort_by, ascending=ascending, kind="stable", na_position="last").drop(
                columns=[column for column in ["_trial_sort", "_candidate_sort", "_row_order"] if column in ordered.columns]
            )
        if "Trial" in ordered.columns:
            ordered = ordered.assign(_trial_sort=pd.to_numeric(ordered["Trial"], errors="coerce"))
            sort_by = ["_trial_sort", "_row_order"]
            ascending = [False, False]
            if "candidate_id" in ordered.columns:
                ordered = ordered.assign(_candidate_sort=pd.to_numeric(ordered["candidate_id"], errors="coerce"))
                sort_by.insert(1, "_candidate_sort")
                ascending.insert(1, False)
            return ordered.sort_values(by=sort_by, ascending=ascending, kind="stable", na_position="last").drop(
                columns=[column for column in ["_trial_sort", "_candidate_sort", "_row_order"] if column in ordered.columns]
            )
        return ordered.sort_values(by="_row_order", ascending=False, kind="stable").drop(columns="_row_order")

    def _upsert_experiment_rows(self, project: ProjectState, rows: list[dict[str, object]]) -> pd.DataFrame:
        current = read_sheet(project, SHEET_EXPERIMENTS)
        if current.empty:
            current = self._experiment_schema_frame(project)
        else:
            current = self._align_experiment_columns(project, current.copy())
        next_experiment_id = self._next_experiment_id(project, experiment_df=current)
        next_candidate_id = self._next_candidate_id(project, experiment_df=current)

        for row in rows:
            normalized = self._normalize_experiment_row(project, row)
            matched_index = self._match_experiment_row(
                current,
                normalized,
                original_row=row,
                latest_batch_id=int(project.latest_batch_id or 0),
            )
            if matched_index is None:
                if normalized.get("experiment_id") in (None, ""):
                    normalized["experiment_id"] = next_experiment_id
                    next_experiment_id += 1
                # Assign candidate_id if not provided OR if the provided one
                # already exists in the current frame (avoids creating a
                # duplicate row that shares a candidate_id with an existing
                # row — the append branch updates the existing row instead).
                existing_candidate_id = normalized.get("candidate_id")
                candidate_id_taken = (
                    existing_candidate_id not in (None, "")
                    and "candidate_id" in current.columns
                    and not current.empty
                    and bool(
                        pd.to_numeric(current["candidate_id"], errors="coerce")
                        .eq(self._coerce_int_like(existing_candidate_id))
                        .any()
                    )
                )
                if normalized.get("candidate_id") in (None, "") or candidate_id_taken:
                    normalized["candidate_id"] = next_candidate_id
                    next_candidate_id += 1
                # Row-by-row append (not a batched concat) on purpose: later
                # rows in the same batch must be able to match rows appended
                # just above. concat_rows avoids the deprecated all-NA concat.
                current = concat_rows(current, [normalized])
                continue
            for key, value in normalized.items():
                if value in (None, "") and key not in {"status", "source"}:
                    continue
                current.at[matched_index, key] = value

        write_sheet(project, SHEET_EXPERIMENTS, current)
        return current

    def _sync_design_status(self, project: ProjectState, rows: list[dict[str, object]]) -> None:
        design_df = read_sheet(project, SHEET_DESIGNS)
        if design_df.empty or "Trial" not in design_df.columns or "candidate_id" not in design_df.columns:
            return
        updated = False
        for row in rows:
            trial = self._coerce_int_like(row.get("Trial"))
            candidate_id = self._coerce_int_like(row.get("candidate_id"))
            status = str(row.get("status") or "").strip().lower()
            if trial is None or candidate_id is None or not status:
                continue
            mask = (
                pd.to_numeric(design_df["Trial"], errors="coerce").eq(trial)
                & pd.to_numeric(design_df["candidate_id"], errors="coerce").eq(candidate_id)
            )
            if not mask.any():
                continue
            design_df.loc[mask, "status"] = status
            updated = True
        if updated:
            write_sheet(project, SHEET_DESIGNS, design_df)

    def _validate_llm_declared_result_columns(self, project: ProjectState, rows: list[dict]) -> None:
        """Require explicit LLM objective declarations instead of guessing goals from columns."""
        if project.objectives:
            return
        non_completed = all(str(row.get("status") or "completed").lower() == "skipped" for row in rows)
        if non_completed:
            return
        raise ValueError(
            "LLM 返回了实验记录，但项目没有 objectives；请让 LLM 在 append_results/import 输出中显式声明 objectives，"
            "不要由程序规则猜测目标列或优化方向。"
        )

    def _sync_desirability_to_mpo(
        self,
        project: ProjectState,
        desirability_updates: dict[str, list[dict[str, float | str]]]
    ) -> ProjectState:
        """Sync auto-calculated desirability functions to the MPO sheet.

        This ensures the MPO sheet reflects the actual desirability functions
        being used for scoring, providing transparency to the user.

        Args:
            project: The project state
            desirability_updates: Dict mapping objective name to its desirability points

        Returns:
            Updated project state
        """
        if not desirability_updates:
            return project

        updated = False
        for objective in project.objectives:
            if objective.name in desirability_updates:
                new_desirability = normalize_desirability_text(desirability_updates[objective.name])
                if new_desirability != objective.desirability:
                    objective.desirability = new_desirability
                    updated = True

        if updated:
            create_or_update_workbook(project)

        return project

    def _experiment_schema_frame(self, project: ProjectState) -> pd.DataFrame:
        columns = ["experiment_id", "Trial", "candidate_id", "source", "status"]
        columns.extend(factor.name for factor in project.factors)
        columns.extend(objective.name for objective in project.objectives)
        columns.extend(["FuzzyScore", "COMMENT"])
        return pd.DataFrame(columns=columns)

    def _align_experiment_columns(self, project: ProjectState, frame: pd.DataFrame) -> pd.DataFrame:
        schema = self._experiment_schema_frame(project)
        for column in schema.columns:
            if column not in frame.columns:
                frame[column] = None
        # Ensure numeric columns can handle decimal values by converting to float
        # This includes ID columns and factor columns that might have decimal values
        numeric_columns = {"Trial", "candidate_id", "experiment_id"}
        numeric_columns.update({f.name for f in project.factors if f.kind in ("continuous", "ordinal")})
        for column in numeric_columns:
            if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column]):
                frame[column] = frame[column].astype(float)
        return frame[schema.columns]

    def _prepare_column_for_assignment(
        self,
        project: ProjectState,
        frame: pd.DataFrame,
        column: str,
        value: object,
    ) -> pd.DataFrame:
        if column not in frame.columns:
            return frame
        if column in {"COMMENT", "status", "source"}:
            frame[column] = frame[column].astype("object")
            return frame
        factor = next((item for item in project.factors if item.name == column), None)
        if factor is not None and factor.kind == "categorical":
            frame[column] = frame[column].astype("object")
            return frame
        if isinstance(value, str):
            frame[column] = frame[column].astype("object")
            return frame
        # Convert numeric columns to float64 if value is a float to handle decimal values
        if isinstance(value, float) and column in frame.columns:
            if pd.api.types.is_numeric_dtype(frame[column]):
                frame[column] = frame[column].astype(float)
        return frame

    def _normalize_experiment_row(self, project: ProjectState, row: dict[str, object]) -> dict[str, object]:
        schema_columns = self._experiment_schema_frame(project).columns.tolist()
        normalized = {column: None for column in schema_columns}
        normalized.update(row)
        normalized["experiment_id"] = self._coerce_int_like(normalized.get("experiment_id"))
        normalized["Trial"] = self._coerce_int_like(normalized.get("Trial")) or int(project.latest_batch_id or 0)
        candidate_id = self._coerce_int_like(normalized.get("candidate_id"))
        normalized["candidate_id"] = candidate_id
        normalized["source"] = str(normalized.get("source") or "manual")
        normalized["status"] = str(normalized.get("status") or "completed").lower()
        return normalized

    def _match_experiment_row(
        self,
        frame: pd.DataFrame,
        row: dict[str, object],
        *,
        original_row: dict[str, object] | None = None,
        latest_batch_id: int | None = None,
    ) -> int | None:
        if frame.empty or "Trial" not in frame.columns:
            return None
        experiment_id = self._coerce_int_like(row.get("experiment_id"))
        if experiment_id is not None and "experiment_id" in frame.columns:
            experiment_mask = pd.to_numeric(frame["experiment_id"], errors="coerce").eq(experiment_id)
            matches = frame.index[experiment_mask].tolist()
            if matches:
                return int(matches[-1])
        # Detect whether the CALLER actually provided a Trial. The normalized
        # row always carries a Trial (the normalizer defaults missing Trials
        # to the project's latest_batch_id), so a normalized Trial of 0 or
        # latest_batch_id must NOT be treated as an explicit Trial for matching
        # purposes — otherwise appending a result row that omits Trial
        # (providing only candidate_id) duplicates the row instead of updating
        # it, leaving two rows sharing one candidate_id.
        trial_value: object = None
        if original_row is not None:
            trial_value = original_row.get("Trial")
        trial = self._coerce_int_like(trial_value)
        candidate_id = self._coerce_int_like(row.get("candidate_id"))
        if trial is None:
            # Result row without a Trial: fall back to matching on candidate_id
            # WITHIN THE LATEST BATCH only. candidate_id is batch-local, so a
            # candidate_id reused across batches (e.g. candidate_id=1 in
            # batch 1 and again in batch 2) must NOT match the old batch —
            # the new row appends to the new batch instead of updating the old
            # one. Only re-appending to the SAME batch (latest_batch_id) is a
            # duplicate that should be updated in place.
            if candidate_id is None or "candidate_id" not in frame.columns:
                return None
            target_batch = latest_batch_id if latest_batch_id is not None else 0
            candidate_mask = pd.to_numeric(frame["candidate_id"], errors="coerce").eq(candidate_id)
            trial_mask = pd.to_numeric(frame["Trial"], errors="coerce").eq(target_batch)
            mask = candidate_mask & trial_mask
            matches = frame.index[mask].tolist()
            if matches:
                return int(matches[-1])
            return None
        trial_mask = pd.to_numeric(frame["Trial"], errors="coerce").eq(trial)
        if candidate_id is not None and "candidate_id" in frame.columns:
            candidate_mask = pd.to_numeric(frame["candidate_id"], errors="coerce").eq(candidate_id)
            mask = trial_mask & candidate_mask
            matches = frame.index[mask].tolist()
            if matches:
                return int(matches[-1])
        return None

    def _resolve_result_row_references(
        self,
        project: ProjectState,
        rows: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        current = read_sheet(project, SHEET_EXPERIMENTS)
        if current.empty:
            return rows
        current = self._align_experiment_columns(project, current.copy())
        resolved_rows: list[dict[str, object]] = []
        for row in rows:
            resolved = dict(row)
            match = self._match_reference_row(current, resolved)
            if match is not None:
                if "experiment_id" in match.index and resolved.get("experiment_id") in (None, ""):
                    resolved["experiment_id"] = self._coerce_int_like(match.get("experiment_id"))
                if resolved.get("Trial") in (None, "", 0):
                    resolved["Trial"] = self._coerce_int_like(match.get("Trial"))
                if resolved.get("candidate_id") in (None, "") or resolved.get("experiment_id") == resolved.get("candidate_id"):
                    resolved["candidate_id"] = self._coerce_int_like(match.get("candidate_id"))
            resolved_rows.append(resolved)
        return resolved_rows

    def _match_reference_row(self, frame: pd.DataFrame, row: dict[str, object]) -> pd.Series | None:
        has_experiment_id = "experiment_id" in row and row.get("experiment_id") not in (None, "")
        experiment_id = self._coerce_int_like(row.get("experiment_id")) if has_experiment_id else None
        trial = self._coerce_int_like(row.get("Trial"))
        candidate_id = self._coerce_int_like(row.get("candidate_id"))
        if has_experiment_id and experiment_id is not None and "experiment_id" in frame.columns:
            mask = pd.to_numeric(frame["experiment_id"], errors="coerce").eq(experiment_id)
            if mask.any():
                return frame.loc[mask].iloc[-1]
        if trial is not None and candidate_id is not None and "candidate_id" in frame.columns:
            mask = (
                pd.to_numeric(frame["Trial"], errors="coerce").eq(trial)
                & pd.to_numeric(frame["candidate_id"], errors="coerce").eq(candidate_id)
            )
            if mask.any():
                return frame.loc[mask].iloc[-1]
        return None

    def _plan_row_mask(
        self,
        frame: pd.DataFrame,
        *,
        experiment_id: int | None,
        trial: int | None,
        candidate_id: int | None,
    ) -> pd.Series:
        if frame.empty:
            return pd.Series(dtype=bool)
        if experiment_id is not None and "experiment_id" in frame.columns:
            mask = pd.to_numeric(frame["experiment_id"], errors="coerce").eq(experiment_id)
            if mask.any():
                return mask
        if trial is not None and candidate_id is not None and "Trial" in frame.columns and "candidate_id" in frame.columns:
            mask = (
                pd.to_numeric(frame["Trial"], errors="coerce").eq(trial)
                & pd.to_numeric(frame["candidate_id"], errors="coerce").eq(candidate_id)
            )
            if mask.any():
                return mask
        return pd.Series([False] * len(frame), index=frame.index)

    def _coerce_int_like(self, value: object) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _next_experiment_id(
        self,
        project: ProjectState,
        *,
        experiment_df: pd.DataFrame | None = None,
        design_df: pd.DataFrame | None = None,
    ) -> int:
        frames = [
            experiment_df if experiment_df is not None else read_sheet(project, SHEET_EXPERIMENTS),
            design_df if design_df is not None else read_sheet(project, SHEET_DESIGNS),
        ]
        max_value = 0
        for frame in frames:
            if frame.empty or "experiment_id" not in frame.columns:
                continue
            values = pd.to_numeric(frame["experiment_id"], errors="coerce").dropna()
            if values.empty:
                continue
            max_value = max(max_value, int(values.max()))
        return max_value + 1

    def _next_candidate_id(
        self,
        project: ProjectState,
        *,
        experiment_df: pd.DataFrame | None = None,
        design_df: pd.DataFrame | None = None,
    ) -> int:
        """Get the next available candidate_id.

        This finds the maximum candidate_id across experiments and designs,
        and returns the next sequential ID. This ensures that:
        1. New experiments get IDs that continue from the current maximum
        2. After deleting middle experiments, new IDs start from the new max

        Args:
            project: The project state
            experiment_df: Optional experiment dataframe (to avoid re-reading)
            design_df: Optional design dataframe (to avoid re-reading)

        Returns:
            The next candidate_id to assign
        """
        frames = [
            experiment_df if experiment_df is not None else read_sheet(project, SHEET_EXPERIMENTS),
            design_df if design_df is not None else read_sheet(project, SHEET_DESIGNS),
        ]
        max_value = 0
        for frame in frames:
            if frame.empty or "candidate_id" not in frame.columns:
                continue
            values = pd.to_numeric(frame["candidate_id"], errors="coerce").dropna()
            if values.empty:
                continue
            max_value = max(max_value, int(values.max()))
        return max_value + 1

    def _latest_design_batch_id(self, designs_df: pd.DataFrame) -> int | None:
        if designs_df.empty or "Trial" not in designs_df.columns:
            return None
        trial_values = pd.to_numeric(designs_df["Trial"], errors="coerce").dropna()
        if trial_values.empty:
            return None
        return int(trial_values.max())

    def _latest_design_rows(self, designs_df: pd.DataFrame, batch_id: int | None) -> list[dict[str, object]]:
        if designs_df.empty or batch_id is None or "Trial" not in designs_df.columns:
            return []
        trial_values = pd.to_numeric(designs_df["Trial"], errors="coerce")
        latest_df = designs_df.loc[trial_values == batch_id].copy()
        if latest_df.empty:
            return []
        if "candidate_id" in latest_df.columns:
            candidate_values = pd.to_numeric(latest_df["candidate_id"], errors="coerce")
            latest_df = latest_df.assign(_candidate_sort=candidate_values).sort_values(
                by="_candidate_sort",
                kind="stable",
                na_position="last",
            ).drop(columns="_candidate_sort")
        return latest_df.head(8).fillna("").to_dict(orient="records")


def _frames_equivalent(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Return True if two experiment frames are equivalent for persistence
    purposes — same shape and same cell values under a normalized comparison
    (so a write is a no-op). Used by _refresh_fuzzy_scores to skip write-on-read
    on every dashboard tick when scores are unchanged."""
    if a is b:
        return True
    if a is None or b is None:
        return False
    if list(a.columns) != list(b.columns):
        return False
    if len(a) != len(b):
        return False
    if a.index.shape != b.index.shape:
        return False
    try:
        left = a.fillna("").astype(object)
        right = b.fillna("").astype(object)
        return bool((left.values == right.values).all())
    except Exception:
        return False


def parse_factor_text(spec_text: str) -> FactorSpec:
    """Parse a free-text factor definition (Chinese or English: name, range
    or categories, optional step/unit) into a FactorSpec; raises ValueError
    on unparseable input."""
    text = spec_text.strip()
    text = re.sub(r"^(因素定义|参数定义|列定义)\s*[:：]?\s*", "", text)
    if ":" in text:
        name, detail = [part.strip() for part in text.split(":", 1)]
    else:
        tokens = text.split()
        if len(tokens) < 2:
            raise ValueError(f"无法解析因素定义: {spec_text}")
        name, detail = tokens[0], " ".join(tokens[1:])

    if any(keyword in detail for keyword in ["类别", "分类", "categorical"]):
        categories = _split_categories(detail)
        return FactorSpec(name=name, kind="categorical", categories=categories)

    range_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:到|-|~)\s*(-?\d+(?:\.\d+)?)", detail)
    if not range_match:
        raise ValueError(f"无法解析数值范围: {spec_text}")
    min_value = float(range_match.group(1))
    max_value = float(range_match.group(2))
    step_match = re.search(r"(?:步长|step)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", detail, re.IGNORECASE)
    unit_match = re.search(r"单位\s*[:=]?\s*([A-Za-z%°/\u4e00-\u9fff]+)", detail)
    unit = unit_match.group(1) if unit_match else None
    if any(keyword in detail for keyword in ["离散", "ordinal"]):
        return FactorSpec(
            name=name,
            kind="ordinal",
            min_value=min_value,
            max_value=max_value,
            step=float(step_match.group(1)) if step_match else 1.0,
            unit=unit,
        )
    return FactorSpec(name=name, kind="continuous", min_value=min_value, max_value=max_value, unit=unit)


def parse_objective_text(spec_text: str) -> ObjectiveSpec:
    """Parse a free-text objective definition (goal keyword and optional
    target value) into an ObjectiveSpec; defaults to maximize."""
    text = spec_text.strip()
    text = re.sub(r"^(把)?目标(改成|修改为|设为|设置为|是)?", "", text).strip(" :：")
    goal: GoalType = "maximize"
    if "最小" in text or "min" in text.lower():
        goal = "minimize"
    if "接近" in text or "目标值" in text or "target" in text.lower():
        goal = "target"
    target_value = None
    target_match = re.search(r"(?:目标值|target)\s*[:=]?\s*([^\s,，;；]+)", text, re.IGNORECASE)
    if target_match:
        token = target_match.group(1)
        target_value = float(token) if _looks_like_number(token) else token
    name = re.sub(r"(最大化|最小化|最大|最小|接近目标值.*|目标值.*)", "", text).strip(" :：")
    if not name:
        name = text
    return ObjectiveSpec(name=name, goal=goal, target_value=target_value)


def _split_categories(text: str) -> list[str]:
    after_marker = text
    for marker in ["类别", "分类", "包括", "取值", "=", "为"]:
        if marker in after_marker:
            after_marker = after_marker.split(marker, 1)[1]
    parts = re.split(r"[,，/|、\s]+", after_marker.strip())
    return [part for part in parts if part and part not in {"类别", "分类"}]


def _looks_like_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False
