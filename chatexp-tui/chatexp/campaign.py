"""Real-lab campaign tooling: turn proposed designs into an executable,
auditable campaign with three artifacts:

1. ``build_campaign_protocol`` — a lab worksheet for the next proposed batch:
   every run's exact factor levels (legal, snapped), the measurements required
   for each objective (name + unit + goal), and blank result cells for the
   operator to fill in.
2. ``import_campaign_results`` — strict validation of operator-entered results
   against the protocol: every protocol run must appear exactly once, factor
   values must match the proposed legal levels, objective values must be
   present and numeric. Rejections carry a reason; nothing partial is written.
3. ``campaign ledger`` events — protocol/observation/decision events appended
   to the audit event log, so a real campaign's provenance chain (proposal →
   worksheet → measurement entry → decision) is reconstructible.

The CLI entry (``python -m chatexp.campaign``) prints a protocol worksheet for
a project's latest pending designs. ``import_campaign_results`` and the ledger
helpers are exercised by the test suite and external benchmark tooling; they
are not yet wired into the app's design-confirmation flow.
"""

from __future__ import annotations

import argparse
import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .event_log import append_event
from .models import FactorSpec, ObjectiveSpec, ProjectState


@dataclass
class CampaignProtocol:
    batch_id: int
    factors: list[dict[str, Any]]
    objectives: list[dict[str, Any]]
    runs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": 1,
            "batch_id": self.batch_id,
            "factors": self.factors,
            "objectives": self.objectives,
            "runs": self.runs,
        }


def build_campaign_protocol(project: ProjectState, design_rows: list[dict[str, Any]]) -> CampaignProtocol:
    """Create the lab worksheet for a proposed batch.

    ``design_rows`` are the pending DESIGNS rows (factor values + Trial +
    candidate_id). Factor values are re-snapped to legal levels so the
    worksheet can be executed verbatim at the bench.
    """
    factor_payload = [
        {
            "name": factor.name,
            "kind": factor.kind,
            "unit": factor.unit,
            "levels": _factor_levels(factor),
        }
        for factor in project.factors
    ]
    objective_payload = [
        {
            "name": objective.name,
            "goal": objective.goal,
            "unit": objective.unit,
            "direction": _measure_direction(objective),
        }
        for objective in project.objectives
    ]
    protocol = CampaignProtocol(
        batch_id=int(project.latest_batch_id) + 1,
        factors=factor_payload,
        objectives=objective_payload,
    )
    factor_names = [factor.name for factor in project.factors]
    for row in design_rows:
        run: dict[str, Any] = {
            "candidate_id": row.get("candidate_id"),
            "values": {name: _snap_value(project, name, row.get(name)) for name in factor_names},
            "measure": {objective.name: None for objective in project.objectives},
            "status": "pending",
        }
        protocol.runs.append(run)
    return protocol


def _factor_levels(factor: FactorSpec) -> list[Any] | None:
    if factor.kind == "categorical":
        return list(factor.categories or [])
    if factor.kind == "ordinal" and factor.step:
        start = float(factor.min_value)
        stop = float(factor.max_value)
        step = float(factor.step)
        half = step / 2
        levels: list[float] = []
        current = start
        while current <= stop + half:
            levels.append(round(current, 6))
            current += step
        return levels
    return None


def _measure_direction(objective: ObjectiveSpec) -> str:
    if objective.goal == "maximize":
        return "higher is better"
    if objective.goal == "minimize":
        return "lower is better"
    return f"target {objective.target_value}" if objective.target_value is not None else "target range"


def _snap_value(project: ProjectState, factor_name: str, value: Any) -> Any:
    factor = next((item for item in project.factors if item.name == factor_name), None)
    if factor is None or value is None:
        return value
    if factor.kind == "categorical":
        return str(value)
    if factor.kind == "ordinal":
        levels = _factor_levels(factor) or []
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return value
        if not levels:
            return numeric
        return min(levels, key=lambda level: abs(level - numeric))
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


# ═════════════════════════════════════════════════════════════════
# Strict results import
# ═════════════════════════════════════════════════════════════════


@dataclass
class CampaignImportResult:
    ok: bool
    rows: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.ok:
            return f"{len(self.rows)} result row(s) accepted."
        return "rejected: " + " | ".join(self.rejections)


def import_campaign_results(
    project: ProjectState,
    protocol: CampaignProtocol,
    records: list[dict[str, Any]],
) -> CampaignImportResult:
    """Validate operator-entered results against the protocol.

    Accepted rows carry the protocol's candidate_id and the snapped factor
    values (guaranteeing the executed recipe) plus the entered measurements.
    Any protocol violation rejects the WHOLE import (all-or-nothing), so a
    partial workbook write can never misrepresent the campaign.
    """
    rejections: list[str] = []
    objective_names = [objective.name for objective in project.objectives]
    factor_names = [factor.name for factor in project.factors]
    expected_runs = {str(run.get("candidate_id")): run for run in protocol.runs}
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        candidate_id = str(record.get("candidate_id", "")).strip()
        if candidate_id not in expected_runs:
            rejections.append(f"row {index}: unknown candidate_id {candidate_id!r} (not in protocol batch {protocol.batch_id})")
            continue
        if candidate_id in seen:
            rejections.append(f"row {index}: candidate_id {candidate_id!r} entered more than once")
            continue
        seen.add(candidate_id)
        run = expected_runs[candidate_id]
        row: dict[str, Any] = {"candidate_id": record.get("candidate_id"), "Trial": protocol.batch_id}
        for name in factor_names:
            entered = record.get(name)
            if entered is None or str(entered).strip() == "":
                # The executed recipe is the protocol's job; the operator does
                # not re-type factor levels.
                row[name] = run["values"].get(name)
                continue
            snapped = _snap_value(project, name, entered)
            expected = run["values"].get(name)
            if isinstance(expected, (int, float)) and isinstance(snapped, (int, float)):
                if abs(float(snapped) - float(expected)) > 1e-9:
                    rejections.append(
                        f"row {index}: factor {name!r} entered {entered} but protocol specifies {expected}"
                    )
                    continue
            row[name] = snapped
        for name in objective_names:
            value = record.get(name)
            text = "" if value is None else str(value).strip()
            if text == "":
                rejections.append(f"row {index}: missing measurement for objective {name!r}")
                continue
            try:
                row[name] = float(text)
            except ValueError:
                rejections.append(f"row {index}: objective {name!r} value {value!r} is not numeric")
        rows.append(row)

    missing = sorted(set(expected_runs) - seen)
    if missing:
        rejections.append(f"protocol runs without results: {', '.join(missing)}")
    if rejections:
        return CampaignImportResult(ok=False, rejections=rejections)
    return CampaignImportResult(ok=True, rows=rows)


# ═════════════════════════════════════════════════════════════════
# Ledger + CSV helpers
# ═════════════════════════════════════════════════════════════════


def protocol_to_csv(protocol: CampaignProtocol) -> str:
    """Render the worksheet as CSV: factor columns filled, measure columns blank."""
    factor_names = [factor["name"] for factor in protocol.factors]
    objective_names = [objective["name"] for objective in protocol.objectives]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["candidate_id", *factor_names, *objective_names, "comment"])
    for run in protocol.runs:
        writer.writerow([
            run.get("candidate_id"),
            *[run["values"].get(name, "") for name in factor_names],
            *[("" if run["measure"].get(name) is None else run["measure"][name]) for name in objective_names],
            "",
        ])
    return buffer.getvalue()


def records_from_csv_text(text: str) -> list[dict[str, Any]]:
    return list(csv.DictReader(io.StringIO(text)))


def log_protocol_issued(task_dir: str | Path, protocol: CampaignProtocol) -> None:
    append_event(
        task_dir,
        "campaign_protocol_issued",
        batch_id=protocol.batch_id,
        runs=len(protocol.runs),
        factors=[factor["name"] for factor in protocol.factors],
        objectives=[objective["name"] for objective in protocol.objectives],
    )


def log_results_imported(task_dir: str | Path, protocol: CampaignProtocol, accepted: int) -> None:
    append_event(
        task_dir,
        "campaign_results_imported",
        batch_id=protocol.batch_id,
        accepted_rows=accepted,
    )


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a campaign protocol worksheet for a project's latest proposed batch."
    )
    parser.add_argument("workbook", help="project workbook name/path")
    args = parser.parse_args()

    from .service import ChatExpService
    from .storage import SHEET_DESIGNS, read_sheet

    workbook = Path(args.workbook)
    workspace = workbook.parent
    service = ChatExpService(workspace)
    project = service.load_project(workbook.name)
    designs = read_sheet(project, SHEET_DESIGNS)
    pending = designs[designs.get("status", "").astype(str).str.lower() == "proposed"]
    if pending.empty:
        raise SystemExit("No proposed (pending) designs found; run design generation first.")
    records = pending.to_dict(orient="records")
    protocol = build_campaign_protocol(project, records)
    print(protocol_to_csv(protocol))


if __name__ == "__main__":
    main()


__all__ = [
    "CampaignImportResult",
    "CampaignProtocol",
    "build_campaign_protocol",
    "import_campaign_results",
    "log_protocol_issued",
    "log_results_imported",
    "protocol_to_csv",
    "records_from_csv_text",
]
