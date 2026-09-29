"""Phase 2: validate the intervention itself, before trusting any agent metric.

The blueprint is explicit that the state model must be measured on its own terms
first: *"Before measuring agents, test the intervention."* Aggregate numbers
alone cannot tell a weak representation apart from a delivery bug, which is
exactly the ambiguity that made the 11-task pilot uninterpretable.

This module produces the per-task diagnostic the blueprint asks for:

| Task | Anchor | Route | Tables | Records | Delivery | Precision | Recall |

Each column answers one question in the delivery chain:

``Anchor``   - did a prompt mention resolve to exactly one real row?
``Route``    - was the route structurally valid (Phase 1.3)?
``Tables``   - how many distinct tables were actually read?
``Records``  - how many rows reached the agent?
``Delivery`` - was the intervention actually injected?
``Precision``/``Recall`` - post-hoc, prompt/seed-derived, and therefore *analysis
               only*: this module must stay unreachable from the Condition-B
               import chain (SEC-001 / GUD-008).

The engine targets in :data:`DELIVERY_TARGET`, :data:`PRECISION_TARGET`, and
:data:`RECALL_TARGET` are **engineering acceptance thresholds**, not research
conclusions. :func:`evaluate_targets` reports pass/fail against them without ever
asserting that the representation is "good enough" as a scientific claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from csm_env.query.models import RouteStatus

from .relevance import RelevanceSet
from .results import TaskRunRecord
from .stats import state_model_delivered

#: Phase 2 acceptance thresholds. These are engineering targets used to decide
#: whether the *pipeline* is trustworthy, not claims about agent performance.
DELIVERY_TARGET = 1.0
PRECISION_TARGET = 0.80
RECALL_TARGET = 0.80

#: P5.1/A2: the B1-vs-B2 contrast is interpretable only when the minimal arm
#: actually delivered a *reduced* state on every analyzed run. A gate-failed
#: run (selector refused -> broad state delivered) is B1-equivalent, so it must
#: not count toward the B2 payload. The contrast target is met when no delivered
#: minimal run is gate-failed and the B2 payload is strictly smaller than B1.
CONTRAST_TARGET = 1.0

#: Route statuses that count as structurally valid for the ``Route`` column.
VALID_ROUTE_STATUSES = (
    RouteStatus.COMPLETE,
    RouteStatus.UNRESOLVED,
    RouteStatus.BUDGET_EXHAUSTED,
)

#: Column order of the rendered Markdown table, matching the blueprint.
REPORT_COLUMNS = (
    "Task",
    "Anchor",
    "Route",
    "Tables",
    "Records",
    "Delivery",
    "Precision",
    "Recall",
)


@dataclass(frozen=True)
class StateDiagnostic:
    """One task's end-to-end delivery-chain evidence.

    Attributes:
        task_id: The task this row describes.
        run_index: Repeat index within the sweep.
        anchor_table: Resolved anchor table, or ``None``.
        anchor_row_id: Resolved anchor primary key, or ``None``.
        anchor_strategy: Which resolution tier produced the match.
        route_status: Route status reported by the router.
        structurally_valid: Whether the route was sound (Phase 1.3 rule).
        retrieved_tables: Number of distinct tables read.
        retrieved_table_names: Those table names.
        retrieved_records: Number of rows reaching the agent.
        retrieved_tokens: Estimated tokens of the injected block.
        delivered: Whether the intervention was actually injected.
        precision: Post-hoc retrieval precision, when measurable.
        recall: Post-hoc retrieval recall, when measurable.
        precision_granularity: ``row``, ``table`` or ``n/a``.
        error: Any recorded error message.
    """

    task_id: str
    run_index: int
    anchor_table: Optional[str]
    anchor_row_id: Optional[str]
    anchor_strategy: str
    route_status: str
    structurally_valid: bool
    retrieved_tables: int
    retrieved_table_names: tuple
    retrieved_records: int
    retrieved_tokens: int
    delivered: bool
    precision: Optional[float]
    recall: Optional[float]
    precision_granularity: str
    error: Optional[str] = None
    #: P5.1/A2: the minimal selector gate fired; the broad state was delivered
    #: and the run must not count toward the B1-vs-B2 contrast.
    selection_gate_failed: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_index": self.run_index,
            "anchor_table": self.anchor_table,
            "anchor_row_id": self.anchor_row_id,
            "anchor_strategy": self.anchor_strategy,
            "route_status": self.route_status,
            "structurally_valid": self.structurally_valid,
            "retrieved_tables": self.retrieved_tables,
            "retrieved_table_names": list(self.retrieved_table_names),
            "retrieved_records": self.retrieved_records,
            "retrieved_tokens": self.retrieved_tokens,
            "delivered": self.delivered,
            "precision": self.precision,
            "recall": self.recall,
            "precision_granularity": self.precision_granularity,
            "error": self.error,
            "selection_gate_failed": self.selection_gate_failed,
        }


def _anchor_fields(telemetry: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract anchor identity, tolerating the flat telemetry shape."""
    anchor = telemetry.get("anchor")
    if isinstance(anchor, Mapping):
        return {
            "anchor_table": anchor.get("table"),
            "anchor_row_id": anchor.get("row_id"),
            "anchor_strategy": str(anchor.get("strategy", "")),
        }
    return {
        "anchor_table": telemetry.get("anchor_table"),
        "anchor_row_id": telemetry.get("anchor_row_id"),
        "anchor_strategy": str(telemetry.get("anchor_strategy", "")),
    }



def diagnose_run(
    record: TaskRunRecord,
    relevance: Optional[RelevanceSet] = None,
) -> Optional[StateDiagnostic]:
    """Build one task's delivery-chain diagnostic from a persisted record.

    Args:
        record: A Condition-B run record.
        relevance: Optional initial-state prompt-entity relevance set. Supplying it adds precision
            and recall. It is derived from verifier SQL and is therefore used
            for reporting only, never for the intervention.

    Returns:
        The diagnostic, or ``None`` when the record carries no state telemetry
        (for example a Condition-A record, or a run that failed before retrieval).
    """
    telemetry = record.state_retrieval
    if not telemetry:
        return None

    route_status = str(telemetry.get("state_route_status", ""))
    table_names = tuple(
        str(name) for name in (telemetry.get("retrieved_table_names") or ())
    )
    anchor = _anchor_fields(telemetry)

    precision: Optional[float] = None
    recall: Optional[float] = None
    granularity = "n/a"
    if relevance is not None and relevance.confidence != "empty":
        retrieved_rows = {
            (str(table), str(row_id))
            for table, row_id in (telemetry.get("retrieved_rows") or [])
            if isinstance((table, row_id), (list, tuple)) and len((table, row_id)) == 2
        }
        if relevance.rows:
            granularity = "row"
            relevant = set(relevance.rows)
            retrieved = retrieved_rows
        else:
            granularity = "table"
            relevant = set(relevance.tables)
            retrieved = set(table_names)
        overlap = len(retrieved & relevant)
        precision = (overlap / len(retrieved)) if retrieved else None
        recall = (overlap / len(relevant)) if relevant else None

    return StateDiagnostic(
        task_id=record.task_id,
        run_index=record.run_index,
        anchor_table=anchor["anchor_table"],
        anchor_row_id=anchor["anchor_row_id"],
        anchor_strategy=anchor["anchor_strategy"],
        route_status=route_status,
        structurally_valid=route_status in VALID_ROUTE_STATUSES,
        retrieved_tables=int(telemetry.get("retrieved_tables", 0) or 0),
        retrieved_table_names=table_names,
        retrieved_records=int(telemetry.get("retrieved_records", 0) or 0),
        retrieved_tokens=int(telemetry.get("context_tokens_injected", 0) or 0),
        delivered=state_model_delivered(record),
        precision=precision,
        recall=recall,
        precision_granularity=granularity,
        error=record.error_message or telemetry.get("state_error"),
        selection_gate_failed=bool(telemetry.get("selection_gate_failed", False)),
    )



def build_state_report(
    records: Sequence[TaskRunRecord],
    relevance_map: Optional[Mapping[str, RelevanceSet]] = None,
    *,
    condition_name: str,
) -> Dict[str, Any]:
    """Aggregate the Phase 2 diagnostic across every Condition-B run.

    Args:
        records: All persisted run records.
        relevance_map: Optional initial-state prompt-entity relevance sets, keyed by task id.
        condition_name: The treatment condition to report on.

    Returns:
        A JSON-serialisable report carrying the per-task rows, the delivery
        rate, mean precision/recall, and an explicit pass/fail against the
        engineering targets.
    """
    relevance_lookup = relevance_map or {}
    diagnostics: List[StateDiagnostic] = []
    runs_seen = 0
    for record in records:
        if record.condition != condition_name:
            continue
        runs_seen += 1
        diagnostic = diagnose_run(record, relevance_lookup.get(record.task_id))
        if diagnostic is not None:
            diagnostics.append(diagnostic)

    diagnostics.sort(key=lambda item: (item.task_id, item.run_index))
    delivered = sum(1 for item in diagnostics if item.delivered)
    total = len(diagnostics)

    def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
        collected = [value for value in values if value is not None]
        return (sum(collected) / len(collected)) if collected else None

    delivery_rate = (delivered / total) if total else 0.0
    mean_precision = _mean([item.precision for item in diagnostics])
    mean_recall = _mean([item.recall for item in diagnostics])
    delivered_tokens = [item.retrieved_tokens for item in diagnostics if item.delivered]
    mean_retrieved_tokens = (
        (sum(delivered_tokens) / len(delivered_tokens)) if delivered_tokens else None
    )

    # P5.1/A2: a run is *contrast-valid* only when it delivered the minimal
    # state without the gate firing (a gate-fired run delivered the broad
    # state, i.e. it is B1-equivalent and must not count toward B2).
    delivered_items = [item for item in diagnostics if item.delivered]
    gate_failed_delivered = sum(1 for item in delivered_items if item.selection_gate_failed)
    contrast_valid = (
        bool(delivered_items) and gate_failed_delivered == 0
        and delivered_items[0].delivered
        and all(not item.selection_gate_failed for item in delivered_items)
    )

    return {
        "condition": condition_name,
        "runs_seen": runs_seen,
        "runs_with_telemetry": total,
        "delivered": delivered,
        "delivery_rate": delivery_rate,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_retrieved_tokens": mean_retrieved_tokens,
        "selection_gate_failed_delivered": gate_failed_delivered,
        "contrast_valid": contrast_valid,
        "anchors_resolved": sum(1 for item in diagnostics if item.anchor_table),
        "routes_structurally_valid": sum(1 for item in diagnostics if item.structurally_valid),
        "targets": {
            "delivery": {
                "target": DELIVERY_TARGET,
                "observed": delivery_rate,
                "met": delivery_rate >= DELIVERY_TARGET,
            },
            "contrast": {
                "target": CONTRAST_TARGET,
                "observed": (1.0 if contrast_valid else 0.0),
                "met": contrast_valid,
            },
            "precision": {
                "target": PRECISION_TARGET,
                "observed": mean_precision,
                "met": mean_precision is not None and mean_precision >= PRECISION_TARGET,
            },
            "recall": {
                "target": RECALL_TARGET,
                "observed": mean_recall,
                "met": mean_recall is not None and mean_recall >= RECALL_TARGET,
            },
        },
        "per_task": [item.as_dict() for item in diagnostics],
    }


def evaluate_targets(report: Mapping[str, Any]) -> Dict[str, bool]:
    """Return the pass/fail verdict for each engineering target.

    Targets gate the *pipeline*, not the research claim. A failed target means
    "fix the state model before measuring agents", which is precisely the
    ordering the blueprint insists on.
    """
    targets = report.get("targets") or {}
    return {
        name: bool((targets.get(name) or {}).get("met", False))
        for name in ("delivery", "contrast", "precision", "recall")
    }


def render_state_report_markdown(report: Mapping[str, Any]) -> str:
    """Render the blueprint's per-task diagnostic table as Markdown."""
    def _cell(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines = [
        "| " + " | ".join(REPORT_COLUMNS) + " |",
        "|" + "|".join(["---"] * len(REPORT_COLUMNS)) + "|",
    ]
    for item in report.get("per_task") or ():
        anchor = (
            f"{item['anchor_table']}:{item['anchor_row_id']}"
            if item.get("anchor_table")
            else "unresolved"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("task_id", ""))[-12:],
                    anchor,
                    str(item.get("route_status", "") or "n/a"),
                    _cell(item.get("retrieved_tables")),
                    _cell(item.get("retrieved_records")),
                    _cell(item.get("delivered")),
                    _cell(item.get("precision")),
                    _cell(item.get("recall")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)