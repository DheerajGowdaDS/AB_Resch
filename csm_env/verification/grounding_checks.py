from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ..schema_spec.registry import SchemaRegistry
from ..state.consistency import check_state
from ..state.models import GroundedState
from .oracle import DirectSqlOracle
from .schema_checks import CheckResult


async def check_v7_identifier_propagation(
    registry: SchemaRegistry,
    oracle: DirectSqlOracle,
    state: GroundedState,
    fk,
) -> CheckResult:
    """V7: every propagated identifier equals ID_next = DB(row, FK).

    For each relation derived from fk, the oracle independently reads the
    source row and confirms the FK value matches the observed target id.
    """
    failures: List[str] = []
    for relation in state.relations:
        if relation.edge_id != fk.edge_id:
            continue
        source_record = state.record(relation.source_table, relation.source_id)
        if source_record is None:
            failures.append(f"relation source {relation.source_id} not in state")
            continue
        db_row = await oracle.get_row(relation.source_table, relation.source_id)
        if db_row is None:
            failures.append(
                f"oracle cannot find source row {relation.source_table}:{relation.source_id}"
            )
            continue
        db_value = db_row.get(fk.source_column)
        if str(db_value) != str(relation.target_id):
            failures.append(
                f"propagated target {relation.target_id!r} != DB {fk.source_column}={db_value!r}"
            )
    return CheckResult("V7", not failures, failures)


def check_v8_fact_grounding(state: GroundedState, required_fact_count: int = 0) -> CheckResult:
    """V8: every fact carries complete provenance; no value exists without a source row."""
    failures: List[str] = []
    mandatory = (
        "database_id", "schema_version", "table", "column", "row_pk", "query_id", "route_id", "observed_at",
    )
    for fact in state.facts:
        for field_name in mandatory:
            value = getattr(fact.provenance, field_name, None)
            if value is None or value == "":
                failures.append(
                    f"fact {fact.provenance.table}.{fact.provenance.column} "
                    f"missing provenance field {field_name}"
                )
    if required_fact_count and len(state.facts) < required_fact_count:
        failures.append(
            f"expected at least {required_fact_count} grounded facts, found {len(state.facts)}"
        )
    return CheckResult("V8", not failures, failures)


def check_v9_state_consistency(state: GroundedState, registry: SchemaRegistry) -> CheckResult:
    """V9: state consistency via the dedicated consistency checker."""
    report = check_state(state, registry)
    return CheckResult("V9", report.consistent, list(report.problems))


def check_v10_freshness(
    state_after: GroundedState,
    registry: SchemaRegistry,
    oracle: DirectSqlOracle,
    changed_row: Dict[str, Any],
) -> CheckResult:
    """V10: a rebuilt state after a DB modification reflects the new value.

    `changed_row` is the independently-read current row for the anchor
    (oracle-read). The state must agree with it on every shared column.
    """
    failures: List[str] = []
    anchor = state_after.anchor
    if not anchor.values:
        return CheckResult("V10", False, ["anchor has no values to compare"])
    for column, expected in changed_row.items():
        if column not in anchor.values:
            continue
        actual = anchor.values.get(column)
        if actual != expected and str(actual) != str(expected):
            failures.append(
                f"anchor {anchor.table}:{anchor.row_id}.{column}: state {actual!r} != DB {expected!r}"
            )
    if state_after.observation_generation < 1:
        failures.append("state has no observation generation")
    return CheckResult("V10", not failures, failures)
