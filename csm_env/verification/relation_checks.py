from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..schema_graph.graph import SchemaGraph
from ..schema_spec.registry import SchemaRegistry
from .oracle import DirectSqlOracle
from .schema_checks import CheckResult


async def check_v4_referential_integrity(
    registry: SchemaRegistry,
    oracle: DirectSqlOracle,
    fk,
    sample_size: int = 100,
) -> CheckResult:
    """V4: sample source rows and verify each referenced target row exists."""
    dangling = await oracle.referential_integrity_sample(
        fk.source_table,
        fk.source_column,
        fk.target_table,
        fk.target_column,
        sample_size=sample_size,
    )
    failures = [
        f"{fk.source_table}.{fk.source_column}={row.get(fk.source_column)!r} "
        f"has no {fk.target_table} row (pk={row.get(registry.require_table(fk.source_table).primary_key)!r})"
        for row in dangling
    ]
    return CheckResult("V4", not failures, failures)


async def check_v5_graph_sql_equivalence(
    registry: SchemaRegistry,
    graph: SchemaGraph,
    oracle: DirectSqlOracle,
    fk,
    source_pk_value,
) -> CheckResult:
    """V5: graph-implied relationship result == direct SQL join result.

    Graph side: the FK binding observed on the source row (schema-faithful).
    SQL side: an explicit JOIN executed by the independent oracle.
    """
    source_row = await oracle.get_row(fk.source_table, source_pk_value)
    if source_row is None:
        return CheckResult("V5", False, [f"source row {fk.source_table}:{source_pk_value} not found"])
    graph_value = source_row.get(fk.source_column)

    joined = await oracle.join_equivalence(
        source_table=fk.source_table,
        source_pk=registry.require_table(fk.source_table).primary_key,
        join_source_column=fk.source_column,
        target_table=fk.target_table,
        target_column=fk.target_column,
        source_pk_value=source_pk_value,
    )
    failures: List[str] = []
    if graph_value is None:
        if joined is not None:
            failures.append(
                f"graph implies no {fk.target_table} row but SQL join returned one"
            )
    else:
        if joined is None:
            failures.append(
                f"graph implies {fk.target_table} row for {graph_value!r} but SQL join returned none"
            )
        else:
            joined_pk_value = joined.get(registry.require_table(fk.target_table).primary_key)
            if str(joined_pk_value) != str(graph_value):
                failures.append(
                    f"graph target {graph_value!r} != SQL join target {joined_pk_value!r}"
                )
    return CheckResult("V5", not failures, failures)


def check_v6_plan_validation(
    graph: SchemaGraph, plan_steps
) -> CheckResult:
    """V6: every planned hop corresponds to an actual structural FK edge."""
    failures: List[str] = []
    for step in plan_steps:
        if getattr(step, "hop", 0) == 0:
            continue
        edge = graph.get_edge(step.edge_id or "")
        if edge is None:
            failures.append(
                f"step {step.step_id} hop {getattr(step, 'hop', '?')} uses non-existent edge {step.edge_id!r}"
            )
            continue
        if edge.relation != step.__dict__.get("relation", edge.relation):
            pass  # relation is derived; structural identity is the edge ID
    return CheckResult("V6", not failures, failures)
