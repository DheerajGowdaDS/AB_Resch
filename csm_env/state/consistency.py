from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..schema_spec.registry import SchemaRegistry
from .models import GroundedState


@dataclass(frozen=True)
class ConsistencyReport:
    consistent: bool
    problems: Tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.consistent


def check_state(state: GroundedState, registry: SchemaRegistry) -> ConsistencyReport:
    """Check a grounded state for internal contradictions (V9).

    Contradictions are reported; they are never silently resolved.
    Checks: returned-row binding agreement, relation endpoint agreement,
    duplicate PK conflicts, and anchor consistency.
    """
    problems: List[str] = []

    records = {(record.table, record.row_id): record for record in state.records}
    if state.contradictions:
        for contradiction in state.contradictions:
            problems.append(
                f"contradiction {contradiction.table}:{contradiction.row_id}.{contradiction.column}: "
                f"{contradiction.first_value!r} vs {contradiction.second_value!r}"
            )
    if len(records) != len(state.records):
        problems.append("duplicate primary keys present in records")

    # Relation endpoints must exist in the record set.
    for relation in state.relations:
        source_key = (relation.source_table, relation.source_id)
        target_key = (relation.target_table, relation.target_id)
        if source_key not in records:
            problems.append(
                f"relation {relation.edge_id} source endpoint {source_key} not observed"
            )
        if target_key not in records:
            problems.append(
                f"relation {relation.edge_id} target endpoint {target_key} not observed"
            )

    # Binding agreement: NULL FK values must be recorded unresolved; a
    # referenced row outside the activated subset is a coverage gap, not a
    # contradiction (states are task-relevant, never full snapshots).
    unresolved_keys = set(state.unresolved)
    for record in state.records:
        table_spec = registry.require_table(record.table)
        for fk in registry.outgoing_fks(record.table):
            value = record.values.get(fk.source_column)
            if value is None and fk.target_table not in unresolved_keys:
                problems.append(
                    f"{record.table}:{record.row_id}.{fk.source_column} is NULL but "
                    f"{fk.target_table} is not recorded unresolved"
                )

    # Anchor must be present.
    anchor_key = (state.anchor.table, state.anchor.row_id)
    if state.anchor.values and anchor_key not in records:
        problems.append(f"anchor {anchor_key} has values but is not among records")

    return ConsistencyReport(consistent=not problems, problems=tuple(problems))
