from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..query.models import RouteOutcome, StepStatus
from ..schema_spec.registry import SchemaRegistry
from .models import (
    FactProvenance,
    FreshnessInfo,
    GroundedState,
    StateFact,
    StateContradiction,
    StateProvenance,
    StateRecord,
    StateRelation,
    StateStatus,
)
from .provenance import build_fact_provenance, build_state_provenance


class StateBuilder:
    """Aggregates route outcomes into a task-relevant GroundedState.

    Every included fact carries complete provenance; facts without a source
    row are never inferred (from task text or LLM output). NULL database
    values remain typed values inside records; unresolved FKs and
    required-but-unobserved tables are recorded separately.
    """

    def __init__(
        self,
        registry: SchemaRegistry,
        database_id: str = "unknown",
        clock=None,
    ) -> None:
        self._registry = registry
        self._database_id = database_id
        self._clock = clock
        self._generation = 0
        self._fk_by_edge_id = {fk.edge_id: fk for fk in registry.manifest.foreign_keys}

    def build(self, outcome: RouteOutcome) -> GroundedState:
        self._generation += 1
        plan = outcome.plan
        observed_at = self._clock() if self._clock else _default_clock()

        rows_by_step = {result.step_id: result for result in outcome.results}
        records: List[StateRecord] = []
        record_index: Dict[Tuple[str, str], StateRecord] = {}
        relations: List[StateRelation] = []
        facts: List[StateFact] = []
        contradictions: List[StateContradiction] = []
        fact_index: Dict[Tuple[str, str, str], StateFact] = {}

        for step in plan.steps:
            result = rows_by_step.get(step.step_id)
            if result is None or result.status != StepStatus.OK:
                continue
            table_spec = self._registry.require_table(step.table)
            for row in result.rows:
                pk_value = row.get(table_spec.primary_key)
                if pk_value is None:
                    continue
                row_id = str(pk_value)
                key = (step.table, row_id)
                existing_record = record_index.get(key)
                if existing_record is not None:
                    # Duplicate observations are allowed only when values agree.
                    # Conflicts are explicit research signals; never silently pick
                    # one observation over the other.
                    # Compare the union of observed columns. A later observation
                    # may reveal a field omitted by the first query; it must be
                    # added to the record rather than silently discarded.
                    merged_values = dict(existing_record.values)
                    for column in set(existing_record.values) | set(row):
                        if column not in existing_record.values:
                            merged_values[column] = row[column]
                            continue
                        if column not in row:
                            continue
                        previous = existing_record.values[column]
                        if self._values_equivalent(previous, row[column], table_spec, column):
                            continue
                        previous_fact = fact_index.get((step.table, row_id, column))
                        contradictions.append(
                            StateContradiction(
                                table=step.table,
                                row_id=row_id,
                                column=column,
                                first_value=previous,
                                second_value=row[column],
                                first_query_id=previous_fact.provenance.query_id if previous_fact else "",
                                second_query_id=step.step_id,
                            )
                        )
                    record_index[key] = StateRecord(
                        table=existing_record.table,
                        primary_key=existing_record.primary_key,
                        row_id=existing_record.row_id,
                        values=merged_values,
                    )
                    records[:] = [record_index.get((r.table, r.row_id), r) for r in records]
                    # Preserve provenance for newly observed columns.
                    for column in row:
                        if (step.table, row_id, column) in fact_index:
                            continue
                        provenance = build_fact_provenance(
                            database_id=self._database_id,
                            schema_version=self._registry.manifest.schema_version,
                            table=step.table, column=column, row_pk=row_id,
                            query_id=step.step_id, route_id=plan.route_id, observed_at=observed_at,
                        )
                        fact = StateFact(value=row[column], provenance=provenance)
                        facts.append(fact)
                        fact_index[(step.table, row_id, column)] = fact
                    continue

                record = StateRecord(
                    table=step.table,
                    primary_key=table_spec.primary_key,
                    row_id=row_id,
                    values=dict(row),
                )
                record_index[key] = record
                records.append(record)

                for column in table_spec.column_names():
                    if column not in row:
                        continue
                    provenance = build_fact_provenance(
                        database_id=self._database_id,
                        schema_version=self._registry.manifest.schema_version,
                        table=step.table,
                        column=column,
                        row_pk=row_id,
                        query_id=step.step_id,
                        route_id=plan.route_id,
                        observed_at=observed_at,
                    )
                    fact = StateFact(value=row[column], provenance=provenance)
                    facts.append(fact)
                    fact_index[(step.table, row_id, column)] = fact

                # Typed relation to the dependency row via the structural FK
                # this step followed (either direction), preserving lineage.
                if step.edge_id and step.depends_on:
                    fk = self._fk_by_edge_id.get(step.edge_id)
                    parent_step = plan.step(step.depends_on)
                    if fk is not None and parent_step is not None:
                        self._relate(fk, parent_step, step, row, table_spec, rows_by_step, relations)

        state_provenance = build_state_provenance(
            database_id=self._database_id,
            schema_version=self._registry.manifest.schema_version,
            route_id=plan.route_id,
            query_ids=tuple(sorted({result.step_id for result in outcome.results})),
            observed_at=observed_at,
        )

        unresolved_details = tuple(
            (unresolved.table, unresolved.source_column or "", unresolved.reason)
            for unresolved in outcome.unresolved
        )
        unresolved_ids = tuple(sorted({item[0] for item in unresolved_details}))

        status = self._status(outcome, record_index, unresolved_details, plan, contradictions)

        state_id = (
            f"{plan.anchor_table}:{plan.anchor_value}"
            f"@{self._registry.manifest.schema_version}"
        )
        return GroundedState(
            state_id=state_id,
            anchor=record_index.get(
                (plan.anchor_table, str(plan.anchor_value)),
                StateRecord(
                    table=plan.anchor_table,
                    primary_key=self._registry.require_table(plan.anchor_table).primary_key,
                    row_id=str(plan.anchor_value),
                    values={},
                ),
            ),
            records=tuple(sorted(records, key=lambda r: (r.table, r.row_id))),
            relations=tuple(sorted(set(relations), key=lambda r: r.identity())),
            unresolved=unresolved_ids,
            unresolved_details=unresolved_details,
            provenance=state_provenance,
            status=status,
            schema_version=self._registry.manifest.schema_version,
            observation_generation=self._generation,
            facts=tuple(facts),
            contradictions=tuple(contradictions),
        )

    def freshness(self, state: GroundedState) -> FreshnessInfo:
        return FreshnessInfo(
            observation_generation=state.observation_generation,
            observed_at=state.provenance.observed_at,
            database_id=state.provenance.database_id,
            schema_version=state.provenance.schema_version,
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _relate(
        self,
        fk,
        parent_step,
        child_step,
        child_row: Dict[str, Any],
        child_spec,
        rows_by_step,
        relations: List[StateRelation],
    ) -> None:
        """Create StateRelations between child rows and dependency rows.

        For child==fk.source_table: child carries fk.source_column and the
        parent carries fk.target_column. For child==fk.target_table the
        columns swap. Relation direction always follows the FK definition:
        source_table -> target_table.
        """
        parent_result = rows_by_step.get(parent_step.step_id)
        if parent_result is None:
            return
        parent_spec = self._registry.require_table(parent_step.table)

        child_is_source = fk.source_table == child_step.table
        child_column = fk.source_column if child_is_source else fk.target_column
        parent_column = fk.target_column if child_is_source else fk.source_column
        if child_column not in child_spec.column_names() or parent_column not in parent_spec.column_names():
            return

        child_pk_value = child_row.get(child_spec.primary_key)
        if child_pk_value is None:
            return
        child_fk_value = child_row.get(child_column)
        if child_fk_value is None:
            return

        source_table = fk.source_table
        target_table = fk.target_table
        source_id = str(child_pk_value) if child_is_source else None
        target_id = str(child_pk_value) if not child_is_source else None

        for parent_row in parent_result.rows:
            parent_pk_value = parent_row.get(parent_spec.primary_key)
            if parent_pk_value is None:
                continue
            if str(parent_row.get(parent_column)) != str(child_fk_value):
                continue
            relations.append(
                StateRelation(
                    source_table=source_table,
                    source_id=source_id if source_id is not None else str(parent_pk_value),
                    target_table=target_table,
                    target_id=target_id if target_id is not None else str(parent_pk_value),
                    relation=fk.relation,
                    edge_id=fk.edge_id,
                )
            )

    @staticmethod
    def _values_equivalent(previous: Any, current: Any, table_spec, column: str) -> bool:
        if previous is None or current is None:
            return previous is current
        if type(previous) is type(current):
            return previous == current
        # Allow only documented identifier materialization differences.
        spec = next((c for c in table_spec.columns if c.name == column), None)
        if spec is not None and spec.type_name in {"string", "integer"}:
            return str(previous) == str(current)
        return False

    def _status(
        self,
        outcome: RouteOutcome,
        record_index: Dict[Tuple[str, str], StateRecord],
        unresolved_details: Tuple[Tuple[str, str, str], ...],
        plan,
        contradictions: List[StateContradiction],
    ) -> str:
        # The route status is already requirement-aware (only unresolved
        # bindings for REQUIRED tables downgrade to UNRESOLVED). Non-required
        # NULL FKs stay recorded in unresolved_details without failing the
        # state; errors always propagate.
        if any(result.status == "error" for result in outcome.results):
            return StateStatus.FAILED
        # Builder-detected contradictions are fatal for a grounded state.
        # Do not silently resolve conflicting observations.
        if contradictions:
            return StateStatus.FAILED
        return outcome.status


def _default_clock():
    from .provenance import utc_now_iso

    return utc_now_iso()
