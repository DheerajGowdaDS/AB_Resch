from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


class StateStatus:
    COMPLETE = "COMPLETE"
    UNRESOLVED = "UNRESOLVED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class StateRecord:
    """One observed database row inside a grounded state."""

    table: str
    primary_key: str
    row_id: str
    values: Dict[str, Any] = field(default_factory=dict)

    def identity(self) -> str:
        return f"{self.table}:{self.row_id}"


@dataclass(frozen=True)
class StateRelation:
    """A typed relationship between two observed records.

    `edge_id` preserves lineage to the structural FK that was followed
    (GUD-003); relations pointing at unobserved endpoints are rejected by
    the builder.
    """

    source_table: str
    source_id: str
    target_table: str
    target_id: str
    relation: str
    edge_id: str

    def identity(self) -> Tuple[str, str, str, str, str]:
        return (self.source_table, self.source_id, self.target_table, self.target_id, self.relation)


@dataclass(frozen=True)
class StateProvenance:
    """Shared lineage for one state construction run (approved field set)."""

    database_id: str
    schema_version: str
    route_id: str
    observed_at: str  # UTC ISO-8601, timezone-aware
    query_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class FactProvenance:
    """Per-fact lineage. Required for every database-derived fact."""

    database_id: str
    schema_version: str
    table: str
    column: str
    row_pk: str
    query_id: str
    route_id: str
    observed_at: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "database_id": self.database_id,
            "schema_version": self.schema_version,
            "table": self.table,
            "column": self.column,
            "row_pk": self.row_pk,
            "query_id": self.query_id,
            "route_id": self.route_id,
            "observed_at": self.observed_at,
        }


@dataclass(frozen=True)
class StateFact:
    """One grounded fact: value + mandatory provenance."""

    value: Any
    provenance: FactProvenance


@dataclass(frozen=True)
class StateContradiction:
    """Conflicting observations for the same logical record/field."""

    table: str
    row_id: str
    column: str
    first_value: Any
    second_value: Any
    first_query_id: str
    second_query_id: str


@dataclass(frozen=True)
class GroundedState:
    """Task-relevant, database-derived view. Never a full snapshot (CON-010).

    UNKNOWN is represented by the absence of a fact; missing values are
    never guessed. Unresolved FKs are recorded separately in `unresolved`.
    """

    state_id: str
    anchor: StateRecord
    records: Tuple[StateRecord, ...]
    relations: Tuple[StateRelation, ...]
    unresolved: Tuple[str, ...]
    unresolved_details: Tuple[Tuple[str, str, str], ...]  # (table, column, reason)
    provenance: StateProvenance
    status: str
    schema_version: str
    observation_generation: int = 1
    facts: Tuple[StateFact, ...] = ()
    contradictions: Tuple[StateContradiction, ...] = ()

    def record(self, table: str, row_id: Any) -> Optional[StateRecord]:
        for candidate in self.records:
            if candidate.table == table and candidate.row_id == str(row_id):
                return candidate
        return None

    def fact_value(self, table: str, row_id: Any, column: str) -> Any:
        """Return the grounded value, or None when absent (UNKNOWN)."""
        for fact in self.facts:
            if (
                fact.provenance.table == table
                and fact.provenance.row_pk == str(row_id)
                and fact.provenance.column == column
            ):
                return fact.value
        return None


@dataclass(frozen=True)
class FreshnessInfo:
    """Freshness metadata exposed to callers (blueprint V10)."""

    observation_generation: int
    observed_at: str
    database_id: str
    schema_version: str
