from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..schema_spec.models import ForeignKeySpec, TableSpec


@dataclass(frozen=True)
class TableNode:
    """Structural node for one database table.

    Holds only schema metadata - never record IDs or row data. Record-level
    data lives in csm_env.graph.EnvironmentGraph, which is a separate,
    derived cache.
    """

    table: str
    primary_key: str
    columns: Tuple[str, ...]
    node_type: str

    @classmethod
    def from_spec(cls, spec: TableSpec) -> "TableNode":
        return cls(
            table=spec.table,
            primary_key=spec.primary_key,
            columns=spec.column_names(),
            node_type=spec.node_type,
        )


@dataclass(frozen=True)
class SchemaEdge:
    """Structural FK edge. Structural metadata is authoritative."""

    edge_id: str
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relation: str

    @classmethod
    def from_spec(cls, fk: ForeignKeySpec) -> "SchemaEdge":
        return cls(
            edge_id=fk.edge_id,
            source_table=fk.source_table,
            source_column=fk.source_column,
            target_table=fk.target_table,
            target_column=fk.target_column,
            relation=fk.relation,
        )


@dataclass(frozen=True)
class SemanticAnnotation:
    """Derived semantic label attached to a structural edge (GUD-003).

    Semantic labels never exist without a structural edge; `edge_id` records
    the lineage so reasoning can always be traced back to the real FK.
    """

    edge_id: str
    label: str
    note: str = ""
