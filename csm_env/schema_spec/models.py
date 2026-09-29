from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple


ColumnType = Literal["string", "int", "float", "bool", "datetime", "text", "unknown"]

COLUMN_TYPES: Tuple[str, ...] = ("string", "int", "float", "bool", "datetime", "text", "unknown")


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    type: ColumnType = "unknown"

    def accepts_runtime_value(self, value: object) -> bool:
        """Return whether a DB-returned Python value is compatible with this schema type.

        Identifier columns are intentionally permissive for string/int DB drivers:
        ServiceNow-style IDs are semantically identifiers even when a fixture or
        backend materializes them as integers. NULL is valid for nullable columns.
        """
        if value is None or self.type == "unknown":
            return True
        import datetime as _dt
        if self.type == "bool":
            return isinstance(value, bool)
        if self.type == "int":
            return isinstance(value, int) and not isinstance(value, bool)
        if self.type == "float":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if self.type == "datetime":
            return isinstance(value, (_dt.datetime, _dt.date, str))
        if self.type in ("string", "text"):
            return isinstance(value, str)
        return True


@dataclass(frozen=True)
class TableSpec:
    table: str
    node_type: str
    primary_key: str
    columns: Tuple[ColumnSpec, ...]

    def column_names(self) -> Tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def column(self, name: str) -> Optional[ColumnSpec]:
        for column in self.columns:
            if column.name == name:
                return column
        return None


@dataclass(frozen=True)
class ForeignKeySpec:
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relation: str

    @property
    def edge_id(self) -> str:
        return f"{self.source_table}.{self.source_column}->{self.target_table}.{self.target_column}"


class SchemaSourceKind:
    STATIC_REGISTRY = "static_registry"
    RUNTIME_INTROSPECTION = "runtime_introspection"


@dataclass(frozen=True)
class SchemaSource:
    kind: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.kind not in (SchemaSourceKind.STATIC_REGISTRY, SchemaSourceKind.RUNTIME_INTROSPECTION):
            raise ValueError(f"Unsupported schema source kind: {self.kind!r}")


SchemaSource.STATIC_REGISTRY = SchemaSource(kind=SchemaSourceKind.STATIC_REGISTRY, detail="csm_env.schema")


@dataclass(frozen=True)
class SchemaManifest:
    schema_version: str
    tables: Tuple[TableSpec, ...]
    foreign_keys: Tuple[ForeignKeySpec, ...]
    source: SchemaSource = field(default_factory=lambda: SchemaSource.STATIC_REGISTRY)
    derived_from: str = ""

    def table(self, name: str) -> Optional[TableSpec]:
        for spec in self.tables:
            if spec.table == name:
                return spec
        return None


@dataclass(frozen=True)
class SchemaCompatibility:
    requested_version: str
    manifest_version: str
    compatible: bool
