from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .models import (
    ColumnSpec,
    ForeignKeySpec,
    SchemaCompatibility,
    SchemaManifest,
    TableSpec,
)

_MAJOR_INDEX = 0


class SchemaRegistry:
    """Indexed lookup over a validated schema manifest.

    Duplicate table names, primary keys, or FK definitions are rejected at
    construction time so downstream phases never see ambiguous metadata.
    """

    def __init__(self, manifest: SchemaManifest) -> None:
        self.manifest = manifest
        self._tables: Dict[str, TableSpec] = {}
        self._node_types: Dict[str, TableSpec] = {}
        self._columns: Dict[str, Dict[str, ColumnSpec]] = {}
        self._outgoing: Dict[str, List[ForeignKeySpec]] = {}
        self._incoming: Dict[str, List[ForeignKeySpec]] = {}
        self._fk_by_identity: Dict[Tuple[str, str, str, str], ForeignKeySpec] = {}
        self._fk_by_relation: Dict[Tuple[str, str, str], ForeignKeySpec] = {}

        for spec in manifest.tables:
            key = spec.table.lower()
            if key in self._tables:
                raise ValueError(f"Duplicate table definition: {spec.table}")
            if spec.primary_key not in spec.column_names():
                raise ValueError(
                    f"Table {spec.table}: primary key {spec.primary_key!r} is not a declared column"
                )
            self._tables[key] = spec
            self._node_types[spec.node_type] = spec
            self._columns[key] = {column.name: column for column in spec.columns}

        for fk in manifest.foreign_keys:
            identity = (fk.source_table, fk.source_column, fk.target_table, fk.target_column)
            if identity in self._fk_by_identity:
                raise ValueError(f"Duplicate FK definition: {identity}")
            relation_key = (fk.source_table, fk.source_column, fk.relation)
            if relation_key in self._fk_by_relation:
                raise ValueError(f"Duplicate relation label for FK: {relation_key}")
            self._fk_by_identity[identity] = fk
            self._fk_by_relation[relation_key] = fk
            self._outgoing.setdefault(fk.source_table, []).append(fk)
            self._incoming.setdefault(fk.target_table, []).append(fk)

    @classmethod
    def from_static(cls) -> "SchemaRegistry":
        from .csm_schema import MANIFEST

        return cls(MANIFEST)

    def compatibility(self, requested_version: str) -> SchemaCompatibility:
        major = manifest_major(self.manifest.schema_version)
        requested_major = manifest_major(requested_version)
        return SchemaCompatibility(
            requested_version=requested_version,
            manifest_version=self.manifest.schema_version,
            compatible=major == requested_major,
        )

    def tables(self) -> Tuple[str, ...]:
        return tuple(sorted(self._tables))

    def table(self, name: str) -> Optional[TableSpec]:
        return self._tables.get(name.lower())

    def require_table(self, name: str) -> TableSpec:
        spec = self._tables.get(name.lower())
        if spec is None:
            raise KeyError(f"Unknown table: {name}")
        return spec

    def require_column(self, table: str, column: str) -> ColumnSpec:
        table_spec = self.require_table(table)
        column_spec = self._columns[table_spec.table].get(column)
        if column_spec is None:
            raise KeyError(f"Unknown column {column!r} for table {table_spec.table}")
        return column_spec

    def outgoing_fks(self, table: str) -> Tuple[ForeignKeySpec, ...]:
        return tuple(self._outgoing.get(table.lower(), ()))

    def incoming_fks(self, table: str) -> Tuple[ForeignKeySpec, ...]:
        return tuple(self._incoming.get(table.lower(), ()))

    def foreign_keys(self) -> Tuple[ForeignKeySpec, ...]:
        return tuple(self.manifest.foreign_keys)


def manifest_major(version: str) -> int:
    part = version.split(".", 1)[0]
    try:
        return int(part)
    except ValueError as exc:
        raise ValueError(f"Invalid schema version: {version!r}") from exc
