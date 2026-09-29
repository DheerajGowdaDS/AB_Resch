from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..schema_spec.models import SchemaManifest
from ..transport.reader import SQLReader


class DirectSqlOracle:
    """Independent ground-truth oracle built directly from SQL.

    Independence contract (TASK-039 / GUD-005): this module imports only the
    schema manifest and the raw reader. It must never import SchemaGraph,
    GraphQueryRouter, StateBuilder, or their helpers; verification that used
    the system under test would be circular (RISK-008).

    Queries are explicit, hand-written per check, using manifest-declared
    tables/columns only. Values are escaped through the same single
    literal-rendering rule used everywhere (no user-controlled identifiers).
    """

    def __init__(self, reader: SQLReader, manifest: SchemaManifest, database_id: str = "unknown") -> None:
        self._reader = reader
        self._manifest = manifest
        self.database_id = database_id

    def _spec(self, table: str):
        for candidate in self._manifest.tables:
            if candidate.table == table:
                return candidate
        raise KeyError(f"Unknown table: {table}")

    def _literal(self, value: Any) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return repr(value)
        if isinstance(value, str):
            return "'" + value.replace("'", "''") + "'"
        raise TypeError(f"Unsupported oracle value type: {type(value).__name__}")

    def _select(self, table: str, columns: Optional[Tuple[str, ...]] = None) -> str:
        spec = self._spec(table)
        names = columns if columns is not None else spec.column_names()
        for name in names:
            if name not in spec.column_names():
                raise KeyError(f"Unknown column {name!r} for {table}")
        return f"SELECT {', '.join(names)} FROM {table}"

    async def get_row(self, table: str, pk_value: Any) -> Optional[Dict[str, Any]]:
        spec = self._spec(table)
        query = (
            f"{self._select(table)} WHERE {spec.primary_key} = "
            f"{self._literal(pk_value)} LIMIT 1;"
        )
        rows = await self._reader.fetch_rows(query)
        return rows[0] if rows else None

    async def get_rows_by_fk(
        self, table: str, column: str, value: Any
    ) -> List[Dict[str, Any]]:
        spec = self._spec(table)  # validates table
        if column not in spec.column_names():
            raise KeyError(f"Unknown column {column!r} for {table}")
        query = f"{self._select(table)} WHERE {column} = {self._literal(value)};"
        return await self._reader.fetch_rows(query)

    async def referential_integrity_sample(
        self,
        source_table: str,
        source_column: str,
        target_table: str,
        target_column: str,
        sample_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """V4 helper: sample source rows whose FK value has no target row."""
        source_spec = self._spec(source_table)
        target_spec = self._spec(target_table)
        if source_column not in source_spec.column_names():
            raise KeyError(f"Unknown column {source_column!r} for {source_table}")
        if target_column not in target_spec.column_names():
            raise KeyError(f"Unknown column {target_column!r} for {target_table}")
        query = (
            f"SELECT {source_spec.primary_key}, {source_column} FROM {source_table};"
        )
        rows = await self._reader.fetch_rows(query)
        dangling: List[Dict[str, Any]] = []
        for row in rows[:sample_size]:
            value = row.get(source_column)
            if value is None:
                continue
            target = await self.get_row(target_table, value)
            if target is None:
                dangling.append(row)
        return dangling

    async def join_equivalence(
        self,
        source_table: str,
        source_pk: str,
        join_source_column: str,
        target_table: str,
        target_column: str,
        source_pk_value: Any,
    ) -> Optional[Dict[str, Any]]:
        """V5 helper: direct join result, independent of any graph routing."""
        self._spec(source_table)
        self._spec(target_table)
        query = (
            f"SELECT t.* FROM {source_table} s JOIN {target_table} t "
            f"ON s.{join_source_column} = t.{target_column} "
            f"WHERE s.{source_pk} = {self._literal(source_pk_value)} LIMIT 1;"
        )
        rows = await self._reader.fetch_rows(query)
        return rows[0] if rows else None
