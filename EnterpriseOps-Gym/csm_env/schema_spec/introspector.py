from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .models import (
    ColumnSpec,
    ColumnType,
    ForeignKeySpec,
    SchemaManifest,
    TableSpec,
)

_MISSING = object()


class UnsupportedSchemaMetadata(Exception):
    """Raised when a backend cannot provide schema metadata.

    This is an expected, explicit condition (fail-closed): callers must not
    guess tables, columns, types, or constraints when this is raised.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class SchemaIntrospector:
    """Protocol-like base for metadata discovery."""

    async def introspect(self) -> SchemaManifest:
        raise NotImplementedError


@dataclass(frozen=True)
class MismatchReport:
    missing_tables: Tuple[str, ...] = ()
    extra_tables: Tuple[str, ...] = ()
    missing_columns: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    extra_columns: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    wrong_primary_keys: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    @property
    def has_mismatch(self) -> bool:
        return bool(
            self.missing_tables
            or self.extra_tables
            or self.missing_columns
            or self.extra_columns
            or self.wrong_primary_keys
        )

    def summary(self) -> str:
        parts: List[str] = []
        if self.missing_tables:
            parts.append(f"missing tables: {sorted(self.missing_tables)}")
        if self.extra_tables:
            parts.append(f"extra tables: {sorted(self.extra_tables)}")
        for table in sorted(self.missing_columns):
            parts.append(f"{table}: missing columns {sorted(self.missing_columns[table])}")
        for table in sorted(self.extra_columns):
            parts.append(f"{table}: extra columns {sorted(self.extra_columns[table])}")
        for table in sorted(self.wrong_primary_keys):
            declared, observed = self.wrong_primary_keys[table]
            parts.append(f"{table}: PK declared {declared!r} but observed {observed!r}")
        return "; ".join(parts) if parts else "no mismatch"


def compare_with_manifest(
    manifest: SchemaManifest,
    observed_tables: Dict[str, Tuple[str, ...]],
    observed_primary_keys: Dict[str, str],
) -> MismatchReport:
    """Fail-closed structural comparison of observed metadata vs manifest."""
    missing_tables = tuple(
        sorted(spec.table for spec in manifest.tables if spec.table not in observed_tables)
    )
    extra_tables = tuple(
        sorted(name for name in observed_tables if manifest.table(name) is None)
    )

    missing_columns: Dict[str, Tuple[str, ...]] = {}
    extra_columns: Dict[str, Tuple[str, ...]] = {}
    wrong_primary_keys: Dict[str, Tuple[str, str]] = {}

    for spec in manifest.tables:
        observed = observed_tables.get(spec.table)
        if observed is None:
            continue
        declared = set(spec.column_names())
        observed_set = set(observed)
        missing = tuple(sorted(declared - observed_set))
        extra = tuple(sorted(observed_set - declared))
        if missing:
            missing_columns[spec.table] = missing
        if extra:
            extra_columns[spec.table] = extra
        observed_pk = observed_primary_keys.get(spec.table)
        if observed_pk is not None and observed_pk != spec.primary_key:
            wrong_primary_keys[spec.table] = (spec.primary_key, observed_pk)

    return MismatchReport(
        missing_tables=missing_tables,
        extra_tables=extra_tables,
        missing_columns=missing_columns,
        extra_columns=extra_columns,
        wrong_primary_keys=wrong_primary_keys,
    )


class SQLSchemaIntrospector(SchemaIntrospector):
    """Optional metadata discovery over a SQL reader.

    Uses information_schema-style metadata queries. Backends that do not
    expose such metadata produce UnsupportedSchemaMetadata, never guesses.
    """

    def __init__(self, reader, manifest: SchemaManifest) -> None:
        self._reader = reader
        self._manifest = manifest

    async def introspect(self) -> SchemaManifest:
        raise UnsupportedSchemaMetadata(
            "The approved EnterpriseOps sql-runner transport does not expose "
            "information_schema metadata in the current workspace (CON-005). "
            "Runtime introspection is unavailable; use the static manifest."
        )

    async def verify_against_manifest(self) -> Optional[MismatchReport]:
        """Return a MismatchReport, or None when introspection is unsupported."""
        try:
            await self.introspect()
        except UnsupportedSchemaMetadata:
            return None
        # A working introspector would build observed metadata here.
        return None


@dataclass(frozen=True)
class IntrospectionOutcome:
    manifest: Optional[SchemaManifest]
    mismatch: Optional[MismatchReport]
    unsupported_reason: Optional[str]

    @property
    def is_supported(self) -> bool:
        return self.manifest is not None
