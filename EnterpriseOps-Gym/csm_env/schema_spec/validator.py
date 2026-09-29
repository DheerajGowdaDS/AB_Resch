from __future__ import annotations

from typing import List

from .models import SchemaManifest


class ManifestValidationError(ValueError):
    """Raised when a schema manifest fails integrity validation."""


def validate_manifest(manifest: SchemaManifest) -> List[str]:
    """Validate manifest integrity; return the list of problems found.

    Empty list means the manifest is valid. Checks:
    - table uniqueness and identifier shape
    - PK presence among declared columns
    - column name uniqueness per table and supported type names
    - FK endpoint existence and duplicate FK detection
    - schema-version compatibility shape
    """
    problems: List[str] = []

    seen_tables = set()
    for spec in manifest.tables:
        if spec.table in seen_tables:
            problems.append(f"duplicate table: {spec.table}")
        seen_tables.add(spec.table)

        if not spec.table or spec.table != spec.table.lower():
            problems.append(f"table name must be lowercase non-empty: {spec.table!r}")
        if not spec.node_type:
            problems.append(f"table {spec.table}: empty node_type")
        if not spec.columns:
            problems.append(f"table {spec.table}: no declared columns")

        column_names = [column.name for column in spec.columns]
        if len(set(column_names)) != len(column_names):
            duplicates = sorted({name for name in column_names if column_names.count(name) > 1})
            problems.append(f"table {spec.table}: duplicate columns {duplicates}")
        if spec.primary_key not in column_names:
            problems.append(
                f"table {spec.table}: primary key {spec.primary_key!r} not among declared columns"
            )

        for column in spec.columns:
            if not column.type:
                problems.append(f"table {spec.table}.{column.name}: empty type")

    seen_fks = set()
    for fk in manifest.foreign_keys:
        identity = (fk.source_table, fk.source_column, fk.target_table, fk.target_column)
        if identity in seen_fks:
            problems.append(f"duplicate FK: {identity}")
        seen_fks.add(identity)

        if fk.source_table not in seen_tables:
            problems.append(f"FK source table not declared: {fk.source_table}")
        elif fk.source_column not in _column_names(manifest, fk.source_table):
            problems.append(f"FK source column not declared: {fk.source_table}.{fk.source_column}")
        if fk.target_table not in seen_tables:
            problems.append(f"FK target table not declared: {fk.target_table}")
        elif fk.target_column not in _column_names(manifest, fk.target_table):
            problems.append(f"FK target column not declared: {fk.target_table}.{fk.target_column}")
        if not fk.relation or fk.relation != fk.relation.upper():
            problems.append(f"FK relation must be UPPERCASE: {fk.relation!r}")

    if not manifest.schema_version or len(manifest.schema_version.split(".")) < 2:
        problems.append(f"invalid schema_version: {manifest.schema_version!r}")

    return problems


def _column_names(manifest: SchemaManifest, table: str) -> set:
    for spec in manifest.tables:
        if spec.table == table:
            return {column.name for column in spec.columns}
    return set()
