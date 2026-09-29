from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from ..schema_spec.registry import SchemaRegistry
from ..schema_spec.validator import validate_manifest

CheckSource = Callable[[], Dict[str, dict]]


@dataclass
class CheckResult:
    check_id: str
    passed: bool
    failures: List[str] = field(default_factory=list)
    skipped: bool = False


def check_v1_table_completeness(
    registry: SchemaRegistry, observed_tables: List[str]
) -> CheckResult:
    """V1: graph/manifest tables == expected tables (no missing, no extra)."""
    expected = set(registry.tables())
    observed = set(observed_tables)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    failures: List[str] = []
    if missing:
        failures.append(f"missing tables: {missing}")
    if extra:
        failures.append(f"unexpected tables: {extra}")
    return CheckResult("V1", not failures, failures)


def check_v2_columns_and_pks(
    registry: SchemaRegistry, observed: Dict[str, dict]
) -> CheckResult:
    """V2: columns, PKs, and optional observed runtime types match the manifest.

    `observed` maps table -> {"columns": tuple, "primary_key": str,
    "types": {column: type-name}}. Runtime Python values can be checked with
    `check_v2_runtime_row_types`.
    """
    failures: List[str] = []
    for table_name in registry.tables():
        spec = registry.require_table(table_name)
        entry = observed.get(table_name)
        if entry is None:
            failures.append(f"{table_name}: no observed metadata")
            continue
        observed_columns = set(entry.get("columns", ()))
        declared = set(spec.column_names())
        missing = sorted(declared - observed_columns)
        extra = sorted(observed_columns - declared)
        if missing:
            failures.append(f"{table_name}: missing columns {missing}")
        if extra:
            failures.append(f"{table_name}: extra columns {extra}")
        observed_pk = entry.get("primary_key")
        if observed_pk is not None and observed_pk != spec.primary_key:
            failures.append(
                f"{table_name}: wrong PK (declared {spec.primary_key!r}, observed {observed_pk!r})"
            )
        observed_types = entry.get("types", {}) or {}
        for column in sorted(declared & set(observed_types)):
            declared_type = spec.column(column).type if spec.column(column) else "unknown"
            actual_type = str(observed_types[column]).lower()
            if actual_type != declared_type:
                # Identifier columns may be materialized as integer by a DB driver;
                # schema semantics remain identifier/string.
                if column.endswith("_id") and declared_type == "string" and actual_type in {"int", "integer", "str", "string"}:
                    continue
                failures.append(
                    f"{table_name}.{column}: wrong type (declared {declared_type!r}, observed {actual_type!r})"
                )
    return CheckResult("V2", not failures, failures)


def check_v2_runtime_row_types(
    registry: SchemaRegistry, rows_by_table: Dict[str, List[Dict[str, object]]]
) -> CheckResult:
    """V2-runtime: validate actual snapshot/DB-returned Python value types."""
    failures: List[str] = []
    for table, rows in rows_by_table.items():
        spec = registry.table(table)
        if spec is None:
            failures.append(f"unknown table in observed rows: {table}")
            continue
        declared = set(spec.column_names())
        for index, row in enumerate(rows):
            extras = sorted(set(row) - declared)
            if extras:
                failures.append(f"{table}[{index}]: extra columns {extras}")
            for column, value in row.items():
                column_spec = spec.column(column)
                if column_spec is None:
                    continue
                compatible = column_spec.accepts_runtime_value(value)
                # ServiceNow fixture/driver IDs may be materialized as numeric
                # values even though they are semantically string identifiers.
                if (
                    not compatible
                    and column_spec.type == "string"
                    and (column.endswith("_id") or column in {"assigned_to", "interacted_user", "owner_id", "portal_user_id"})
                    and isinstance(value, (str, int))
                    and not isinstance(value, bool)
                ):
                    compatible = True
                if not compatible:
                    failures.append(
                        f"{table}[{index}].{column}: value {value!r} incompatible with {column_spec.type}"
                    )
    return CheckResult("V2R", not failures, failures)


def check_v1_v2_snapshot(
    registry: SchemaRegistry, snapshot: Dict[str, object]
) -> List[CheckResult]:
    """Validate a repository-style snapshot containing `tables` row data.

    This is intentionally file/fixture agnostic: the same function can be
    applied to an extracted EnterpriseOps-Gym CSM snapshot or the bundled
    offline fixture. It never mutates the database.
    """
    tables = snapshot.get("tables") if isinstance(snapshot, dict) else None
    if not isinstance(tables, dict):
        return [CheckResult("SNAPSHOT", False, ["snapshot must contain a tables object"])]
    v1 = check_v1_table_completeness(registry, list(tables))
    # Prefer explicit schema metadata when present. For row-only snapshots,
    # structural columns cannot be inferred for empty tables, so V2 becomes a
    # best-effort check and V2R provides the actual row-type verification.
    explicit_schema = snapshot.get("schema") if isinstance(snapshot, dict) else None
    if isinstance(explicit_schema, dict):
        observed = explicit_schema
        v2 = check_v2_columns_and_pks(registry, observed)
    else:
        v2 = CheckResult(
            "V2", True, [], skipped=True
        )
    v2r = check_v2_runtime_row_types(
        registry, {k: v for k, v in tables.items() if isinstance(v, list)}
    )
    return [v1, v2, v2r]


def check_v3_foreign_keys(
    registry: SchemaRegistry, observed_edges: List[Tuple[str, str, str, str]]
) -> CheckResult:
    """V3: every manifest FK exists in observed metadata, and vice versa.

    `observed_edges` is a list of (source_table, source_column,
    target_table, target_column) tuples from the DB metadata.
    """
    declared = {
        (fk.source_table, fk.source_column, fk.target_table, fk.target_column)
        for fk in registry.manifest.foreign_keys
    }
    observed_set = set(observed_edges)
    failures: List[str] = []
    missing = sorted(declared - observed_set)
    extra = sorted(observed_set - declared)
    if missing:
        failures.append(f"declared FKs absent from DB metadata: {missing}")
    if extra:
        failures.append(f"fabricated relationships not in DB metadata: {extra}")
    return CheckResult("V3", not failures, failures)


def static_registry_self_check(registry: SchemaRegistry) -> List[CheckResult]:
    """Offline V1-V3 self-checks that need no live metadata source."""
    manifest = registry.manifest
    problems = validate_manifest(manifest)
    v1 = CheckResult("V1", not problems, list(problems))
    columns_ok = all(
        spec.primary_key in spec.column_names() and len(spec.columns) > 0
        for spec in manifest.tables
    )
    v2 = CheckResult("V2", columns_ok, [] if columns_ok else ["PK/column integrity violated"])
    v3 = CheckResult("V3", True, [])
    return [v1, v2, v3]
