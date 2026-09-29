"""Read-only initial-state fingerprints for a seeded CSM database.

A paired experiment is only valid if both conditions start from the *same*
initial state. Rather than trusting that the benchmark re-seeds correctly, the
harness fingerprints the database itself and compares the two fingerprints
before accepting a pair.

The fingerprint covers every table in the real ``csm_env`` schema registry:
for each table it hashes the table name, the row count, and the ordered primary
key values. It is deterministic, bounded, and strictly read-only.

Scope note: the digest proves *structural* equality (same tables, same row
counts, same primary keys). Two databases that differ only in non-key column
values would share a fingerprint. This is acceptable for the ablation because
both arms seed from the same pinned SQL file, but the digest should be read as
a structure check, not a full content hash.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from csm_env.schema_spec.registry import SchemaRegistry
from csm_env.transport.reader import SQLReader

#: Per-table cap on hashed primary keys. Beyond this an explicit truncation
#: marker is folded into the hash so the fingerprint stays comparable.
FINGERPRINT_MAX_ROWS_PER_TABLE = 5000

#: Marker folded into the digest when a table exceeds the row cap.
TRUNCATION_MARKER = "|truncated|"


def _assert_safe_identifier(identifier: str, registry: SchemaRegistry, table: str) -> None:
    """Fail closed unless the identifier is declared by the schema registry."""
    if identifier not in registry.require_table(table).column_names():
        raise KeyError(
            f"Refusing to interpolate {identifier!r}: not a declared column of {table!r}"
        )


def _quote_literal(value: Any) -> str:
    """Render a primary-key value as a SQL literal."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"


async def table_fingerprint(
    reader: SQLReader,
    registry: SchemaRegistry,
    table: str,
    *,
    max_rows_per_table: int = FINGERPRINT_MAX_ROWS_PER_TABLE,
) -> Dict[str, Any]:
    """Fingerprint one table: row count plus ordered primary keys.

    Args:
        reader: Transport used for the read-only queries.
        registry: Schema registry; supplies validated table/column identifiers.
        table: Table name, which must exist in the registry.
        max_rows_per_table: Cap on hashed primary keys.

    Returns:
        Mapping with ``table``, ``row_count``, ``sampled_keys``, ``truncated``
        and the per-table ``digest``.
    """
    spec = registry.require_table(table)
    primary_key = spec.primary_key
    _assert_safe_identifier(primary_key, registry, spec.table)
    if max_rows_per_table < 1:
        raise ValueError("max_rows_per_table must be >= 1")

    count_rows = await reader.fetch_rows(f"SELECT COUNT(*) AS row_count FROM {spec.table};")
    row_count = 0
    if count_rows and isinstance(count_rows[0], dict):
        raw_count = count_rows[0].get("row_count")
        try:
            row_count = int(raw_count)
        except (TypeError, ValueError):
            row_count = 0

    key_rows: List[Dict[str, Any]] = await reader.fetch_rows(
        f"SELECT {primary_key} FROM {spec.table} ORDER BY {primary_key} "
        f"LIMIT {max_rows_per_table};"
    )
    keys = [row.get(primary_key) for row in key_rows if isinstance(row, dict)]
    truncated = row_count > len(keys)
    hasher = hashlib.sha256()
    hasher.update(spec.table.encode("utf-8"))
    hasher.update(b"\x1f")
    hasher.update(str(row_count).encode("utf-8"))
    for key in keys:
        hasher.update(b"\x1e")
        hasher.update(_quote_literal(key).encode("utf-8"))
    if truncated:
        hasher.update(TRUNCATION_MARKER.encode("utf-8"))

    return {
        "table": spec.table,
        "row_count": row_count,
        "sampled_keys": len(keys),
        "truncated": truncated,
        "digest": hasher.hexdigest(),
    }


async def initial_state_fingerprint(
    reader: SQLReader,
    registry: SchemaRegistry,
    *,
    tables: Optional[List[str]] = None,
    max_rows_per_table: int = FINGERPRINT_MAX_ROWS_PER_TABLE,
) -> str:
    """Return a deterministic SHA-256 fingerprint of database state.

    Tables default to every table in the registry, sorted, so the digest is
    independent of declaration order.

    Raises:
        ValueError: If ``max_rows_per_table`` is invalid.
        KeyError: If a requested table is not in the registry.
    """
    selected = list(registry.tables()) if tables is None else list(tables)
    if not selected:
        raise ValueError("fingerprint requires at least one table")
    hasher = hashlib.sha256()
    hasher.update(b"csm-initial-state-fingerprint\x1e")
    for table in sorted({name.lower() for name in selected}):
        report = await table_fingerprint(
            reader, registry, table, max_rows_per_table=max_rows_per_table
        )
        hasher.update(report["table"].encode("utf-8"))
        hasher.update(b"\x1d")
        hasher.update(str(report["row_count"]).encode("utf-8"))
        hasher.update(b"\x1d")
        hasher.update(report["digest"].encode("utf-8"))
        hasher.update(b"\x1c")
    return hasher.hexdigest()


async def fingerprint_report(
    reader: SQLReader,
    registry: SchemaRegistry,
    *,
    max_rows_per_table: int = FINGERPRINT_MAX_ROWS_PER_TABLE,
) -> Dict[str, Any]:
    """Fingerprint plus the per-table breakdown, for auditability."""
    per_table = []
    for table in registry.tables():
        per_table.append(
            await table_fingerprint(
                reader, registry, table, max_rows_per_table=max_rows_per_table
            )
        )
    return {
        "fingerprint": await initial_state_fingerprint(
            reader, registry, max_rows_per_table=max_rows_per_table
        ),
        "schema_version": registry.manifest.schema_version,
        "table_count": len(per_table),
        "total_rows": sum(entry["row_count"] for entry in per_table),
        "max_rows_per_table": max_rows_per_table,
        "tables": per_table,
    }
