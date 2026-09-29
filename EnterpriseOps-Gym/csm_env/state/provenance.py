from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .models import FactProvenance, StateProvenance

# Patterns whose values must never reach diagnostics or serialized output
# (SEC-003). Keys are matched case-insensitively.
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(token|secret|password|authorization|api[_-]?key|credential|cookie)", re.IGNORECASE
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_fact_provenance(
    *,
    database_id: str,
    schema_version: str,
    table: str,
    column: str,
    row_pk: str,
    query_id: str,
    route_id: str,
    observed_at: Optional[str] = None,
) -> FactProvenance:
    """Build complete fact provenance; missing fields are a contract error."""
    missing = [
        name
        for name, value in (
            ("database_id", database_id),
            ("schema_version", schema_version),
            ("table", table),
            ("column", column),
            ("row_pk", row_pk),
            ("query_id", query_id),
            ("route_id", route_id),
            ("observed_at", observed_at or utc_now_iso()),
        )
        if value is None or value == ""
    ]
    if missing:
        raise ValueError(f"Provenance fields missing or empty: {missing}")
    return FactProvenance(
        database_id=database_id,
        schema_version=schema_version,
        table=table,
        column=column,
        row_pk=str(row_pk),
        query_id=query_id,
        route_id=route_id,
        observed_at=observed_at or utc_now_iso(),
    )


def build_state_provenance(
    *,
    database_id: str,
    schema_version: str,
    route_id: str,
    query_ids,
    observed_at: Optional[str] = None,
) -> StateProvenance:
    return StateProvenance(
        database_id=database_id,
        schema_version=schema_version,
        route_id=route_id,
        observed_at=observed_at or utc_now_iso(),
        query_ids=tuple(query_ids),
    )


def redact(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of mapping with sensitive keys removed.

    Redaction is recursive over nested dicts; list values are preserved with
    per-item redaction applied.
    """
    cleaned: Dict[str, Any] = {}
    for key, value in mapping.items():
        if _SENSITIVE_KEY_PATTERN.search(str(key)):
            continue
        if isinstance(value, dict):
            cleaned[key] = redact(value)
        elif isinstance(value, list):
            cleaned[key] = [
                redact(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned
