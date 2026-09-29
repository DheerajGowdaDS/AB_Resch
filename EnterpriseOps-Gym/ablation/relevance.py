"""Post-hoc relevance sets, derived from verifier SQL. Analysis only.

This module is deliberately isolated from the intervention: ``state_model.py``
must not import it (GUD-008, SEC-001), and TEST-013 asserts that. It exists so
the report can answer "why did it work" by scoring *state retrieval precision
and recall* after the fact.

Nothing here influences what the agent saw.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from csm_env.schema_spec.registry import SchemaRegistry
from .task_registry import resolve_task_config_path

PathLike = Union[str, Path]

#: Confidence levels reported per relevance set.
CONFIDENCE_FULL = "full"
CONFIDENCE_TABLE_ONLY = "table_only"
CONFIDENCE_EMPTY = "empty"

_TABLE_RE = re.compile(
    r"\b(?:FROM|JOIN|UPDATE|INTO)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE
)
_PK_LITERAL_RE_CACHE: Dict[str, re.Pattern] = {}


def _pk_literal_pattern(primary_key: str) -> re.Pattern:
    """Cache one compiled pattern per primary-key column name."""
    pattern = _PK_LITERAL_RE_CACHE.get(primary_key)
    if pattern is None:
        pattern = re.compile(
            rf"\b{re.escape(primary_key)}\s*=\s*'([^']{{1,64}})'",
            re.IGNORECASE,
        )
        _PK_LITERAL_RE_CACHE[primary_key] = pattern
    return pattern


@dataclass(frozen=True)
class RelevanceSet:
    """What a correct state retrieval would plausibly have covered.

    Attributes:
        task_id: Task the set belongs to.
        tables: Tables referenced by verifier SQL that exist in the CSM registry.
        rows: ``(table, primary_key_value)`` pairs extracted from verifier SQL.
        confidence: One of the ``CONFIDENCE_*`` constants.
        unparsed: Human-readable reasons a verifier could not be interpreted.
        verifier_count: Number of verifiers inspected.
    """

    task_id: str
    tables: Tuple[str, ...]
    rows: Tuple[Tuple[str, str], ...]
    confidence: str
    unparsed: Tuple[str, ...]
    verifier_count: int
    scope: str = "final_state_verifier"
    source: str = "verifier_sql"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tables": list(self.tables),
            "rows": [list(row) for row in self.rows],
            "confidence": self.confidence,
            "unparsed": list(self.unparsed),
            "verifier_count": self.verifier_count,
            "scope": self.scope,
            "source": self.source,
        }


def extract_relevant_entities(
    task_config_path: PathLike,
    *,
    registry: Optional[SchemaRegistry] = None,
) -> RelevanceSet:
    """Derive the relevance set for one task from its verifier SQL.

    Table-level extraction is reliable. Row-level extraction succeeds only when a
    verifier pins a primary key with a string literal; otherwise the set is
    marked table-only and row-level precision/recall is reported with a
    table-granularity fallback rather than a misleading number.
    """
    resolved_path = resolve_task_config_path(task_config_path)
    payload: Mapping[str, Any] = json.loads(
        resolved_path.read_text(encoding="utf-8")
    )
    resolved_registry = registry or SchemaRegistry.from_static()
    task_id = resolved_path.stem

    verifiers = payload.get("verifiers") or []
    tables: List[str] = []
    rows: List[Tuple[str, str]] = []
    unparsed: List[str] = []

    for index, verifier in enumerate(verifiers):
        if not isinstance(verifier, dict):
            continue
        validation_config = verifier.get("validation_config") or {}
        sql_values = [
            validation_config.get(key) for key in ("query", "sql_query")
        ]
        sql_values = [value for value in sql_values if isinstance(value, str)]
        if not sql_values:
            unparsed.append(f"verifier[{index}] has no SQL to analyze")
            continue
        for sql in sql_values:
            for raw_table in _TABLE_RE.findall(sql):
                table = raw_table.lower()
                if resolved_registry.table(table) is None:
                    continue
                if table not in tables:
                    tables.append(table)
                spec = resolved_registry.require_table(table)
                for match in _pk_literal_pattern(spec.primary_key).finditer(sql):
                    row = (table, match.group(1))
                    if row not in rows:
                        rows.append(row)

    if rows:
        confidence = CONFIDENCE_FULL
    elif tables:
        confidence = CONFIDENCE_TABLE_ONLY
    else:
        confidence = CONFIDENCE_EMPTY

    return RelevanceSet(
        task_id=task_id,
        tables=tuple(tables),
        rows=tuple(rows),
        confidence=confidence,
        unparsed=tuple(unparsed),
        verifier_count=len(verifiers),
    )


def build_relevance_map(
    task_config_paths: Sequence[PathLike],
    *,
    registry: Optional[SchemaRegistry] = None,
) -> Dict[str, RelevanceSet]:
    """Derive relevance sets for a whole evaluation set, keyed by task id."""
    resolved_registry = registry or SchemaRegistry.from_static()
    return {
        Path(path).stem: extract_relevant_entities(path, registry=resolved_registry)
        for path in task_config_paths
    }
