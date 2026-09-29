"""Evaluator-only relevance for the *initial* task state.

Experiment 1 injects state before the agent acts.  Therefore retrieval-quality
metrics must not compare that pre-action state with final-state verifier values.
This module builds an evaluation-only relevance set from information the agent
could legitimately use at time t=0: the task's user prompt plus the seeded DB.

Method:
* use prompt-derived literals/proper nouns/serials/emails;
* search only prompt-matched CSM tables and identity/name-like columns in the
  seeded SQL snapshot;
* never read verifier SQL and never sit on the Condition-B import chain.

The result is deliberately called ``prompt_entity`` relevance rather than a
"gold solution state": it measures explicit initial entities, not every hidden
fact needed to solve the task.  It is therefore diagnostic, while benchmark
TSR/VPR remain the research outcomes.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from csm_env.schema_spec.models import ColumnType
from csm_env.schema_spec.registry import SchemaRegistry

from .seeds import SeedCache
from .task_registry import EvalSetManifest, TaskRecord, resolve_task_config_path
from .relevance import (
    CONFIDENCE_EMPTY,
    CONFIDENCE_FULL,
    RelevanceSet,
)

PathLike = Union[str, Path]

# Identity/name-like columns are intentionally narrower than all text columns;
# searching fields such as descriptions would turn generic prompt words into a
# large and noisy "relevance" set.
_NAME_COLUMNS = {
    "name", "title", "number", "kb_number", "serial_number", "email",
    "first_name", "last_name", "street", "plot_no",
    "product_name", "account_name",
}
_LITERAL_CLEAN = re.compile(r"\s+")


def _quote_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _sqlite_type(column_type: ColumnType) -> str:
    return {
        "int": "INTEGER",
        "float": "REAL",
        "bool": "INTEGER",
        "datetime": "TEXT",
        "string": "TEXT",
        "text": "TEXT",
        "unknown": "TEXT",
    }.get(column_type, "TEXT")


def _load_seed_db(seed_path: Path, registry: SchemaRegistry) -> sqlite3.Connection:
    """Load a benchmark seed SQL file into a temporary SQLite database."""
    conn = sqlite3.connect(":memory:")
    ddl = []
    for spec in registry.manifest.tables:
        cols = []
        for column in spec.columns:
            declaration = f"{_quote_identifier(column.name)} {_sqlite_type(column.type)}"
            if column.name == spec.primary_key:
                declaration += " PRIMARY KEY"
            cols.append(declaration)
        ddl.append(f"CREATE TABLE {_quote_identifier(spec.table)} ({', '.join(cols)});")
    conn.executescript("\n".join(ddl))
    conn.executescript(seed_path.read_text(encoding="utf-8", errors="replace"))
    return conn


def _norm(value: Any) -> str:
    return _LITERAL_CLEAN.sub(" ", str(value).strip().lower())


def _candidate_values(task: TaskRecord) -> Tuple[str, ...]:
    values = []
    seen = set()
    for value in (*task.signals.literals, *task.signals.proper_nouns, *task.signals.serial_suffixes):
        value = str(value).strip()
        if len(value) < 3:
            continue
        normalized = _norm(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            values.append(value)
    return tuple(values)


def _search_prompt_entities(
    conn: sqlite3.Connection,
    task: TaskRecord,
    registry: SchemaRegistry,
) -> Tuple[Tuple[str, str], ...]:
    """Find explicit prompt entities in the seeded initial state."""
    found = set()
    candidate_values = _candidate_values(task)
    matched_tables = set(task.signals.matched_tables)
    all_tables = {spec.table for spec in registry.manifest.tables}
    # Search explicitly matched tables first, then the remaining CSM tables.
    # This avoids missing a task entity such as a named User when the prompt's
    # table matcher only identified the destination table (e.g. customer_case).
    target_tables = tuple(sorted(matched_tables)) + tuple(sorted(all_tables - matched_tables))

    if not candidate_values:
        return tuple(sorted(found))

    for table in target_tables:
        spec = registry.table(table)
        if spec is None:
            continue
        searchable = [
            c for c in spec.columns
            if c.name.lower() in _NAME_COLUMNS
            or c.name == spec.primary_key
        ]
        if not searchable:
            continue

        table_q = _quote_identifier(spec.table)
        pk = _quote_identifier(spec.primary_key)

        # User names are represented as first_name + last_name rather than a
        # single ``name`` column. Match the combined identity so ``Curtis
        # McCoy`` does not become thousands of unrelated first/last-name hits.
        if spec.table == "user":
            names = conn.execute(
                f"SELECT {pk}, {_quote_identifier('first_name')}, {_quote_identifier('last_name')} "
                f"FROM {table_q}"
            ).fetchall()
            for row_id, first_name, last_name in names:
                full_name = _norm(f"{first_name or ''} {last_name or ''}")
                for candidate in candidate_values:
                    candidate_norm = _norm(candidate)
                    if len(candidate_norm.split()) >= 2 and full_name == candidate_norm:
                        found.add((spec.table, str(row_id)))
                        break

        for column in searchable:
            col = _quote_identifier(column.name)
            # Exact normalized matching is performed in Python after a SELECT;
            # SQLite's LIKE/collation semantics vary and are not suitable for a
            # reproducible evaluator metric.
            rows = conn.execute(
                f"SELECT {pk}, {col} FROM {table_q} WHERE {col} IS NOT NULL"
            ).fetchall()
            for row_id, value in rows:
                value_norm = _norm(value)
                for candidate in candidate_values:
                    candidate_norm = _norm(candidate)
                    if value_norm == candidate_norm:
                        found.add((spec.table, str(row_id)))
                        break
                    # Multi-word names such as ``Dell PowerEdge R740`` may be
                    # decorated in the DB with a variant/config suffix. Allow
                    # containment only for substantive multi-token candidates;
                    # this deliberately excludes generic one-word terms such as
                    # ``London`` that caused massive false location matches.
                    if (
                        len(candidate_norm.split()) >= 2
                        and len(candidate_norm) >= 10
                        and candidate_norm in value_norm
                    ):
                        found.add((spec.table, str(row_id)))
                        break

    return tuple(sorted(found))




def build_initial_relevance_map(
    eval_set: EvalSetManifest,
    *,
    archive: PathLike,
    registry: Optional[SchemaRegistry] = None,
) -> Dict[str, RelevanceSet]:
    """Build initial-state relevance sets from prompt entities and seed data."""
    resolved_registry = registry or SchemaRegistry.from_static()
    archive_path = Path(archive)
    if not archive_path.is_absolute():
        # Accept paths supplied either from the gym checkout or from the
        # workspace root; ``SeedCache`` resolves relative archives from the
        # package root, so make the caller's existing path explicit first.
        if archive_path.is_file() or archive_path.is_dir():
            archive_path = archive_path.resolve()
        else:
            package_root = Path(__file__).resolve().parents[1]
            archive_path = (package_root / archive_path).resolve()
    cache = SeedCache(archive=archive_path)
    output: Dict[str, RelevanceSet] = {}

    # One temporary directory/cache per task keeps the evaluator deterministic
    # and avoids retaining the generated SQLite snapshots in experiment output.
    for task in eval_set:
        seed_file = task.seed_database_file
        try:
            seed_path = cache.materialize(seed_file)
        except FileNotFoundError:
            output[task.task_id] = RelevanceSet(
                task_id=task.task_id,
                tables=tuple(sorted(task.signals.matched_tables)),
                rows=(),
                confidence=CONFIDENCE_EMPTY,
                unparsed=(f"seed unavailable: {seed_file}",),
                verifier_count=task.verifier_count,
                scope="initial_prompt_entities",
                source="task_prompt+seed_sql",
            )
            continue

        try:
            conn = _load_seed_db(seed_path, resolved_registry)
            rows = _search_prompt_entities(conn, task, resolved_registry)
            conn.close()
        except Exception as exc:
            output[task.task_id] = RelevanceSet(
                task_id=task.task_id,
                tables=tuple(sorted(task.signals.matched_tables)),
                rows=(),
                confidence=CONFIDENCE_EMPTY,
                unparsed=(f"initial-state relevance error: {type(exc).__name__}: {exc}",),
                verifier_count=task.verifier_count,
                scope="initial_prompt_entities",
                source="task_prompt+seed_sql",
            )
            continue

        confidence = CONFIDENCE_FULL if rows else CONFIDENCE_EMPTY
        output[task.task_id] = RelevanceSet(
            task_id=task.task_id,
            tables=tuple(sorted(task.signals.matched_tables)),
            rows=rows,
            confidence=confidence,
            unparsed=(),
            verifier_count=task.verifier_count,
            scope="initial_prompt_entities",
            source="task_prompt+seed_sql",
        )
    return output


__all__ = ["build_initial_relevance_map"]
