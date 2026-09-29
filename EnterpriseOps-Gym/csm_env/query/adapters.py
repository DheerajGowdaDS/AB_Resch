from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..schema_spec.registry import SchemaRegistry
from ..transport.reader import SQLReader
from .models import Filter, QuerySpec
from .validation import validate_read_only_select

#: Escape character used for ``LIKE`` wildcards.
LIKE_ESCAPE_CHAR = "\\"

#: Wildcards that must be neutralised inside a user-supplied ``LIKE`` value.
LIKE_WILDCARDS: Tuple[str, ...] = ("%", "_")


def render_literal(value: Any) -> str:
    """Render one scalar SQL literal with strict escaping."""
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
    raise TypeError(f"Unsupported filter value type for SQL rendering: {type(value).__name__}")


def escape_like_pattern(value: str) -> str:
    """Escape a value for safe use inside a ``LIKE`` pattern.

    Two classes of character must be neutralised, and omitting either is a real
    defect rather than a theoretical one:

    * wildcards (``%`` and ``_``) plus the escape character itself, so a
      literal serial such as ``A_B`` cannot match ``AXB``;
    * **the apostrophe**, which would otherwise terminate the string literal and
      produce malformed SQL. The live Experiment-1 pilot failed a task exactly
      this way: the prompt proper noun ``Wayne Enterprises' Windows Server``
      yielded the pattern ``'%Wayne Enterprises' Windows Server%'``, which the
      CSM server rejected with ``400 Bad Request``. Doubling the quote is what
      keeps the literal intact.

    Args:
        value: Raw text taken from a task prompt.

    Returns:
        The escaped body, without the surrounding ``%`` wildcards and without
        the surrounding SQL quotes.
    """
    escaped = str(value)
    escaped = escaped.replace(LIKE_ESCAPE_CHAR, LIKE_ESCAPE_CHAR * 2)
    for wildcard in LIKE_WILDCARDS:
        escaped = escaped.replace(wildcard, LIKE_ESCAPE_CHAR + wildcard)
    return escaped.replace("'", "''")


def render_like_pattern(value: str) -> str:
    """Render a bounded, contains-style ``LIKE`` pattern for a raw value."""
    return "'%" + escape_like_pattern(value) + "%'"


def render_filter(column: str, value: Any) -> str:
    """Render equality or membership semantics for one validated column.

    A tuple/list/set means SQL IN (...). This is the core fix for GGQR fan-out:
    multiple propagated foreign-key identifiers are represented as one set-valued
    filter instead of an impossible `column=a AND column=b` predicate.
    """
    if isinstance(value, (list, tuple, set, frozenset)):
        values = list(value)
        if not values:
            return "FALSE"
        # Deterministic ordering for sets; caller-provided list/tuple order is kept.
        if isinstance(value, (set, frozenset)):
            values = sorted(values, key=lambda item: (type(item).__name__, repr(item)))
        return f"{column} IN (" + ", ".join(render_literal(item) for item in values) + ")"
    return f"{column} = {render_literal(value)}"


def render_query(registry: SchemaRegistry, spec: QuerySpec) -> str:
    """Render a QuerySpec into SQL.

    Identifiers come only from the schema registry; values pass through
    render_literal. User input can never become an identifier (SEC-002).
    """
    table = registry.require_table(spec.table)
    declared = set(table.column_names())

    columns = spec.columns if spec.columns is not None else table.column_names()
    columns = tuple(columns)
    if not columns:
        raise ValueError(f"QuerySpec for {spec.table} has no columns to select")
    for column in columns:
        if column not in declared:
            raise KeyError(f"Unknown column {column!r} for table {table.table}")

    if spec.limit < 1:
        raise ValueError("QuerySpec limit must be >= 1")

    parts: List[str] = [f"SELECT {', '.join(columns)}", f"FROM {table.table}"]
    if spec.filters:
        rendered = []
        for one_filter in spec.filters:
            if one_filter.column not in declared:
                raise KeyError(f"Unknown filter column {one_filter.column!r} for table {table.table}")
            rendered.append(render_filter(one_filter.column, one_filter.value))
        parts.append("WHERE " + " AND ".join(rendered))
    parts.append(f"LIMIT {int(spec.limit)}")
    return " ".join(parts) + ";"


class TableAdapter:
    """Schema-bound, read-only access to one table.

    All identifiers are validated against the manifest; the only SQL
    rendering happens in render_query. GGQR plans carry QuerySpec objects,
    never SQL text.
    """

    def __init__(self, registry: SchemaRegistry, reader: SQLReader) -> None:
        self._registry = registry
        self._reader = reader

    @property
    def table(self) -> str:
        raise NotImplementedError

    async def query(self, spec: QuerySpec) -> List[Dict[str, Any]]:
        """Render, validate, and execute one schema-bound read.

        The validation step is the Phase 1.4 guarantee: a malformed statement is
        converted into a typed :class:`~csm_env.query.validation.SqlValidationError`
        here, *before* any network call, so no broken query can reach the live
        CSM server and be misreported as a transport fault.
        """
        if spec.table != self.table:
            raise ValueError(f"Adapter for {self.table} cannot query {spec.table}")
        query = validate_read_only_select(render_query(self._registry, spec))
        return await self._reader.fetch_rows(query)

    async def query_by_pk(self, value: Any, limit: int = 2) -> List[Dict[str, Any]]:
        table = self._registry.require_table(self.table)
        spec = QuerySpec(
            table=self.table,
            filters=(Filter(column=table.primary_key, value=value),),
            limit=limit,
        )
        return await self.query(spec)

    async def query_by_fk(self, column: str, value: Any, limit: int = 50) -> List[Dict[str, Any]]:
        # Validates the column against the manifest before rendering.
        self._registry.require_column(self.table, column)
        spec = QuerySpec(
            table=self.table,
            filters=(Filter(column=column, value=value),),
            limit=limit,
        )
        return await self.query(spec)


class ManagedTableAdapter(TableAdapter):
    """Adapter bound to one specific table at construction."""

    def __init__(self, registry: SchemaRegistry, reader: SQLReader, table: str) -> None:
        super().__init__(registry, reader)
        self._table = registry.require_table(table).table

    @property
    def table(self) -> str:  # noqa: D102 - see TableAdapter
        return self._table


class TableAdapterFactory:
    """Builds per-table adapters on demand, all sharing one reader."""

    def __init__(self, registry: SchemaRegistry, reader: SQLReader) -> None:
        self._registry = registry
        self._reader = reader
        self._adapters: Dict[str, TableAdapter] = {}

    def adapter_for(self, table: str) -> TableAdapter:
        key = table.lower()
        if key not in self._adapters:
            self._adapters[key] = ManagedTableAdapter(self._registry, self._reader, key)
        return self._adapters[key]
