from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


class TransportError(RuntimeError):
    """Raised for malformed transport responses or wrong-shape results."""


@runtime_checkable
class SQLReader(Protocol):
    async def fetch_rows(self, query: str) -> List[Dict[str, Any]]: ...


class EnterpriseOpsSQLRunnerReader:
    """Wraps the existing EnterpriseOpsSQLRunner behind SQLReader.

    Keeps HTTP/header/timeout concerns out of routing logic and converts
    non-dict/non-list responses into a typed TransportError.
    """

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    async def fetch_rows(self, query: str) -> List[Dict[str, Any]]:
        from ..sql_runner import extract_rows

        try:
            result = await self._runner.execute(query)
        except TypeError as exc:
            raise TransportError(f"Transport returned unsupported shape: {exc}") from exc
        if not isinstance(result, dict):
            raise TransportError(
                f"Expected dict envelope from sql-runner, got {type(result).__name__}"
            )
        rows = extract_rows(result)
        if rows and not all(isinstance(row, dict) for row in rows):
            raise TransportError("Transport produced non-dict rows")
        return rows


class StaticRowsReader:
    """Deterministic in-memory reader for tests and offline tools."""

    def __init__(self, rows_by_table: Dict[str, List[Dict[str, Any]]]) -> None:
        self._rows_by_table = {table: list(rows) for table, rows in rows_by_table.items()}

    async def fetch_rows(self, query: str) -> List[Dict[str, Any]]:
        import re

        match = re.search(r"FROM\s+(\w+)", query, re.IGNORECASE)
        if not match:
            raise TransportError(f"Cannot route query to fixture table: {query!r}")
        table = match.group(1).lower()
        return [dict(row) for row in self._rows_by_table.get(table, [])]
