from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from csm_env.schema_spec import SchemaRegistry
from csm_env.transport.reader import StaticRowsReader

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "csm_minimal.json"


def load_fixture() -> Dict[str, Any]:
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_fixture_rows() -> Dict[str, List[Dict[str, Any]]]:
    return load_fixture()["tables"]


class FilteredStaticReader(StaticRowsReader):
    """Static reader honoring simple WHERE equality, IN, AND predicates."""

    async def fetch_rows(self, query: str) -> List[Dict[str, Any]]:
        import re

        rows = await super().fetch_rows(query)
        where = re.search(r"WHERE\s+(.+?)\s+LIMIT\s+\d+", query, re.IGNORECASE | re.DOTALL)
        if not where:
            return rows

        expr = where.group(1).strip()
        predicates = [piece.strip() for piece in re.split(r"\s+AND\s+", expr, flags=re.IGNORECASE)]

        def parse_literal(raw: str):
            raw = raw.strip()
            if raw.upper() == "NULL":
                return None
            if raw.upper() in ("TRUE", "FALSE"):
                return raw.upper() == "TRUE"
            if raw.startswith("'") and raw.endswith("'"):
                return raw[1:-1].replace("''", "'")
            if re.fullmatch(r"-?\d+", raw):
                return int(raw)
            if re.fullmatch(r"-?\d+\.\d+", raw):
                return float(raw)
            raise ValueError(f"fixture cannot parse literal {raw!r}")

        def matches(row, predicate: str) -> bool:
            if predicate.upper() == "FALSE":
                return False
            in_match = re.fullmatch(r"(\w+)\s+IN\s*\((.*)\)", predicate, re.IGNORECASE | re.DOTALL)
            if in_match:
                column = in_match.group(1)
                raw_values = [part.strip() for part in in_match.group(2).split(",") if part.strip()]
                values = [parse_literal(raw) for raw in raw_values]
                return any(row.get(column) == value or (value is not None and str(row.get(column)) == str(value)) for value in values)

            eq_match = re.fullmatch(r"(\w+)\s*=\s*(.+)", predicate, re.IGNORECASE | re.DOTALL)
            if not eq_match:
                return True
            column = eq_match.group(1)
            value = parse_literal(eq_match.group(2))
            return row.get(column) == value or (value is not None and str(row.get(column)) == str(value))

        return [row for row in rows if all(matches(row, predicate) for predicate in predicates)]


@pytest.fixture()
def fixture_rows() -> Dict[str, List[Dict[str, Any]]]:
    return load_fixture_rows()


@pytest.fixture()
def registry() -> SchemaRegistry:
    return SchemaRegistry.from_static()


@pytest.fixture()
def reader() -> FilteredStaticReader:
    return FilteredStaticReader(load_fixture_rows())
