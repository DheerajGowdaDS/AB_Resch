"""Local test doubles for the ablation harness.

These are behavioural stand-ins for the *transports* the harness talks to
(``SQLReader``, the ``csm_env`` facade, the LLM client, the verifier engine).
They are deliberately small: every test still uses the real production types
(``GroundedState``, ``TaskRecord``, ``ReactOrchestrator``, ``VerifierConfig``),
so nothing is asserted against a re-implementation of the code under test.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from csm_env.state.models import (
    FactProvenance,
    GroundedState,
    StateFact,
    StateProvenance,
    StateRecord,
    StateRelation,
)
from csm_env.transport.reader import TransportError


class FakeSQLReader:
    """In-memory ``SQLReader`` that records every statement it is asked to run.

    Supports the harness's ``COUNT(*)`` aggregates and schema-bounded anchor
    queries of the form ``SELECT <pk> FROM <table> WHERE ...`` using ``=``,
    ``LIKE``, ``OR`` and bounded composite ``AND`` predicates.
    """

    def __init__(
        self, rows_by_table: Optional[Mapping[str, Sequence[Dict[str, Any]]]] = None
    ) -> None:
        self.rows_by_table: Dict[str, List[Dict[str, Any]]] = {
            table: [dict(row) for row in rows] for table, rows in (rows_by_table or {}).items()
        }
        self.statements: List[str] = []

    def add_row(self, table: str, row: Dict[str, Any]) -> None:
        self.rows_by_table.setdefault(table, []).append(dict(row))

    @property
    def read_only(self) -> bool:
        """Whether every recorded statement was a read."""
        return all(statement.strip().upper().startswith("SELECT") for statement in self.statements)

    @property
    def used_select_star(self) -> bool:
        return any("SELECT *" in statement.upper() for statement in self.statements)

    async def fetch_rows(self, query: str) -> List[Dict[str, Any]]:
        self.statements.append(query)
        if not query.strip().upper().startswith("SELECT"):
            raise TransportError(f"FakeSQLReader only serves SELECT, got: {query!r}")
        table_match = re.search(r"FROM\s+([A-Za-z_][A-Za-z0-9_]*)", query, re.IGNORECASE)
        if not table_match:
            raise TransportError(f"Cannot route query to a table: {query!r}")
        table = table_match.group(1).lower()
        rows = list(self.rows_by_table.get(table, []))

        if re.search(r"COUNT\s*\(\s*\*\s*\)", query, re.IGNORECASE):
            return [{"row_count": len(rows)}]

        pk_match = re.match(r"\s*SELECT\s+([A-Za-z_][A-Za-z0-9_]*)\s+FROM", query, re.IGNORECASE)
        primary_key = pk_match.group(1) if pk_match else None

        def matches(row: Dict[str, Any]) -> bool:
            predicates = re.findall(
                r"([A-Za-z_][A-Za-z0-9_]*)\s*(=|LIKE)\s*'((?:''|[^'])*)'",
                query,
                re.IGNORECASE,
            )
            if not predicates:
                return True
            # The production anchor query is an OR-of-terms, plus an optional
            # first_name/last_name AND pair. The fake mirrors those semantics
            # without becoming a SQL parser.
            ors = re.split(r"\s+OR\s+", query, flags=re.IGNORECASE)
            for branch in ors:
                terms = re.findall(
                    r"([A-Za-z_][A-Za-z0-9_]*)\s*(=|LIKE)\s*'((?:''|[^'])*)'",
                    branch,
                    re.IGNORECASE,
                )
                if not terms:
                    continue
                if all(
                    (str(row.get(col)) == val.replace("''", "'"))
                    if op == "="
                    else (val.replace("''", "'").strip("%") .lower() in str(row.get(col, "")).lower())
                    for col, op, val in terms
                ):
                    return True
            return False

        rows = [row for row in rows if matches(row)]
        if primary_key:
            rows = [
                {primary_key: row.get(primary_key)}
                for row in rows
                if row.get(primary_key) is not None
            ]
        limit = re.search(r"LIMIT\s+(\d+)", query, re.IGNORECASE)
        if limit:
            rows = rows[: int(limit.group(1))]
        return rows


def make_grounded_state(
    *,
    state_id: str = "state-1",
    table: str = "customer_case",
    row_id: str = "1233",
    extra_records: Sequence[Tuple[str, str]] = (),
    database_id: str = "db_test",
    status: str = "COMPLETE",
) -> GroundedState:
    """Build a real ``GroundedState`` without touching a database."""
    anchor = StateRecord(
        table=table, primary_key="case_id", row_id=row_id, values={"state": "open"}
    )
    records: List[StateRecord] = [anchor]
    for extra_table, extra_id in extra_records:
        records.append(
            StateRecord(
                table=extra_table,
                primary_key=f"{extra_table}_id",
                row_id=extra_id,
                values={"name": f"{extra_table}-{extra_id}"},
            )
        )
    observed_at = "2026-09-26T00:00:00+00:00"
    provenance = StateProvenance(
        database_id=database_id,
        schema_version="1.0.0",
        route_id="route-1",
        observed_at=observed_at,
        query_ids=("q1",),
    )
    facts = tuple(
        StateFact(
            value=record.values.get("state") or record.values.get("name"),
            provenance=FactProvenance(
                database_id=database_id,
                schema_version="1.0.0",
                table=record.table,
                column=next(iter(record.values), "id"),
                row_pk=record.row_id,
                query_id="q1",
                route_id="route-1",
                observed_at=observed_at,
            ),
        )
        for record in records
    )
    relations = (
        StateRelation(
            source_table=table,
            source_id=row_id,
            target_table=extra_records[0][0],
            target_id=extra_records[0][1],
            relation="BELONGS_TO",
            edge_id=f"{table}.account_id->account.account_id",
        ),
    ) if extra_records else ()
    return GroundedState(
        state_id=state_id,
        anchor=anchor,
        records=tuple(records),
        relations=relations,
        unresolved=(),
        unresolved_details=(),
        provenance=provenance,
        status=status,
        schema_version="1.0.0",
        observation_generation=1,
        facts=facts,
        contradictions=(),
    )


class FakeCsmEnvAPI:
    """Minimal stand-in for ``CSMEnvironmentAPI`` returning real representations.

    ``build_state`` is async and ``environment.sql`` exists so the production
    ``StateModelAdapter`` constructor path is exercised unchanged.
    """

    def __init__(self, state: GroundedState) -> None:
        self._state = state
        self.calls: List[Tuple[str, Any, Any]] = []
        self.environment = type("FakeEnvironment", (), {"sql": None})()

    async def build_state(
        self, task_type: str, reference_id: Any, reference_type: Optional[str] = None
    ) -> GroundedState:
        self.calls.append(("build_state", reference_id, reference_type))
        return self._state

    def represent_state(self, state: GroundedState, **_unused) -> Dict[str, str]:
        self.calls.append(("represent_state", state.state_id, None))
        text = " ".join(
            f"{record.table} {record.row_id} is present." for record in state.records
        )
        return {"text": text, "graph": f"graph:{state.state_id}", "json": "{}"}


class FakeLLMClient:
    """Returns one assistant turn with no tool calls, ending the ReAct loop."""

    def __init__(self, content: str = "done") -> None:
        self.content = content
        self.prompts: List[Any] = []

    async def invoke_with_tools(self, messages: Sequence[Any], tools: Sequence[Any]) -> Any:
        from langchain_core.messages import AIMessage

        self.prompts.append(list(messages))
        return AIMessage(content=self.content)


class FakeVerifierEngine:
    """Verifier engine that passes everything, recording what it was asked."""

    def __init__(self, passed: bool = True) -> None:
        self.passed = passed
        self.calls: List[Tuple[Optional[str], Any]] = []

    async def execute_verifier(
        self,
        verifier: Any,
        model_response: Dict[str, Any],
        database_id: Any,
        context: Optional[Dict[str, Any]] = None,
        gym_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.calls.append((verifier.name, database_id))
        return {"passed": self.passed, "expected": None, "actual": None}


class FakeMcpClient:
    """An ``MCPClient`` shape small enough to attach to an orchestrator."""

    def __init__(
        self, base_url: str = "http://localhost:8001", database_id: str = "db_test"
    ) -> None:
        self.base_url = base_url
        self.database_id = database_id
        self.context: Dict[str, Any] = {}
        self.auth_config = None

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True, "result": {}}
