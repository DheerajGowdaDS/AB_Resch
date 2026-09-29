from __future__ import annotations

import asyncio

import pytest

import csm_env
from csm_env import (
    CSMEnvironmentAPI,
    CSMEnvironmentRepresentation,
    EnterpriseOpsSQLRunner,
    GGQREnvironment,
    GroundedState,
    QueryBudget,
    SchemaRegistry,
    TaskRequest,
    from_enterpriseops_mcp_client,
)
from csm_env.query.models import RouteStatus
from csm_env.representation import verify_fidelity
from csm_env.state import StateStatus, check_state
from tests_support.helpers import make_router
from conftest import FilteredStaticReader, load_fixture_rows


# ---------------------------------------------------------------------------
# End-to-end offline pipeline (blueprint section 13)
# ---------------------------------------------------------------------------


def test_end_to_end_task_to_verified_state(registry, reader):
    ggqr = GGQREnvironment(reader, database_id="fixture-db", registry=registry)

    async def run():
        state = await ggqr.build_state_for("resolve_case", 1233, "customer_case")
        return state

    state = asyncio.run(run())
    assert isinstance(state, GroundedState)
    assert state.status == StateStatus.COMPLETE

    # Every fact traceable to a table/column/row.
    assert state.facts
    fact = state.facts[0].provenance
    assert fact.table and fact.column and fact.row_pk

    # Representations agree.
    representations = ggqr.represent(state)
    assert set(representations) == {"json", "graph", "text"}
    assert ggqr.fidelity(state).passed

    # Structural path validity.
    report = check_state(state, registry)
    assert report.consistent, report.problems


def test_end_to_end_equivalence_against_direct_sql(registry, reader):
    ggqr = GGQREnvironment(reader, database_id="fixture-db", registry=registry)
    result = asyncio.run(ggqr.verify_case_equivalence(1233))
    assert result["equivalent"], result["failures"]
    assert result["state_status"] == StateStatus.COMPLETE


# ---------------------------------------------------------------------------
# Compatibility: every existing public import and behavior (TEST-016)
# ---------------------------------------------------------------------------


def test_public_imports_unchanged():
    assert csm_env.CSMEnvironmentAPI is CSMEnvironmentAPI
    assert csm_env.CSMEnvironmentRepresentation is CSMEnvironmentRepresentation
    assert csm_env.EnterpriseOpsSQLRunner is EnterpriseOpsSQLRunner
    assert callable(from_enterpriseops_mcp_client)
    from csm_env import ENTITIES, FKS, TABLE_COLUMNS, Edge, EnvironmentGraph, Node

    assert len(ENTITIES) == 17 and len(FKS) == 29 and len(TABLE_COLUMNS) == 17
    assert EnvironmentGraph and Node and Edge


def test_legacy_record_graph_flow_unchanged():
    from csm_env.builder import CSMEnvironmentRepresentation as Rep
    from csm_env.graph import EnvironmentGraph

    class FakeSQL:
        async def fetch_rows(self, query):
            import re

            m = re.search(r"FROM\s+(\w+).*?(?:WHERE\s+(\w+)\s*=\s*([^\s;]+))?", query, re.I | re.S)
            rows = load_fixture_rows().get(m.group(1), [])
            if m.group(2):
                col = m.group(2)
                value = m.group(3).strip("'")
                rows = [r for r in rows if str(r.get(col)) == value]
            return rows

    env = Rep(FakeSQL())
    context = asyncio.run(env.get_case_context(1233, max_hops=2))
    assert context["case"]["state"] == "open"
    relations = {r["relation"] for r in context["relations"]}
    assert {"BELONGS_TO", "CONCERNS_PRODUCT", "ASSIGNED_TO"}.issubset(relations)
    ground_truth = asyncio.run(env.get_current_state("customer_case", 1233))
    assert ground_truth["priority"] == "P1"
    assert isinstance(env.graph, EnvironmentGraph)


def test_mcp_client_adapter_legacy_and_ggqr():
    class FakeMCPClient:
        base_url = "http://localhost:8001"
        database_id = "fixture-db"

    legacy = from_enterpriseops_mcp_client(FakeMCPClient())
    assert not hasattr(legacy, "ggqr") or getattr(legacy, "ggqr", None) is None

    enabled = from_enterpriseops_mcp_client(FakeMCPClient(), enable_ggqr=True)
    assert enabled.ggqr is not None
    assert isinstance(enabled.ggqr, GGQREnvironment)


def test_api_additive_surface_requires_attach():
    class FakeSQL:
        async def fetch_rows(self, query):
            return []

    api = CSMEnvironmentAPI(CSMEnvironmentRepresentation(FakeSQL()))
    with pytest.raises(RuntimeError, match="GGQR is not attached"):
        asyncio.run(api.route_state("resolve_case", 1233))


def test_api_additive_surface_end_to_end(registry, reader):
    class FakeSQL:
        async def fetch_rows(self, query):
            return await reader.fetch_rows(query)

    representation = CSMEnvironmentRepresentation(FakeSQL())
    representation.ggqr = GGQREnvironment(reader, database_id="fixture-db", registry=registry)
    api = CSMEnvironmentAPI(representation, ggqr_environment=representation.ggqr)

    outcome = asyncio.run(api.route_state("resolve_case", 1233, "customer_case"))
    assert outcome.status == RouteStatus.COMPLETE

    state = asyncio.run(api.build_state("resolve_case", 1233, "case"))
    assert state.status == StateStatus.COMPLETE
    rendered = api.represent_state(state)
    assert "json" in rendered and "text" in rendered


def test_budget_defaults_are_approved_values():
    budget = QueryBudget()
    assert budget.max_hops == 2
    assert budget.max_tables == 12
    assert budget.max_queries == 32
    assert budget.max_rows_per_query == 50
    assert budget.max_total_rows == 500
    assert budget.max_parallel == 8


def test_schema_version_compatibility_gate():
    registry = SchemaRegistry.from_static()
    assert registry.compatibility("1.0.0").compatible is True
    assert registry.compatibility("2.0.0").compatible is False
