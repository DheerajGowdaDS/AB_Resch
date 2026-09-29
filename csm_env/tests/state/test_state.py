from __future__ import annotations

import asyncio

import pytest

from csm_env.query.models import RouteStatus, QueryBudget, QueryResult, QueryStep, StepStatus, QueryPlan, RouteOutcome
from csm_env.state import (
    GroundedState,
    StateBuilder,
    StateStatus,
    check_state,
    redact,
)
from tests_support.helpers import make_router
from conftest import load_fixture_rows, FilteredStaticReader  # same dir as tests
from csm_env.schema_spec import SchemaRegistry


def router_for(registry, reader):
    from tests_support.helpers import make_router

    return make_router(registry, reader)


@pytest.fixture()
def router(registry, reader):
    return make_router(registry, reader)


def _build_state(router, registry, task_type="resolve_case", reference=1233):
    outcome = asyncio.run(
        router.route(
            TaskRequest_for(task_type, reference)
        )
    )
    builder = StateBuilder(registry, database_id="fixture-db")
    return outcome, builder.build(outcome)


def TaskRequest_for(task_type, reference):
    from csm_env.query.models import TaskRequest

    return TaskRequest(task_type=task_type, reference_type="customer_case", reference_id=reference)


def test_state_builds_with_provenance(router, registry):
    outcome, state = _build_state(router, registry)
    assert state.status == StateStatus.COMPLETE
    tables = {record.table for record in state.records}
    assert {"customer_case", "account", "entitlement", "case_sla", "product"}.issubset(tables)
    assert state.facts, "facts must carry provenance"
    fact = state.facts[0]
    assert fact.provenance.database_id == "fixture-db"
    assert fact.provenance.table
    assert fact.provenance.column
    assert fact.provenance.row_pk
    assert fact.provenance.query_id
    assert fact.provenance.route_id
    assert "T" in fact.provenance.observed_at


def test_state_relations_point_to_observed_rows(router, registry):
    _, state = _build_state(router, registry)
    records = {(r.table, r.row_id) for r in state.records}
    assert state.relations
    for relation in state.relations:
        assert (relation.source_table, relation.source_id) in records
        assert (relation.target_table, relation.target_id) in records


def test_state_contains_expected_semantic_relations(router, registry):
    """Blueprint section 18 artifact: the case_overview schema paths
    (case->account, case->contact, case->product, case->user) must survive
    state construction. Guards against silently dropped relations."""
    outcome = asyncio.run(
        router.route(TaskRequest_for("case_overview", 1233))
    )
    state = StateBuilder(registry, database_id="fixture-db").build(outcome)
    relations = {relation.relation for relation in state.relations}
    assert {"BELONGS_TO", "REPORTED_BY", "CONCERNS_PRODUCT", "ASSIGNED_TO"}.issubset(relations)


def test_state_fact_value_lookup(router, registry):
    _, state = _build_state(router, registry)
    assert state.fact_value("customer_case", 1233, "state") == "open"
    assert state.fact_value("account", 10, "name") == "SynthCorp"
    assert state.fact_value("customer_case", 1233, "not_a_column") is None


def test_state_unresolved_for_missing_reference(router, registry):
    outcome = asyncio.run(
        router.route(
            TaskRequest_for("resolve_case", 999999)
        )
    )
    state = StateBuilder(registry, database_id="fixture-db").build(outcome)
    assert state.status == StateStatus.UNRESOLVED
    assert "customer_case" in state.unresolved


def test_null_fk_surfaces_in_state(router, registry):
    outcome = asyncio.run(
        router.route(
            TaskRequest_for("case_overview", 1240)
        )
    )
    state = StateBuilder(registry, database_id="fixture-db").build(outcome)
    null_reasons = [d for d in state.unresolved_details if d[2] == "null_fk"]
    assert null_reasons, "case 1240 has NULL contact_id / assigned_to"


def test_consistency_passes_on_good_state(router, registry):
    _, state = _build_state(router, registry)
    report = check_state(state, registry)
    assert report.consistent, report.problems


def test_consistency_detects_broken_relation(registry, reader):
    outcome = asyncio.run(
        router_for(registry, reader).route(TaskRequest_for("resolve_case", 1233))
    )
    state = StateBuilder(registry, database_id="fixture-db").build(outcome)
    from dataclasses import replace
    from csm_env.state.models import StateRelation

    first = state.relations[0]
    broken = (replace(first, target_id="ghost"),) + tuple(state.relations[1:])
    tampered = replace(state, relations=broken)
    report = check_state(tampered, registry)
    assert not report.consistent
    assert any("ghost" in problem for problem in report.problems)


def test_redaction_removes_secrets():
    dirty = {
        "table": "account",
        "Authorization": "Bearer abc",
        "api_key": "xyz",
        "nested": {"token": "t", "value": 1},
    }
    clean = redact(dirty)
    assert clean == {"table": "account", "nested": {"value": 1}}


def test_freshness_generations(router, registry):
    outcome = asyncio.run(router.route(TaskRequest_for("resolve_case", 1233)))
    builder = StateBuilder(registry, database_id="fixture-db")
    first = builder.build(outcome)
    second = builder.build(outcome)
    assert first.observation_generation == 1
    assert second.observation_generation == 2
    info = builder.freshness(second)
    assert info.observation_generation == 2
    assert info.database_id == "fixture-db"


def test_state_rebuild_after_change(registry):
    # Mutable fake reader: simulate the database changing between builds (V10).
    from tests_support.helpers import make_router

    rows = load_fixture_rows()
    reader = FilteredStaticReader(rows)

    router = make_router(registry, reader)
    outcome_before = asyncio.run(
        router.route(TaskRequest_for("resolve_case", 1233))
    )
    builder = StateBuilder(registry, database_id="fixture-db")
    state_before = builder.build(outcome_before)
    assert state_before.fact_value("customer_case", 1233, "state") == "open"

    # Mutate the fixture database.
    rows["customer_case"][0]["state"] = "resolved"

    outcome_after = asyncio.run(
        router.route(TaskRequest_for("resolve_case", 1233))
    )
    state_after = builder.build(outcome_after)
    assert state_after.fact_value("customer_case", 1233, "state") == "resolved"
    assert state_after.observation_generation > state_before.observation_generation


def test_state_builder_records_duplicate_conflict_without_silent_resolution(registry, reader):
    from dataclasses import replace
    from csm_env.query.models import QueryBudget, QueryPlan, QueryResult, QueryStep, RouteOutcome, StepStatus
    from csm_env.state import StateBuilder

    plan = QueryPlan(
        anchor_table="account",
        anchor_key="account_id",
        anchor_value=10,
        steps=(
            QueryStep("s000", "account", (), None, 1, hop=0),
            QueryStep("s001", "account", (), None, 1, depends_on="s000", hop=1, edge_id="account.account_id->customer_case.account_id", direction="incoming", filter_column="account_id", binding_column="account_id"),
        ),
        budget=QueryBudget(max_hops=1),
        task_type="dup_test",
        route_id="route-dup",
    )
    first = {"account_id": 10, "name": "SynthCorp", "active": True}
    second = {"account_id": 10, "name": "Different", "active": True}
    outcome = RouteOutcome(
        status="COMPLETE",
        plan=plan,
        results=[
            QueryResult("s000", "account", [first], StepStatus.OK),
            QueryResult("s001", "account", [second], StepStatus.OK),
        ],
    )
    state = StateBuilder(registry, database_id="fixture-db").build(outcome)
    assert state.contradictions
    assert state.contradictions[0].column == "name"
    assert state.status == "FAILED"


def test_duplicate_observation_adds_new_columns_and_detects_real_type_conflict(registry):
    from csm_env.query.models import QueryPlan, QueryResult, QueryStep, RouteOutcome, TaskRequirements, StepStatus
    from csm_env.state.builder import StateBuilder
    from csm_env.state.models import StateStatus
    plan = QueryPlan(
        anchor_table="account", anchor_key="account_id", anchor_value=10,
        steps=(QueryStep(step_id="q1", table="account", filters=(), columns=None, limit=50),
               QueryStep(step_id="q2", table="account", filters=(), columns=None, limit=50)),
        budget=QueryBudget(), task_type="account_profile", route_id="r1"
    )
    outcome = RouteOutcome(status="COMPLETE", plan=plan, results=[
        QueryResult(step_id="q1", table="account", rows=[{"account_id": 10, "name": "SynthCorp"}], status=StepStatus.OK),
        QueryResult(step_id="q2", table="account", rows=[{"account_id": 10, "name": "Other", "active": True}], status=StepStatus.OK),
    ])
    state = StateBuilder(registry, database_id="fixture").build(outcome)
    rec = state.record("account", 10)
    assert rec is not None and rec.values["active"] is True
    assert state.status == StateStatus.FAILED
    assert any(c.column == "name" for c in state.contradictions)
