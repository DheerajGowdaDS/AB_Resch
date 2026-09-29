from __future__ import annotations

import asyncio

import pytest

from csm_env.query.models import TaskRequest
from csm_env.schema_graph import build_schema_graph
from csm_env.schema_spec import SchemaRegistry
from csm_env.state import StateBuilder
from csm_env.verification import (
    DirectSqlOracle,
    check_v10_freshness,
    check_v1_table_completeness,
    check_v2_columns_and_pks,
    check_v3_foreign_keys,
    check_v4_referential_integrity,
    check_v5_graph_sql_equivalence,
    check_v6_plan_validation,
    check_v7_identifier_propagation,
    check_v8_fact_grounding,
    check_v9_state_consistency,
    compare_states,
)
from tests_support.helpers import make_router
from conftest import load_fixture_rows, FilteredStaticReader


@pytest.fixture()
def registry():
    return SchemaRegistry.from_static()


@pytest.fixture()
def oracle(registry, reader):
    return DirectSqlOracle(
        reader, registry.manifest, database_id="fixture-db"
    )


@pytest.fixture()
def state(registry, reader):
    router = make_router(registry, reader)
    outcome = asyncio.run(
        router.route(TaskRequest(task_type="resolve_case", reference_type="customer_case", reference_id=1233))
    )
    return StateBuilder(registry, database_id="fixture-db").build(outcome)


def test_oracle_reads_independent_rows(registry, oracle):
    row = asyncio.run(oracle.get_row("customer_case", 1233))
    assert row is not None
    assert row["state"] == "open"
    assert asyncio.run(oracle.get_row("customer_case", 424242)) is None


def test_oracle_rejects_unknown_identifiers(oracle):
    with pytest.raises(KeyError):
        asyncio.run(oracle.get_row("not_a_table", 1))
    with pytest.raises(KeyError):
        asyncio.run(oracle.get_rows_by_fk("account", "nope", 1))


def test_v1_table_completeness(registry):
    result = check_v1_table_completeness(registry, registry.tables())
    assert result.passed
    result = check_v1_table_completeness(registry, ["account", "ghost_table"])
    assert not result.passed
    assert any("missing tables" in failure for failure in result.failures)


def test_v2_columns_and_pks(registry):
    observed = {
        spec.table: {"columns": spec.column_names(), "primary_key": spec.primary_key}
        for spec in registry.manifest.tables
    }
    assert check_v2_columns_and_pks(registry, observed).passed
    observed["account"]["primary_key"] = "row_id"
    result = check_v2_columns_and_pks(registry, observed)
    assert not result.passed
    assert any("wrong PK" in failure for failure in result.failures)


def test_v3_foreign_keys(registry):
    edges = [
        (fk.source_table, fk.source_column, fk.target_table, fk.target_column)
        for fk in registry.manifest.foreign_keys
    ]
    assert check_v3_foreign_keys(registry, edges).passed
    fabricated = edges + [("customer_case", "case_id", "contract", "contract_id")]
    result = check_v3_foreign_keys(registry, fabricated)
    assert not result.passed
    assert any("fabricated" in failure for failure in result.failures)


def test_v4_referential_integrity(registry, oracle):
    fk = next(fk for fk in registry.manifest.foreign_keys if fk.edge_id == "customer_case.account_id->account.account_id")
    result = asyncio.run(check_v4_referential_integrity(registry, oracle, fk, sample_size=10))
    assert result.passed, result.failures


def test_v4_detects_dangling_reference(registry, oracle):
    # Simulate a dangling FK in a shared mutable fixture database.
    from conftest import FilteredStaticReader, load_fixture_rows
    from csm_env.verification import DirectSqlOracle as O

    rows = load_fixture_rows()
    rows["customer_case"][0]["account_id"] = 424242
    dangling_oracle = DirectSqlOracle(FilteredStaticReader(rows), registry.manifest, database_id="fixture-db")
    fk = next(fk for fk in registry.manifest.foreign_keys if fk.edge_id == "customer_case.account_id->account.account_id")
    result = asyncio.run(check_v4_referential_integrity(registry, dangling_oracle, fk, sample_size=10))
    assert not result.passed


def test_v5_graph_sql_equivalence(registry, oracle):
    fk = next(fk for fk in registry.manifest.foreign_keys if fk.edge_id == "customer_case.account_id->account.account_id")
    result = asyncio.run(check_v5_graph_sql_equivalence(registry, build_schema_graph(registry), oracle, fk, 1233))
    assert result.passed, result.failures


def test_v6_plan_validation(registry, reader):
    router = make_router(registry, reader)
    # Plan validation must accept the real plan and reject a fabricated hop.
    outcome = asyncio.run(
        router.route(TaskRequest(task_type="resolve_case", reference_type="customer_case", reference_id=1233))
    )
    graph = build_schema_graph(registry)
    result = check_v6_plan_validation(graph, outcome.plan.steps)
    assert result.passed

    from csm_env.query.models import QueryStep

    fake = QueryStep(
        step_id="s999", table="entitlement", filters=(), columns=None, limit=1,
        depends_on="s000", edge_id="customer_case.case_id->entitlement.entitlement_id",
        direction="outgoing", hop=1,
    )
    result = check_v6_plan_validation(graph, [outcome.plan.steps[0], fake])
    assert not result.passed


def test_v7_identifier_propagation(registry, oracle, state):
    fk = next(fk for fk in registry.manifest.foreign_keys if fk.edge_id == "customer_case.account_id->account.account_id")
    result = asyncio.run(check_v7_identifier_propagation(registry, oracle, state, fk))
    assert result.passed, result.failures


def test_v7_detects_propagated_id_drift(registry, reader, state):
    from dataclasses import replace

    drifted = tuple(
        replace(relation, target_id=99999) if relation.edge_id == "customer_case.account_id->account.account_id" else relation
        for relation in state.relations
    )
    tampered = replace(state, relations=drifted)
    fk = next(fk for fk in registry.manifest.foreign_keys if fk.edge_id == "customer_case.account_id->account.account_id")
    result = asyncio.run(check_v7_identifier_propagation(registry, DirectSqlOracle(reader, registry.manifest, database_id="fixture-db"), tampered, fk))
    assert not result.passed


def test_v8_fact_grounding(state):
    result = check_v8_fact_grounding(state, required_fact_count=10)
    assert result.passed, result.failures


def test_v8_detects_incomplete_provenance(state):
    from dataclasses import replace

    from csm_env.state.models import FactProvenance

    broken = tuple(
        replace(fact, provenance=replace(fact.provenance, query_id=""))
        if index == 0
        else fact
        for index, fact in enumerate(state.facts)
    )
    tampered = replace(state, facts=broken)
    result = check_v8_fact_grounding(tampered)
    assert not result.passed
    assert any("query_id" in failure for failure in result.failures)


def test_v9_state_consistency(registry, state):
    result = check_v9_state_consistency(state, registry)
    assert result.passed, result.failures


def test_v10_freshness_after_mutation(registry, oracle):
    from conftest import FilteredStaticReader, load_fixture_rows

    rows = load_fixture_rows()
    shared_reader = FilteredStaticReader(rows)
    router = make_router(registry, shared_reader)
    outcome = asyncio.run(
        router.route(TaskRequest(task_type="resolve_case", reference_type="customer_case", reference_id=1233))
    )
    builder = StateBuilder(registry, database_id="fixture-db")
    state_before = builder.build(outcome)
    assert state_before.fact_value("customer_case", 1233, "state") == "open"

    # Database changes externally; the same store serves the rebuild and
    # the independent oracle read.
    rows["customer_case"][0]["state"] = "resolved"
    shared_oracle = DirectSqlOracle(shared_reader, registry.manifest, database_id="fixture-db")
    current_row = asyncio.run(shared_oracle.get_row("customer_case", 1233))

    outcome_after = asyncio.run(
        router.route(TaskRequest(task_type="resolve_case", reference_type="customer_case", reference_id=1233))
    )
    state_after = builder.build(outcome_after)
    assert state_after.fact_value("customer_case", 1233, "state") == "resolved"
    result = check_v10_freshness(state_after, registry, oracle, current_row)
    assert result.passed, result.failures


def test_end_to_end_comparison_state_a_equals_state_b(registry, reader, oracle):
    """State A (pipeline) vs State B (direct SQL): exact match required."""
    router = make_router(registry, reader)
    outcome = asyncio.run(
        router.route(TaskRequest(task_type="resolve_case", reference_type="customer_case", reference_id=1233))
    )
    state_a = StateBuilder(registry, database_id="fixture-db").build(outcome)

    # State B: derived purely through the independent oracle for the tables
    # the task actually activates. Entitlement rows come from a direct FK
    # read (the same shape the DB answers, no pipeline code involved).
    oracle_facts = {}
    for table, pk_value in (
        ("customer_case", 1233),
        ("account", 10),
        ("product", 20),
        ("case_sla", 300),
    ):
        row = asyncio.run(oracle.get_row(table, pk_value))
        if row is None:
            continue
        for column, value in row.items():
            oracle_facts[(table, str(pk_value), column)] = _ser(value)

    for row in asyncio.run(oracle.get_rows_by_fk("entitlement", "account_id", 10)):
        pk = str(row["entitlement_id"])
        for column, value in row.items():
            oracle_facts[("entitlement", pk, column)] = _ser(value)

    report = compare_states(state_a, oracle_facts)
    assert report.equivalent, (report.summary(), report.value_mismatches[:3])
    assert report.matches > 30


def test_end_to_end_comparison_detects_mismatch(registry, reader, oracle):
    router = make_router(registry, reader)
    outcome = asyncio.run(
        router.route(TaskRequest(task_type="resolve_case", reference_type="customer_case", reference_id=1233))
    )
    state_a = StateBuilder(registry, database_id="fixture-db").build(outcome)

    oracle_facts = {("customer_case", "1233", "state"): "resolved"}
    report = compare_states(state_a, oracle_facts)
    assert not report.equivalent
    assert report.value_mismatches


def _ser(value):
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def test_snapshot_verification_runs_against_fixture(registry, fixture_rows):
    from csm_env.verification import check_v1_v2_snapshot

    checks = check_v1_v2_snapshot(registry, {"tables": fixture_rows})
    assert checks[0].passed
    assert checks[2].passed, checks[2].failures
