"""State Model 2.0 blueprint mechanisms: minimal-state selection and reporting.

Blueprint sections 2, 5-10, 12 and 15: relevance scoring with externalized
weights, greedy weighted set cover, attribute pruning, token budget, the
coverage gate, coverage-driven escalation, and the N-arm report with the
restored fail-closed delivery gate. All offline, using the local doubles.
"""

from __future__ import annotations

import dataclasses

import pytest

from ablation.conditions import (
    CONDITION_A,
    CONDITION_B1,
    CONDITION_B2,
    parse_conditions,
)
from ablation.minimal_state import (
    DEFAULT_WEIGHTS,
    RelevanceWeights,
    build_minimal_state,
    coverage_facts,
    estimate_state_tokens,
    greedy_minimal_cover,
    prune_record_attributes,
    realized_relation_facts,
    score_candidate,
    task_intent,
    task_required_attributes,
    task_required_tables,
)
from ablation.report import build_report, summarize_state_injection
from ablation.state_model import StateModelAdapter
from ablation.task_registry import TaskRecord, analyze_prompt
from csm_env import SchemaRegistry
from csm_env.query.models import RouteStrategy
from csm_env.state.models import StateRecord
from tests.tests_support.ablation_doubles import (
    FakeCsmEnvAPI,
    FakeSQLReader,
    make_grounded_state,
)

PROMPT = (
    "Stark Industries needs urgent help. Update the entitlement support level to "
    "Enterprise 24x7 for their product and check the case priority and assignment group."
)


@pytest.fixture(scope="module")
def registry() -> SchemaRegistry:
    return SchemaRegistry.from_static()


def build_task(registry: SchemaRegistry, **overrides) -> TaskRecord:
    signals = analyze_prompt(PROMPT, ["find_user", "update_case"], task_id="t2", registry=registry)
    base = TaskRecord(
        task_id="t2",
        task_config_path="data/revised/csm/t2.json",
        seed_database_file="seed.sql",
        database_id=None,
        initial_state_fingerprint=None,
        complexity_category="TYPE_4_CROSS_ENTITY",
        category_rationale="test fixture",
        reference_entities=signals.matched_tables,
        reference_rows=(),
        reference_names=(("account", "Stark Industries"),),
        verifier_count=0,
        selected_tools_count=2,
        number_of_runs=1,
        reset_database_between_runs=True,
        ggqr_task_type="check_entitlement",
        signals=signals,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


# ---------------------------------------------------------------------------
# Blueprint section 7: relevance scoring with externalized weights
# ---------------------------------------------------------------------------


def test_scoring_prefers_anchor_then_required_then_noise(registry):
    state = make_grounded_state(
        table="customer_case",
        row_id="1233",
        extra_records=(("account", "7"), ("entitlement", "3"), ("location", "9"), ("notification", "800")),
    )
    task = build_task(registry, reference_entities=("customer_case", "account", "entitlement"))
    required = task_required_tables(task) | {"customer_case"}
    hinted = frozenset()

    scores = {
        (record.table, record.row_id): score_candidate(
            record, state=state, required_tables=required, hinted_tables=hinted,
            attributes=("priority", "support_level"),
        ).score
        for record in state.records
    }  # noqa: E501
    assert scores[("customer_case", "1233")] == max(scores.values())
    assert scores[("account", "7")] > scores[("location", "9")]
    assert scores[("entitlement", "3")] > scores[("notification", "800")]
    assert scores[("notification", "800")] < 0.6, "structural noise must score low"


def test_weights_are_externalized_and_validated():
    custom = RelevanceWeights(anchor_row=2.0, required_entity=1.0, hop=0.5)
    assert custom.anchor_row == 2.0
    with pytest.raises(ValueError):
        RelevanceWeights(anchor_row=0.0).validate()
    with pytest.raises(ValueError):
        RelevanceWeights(hop=-1.0).validate()
    assert DEFAULT_WEIGHTS.anchor_table > 0 and DEFAULT_WEIGHTS.hop >= 0


def test_task_attributes_and_intent_come_from_prompt_only(registry):
    task = build_task(registry)
    attributes = task_required_attributes(task)
    assert "priority" in attributes
    assert "support_level" in attributes
    assert "assignment_group" in attributes
    assert task_intent(task) in {("read", "update"), ("read",)}


# ---------------------------------------------------------------------------
# Blueprint section 8: greedy weighted set cover
# ---------------------------------------------------------------------------


def test_greedy_cover_keeps_every_required_table(registry):
    """The cover must not silently drop a required table the broad state saw."""
    state = make_grounded_state(
        table="customer_case",
        row_id="1233",
        extra_records=(("account", "7"), ("entitlement", "3")),
    )
    task = build_task(registry, reference_entities=("customer_case", "account", "entitlement"))
    required = task_required_tables(task) | {"customer_case"}

    scores = [
        score_candidate(
            record,
            state=state,
            required_tables=required,
            hinted_tables=frozenset(),
            attributes=(),
        )
        for record in state.records
    ]
    outcome = greedy_minimal_cover(scores, state=state, required_tables=required)
    kept_tables = {record.table for record in outcome.selected}
    assert {table for table in required if f"table:{table}" in outcome.covered} <= kept_tables
    assert outcome.uncovered_required == frozenset()


def test_greedy_cover_drops_rows_that_cover_nothing(registry):
    state = make_grounded_state(
        table="customer_case",
        row_id="1233",
        extra_records=(("account", "7"), ("location", "9"), ("notification", "800")),
    )
    task = build_task(registry, reference_entities=("customer_case", "account"))
    required = task_required_tables(task) | {"customer_case"}
    scores = [
        score_candidate(
            record,
            state=state,
            required_tables=required,
            hinted_tables=frozenset(),
            attributes=(),
        )
        for record in state.records
    ]
    outcome = greedy_minimal_cover(scores, state=state, required_tables=required)
    dropped_tables = {table for table, _row in outcome.dropped}
    assert "location" in dropped_tables and "notification" in dropped_tables


def test_coverage_facts_identify_anchor_and_required_tables(registry):
    state = make_grounded_state()
    anchor_record = state.records[0]
    covered = coverage_facts(
        anchor_record,
        anchor=state.anchor,
        required_tables=frozenset({"customer_case"}),
        required_attributes=("state",),
    )
    assert "anchor" in covered
    assert "table:customer_case" in covered
    # P2/A5: an attribute the row carries is now a first-class cover fact.
    assert "attr:customer_case.state" in covered


# ---------------------------------------------------------------------------
# Blueprint section 9: attribute pruning
# ---------------------------------------------------------------------------


def test_attribute_pruning_keeps_pk_fks_and_task_attributes(registry):
    record = StateRecord(
        table="customer_case",
        primary_key="case_id",
        row_id="1233",
        values={
            "case_id": 1233,
            "account_id": 7,
            "priority": "critical",
            "state": "open",
            "escalation": 0,
            "channel": "phone",
            "sys_created_on": "2026-01-01",
            "sys_updated_on": "2026-01-02",
        },
    )
    pruned = prune_record_attributes(
        record,
        registry=registry,
        required_attributes=("priority",),
        required_tables=("customer_case", "account"),
    )
    assert "case_id" in pruned.values and "account_id" in pruned.values
    assert "priority" in pruned.values and "state" in pruned.values
    assert "channel" not in pruned.values
    assert "sys_created_on" not in pruned.values


def test_attribute_pruning_never_empties_a_row(registry):
    record = StateRecord(
        table="contact",
        primary_key="contact_id",
        row_id="5",
        values={"contact_id": 5, "is_primary": True},
    )
    pruned = prune_record_attributes(
        record,
        registry=registry,
        required_attributes=("priority",),
        required_tables=("contact",),
    )
    assert pruned.values, "a kept row must stay identifiable"
    assert pruned.values.get("contact_id") == 5


# ---------------------------------------------------------------------------
# Blueprint sections 2 and 12: token budget and coverage gate
# ---------------------------------------------------------------------------


def test_token_estimator_is_deterministic_and_positive(registry):
    state = make_grounded_state(extra_records=(("account", "7"),))
    first = estimate_state_tokens(state)
    assert first > 0
    assert estimate_state_tokens(state) == first


def test_token_budget_accepts_a_budget_that_fits_the_selection(registry):
    """A budget at the selection's own cost passes; below the anchor floor fails."""
    state = make_grounded_state(
        table="customer_case",
        row_id="1233",
        extra_records=(("account", "7"), ("location", "9"), ("notification", "800")),
    )
    task = build_task(registry, reference_entities=("customer_case",))
    unconstrained = build_minimal_state(task, state, registry=registry)
    fitted = build_minimal_state(
        task, state, registry=registry, max_tokens=unconstrained.token_estimate
    )
    assert fitted.token_estimate <= unconstrained.token_estimate
    assert "customer_case" in {record.table for record in fitted.state.records}
    # The anchor row has a non-zero token floor; a budget below it can never
    # be met without losing grounding, so it fails closed.
    with pytest.raises(ValueError, match="token budget"):
        build_minimal_state(task, state, registry=registry, max_tokens=1)


def test_token_budget_fails_closed_when_pruning_would_break_grounding():
    """A single-row state cannot satisfy a tiny budget without losing grounding."""
    state = make_grounded_state(table="customer_case", row_id="1233")
    task = dataclasses.replace(
        build_task(registry := SchemaRegistry.from_static()),
        reference_entities=("customer_case",),
    )
    with pytest.raises(ValueError, match="token budget"):
        build_minimal_state(task, state, registry=registry, max_tokens=1)


def test_coverage_gate_fails_closed_below_tau(registry):
    """Under-coverage must raise, never return a silently weaker state.

    The gate guards the tables the *broad* state observed: the trigger here is
    a row cap that forces the cover to drop a required table, which is exactly
    the silent-drop defect the blueprint forbids.
    """
    state = make_grounded_state(
        table="customer_case",
        row_id="1233",
        extra_records=(("account", "7"),),
    )
    task = build_task(
        registry,
        reference_entities=("customer_case", "account"),
        reference_rows=(("customer_case", "1233"),),
    )
    with pytest.raises(ValueError, match="coverage gate failed"):
        build_minimal_state(task, state, registry=registry, max_rows=1, coverage_threshold=1.0)
    # Relaxing tau to what the capped selection can observe is honest. The
    # universe is {anchor, table:customer_case, table:account}; a 1-row cap
    # keeps the anchor (covering anchor + table:customer_case) = 2/3.
    result = build_minimal_state(
        task, state, registry=registry, max_rows=1, coverage_threshold=0.0
    )
    assert result.coverage_ratio == pytest.approx(2 / 3)


def test_build_minimal_state_end_to_end_keeps_minimum(registry):
    state = make_grounded_state(
        table="customer_case",
        row_id="1233",
        extra_records=(("account", "7"), ("entitlement", "3"), ("location", "9"), ("notification", "800")),
    )
    task = build_task(registry)
    result = build_minimal_state(task, state, registry=registry)
    kept_tables = {record.table for record in result.state.records}
    # The minimal arm keeps exactly the task-relevant tables: location and
    # notification are structural noise and must be dropped, not merely
    # "not required".
    assert kept_tables == {"customer_case", "account", "entitlement"}
    assert result.coverage_ratio == 1.0
    assert result.dropped_rows == (("location", "9"), ("notification", "800"))
    assert result.pruned_columns, "attribute pruning must engage"
    # Facts keep their provenance through the selection.
    for fact in result.state.facts:
        assert fact.provenance.table in kept_tables


# ---------------------------------------------------------------------------
# Blueprint section 10: coverage-driven escalation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalation_continues_until_required_tables_are_observed(registry):
    class DepthSensitiveAPI:
        """Returns more of the graph as the hop budget grows."""

        def __init__(self) -> None:
            self.environment = type("FakeEnvironment", (), {"sql": None})()
            self.budgets: list = []

        async def build_state(self, task_type, reference_id, reference_type=None, budget=None):
            self.budgets.append(budget.max_hops)
            if budget.max_hops < 3:
                return make_grounded_state(table="customer_case", row_id="1233")
            return make_grounded_state(
                table="customer_case",
                row_id="1233",
                extra_records=(("account", "7"), ("entitlement", "3")),
            )

        def represent_state(self, state, **_unused):
            return {"text": "ok", "graph": "", "json": ""}

    task = build_task(
        registry,
        reference_entities=("customer_case", "account", "entitlement"),
        reference_rows=(("customer_case", "1233"),),
    )
    api = DepthSensitiveAPI()
    adapter = StateModelAdapter(api, task, registry=registry, reader=FakeSQLReader({}))
    context = await adapter.context_for(minimal_state=False)

    assert api.budgets == [2, 3], "escalation must continue past a partial rung"
    assert {"account", "entitlement"} <= set(context.retrieved_tables)
    assert context.resolved_max_hops == 3


@pytest.mark.asyncio
async def test_escalation_delivers_deepest_state_when_coverage_is_impossible(registry):
    class PartialAPI:
        def __init__(self) -> None:
            self.environment = type("FakeEnvironment", (), {"sql": None})()

        async def build_state(self, task_type, reference_id, reference_type=None, budget=None):
            return make_grounded_state(table="customer_case", row_id="1233")

        def represent_state(self, state, **_unused):
            return {"text": "ok", "graph": "", "json": ""}

    task = build_task(
        registry,
        reference_rows=(("customer_case", "1233"),),
        reference_entities=("customer_case", "entitlement"),  # never reachable here
    )
    adapter = StateModelAdapter(
        PartialAPI(), task, registry=registry, reader=FakeSQLReader({}),
        budget=None,
    )
    context = await adapter.context_for(minimal_state=False)
    assert context.retrieved_record_count == 1, "partial coverage still delivers"
    assert context.route_status == "COMPLETE"


# ---------------------------------------------------------------------------
# Blueprint section 15: N-arm report and the restored delivery gate
# ---------------------------------------------------------------------------


def _report_record(condition: str, **overrides):
    from ablation.results import TaskRunRecord
    from dataclasses import replace

    base = TaskRunRecord(
        experiment="experiment_1",
        run_uid=f"uid-{condition}",
        condition=condition,
        task_id="task-1",
        run_index=1,
        complexity_category="TYPE_3_MULTI_HOP",
        seed_database_file="seed.sql",
        runtime_database_id="db_1",
        initial_state_fingerprint="fp",
        model_provider="openai",
        model_name="gpt-x",
        temperature=0.0,
        max_steps=15,
        tool_mode="oracle",
        orchestrator="react",
        overall_success=True,
        verifier_summary_collapsed={"total": 2, "passed": 2},
        verifier_summary_indexed={"total": 2, "passed": 2},
        verifier_results_collapsed={},
        verifier_results_indexed={},
        agent_error=False,
        error_message=None,
        steps_taken=4,
        tool_call_count=4,
        read_query_count=2,
        invalid_action_count=0,
        latency_ms=100,
        input_tokens=10,
        output_tokens=5,
        state_retrieval=None,
        started_at="",
        finished_at="",
    )
    record = replace(base, **overrides) if overrides else base
    if record.condition in (CONDITION_B1.name, CONDITION_B2.name) and "state_retrieval" not in overrides:
        record = replace(
            record,
            state_retrieval={
                "state_model_available": True,
                "retrieved_records": 4,
                "retrieved_tokens": 120,
                "context_tokens_injected": 120,
                "anchor_status": "RESOLVED",
                "state_route_status": "COMPLETE",
            },
        )
    return record


def test_three_arm_report_names_treatments_and_pairs():
    records = [
        _report_record(CONDITION_A.name, overall_success=False),
        _report_record(CONDITION_B1.name, overall_success=True),
        _report_record(CONDITION_B2.name, overall_success=True),
    ]
    report = build_report(records, conditions=(CONDITION_A, CONDITION_B1, CONDITION_B2))
    assert report["design"]["conditions"] == [
        CONDITION_A.name,
        CONDITION_B1.name,
        CONDITION_B2.name,
    ]
    assert report["design"]["treatment"] == CONDITION_B1.name
    assert report["design"]["manipulated_variable"] == "state_model_availability_and_selection_mode"
    assert set(report["headline"]) >= {
        CONDITION_A.name,
        CONDITION_B1.name,
        CONDITION_B2.name,
        "delta_tsr",
    }
    pairwise = report["pairwise"]
    assert set(pairwise) == {
        f"{CONDITION_A.name}_vs_{CONDITION_B1.name}",
        f"{CONDITION_A.name}_vs_{CONDITION_B2.name}",
        f"{CONDITION_B1.name}_vs_{CONDITION_B2.name}",
    }
    assert report["record_counts"]["by_condition"][CONDITION_B2.name] == 1


def test_delivery_gate_gates_every_state_model_arm():
    """A B2 delivery failure must withhold the effect, exactly like B1."""
    records = [
        _report_record(CONDITION_A.name),
        _report_record(
            CONDITION_B2.name,
            state_retrieval={
                "state_model_available": False,
                "retrieved_records": 0,
                "retrieved_tokens": 0,
                "context_tokens_injected": 0,
                "anchor_status": "UNRESOLVED",
                "state_route_status": "FAILED",
            },
        ),
    ]
    report = build_report(records, conditions=(CONDITION_A, CONDITION_B2))
    assert report["state_injection"]["condition"] == CONDITION_B2.name
    assert report["experimental_validity"]["clean_intervention_delivery"] is False
    assert report["headline"]["delta_tsr"] is None
    summary = summarize_state_injection(records, condition_name=CONDITION_B2.name)
    assert summary["runs"] == 1 and summary["injected"] == 0


def test_injection_summary_selects_condition_by_name_not_position():
    records = [
        _report_record(CONDITION_A.name),
        _report_record(CONDITION_B1.name),
        _report_record(CONDITION_B2.name),
    ]
    b2 = summarize_state_injection(records, condition_name=CONDITION_B2.name)
    assert b2["runs"] == 1 and b2["injected"] == 1
    assert b2["condition"] == CONDITION_B2.name


def test_parse_conditions_three_arm_round_trip():
    conditions = parse_conditions("A,B1,B2")
    assert conditions == (CONDITION_A, CONDITION_B1, CONDITION_B2)


# ---------------------------------------------------------------------------
# P1: attribute pruning must keep facts consistent with record values# P1: attribute pruning must keep facts consistent with record values
# ---------------------------------------------------------------------------


def _state_with_full_facts(registry):
    """A customer_case state whose facts cover every column of the records.

    Mirrors what ``StateBuilder`` produces: one fact per observed column, with
    provenance scoped to the record.
    """
    from csm_env.state.models import (
        FactProvenance,
        StateFact,
        StateProvenance,
        StateRecord,
    )

    case = StateRecord(
        table="customer_case",
        primary_key="case_id",
        row_id="1233",
        values={
            "case_id": 1233,
            "account_id": 7,
            "priority": "critical",
            "state": "open",
            "channel": "phone",
            "sys_created_on": "2026-01-01",
        },
    )
    account = StateRecord(
        table="account",
        primary_key="account_id",
        row_id="7",
        values={"account_id": 7, "name": "Stark", "sys_created_on": "2026-01-01"},
    )
    records = (case, account)
    facts = []
    for record in records:
        for column, value in record.values.items():
            facts.append(
                StateFact(
                    value=value,
                    provenance=FactProvenance(
                        database_id="db",
                        schema_version="1.0.0",
                        table=record.table,
                        column=column,
                        row_pk=record.row_id,
                        query_id="s000",
                        route_id="r",
                        observed_at="2026-01-01T00:00:00+00:00",
                    ),
                )
            )
    from csm_env.state.models import GroundedState, StateStatus

    prov = StateProvenance(
        database_id="db",
        schema_version="1.0.0",
        route_id="r",
        observed_at="2026-01-01T00:00:00+00:00",
        query_ids=("s000",),
    )
    return GroundedState(
        state_id="s",
        anchor=case,
        records=records,
        relations=(),
        unresolved=(),
        unresolved_details=(),
        provenance=prov,
        status=StateStatus.COMPLETE,
        schema_version="1.0.0",
        observation_generation=1,
        facts=tuple(facts),
        contradictions=(),
    )


def test_p1_pruned_columns_do_not_survive_as_facts(registry):
    """N1: after attribute pruning, no fact may reference a pruned column.

    This is the invariant the representation contract depends on: the
    agent-facing text (facts) and the record values stay consistent, so a
    column the selection removed is gone from the payload entirely.
    """
    state = _state_with_full_facts(registry)
    task = build_task(registry)
    result = build_minimal_state(
        task,
        state,
        registry=registry,
        required_tables=frozenset({"customer_case", "account"}),
        attribute_keys=("priority", "state"),
    )
    kept_columns = {
        (r.table.lower(), r.row_id): {str(c).lower() for c in r.values}
        for r in result.state.records
    }
    for fact in result.state.facts:
        key = (fact.provenance.table.lower(), str(fact.provenance.row_pk))
        column = str(fact.provenance.column).lower()
        if key in kept_columns:
            assert column in kept_columns[key], (
                f"fact {column!r} survived but was pruned from record {key}"
            )


def test_p1_token_estimate_is_rebuilt_state_footprint(registry):
    """N4/N6: the reported token estimate is the minimal state's own footprint."""
    state = make_grounded_state(
        table="customer_case",
        row_id="1233",
        extra_records=(("account", "7"), ("location", "9"), ("notification", "800")),
    )
    task = build_task(registry)
    result = build_minimal_state(task, state, registry=registry)
    assert result.token_estimate == estimate_state_tokens(result.state)


# ---------------------------------------------------------------------------
# P2: required relations gate coverage and stay realized between kept rows
# ---------------------------------------------------------------------------


def test_p2_relation_fact_covers_required_relation(registry):
    """A required relation realized by the kept rows is a cover fact."""
    state = make_grounded_state(
        table="customer_case",
        row_id="1233",
        extra_records=(("account", "7"),),
    )
    task = build_task(registry, reference_entities=("customer_case", "account"))
    covered, uncovered = realized_relation_facts(
        state,
        {(r.table.lower(), r.row_id) for r in state.records},
        (("customer_case", "account"),),
    )
    assert "relation:customer_case:account" in covered
    assert "relation:customer_case:account" not in uncovered


def test_p2_missing_relation_fails_coverage_gate(registry):
    """A required relation the selection cannot realize fails the coverage gate.

    The broad state observes both endpoints (so the relation is in the universe),
    but a 1-row cap means relation-completion cannot add the second endpoint, so
    the relation is uncovered and the gate fails closed.
    """
    state = make_grounded_state(
        table="customer_case", row_id="1233", extra_records=(("account", "7"),)
    )
    task = build_task(registry, reference_entities=("customer_case", "account"))
    with pytest.raises(ValueError, match="coverage gate failed"):
        build_minimal_state(
            task,
            state,
            registry=registry,
            required_tables=frozenset({"customer_case", "account"}),
            required_relations=(("customer_case", "account"),),
            attribute_keys=(),
            max_rows=1,
            coverage_threshold=1.0,
        )


def test_p2_relation_completion_selects_missing_endpoint(registry):
    """Relation completion keeps both endpoints of a required relation."""
    state = make_grounded_state(
        table="customer_case",
        row_id="1233",
        extra_records=(("account", "7"), ("location", "9")),
    )
    task = build_task(registry, reference_entities=("customer_case", "account"))
    result = build_minimal_state(
        task,
        state,
        registry=registry,
        required_tables=frozenset({"customer_case", "account"}),
        required_relations=(("customer_case", "account"),),
        attribute_keys=(),
    )
    kept = {(r.table.lower(), r.row_id) for r in result.state.records}
    assert ("customer_case", "1233") in kept
    assert ("account", "7") in kept


# ---------------------------------------------------------------------------
# P5.1/A2: gate-failed minimal runs are flagged so the contrast stays honest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a2_gate_failed_context_is_flagged(registry):
    """context_for must flag a gate-fired run so the report can exclude it."""
    from ablation.state_model import StateModelAdapter
    from tests.tests_support.ablation_doubles import FakeSQLReader

    state = make_grounded_state(
        table="customer_case",
        row_id="1233",
        extra_records=(("account", "7"),),
    )
    task = build_task(
        registry,
        reference_entities=("customer_case", "account"),
        reference_rows=(("customer_case", "1233"),),
    )

    class _API:
        def __init__(self, state):
            self._state = state
            self.environment = type("E", (), {"sql": None})()

        async def build_state(self, task_type, reference_id, reference_type=None, budget=None):
            return self._state

        def represent_state(self, s, **_unused):
            return {"text": "ok", "graph": "", "json": ""}

    adapter = StateModelAdapter(_API(state), task, registry=registry, reader=FakeSQLReader({}))
    # A 1-row cap forces the coverage gate to fire; the adapter must fall back
    # to the broad state AND flag the run. Patch the selector module (the
    # adapter re-imports it locally on each call), not the adapter module.
    import ablation.minimal_state as ms

    original = ms.build_minimal_state
    ms.build_minimal_state = lambda *a, **k: (_ for _ in ()).throw(
        ValueError("minimal state coverage gate failed: ['table:account'] uncovered")
    )
    try:
        context = await adapter.context_for(minimal_state=True)
    finally:
        ms.build_minimal_state = original
    assert context.selection_gate_failed is True
    # The broad state is still delivered (record count is the full state's).
    assert context.retrieved_record_count == len(state.records)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "minimal,expected",
    [
        (True, RouteStrategy.FRONTIER),
        (False, RouteStrategy.BROAD),
    ],
)
async def test_b2_retrieves_with_frontier_and_b1_with_broad(
    registry, minimal, expected
):
    """State Model 2.0 sections 2/10/13: the arms differ at *retrieval* too.

    B2 must ask the router for the frontier strategy, so the router stops at the
    first prefix meeting ``Coverage >= tau`` and ``lambda_q * |queries|`` is
    genuinely minimized. B1 must keep the eager strategy, which pins its
    retrieval cost to State Model 1.0 -- so the only manipulated variable left
    is the *selection* of what is delivered.
    """
    from ablation.state_model import StateModelAdapter
    from tests.tests_support.ablation_doubles import FakeSQLReader

    state = make_grounded_state(
        table="customer_case",
        row_id="1233",
        extra_records=(("account", "7"),),
    )
    task = build_task(
        registry,
        reference_entities=("customer_case", "account"),
        reference_rows=(("customer_case", "1233"),),
    )

    class _StrategyAPI:
        def __init__(self, state):
            self._state = state
            self.strategies = []
            self.environment = type("E", (), {"sql": None})()

        async def build_state(
            self, task_type, reference_id, reference_type=None, budget=None, strategy=None
        ):
            self.strategies.append(strategy)
            return self._state

        def represent_state(self, s, **_unused):
            return {"text": "ok", "graph": "", "json": ""}

    api = _StrategyAPI(state)
    adapter = StateModelAdapter(api, task, registry=registry, reader=FakeSQLReader({}))
    await adapter.context_for(minimal_state=minimal)

    assert api.strategies, "the adapter never reached the environment"
    assert expected in api.strategies, (
        f"minimal_state={minimal} requested {api.strategies!r}, expected {expected!r}"
    )

