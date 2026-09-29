"""Condition B state adapter, fairness guard, and orchestrator injection."""

from __future__ import annotations

import dataclasses

import pytest

from ablation.state_model import (
    ANCHOR_TABLE_PRIORITY,
    StateContextPolicy,
    StateModelAdapter,
    assert_no_verifier_metadata,
    estimate_tokens,
    task_conditioned_minimal_state,
)
from ablation.state_model_orchestrator import (
    DEFAULT_STATE_QUERY_BUDGET,
    STATE_CONTEXT_HEADER,
    StateModelDeliveryError,
    StateModelReactOrchestrator,
    compose_state_user_prompt,
)
from ablation.task_registry import TaskRecord, analyze_prompt
from benchmark.models import BenchmarkConfig
from csm_env import SchemaRegistry
from tests.tests_support.ablation_doubles import (
    FakeCsmEnvAPI,
    FakeLLMClient,
    FakeMcpClient,
    FakeSQLReader,
    make_grounded_state,
)

PROMPT = (
    "Marc Henry needs to address a critical situation for Stark Industries. Update the existing "
    "entitlement for their Palo Alto Networks PA-3220 Variant 8 to our highest Enterprise support "
    "level, ensuring 24x7 coverage. Register a new critical case for this device and assign it."
)


@pytest.fixture(scope="module")
def registry() -> SchemaRegistry:
    return SchemaRegistry.from_static()


def build_task(registry: SchemaRegistry, **overrides) -> TaskRecord:
    """A controlled TaskRecord with prompt-derived signals but no verifier data."""
    signals = analyze_prompt(PROMPT, ["find_user", "update_case"], task_id="t1", registry=registry)
    base = TaskRecord(
        task_id="t1",
        task_config_path="data/revised/csm/t1.json",
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


@pytest.mark.asyncio
async def test_fingerprint_is_deterministic_and_read_only(registry):
    from ablation.fingerprint import initial_state_fingerprint

    reader = FakeSQLReader(
        {
            "customer_case": [{"case_id": 1, "state": "open"}, {"case_id": 2, "state": "closed"}],
            "account": [{"account_id": 7, "name": "Stark Industries"}],
        }
    )
    first = await initial_state_fingerprint(reader, registry)
    second = await initial_state_fingerprint(reader, registry)
    assert first == second
    assert reader.read_only, "fingerprinting must never mutate the database"
    assert not reader.used_select_star

    reader.add_row("customer_case", {"case_id": 3, "state": "new"})
    assert await initial_state_fingerprint(reader, registry) != first


@pytest.mark.asyncio
async def test_fingerprint_covers_every_registered_table(registry):
    from ablation.fingerprint import initial_state_fingerprint

    reader = FakeSQLReader({table: [] for table in registry.tables()})
    digest = await initial_state_fingerprint(reader, registry)
    assert len(digest) == 64
    covered = {
        statement.split("FROM ")[1].split(" ")[0].rstrip(";") for statement in reader.statements
    }
    assert covered == set(registry.tables())


def test_state_adapter_rejects_verifier_metadata():
    with pytest.raises(TypeError, match="raw task-config mapping"):
        assert_no_verifier_metadata({"user_prompt": "x", "verifiers": []})
    with pytest.raises(ValueError, match="verifier-only key"):
        assert_no_verifier_metadata({"expected_value": "critical"})
    with pytest.raises(ValueError, match="verifier-only key"):
        assert_no_verifier_metadata({"nested": {"validation_config": {}}})

    leaky = type("Leaky", (), {"verifiers": [], "reference_rows": ()})()
    with pytest.raises(ValueError, match="verifier-only attribute"):
        assert_no_verifier_metadata(leaky)


def test_state_adapter_refuses_leaky_task(registry):
    leaky = type("Leaky", (), {"verifiers": [], "reference_rows": ()})()
    with pytest.raises(ValueError):
        StateModelAdapter(object(), leaky, registry=registry)


@pytest.mark.asyncio
async def test_anchor_resolves_by_name_against_real_rows(registry):
    reader = FakeSQLReader(
        {
            "account": [{"account_id": 7, "name": "Stark Industries"}],
            "product": [{"product_id": 42, "name": "Palo Alto Networks PA-3220 Variant 8"}],
        }
    )
    adapter = StateModelAdapter(object(), build_task(registry), registry=registry, reader=reader)
    anchor = await adapter.resolve_anchor()

    assert anchor.resolved
    assert (anchor.table, anchor.row_id) == ("account", "7")
    assert anchor.strategy == "name_match"
    assert "Stark Industries" in anchor.evidence
    assert reader.read_only


@pytest.mark.asyncio
async def test_full_person_name_resolves_in_one_table_lookup(registry):
    reader = FakeSQLReader(
        {
            "user": [
                {"user_id": 47, "first_name": "Michael", "last_name": "Ward"},
            ]
        }
    )
    signals = analyze_prompt(
        "Michael Ward needs this assignment changed.",
        ["find_user"],
        task_id="person",
        registry=registry,
    )
    task = build_task(
        registry,
        task_id="person",
        signals=signals,
        reference_entities=("user",),
        reference_names=(("user", "Michael Ward"),),
    )
    adapter = StateModelAdapter(object(), task, registry=registry, reader=reader)
    anchor = await adapter.resolve_anchor()

    assert anchor.resolved
    assert (anchor.table, anchor.row_id) == ("user", "47")
    assert anchor.lookups == 1
    assert len(reader.statements) == 1


@pytest.mark.asyncio
async def test_manifest_anchor_short_circuits_lookups(registry):
    reader = FakeSQLReader({"account": [{"account_id": 7, "name": "Stark Industries"}]})
    task = build_task(registry, reference_rows=(("account", "7"),))
    adapter = StateModelAdapter(object(), task, registry=registry, reader=reader)
    anchor = await adapter.resolve_anchor()

    assert anchor.strategy == "manifest_anchor"
    assert anchor.lookups == 0
    assert reader.statements == []


@pytest.mark.asyncio
async def test_unresolvable_anchor_is_reported_not_guessed(registry):
    reader = FakeSQLReader({"account": []})
    task = build_task(registry)
    adapter = StateModelAdapter(object(), task, registry=registry, reader=reader, max_lookups=6)
    context = await adapter.context_for(minimal_state=False)

    assert context.route_status == "UNRESOLVED"
    assert context.is_empty
    assert context.retrieved_record_count == 0
    assert context.error
    assert reader.read_only


@pytest.mark.asyncio
async def test_context_carries_only_environment_entities(registry):
    reader = FakeSQLReader({"account": [{"account_id": 7, "name": "Stark Industries"}]})
    api = FakeCsmEnvAPI(make_grounded_state(extra_records=(("account", "7"),)))
    adapter = StateModelAdapter(api, build_task(registry), registry=registry, reader=reader)
    context = await adapter.context_for(minimal_state=False)

    assert context.route_status == "COMPLETE"
    assert context.reference_type == "account"
    assert context.reference_id == "7"
    for table in context.retrieved_tables:
        assert registry.require_table(table) is not None
    assert context.retrieved_token_count == estimate_tokens(context.rendered_text)
    assert api.calls[0][0] == "build_state"
    assert api.calls[0][2] == "account"


def test_anchor_table_priority_covers_the_registry(registry):
    assert set(ANCHOR_TABLE_PRIORITY) == set(registry.tables())


def test_token_estimate_is_deterministic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("x" * 400) == 100


def test_state_context_policy_validates():
    assert StateContextPolicy.validate(StateContextPolicy.PRE_ACTION_ONCE)
    assert StateContextPolicy.validate(StateContextPolicy.PRE_ACTION_PER_STEP)
    with pytest.raises(ValueError):
        StateContextPolicy.validate("every_step")


def test_condition_names_include_broad_and_minimal_variants():
    from ablation.conditions import (
        CONDITION_A,
        CONDITION_B1,
        CONDITION_B2,
        condition_from_name,
        parse_conditions,
    )

    assert CONDITION_A.state_model is False
    assert CONDITION_B1.state_model is True and CONDITION_B1.minimal_state is False
    assert CONDITION_B2.state_model is True and CONDITION_B2.minimal_state is True
    assert condition_from_name("B1") == CONDITION_B1
    assert condition_from_name("B2") == CONDITION_B2
    assert parse_conditions("A,B1,B2") == (CONDITION_A, CONDITION_B1, CONDITION_B2)


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        system_prompt="SYSTEM POLICY",
        user_prompt="TASK PROMPT",
        verifiers=[],
        number_of_runs=1,
    )


def _orchestrator_kwargs() -> dict:
    return {
        "llm_client": FakeLLMClient(),
        "mcp_clients": {"sn-csm-server": FakeMcpClient()},
        "tool_to_server_mapping": {},
        "available_tools": [],
        "config": _config(),
        "max_iterations": 3,
    }


def _user_messages(result: dict) -> list:
    return [
        entry["content"] for entry in result["conversation_flow"] if entry["type"] == "user_message"
    ]


@pytest.mark.asyncio
async def test_condition_a_injects_nothing():
    from ablation.runner import ProbedReactOrchestrator

    orchestrator = ProbedReactOrchestrator(**_orchestrator_kwargs())
    result = await orchestrator.execute()
    user_messages = _user_messages(result)
    assert user_messages == ["TASK PROMPT"]
    assert STATE_CONTEXT_HEADER not in user_messages[0]
    assert orchestrator.get_result_metadata()["state_retrieval"] is None


@pytest.mark.asyncio
async def test_condition_a_records_the_pre_run_fingerprint():
    from ablation.runner import ProbedReactOrchestrator

    async def probe(client):
        assert client.database_id == "db_test"
        return "fingerprint-A"

    orchestrator = ProbedReactOrchestrator(**_orchestrator_kwargs(), probe=probe)
    await orchestrator.execute()
    metadata = orchestrator.get_result_metadata()
    assert metadata["pre_state_fingerprint"] == "fingerprint-A"
    assert metadata["state_retrieval"] is None


@pytest.mark.asyncio
async def test_fingerprint_probe_failure_does_not_fail_the_run():
    from ablation.runner import ProbedReactOrchestrator

    async def probe(client):
        raise RuntimeError("sql runner unreachable")

    orchestrator = ProbedReactOrchestrator(**_orchestrator_kwargs(), probe=probe)
    result = await orchestrator.execute()
    assert _user_messages(result) == ["TASK PROMPT"]
    assert orchestrator.get_result_metadata()["pre_state_fingerprint"] is None


@pytest.mark.asyncio
async def test_condition_b_injects_exactly_one_labelled_block(registry):
    orchestrator = StateModelReactOrchestrator(
        **_orchestrator_kwargs(),
        state_task=build_task(registry, reference_rows=(("account", "7"),)),
        state_environment_factory=lambda client, **kwargs: FakeCsmEnvAPI(
            make_grounded_state(extra_records=(("account", "7"),))
        ),
    )
    result = await orchestrator.execute()
    user_messages = _user_messages(result)
    assert len(user_messages) == 1
    injected = user_messages[0]
    assert injected.startswith("TASK PROMPT")
    assert injected.count(STATE_CONTEXT_HEADER) == 1
    assert orchestrator.get_result_metadata()["state_retrieval"]["retrieved_records"] == 2
    # The orchestrator must not leave a mutated config behind.
    assert orchestrator.config.user_prompt == "TASK PROMPT"


@pytest.mark.asyncio
async def test_condition_b_keeps_identical_step_budget_and_tools(registry):
    from ablation.runner import ProbedReactOrchestrator, ProbedStateModelReactOrchestrator

    baseline = ProbedReactOrchestrator(**_orchestrator_kwargs())
    treatment = ProbedStateModelReactOrchestrator(
        **_orchestrator_kwargs(),
        state_task=build_task(registry, reference_rows=(("account", "7"),)),
        state_environment_factory=lambda client, **kwargs: FakeCsmEnvAPI(make_grounded_state()),
    )
    await baseline.execute()
    await treatment.execute()
    assert baseline.max_iterations == treatment.max_iterations
    assert baseline.available_tools == treatment.available_tools
    assert baseline.state_context_policy == treatment.state_context_policy == (
        StateContextPolicy.PRE_ACTION_ONCE
    )
    assert baseline.get_result_metadata()["pre_state_fingerprint"] is None
    assert treatment.get_result_metadata()["pre_state_fingerprint"] is None
    assert treatment.get_result_metadata()["state_retrieval"]["retrieved_records"] == 1


@pytest.mark.asyncio
async def test_state_failure_fails_closed_and_never_falls_back_to_baseline(registry):
    class Exploding:
        """A state model that resolves its anchor but cannot build state."""

        def __init__(self) -> None:
            self.environment = type("FakeEnvironment", (), {"sql": None})()

        async def build_state(self, *args, **kwargs):
            raise RuntimeError("router unavailable")

        def represent_state(self, state, **_unused):
            return {"text": "", "graph": "", "json": ""}

    llm = FakeLLMClient()
    orchestrator = StateModelReactOrchestrator(
        llm_client=llm,
        mcp_clients={"sn-csm-server": FakeMcpClient()},
        tool_to_server_mapping={},
        available_tools=[],
        config=_config(),
        max_iterations=3,
        state_task=build_task(registry, reference_rows=(("account", "7"),)),
        state_environment_factory=lambda client, **kwargs: Exploding(),
    )
    with pytest.raises(StateModelDeliveryError, match="state model delivery failed"):
        await orchestrator.execute()
    assert llm.prompts == []
    telemetry = orchestrator.get_result_metadata()["state_retrieval"]
    assert telemetry["state_model_available"] is False
    assert telemetry["state_route_status"] == "FAILED"
    assert "router unavailable" in telemetry["state_error"]


def test_condition_b_uses_three_hop_default_budget():
    assert DEFAULT_STATE_QUERY_BUDGET.max_hops == 3


def test_task_conditioned_minimal_state_prunes_irrelevant_records(registry):
    task = build_task(
        registry,
        reference_entities=("account",),
        reference_names=(("account", "Stark Industries"),),
    )
    state = make_grounded_state(
        table="customer_case",
        row_id="1233",
        extra_records=(("account", "7"), ("location", "9"), ("notification", "800")),
    )

    minimal = task_conditioned_minimal_state(task, state)

    tables = {record.table for record in minimal.records}
    assert tables == {"customer_case", "account"}
    assert minimal.anchor.table == "customer_case"
    assert any(record.table == "account" for record in minimal.records)
    assert "location" not in tables
    assert "notification" not in tables


@pytest.mark.asyncio
async def test_empty_state_context_fails_closed(registry):
    class EmptyState:
        def __init__(self) -> None:
            self.environment = type("FakeEnvironment", (), {"sql": None})()

        async def build_state(self, *args, **kwargs):
            return make_grounded_state(status="UNRESOLVED")

        def represent_state(self, state, **_unused):
            return {"text": "", "graph": "", "json": ""}

    llm = FakeLLMClient()
    orchestrator = StateModelReactOrchestrator(
        llm_client=llm,
        mcp_clients={"sn-csm-server": FakeMcpClient()},
        tool_to_server_mapping={},
        available_tools=[],
        config=_config(),
        max_iterations=3,
        state_task=build_task(registry, reference_rows=(("account", "7"),)),
        state_environment_factory=lambda client, **kwargs: EmptyState(),
    )
    with pytest.raises(StateModelDeliveryError, match="route=UNRESOLVED"):
        await orchestrator.execute()
    assert llm.prompts == []


def test_compose_state_user_prompt_preserves_the_task():
    from ablation.state_model import AnchorResolution, StateContext

    context = StateContext(
        task_id="t1",
        task_type="resolve_case",
        reference_type="account",
        reference_id="7",
        route_status="COMPLETE",
        rendered_text="account 7 is present.",
        retrieved_tables=("account",),
        retrieved_rows=(("account", "7"),),
        retrieved_record_count=1,
        retrieved_fact_count=1,
        retrieved_token_count=5,
        retrieval_latency_ms=1.0,
        unresolved=(),
        contradictions=(),
        schema_version="1.0.0",
        observed_at="now",
        token_estimator="chars_div_4",
        anchor=AnchorResolution("t1", "account", "7", "RESOLVED", "name_match", "e", 1),
    )
    composed = compose_state_user_prompt("TASK PROMPT", context)
    assert composed.startswith("TASK PROMPT")
    assert "account 7 is present." in composed
    assert STATE_CONTEXT_HEADER in composed
