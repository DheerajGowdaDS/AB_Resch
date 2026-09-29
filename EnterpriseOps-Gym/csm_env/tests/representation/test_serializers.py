from __future__ import annotations

import asyncio
import json

import pytest

from csm_env.query.models import TaskRequest
from csm_env.representation import (
    DEFAULT_TEXT_MAX_CHARS,
    FidelityReport,
    to_dict,
    to_graph_text,
    to_json,
    to_text,
    verify_fidelity,
)
from csm_env.state import StateBuilder
from tests_support.helpers import make_router


@pytest.fixture()
def state(registry, reader):
    router = make_router(registry, reader)
    outcome = asyncio.run(
        router.route(TaskRequest(task_type="resolve_case", reference_type="customer_case", reference_id=1233))
    )
    return StateBuilder(registry, database_id="fixture-db").build(outcome)


def test_json_output_is_deterministic(state):
    first = to_json(state)
    second = to_json(state)
    assert first == second
    parsed = json.loads(first)
    assert parsed["representation_version"] == "1.1.0"
    assert parsed["state_id"].startswith("customer_case:1233")


def test_json_round_trip_preserves_critical_facts(state):
    parsed = json.loads(to_json(state))
    records = {(r["table"], r["row_id"]): r for r in parsed["records"]}
    assert records[("customer_case", "1233")]["values"]["state"] == "open"
    assert records[("account", "10")]["values"]["name"] == "SynthCorp"
    assert parsed["anchor"]["table"] == "customer_case"
    assert parsed["relations"], "relations must be present in JSON"


def test_graph_text_contains_relations(state):
    text = to_graph_text(state)
    assert "ANCHOR customer_case:1233" in text
    assert "-[BELONGS_TO]->" in text


def test_text_output_summarizes_state(state):
    text = to_text(state)
    assert "customer_case" in text
    assert "SynthCorp" in text
    assert "Provenance:" in text


def test_text_truncation_is_marked(state):
    text = to_text(state, max_chars=80)
    assert len(text) <= 80
    assert text.endswith("[TRUNCATED]")


def test_serialization_does_not_mutate_state(state):
    records_before = len(state.records)
    facts_before = len(state.facts)
    to_dict(state)
    to_text(state)
    to_graph_text(state)
    assert len(state.records) == records_before
    assert len(state.facts) == facts_before


def test_fidelity_passes_on_complete_state(state):
    report = verify_fidelity(state)
    assert isinstance(report, FidelityReport)
    assert report.passed, report.failures
    assert report.representation_version == "1.1.0"


def test_json_redacts_sensitive_keys(registry, reader):
    router = make_router(registry, reader)
    outcome = asyncio.run(
        router.route(TaskRequest(task_type="resolve_case", reference_type="customer_case", reference_id=1233))
    )
    state = StateBuilder(registry, database_id="fixture-db").build(outcome)
    payload = to_dict(state, redact_secrets=True)
    serialized = json.dumps(payload)
    assert "authorization" not in serialized.lower()
    assert "api_key" not in serialized.lower()
