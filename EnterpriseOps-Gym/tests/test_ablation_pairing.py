"""REQ-010 pairing gate, explicit step budget, and report summaries."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ablation.conditions import CONDITION_A, CONDITION_B
from ablation.report import build_report, summarize_state_injection
from ablation.results import TaskRunRecord, write_run_record
from ablation.runner import (
    reconcile_fingerprint_pairs,
    reconcile_pair_fingerprints,
)
from ablation.stats import pair_records


def make_record(**overrides) -> TaskRunRecord:
    """A minimal, valid record with sensible defaults (same shape as runner tests)."""
    base = TaskRunRecord(
        experiment="experiment_1",
        run_uid="uid-pair",
        condition=CONDITION_A.name,
        task_id="task-1",
        run_index=1,
        complexity_category="TYPE_3_MULTI_HOP",
        seed_database_file="seed.sql",
        runtime_database_id="db_1",
        initial_state_fingerprint="fp",
        model_provider="openai",
        model_name="gpt-x",
        temperature=0.0,
        max_steps=50,
        tool_mode="oracle",
        orchestrator="react",
        overall_success=True,
        verifier_summary_collapsed={"total": 2, "passed": 2},
        verifier_summary_indexed={"total": 2, "passed": 2},
        verifier_results_collapsed={},
        verifier_results_indexed={},
        agent_error=False,
        error_message=None,
        steps_taken=5,
        tool_call_count=5,
        read_query_count=2,
        invalid_action_count=0,
        latency_ms=1000,
        input_tokens=10,
        output_tokens=5,
        state_retrieval=None,
        started_at="",
        finished_at="",
    )
    record = replace(base, **overrides) if overrides else base
    if record.condition == CONDITION_B.name and "state_retrieval" not in overrides:
        record = replace(
            record,
            state_retrieval={
                "state_model_available": True,
                "retrieved_records": 1,
                "retrieved_tokens": 20,
                "context_tokens_injected": 20,
                "anchor_status": "RESOLVED",
                "state_route_status": "COMPLETE",
            },
        )
    return record


def _orchestrator_kwargs() -> dict:
    """Local copy of the state-model tests' orchestrator kwargs."""
    from benchmark.models import BenchmarkConfig
    from tests.tests_support.ablation_doubles import FakeLLMClient, FakeMcpClient

    return {
        "llm_client": FakeLLMClient(),
        "mcp_clients": {"sn-csm-server": FakeMcpClient()},
        "tool_to_server_mapping": {},
        "available_tools": [],
        "config": BenchmarkConfig(
            system_prompt="SYSTEM POLICY",
            user_prompt="TASK PROMPT",
            verifiers=[],
            number_of_runs=1,
        ),
    }


# ---------- reconcile_pair_fingerprints ----------


def test_matching_fingerprints_stamp_the_pair():
    baseline = make_record(condition=CONDITION_A.name, initial_state_fingerprint="fp-1")
    treatment = make_record(condition=CONDITION_B.name, initial_state_fingerprint="fp-1")
    left, right, matched = reconcile_pair_fingerprints(baseline, treatment)
    assert matched is True
    assert left.fingerprint_match is True and right.fingerprint_match is True
    assert left.fingerprint_mismatch_reason is None
    assert not left.agent_error and not right.agent_error
    # Inputs are never mutated.
    assert baseline.fingerprint_match is None


def test_mismatched_fingerprints_flag_both_records():
    baseline = make_record(
        condition=CONDITION_A.name, initial_state_fingerprint="fp-1", overall_success=True
    )
    treatment = make_record(condition=CONDITION_B.name, initial_state_fingerprint="fp-2")
    left, right, matched = reconcile_pair_fingerprints(baseline, treatment)
    assert matched is False
    assert left.agent_error and right.agent_error
    assert left.fingerprint_match is False and right.fingerprint_match is False
    assert "fp-1 != fp-2" in left.error_message
    assert left.fingerprint_mismatch_reason == left.error_message


def test_missing_fingerprint_flags_the_pair_as_unverifiable():
    baseline = make_record(condition=CONDITION_A.name, initial_state_fingerprint=None)
    treatment = make_record(condition=CONDITION_B.name, initial_state_fingerprint="fp-2")
    left, right, matched = reconcile_pair_fingerprints(baseline, treatment)
    assert matched is False
    assert left.agent_error and right.agent_error
    assert "could not be verified" in left.error_message


# ---------- reconcile_fingerprint_pairs ----------


def test_reconcile_flags_mismatched_pairs_in_place():
    records = [
        make_record(
            task_id="t1",
            condition=CONDITION_A.name,
            initial_state_fingerprint="fp-a",
            overall_success=True,
        ),
        make_record(
            task_id="t1",
            condition=CONDITION_B.name,
            initial_state_fingerprint="fp-b",
            overall_success=True,
        ),
        make_record(
            task_id="t2",
            condition=CONDITION_A.name,
            initial_state_fingerprint="fp-x",
            overall_success=False,
        ),
        make_record(
            task_id="t2",
            condition=CONDITION_B.name,
            initial_state_fingerprint="fp-x",
            overall_success=True,
        ),
    ]
    reconciled, summaries = reconcile_fingerprint_pairs(records)
    assert len(summaries) == 2
    flagged = [pair for pair in summaries if not pair["fingerprint_match"]]
    assert len(flagged) == 1 and flagged[0]["task_id"] == "t1"

    by_key = {(r.task_id, r.condition): r for r in reconciled}
    assert by_key[("t1", CONDITION_A.name)].agent_error
    assert by_key[("t1", CONDITION_B.name)].agent_error
    assert not by_key[("t2", CONDITION_A.name)].agent_error

    # The good pair is untouched by the gate and now scores on its own.
    outcomes = pair_records(reconciled)
    assert outcomes.total == 1
    assert outcomes.treatment_only == 1
    assert outcomes.excluded == 1


def test_reconcile_skips_already_reviewed_pairs():
    records = [
        make_record(
            condition=CONDITION_A.name,
            initial_state_fingerprint="fp-1",
            fingerprint_match=False,
            fingerprint_mismatch_reason="already flagged",
            agent_error=True,
        ),
        make_record(
            condition=CONDITION_B.name,
            initial_state_fingerprint="fp-1",
            fingerprint_match=False,
            fingerprint_mismatch_reason="already flagged",
            agent_error=True,
        ),
    ]
    reconciled, summaries = reconcile_fingerprint_pairs(records)
    assert summaries == [
        {
            "task_id": records[0].task_id,
            "run_index": records[0].run_index,
            "fingerprint_match": False,
            "reason": "already flagged",
        }
    ]
    assert all(record.agent_error for record in reconciled)


def test_reconcile_ignores_singletons():
    records = [make_record(condition=CONDITION_A.name, initial_state_fingerprint=None)]
    reconciled, summaries = reconcile_fingerprint_pairs(records)
    assert summaries == []
    assert reconciled == records


def test_gate_excludes_mismatch_from_mcnemar_inputs():
    """A mismatched pair must not contribute a success to either arm."""
    records = [
        make_record(
            condition=CONDITION_A.name, initial_state_fingerprint="fp-1", overall_success=True
        ),
        make_record(
            condition=CONDITION_B.name, initial_state_fingerprint="fp-9", overall_success=True
        ),
    ]
    reconciled, _ = reconcile_fingerprint_pairs(records)
    outcomes = pair_records(reconciled)
    assert outcomes.total == 0
    assert outcomes.excluded == 1


# ---------- explicit step budget ----------


def test_max_steps_is_forwarded_to_both_arms():
    from ablation.runner import ProbedReactOrchestrator, ProbedStateModelReactOrchestrator
    from ablation.state_model import StateContextPolicy

    kwargs = _orchestrator_kwargs()
    baseline = ProbedReactOrchestrator(**kwargs, max_iterations=17)
    treatment = ProbedStateModelReactOrchestrator(
        **kwargs,
        max_iterations=17,
        state_task=None,
        state_environment_factory=lambda client, **kw: object(),
    )
    assert baseline.max_iterations == treatment.max_iterations == 17
    assert baseline.state_context_policy == treatment.state_context_policy == (
        StateContextPolicy.PRE_ACTION_ONCE
    )


def test_record_from_run_carries_the_observed_budget():
    from ablation.runner import MAX_STEPS_DEFAULT, record_from_run
    from benchmark.models import LLMConfig
    from ablation.task_registry import analyze_prompt
    from csm_env import SchemaRegistry

    registry = SchemaRegistry.from_static()
    signals = analyze_prompt("Register a case for Stark Industries.", ["create_new_case"], task_id="t1", registry=registry)
    from ablation.task_registry import TaskRecord

    task = TaskRecord(
        task_id="t1",
        task_config_path="p.json",
        seed_database_file="s.sql",
        database_id=None,
        initial_state_fingerprint=None,
        complexity_category="TYPE_1_SINGLE_ENTITY",
        category_rationale="r",
        reference_entities=signals.matched_tables,
        reference_rows=(),
        reference_names=(),
        verifier_count=1,
        selected_tools_count=1,
        number_of_runs=1,
        reset_database_between_runs=True,
        ggqr_task_type="resolve_case",
        signals=signals,
    )
    llm = LLMConfig(llm_provider="openai", llm_model="gpt-x", llm_api_key="k")
    record = record_from_run(
        condition=CONDITION_A,
        task=task,
        run_index=1,
        llm_config=llm,
        result={},
        run={},
        tool_mode="oracle",
        orchestrator_name="react",
        max_steps=23,
    )
    assert record.max_steps == 23
    assert record.max_steps != MAX_STEPS_DEFAULT


# ---------- report summaries ----------


def _injected_record(**overrides):
    return make_record(
        condition=CONDITION_B.name,
        state_retrieval={
            "state_model_available": True,
            "retrieved_records": 4,
            "retrieved_tokens": 120,
            "context_tokens_injected": 120,
            "anchor_status": "RESOLVED",
            "state_route_status": "COMPLETE",
        },
        **overrides,
    )


def test_injection_summary_counts_delivered_blocks():
    records = [
        _injected_record(),
        make_record(
            condition=CONDITION_B.name,
            state_retrieval={
                "state_model_available": True,
                "retrieved_records": 0,
                "retrieved_tokens": 0,
                "context_tokens_injected": 0,
                "anchor_status": "UNRESOLVED",
                "state_route_status": "UNRESOLVED",
            },
        ),
        make_record(condition=CONDITION_B.name, state_retrieval=None),
    ]
    summary = summarize_state_injection(records)
    assert summary["runs"] == 3
    assert summary["injected"] == 1
    assert summary["injection_rate"] == pytest.approx(1 / 3)
    assert summary["empty_context"] == 1
    assert summary["unresolved_anchor"] == 1


def test_injection_summary_handles_an_empty_treatment_arm():
    summary = summarize_state_injection([make_record(condition=CONDITION_A.name)])
    assert summary["runs"] == 0
    assert summary["injection_rate"] is None


def test_report_includes_pairing_and_injection_sections(tmp_path):
    records = [
        make_record(
            task_id="t1",
            condition=CONDITION_A.name,
            initial_state_fingerprint="fp-1",
            overall_success=True,
        ),
        make_record(
            task_id="t1",
            condition=CONDITION_B.name,
            initial_state_fingerprint="fp-1",
            overall_success=False,
            state_retrieval={
                "state_model_available": True,
                "retrieved_records": 4,
                "retrieved_tokens": 120,
                "context_tokens_injected": 120,
                "anchor_status": "RESOLVED",
                "state_route_status": "COMPLETE",
            },
        ),
    ]
    for record in records:
        write_run_record(record, tmp_path)

    report = build_report(records)
    pairing = report["fingerprint_pairing"]
    assert pairing["pairs_reviewed"] == 1
    assert pairing["pairs_matched"] == 1
    assert pairing["pairs_flagged"] == 0
    assert report["state_injection"]["runs"] == 1
    assert report["state_injection"]["injected"] == 1


def test_report_pairing_gate_excludes_mismatch_from_headline():
    records = [
        make_record(
            condition=CONDITION_A.name,
            initial_state_fingerprint="fp-1",
            overall_success=True,
        ),
        make_record(
            condition=CONDITION_B.name,
            initial_state_fingerprint="fp-2",
            overall_success=True,
        ),
    ]
    report = build_report(records)
    pairing = report["fingerprint_pairing"]
    assert pairing["pairs_flagged"] == 1
    # The mismatched pair is excluded from every success metric: only the
    # agent-error records remain, so the scoreable-run count is zero and the
    # headline difference is withheld rather than rendered as "no effect".
    assert report["headline"][CONDITION_A.name]["tsr"] == 0.0
    assert report["headline"]["delta_tsr"] is None
    assert report["paired"]["outcomes"]["total_pairs"] == 0
    assert report["paired"]["outcomes"]["excluded"] == 1
