"""Offline tests for the Phase 2/3/4 diagnostic gates.

Phase 2 validates the intervention itself, Phase 3 proves the evaluator is
unambiguous, and Phase 4 proves the step budget is not censoring runs. All three
recompute from records only, so no server or LLM is required.
"""

from __future__ import annotations

import pytest

from ablation.horizon import (
    CENSORED_FRACTION_THRESHOLD,
    analyse_horizon,
    recommend_max_steps,
)
from ablation.results import TaskRunRecord
from ablation.state_validation import (
    build_state_report,
    diagnose_run,
    evaluate_targets,
    render_state_report_markdown,
)
from ablation.verifier_validation import (
    build_verifier_validation_report,
    task_success,
    validate_task_verifiers,
)

#: A real pinned CSM task known to declare duplicate verifier names.
TASK_WITH_DUPLICATES = (
    "data/revised/csm/task_20251205_153330_906_a8eea1c0_8c7a6205.json"
)


def _record(**overrides) -> TaskRunRecord:
    """Build a persisted-style record with sensible defaults."""
    payload = {
        "experiment": "experiment_1",
        "run_uid": "uid",
        "condition": "B_state_model",
        "task_id": "task_x",
        "run_index": 1,
        "complexity_category": "TYPE_1_SINGLE_ENTITY",
        "seed_database_file": "seed.sql",
        "runtime_database_id": "db",
        "initial_state_fingerprint": "fp",
        "model_provider": "p",
        "model_name": "m",
        "temperature": 0.0,
        "max_steps": 15,
        "tool_mode": "oracle",
        "orchestrator": "react",
        "overall_success": False,
        "verifier_summary_collapsed": {},
        "verifier_summary_indexed": {},
        "verifier_results_collapsed": {},
        "verifier_results_indexed": {},
        "agent_error": False,
        "error_message": None,
        "steps_taken": 3,
        "tool_call_count": 2,
        "read_query_count": 1,
        "invalid_action_count": 0,
        "latency_ms": 10,
        "input_tokens": 1,
        "output_tokens": 1,
        "state_retrieval": {
            "state_model_available": True,
            "state_route_status": "COMPLETE",
            "anchor_status": "RESOLVED",
            "anchor_table": "customer_case",
            "anchor_row_id": "1",
            "retrieved_records": 4,
            "retrieved_tables": 2,
            "retrieved_table_names": ["customer_case", "account"],
            "retrieved_rows": [["customer_case", "1"], ["account", "1"]],
            "context_tokens_injected": 100,
        },
        "started_at": "t0",
        "finished_at": "t1",
    }
    payload.update(overrides)
    return TaskRunRecord.from_dict(payload)


# ---------------------------------------------------------------------------
# Phase 2 - state-model validation
# ---------------------------------------------------------------------------


def test_diagnose_run_reads_the_delivery_chain():
    diagnostic = diagnose_run(_record())
    assert diagnostic.anchor_table == "customer_case"
    assert diagnostic.route_status == "COMPLETE"
    assert diagnostic.structurally_valid is True
    assert diagnostic.retrieved_tables == 2
    assert diagnostic.retrieved_records == 4
    assert diagnostic.delivered is True


def test_diagnose_run_returns_none_without_telemetry():
    assert diagnose_run(_record(state_retrieval=None, condition="A_baseline")) is None


def test_state_report_computes_delivery_rate_and_targets():
    failed = {
        "state_model_available": False,
        "state_route_status": "FAILED",
        "anchor_status": "UNRESOLVED",
        "retrieved_records": 0,
        "retrieved_tables": 0,
        "context_tokens_injected": 0,
    }
    report = build_state_report(
        [_record(), _record(run_index=2, state_retrieval=failed)],
        condition_name="B_state_model",
    )
    assert report["runs_with_telemetry"] == 2
    assert report["delivered"] == 1
    assert report["delivery_rate"] == 0.5
    assert report["targets"]["delivery"]["met"] is False


def test_state_report_meets_targets_at_full_delivery():
    report = build_state_report([_record()], condition_name="B_state_model")
    assert report["delivery_rate"] == 1.0
    assert evaluate_targets(report)["delivery"] is True



# ---------------------------------------------------------------------------
# Phase 3 - evaluator validity
# ---------------------------------------------------------------------------


def test_task_success_requires_every_verifier_to_pass():
    assert task_success({"a": {"passed": True}, "b": {"passed": True}}) is True
    assert task_success({"a": {"passed": True}, "b": {"passed": False}}) is False
    # An empty mapping is not a success; it is missing evidence.
    assert task_success({}) is False


def test_duplicate_names_are_detected_on_a_pinned_task():
    validation = validate_task_verifiers(TASK_WITH_DUPLICATES)
    assert validation.verifier_count == 11
    assert validation.duplicate_names
    assert validation.lost_to_collision > 0
    # The index-keyed view must retain every declared verifier.
    assert len(validation.reconciled_keys) == validation.verifier_count
    assert len(validation.original_keys) < validation.verifier_count


def test_collision_that_cannot_flip_verdict_is_reported_consistent():
    record = _record(
        task_id="t",
        verifier_results_collapsed={"a": {"passed": True}, "b": {"passed": True}},
        verifier_results_indexed={"0:a": {"passed": True}, "1:a": {"passed": True}},
    )
    validation = validate_task_verifiers(
        TASK_WITH_DUPLICATES, records=[record], task_id="t"
    )
    assert validation.observed is True
    assert validation.task_success_differs is False
    assert validation.is_ambiguous is False


def test_collision_that_flips_verdict_is_reported_ambiguous():
    # Collapsing hides the failing verifier, so the two views disagree.
    record = _record(
        task_id="t",
        verifier_results_collapsed={"a": {"passed": True}},
        verifier_results_indexed={"0:a": {"passed": True}, "1:a": {"passed": False}},
    )
    validation = validate_task_verifiers(
        TASK_WITH_DUPLICATES, records=[record], task_id="t"
    )
    assert validation.task_success_original is True
    assert validation.task_success_reconciled is False
    assert validation.task_success_differs is True
    assert validation.is_ambiguous is True


def test_verifier_report_gates_on_observed_divergence():
    record = _record(
        task_id="t",
        verifier_results_collapsed={"a": {"passed": True}},
        verifier_results_indexed={"0:a": {"passed": True}, "1:a": {"passed": False}},
    )
    report = build_verifier_validation_report(
        [TASK_WITH_DUPLICATES], records=[record], task_ids={TASK_WITH_DUPLICATES: "t"}
    )
    assert report["evaluator_unambiguous"] is False
    assert report["observed_divergent_task_ids"] == ["t"]


def test_verifier_report_is_unambiguous_without_observed_divergence():
    report = build_verifier_validation_report([TASK_WITH_DUPLICATES], records=[])
    assert report["evaluator_unambiguous"] is True
    # Duplicates are still surfaced as latent risk rather than hidden.
    assert report["tasks_with_duplicate_names"] == 1
    assert report["latent_risk_task_ids"]


def test_verifier_report_rejects_duplicate_task_ids():
    with pytest.raises(ValueError, match="Duplicate task id"):
        build_verifier_validation_report([TASK_WITH_DUPLICATES, TASK_WITH_DUPLICATES])


# ---------------------------------------------------------------------------
# Phase 4 - step-budget horizon
# ---------------------------------------------------------------------------


def test_horizon_flags_a_budget_that_censors_runs():
    records = [_record(steps_taken=5, max_steps=5) for _ in range(4)]
    verdict = analyse_horizon(records)
    assert verdict.budget_capped == 4
    assert verdict.censored_failures == 4
    assert verdict.censored_fraction == 1.0
    assert verdict.sufficient is False


def test_horizon_accepts_a_budget_with_headroom():
    records = [
        _record(steps_taken=4, max_steps=15, overall_success=True) for _ in range(4)
    ]
    verdict = analyse_horizon(records)
    assert verdict.censored_failures == 0
    assert verdict.sufficient is True
    assert verdict.censored_fraction <= CENSORED_FRACTION_THRESHOLD


def test_budget_capped_but_successful_is_not_censored():
    """A run that used the whole budget and still succeeded is not censored."""
    assert analyse_horizon(
        [_record(steps_taken=5, max_steps=5, overall_success=True)]
    ).censored_failures == 0


def test_horizon_with_no_records_is_not_sufficient():
    verdict = analyse_horizon([])
    assert verdict.runs == 0
    assert verdict.sufficient is False


def test_recommend_refuses_to_extrapolate_from_in_place_censoring():
    records = [_record(steps_taken=5, max_steps=5) for _ in range(3)]
    recommendation = recommend_max_steps(records, proposed=15)
    assert recommendation["recommended_max_steps"] is None
    assert recommendation["validated"] is False
    assert "cannot justify a larger horizon" in recommendation["reason"]


def test_recommend_keeps_the_budget_when_sufficient():
    records = [
        _record(steps_taken=6, max_steps=15, overall_success=True) for _ in range(3)
    ]
    recommendation = recommend_max_steps(records, proposed=15)
    assert recommendation["recommended_max_steps"] == 15
    assert recommendation["verdict"]["sufficient"] is True

def test_state_report_markdown_has_the_blueprint_columns():
    report = build_state_report([_record()], condition_name="B_state_model")
    header = render_state_report_markdown(report).splitlines()[0]
    for column in (
        "Task", "Anchor", "Route", "Tables", "Records", "Delivery", "Precision", "Recall"
    ):
        assert column in header
