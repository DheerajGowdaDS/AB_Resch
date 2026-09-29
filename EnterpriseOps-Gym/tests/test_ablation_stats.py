"""Metrics, paired statistics, and reconciliation with the official scorer."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from ablation.conditions import CONDITION_A, CONDITION_B
from ablation.relevance import CONFIDENCE_FULL, CONFIDENCE_TABLE_ONLY, RelevanceSet
from ablation.results import TaskRunRecord
from ablation.score_bridge import emit_compute_score_layout
from ablation.stats import (
    agent_error_rate,
    breakdown_by_complexity,
    breakdown_by_condition,
    mcnemar_exact_pvalue,
    mcnemar_test,
    pair_records,
    paired_bootstrap_ci,
    retrieval_quality,
    tsr,
    vpr,
    verifier_level_pass_rate_pooled,
)
from ablation.task_registry import COMPLEXITY_CATEGORIES


def make_record(**overrides) -> TaskRunRecord:
    base = TaskRunRecord(
        experiment="experiment_1",
        run_uid="uid",
        condition=CONDITION_A.name,
        task_id="task-1",
        run_index=1,
        complexity_category="TYPE_3_MULTI_HOP",
        seed_database_file="seed.sql",
        runtime_database_id="db",
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


def test_tsr_and_vpr_match_compute_score_semantics():
    records = [
        make_record(overall_success=True, verifier_summary_collapsed={"total": 4, "passed": 4}),
        make_record(overall_success=False, verifier_summary_collapsed={"total": 4, "passed": 2}),
        make_record(overall_success=False, verifier_summary_collapsed={"total": 4, "passed": 0}),
    ]
    assert tsr(records) == pytest.approx(1 / 3)
    assert vpr(records) == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert verifier_level_pass_rate_pooled(records) == pytest.approx(6 / 12)


def test_agent_errors_are_excluded_from_success_metrics():
    records = [
        make_record(overall_success=True),
        make_record(overall_success=False, agent_error=True, error_message="crash"),
    ]
    assert tsr(records) == pytest.approx(1.0)
    assert agent_error_rate(records) == pytest.approx(0.5)


def test_pair_records_builds_the_2x2_table():
    """Pairs are keyed by (task_id, run_index), so each case gets a distinct id."""
    records = [
        make_record(task_id="t1", condition=CONDITION_A.name, overall_success=True),
        make_record(task_id="t1", condition=CONDITION_B.name, overall_success=True),
        make_record(task_id="t2", condition=CONDITION_A.name, overall_success=True),
        make_record(task_id="t2", condition=CONDITION_B.name, overall_success=False),
        make_record(task_id="t3", condition=CONDITION_A.name, overall_success=False),
        make_record(task_id="t3", condition=CONDITION_B.name, overall_success=True),
        make_record(task_id="t4", condition=CONDITION_A.name, overall_success=False),
        make_record(task_id="t4", condition=CONDITION_B.name, overall_success=False),
    ]
    outcomes = pair_records(records)
    assert outcomes.both_pass == 1
    assert outcomes.baseline_only == 1
    assert outcomes.treatment_only == 1
    assert outcomes.both_fail == 1
    assert outcomes.total == 4
    assert outcomes.discordant == 2


def test_pair_records_treats_repeats_as_independent_pairs():
    records = [
        make_record(task_id="t1", run_index=1, condition=CONDITION_A.name, overall_success=False),
        make_record(task_id="t1", run_index=1, condition=CONDITION_B.name, overall_success=True),
        make_record(task_id="t1", run_index=2, condition=CONDITION_A.name, overall_success=False),
        make_record(task_id="t1", run_index=2, condition=CONDITION_B.name, overall_success=True),
    ]
    outcomes = pair_records(records)
    assert outcomes.total == 2
    assert outcomes.treatment_only == 2


def test_pair_records_excludes_errored_pairs():
    records = [
        make_record(condition=CONDITION_A.name, overall_success=True),
        make_record(condition=CONDITION_B.name, overall_success=False, agent_error=True),
    ]
    outcomes = pair_records(records)
    assert outcomes.total == 0
    assert outcomes.excluded == 1


@pytest.mark.parametrize(
    "baseline_only,treatment_only,expected",
    [
        (0, 5, 0.0625),  # 2 * C(5,0) / 2^5
        (5, 0, 0.0625),
        (3, 3, 1.0),  # 2 * 42/64, capped at 1
        (10, 4, 2 * 1471 / 16384),  # sum_{k<=4} C(14,k) = 1471
    ],
)
def test_mcnemar_exact_matches_hand_computed_values(baseline_only, treatment_only, expected):
    assert mcnemar_exact_pvalue(baseline_only, treatment_only) == pytest.approx(
        min(1.0, expected), abs=1e-12
    )


def test_mcnemar_exact_caps_at_one():
    expected = 2 * sum(math.comb(24, k) for k in range(13)) / (2**24)
    assert expected > 1.0, "this case is chosen because the two-sided tail exceeds 1"
    assert mcnemar_exact_pvalue(12, 12) == 1.0


def test_mcnemar_exact_is_symmetric():
    assert mcnemar_exact_pvalue(2, 9) == mcnemar_exact_pvalue(9, 2)
    assert mcnemar_exact_pvalue(0, 0) == 1.0


def test_mcnemar_switches_to_chi_square_above_the_exact_threshold():
    assert mcnemar_test(pair_records([]))["method"] == "none"

    records = []
    for index in range(40):
        records.append(
            make_record(task_id=f"t{index}", condition=CONDITION_A.name, overall_success=True)
        )
        records.append(
            make_record(task_id=f"t{index}", condition=CONDITION_B.name, overall_success=False)
        )
    result = mcnemar_test(pair_records(records))
    assert result["method"] == "chi2_continuity_corrected"
    assert result["discordant"] == 40
    assert 0.0 <= result["p_value"] <= 1.0


def test_paired_bootstrap_is_reproducible_and_centred():
    records = []
    for index in range(10):
        records.append(
            make_record(task_id=f"t{index}", condition=CONDITION_A.name, overall_success=index < 3)
        )
        records.append(
            make_record(task_id=f"t{index}", condition=CONDITION_B.name, overall_success=index < 7)
        )
    outcomes = pair_records(records)
    first = paired_bootstrap_ci(outcomes, n_resamples=500, seed=13)
    second = paired_bootstrap_ci(outcomes, n_resamples=500, seed=13)
    assert first == second
    assert first["delta"] == pytest.approx(0.4)
    assert first["ci_low"] <= first["delta"] <= first["ci_high"]
    assert first["n"] == 10
    assert first["seed"] == 13


def test_paired_bootstrap_validates_arguments():
    outcomes = pair_records([make_record(), make_record(condition=CONDITION_B.name)])
    with pytest.raises(ValueError):
        paired_bootstrap_ci(outcomes, n_resamples=0)
    with pytest.raises(ValueError):
        paired_bootstrap_ci(outcomes, alpha=1.5)


def test_breakdown_reports_na_for_empty_cells():
    records = [make_record(condition=CONDITION_A.name, overall_success=True)]
    table = breakdown_by_complexity(
        records, conditions=(CONDITION_A, CONDITION_B), categories=COMPLEXITY_CATEGORIES
    )
    assert table["TYPE_3_MULTI_HOP"][CONDITION_A.name]["n"] == 1
    assert table["TYPE_3_MULTI_HOP"][CONDITION_B.name]["n"] == 0
    assert table["TYPE_3_MULTI_HOP"][CONDITION_B.name]["tsr"] is None
    assert table["TYPE_2_ONE_HOP"][CONDITION_A.name]["tsr"] is None


def test_breakdown_by_condition_reports_headline_metrics():
    records = [
        make_record(condition=CONDITION_A.name, overall_success=True),
        make_record(condition=CONDITION_B.name, overall_success=False),
    ]
    summary = breakdown_by_condition(records, (CONDITION_A, CONDITION_B))
    assert summary[CONDITION_A.name]["tsr"] == pytest.approx(1.0)
    assert summary[CONDITION_B.name]["tsr"] == pytest.approx(0.0)
    assert summary[CONDITION_B.name]["state_model"] is True


def test_retrieval_quality_uses_rows_then_falls_back_to_tables():
    record = make_record(
        condition=CONDITION_B.name,
        state_retrieval={
            "state_model_available": True,
            "retrieved_records": 2,
            "retrieved_tokens": 20,
            "context_tokens_injected": 20,
            "anchor_status": "RESOLVED",
            "state_route_status": "COMPLETE",
            "retrieved_rows": [["customer_case", "1233"], ["account", "7"]],
            # `retrieved_tables` is a COUNT in the telemetry payload; the names
            # live in `retrieved_table_names`. Mixing these up was a real bug.
            "retrieved_tables": 2,
            "retrieved_table_names": ["customer_case", "account"],
        },
    )
    row_level = RelevanceSet(
        "task-1", ("customer_case",), (("customer_case", "1233"),), CONFIDENCE_FULL, (), 2
    )
    quality = retrieval_quality(record, row_level)
    assert quality.granularity == "row"
    assert quality.precision == pytest.approx(0.5)
    assert quality.recall == pytest.approx(1.0)

    table_level = RelevanceSet(
        "task-1", ("customer_case", "entitlement"), (), CONFIDENCE_TABLE_ONLY, (), 2
    )
    fallback = retrieval_quality(record, table_level)
    assert fallback.granularity == "table"
    assert fallback.precision == pytest.approx(0.5)
    assert fallback.recall == pytest.approx(0.5)


def test_retrieval_quality_skips_condition_a():
    record = make_record(condition=CONDITION_A.name, state_retrieval=None)
    relevance = RelevanceSet("task-1", ("customer_case",), (), CONFIDENCE_TABLE_ONLY, (), 1)
    assert retrieval_quality(record, relevance) is None


def test_retrieval_quality_tolerates_a_count_in_retrieved_tables():
    """A count in `retrieved_tables` must not be iterated as a list of names."""
    record = make_record(
        condition=CONDITION_B.name,
        state_retrieval={
            "state_model_available": True,
            "retrieved_records": 1,
            "retrieved_tokens": 20,
            "context_tokens_injected": 20,
            "anchor_status": "RESOLVED",
            "state_route_status": "COMPLETE",
            "retrieved_rows": [["customer_case", "1233"]],
            "retrieved_tables": 3,
            "retrieved_table_names": ["customer_case"],
        },
    )
    relevance = RelevanceSet(
        "task-1", ("customer_case",), (("customer_case", "1233"),), CONFIDENCE_FULL, (), 1
    )
    quality = retrieval_quality(record, relevance)
    assert quality is not None
    assert quality.granularity == "row"
    assert quality.precision == pytest.approx(1.0)


def test_state_telemetry_exposes_counts_and_names_separately():
    from ablation.state_model_orchestrator import StateUsageTelemetry

    payload = StateUsageTelemetry(
        available=True,
        state_retrieval_count=1,
        retrieved_records=2,
        retrieved_tables=2,
        retrieved_token_count=10,
        context_tokens_injected=10,
        state_route_status="COMPLETE",
        anchor_status="RESOLVED",
        anchor_table="customer_case",
        anchor_row_id="888",
        retrieval_latency_ms=1.0,
        error=None,
        retrieved_table_names=("customer_case", "account"),
        retrieved_rows=(("customer_case", "888"), ("account", "7")),
    ).as_dict()
    assert payload["retrieved_tables"] == 2
    assert payload["retrieved_table_names"] == ["customer_case", "account"]
    assert payload["retrieved_rows"] == [["customer_case", "888"], ["account", "7"]]


def test_score_bridge_matches_the_official_scorer(tmp_path):
    import compute_score

    records = [
        make_record(
            task_id="t1", condition=CONDITION_A.name, overall_success=True,
            verifier_summary_collapsed={"total": 4, "passed": 4},
        ),
        make_record(
            task_id="t2", condition=CONDITION_A.name, overall_success=False,
            verifier_summary_collapsed={"total": 2, "passed": 1},
        ),
        make_record(
            task_id="t1", condition=CONDITION_B.name, overall_success=True,
            verifier_summary_collapsed={"total": 4, "passed": 4},
        ),
    ]
    written = emit_compute_score_layout(records, tmp_path)
    assert set(written) == {CONDITION_A.name, CONDITION_B.name}

    baseline_records = [r for r in records if r.condition == CONDITION_A.name]
    baseline = compute_score.process_mode(str(tmp_path / CONDITION_A.name), "A")
    assert baseline["avg_overall_success_rate"] == pytest.approx(tsr(baseline_records) * 100.0)
    assert baseline["avg_verifier_pass_rate"] == pytest.approx(vpr(baseline_records) * 100.0)
    assert baseline["files_with_errors"] == 0

    treatment = compute_score.process_mode(str(tmp_path / CONDITION_B.name), "B")
    assert treatment["avg_overall_success_rate"] == pytest.approx(100.0)


def test_score_bridge_counts_error_files(tmp_path):
    import compute_score

    records = [
        make_record(condition=CONDITION_A.name, overall_success=True),
        make_record(
            condition=CONDITION_A.name, overall_success=False, agent_error=True, error_message="boom"
        ),
    ]
    emit_compute_score_layout(records, tmp_path)
    baseline = compute_score.process_mode(str(tmp_path / CONDITION_A.name), "A")
    assert baseline["files_with_errors"] == 1


def test_report_is_regenerated_from_records_only(tmp_path):
    from ablation.relevance import CONFIDENCE_TABLE_ONLY, RelevanceSet
    from ablation.report import report_from_directory
    from ablation.results import write_run_record

    records = [
        make_record(
            task_id="t1", condition=CONDITION_A.name, overall_success=True,
            verifier_summary_collapsed={"total": 4, "passed": 4},
        ),
        make_record(
            task_id="t1", condition=CONDITION_B.name, overall_success=True,
            verifier_summary_collapsed={"total": 4, "passed": 4},
            state_retrieval={
                "state_model_available": True,
                "retrieved_rows": [["customer_case", "1233"]],
                "retrieved_tables": ["customer_case"],
                "retrieved_records": 1,
                "retrieved_tokens": 20,
                "context_tokens_injected": 20,
                "anchor_status": "RESOLVED",
                "state_route_status": "COMPLETE",
            },
        ),
        make_record(
            task_id="t2", condition=CONDITION_A.name, overall_success=False,
            verifier_summary_collapsed={"total": 3, "passed": 1},
        ),
        make_record(
            task_id="t2", condition=CONDITION_B.name, overall_success=True,
            verifier_summary_collapsed={"total": 3, "passed": 3},
        ),
    ]
    for record in records:
        write_run_record(record, tmp_path)

    relevance = {
        "t1": RelevanceSet("t1", ("customer_case",), (), CONFIDENCE_TABLE_ONLY, (), 4),
        "t2": RelevanceSet("t2", ("entitlement",), (), CONFIDENCE_TABLE_ONLY, (), 3),
    }
    paths = report_from_directory(tmp_path, tmp_path, relevance_map=relevance)
    assert paths["json"].is_file() and paths["markdown"].is_file()

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["headline"][CONDITION_A.name]["tsr"] == pytest.approx(0.5)
    assert payload["headline"][CONDITION_B.name]["tsr"] == pytest.approx(1.0)
    assert payload["headline"]["delta_tsr"] == pytest.approx(0.5)
    assert payload["paired"]["delta_tsr"] == pytest.approx(0.5)
    assert payload["paired"]["outcomes"]["treatment_only_pass"] == 1
    assert 0.0 <= payload["paired"]["mcnemar"]["p_value"] <= 1.0
    assert payload["retrieval_quality"][CONDITION_B.name]["runs_measured"] == 2
    assert payload["retrieval_quality"][CONDITION_B.name]["runs_excluded"] == 0

    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "## Headline" in markdown
    assert "## Paired result" in markdown
    assert "McNemar p-value" in markdown
    assert "Verifier reconciliation" in markdown
    assert CONDITION_A.name in markdown and CONDITION_B.name in markdown


def test_report_handles_an_empty_run_set(tmp_path):
    from ablation.report import build_report

    report = build_report([])
    assert report["paired"]["bootstrap"]["n"] == 0
    assert report["headline"]["delta_tsr"] is None


def test_pair_records_excludes_condition_b_without_delivered_state():
    records = [
        make_record(task_id="t1", condition=CONDITION_A.name, overall_success=False),
        make_record(
            task_id="t1",
            condition=CONDITION_B.name,
            overall_success=True,
            state_retrieval={
                "state_model_available": False,
                "retrieved_records": 0,
                "retrieved_tokens": 0,
                "context_tokens_injected": 0,
                "anchor_status": "UNRESOLVED",
                "state_route_status": "UNRESOLVED",
            },
        ),
    ]
    outcomes = pair_records(records)
    assert outcomes.total == 0
    assert outcomes.excluded == 1
