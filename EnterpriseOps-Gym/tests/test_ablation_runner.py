"""Goal loop, record persistence, redaction, and telemetry extraction."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ablation.conditions import CONDITION_A, CONDITION_B
from ablation.goals import (
    STAGES,
    Goal,
    GoalEngine,
    GoalReporter,
    GoalState,
    build_goals,
)
from ablation.results import (
    REDACTED,
    TaskRunRecord,
    is_recorded,
    load_run_records,
    redact,
    run_filename,
    write_run_record,
)
from ablation.runner import extract_telemetry, is_read_tool


def make_record(**overrides) -> TaskRunRecord:
    """A minimal, valid record with sensible defaults."""
    base = TaskRunRecord(
        experiment="experiment_1",
        run_uid="uid-1",
        condition=CONDITION_A.name,
        task_id="task-1",
        run_index=1,
        complexity_category="TYPE_3_MULTI_HOP",
        seed_database_file="seed.sql",
        runtime_database_id="db_1",
        initial_state_fingerprint="fp-1",
        model_provider="openai",
        model_name="gpt-x",
        temperature=0.0,
        max_steps=50,
        tool_mode="oracle",
        orchestrator="react",
        overall_success=True,
        verifier_summary_collapsed={"total": 4, "passed": 4},
        verifier_summary_indexed={"total": 6, "passed": 6},
        verifier_results_collapsed={"a": {"passed": True}},
        verifier_results_indexed={"0:a": {"passed": True}},
        agent_error=False,
        error_message=None,
        steps_taken=4,
        tool_call_count=6,
        read_query_count=3,
        invalid_action_count=0,
        latency_ms=1234,
        input_tokens=100,
        output_tokens=20,
        state_retrieval=None,
        started_at="2026-09-26T00:00:00+00:00",
        finished_at="2026-09-26T00:00:01+00:00",
    )
    return replace(base, **overrides) if overrides else base


# ---------- goal loop ----------


def test_build_goals_is_a_deterministic_matrix():
    goals = build_goals([make_record()], [CONDITION_A, CONDITION_B], num_runs=3)
    assert len(goals) == 6
    assert [goal.condition for goal in goals[:3]] == [CONDITION_A.name] * 3
    assert [goal.run_index for goal in goals[:3]] == [1, 2, 3]
    with pytest.raises(ValueError):
        build_goals([make_record()], [CONDITION_A], num_runs=0)


@pytest.mark.asyncio
async def test_goal_engine_runs_and_reports_each_goal():
    seen: list = []

    async def executor(goal: Goal, reporter: GoalReporter) -> TaskRunRecord:
        seen.append(goal.key)
        reporter.measure("agent", 12.5)
        reporter.transition(GoalState.RECORDED, "ok")
        return make_record(task_id=goal.task_id, condition=goal.condition)

    engine = GoalEngine(executor, max_concurrency=2)
    goals = build_goals([make_record()], [CONDITION_A, CONDITION_B], num_runs=2)
    report = await engine.run(goals)

    assert len(seen) == 4
    assert len(report.completed) == 4
    assert not report.failed
    assert report.stage_summary()["agent"]["mean_ms"] == pytest.approx(12.5)
    assert report.as_dict()["concurrency"] == 2


@pytest.mark.asyncio
async def test_goal_engine_isolates_a_failing_goal():
    async def executor(goal: Goal, reporter: GoalReporter) -> TaskRunRecord:
        if goal.condition == CONDITION_B.name:
            raise RuntimeError("state model exploded")
        return make_record(task_id=goal.task_id, condition=goal.condition)

    report = await GoalEngine(executor, max_concurrency=1).run(
        build_goals([make_record()], [CONDITION_A, CONDITION_B], num_runs=1)
    )
    assert len(report.completed) == 1
    assert len(report.failed) == 1
    assert "state model exploded" in report.failed[0].error


@pytest.mark.asyncio
async def test_goal_engine_skips_recorded_goals(tmp_path):
    calls: list = []

    async def executor(goal: Goal, reporter: GoalReporter) -> TaskRunRecord:
        calls.append(goal.key)
        record = make_record(task_id=goal.task_id, condition=goal.condition, run_index=goal.run_index)
        write_run_record(record, tmp_path)
        return record

    goals = build_goals([make_record()], [CONDITION_A], num_runs=2)
    first = await GoalEngine(executor, output_dir=str(tmp_path)).run(goals)
    assert len(first.completed) == 2

    second = await GoalEngine(executor, output_dir=str(tmp_path)).run(goals)
    assert len(second.skipped) == 2
    assert not second.completed
    assert len(calls) == 2, "recorded goals must not be re-executed"


def test_goal_reporter_rejects_unknown_state_and_stage():
    reporter = GoalReporter(Goal("t", "A", 1))
    with pytest.raises(ValueError):
        reporter.transition("TELEPORTING")
    with pytest.raises(ValueError):
        reporter.measure("teleportation", 1.0)
    for stage in STAGES:
        reporter.measure(stage, 1.0)
    assert set(reporter.timings) == set(STAGES)


def test_goal_engine_rejects_bad_concurrency():
    async def executor(goal, reporter):  # pragma: no cover - never called
        raise AssertionError

    with pytest.raises(ValueError):
        GoalEngine(executor, max_concurrency=0)


# ---------- records, persistence, redaction ----------


def test_run_record_schema_round_trip():
    record = make_record()
    restored = TaskRunRecord.from_dict(record.as_dict())
    assert restored.as_dict() == record.as_dict()
    assert restored.verifier_pass_rate == pytest.approx(1.0)


def test_run_record_requires_core_fields():
    payload = make_record().as_dict()
    del payload["condition"]
    with pytest.raises(ValueError, match="missing"):
        TaskRunRecord.from_dict(payload)


def test_records_persist_and_resume(tmp_path):
    record = make_record()
    target = write_run_record(record, tmp_path)
    assert target.name == run_filename(record.task_id, record.condition, record.run_index)
    assert target.is_file()
    assert is_recorded(tmp_path, record.task_id, record.condition, record.run_index)

    index = (tmp_path / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(index) == 1
    assert json.loads(index[0])["task_id"] == "task-1"

    loaded = load_run_records(tmp_path)
    assert len(loaded) == 1 and loaded[0].task_id == "task-1"


def test_credentials_are_redacted():
    payload = redact(
        {
            "llm_api_key": "sk-super-secret",
            "nested": {"api_key": "another", "token": "tok-123", "model": "gpt-x"},
            "list": [{"token": "t"}],
        }
    )
    assert payload["llm_api_key"] == REDACTED
    assert payload["nested"]["api_key"] == REDACTED
    assert payload["nested"]["token"] == REDACTED
    assert payload["nested"]["model"] == "gpt-x"
    assert payload["list"][0]["token"] == REDACTED


def test_persisted_record_never_contains_a_secret(tmp_path):
    write_run_record(make_record(), tmp_path)
    for path in tmp_path.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "sk-super-secret" not in text
        assert REDACTED in text or "api_key" not in text


# ---------- telemetry extraction ----------


def test_read_tool_classification():
    assert is_read_tool("find_user")
    assert is_read_tool("retrieve_installed_products")
    assert not is_read_tool("update_case")
    assert not is_read_tool("create_new_case")
    assert not is_read_tool("link_new_case_sla")


def test_extract_telemetry_counts_and_nulls_absent_usage():
    run = {
        "conversation_flow": [
            {"type": "system_message", "content": "s"},
            {"type": "user_message", "content": "u"},
            {"type": "ai_message", "content": "a"},
            {
                "type": "ai_message",
                "content": "b",
                "usage_metadata": {"input_tokens": 5, "output_tokens": 2},
            },
        ],
        "tool_results": [
            {"tool_name": "find_user", "result": {"success": True}},
            {"tool_name": "update_case", "result": {"success": True}},
            {"tool_name": "create_new_case", "result": {"success": False}},
        ],
    }
    telemetry = extract_telemetry(run)
    assert telemetry["steps_taken"] == 2
    assert telemetry["tool_call_count"] == 3
    assert telemetry["read_query_count"] == 1
    assert telemetry["invalid_action_count"] == 1
    assert telemetry["input_tokens"] == 5
    assert telemetry["output_tokens"] == 2

    no_usage = {"conversation_flow": [{"type": "ai_message", "content": "a"}], "tool_results": []}
    assert extract_telemetry(no_usage)["input_tokens"] is None
