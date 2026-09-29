"""Tests for the Phase 1-4 research freeze: manifest, gold state, horizon."""

from __future__ import annotations

import json

import pytest

from ablation.experiment_manifest import (
    VERSION_LABELS,
    VERSION_PREFIX_PILOT,
    VERSION_STATE_MODEL_FIXED,
    ExperimentManifest,
    assert_single_version,
    build_manifest,
    write_manifest,
)
from ablation.gold_state import build_gold_state, build_gold_state_map, compare_to_gold
from ablation.horizon import (
    CANDIDATE_HORIZONS,
    censor_fraction_at,
    select_common_horizon,
)
from ablation.results import TaskRunRecord
from ablation.task_registry import load_eval_set

#: Real pinned tasks used as fixtures.
TASK_DUPLICATE_NAMES = "data/revised/csm/task_20251205_153330_906_a8eea1c0_8c7a6205.json"
TASK_ALIASED = "data/revised/csm/task_20260101_122222_300_ad5a67e3_3757206c.json"
EVAL_SET = "ablation/manifests/csm_eval_set.json"


def _record(**overrides) -> TaskRunRecord:
    payload = {
        "experiment": "experiment_1", "run_uid": "u", "condition": "B_state_model",
        "task_id": "t", "run_index": 1, "complexity_category": "TYPE_1_SINGLE_ENTITY",
        "seed_database_file": "s", "runtime_database_id": "d",
        "initial_state_fingerprint": "f", "model_provider": "openrouter",
        "model_name": "space-bunny-alpha", "temperature": 0.0, "max_steps": 15,
        "tool_mode": "oracle", "orchestrator": "react", "overall_success": False,
        "verifier_summary_collapsed": {}, "verifier_summary_indexed": {},
        "verifier_results_collapsed": {}, "verifier_results_indexed": {},
        "agent_error": False, "error_message": None, "steps_taken": 5,
        "tool_call_count": 0, "read_query_count": 0, "invalid_action_count": 0,
        "latency_ms": 1, "input_tokens": None, "output_tokens": None,
        "state_retrieval": None, "started_at": "a", "finished_at": "b",
    }
    payload.update(overrides)
    return TaskRunRecord.from_dict(payload)


# ---------------------------------------------------------------------------
# Phase 1 - versioned manifest
# ---------------------------------------------------------------------------


def _manifest(**overrides) -> ExperimentManifest:
    common = dict(
        experiment_id="experiment_1", version=VERSION_STATE_MODEL_FIXED,
        model_provider="openrouter", model="space-bunny-alpha", temperature=0.0,
        orchestrator="react", tool_mode="oracle", max_steps=15, seed=0,
        csm_env_schema_version="1.0.0", csm_env_sha256="a", benchmark_sha256="b",
        ablation_sha256="c", task_set_sha256="d", seed_archive_sha256="e",
        interpreter="CPython 3.11.8", platform="win32", task_ids=("b", "a"),
    )
    common.update(overrides)
    return ExperimentManifest(**common)


def test_version_labels_distinguish_pre_fix_from_repaired():
    assert VERSION_PREFIX_PILOT in VERSION_LABELS
    assert VERSION_STATE_MODEL_FIXED in VERSION_LABELS
    assert VERSION_PREFIX_PILOT != VERSION_STATE_MODEL_FIXED


def test_manifest_hash_is_stable_and_content_addressed():
    first = _manifest()
    # Task-id ordering must not change the hash.
    assert _manifest(task_ids=("a", "b")).manifest_hash == first.manifest_hash
    # A real configuration change must change it.
    assert _manifest(max_steps=20).manifest_hash != first.manifest_hash


def test_manifest_rejects_unknown_version():
    with pytest.raises(ValueError, match="Unknown experiment version"):
        _manifest(version="V9-INVENTED")


def test_manifest_rejects_non_positive_budget():
    with pytest.raises(ValueError, match="max_steps"):
        _manifest(max_steps=0)


def test_build_manifest_hashes_the_real_workspace():
    manifest = build_manifest(
        experiment_id="experiment_1", version=VERSION_STATE_MODEL_FIXED,
        model_provider="openrouter", model="space-bunny-alpha", temperature=0.0,
        orchestrator="react", tool_mode="oracle", max_steps=15, seed=0,
        eval_set_path=EVAL_SET, seed_archive="gym_dbs.zip",
    )
    for digest in (manifest.csm_env_sha256, manifest.benchmark_sha256,
                   manifest.task_set_sha256, manifest.seed_archive_sha256):
        assert len(digest) == 64
    assert manifest.num_tasks == 11
    assert manifest.manifest_hash


def test_manifest_round_trips_through_disk(tmp_path):
    manifest = build_manifest(
        experiment_id="experiment_1", version=VERSION_STATE_MODEL_FIXED,
        model_provider="openrouter", model="space-bunny-alpha", temperature=0.0,
        orchestrator="react", tool_mode="oracle", max_steps=15, seed=0,
        eval_set_path=EVAL_SET,
    )
    path = write_manifest(manifest, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["manifest_hash"] == manifest.manifest_hash
    assert payload["version"] == VERSION_STATE_MODEL_FIXED


def test_mixing_experimental_versions_is_rejected():
    """Pooling pre-fix and post-fix records would compare two interventions."""
    records = [
        _record(experiment_version=VERSION_PREFIX_PILOT),
        _record(experiment_version=VERSION_STATE_MODEL_FIXED),
    ]
    with pytest.raises(ValueError, match="mix experimental versions"):
        assert_single_version(records)


def test_single_version_records_pass():
    assert_single_version(
        [_record(experiment_version=VERSION_STATE_MODEL_FIXED) for _ in range(3)]
    )


# ---------------------------------------------------------------------------
# Fairness boundary - the gold state is evaluator-only (SEC-001)
# ---------------------------------------------------------------------------


def test_gold_state_is_unreachable_from_the_intervention():
    """The Condition-B import chain must not touch verifier-derived gold data.

    ``gold_state`` is built from verifier SQL. If the intervention could reach
    it, Condition B would be reading the answer key, and the causal comparison
    would be meaningless. The check is on executable code, matching the existing
    ``relevance`` boundary test.
    """
    import ast
    from pathlib import Path

    gym_root = Path(__file__).resolve().parents[1]
    forbidden = ("gold_state", "relevance")
    for module in ("state_model.py", "state_model_orchestrator.py", "runner.py"):
        source = (gym_root / "ablation" / module).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        assert not any(
            any(name in module_name for name in forbidden) for module_name in imported
        ), f"{module} imports evaluator-only module: {imported}"


def test_gold_state_module_never_reads_task_verifiers_at_runtime():
    """`gold_state` may read verifier SQL; nothing else in it may read the DB.

    It is a pure derivation over task JSON, which is what keeps it replayable
    and free of any live-server dependency.
    """
    from pathlib import Path

    gym_root = Path(__file__).resolve().parents[1]
    source = (gym_root / "ablation" / "gold_state.py").read_text(encoding="utf-8")
    for forbidden in ("httpx", "EnterpriseOpsSQLRunner", "create_database_from_file"):
        assert forbidden not in source, f"gold_state must not use {forbidden}"


# ---------------------------------------------------------------------------
# Phase 2 - evaluator-only gold state
# ---------------------------------------------------------------------------


def test_gold_state_extracts_facts_from_verifier_sql():
    gold = build_gold_state(TASK_DUPLICATE_NAMES)
    assert gold.facts
    assert gold.verifier_count == 11
    assert any(f.table == "customer_case" for f in gold.facts)


def test_aliases_are_resolved_so_no_gold_set_is_empty():
    """Regression: aliased predicates were dropped, silently zeroing gold facts.

    Two of the eleven tasks have fully aliased verifier SQL; without alias
    resolution their gold set was empty and their recall undefined.
    """
    gold = build_gold_state(TASK_ALIASED)
    assert gold.facts, "aliased task must still produce gold facts"
    assert any(f.table == "account" for f in gold.facts)


def test_every_pinned_task_has_a_non_empty_gold_set():
    manifest = load_eval_set(EVAL_SET)
    states = build_gold_state_map([r.task_config_path for r in manifest])
    assert len(states) == len(manifest)
    empty = [tid for tid, gold in states.items() if not gold.facts]
    assert not empty, f"tasks with empty gold sets: {empty}"


def test_gold_facts_exclude_aggregate_columns():
    """COUNT(*) is scaffolding, not a task fact."""
    gold = build_gold_state(TASK_DUPLICATE_NAMES)
    assert not any(f.column in {"count", "row_count"} for f in gold.facts)


class _Fact:
    def __init__(self, table, column, value):
        self.value = value
        self.provenance = type(
            "P", (), {"table": table, "column": column, "row_pk": "1"}
        )()


class _State:
    def __init__(self, facts):
        self.facts = facts


def test_fact_computation_is_exact():
    gold = build_gold_state(TASK_DUPLICATE_NAMES)
    target = gold.facts[0]
    state = _State([_Fact(target.table, target.column, target.value),
                    _Fact("account", "name", "unrelated")])
    result = compare_to_gold(gold, state)
    assert result.matched_fact_count == 1
    assert result.retrieved_fact_count == 2
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(1 / len(gold.facts))
    assert len(result.unresolved_facts) == len(gold.facts) - 1


def test_fact_comparison_with_no_state_is_all_recall_zero():
    result = compare_to_gold(build_gold_state(TASK_DUPLICATE_NAMES), None)
    assert result.retrieved_fact_count == 0
    assert result.precision is None
    assert result.recall == 0.0
    assert result.matched_fact_count == 0


# ---------------------------------------------------------------------------
# Phase 4 - empirical horizon protocol
# ---------------------------------------------------------------------------


def test_candidate_horizons_match_the_blueprint_ladder():
    assert CANDIDATE_HORIZONS == (10, 15, 20, 25, 30)


def test_censor_fraction_is_monotone_in_the_budget():
    """Raising the ceiling can only reduce censoring - never increase it.

    This monotonicity is what makes ladder selection valid: the first candidate
    that clears the threshold is the smallest defensible budget.
    """
    records = [
        _record(steps_taken=4, max_steps=5, overall_success=False),
        _record(steps_taken=5, max_steps=5, overall_success=False),
    ]
    # 4 censors both runs, 5 censors one, 10 censors neither.
    assert censor_fraction_at(records, 4) == pytest.approx(1.0)
    assert censor_fraction_at(records, 5) == pytest.approx(0.5)
    assert censor_fraction_at(records, 10) == pytest.approx(0.0)
    fractions = [censor_fraction_at(records, n) for n in (4, 5, 10)]
    assert fractions[0] >= fractions[1] >= fractions[2]


def test_successful_run_at_ceiling_is_not_censored():
    records = [_record(steps_taken=5, max_steps=5, overall_success=True)]
    assert censor_fraction_at(records, 5) == 0.0


def test_horizon_selection_prefers_smallest_acceptable_budget():
    records = [
        _record(steps_taken=5, max_steps=5, overall_success=False),
        _record(steps_taken=2, max_steps=5, overall_success=True),
        _record(steps_taken=1, max_steps=5, overall_success=True),
    ]
    selection = select_common_horizon(records)
    assert selection["selected_max_steps"] == 10
    assert selection["reason"]
    row = next(r for r in selection["candidates"] if r["max_steps"] == 10)
    assert row["acceptable"] is True


def test_horizon_selection_reports_failure_when_nothing_acceptable():
    """High censoring at every rung, but the runs did stop *below* their ceiling.

    ``steps_taken=30`` with ``max_steps=50`` means the agent genuinely used 30
    steps and still failed, so every candidate rung censors it. This is the
    honest way to exhaust the ladder, as distinct from the in-place truncation
    the guard rejects above.
    """
    records = [_record(steps_taken=30, max_steps=50, overall_success=False)
               for _ in range(3)]
    selection = select_common_horizon(records)
    assert selection["selected_max_steps"] == max(CANDIDATE_HORIZONS)
    assert "re-measured" in selection["reason"]
    assert all(row["acceptable"] is False for row in selection["candidates"])


def test_horizon_selection_requires_candidates():
    with pytest.raises(ValueError):
        select_common_horizon([], candidates=())


def test_horizon_selection_refuses_in_place_censored_records():
    """Regression: re-scoring truncated runs reported a vacuous 0% censoring.

    Every run in the pre-fix pilot is exactly ``(steps=5, max_steps=5)`` - it was
    physically terminated at the ceiling, so nothing is known about what a larger
    budget would have allowed. Selecting 10 from that data would manufacture
    confidence; the ladder must refuse instead.
    """
    records = [_record(steps_taken=5, max_steps=5, overall_success=False)
               for _ in range(18)]
    selection = select_common_horizon(records)
    assert selection["selected_max_steps"] is None
    assert selection["candidates"] == []
    assert "cannot" in selection["reason"]


def test_horizon_selection_works_once_budget_headroom_is_observed():
    """The same data, but observed under a generous ceiling, is usable."""
    records = [
        _record(steps_taken=5, max_steps=30, overall_success=False),
        _record(steps_taken=12, max_steps=30, overall_success=True),
        _record(steps_taken=2, max_steps=30, overall_success=True),
    ]
    selection = select_common_horizon(records)
    assert selection["selected_max_steps"] == 10
    assert selection["candidates"]
