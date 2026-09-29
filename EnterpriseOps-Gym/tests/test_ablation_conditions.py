"""Conditions, evaluation-set construction, and manifest integrity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ablation.conditions import (
    CONDITION_A,
    CONDITION_B,
    CONDITION_B1,
    CONDITION_B2,
    CONDITIONS,
    condition_for,
    condition_from_name,
    is_state_model_condition,
    parse_conditions,
)
from ablation.task_registry import (
    COMPLEXITY_CATEGORIES,
    build_eval_set,
    build_requirements_manifest,
    load_eval_set,
    load_task_config,
    save_eval_set,
)
from csm_env import SchemaRegistry

GYM_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = GYM_ROOT / "data" / "revised" / "csm"


@pytest.fixture(scope="module")
def registry() -> SchemaRegistry:
    return SchemaRegistry.from_static()


@pytest.fixture(scope="module")
def eval_set(registry: SchemaRegistry):
    if not TASKS_DIR.is_dir():
        pytest.skip("local CSM task directory is not available")
    return build_eval_set(TASKS_DIR, registry=registry)


def test_conditions_expose_a_broad_and_minimal():
    """The State Model 2.0 split: one baseline plus two state-model arms."""
    assert [condition.name for condition in CONDITIONS] == [
        "A_baseline",
        "B1_state_model_broad",
        "B2_state_model_minimal",
    ]
    assert CONDITION_A.state_model is False
    assert CONDITION_B1.state_model is True and CONDITION_B1.minimal_state is False
    assert CONDITION_B2.state_model is True and CONDITION_B2.minimal_state is True
    assert condition_from_name("A_baseline") is CONDITION_A
    assert condition_from_name("B1_state_model_broad") is CONDITION_B1
    assert condition_from_name("B2_state_model_minimal") is CONDITION_B2


def test_historical_b_alias_keeps_broad_semantics():
    """Archived ``B_state_model`` records must keep their original meaning.

    ``CONDITION_B`` and the ``B_state_model`` name resolve to the broad arm,
    so already-published results are never silently re-interpreted as the
    minimal arm.
    """
    assert CONDITION_B is CONDITION_B1
    assert condition_from_name("B_state_model") is CONDITION_B1
    assert condition_from_name("B") is CONDITION_B1
    assert condition_from_name("B_state_model").minimal_state is False


def test_condition_for_tolerates_archived_names():
    assert condition_for("B_state_model") is CONDITION_B1
    archived = condition_for("B_renamed_long_ago")
    assert archived.state_model is False, "unknown names never satisfy a delivery gate"


def test_is_state_model_condition_gates_both_b_arms():
    assert is_state_model_condition(CONDITION_B1) is True
    assert is_state_model_condition(CONDITION_B2) is True
    assert is_state_model_condition(CONDITION_A) is False


def test_transition_condition_is_refused():
    with pytest.raises(KeyError, match="Experiment 2"):
        condition_from_name("C_transition")
    with pytest.raises(KeyError):
        condition_from_name("nonsense")


def test_parse_conditions_accepts_letters_and_names():
    assert [c.name for c in parse_conditions("A,B")] == [
        "A_baseline",
        "B1_state_model_broad",
    ]
    assert [c.name for c in parse_conditions("A,B1,B2")] == [
        "A_baseline",
        "B1_state_model_broad",
        "B2_state_model_minimal",
    ]
    assert [c.name for c in parse_conditions("B_state_model")] == ["B1_state_model_broad"]
    assert [c.name for c in parse_conditions("B2")] == ["B2_state_model_minimal"]
    with pytest.raises(ValueError):
        parse_conditions("")
    with pytest.raises(ValueError):
        parse_conditions("A,A")
    with pytest.raises(ValueError):
        parse_conditions("A,B1,B1")


def test_eval_set_manifest_loads_and_validates(eval_set, registry):
    assert len(eval_set) > 0
    for record in eval_set:
        assert record.complexity_category in COMPLEXITY_CATEGORIES
        assert record.ggqr_task_type
        for table in record.reference_entities:
            assert registry.require_table(table) is not None
        assert record.seed_database_file
        assert record.category_rationale


def test_eval_set_categories_are_counted(eval_set):
    counts = eval_set.category_counts()
    assert set(counts) == set(COMPLEXITY_CATEGORIES)
    assert sum(counts.values()) == len(eval_set)


def test_eval_set_round_trips_through_disk(eval_set, tmp_path):
    target = save_eval_set(eval_set, tmp_path / "csm_eval_set.json")
    reloaded = load_eval_set(target)
    assert len(reloaded) == len(eval_set)
    assert reloaded.get(eval_set.tasks[0].task_id).complexity_category == (
        eval_set.tasks[0].complexity_category
    )


def test_duplicate_task_ids_are_rejected(tmp_path):
    payload = {"tasks": [], "eval_set_id": "x"}
    target = tmp_path / "bad.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="no tasks"):
        load_eval_set(target)


def test_unknown_complexity_category_is_rejected(eval_set, tmp_path):
    payload = eval_set.as_dict()
    payload["tasks"][0]["complexity_category"] = "TYPE_9_IMPOSSIBLE"
    target = tmp_path / "bad_category.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown complexity category"):
        load_eval_set(target)


def test_unsupported_tool_mode_is_refused():
    config = load_task_config(sorted(TASKS_DIR.glob("task_*.json"))[0])
    from ablation.task_registry import apply_tool_mode

    assert apply_tool_mode(config, "oracle") is config
    with pytest.raises(ValueError, match="Unsupported tool mode"):
        apply_tool_mode(config, "+10_tools")


def test_requirements_manifest_matches_real_registry(eval_set, registry):
    manifest = build_requirements_manifest(eval_set, registry=registry)
    assert set(manifest["task_assignment"]) == {record.task_id for record in eval_set}
    assert set(manifest["tasks"]) == {
        "account_profile",
        "case_overview",
        "check_entitlement",
        "find_knowledge",
        "resolve_case",
    }
    for task_type, tables in manifest["tasks"].items():
        for table in tables:
            assert registry.require_table(table) is not None, f"{task_type}:{table}"


def test_requirements_manifest_is_accepted_by_csm_env(eval_set, registry, tmp_path):
    """The generated manifest must be consumable by the real loader."""
    from ablation.task_registry import save_json
    from csm_env.query.requirements import RequirementRegistry

    manifest = build_requirements_manifest(eval_set, registry=registry)
    target = save_json(manifest, tmp_path / "csm_requirements.json")
    loaded = RequirementRegistry.from_json(registry, target)
    for record in eval_set:
        required = loaded.requirements_for(record.ggqr_task_type).required_tables
        assert required
