from pathlib import Path

from ablation.initial_relevance import build_initial_relevance_map
from ablation.task_registry import load_eval_set
from ablation.relevance import RelevanceSet
from ablation.experiment_manifest import VERSION_STATE_MODEL_REFINED, VERSION_LABELS

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "ablation" / "manifests" / "csm_eval_set.json"
ARCHIVE = ROOT / "gym_dbs.zip"


def test_initial_relevance_is_seed_prompt_scoped_and_not_verifier_scoped():
    manifest = load_eval_set(EVAL)
    result = build_initial_relevance_map(manifest, archive=ARCHIVE)
    assert result
    for relevance in result.values():
        assert relevance.scope == "initial_prompt_entities"
        assert relevance.source == "task_prompt+seed_sql"


def test_relevance_set_keeps_final_state_default_compatible():
    relevance = RelevanceSet("task", ("customer_case",), (), "table_only", (), 1)
    assert relevance.scope == "final_state_verifier"
    assert relevance.source == "verifier_sql"


def test_refined_version_is_distinct_and_supported():
    assert VERSION_STATE_MODEL_REFINED in VERSION_LABELS


def test_initial_relevance_accepts_workspace_relative_archive(tmp_path):
    from pathlib import Path
    from ablation.initial_relevance import build_initial_relevance_map
    from ablation.task_registry import load_eval_set

    eval_set = load_eval_set(Path("ablation/manifests/csm_eval_set.json"))
    result = build_initial_relevance_map(eval_set, archive=Path("gym_dbs.zip"))
    assert len(result) == len(eval_set)
    assert all(item.source == "task_prompt+seed_sql" for item in result.values())
    assert all(item.scope == "initial_prompt_entities" for item in result.values())


def test_initial_relevance_searches_all_csm_identity_tables():
    from pathlib import Path
    from ablation.initial_relevance import build_initial_relevance_map
    from ablation.task_registry import load_eval_set

    eval_set = load_eval_set(Path("ablation/manifests/csm_eval_set.json"))
    task = next(t for t in eval_set if t.task_id.endswith("910b176a"))
    result = build_initial_relevance_map(eval_set, archive=Path("gym_dbs.zip"))[task.task_id]
    # The prompt names Perry Watts, while the task registry only matches the
    # destination customer_case table. The evaluator must still find the User.
    assert any(table == "user" for table, _ in result.rows)
