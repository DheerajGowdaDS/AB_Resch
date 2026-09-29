"""Fairness boundary, additivity, verifier-collision handling, and readiness."""

from __future__ import annotations

import ast
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ablation.parity import compare_csm_env_copies, require_parity, sha256_file
from ablation.task_registry import build_eval_set
from ablation.verify_pinning import validate_eval_set_verifiers, validate_verifier_set
from csm_env import SchemaRegistry
from tests.tests_support.ablation_doubles import FakeVerifierEngine

GYM_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = GYM_ROOT.parent
TASKS_DIR = GYM_ROOT / "data" / "revised" / "csm"
PINS_PATH = GYM_ROOT / "ablation" / "manifests" / "pinned_revisions.json"


def test_pinned_revisions_match_the_working_tree():
    """Pin drift must fail the suite, not surface after the fact.

    ``verify_pinning`` only validates *verifier sets*; nothing else compared the
    recorded content hashes against the working tree, so
    ``pinned_revisions.json`` silently drifted twice while the suite stayed
    green. A reported experiment is only auditable if the pin matches the code
    that produced it.
    """
    payload = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    pins = payload["ablation_files_sha256"]
    assert pins, "pinned_revisions.json carries no ablation hashes"

    stale = sorted(
        name
        for name, digest in pins.items()
        if not (GYM_ROOT / name).is_file() or sha256_file(GYM_ROOT / name) != digest
    )
    assert not stale, (
        "pinned_revisions.json is stale for: "
        f"{stale}. Re-run `python run_ablation.py --init-manifests` before reporting."
    )

#: The pinned task whose verifier names collide, confirmed by inspection.
COLLIDING_TASK = "task_20251205_153330_906_a8eea1c0_8c7a6205"
CLEAN_TASK = "task_20251209_132736_542_99ba2325_6705caf4"


def _task_path(task_id: str) -> Path:
    return TASKS_DIR / f"{task_id}.json"


def _skip_without_tasks() -> None:
    if not TASKS_DIR.is_dir():
        pytest.skip("local CSM task directory is not available")


# ---------- verifier collision detection ----------


def test_verifier_validation_detects_duplicate_names():
    _skip_without_tasks()
    report = validate_verifier_set(_task_path(COLLIDING_TASK))
    assert report.declared_count == 11
    assert report.has_duplicate_names
    assert report.duplicate_names
    assert report.collapsed_count < report.declared_count
    assert report.is_clean is False


def test_verifier_validation_reports_clean_tasks():
    _skip_without_tasks()
    report = validate_verifier_set(_task_path(CLEAN_TASK))
    assert report.declared_count == report.unique_name_count
    assert not report.has_duplicate_names
    assert report.is_clean is True


def test_eval_set_verifier_reports_cover_every_task():
    _skip_without_tasks()
    eval_set = build_eval_set(TASKS_DIR, registry=SchemaRegistry.from_static())
    reports = validate_eval_set_verifiers([r.task_config_path for r in eval_set])
    assert len(reports) == len(eval_set)
    assert any(item.has_duplicate_names for item in reports.values())


def test_verifier_bridge_recovers_collapsed_verifiers():
    from ablation.verifier_bridge import run_verifiers_indexed, summarize

    verifiers = [
        {"verifier_type": "database_state", "name": "create_new_case",
         "gym_name": "sn-csm-server",
         "validation_config": {"query": "SELECT 1;", "expected_value": 1}},
        {"verifier_type": "database_state", "name": "create_new_case",
         "gym_name": "sn-csm-server",
         "validation_config": {"query": "SELECT 2;", "expected_value": 2}},
        {"verifier_type": "database_state", "name": "update_case",
         "gym_name": "sn-csm-server",
         "validation_config": {"query": "SELECT 3;", "expected_value": 3}},
    ]
    gym_configs = [
        {"mcp_server_name": "sn-csm-server", "database_id": "db_1", "context": {"x-user-email": "a"}}
    ]
    engine = FakeVerifierEngine(passed=True)
    report = asyncio.run(
        run_verifiers_indexed(
            engine, verifiers, {"final_response": "done", "tool_results": []}, gym_configs
        )
    )

    assert report.collapsed_count == 2, "the official name-keyed view loses a verifier"
    assert report.indexed_count == 3
    assert report.lost_to_collision == 1
    assert set(report.indexed_results) == {
        "0:create_new_case",
        "1:create_new_case",
        "2:update_case",
    }
    assert len(engine.calls) == 3, "every verifier must actually execute"
    assert summarize({}) == {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}


def test_verifier_bridge_skips_unconfigured_gyms():
    from ablation.verifier_bridge import run_verifiers_indexed

    verifiers = [
        {"verifier_type": "database_state", "name": "v", "gym_name": "other-server",
         "validation_config": {"query": "SELECT 1;"}}
    ]
    engine = FakeVerifierEngine()
    report = asyncio.run(
        run_verifiers_indexed(engine, verifiers, {"final_response": "", "tool_results": []}, [])
    )
    assert report.indexed_count == 0
    assert not engine.calls


# ---------- fairness isolation ----------


def test_harness_does_not_import_verifier_derived_state():
    """`state_model.py` must be structurally blind to the analysis layer.

    The check is on executable code, not raw text: the guard legitimately names
    the forbidden fields in its deny-list, so what matters is that no module
    *reads* them.
    """
    source = (GYM_ROOT / "ablation" / "state_model.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not any("relevance" in name for name in imported)

    forbidden = {"validation_config", "expected_value", "verifiers", "comparison_type"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if node.slice.value in forbidden:
                raise AssertionError(f"state adapter reads {node.slice.value!r}")
        if isinstance(node, ast.Attribute) and node.attr in forbidden:
            raise AssertionError(f"state adapter reads attribute {node.attr!r}")

    # The whole Condition B import chain must stay clean of the analysis layer.
    for module in ("state_model", "state_model_orchestrator", "runner", "goals", "conditions"):
        text = (GYM_ROOT / "ablation" / f"{module}.py").read_text(encoding="utf-8")
        assert "from .relevance" not in text and "import relevance" not in text


def test_state_model_does_not_accept_a_task_config_payload():
    """The adapter's constructor takes a TaskRecord, never the task file."""
    import inspect

    from ablation.state_model import StateModelAdapter

    parameters = list(inspect.signature(StateModelAdapter.__init__).parameters)
    assert parameters[:3] == ["self", "api", "task"]
    assert "config" not in parameters and "payload" not in parameters


def test_eval_set_prompts_do_not_require_verifier_fields():
    """Intervention inputs are derivable from the prompt and tool list alone."""
    _skip_without_tasks()
    registry = SchemaRegistry.from_static()
    eval_set = build_eval_set(TASKS_DIR, registry=registry)
    for path in sorted(TASKS_DIR.glob("task_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        prompt_only_keys = {"user_prompt", "selected_tools", "gym_servers_config"}
        assert "verifiers" not in prompt_only_keys
        assert payload["user_prompt"]
        record = eval_set.get(path.stem)
        assert record.ggqr_task_type
        assert record.reference_entities, "anchor candidates exist without verifiers"
        assert record.complexity_category


def test_relevance_module_is_analysis_only():
    from ablation.relevance import extract_relevant_entities

    _skip_without_tasks()
    registry = SchemaRegistry.from_static()
    relevance = extract_relevant_entities(_task_path(COLLIDING_TASK), registry=registry)
    assert relevance.verifier_count == 11
    assert relevance.tables, "verifier SQL should yield known CSM tables"
    assert all(registry.table(table) is not None for table in relevance.tables)


# ---------- additivity ----------


def test_harness_did_not_modify_the_frozen_packages():
    """The two csm_env copies must still be byte-identical."""
    report = compare_csm_env_copies(WORKSPACE_ROOT / "csm_env", GYM_ROOT / "csm_env")
    assert report.is_identical, report.as_dict()
    require_parity(WORKSPACE_ROOT / "csm_env", GYM_ROOT / "csm_env")


def test_benchmark_modules_are_untouched_by_the_harness():
    """No harness module may monkey-patch benchmark internals."""
    for module in ("runner", "verifier_bridge", "state_model_orchestrator"):
        text = (GYM_ROOT / "ablation" / f"{module}.py").read_text(encoding="utf-8")
        assert "monkeypatch" not in text
        assert "setattr(" not in text


def test_orchestrators_react_is_not_edited():
    """The ReAct loop must remain the upstream implementation."""
    react = (GYM_ROOT / "orchestrators" / "react.py").read_text(encoding="utf-8")
    assert "class ReactOrchestrator(AgentOrchestrator):" in react
    assert "CSM_STATE_CONTEXT" not in react
    assert "ablation" not in react


def test_compute_score_is_not_edited():
    scorer = (GYM_ROOT / "compute_score.py").read_text(encoding="utf-8")
    assert "ablation" not in scorer
    assert "def get_score(sample):" in scorer


# ---------- readiness ----------


def test_preflight_and_dry_run_agree_offline():
    from ablation.preflight import filter_tasks, offline_preflight

    _skip_without_tasks()
    registry = SchemaRegistry.from_static()
    eval_set = build_eval_set(TASKS_DIR, registry=registry)
    eval_set, excluded = filter_tasks(eval_set, allow_missing_seeds=True)
    report = offline_preflight(eval_set, registry=registry)

    names = {check.name: check for check in report.checks}
    assert names["csm_env_schema"].ok
    assert names["csm_env_parity"].ok
    assert names["seed_archive"].ok, "the filtered set has every seed present"
    # Duplicates and a missing LLM config are reported but not blocking.
    assert not names["verifier_sets"].ok and not names["verifier_sets"].blocking
    assert not names["llm_config"].ok and not names["llm_config"].blocking
    assert report.ok, "the offline gate must pass once filtering is applied"
    assert excluded, "at least one task lacks its seed in this archive"


def test_eval_manifest_uses_portable_task_paths():
    import json

    manifest = GYM_ROOT / "ablation" / "manifests" / "csm_eval_set.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert not str(payload["generated_from"]).startswith(("C:\\", "D:\\"))
    for task in payload["tasks"]:
        path = task["task_config_path"]
        assert not (len(path) >= 2 and path[1] == ":"), path
        assert "\\" not in path, path
        assert (GYM_ROOT / path).is_file(), path


def test_dry_run_prints_the_matrix_without_touching_the_network():
    completed = subprocess.run(
        [
            sys.executable, "run_ablation.py", "--dry-run", "--num-runs", "1",
            "--allow-missing-seeds", "--output-folder", "out/test_dry_run",
        ],
        cwd=str(GYM_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    assert "PREFLIGHT PASS" in completed.stdout
    assert "Run matrix:" in completed.stdout
    assert "A_baseline/r1" in completed.stdout
    assert "B1_state_model_broad/r1" in completed.stdout
    assert "B2_state_model_minimal/r1" in completed.stdout
    assert not (GYM_ROOT / "out" / "test_dry_run").exists(), "dry run must not write records"


def test_existing_gym_wiring_smoke_test_still_passes():
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_csm_wiring_smoke.py", "-q",
         "-p", "no:cacheprovider"],
        cwd=str(GYM_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
