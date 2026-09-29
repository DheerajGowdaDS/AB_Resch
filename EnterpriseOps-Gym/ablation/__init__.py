"""Experiment 1 ablation harness for EnterpriseOps-Gym CSM tasks.

This package is **additive glue**. It composes the unmodified benchmark
(``benchmark``, ``orchestrators``, ``compute_score.py``) with the frozen
``csm_env`` state model and measures the paired A/B effect of that state model
on task success.

Design axes:

* **Goal-driven loop** — every ``(task, run, condition)`` triple is a
  :class:`ablation.goals.Goal` that moves through an explicit state machine
  (``SEED -> AGENT -> VERIFY -> RECORD -> SCORE``). ``ablation.goals.GoalEngine``
  drives that loop, supports bounded concurrency, resumable iteration, and
  per-goal performance tracking.
* **No static mocks** — the evaluation set, complexity categories, GGQR anchors,
  requirement mapping, revision pins, database fingerprints, and relevance sets
  are all *derived at runtime* from the real task JSON files, the real
  ``csm_env`` schema registry, and the real CSM database. Nothing about the CSM
  domain is hard-coded into the harness.
* **Fairness boundary** — the Condition B state adapter is structurally
  incapable of reading verifier-only fields (SEC-001). Verifier data is used
  exclusively by the post-hoc analysis layer (``ablation.relevance``).

Public names are exported lazily so that importing ``ablation`` never pulls in
``langchain`` or the benchmark clients unless they are actually used.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CONDITION_A",
    "CONDITION_B",
    "CONDITION_B1",
    "CONDITION_B2",
    "CONDITIONS",
    "Condition",
    "condition_from_name",
    "is_state_model_condition",
    "COMPLEXITY_CATEGORIES",
    "EvalSetManifest",
    "TaskRecord",
    "build_eval_set",
    "load_eval_set",
    "load_task_config",
    "VerifierSetReport",
    "validate_verifier_set",
    "initial_state_fingerprint",
    "ParityReport",
    "compare_csm_env_copies",
    "StateContext",
    "StateContextPolicy",
    "StateModelAdapter",
    "task_conditioned_minimal_state",
    "build_minimal_state",
    "StateUsageTelemetry",
    "StateModelReactOrchestrator",
    "STATE_CONTEXT_HEADER",
    "IndexedVerificationReport",
    "run_verifiers_indexed",
    "TaskRunRecord",
    "load_run_records",
    "write_run_record",
    "Goal",
    "GoalEngine",
    "GoalState",
    "PairOutcome",
    "run_condition",
    "run_pair",
    "reconcile_pair_fingerprints",
    "reconcile_fingerprint_pairs",
    "build_report",
    "write_report",
    "estimate_tokens",
]

_LAZY_EXPORTS = {
    "CONDITION_A": "ablation.conditions",
    "CONDITION_B": "ablation.conditions",
    "CONDITION_B1": "ablation.conditions",
    "CONDITION_B2": "ablation.conditions",
    "CONDITIONS": "ablation.conditions",
    "Condition": "ablation.conditions",
    "condition_from_name": "ablation.conditions",
    "is_state_model_condition": "ablation.conditions",
    "COMPLEXITY_CATEGORIES": "ablation.task_registry",
    "EvalSetManifest": "ablation.task_registry",
    "TaskRecord": "ablation.task_registry",
    "build_eval_set": "ablation.task_registry",
    "load_eval_set": "ablation.task_registry",
    "load_task_config": "ablation.task_registry",
    "VerifierSetReport": "ablation.verify_pinning",
    "validate_verifier_set": "ablation.verify_pinning",
    "initial_state_fingerprint": "ablation.fingerprint",
    "ParityReport": "ablation.parity",
    "compare_csm_env_copies": "ablation.parity",
    "StateContext": "ablation.state_model",
    "StateContextPolicy": "ablation.state_model",
    "StateModelAdapter": "ablation.state_model",
    "task_conditioned_minimal_state": "ablation.state_model",
    "build_minimal_state": "ablation.minimal_state",
    "estimate_tokens": "ablation.state_model",
    "StateUsageTelemetry": "ablation.state_model_orchestrator",
    "StateModelReactOrchestrator": "ablation.state_model_orchestrator",
    "STATE_CONTEXT_HEADER": "ablation.state_model_orchestrator",
    "IndexedVerificationReport": "ablation.verifier_bridge",
    "run_verifiers_indexed": "ablation.verifier_bridge",
    "TaskRunRecord": "ablation.results",
    "load_run_records": "ablation.results",
    "write_run_record": "ablation.results",
    "Goal": "ablation.goals",
    "GoalEngine": "ablation.goals",
    "GoalState": "ablation.goals",
    "PairOutcome": "ablation.runner",
    "run_condition": "ablation.runner",
    "run_pair": "ablation.runner",
    "reconcile_pair_fingerprints": "ablation.runner",
    "reconcile_fingerprint_pairs": "ablation.runner",
    "build_report": "ablation.report",
    "write_report": "ablation.report",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
