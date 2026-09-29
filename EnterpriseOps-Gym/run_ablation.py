#!/usr/bin/env python3
"""Experiment 1 ablation driver: manifest bootstrap, dry run, sweep, report.

This is the single entry point, and it is intentionally thin: every decision
lives in the ``ablation`` package so the same logic is reachable from tests and
from other drivers.

Modes (mutually exclusive, chosen by flag):

* ``--init-manifests``  derive the evaluation set, requirements and revision pins
  from the real task files and schema registry, and write them;
* ``--resolve-anchors`` seed each task database once, fingerprint it, and resolve
  every GGQR anchor from its prompt (read-only, then deletes the database);
* ``--dry-run``         run the offline readiness gate and print the exact
  ``task x run x condition`` matrix without touching the network or an LLM;
* ``--report-only``     recompute the report from persisted records;
* ``--diagnose``        emit the Phase 2/3/4 gates (state-model validity,
  evaluator validity, step-budget horizon) from persisted records. Offline, and
  exits non-zero when any gate fails;
* default              execute the sweep, then score and report.

Examples (PowerShell, from the ``EnterpriseOps-Gym`` directory)::

    python run_ablation.py --init-manifests
    python run_ablation.py --dry-run --llm-config conf/llm/my-model.json --output-folder out/experiment_1
    python run_ablation.py --conditions A,B --num-runs 3 --llm-config conf/llm/my-model.json --output-folder out/experiment_1
    python run_ablation.py --diagnose --output-folder out/experiment_1
    python run_ablation.py --report-only --output-folder out/experiment_1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from csm_env import SchemaRegistry

from ablation.conditions import Condition, parse_conditions
from ablation.goals import GoalEngine, build_goals
from ablation.preflight import (
    bootstrap_manifests,
    filter_tasks,
    resolve_live_anchors,
    run_preflight,
)
from ablation.initial_relevance import build_initial_relevance_map
from ablation.report import report_from_directory
from ablation.results import load_run_records
from ablation.runner import default_fingerprint_probe, make_goal_executor
from ablation.score_bridge import emit_from_directory
from ablation.stats import DEFAULT_BOOTSTRAP_RESAMPLES
from ablation.task_registry import build_eval_set, load_eval_set, save_json

GYM_ROOT = Path(__file__).resolve().parent
MANIFESTS_DIR = GYM_ROOT / "ablation" / "manifests"
DEFAULT_TASKS_DIR = GYM_ROOT / "data" / "revised" / "csm"
DEFAULT_EVAL_SET = MANIFESTS_DIR / "csm_eval_set.json"
DEFAULT_REQUIREMENTS = MANIFESTS_DIR / "csm_requirements.json"
DEFAULT_OUTPUT = GYM_ROOT / "out" / "experiment_1"
DEFAULT_ARCHIVE = GYM_ROOT / "gym_dbs.zip"


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="run_ablation.py",
        description="EnterpriseOps-Gym CSM paired A/B ablation of the csm_env state model.",
    )
    parser.add_argument(
        "--conditions",
        default="A,B1,B2",
        help="Conditions to run, e.g. 'A,B1,B2' (State Model 2.0 default) or 'A,B1'",
    )
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET), help="Evaluation set manifest")
    parser.add_argument("--tasks-dir", default=str(DEFAULT_TASKS_DIR), help="Real task JSON folder")
    parser.add_argument("--manifests-dir", default=str(MANIFESTS_DIR), help="Manifest output folder")
    parser.add_argument(
        "--requirements-path", default=str(DEFAULT_REQUIREMENTS), help="GGQR requirements manifest"
    )
    parser.add_argument("--tool-mode", default="oracle", help="Tool mode; only 'oracle' is supported")
    parser.add_argument("--orchestrator", default="react", help="Frozen orchestrator family")
    parser.add_argument("--num-runs", type=int, default=3, help="Repeats per task per condition")
    parser.add_argument("--llm-config", default=None, help="LLM config JSON in conf.example format")
    parser.add_argument("--output-folder", default=str(DEFAULT_OUTPUT), help="Where records land")
    parser.add_argument("--concurrency", type=int, default=1, help="Max goals in flight; 1 keeps seeding serial")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Agent step budget (max_iterations) applied identically to both arms; provisional until horizon calibration",
    )
    parser.add_argument(
        "--strict-fingerprint",
        action="store_true",
        help="Fail the run when the pre-run fingerprint probe errors instead of degrading to None",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for bootstrap resampling")
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES, help="Bootstrap draws"
    )
    parser.add_argument("--task-ids", nargs="*", default=None, help="Restrict to these task ids")
    parser.add_argument(
        "--allow-missing-seeds",
        action="store_true",
        help="Drop tasks whose seed SQL is absent from the archive instead of failing",
    )
    parser.add_argument("--base-url", default=None, help="CSM server URL; defaults to task config")
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE), help="Seed archive (gym_dbs.zip)")
    parser.add_argument(
        "--init-manifests", action="store_true", help="Build manifests from the real dataset"
    )
    parser.add_argument(
        "--resolve-anchors", action="store_true", help="Resolve live anchors and fingerprints"
    )
    parser.add_argument("--dry-run", action="store_true", help="Offline gate plus run matrix")
    parser.add_argument(
        "--report-only", action="store_true", help="Recompute the report from existing records"
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help=(
            "Emit the Phase 2/3/4 gates (state validity, evaluator validity, "
            "step-budget horizon) from existing records. Offline; exits non-zero "
            "when a gate fails"
        ),
    )
    parser.add_argument("--skip-preflight", action="store_true", help="Bypass the readiness gate")
    parser.add_argument(
        "--emit-score-layout", action="store_true", help="Also write the compute_score.py layout"
    )
    return parser


def _mode(args: argparse.Namespace) -> str:
    """Resolve the mutually exclusive run mode."""
    chosen = [
        mode
        for mode, flag in (
            ("init", args.init_manifests),
            ("resolve", args.resolve_anchors),
            ("dry-run", args.dry_run),
            ("report", args.report_only),
            ("diagnose", args.diagnose),
        )
        if flag
    ]
    if len(chosen) > 1:
        raise SystemExit(
            "--init-manifests, --resolve-anchors, --dry-run, --report-only and "
            "--diagnose are exclusive"
        )
    return chosen[0] if chosen else "sweep"


def _load_llm_config(path: Optional[str]) -> Any:
    """Load exactly one LLM config; the design fixes the model per sweep."""
    if not path:
        raise SystemExit("--llm-config is required to execute runs")
    from benchmark_utils import load_llm_configs

    configs = load_llm_configs(path)
    if not configs:
        raise SystemExit(f"no LLM configuration found in {path}")
    return configs[0]


def _load_eval_set(args: argparse.Namespace) -> Any:
    """Load the evaluation set, deriving it from real tasks when absent."""
    if Path(args.eval_set).is_file():
        return load_eval_set(args.eval_set)
    print(f"eval-set manifest not found; building from {args.tasks_dir}")
    return build_eval_set(args.tasks_dir, tool_mode=args.tool_mode)


def load_pinned_revisions(manifests_dir: str) -> Optional[Dict[str, Any]]:
    """Load the pinned-revision manifest when it exists."""
    path = Path(manifests_dir) / "pinned_revisions.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _print_matrix(eval_set: Any, conditions: Sequence[Condition], num_runs: int) -> None:
    """Print the exact ``task x run x condition`` matrix that would execute."""
    goals = build_goals(list(eval_set), conditions, num_runs=num_runs)
    print(
        f"\nRun matrix: {len(goals)} goals "
        f"({len(eval_set)} tasks x {len(conditions)} conditions x {num_runs} runs)"
    )
    for goal in goals:
        print(f"  {goal.describe()}  [{goal.complexity_category}]")


async def _async_main(args: argparse.Namespace) -> int:
    """Resolve the run mode and execute it."""
    registry = SchemaRegistry.from_static()
    mode = _mode(args)

    if mode == "init":
        eval_set = build_eval_set(args.tasks_dir, registry=registry, tool_mode=args.tool_mode)
        eval_set, excluded = filter_tasks(
            eval_set, archive=args.archive, allow_missing_seeds=False
        )
        if excluded:
            print(f"Excluded {len(excluded)} task(s) from the persisted manifest because their seed SQL is absent:")
            for item in excluded:
                print(f"  {item['task_id']}: {item['reason']}")
        if not eval_set:
            raise SystemExit("no seed-backed tasks remain to initialize the experiment manifest")
        paths = bootstrap_manifests(
            eval_set, registry=registry, manifests_dir=args.manifests_dir, archive=args.archive
        )
        print("Wrote manifests:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
        print(f"\nCategories: {eval_set.category_counts()}")
        return 0

    eval_set = _load_eval_set(args)
    eval_set, excluded = filter_tasks(
        eval_set,
        task_ids=args.task_ids,
        archive=args.archive,
        allow_missing_seeds=args.allow_missing_seeds,
    )
    if excluded:
        print(f"Excluded {len(excluded)} task(s):")
        for item in excluded:
            print(f"  {item['task_id']}: {item['reason']}")
    if not eval_set:
        raise SystemExit("no runnable tasks remain after filtering")

    conditions = parse_conditions(args.conditions)

    if mode == "resolve":
        resolved, resolutions = await resolve_live_anchors(
            eval_set,
            base_url=args.base_url or "",
            registry=registry,
            requirements_path=args.requirements_path,
            archive=args.archive,
        )
        save_json(resolved.as_dict(), args.eval_set)
        save_json(resolutions, Path(args.manifests_dir) / "anchor_resolutions.json")
        for task_id, info in resolutions.items():
            evidence = (info.get("anchor") or {}).get("evidence", "")
            print(f"{task_id}: {info.get('status')} {evidence}")
        return 0

    if mode == "report":
        output_dir = Path(args.output_folder)
        records = load_run_records(output_dir)
        if not records:
            raise SystemExit(f"no records found in {output_dir}")
        paths = report_from_directory(
            output_dir,
            output_dir,
            eval_set=eval_set,
            relevance_map=build_initial_relevance_map(eval_set, archive=args.archive),
            conditions=conditions,
            pinned_revisions=load_pinned_revisions(args.manifests_dir),
        )
        print(f"Report: {paths['markdown']} and {paths['json']}")
        return 0

    if mode == "diagnose":
        return _run_diagnostics(args, eval_set, conditions)

    if not args.skip_preflight:
        report = await run_preflight(
            eval_set,
            registry=registry,
            llm_config=args.llm_config,
            base_url=args.base_url,
            archive=args.archive,
        )
        print(report.render())
        if not report.ok:
            print("\nPreflight failed. Fix the blocking checks, or narrow the task set.")
            return 2

    if mode == "dry-run":
        _print_matrix(eval_set, conditions, args.num_runs)
        return 0

    return await _run_sweep(args, eval_set, conditions, registry)


def _run_diagnostics(args: argparse.Namespace, eval_set: Any, conditions: Sequence[Condition]) -> int:
    """Emit the Phase 2, 3, and 4 diagnostic artefacts from persisted records.

    This mode is the blueprint's "validate before you measure" gate. It needs no
    live server and no LLM: everything is recomputed from run records, so it can
    be run on the output of any completed sweep. A non-zero exit code means at
    least one gate failed, which is what makes it usable in CI.
    """
    from ablation.horizon import recommend_max_steps
    from ablation.state_validation import (
        build_state_report,
        evaluate_targets,
        render_state_report_markdown,
    )
    from ablation.verifier_validation import (
        build_verifier_validation_report,
        render_verifier_validation_markdown,
    )

    output_dir = Path(args.output_folder)
    records = load_run_records(output_dir)
    if not records:
        raise SystemExit(f"no records found in {output_dir}")
    relevance_map = build_initial_relevance_map(eval_set, archive=args.archive)
    # State Model 2.0: precision/recall are the metrics the *minimal* arm
    # exists to move, so when several state-model arms are present the Phase 2
    # intervention diagnostics target the minimal arm (B2). A two-arm sweep
    # keeps targeting its only treatment.
    state_model_arms = [condition for condition in conditions if condition.state_model]
    if not state_model_arms:
        raise SystemExit("diagnostics need the treatment condition (--conditions A,B1)")
    treatment = next(
        (condition for condition in state_model_arms if condition.minimal_state),
        state_model_arms[0],
    )
    additional_arms = [arm for arm in state_model_arms if arm.name != treatment.name]

    ok = True

    # --- Phase 2: the intervention, measured on its own terms -----------------
    state_report = build_state_report(
        records, relevance_map, condition_name=treatment.name
    )
    save_json(state_report, output_dir / "state_validation_report.json")
    (output_dir / "state_validation_report.md").write_text(
        "# Phase 2 - state-model validation\n\n"
        "Anchor -> Route -> Tables -> Records -> Delivery -> Precision -> Recall.\n"
        "Precision/recall are evaluator-only initial-state prompt/seed metrics and never feed the agent.\n\n"
        + render_state_report_markdown(state_report)
        + "\n",
        encoding="utf-8",
    )
    targets = evaluate_targets(state_report)
    print(f"[Phase 2] treatment={treatment.name} "
          f"delivery={state_report['delivery_rate']:.1%} "
          f"precision={state_report['mean_precision']} "
          f"recall={state_report['mean_recall']}")
    print(f"[Phase 2] targets met: {targets}")
    if not targets["delivery"]:
        print("[Phase 2] FAIL delivery < 100%; fix the state model before measuring agents")
        ok = False
    # P5.1/A2: the B1-vs-B2 contrast is only interpretable when the minimal arm
    # delivered a *reduced* payload without the gate firing. Report the contrast
    # target for the treatment, and when a broad arm (B1) is present cross-check
    # the mean delivered tokens so "minimality never happened" is not read as
    # "minimality doesn't help".
    if not targets["contrast"]:
        print(
            f"[Phase 2] WARN {treatment.name} contrast invalid: "
            f"{state_report.get('selection_gate_failed_delivered')} delivered run(s) "
            "had the minimal gate fire (broad state delivered); B1-vs-B2 contrast is diluted"
        )
    if treatment.minimal_state:
        broad = next(
            (arm for arm in additional_arms if not arm.minimal_state), None
        )
        if broad is not None:
            broad_report = build_state_report(records, relevance_map, condition_name=broad.name)
            minimal_tokens = state_report.get("mean_retrieved_tokens")
            broad_tokens = broad_report.get("mean_retrieved_tokens")
            reduced = (
                minimal_tokens is not None
                and broad_tokens is not None
                and minimal_tokens <= broad_tokens
            )
            print(
                f"[Phase 2] contrast payload {broad.name}={broad_tokens} -> "
                f"{treatment.name}={minimal_tokens} (reduced={reduced})"
            )
            if not reduced:
                print(
                    f"[Phase 2] FAIL {treatment.name} did not reduce the payload vs {broad.name}; "
                    "the B1-vs-B2 contrast cannot be interpreted"
                )
                ok = False

    # Companion Phase 2 report for every additional state-model arm so B1 and
    # B2 delivery/precision/recall are always both visible in a 3-arm sweep.
    for arm in additional_arms:
        arm_report = build_state_report(
            records, relevance_map, condition_name=arm.name
        )
        save_json(arm_report, output_dir / f"state_validation_report_{arm.name}.json")
        (output_dir / f"state_validation_report_{arm.name}.md").write_text(
            f"# Phase 2 - state-model validation ({arm.name})\n\n"
            + render_state_report_markdown(arm_report)
            + "\n",
            encoding="utf-8",
        )
        print(f"[Phase 2] {arm.name}: delivery={arm_report['delivery_rate']:.1%} "
              f"precision={arm_report['mean_precision']} "
              f"recall={arm_report['mean_recall']}")

    # --- Phase 3: is the evaluator itself unambiguous? ------------------------
    verifier_report = build_verifier_validation_report(
        [record.task_config_path for record in eval_set], records=records
    )
    save_json(verifier_report, output_dir / "verifier_validation_report.json")
    (output_dir / "verifier_validation_report.md").write_text(
        "# Phase 3 - evaluator validity\n\n"
        "Official (name-keyed) vs reconciled (index-keyed) verifier semantics.\n\n"
        + render_verifier_validation_markdown(verifier_report)
        + "\n",
        encoding="utf-8",
    )
    print(f"[Phase 3] tasks_with_duplicate_names="
          f"{verifier_report['tasks_with_duplicate_names']} "
          f"results_lost={verifier_report['tasks_losing_results_to_collision']} "
          f"unambiguous={verifier_report['evaluator_unambiguous']}")
    if not verifier_report["evaluator_unambiguous"]:
        print("[Phase 3] FAIL duplicate verifier names changed a task verdict; "
              "the final experiment would depend on an evaluator ambiguity")
        ok = False

    # --- Phase 4: does the step budget let the benchmark finish? --------------
    recommendation = recommend_max_steps(records, proposed=args.max_steps)
    save_json(recommendation, output_dir / "horizon_report.json")
    verdict = recommendation["verdict"]
    print(f"[Phase 4] max_steps={verdict['max_steps']} "
          f"censored={verdict['censored_failures']}/{verdict['runs']} "
          f"({verdict['censored_fraction']:.1%}) -> recommend "
          f"{recommendation['recommended_max_steps']}")
    if not verdict["sufficient"]:
        print("[Phase 4] FAIL the step budget is censoring runs; task success is "
              "measuring the budget, not the agent")
        ok = False

    print(f"\nDiagnostics written to {output_dir}")
    return 0 if ok else 1


async def _run_sweep(
    args: argparse.Namespace, eval_set: Any, conditions: Sequence[Condition], registry: Any
) -> int:
    """Execute the goal matrix, then score and report."""
    output_dir = Path(args.output_folder)
    llm_config = _load_llm_config(args.llm_config)
    executor = make_goal_executor(
        list(eval_set),
        llm_config,
        output_dir,
        tool_mode=args.tool_mode,
        requirements_path=args.requirements_path,
        probe=default_fingerprint_probe(registry, strict=args.strict_fingerprint),
        conditions_by_name={condition.name: condition for condition in conditions},
        max_steps=args.max_steps,
        archive=args.archive,
    )

    def on_transition(goal: Any, state: str, detail: str) -> None:
        suffix = f" ({detail})" if detail else ""
        print(f"  [{state}] {goal.describe()}{suffix}", flush=True)

    engine = GoalEngine(
        executor,
        output_dir=str(output_dir),
        max_concurrency=args.concurrency,
        on_transition=on_transition,
    )
    goals = build_goals(list(eval_set), conditions, num_runs=args.num_runs)
    print(f"Executing {len(goals)} goals at concurrency {args.concurrency}...")
    engine_report = await engine.run(goals)
    print(
        f"Goals: {len(engine_report.completed)} completed, "
        f"{len(engine_report.failed)} failed, {len(engine_report.skipped)} skipped"
    )
    save_json(engine_report.as_dict(), output_dir / "engine_report.json")

    relevance_map = build_initial_relevance_map(eval_set, archive=args.archive)
    paths = report_from_directory(
        output_dir,
        output_dir,
        eval_set=eval_set,
        relevance_map=relevance_map,
        conditions=conditions,
        engine_report=engine_report.as_dict(),
        pinned_revisions=load_pinned_revisions(args.manifests_dir),
    )
    print(f"Report: {paths['markdown']} and {paths['json']}")

    if args.emit_score_layout:
        for condition, path in emit_from_directory(output_dir, output_dir / "compute_score").items():
            print(f"compute_score layout for {condition}: {path}")

    records = load_run_records(output_dir)
    errored = [record for record in records if record.agent_error]
    if errored:
        print(f"{len(errored)} record(s) carry an agent error; see report.md")
    return 0 if not errored else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    if args.num_runs < 1:
        raise SystemExit("--num-runs must be >= 1")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")
    if args.orchestrator != "react":
        raise SystemExit(
            f"--orchestrator {args.orchestrator!r} is not frozen for Experiment 1; only 'react' is allowed"
        )
    random.seed(args.seed)
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    sys.exit(main())
