"""Report generation: the paper tables, derived only from persisted records.

Everything reported here is recomputed from ``TaskRunRecord`` files, so a reader
can regenerate every number from raw records with no live server.

State Model 2.0: the report is N-arm aware. ``A, B1, B2`` produces three
headline arms plus pairwise comparisons (A vs B1, A vs B2, B1 vs B2), the
treatment is selected by *name* (never positional indexing), and the
fail-closed delivery gate keys on the :attr:`Condition.state_model` property
rather than a condition-name literal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .conditions import (
    CONDITION_A,
    CONDITION_B1,
    Condition,
    condition_for,
    is_state_model_condition,
)
from .relevance import RelevanceSet
from .results import TaskRunRecord, load_run_records, utc_now
from .runner import reconcile_fingerprint_pairs
from .stats import (
    DEFAULT_ALPHA,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    breakdown_by_bucket,
    breakdown_by_complexity,
    breakdown_by_condition,
    mcnemar_test,
    pair_records,
    paired_bootstrap_ci,
    retrieval_summary,
    state_model_delivered,
)
from .task_registry import COMPLEXITY_CATEGORIES, EvalSetManifest

PathLike = Union[str, Path]

NA = "n/a"

#: Pairwise comparisons reported whenever both arms are present.
#: A vs each state-model arm, and B1 vs B2 (the State Model 2.0 contrast).
PAIRWISE_BASELINE = CONDITION_A


def _pct(value: Optional[float]) -> str:
    return NA if value is None else f"{value * 100:.1f}%"


def _num(value: Optional[float], digits: int = 1) -> str:
    return NA if value is None else f"{value:.{digits}f}"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _select_state_model_conditions(
    conditions: Sequence[Condition],
) -> Tuple[Optional[Condition], ...]:
    """The state-model arms among ``conditions``, in the given order."""
    return tuple(condition for condition in conditions if is_state_model_condition(condition))


def _select_baseline(conditions: Sequence[Condition]) -> Optional[Condition]:
    """The baseline arm, chosen by *property*, not position (P4, audit item-7).

    The baseline is the arm that does **not** expose the state model - the
    native-agent condition. Choosing it by the :attr:`Condition.state_model`
    property keeps the report correct no matter how the caller orders the
    conditions, so ``--conditions`` ordering can never silently move the
    baseline. If no native arm is present (a state-model-only sweep) there is
    no baseline and ``None`` is returned.
    """
    for condition in conditions:
        if not is_state_model_condition(condition):
            return condition
    return None


def summarize_state_injection(
    records: Sequence[TaskRunRecord],
    *,
    condition_name: Optional[str] = None,
) -> Dict[str, Any]:
    """How often a state-model arm actually delivered the injected state block.

    The intervention degrades to plain ReAct whenever the anchor is unresolved
    or the route fails, so the headline ΔTSR is a *diluted* effect unless the
    injection rate is high. This summary makes that dilution visible.

    Args:
        records: All persisted run records.
        condition_name: The treatment condition to summarize. When ``None``,
            the first state-model condition observed in the records is used,
            preserving the historical single-treatment behaviour.
    """
    if condition_name is None:
        condition_name = next(
            (
                record.condition
                for record in records
                if is_state_model_condition(condition_for(record.condition))
            ),
            None,
        )
    if condition_name is None:
        treatment_runs: List[TaskRunRecord] = []
    else:
        treatment_runs = [r for r in records if r.condition == condition_name]
    if not treatment_runs:
        return {
            "condition": condition_name,
            "runs": 0,
            "injected": 0,
            "injection_rate": None,
            "delivery_failures": 0,
            "causal_estimate_ready": False,
            "empty_context": 0,
            "no_telemetry": 0,
            "unresolved_anchor": 0,
            "failed_route": 0,
            "mean_retrieved_records": None,
            "mean_retrieved_tokens": None,
        }

    def _telemetry(record: TaskRunRecord) -> Dict[str, Any]:
        return record.state_retrieval or {}

    injected = sum(1 for record in treatment_runs if state_model_delivered(record))
    with_telemetry = [record for record in treatment_runs if record.state_retrieval]
    empty_context = sum(
        1
        for record in with_telemetry
        if not state_model_delivered(record)
        and int(_telemetry(record).get("retrieved_records", 0) or 0) == 0
    )
    no_telemetry = len(treatment_runs) - len(with_telemetry)
    unresolved_anchor = sum(
        1
        for record in treatment_runs
        if str(_telemetry(record).get("anchor_status", "")) == "UNRESOLVED"
    )
    failed_route = sum(
        1
        for record in treatment_runs
        if str(_telemetry(record).get("state_route_status", "")) == "FAILED"
    )
    delivered = [record for record in treatment_runs if state_model_delivered(record)]
    record_counts = [int(_telemetry(record).get("retrieved_records", 0) or 0) for record in delivered]
    token_counts = [int(_telemetry(record).get("retrieved_tokens", 0) or 0) for record in delivered]
    return {
        "condition": condition_name,
        "runs": len(treatment_runs),
        "injected": injected,
        "injection_rate": injected / len(treatment_runs),
        "delivery_failures": len(treatment_runs) - injected,
        "causal_estimate_ready": injected == len(treatment_runs) and not any(
            record.agent_error for record in treatment_runs
        ),
        "empty_context": empty_context,
        "no_telemetry": no_telemetry,
        "unresolved_anchor": unresolved_anchor,
        "failed_route": failed_route,
        "mean_retrieved_records": (sum(record_counts) / len(record_counts)) if record_counts else None,
        "mean_retrieved_tokens": (sum(token_counts) / len(token_counts)) if token_counts else None,
    }





def _pairwise_comparison(
    records: Sequence[TaskRunRecord],
    *,
    baseline: Condition,
    treatment: Condition,
    injection_summary: Optional[Dict[str, Any]],
    bootstrap_resamples: int,
    alpha: float,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    """One named pairwise comparison between two arms."""
    outcomes = pair_records(records, baseline=baseline, treatment=treatment)
    mcnemar = mcnemar_test(outcomes)
    bootstrap = paired_bootstrap_ci(
        outcomes, n_resamples=bootstrap_resamples, alpha=alpha, seed=bootstrap_seed
    )
    clean = (
        not is_state_model_condition(treatment)
        or injection_summary is None
        or bool(injection_summary.get("causal_estimate_ready"))
    )
    return {
        "baseline": baseline.name,
        "treatment": treatment.name,
        "outcomes": outcomes.as_dict(),
        "mcnemar": mcnemar,
        "bootstrap": bootstrap,
        "delta_tsr": bootstrap["delta"],
        "ci_low": bootstrap["ci_low"],
        "ci_high": bootstrap["ci_high"],
        "clean_intervention_delivery": clean,
    }


def build_report(
    records: Sequence[TaskRunRecord],
    *,
    eval_set: Optional[EvalSetManifest] = None,
    relevance_map: Optional[Mapping[str, RelevanceSet]] = None,
    conditions: Sequence[Condition] = (CONDITION_A, CONDITION_B1),
    engine_report: Optional[Dict[str, Any]] = None,
    pinned_revisions: Optional[Mapping[str, Any]] = None,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """Assemble the full Experiment 1 report as a JSON-ready mapping.

    Args:
        conditions: The arms of this sweep. The first state-model condition is
            the headline treatment (for a two-arm ``A,B`` sweep that is B1;
            for ``A,B1,B2`` the pairwise table carries both treatments).
    """
    conditions = tuple(conditions)
    if not conditions:
        raise ValueError("build_report requires at least one condition")
    # P4 (audit item-7): the baseline is the *non-state-model* arm, chosen by
    # property so the report is invariant to condition ordering. A positional
    # fallback (conditions[0]) only applies when a state-model-only sweep has
    # no native baseline to anchor against.
    baseline = _select_baseline(conditions) or conditions[0]
    state_model_arms = _select_state_model_conditions(conditions)
    treatment = state_model_arms[0] if state_model_arms else conditions[-1]

    # REQ-010 pairing gate: flag fingerprint-mismatched pairs as agent errors
    # before any metric is computed, so every number below excludes them.
    records, fingerprint_pairs = reconcile_fingerprint_pairs(records, conditions=conditions)
    matched_pairs = sum(1 for pair in fingerprint_pairs if pair["fingerprint_match"])
    fingerprint_pairing = {
        "pairs_reviewed": len(fingerprint_pairs),
        "pairs_matched": matched_pairs,
        "pairs_flagged": len(fingerprint_pairs) - matched_pairs,
        "reasons": [
            {
                "task_id": pair["task_id"],
                "run_index": pair["run_index"],
                "reason": pair["reason"],
            }
            for pair in fingerprint_pairs
            if not pair["fingerprint_match"]
        ],
    }

    by_condition = breakdown_by_condition(records, conditions)
    outcomes = pair_records(records, baseline=baseline, treatment=treatment)
    mcnemar = mcnemar_test(outcomes)
    bootstrap = paired_bootstrap_ci(
        outcomes, n_resamples=bootstrap_resamples, alpha=alpha, seed=bootstrap_seed
    )
    complexity = breakdown_by_complexity(
        records, conditions=conditions, categories=COMPLEXITY_CATEGORIES
    )

    baseline_stats = by_condition.get(baseline.name, {})
    treatment_stats = by_condition.get(treatment.name, {})

    # A difference is only meaningful when both arms have at least one scoreable
    # run. "No data" must never be rendered as "no effect".
    def scoreable_runs(condition_name: str) -> int:
        return sum(
            1
            for record in records
            if record.condition == condition_name and not record.agent_error
        )

    injection_summary = summarize_state_injection(records, condition_name=treatment.name)
    has_both_arms = scoreable_runs(baseline.name) > 0 and scoreable_runs(treatment.name) > 0
    # Fail-closed delivery gate, keyed on the condition *property*, not a name
    # literal: every state-model arm must have delivered its intervention on
    # every analyzed run before a causal effect estimate is reported.
    has_clean_intervention = not is_state_model_condition(treatment) or bool(
        injection_summary.get("causal_estimate_ready")
    )
    delta_tsr = delta_vpr = delta_error = None
    if has_both_arms and has_clean_intervention:
        delta_tsr = treatment_stats["tsr"] - baseline_stats["tsr"]
        delta_vpr = treatment_stats["vpr"] - baseline_stats["vpr"]
        delta_error = (
            treatment_stats["agent_error_rate"] - baseline_stats["agent_error_rate"]
        )

    # Pairwise comparisons across every state-model arm (State Model 2.0):
    # A vs B1, A vs B2, and B1 vs B2 when all three arms are present. Each
    # unordered pair is reported once, with the earlier arm as the baseline.
    comparison_pairs: List[Tuple[Condition, Condition]] = [
        (baseline, arm) for arm in state_model_arms if arm.name != baseline.name
    ]
    for index, arm in enumerate(state_model_arms):
        for other in state_model_arms[index + 1 :]:
            comparison_pairs.append((arm, other))
    pairwise: Dict[str, Any] = {}
    for comparison_baseline, treatment_arm in comparison_pairs:
        arm_injection = summarize_state_injection(records, condition_name=treatment_arm.name)
        pairwise[f"{comparison_baseline.name}_vs_{treatment_arm.name}"] = _pairwise_comparison(
            records,
            baseline=comparison_baseline,
            treatment=treatment_arm,
            injection_summary=arm_injection,
            bootstrap_resamples=bootstrap_resamples,
            alpha=alpha,
            bootstrap_seed=bootstrap_seed,
        )

    collapsed_total = sum(
        int((record.verifier_summary_collapsed or {}).get("total", 0) or 0) for record in records
    )
    indexed_total = sum(
        int((record.verifier_summary_indexed or {}).get("total", 0) or 0) for record in records
    )
    reconciliation = {
        "collapsed_verifiers": collapsed_total,
        "indexed_verifiers": indexed_total,
        "lost_to_collision": indexed_total - collapsed_total,
        "runs_where_collision_changed_outcome": sum(
            1
            for record in records
            if bool(record.verifier_summary_indexed)
            and record.verifier_summary_indexed.get("passed")
            != record.verifier_summary_collapsed.get("passed")
        ),
    }

    retrieval_quality: Dict[str, Any] = {}
    for treatment_arm in state_model_arms:
        retrieval_quality[treatment_arm.name] = retrieval_summary(
            records, relevance_map or {}, condition_name=treatment_arm.name
        )

    report: Dict[str, Any] = {
        "experiment": "experiment_1",
        "generated_at": utc_now(),
        "design": {
            "baseline": baseline.name,
            "treatment": treatment.name,
            "conditions": [condition.name for condition in conditions],
            "manipulated_variable": (
                "state_model_availability_and_selection_mode"
                if len(state_model_arms) > 1
                else "state_model_availability"
            ),
            "tool_mode": sorted({record.tool_mode for record in records}) or ["oracle"],
            "orchestrator": sorted({record.orchestrator for record in records}) or ["react"],
            "models": sorted(
                {f"{record.model_provider}/{record.model_name}" for record in records}
            ),
            "temperatures": sorted({record.temperature for record in records}),
            "run_indices": sorted({record.run_index for record in records}),
        },
        "headline": {
            **{condition.name: by_condition.get(condition.name, {}) for condition in conditions},
            "delta_tsr": delta_tsr,
            "delta_vpr": delta_vpr,
            "delta_agent_error_rate": delta_error,
        },
        "paired": {
            "outcomes": outcomes.as_dict(),
            "mcnemar": mcnemar,
            "bootstrap": bootstrap,
            "delta_tsr": bootstrap["delta"],
            "ci_low": bootstrap["ci_low"],
            "ci_high": bootstrap["ci_high"],
        },
        "pairwise": pairwise,
        "fingerprint_pairing": fingerprint_pairing,
        "state_injection": injection_summary,
        "state_injection_by_condition": {
            arm.name: summarize_state_injection(records, condition_name=arm.name)
            for arm in state_model_arms
        },
        "experimental_validity": {
            "clean_intervention_delivery": has_clean_intervention,
            "causal_effect_estimate_ready": has_clean_intervention and outcomes.excluded == 0,
            "reason": (
                "The state-model arm did not deliver the intervention on every run; effect size is withheld."
                if not has_clean_intervention
                else "Every state-model run delivered a usable state intervention and paired analysis is eligible."
            ),
        },
        "by_complexity": complexity,
        "by_steps": {
            condition.name: breakdown_by_bucket(
                records,
                condition_name=condition.name,
                attribute="steps_taken",
                bounds=(3, 6, 10, 15),
            )
            for condition in conditions
        },
        "verifier_reconciliation": reconciliation,
        "retrieval_quality": retrieval_quality,
        "record_counts": {
            "total": len(records),
            "by_condition": {
                condition.name: sum(1 for r in records if r.condition == condition.name)
                for condition in conditions
            },
        },
    }

    if eval_set is not None:
        report["evaluation_set"] = {
            "eval_set_id": eval_set.eval_set_id,
            "task_count": len(eval_set),
            "categories": eval_set.category_counts(),
            "schema_version": eval_set.schema_version,
        }
    if engine_report is not None:
        report["performance"] = engine_report
    if pinned_revisions is not None:
        report["pinned_revisions"] = dict(pinned_revisions)
    report["analysis_notes"] = [
        "Primary outcome is task success (all verifiers pass), the benchmark's own definition.",
        "Stratified tables are exploratory; the pre-registered comparison is the paired one.",
        "Pooled verifier pass rate is a companion metric, not the official VPR.",
        "State-model arms are fail-closed: runs without a usable state intervention are execution errors, not baseline fallbacks.",
        "The causal effect is reported only when the state intervention is delivered on every analyzed state-model run.",
    ]
    return report


def render_markdown(report: Mapping[str, Any], conditions: Sequence[Condition]) -> str:
    """Render the human-readable report."""
    conditions = tuple(conditions)
    baseline = _select_baseline(conditions) or conditions[0]
    state_model_arms = _select_state_model_conditions(conditions)
    treatment = state_model_arms[0] if state_model_arms else conditions[-1]
    headline = report["headline"]
    paired = report["paired"]
    lines: List[str] = []

    lines.append("# Experiment 1 - csm_env state-model ablation (CSM, EnterpriseOps-Gym)")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    rows = []
    for condition in conditions:
        stats = headline.get(condition.name) or {}
        rows.append(
            [
                condition.name,
                condition.label,
                str(stats.get("runs", 0)),
                _pct(stats.get("tsr")),
                _pct(stats.get("vpr")),
                _pct(stats.get("agent_error_rate")),
            ]
        )
    delta = report["experimental_validity"]
    lines.append("")
    lines.append(
        _md_table(
            ["Condition", "Arm", "Runs", "Task Success Rate", "Verifier Pass Rate", "Error Rate"],
            rows,
        )
    )
    lines.append("")
    if report["design"].get("conditions"):
        lines.append(
            f"- Baseline: {report['design']['baseline']}; headline treatment: "
            f"{report['design']['treatment']}; arms: "
            f"{', '.join(report['design']['conditions'])}"
        )
    if delta.get("clean_intervention_delivery"):
        lines.append(f"- ΔTSR (treatment - baseline): {_pct(headline.get('delta_tsr'))}")
        lines.append(f"- ΔVPR: {_pct(headline.get('delta_vpr'))}")
    else:
        lines.append(
            "- ΔTSR/ΔVPR withheld: the state intervention was not delivered on every analyzed run."
        )
    lines.append("")

    pairwise = report.get("pairwise") or {}
    if pairwise:
        lines.append("## Pairwise comparisons")
        lines.append("")
        pair_rows = []
        for name, comparison in pairwise.items():
            clean = comparison.get("clean_intervention_delivery")
            pair_rows.append(
                [
                    name.replace("_vs_", " vs "),
                    str(comparison["outcomes"].get("total_pairs", 0)),
                    _pct(comparison.get("delta_tsr")),
                    f"{comparison['mcnemar']['p_value']:.4g}",
                    "clean" if clean else "withheld (delivery)",
                ]
            )
        lines.append(
            _md_table(
                ["Comparison", "Pairs", "ΔTSR", "McNemar p", "Delivery"],
                pair_rows,
            )
        )
        lines.append("")

    lines.append("## Paired result")
    lines.append("")
    validity = report.get("experimental_validity") or {}
    lines.append(
        f"- Causal effect estimate ready: {validity.get('causal_effect_estimate_ready', False)}"
    )
    if validity.get("reason"):
        lines.append(f"- Validity note: {validity['reason']}")
    mcnemar = paired["mcnemar"]
    bootstrap = paired["bootstrap"]
    outcomes = paired["outcomes"]
    lines.append(f"- Paired delta TSR: {_pct(paired['delta_tsr'])}")
    lines.append(
        f"- 95% CI: [{_pct(paired['ci_low'])}, {_pct(paired['ci_high'])}] "
        f"(seed {bootstrap['seed']}, {bootstrap['n_resamples']} resamples, n={bootstrap['n']})"
    )
    lines.append(
        f"- McNemar p-value: {mcnemar['p_value']:.4g} ({mcnemar['method']}, "
        f"{mcnemar['discordant']} discordant pairs)"
    )
    lines.append(
        f"- Pairs: both pass {outcomes['both_pass']}, A only {outcomes['baseline_only_pass']}, "
        f"treatment only {outcomes['treatment_only_pass']}, both fail {outcomes['both_fail']}, "
        f"excluded {outcomes['excluded']}"
    )
    lines.append("")

    pairing = report.get("fingerprint_pairing") or {}
    if pairing.get("pairs_reviewed"):
        lines.append("## Initial-state pairing gate (REQ-010)")
        lines.append("")
        lines.append(f"- Pairs reviewed: {pairing.get('pairs_reviewed', 0)}")
        lines.append(f"- Pairs matched: {pairing.get('pairs_matched', 0)}")
        lines.append(f"- Pairs flagged and excluded: {pairing.get('pairs_flagged', 0)}")
        if pairing.get("reasons"):
            lines.append("")
            lines.append(
                _md_table(
                    ["Task", "Run", "Reason"],
                    [
                        [reason["task_id"], reason["run_index"], reason["reason"]]
                        for reason in pairing["reasons"]
                    ],
                )
            )
        lines.append("")

    injection_by_condition = report.get("state_injection_by_condition") or {}
    for condition in state_model_arms:
        injection = injection_by_condition.get(condition.name) or {}
        if not injection.get("runs"):
            continue
        lines.append(f"## State injection delivery ({condition.name})")
        lines.append("")
        lines.append(
            f"- Runs with the state block injected: {injection['injected']}/{injection['runs']} "
            f"({_pct(injection.get('injection_rate'))})"
        )
        lines.append(f"- Delivery failures: {injection.get('delivery_failures', 0)}")
        lines.append(
            f"- Causal effect estimate ready: {injection.get('causal_estimate_ready', False)}"
        )
        lines.append(f"- Empty contexts: {injection.get('empty_context', 0)}")
        lines.append(f"- Runs without state telemetry: {injection.get('no_telemetry', 0)}")
        lines.append(f"- Unresolved anchors: {injection.get('unresolved_anchor', 0)}")
        lines.append(f"- Failed routes: {injection.get('failed_route', 0)}")
        lines.append(f"- Mean retrieved records: {_num(injection.get('mean_retrieved_records'), 1)}")
        lines.append(f"- Mean retrieved tokens: {_num(injection.get('mean_retrieved_tokens'), 0)}")
        lines.append("")

    lines.append("## By state-dependence category (exploratory)")
    lines.append("")
    category_rows = []
    for category, per_condition in report["by_complexity"].items():
        row = [category]
        for condition in conditions:
            cell = per_condition.get(condition.name) or {}
            row.append(str(cell.get("n", 0)))
            row.append(_pct(cell.get("tsr")))
            row.append(_pct(cell.get("vpr")))
        category_rows.append(row)
    header = ["Category"]
    for condition in conditions:
        header.extend([f"{condition.name} n", f"{condition.name} TSR", f"{condition.name} VPR"])
    lines.append(_md_table(header, category_rows))
    lines.append("")
    return "\n".join(lines) + "\n" + _render_tail(report, treatment)


def _render_tail(report: Mapping[str, Any], treatment: Condition) -> str:
    """Sections after the category table: reconciliation, quality, performance."""
    lines: List[str] = []

    lines.append("## Verifier reconciliation (duplicate-name collision)")
    lines.append("")
    reconciliation = report["verifier_reconciliation"]
    lines.append(f"- Collapsed verifier results recorded: {reconciliation['collapsed_verifiers']}")
    lines.append(f"- Index-keyed verifier results: {reconciliation['indexed_verifiers']}")
    lines.append(f"- Lost to name collision: {reconciliation['lost_to_collision']}")
    lines.append(
        "- Runs where the collision changed the pass count: "
        f"{reconciliation['runs_where_collision_changed_outcome']}"
    )
    lines.append("")

    retrieval_quality = report.get("retrieval_quality") or {}
    retrieval = retrieval_quality.get(treatment.name) or {}
    if retrieval.get("runs_measured"):
        lines.append(f"## State retrieval quality ({treatment.name}, post-hoc)")
        lines.append("")
        lines.append(f"- Granularity: {retrieval['granularity']}")
        lines.append(f"- Mean precision: {_num(retrieval['mean_precision'], 3)}")
        lines.append(f"- Mean recall: {_num(retrieval['mean_recall'], 3)}")
        lines.append(
            f"- Runs measured: {retrieval['runs_measured']} "
            f"(excluded {retrieval['runs_excluded']})"
        )
        lines.append("")

    performance = report.get("performance") or {}
    if performance:
        lines.append("## Execution performance")
        lines.append("")
        lines.append(
            f"- Goals: {performance.get('total_goals')} total, "
            f"{performance.get('completed')} completed, "
            f"{performance.get('failed')} failed, {performance.get('skipped')} skipped"
        )
        lines.append(
            f"- Wall clock: {_num(performance.get('wall_clock_ms', 0) / 1000.0)} s "
            f"at concurrency {performance.get('concurrency')}"
        )
        lines.append(
            f"- Throughput: {_num(performance.get('goals_per_minute', 0), 2)} goals/min"
        )
        stage_rows = [
            [
                stage,
                _num(values.get("mean_ms")),
                _num(values.get("total_ms")),
                str(values.get("count")),
            ]
            for stage, values in (performance.get("stage_summary") or {}).items()
        ]
        if stage_rows:
            lines.append("")
            lines.append(_md_table(["Stage", "Mean ms", "Total ms", "Count"], stage_rows))
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    for note in report.get("analysis_notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def write_report(
    report: Mapping[str, Any],
    output_dir: PathLike,
    conditions: Sequence[Condition] = (CONDITION_A, CONDITION_B1),
) -> Dict[str, Path]:
    """Write ``report.json`` and ``report.md`` and return their paths."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "report.json"
    md_path = directory / "report.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(report, conditions), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def report_from_directory(
    records_dir: PathLike,
    output_dir: PathLike,
    *,
    eval_set: Optional[EvalSetManifest] = None,
    relevance_map: Optional[Mapping[str, RelevanceSet]] = None,
    conditions: Sequence[Condition] = (CONDITION_A, CONDITION_B1),
    engine_report: Optional[Mapping[str, Any]] = None,
    pinned_revisions: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Path]:
    """Load records from disk and write the report. Requires no live server."""
    records = load_run_records(records_dir)
    report = build_report(
        records,
        eval_set=eval_set,
        relevance_map=relevance_map,
        conditions=conditions,
        engine_report=dict(engine_report) if engine_report else None,
        pinned_revisions=pinned_revisions,
    )
    return write_report(report, output_dir, conditions)
