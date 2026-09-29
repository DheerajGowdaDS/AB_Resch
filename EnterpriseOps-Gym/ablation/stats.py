"""Experiment 1 metrics and paired statistics.

Metric semantics deliberately mirror ``compute_score.py``:

* **TSR** (primary) = mean over runs of the all-verifiers-passed indicator;
* **VPR** (secondary) = mean over runs of that run's verifier pass rate;
* **error rate** = fraction of runs carrying an agent error.

The paired analysis is the point of the harness: McNemar's exact test for the
binary outcome and a seeded paired bootstrap for the CI. Everything is
standard-library only, so no new runtime dependency is introduced.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .conditions import CONDITION_A, CONDITION_B1, Condition
from csm_env.query.models import RouteStatus

from .relevance import CONFIDENCE_EMPTY, RelevanceSet
from .results import TaskRunRecord

DEFAULT_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_ALPHA = 0.05

#: Below this many discordant pairs the exact binomial test is used.
EXACT_TEST_MAX_DISCORDANT = 25


def state_model_delivered(record: TaskRunRecord) -> bool:
    """Whether a state-model run actually delivered a usable intervention.

    This is the *record-side* view of the same rule the orchestrator enforces at
    injection time, so the report can never disagree with what the agent saw. It
    accepts any route status that is not ``FAILED``: a required table returning
    zero rows is a true statement about the database, not an undelivered
    intervention (see :func:`ablation.state_model_orchestrator.
    state_context_delivered` for the rationale).

    State Model 2.0: the gate keys on the condition's ``state_model`` property,
    so every state-model arm (B1 and B2) is gated, and non-state-model records
    (A, or archived names) pass through untouched.
    """
    from .conditions import condition_for

    if not condition_for(record.condition).state_model:
        return True
    telemetry = record.state_retrieval or {}
    if record.agent_error:
        return False
    route_status = str(telemetry.get("state_route_status", ""))
    if route_status == RouteStatus.FAILED or not route_status:
        return False
    return (
        bool(telemetry.get("state_model_available"))
        and str(telemetry.get("anchor_status", "")) == "RESOLVED"
        and int(telemetry.get("retrieved_records", 0) or 0) > 0
        and int(telemetry.get("context_tokens_injected", 0) or 0) > 0
    )


def _clean(records: Sequence[TaskRunRecord]) -> List[TaskRunRecord]:
    """Records that are scoreable under the frozen experiment intervention."""
    return [record for record in records if not record.agent_error and state_model_delivered(record)]


def tsr(records: Sequence[TaskRunRecord]) -> float:
    """Task Success Rate: mean of the all-verifiers-passed indicator."""
    usable = _clean(records)
    if not usable:
        return 0.0
    return sum(1 for record in usable if record.overall_success) / len(usable)


def vpr(records: Sequence[TaskRunRecord], *, indexed: bool = False) -> float:
    """Verifier-Level Pass Rate: mean of per-run verifier pass rates."""
    usable = _clean(records)
    if not usable:
        return 0.0
    rates: List[float] = []
    for record in usable:
        summary = record.verifier_summary_indexed if indexed else record.verifier_summary_collapsed
        total = int(summary.get("total", 0) or 0)
        if total:
            rates.append(float(summary.get("passed", 0) or 0) / total)
    if not rates:
        return 0.0
    return sum(rates) / len(rates)


def verifier_level_pass_rate_pooled(records: Sequence[TaskRunRecord]) -> float:
    """Pooled verifier pass ratio.

    Deliberately *not* the official VPR: ``compute_score.py`` averages per-file
    rates. Reported only as a clearly labelled companion metric (CON-010).
    """
    usable = _clean(records)
    total = 0
    passed = 0
    for record in usable:
        summary = record.verifier_summary_collapsed or {}
        total += int(summary.get("total", 0) or 0)
        passed += int(summary.get("passed", 0) or 0)
    return (passed / total) if total else 0.0


def agent_error_rate(records: Sequence[TaskRunRecord]) -> float:
    """Fraction of runs carrying an agent error (a reliability metric)."""
    if not records:
        return 0.0
    return sum(1 for record in records if record.agent_error) / len(records)


def mean_extra_metrics(records: Sequence[TaskRunRecord]) -> Dict[str, Optional[float]]:
    """Mean of the secondary telemetry, ignoring absent token usage."""
    usable = _clean(records)
    if not usable:
        return {}

    def mean(values: Iterable[float]) -> Optional[float]:
        collected = list(values)
        return (sum(collected) / len(collected)) if collected else None

    return {
        "steps_taken": mean(record.steps_taken for record in usable),
        "tool_call_count": mean(record.tool_call_count for record in usable),
        "read_query_count": mean(record.read_query_count for record in usable),
        "invalid_action_count": mean(record.invalid_action_count for record in usable),
        "latency_ms": mean(record.latency_ms for record in usable),
        "input_tokens": mean(
            record.input_tokens for record in usable if record.input_tokens is not None
        ),
        "output_tokens": mean(
            record.output_tokens for record in usable if record.output_tokens is not None
        ),
        "state_records": mean(
            float((record.state_retrieval or {}).get("retrieved_records", 0))
            for record in usable
            if record.state_retrieval
        ),
        "state_tokens": mean(
            float((record.state_retrieval or {}).get("retrieved_tokens", 0))
            for record in usable
            if record.state_retrieval
        ),
    }


@dataclass(frozen=True)
class PairedOutcomes:
    """The 2x2 outcome table of a matched A/B comparison.

    Attributes:
        both_pass: A pass, B pass.
        baseline_only: A pass, B fail - the state model regressed this pair.
        treatment_only: A fail, B pass - the state model fixed this pair.
        both_fail: A fail, B fail.
        excluded: Pairs dropped because of an agent error or a bad pairing.
    """

    both_pass: int
    baseline_only: int
    treatment_only: int
    both_fail: int
    excluded: int = 0

    @property
    def total(self) -> int:
        return self.both_pass + self.baseline_only + self.treatment_only + self.both_fail

    @property
    def discordant(self) -> int:
        return self.baseline_only + self.treatment_only

    def as_dict(self) -> Dict[str, int]:
        return {
            "both_pass": self.both_pass,
            "baseline_only_pass": self.baseline_only,
            "treatment_only_pass": self.treatment_only,
            "both_fail": self.both_fail,
            "excluded": self.excluded,
            "total_pairs": self.total,
        }


def pair_records(
    records: Sequence[TaskRunRecord],
    *,
    baseline: Condition = CONDITION_A,
    treatment: Condition = CONDITION_B1,
) -> PairedOutcomes:
    """Build the paired outcome table, dropping unusable pairs.

    Pairs are matched on ``(task_id, run_index)``, so repeats contribute
    independent pairs rather than being averaged away.
    """
    baseline_by_key = {
        (record.task_id, record.run_index): record
        for record in records
        if record.condition == baseline.name
    }
    treatment_by_key = {
        (record.task_id, record.run_index): record
        for record in records
        if record.condition == treatment.name
    }

    both_pass = baseline_only = treatment_only = both_fail = excluded = 0
    for key in sorted(set(baseline_by_key) & set(treatment_by_key)):
        left = baseline_by_key[key]
        right = treatment_by_key[key]
        if left.agent_error or right.agent_error or not state_model_delivered(right):
            excluded += 1
            continue
        left_ok = bool(left.overall_success)
        right_ok = bool(right.overall_success)
        if left_ok and right_ok:
            both_pass += 1
        elif left_ok:
            baseline_only += 1
        elif right_ok:
            treatment_only += 1
        else:
            both_fail += 1
    return PairedOutcomes(
        both_pass=both_pass,
        baseline_only=baseline_only,
        treatment_only=treatment_only,
        both_fail=both_fail,
        excluded=excluded,
    )


def mcnemar_exact_pvalue(baseline_only: int, treatment_only: int) -> float:
    """Two-sided exact binomial p-value for the discordant pairs.

    Under the null the discordant pairs split 50/50, so the probability of the
    observed split or a more extreme one is twice the lower tail.
    """
    n = baseline_only + treatment_only
    if n == 0:
        return 1.0
    smaller = min(baseline_only, treatment_only)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def mcnemar_test(outcomes: PairedOutcomes) -> Dict[str, Any]:
    """McNemar's test on the paired binary outcome.

    Uses the exact binomial test for fewer than 25 discordant pairs and the
    continuity-corrected chi-square otherwise, and reports which was used.
    """
    b = outcomes.baseline_only
    c = outcomes.treatment_only
    if b + c == 0:
        return {
            "test": "mcnemar",
            "method": "none",
            "statistic": 0.0,
            "p_value": 1.0,
            "discordant": 0,
            "baseline_only_pass": b,
            "treatment_only_pass": c,
        }

    if b + c < EXACT_TEST_MAX_DISCORDANT:
        return {
            "test": "mcnemar",
            "method": "exact_binomial",
            "statistic": float(min(b, c)),
            "p_value": mcnemar_exact_pvalue(b, c),
            "discordant": b + c,
            "baseline_only_pass": b,
            "treatment_only_pass": c,
        }

    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
    p_value = math.erfc(math.sqrt(chi2 / 2.0))
    return {
        "test": "mcnemar",
        "method": "chi2_continuity_corrected",
        "statistic": chi2,
        "p_value": p_value,
        "discordant": b + c,
        "baseline_only_pass": b,
        "treatment_only_pass": c,
    }


def success_deltas(outcomes: PairedOutcomes) -> List[float]:
    """Per-pair ``Success_B - Success_A`` values implied by the 2x2 table."""
    return (
        [0.0] * outcomes.both_pass
        + [-1.0] * outcomes.baseline_only
        + [1.0] * outcomes.treatment_only
        + [0.0] * outcomes.both_fail
    )


def paired_bootstrap_ci(
    outcomes: PairedOutcomes,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Dict[str, Any]:
    """Percentile bootstrap CI for the paired difference in success rate.

    Pairs are resampled with replacement, which is the correct unit because the
    design is paired. The seed is returned so a result can be reproduced exactly.
    """
    deltas = success_deltas(outcomes)
    n = len(deltas)
    if n == 0:
        return {
            "delta": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n": 0,
            "n_resamples": 0,
            "alpha": alpha,
            "seed": seed,
        }
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    observed = sum(deltas) / n
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    low_index = max(0, int(math.floor((alpha / 2.0) * n_resamples)))
    high_index = min(n_resamples - 1, int(math.ceil((1.0 - alpha / 2.0) * n_resamples)) - 1)
    return {
        "delta": observed,
        "ci_low": means[low_index],
        "ci_high": means[high_index],
        "n": n,
        "n_resamples": n_resamples,
        "alpha": alpha,
        "seed": seed,
    }


def breakdown_by_condition(
    records: Sequence[TaskRunRecord], conditions: Sequence[Condition]
) -> Dict[str, Dict[str, Any]]:
    """Per-condition headline metrics."""
    summary: Dict[str, Dict[str, Any]] = {}
    for condition in conditions:
        subset = [record for record in records if record.condition == condition.name]
        summary[condition.name] = {
            "label": condition.label,
            "state_model": condition.state_model,
            "runs": len(subset),
            "tsr": tsr(subset),
            "vpr": vpr(subset),
            "vpr_indexed": vpr(subset, indexed=True),
            "vpr_pooled_non_official": verifier_level_pass_rate_pooled(subset),
            "agent_error_rate": agent_error_rate(subset),
            "extra": mean_extra_metrics(subset),
        }
    return summary


def breakdown_by_complexity(
    records: Sequence[TaskRunRecord],
    *,
    conditions: Sequence[Condition],
    categories: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """TSR/VPR per complexity category per condition.

    Empty cells are reported as ``n/a`` rather than ``0``, so an absent category
    is never mistaken for a total failure.
    """
    table: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for category in categories:
        row: Dict[str, Dict[str, Any]] = {}
        for condition in conditions:
            subset = [
                record
                for record in records
                if record.complexity_category == category
                and record.condition == condition.name
            ]
            clean = _clean(subset)
            if not clean:
                row[condition.name] = {"n": 0, "tsr": None, "vpr": None}
            else:
                row[condition.name] = {
                    "n": len(clean),
                    "tsr": tsr(subset),
                    "vpr": vpr(subset),
                }
        table[category] = row
    return table


def breakdown_by_bucket(
    records: Sequence[TaskRunRecord],
    *,
    condition_name: str,
    attribute: str,
    bounds: Sequence[float],
) -> List[Dict[str, Any]]:
    """Bucket one numeric attribute into half-open ranges and score each bucket."""
    edges = list(bounds) + [float("inf")]
    lower = float("-inf")
    rows: List[Dict[str, Any]] = []
    for edge in edges:
        subset = [
            record
            for record in records
            if record.condition == condition_name
            and lower <= float(getattr(record, attribute, 0) or 0) < edge
        ]
        clean = _clean(subset)
        rows.append(
            {
                "attribute": attribute,
                "range": f"[{lower}, {edge})",
                "n": len(clean),
                "tsr": tsr(subset) if clean else None,
                "vpr": vpr(subset) if clean else None,
            }
        )
        lower = edge
    return rows


@dataclass(frozen=True)
class RetrievalQuality:
    """State-retrieval precision/recall for one run.

    ``granularity`` records whether the comparison happened over
    ``(table, row_id)`` pairs or fell back to tables only, so a weaker measure
    is never presented as a stronger one.
    """

    task_id: str
    condition: str
    granularity: str
    precision: Optional[float]
    recall: Optional[float]
    retrieved: int
    relevant: int
    overlap: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "condition": self.condition,
            "granularity": self.granularity,
            "precision": self.precision,
            "recall": self.recall,
            "retrieved": self.retrieved,
            "relevant": self.relevant,
            "overlap": self.overlap,
        }


def retrieval_quality(
    record: TaskRunRecord, relevance: RelevanceSet
) -> Optional[RetrievalQuality]:
    """Compare one run's retrieved state against the post-hoc relevance set.

    Returns ``None`` when the run has no state retrieval (Condition A) or the
    relevance set is empty, so those cases are excluded rather than counted as
    zero precision.
    """
    if not record.state_retrieval or not state_model_delivered(record):
        return None
    if relevance.confidence == CONFIDENCE_EMPTY:
        return None

    retrieved_rows = {
        (str(table), str(row_id))
        for table, row_id in (record.state_retrieval.get("retrieved_rows") or [])
        if isinstance((table, row_id), (list, tuple)) and len((table, row_id)) == 2
    }
    # `retrieved_tables` is a count in the telemetry payload; the names live in
    # the parallel `retrieved_table_names` field.
    retrieved_tables = {
        str(table)
        for table in (record.state_retrieval.get("retrieved_table_names") or [])
    }

    if relevance.rows:
        relevant = set(relevance.rows)
        granularity = "row"
        overlap = len(retrieved_rows & relevant)
        retrieved_count = len(retrieved_rows)
    else:
        relevant = set(relevance.tables)
        granularity = "table"
        overlap = len(retrieved_tables & relevant)
        retrieved_count = len(retrieved_tables)

    precision = (overlap / retrieved_count) if retrieved_count else None
    recall = (overlap / len(relevant)) if relevant else None
    return RetrievalQuality(
        task_id=record.task_id,
        condition=record.condition,
        granularity=granularity,
        precision=precision,
        recall=recall,
        retrieved=retrieved_count,
        relevant=len(relevant),
        overlap=overlap,
    )


def retrieval_summary(
    records: Sequence[TaskRunRecord],
    relevance_map: Mapping[str, RelevanceSet],
    *,
    condition_name: str,
) -> Dict[str, Any]:
    """Mean retrieval precision/recall for one condition, with exclusions."""
    qualities: List[RetrievalQuality] = []
    excluded = 0
    for record in records:
        if record.condition != condition_name:
            continue
        relevance = relevance_map.get(record.task_id)
        if relevance is None:
            excluded += 1
            continue
        quality = retrieval_quality(record, relevance)
        if quality is None:
            excluded += 1
            continue
        qualities.append(quality)

    def mean(values: Iterable[Optional[float]]) -> Optional[float]:
        collected = [value for value in values if value is not None]
        return (sum(collected) / len(collected)) if collected else None

    return {
        "runs_measured": len(qualities),
        "runs_excluded": excluded,
        "granularity": qualities[0].granularity if qualities else "n/a",
        "mean_precision": mean(q.precision for q in qualities),
        "mean_recall": mean(q.recall for q in qualities),
        "per_run": [q.as_dict() for q in qualities],
    }
