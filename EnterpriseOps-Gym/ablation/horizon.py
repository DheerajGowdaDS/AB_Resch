"""Phase 4: choose a step budget that lets the benchmark actually finish.

The pilot ran with ``max_steps = 5`` and produced ``TSR = 0/11``. That is not a
result; it is a measurement artifact. The CSM tasks are multi-step workflows
("relocate this device, then notify two account contacts"), so a five-step budget
terminates them mid-procedure and the primary metric cannot discriminate between
the two arms at all.

Picking a larger number is easy and insufficient. The blueprint asks to *validate*
that the chosen limit "does not artificially terminate substantial portions of the
benchmark". This module does exactly that by measuring, from persisted records:

* how many runs **hit the ceiling** (``steps_taken >= max_steps``);
* of those, how many failed while budget-capped.

A run that used every available step and then failed is *censored*: we do not
know whether it would have succeeded with more budget. :func:`analyse_horizon`
reports the censored fraction so the budget is chosen against evidence rather
than intuition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

#: Censored fraction above which a budget is judged too shallow.
#:
#: One third is deliberately conservative. If more than a third of runs are
#: terminated by the ceiling, task success is measuring the budget rather than
#: the agent, and any Delta-TSR computed from it is uninterpretable.
CENSORED_FRACTION_THRESHOLD = 1 / 3

#: Candidate step budgets for the empirical horizon protocol (Phase 4).
#:
#: The blueprint requires the budget to be *derived* from evidence rather than
#: assumed. These are the rungs :func:`select_common_horizon` walks, from the
#: first plausible budget to a clearly generous one. Selection returns a single
#: number, never a per-condition value, because the horizon must be identical
#: for both arms - a budget that differed between A and B would introduce a
#: second manipulated variable.
CANDIDATE_HORIZONS: Tuple[int, ...] = (10, 15, 20, 25, 30)


@dataclass(frozen=True)
class HorizonVerdict:
    """Whether a step budget is adequate for a set of runs.

    Attributes:
        max_steps: The budget under evaluation.
        runs: Number of runs examined.
        budget_capped: Runs that consumed the full budget.
        censored_failures: Runs that hit the ceiling *and* failed.
        censored_fraction: ``censored_failures / runs``.
        observed_max_steps: Largest ``steps_taken`` seen.
        sufficient: Whether the censored fraction is within the threshold.
    """

    max_steps: int
    runs: int
    budget_capped: int
    censored_failures: int
    censored_fraction: float
    observed_max_steps: int
    sufficient: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "runs": self.runs,
            "budget_capped": self.budget_capped,
            "censored_failures": self.censored_failures,
            "censored_fraction": self.censored_fraction,
            "observed_max_steps": self.observed_max_steps,
            "sufficient": self.sufficient,
        }


def analyse_horizon(
    records: Sequence[Any],
    *,
    condition_name: Optional[str] = None,
) -> HorizonVerdict:
    """Measure how often the step budget censored a run.

    Args:
        records: Persisted run records.
        condition_name: Restrict the analysis to one arm. Both arms share one
            budget, so leaving this ``None`` measures the pooled horizon.

    Returns:
        The horizon verdict for the given records.
    """
    selected = [
        record
        for record in records
        if condition_name is None or record.condition == condition_name
    ]
    total = len(selected)
    if total == 0:
        return HorizonVerdict(
            max_steps=0,
            runs=0,
            budget_capped=0,
            censored_failures=0,
            censored_fraction=0.0,
            observed_max_steps=0,
            sufficient=False,
        )

    budget_capped = 0
    censored = 0
    observed_max = 0
    for record in selected:
        steps = int(record.steps_taken or 0)
        budget = int(record.max_steps or 0)
        observed_max = max(observed_max, steps)
        if budget and steps >= budget:
            budget_capped += 1
            if not record.overall_success:
                censored += 1

    fraction = (censored / total) if total else 0.0
    return HorizonVerdict(
        max_steps=int(selected[0].max_steps or 0),
        runs=total,
        budget_capped=budget_capped,
        censored_failures=censored,
        censored_fraction=fraction,
        observed_max_steps=observed_max,
        sufficient=fraction <= CENSORED_FRACTION_THRESHOLD,
    )


def recommend_max_steps(
    records: Sequence[Any],
    *,
    condition_name: Optional[str] = None,
    proposed: int = 15,
    step: int = 5,
) -> Dict[str, Any]:
    """Assess ``proposed`` without inventing a larger budget from censored runs.

    Historical records are only informative about a larger horizon when at least
    some runs were observed with headroom above their own ceiling. When every run
    stopped at its own ceiling (the old max_steps=5 pilot), no larger budget can be
    inferred from those records; a fresh horizon calibration is required.
    """
    if proposed < 1:
        raise ValueError("proposed must be >= 1")
    verdict = analyse_horizon(records, condition_name=condition_name)
    usable = [
        record for record in records
        if (condition_name is None or record.condition == condition_name)
        and not getattr(record, "agent_error", False)
    ]
    if verdict.runs == 0:
        return {
            "recommended_max_steps": proposed,
            "verdict": verdict.as_dict(),
            "reason": "no records available; proposed budget requires fresh calibration",
            "validated": False,
        }

    all_at_ceiling = bool(usable) and all(
        int(record.steps_taken or 0) >= int(record.max_steps or 0)
        for record in usable
    )
    if all_at_ceiling:
        return {
            "recommended_max_steps": None,
            "verdict": verdict.as_dict(),
            "reason": (
                "every usable historical run terminated at its own ceiling; the "
                "records cannot justify a larger horizon. Run fresh horizon "
                "calibration with a generous temporary ceiling (for example 30) "
                "and select the smallest common candidate below the censoring "
                "threshold."
            ),
            "validated": False,
        }

    if verdict.sufficient:
        return {
            "recommended_max_steps": verdict.max_steps,
            "verdict": verdict.as_dict(),
            "reason": (
                f"observed censored fraction {verdict.censored_fraction:.1%} is within "
                "the predefined threshold; the current budget is supported by "
                "observed headroom."
            ),
            "validated": True,
        }

    headroom = max(step, (verdict.observed_max_steps // step + 1) * step)
    return {
        "recommended_max_steps": max(proposed, headroom),
        "verdict": verdict.as_dict(),
        "reason": (
            f"{verdict.censored_failures}/{verdict.runs} runs hit the ceiling and "
            "failed; a fresh run at a larger temporary ceiling is required before "
            "selecting the primary experiment horizon."
        ),
        "validated": False,
    }


def censor_fraction_at(records: Sequence[Any], max_steps: int) -> float:
    """Censored fraction if the sweep had used ``max_steps`` as its budget.

    A run is censored when it consumed the whole budget *and* failed, because
    only then is the outcome attributable to the ceiling rather than the agent.

    Args:
        records: Persisted run records.
        max_steps: Candidate budget to evaluate.

    Returns:
        The censored fraction, or ``1.0`` when no records are available (an
        unmeasured budget is treated as unusable rather than acceptable).
    """
    usable = [record for record in records if not getattr(record, "agent_error", False)]
    if not usable:
        return 1.0
    censored = sum(
        1
        for record in usable
        if int(record.steps_taken or 0) >= max_steps and not record.overall_success
    )
    return censored / len(usable)


def select_common_horizon(
    records: Sequence[Any],
    *,
    candidates: Sequence[int] = CANDIDATE_HORIZONS,
    threshold: float = CENSORED_FRACTION_THRESHOLD,
) -> Dict[str, Any]:
    """Choose the smallest budget at which censoring becomes acceptable.

    This replaces the previous heuristic, which derived a budget from
    ``observed_max_steps``. That derivation is unsound when censoring is heavy:
    with an 81.8% censored fraction every run stops at the ceiling, so the
    observed maximum equals the ceiling and carries no information about what
    the agent needed - which is how ``max_steps=5`` produced a spurious
    recommendation of 50.

    Instead, each candidate budget is re-scored against the *same* records.
    Raising the ceiling can only reduce censoring, so the smallest candidate
    whose fraction clears the threshold is selected. The result is a single
    value applied to both arms.

    Args:
        records: Persisted run records from the horizon calibration.
        candidates: Budgets to consider, in preference order.
        threshold: Maximum acceptable censored fraction.

    Returns:
        A JSON-serialisable selection record.

    Raises:
        ValueError: If ``candidates`` is empty.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")

    usable = [record for record in records if not getattr(record, "agent_error", False)]

    # Guard against in-place censoring. Every run that stopped *at* its own
    # ceiling carries no evidence about what a larger budget would have allowed:
    # re-scoring such records at 10 steps reports 0% censoring trivially, because
    # the run was physically terminated at 5 and never had the chance to use the
    # extra steps. Selecting a budget from that data would manufacture false
    # confidence, so the ladder refuses instead.
    all_at_ceiling = bool(usable) and all(
        int(record.steps_taken or 0) >= int(record.max_steps or 0) for record in usable
    )
    if all_at_ceiling:
        return {
            "selected_max_steps": None,
            "threshold": threshold,
            "candidates": [],
            "runs_considered": len(usable),
            "reason": (
                "every run terminated at its own ceiling, so these records cannot "
                "inform a horizon decision. A fresh calibration run at a generous "
                "budget (for example 30) must be executed first, so that actual "
                "step consumption is observed rather than assumed."
            ),
        }

    table = [
        {
            "max_steps": int(candidate),
            "censor_fraction": censor_fraction_at(records, int(candidate)),
            "acceptable": censor_fraction_at(records, int(candidate)) <= threshold,
        }
        for candidate in candidates
    ]
    acceptable = [row for row in table if row["acceptable"]]
    if acceptable:
        selected = acceptable[0]["max_steps"]
        reason = (
            f"smallest candidate budget whose censored fraction "
            f"({acceptable[0]['censor_fraction']:.1%}) is within the "
            f"{threshold:.0%} threshold"
        )
    else:
        selected = max(table, key=lambda row: row["max_steps"])["max_steps"]
        reason = (
            "no candidate budget reached the threshold; using the largest rung. "
            "The horizon must be re-measured with a calibration run before it is "
            "used for the primary experiment."
        )

    return {
        "selected_max_steps": selected,
        "threshold": threshold,
        "candidates": table,
        "runs_considered": len(usable),
        "reason": reason,
    }
