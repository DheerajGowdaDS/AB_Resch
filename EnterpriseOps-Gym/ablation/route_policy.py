"""Route-budget policy: escalate hops until the required tables are reachable.

Phase 1.3 of the blueprint asks for the *shortest valid schema path* rather
than "an acceptively restrictive fixed hop count". A single fixed budget cannot
satisfy both ends of the CSM graph:

* ``customer_case -> account`` is one hop and needs no help;
* ``location -> user -> user_group_member -> user_group`` is three hops, and a
  ``max_hops=2`` budget makes that required table look *unreachable* even though
  a perfectly valid structural path exists.

This module turns hop count into an ordered **search**: attempt the route with
an escalating ladder of budgets and keep the first rung that both plans and
executes. Determinism is preserved because the ladder is ordered and the first
success wins, so the same task always resolves to the same budget.

The ladder never fabricates a relationship. If no rung reaches a required table
through declared foreign keys, routing fails closed exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from csm_env.query.models import QueryBudget, RouteOutcome, StepStatus, TaskRequirements

#: Hop counts tried in order. The first rungs mirror the historically used
#: budgets; the larger ones exist so multi-hop CSM relations become reachable
#: without weakening the planner's structural guarantees.
DEFAULT_HOP_LADDER: Tuple[int, ...] = (2, 3, 4)

#: Bounds for the rest of the budget. These are inherited unchanged from the
#: production defaults, so escalation widens *depth* only, never result volume.
DEFAULT_MAX_TABLES = 12
DEFAULT_MAX_QUERIES = 32
DEFAULT_MAX_ROWS_PER_QUERY = 50
DEFAULT_MAX_TOTAL_ROWS = 500
DEFAULT_MAX_PARALLEL = 8

#: Refuse to escalate without bound even if a caller supplies a huge ladder.
MAX_LADDER_LENGTH = 8

#: An absolute ceiling on relational depth. CSM's widest useful chain is three
#: hops; four is already generous. Beyond this a route is almost certainly
#: retrieving noise rather than task-relevant state.
MAX_SUPPORTED_HOPS = 6


class RouteBudgetExhausted(RuntimeError):
    """No rung of the ladder produced a structurally valid route."""


def build_ladder(
    hops: Sequence[int] = DEFAULT_HOP_LADDER,
    *,
    max_tables: int = DEFAULT_MAX_TABLES,
    max_queries: int = DEFAULT_MAX_QUERIES,
    max_rows_per_query: int = DEFAULT_MAX_ROWS_PER_QUERY,
    max_total_rows: int = DEFAULT_MAX_TOTAL_ROWS,
    max_parallel: int = DEFAULT_MAX_PARALLEL,
) -> Tuple[QueryBudget, ...]:
    """Build the ordered, deduplicated budget ladder.

    Args:
        hops: Hop counts in preference order. Duplicates and non-positive values
            are dropped; values above :data:`MAX_SUPPORTED_HOPS` are clamped.
        max_tables: Table ceiling carried by every rung.
        max_queries: Query ceiling carried by every rung.
        max_rows_per_query: Per-query row ceiling carried by every rung.
        max_total_rows: Route-wide row ceiling carried by every rung.
        max_parallel: Concurrency ceiling carried by every rung.

    Returns:
        One :class:`QueryBudget` per usable hop count, shallowest first.

    Raises:
        ValueError: If ``max_tables`` is below 1, or no usable hop count remains.
    """
    if max_tables < 1:
        raise ValueError("max_tables must be >= 1")

    seen = set()
    budgets = []
    for hop in list(hops)[:MAX_LADDER_LENGTH]:
        clamped = min(int(hop), MAX_SUPPORTED_HOPS)
        if clamped < 1 or clamped in seen:
            continue
        seen.add(clamped)
        budgets.append(
            QueryBudget(
                max_hops=clamped,
                max_tables=max_tables,
                max_queries=max_queries,
                max_rows_per_query=max_rows_per_query,
                max_total_rows=max_total_rows,
                max_parallel=max_parallel,
            )
        )
    if not budgets:
        raise ValueError(f"hop ladder {list(hops)} contains no usable hop count")
    return tuple(budgets)


@dataclass(frozen=True)
class RouteAttempt:
    """One rung of the ladder, and what it achieved.

    Attributes:
        budget: The budget this rung used.
        outcome: The route outcome, or ``None`` when planning failed outright.
        planned: Whether every required table was reached by a schema path.
        executed: Whether every planned statement was accepted by the server.
        anchor_resolved: Whether the anchor produced a real row.
        accepted: Whether this rung is good enough to keep.
    """

    budget: QueryBudget
    outcome: Optional[RouteOutcome]
    planned: bool
    executed: bool
    anchor_resolved: bool
    accepted: bool

    @property
    def max_hops(self) -> int:
        return self.budget.max_hops

    @property
    def uncovered_required(self) -> Tuple[str, ...]:
        """Required tables that returned no rows, for reporting only.

        The requirement list is read from the route's own diagnostics when
        present, and otherwise reconstructed from the plan's task type is not
        possible, so this stays empty rather than guessing. Callers that hold the
        :class:`TaskRequirements` should use
        :meth:`csm_env.query.models.RouteOutcome.unresolved_required_tables`.
        """
        if self.outcome is None:
            return ()
        return tuple(sorted(self.outcome.unresolved_required_tables()))

    def as_dict(self) -> dict:
        return {
            "max_hops": self.max_hops,
            "planned": self.planned,
            "executed": self.executed,
            "anchor_resolved": self.anchor_resolved,
            "accepted": self.accepted,
            "route_status": self.outcome.status if self.outcome else None,
            "observed_tables": (
                sorted(self.outcome.observed_tables()) if self.outcome else []
            ),
            "uncovered_required": list(self.uncovered_required),
        }


def classify_attempt(
    outcome: Optional[RouteOutcome], requirements: TaskRequirements
) -> RouteAttempt:
    """Score one route outcome against the acceptance rule.

    A rung is accepted when the route is *structurally valid* for the
    requirements. Row coverage deliberately does not participate: an empty
    ``entitlement`` table is a real database state, and a required table reached
    through a valid foreign-key path but holding no matching row has still been
    *reached*. The pilot conflated the two and suppressed three of eleven
    Condition-B runs for no fault of the intervention.
    """
    if outcome is None:
        return RouteAttempt(
            budget=QueryBudget(),
            outcome=None,
            planned=False,
            executed=False,
            anchor_resolved=False,
            accepted=False,
        )

    required = set(requirements.required_tables)
    planned = required.issubset(outcome.planned_tables())
    executed = not outcome.errored_tables()
    anchor_resolved = any(
        result.table == outcome.plan.anchor_table
        and result.status == StepStatus.OK
        and bool(result.rows)
        for result in outcome.results
    )
    return RouteAttempt(
        budget=outcome.plan.budget,
        outcome=outcome,
        planned=planned,
        executed=executed,
        anchor_resolved=anchor_resolved,
        accepted=planned and executed and anchor_resolved,
    )


def select_attempt(attempts: Sequence[RouteAttempt]) -> RouteAttempt:
    """Return the first accepted attempt.

    Raises:
        RouteBudgetExhausted: If no rung produced an acceptable route. The
            message states what each rung achieved, so a delivery failure is
            diagnosable from the run record alone without a live server.
    """
    for attempt in attempts:
        if attempt.accepted:
            return attempt
    detail = "; ".join(
        f"max_hops={attempt.max_hops}"
        f"(planned={attempt.planned}, executed={attempt.executed}, "
        f"anchor={attempt.anchor_resolved})"
        for attempt in attempts
    )
    raise RouteBudgetExhausted(
        "no hop budget produced a structurally valid route to the required "
        f"tables: {detail}. The planner refuses to fabricate relationships."
    )


def attempts_as_dict(attempts: Sequence[RouteAttempt]) -> dict:
    """Serialise the whole ladder for the run record."""
    return {
        "ladder": [attempt.as_dict() for attempt in attempts],
        "accepted_max_hops": next(
            (attempt.max_hops for attempt in attempts if attempt.accepted), None
        ),
    }

