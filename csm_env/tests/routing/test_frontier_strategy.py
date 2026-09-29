"""State Model 2.0 section 2/10/13: the cost objective and frontier retrieval.

These tests pin the structural change that makes ``Cost(S)`` a real objective:

* ``CostWeights``/``compute_cost`` implement
  ``lambda_r*|rows| + lambda_t*|tokens| + lambda_q*|queries|`` with externalized
  weights and a deterministic token estimator;
* ``cheapest_covering_prefix`` is the literal ``argmin`` over prefixes that
  satisfy coverage;
* ``coverage_report`` scopes the universe to the *planned* tables, so a required
  table that has not been fetched yet keeps the loop going;
* ``RouteStrategy.FRONTIER`` actually stops early, reports its cost/coverage, and
  never achieves *less* coverage than the eager BROAD strategy.
"""

from __future__ import annotations

import asyncio

import pytest

from csm_env.query.cost import (
    CostWeights,
    cheapest_covering_prefix,
    compute_cost,
    estimate_tokens,
)
from csm_env.query.coverage import coverage_report, table_fact
from csm_env.query.models import (
    QueryBudget,
    RouteStatus,
    RouteStrategy,
    TaskRequest,
    TaskRequirements,
)
from csm_env.query.requirements import RequirementRegistry
from tests_support.helpers import make_router


# ---------------------------------------------------------------------------
# Cost(S) objective (blueprint section 2)
# ---------------------------------------------------------------------------


def test_cost_is_the_documented_linear_combination():
    weights = CostWeights(lambda_rows=1.0, lambda_tokens=0.05, lambda_queries=8.0)
    cost = compute_cost(rows=10, tokens=200, queries=3, weights=weights)
    assert cost.total == pytest.approx(10 * 1.0 + 200 * 0.05 + 3 * 8.0)
    assert cost.query_cost == pytest.approx(24.0)


def test_cost_weights_reject_degenerate_sets():
    with pytest.raises(ValueError):
        CostWeights(lambda_rows=0, lambda_tokens=0, lambda_queries=0).validate()
    with pytest.raises(ValueError):
        CostWeights(lambda_queries=-1).validate()


def test_token_estimator_is_deterministic_and_additive():
    rows = [{"a": "1", "b": "two"}]
    assert estimate_tokens(rows) == estimate_tokens(rows)
    assert estimate_tokens([{}, {}]) == 0
    # Doubling the payload can never make the estimate smaller, and the
    # estimator is monotone in payload size.
    assert estimate_tokens(rows + rows) >= estimate_tokens(rows)
    assert estimate_tokens([{"a": "1"}]) < estimate_tokens(rows)


def test_cheapest_covering_prefix_is_argmin_and_breaks_ties_early():
    cheap = compute_cost(rows=1, tokens=1, queries=1)
    costly = compute_cost(rows=9, tokens=9, queries=9)
    # Only indices 1 and 2 meet coverage; the cheaper of them wins.
    chosen = cheapest_covering_prefix([(0, cheap, False), (1, costly, False), (2, cheap, True)])
    assert chosen is not None and chosen[0] == 2
    # No prefix meets coverage -> None, never a silent fallback.
    assert cheapest_covering_prefix([(0, cheap, False)]) is None
    # Tie on cost breaks toward the earliest prefix.
    tied = cheapest_covering_prefix([(1, cheap, True), (2, cheap, True)])
    assert tied is not None and tied[0] == 1


# ---------------------------------------------------------------------------
# Coverage (blueprint section 12)
# ---------------------------------------------------------------------------


def test_coverage_universe_is_scoped_to_planned_tables(registry):
    """A planned-but-unretrieved required table must stay in the denominator."""
    req = RequirementRegistry(registry).requirements_for("resolve_case")
    report = coverage_report(
        requirement=req,
        registry=registry,
        rows_by_table={"customer_case": [{"case_id": 1233}]},
        anchor=("customer_case", 1233),
        planned_tables=("customer_case", "account", "entitlement", "case_sla", "product"),
    )
    assert table_fact("account") in report.universe
    assert table_fact("account") not in report.covered
    assert report.ratio < 1.0
    assert not report.meets(1.0)

    # Retrieving those tables can only ever raise coverage.
    extended = coverage_report(
        requirement=req,
        registry=registry,
        rows_by_table={
            "customer_case": [{"case_id": 1233, "account_id": 10}],
            "account": [{"account_id": 10}],
            "entitlement": [{"entitlement_id": 1}],
            "case_sla": [{"sla_id": 1}],
            "product": [{"product_id": 1}],
        },
        anchor=("customer_case", 1233),
        planned_tables=("customer_case", "account", "entitlement", "case_sla", "product"),
    )
    assert extended.ratio > report.ratio


def test_coverage_without_planned_scope_degrades_to_retrieved(registry):
    """Historical behaviour: scope defaults to the tables already retrieved."""
    req = RequirementRegistry(registry).requirements_for("resolve_case")
    report = coverage_report(
        requirement=req,
        registry=registry,
        rows_by_table={
            # A full customer_case row, as the projection would return it.
            "customer_case": [
                {
                    "case_id": 1233,
                    "priority": "P1",
                    "state": "open",
                    "assignment_group_id": 7,
                    "escalation": True,
                    "escalation_reason": "impact",
                }
            ]
        },
        anchor=("customer_case", 1233),
    )
    # `account` was never retrieved, so it is outside the universe entirely --
    # the loop cannot be forced to fetch anything by this degraded mode.
    assert table_fact("account") not in report.universe
    assert report.meets(1.0)


def test_required_relation_enters_the_universe_only_when_both_ends_planned(registry):
    req = TaskRequirements(
        task_type="resolve_case",
        required_tables=("customer_case",),
        required_relations=(("customer_case", "account"),),
    )
    only_one = coverage_report(
        requirement=req,
        registry=registry,
        rows_by_table={"customer_case": [{"case_id": 1, "account_id": 10}]},
        planned_tables=("customer_case",),
    )
    assert not any(fact.startswith("relation:") for fact in only_one.universe)

    both = coverage_report(
        requirement=req,
        registry=registry,
        rows_by_table={
            "customer_case": [{"case_id": 1, "account_id": 10}],
            "account": [{"account_id": 10}],
        },
        planned_tables=("customer_case", "account"),
    )
    assert "relation:customer_case:account" in both.covered
    assert both.meets(1.0)


# ---------------------------------------------------------------------------
# Frontier retrieval (blueprint sections 10 and 13)
# ---------------------------------------------------------------------------


def _request(strategy=None, budget=None) -> TaskRequest:
    return TaskRequest(
        task_type="resolve_case",
        reference_type="customer_case",
        reference_id=1233,
        strategy=strategy,
        budget=budget,
    )


def test_unknown_strategy_fails_closed(registry, reader):
    router = make_router(registry, reader)
    with pytest.raises(ValueError, match="Unknown route strategy"):
        asyncio.run(router.route(_request(strategy="magic")))


def test_frontier_reports_cost_coverage_and_planned_queries(registry, reader):
    router = make_router(registry, reader)
    outcome = asyncio.run(router.route(_request(strategy=RouteStrategy.FRONTIER)))

    assert outcome.status == RouteStatus.COMPLETE
    diagnostics = outcome.diagnostics
    assert diagnostics["strategy"] == RouteStrategy.FRONTIER
    # Section 12 budget validation is now visible on every route.
    assert diagnostics["queries_planned"] == len(outcome.plan.steps)
    assert diagnostics["queries_saved"] >= 0
    assert diagnostics["budget"]["coverage_threshold"] == 1.0

    # Section 2: Cost(S) is measured, not asserted.
    cost = diagnostics["cost"]
    assert cost["rows"] == sum(len(result.rows) for result in outcome.results)
    assert cost["queries"] == diagnostics["queries_executed"]
    assert cost["total"] > 0
    assert diagnostics["cost_prefixes"]

    # Section 12: coverage gate met at tau = 1.0.
    assert diagnostics["meets_coverage"] is True
    assert diagnostics["coverage_ratio"] == pytest.approx(1.0)

    # Section 10: the loop recorded the order it expanded relations in.
    assert diagnostics["frontier_order"]
    assert set(diagnostics["frontier_order"]).issubset(
        {step.step_id for step in outcome.plan.steps}
    )


def test_frontier_stops_early_when_tau_allows(registry, reader):
    """A looser coverage contract must buy back queries -- that is lambda_q."""
    loose = QueryBudget(max_hops=2, coverage_threshold=0.4)
    router = make_router(registry, reader)

    strict = asyncio.run(router.route(_request(strategy=RouteStrategy.FRONTIER)))
    relaxed = asyncio.run(
        router.route(_request(strategy=RouteStrategy.FRONTIER, budget=loose))
    )

    assert relaxed.status in (RouteStatus.COMPLETE, RouteStatus.UNRESOLVED)
    planned = len(relaxed.plan.steps)
    assert relaxed.diagnostics["queries_executed"] < planned, (
        f"expected early stop below tau=1.0, executed {relaxed.diagnostics['queries_executed']} "
        f"of {planned} planned"
    )
    assert relaxed.diagnostics["queries_saved"] > 0
    assert relaxed.diagnostics["cost"]["queries"] < strict.diagnostics["cost"]["queries"]


def test_frontier_returns_no_less_coverage_than_broad(registry, reader):
    router = make_router(registry, reader)
    broad = asyncio.run(router.route(_request(strategy=RouteStrategy.BROAD)))
    frontier = asyncio.run(router.route(_request(strategy=RouteStrategy.FRONTIER)))

    assert broad.status == RouteStatus.COMPLETE
    assert frontier.status == RouteStatus.COMPLETE

    broad_tables = {result.table for result in broad.results if result.rows}
    frontier_tables = {result.table for result in frontier.results if result.rows}

    # Both must satisfy the same requirement set...
    assert {"customer_case", "account", "entitlement", "case_sla", "product"}.issubset(
        frontier_tables
    )
    # ...and frontier may only ever retrieve a subset of what broad retrieved.
    assert frontier_tables.issubset(broad_tables)
    assert frontier.diagnostics["queries_executed"] <= broad.diagnostics["queries_executed"]


def test_broad_strategy_diagnostics_stay_backward_compatible(registry, reader):
    router = make_router(registry, reader)
    outcome = asyncio.run(router.route(_request(strategy=RouteStrategy.BROAD)))
    diagnostics = outcome.diagnostics
    # New keys are additive; every pre-existing key is still present and intact.
    for key in (
        "queries_executed",
        "total_rows",
        "tables_activated",
        "required_tables",
        "structurally_valid",
        "uncovered_required",
        "errored_tables",
        "budget",
    ):
        assert key in diagnostics, f"missing legacy diagnostic {key!r}"
    assert diagnostics["strategy"] == RouteStrategy.BROAD
    assert diagnostics["queries_saved"] == 0

