"""Offline tests for the Phase 1 state-model reliability fixes.

These cover the four defects that suppressed Condition-B delivery in the
11-task pilot, plus the structural/coverage separation the blueprint's Phase 1.3
requires. No CSM server, no LLM, no network: every test uses local doubles.
"""

from __future__ import annotations

import pytest

from ablation.route_policy import (
    DEFAULT_HOP_LADDER,
    RouteBudgetExhausted,
    build_ladder,
    classify_attempt,
    select_attempt,
)
from ablation.state_model import _like_pattern, _quote_literal
from ablation.state_model_orchestrator import (
    DELIVERABLE_ROUTE_STATUSES,
    state_context_delivered,
)
from csm_env.query.models import (
    QueryBudget,
    QueryPlan,
    QueryResult,
    QueryStep,
    RouteOutcome,
    RouteStatus,
    StepStatus,
    TaskRequirements,
)
from csm_env.query.validation import SqlValidationError, validate_read_only_select


# ---------------------------------------------------------------------------
# Phase 1.4 - SQL generation / execution edge cases
# ---------------------------------------------------------------------------


def test_like_pattern_escapes_apostrophe_from_task_prompt():
    """Regression for the live 400 Bad Request.

    The prompt proper noun ``Wayne Enterprises' Windows Server`` previously
    produced the unbalanced literal ``'%Wayne Enterprises' Windows Server%'``.
    """
    pattern = _like_pattern("Wayne Enterprises' Windows Server")
    assert pattern == "'%Wayne Enterprises'' Windows Server%'"
    query = f"SELECT installed_product_id FROM installed_product WHERE name LIKE {pattern} LIMIT 5;"
    assert validate_read_only_select(query) == query


def test_quote_literal_doubles_apostrophe():
    assert _quote_literal("O'Brien") == "'O''Brien'"


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT * FROM user LIMIT 5;",
        "SELECT user.* FROM user;",
        "SELECT a FROM t WHERE x = 1; DROP TABLE user;",
        "DELETE FROM user;",
        "SELECT a FROM t -- comment",
        "SELECT a FROM t WHERE (x = 1 LIMIT 5;",
        "SELECT a FROM t WHERE n = 'O'Brien' LIMIT 5;",
        "",
    ],
)
def test_validation_rejects_malformed_statements(statement):
    """No malformed query may reach the live server (Phase 1.4)."""
    with pytest.raises(SqlValidationError):
        validate_read_only_select(statement)




@pytest.mark.parametrize(
    "statement",
    [
        "SELECT installed_product_id FROM installed_product LIMIT 5;",
        "SELECT a, b FROM t WHERE n = ''O''''Brien'' LIMIT 5;",
        "SELECT count(*) FROM t;",
        "SELECT DISTINCT a FROM t;",
    ],
)
def test_validation_accepts_well_formed_statements(statement):
    assert validate_read_only_select(statement) == statement


def test_validation_rejects_non_string():
    with pytest.raises(TypeError):
        validate_read_only_select(["SELECT a FROM t;"])


# ---------------------------------------------------------------------------
# Phase 1.3 - route resolution and the structural/coverage split
# ---------------------------------------------------------------------------


def test_ladder_is_ordered_shallowest_first_and_deduplicated():
    ladder = build_ladder((2, 3, 2, 4))
    assert [budget.max_hops for budget in ladder] == [2, 3, 4]
    assert DEFAULT_HOP_LADDER[0] < DEFAULT_HOP_LADDER[-1]


def test_ladder_clamps_hops_and_rejects_empty():
    assert build_ladder((99,))[0].max_hops == 6
    with pytest.raises(ValueError):
        build_ladder((0, -1))
    with pytest.raises(ValueError):
        build_ladder(max_tables=0)


def _outcome(planned, observed, *, anchor_rows=1, errors=()):
    """Build a minimal RouteOutcome for classification tests."""
    steps = tuple(
        QueryStep(step_id=f"s{i}", table=table, filters=(), columns=None, limit=10)
        for i, table in enumerate(planned)
    )
    plan = QueryPlan(
        anchor_table=planned[0],
        anchor_key="id",
        anchor_value=1,
        steps=steps,
        budget=QueryBudget(),
        task_type="t",
        route_id="r",
    )
    results = []
    for step in steps:
        if step.table in errors:
            results.append(QueryResult(step.step_id, step.table, status=StepStatus.ERROR))
        elif step.table in observed:
            results.append(
                QueryResult(step.step_id, step.table, rows=[{"id": 1}], status=StepStatus.OK)
            )
        else:
            results.append(QueryResult(step.step_id, step.table, status=StepStatus.EMPTY))
    if not anchor_rows:
        results[0] = QueryResult(results[0].step_id, results[0].table, status=StepStatus.EMPTY)
    return RouteOutcome(status=RouteStatus.COMPLETE, plan=plan, results=results)


def _requirements(*tables):
    return TaskRequirements(task_type="t", required_tables=tuple(tables))


def test_structurally_valid_route_with_empty_required_table_is_accepted():
    """The core Phase 1.3 fix.

    A required table that was *reached* but held no rows is a true statement
    about the database, not an undeliverable intervention.
    """
    outcome = _outcome(planned=("a", "b", "c"), observed=("a", "b"))
    attempt = classify_attempt(outcome, _requirements("a", "b", "c"))
    assert attempt.planned is True
    assert attempt.executed is True
    assert attempt.anchor_resolved is True
    assert attempt.accepted is True
    assert set(attempt.uncovered_required) == set()
    # With the requirement list in hand, the coverage gap is exact.
    assert set(outcome.unresolved_required_tables(_requirements("a", "b", "c"))) == {"c"}


def test_route_missing_a_required_table_is_not_accepted():
    outcome = _outcome(planned=("a", "b"), observed=("a", "b"))
    assert classify_attempt(outcome, _requirements("a", "b", "c")).accepted is False


def test_route_with_an_execution_error_is_not_accepted():
    outcome = _outcome(planned=("a", "b"), observed=("a", "b"), errors=("b",))
    assert classify_attempt(outcome, _requirements("a", "b")).accepted is False


def test_route_without_anchor_row_is_not_accepted():
    outcome = _outcome(planned=("a", "b"), observed=("a", "b"), anchor_rows=0)
    assert classify_attempt(outcome, _requirements("a", "b")).accepted is False


def test_select_attempt_accepts_an_acceptable_rung():
    requirements = _requirements("a", "b")
    chosen = select_attempt(
        [classify_attempt(_outcome(("a", "b"), ("a", "b")), requirements)]
    )
    assert chosen is not None and chosen.accepted is True


def test_select_attempt_raises_when_no_rung_works():
    requirements = _requirements("a", "zzz")
    outcome = _outcome(planned=("a",), observed=("a",))
    with pytest.raises(RouteBudgetExhausted, match="refuses to fabricate"):
        select_attempt([classify_attempt(outcome, requirements)])


# ---------------------------------------------------------------------------
# Phase 1.1 - the delivery rule
# ---------------------------------------------------------------------------


class _Context:
    """Minimal stand-in exposing the StateContext attributes the gate reads."""

    def __init__(self, route_status, *, empty=False, anchor=True, records=1, tokens=10):
        self.route_status = route_status
        self.is_empty = empty
        self.anchor = type("A", (), {"resolved": anchor})()
        self.retrieved_record_count = records
        self.retrieved_token_count = tokens


@pytest.mark.parametrize("status", DELIVERABLE_ROUTE_STATUSES)
def test_delivery_accepts_non_failed_routes(status):
    assert state_context_delivered(_Context(status)) is True


def test_delivery_rejects_failed_route():
    assert state_context_delivered(_Context(RouteStatus.FAILED)) is False


@pytest.mark.parametrize(
    "kwargs",
    [{"empty": True}, {"anchor": False}, {"records": 0}, {"tokens": 0}],
)
def test_delivery_requires_a_usable_context(kwargs):
    assert state_context_delivered(_Context(RouteStatus.COMPLETE, **kwargs)) is False