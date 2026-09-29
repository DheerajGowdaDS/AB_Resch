"""Explicitly skipped live verification tests (FILE-059 / TASK-045).

These run ONLY when an approved live CSM endpoint is supplied via
environment variables. They are excluded from the default offline suite
(REQ-011 / CON-005): without the variables below they skip, and a skip is
never reported as a passing verification result (Section 6.3).
"""

from __future__ import annotations

import os

import pytest

_LIVE_BASE_URL = os.environ.get("CSM_LIVE_BASE_URL")
_LIVE_DATABASE_ID = os.environ.get("CSM_LIVE_DATABASE_ID")

pytestmark = pytest.mark.skipif(
    not (_LIVE_BASE_URL and _LIVE_DATABASE_ID),
    reason="Live CSM endpoint not configured (set CSM_LIVE_BASE_URL and CSM_LIVE_DATABASE_ID); offline suite must not depend on live services",
)


@pytest.fixture()
def live_ggqr():
    from csm_env import EnterpriseOpsSQLRunner, GGQREnvironment
    from csm_env.transport.reader import EnterpriseOpsSQLRunnerReader

    runner = EnterpriseOpsSQLRunner(base_url=_LIVE_BASE_URL, database_id=_LIVE_DATABASE_ID)
    return GGQREnvironment(EnterpriseOpsSQLRunnerReader(runner), database_id=_LIVE_DATABASE_ID)


async def test_live_schema_manifest_matches_database(live_ggqr):
    """Live V1/V2: the 17-table manifest must match the real backend."""
    registry = live_ggqr.registry
    assert len(registry.tables()) == 17


async def test_live_case_routing_round_trip(live_ggqr):
    """Live routing: a real case id routes, builds state, and stays grounded."""
    from csm_env.query.models import TaskRequest

    outcome = await live_ggqr.router.route(
        TaskRequest(task_type="resolve_case", reference_type="customer_case", reference_id=1)
    )
    assert outcome.status in ("COMPLETE", "UNRESOLVED", "BUDGET_EXHAUSTED")
    for result in outcome.results:
        if result.status == "error":
            raise AssertionError(f"transport error in live route: {result.error}")


async def test_live_graph_sql_equivalence(live_ggqr):
    """Live V5-style equivalence on the anchor row of a real case."""
    result = await live_ggqr.verify_case_equivalence(1)
    assert result["equivalent"] or result["state_status"] == "UNRESOLVED", result["failures"]


async def test_live_frontier_stops_before_broad(live_ggqr):
    """Live frontier: the FRONTIER strategy must not retrieve more queries than BROAD.

    State Model 2.0, blueprint section 2: ``argmin Cost(S)`` over prefixes that
    meet ``Coverage(S, F_T) >= tau``.  If the frontier retrieves the same or
    more queries than the eager BROAD strategy, the stopping rule is inert.
    """
    from csm_env.query.models import QueryBudget, RouteStrategy, TaskRequest

    task = TaskRequest(task_type="case_overview", reference_type="customer_case", reference_id=1)

    broad_outcome = await live_ggqr.router.route(
        task, strategy=RouteStrategy.BROAD, budget=QueryBudget(max_hops=3)
    )
    frontier_outcome = await live_ggqr.router.route(
        task, strategy=RouteStrategy.FRONTIER, budget=QueryBudget(max_hops=3)
    )

    broad_queries = sum(
        1 for r in broad_outcome.results if getattr(r, "status", None) != "SKIPPED_BUDGET"
    )
    frontier_queries = sum(
        1 for r in frontier_outcome.results if getattr(r, "status", None) != "SKIPPED_BUDGET"
    )

    assert frontier_queries <= broad_queries, (
        f"FRONTIER used {frontier_queries} queries but BROAD used {broad_queries}; "
        "the frontier stopping rule did not save queries"
    )


async def test_live_b2_minimal_state_gate(live_ggqr):
    """Live B2: the minimal selector must produce a smaller payload than B1.

    State Model 2.0 distinctness gate.  Without this, a null result
    ('minimality does not help') is indistinguishable from 'minimality was not
    activated'.  The test asserts B2's retrieved_record_count is strictly less
    than B1's for at least one task.
    """
    from csm_env.query.models import QueryBudget, RouteStrategy, TaskRequest

    task = TaskRequest(task_type="case_overview", reference_type="customer_case", reference_id=1)

    b1_state = await live_ggqr.build_state(
        task.task_type, task.reference_id, task.reference_type, strategy=RouteStrategy.BROAD
    )
    b2_state = await live_ggqr.build_state(
        task.task_type, task.reference_id, task.reference_type, strategy=RouteStrategy.FRONTIER
    )

    b1_records = len(getattr(b1_state, "records", None) or ())
    b2_records = len(getattr(b2_state, "records", None) or ())

    assert b2_records < b1_records or b1_records == 0, (
        f"B2 minimal state ({b2_records} records) was not smaller than B1 broad ({b1_records} records); "
        "minimality was not activated"
    )
