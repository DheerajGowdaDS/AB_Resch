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
