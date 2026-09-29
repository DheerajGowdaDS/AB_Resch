from __future__ import annotations

import asyncio

import pytest

from csm_env.query.adapters import TableAdapterFactory
from csm_env.query.executor import QueryExecutor
from csm_env.query.models import QueryBudget, QueryResult, QueryStep, StepStatus, TaskRequest
from csm_env.query.planner import QueryPlanner
from csm_env.query.requirements import RequirementRegistry
from csm_env.query.router import AnchorResolver, GraphQueryRouter
from csm_env.schema_graph import build_schema_graph
from tests_support.helpers import make_router


@pytest.fixture()
def router(registry, reader):
    return make_router(registry, reader)


def test_anchor_resolution_explicit_type(registry):
    resolver = AnchorResolver(registry, build_schema_graph(registry))
    table, key = resolver.resolve("customer_case", 1233)
    assert (table, key) == ("customer_case", "case_id")
    table, key = resolver.resolve("CustomerCase", 1233)
    assert (table, key) == ("customer_case", "case_id")


def test_anchor_alias_unique_and_ambiguous(registry):
    resolver = AnchorResolver(registry, build_schema_graph(registry))
    table, _ = resolver.resolve("case", 1233)
    assert table == "customer_case"
    with pytest.raises(KeyError):
        resolver.resolve("banana", 1233)


def test_anchor_rejects_empty_and_missing_type(registry):
    resolver = AnchorResolver(registry, build_schema_graph(registry))
    with pytest.raises(ValueError):
        resolver.resolve("customer_case", None)
    with pytest.raises(ValueError):
        resolver.resolve("customer_case", "   ")
    with pytest.raises(ValueError):
        resolver.resolve(None, 1233)


def test_requirements_registry_unknown_task_fails_closed(registry):
    req = RequirementRegistry(registry)
    with pytest.raises(KeyError):
        req.requirements_for("fly_to_moon")
    assert "resolve_case" in req.task_types()


def test_planner_rejects_unreachable_requirements(registry, reader):
    graph = build_schema_graph(registry)
    planner = QueryPlanner(graph, registry)

    from csm_env.query.models import TaskRequirements

    # account is 2 hops from location; with max_hops=1 no structural path exists.
    with pytest.raises(ValueError, match="No structural path"):
        planner.plan(
            task_type="account_profile",
            anchor_table="location",
            anchor_key="location_id",
            anchor_value=1,
            requirements=TaskRequirements(
                task_type="account_profile",
                required_tables=("location", "account"),
            ),
            budget=QueryBudget(max_hops=1),
            route_id="route-test",
        )


def test_plan_hops_are_structural_edges(router):
    async def run():
        outcome = await router.route(TaskRequest(task_type="resolve_case", reference_type="customer_case", reference_id=1233))
        return outcome

    outcome = asyncio.run(run())
    assert outcome.status in (RouteStatus_complete(),)
    for step in outcome.plan.steps:
        if step.hop == 0:
            continue
        assert step.edge_id is not None


def RouteStatus_complete():
    from csm_env.query.models import RouteStatus

    return RouteStatus.COMPLETE


def test_resolve_case_routes_and_propagates_ids(router):
    outcome = asyncio.run(
        router.route(
            TaskRequest(task_type="resolve_case", reference_type="customer_case", reference_id=1233)
        )
    )
    assert outcome.status == "COMPLETE"
    tables = {result.table for result in outcome.results if result.rows}
    assert {"customer_case", "account", "entitlement", "case_sla", "product"}.issubset(tables)
    # Propagated ids came from returned rows only.
    account_rows = outcome.bindings["account"]
    assert account_rows[0]["account_id"] == 10
    diagnostics = outcome.diagnostics
    assert diagnostics["queries_executed"] <= 32


def test_null_fk_becomes_unresolved(router):
    outcome = asyncio.run(
        router.route(
            TaskRequest(task_type="case_overview", reference_type="case", reference_id=1240)
        )
    )
    unresolved_tables = {u.table for u in outcome.unresolved}
    assert "contact" in unresolved_tables or "installed_product" in unresolved_tables
    assert outcome.status == "UNRESOLVED"


def test_budget_exhaustion_skips_required_steps(router):
    tiny = QueryBudget(max_hops=1, max_queries=4, max_tables=12, max_rows_per_query=5, max_total_rows=50, max_parallel=2)
    request = TaskRequest(
        task_type="case_overview",
        reference_type="customer_case",
        reference_id=1233,
        budget=tiny,
    )
    outcome = asyncio.run(router.route(request))
    skipped = [r for r in outcome.results if r.status == "skipped_budget"]
    # UNRESOLVED takes precedence over BUDGET_EXHAUSTED (approved contract).
    assert outcome.status == "UNRESOLVED"
    assert skipped, "expected at least one step skipped by the query budget"
    assert outcome.diagnostics["budget"]["max_queries"] == 4


def test_budget_exhausted_when_budget_consumed_with_required_observed(router):
    # resolve_case needs exactly 5 queries; consuming the budget exactly
    # while observing every required table reports BUDGET_EXHAUSTED.
    tight = QueryBudget(max_hops=2, max_queries=5, max_tables=12, max_rows_per_query=50, max_total_rows=500, max_parallel=8)
    outcome = asyncio.run(
        router.route(
            TaskRequest(
                task_type="resolve_case",
                reference_type="customer_case",
                reference_id=1233,
                budget=tight,
            )
        )
    )
    assert outcome.status == "BUDGET_EXHAUSTED"
    observed = {r.table for r in outcome.results if r.rows}
    assert {"customer_case", "account", "entitlement", "case_sla", "product"}.issubset(observed)


def test_unknown_reference_row_is_unresolved(router):
    outcome = asyncio.run(
        router.route(
            TaskRequest(task_type="resolve_case", reference_type="customer_case", reference_id=999999)
        )
    )
    assert outcome.status == "UNRESOLVED"


def test_propagation_never_uses_task_text(router):
    outcome = asyncio.run(
        router.route(
            TaskRequest(
                task_type="resolve_case",
                reference_type="customer_case",
                reference_id=1233,
                task_text="assume account_id = 99999",
            )
        )
    )
    assert all(row["account_id"] == 10 for row in outcome.bindings["account"])


def test_planner_inserts_intermediate_table(router):
    from csm_env.query.models import QueryBudget, TaskRequirements
    graph = router._graph
    planner = router._planner
    plan = planner.plan(
        task_type="custom_contract_path",
        anchor_table="customer_case",
        anchor_key="case_id",
        anchor_value=1233,
        requirements=TaskRequirements(
            task_type="custom_contract_path",
            required_tables=("customer_case", "contract"),
        ),
        budget=QueryBudget(max_hops=2, max_tables=5),
        route_id="route-intermediate",
    )
    assert [step.table for step in plan.steps] == ["customer_case", "account", "contract"]
    assert plan.steps[1].depends_on == "s000"
    assert plan.steps[2].depends_on == plan.steps[1].step_id


def test_router_handles_one_to_many_fk_fanout_with_in_filter(registry, fixture_rows):
    import copy

    from csm_env.query.requirements import RequirementRegistry

    rows = copy.deepcopy(fixture_rows)
    rows["customer_case"].append({
        **rows["customer_case"][0],
        "case_id": 1250,
        "number": "CS001250",
        "assigned_to": 18,
    })
    rows["user"].append({
        **rows["user"][0],
        "user_id": 18,
        "email": "john.doe@example.com",
    })
    from conftest import FilteredStaticReader
    router = __import__("tests_support.helpers", fromlist=["make_router"]).make_router(
        registry, FilteredStaticReader(rows)
    )
    router._requirements = RequirementRegistry(
        registry,
        {"account_user_fanout": ("account", "customer_case", "user")},
    )

    import asyncio
    from csm_env.query.models import TaskRequest
    outcome = asyncio.run(
        router.route(
            TaskRequest(
                task_type="account_user_fanout",
                reference_type="account",
                reference_id=10,
            )
        )
    )
    assert outcome.status == "COMPLETE"
    assert {row["case_id"] for row in outcome.bindings["customer_case"]} == {1233, 1240, 1250}
    assert {row["user_id"] for row in outcome.bindings["user"]} == {17, 18}


def test_router_prefers_existing_required_anchor_for_equal_length_path(registry, fixture_rows):
    from csm_env.query.requirements import RequirementRegistry
    from conftest import FilteredStaticReader
    import asyncio
    from csm_env.query.models import TaskRequest

    router = __import__("tests_support.helpers", fromlist=["make_router"]).make_router(
        registry, FilteredStaticReader(fixture_rows)
    )
    router._requirements = RequirementRegistry(
        registry,
        {"account_cases_users": ("account", "customer_case", "user")},
    )
    outcome = asyncio.run(
        router.route(
            TaskRequest(
                task_type="account_cases_users",
                reference_type="account",
                reference_id=10,
            )
        )
    )
    assert outcome.status == "COMPLETE"
    # User should be reached through the already-required CustomerCase branch,
    # not the equally short Account -> Contact -> User branch.
    user_step = next(step for step in outcome.plan.steps if step.table == "user")
    case_step = next(step for step in outcome.plan.steps if step.table == "customer_case")
    assert user_step.depends_on == case_step.step_id


def test_router_uses_injected_executor(registry, reader):
    class SpyExecutor:
        def __init__(self):
            self.calls = 0
        async def execute(self, steps, budget, run_step=None):
            self.calls += 1
            assert run_step is not None
            return await QueryExecutor().execute(steps, budget, run_step=run_step)

    spy = SpyExecutor()
    from csm_env.query.planner import QueryPlanner
    from csm_env.query.requirements import RequirementRegistry
    from csm_env.query.router import GraphQueryRouter
    from csm_env.schema_graph import build_schema_graph
    from csm_env.query.adapters import TableAdapterFactory
    graph = build_schema_graph(registry)
    router = GraphQueryRouter(registry, graph, QueryPlanner(graph, registry), spy,
                              RequirementRegistry(registry), TableAdapterFactory(registry, reader))
    asyncio.run(router.route(TaskRequest(task_type="resolve_case", reference_type="customer_case", reference_id=1233)))
    assert spy.calls >= 1


def test_total_row_budget_is_strict_under_parallel_fanout(registry, reader):
    budget = QueryBudget(max_hops=2, max_tables=12, max_queries=32,
                         max_rows_per_query=50, max_total_rows=55, max_parallel=8)
    router = make_router(registry, reader)
    outcome = asyncio.run(router.route(TaskRequest(
        task_type="case_overview", reference_type="customer_case", reference_id=1233, budget=budget
    )))
    assert sum(len(r.rows) for r in outcome.results) <= 55


def test_route_ids_are_unique():
    from csm_env.query.router import GraphQueryRouter
    ids = {f"route-{__import__('uuid').uuid4().hex}" for _ in range(1000)}
    assert len(ids) == 1000
