from __future__ import annotations

from csm_env.query.adapters import TableAdapterFactory
from csm_env.query.executor import QueryExecutor
from csm_env.query.models import QueryBudget
from csm_env.query.planner import QueryPlanner
from csm_env.query.requirements import RequirementRegistry
from csm_env.query.router import GraphQueryRouter
from csm_env.schema_graph import build_schema_graph
from csm_env.transport.reader import SQLReader


def make_router(registry, reader: SQLReader, budget: QueryBudget | None = None) -> GraphQueryRouter:
    graph = build_schema_graph(registry)
    planner = QueryPlanner(graph, registry)
    requirements = RequirementRegistry(registry)
    adapters = TableAdapterFactory(registry, reader)

    async def run_step(step, budget):
        started = __import__("time").perf_counter()
        try:
            adapter = adapters.adapter_for(step.table)
            from csm_env.query.models import QuerySpec

            rows = await adapter.query(
                QuerySpec(table=step.table, filters=step.filters, columns=step.columns, limit=step.limit)
            )
            from csm_env.query.models import QueryResult, StepStatus

            return QueryResult(
                step_id=step.step_id,
                table=step.table,
                rows=rows,
                status=StepStatus.OK if rows else StepStatus.EMPTY,
                latency_ms=(__import__("time").perf_counter() - started) * 1000.0,
                edge_id=step.edge_id,
            )
        except Exception as exc:  # noqa: BLE001
            from csm_env.query.models import QueryResult, StepStatus

            return QueryResult(
                step_id=step.step_id,
                table=step.table,
                status=StepStatus.ERROR,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(__import__("time").perf_counter() - started) * 1000.0,
                edge_id=step.edge_id,
            )

    executor = QueryExecutor(run_step)
    return GraphQueryRouter(
        registry=registry,
        graph=graph,
        planner=planner,
        executor=executor,
        requirements=requirements,
        adapter_factory=adapters,
        budget=budget,
    )
