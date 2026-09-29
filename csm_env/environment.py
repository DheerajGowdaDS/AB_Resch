from __future__ import annotations

from typing import Any, Dict, List, Optional

from .query.adapters import TableAdapterFactory
from .query.executor import QueryExecutor
from .query.models import QueryBudget, QueryResult, QuerySpec, QueryStep, StepStatus, TaskRequest
from .query.planner import QueryPlanner
from .query.requirements import RequirementRegistry
from .query.router import AnchorResolver, GraphQueryRouter
from .representation import to_dict, to_graph_text, to_json, to_text, verify_fidelity
from .schema_graph import SchemaGraph, build_schema_graph
from .schema_spec.registry import SchemaRegistry
from .state.builder import StateBuilder
from .state.consistency import check_state
from .state.models import GroundedState
from .transport.reader import SQLReader
from .verification.oracle import DirectSqlOracle


class GGQREnvironment:
    """Additive service exposing the GGQR pipeline over one SQL reader.

    Composes the schema registry, structural graph, deterministic router,
    state builder, representations, and the independent verification branch.
    Existing classes (CSMEnvironmentRepresentation / CSMEnvironmentAPI) are
    untouched; this is the new capability surface from the approved plan.
    """

    def __init__(
        self,
        reader: SQLReader,
        *,
        database_id: str = "unknown",
        registry: Optional[SchemaRegistry] = None,
        budget: Optional[QueryBudget] = None,
        requirement_tables: Optional[dict] = None,
        requirement_registry: Optional[RequirementRegistry] = None,
    ) -> None:
        self.registry = registry or SchemaRegistry.from_static()
        self.graph: SchemaGraph = build_schema_graph(self.registry)
        self.budget = budget or QueryBudget()
        self.database_id = database_id

        self.adapter_factory = TableAdapterFactory(self.registry, reader)
        # A fully-built registry (e.g. from a manifest with attribute/relation/
        # intent blocks, via ``RequirementRegistry.from_json``) is used as-is;
        # otherwise fall back to the tables-only constructor so the built-in
        # task-type defaults still apply (P0/A3).
        self.requirements = (
            requirement_registry
            if requirement_registry is not None
            else RequirementRegistry(self.registry, requirement_tables)
        )
        self.planner = QueryPlanner(self.graph, self.registry)
        self._reader = reader

        # Router owns the request-specific runner because it must enforce
        # strict per-route query/row budgets and bindings. The executor remains
        # the injected concurrency component.
        self.executor = QueryExecutor()
        self.router = GraphQueryRouter(
            registry=self.registry,
            graph=self.graph,
            planner=self.planner,
            executor=self.executor,
            requirements=self.requirements,
            adapter_factory=self.adapter_factory,
            budget=self.budget,
        )
        self.state_builder = StateBuilder(self.registry, database_id=database_id)
        self.oracle = DirectSqlOracle(reader, self.registry.manifest, database_id=database_id)

    # ---------- routing + state ----------

    async def route(self, request: TaskRequest):
        return await self.router.route(request)

    async def build_state(self, request: TaskRequest) -> GroundedState:
        outcome = await self.router.route(request)
        return self.state_builder.build(outcome)

    async def build_state_for(
        self,
        task_type: str,
        reference_id: Any,
        reference_type: Optional[str] = None,
        budget: Optional[QueryBudget] = None,
        *,
        required_attributes: Optional[Tuple[str, ...]] = None,
        required_relations: Optional[Tuple[Tuple[str, str], ...]] = None,
        intent: Optional[Tuple[str, ...]] = None,
        strategy: Optional[str] = None,
    ) -> GroundedState:
        """Build the grounded state for one task/anchor pair.

        ``strategy`` selects the retrieval strategy of blueprint section 2:
        ``RouteStrategy.BROAD`` (eager, historical) or ``RouteStrategy.FRONTIER``
        (incremental, stops once ``Coverage >= tau`` so ``lambda_q`` is
        optimized). ``None`` keeps the environment default.
        """
        return await self.build_state(
            TaskRequest(
                task_type=task_type,
                reference_id=reference_id,
                reference_type=reference_type,
                budget=budget,
                strategy=strategy,
                intent=tuple(intent) if intent else (),
                required_attributes=tuple(required_attributes) if required_attributes else (),
                required_relations=(
                    tuple((str(a), str(b)) for a, b in required_relations)
                    if required_relations
                    else ()
                ),
            )
        )

    # ---------- representation ----------

    @staticmethod
    def represent(state: GroundedState, *, max_chars: int = 4000) -> Dict[str, str]:
        # ``max_chars`` (P1/N4) lets a caller align the renderer's text ceiling
        # with the selector's token budget so the two ceilings stay coherent.
        # The default 4000 preserves the historical broad-state behaviour.
        return {
            "json": to_json(state),
            "graph": to_graph_text(state),
            "text": to_text(state, max_chars=max_chars),
        }

    @staticmethod
    def fidelity(state: GroundedState):
        return verify_fidelity(state)

    # ---------- verification ----------

    async def verify_case_equivalence(self, case_id: Any) -> Dict[str, Any]:
        """Graph-derived state vs direct-SQL facts for one case (V5-style)."""
        state = await self.build_state_for("resolve_case", case_id, "customer_case")
        failures: List[str] = []

        def _state_fact(column: str):
            for fact in state.facts:
                if (
                    fact.provenance.table == "customer_case"
                    and fact.provenance.row_pk == str(case_id)
                    and fact.provenance.column == column
                ):
                    return fact.value
            return _MISSING

        row = await self.oracle.get_row("customer_case", case_id)
        if row is None:
            failures.append(f"oracle has no customer_case row {case_id!r}")
        else:
            for column, value in row.items():
                state_value = _state_fact(column)
                if state_value is _MISSING:
                    failures.append(f"state missing fact customer_case.{column}")
                elif str(state_value) != str(value):
                    failures.append(
                        f"customer_case.{column}: state {state_value!r} != DB {value!r}"
                    )
        return {
            "state_status": state.status,
            "equivalent": not failures,
            "failures": failures,
            "consistency": check_state(state, self.registry),
        }


def _now() -> float:
    import time

    return time.perf_counter()


_MISSING = object()
