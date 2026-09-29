from __future__ import annotations

import time
import uuid
import asyncio
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..schema_graph.graph import SchemaGraph
from .bindings import UnresolvedBinding
from .cost import (
    DEFAULT_COST_WEIGHTS,
    CostWeights,
    cheapest_covering_prefix,
    compute_cost,
    estimate_tokens_of_results,
)
from .coverage import coverage_report
from .executor import QueryExecutor
from .models import (
    Filter,
    QueryBudget,
    QueryPlan,
    QueryResult,
    QuerySpec,
    QueryStep,
    RouteOutcome,
    RouteStatus,
    RouteStrategy,
    StepStatus,
    TaskRequest,
    TaskRequirements,
)
from .planner import QueryPlanner
from .requirements import RequirementRegistry

# Deterministic alias table per the approved Phase 0 policy record.
_EXTRA_ALIASES: Dict[str, Tuple[str, ...]] = {
    "customer_case": ("case", "cases"),
    "user_group": ("group", "groups"),
    "installed_product": ("installedproduct",),
}


class AnchorResolver:
    """Resolve a reference to exactly one anchor table; never guess."""

    def __init__(self, registry, graph: SchemaGraph) -> None:
        self._registry = registry
        self._graph = graph

    def resolve(self, reference_type: Optional[str], reference_id: Any) -> Tuple[str, str]:
        if reference_id is None or (isinstance(reference_id, str) and not reference_id.strip()):
            raise ValueError("reference_id must be a non-empty value")
        if not reference_type:
            raise ValueError(
                "reference_type is required; routing never guesses the anchor table"
            )

        probe = reference_type.strip()
        candidates: Set[str] = set()
        for table in self._graph.nodes():
            lowered = probe.lower()
            if lowered in (table.table.lower(), table.node_type.lower()) or lowered in _EXTRA_ALIASES.get(
                table.table, ()
            ):
                candidates.add(table.table)

        if not candidates:
            raise KeyError(f"Unknown reference type: {reference_type!r}")
        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous reference type {reference_type!r}: matches {sorted(candidates)}"
            )

        table_name = candidates.pop()
        anchor = self._registry.require_table(table_name)
        return anchor.table, anchor.primary_key


class GraphQueryRouter:
    """Deterministic Graph-Guided Query Routing (GGQR).

    Pipeline: validate -> resolve anchor -> plan over structural edges ->
    execute dependency-ordered waves -> propagate identifiers only from
    returned rows (ID_next = DB(row, FK)).
    """

    def __init__(
        self,
        registry,
        graph: SchemaGraph,
        planner: QueryPlanner,
        executor: QueryExecutor,
        requirements: RequirementRegistry,
        adapter_factory,
        budget: Optional[QueryBudget] = None,
        strategy: Optional[str] = None,
    ) -> None:
        self._registry = registry
        self._graph = graph
        self._planner = planner
        self._executor = executor
        self._requirements = requirements
        self._adapters = adapter_factory
        self._budget = budget or QueryBudget()
        self._resolver = AnchorResolver(registry, graph)
        resolved_strategy = strategy or RouteStrategy.DEFAULT
        if resolved_strategy not in RouteStrategy.ALL:
            raise ValueError(
                f"Unknown route strategy {resolved_strategy!r}; "
                f"expected one of {list(RouteStrategy.ALL)}"
            )
        self._strategy = resolved_strategy
        self._cost_weights = DEFAULT_COST_WEIGHTS

    async def route(self, request: TaskRequest) -> RouteOutcome:
        route_id = f"route-{uuid.uuid4().hex}"
        budget = request.budget or self._budget
        strategy = request.strategy or self._strategy
        if strategy not in RouteStrategy.ALL:
            raise ValueError(
                f"Unknown route strategy {strategy!r}; "
                f"expected one of {list(RouteStrategy.ALL)}"
            )

        # State Model 2.0 (blueprint section 3): request-level task
        # normalization (attributes, intent) refines the registered task-type
        # requirements instead of bypassing them.
        requirement = self._requirements.requirements_for(
            request.task_type,
            attribute_overrides=tuple(getattr(request, "required_attributes", ()) or ()),
            intent_overrides=tuple(getattr(request, "intent", ()) or ()),
            relation_overrides=tuple(getattr(request, "required_relations", ()) or ()),
        )
        anchor_table, anchor_key = self._resolver.resolve(
            request.reference_type, request.reference_id
        )

        plan = self._planner.plan(
            task_type=request.task_type,
            anchor_table=anchor_table,
            anchor_key=anchor_key,
            anchor_value=request.reference_id,
            requirements=requirement,
            budget=budget,
            route_id=route_id,
        )

        bindings: Dict[str, List[Dict[str, Any]]] = {}
        unresolved: List[UnresolvedBinding] = []
        total_rows = 0
        queries_used = 0
        budget_lock = asyncio.Lock()

        async def run_step(step: QueryStep, wave_budget: QueryBudget) -> QueryResult:
            nonlocal total_rows, queries_used
            started = time.perf_counter()
            async with budget_lock:
                if total_rows >= wave_budget.max_total_rows:
                    return QueryResult(
                        step_id=step.step_id, table=step.table,
                        status=StepStatus.SKIPPED_BUDGET,
                        error="total row budget exhausted", edge_id=step.edge_id,
                    )
                if queries_used >= wave_budget.max_queries:
                    return QueryResult(
                        step_id=step.step_id, table=step.table,
                        status=StepStatus.SKIPPED_BUDGET,
                        error="query budget exhausted", edge_id=step.edge_id,
                    )
                queries_used += 1
                remaining_rows = wave_budget.max_total_rows - total_rows
                reserved_limit = min(wave_budget.max_rows_per_query, remaining_rows)
                if reserved_limit <= 0:
                    return QueryResult(
                        step_id=step.step_id, table=step.table,
                        status=StepStatus.SKIPPED_BUDGET,
                        error="total row budget exhausted", edge_id=step.edge_id,
                    )
                # Reserve the maximum number of rows this in-flight query may
                # contribute. This makes max_total_rows strict under concurrency.
                total_rows += reserved_limit
            try:
                adapter = self._adapters.adapter_for(step.table)
                rows = await adapter.query(
                    QuerySpec(
                        table=step.table,
                        filters=step.filters,
                        columns=step.columns,
                        limit=min(step.limit, reserved_limit),
                    )
                )
                bounded = rows[:reserved_limit]
                async with budget_lock:
                    total_rows -= max(0, reserved_limit - len(bounded))
                return QueryResult(
                    step_id=step.step_id,
                    table=step.table,
                    rows=bounded,
                    status=StepStatus.OK if bounded else StepStatus.EMPTY,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    edge_id=step.edge_id,
                )
            except Exception as exc:  # noqa: BLE001 - recorded on the result
                async with budget_lock:
                    total_rows -= reserved_limit
                return QueryResult(
                    step_id=step.step_id,
                    table=step.table,
                    status=StepStatus.ERROR,
                    error=f"{type(exc).__name__}: {exc}",
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    edge_id=step.edge_id,
                )

        executor = self._executor

        # Wave 0: anchor.
        anchor_step = plan.steps[0]
        anchor_result = (await executor.execute([anchor_step], budget, run_step=run_step))[0]
        if anchor_result.status == StepStatus.ERROR:
            return self._finish(
                plan=plan,
                results=[anchor_result],
                bindings=bindings,
                unresolved=unresolved,
                requirement=requirement,
                budget=budget,
                status=RouteStatus.FAILED,
                strategy=strategy,
            )
        if anchor_result.status != StepStatus.OK:
            unresolved.append(
                UnresolvedBinding(
                    table=anchor_table,
                    source_column=anchor_key,
                    reason="no_rows",
                    source_value=request.reference_id,
                )
            )
            return self._finish(
                plan=plan,
                results=[anchor_result],
                bindings=bindings,
                unresolved=unresolved,
                requirement=requirement,
                budget=budget,
                status=RouteStatus.UNRESOLVED,
                strategy=strategy,
            )

        bindings[anchor_table] = list(anchor_result.rows)
        self._collect_null_fks(anchor_table, anchor_result.rows, unresolved)
        results: List[QueryResult] = [anchor_result]

        # State Model 2.0, sections 2/10/13. Under FRONTIER the router retrieves
        # one step at a time, cheapest and most coverage-additive first, and
        # stops as soon as Coverage(S, F_T) >= tau. Because cost is monotone in
        # queries/rows/tokens, that first covering prefix *is*
        # S* = argmin Cost(S) s.t. coverage -- which is what makes the
        # lambda_q * |queries| term meaningful rather than decorative.
        if strategy == RouteStrategy.FRONTIER:
            return await self._execute_frontier(
                plan=plan,
                budget=budget,
                requirement=requirement,
                run_step=run_step,
                executor=executor,
                bindings=bindings,
                unresolved=unresolved,
                results=results,
                anchor_table=anchor_table,
            )

        # Later waves: execute all pending steps grouped by hop.
        step_table = {step.step_id: step.table for step in plan.steps}
        remaining = list(plan.steps[1:])
        while remaining:
            current_hop = remaining[0].hop
            wave = [step for step in remaining if step.hop == current_hop]
            remaining = [step for step in remaining if step.hop != current_hop]

            # Same-wave bindings: an earlier step in this wave may already
            # have produced bindings this wave depends on (processed in
            # step-ID order). Steps whose dependency is still pending are
            # deferred to the next loop pass.
            executable: List[QueryStep] = []
            deferred: List[QueryStep] = []
            wave_results: List[QueryResult] = []

            def _prepare(step: QueryStep) -> None:
                dependency_rows = bindings.get(step_table.get(step.depends_on, ""), [])
                if not dependency_rows:
                    unresolved.append(
                        UnresolvedBinding(
                            table=step.table,
                            source_column=step.binding_column,
                            reason="null_fk",
                            source_table=step.depends_on,
                        )
                    )
                    return
                values = self._binding_values(dependency_rows, step)
                if not values:
                    unresolved.append(
                        UnresolvedBinding(
                            table=step.table,
                            source_column=step.binding_column,
                            reason="null_fk",
                            source_table=step.depends_on,
                        )
                    )
                    return
                filter_column = step.filter_column or self._registry.require_table(step.table).primary_key
                executable.append(
                    QueryStep(
                        step_id=step.step_id,
                        table=step.table,
                        # One set-valued filter is rendered as IN (...), preserving
                        # one-to-many FK fan-out semantics without contradictory
                        # `column=a AND column=b` predicates.
                        filters=(Filter(column=filter_column, value=tuple(values)),),
                        columns=step.columns,
                        limit=budget.max_rows_per_query,
                        depends_on=step.depends_on,
                        edge_id=step.edge_id,
                        direction=step.direction,
                        hop=step.hop,
                        filter_column=step.filter_column,
                        binding_column=step.binding_column,
                    )
                )

            progress = True
            pending = list(wave)
            while progress and pending:
                progress = False
                still_pending: List[QueryStep] = []
                for step in pending:
                    dep_table = step_table.get(step.depends_on, "")
                    if dep_table in bindings:
                        _prepare(step)
                        progress = True
                    else:
                        still_pending.append(step)
                if still_pending == pending:
                    pending = still_pending
                    break
                pending = still_pending

            # Anything still pending has no bindings: skip with diagnostics.
            for step in pending:
                results.append(
                    QueryResult(
                        step_id=step.step_id,
                        table=step.table,
                        status=StepStatus.SKIPPED_BUDGET,
                        error="dependency produced no bindings",
                        edge_id=step.edge_id,
                    )
                )

            if not executable:
                continue

            executed = await executor.execute(executable, budget, run_step=run_step)
            for result in executed:
                wave_results.append(result)
                results.append(result)
                if result.status == StepStatus.OK and result.rows:
                    bindings.setdefault(result.table, []).extend(result.rows)
                    self._collect_null_fks(result.table, result.rows, unresolved)
                elif result.status == StepStatus.EMPTY:
                    unresolved.append(
                        UnresolvedBinding(
                            table=result.table,
                            source_column=None,
                            reason="no_rows",
                            source_table=result.step_id,
                        )
                    )

        status = self._status(results, bindings, unresolved, requirement, budget)
        return self._finish(
            plan=plan,
            results=results,
            bindings=bindings,
            unresolved=unresolved,
            requirement=requirement,
            budget=budget,
            status=status,
            strategy=strategy,
        )

    # ------------------------------------------------------------------
    # Frontier strategy (blueprint sections 2, 10, 13)
    # ------------------------------------------------------------------

    def _prepare_step(
        self,
        step: QueryStep,
        step_table: Dict[str, str],
        bindings: Dict[str, List[Dict[str, Any]]],
        unresolved: List[UnresolvedBinding],
        budget: QueryBudget,
    ) -> Optional[QueryStep]:
        """Resolve one planned step against the rows retrieved so far.

        Mirrors the broad path's preparation: identifiers are propagated only
        from returned rows (``ID_next = DB(row, FK)``) and rendered as one
        set-valued filter so one-to-many fan-out stays a single ``IN (...)``.

        Returns:
            The executable step, or ``None`` when the dependency produced no
            usable binding. The failure is recorded as unresolved -- a binding
            is never guessed.
        """
        dependency_rows = bindings.get(step_table.get(step.depends_on or "", ""), [])
        if not dependency_rows:
            unresolved.append(
                UnresolvedBinding(
                    table=step.table,
                    source_column=step.binding_column,
                    reason="null_fk",
                    source_table=step.depends_on,
                )
            )
            return None

        values = self._binding_values(dependency_rows, step)
        if not values:
            unresolved.append(
                UnresolvedBinding(
                    table=step.table,
                    source_column=step.binding_column,
                    reason="null_fk",
                    source_table=step.depends_on,
                )
            )
            return None

        filter_column = step.filter_column or self._registry.require_table(
            step.table
        ).primary_key
        return QueryStep(
            step_id=step.step_id,
            table=step.table,
            filters=(Filter(column=filter_column, value=tuple(values)),),
            columns=step.columns,
            limit=budget.max_rows_per_query,
            depends_on=step.depends_on,
            edge_id=step.edge_id,
            direction=step.direction,
            hop=step.hop,
            filter_column=step.filter_column,
            binding_column=step.binding_column,
            priority=step.priority,
        )

    async def _execute_frontier(
        self,
        *,
        plan: QueryPlan,
        budget: QueryBudget,
        requirement: TaskRequirements,
        run_step: Callable[..., Any],
        executor: QueryExecutor,
        bindings: Dict[str, List[Dict[str, Any]]],
        unresolved: List[UnresolvedBinding],
        results: List[QueryResult],
        anchor_table: str,
    ) -> RouteOutcome:
        """Retrieve incrementally and stop at the first covering prefix.

        This is the literal ``S* = argmin Cost(S)`` subject to
        ``Coverage(S, F_T) >= tau`` from blueprint section 2, realized as an
        anytime loop (section 10):

        1. measure coverage and cost over the rows retrieved so far;
        2. stop when coverage meets ``tau``;
        3. otherwise expand the *best next relation*: the executable step whose
           table can add an uncovered fact, ordered by the section-5 edge
           weight, then hop, then step id (so the outcome is deterministic);
        4. retrieve it, re-measure, re-check.

        Because ``Cost(S)`` is non-decreasing in queries, rows and tokens, the
        first covering prefix is always the cheapest one, so the ``argmin`` is
        provably the stopping point rather than an approximation. Every prefix
        is recorded in ``diagnostics['cost_prefixes']`` for later audit.

        Bounded by the plan, the query/row budget (``run_step`` refuses with
        ``SKIPPED_BUDGET``), and ``budget.max_expansion_rounds`` consecutive
        steps that add no coverage.
        """
        step_table = {step.step_id: step.table for step in plan.steps}
        remaining: List[QueryStep] = list(plan.steps[1:])
        tau = float(budget.coverage_threshold)
        weights = self._cost_weights
        planned_tables = {step.table for step in plan.steps}
        frontier_order: List[str] = []

        def rows_by_table() -> Dict[str, List[Dict[str, Any]]]:
            collected: Dict[str, List[Dict[str, Any]]] = {}
            for result in results:
                if result.status == StepStatus.OK and result.rows:
                    collected.setdefault(result.table, []).extend(result.rows)
            return collected

        def measure():
            report = coverage_report(
                requirement=requirement,
                registry=self._registry,
                rows_by_table=rows_by_table(),
                anchor=(anchor_table, plan.anchor_value),
                planned_tables=planned_tables,
            )
            cost = compute_cost(
                rows=sum(len(result.rows) for result in results),
                tokens=estimate_tokens_of_results(results),
                queries=sum(
                    1
                    for result in results
                    if result.status != StepStatus.SKIPPED_BUDGET
                ),
                weights=weights,
            )
            return report, cost

        report, cost = measure()
        prefixes: List[Tuple[int, Any, bool]] = [
            (len(results), cost, report.meets(tau))
        ]
        stalled = 0

        while remaining and not report.meets(tau):
            executable = [
                step
                for step in remaining
                if step.depends_on is None
                or step_table.get(step.depends_on, "") in bindings
            ]
            if not executable:
                # Every remaining step depends on a table we never reached:
                # this frontier is exhausted, so record it and stop.
                break

            uncovered = set(report.uncovered)

            def rank(step: QueryStep) -> Tuple[int, float, int, str]:
                adds_coverage = 0 if any(step.table in fact for fact in uncovered) else 1
                return (adds_coverage, step.priority, step.hop, step.step_id)

            chosen = min(executable, key=rank)
            prepared = self._prepare_step(
                chosen, step_table, bindings, unresolved, budget
            )

            if prepared is None:
                results.append(
                    QueryResult(
                        step_id=chosen.step_id,
                        table=chosen.table,
                        status=StepStatus.SKIPPED_BUDGET,
                        error="dependency produced no bindings",
                        edge_id=chosen.edge_id,
                    )
                )
            else:
                executed = await executor.execute(
                    [prepared], budget, run_step=run_step
                )
                result = executed[0]
                results.append(result)
                frontier_order.append(chosen.step_id)
                if result.status == StepStatus.OK and result.rows:
                    bindings.setdefault(result.table, []).extend(result.rows)
                    self._collect_null_fks(result.table, result.rows, unresolved)
                elif result.status == StepStatus.EMPTY:
                    unresolved.append(
                        UnresolvedBinding(
                            table=result.table,
                            source_column=None,
                            reason="no_rows",
                            source_table=result.step_id,
                        )
                    )

            remaining = [step for step in remaining if step.step_id != chosen.step_id]

            previous_covered = report.covered
            report, cost = measure()
            prefixes.append((len(results), cost, report.meets(tau)))

            if report.covered == previous_covered:
                stalled += 1
                if stalled >= budget.max_expansion_rounds:
                    break
            else:
                stalled = 0

        # argmin Cost(S) over the prefixes that met coverage.
        chosen_prefix = cheapest_covering_prefix(prefixes)
        if chosen_prefix is not None:
            index = chosen_prefix[0]
            if index != len(results):
                results = results[:index]
                bindings = {}
                for result in results:
                    if result.status == StepStatus.OK and result.rows:
                        bindings.setdefault(result.table, []).extend(result.rows)
            report, cost = measure()

        if any(result.status == StepStatus.ERROR for result in results):
            status = RouteStatus.FAILED
        elif not report.meets(tau):
            # Coverage below tau is a real shortfall: fail visibly rather than
            # hand over a state the objective does not accept.
            status = RouteStatus.UNRESOLVED
        else:
            status = self._status(results, bindings, unresolved, requirement, budget)

        return self._finish(
            plan=plan,
            results=results,
            bindings=bindings,
            unresolved=unresolved,
            requirement=requirement,
            budget=budget,
            status=status,
            extra={
                "coverage": report.as_dict(),
                "coverage_ratio": report.ratio,
                "coverage_threshold": tau,
                "meets_coverage": report.meets(tau),
                "cost": cost.as_dict(),
                "cost_prefixes": [
                    {
                        "results": index,
                        "total": prefix_cost.total,
                        "rows": prefix_cost.rows,
                        "tokens": prefix_cost.tokens,
                        "queries": prefix_cost.queries,
                        "meets_coverage": meets,
                    }
                    for index, prefix_cost, meets in prefixes
                ],
                "frontier_order": frontier_order,
            },
            strategy=RouteStrategy.FRONTIER,
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _collect_null_fks(
        self, table: str, rows: List[Dict[str, Any]], unresolved: List[UnresolvedBinding]
    ) -> None:
        """V7: NULL FK values observed in returned rows are unresolved, never guessed."""
        for fk in self._graph.outgoing(table):
            for row in rows:
                if fk.source_column in row and row[fk.source_column] is None:
                    unresolved.append(
                        UnresolvedBinding(
                            table=fk.target_table,
                            source_column=fk.source_column,
                            reason="null_fk",
                            source_table=table,
                            source_value=None,
                        )
                    )

    def _binding_values(self, rows: List[Dict[str, Any]], step: QueryStep) -> List[Any]:
        if step.binding_column is None:
            return []
        values: List[Any] = []
        seen: Set[Any] = set()
        for row in rows:
            value = row.get(step.binding_column, _MISSING)
            if value is _MISSING or value is None:
                continue
            if value in seen:
                continue
            seen.add(value)
            values.append(value)
        return values

    def _status(
        self,
        results: List[QueryResult],
        bindings: Dict[str, List[Dict[str, Any]]],
        unresolved: List[UnresolvedBinding],
        requirement: TaskRequirements,
        budget: QueryBudget,
    ) -> str:
        if any(result.status == StepStatus.ERROR for result in results):
            return RouteStatus.FAILED
        observed = {
            result.table for result in results if result.status == StepStatus.OK and result.rows
        }
        missing_required = [table for table in requirement.required_tables if table not in observed]
        # A NULL FK on one of many source rows does not make a required
        # destination unresolved when another source row yielded a valid
        # binding and the destination was successfully observed. Only an
        # actually missing required table, no-row result, or explicit required
        # coverage gap should downgrade the route.
        required_unresolved = [
            u for u in unresolved
            if u.table in requirement.required_tables
            and u.reason in {"no_rows", "required_table_unobserved"}
        ]
        if missing_required or required_unresolved:
            return RouteStatus.UNRESOLVED
        if len(results) >= budget.max_queries:
            return RouteStatus.BUDGET_EXHAUSTED
        return RouteStatus.COMPLETE

    def _finish(
        self,
        plan: QueryPlan,
        results: List[QueryResult],
        bindings: Dict[str, List[Dict[str, Any]]],
        unresolved: List[UnresolvedBinding],
        requirement: TaskRequirements,
        budget: QueryBudget,
        status: str,
        extra: Optional[Dict[str, Any]] = None,
        strategy: Optional[str] = None,
    ) -> RouteOutcome:
        ordered = sorted(results, key=lambda r: r.step_id)
        observed = {
            result.table for result in ordered if result.status == StepStatus.OK and result.rows
        }
        planned = {step.table for step in plan.steps}
        errored = {result.table for result in ordered if result.status == StepStatus.ERROR}
        diagnostics: Dict[str, Any] = {
            "queries_executed": len(ordered),
            "total_rows": sum(len(result.rows) for result in ordered),
            "tables_activated": len(observed),
            "required_tables": list(requirement.required_tables),
            # Phase 1.3 diagnostics. `structurally_valid` answers "could this
            # route satisfy the requirements?", while `uncovered_required`
            # answers "which required tables happened to be empty?". Keeping
            # the two apart is what stops a legitimately empty table from
            # being reported as a failed intervention.
            "structurally_valid": not errored and planned.issuperset(
                requirement.required_tables
            ),
            "uncovered_required": sorted(
                table for table in requirement.required_tables if table not in observed
            ),
            "errored_tables": sorted(errored),
            # State Model 2.0 (sections 2, 10, 12): the retrieval strategy
            # actually used, how many queries the plan contained versus how many
            # were executed, and the Cost(S)/coverage budget terms. Under the
            # FRONTIER strategy `queries_saved` is the physical evidence that
            # `lambda_q * |queries|` was optimized rather than merely claimed.
            "strategy": strategy or self._strategy,
            "queries_planned": len(plan.steps),
            "queries_saved": max(0, len(plan.steps) - len(ordered)),
            "cost_weights": self._cost_weights.as_dict(),
            "budget": {
                "max_hops": budget.max_hops,
                "max_tables": budget.max_tables,
                "max_queries": budget.max_queries,
                "max_rows_per_query": budget.max_rows_per_query,
                "max_total_rows": budget.max_total_rows,
                "max_parallel": budget.max_parallel,
                "max_tokens": budget.max_tokens,
                "coverage_threshold": budget.coverage_threshold,
            },
        }
        if extra:
            diagnostics.update(extra)
        return RouteOutcome(
            status=status,
            plan=plan,
            results=ordered,
            bindings=bindings,
            unresolved=unresolved,
            diagnostics=diagnostics,
        )


_MISSING = object()
