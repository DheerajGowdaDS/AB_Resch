from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Route / step statuses
# ---------------------------------------------------------------------------


class RouteStatus:
    COMPLETE = "COMPLETE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    UNRESOLVED = "UNRESOLVED"
    FAILED = "FAILED"

    ALL = (COMPLETE, BUDGET_EXHAUSTED, UNRESOLVED, FAILED)


class StepStatus:
    OK = "ok"
    EMPTY = "empty"
    ERROR = "error"
    SKIPPED_BUDGET = "skipped_budget"


class RouteStrategy:
    """How much of the plan is retrieved (State Model 2.0, blueprint section 2).

    ``BROAD`` is the historical eager strategy: every planned step is executed,
    wave by wave, and the caller prunes afterwards. ``FRONTIER`` retrieves
    *incrementally* and stops as soon as the task's required facts are covered,
    so ``Cost(S) = lambda_r*rows + lambda_t*tokens + lambda_q*queries`` is
    minimized over the executed prefix -- which is what makes ``lambda_q``
    optimizable at all (blueprint sections 2, 10, 13).
    """

    BROAD = "broad"
    FRONTIER = "frontier"

    ALL = (BROAD, FRONTIER)

    DEFAULT = BROAD


# ---------------------------------------------------------------------------
# Filters and query specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Filter:
    column: str
    value: Any


@dataclass(frozen=True)
class QueryBudget:
    max_hops: int = 2
    max_tables: int = 12
    max_queries: int = 32
    max_rows_per_query: int = 50
    max_total_rows: int = 500
    max_parallel: int = 8
    #: Soft token ceiling for the assembled state (State Model 2.0, blueprint
    #: section 2/12). Enforced by pruning before assembly, not by truncating
    #: rendered text after the fact.
    max_tokens: int = 20000
    #: ``tau`` of the coverage gate ``Coverage(S, F_T) >= tau`` (blueprint
    #: section 2/12). The frontier strategy stops retrieving once this is met;
    #: below it a frontier route reports ``RouteStatus.UNRESOLVED`` rather than
    #: claiming success.
    coverage_threshold: float = 1.0
    #: Maximum retrieval depth explored when a frontier route must escalate
    #: beyond the plan's own hop budget (blueprint section 10).
    max_expansion_rounds: int = 4


@dataclass(frozen=True)
class QuerySpec:
    """Structured query description. Never contains SQL text."""

    table: str
    filters: Tuple[Filter, ...] = ()
    columns: Optional[Tuple[str, ...]] = None
    limit: int = 50


@dataclass(frozen=True)
class TaskRequest:
    """Structured GGQR input (accepted 2026-09-24).

    ``intent``, ``required_attributes`` and ``task_text`` carry the task
    normalization output (State Model 2.0, blueprint section 3). They are
    derived from the task prompt only — never from verifier data — and are
    consumed by the planner for task-conditioned column projection.
    """

    task_type: str
    reference_id: Any
    reference_type: Optional[str] = None
    task_text: Optional[str] = None
    budget: Optional[QueryBudget] = None
    intent: Tuple[str, ...] = ()
    required_attributes: Tuple[str, ...] = ()
    required_relations: Tuple[Tuple[str, str], ...] = ()
    #: ``RouteStrategy.BROAD`` (eager, historical) or ``RouteStrategy.FRONTIER``
    #: (incremental, coverage-stopping). ``None`` uses the router default.
    strategy: Optional[str] = None


@dataclass(frozen=True)
class TaskRequirements:
    """What a task type needs from the database.

    Beyond the required tables, ``required_attributes`` names task-relevant
    columns (driving attribute pruning) and ``required_relations`` names
    ``(source_table, target_table)`` pairs the task's relational structure
    needs (blueprint sections 3 and 5).
    """

    task_type: str
    required_tables: Tuple[str, ...]
    required_attributes: Tuple[str, ...] = ()
    required_relations: Tuple[Tuple[str, str], ...] = ()
    intent: Tuple[str, ...] = ()

    def satisfied_by(self, observed_tables: set) -> bool:
        return set(self.required_tables).issubset(observed_tables)


# ---------------------------------------------------------------------------
# Plans, steps, results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryStep:
    """One planned query. Every non-anchor step follows a structural edge."""

    step_id: str
    table: str
    filters: Tuple[Filter, ...]
    columns: Optional[Tuple[str, ...]]
    limit: int
    depends_on: Optional[str] = None
    edge_id: Optional[str] = None
    direction: Optional[str] = None  # "outgoing" | "incoming"
    hop: int = 0
    # filter_column: column on THIS step's table to filter on.
    # binding_column: column on the DEPENDENCY row that provides the value.
    filter_column: Optional[str] = None
    binding_column: Optional[str] = None
    #: Retrieval priority assigned by the planner (State Model 2.0, blueprint
    #: section 5). Lower cost is preferred; the frontier strategy executes the
    #: cheapest step that can still add coverage. Defaults to 0.0 so hand-built
    #: steps stay valid.
    priority: float = 0.0


@dataclass(frozen=True)
class QueryPlan:
    anchor_table: str
    anchor_key: str
    anchor_value: Any
    steps: Tuple[QueryStep, ...]
    budget: QueryBudget
    task_type: str
    route_id: str

    def step(self, step_id: str) -> Optional[QueryStep]:
        for candidate in self.steps:
            if candidate.step_id == step_id:
                return candidate
        return None


@dataclass
class QueryResult:
    step_id: str
    table: str
    rows: List[Dict[str, Any]] = field(default_factory=list)
    status: str = StepStatus.OK
    error: Optional[str] = None
    latency_ms: float = 0.0
    edge_id: Optional[str] = None


@dataclass
class RouteOutcome:
    status: str
    plan: QueryPlan
    results: List[QueryResult] = field(default_factory=list)
    bindings: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    unresolved: List["UnresolvedBinding"] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def observed_tables(self) -> set:
        return {
            result.table
            for result in self.results
            if result.status == StepStatus.OK and result.rows
        }

    def planned_tables(self) -> set:
        """Tables the plan intended to read, whether or not they returned rows."""
        return {step.table for step in self.plan.steps}

    def errored_tables(self) -> set:
        """Tables whose query failed to execute."""
        return {
            result.table
            for result in self.results
            if result.status == StepStatus.ERROR
        }

    def unresolved_required_tables(self, requirements: Optional[TaskRequirements] = None) -> set:
        """Required tables that returned no usable rows.

        This is a *coverage* property, not a *correctness* property: an empty
        ``entitlement`` result is a true statement about the database.
        """
        required = (
            tuple(requirements.required_tables)
            if requirements is not None
            else tuple(self.diagnostics.get("required_tables", ()) or ())
        )
        return {table for table in required if table not in self.observed_tables()}

    def structurally_valid(self, requirements: Optional[TaskRequirements] = None) -> bool:
        """Whether the route was *capable* of satisfying the requirements.

        Three conditions must hold, and all three are about the plan and its
        execution rather than about the data:

        1. no step raised an execution error, so every planned statement was
           accepted by the server;
        2. every required table was actually planned, i.e. a valid schema path
           to it was discovered;
        3. the route was not abandoned for a structural reason (the anchor itself
           did not resolve).

        A structurally valid route may still observe nothing. That distinction is
        what Phase 1.3 requires: *"required table reachable AND route
        structurally valid AND query executes"*. Treating a legitimately empty
        required table as an undeliverable intervention was the defect that
        suppressed three of the eleven Condition-B runs in the pilot, because the
        delivery gate demanded ``status == COMPLETE`` rather than a valid route.
        """
        if any(result.status == StepStatus.ERROR for result in self.results):
            return False

        required = (
            tuple(requirements.required_tables)
            if requirements is not None
            else tuple(self.diagnostics.get("required_tables", ()) or ())
        )
        if not required:
            return False

        # The anchor must have resolved to a real row, otherwise there is no
        # state to represent at all.
        anchor_table = self.plan.anchor_table
        anchor_observed = any(
            result.table == anchor_table
            and result.status == StepStatus.OK
            and bool(result.rows)
            for result in self.results
        )
        if not anchor_observed:
            return False

        return set(required).issubset(self.planned_tables())
