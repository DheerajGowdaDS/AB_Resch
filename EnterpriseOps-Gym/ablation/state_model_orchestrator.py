"""Condition B orchestrator: ReAct plus the frozen ``csm_env`` state context.

The ReAct loop itself is **not** copied or modified. This subclass injects one
clearly labelled, auditable state block into the task message and then delegates
to ``ReactOrchestrator.execute()``. Condition A therefore runs the exact same
loop with the exact same tools, prompts and step budget; the only difference is
the presence of that block.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

from orchestrators.react import ReactOrchestrator

from csm_env import QueryBudget
from csm_env.query.models import RouteStatus

from .state_model import StateContext, StateContextPolicy, StateModelAdapter

#: Marker that makes the injected block greppable in ``conversation_flow``.
STATE_CONTEXT_HEADER = "[CSM_STATE_CONTEXT]"

#: Separates the original task text from the injected evidence.
STATE_CONTEXT_RULE = "-" * 8

DEFAULT_CSM_SERVER_NAME = "sn-csm-server"
#: CSM Experiment-1 needs one additional structural hop for valid relations such as
#: location -> user -> user_group_member -> user_group. Explicit caller budgets
#: still override this default.
DEFAULT_STATE_QUERY_BUDGET = QueryBudget(max_hops=3)

PathLike = Union[str, Path]

#: Signature of the environment factory: (mcp_client) -> CSMEnvironmentAPI.
EnvironmentFactory = Callable[[Any], Any]


class StateModelDeliveryError(RuntimeError):
    """Condition-B intervention could not be delivered to the agent.

    Experiment 1 is a causal ablation. Falling back to native ReAct when the
    treatment context is missing would silently turn B into a mixture of B and A.
    The intervention therefore fails closed instead.
    """


#: Route statuses that represent a *structurally* sound route. ``UNRESOLVED`` and
#: ``BUDGET_EXHAUSTED`` are included deliberately: both mean some required table
#: came back empty, which is a true statement about the database rather than a
#: defect in the intervention. Only ``FAILED`` indicates a real routing fault.
DELIVERABLE_ROUTE_STATUSES: Tuple[str, ...] = (
    RouteStatus.COMPLETE,
    RouteStatus.UNRESOLVED,
    RouteStatus.BUDGET_EXHAUSTED,
)


def state_context_delivered(context: StateContext) -> bool:
    """Whether a state context counts as a usable Condition-B intervention.

    This is the single delivery rule for the whole experiment (Phase 1.1), and it
    is deliberately different from the pilot's rule. The pilot required
    ``route_status == COMPLETE``, which meant a required table holding zero rows
    silently downgraded a perfectly good retrieval into an "undelivered"
    intervention. Three of eleven B runs were lost that way, and the report
    correctly refused to state a causal effect as a result.

    The rule now asks the Phase 1.3 question - *is the route sound?* - rather
    than *did every table happen to return rows?*:

    * the anchor resolved to a real row;
    * the route status is not ``FAILED`` (no execution error);
    * at least one record was retrieved;
    * the rendered representation is non-empty and non-zero in tokens.

    Coverage gaps remain visible through ``StateContext.unresolved``, so nothing
    is hidden; they simply stop masquerading as delivery failures.
    """
    return (
        not context.is_empty
        and context.route_status in DELIVERABLE_ROUTE_STATUSES
        and context.anchor.resolved
        and context.retrieved_record_count > 0
        and context.retrieved_token_count > 0
    )


@dataclass(frozen=True)
class StateUsageTelemetry:
    """Per-run usage of the state model, merged into the benchmark result.

    ``retrieved_tables`` is the *count* of distinct tables. The parallel
    ``retrieved_table_names`` and ``retrieved_rows`` carry the identity of what
    was actually retrieved, which is what the evaluator-only initial-state relevance analysis
    needs; keeping the count and the names as separate fields avoids the
    ambiguity that made a list-vs-int mismatch possible.
    """

    available: bool
    state_retrieval_count: int
    retrieved_records: int
    retrieved_tables: int
    retrieved_token_count: int
    context_tokens_injected: int
    state_route_status: str
    anchor_status: str
    anchor_table: Optional[str]
    anchor_row_id: Optional[str]
    retrieval_latency_ms: float
    error: Optional[str]
    retrieved_table_names: Tuple[str, ...] = ()
    retrieved_rows: Tuple[Tuple[str, str], ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "state_model_available": self.available,
            "state_model_delivered": self.available,
            "state_retrieval_count": self.state_retrieval_count,
            "retrieved_records": self.retrieved_records,
            "retrieved_tables": self.retrieved_tables,
            "retrieved_table_names": list(self.retrieved_table_names),
            "retrieved_rows": [list(row) for row in self.retrieved_rows],
            "retrieved_tokens": self.retrieved_token_count,
            "context_tokens_injected": self.context_tokens_injected,
            "state_route_status": self.state_route_status,
            "anchor_status": self.anchor_status,
            "anchor_table": self.anchor_table,
            "anchor_row_id": self.anchor_row_id,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "state_error": self.error,
        }

    @classmethod
    def unavailable(cls, error: str) -> "StateUsageTelemetry":
        return cls(
            available=False,
            state_retrieval_count=0,
            retrieved_records=0,
            retrieved_tables=0,
            retrieved_token_count=0,
            context_tokens_injected=0,
            state_route_status=RouteStatus.FAILED,
            anchor_status="UNRESOLVED",
            anchor_table=None,
            anchor_row_id=None,
            retrieval_latency_ms=0.0,
            error=error,
        )


def default_environment_factory(
    mcp_client: Any,
    *,
    requirements_path: Optional[PathLike] = None,
    budget: Optional[QueryBudget] = None,
) -> Any:
    """Build the ``csm_env`` facade for the benchmark's live MCP client.

    Uses the existing additive wiring in ``csm_integration`` so the state model
    reads the same SQL-runner path, headers and ``database_id`` as the agent.
    """
    from csm_integration import build_csm_environment

    if requirements_path is None:
        return build_csm_environment(mcp_client, enable_ggqr=True, budget=budget)
    return build_csm_environment(
        mcp_client,
        enable_ggqr=True,
        budget=budget,
        requirements_path=requirements_path,
    )


def compose_state_user_prompt(original_prompt: str, context: StateContext) -> str:
    """Append the labelled state block to the task prompt.

    The task text is preserved byte-for-byte as the leading content, so the
    benchmark prompt is unchanged; only an additive, clearly marked block
    follows it.
    """
    body = context.rendered_text.strip()
    return (
        f"{original_prompt}\n\n{STATE_CONTEXT_RULE}\n"
        f"{STATE_CONTEXT_HEADER} grounded state for {context.task_id} "
        f"(anchor {context.reference_type}:{context.reference_id}, "
        f"route {context.route_status}, {context.retrieved_record_count} records, "
        f"schema {context.schema_version})\n{body}\n"
    )


class StateModelReactOrchestrator(ReactOrchestrator):
    """ReAct with one pre-action grounded-state injection.

    Args:
        state_task: The ``TaskRecord`` describing the current task.
        state_environment_factory: Optional override used by tests. Defaults to
            :func:`default_environment_factory`, called with the live MCP client.
        state_requirements_path: Path to the GGQR requirements manifest.
        state_budget: Optional ``QueryBudget`` for the GGQR route.
        state_gym_name: Which ``mcp_clients`` entry to attach the state model to.
        state_context_policy: ``pre_action_once`` (default) or
            ``pre_action_per_step``.
    """

    def __init__(
        self,
        *args: Any,
        state_task: Any = None,
        state_environment_factory: Optional[EnvironmentFactory] = None,
        state_requirements_path: Optional[PathLike] = None,
        state_budget: Optional[QueryBudget] = None,
        state_gym_name: str = DEFAULT_CSM_SERVER_NAME,
        state_context_policy: str = StateContextPolicy.PRE_ACTION_ONCE,
        state_minimal: bool = False,
        **kwargs: Any,
    ) -> None:
        """Args:
        state_minimal: Selects the minimal-state (B2) rendering. The default
            is deliberately ``False`` (broad, B1): the manipulated variable
            must be set explicitly by the condition, never by omission, so a
            call path that forgets the flag silently running the *minimal*
            arm can confound the B1/B2 comparison.
        """
        super().__init__(*args, **kwargs)
        self.state_task = state_task
        self.state_environment_factory = state_environment_factory or default_environment_factory
        self.state_requirements_path = state_requirements_path
        self.state_budget = state_budget if state_budget is not None else DEFAULT_STATE_QUERY_BUDGET
        self.state_gym_name = state_gym_name
        self.state_context_policy = StateContextPolicy.validate(state_context_policy)
        self.state_minimal = bool(state_minimal)
        self._telemetry: Optional[StateUsageTelemetry] = None

    def _client(self) -> Any:
        clients = self.mcp_clients or {}
        if self.state_gym_name in clients:
            return clients[self.state_gym_name]
        if not clients:
            raise KeyError("No MCP clients are attached to this orchestrator")
        return next(iter(clients.values()))

    async def resolve_state_context(self) -> Optional[StateContext]:
        """Materialize the state context, or ``None`` when unavailable.

        The environment is built lazily, at execute time, so it always observes
        the ``database_id`` the benchmark just seeded.
        """
        if self.state_task is None:
            self._telemetry = StateUsageTelemetry.unavailable("no task attached")
            return None
        try:
            if callable(self.state_environment_factory):
                api = self.state_environment_factory(
                    self._client(),
                    requirements_path=self.state_requirements_path,
                    budget=self.state_budget,
                )
            else:
                api = self.state_environment_factory
            adapter = StateModelAdapter(api, self.state_task)
            return await adapter.context_for(minimal_state=self.state_minimal)
        except Exception as exc:  # noqa: BLE001 - reported, never silently swallowed
            self._telemetry = StateUsageTelemetry.unavailable(f"{type(exc).__name__}: {exc}")
            return None

    def _record_telemetry(self, context: Optional[StateContext]) -> None:
        if context is None:
            if self._telemetry is None:
                self._telemetry = StateUsageTelemetry.unavailable("context unavailable")
            return
        delivered = state_context_delivered(context)
        self._telemetry = StateUsageTelemetry(
            available=delivered,
            state_retrieval_count=1,
            retrieved_records=context.retrieved_record_count,
            retrieved_tables=len(context.retrieved_tables),
            retrieved_token_count=context.retrieved_token_count,
            context_tokens_injected=context.retrieved_token_count if delivered else 0,
            state_route_status=context.route_status,
            anchor_status=context.anchor.status,
            anchor_table=context.reference_type,
            anchor_row_id=context.reference_id,
            retrieval_latency_ms=context.retrieval_latency_ms,
            error=context.error,
            retrieved_table_names=context.retrieved_tables,
            retrieved_rows=context.retrieved_rows,
        )

    async def execute(self) -> Dict[str, Any]:
        """Materialize state, inject it, then run the untouched ReAct loop.

        Condition B is fail-closed: a missing, empty, or unresolved state
        context is an intervention failure, not an invitation to run A. Note
        that a ``UNRESOLVED`` *route* status no longer fails the gate on its
        own; see :func:`state_context_delivered` for why.
        """
        context = await self.resolve_state_context()
        self._record_telemetry(context)

        if context is None:
            detail = (self._telemetry.error if self._telemetry else None) or "context unavailable"
            raise StateModelDeliveryError(f"state model delivery failed: {detail}")

        if not state_context_delivered(context):
            detail = context.error or "usable grounded state context required"
            raise StateModelDeliveryError(
                f"state model delivery failed: route={context.route_status}, "
                f"anchor={context.anchor.status}, "
                f"records={context.retrieved_record_count}, "
                f"tokens={context.retrieved_token_count}; {detail}"
            )

        original_config = self.config
        self.config = replace(
            original_config,
            user_prompt=compose_state_user_prompt(original_config.user_prompt, context),
        )
        try:
            return await super().execute()
        finally:
            self.config = original_config

    def get_result_metadata(self) -> Dict[str, Any]:
        """Merge state usage into the benchmark's run result."""
        telemetry = self._telemetry or StateUsageTelemetry.unavailable("not executed")
        return {"state_retrieval": telemetry.as_dict()}
