from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .builder import CSMEnvironmentRepresentation


class CSMEnvironmentAPI:
    """Stable interface for agents/planners/evaluators.

    Downstream modules should depend on this API instead of depending on the
    graph implementation or SQL schema directly.

    Additive GGQR surface (approved plan Phase 8): when the underlying
    representation was constructed through :func:`from_enterpriseops_mcp_client`
    with GGQR enabled, `route_state`, `represent_state`, and `verify_case`
    are available; otherwise they raise a typed error instead of failing
    silently.
    """

    def __init__(self, environment: CSMEnvironmentRepresentation, ggqr_environment: Any = None) -> None:
        self.environment = environment
        self._ggqr = ggqr_environment

    # ---------- legacy surface (unchanged behavior) ----------

    async def get_ground_truth(self, entity_type: str, entity_id: Any) -> Dict[str, Any]:
        return await self.environment.get_current_state(entity_type, entity_id)

    async def get_case_context(self, case_id: Any, hops: int = 2) -> Dict[str, Any]:
        return await self.environment.get_case_context(case_id, max_hops=hops)

    async def get_neighbors(
        self,
        entity_type: str,
        entity_id: Any,
        relation: Optional[str] = None,
        direction: str = "both",
        refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        if refresh:
            await self.environment.hydrate_entity(entity_type, entity_id, include_links=True)
        return self.environment.query_local_graph(entity_type, entity_id, relation, direction)

    async def search_cases(self, *, state: Optional[str] = None, priority: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        predicates = []
        if state is not None:
            predicates.append(f"state = {self.environment._sql_literal(state)}")
        if priority is not None:
            predicates.append(f"priority = {self.environment._sql_literal(priority)}")
        where = f" WHERE {' AND '.join(predicates)}" if predicates else ""
        limit = max(1, min(int(limit), 500))
        return await self.environment.query(
            "SELECT " + ", ".join(
                [
                    "case_id", "number", "account_id", "contact_id", "product_id",
                    "installed_product_id", "channel", "priority", "state",
                    "short_description", "assignment_group_id", "assigned_to",
                    "escalation", "escalation_reason", "reopen_count",
                ]
            ) + f" FROM customer_case{where} ORDER BY case_id LIMIT {limit};"
    )

    # ---------- additive GGQR surface ----------

    def attach_ggqr(self, ggqr_environment: Any) -> None:
        """Attach a GGQREnvironment to enable the additive routing surface."""
        self._ggqr = ggqr_environment

    def _require_ggqr(self):
        if self._ggqr is None:
            raise RuntimeError(
                "GGQR is not attached. Construct the API via "
                "csm_env.from_enterpriseops_mcp_client(..., enable_ggqr=True) "
                "or call attach_ggqr(ggqr_environment)."
            )
        return self._ggqr

    async def route_state(self, task_type: str, reference_id: Any, reference_type: Optional[str] = None):
        ggqr = self._require_ggqr()
        from .query.models import TaskRequest

        return await ggqr.route(
            TaskRequest(task_type=task_type, reference_id=reference_id, reference_type=reference_type)
        )

    async def build_state(
        self,
        task_type: str,
        reference_id: Any,
        reference_type: Optional[str] = None,
        budget: Optional[Any] = None,
        *,
        strategy: Optional[str] = None,
        required_attributes: Optional[Tuple[str, ...]] = None,
        required_relations: Optional[Tuple[Tuple[str, str], ...]] = None,
        intent: Optional[Tuple[str, ...]] = None,
    ):
        """Build the grounded state for one task/anchor pair.

        Args:
            task_type: Registered GGQR task type, e.g. ``resolve_case``.
            reference_id: Anchor primary-key value.
            reference_type: Anchor table name.
            budget: Optional per-request :class:`~csm_env.query.models.QueryBudget`.
                Supplying this lets a caller widen relational depth for a task
                whose required tables are more than the default hop count away,
                without mutating the shared environment default. ``None`` keeps
                the environment's configured budget.
            required_attributes / required_relations / intent: Optional task-
                normalization carriers (State Model 2.0, blueprint section 3)
                that are unioned with the registered task-type requirements
                before the route is planned. Left as ``None`` keeps the
                registered defaults.
        """
        ggqr = self._require_ggqr()
        return await ggqr.build_state_for(
            task_type,
            reference_id,
            reference_type,
            budget=budget,
            strategy=strategy,
            required_attributes=required_attributes,
            required_relations=required_relations,
            intent=intent,
        )

    def represent_state(self, state, *, max_chars: int = 4000) -> Dict[str, str]:
        return type(self._require_ggqr()).represent(state, max_chars=max_chars)

    def verify_case(self, case_id: Any):
        ggqr = self._require_ggqr()
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            raise RuntimeError("verify_case cannot be called from a running event loop; await route_state/build_state instead")
        return asyncio.run(ggqr.verify_case_equivalence(case_id))
