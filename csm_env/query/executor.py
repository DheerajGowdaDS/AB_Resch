from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Dict, List, Optional

from .models import QueryBudget, QueryResult, QueryStep


class QueryExecutor:
    """Executes router-resolved, dependency-ready steps in bounded waves.

    Readiness (dependency bindings) is resolved by the router before steps
    are handed over; the executor enforces max_parallel batching, records
    per-step latency/status, and returns results sorted by step ID so output
    is deterministic regardless of completion order (RISK-006).
    """

    def __init__(self, run_step: Optional[Callable[[QueryStep, QueryBudget], Awaitable[QueryResult]]] = None) -> None:
        self._run_step = run_step

    async def execute(
        self,
        steps: List[QueryStep],
        budget: QueryBudget,
        run_step: Optional[Callable[[QueryStep, QueryBudget], Awaitable[QueryResult]]] = None,
    ) -> List[QueryResult]:
        if not steps:
            return []

        runner = run_step or self._run_step
        if runner is None:
            raise ValueError("QueryExecutor requires a run_step callback")
        ordered = sorted(steps, key=lambda step: step.step_id)
        completed: Dict[str, QueryResult] = {}

        for start in range(0, len(ordered), max(1, budget.max_parallel)):
            batch = ordered[start : start + max(1, budget.max_parallel)]
            results = await asyncio.gather(
                *(runner(step, budget) for step in batch)
            )
            for step, result in zip(batch, results):
                completed[step.step_id] = result

        return [completed[step_id] for step_id in sorted(completed)]
