"""Goal-driven execution loop for the ablation sweep.

The experiment is expressed as a set of *goals*. One goal is one experimental
unit: ``(task, run_index, condition)``. Each goal walks an explicit state
machine, every transition is timestamped, and each stage is timed separately so
the report can say exactly where wall-clock time went.

``GoalEngine`` adds the properties a real sweep needs:

* **resumability** - goals whose record already exists are skipped, mirroring
  ``benchmark_utils.skip_sample``;
* **scalability** - bounded concurrency via a semaphore, so a bigger evaluation
  set or more runs needs no code change;
* **isolation** - failures are captured per goal, so one bad task cannot abort
  the sweep and silently truncate the statistics.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .conditions import Condition
from .results import TaskRunRecord, is_recorded, utc_now


class GoalState:
    """States a goal moves through. Ordered by pipeline position."""

    PENDING = "PENDING"
    FINGERPRINTED = "FINGERPRINTED"
    SEEDED = "SEEDED"
    AGENT_RUNNING = "AGENT_RUNNING"
    VERIFIED = "VERIFIED"
    RECORDED = "RECORDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"

    TERMINAL = (RECORDED, FAILED, SKIPPED)
    ALL = (
        PENDING,
        FINGERPRINTED,
        SEEDED,
        AGENT_RUNNING,
        VERIFIED,
        RECORDED,
        FAILED,
        SKIPPED,
    )


#: Stages the runner reports timings for, in pipeline order.
STAGES: Tuple[str, ...] = ("fingerprint", "seed", "agent", "verify", "record")


@dataclass(frozen=True)
class Goal:
    """One experimental unit of work."""

    task_id: str
    condition: str
    run_index: int
    task_config_path: str = ""
    complexity_category: str = ""

    @property
    def key(self) -> Tuple[str, str, int]:
        return (self.task_id, self.condition, self.run_index)

    def describe(self) -> str:
        return f"{self.task_id}/{self.condition}/r{self.run_index}"

    @classmethod
    def from_record(
        cls, record: TaskRunRecord, *, task_config_path: str = ""
    ) -> "Goal":
        return cls(
            task_id=record.task_id,
            condition=record.condition,
            run_index=record.run_index,
            task_config_path=task_config_path,
            complexity_category=record.complexity_category,
        )


def build_goals(
    task_records: Sequence[Any],
    conditions: Sequence[Condition],
    *,
    num_runs: int,
    start_run: int = 1,
) -> List[Goal]:
    """Expand an evaluation set into the full goal matrix.

    The order is deterministic: task, then condition, then run index.
    """
    if num_runs < 1:
        raise ValueError("num_runs must be >= 1")
    goals: List[Goal] = []
    for record in task_records:
        for condition in conditions:
            for run_index in range(start_run, start_run + num_runs):
                goals.append(
                    Goal(
                        task_id=record.task_id,
                        condition=condition.name,
                        run_index=run_index,
                        task_config_path=getattr(record, "task_config_path", ""),
                        complexity_category=getattr(record, "complexity_category", ""),
                    )
                )
    return goals


class GoalReporter:
    """Per-goal progress sink handed to the runner."""

    def __init__(
        self,
        goal: Goal,
        on_transition: Optional[Callable[[Goal, str, str], None]] = None,
    ) -> None:
        self.goal = goal
        self._on_transition = on_transition
        self._history: List[str] = [GoalState.PENDING]
        self._timings: Dict[str, float] = {}
        self.error: Optional[str] = None
        self.started_at = utc_now()

    def transition(self, state: str, detail: str = "") -> None:
        """Record a state change; unknown states are rejected."""
        if state not in GoalState.ALL:
            raise ValueError(f"Unknown goal state: {state!r}")
        self._history.append(state)
        if self._on_transition is not None:
            self._on_transition(self.goal, state, detail)

    def measure(self, stage: str, milliseconds: float) -> None:
        """Record a stage duration, accumulating repeated stages."""
        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage!r}; expected one of {list(STAGES)}")
        self._timings[stage] = self._timings.get(stage, 0.0) + float(milliseconds)

    def fail(self, error: BaseException | str) -> None:
        """Mark the goal failed with a concrete reason."""
        self.error = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
        self.transition(GoalState.FAILED, self.error)

    @property
    def history(self) -> Tuple[str, ...]:
        return tuple(self._history)

    @property
    def timings(self) -> Dict[str, float]:
        return dict(self._timings)


@dataclass
class GoalTrace:
    """Outcome of one goal execution."""

    goal: Goal
    state: str
    history: Tuple[str, ...]
    timings_ms: Dict[str, float]
    error: Optional[str]
    wall_clock_ms: float
    record: Optional[TaskRunRecord] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.goal.task_id,
            "condition": self.goal.condition,
            "run_index": self.goal.run_index,
            "state": self.state,
            "history": list(self.history),
            "timings_ms": dict(self.timings_ms),
            "error": self.error,
            "wall_clock_ms": self.wall_clock_ms,
            "recorded": self.record is not None,
        }


@dataclass
class EngineReport:
    """Aggregate outcome of a sweep."""

    traces: List[GoalTrace] = field(default_factory=list)
    wall_clock_ms: float = 0.0
    concurrency: int = 1

    @property
    def completed(self) -> List[GoalTrace]:
        return [t for t in self.traces if t.state == GoalState.RECORDED]

    @property
    def failed(self) -> List[GoalTrace]:
        return [t for t in self.traces if t.state == GoalState.FAILED]

    @property
    def skipped(self) -> List[GoalTrace]:
        return [t for t in self.traces if t.state == GoalState.SKIPPED]

    @property
    def records(self) -> List[TaskRunRecord]:
        return [t.record for t in self.traces if t.record is not None]

    def stage_summary(self) -> Dict[str, Dict[str, float]]:
        """Mean and total milliseconds per stage across executed goals."""
        summary: Dict[str, Dict[str, float]] = {}
        traces = self.completed or self.traces
        for stage in STAGES:
            values = [
                trace.timings_ms[stage] for trace in traces if stage in trace.timings_ms
            ]
            if values:
                summary[stage] = {
                    "mean_ms": sum(values) / len(values),
                    "total_ms": sum(values),
                    "count": len(values),
                }
        return summary

    def as_dict(self) -> Dict[str, Any]:
        return {
            "concurrency": self.concurrency,
            "wall_clock_ms": self.wall_clock_ms,
            "total_goals": len(self.traces),
            "completed": len(self.completed),
            "failed": len(self.failed),
            "skipped": len(self.skipped),
            "goals_per_minute": (
                len(self.completed) / (self.wall_clock_ms / 60000.0)
                if self.wall_clock_ms > 0
                else 0.0
            ),
            "stage_summary": self.stage_summary(),
            "traces": [trace.as_dict() for trace in self.traces],
        }


#: Signature of the per-goal coroutine driven by the engine.
GoalExecutor = Callable[[Goal, GoalReporter], Awaitable[Optional[TaskRunRecord]]]


class GoalEngine:
    """Drives the goal loop with bounded concurrency and per-goal isolation.

    Args:
        executor: Coroutine invoked for each non-skipped goal.
        output_dir: When set, goals with an existing record are skipped, which is
            what makes a sweep resumable.
        max_concurrency: Maximum goals in flight. ``1`` keeps database seeding
            strictly serial, which is the safe default.
        on_transition: Optional observer for state changes.
    """

    def __init__(
        self,
        executor: GoalExecutor,
        *,
        output_dir: Optional[str] = None,
        max_concurrency: int = 1,
        on_transition: Optional[Callable[[Goal, str, str], None]] = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._executor = executor
        self._output_dir = output_dir
        self._concurrency = max_concurrency
        self._on_transition = on_transition

    async def _run_one(self, goal: Goal, semaphore: asyncio.Semaphore) -> GoalTrace:
        reporter = GoalReporter(goal, self._on_transition)
        started = time.perf_counter()
        if self._output_dir is not None and is_recorded(
            self._output_dir, goal.task_id, goal.condition, goal.run_index
        ):
            reporter.transition(GoalState.SKIPPED, "record already exists")
            return GoalTrace(
                goal=goal,
                state=GoalState.SKIPPED,
                history=reporter.history,
                timings_ms=reporter.timings,
                error=None,
                wall_clock_ms=(time.perf_counter() - started) * 1000.0,
            )

        record: Optional[TaskRunRecord] = None
        state = GoalState.FAILED
        async with semaphore:
            try:
                record = await self._executor(goal, reporter)
                state = GoalState.RECORDED if record is not None else GoalState.FAILED
                if record is None:
                    reporter.transition(GoalState.FAILED, "executor returned no record")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one goal must not abort a sweep
                reporter.fail(exc)
                record = None
        return GoalTrace(
            goal=goal,
            state=state,
            history=reporter.history,
            timings_ms=reporter.timings,
            error=reporter.error,
            wall_clock_ms=(time.perf_counter() - started) * 1000.0,
            record=record,
        )

    async def run(self, goals: Iterable[Goal]) -> EngineReport:
        """Execute every goal and return the aggregate report.

        Goals are deduplicated by key, so an overlapping matrix can never
        double-run an experimental unit.
        """
        unique: Dict[Tuple[str, str, int], Goal] = {}
        for goal in goals:
            unique.setdefault(goal.key, goal)
        ordered = list(unique.values())
        if not ordered:
            return EngineReport(traces=[], wall_clock_ms=0.0, concurrency=self._concurrency)

        semaphore = asyncio.Semaphore(self._concurrency)
        started = time.perf_counter()
        traces = await asyncio.gather(
            *(self._run_one(goal, semaphore) for goal in ordered)
        )
        return EngineReport(
            traces=list(traces),
            wall_clock_ms=(time.perf_counter() - started) * 1000.0,
            concurrency=self._concurrency,
        )
