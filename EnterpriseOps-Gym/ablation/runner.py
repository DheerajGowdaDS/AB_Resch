"""Paired A/B execution against the unmodified benchmark.

Composition rules honoured here:

* the benchmark does the seeding, the agent run and the official verification;
* the harness only *observes* and *records*, plus the index-keyed verifier pass
  and a read-only pre-run fingerprint taken while the seeded database is alive;
* both conditions execute through the same code path, so the only difference is
  whether the state block is injected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, Union

from benchmark.models import BenchmarkConfig, LLMConfig
from csm_env import QueryBudget, SchemaRegistry
from orchestrators.react import ReactOrchestrator

from .conditions import CONDITION_A, CONDITION_B1, Condition
from .experiment_manifest import VERSION_STATE_MODEL_REFINED
from .fingerprint import initial_state_fingerprint
from .goals import GoalReporter, GoalState
from .results import TaskRunRecord, new_run_uid, utc_now, write_run_record
from .seeds import DEFAULT_ARCHIVE, SeedCache
from .state_model_orchestrator import StateModelReactOrchestrator
from .task_registry import TaskRecord, apply_tool_mode, load_task_config
from .verifier_bridge import IndexedVerifierExecutor

PathLike = Union[str, Path]

#: Tool-name prefixes that indicate a read-only action. Derived from the tool
#: names the benchmark actually exposes (``find_*``, ``retrieve_*``, ...), not
#: from a hard-coded list of CSM operations.
READ_TOOL_PREFIXES: Tuple[str, ...] = (
    "find_",
    "get_",
    "list_",
    "search_",
    "retrieve_",
    "view_",
    "read_",
    "check_",
    "query_",
    "lookup_",
)

#: Default step budget for the paired sweep.
#:
#: Phase 4 of the blueprint: the pilot ran with ``max_steps=5`` and every one of
#: the eleven tasks failed, which makes the primary metric (task success) unable
#: to discriminate between the arms. The CSM tasks are multi-step workflows, so a
#: shallow budget terminates them mid-procedure. Fifteen is the recommended
#: starting budget.
#:
#: The value is forwarded explicitly to *both* arms by
#: :func:`ablation.runner.build_orchestrator`, so the budget is identical by
#: construction. Raising it for Condition B alone would break the only
#: controlled variable in the experiment.
MAX_STEPS_DEFAULT = 15

#: Upper bound accepted by the CLI. A larger budget mostly raises cost; it does
#: not add signal once runs stop being budget-terminated.
MAX_STEPS_CEILING = 100


def is_read_tool(tool_name: str) -> bool:
    """Whether a tool name looks read-only by convention."""
    lowered = (tool_name or "").lower()
    return lowered.startswith(READ_TOOL_PREFIXES)


def extract_telemetry(run: Dict[str, Any]) -> Dict[str, Any]:
    """Derive per-run counters from a benchmark run result.

    Token usage is reported only when the provider actually returned it; absent
    usage stays ``None`` rather than becoming a misleading zero.
    """
    conversation = run.get("conversation_flow") or []
    tool_results = run.get("tool_results") or []

    steps = 0
    input_tokens: Optional[int] = 0
    output_tokens: Optional[int] = 0
    saw_usage = False
    for entry in conversation:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") == "ai_message":
            steps += 1
        usage = entry.get("usage_metadata") or {}
        if isinstance(usage, dict) and usage:
            saw_usage = True
            input_tokens = (input_tokens or 0) + int(usage.get("input_tokens", 0) or 0)
            output_tokens = (output_tokens or 0) + int(usage.get("output_tokens", 0) or 0)

    read_calls = 0
    invalid = 0
    for item in tool_results:
        if not isinstance(item, dict):
            continue
        if is_read_tool(str(item.get("tool_name", ""))):
            read_calls += 1
        result = item.get("result")
        if isinstance(result, dict) and result.get("success") is False:
            invalid += 1

    return {
        "steps_taken": steps,
        "tool_call_count": len(tool_results),
        "read_query_count": read_calls,
        "invalid_action_count": invalid,
        "input_tokens": input_tokens if saw_usage else None,
        "output_tokens": output_tokens if saw_usage else None,
    }


#: Signature of the pre-run fingerprint probe: (mcp_client) -> digest or None.
FingerprintProbe = Callable[[Any], Awaitable[Optional[str]]]


def default_fingerprint_probe(
    registry: Optional[SchemaRegistry] = None, *, strict: bool = False
) -> FingerprintProbe:
    """Build the default probe: a read-only fingerprint over the live database.

    It reuses ``csm_env``'s own SQL runner so the digest covers exactly the
    tables the state model can see.

    Args:
        strict: When true, a probe failure raises instead of degrading to
            ``None``. A ``None`` fingerprint makes the pairing unverifiable, so
            strict mode turns a systematically broken probe into a loud error
            rather than a sweep full of unpairable records.
    """
    resolved_registry = registry or SchemaRegistry.from_static()

    async def probe(mcp_client: Any) -> Optional[str]:
        from csm_env.sql_runner import EnterpriseOpsSQLRunner
        from csm_env.transport.reader import EnterpriseOpsSQLRunnerReader

        try:
            runner = EnterpriseOpsSQLRunner(
                base_url=mcp_client.base_url,
                database_id=mcp_client.database_id,
                auth_config=getattr(mcp_client, "auth_config", None),
                context=getattr(mcp_client, "context", None),
            )
            reader = EnterpriseOpsSQLRunnerReader(runner)
            return await initial_state_fingerprint(reader, resolved_registry)
        except Exception:
            if strict:
                raise
            return None

    return probe


class ProbedReactOrchestrator(StateModelReactOrchestrator):
    """ReAct with a read-only pre-run fingerprint and no state injection.

    This is Condition A. It exists only to measure the initial state of the
    database the benchmark just seeded, so the pairing can be verified. It adds
    no message, no tool, and no behavioural change whatsoever.
    """

    def __init__(self, *args: Any, probe: Optional[FingerprintProbe] = None, **kwargs: Any) -> None:
        kwargs.pop("state_task", None)
        super().__init__(*args, **kwargs)
        self.probe = probe
        self.pre_state_fingerprint: Optional[str] = None

    async def execute(self) -> Dict[str, Any]:
        """Fingerprint the seeded database, then run the untouched loop."""
        if self.probe is not None:
            try:
                self.pre_state_fingerprint = await self.probe(self._client())
            except Exception:  # noqa: BLE001 - a probe failure must not fail the run
                self.pre_state_fingerprint = None
        return await ReactOrchestrator.execute(self)

    def get_result_metadata(self) -> Dict[str, Any]:
        """No state usage for Condition A; the fingerprint is harness metadata."""
        return {"state_retrieval": None, "pre_state_fingerprint": self.pre_state_fingerprint}


class ProbedStateModelReactOrchestrator(StateModelReactOrchestrator):
    """Condition B: the state injection plus the same pre-run fingerprint."""

    def __init__(self, *args: Any, probe: Optional[FingerprintProbe] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.probe = probe
        self.pre_state_fingerprint: Optional[str] = None

    async def resolve_state_context(self) -> Optional[Any]:
        """Fingerprint first, then materialize the grounded state."""
        if self.probe is not None:
            try:
                self.pre_state_fingerprint = await self.probe(self._client())
            except Exception:  # noqa: BLE001 - a probe failure must not fail the run
                self.pre_state_fingerprint = None
        return await super().resolve_state_context()

    def get_result_metadata(self) -> Dict[str, Any]:
        """Merge state usage and the fingerprint into the run result."""
        metadata = super().get_result_metadata()
        metadata["pre_state_fingerprint"] = self.pre_state_fingerprint
        return metadata


class AblationExecutor(IndexedVerifierExecutor):
    """Benchmark executor that also surfaces harness metadata on each run.

    The orchestrator instances the parent constructs are captured so their
    fingerprint and state telemetry can be attached to the run result.
    """

    def __init__(self, *args: Any, capture: Optional[List[Any]] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.captured_orchestrators: List[Any] = capture if capture is not None else []

    async def execute_single_run(self, run_number: int) -> Dict[str, Any]:
        """Attach pre-run fingerprint, runtime database id, and state telemetry.

        Interventions can fail before the base executor has a verifier result.
        Capture that failure as a normal run record so the state-delivery reason
        and telemetry are preserved for the ablation report instead of being
        replaced by a generic outer-loop error.
        """
        try:
            result = await super().execute_single_run(run_number)
        except Exception as exc:  # noqa: BLE001 - convert to an auditable run result
            result = {
                "run_number": run_number,
                "error": f"{type(exc).__name__}: {exc}",
                "overall_success": False,
                "verification_results": {},
                "verification_summary": {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "pass_rate": 0.0,
                },
                "execution_time_ms": 0,
                "conversation_flow": [],
                "tools_used": [],
                "tool_results": [],
                "model_response": None,
            }
        harness: Dict[str, Any] = {
            "pre_state_fingerprint": None,
            "state_retrieval": None,
            "runtime_database_id": None,
        }
        # The seeded database id is only known here, while the run is alive;
        # execute_benchmark deletes the database in its finally block. The base
        # executor assigns it to the MCP client but never writes it back into
        # gym_configs, so the client is the authoritative source here.
        for gym in self.gym_configs or ():
            client = self.mcp_clients.get(gym.get("mcp_server_name"))
            database_id = getattr(client, "database_id", None) or gym.get("database_id")
            if database_id:
                harness["runtime_database_id"] = database_id
                break
        if self.captured_orchestrators:
            instance = self.captured_orchestrators[-1]
            metadata = instance.get_result_metadata() or {}
            harness["pre_state_fingerprint"] = metadata.get("pre_state_fingerprint")
            harness["state_retrieval"] = metadata.get("state_retrieval")
        result["harness"] = harness
        return result


def orchestrator_factory(
    base_class: Any, capture: List[Any], **extra_kwargs: Any
) -> Callable[..., Any]:
    """Return a factory the benchmark can call with keyword arguments.

    ``BenchmarkExecutor.execute_single_run`` constructs the orchestrator with
    keyword arguments only, so this wrapper is a drop-in replacement for the
    class and records each instance for later inspection.
    """

    def factory(**kwargs: Any) -> Any:
        instance = base_class(**kwargs, **extra_kwargs)
        capture.append(instance)
        return instance

    factory.__name__ = f"captured_{getattr(base_class, '__name__', 'orchestrator')}"
    return factory


def build_orchestrator(
    condition: Condition,
    *,
    capture: List[Any],
    task: Optional[TaskRecord],
    probe: Optional[FingerprintProbe] = None,
    requirements_path: Optional[PathLike] = None,
    budget: Optional[QueryBudget] = None,
    max_steps: int = MAX_STEPS_DEFAULT,
) -> Callable[..., Any]:
    """Return the orchestrator factory for one condition.

    ``max_steps`` is forwarded as an explicit keyword to both arms, so the step
    budget is identical by construction rather than by coincidence with the
    benchmark's default (the orchestrator base class accepts it as
    ``max_iterations``).
    """
    if condition.state_model:
        return orchestrator_factory(
            ProbedStateModelReactOrchestrator,
            capture,
            probe=probe,
            state_task=task,
            state_requirements_path=requirements_path,
            state_budget=budget,
            state_minimal=condition.minimal_state,
            max_iterations=max_steps,
        )
    return orchestrator_factory(
        ProbedReactOrchestrator,
        capture,
        probe=probe,
        max_iterations=max_steps,
    )


def _runtime_database_id(result: Dict[str, Any], run: Dict[str, Any]) -> Optional[str]:
    """Best-effort runtime database id for the run that just executed.

    Prefers the id the executor recorded while the seeded database was alive,
    then falls back to the benchmark config block.
    """
    harness_id = (run.get("harness") or {}).get("runtime_database_id")
    if harness_id:
        return str(harness_id)
    for gym in (result.get("benchmark_config") or {}).get("gym_servers") or ():
        database_id = gym.get("database_id")
        if database_id:
            return str(database_id)
    return None


def record_from_run(
    *,
    condition: Condition,
    task: TaskRecord,
    run_index: int,
    llm_config: LLMConfig,
    result: Dict[str, Any],
    run: Dict[str, Any],
    tool_mode: str,
    orchestrator_name: str,
    goal_history: Sequence[str] = (),
    timings_ms: Optional[Dict[str, float]] = None,
    max_steps: int = MAX_STEPS_DEFAULT,
) -> TaskRunRecord:
    """Fold one benchmark run into a :class:`TaskRunRecord`."""
    telemetry = extract_telemetry(run)
    harness = run.get("harness") or {}
    indexed = run.get("verification_indexed") or {}
    error_message = run.get("error")
    agent_error = bool(error_message) or bool(run.get("error"))

    return TaskRunRecord(
        experiment="experiment_1",
        run_uid=new_run_uid(),
        condition=condition.name,
        task_id=task.task_id,
        run_index=run_index,
        complexity_category=task.complexity_category,
        seed_database_file=task.seed_database_file,
        runtime_database_id=_runtime_database_id(result, run),
        initial_state_fingerprint=harness.get("pre_state_fingerprint"),
        model_provider=getattr(llm_config, "llm_provider", ""),
        model_name=getattr(llm_config, "llm_model", ""),
        temperature=float(getattr(llm_config, "temperature", 0.0) or 0.0),
        max_steps=max_steps,
        tool_mode=tool_mode,
        orchestrator=orchestrator_name,
        overall_success=bool(run.get("overall_success", False)),
        verifier_summary_collapsed=dict(run.get("verification_summary", {}) or {}),
        verifier_summary_indexed=dict(indexed.get("indexed_summary", {}) or {}),
        verifier_results_collapsed=dict(run.get("verification_results", {}) or {}),
        verifier_results_indexed=dict(indexed.get("results_indexed", {}) or {}),
        agent_error=agent_error,
        error_message=error_message,
        latency_ms=int(run.get("execution_time_ms", 0) or 0),
        state_retrieval=harness.get("state_retrieval"),
        started_at=str(run.get("started_at", "")),
        finished_at=utc_now(),
        goal_states=tuple(goal_history),
        timings_ms=dict(timings_ms or {}),
        experiment_version=VERSION_STATE_MODEL_REFINED,
        **telemetry,
    )


async def run_condition(
    condition: Condition,
    task: TaskRecord,
    run_index: int,
    llm_config: LLMConfig,
    output_dir: PathLike,
    *,
    tool_mode: str = "oracle",
    requirements_path: Optional[PathLike] = None,
    budget: Optional[QueryBudget] = None,
    probe: Optional[FingerprintProbe] = None,
    orchestrator_name: str = "react",
    max_steps: int = MAX_STEPS_DEFAULT,
    archive: PathLike = DEFAULT_ARCHIVE,
    seed_cache: Optional["SeedCache"] = None,
) -> TaskRunRecord:
    """Execute one task under one condition and persist the record.

    The benchmark owns database seeding, the agent loop, and the official
    verification. Repeats are owned by the harness, so the task's own
    ``number_of_runs`` is forced to 1 (CON-005).

    The task's ``seed_database_file`` names a member of ``gym_dbs.zip`` while the
    benchmark reads seed SQL from disk, so the seed is materialized first
    (:mod:`ablation.seeds`). Without that step ``create_database_from_file``
    returns ``None``, the agent runs against no database, and every verifier
    fails for a reason that has nothing to do with the intervention.

    Raises:
        ValueError: If the task's tool mode is unsupported.
        FileNotFoundError: If the task's seed SQL cannot be materialized.
    """
    config: BenchmarkConfig = apply_tool_mode(load_task_config(task.task_config_path), tool_mode)
    config.number_of_runs = 1
    (seed_cache or SeedCache(archive)).bind_config(config)

    capture: List[Any] = []
    factory = build_orchestrator(
        condition,
        capture=capture,
        task=task,
        probe=probe,
        requirements_path=requirements_path,
        budget=budget,
        max_steps=max_steps,
    )
    executor = AblationExecutor(
        config,
        llm_config=llm_config,
        orchestrator_class=factory,
        orchestrator_kwargs={},
        config_path=task.task_config_path,
        capture=capture,
    )

    started = time.perf_counter()
    result = await executor.execute_benchmark()
    agent_ms = (time.perf_counter() - started) * 1000.0

    runs = list(result.get("runs") or [])
    if not runs:
        raise RuntimeError(
            f"{task.task_id}/{condition.name}/r{run_index}: benchmark produced no runs"
        )
    run = runs[0]
    indexed = run.get("verification_indexed") or {}

    # Integrity gate: a run is only scoreable if the benchmark actually
    # evaluated verifiers against a seeded database. Seeding failures otherwise
    # masquerade as a legitimate "agent failed every check".
    collapsed_total = int((run.get("verification_summary") or {}).get("total", 0) or 0)
    indexed_total = int((indexed.get("indexed_summary") or {}).get("total", 0) or 0)
    seeding_failure = collapsed_total == 0 and indexed_total == 0

    record = record_from_run(
        condition=condition,
        task=task,
        run_index=run_index,
        llm_config=llm_config,
        result=result,
        run=run,
        tool_mode=tool_mode,
        orchestrator_name=orchestrator_name,
        max_steps=max_steps,
        goal_history=(GoalState.AGENT_RUNNING, GoalState.VERIFIED, GoalState.RECORDED),
        timings_ms={
            "agent": agent_ms,
            "verify": float(run.get("execution_time_ms", 0) or 0),
            "lost_to_collision": float(indexed.get("lost_to_collision", 0) or 0),
        },
    )
    if seeding_failure and not record.agent_error:
        # A run that evaluated no verifiers never touched a seeded database.
        # Recording it as a normal failure would silently corrupt the headline
        # metric, so it is quarantined as an execution error instead. Preserve an
        # existing orchestrator/intervention error when one is already present.
        from dataclasses import replace

        record = replace(
            record,
            agent_error=True,
            error_message=(
                "no verifier was evaluated: the task database was not seeded "
                "(check the seed archive and the CSM server)"
            ),
        )
    write_run_record(record, output_dir)
    return record


@dataclass
class PairOutcome:
    """The two records of one matched task instance, plus pairing health."""

    task_id: str
    run_index: int
    baseline: TaskRunRecord
    treatment: TaskRunRecord
    fingerprint_match: bool
    reason: str = ""

    @property
    def usable(self) -> bool:
        """A pair counts only when the benchmark pair and B intervention are clean."""
        from .stats import state_model_delivered

        return (
            self.fingerprint_match
            and not self.baseline.agent_error
            and not self.treatment.agent_error
            and state_model_delivered(self.treatment)
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_index": self.run_index,
            "fingerprint_match": self.fingerprint_match,
            "usable": self.usable,
            "reason": self.reason,
            "baseline": self.baseline.as_dict(),
            "treatment": self.treatment.as_dict(),
        }


async def run_pair(
    task: TaskRecord,
    run_index: int,
    llm_config: LLMConfig,
    output_dir: PathLike,
    *,
    baseline: Condition = CONDITION_A,
    treatment: Condition = CONDITION_B1,
    tool_mode: str = "oracle",
    requirements_path: Optional[PathLike] = None,
    budget: Optional[QueryBudget] = None,
    probe: Optional[FingerprintProbe] = None,
) -> PairOutcome:
    """Run both conditions on the same task instance and check the pairing.

    Constants are held fixed by construction: the same task file, the same LLM
    config, the same tool mode, the same step budget and the same orchestrator
    family. The benchmark re-seeds the database before each run, and the
    pre-run fingerprints are compared afterwards. A mismatch marks both records
    as agent errors and the pair is reported as unusable (REQ-010).

    Raises:
        RuntimeError: If the benchmark produced no runs for either condition.
    """
    shared = {
        "tool_mode": tool_mode,
        "requirements_path": requirements_path,
        "budget": budget,
        "probe": probe,
    }
    baseline_record = await run_condition(
        baseline, task, run_index, llm_config, output_dir, **shared
    )
    treatment_record = await run_condition(
        treatment, task, run_index, llm_config, output_dir, **shared
    )

    left = baseline_record.initial_state_fingerprint
    right = treatment_record.initial_state_fingerprint
    if left is None or right is None:
        fingerprint_match = False
        reason = "fingerprint unavailable; pairing could not be verified"
    else:
        fingerprint_match = left == right
        reason = (
            ""
            if fingerprint_match
            else f"initial-state fingerprint mismatch: {left} != {right}"
        )

    outcome = PairOutcome(
        task_id=task.task_id,
        run_index=run_index,
        baseline=baseline_record,
        treatment=treatment_record,
        fingerprint_match=fingerprint_match,
        reason=reason,
    )
    if not fingerprint_match:
        outcome = PairOutcome(
            task_id=outcome.task_id,
            run_index=outcome.run_index,
            baseline=_flag_error(baseline_record, reason),
            treatment=_flag_error(treatment_record, reason),
            fingerprint_match=False,
            reason=reason,
        )
    return outcome


def _flag_error(record: TaskRunRecord, reason: str) -> TaskRunRecord:
    """Return a copy of ``record`` marked as an agent error.

    Used exclusively for pairing-gate failures, so the fingerprint outcome is
    stamped alongside the agent-error flag.
    """
    from dataclasses import replace

    return replace(
        record,
        agent_error=True,
        error_message=reason or "pairing failed",
        fingerprint_match=False,
        fingerprint_mismatch_reason=reason or "pairing failed",
    )


def reconcile_pair_fingerprints(
    baseline_record: TaskRunRecord, treatment_record: TaskRunRecord
) -> Tuple[TaskRunRecord, TaskRunRecord, bool]:
    """Enforce the REQ-010 pairing gate on one matched pair of records.

    The two records of a ``(task_id, run_index)`` pair are accepted only when
    both pre-run fingerprints are present and equal. A missing fingerprint means
    the pairing could not be verified; a differing fingerprint means the two
    arms did not start from the same initial state. In both cases the records
    are flagged as agent errors so ``pair_records`` excludes them from the
    primary metric, matching what :func:`run_pair` does at execution time.

    Returns the (possibly flagged) ``(baseline, treatment, fingerprint_match)``
    triple; it never mutates its inputs.
    """
    left = baseline_record.initial_state_fingerprint
    right = treatment_record.initial_state_fingerprint
    if left is not None and right is not None and left == right:
        from dataclasses import replace

        return (
            replace(baseline_record, fingerprint_match=True, fingerprint_mismatch_reason=None),
            replace(treatment_record, fingerprint_match=True, fingerprint_mismatch_reason=None),
            True,
        )

    if left is None or right is None:
        reason = "fingerprint unavailable; pairing could not be verified"
    else:
        reason = f"initial-state fingerprint mismatch: {left} != {right}"
    return (
        _flag_error(baseline_record, reason),
        _flag_error(treatment_record, reason),
        False,
    )


def reconcile_fingerprint_pairs(
    records: Sequence[TaskRunRecord],
    *,
    conditions: Optional[Sequence[Condition]] = None,
) -> Tuple[List[TaskRunRecord], List[Dict[str, Any]]]:
    """Apply the pairing gate to every matched ``(task_id, run_index)`` pair.

    State Model 2.0: the gate is N-arm aware. ``conditions`` names the arms of
    the sweep (default: baseline vs the broad state-model arm, preserving the
    historical two-arm behaviour). Every non-baseline condition is paired
    against the baseline, so ``A, B1, B2`` reviews both A-B1 and A-B2 pairs.

    Returns the reconciled record list (with flagged replacements) plus one
    plain-dict entry per reviewed pair for the report. Singleton records whose
    condition counterpart is absent are returned untouched and counted as
    ``unpaired``; they are already excluded from ``pair_records`` by the
    intersection of keys.
    """
    # P4 (audit item-7): choose the baseline by *property* (the non-state-model
    # arm), not by position, so the pairing is invariant to condition ordering.
    # Falls back to the historical two-arm A/B1 behaviour when no conditions are
    # supplied.
    if conditions:
        native = [condition for condition in conditions if not condition.state_model]
        baseline_name = native[0].name if native else conditions[0].name
        baseline_set = {condition.name for condition in native} if native else {baseline_name}
        treatment_names = tuple(
            condition.name for condition in conditions if condition.name not in baseline_set
        )
    else:
        baseline_name = CONDITION_A.name
        treatment_names = (CONDITION_B1.name,)

    baseline_by_key: Dict[Tuple[str, int], TaskRunRecord] = {}
    treatments_by_key: Dict[Tuple[str, int], Dict[str, TaskRunRecord]] = {}
    for record in records:
        key = (record.task_id, record.run_index)
        if record.condition == baseline_name:
            baseline_by_key[key] = record
        elif record.condition in treatment_names:
            treatments_by_key.setdefault(key, {})[record.condition] = record

    reconciled = list(records)
    summaries: List[Dict[str, Any]] = []
    reviewed_keys: set = set()
    for key in sorted(set(baseline_by_key) & set(treatments_by_key)):
        baseline_record = baseline_by_key[key]
        for treatment_name in treatment_names:
            treatment_record = treatments_by_key[key].get(treatment_name)
            if treatment_record is None:
                continue
            reviewed_keys.add(key)
            if (
                baseline_record.fingerprint_match is not None
                and treatment_record.fingerprint_match is not None
            ):
                summaries.append(
                    {
                        "task_id": baseline_record.task_id,
                        "run_index": baseline_record.run_index,
                        "fingerprint_match": baseline_record.fingerprint_match,
                        "reason": baseline_record.fingerprint_mismatch_reason or "",
                    }
                )
                continue

            left, right, matched = reconcile_pair_fingerprints(baseline_record, treatment_record)
            reason = "" if matched else (left.error_message or "pairing failed")
            summaries.append(
                {
                    "task_id": left.task_id,
                    "run_index": left.run_index,
                    "fingerprint_match": matched,
                    "reason": reason,
                }
            )
            reconciled = _replaced(reconciled, baseline_record, left)
            reconciled = _replaced(reconciled, treatment_record, right)
    return reconciled, summaries


def _replaced(
    records: Sequence[TaskRunRecord], old: TaskRunRecord, new: TaskRunRecord
) -> List[TaskRunRecord]:
    """Return ``records`` with the first occurrence of ``old`` swapped for ``new``."""
    updated: List[TaskRunRecord] = []
    swapped = False
    for record in records:
        if not swapped and record is old:
            updated.append(new)
            swapped = True
        else:
            updated.append(record)
    return updated


def make_goal_executor(
    eval_set: Sequence[TaskRecord],
    llm_config: LLMConfig,
    output_dir: PathLike,
    *,
    tool_mode: str = "oracle",
    requirements_path: Optional[PathLike] = None,
    budget: Optional[QueryBudget] = None,
    probe: Optional[FingerprintProbe] = None,
    conditions_by_name: Optional[Dict[str, Condition]] = None,
    max_steps: int = MAX_STEPS_DEFAULT,
    archive: PathLike = DEFAULT_ARCHIVE,
) -> Callable[[Any, GoalReporter], Awaitable[TaskRunRecord]]:
    """Adapt the goal loop to :func:`run_condition`.

    The returned coroutine is what ``GoalEngine`` drives. It resolves the task and
    condition from the goal key, so the goal matrix stays the single source of
    truth for what gets executed.
    """
    tasks_by_id = {record.task_id: record for record in eval_set}
    lookup = conditions_by_name or {
        CONDITION_A.name: CONDITION_A,
        CONDITION_B1.name: CONDITION_B1,
    }

    async def executor(goal: Any, reporter: GoalReporter) -> TaskRunRecord:
        task = tasks_by_id.get(goal.task_id)
        if task is None:
            raise KeyError(f"Goal references an unknown task: {goal.task_id}")
        condition = lookup.get(goal.condition)
        if condition is None:
            raise KeyError(f"Goal references an unknown condition: {goal.condition}")
        started = time.perf_counter()
        record = await run_condition(
            condition,
            task,
            goal.run_index,
            llm_config,
            output_dir,
            tool_mode=tool_mode,
            requirements_path=requirements_path,
            budget=budget,
            probe=probe,
            max_steps=max_steps,
            archive=archive,
        )
        reporter.measure("agent", (time.perf_counter() - started) * 1000.0)
        reporter.transition(GoalState.RECORDED, record.run_uid)
        return record

    return executor
