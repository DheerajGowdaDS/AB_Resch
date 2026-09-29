"""The persisted experiment record: one JSON file per ``(task, run, condition)``.

Records are append-only and resumable: ``is_recorded`` mirrors
``benchmark_utils.skip_sample`` so an interrupted sweep restarts cheaply. LLM
credentials are redacted before anything is written.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

PathLike = Union[str, Path]

#: Replacement written instead of a credential value.
REDACTED = "***REDACTED***"

#: Field names whose values must never be persisted.
_SECRET_FIELDS = frozenset(
    {"llm_api_key", "api_key", "apikey", "token", "access_token", "auth_token", "password"}
)

INDEX_FILENAME = "index.jsonl"

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def new_run_uid() -> str:
    """Return a fresh run identifier."""
    return uuid.uuid4().hex


def utc_now() -> str:
    """Return an aware UTC timestamp in ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat()


def redact(value: Any) -> Any:
    """Recursively replace credential-looking values with :data:`REDACTED`."""
    if isinstance(value, Mapping):
        redacted: Dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in _SECRET_FIELDS:
                redacted[str(key)] = REDACTED
            else:
                redacted[str(key)] = redact(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


def run_filename(task_id: str, condition: str, run_index: int) -> str:
    """Deterministic, filesystem-safe record name."""
    safe_task = _UNSAFE_FILENAME.sub("_", task_id)
    safe_condition = _UNSAFE_FILENAME.sub("_", condition)
    return f"run__{safe_task}__{safe_condition}__r{int(run_index)}.json"


@dataclass(frozen=True)
class TaskRunRecord:
    """Everything measured about one experimental unit.

    ``overall_success`` uses the benchmark's own definition (all verifiers pass).
    Both the official collapsed verifier summary and the index-keyed corrected
    summary are kept so the report can reconcile them (GUD-007).
    """

    experiment: str
    run_uid: str
    condition: str
    task_id: str
    run_index: int
    complexity_category: str
    seed_database_file: str
    runtime_database_id: Optional[str]
    initial_state_fingerprint: Optional[str]
    model_provider: str
    model_name: str
    temperature: float
    max_steps: int
    tool_mode: str
    orchestrator: str
    overall_success: bool
    verifier_summary_collapsed: Dict[str, Any]
    verifier_summary_indexed: Dict[str, Any]
    verifier_results_collapsed: Dict[str, Any]
    verifier_results_indexed: Dict[str, Any]
    agent_error: bool
    error_message: Optional[str]
    steps_taken: int
    tool_call_count: int
    read_query_count: int
    invalid_action_count: int
    latency_ms: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    state_retrieval: Optional[Dict[str, Any]]
    started_at: str
    finished_at: str
    goal_states: Tuple[str, ...] = ()
    timings_ms: Dict[str, float] = field(default_factory=dict)
    #: Blueprint Phase 1. Which experimental system produced this record.
    #: ``V1-PREFIX-PILOT`` records predate the anchor/route/SQL repairs and must
    #: never be pooled with earlier pilot/fixed-version records.
    experiment_version: Optional[str] = None
    #: Digest of the :class:`~ablation.experiment_manifest.ExperimentManifest`
    #: that produced this record, so any result is traceable to its exact
    #: configuration.
    manifest_hash: Optional[str] = None
    #: Outcome of the REQ-010 pairing gate. ``None`` means not yet reviewed;
    #: ``False`` carries the reason in ``fingerprint_mismatch_reason``.
    fingerprint_match: Optional[bool] = None
    fingerprint_mismatch_reason: Optional[str] = None

    @property
    def key(self) -> Tuple[str, str, int]:
        return (self.task_id, self.condition, self.run_index)

    @property
    def verifier_pass_rate(self) -> Optional[float]:
        """Pass rate of the official collapsed summary when it has any entries."""
        for summary in (self.verifier_summary_collapsed, self.verifier_summary_indexed):
            total = int(summary.get("total", 0) or 0)
            if total:
                return float(summary.get("passed", 0) or 0) / total
        return None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "experiment": self.experiment,
            "run_uid": self.run_uid,
            "condition": self.condition,
            "task_id": self.task_id,
            "run_index": self.run_index,
            "complexity_category": self.complexity_category,
            "seed_database_file": self.seed_database_file,
            "runtime_database_id": self.runtime_database_id,
            "initial_state_fingerprint": self.initial_state_fingerprint,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_steps": self.max_steps,
            "tool_mode": self.tool_mode,
            "orchestrator": self.orchestrator,
            "overall_success": bool(self.overall_success),
            "verifier_summary_collapsed": dict(self.verifier_summary_collapsed),
            "verifier_summary_indexed": dict(self.verifier_summary_indexed),
            "verifier_results_collapsed": dict(self.verifier_results_collapsed),
            "verifier_results_indexed": dict(self.verifier_results_indexed),
            "agent_error": bool(self.agent_error),
            "error_message": self.error_message,
            "steps_taken": int(self.steps_taken),
            "tool_call_count": int(self.tool_call_count),
            "read_query_count": int(self.read_query_count),
            "invalid_action_count": int(self.invalid_action_count),
            "latency_ms": int(self.latency_ms),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "state_retrieval": dict(self.state_retrieval) if self.state_retrieval else None,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "goal_states": list(self.goal_states),
            "timings_ms": dict(self.timings_ms),
            "experiment_version": self.experiment_version,
            "manifest_hash": self.manifest_hash,
            "fingerprint_match": self.fingerprint_match,
            "fingerprint_mismatch_reason": self.fingerprint_mismatch_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskRunRecord":
        missing = [
            name
            for name in ("condition", "task_id", "run_index", "overall_success")
            if name not in payload
        ]
        if missing:
            raise ValueError(f"TaskRunRecord payload is missing {missing}")

        def get(key: str, default: Any = None) -> Any:
            return payload.get(key, default)

        return cls(
            experiment=get("experiment", "experiment_1"),
            run_uid=get("run_uid") or new_run_uid(),
            condition=payload["condition"],
            task_id=payload["task_id"],
            run_index=int(payload["run_index"]),
            complexity_category=get("complexity_category", ""),
            seed_database_file=get("seed_database_file", ""),
            runtime_database_id=get("runtime_database_id"),
            initial_state_fingerprint=get("initial_state_fingerprint"),
            model_provider=get("model_provider", ""),
            model_name=get("model_name", ""),
            temperature=float(get("temperature", 0.0) or 0.0),
            max_steps=int(get("max_steps", 0) or 0),
            tool_mode=get("tool_mode", "oracle"),
            orchestrator=get("orchestrator", "react"),
            overall_success=bool(payload["overall_success"]),
            verifier_summary_collapsed=dict(get("verifier_summary_collapsed", {}) or {}),
            verifier_summary_indexed=dict(get("verifier_summary_indexed", {}) or {}),
            verifier_results_collapsed=dict(get("verifier_results_collapsed", {}) or {}),
            verifier_results_indexed=dict(get("verifier_results_indexed", {}) or {}),
            agent_error=bool(get("agent_error", False)),
            error_message=get("error_message"),
            steps_taken=int(get("steps_taken", 0) or 0),
            tool_call_count=int(get("tool_call_count", 0) or 0),
            read_query_count=int(get("read_query_count", 0) or 0),
            invalid_action_count=int(get("invalid_action_count", 0) or 0),
            latency_ms=int(get("latency_ms", 0) or 0),
            input_tokens=get("input_tokens"),
            output_tokens=get("output_tokens"),
            state_retrieval=get("state_retrieval"),
            started_at=get("started_at", ""),
            finished_at=get("finished_at", ""),
            goal_states=tuple(get("goal_states", ()) or ()),
            timings_ms=dict(get("timings_ms", {}) or {}),
            fingerprint_match=get("fingerprint_match"),
            fingerprint_mismatch_reason=get("fingerprint_mismatch_reason"),
            experiment_version=get("experiment_version"),
            manifest_hash=get("manifest_hash"),
        )


def record_path(output_dir: PathLike, task_id: str, condition: str, run_index: int) -> Path:
    """Absolute path of one record file."""
    return Path(output_dir) / run_filename(task_id, condition, run_index)


def is_recorded(output_dir: PathLike, task_id: str, condition: str, run_index: int) -> bool:
    """Whether a record already exists, enabling resumable sweeps."""
    return record_path(output_dir, task_id, condition, run_index).exists()


def write_run_record(record: TaskRunRecord, output_dir: PathLike) -> Path:
    """Persist one record and append it to the run index.

    Credentials are redacted first. The JSON file is the full record; the index
    line is the resumable ledger.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = record_path(directory, record.task_id, record.condition, record.run_index)
    target.write_text(
        json.dumps(redact(record.as_dict()), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    index_entry = {
        "task_id": record.task_id,
        "condition": record.condition,
        "run_index": record.run_index,
        "run_uid": record.run_uid,
        "overall_success": record.overall_success,
        "agent_error": record.agent_error,
        "fingerprint_match": record.fingerprint_match,
        "file": target.name,
        "complexity_category": record.complexity_category,
    }
    with open(directory / INDEX_FILENAME, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(index_entry, sort_keys=True, default=str) + "\n")
    return target


def load_run_records(output_dir: PathLike) -> List[TaskRunRecord]:
    """Load every persisted record from an output directory, sorted."""
    directory = Path(output_dir)
    if not directory.is_dir():
        return []
    records: List[TaskRunRecord] = []
    for path in sorted(directory.glob("run__*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        records.append(TaskRunRecord.from_dict(payload))
    return records
