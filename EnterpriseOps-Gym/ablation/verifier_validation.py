"""Phase 3: prove the evaluator is unambiguous before the final A/B run.

The pilot found that four of eleven tasks declare verifiers sharing a ``name``
(3, 1, 5, and 9 duplicated names respectively). ``benchmark/executor.py`` stores
results in a dict keyed by ``verifier.name``, so a collision silently drops a
result - upstream issue #23. The harness also computes an index-keyed view, but
*having* two numbers is not the same as *knowing* they agree.

This module produces ``verifier_validation_report.json`` answering, per task:

* ``verifier_count``  - how many verifiers the task declares;
* ``duplicate_names`` - which names collide, and how many results they destroy;
* ``original_keys``   - the keys the official collapsed summary will hold;
* ``reconciled_keys`` - the positional keys the corrected summary holds;
* ``task_success_*``  - whether collapsing by name can change the final verdict.

The final verdict is the important column. A collision that cannot flip the
outcome is a reporting nuisance; one that can is a validity threat, because the
primary metric is defined as *all verifiers pass*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .results import TaskRunRecord
from .task_registry import resolve_task_config_path
from .verifier_bridge import verifier_key

PathLike = Union[str, Path]

#: Keys inspected, in order, when reading a recorded verifier outcome.
_PASS_KEYS: Tuple[str, ...] = ("passed", "success", "result")


def _declared_verifiers(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Return the task's verifier list, validating its shape.

    Raises:
        ValueError: If ``verifiers`` is absent or is not a list.
    """
    verifiers = payload.get("verifiers")
    if verifiers is None:
        raise ValueError("task config has no 'verifiers' key")
    if not isinstance(verifiers, list):
        raise ValueError("'verifiers' must be a list")
    return [item for item in verifiers if isinstance(item, Mapping)]


def _verifier_name(verifier: Mapping[str, Any], index: int) -> str:
    """The name the benchmark would key on, matching its own substitution."""
    name = verifier.get("name")
    return name if isinstance(name, str) and name else f"verifier_{index + 1}"


def _result_passed(result: Any) -> bool:
    """Whether one recorded verifier result counts as a pass."""
    if isinstance(result, Mapping):
        for key in _PASS_KEYS:
            if key in result:
                return bool(result[key])
        return False
    return bool(result)


def task_success(results: Mapping[str, Any]) -> bool:
    """All-verifiers-passed, the benchmark's own task-success definition.

    An empty result mapping is *not* a success: a task that produced no verifier
    evidence has not demonstrated completion, and counting it as a pass would
    inflate the primary metric.
    """
    if not results:
        return False
    return all(_result_passed(value) for value in results.values())



@dataclass(frozen=True)
class TaskVerifierValidation:
    """Official-vs-indexed reconciliation for a single task.

    Attributes:
        task_id: The task under validation.
        verifier_count: Number of verifiers declared in the task file.
        duplicate_names: Names shared by two or more verifiers, sorted.
        duplicate_name_count: How many names collide.
        lost_to_collision: Declared verifiers minus distinct original keys.
        original_keys: Keys the official collapsed summary holds.
        reconciled_keys: Positional keys the corrected summary holds.
        task_success_original: All-verifiers-passed under the collapsed keys.
        task_success_reconciled: All-verifiers-passed under the positional keys.
        task_success_differs: Whether the collision can flip the verdict.
        observed: Whether a real run record was available to compare.
    """

    task_id: str
    verifier_count: int
    duplicate_names: Tuple[str, ...]
    duplicate_name_count: int
    lost_to_collision: int
    original_keys: Tuple[str, ...]
    reconciled_keys: Tuple[str, ...]
    task_success_original: Optional[bool]
    task_success_reconciled: Optional[bool]
    task_success_differs: Optional[bool]
    observed: bool

    @property
    def is_ambiguous(self) -> bool:
        """Whether this task can produce two different verdicts.

        Ambiguity is *demonstrated* only when a run record shows the two keyings
        disagreeing. Without a record, a duplicate name is reported as a latent
        risk rather than a proven one, so the report never overstates the threat.
        """
        if self.task_success_differs is None:
            return bool(self.duplicate_names)
        return bool(self.task_success_differs)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "verifier_count": self.verifier_count,
            "duplicate_names": list(self.duplicate_names),
            "duplicate_name_count": self.duplicate_name_count,
            "lost_to_collision": self.lost_to_collision,
            "original_keys": list(self.original_keys),
            "reconciled_keys": list(self.reconciled_keys),
            "task_success_original": self.task_success_original,
            "task_success_reconciled": self.task_success_reconciled,
            "task_success_differs": self.task_success_differs,
            "observed": self.observed,
            "is_ambiguous": self.is_ambiguous,
        }


def validate_task_verifiers(
    task_config_path: PathLike,
    *,
    records: Sequence[TaskRunRecord] = (),
    task_id: Optional[str] = None,
) -> TaskVerifierValidation:
    """Reconcile one task's declared verifiers against its observed runs.

    Args:
        task_config_path: Path to an EnterpriseOps-Gym task JSON.
        records: Run records, used to compare the two keyings when available.
        task_id: Override for the report identifier.

    Returns:
        The reconciliation for this task.

    Raises:
        ValueError: If the payload is not a task object, or ``verifiers`` is
            missing or not a list.
    """
    path = resolve_task_config_path(task_config_path)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: task config must be a JSON object")

    verifiers = _declared_verifiers(payload)
    resolved_id = task_id or Path(path).stem

    names = [_verifier_name(verifier, index) for index, verifier in enumerate(verifiers)]
    counts: Dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    duplicates = tuple(sorted(name for name, count in counts.items() if count > 1))

    original: Optional[bool] = None
    reconciled: Optional[bool] = None
    for record in records:
        if record.task_id != resolved_id:
            continue
        if record.verifier_results_collapsed:
            original = task_success(record.verifier_results_collapsed)
        if record.verifier_results_indexed:
            reconciled = task_success(record.verifier_results_indexed)
        if original is not None and reconciled is not None:
            break

    differs = (
        None if (original is None or reconciled is None) else (original != reconciled)
    )

    return TaskVerifierValidation(
        task_id=resolved_id,
        verifier_count=len(verifiers),
        duplicate_names=duplicates,
        duplicate_name_count=len(duplicates),
        lost_to_collision=max(0, len(verifiers) - len(counts)),
        original_keys=tuple(sorted(counts)),
        reconciled_keys=tuple(
            verifier_key(dict(verifier), index)
            for index, verifier in enumerate(verifiers)
        ),
        task_success_original=original,
        task_success_reconciled=reconciled,
        task_success_differs=differs,
        observed=original is not None and reconciled is not None,
    )



def build_verifier_validation_report(
    task_config_paths: Sequence[PathLike],
    *,
    records: Sequence[TaskRunRecord] = (),
    task_ids: Optional[Mapping[PathLike, str]] = None,
) -> Dict[str, Any]:
    """Validate every task's verifier set and summarise the evaluator risk.

    Args:
        task_config_paths: Task JSON paths forming the evaluation set.
        records: Optional run records, used to check observed equivalence.
        task_ids: Optional explicit id per path. Records are matched against the
            resolved id, so an override must be supplied here too when the
            manifest id differs from the file stem.

    Returns:
        A JSON-serialisable report. ``evaluator_unambiguous`` is the Phase 3
        gate, and it is ``True`` only when no task's duplicate names were
        *observed* to change a final verdict. Unobserved tasks are reported as
        latent risk rather than as a proven threat.

    Raises:
        ValueError: If the same task id appears twice, which would silently drop
            a task from the evaluation set.
    """
    validations: List[TaskVerifierValidation] = []
    seen_ids = set()
    for path in task_config_paths:
        validation = validate_task_verifiers(
            path, records=records, task_id=(task_ids or {}).get(path)
        )
        if validation.task_id in seen_ids:
            raise ValueError(f"Duplicate task id in evaluation set: {validation.task_id}")
        seen_ids.add(validation.task_id)
        validations.append(validation)

    validations.sort(key=lambda item: item.task_id)
    with_duplicates = [item for item in validations if item.duplicate_names]
    observed_divergent = [
        item for item in validations if item.task_success_differs is True
    ]

    return {
        "tasks": len(validations),
        "total_verifiers": sum(item.verifier_count for item in validations),
        "tasks_with_duplicate_names": len(with_duplicates),
        "tasks_losing_results_to_collision": sum(
            item.lost_to_collision for item in validations
        ),
        "tasks_observed": sum(1 for item in validations if item.observed),
        "evaluator_unambiguous": not observed_divergent,
        "latent_risk_task_ids": [
            item.task_id for item in validations if item.is_ambiguous
        ],
        "observed_divergent_task_ids": [item.task_id for item in observed_divergent],
        "per_task": [item.as_dict() for item in validations],
    }


def render_verifier_validation_markdown(report: Mapping[str, Any]) -> str:
    """Render the Phase 3 reconciliation as a Markdown table."""
    lines = [
        "| Task | Verifiers | Duplicate names | Lost | Verdict differs | Verdict |",
        "|---|---|---|---:|---|---|",
    ]
    for item in report.get("per_task") or ():
        differs = item.get("task_success_differs")
        verdict = "ambiguous" if item.get("is_ambiguous") else "consistent"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("task_id", ""))[-12:],
                    str(item.get("verifier_count", 0)),
                    ", ".join(item.get("duplicate_names") or []) or "-",
                    str(item.get("lost_to_collision", 0)),
                    "n/a" if differs is None else ("yes" if differs else "no"),
                    verdict,
                ]
            )
            + " |"
        )
    return "\n".join(lines)