"""Pre-run validation of a task's verifier set.

``BenchmarkExecutor._run_verifiers`` stores results in a dict keyed by
``verifier.name`` (see ``benchmark/executor.py``), so verifiers that share a
name silently overwrite each other. Upstream tracks this as issue #23.

This module *measures* the collision instead of editing the benchmark, so the
official collapsed summary can be reported alongside a corrected one.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from csm_env.schema_spec.registry import SchemaRegistry
from .task_registry import resolve_task_config_path

#: Verifier types the benchmark's ``VerifierEngine`` implements.
SUPPORTED_VERIFIER_TYPES: Tuple[str, ...] = (
    "database_state",
    "response_check",
    "tool_execution",
)

PathLike = Union[str, Path]


@dataclass(frozen=True)
class VerifierSetReport:
    """Validation outcome for one task's verifier list.

    Attributes:
        task_id: Task the verifiers belong to.
        declared_count: Number of verifiers declared in the task file.
        unique_name_count: Number of distinct ``name`` values (or positional
            names when ``name`` is absent).
        duplicate_names: Sorted names that appear more than once. Non-empty means
            the official collapsed summary under-reports the verifier count.
        positional_names: Verifiers that omit ``name``; the benchmark substitutes
            ``verifier_<i+1>`` for these.
        unsupported_verifier_types: Types outside ``SUPPORTED_VERIFIER_TYPES``.
        missing_gym_name_count: Verifiers without ``gym_name``, which makes the
            benchmark fall back to the single-gym database.
        declared_tables: Tables referenced by verifier SQL that exist in the CSM
            schema registry. Analysis-only information.
    """

    task_id: str
    declared_count: int
    unique_name_count: int
    duplicate_names: Tuple[str, ...]
    positional_names: Tuple[int, ...]
    unsupported_verifier_types: Tuple[str, ...]
    missing_gym_name_count: int
    declared_tables: Tuple[str, ...]

    @property
    def has_duplicate_names(self) -> bool:
        return bool(self.duplicate_names)

    @property
    def collapsed_count(self) -> int:
        """Entry count the official dict-keyed summary will actually hold."""
        return self.unique_name_count

    @property
    def is_clean(self) -> bool:
        return not self.has_duplicate_names and not self.unsupported_verifier_types

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "declared_count": self.declared_count,
            "unique_name_count": self.unique_name_count,
            "collapsed_count": self.collapsed_count,
            "duplicate_names": list(self.duplicate_names),
            "positional_names": list(self.positional_names),
            "unsupported_verifier_types": list(self.unsupported_verifier_types),
            "missing_gym_name_count": self.missing_gym_name_count,
            "declared_tables": list(self.declared_tables),
            "is_clean": self.is_clean,
        }


def _read_payload(task_config_path: PathLike) -> Dict[str, Any]:
    resolved_path = resolve_task_config_path(task_config_path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{resolved_path}: task config must be a JSON object")
    return payload


def _verifier_name(verifier: Dict[str, Any], index: int) -> str:
    """Mirror the benchmark's naming rule for a verifier entry."""
    name = verifier.get("name")
    return name if isinstance(name, str) and name else f"verifier_{index + 1}"


def _referenced_tables(sql: str) -> List[str]:
    """Extract table names from a verifier SQL statement."""
    import re

    if not sql:
        return []
    found: List[str] = []
    for match in re.finditer(
        r"\b(?:FROM|JOIN|UPDATE|INTO)\s+([A-Za-z_][A-Za-z0-9_]*)", sql, re.IGNORECASE
    ):
        table = match.group(1).lower()
        if table not in found:
            found.append(table)
    return found


def validate_verifier_set(
    task_config_path: PathLike,
    *,
    registry: Optional[SchemaRegistry] = None,
    task_id: Optional[str] = None,
) -> VerifierSetReport:
    """Validate one task's verifier list without executing anything.

    Args:
        task_config_path: Path to an EnterpriseOps-Gym task JSON.
        registry: Optional schema registry, used to keep only known CSM tables
            in ``declared_tables``.
        task_id: Override for the report's task identifier.

    Raises:
        ValueError: If the payload is not a task object, or ``verifiers`` is
            missing or not a list.
    """
    payload = _read_payload(task_config_path)
    if "verifiers" not in payload:
        raise ValueError(f"{task_config_path}: task config has no 'verifiers' key")
    verifiers = payload["verifiers"]
    if not isinstance(verifiers, list):
        raise ValueError(f"{task_config_path}: 'verifiers' must be a list")

    names: List[str] = []
    positional: List[int] = []
    unsupported: List[str] = []
    missing_gym = 0
    tables: List[str] = []

    for index, verifier in enumerate(verifiers):
        if not isinstance(verifier, dict):
            raise ValueError(
                f"{task_config_path}: verifier at index {index} is not an object"
            )
        names.append(_verifier_name(verifier, index))
        if not (isinstance(verifier.get("name"), str) and verifier["name"]):
            positional.append(index)

        verifier_type = verifier.get("verifier_type")
        if verifier_type not in SUPPORTED_VERIFIER_TYPES:
            unsupported.append(str(verifier_type))
        if not verifier.get("gym_name"):
            missing_gym += 1
        validation_config = verifier.get("validation_config") or {}
        if isinstance(validation_config, dict):
            for sql_key in ("query", "sql_query"):
                sql = validation_config.get(sql_key)
                if isinstance(sql, str):
                    for table in _referenced_tables(sql):
                        if table not in tables:
                            tables.append(table)

    known_tables = tables
    if registry is not None:
        known_tables = [table for table in tables if registry.table(table) is not None]

    return VerifierSetReport(
        task_id=task_id or Path(task_config_path).stem,
        declared_count=len(verifiers),
        unique_name_count=len(set(names)),
        duplicate_names=tuple(sorted(name for name, c in Counter(names).items() if c > 1)),
        positional_names=tuple(positional),
        unsupported_verifier_types=tuple(sorted(set(unsupported))),
        missing_gym_name_count=missing_gym,
        declared_tables=tuple(known_tables),
    )


def validate_eval_set_verifiers(
    task_config_paths: List[PathLike],
    *,
    registry: Optional[SchemaRegistry] = None,
) -> Dict[str, VerifierSetReport]:
    """Validate every task in an evaluation set, keyed by task id.

    Raises:
        ValueError: On duplicate task ids across the supplied paths.
    """
    reports: Dict[str, VerifierSetReport] = {}
    for path in task_config_paths:
        report = validate_verifier_set(path, registry=registry, task_id=Path(path).stem)
        if report.task_id in reports:
            raise ValueError(f"Duplicate task id in evaluation set: {report.task_id}")
        reports[report.task_id] = report
    return reports
