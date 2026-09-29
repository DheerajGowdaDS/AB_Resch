"""Versioned experiment manifest: which binary produced which number.

Blueprint Phase 1. The 11-task pilot was executed by code that predates the
state-model repairs, so its 22 persisted records are **not** attributable to the
current intervention. Mixing them with post-fix runs would silently corrupt
every comparison, because the two populations were produced by different
systems.

This module makes that distinction structural rather than documentary:

* named *version labels* (:data:`VERSION_PREFIX_PILOT`,
  :data:`VERSION_STATE_MODEL_FIXED`, :data:`VERSION_STATE_MODEL_REFINED`) so a record states which system produced it;
* a frozen :class:`ExperimentManifest` capturing every constant the experiment
  holds fixed - model, temperature, orchestrator, tool mode, step budget, seed -
  plus content hashes of the code, the intervention, the benchmark, and the task
  set;
* a :attr:`ExperimentManifest.manifest_hash`, stamped into every run record, so
  any result can be traced to the exact configuration that produced it.

The hash is over the manifest's *content*, not its file, so it is stable across
machines and safe to compare between runs.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from .parity import hash_package, sha256_file

PathLike = Union[str, Path]

GYM_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = GYM_ROOT.parent

#: The pre-repair pilot. Its 22 records were produced before the anchor, route,
#: and SQL-escaping fixes, and must never be pooled with post-fix records.
VERSION_PREFIX_PILOT = "V1-PREFIX-PILOT"

#: The repaired intervention: per-tier anchor budget, identity-aware table
#: ordering, possessive decomposition, hop ladder, and pre-execution SQL
#: validation.
VERSION_STATE_MODEL_FIXED = "V1.1-STATE-MODEL-FIXED"
VERSION_STATE_MODEL_REFINED = "V1.2-STATE-MODEL-REFINED"

#: Every recognised version label, oldest first.
VERSION_LABELS: Tuple[str, ...] = (VERSION_PREFIX_PILOT, VERSION_STATE_MODEL_FIXED, VERSION_STATE_MODEL_REFINED)


@dataclass(frozen=True)
class ExperimentManifest:
    """Immutable description of one experimental configuration.

    Attributes:
        experiment_id: Stable experiment name, e.g. ``experiment_1``.
        version: One of :data:`VERSION_LABELS`.
        model_provider: LLM provider, e.g. ``openrouter``.
        model: LLM model identifier.
        temperature: Sampling temperature; Experiment 1 fixes this at 0.
        orchestrator: Agent loop; Experiment 1 permits only ``react``.
        tool_mode: Tool selection mode; Experiment 1 fixes ``oracle``.
        max_steps: Step budget, identical across arms by construction.
        seed: Random seed for the sweep.
        csm_env_schema_version: Schema version of the intervention.
        csm_env_sha256: Digest over the intervention package.
        benchmark_sha256: Digest over the benchmark/evaluator modules.
        ablation_sha256: Digest over the harness modules.
        task_set_sha256: Digest over the task JSON files.
        seed_archive_sha256: Digest of ``gym_dbs.zip``.
        interpreter: Python version string.
        platform: Platform string.
        num_tasks: Task count in this configuration.
        notes: Free-form provenance notes.
    """

    experiment_id: str
    version: str
    model_provider: str
    model: str
    temperature: float
    orchestrator: str
    tool_mode: str
    max_steps: int
    seed: int
    csm_env_schema_version: str
    csm_env_sha256: str
    benchmark_sha256: str
    ablation_sha256: str
    task_set_sha256: str
    seed_archive_sha256: str
    interpreter: str
    platform: str
    num_tasks: int = 0
    notes: str = ""
    task_ids: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.version not in VERSION_LABELS:
            raise ValueError(
                f"Unknown experiment version {self.version!r}; "
                f"expected one of {list(VERSION_LABELS)}"
            )
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")

    @property
    def manifest_hash(self) -> str:
        """Stable digest of the manifest content.

        Excludes ``notes`` and ``task_ids`` ordering noise is normalised, so two
        manifests describing the same experimental setup hash identically.
        """
        payload = self.as_dict(include_hash=False)
        payload.pop("notes", None)
        payload["task_ids"] = sorted(self.task_ids)
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self, *, include_hash: bool = True) -> Dict[str, Any]:
        """Serialise the manifest, optionally including its hash."""
        payload = asdict(self)
        payload["task_ids"] = list(self.task_ids)
        if include_hash:
            payload["manifest_hash"] = self.manifest_hash
        return payload


def _digest_of_hashes(hashes: Mapping[str, str]) -> str:
    """Collapse a per-file digest mapping into one stable digest."""
    digest = hashlib.sha256()
    for name in sorted(hashes):
        digest.update(name.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(hashes[name].encode("utf-8"))
    return digest.hexdigest()


def _hash_directory_modules(directory: Path) -> Dict[str, str]:
    """Hash every ``.py`` module directly inside ``directory``."""
    return {
        path.name: sha256_file(path)
        for path in sorted(directory.glob("*.py"))
        if path.is_file()
    }


def build_manifest(
    *,
    experiment_id: str,
    version: str,
    model_provider: str,
    model: str,
    temperature: float,
    orchestrator: str,
    tool_mode: str,
    max_steps: int,
    seed: int,
    task_ids: Sequence[str] = (),
    eval_set_path: Optional[PathLike] = None,
    seed_archive: Optional[PathLike] = None,
    gym_root: PathLike = GYM_ROOT,
    notes: str = "",
) -> ExperimentManifest:
    """Construct a manifest by hashing the real workspace.

    Args:
        experiment_id: Experiment name.
        version: One of :data:`VERSION_LABELS`.
        model_provider: LLM provider.
        model: LLM model identifier.
        temperature: Sampling temperature.
        orchestrator: Agent loop identifier.
        tool_mode: Tool selection mode.
        max_steps: Step budget.
        seed: Sweep seed.
        task_ids: Task identifiers participating in this configuration.
        eval_set_path: Optional manifest whose task files are hashed.
        seed_archive: Optional ``gym_dbs.zip`` to hash.
        gym_root: Repository root used to resolve relative paths.
        notes: Free-form provenance notes.

    Returns:
        A fully populated :class:`ExperimentManifest`.

    Raises:
        ValueError: If the version label is unknown or ``max_steps`` < 1.
    """
    from csm_env import SchemaRegistry

    root = Path(gym_root)
    csm_env_dir = root / "csm_env" if (root / "csm_env").is_dir() else WORKSPACE_ROOT / "csm_env"

    task_hashes: Dict[str, str] = {}
    if eval_set_path is not None and Path(eval_set_path).is_file():
        from .task_registry import load_eval_set, resolve_task_config_path

        manifest = load_eval_set(eval_set_path)
        for record in manifest:
            path = resolve_task_config_path(record.task_config_path)
            if path.is_file():
                task_hashes[record.task_id] = sha256_file(path)

    archive_digest = ""
    if seed_archive is not None and Path(seed_archive).is_file():
        archive_digest = sha256_file(seed_archive)

    return ExperimentManifest(
        experiment_id=experiment_id,
        version=version,
        model_provider=model_provider,
        model=model,
        temperature=float(temperature),
        orchestrator=orchestrator,
        tool_mode=tool_mode,
        max_steps=int(max_steps),
        seed=int(seed),
        csm_env_schema_version=SchemaRegistry.from_static().manifest.schema_version,
        csm_env_sha256=_digest_of_hashes(hash_package(csm_env_dir)),
        benchmark_sha256=_digest_of_hashes(
            _hash_directory_modules(root / "benchmark")
        ),
        ablation_sha256=_digest_of_hashes(
            _hash_directory_modules(root / "ablation")
        ),
        task_set_sha256=_digest_of_hashes(task_hashes),
        seed_archive_sha256=archive_digest,
        interpreter=f"{platform.python_implementation()} {platform.python_version()}",
        platform=platform.platform(),
        num_tasks=len(task_ids) or len(task_hashes),
        notes=notes,
        task_ids=tuple(task_ids),
    )


def write_manifest(manifest: ExperimentManifest, output_dir: PathLike) -> Path:
    """Persist ``manifest.json`` into ``output_dir`` and return its path."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "manifest.json"
    target.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return target


def assert_single_version(records: Sequence[Any]) -> Dict[str, Any]:
    """Fail when a record directory mixes experimental versions.

    Pooling ``V1-PREFIX-PILOT`` records with ``V1.1-STATE-MODEL-FIXED`` records
    would compare two different interventions and attribute the difference to the
    state model. This is the enforcement point for that invariant.

    Args:
        records: Persisted run records carrying an ``experiment_version``.

    Returns:
        A summary of the versions present.

    Raises:
        ValueError: If more than one version label is present.
    """
    versions: Dict[str, int] = {}
    unversioned = 0
    for record in records:
        version = getattr(record, "experiment_version", None)
        if not version:
            unversioned += 1
            continue
        versions[version] = versions.get(version, 0) + 1
    if len(versions) > 1:
        raise ValueError(
            "records mix experimental versions "
            f"({versions}); the pre-fix pilot and the repaired intervention must "
            "be analysed separately"
        )
    return {"versions": versions, "unversioned": unversioned}


def load_manifest(path: PathLike) -> ExperimentManifest:
    """Load a manifest previously written by :func:`write_manifest`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload.pop("manifest_hash", None)
    payload["task_ids"] = tuple(payload.get("task_ids", ()))
    return ExperimentManifest(**payload)
