"""Readiness gate and revision pinning.

Two modes, deliberately separated:

* **offline** - manifest, schema, parity, archive, verifier and config checks
  that need neither the CSM server nor an LLM. This is what ``--dry-run`` uses,
  so a misconfigured sweep is caught before any tokens are spent.
* **live** - :func:`resolve_live_anchors` seeds a task database, fingerprints it,
  and resolves each task's GGQR anchor from its prompt. It only ever issues
  read-only SQL and deletes the database it created.
"""

from __future__ import annotations

import platform
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .parity import compare_csm_env_copies, sha256_file
from .task_registry import EvalSetManifest, build_requirements_manifest, save_json, resolve_task_config_path, with_anchor
from .verify_pinning import validate_eval_set_verifiers

PathLike = Union[str, Path]

GYM_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = GYM_ROOT.parent

#: Files whose content defines the evaluator, pinned by hash because this
#: checkout carries no git metadata (CON-003).
PINNED_BENCHMARK_FILES: Tuple[str, ...] = (
    "benchmark/executor.py",
    "benchmark/verifier.py",
    "benchmark/mcp_client.py",
    "benchmark/models.py",
    "benchmark/llm_client.py",
    "orchestrators/react.py",
    "orchestrators/base.py",
    "compute_score.py",
    "evaluate.py",
    "benchmark_utils.py",
    "csm_integration.py",
    "utils/task_queue_worker.py",
)

DEFAULT_SEED_ARCHIVE = "gym_dbs.zip"


@dataclass(frozen=True)
class PreflightCheck:
    """One readiness check."""

    name: str
    ok: bool
    detail: str
    blocking: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "blocking": self.blocking,
        }


@dataclass
class PreflightReport:
    """Aggregate readiness verdict."""

    checks: List[PreflightCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks if check.blocking)

    @property
    def failures(self) -> List[PreflightCheck]:
        return [check for check in self.checks if not check.ok and check.blocking]

    def add(self, name: str, ok: bool, detail: str, *, blocking: bool = True) -> None:
        self.checks.append(PreflightCheck(name, ok, detail, blocking))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [check.as_dict() for check in self.checks],
            "failures": [check.name for check in self.failures],
        }

    def render(self) -> str:
        rows = []
        for check in self.checks:
            status = "PASS" if check.ok else ("WARN" if not check.blocking else "FAIL")
            rows.append(f"{status:<5} {check.name}: {check.detail}")
        rows.append(f"PREFLIGHT {'PASS' if self.ok else 'FAIL'}")
        return "\n".join(rows)


def check_schema(registry: Any, report: PreflightReport) -> None:
    """Confirm the CSM schema registry loads and is non-empty."""
    try:
        tables = registry.tables()
        report.add(
            "csm_env_schema",
            len(tables) > 0,
            f"schema {registry.manifest.schema_version} with {len(tables)} tables",
        )
    except Exception as exc:  # noqa: BLE001 - a check must not abort the gate
        report.add("csm_env_schema", False, f"{type(exc).__name__}: {exc}")


def check_parity(report: PreflightReport) -> None:
    """Confirm the two ``csm_env`` copies are identical."""
    try:
        parity = compare_csm_env_copies(WORKSPACE_ROOT / "csm_env", GYM_ROOT / "csm_env")
        report.add(
            "csm_env_parity",
            parity.is_identical,
            (
                f"{parity.identical_count} modules identical"
                if parity.is_identical
                else f"diverged: only_root={list(parity.only_in_root)} "
                f"only_gym={list(parity.only_in_gym)} changed={list(parity.changed)}"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        report.add("csm_env_parity", False, f"{type(exc).__name__}: {exc}")


def check_seed_archive(
    eval_set: EvalSetManifest, report: PreflightReport, archive: PathLike = DEFAULT_SEED_ARCHIVE
) -> None:
    """Confirm every task's seed SQL exists inside the archive."""
    archive_path = Path(archive)
    if not archive_path.is_absolute():
        archive_path = GYM_ROOT / archive_path
    if not archive_path.is_file():
        report.add("seed_archive", False, f"missing archive: {archive_path}")
        return
    try:
        with zipfile.ZipFile(archive_path) as bundle:
            names = set(bundle.namelist())
    except zipfile.BadZipFile as exc:
        report.add("seed_archive", False, f"corrupt archive {archive_path}: {exc}")
        return

    missing = [record.task_id for record in eval_set if record.seed_database_file not in names]
    report.add(
        "seed_archive",
        not missing,
        f"{len(eval_set) - len(missing)}/{len(eval_set)} seed files present"
        + (f"; missing for {missing}" if missing else ""),
    )


def check_eval_set(eval_set: EvalSetManifest, report: PreflightReport) -> None:
    """Validate the evaluation set and every task's verifier set."""
    report.add(
        "eval_set",
        len(eval_set) > 0,
        f"{len(eval_set)} tasks, categories {eval_set.category_counts()}",
    )
    try:
        reports = validate_eval_set_verifiers([r.task_config_path for r in eval_set])
    except Exception as exc:  # noqa: BLE001
        report.add("verifier_sets", False, f"{type(exc).__name__}: {exc}")
        return

    collisions = [item for item in reports.values() if item.has_duplicate_names]
    report.add(
        "verifier_sets",
        not collisions,
        f"{len(reports)} tasks validated; {len(collisions)} with duplicate verifier names "
        "(reconciled via index keying, not hidden)",
        blocking=False,
    )


def check_llm_config(path: Optional[PathLike], report: PreflightReport) -> None:
    """Confirm an LLM config exists and carries a model and a key."""
    if path is None:
        report.add("llm_config", False, "no --llm-config supplied", blocking=False)
        return
    config_path = Path(path)
    if not config_path.is_file():
        report.add("llm_config", False, f"missing file: {config_path}", blocking=False)
        return
    try:
        from benchmark_utils import load_llm_configs

        configs = load_llm_configs(str(config_path))
    except Exception as exc:  # noqa: BLE001
        report.add("llm_config", False, f"{type(exc).__name__}: {exc}", blocking=False)
        return
    ok = all(config.llm_model and config.llm_api_key for config in configs)
    report.add(
        "llm_config",
        ok,
        f"{len(configs)} model(s): "
        + ", ".join(f"{c.llm_provider}/{c.llm_model}@T{c.temperature}" for c in configs),
        blocking=False,
    )


async def check_server(base_url: str, report: PreflightReport) -> None:
    """Probe the CSM server health endpoint. Read-only, non-blocking."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url.rstrip('/')}/health")
        report.add(
            "csm_server",
            response.status_code == 200,
            f"{base_url} -> HTTP {response.status_code}",
            blocking=False,
        )
    except Exception as exc:  # noqa: BLE001
        report.add(
            "csm_server",
            False,
            f"{base_url} unreachable: {type(exc).__name__}",
            blocking=False,
        )


def write_pinned_revisions(
    eval_set: EvalSetManifest,
    *,
    registry: Any,
    archive: PathLike = DEFAULT_SEED_ARCHIVE,
    gym_root: PathLike = GYM_ROOT,
    reference_csm_task_count: int = 186,
) -> Dict[str, Any]:
    """Build the revision pin payload for the whole experiment.

    Revisions are content hashes, because this checkout has no git metadata. The
    dataset shortfall against the published CSM task count is recorded here
    rather than left implicit (CON-004).
    """
    root = Path(gym_root)
    archive_path = Path(archive)
    if not archive_path.is_absolute():
        archive_path = root / archive_path

    benchmark_hashes: Dict[str, str] = {}
    for relative in PINNED_BENCHMARK_FILES:
        path = root / relative
        benchmark_hashes[relative] = sha256_file(path) if path.is_file() else "missing"

    ablation_hashes: Dict[str, str] = {}
    ablation_root = root / "ablation"
    for path in sorted(ablation_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        ablation_hashes[relative] = sha256_file(path)

    task_hashes = {
        record.task_id: sha256_file(resolve_task_config_path(record.task_config_path))
        for record in eval_set
    }
    try:
        parity = compare_csm_env_copies(WORKSPACE_ROOT / "csm_env", GYM_ROOT / "csm_env")
    except Exception:  # noqa: BLE001 - parity has its own check
        parity = None

    parity_payload = parity.as_dict() if parity is not None else None
    if parity_payload is not None:
        parity_payload["root_package"] = "../csm_env"
        parity_payload["gym_package"] = "csm_env"
    try:
        archive_display = archive_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        archive_display = str(archive_path)

    return {
        "generated_at_epoch": int(time.time()),
        "benchmark_files_sha256": benchmark_hashes,
        "ablation_files_sha256": ablation_hashes,
        "task_files_sha256": task_hashes,
        "seed_archive": {
            "path": archive_display,
            "sha256": sha256_file(archive_path) if archive_path.is_file() else "missing",
        },
        "csm_env": {
            "schema_version": registry.manifest.schema_version,
            "table_count": len(registry.tables()),
            "parity": parity_payload,
        },
        "dataset": {
            "local_csm_task_count": len(eval_set),
            "published_csm_task_count": reference_csm_task_count,
            "shortfall": max(0, reference_csm_task_count - len(eval_set)),
            "categories": eval_set.category_counts(),
        },
        "interpreter": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
    }


def offline_preflight(
    eval_set: EvalSetManifest,
    *,
    registry: Any,
    llm_config: Optional[PathLike] = None,
    archive: PathLike = DEFAULT_SEED_ARCHIVE,
) -> PreflightReport:
    """Run every check that needs neither the CSM server nor an LLM."""
    report = PreflightReport()
    check_schema(registry, report)
    check_parity(report)
    check_eval_set(eval_set, report)
    check_seed_archive(eval_set, report, archive)
    check_llm_config(llm_config, report)
    return report


async def run_preflight(
    eval_set: EvalSetManifest,
    *,
    registry: Any,
    llm_config: Optional[PathLike] = None,
    base_url: Optional[str] = None,
    archive: PathLike = DEFAULT_SEED_ARCHIVE,
) -> PreflightReport:
    """Offline checks plus an optional live server probe."""
    report = offline_preflight(
        eval_set, registry=registry, llm_config=llm_config, archive=archive
    )
    if base_url:
        await check_server(base_url, report)
    return report


def _client_for(base_url: str, database_id: str, context: Optional[Dict[str, Any]]) -> Any:
    """Build a benchmark MCP client bound to a specific database."""
    from benchmark.mcp_client import MCPClient

    return MCPClient(base_url=base_url, database_id=database_id, context=context)


async def resolve_live_anchors(
    eval_set: EvalSetManifest,
    *,
    base_url: str,
    registry: Any,
    requirements_path: Optional[PathLike] = None,
    archive: PathLike = DEFAULT_SEED_ARCHIVE,
    gym_root: PathLike = GYM_ROOT,
) -> Tuple[EvalSetManifest, Dict[str, Any]]:
    """Seed, fingerprint and resolve every task's anchor against the real CSM DB.

    Each task gets a throwaway database built from its own seed SQL, which is
    deleted afterwards. Every query issued here is read-only.

    Returns:
        The evaluation set with resolved anchors and fingerprints, plus a
        per-task resolution report.
    """
    import tempfile

    from benchmark.mcp_client import create_database_from_file, delete_database
    from csm_env import EnterpriseOpsSQLRunner
    from csm_env.transport.reader import EnterpriseOpsSQLRunnerReader
    from csm_integration import build_csm_environment

    from .fingerprint import initial_state_fingerprint
    from .state_model import StateModelAdapter
    from .task_registry import load_task_config

    archive_path = Path(archive)
    if not archive_path.is_absolute():
        archive_path = Path(gym_root) / archive_path

    updated: List[TaskRecord] = []
    resolutions: Dict[str, Any] = {}
    with zipfile.ZipFile(archive_path) as bundle, tempfile.TemporaryDirectory() as workdir:
        for record in eval_set:
            config = load_task_config(record.task_config_path)
            server = (config.gym_servers_config or [{}])[0]
            url = base_url or str(server.get("mcp_server_url"))
            context = server.get("context") or None

            sql_path = Path(workdir) / f"{record.task_id}.sql"
            sql_path.write_bytes(bundle.read(record.seed_database_file))
            database_id = create_database_from_file(url, str(sql_path))
            if not database_id:
                resolutions[record.task_id] = {"status": "seed_failed"}
                updated.append(record)
                continue
            try:
                runner = EnterpriseOpsSQLRunner(
                    base_url=url, database_id=database_id, context=context
                )
                reader = EnterpriseOpsSQLRunnerReader(runner)
                fingerprint = await initial_state_fingerprint(reader, registry)
                api = build_csm_environment(
                    _client_for(url, database_id, context),
                    enable_ggqr=True,
                    requirements_path=requirements_path,
                )
                adapter = StateModelAdapter(api, record, registry=registry, reader=reader)
                anchor = await adapter.resolve_anchor()
                resolutions[record.task_id] = {
                    "status": anchor.status,
                    "anchor": anchor.as_dict(),
                    "initial_state_fingerprint": fingerprint,
                }
                updated.append(
                    with_anchor(
                        record,
                        reference_rows=(
                            [(anchor.table, anchor.row_id or "")] if anchor.resolved else []
                        ),
                        initial_state_fingerprint=fingerprint,
                        database_id=database_id,
                    )
                )
            finally:
                delete_database(url, database_id)

    return (
        EvalSetManifest(
            eval_set_id=eval_set.eval_set_id,
            generated_from=eval_set.generated_from,
            schema_version=eval_set.schema_version,
            tasks=tuple(updated),
        ),
        resolutions,
    )


def filter_tasks(
    eval_set: EvalSetManifest,
    *,
    task_ids: Optional[Sequence[str]] = None,
    archive: PathLike = DEFAULT_SEED_ARCHIVE,
    allow_missing_seeds: bool = False,
    gym_root: PathLike = GYM_ROOT,
) -> Tuple[EvalSetManifest, List[Dict[str, str]]]:
    """Select the runnable subset of an evaluation set.

    A task is dropped when it is not explicitly requested, or when its seed SQL
    is absent from the archive and the caller did not opt in. Exclusions are
    returned so the caller can record exactly what was left out and why.

    Raises:
        ValueError: If ``task_ids`` names a task that is not in the set.
    """
    available = {record.task_id for record in eval_set}
    if task_ids:
        unknown = [task_id for task_id in task_ids if task_id not in available]
        if unknown:
            raise ValueError(f"Unknown task ids: {unknown}")

    archive_path = Path(archive)
    if not archive_path.is_absolute():
        archive_path = Path(gym_root) / archive_path
    seed_names: Optional[set] = None
    if archive_path.is_file():
        try:
            with zipfile.ZipFile(archive_path) as bundle:
                seed_names = set(bundle.namelist())
        except zipfile.BadZipFile:
            seed_names = None

    kept: List[TaskRecord] = []
    excluded: List[Dict[str, str]] = []
    for record in eval_set:
        if task_ids and record.task_id not in task_ids:
            excluded.append({"task_id": record.task_id, "reason": "not requested"})
            continue
        if seed_names is not None and record.seed_database_file not in seed_names:
            reason = f"seed file absent from {archive_path.name}: {record.seed_database_file}"
            if allow_missing_seeds:
                excluded.append({"task_id": record.task_id, "reason": reason + " (allowed)"})
                continue
            excluded.append({"task_id": record.task_id, "reason": reason})
            continue
        kept.append(record)

    return (
        EvalSetManifest(
            eval_set_id=eval_set.eval_set_id,
            generated_from=eval_set.generated_from,
            schema_version=eval_set.schema_version,
            tasks=tuple(kept),
        ),
        excluded,
    )


def bootstrap_manifests(
    eval_set: EvalSetManifest,
    *,
    registry: Any,
    manifests_dir: PathLike,
    archive: PathLike = DEFAULT_SEED_ARCHIVE,
    gym_root: PathLike = GYM_ROOT,
) -> Dict[str, str]:
    """Write the eval-set, requirements and pinned-revision manifests."""
    directory = Path(manifests_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "eval_set": str(
            save_json(eval_set.as_dict(), directory / "csm_eval_set.json")
        ),
        "requirements": str(
            save_json(
                build_requirements_manifest(eval_set, registry=registry),
                directory / "csm_requirements.json",
            )
        ),
        "pinned_revisions": str(
            save_json(
                write_pinned_revisions(
                    eval_set, registry=registry, archive=archive, gym_root=gym_root
                ),
                directory / "pinned_revisions.json",
            )
        ),
    }
