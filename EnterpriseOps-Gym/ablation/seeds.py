"""Materializing seed SQL so the benchmark can actually seed a database.

A task's ``seed_database_file`` names a path *inside* ``gym_dbs.zip``, but
``benchmark.mcp_client.create_database_from_file`` reads the SQL from the
filesystem. The benchmark therefore needs those files extracted first.

This module bridges the two without touching the benchmark: it extracts each
referenced archive member once into a local cache and rewrites the loaded
``BenchmarkConfig`` to point at the extracted file.
"""

from __future__ import annotations

import copy
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from benchmark.models import BenchmarkConfig

PathLike = Union[str, Path]

#: Default seed archive shipped with the gym checkout.
DEFAULT_ARCHIVE = "gym_dbs.zip"


class SeedCache:
    """Extracts and caches the seed SQL referenced by task configs.

    Args:
        archive: Path to ``gym_dbs.zip`` (or an already-extracted directory).
        cache_dir: Where extracted SQL is written. Defaults to
            ``<gym root>/.seed_cache``.
    """

    def __init__(
        self,
        archive: PathLike = DEFAULT_ARCHIVE,
        *,
        cache_dir: Optional[PathLike] = None,
        gym_root: Optional[PathLike] = None,
    ) -> None:
        root = Path(gym_root) if gym_root is not None else Path(__file__).resolve().parents[1]
        self.archive = Path(archive)
        if not self.archive.is_absolute():
            self.archive = root / self.archive
        self.cache_dir = Path(cache_dir) if cache_dir is not None else root / ".seed_cache"
        self._members: Optional[set] = None

    # ---------- internals ----------

    def _archive_members(self) -> Optional[set]:
        if self._members is None:
            if self.archive.is_dir():
                members: set = set()
                for path in self.archive.rglob("*.sql"):
                    members.add(path.relative_to(self.archive).as_posix())
                self._members = members or None
            elif self.archive.is_file():
                with zipfile.ZipFile(self.archive) as bundle:
                    self._members = {name for name in bundle.namelist() if name.endswith(".sql")}
            else:
                self._members = None
        return self._members

    # ---------- public surface ----------

    def contains(self, seed_file: str) -> bool:
        """Whether this seed can be resolved (on disk, or present in the archive)."""
        if not seed_file:
            return False
        candidate = Path(seed_file)
        if candidate.is_file():
            return True
        return bool(self._archive_members() and seed_file in self._archive_members())

    def materialize(self, seed_file: str) -> Path:
        """Return a local filesystem path for ``seed_file``.

        Already-present files are returned untouched. Otherwise the archive
        member is extracted once into the cache and reused afterwards.

        Raises:
            FileNotFoundError: If the seed is neither on disk nor in the archive.
        """
        candidate = Path(seed_file)
        if candidate.is_file():
            return candidate

        members = self._archive_members()
        if not members or seed_file not in members:
            raise FileNotFoundError(
                f"Seed SQL not found on disk or in {self.archive}: {seed_file}"
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self.cache_dir / Path(seed_file).name
        if not target.is_file():
            if self.archive.is_dir():
                source = self.archive / seed_file
                shutil.copyfile(source, target)
            else:
                with zipfile.ZipFile(self.archive) as bundle:
                    with bundle.open(seed_file) as reader, open(target, "wb") as writer:
                        shutil.copyfileobj(reader, writer)
        return target

    def bind_config(self, config: BenchmarkConfig) -> List[Dict[str, Any]]:
        """Return a copy of ``config`` whose seed paths exist on disk.

        Raises:
            FileNotFoundError: If any referenced seed cannot be materialized.
        """
        servers = config.gym_servers_config or []
        if not servers:
            return []

        bound = copy.deepcopy(servers)
        for entry in bound:
            seed_file = entry.get("seed_database_file")
            if not seed_file:
                continue
            entry["seed_database_file"] = str(self.materialize(seed_file))
        config.gym_servers_config = bound
        return bound

    def preflight(self, seed_files: List[str]) -> Dict[str, str]:
        """Report the resolved path for each seed, without mutating anything."""
        resolved: Dict[str, str] = {}
        for seed_file in seed_files:
            resolved[seed_file] = str(self.materialize(seed_file))
        return resolved
