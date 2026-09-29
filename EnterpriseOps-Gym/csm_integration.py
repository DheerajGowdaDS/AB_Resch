"""Additive wiring between EnterpriseOps-Gym and the ``csm_env`` layer.

This module is glue only: it does not modify any benchmark, verifier, or
``csm_env`` logic. It composes the existing ``benchmark.mcp_client.MCPClient``
configuration surface (base_url, database_id, auth_config, context) with
``csm_env``'s public entry points so agents, planners, and evaluators can
build a database-backed CSM environment representation from the same client
the benchmark and its verifiers already use.

Two ways in:

- :func:`build_csm_environment` adapts an existing ``MCPClient`` instance
  (the one a benchmark run already created for a seeded database).
- :func:`build_csm_environment_from_task` consumes an external
  EnterpriseOps-Gym CSM task JSON (``gym_servers_config``) and constructs the
  client first. ``database_id`` is supplied by the caller because the gym
  assigns it when the database is seeded at run time.

External task-requirement manifests (``{"tasks": {task_type: [table, ...]}}``)
are consumed through the public ``requirement_tables`` hook and are validated
fail-closed by ``csm_env``'s ``RequirementRegistry``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from benchmark.mcp_client import MCPClient

from csm_env import (
    CSMEnvironmentAPI,
    CSMEnvironmentRepresentation,
    QueryBudget,
    from_enterpriseops_mcp_client,
)
from csm_env.environment import GGQREnvironment
from csm_env.sql_runner import EnterpriseOpsSQLRunner
from csm_env.transport.reader import EnterpriseOpsSQLRunnerReader

DEFAULT_CSM_SERVER_NAME = "sn-csm-server"

PathLike = Union[str, Path]


def build_csm_environment(
    mcp_client: MCPClient,
    *,
    enable_ggqr: bool = True,
    budget: Optional[QueryBudget] = None,
    requirements_path: Optional[PathLike] = None,
) -> CSMEnvironmentAPI:
    """Wire a benchmark ``MCPClient`` into the ``csm_env`` representation.

    Returns the stable :class:`CSMEnvironmentAPI` facade. When
    ``enable_ggqr`` is true the underlying representation also carries the
    additive ``ggqr`` environment. ``requirements_path`` optionally points at
    an external task-requirement manifest consumed by the GGQR router.
    """
    if requirements_path is not None and not enable_ggqr:
        raise ValueError("requirements_path requires enable_ggqr=True")

    if requirements_path is None:
        representation = from_enterpriseops_mcp_client(
            mcp_client, enable_ggqr=enable_ggqr, budget=budget
        )
        return CSMEnvironmentAPI(
            representation, getattr(representation, "ggqr", None)
        )

    # Requirement-manifest path: the same composition as
    # from_enterpriseops_mcp_client, plus an externally supplied
    # requirement manifest. Loading it through ``RequirementRegistry.from_json``
    # honours the optional ``required_attributes`` / ``required_relations`` /
    # ``intent`` blocks when present (State Model 2.0, P0/A3); a tables-only
    # manifest keeps the built-in task-type defaults for those dimensions.
    from csm_env.schema_spec.registry import SchemaRegistry
    from csm_env.query.requirements import RequirementRegistry

    requirement_registry = RequirementRegistry.from_json(
        SchemaRegistry.from_static(), requirements_path
    )
    sql = EnterpriseOpsSQLRunner(
        base_url=mcp_client.base_url,
        database_id=mcp_client.database_id,
        auth_config=getattr(mcp_client, "auth_config", None),
        context=getattr(mcp_client, "context", None),
    )
    representation = CSMEnvironmentRepresentation(sql)
    ggqr = GGQREnvironment(
        EnterpriseOpsSQLRunnerReader(sql),
        database_id=mcp_client.database_id,
        budget=budget,
        requirement_registry=requirement_registry,
    )
    representation.ggqr = ggqr
    return CSMEnvironmentAPI(representation, ggqr)


def mcp_client_from_task_config(
    task_path: PathLike,
    *,
    gym_server_name: Optional[str] = None,
    database_id: Optional[str] = None,
    auth_config: Optional[Dict[str, Any]] = None,
) -> MCPClient:
    """Build a benchmark ``MCPClient`` from an EnterpriseOps-Gym task JSON.

    Reads the task file's ``gym_servers_config`` entry (selected by
    ``mcp_server_name``; defaults to the first entry) and maps it onto the
    standard client constructor. ``database_id`` is a caller input because
    the gym assigns it when the task database is seeded.
    """
    payload = json.loads(Path(task_path).read_text(encoding="utf-8"))
    servers = payload.get("gym_servers_config")
    if not isinstance(servers, list) or not servers:
        raise ValueError(f"{task_path}: missing or empty gym_servers_config")

    selected: Optional[Dict[str, Any]] = None
    for entry in servers:
        if not isinstance(entry, dict):
            continue
        if gym_server_name is None or entry.get("mcp_server_name") == gym_server_name:
            selected = entry
            break
    if selected is None:
        available = [entry.get("mcp_server_name") for entry in servers]
        raise KeyError(
            f"gym server {gym_server_name!r} not found in {task_path}; "
            f"available: {available}"
        )

    base_url = selected.get("mcp_server_url")
    if not base_url:
        raise ValueError(f"{task_path}: selected gym server has no mcp_server_url")

    return MCPClient(
        base_url=base_url,
        auth_config=auth_config,
        mcp_endpoint=payload.get("mcp_endpoint", "/mcp"),
        database_id=database_id,
        context=selected.get("context") or None,
    )


def build_csm_environment_from_task(
    task_path: PathLike,
    *,
    gym_server_name: Optional[str] = None,
    database_id: Optional[str] = None,
    auth_config: Optional[Dict[str, Any]] = None,
    enable_ggqr: bool = True,
    budget: Optional[QueryBudget] = None,
    requirements_path: Optional[PathLike] = None,
) -> CSMEnvironmentAPI:
    """One-call wiring: task JSON -> MCPClient -> CSM environment facade."""
    client = mcp_client_from_task_config(
        task_path,
        gym_server_name=gym_server_name,
        database_id=database_id,
        auth_config=auth_config,
    )
    return build_csm_environment(
        client,
        enable_ggqr=enable_ggqr,
        budget=budget,
        requirements_path=requirements_path,
    )


