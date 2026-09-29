"""Offline smoke tests for the EnterpriseOps-Gym <-> csm_env wiring.

No network access and no live CSM server: the gym ``MCPClient`` is only
constructed, never connected. These tests verify that the wiring module
composes the two codebases without editing either one's logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import csm_env
from benchmark.mcp_client import MCPClient
from csm_env import (
    CSMEnvironmentAPI,
    CSMEnvironmentRepresentation,
    QueryBudget,
    from_enterpriseops_mcp_client,
)
from csm_env.environment import GGQREnvironment

from csm_integration import (
    DEFAULT_CSM_SERVER_NAME,
    build_csm_environment,
    build_csm_environment_from_task,
    mcp_client_from_task_config,
)

GYM_ROOT = Path(__file__).resolve().parents[1]
CSM_TASK = (
    GYM_ROOT
    / "data"
    / "revised"
    / "csm"
    / "task_20251205_153330_906_a8eea1c0_8c7a6205.json"
)


def _offline_client() -> MCPClient:
    return MCPClient(
        base_url="http://localhost:8001/",
        database_id="db_smoke_test",
        auth_config={"type": "bearer", "token": "smoke-token"},
        context={"x-user-email": "agent@example.com"},
    )


def test_csm_env_public_surface_importable_inside_gym():
    for name in csm_env.__all__:
        assert hasattr(csm_env, name), f"missing public export: {name}"


def test_mcp_client_config_propagates_without_network():
    client = _offline_client()
    api = build_csm_environment(client, budget=QueryBudget(max_queries=5))

    assert isinstance(api, CSMEnvironmentAPI)
    representation = api.environment
    assert isinstance(representation, CSMEnvironmentRepresentation)

    sql = representation.sql
    # MCPClient rstrips the base URL; the SQL runner must see the same value.
    assert sql.base_url == "http://localhost:8001"
    assert sql.database_id == "db_smoke_test"
    assert sql.auth_config == {"type": "bearer", "token": "smoke-token"}
    assert sql.context == {"x-user-email": "agent@example.com"}

    ggqr = getattr(representation, "ggqr", None)
    assert isinstance(ggqr, GGQREnvironment)
    assert ggqr.database_id == "db_smoke_test"
    assert ggqr.budget.max_queries == 5


def test_legacy_surface_without_ggqr_fails_closed():
    api = build_csm_environment(_offline_client(), enable_ggqr=False)
    assert not hasattr(api.environment, "ggqr")
    with pytest.raises(RuntimeError, match="GGQR is not attached"):
        api.represent_state(object())


def test_gym_csm_task_json_consumed():
    client = mcp_client_from_task_config(
        CSM_TASK,
        gym_server_name=DEFAULT_CSM_SERVER_NAME,
        database_id="db_task_smoke",
    )
    assert client.base_url == "http://localhost:8001"
    assert client.database_id == "db_task_smoke"
    assert client.context == {"x-user-email": "joanne.simpson@servicenow.com"}
    assert client.mcp_endpoint == "/mcp"


def test_environment_built_from_gym_csm_task_json():
    api = build_csm_environment_from_task(CSM_TASK, database_id="db_task_smoke")
    assert isinstance(api, CSMEnvironmentAPI)
    assert api.environment.sql.base_url == "http://localhost:8001"
    assert api.environment.sql.database_id == "db_task_smoke"
    assert api.environment.sql.context == {
        "x-user-email": "joanne.simpson@servicenow.com"
    }
    assert isinstance(api.environment.ggqr, GGQREnvironment)


def test_unknown_gym_server_name_fails_closed():
    with pytest.raises(KeyError, match="not found"):
        mcp_client_from_task_config(
            CSM_TASK, gym_server_name="sn-nonexistent-server"
        )


def test_requirements_manifest_consumed(tmp_path):
    manifest = tmp_path / "requirements.json"
    manifest.write_text(
        json.dumps(
            {
                "tasks": {
                    "resolve_case": ["customer_case", "account"],
                    "entitlement_audit": ["entitlement", "contract", "account"],
                }
            }
        ),
        encoding="utf-8",
    )
    api = build_csm_environment(
        _offline_client(), requirements_path=manifest
    )
    requirements = api.environment.ggqr.requirements
    assert requirements.task_types() == ("entitlement_audit", "resolve_case")
    assert requirements.requirements_for("resolve_case").required_tables == (
        "customer_case",
        "account",
    )
    # Unknown task types still fail closed inside csm_env.
    with pytest.raises(KeyError):
        requirements.requirements_for("fly_to_moon")


def test_requirements_manifest_unknown_table_fails_closed(tmp_path):
    bad = tmp_path / "bad_requirements.json"
    bad.write_text(
        json.dumps({"tasks": {"bad_task": ["not_a_csm_table"]}}),
        encoding="utf-8",
    )
    with pytest.raises(KeyError):
        build_csm_environment(_offline_client(), requirements_path=bad)


def test_requirements_path_requires_ggqr(tmp_path):
    manifest = tmp_path / "requirements.json"
    manifest.write_text(json.dumps({"tasks": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="enable_ggqr"):
        build_csm_environment(
            _offline_client(), enable_ggqr=False, requirements_path=manifest
        )


def test_direct_integration_entry_point_still_works():
    representation = from_enterpriseops_mcp_client(_offline_client())
    assert isinstance(representation, CSMEnvironmentRepresentation)
    assert not hasattr(representation, "ggqr")
