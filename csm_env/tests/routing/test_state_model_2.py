"""State Model 2.0 blueprint mechanisms: attribute projection, weighted paths.

Covers blueprint sections 3, 5, 7 and 9: task normalization carriers on
``TaskRequirements``/``TaskRequest``, task-conditioned column projection, and
relation-aware path search. These tests run against the static CSM schema.
"""

from __future__ import annotations

import pytest

from csm_env.query.models import QueryBudget, TaskRequest, TaskRequirements
from csm_env.query.planner import QueryPlanner, project_columns
from csm_env.query.requirements import RequirementRegistry
from csm_env.schema_graph import build_schema_graph
from csm_env.schema_spec.registry import SchemaRegistry


@pytest.fixture()
def registry() -> SchemaRegistry:
    return SchemaRegistry.from_static()


# ---------------------------------------------------------------------------
# Blueprint section 9: attribute-level projection
# ---------------------------------------------------------------------------


def test_projection_keeps_pk_fks_and_task_attributes(registry):
    columns = project_columns(
        "customer_case",
        registry=registry,
        required_attributes=("priority", "assignment_group"),
        required_tables=("customer_case", "account", "user_group"),
    )
    assert columns is not None
    assert "case_id" in columns
    # FKs to required tables survive; irrelevant FKs to non-required tables
    # are pruned when the task names its relational scope.
    assert "account_id" in columns
    assert "assignment_group_id" in columns
    assert "priority" in columns
    assert "contact_id" not in columns
    assert "escalation" not in columns


def test_projection_keeps_identity_and_interpretation_columns(registry):
    columns = project_columns(
        "entitlement",
        registry=registry,
        required_attributes=("support_level",),
        required_tables=("entitlement",),
    )
    assert columns is not None
    assert "entitlement_id" in columns
    assert "support_level" in columns
    # P0/A1: attribute keys now resolve through the SHARED lexicon, so the
    # "support_level" key keeps its whole cluster (support_level +
    # coverage_hours), exactly as the requirement registry resolves it.
    assert "coverage_hours" in columns
    assert "sys_created_on" not in columns
    assert "max_cases_per_month" not in columns


def test_projection_without_task_context_keeps_all_fks(registry):
    columns = project_columns("account", registry=registry)
    # account has no outgoing FKs, so with identity columns this is a
    # meaningful minimal projection rather than the full column list.
    assert columns is not None
    assert "account_id" in columns
    assert "name" in columns
    assert "email_domain" not in columns, "email_domain does not name a row"
    assert "sys_created_on" not in columns and "sys_updated_on" not in columns


def test_projection_never_reaches_sql_layer_with_unknown_columns(registry):
    columns = project_columns(
        "customer_case",
        registry=registry,
        required_attributes=("priority", "nonexistent_attribute"),
        required_tables=("customer_case",),
    )
    assert columns is not None
    for column in columns:
        registry.require_column("customer_case", column)


def test_projection_prunes_timestamps_on_a_structural_table(registry):
    # user_group_member carries only a PK, two FKs and timestamps; the
    # projection keeps the structural set and drops the timestamps.
    columns = project_columns(
        "user_group_member",
        registry=registry,
        required_attributes=(),
        required_tables=(),
    )
    assert columns == ("member_id", "group_id", "user_id")


# ---------------------------------------------------------------------------
# Blueprint section 3: registry attribute/relation carriers
# ---------------------------------------------------------------------------


def test_requirements_carry_attributes_relations_and_intent(registry):
    requirements = RequirementRegistry(registry).requirements_for("resolve_case")
    assert "customer_case" in requirements.required_tables
    assert "priority" in requirements.required_attributes
    assert ("customer_case", "account") in requirements.required_relations
    assert "read" in requirements.intent


def test_unknown_attribute_fails_closed(registry):
    with pytest.raises(KeyError, match="matches no column"):
        RequirementRegistry(
            registry,
            {"custom": ("customer_case",)},
            attributes_by_task={"custom": ("totally_bogus_column",)},
        )


def test_attribute_overrides_merge_without_dropping_registered(registry):
    registry_obj = RequirementRegistry(registry)
    merged = registry_obj.requirements_for(
        "resolve_case", attribute_overrides=("channel",)
    )
    assert "channel" in merged.required_attributes
    assert "priority" in merged.required_attributes


def test_unknown_relation_pair_fails_closed(registry):
    with pytest.raises(KeyError, match="declares no foreign key"):
        RequirementRegistry(
            registry,
            {"custom": ("customer_case",)},
            relations_by_task={"custom": (("customer_case", "sla_definition"),)},
        )


def test_json_loader_accepts_tables_only_manifests(registry, tmp_path):
    import json

    path = tmp_path / "requirements.json"
    path.write_text(
        json.dumps({"tasks": {"custom": ["customer_case", "account"]}}),
        encoding="utf-8",
    )
    loaded = RequirementRegistry.from_json(registry, path)
    requirements = loaded.requirements_for("custom")
    assert requirements.required_tables == ("customer_case", "account")
    # A tables-only manifest intentionally starts with no attribute requests;
    # the planner's structural projection (PK + FKs + identity) still applies.
    assert requirements.required_attributes == ()


# ---------------------------------------------------------------------------
# Blueprint section 5: relation-aware weighted path search
# ---------------------------------------------------------------------------


def test_plan_prefers_task_required_relation_at_equal_cost(registry):
    planner = QueryPlanner(build_schema_graph(registry), registry)
    requirements = TaskRequirements(
        task_type="custom",
        required_tables=("customer_case", "user"),
        required_relations=(("customer_case", "user"),),
    )
    plan = planner.plan(
        task_type="custom",
        anchor_table="customer_case",
        anchor_key="case_id",
        anchor_value=1233,
        requirements=requirements,
        budget=QueryBudget(max_hops=2),
        route_id="route-weighted",
    )
    user_step = next(step for step in plan.steps if step.table == "user")
    assert user_step.hop == 1, "the direct ASSIGNED_TO FK must win when the task names it"


def test_plan_without_relations_stays_deterministic(registry):
    planner = QueryPlanner(build_schema_graph(registry), registry)
    requirements = TaskRequirements(
        task_type="custom",
        required_tables=("customer_case", "contract"),
    )
    plan = planner.plan(
        task_type="custom",
        anchor_table="customer_case",
        anchor_key="case_id",
        anchor_value=1233,
        requirements=requirements,
        budget=QueryBudget(max_hops=3),
        route_id="route-unweighted",
    )
    assert [step.table for step in plan.steps] == [
        "customer_case",
        "account",
        "contract",
    ]


def test_request_attributes_flow_through_requirements_merge(registry):
    registry_obj = RequirementRegistry(registry)
    merged = registry_obj.requirements_for(
        "resolve_case",
        attribute_overrides=("escalation",),
        intent_overrides=("update",),
    )
    assert "escalation" in merged.required_attributes
    assert "update" in merged.intent


def test_budget_carries_max_tokens():
    budget = QueryBudget()
    assert budget.max_tokens == 20000
    assert QueryBudget(max_tokens=500).max_tokens == 500


# ---------------------------------------------------------------------------
# P0/A1: shared attribute resolution + the task's own FK survives projection
# ---------------------------------------------------------------------------


def test_shared_resolver_realizes_assignment_group_to_fk(registry):
    """The shared lexicon must realize the prompt shorthand to the concrete FK.

    This is what keeps the query layer and the state layer from drifting apart:
    ``assignment_group`` -> ``assignment_group_id`` in both.
    """
    from csm_env.query.requirements import resolve_attribute_columns

    resolved = resolve_attribute_columns(registry, ("customer_case",), ("assignment_group",))
    assert "assignment_group_id" in resolved


def test_project_columns_keeps_task_assignment_group_fk(registry):
    """N2: pruning for "check the assigned group" must not drop assignment_group_id."""
    columns = project_columns(
        "customer_case",
        registry=registry,
        required_attributes=("priority", "assignment_group", "assigned_to"),
        required_tables=("customer_case", "account"),
    )
    assert columns is not None
    # The FK the task explicitly asks about must survive the projection.
    assert "assignment_group_id" in columns
    assert "priority" in columns


def test_project_columns_and_registry_use_one_resolver(registry):
    """P0/A1: the planner and the requirement registry agree on attribute keys."""
    from csm_env.query.requirements import resolve_attribute_columns

    keys = ("support_level", "assignment_group")
    planner_cols = project_columns(
        "customer_case",
        registry=registry,
        required_attributes=keys,
        required_tables=("customer_case",),
    )
    shared_cols = resolve_attribute_columns(registry, ("customer_case",), keys)
    # Every column the shared resolver realizes on customer_case must be in the
    # planner's projection (the planner's projection is a superset for that table).
    assert set(shared_cols) <= set(planner_cols or ())
