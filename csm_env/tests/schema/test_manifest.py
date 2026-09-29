from __future__ import annotations

import pytest

from csm_env.schema import ENTITY_BY_TABLE, ENTITY_BY_TYPE, ENTITIES, FK_BY_SOURCE, FK_BY_TARGET, FKS, TABLE_COLUMNS
from csm_env.schema_spec import (
    EXPECTED_TABLE_COUNT,
    MANIFEST,
    ColumnSpec,
    ForeignKeySpec,
    SchemaRegistry,
    TableSpec,
    validate_manifest,
)


def test_static_manifest_loads_17_tables():
    registry = SchemaRegistry.from_static()
    assert len(registry.tables()) == EXPECTED_TABLE_COUNT == 17
    for name in (
        "account", "customer_case", "entitlement", "contract", "product",
        "installed_product", "contact", "user", "user_group", "location",
        "knowledge", "case_sla", "case_knowledge", "notification",
        "interaction", "sla_definition", "user_group_member",
    ):
        assert registry.table(name) is not None


def test_manifest_pk_is_declared_column():
    registry = SchemaRegistry.from_static()
    for table_name in registry.tables():
        spec = registry.require_table(table_name)
        assert spec.primary_key in spec.column_names()


def test_registry_rejects_duplicate_tables():
    spec = TableSpec(
        table="account",
        node_type="Account",
        primary_key="account_id",
        columns=(ColumnSpec("account_id", "string"),),
    )
    from csm_env.schema_spec.models import SchemaManifest, SchemaSource

    manifest = SchemaManifest(
        schema_version="1.0.0",
        tables=(spec, spec),
        foreign_keys=(),
        source=SchemaSource.STATIC_REGISTRY,
    )
    with pytest.raises(ValueError, match="Duplicate table"):
        SchemaRegistry(manifest)


def test_registry_rejects_pk_not_in_columns():
    spec = TableSpec(
        table="account",
        node_type="Account",
        primary_key="row_id",
        columns=(ColumnSpec("account_id", "string"),),
    )
    from csm_env.schema_spec.models import SchemaManifest, SchemaSource

    manifest = SchemaManifest(
        schema_version="1.0.0",
        tables=(spec,),
        foreign_keys=(),
        source=SchemaSource.STATIC_REGISTRY,
    )
    with pytest.raises(ValueError, match="primary key"):
        SchemaRegistry(manifest)


def test_registry_rejects_duplicate_fk():
    from csm_env.schema_spec.models import ColumnSpec as CS, SchemaManifest, SchemaSource, TableSpec as TS

    table = TS("account", "Account", "account_id", (CS("account_id", "string"),))
    fk = ForeignKeySpec("account", "account_id", "account", "account_id", "SELF")
    manifest = SchemaManifest(
        schema_version="1.0.0",
        tables=(table,),
        foreign_keys=(fk, fk),
        source=SchemaSource.STATIC_REGISTRY,
    )
    with pytest.raises(ValueError, match="Duplicate FK"):
        SchemaRegistry(manifest)


def test_validate_manifest_reports_fk_problems():
    from csm_env.schema_spec.models import ColumnSpec as CS, SchemaManifest, SchemaSource, TableSpec as TS

    table = TS("account", "Account", "account_id", (CS("account_id", "string"),))
    bad_fk = ForeignKeySpec("account", "nope", "missing_table", "also_nope", "lowercase")
    manifest = SchemaManifest(
        schema_version="1.0.0",
        tables=(table,),
        foreign_keys=(bad_fk,),
        source=SchemaSource.STATIC_REGISTRY,
    )
    problems = validate_manifest(manifest)
    assert any("FK source column" in p for p in problems)
    assert any("FK target table" in p for p in problems)
    assert any("UPPERCASE" in p for p in problems)


def test_static_manifest_is_self_consistent():
    assert validate_manifest(MANIFEST) == []


def test_compatibility_major_version():
    registry = SchemaRegistry.from_static()
    assert registry.compatibility("1.9.9").compatible is True
    assert registry.compatibility("2.0.0").compatible is False


def test_legacy_schema_shapes_preserved():
    assert len(ENTITIES) == 17
    assert len(TABLE_COLUMNS) == 17
    assert ENTITY_BY_TABLE["customer_case"].primary_key == "case_id"
    assert ENTITY_BY_TYPE["CustomerCase"].table == "customer_case"
    assert TABLE_COLUMNS["interaction"] == (
        "interaction_id", "channel", "account_id", "contact_id", "case_id",
        "interacted_user", "status", "started_at", "ended_at", "sys_created_on",
    )
    assert len(FKS) == 29
    source_tables = {fk.source_table for fk in FKS}
    assert source_tables.issubset({e.table for e in ENTITIES})
    assert FK_BY_SOURCE["customer_case"]
    assert FK_BY_TARGET["account"]


def test_all_declared_fk_columns_exist_in_table_columns():
    for fk in FKS:
        assert fk.source_column in TABLE_COLUMNS[fk.source_table]
        assert fk.target_column in TABLE_COLUMNS[fk.target_table]


def test_snapshot_runtime_type_validation_accepts_identifier_ints_and_string_coverage_hours(registry):
    from csm_env.verification import check_v2_runtime_row_types

    result = check_v2_runtime_row_types(
        registry,
        {
            "account": [{"account_id": 10, "name": "SynthCorp", "active": True}],
            "entitlement": [{"entitlement_id": 50, "account_id": 10, "product_id": 20, "coverage_hours": "24x7", "active": True}],
        },
    )
    assert result.passed, result.failures


def test_snapshot_runtime_type_validation_rejects_bad_boolean(registry):
    from csm_env.verification import check_v2_runtime_row_types

    result = check_v2_runtime_row_types(
        registry,
        {"account": [{"account_id": 10, "name": "SynthCorp", "active": "yes"}]},
    )
    assert not result.passed
    assert any("active" in failure for failure in result.failures)


def test_requirement_registry_from_json(tmp_path, registry):
    import json
    from csm_env.query.requirements import RequirementRegistry

    path = tmp_path / "requirements.json"
    path.write_text(json.dumps({"tasks": {"custom_task": ["customer_case", "account"]}}), encoding="utf-8")
    loaded = RequirementRegistry.from_json(registry, path)
    assert loaded.requirements_for("custom_task").required_tables == ("customer_case", "account")


def test_row_only_snapshot_marks_structural_schema_check_skipped(registry, fixture_rows):
    from csm_env.verification import check_v1_v2_snapshot

    checks = check_v1_v2_snapshot(registry, {"tables": fixture_rows})
    assert checks[0].passed
    assert checks[1].skipped
    assert checks[2].passed
