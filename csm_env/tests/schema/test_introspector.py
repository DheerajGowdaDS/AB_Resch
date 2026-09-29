from __future__ import annotations

import pytest

from csm_env.schema_spec import (
    MANIFEST,
    SQLSchemaIntrospector,
    UnsupportedSchemaMetadata,
    compare_with_manifest,
)


def test_sql_introspector_is_fail_closed_unsupported():
    introspector = SQLSchemaIntrospector(reader=None, manifest=MANIFEST)

    with pytest.raises(UnsupportedSchemaMetadata):
        __import__("asyncio").run(introspector.introspect())


def test_verify_against_manifest_returns_none_when_unsupported():
    introspector = SQLSchemaIntrospector(reader=None, manifest=MANIFEST)

    import asyncio

    assert asyncio.run(introspector.verify_against_manifest()) is None


def test_compare_reports_missing_and_extra_tables():
    observed = {name: ("account_id",) for name in (
        "account", "contact",
    )}
    observed["rogue_table"] = ("x",)
    observed_pks = {"account": "account_id", "contact": "contact_id", "rogue_table": "x"}

    report = compare_with_manifest(MANIFEST, observed, observed_pks)

    assert report.has_mismatch
    assert len(report.missing_tables) == 15
    assert "rogue_table" in report.extra_tables
    assert "customer_case" in report.missing_tables


def test_compare_reports_column_and_pk_mismatch():
    observed = {"account": ("account_id", "name")}
    report = compare_with_manifest(MANIFEST, observed, {"account": "row_id"})

    assert report.missing_columns["account"] == (
        "account_type", "active", "email_domain", "sys_created_on", "sys_updated_on",
    )
    assert report.wrong_primary_keys["account"] == ("account_id", "row_id")


def test_compare_passes_on_identical_metadata():
    observed = {spec.table: spec.column_names() for spec in MANIFEST.tables}
    pks = {spec.table: spec.primary_key for spec in MANIFEST.tables}

    report = compare_with_manifest(MANIFEST, dict(observed), dict(pks))

    assert not report.has_mismatch
    assert report.summary() == "no mismatch"
