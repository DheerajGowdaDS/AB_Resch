from __future__ import annotations

import pytest

from csm_env.query.adapters import ManagedTableAdapter, TableAdapterFactory, render_literal, render_query
from csm_env.query.models import Filter, QuerySpec
from csm_env.transport.reader import StaticRowsReader, TransportError


def test_render_literal_types():
    assert render_literal(None) == "NULL"
    assert render_literal(True) == "TRUE"
    assert render_literal(False) == "FALSE"
    assert render_literal(5) == "5"
    assert render_literal("o'brien") == "'o''brien'"


def test_render_literal_rejects_objects():
    with pytest.raises(TypeError):
        render_literal(object())


def test_render_query_selects_explicit_columns(registry):
    query = render_query(
        registry,
        QuerySpec(table="customer_case", columns=("case_id", "state"), limit=5),
    )
    assert query == "SELECT case_id, state FROM customer_case LIMIT 5;"


def test_render_query_with_filters(registry):
    query = render_query(
        registry,
        QuerySpec(
            table="entitlement",
            filters=(Filter("account_id", 10), Filter("product_id", 20)),
            limit=10,
        ),
    )
    assert query == (
        "SELECT entitlement_id, account_id, product_id, contract_id, support_level, "
        "coverage_hours, max_cases_per_month, active, sys_created_on, sys_updated_on "
        "FROM entitlement WHERE account_id = 10 AND product_id = 20 LIMIT 10;"
    )


def test_render_query_rejects_unknown_table_and_column(registry):
    with pytest.raises(KeyError):
        render_query(registry, QuerySpec(table="not_a_table", limit=1))
    with pytest.raises(KeyError):
        render_query(registry, QuerySpec(table="account", columns=("nope",), limit=1))
    with pytest.raises(KeyError):
        render_query(
            registry,
            QuerySpec(table="account", filters=(Filter("nope", 1),), limit=1),
        )


def test_render_query_rejects_user_identifier_injection(registry):
    with pytest.raises(KeyError):
        render_query(
            registry,
            QuerySpec(table="account", columns=("account_id; DROP TABLE account",), limit=1),
        )


def test_render_query_requires_positive_limit(registry):
    with pytest.raises(ValueError):
        render_query(registry, QuerySpec(table="account", limit=0))


async def _collect(adapter):
    return await adapter.query_by_pk(1233)


def test_adapter_query_by_pk(reader, registry):
    import asyncio

    adapter = ManagedTableAdapter(registry, reader, "customer_case")
    rows = asyncio.run(_collect(adapter))
    assert len(rows) == 1
    assert rows[0]["case_id"] == 1233
    assert rows[0]["state"] == "open"


def test_adapter_query_by_fk_validates_column(reader, registry):
    import asyncio

    adapter = ManagedTableAdapter(registry, reader, "entitlement")
    rows = asyncio.run(adapter.query_by_fk("account_id", 10))
    assert {row["entitlement_id"] for row in rows} == {50, 51}

    with pytest.raises(KeyError):
        asyncio.run(adapter.query_by_fk("not_a_column", 10))


def test_adapter_rejects_foreign_table_query(reader, registry):
    import asyncio

    adapter = ManagedTableAdapter(registry, reader, "account")
    with pytest.raises(ValueError):
        asyncio.run(adapter.query(QuerySpec(table="product", limit=1)))


def test_adapter_factory_caches_adapters(reader, registry):
    factory = TableAdapterFactory(registry, reader)
    first = factory.adapter_for("account")
    second = factory.adapter_for("account")
    assert first is second
    assert first.table == "account"


def test_static_reader_routes_by_table():
    import asyncio

    reader = StaticRowsReader({"account": [{"account_id": 1}], "product": [{"product_id": 2}]})

    async def run():
        return (
            await reader.fetch_rows("SELECT * FROM account;"),
            await reader.fetch_rows("SELECT * FROM product;"),
            await reader.fetch_rows("SELECT * FROM missing;"),
        )

    accounts, products, missing = asyncio.run(run())
    assert accounts == [{"account_id": 1}]
    assert products == [{"product_id": 2}]
    assert missing == []


def test_transport_error_on_non_dict_envelope():
    from csm_env.transport.reader import EnterpriseOpsSQLRunnerReader

    class BadRunner:
        async def execute(self, query):
            return ["not", "a", "dict"]

    import asyncio

    with pytest.raises(TransportError):
        asyncio.run(EnterpriseOpsSQLRunnerReader(BadRunner()).fetch_rows("SELECT 1;"))


def test_render_query_multiple_values_uses_in(registry):
    query = render_query(
        registry,
        QuerySpec(
            table="user",
            filters=(Filter("user_id", (17, 18, 19)),),
            limit=10,
        ),
    )
    assert "WHERE user_id IN (17, 18, 19) LIMIT 10;" in query
    assert "user_id = 17 AND user_id = 18" not in query
