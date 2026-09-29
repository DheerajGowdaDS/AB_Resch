"""Tests for anchor-resolution budget discipline (Phase 1.2).

Every rule here was derived from a live probe against the real CSM server and
its real seed databases, not from theory. Each test names the failure it prevents.
"""

from __future__ import annotations

import pytest

from ablation.state_model import (
    MAX_TABLES_PER_CANDIDATE,
    TIER_BUDGET_SHARES,
    _has_identity_column,
    _search_columns,
    _split_compound,
)
from csm_env import SchemaRegistry


# ---------------------------------------------------------------------------
# Possessive decomposition
# ---------------------------------------------------------------------------


def test_possessive_compound_splits_into_two_real_entities():
    """Regression: the anchor for this task is `account.name = 'Wayne Enterprises'`.

    The whole phrase matches nothing, so without splitting the anchor is lost.
    """
    assert _split_compound("Wayne Enterprises' Windows Server") == [
        "Wayne Enterprises",
        "Windows Server",
    ]


def test_multiword_names_are_not_shattered():
    """`Windows Server` is a useful substring; splitting it would be harmful.

    Live data has `product.name = 'Windows Server 2022 Datacenter'`, so the
    semantic tier needs the two-word phrase intact.
    """
    assert _split_compound("Windows Server") == []
    assert _split_compound("Larson Operations Desk") == []
    assert _split_compound("SUSE Linux Enterprise Server") == []


@pytest.mark.parametrize("value", ["", "   ", "IBM", "It's a thing", "a b"])
def test_non_compound_values_yield_nothing(value):
    assert _split_compound(value) == []


# ---------------------------------------------------------------------------
# Searchability: only probe tables that can match
# ---------------------------------------------------------------------------


def test_customer_case_carries_a_resolvable_identifier(registry=None):
    """`customer_case.number` holds `CS-0000888` and must stay searchable.

    Leaving "number" out of the identity fragments made case-number anchors
    unreachable and regressed a task that had been delivering.
    """
    resolved = registry or SchemaRegistry.from_static()
    spec = resolved.require_table("customer_case")
    assert _has_identity_column(spec) is True
    assert "number" in _search_columns(spec, for_name=True)


def test_tables_without_identity_columns_are_excluded_from_name_search():
    resolved = SchemaRegistry.from_static()
    # `interaction` exposes only channel/interacted_user/status: nothing there can
    # ever equal an entity name, so a name probe must not spend a lookup on it.
    assert _has_identity_column(resolved.require_table("interaction")) is False
    assert _has_identity_column(resolved.require_table("entitlement")) is False


def test_name_search_drops_pure_foreign_keys():
    """`contact` declares only account_id/portal_user_id, so it can never match a name."""
    resolved = SchemaRegistry.from_static()
    spec = resolved.require_table("contact")
    assert _search_columns(spec, for_name=True) == ()


def test_identifier_search_keeps_foreign_keys():
    """An identifier probe *can* legitimately match a key column."""
    resolved = SchemaRegistry.from_static()
    spec = resolved.require_table("contact")
    assert "account_id" in _search_columns(spec)


# ---------------------------------------------------------------------------
# Budget reservation
# ---------------------------------------------------------------------------


def test_every_resolution_tier_gets_a_reserved_share():
    """Regression: the semantic tier never ran, so create-new-entity tasks failed.

    The live trace showed all 48 lookups going to exact equality on a task whose
    entities did not exist yet, so only a substring match could have worked.
    """
    for tier in ("identifier", "name_match", "semantic_lookup"):
        assert TIER_BUDGET_SHARES.get(tier, 0) > 0
    assert sum(TIER_BUDGET_SHARES.values()) == pytest.approx(1.0)


def test_semantic_tier_gets_the_largest_share():
    """It is the only tier that can match a substring, which CSM names need."""
    assert TIER_BUDGET_SHARES["semantic_lookup"] == max(TIER_BUDGET_SHARES.values())


def test_table_cap_covers_every_identity_table():
    """Budget must never be the reason a resolvable anchor is missed.

    CSM has nine identity-capable tables; the cap is set so all of them fit
    inside one candidate's fan-out.
    """
    resolved = SchemaRegistry.from_static()
    identity_tables = [
        spec.table
        for spec in resolved.manifest.tables
        if _has_identity_column(spec)
    ]
    assert len(identity_tables) <= MAX_TABLES_PER_CANDIDATE
