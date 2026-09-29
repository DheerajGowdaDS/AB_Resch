from __future__ import annotations

import pytest

from csm_env.schema_spec import SchemaRegistry
from csm_env.schema_graph import SchemaEdge, SchemaGraph, TableNode, build_schema_graph


@pytest.fixture()
def registry() -> SchemaRegistry:
    return SchemaRegistry.from_static()


@pytest.fixture()
def graph(registry: SchemaRegistry) -> SchemaGraph:
    return build_schema_graph(registry)


def test_graph_contains_exactly_manifest_tables(graph: SchemaGraph, registry: SchemaRegistry):
    assert {node.table for node in graph.nodes()} == set(registry.tables())
    assert len(graph.nodes()) == 17


def test_graph_contains_exactly_declared_fks(graph: SchemaGraph, registry: SchemaRegistry):
    expected_ids = {fk.edge_id for fk in registry.manifest.foreign_keys}
    actual_ids = {edge.edge_id for edge in graph.edges()}
    assert actual_ids == expected_ids
    assert len(actual_ids) == 29


def test_edge_directionality(graph: SchemaGraph):
    case_out = {edge.target_table for edge in graph.outgoing("customer_case")}
    assert case_out == {"account", "contact", "product", "installed_product", "user", "user_group"}
    assert graph.incoming("customer_case")
    assert all(
        edge.target_table == "customer_case" for edge in graph.incoming("customer_case")
    )


def test_neighbors_both_directions(graph: SchemaGraph):
    names = {node.table for node in graph.neighbors("account")}
    assert "contract" in names
    assert "entitlement" in names
    assert "customer_case" in names


def test_get_edge_and_node(graph: SchemaGraph):
    edge = graph.get_edge("customer_case.account_id->account.account_id")
    assert edge is not None
    assert edge.relation == "BELONGS_TO"
    node = graph.node("customer_case")
    assert node.primary_key == "case_id"
    assert node.node_type == "CustomerCase"
    assert "case_id" in node.columns


def test_no_fabricated_edges(graph: SchemaGraph, registry: SchemaRegistry):
    for edge in graph.edges():
        table = registry.require_table(edge.source_table)
        assert edge.source_column in table.column_names()
        target = registry.require_table(edge.target_table)
        assert edge.target_column in target.column_names()


def test_rejects_edge_with_unknown_endpoint():
    node = TableNode.from_spec(
        __import__("csm_env.schema_spec.models", fromlist=["TableSpec"]).TableSpec(
            table="a", node_type="A", primary_key="id", columns=(
                __import__("csm_env.schema_spec.models", fromlist=["ColumnSpec"]).ColumnSpec("id", "string"),
            )
        )
    )
    orphan = SchemaEdge("a.id->b.id", "a", "id", "b", "id", "REL")
    with pytest.raises(ValueError, match="endpoint not in graph"):
        SchemaGraph(nodes={"a": node}, edges={"a.id->b.id": orphan})


def test_semantic_annotations_reference_structural_edges(graph: SchemaGraph):
    annotations = graph.annotations()
    edge_ids = {edge.edge_id for edge in graph.edges()}
    assert len(annotations) == len(edge_ids)
    for annotation in annotations:
        assert annotation.edge_id in edge_ids


def test_builder_rejects_invalid_manifest():
    from csm_env.schema_spec.models import (
        ColumnSpec,
        ForeignKeySpec,
        SchemaManifest,
        SchemaSource,
        TableSpec,
    )

    table = TableSpec("account", "Account", "account_id", (ColumnSpec("account_id", "string"),))
    bad_fk = ForeignKeySpec("account", "nope", "ghost", "nope", "BAD")
    manifest = SchemaManifest(
        schema_version="1.0.0",
        tables=(table,),
        foreign_keys=(bad_fk,),
        source=SchemaSource.STATIC_REGISTRY,
    )
    registry = SchemaRegistry(manifest)
    with pytest.raises(ValueError, match="invalid manifest"):
        build_schema_graph(registry)


def test_record_graph_stays_separate():
    import csm_env.graph as record_graph_module

    import csm_env.schema_graph as schema_graph_module

    assert record_graph_module is not schema_graph_module
    assert not hasattr(schema_graph_module, "EnvironmentGraph")
