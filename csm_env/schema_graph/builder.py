from __future__ import annotations

from ..schema_spec.registry import SchemaRegistry
from ..schema_spec.validator import validate_manifest
from .graph import SchemaGraph
from .models import SchemaEdge, TableNode


def build_schema_graph(registry: SchemaRegistry) -> SchemaGraph:
    """Build the structural schema graph from a registry.

    Structural FK edges are generated first; semantic labels exist only as
    annotations derived from each edge's relation. Duplicate edge IDs are
    rejected (they would mean duplicate FK definitions upstream).
    """
    problems = validate_manifest(registry.manifest)
    if problems:
        raise ValueError(f"Cannot build schema graph from invalid manifest: {problems}")

    nodes = {spec.table: TableNode.from_spec(spec) for spec in registry.manifest.tables}
    edges: dict[str, SchemaEdge] = {}
    for fk in registry.manifest.foreign_keys:
        edge = SchemaEdge.from_spec(fk)
        if edge.edge_id in edges:
            raise ValueError(f"Duplicate schema edge: {edge.edge_id}")
        edges[edge.edge_id] = edge
    return SchemaGraph(nodes=nodes, edges=edges)
