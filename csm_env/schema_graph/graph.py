from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..schema_spec.models import ForeignKeySpec
from .models import SchemaEdge, SemanticAnnotation, TableNode


class SchemaGraph:
    """Schema-only graph: TableNodes + FK-backed SchemaEdges.

    Validated against the registry at construction; rejects endpoints or
    relations absent from the manifest. Contains no record data.
    """

    def __init__(self, nodes: Dict[str, TableNode], edges: Dict[str, SchemaEdge]) -> None:
        self._nodes = dict(nodes)
        self._edges = dict(edges)
        self._outgoing: Dict[str, List[SchemaEdge]] = {}
        self._incoming: Dict[str, List[SchemaEdge]] = {}
        for edge in self._edges.values():
            if edge.source_table not in self._nodes:
                raise ValueError(f"Edge endpoint not in graph: {edge.source_table}")
            if edge.target_table not in self._nodes:
                raise ValueError(f"Edge endpoint not in graph: {edge.target_table}")
            self._outgoing.setdefault(edge.source_table, []).append(edge)
            self._incoming.setdefault(edge.target_table, []).append(edge)
        for mapping in (self._outgoing, self._incoming):
            for key in mapping:
                mapping[key].sort(key=lambda e: e.edge_id)

    def nodes(self) -> Tuple[TableNode, ...]:
        return tuple(self._nodes[table] for table in sorted(self._nodes))

    def edges(self) -> Tuple[SchemaEdge, ...]:
        return tuple(self._edges[edge_id] for edge_id in sorted(self._edges))

    def node(self, table: str) -> Optional[TableNode]:
        return self._nodes.get(table.lower())

    def get_edge(self, edge_id: str) -> Optional[SchemaEdge]:
        return self._edges.get(edge_id)

    def outgoing(self, table: str) -> Tuple[SchemaEdge, ...]:
        return tuple(self._outgoing.get(table.lower(), ()))

    def incoming(self, table: str) -> Tuple[SchemaEdge, ...]:
        return tuple(self._incoming.get(table.lower(), ()))

    def neighbors(self, table: str) -> Tuple[TableNode, ...]:
        names = {edge.target_table for edge in self.outgoing(table)}
        names |= {edge.source_table for edge in self.incoming(table)}
        return tuple(self._nodes[name] for name in sorted(names))

    def annotations(self) -> Tuple[SemanticAnnotation, ...]:
        return tuple(
            SemanticAnnotation(edge_id=edge.edge_id, label=edge.relation)
            for edge in self.edges()
        )
