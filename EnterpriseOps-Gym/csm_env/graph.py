from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class Node:
    id: str
    type: str
    table: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    relation: str
    properties: Dict[str, Any] = field(default_factory=dict)


class EnvironmentGraph:
    """Small dependency-free property graph used as the environment view.

    The graph is intentionally a state/cache representation. The relational
    database remains the source of truth; callers should refresh nodes before
    making ground-truth claims when freshness matters.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self._edge_keys: set[Tuple[str, str, str]] = set()

    @staticmethod
    def node_key(node_type: str, entity_id: Any) -> str:
        return f"{node_type}:{entity_id}"

    def add_node(self, node: Node) -> str:
        key = self.node_key(node.type, node.id)
        existing = self.nodes.get(key)
        if existing:
            existing.properties.update(node.properties)
        else:
            self.nodes[key] = node
        return key

    def add_edge(self, edge: Edge) -> None:
        key = (edge.source, edge.target, edge.relation)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self.edges.append(edge)

    def get_node(self, node_type: str, entity_id: Any) -> Optional[Node]:
        return self.nodes.get(self.node_key(node_type, entity_id))

    def neighbors(
        self,
        node_type: str,
        entity_id: Any,
        relation: Optional[str] = None,
        direction: str = "both",
    ) -> List[Tuple[Node, Edge]]:
        key = self.node_key(node_type, entity_id)
        output: List[Tuple[Node, Edge]] = []
        for edge in self.edges:
            forward = direction in ("out", "both") and edge.source == key
            backward = direction in ("in", "both") and edge.target == key
            if not (forward or backward):
                continue
            if relation and edge.relation != relation:
                continue
            other_key = edge.target if forward else edge.source
            node = self.nodes.get(other_key)
            if node:
                output.append((node, edge))
        return output

    def subgraph_for(self, node_type: str, entity_id: Any, max_hops: int = 2) -> "EnvironmentGraph":
        root = self.node_key(node_type, entity_id)
        selected = {root}
        frontier = {root}
        for _ in range(max_hops):
            nxt = set()
            for edge in self.edges:
                if edge.source in frontier:
                    nxt.add(edge.target)
                if edge.target in frontier:
                    nxt.add(edge.source)
            nxt -= selected
            selected |= nxt
            frontier = nxt
            if not frontier:
                break

        result = EnvironmentGraph()
        for key in selected:
            if key in self.nodes:
                result.nodes[key] = self.nodes[key]
        for edge in self.edges:
            if edge.source in selected and edge.target in selected:
                result.add_edge(edge)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {"id": n.id, "type": n.type, "table": n.table, "properties": n.properties}
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "relation": e.relation,
                    "properties": e.properties,
                }
                for e in self.edges
            ],
        }

    def __len__(self) -> int:
        return len(self.nodes)
