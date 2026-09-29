from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .graph import Edge, EnvironmentGraph, Node
from .schema import (
    ENTITY_BY_TABLE,
    ENTITY_BY_TYPE,
    FK_BY_SOURCE,
    FK_BY_TARGET,
    TABLE_COLUMNS,
)
from .sql_runner import EnterpriseOpsSQLRunner


class CSMEnvironmentRepresentation:
    """Queryable, database-backed CSM environment representation.

    The graph is a cache/view. Every public *ground-truth* getter can bypass the
    cache and read directly from the live CSM database through /api/sql-runner.
    """

    def __init__(self, sql: EnterpriseOpsSQLRunner) -> None:
        self.sql = sql
        self.graph = EnvironmentGraph()

    # ---------- safe SQL helpers ----------
    @staticmethod
    def _sql_literal(value: Any) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, (int, float)):
            return str(value)
        text = str(value).replace("'", "''")
        return f"'{text}'"

    @staticmethod
    def _validated_table(entity: str) -> str:
        if entity in ENTITY_BY_TABLE:
            return entity
        if entity in ENTITY_BY_TYPE:
            return ENTITY_BY_TYPE[entity].table
        raise KeyError(f"Unknown CSM entity: {entity}")

    # ---------- ground truth ----------
    async def get_current_state(self, entity: str, entity_id: Any) -> Dict[str, Any]:
        """Fetch the exact current row from the live database."""
        table = self._validated_table(entity)
        spec = ENTITY_BY_TABLE[table]
        query = (
            f"SELECT {', '.join(TABLE_COLUMNS[table])} "
            f"FROM {table} "
            f"WHERE {spec.primary_key} = {self._sql_literal(entity_id)} "
            "LIMIT 1;"
        )
        rows = await self.sql.fetch_rows(query)
        if not rows:
            raise KeyError(f"No {spec.node_type} with id={entity_id!r} exists in the database")
        return self._clean_row(rows[0])

    async def query(self, sql_query: str) -> List[Dict[str, Any]]:
        """Escape hatch for research code that needs a custom read-only query."""
        return await self.sql.fetch_rows(sql_query)

    # ---------- graph hydration ----------
    async def hydrate_entity(self, entity: str, entity_id: Any, include_links: bool = True) -> Node:
        table = self._validated_table(entity)
        spec = ENTITY_BY_TABLE[table]
        row = await self.get_current_state(table, entity_id)
        node = self._upsert_node(spec.table, row)

        if include_links:
            await self._hydrate_links(spec.table, row)
        return node

    async def hydrate_case(self, case_id: Any, hops: int = 2) -> EnvironmentGraph:
        """Build a case-centered subgraph from current DB state."""
        await self.hydrate_entity("customer_case", case_id, include_links=True)
        frontier = [("CustomerCase", case_id)]
        visited = {("CustomerCase", str(case_id))}

        for _ in range(max(0, hops - 1)):
            next_frontier: list[tuple[str, Any]] = []
            for node_type, node_id in frontier:
                for neighbor, _edge in self.graph.neighbors(node_type, node_id, direction="both"):
                    key = (neighbor.type, str(neighbor.id))
                    if key in visited:
                        continue
                    visited.add(key)
                    await self.hydrate_entity(neighbor.type, neighbor.id, include_links=False)
                    await self._hydrate_links(neighbor.table, neighbor.properties)
                    next_frontier.append((neighbor.type, neighbor.id))
            frontier = next_frontier
            if not frontier:
                break

        return self.graph.subgraph_for("CustomerCase", case_id, max_hops=hops)

    async def snapshot_table(self, entity: str) -> int:
        """Materialize one CSM table into the graph. Returns inserted row count."""
        table = self._validated_table(entity)
        cols = TABLE_COLUMNS[table]
        rows = await self.sql.fetch_rows(f"SELECT {', '.join(cols)} FROM {table};")
        for row in rows:
            self._upsert_node(table, self._clean_row(row))
        return len(rows)

    async def snapshot_all(self) -> Dict[str, int]:
        """Materialize the entire CSM database.

        This is intentionally explicit because a full CSM snapshot is large.
        For agent-time usage, prefer hydrate_case/hydrate_entity.
        """
        counts: Dict[str, int] = {}
        for spec in ENTITY_BY_TABLE.values():
            counts[spec.table] = await self.snapshot_table(spec.table)
        for edge in list(self.graph.edges):
            await self._add_semantic_projection(edge)
        return counts

    # ---------- contextual views ----------
    async def get_case_context(self, case_id: Any, max_hops: int = 2) -> Dict[str, Any]:
        graph = await self.hydrate_case(case_id, hops=max_hops)
        case = graph.get_node("CustomerCase", case_id)
        if not case:
            raise KeyError(case_id)

        result: Dict[str, Any] = {
            "case": case.properties,
            "related_entities": [],
            "relations": [],
        }
        for node in graph.nodes.values():
            if node.id == str(case_id) and node.type == "CustomerCase":
                continue
            result["related_entities"].append(
                {"id": node.id, "type": node.type, "table": node.table, "properties": node.properties}
            )
        for edge in graph.edges:
            result["relations"].append(
                {
                    "source": edge.source,
                    "relation": edge.relation,
                    "target": edge.target,
                    "properties": edge.properties,
                }
            )
        return result

    def query_local_graph(
        self,
        node_type: str,
        entity_id: Any,
        relation: Optional[str] = None,
        direction: str = "both",
    ) -> List[Dict[str, Any]]:
        return [
            {
                "entity": {"id": n.id, "type": n.type, "table": n.table, "properties": n.properties},
                "edge": {"relation": e.relation, "properties": e.properties},
            }
            for n, e in self.graph.neighbors(node_type, entity_id, relation=relation, direction=direction)
        ]

    # ---------- internals ----------
    @staticmethod
    def _clean_row(row: Dict[str, Any]) -> Dict[str, Any]:
        return {str(k): v for k, v in row.items()}

    def _upsert_node(self, table: str, row: Dict[str, Any]) -> Node:
        spec = ENTITY_BY_TABLE[table]
        entity_id = row.get(spec.primary_key)
        if entity_id is None:
            raise ValueError(f"Row for {table} has no primary key {spec.primary_key}: {row}")
        properties = dict(row)
        properties["_source"] = "EnterpriseOps-Gym CSM database"
        properties["_observed_at"] = datetime.now(timezone.utc).isoformat()
        node = Node(id=str(entity_id), type=spec.node_type, table=table, properties=properties)
        self.graph.add_node(node)
        return node

    async def _hydrate_links(self, table: str, row: Dict[str, Any]) -> None:
        source_spec = ENTITY_BY_TABLE[table]
        source_key = self.graph.node_key(source_spec.node_type, row[source_spec.primary_key])

        # Outgoing foreign keys: source row -> target row.
        for fk in FK_BY_SOURCE.get(table, []):
            value = row.get(fk.source_column)
            if value is None:
                continue
            target = await self._get_one(fk.target_table, fk.target_column, value)
            if not target:
                continue
            target_node = self._upsert_node(fk.target_table, target)
            self.graph.add_edge(
                Edge(source=source_key, target=self.graph.node_key(target_node.type, target_node.id), relation=fk.relation)
            )

        # Incoming foreign keys: related source rows -> this row.
        for fk in FK_BY_TARGET.get(table, []):
            source_rows = await self._get_many(fk.source_table, fk.source_column, row[fk.target_column])
            for source_row in source_rows:
                source_node = self._upsert_node(fk.source_table, source_row)
                self.graph.add_edge(
                    Edge(
                        source=self.graph.node_key(source_node.type, source_node.id),
                        target=source_key,
                        relation=fk.relation,
                    )
                )

        # Add useful semantic shortcuts while retaining the database-faithful
        # association nodes (case_knowledge, case_sla, user_group_member).
        await self._project_semantic_links(table, row)

    async def _project_semantic_links(self, table: str, row: Dict[str, Any]) -> None:
        if table == "case_knowledge":
            case_id = row.get("case_id")
            knowledge_id = row.get("knowledge_id")
            if case_id is not None and knowledge_id is not None:
                self.graph.add_edge(
                    Edge(
                        source=self.graph.node_key("CustomerCase", case_id),
                        target=self.graph.node_key("Knowledge", knowledge_id),
                        relation="HAS_KNOWLEDGE",
                        properties={"used_as": row.get("used_as")},
                    )
                )
        elif table == "user_group_member":
            user_id = row.get("user_id")
            group_id = row.get("group_id")
            if user_id is not None and group_id is not None:
                self.graph.add_edge(
                    Edge(
                        source=self.graph.node_key("User", user_id),
                        target=self.graph.node_key("UserGroup", group_id),
                        relation="MEMBER_OF",
                        properties={"membership_id": row.get("member_id")},
                    )
                )

    async def _add_semantic_projection(self, edge: Edge) -> None:
        # Kept for future projection rules that depend on a complete snapshot.
        return None

    async def _get_one(self, table: str, column: str, value: Any) -> Optional[Dict[str, Any]]:
        rows = await self.sql.fetch_rows(
            f"SELECT {', '.join(TABLE_COLUMNS[table])} FROM {table} "
            f"WHERE {column} = {self._sql_literal(value)} LIMIT 1;"
        )
        return self._clean_row(rows[0]) if rows else None

    async def _get_many(self, table: str, column: str, value: Any) -> List[Dict[str, Any]]:
        rows = await self.sql.fetch_rows(
            f"SELECT {', '.join(TABLE_COLUMNS[table])} FROM {table} "
            f"WHERE {column} = {self._sql_literal(value)};"
        )
        return [self._clean_row(row) for row in rows]
