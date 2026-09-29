from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..schema_graph.graph import SchemaGraph
from .models import Filter, QueryBudget, QueryPlan, QueryStep, TaskRequirements
from .requirements import resolve_attribute_columns

PLAN_COLUMNS_LIMIT = 24

#: Base cost of traversing one structural edge (the hop penalty of the
#: blueprint's ``w(e) = task relevance + relationship necessity + hop penalty``,
#: section 5). The two additive component constants are defined inside
#: :meth:`QueryPlanner._shortest_path` where their meaning is documented.
HOP_COST = 1.0

#: Relevance penalty for an edge that does not touch a required table.
IRRELEVANT_EDGE_PENALTY = 0.25

#: Relevance penalty for an edge that touches a required table but does not
#: realize a required relation.
REQUIRED_TABLE_PENALTY = 0.05

#: Penalty for an edge that is one of several structural alternatives.
NECESSITY_PENALTY = 0.05

#: Column fragments that are *always* needed to interpret a row, regardless of
#: task. The primary key and required foreign keys are added structurally;
#: these fragments cover the display/identity columns a human or an agent needs
#: to recognize what a row is (blueprint section 9: "attributes needed to
#: interpret the relationship").
_IDENTITY_COLUMN_FRAGMENTS: Tuple[str, ...] = (
    "name",
    "title",
    "number",
    "serial",
    "email",
    "first_name",
    "last_name",
    "state",
    "status",
    "priority",
    "support_level",
    "active",
)


def project_columns(
    table: str,
    *,
    registry,
    required_attributes: Sequence[str] = (),
    required_tables: Sequence[str] = (),
    keep_all_fks: bool = False,
) -> Optional[Tuple[str, ...]]:
    """Task-conditioned column projection (State Model 2.0, blueprint section 9).

    The kept set is exactly:

    1. the table's primary key;
    2. every foreign key needed to realize the task's required relations
       (or all outgoing FKs when the table's relation targets are unknown, or
       when ``keep_all_fks`` is set);
    3. task-relevant attributes named by ``required_attributes``;
    4. identity/interpretation columns (:data:`_IDENTITY_COLUMN_FRAGMENTS`).

    ``keep_all_fks`` implements the ``intent`` dimension of the requirement
    object (blueprint section 3). A task whose intent includes ``update`` must
    be able to *write* the row it was given, so its foreign keys are mutable
    context that must survive attribute pruning; narrowing them to only the
    required tables would drop the very handles the update needs.

    Everything else — prices, timestamps, free-text bodies — is pruned before
    the query is built, which reduces token cost without losing grounding.

    Returns ``None`` when the projection would keep every column anyway (so
    the adapter's default full projection is used) and an explicit tuple
    otherwise. Unknown attribute names are ignored rather than fatal: the
    requirement registry already fails closed on malformed attribute specs.
    """
    spec = registry.require_table(table)
    all_columns = spec.column_names()
    keep: Set[str] = {spec.primary_key}

    # Foreign keys. Required-table knowledge lets us keep only the FKs the
    # task's relational structure actually follows. ``keep_all_fks`` suspends
    # that narrowing for an ``update`` intent, whose FKs are mutable context.
    required_lower = {table_name.lower() for table_name in required_tables}
    kept_any_fk = False
    for fk in registry.outgoing_fks(table):
        if keep_all_fks or not required_lower or fk.target_table.lower() in required_lower:
            keep.add(fk.source_column)
            kept_any_fk = True
    if not kept_any_fk:
        # No required-table context: keep all structural FKs so relation
        # interpretation never silently degrades.
        for fk in registry.outgoing_fks(table):
            keep.add(fk.source_column)

    # Task-relevant attributes. Resolved through the SHARED lexicon resolver
    # (P0/A1) so the query layer and the state layer can never drift: the
    # prompt shorthand "assignment_group" realizes the concrete
    # "assignment_group_id" FK exactly as the requirement registry does. The
    # keys are resolved against this table's own columns.
    for attribute in resolve_attribute_columns(registry, (table,), required_attributes):
        keep.add(attribute)

    # Identity/interpretation columns. Matching is exact or on a ``_fragment``
    # suffix: an any-token test would wrongly keep ``email_domain`` (which
    # does not name a row) because it contains ``email``, while a suffix test
    # correctly keeps ``lifecycle_state``, ``serial_number`` and
    # ``first_name``/``last_name``.
    for column in all_columns:
        if column in _IDENTITY_COLUMN_FRAGMENTS:
            keep.add(column)
            continue
        if column.endswith(tuple(f"_{fragment}" for fragment in _IDENTITY_COLUMN_FRAGMENTS)):
            keep.add(column)

    if len(keep) >= len(all_columns):
        return None
    ordered = tuple(column for column in all_columns if column in keep)
    return ordered[:PLAN_COLUMNS_LIMIT]


class QueryPlanner:
    """Build a deterministic schema-graph query plan with path discovery.

    Requirements name destination tables. The planner discovers shortest
    structural FK paths from the anchor to each destination, includes required
    intermediate bridge tables automatically, and then takes the union of
    those paths. It therefore does not require callers to enumerate join
    bridges such as CustomerCase -> Account -> Entitlement.

    State Model 2.0 (blueprint section 5): when the requirements carry
    ``required_relations``, the path search prefers edges that realize those
    task-relevant pairs over equally short structural alternatives, so the
    plan connects exactly the entities the task asks about rather than any
    valid schema path.
    """

    def __init__(self, graph: SchemaGraph, registry) -> None:
        self._graph = graph
        self._registry = registry
        self._degree_cache: Dict[str, int] = {}

    def _degree(self, table: str) -> int:
        """Structural degree (in + out) of one table, memoized per planner."""
        cached = self._degree_cache.get(table)
        if cached is None:
            cached = len(self._graph.outgoing(table)) + len(self._graph.incoming(table))
            self._degree_cache[table] = cached
        return cached

    def _is_only_link(self, edge) -> bool:
        """Whether an edge is the only structural link at one of its endpoints.

        Such an edge is *unavoidable*: any path to that endpoint must traverse
        it, so charging a choice penalty would be wrong. This is the
        "relationship necessity" component of the blueprint's edge weight.
        """
        return (
            self._degree(edge.source_table) <= 1
            or self._degree(edge.target_table) <= 1
        )

    def plan(
        self,
        task_type: str,
        anchor_table: str,
        anchor_key: str,
        anchor_value,
        requirements: TaskRequirements,
        budget: QueryBudget,
        route_id: str,
    ) -> QueryPlan:
        if budget.max_hops < 0 or budget.max_tables < 1:
            raise ValueError("query budget max_hops must be >= 0 and max_tables >= 1")

        required = {table.lower() for table in requirements.required_tables}
        unknown = sorted(table for table in required if self._registry.table(table) is None)
        if unknown:
            raise KeyError(f"Requirement references unknown tables: {unknown}")
        required.discard(anchor_table.lower())

        # Task-conditioned column projection (blueprint section 9). Attributes
        # come from the requirements; required tables scope the FK projection.
        required_attributes = tuple(requirements.required_attributes)
        required_tables_for_projection = tuple(sorted(required | {anchor_table.lower()}))

        # Discover a path for each required destination against the tree built so far.
        # Existing required nodes are preferred as routing anchors. This prevents
        # ambiguous equal-length routes from selecting an unrelated bridge (for
        # example Account -> Contact -> User) when CustomerCase is already a
        # required node and provides the intended Account -> CustomerCase -> User
        # fan-out.
        parents: Dict[str, Optional[str]] = {anchor_table: None}
        parent_edge: Dict[str, object] = {}
        parent_direction: Dict[str, str] = {}
        parent_filter_column: Dict[str, str] = {}
        parent_binding_column: Dict[str, str] = {}
        distance: Dict[str, int] = {anchor_table: 0}
        order: List[str] = [anchor_table]

        # Context used by the section-5 edge weight: the task's required tables
        # plus the anchor (the anchor is always structurally relevant).
        required_context: Set[str] = set(required) | {anchor_table.lower()}
        relation_pairs: Set[Tuple[str, str]] = {
            (str(source).lower(), str(target).lower())
            for source, target in requirements.required_relations
        }

        for target in sorted(required):
            path = self._shortest_path_from_tree(
                target,
                parents.keys(),
                distance,
                required,
                budget.max_hops,
                required_relations=requirements.required_relations,
                required_context=required_context,
            )
            if path is None:
                raise ValueError(
                    f"No structural path from {anchor_table} to required table {target!r} "
                    f"within max_hops={budget.max_hops}; refusing to fabricate relationships"
                )

            source, path_nodes = path
            predecessor = source
            for table, edge, direction, filter_column, binding_column in path_nodes:
                if table in parents:
                    predecessor = table
                    continue
                parents[table] = predecessor
                parent_edge[table] = edge
                parent_direction[table] = direction
                parent_filter_column[table] = filter_column
                parent_binding_column[table] = binding_column
                distance[table] = distance[predecessor] + 1
                order.append(table)
                predecessor = table

        if len(order) > budget.max_tables:
            raise ValueError(
                f"Required path closure needs {len(order)} tables, exceeding "
                f"max_tables={budget.max_tables}"
            )

        # The shortest path representation can be simplified: for each target,
        # paths are materialized in forward order. The union insertion above
        # gives parent-before-child ordering because every path starts at the
        # anchor and adds ancestors before descendants.
        anchor_step = QueryStep(
            step_id="s000",
            table=anchor_table,
            filters=(Filter(column=anchor_key, value=anchor_value),),
            columns=project_columns(
                anchor_table,
                registry=self._registry,
                required_attributes=required_attributes,
                required_tables=required_tables_for_projection,
            ),
            limit=1,
            hop=0,
        )
        steps: List[QueryStep] = [anchor_step]
        step_id_by_table = {anchor_table: "s000"}
        for table in order[1:]:
            step_id = f"s{len(steps):03d}"
            parent = parents[table]
            steps.append(
                QueryStep(
                    step_id=step_id,
                    table=table,
                    filters=(),
                    columns=project_columns(
                        table,
                        registry=self._registry,
                        required_attributes=required_attributes,
                        required_tables=required_tables_for_projection,
                    ),
                    limit=budget.max_rows_per_query,
                    depends_on=step_id_by_table[parent],
                    edge_id=parent_edge[table].edge_id,
                    direction=parent_direction[table],
                    hop=distance[table],
                    filter_column=parent_filter_column[table],
                    binding_column=parent_binding_column[table],
                    # Section-5 weight of the hop that introduces this step.
                    # The frontier strategy executes the cheapest step that can
                    # still add coverage, so this is the retrieval priority.
                    priority=self.edge_weight(
                        parent_direction[table],
                        parent_edge[table],
                        required_tables=required_context,
                        required_relations=relation_pairs,
                    ),
                )
            )
            step_id_by_table[table] = step_id

        self._validate_plan(steps)
        return QueryPlan(
            anchor_table=anchor_table,
            anchor_key=anchor_key,
            anchor_value=anchor_value,
            steps=tuple(steps),
            budget=budget,
            task_type=task_type,
            route_id=route_id,
        )

    def _shortest_path_from_tree(
        self,
        target_table: str,
        existing_tables,
        distances: Dict[str, int],
        required: Set[str],
        max_hops: int,
        *,
        required_relations: Tuple[Tuple[str, str], ...] = (),
        required_context: Optional[Set[str]] = None,
    ) -> Optional[Tuple[str, List[Tuple[str, object, str, str, str]]]]:
        """Find the cheapest path to a target from the current planned tree.

        Required tables already in the tree are preferred over incidental bridge
        tables when total hop cost ties. This makes multi-target plans reuse the
        task's semantic anchors and preserves one-to-many fan-out opportunities.

        State Model 2.0: when the task names required relations, a path whose
        edges realize those pairs is preferred over an equally short path that
        does not (weighted shortest path, blueprint section 5).
        """
        if target_table in existing_tables:
            return target_table, []

        relation_pairs = {
            (source.lower(), target.lower()) for source, target in required_relations
        }

        candidates = []
        for source in sorted(existing_tables):
            remaining_hops = max_hops - distances.get(source, 0)
            if remaining_hops < 1:
                continue
            path = self._shortest_path(
                source,
                target_table,
                remaining_hops,
                required_relations=relation_pairs,
                required_tables=(
                    set(required) if required_context is None else set(required_context)
                ),
            )
            if path is None:
                continue
            total = distances.get(source, 0) + len(path)
            # Prefer total distance, then required source, then
            # relation-relevant edges, then source name.
            relevance = 1 if self._path_is_relation_relevant(path, relation_pairs) else 0
            candidates.append((total, 0 if source in required else 1, source, relevance, path))
        if not candidates:
            return None
        _, _, source, _, path = min(
            candidates, key=lambda item: (item[0], item[1], -item[3], item[2])
        )
        return source, path

    @staticmethod
    def _path_is_relation_relevant(
        path: Sequence[Tuple[str, object, str, str, str]],
        relation_pairs: Set[Tuple[str, str]],
    ) -> bool:
        """Whether any edge of the path realizes a task-required relation pair."""
        if not relation_pairs:
            return False
        for _table, edge, direction, _filter_column, _binding_column in path:
            if direction == "outgoing":
                pair = (edge.source_table.lower(), edge.target_table.lower())
            else:
                pair = (edge.target_table.lower(), edge.source_table.lower())
            if pair in relation_pairs:
                return True
        return False

    def edge_weight(
        self,
        direction: str,
        edge,
        *,
        required_tables: Set[str] = frozenset(),
        required_relations: Set[Tuple[str, str]] = frozenset(),
    ) -> float:
        """``w(e)`` for one structural edge (blueprint section 5).

        Three additive components, exactly as specified:

        1. hop penalty - :data:`HOP_COST`, the base cost of one traversal;
        2. task relevance - ``0`` when the edge realizes a required relation,
           :data:`REQUIRED_TABLE_PENALTY` when it touches a required table, and
           :data:`IRRELEVANT_EDGE_PENALTY` otherwise;
        3. relationship necessity - ``0`` when the edge is the only structural
           link at one of its endpoints (unavoidable), else
           :data:`NECESSITY_PENALTY` for being one of several alternatives.

        With no required tables or relations the weight is exactly
        :data:`HOP_COST`, so context-free planning reproduces the historical
        hop-count search.
        """
        if not (required_tables or required_relations):
            return HOP_COST

        if direction == "outgoing":
            pair = (edge.source_table.lower(), edge.target_table.lower())
        else:
            pair = (edge.target_table.lower(), edge.source_table.lower())

        if required_relations and pair in required_relations:
            relevance = 0.0
        elif pair[0] in required_tables or pair[1] in required_tables:
            relevance = REQUIRED_TABLE_PENALTY
        else:
            relevance = IRRELEVANT_EDGE_PENALTY

        necessity = 0.0 if self._is_only_link(edge) else NECESSITY_PENALTY
        return HOP_COST + relevance + necessity

    def _shortest_path(
        self,
        anchor_table: str,
        target_table: str,
        max_hops: int,
        *,
        required_relations: Set[Tuple[str, str]] = frozenset(),
        required_tables: Set[str] = frozenset(),
    ) -> Optional[List[Tuple[str, object, str, str, str]]]:
        """Return path excluding anchor: (table, edge, direction, filter, binding).

        Weighted search (blueprint section 5). Each edge costs

            ``hop penalty + task relevance + relationship necessity``

        where

        1. **hop penalty** is :data:`HOP_COST` - the base cost of one traversal;
        2. **task relevance** is ``0`` for an edge that realizes a required
           relation, :data:`REQUIRED_TABLE_PENALTY` for an edge touching a
           required table, and :data:`IRRELEVANT_EDGE_PENALTY` otherwise;
        3. **relationship necessity** is ``0`` for an edge that is the only
           structural link at one of its endpoints (unavoidable) and
           :data:`NECESSITY_PENALTY` for a choice among alternatives.

        The components only engage when the task supplies discriminating
        context: with no required tables or relations every edge costs exactly
        :data:`HOP_COST`, which reproduces the historical hop-count search byte
        for byte. Reachability is never lost - the penalties are additive
        pressures, not filters.
        """
        if anchor_table == target_table:
            return []

        def edge_cost(direction: str, edge) -> float:
            return self.edge_weight(
                direction,
                edge,
                required_tables=required_tables,
                required_relations=required_relations,
            )

        # queue item: (current_table, path, cost)
        queue = deque([(anchor_table, [], 0.0)])
        best_cost: Dict[str, float] = {anchor_table: 0.0}
        while queue:
            current, path, cost = queue.popleft()
            if len(path) >= max_hops:
                continue
            for neighbor, filter_column, binding_column, edge, direction in self._candidates(current):
                step_cost = cost + edge_cost(direction, edge)
                if neighbor in best_cost and best_cost[neighbor] <= step_cost:
                    continue
                best_cost[neighbor] = step_cost
                next_path = path + [(neighbor, edge, direction, filter_column, binding_column)]
                if neighbor == target_table:
                    return next_path
                queue.append((neighbor, next_path, step_cost))
        return None

    def _candidates(self, table: str) -> List[Tuple[str, str, str, object, str]]:
        output: List[Tuple[str, str, str, object, str]] = []
        for edge in self._graph.outgoing(table):
            output.append((edge.target_table, edge.target_column, edge.source_column, edge, "outgoing"))
        for edge in self._graph.incoming(table):
            output.append((edge.source_table, edge.source_column, edge.target_column, edge, "incoming"))
        return sorted(output, key=lambda item: (item[0], item[3].edge_id, item[4]))

    def _columns_for(self, table: str) -> Optional[Tuple[str, ...]]:
        """Full table projection, capped at :data:`PLAN_COLUMNS_LIMIT`.

        Retained for callers that plan without task context; the task-
        conditioned path uses :func:`project_columns` instead.
        """
        names = self._registry.require_table(table).column_names()
        return names[:PLAN_COLUMNS_LIMIT] if len(names) > PLAN_COLUMNS_LIMIT else names

    def _validate_plan(self, steps: List[QueryStep]) -> None:
        table_of = {step.step_id: step.table for step in steps}
        for step in steps:
            if step.hop == 0:
                continue
            if step.depends_on not in table_of:
                raise ValueError(f"Step {step.step_id} depends on unknown step {step.depends_on!r}")
            edge = self._graph.get_edge(step.edge_id or "")
            if edge is None:
                raise ValueError(f"Step {step.step_id} uses non-existent edge {step.edge_id!r}")
            dependency_table = table_of[step.depends_on]
            forward = edge.source_table == dependency_table and edge.target_table == step.table
            backward = edge.target_table == dependency_table and edge.source_table == step.table
            if not (forward or backward):
                raise ValueError(
                    f"Step {step.step_id} endpoints {dependency_table}->{step.table} do not match {edge.edge_id}"
                )
            expected_filter, expected_binding = (
                (edge.target_column, edge.source_column)
                if forward
                else (edge.source_column, edge.target_column)
            )
            if step.filter_column != expected_filter or step.binding_column != expected_binding:
                raise ValueError(f"Step {step.step_id} key columns do not match edge {edge.edge_id}")
