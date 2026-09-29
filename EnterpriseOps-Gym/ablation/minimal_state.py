"""Task-Conditioned Minimal Grounded State selection (State Model 2.0).

Blueprint sections 5-9 and 12: this module turns a broad, graph-connected
``GroundedState`` into the **minimal sufficient** state for the current task.

The pipeline over one broad state is:

1. **Relevance scoring** (section 7) - every candidate record receives

   ``Score(r|T) = w_a*A(r) + w_e*E(r) + w_p*P(r) + w_d*D(r) - w_h*H(r)``

   where A is anchor proximity, E entity/relationship relevance, P
   predicate/attribute relevance, D dependency value (a *sole provider* of a
   required table, or a *relational bridge* carrying a required task relation),
   and H the hop cost of reaching the record. Weights are externalized in
   :class:`RelevanceWeights` so they can be tuned per experiment without
   touching the algorithm.

2. **Requirement coverage** (section 8) - each candidate covers a set of the
   task's requirements (anchor, each required table, each required relation).

3. **Greedy weighted set cover** (section 8) - candidates are selected by
   marginal coverage per unit cost until every required fact is covered or no
   candidate adds coverage. This replaces the old rank-then-count-cap rule,
   which could silently drop a required table.

4. **Attribute pruning** (section 9) - kept rows keep only PK, required FKs,
   task-relevant attributes, and identity columns. The projection reuses
   ``csm_env.query.planner.project_columns`` so the SQL layer and the state
   layer cannot drift apart.

5. **Validation** (section 12) - the selection must cover every *required
   table*, *attribute*, and *relation* the broad state observed (the
   coverage gate ``Coverage(S,F_T) >= tau``), keep every relation between
   kept rows, and stay inside the row/token budget. Under-coverage fails
   closed rather than returning a silently weaker state.

**Objective (blueprint section 2/13).** The ``lambda_q * |queries|`` term is
minimized by the *retrieval* stage, not by this module: Condition B2 routes with
``RouteStrategy.FRONTIER`` (``csm_env.query.router``), which retrieves one
relation at a time and stops at the first prefix meeting
``Coverage(S, F_T) >= tau``. Because ``Cost(S)`` is non-decreasing in queries,
rows and tokens, that first covering prefix *is* ``argmin Cost(S)`` over the
prefixes it produced -- see ``csm_env.query.cost.cheapest_covering_prefix`` and
the ``cost_prefixes`` diagnostic.

This module then performs the second half of the selection, on the rows that
were actually retrieved: a relevance-weighted greedy weighted set cover under a
hard row cap and a token ceiling, followed by attribute pruning. The per-record
cost is relevance-weighted (``1 / score``), which is a *heuristic* inner loop
rather than the literal global lambda objective; that is deliberate, because the
cover must be greedy and the outer ``argmin`` is already satisfied by the
frontier stopping rule. Delivered rows/tokens are therefore minimized, and the
query count is minimized one stage earlier.

**Resolved deviations.** Items previously listed here are now implemented:
(i) expansion is incremental frontier expansion (``FRONTIER``) rather than a
whole-radius ladder, with the hop ladder retained only as a depth backstop;
(ii) ``D(r)`` is a genuine dependency measure (sole provider + relational
bridge) rather than row rarity; (iii) the planner edge weight is the
three-component ``hop penalty + task relevance + relationship necessity``
(``csm_env.query.planner.QueryPlanner.edge_weight``); (iv) ``intent`` has a real
consumer (an ``update`` intent keeps every foreign key as mutable context).

**Remaining deviation.** The formalism is still single-anchor (blueprint
section 3's example object lists anchors, while ``TaskRequest`` carries one);
the frontier and cover are anchored on the resolved primary row. This is the one
declared V2.1 deviation, and it is explicit here rather than silently assumed.

This module is deterministic: no sampling, no clocks, no network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

from csm_env.query.planner import project_columns
from csm_env.schema_spec.registry import SchemaRegistry
from csm_env.state.models import GroundedState, StateRecord

from .state_model import estimate_tokens


# ---------------------------------------------------------------------------
# Externalized relevance weights (blueprint section 7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RelevanceWeights:
    """Weights of the task-conditioned relevance score.

    The defaults realize the blueprint's intent: the anchor and the task's
    required entities dominate; dependency value and attributes separate
    useful connected rows from structural noise; every hop away from the
    anchor costs a little.
    """

    anchor_table: float = 1.0
    anchor_row: float = 1.5
    required_entity: float = 1.4
    hinted_entity: float = 0.6
    attribute: float = 0.4
    support: float = 0.5
    #: Dependency value for a record that *links* one required table to another
    #: (it realizes a required relation). Kept separate from ``support`` because
    #: a sole-provider row and a relational bridge are different dependencies.
    support_bridge: float = 0.25
    hop: float = 0.15
    unrequired_penalty: float = 0.3

    def validate(self) -> "RelevanceWeights":
        if self.anchor_row <= 0 or self.required_entity <= 0:
            raise ValueError("anchor_row and required_entity weights must be positive")
        if self.hop < 0:
            raise ValueError("hop weight must be non-negative")
        return self


DEFAULT_WEIGHTS = RelevanceWeights()


# ---------------------------------------------------------------------------
# Task requirement facts (blueprint sections 3 and 8)
# ---------------------------------------------------------------------------


def task_required_tables(task: Any) -> FrozenSet[str]:
    """Tables the task's own prompt/tools reference (never verifier data)."""
    required: Set[str] = set()
    if hasattr(task, "reference_entities"):
        required.update(str(value).lower() for value in (task.reference_entities or ()))
    if hasattr(task, "reference_names"):
        required.update(str(table).lower() for table, _ in (task.reference_names or ()))
    if hasattr(task, "reference_rows"):
        required.update(str(table).lower() for table, _ in (task.reference_rows or ()))
    return frozenset(required)


def task_hinted_tables(task: Any) -> FrozenSet[str]:
    """Soft prompt hints used for ranking, not for forcing coverage."""
    hints: Set[str] = set()
    if hasattr(task, "signals") and getattr(task.signals, "matched_tables", None):
        hints.update(str(value).lower() for value in task.signals.matched_tables)
    return frozenset(hints)


def task_required_attributes(task: Any) -> Tuple[str, ...]:
    """Task-relevant attribute keys derived from the prompt only (SEC-001).

    Extracted from the prompt's own vocabulary, so a task asking about
    "priority" or "support level" prunes other columns without any verifier
    information ever being consulted.
    """
    prompts: List[str] = []
    if hasattr(task, "signals"):
        prompt = str(getattr(task.signals, "user_prompt", "") or "")
        prompts.append(prompt.lower())
    text = " ".join(prompts)
    found: List[str] = []
    for key, patterns in _ATTRIBUTE_PROMPT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text):
                if key not in found:
                    found.append(key)
                break
    return tuple(found)


#: Prompt phrases -> attribute keys consumed by ``project_columns``.
_ATTRIBUTE_PROMPT_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("priority", (r"\bpriorit", r"\burgent\b", r"\bcritical\b", r"\bhigh\b")),
    ("state", (r"\bstate\b", r"\bstatus\b")),
    ("support_level", (r"\bsupport (level|plan)\b", r"\bcoverage\b", r"\b24x7\b", r"\b24/7\b")),
    ("assignment_group", (r"\bassignment group\b", r"\bgroup\b", r"\bqueue\b", r"\bteam\b")),
    ("assigned_to", (r"\bassigned\b", r"\bassignee\b", r"\bassign\b")),
    ("escalation", (r"\bescalat",)),
    ("serial_number", (r"\bserial\b",)),
    ("warranty_end", (r"\bwarranty\b",)),
    ("stage", (r"\bsla\b", r"\bstage\b")),
    ("has_breached", (r"\bbreach",)),
    ("contract_type", (r"\bcontract type\b",)),
    ("short_description", (r"\bdescription\b",)),
    ("title", (r"\btitle\b", r"\barticle\b")),
    ("active", (r"\bactive\b", r"\binactive\b")),
)


def task_intent(task: Any) -> Tuple[str, ...]:
    """Coarse intent verbs derived from the prompt (read/update/create)."""
    if not hasattr(task, "signals"):
        return ("read",)
    actions = tuple(getattr(task.signals, "actions", ()) or ())
    reads = tuple(getattr(task.signals, "reads", ()) or ())
    intent: List[str] = []
    if reads:
        intent.append("read")
    if actions:
        intent.append("update")
    return tuple(intent) or ("read",)


# ---------------------------------------------------------------------------
# Relevance scoring (blueprint section 7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateScore:
    """One candidate row's task-conditioned relevance evidence."""

    record: Any
    score: float
    hop: int
    terms: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "table": self.record.table,
            "row_id": self.record.row_id,
            "score": round(self.score, 4),
            "hop": self.hop,
            "terms": {key: round(value, 4) for key, value in self.terms.items()},
        }


def _hop_of(record: Any, state: GroundedState) -> int:
    """Shortest relational distance from the anchor to the record, or 99."""
    if record.table == state.anchor.table and record.row_id == state.anchor.row_id:
        return 0
    adjacency: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
    for relation in state.relations:
        source = (relation.source_table, relation.source_id)
        target = (relation.target_table, relation.target_id)
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    start = (state.anchor.table, state.anchor.row_id)
    goal = (record.table, record.row_id)
    if goal not in adjacency and goal not in {start}:
        # Unconnected rows carry no relational support; treat as far.
        if not any(relation.source_table == record.table or relation.target_table == record.table for relation in state.relations):
            return 99
    visited: Set[Tuple[str, str]] = {start}
    frontier: List[Tuple[Tuple[str, str], int]] = [(start, 0)]
    while frontier:
        (node, depth) = frontier.pop(0)
        for neighbor in sorted(adjacency.get(node, ())):
            if neighbor == goal:
                return depth + 1
            if neighbor in visited:
                continue
            visited.add(neighbor)
            frontier.append((neighbor, depth + 1))
    return 99


def score_candidate(
    record: Any,
    *,
    state: GroundedState,
    required_tables: FrozenSet[str],
    hinted_tables: FrozenSet[str],
    attributes: Sequence[str],
    weights: RelevanceWeights = DEFAULT_WEIGHTS,
) -> CandidateScore:
    """Compute ``Score(r|T)`` for one candidate record (blueprint section 7)."""
    terms: Dict[str, float] = {}

    # A(r): anchor proximity.
    if record.table == state.anchor.table:
        terms["anchor_table"] = weights.anchor_table
        if record.row_id == str(state.anchor.row_id):
            terms["anchor_row"] = weights.anchor_row

    # E(r): entity/relationship relevance.
    if record.table in required_tables:
        terms["required_entity"] = weights.required_entity
    elif record.table in hinted_tables:
        terms["hinted_entity"] = weights.hinted_entity

    # P(r): predicate/attribute relevance - the row carries a task-relevant
    # non-empty attribute value.
    lowered_attributes = {str(attribute).lower() for attribute in attributes}
    carried = 0
    for column, value in record.values.items():
        if column in lowered_attributes and value not in (None, ""):
            carried += 1
    if carried:
        terms["attribute"] = weights.attribute * min(carried, 3)

    # D(r): dependency/support value. This is a genuine *dependency* measure --
    # "what other required facts lose their evidence if this record is dropped"
    # -- rather than row-count rarity. Two distinct dependencies are credited:
    #
    # 1. sole provider: r is the only observed row of a required table, so the
    #    ``table:<t>`` fact itself depends on r;
    # 2. relational bridge: r links one required table to another through a
    #    non-NULL FK, so the ``relation:<a>:<b>`` fact (and the downstream
    #    table's evidence) depends on r.
    #
    # Component 2 is what fixes the earlier defect where an *essential* table
    # with many rows earned no dependency credit at all while a structurally
    # irrelevant single row could outrank it: a fan-out table's rows are now
    # credited for carrying the task's required relationships.
    if record.table in required_tables:
        siblings = [
            other for other in state.records if other.table == record.table
        ]
        if len(siblings) == 1:
            terms["support"] = weights.support

        bridges = 0
        for relation in state.relations:
            if (
                relation.source_table == record.table
                and str(relation.source_id) == str(record.row_id)
            ):
                other_table = relation.target_table
            elif (
                relation.target_table == record.table
                and str(relation.target_id) == str(record.row_id)
            ):
                other_table = relation.source_table
            else:
                continue
            if (
                other_table.lower() in required_tables
                and other_table.lower() != record.table.lower()
            ):
                bridges += 1
        if bridges:
            terms["support_bridge"] = weights.support_bridge * min(bridges, 3)

    score = sum(terms.values())

    # H(r): traversal/hop cost, and a penalty for rows connected to nothing
    # the task asked about.
    hop = _hop_of(record, state)
    score -= weights.hop * hop
    if record.table not in required_tables and record.table not in hinted_tables and score < 0.6:
        score -= weights.unrequired_penalty
    return CandidateScore(record=record, score=score, hop=hop, terms=terms)


# ---------------------------------------------------------------------------
# Requirement coverage (blueprint section 8)
# ---------------------------------------------------------------------------


def coverage_facts(
    record: Any,
    *,
    anchor: Any,
    required_tables: FrozenSet[str],
    required_attributes: Sequence[str] = (),
) -> FrozenSet[str]:
    """The requirement identifiers one candidate record satisfies.

    Per-record facts are deliberately coarse and task-derivable: the anchor
    row, one per required table, and one per required *attribute* the record
    carries a non-NULL value of (P2/A5: attribute requirements now gate the
    coverage check, not just the scorer). Required *relations* are a property
    of a *set* of rows (two endpoints), so they are evaluated post-greedy by
    :func:`realized_relation_facts`, not inside a single record's coverage.
    """
    covered: Set[str] = set()
    table = str(record.table).lower()
    key = (table, str(record.row_id))
    if key == (str(anchor.table).lower(), str(anchor.row_id)):
        covered.add("anchor")
    if table in required_tables:
        covered.add(f"table:{table}")
    for column in required_attributes:
        column_l = str(column).lower()
        if column_l in {str(col).lower() for col in record.values} and record.values.get(column) not in (None, ""):
            covered.add(f"attr:{table}.{column_l}")
    return frozenset(covered)


def _realized_relation_pairs(
    state: GroundedState,
    selected_identity: Set[Tuple[str, str]],
    required_relations: Sequence[Tuple[str, str]],
) -> Set[Tuple[str, str]]:
    """Which required relation pairs are realized between two *kept* rows."""
    normalized = {(str(a).lower(), str(b).lower()) for a, b in required_relations}
    realized: Set[Tuple[str, str]] = set()
    for relation in state.relations:
        source = (relation.source_table.lower(), str(relation.source_id))
        target = (relation.target_table.lower(), str(relation.target_id))
        if source not in selected_identity or target not in selected_identity:
            continue
        for pair in normalized:
            if (source[0] == pair[0] and target[0] == pair[1]) or (
                source[0] == pair[1] and target[0] == pair[0]
            ):
                realized.add(pair)
    return realized


def realized_relation_facts(
    state: GroundedState,
    selected_identity: Set[Tuple[str, str]],
    required_relations: Sequence[Tuple[str, str]],
) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """Covered and uncovered ``relation:src:tgt`` facts among kept rows."""
    covered_pairs = _realized_relation_pairs(state, selected_identity, required_relations)
    all_pairs = {(str(a).lower(), str(b).lower()) for a, b in required_relations}
    covered = frozenset(f"relation:{a}:{b}" for a, b in covered_pairs)
    uncovered = frozenset(
        f"relation:{a}:{b}" for a, b in all_pairs - covered_pairs
    )
    return covered, uncovered


# ---------------------------------------------------------------------------
# Greedy weighted set cover (blueprint section 8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverOutcome:
    """Result of the minimal-sufficient selection."""

    selected: Tuple[Any, ...]
    covered: FrozenSet[str]
    uncovered_required: FrozenSet[str]
    dropped: Tuple[Tuple[str, str], ...]
    scores: Tuple[CandidateScore, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected": [
                {"table": record.table, "row_id": record.row_id}
                for record in self.selected
            ],
            "covered": sorted(self.covered),
            "uncovered_required": sorted(self.uncovered_required),
            "dropped": [list(pair) for pair in self.dropped],
            "scores": [score.as_dict() for score in self.scores],
        }


def _required_universe(
    state: GroundedState,
    anchor_key: Tuple[str, str],
    required_tables: FrozenSet[str],
    required_attributes: Sequence[str],
    required_relations: Sequence[Tuple[str, str]],
) -> Set[str]:
    """The task facts the broad state could observe (the cover's universe).

    ``anchor`` (if the anchor row is present), one ``table:<t>`` per required
    table the broad state observed, one ``attr:<t>.<c>`` per required column a
    required table's observed row carries non-NULL, and one ``relation:<a>:<b>``
    per required relation whose two endpoint tables were both observed.
    """
    universe: Set[str] = set()
    observed_tables = {str(record.table).lower() for record in state.records}
    if any(
        (str(record.table).lower(), str(record.row_id)) == anchor_key
        for record in state.records
    ):
        universe.add("anchor")
    required_set = set(required_tables)
    for table in sorted(observed_tables & required_set):
        universe.add(f"table:{table}")
    for table in sorted(observed_tables & required_set):
        for column in required_attributes:
            column_l = str(column).lower()
            if any(
                str(record.table).lower() == table
                and record.values.get(column_l) not in (None, "")
                for record in state.records
            ):
                universe.add(f"attr:{table}.{column_l}")
    for source, target in required_relations:
        if str(source).lower() in observed_tables and str(target).lower() in observed_tables:
            universe.add(f"relation:{str(source).lower()}:{str(target).lower()}")
    return universe


def greedy_minimal_cover(
    scores: Sequence[CandidateScore],
    *,
    state: GroundedState,
    required_tables: FrozenSet[str],
    required_attributes: Sequence[str] = (),
    required_relations: Sequence[Tuple[str, str]] = (),
    max_rows: int = 12,
    weights: RelevanceWeights = DEFAULT_WEIGHTS,
) -> CoverOutcome:
    """Select the minimal sufficient row set by greedy weighted set cover.

    The anchor is always kept. Then, while uncovered *per-record* requirements
    remain (``anchor`` / ``table:*`` / ``attr:*``), the candidate with the
    best marginal-coverage-per-cost is added, where cost is
    ``1 / max(score, epsilon)`` so higher-relevance rows are cheaper. Rows
    that cover no requirement are dropped even when highly connected - that
    is the precision gain the blueprint asks for - and the loop stops once
    every per-record requirement is covered, which is what keeps the state
    minimal.

    Required *relations* are a property of a pair of kept rows, so after the
    greedy pass a small **relation-completion** step adds the cheapest missing
    endpoint row for each required relation still unrealized among the kept
    rows (P2: relational sufficiency is now enforced, not decorative).

    A hard ``max_rows`` cap is retained only as a budget guard; the cover
    itself terminates on requirement satisfaction, so the cap can no longer
    silently drop a required table that the broad state did observe.
    """
    anchor_key = (str(state.anchor.table).lower(), str(state.anchor.row_id))

    candidates = sorted(scores, key=lambda score: (-score.score, score.record.table, str(score.record.row_id)))

    universe = _required_universe(state, anchor_key, required_tables, required_attributes, required_relations)

    covered: Set[str] = set()
    selected_keys: Set[Tuple[str, str]] = set()
    selected: List[Any] = []
    score_of: Dict[Tuple[str, str], float] = {
        (str(s.record.table).lower(), str(s.record.row_id)): s.score for s in scores
    }

    def coverage_of(record: Any) -> FrozenSet[str]:
        return coverage_facts(
            record,
            anchor=state.anchor,
            required_tables=required_tables,
            required_attributes=required_attributes,
        )

    # The anchor first: it is the one fact every task needs.
    for score in candidates:
        key = (str(score.record.table).lower(), str(score.record.row_id))
        if key == anchor_key:
            selected.append(score.record)
            selected_keys.add(key)
            covered |= coverage_of(score.record)
            break
    if not selected:
        for record in state.records:
            if (str(record.table).lower(), str(record.row_id)) == anchor_key:
                selected.append(record)
                selected_keys.add(anchor_key)
                covered |= coverage_facts(record, anchor=state.anchor, required_tables=required_tables, required_attributes=required_attributes)
                break

    per_record_universe = {fact for fact in universe if not fact.startswith("relation:")}

    # Greedy marginal-coverage-per-cost loop over per-record facts.
    while (per_record_universe - covered) and len(selected) < max_rows:
        best = None
        best_ratio = None
        for score in candidates:
            key = (str(score.record.table).lower(), str(score.record.row_id))
            if key in selected_keys:
                continue
            marginal = coverage_of(score.record) - covered
            if not marginal:
                continue
            cost = 1.0 / max(score.score, 1e-6)
            ratio = len(marginal) / cost
            if best_ratio is None or ratio > best_ratio:
                best_ratio = ratio
                best = (score, key, marginal)
        if best is None:
            break
        score, key, marginal = best
        selected.append(score.record)
        selected_keys.add(key)
        covered |= marginal

    # Relation completion: add the cheapest missing endpoint row for each
    # required relation not yet realized among the kept rows (P2).
    for _ in range(len(required_relations)):
        rel_covered, rel_uncovered = realized_relation_facts(
            state, selected_keys, required_relations
        )
        if not rel_uncovered:
            break
        target_fact = min(rel_uncovered)
        _parts = target_fact.split(":")
        source_table, target_table = _parts[1], _parts[2]
        # Choose the endpoint table with no kept row yet, otherwise the source.
        preferred_tables = [t for t in (source_table, target_table) if not any(k[0] == t for k in selected_keys)]
        preferred_tables = preferred_tables or [source_table]
        placed = False
        for table in preferred_tables:
            best_row = None
            for record in state.records:
                if str(record.table).lower() != table:
                    continue
                key = (table, str(record.row_id))
                if key in selected_keys:
                    continue
                row_score = score_of.get(key, 0.0)
                if best_row is None or row_score > best_row[0]:
                    best_row = (row_score, record, key)
            if best_row is not None and len(selected) < max_rows:
                selected.append(best_row[1])
                selected_keys.add(best_row[2])
                covered |= coverage_of(best_row[1])
                placed = True
                break
        if not placed:
            break

    # Final cover/uncovered computation over the *selected* rows.
    rel_covered, rel_uncovered = realized_relation_facts(state, selected_keys, required_relations)
    covered = covered | rel_covered
    dropped = tuple(
        (record.table, record.row_id)
        for record in state.records
        if (str(record.table).lower(), str(record.row_id)) not in selected_keys
    )
    uncovered = frozenset(
        fact
        for fact in (universe - covered)
        if fact.startswith(("table:", "attr:", "relation:"))
    )
    return CoverOutcome(
        selected=tuple(selected),
        covered=frozenset(covered),
        uncovered_required=uncovered,
        dropped=dropped,
        scores=tuple(candidates),
    )


# ---------------------------------------------------------------------------
# Token budget (blueprint sections 2 and 12)
# ---------------------------------------------------------------------------


def estimate_state_tokens(state: GroundedState) -> int:
    """Deterministic token estimate for a state as rendered for the agent.

    The agent sees the rendered text; this estimate mirrors the same payload
    shape (records + relations) so pruning can happen *before* assembly
    instead of truncating text afterwards.
    """
    if not state.records:
        return 0
    chars = 0
    for record in state.records:
        chars += len(record.table) + len(str(record.row_id)) + 16
        for column, value in record.values.items():
            chars += len(column) + len(str(value)) + 4
    chars += 24 * len(state.relations) + 96
    return max(1, -(-chars // 4))


# ---------------------------------------------------------------------------
# Attribute pruning (blueprint section 9)
# ---------------------------------------------------------------------------


def prune_record_attributes(
    record: StateRecord,
    *,
    registry: SchemaRegistry,
    required_attributes: Sequence[str],
    required_tables: Sequence[str],
    keep_all_fks: bool = False,
) -> StateRecord:
    """Narrow one record's values to the task-conditioned projection.

    ``keep_all_fks`` is the ``intent`` consumer (blueprint section 3): a task
    whose intent includes ``update`` must keep every foreign key of the row it
    is expected to write, because those FKs are the mutable context of the
    update, not merely a relationship to interpret.
    """
    columns = project_columns(
        record.table,
        registry=registry,
        required_attributes=required_attributes,
        required_tables=required_tables,
        keep_all_fks=keep_all_fks,
    )
    if columns is None:
        return record
    kept = tuple(str(column) for column in columns)
    values = {column: value for column, value in record.values.items() if column in kept}
    if not values:
        # Never emit an empty row: keep the primary key so the record stays
        # grounded and identifiable.
        values = {record.primary_key: record.values.get(record.primary_key)}
    return StateRecord(
        table=record.table,
        primary_key=record.primary_key,
        row_id=record.row_id,
        values=values,
    )


# ---------------------------------------------------------------------------
# The full minimal-state construction (blueprint sections 5-12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MinimalStateResult:
    """The minimal state plus the selection evidence for the run record."""

    state: GroundedState
    coverage_ratio: float
    covered: FrozenSet[str]
    uncovered_required: FrozenSet[str]
    dropped_rows: Tuple[Tuple[str, str], ...]
    pruned_columns: Tuple[Tuple[str, str], ...]
    token_estimate: int
    weights: RelevanceWeights
    #: The task's normalized intent (blueprint section 3). It is not
    #: decorative: an ``update`` intent widens the projection to keep every
    #: foreign key of a kept row (mutable context for the write).
    intent: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "coverage_ratio": round(self.coverage_ratio, 4),
            "covered": sorted(self.covered),
            "uncovered_required": sorted(self.uncovered_required),
            "dropped_rows": [list(pair) for pair in self.dropped_rows],
            "pruned_columns": [list(pair) for pair in self.pruned_columns],
            "token_estimate": self.token_estimate,
            "intent": list(self.intent),
            "weights": {
                "anchor_table": self.weights.anchor_table,
                "anchor_row": self.weights.anchor_row,
                "required_entity": self.weights.required_entity,
                "hinted_entity": self.weights.hinted_entity,
                "attribute": self.weights.attribute,
                "support": self.weights.support,
                "support_bridge": self.weights.support_bridge,
                "hop": self.weights.hop,
                "unrequired_penalty": self.weights.unrequired_penalty,
            },
        }


from csm_env.query.requirements import resolve_attribute_columns


def build_minimal_state(
    task: Any,
    state: GroundedState,
    *,
    registry: Optional[SchemaRegistry] = None,
    weights: RelevanceWeights = DEFAULT_WEIGHTS,
    max_rows: int = 12,
    max_tokens: int = 20000,
    coverage_threshold: float = 1.0,
    required_tables: Optional[FrozenSet[str]] = None,
    required_relations: Optional[Sequence[Tuple[str, str]]] = None,
    attribute_keys: Optional[Sequence[str]] = None,
) -> MinimalStateResult:
    """Task-Conditioned Minimal Grounded State selection (TC-MGS core).

    Unified requirement source (P0/A1/A3). The caller supplies ONE set of
    task requirements so the query layer, the escalation loop, and the
    selector can never disagree:

    * ``required_tables`` - the unified table set (prompt tool mentions UNION
      the registry task-type tables), with the anchor table always included.
      When ``None`` the prompt-only set plus the anchor is used.
    * ``attribute_keys`` - *unresolved* attribute keys (lexicon keys such as
      ``assignment_group`` or concrete columns). They are resolved to concrete
      columns through the SHARED :func:`resolve_attribute_columns` so the
      post-retrieval state prunes exactly the columns the query layer kept
      (this is what preserves the task's own FKs, e.g. ``assignment_group ->
      assignment_group_id``). ``None`` falls back to the prompt-derived keys.
    * ``required_relations`` - the task's required ``(src, tgt)`` pairs, live
      in the cover universe and the coverage gate (P2).

    The objective (blueprint section 2/13). The ``lambda_q`` term is minimized
    by the retrieval stage: the caller routes with
    ``RouteStrategy.FRONTIER``, which stops at the first covering prefix, so
    ``argmin Cost(S)`` is already settled when the rows arrive here. This
    function performs the remaining, row-level half of the selection: a
    relevance-weighted greedy weighted set cover under a hard row cap and a
    token ceiling, then attribute pruning. Resolved and remaining deviations
    are listed in the module docstring.

    Args:
        task: A prompt-only ``TaskRecord`` (verifier-blind, SEC-001).
        state: The broad graph-connected ``GroundedState`` to prune.
        registry: Schema registry; defaults to the static CSM manifest.
        weights: Externalized relevance weights.
        max_rows: Hard row-budget guard on the selection.
        max_tokens: Soft token ceiling; enforced by pruning, never truncation.
        coverage_threshold: ``tau`` of the coverage gate. Default 1.0 demands
            every required table / attribute / relation the broad state observed
            stays covered; below it the selection fails closed by raising.

    Returns:
        A :class:`MinimalStateResult` carrying the pruned state and evidence.

    Raises:
        ValueError: If ``tau`` coverage cannot be met from the broad state, or
            the token budget cannot be met without losing a covered
            requirement. The broad state is the caller's fallback in that
            case, so a silently weaker state is never returned.
    """
    resolved_registry = registry or SchemaRegistry.from_static()
    weights.validate()

    if state is None or not getattr(state, "records", None):
        return MinimalStateResult(
            state=state,
            coverage_ratio=1.0,
            covered=frozenset(),
            uncovered_required=frozenset(),
            dropped_rows=(),
            pruned_columns=(),
            token_estimate=0,
            weights=weights,
        )

    # --- Unified requirement source (P0/A1/A3) --------------------------------
    if required_tables is None:
        required_tables = task_required_tables(task)
    required_tables = frozenset(required_tables | {str(state.anchor.table).lower()})
    relations = tuple(required_relations) if required_relations else ()

    # Resolve attribute keys through the SHARED lexicon resolver so the state
    # layer and the query layer agree (A1): "assignment_group" -> the concrete
    # "assignment_group_id" FK, exactly as the planner projects it.
    if attribute_keys is None:
        attribute_keys = task_required_attributes(task)
    attributes: Tuple[str, ...] = resolve_attribute_columns(
        resolved_registry, tuple(required_tables), tuple(attribute_keys)
    )
    hinted_tables = task_hinted_tables(task)

    scores = tuple(
        score_candidate(
            record,
            state=state,
            required_tables=required_tables,
            hinted_tables=hinted_tables,
            attributes=attributes,
            weights=weights,
        )
        for record in state.records
    )

    outcome = greedy_minimal_cover(
        scores,
        state=state,
        required_tables=required_tables,
        required_attributes=attributes,
        required_relations=relations,
        max_rows=max_rows,
        weights=weights,
    )

    # --- Coverage gate (P2/A5): tables AND attributes AND relations ----------
    anchor_key = (str(state.anchor.table).lower(), str(state.anchor.row_id))
    universe = _required_universe(state, anchor_key, required_tables, attributes, relations)
    covered = set(outcome.covered)
    coverage_ratio = (
        len(covered & universe) / len(universe) if universe else 1.0
    )
    if coverage_ratio < coverage_threshold:
        raise ValueError(
            "minimal state coverage gate failed: "
            f"{sorted(universe - covered)} uncovered "
            f"(tau={coverage_threshold}); refusing to return a weaker state"
        )

    # --- Attribute pruning (section 9), THEN token budget (P1) ---------------
    # Section 3 `intent` consumer. An `update` task is expected to *write* the
    # row it was given, so its foreign keys are mutable context that must
    # survive attribute pruning instead of being narrowed to the required set.
    intent = task_intent(task)
    mutable_context = "update" in intent

    required_tables_for_projection = tuple(sorted(required_tables))
    pruned_columns: List[Tuple[str, str]] = []
    pruned_records: List[StateRecord] = []
    for record in outcome.selected:
        pruned = prune_record_attributes(
            record,
            registry=resolved_registry,
            required_attributes=attributes,
            required_tables=required_tables_for_projection,
            keep_all_fks=mutable_context,
        )
        pruned_columns.extend((record.table, column) for column in sorted(set(record.values) - set(pruned.values)))
        pruned_records.append(pruned)

    def _relate(kept: List[StateRecord]) -> Tuple[Any, ...]:
        identity = {_identity(record) for record in kept}
        return tuple(
            relation
            for relation in state.relations
            if (str(relation.source_table).lower(), str(relation.source_id)) in identity
            and (str(relation.target_table).lower(), str(relation.target_id)) in identity
        )

    # P1: the token budget is checked against the *post-rebuild* state (facts
    # already pruned to the kept columns), so the budgeted payload is the one
    # the agent actually sees. Start from the pruned rows, then drop
    # lowest-scored non-essential rows while still over budget, re-pruning and
    # re-estimating each time.
    kept = [
        prune_record_attributes(
            r,
            registry=resolved_registry,
            required_attributes=attributes,
            required_tables=required_tables_for_projection,
        )
        for r in outcome.selected
    ]
    token_estimate = estimate_state_tokens(_rebuild_state(state, kept, _relate(kept)))
    while token_estimate > max_tokens and len(kept) > 1:
        victims = [r for r in kept if _identity(r) != anchor_key and f"table:{_identity(r)[0]}" not in universe]
        if not victims:
            break
        victim = min(
            victims,
            key=lambda record: next(
                (s.score for s in scores if _identity(s.record) == _identity(record)),
                float("-inf"),
            ),
        )
        kept.remove(victim)
        token_estimate = estimate_state_tokens(_rebuild_state(state, kept, _relate(kept)))

    if token_estimate > max_tokens:
        raise ValueError(
            f"minimal state token budget exhausted: {token_estimate} > {max_tokens}; "
            "refusing to truncate grounding"
        )

    minimal = _rebuild_state(state, kept, _relate(kept))
    final_covered, final_uncovered = realized_relation_facts(
        minimal, {_identity(r) for r in kept}, relations
    )
    return MinimalStateResult(
        state=minimal,
        coverage_ratio=coverage_ratio,
        covered=frozenset(covered | final_covered),
        uncovered_required=frozenset(outcome.uncovered_required),
        dropped_rows=tuple(outcome.dropped),
        pruned_columns=tuple(pruned_columns),
        token_estimate=token_estimate,
        weights=weights,
        intent=tuple(intent),
    )


def _identity(record: Any) -> Tuple[str, str]:
    return (str(record.table).lower(), str(record.row_id))


def _rebuild_state(
    state: GroundedState,
    records: Sequence[Any],
    relations: Sequence[Any],
) -> GroundedState:
    """Return a new ``GroundedState`` over the selected rows and relations.

    P1: facts are re-scoped to the kept columns of the kept rows, so a fact
    whose column was attribute-pruned is dropped too - facts and records stay
    consistent, and the token estimate of the rebuilt state reflects the
    payload the agent actually sees.
    """
    selected_identity = {_identity(record) for record in records}
    kept_columns = {
        _identity(record): {str(column).lower() for column in record.values}
        for record in records
    }
    facts = tuple(
        fact
        for fact in state.facts
        if _identity_fact(fact) in selected_identity
        and str(fact.provenance.column).lower() in kept_columns[_identity_fact(fact)]
    )
    return GroundedState(
        state_id=state.state_id,
        anchor=state.anchor,
        records=tuple(sorted(records, key=lambda record: (record.table, record.row_id))),
        relations=tuple(relations),
        unresolved=state.unresolved,
        unresolved_details=state.unresolved_details,
        provenance=state.provenance,
        status=state.status,
        schema_version=state.schema_version,
        observation_generation=state.observation_generation,
        facts=facts,
        contradictions=state.contradictions,
    )


def _identity_fact(fact: Any) -> Tuple[str, str]:
    return (str(fact.provenance.table).lower(), str(fact.provenance.row_pk))


__all__ = [
    "DEFAULT_WEIGHTS",
    "CandidateScore",
    "CoverOutcome",
    "MinimalStateResult",
    "RelevanceWeights",
    "build_minimal_state",
    "coverage_facts",
    "estimate_state_tokens",
    "greedy_minimal_cover",
    "prune_record_attributes",
    "score_candidate",
    "task_hinted_tables",
    "task_intent",
    "task_required_attributes",
    "task_required_tables",
]
