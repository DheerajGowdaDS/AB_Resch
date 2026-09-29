"""Condition B state adapter: the only place ``csm_env`` meets the agent.

This module owns the fairness boundary (SEC-001). It can read the database, the
task's ``user_prompt``, and the task's tool list. It cannot read ``verifiers``,
``validation_config``, ``expected_value``, or verifier names/SQL, and it refuses
to accept a raw task-config mapping at all.

The pipeline is exactly the frozen one:

``SchemaGraph -> GGQR -> bounded retrieval -> GroundedState -> representation``

There is no predicted future, no action simulation, and no planning search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from csm_env.api import CSMEnvironmentAPI
from csm_env.query.adapters import render_like_pattern
from csm_env.query.models import QueryBudget, RouteStrategy
from csm_env.query.validation import SqlValidationError, validate_read_only_select
from csm_env.schema_spec.registry import SchemaRegistry
from csm_env.state.models import GroundedState, StateFact, StateRelation, StateStatus
from csm_env.transport.reader import EnterpriseOpsSQLRunnerReader, SQLReader, TransportError

from .route_policy import DEFAULT_HOP_LADDER, build_ladder

#: Deterministic, dependency-free token estimate. Recorded alongside every
#: measurement so a reader knows exactly how "tokens" was computed.
TOKEN_ESTIMATOR = "chars_div_4"

#: Keys that may never reach the state model. Matched case-insensitively.
FORBIDDEN_KEYS: Tuple[str, ...] = (
    "verifiers",
    "verifier",
    "validation_config",
    "expected_value",
    "comparison_type",
    "comparison_prompt",
    "sql_query",
    "query",
    "description",
    "minimum_comparison_value",
    "minimum_tool_calls",
)

_FORBIDDEN_SET = frozenset(key.lower() for key in FORBIDDEN_KEYS)

#: Anchor search order. Case-centric tables come first because a case anchor
#: gives GGQR the richest bounded neighborhood.
ANCHOR_TABLE_PRIORITY: Tuple[str, ...] = (
    "customer_case",
    "installed_product",
    "account",
    "contact",
    "entitlement",
    "contract",
    "product",
    "user",
    "user_group",
    "location",
    "knowledge",
    "interaction",
    "case_sla",
    "sla_definition",
    "notification",
    "user_group_member",
    "case_knowledge",
)

#: Bounds that keep pre-action state retrieval cheap and predictable.
MAX_ANCHOR_LOOKUPS = 48
MAX_CANDIDATES = 12
LIKE_FETCH_LIMIT = 5

#: Minimal-state budget (State Model 2.0, blueprint sections 2 and 12). The
#: row cap is only a guard: the selector terminates on requirement coverage,
#: so the cap can no longer silently drop a required table. The token ceiling
#: is enforced by pruning before assembly, never by truncating rendered text.
MAX_MINIMAL_ROWS = 12
MAX_MINIMAL_TOKENS = 6000

#: Maximum tables probed for a *single* candidate mention.
#:
#: Phase 1.2's budget discipline. Without a cap the resolver sweeps all 17 CSM
#: tables for candidate 1, then all 17 for candidate 2, and the budget is gone
#: before the constrained-semantic tier ever runs. That is exactly the failure
#: the live probe exposed: ``48 = 17 + 17 + 14`` lookups, every one of them an
#: *exact* match, on a task whose anchor plainly existed.
#:
#: The value is set so that every identity-capable CSM table fits inside one
#: candidate's fan-out. CSM currently has ten such tables, so a cap of twelve
#: guarantees budget is never the reason a resolvable anchor is missed, while
#: still bounding the work per candidate.
MAX_TABLES_PER_CANDIDATE = 12

#: Column-name suffixes that identify a pure foreign key. Such a column is a
#: legitimate anchor target for a *numeric* identifier but can never match a
#: human-readable name, so name tiers skip it and spend the lookup elsewhere.
_FOREIGN_KEY_SUFFIXES: Tuple[str, ...] = ("_id",)

#: Share of the total lookup budget reserved for each resolution tier.
#:
#: Without a reservation the first tier consumes everything. The live trace made
#: this concrete: for the create-new-entities task, all 48 lookups went to exact
#: equality, and the constrained-semantic tier - the *only* tier that could have
#: matched, because the referenced rows do not exist yet and the prompt only
#: appears inside ``knowledge.title = '... SUSE Linux Enterprise Server 15 ...'``
#: - never ran at all. The anchor was scored unresolvable while a valid one sat
#: in the database.
#:
#: The split favours the semantic tier because it is the only one that can match
#: a substring, which is the common case in CSM where product names carry a
#: version suffix (``Windows Server 2019 Standard``).
TIER_BUDGET_SHARES: Dict[str, float] = {
    "identifier": 0.25,
    "name_match": 0.30,
    "semantic_lookup": 0.45,
}


def estimate_tokens(text: str) -> int:
    """Deterministic character-based token estimate.

    Four characters per token is the standard coarse approximation. It is
    intentionally simple so the same text always yields the same number.
    """
    if not text:
        return 0
    return max(1, -(-len(text) // 4))


def task_conditioned_minimal_state(task: Any, state: Any) -> Any:
    """Return the smallest grounded state still covering the task-relevant tables.

    Backward-compatible wrapper around :func:`ablation.minimal_state.
    build_minimal_state`, the full State Model 2.0 selector: relevance-ranked
    candidates, greedy weighted set cover, attribute pruning, and a coverage
    gate. On any under-coverage or budget failure the broad state is returned
    unchanged rather than a silently weaker one.
    """
    if state is None or not getattr(state, "records", None):
        return state
    from .minimal_state import build_minimal_state

    try:
        return build_minimal_state(
            task,
            state,
            max_rows=MAX_MINIMAL_ROWS,
            max_tokens=MAX_MINIMAL_TOKENS,
            coverage_threshold=_tau_for_task(task),
        ).state
    except ValueError:
        # Fail closed to the broad state: the coverage/token gate refused the
        # minimal selection, so delivering the broad state is the honest result.
        return state


def _tau_for_task(task: Any) -> float:
    """Coverage threshold (tau) pre-registered per task type.

    See ``ablation/manifests/experiment_config.json``.  The mapping is
    pre-registered; it is never adapted from run results.
    """
    task_type = str(getattr(getattr(task, "signals", None), "ggqr_task_type", "") or "").strip()
    if not task_type:
        return 1.0
    try:
        import json
        from pathlib import Path

        config_path = Path(__file__).resolve().parent / "manifests" / "experiment_config.json"
        if not config_path.is_file():
            return 1.0
        with open(config_path, encoding="utf-8") as fh:
            config = json.load(fh)
        tau_map = config.get("tau", {})
        if task_type in tau_map and isinstance(tau_map[task_type], (int, float)):
            return float(tau_map[task_type])
    except Exception:
        pass
    return float(config.get("default_tau", 1.0))


def assert_no_verifier_metadata(payload: Any) -> None:
    """Fail closed if verifier-only information is offered to the state model.

    Args:
        payload: Object to inspect. Mappings are scanned recursively; other
            objects are inspected through ``__dict__`` when available.

    Raises:
        TypeError: If ``payload`` looks like a raw task-config mapping. Passing
            the whole task file is the failure mode this guard exists to stop.
        ValueError: If a forbidden key or attribute is found.
    """
    if isinstance(payload, Mapping):
        if "gym_servers_config" in payload or ("verifiers" in payload and "user_prompt" in payload):
            raise TypeError(
                "StateModelAdapter must not receive a raw task-config mapping; pass a "
                "TaskRecord built from prompt and tool data only."
            )
        for key, value in payload.items():
            if str(key).lower() in _FORBIDDEN_SET:
                raise ValueError(
                    f"Refusing verifier-only key {key!r} at the Condition B boundary"
                )
            assert_no_verifier_metadata(value)
        return

    # Inspect both instance and class attributes: a verifier field attached at
    # class level is exactly as leaky as one attached per instance.
    namespaces: List[Dict[str, Any]] = []
    if hasattr(payload, "__dict__"):
        namespaces.append(vars(payload))
    if isinstance(payload, type) or hasattr(payload, "__class__"):
        namespaces.append(vars(type(payload)))
    for namespace in namespaces:
        for name, value in namespace.items():
            if name.startswith("__"):
                continue
            if name.lower() in _FORBIDDEN_SET:
                raise ValueError(
                    f"Refusing verifier-only attribute {name!r} at the Condition B boundary"
                )
            if isinstance(value, (Mapping, list, tuple)):
                assert_no_verifier_metadata(value)


class StateContextPolicy:
    """When the state context is materialized.

    Only :attr:`PRE_ACTION_ONCE` is selected by Experiment 1. Per-step refresh
    exists but is deliberately not chosen: changing refresh frequency would
    confound "does state modeling help" with "how often does it help".
    """

    PRE_ACTION_ONCE = "pre_action_once"
    PRE_ACTION_PER_STEP = "pre_action_per_step"
    ALL = (PRE_ACTION_ONCE, PRE_ACTION_PER_STEP)

    @classmethod
    def validate(cls, policy: str) -> str:
        if policy not in cls.ALL:
            raise ValueError(
                f"Unknown state context policy {policy!r}; expected {list(cls.ALL)}"
            )
        return policy


@dataclass(frozen=True)
class AnchorResolution:
    """Outcome of resolving a prompt mention to a real database row.

    Attributes:
        task_id: Task the anchor belongs to.
        table: Resolved table name, or ``None`` when unresolved.
        row_id: Resolved primary key, or ``None`` when unresolved.
        status: ``RESOLVED`` or ``UNRESOLVED``.
        strategy: ``manifest_anchor``, ``identifier`` or ``name_match``.
        evidence: The prompt literal that produced the match.
        lookups: Number of read-only lookups performed.
    """

    task_id: str
    table: Optional[str]
    row_id: Optional[str]
    status: str
    strategy: str
    evidence: str
    lookups: int

    @property
    def resolved(self) -> bool:
        return self.status == "RESOLVED" and self.table is not None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "table": self.table,
            "row_id": self.row_id,
            "status": self.status,
            "strategy": self.strategy,
            "evidence": self.evidence,
            "lookups": self.lookups,
        }


@dataclass(frozen=True)
class StateContext:
    """One materialization of the grounded state, as shown to the agent."""

    task_id: str
    task_type: str
    reference_type: Optional[str]
    reference_id: Optional[str]
    route_status: str
    rendered_text: str
    retrieved_tables: Tuple[str, ...]
    retrieved_rows: Tuple[Tuple[str, str], ...]
    retrieved_record_count: int
    retrieved_fact_count: int
    retrieved_token_count: int
    retrieval_latency_ms: float
    unresolved: Tuple[str, ...]
    contradictions: Tuple[str, ...]
    schema_version: str
    observed_at: str
    token_estimator: str
    anchor: AnchorResolution
    rendered_graph: str = ""
    rendered_json: str = ""
    error: Optional[str] = None
    #: Phase 1.3: whether every required table was *reachable* through a valid
    #: schema path and every planned statement executed. This is independent of
    #: row coverage: an empty required table lowers ``route_status`` but must not
    #: invalidate the route itself.
    structurally_valid: bool = False
    #: Required tables that were reached but returned no rows, for reporting.
    uncovered_required: Tuple[str, ...] = ()
    #: Hop budget the accepted route used, after ladder escalation.
    resolved_max_hops: Optional[int] = None
    #: P5.1/A2: True when the minimal selector's coverage/token gate failed and
    #: the broad state was delivered instead. Such a run must be excluded from
    #: the B1-vs-B2 contrast (it delivered the broad payload, not the minimal
    #: one), so it is never reported as a successful minimal intervention.
    selection_gate_failed: bool = False
    #

    @property
    def is_empty(self) -> bool:
        return not self.rendered_text.strip()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "reference_type": self.reference_type,
            "reference_id": self.reference_id,
            "route_status": self.route_status,
            "retrieved_tables": list(self.retrieved_tables),
            "retrieved_rows": [list(row) for row in self.retrieved_rows],
            "retrieved_record_count": self.retrieved_record_count,
            "retrieved_fact_count": self.retrieved_fact_count,
            "retrieved_token_count": self.retrieved_token_count,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "unresolved": list(self.unresolved),
            "contradictions": list(self.contradictions),
            "schema_version": self.schema_version,
            "observed_at": self.observed_at,
            "token_estimator": self.token_estimator,
            "anchor": self.anchor.as_dict(),
            "rendered_chars": len(self.rendered_text),
        }


def _quote_literal(value: str) -> str:
    """Render a value as a SQL string literal.

    Doubling the apostrophe is what keeps an embedded quote from terminating the
    literal early. See :func:`csm_env.query.adapters.escape_like_pattern` for the
    live failure this prevents.
    """
    return "'" + str(value).replace("'", "''") + "'"


#: Minimum part length for a decomposed compound. Short fragments such as
#: ``S`` in ``Larson S`` are noise and are dropped rather than searched.
_COMPOUND_MIN_PART = 3

#: Splits a compound proper noun into independently searchable entities.
#:
#: The possessive ``'s`` is the reliable boundary in business prose: it joins
#: the owning entity to the owned one. ``Wayne Enterprises' Windows Server``
#: yields ``Wayne Enterprises`` and ``Windows Server``, both of which appear
#: verbatim in the database, whereas the whole phrase appears nowhere.
#:
#: A plain whitespace split is deliberately *not* performed. Live data showed
#: ``product.name = 'Windows Server 2022 Datacenter'``, so ``Windows Server`` is
#: itself a useful substring; shattering it into ``Windows`` and ``Server``
#: would replace a good candidate with two useless ones. Multi-word names are
#: kept whole and matched by the semantic tier instead.
_COMPOUND_POSSESSIVE: Tuple[str, ...] = ("'s ", "\u2019s ", "' ", "\u2019 ")


def _split_compound(value: str) -> List[str]:
    """Return the entity names embedded in a possessive compound proper noun.

    Args:
        value: A proper noun such as ``"Wayne Enterprises' Windows Server"``.

    Returns:
        Sub-strings worth searching, longest first. Empty unless the phrase is
        genuinely two entities joined by a possessive, in which case the caller
        keeps the original phrase and simply adds these parts.
    """
    text = str(value).strip()
    if not text:
        return []
    for separator in _COMPOUND_POSSESSIVE:
        if separator in text:
            head, _, tail = text.partition(separator)
            head = head.strip()
            tail = tail.strip()
            if len(head) >= _COMPOUND_MIN_PART and len(tail) >= _COMPOUND_MIN_PART:
                return sorted({head, tail}, key=lambda part: (-len(part), part))
    return []


def _like_pattern(value: str) -> str:
    """Render a bounded ``LIKE`` pattern with wildcards *and* quotes escaped.

    The apostrophe case is the important one: a prompt proper noun such as
    ``Wayne Enterprises' Windows Server`` previously produced the unbalanced
    literal ``'%Wayne Enterprises' Windows Server%'``, which the CSM server
    rejected with ``400 Bad Request`` and which cost a Condition-B run its
    delivery. Escaping is delegated to the routing layer so the harness and
    ``csm_env`` cannot drift apart on this rule.
    """
    return render_like_pattern(value)



#: Column-name fragments that make a column a good anchor target, best first.
_COLUMN_PRIORITY: Tuple[str, ...] = (
    "name",
    "serial",
    "model",
    "number",
    "email",
    "title",
    "subject",
    "short_description",
    "description",
    "first_name",
    "last_name",
    "city",
    "address",
    "value",
    "code",
)

_TEXT_TYPES = frozenset({"string", "text", "unknown"})

#: Column-name fragments that carry *identity* - a value a human would recognise
#: as naming an entity.
#:
#: Having any text column is not enough to make a table a plausible name anchor.
#: ``customer_case`` declares ``state``, ``priority`` and ``channel``; none of
#: those can ever equal ``Wayne Enterprises``, yet the table still consumed a
#: share of the lookup budget on every name probe. Requiring an identity column
#: removes those tables from name search entirely, which is what lets ``account``
#: (whose ``name`` *is* ``Wayne Enterprises``) be reached within the cap.
_IDENTITY_COLUMN_FRAGMENTS: Tuple[str, ...] = (
    "name",
    "serial",
    "model",
    "title",
    "email",
    "subject",
    "first_name",
    "last_name",
    # ``customer_case.number`` holds values like ``CS-0000888`` and
    # ``knowledge.kb_number`` holds the article id. Both anchor real rows, so
    # leaving "number" out made an entire class of resolvable anchors - case
    # numbers included - unreachable for the name tiers.
    "number",
)


def _column_tokens(column_name: str) -> frozenset:
    """Split a column name into lowercase underscore-delimited tokens.

    Tokenising is what keeps ``entitlement_id`` from matching the identity
    fragment ``title``, which a plain substring test would do.
    """
    return frozenset(part for part in str(column_name).lower().split("_") if part)


def _has_identity_column(spec: Any) -> bool:
    """Whether a table declares at least one column that names an entity.

    Matching is on underscore-delimited *tokens*, not substrings. A substring
    test is subtly wrong: ``entitlement_id`` contains ``title`` inside
    "en**title**ment", which made a table with no identity column at all look
    like a valid name anchor and consume part of the lookup budget.
    """
    return any(
        _column_tokens(column.name) & frozenset(_IDENTITY_COLUMN_FRAGMENTS)
        for column in spec.columns
    )


def _search_columns(spec: Any, *, for_name: bool = False) -> Tuple[str, ...]:
    """Return a table's anchor-searchable columns, most specific first.

    Only text-ish columns qualify and the primary key is excluded; identifiers
    come from the schema registry so nothing unvalidated is ever interpolated.

    Args:
        spec: The table's ``TableSpec``.
        for_name: When true, drop pure foreign-key columns (``*_id``). A name
            tier can never match ``portal_user_id`` against ``Wayne Enterprises``,
            so searching it only consumes budget. The live probe showed
            ``contact`` offering *only* ``account_id``/``portal_user_id``, which
            made every name probe against it a guaranteed miss. Identifier tiers
            keep foreign keys, because there a numeric value genuinely can match.
    """

    def rank(column: Any) -> Tuple[int, str]:
        name = column.name.lower()
        for index, fragment in enumerate(_COLUMN_PRIORITY):
            if fragment in name:
                return (index, name)
        return (len(_COLUMN_PRIORITY), name)

    candidates = [
        column
        for column in spec.columns
        if column.name != spec.primary_key and column.type in _TEXT_TYPES
    ]
    if for_name:
        candidates = [
            column
            for column in candidates
            if not column.name.lower().endswith(_FOREIGN_KEY_SUFFIXES)
        ]
    return tuple(column.name for column in sorted(candidates, key=rank))


class StateModelAdapter:
    """Materializes the frozen ``csm_env`` state for one task.

    The adapter is constructed *after* the benchmark has seeded the task
    database, so the underlying ``MCPClient``/SQL runner already points at the
    correct ``database_id``.

    Args:
        api: A ``CSMEnvironmentAPI`` built with GGQR enabled.
        task: A ``TaskRecord``. Guarded against verifier-only fields.
        registry: Schema registry; defaults to the static CSM manifest.
        reader: Read-only transport for anchor lookups. Defaults to a reader
            over the API's own SQL runner.
        max_lookups: Upper bound on anchor lookups per resolution.
    """

    def __init__(
        self,
        api: CSMEnvironmentAPI,
        task: Any,
        *,
        registry: Optional[SchemaRegistry] = None,
        reader: Optional[SQLReader] = None,
        max_lookups: int = MAX_ANCHOR_LOOKUPS,
        budget: Optional[QueryBudget] = None,
    ) -> None:
        assert_no_verifier_metadata(task)
        if max_lookups < 1:
            raise ValueError("max_lookups must be >= 1")
        self._api = api
        self._task = task
        self._registry = registry or SchemaRegistry.from_static()
        self._reader: SQLReader = reader or EnterpriseOpsSQLRunnerReader(api.environment.sql)
        self._max_lookups = max_lookups
        self._lookups = 0
        #: When set, escalation is disabled and this single budget is used, so an
        #: explicit caller budget still fully determines the route.
        self._budget = budget
        #: Lookup count at which each tier began, used to enforce the per-tier
        #: reservation in :meth:`_tier_allowance`.
        self._tier_start: Dict[str, int] = {}
        #: Most recent GroundedState, retained for evaluator-side scoring.
        self._last_state: Any = None
        #: Evidence from the last minimal-state selection (coverage, dropped
        #: rows, pruned columns), or ``None`` for broad-state runs.
        self._last_minimal_evidence: Optional[Dict[str, Any]] = None
        #: P5.1/A2: whether the last minimal gate failed and the broad state
        #: was delivered instead.
        self._last_selection_gate_failed: bool = False
        #: Hop count of the ladder rung that produced the accepted state.
        self._accepted_hops: Optional[int] = None

    @property
    def api(self) -> CSMEnvironmentAPI:
        return self._api

    @property
    def task(self) -> Any:
        return self._task

    def _candidate_mentions(self) -> List[Tuple[Optional[str], str]]:
        """Ordered (preferred_table, literal) candidates drawn from the prompt.

        Explicit identifiers win over free-form names, and every candidate comes
        from the task prompt - never from a verifier.
        """
        signals = self._task.signals
        candidates: List[Tuple[Optional[str], str]] = []
        seen: set[str] = set()

        def add(table: Optional[str], value: str) -> None:
            value = (value or "").strip()
            if not value or len(value) > 80:
                return
            key = f"{table}|{value.lower()}"
            if key in seen:
                return
            seen.add(key)
            candidates.append((table, value))

        for table, value in self._task.reference_names:
            add(table, value)
        # Human/entity names are more useful anchors than incidental numeric
        # literals (counts, versions, dates) for tasks that create new records.
        for value in signals.proper_nouns:
            add(None, value)
            # A possessive joins two independent entities: in
            # "Wayne Enterprises' Windows Server" the account is
            # "Wayne Enterprises" and the product is "Windows Server". Neither
            # the whole phrase nor a substring of it matches a stored value, so
            # the compound is decomposed and both halves become candidates. This
            # is what lets the live probe find `account.name = 'Wayne
            # Enterprises'` at all.
            for part in _split_compound(value):
                add(None, part)
        for value in signals.literals:
            add(None, value)
        return candidates[:MAX_CANDIDATES]

    def _attribute_constraints(
        self, table: str, candidate_value: str
    ) -> Tuple[Tuple[str, str], ...]:
        """Build AND-constraint predicates from the prompt for one candidate.

        §4 constrained schema-aware lookup: when the prompt qualifies an entity
        (``Acme's premium customer``), the name portion drives the LIKE/equality
        predicate and the attribute portion is added as a non-negotiable AND
        constraint.  The constraints are derived from the prompt only (never
        verifiers), and every column is validated against the schema registry
        before it is interpolated, so a malformed constraint degrades to a miss
        rather than a 400.

        The current extractor is heuristic: it looks for prompt literals that
        sit alongside attribute-key matches.  Full attribute-value extraction
        requires LLM-grade understanding of which literal is the name and which
        is the constraint; the blueprint's example (``premium customer``) is
        exactly that case.  This implementation provides the plumbing and a
        best-effort heuristic, so the capability is present and testable even
        when the current 11-task eval set does not exercise it.
        """
        from .minimal_state import task_required_attributes

        if not table:
            return ()
        required_attrs = task_required_attributes(self._task)
        if not required_attrs:
            return ()

        spec = self._registry.require_table(table)
        table_columns = {str(col).lower(): str(col) for col in spec.column_names()}
        prompt_text = str(getattr(getattr(self._task, "signals", None), "user_prompt", "") or "")
        candidate_lower = candidate_value.lower()

        identity_fragments = {"name", "serial", "model", "number", "email"}
        stop_words = {"case", "cases", "customer", "account", "user", "product", "contract",
                      "entitlement", "group", "team", "queue", "assignment", "find", "get",
                      "show", "list", "search", "retrieve", "view", "check", "lookup"}
        constraints: List[Tuple[str, str]] = []
        for attr_key in required_attrs:
            attr_lower = str(attr_key).lower()
            if attr_lower not in table_columns:
                continue
            column = table_columns[attr_lower]
            # Heuristic: if the candidate value itself carries an attribute
            # qualifier (e.g. ``Acme premium``), the non-identity tail fragment
            # is treated as a potential constraint value.  This is deliberately
            # conservative: it only fires when the candidate is a multi-word
            # compound, which is exactly the ``X's Y`` / ``X Y`` shape the
            # blueprint's constrained-lookup example uses.
            parts = [p.strip() for p in candidate_value.split() if p.strip()]
            if len(parts) >= 2:
                for part in parts[1:]:
                    if len(part) < 2:
                        continue
                    part_lower = part.lower()
                    if any(frag in part_lower for frag in identity_fragments):
                        continue
                    if part_lower in stop_words:
                        continue
                    if part_lower in candidate_lower and part_lower != candidate_lower:
                        constraints.append((column, part))
                        break
        return tuple(constraints)

    def _candidate_tables(
        self, preferred: Optional[str], *, for_name: bool = False
    ) -> List[str]:
        """Tables to search for one candidate, most promising first.

        The result is capped at :data:`MAX_TABLES_PER_CANDIDATE`, and the cap
        alone is not enough. Ordering by *entity mention* lets a task's own
        ``reference_entities`` monopolise the whole budget with tables that
        cannot possibly match a name: the live probe found the six entity tables
        for this task are ``customer_case``, ``contact``, ``entitlement``,
        ``contract``, ``product``, ``case_sla`` - none of which except ``product``
        has a name column, and all six consumed the budget before ``account``
        (whose ``name`` is ``Wayne Enterprises``) was ever probed.

        So for a name probe the order is *searchability first*, and the task's own
        ``reference_entities`` is deliberately **not** used. Entity mentions
        describe what the task is about, not where its anchor lives: the live
        probe showed a task mentioning twelve entities dropping ``location`` -
        which held ``Larson London Office``, the real anchor - off the end of the
        candidate's table list. Anchor identity tables are a small, fixed set, so
        ordering them by the curated priority list is both deterministic and
        complete. Entity ordering is kept for *identifier* probes, where a
        referenced table is a genuinely useful hint.

        Args:
            preferred: A table named by the candidate itself, if any.
            for_name: True for the name/semantic tiers, which cannot match a
                foreign key.
        """
        def searchable(table: str) -> bool:
            if self._registry.table(table) is None:
                return False
            spec = self._registry.require_table(table)
            if for_name:
                # A name probe can only ever hit an identity column, so a table
                # without one is skipped entirely rather than probed and missed.
                return _has_identity_column(spec) and bool(
                    _search_columns(spec, for_name=True)
                )
            return bool(_search_columns(spec))

        order: List[str] = []
        if preferred and searchable(preferred):
            order.append(preferred)
        if for_name:
            for table in ANCHOR_TABLE_PRIORITY:
                if table not in order and searchable(table):
                    order.append(table)
        else:
            entities = [t for t in self._task.reference_entities if searchable(t)]
            for table in ANCHOR_TABLE_PRIORITY:
                if table in entities and table not in order:
                    order.append(table)
            for table in ANCHOR_TABLE_PRIORITY:
                if table not in order and searchable(table):
                    order.append(table)
        return order[:MAX_TABLES_PER_CANDIDATE]

    async def _lookup(
        self,
        table: str,
        value: str,
        *,
        exact: bool,
        constraints: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> Optional[str]:
        """Search all safe anchor columns in one bounded table query.

        The previous implementation consumed one lookup for every column, so a
        normal CSM task could exhaust the 24-query budget before reaching the
        table containing the actual entity (the live pilot showed exactly this
        failure). One query per candidate table keeps the same fail-closed budget
        while making the search materially less brittle.

        ``constraints`` are additional AND predicates that narrow the lookup
        without consuming extra budget. They are used by the constrained
        schema-aware lookup path (§4): when the prompt qualifies an entity
        (``Acme's premium customer``), the name portion drives the LIKE/equality
        predicate and the attribute portion is added as a non-negotiable AND
        constraint.
        """
        if self._lookups >= self._max_lookups:
            return None
        spec = self._registry.require_table(table)
        # A fuzzy (semantic) probe is a name probe, so foreign-key columns are
        # dropped; an exact probe may legitimately match an id column.
        columns = _search_columns(spec, for_name=not exact)
        predicates: List[str] = []

        if exact and str(value).isdigit():
            # Numeric identifiers commonly map directly to the table primary key.
            predicates.append(f"{spec.primary_key} = {_quote_literal(value)}")

        for column in columns:
            predicates.append(
                f"{column} = {_quote_literal(value)}"
                if exact
                else f"{column} LIKE {_like_pattern(value)}"
            )

        # Full-person-name anchors are common in CSM tasks. Keep this strictly
        # schema-derived: only emit the composite predicate when both columns
        # actually exist.
        column_names = set(spec.column_names())
        if {"first_name", "last_name"}.issubset(column_names):
            parts = [part for part in re.split(r"\s+", str(value).strip()) if part]
            if len(parts) >= 2:
                first = _like_pattern(parts[0])
                last = _like_pattern(" ".join(parts[1:]))
                if exact:
                    predicates.append(
                        f"first_name = {_quote_literal(parts[0])} AND "
                        f"last_name = {_quote_literal(' '.join(parts[1:]))}"
                    )
                else:
                    predicates.append(
                        f"first_name LIKE {first} AND last_name LIKE {last}"
                    )

        # §4 constrained schema-aware lookup: additional AND predicates that
        # qualify the entity by task-relevant attributes. These are derived from
        # the prompt (not verifiers) and validated against the schema registry,
        # so they can never introduce a column the table does not declare.
        if constraints:
            valid_columns = set(spec.column_names())
            for constraint_column, constraint_value in constraints:
                if str(constraint_column).lower() in {
                    str(col).lower() for col in valid_columns
                }:
                    predicates.append(
                        f"{constraint_column} = {_quote_literal(constraint_value)}"
                    )

        if not predicates:
            return None

        self._lookups += 1
        base = " OR ".join(f"({predicate})" for predicate in predicates)
        if constraints:
            constraint_predicates = [
                f"{column} = {_quote_literal(value)}" for column, value in constraints
            ]
            constraints_clause = " AND ".join(f"({predicate})" for predicate in constraint_predicates)
            where = f"({base}) AND {constraints_clause}"
        else:
            where = base
        query = (
            f"SELECT {spec.primary_key} FROM {spec.table} WHERE {where} "
            f"LIMIT {LIKE_FETCH_LIMIT};"
        )
        # Phase 1.4: validate before the network call. Identifiers come from the
        # registry and values from render_literal/escape_like_pattern, so a
        # rejection here means our own construction is wrong; failing here
        # converts a would-be 400 into a recorded, non-fatal anchor miss.
        try:
            query = validate_read_only_select(query)
        except SqlValidationError:
            return None
        try:
            rows = await self._reader.fetch_rows(query)
        except TransportError:
            return None
        for row in rows:
            if isinstance(row, dict) and row.get(spec.primary_key) is not None:
                return str(row[spec.primary_key])
        return None

    async def resolve_anchor(self) -> AnchorResolution:
        """Resolve the task prompt to a real anchor row using read-only queries.

        Phase 1.2 fixes the *order* and the *budget*, not just the escaping. The
        blueprint's ladder is implemented literally:

        1. ``manifest_anchor``  - an explicit id/reference already pinned in the
           manifest, trusted without any query;
        2. ``identifier``       - a pure number, tried against the primary key
           and then against identity columns;
        3. ``name_match``       - an exact name/serial/email equality match;
        4. ``semantic_lookup``  - a constrained substring search, run only for
           candidates that survive the cheaper tiers.

        Crucially the budget is spent *by tier*, not across the full table
        cross-product. The previous ordering interleaved exact and fuzzy probes
        over every table, so a generic mention such as ``Product`` could exhaust
        the whole budget before the search ever reached the table that held the
        real entity. Tiers now escalate, each with its own share of the budget,
        and fuzzy matching is the last resort rather than the first.
        """
        task_id = self._task.task_id
        self._tier_start.clear()

        # --- Tier 1: an explicit, already-pinned anchor -----------------------
        for table, row_id in self._task.reference_rows:
            if self._registry.table(table) is None:
                continue
            return AnchorResolution(
                task_id=task_id,
                table=table,
                row_id=str(row_id),
                status="RESOLVED",
                strategy="manifest_anchor",
                evidence=f"{table}:{row_id}",
                lookups=0,
            )

        # Each tier records where it began, so its allowance is measured against
        # the lookups that tier itself spends.
        self._tier_start["identifier"] = self._lookups
        # --- Tier 2: explicit numeric identifiers ------------------------------
        numeric = [m for m in self._candidate_mentions() if m[1].isdigit()]
        for preferred, value in numeric:
            resolution = await self._probe(
                preferred,
                value,
                exact=True,
                strategy="identifier",
                task_id=task_id,
                allowance=self._tier_allowance("identifier"),
            )
            if resolution is not None:
                return resolution

        # --- Tier 3: exact name/serial/email equality -------------------------
        self._tier_start["name_match"] = self._lookups
        textual = [m for m in self._candidate_mentions() if not m[1].isdigit()]
        for preferred, value in textual:
            constraints = self._attribute_constraints(preferred or "", value)
            resolution = await self._probe(
                preferred,
                value,
                exact=True,
                strategy="name_match",
                task_id=task_id,
                allowance=self._tier_allowance("name_match"),
                constraints=constraints if constraints else None,
            )
            if resolution is not None:
                return resolution

        # --- Tier 4: constrained semantic (substring) lookup -------------------
        # This tier is guaranteed a reserved share of the budget, so a task whose
        # entities must still be created can always fall back to a substring
        # match. Without the reservation, exact probes consumed the whole budget
        # and this tier never ran.
        #
        # §4 constrained schema-aware lookup: when the prompt qualifies an entity
        # (``Acme's premium customer``), the name portion drives the LIKE
        # predicate and the attribute portion is added as an AND constraint.
        # This costs no extra lookups: the constraint is attached to the same
        # query that would have been issued without it.
        self._tier_start["semantic_lookup"] = self._lookups
        for preferred, value in textual:
            constraints = self._attribute_constraints(preferred or "", value)
            resolution = await self._probe(
                preferred,
                value,
                exact=False,
                strategy="semantic_lookup",
                task_id=task_id,
                allowance=self._tier_allowance("semantic_lookup"),
                constraints=constraints if constraints else None,
            )
            if resolution is not None:
                return resolution

        return AnchorResolution(
            task_id=task_id,
            table=None,
            row_id=None,
            status="UNRESOLVED",
            strategy="no_match",
            evidence="",
            lookups=self._lookups,
        )

    def _tier_allowance(self, strategy: str) -> int:
        """Lookups one tier may spend, reserved from the total budget.

        The allowance is computed from the lookups *already consumed* plus the
        tier's reserved share, so tiers cannot starve one another regardless of
        how the earlier ones behaved.
        """
        share = TIER_BUDGET_SHARES.get(strategy, 0.0)
        reserved = max(1, int(self._max_lookups * share))
        return self._tier_start.get(strategy, 0) + reserved

    async def _probe(
        self,
        preferred: Optional[str],
        value: str,
        *,
        exact: bool,
        strategy: str,
        task_id: str,
        allowance: int,
        constraints: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> Optional[AnchorResolution]:
        """Search the candidate tables for one mention under one strategy.

        Returns ``None`` on a miss *or* when the lookup budget is exhausted, so
        the caller simply continues to the next candidate and, ultimately,
        reports ``no_match``.

        The probe kind is derived from the *value*, not from ``exact``: a pure
        digit is an identifier and may legitimately match a key column, whereas
        any textual value is a name. Treating an exact name probe as an
        identifier probe is what let the live trace spend all 48 lookups against
        ``customer_case``/``entitlement``/``contract`` - tables whose columns
        (``state``, ``priority``, ``contract_type``) can never equal
        ``Wayne Enterprises`` - while the table that does hold it was never
        queried.

        ``constraints`` are optional AND predicates (§4 constrained lookup)
        that qualify the entity without consuming extra budget.  A miss with
        constraints degrades to the same unconstrained lookup so a noisy
        heuristic can never silently turn a resolvable anchor into a miss.
        """
        textual = not str(value).isdigit()
        for table in self._candidate_tables(preferred, for_name=textual):
            if self._lookups >= self._max_lookups or self._lookups >= allowance:
                return None
            row_id = await self._lookup(
                table, value, exact=exact, constraints=constraints
            )
            if row_id is not None:
                return AnchorResolution(
                    task_id=task_id,
                    table=table,
                    row_id=row_id,
                    status="RESOLVED",
                    strategy=strategy,
                    evidence=f"{table}~{value}",
                    lookups=self._lookups,
                )
            # §4 safety net: if the constrained lookup missed, retry the same
            # table without constraints before abandoning it.  The heuristic
            # that builds constraints is best-effort and can attach a tail-word
            # qualifier that eliminates a legitimate row; this fallback prevents
            # that from becoming a silent delivery failure.
            if constraints:
                row_id = await self._lookup(table, value, exact=exact)
                if row_id is not None:
                    return AnchorResolution(
                        task_id=task_id,
                        table=table,
                        row_id=row_id,
                        status="RESOLVED",
                        strategy=strategy,
                        evidence=f"{table}~{value}",
                        lookups=self._lookups,
                    )
        return None

    def _empty_context(
        self,
        anchor: AnchorResolution,
        route_status: str,
        started: float,
        error: str,
    ) -> StateContext:
        """Build a zero-retrieval context for a failed or unresolved route."""
        import time

        return StateContext(
            task_id=self._task.task_id,
            task_type=self._task.ggqr_task_type,
            reference_type=anchor.table,
            reference_id=anchor.row_id,
            route_status=route_status,
            rendered_text="",
            retrieved_tables=(),
            retrieved_rows=(),
            retrieved_record_count=0,
            retrieved_fact_count=0,
            retrieved_token_count=0,
            retrieval_latency_ms=(time.perf_counter() - started) * 1000.0,
            unresolved=(),
            contradictions=(),
            schema_version=self._registry.manifest.schema_version,
            observed_at="",
            token_estimator=TOKEN_ESTIMATOR,
            anchor=anchor,
            error=error,
        )

    @property
    def last_state(self) -> Any:
        """The most recently built :class:`GroundedState`, or ``None``.

        Retained so evaluation code can score *retrieval quality* against a
        initial-state relevance set (see :mod:`ablation.initial_relevance`) without re-routing the
        database. This is evaluator-only data: the agent still receives only
        ``rendered_text``.
        """
        return self._last_state

    async def context_for(self, *, minimal_state: bool) -> StateContext:
        """Build the grounded state and its rendered representation.

        The route is attempted against an escalating hop ladder (Phase 1.3)
        rather than a single fixed hop count, because the shortest valid schema
        path to a required table is task-dependent. ``location -> user ->
        user_group_member -> user_group`` needs three hops, while
        ``customer_case -> account`` needs one; one number cannot serve both.

        ``minimal_state=True`` applies the task-conditioned minimal selector
        (State Model 2.0): relevance-ranked candidates, greedy weighted set
        cover, attribute pruning, and the coverage/token gates from
        :mod:`ablation.minimal_state`. ``False`` keeps the full retrieved
        grounded state so broad-state and minimal-state runs can be compared
        under the B1/B2 evaluation split.

        Never raises for a retrieval problem: a failed route is reported as a
        ``FAILED``/``UNRESOLVED`` context so the run stays comparable and the
        failure is visible in the records instead of being silently scored.
        """
        import time

        started = time.perf_counter()
        anchor = await self.resolve_anchor()
        if not anchor.resolved:
            return self._empty_context(
                anchor,
                "UNRESOLVED",
                started,
                "anchor unresolved"
                if anchor.strategy == "no_match"
                else f"anchor resolution {anchor.strategy}",
            )

        unified = self._unified_requirements()
        required_tables = tuple(sorted(unified["tables"]))

        # State Model 2.0, sections 2/10/13. B2 (minimal) retrieves with the
        # FRONTIER strategy, so the router stops at the first prefix that meets
        # Coverage >= tau and `lambda_q * |queries|` is genuinely minimized.
        # B1 (broad) keeps the eager strategy, which pins its retrieval cost to
        # State Model 1.0 so the two arms differ only in *selection*.
        strategy = RouteStrategy.FRONTIER if minimal_state else RouteStrategy.BROAD
        state, attempt = await self._build_state_with_escalation(
            anchor, required_tables=required_tables, strategy=strategy
        )
        if state is None:
            return self._empty_context(
                anchor,
                "FAILED",
                started,
                attempt
                or "no hop budget produced a structurally valid route "
                f"for task type {self._task.ggqr_task_type!r}",
            )
        minimal_evidence: Optional[Dict[str, Any]] = None
        selection_gate_failed = False
        if minimal_state:
            from .minimal_state import build_minimal_state

            try:
                result = build_minimal_state(
                    self._task,
                    state,
                    registry=self._registry,
                    max_rows=MAX_MINIMAL_ROWS,
                    max_tokens=self._minimal_max_tokens(),
                    required_tables=unified["tables"],
                    required_relations=unified["relations"],
                    attribute_keys=unified["attribute_keys"],
                    coverage_threshold=_tau_for_task(self._task),
                )
                state = result.state
                minimal_evidence = result.as_dict()
            except ValueError:
                # P5.1/A2: the coverage/token gate refused the minimal
                # selection. We still deliver the broad state (a usable
                # intervention), but we now *mark* the run so the report can
                # exclude it from the B1-vs-B2 contrast instead of silently
                # counting a broad payload as a minimal one.
                minimal_evidence = {"gate_failed": True}
                selection_gate_failed = True
            if minimal_evidence is not None:
                # Audit trail: which retrieval strategy produced this payload.
                # B1 and B2 must be distinguishable from the record alone.
                minimal_evidence = {**minimal_evidence, "route_strategy": strategy}
        self._last_state = state
        self._last_minimal_evidence = minimal_evidence
        self._last_selection_gate_failed = selection_gate_failed

        # P1/N4: align the renderer's text ceiling with the selector's token
        # budget so the two ceilings stay coherent. The minimal path re-points
        # the ceiling at ``4 * max_tokens`` (one token ~ four chars, matching
        # ``estimate_tokens``); the broad path keeps the historical 4000-char
        # default so B1 is unchanged.
        max_chars = (
            4 * self._minimal_max_tokens() if minimal_state else 4000
        )
        rendered = self._api.represent_state(state, max_chars=max_chars)
        text = rendered.get("text", "")
        records = tuple(sorted((record.table, record.row_id) for record in state.records))
        tables = tuple(sorted({table for table, _ in records}))
        contradictions = tuple(
            f"{item.table}:{item.row_id}.{item.column}" for item in state.contradictions
        )
        return StateContext(
            task_id=self._task.task_id,
            task_type=self._task.ggqr_task_type,
            reference_type=anchor.table,
            reference_id=anchor.row_id,
            route_status=state.status,
            rendered_text=text,
            retrieved_tables=tables,
            retrieved_rows=records,
            retrieved_record_count=len(records),
            retrieved_fact_count=len(state.facts),
            retrieved_token_count=estimate_tokens(text),
            retrieval_latency_ms=(time.perf_counter() - started) * 1000.0,
            unresolved=tuple(state.unresolved),
            contradictions=contradictions,
            schema_version=state.schema_version,
            observed_at=state.provenance.observed_at,
            token_estimator=TOKEN_ESTIMATOR,
            anchor=anchor,
            rendered_graph=rendered.get("graph", ""),
            rendered_json=rendered.get("json", ""),
            structurally_valid=state.status != StateStatus.FAILED,
            uncovered_required=self._uncovered_required(state),
            resolved_max_hops=self._accepted_hops,
            selection_gate_failed=selection_gate_failed,
        )

    async def _build_state_with_escalation(
        self,
        anchor: AnchorResolution,
        *,
        required_tables: Sequence[str] = (),
        strategy: Optional[str] = None,
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Try each rung of the hop ladder until one yields usable state.

        Acceptance is *coverage-driven* (State Model 2.0, blueprint section
        10): a rung is accepted when it observes the task's required tables,
        not merely when it returns any records. Escalation continues while
        required tables remain unobserved, so the agent receives more state
        only when a requirement is still uncovered. Row presence alone still
        accepts a rung when the task names no required tables beyond the
        anchor, keeping the previous behaviour for anchor-only tasks.

        ``strategy`` selects the retrieval strategy of blueprint section 2.
        Condition B2 passes ``RouteStrategy.FRONTIER`` so that the router stops
        retrieving as soon as coverage is met and ``lambda_q * |queries|`` is
        minimized; Condition B1 passes ``None`` and keeps the eager strategy.
        A facade that does not accept the keyword falls back to the historical
        call, so test doubles and older facades keep working.

        Returns the built state plus a diagnostic message. ``(None, message)``
        means every rung failed; the message names each rung so the run record
        explains *why* the intervention could not be delivered.
        """
        ladder = self._hop_ladder()
        last_error: Optional[str] = None
        best_state: Optional[Any] = None
        self._accepted_hops = None
        required = {str(table).lower() for table in required_tables}

        for budget in ladder:
            # Progressive keyword fallbacks. A real facade accepts both the
            # budget and the strategy; an older facade or a test double may
            # accept only one of them. Degrading the *call* must never degrade
            # per-rung escalation, so the budget is the last thing dropped.
            call_attempts: List[Dict[str, Any]] = []
            if strategy:
                call_attempts.append({"budget": budget, "strategy": strategy})
            call_attempts.append({"budget": budget})
            call_attempts.append({})

            state: Optional[Any] = None
            attempted = False
            for kwargs in call_attempts:
                try:
                    state = await self._api.build_state(
                        self._task.ggqr_task_type,
                        anchor.row_id,
                        anchor.table,
                        **kwargs,
                    )
                    attempted = True
                    break
                except TypeError as exc:  # unsupported keyword: try a leaner call
                    last_error = f"facade rejected {sorted(kwargs)}: {exc}"
                    continue
                except Exception as exc:  # noqa: BLE001 - recorded, never dropped
                    last_error = f"max_hops={budget.max_hops}: {type(exc).__name__}: {exc}"
                    break
            if not attempted or state is None:
                continue

            if state is None:
                last_error = f"max_hops={budget.max_hops}: no state returned"
                continue
            # A state that failed outright is not worth re-routing at greater
            # depth; an unresolved one is, because more hops may fill the gap.
            if getattr(state, "status", None) == StateStatus.FAILED:
                last_error = f"max_hops={budget.max_hops}: state FAILED"
                continue
            if not state.records:
                last_error = f"max_hops={budget.max_hops}: no records retrieved"
                continue

            self._accepted_hops = budget.max_hops
            observed = {str(record.table).lower() for record in state.records}
            uncovered = required - observed
            if not uncovered:
                return state, None
            # The rung works but a required table is still unobserved: keep it
            # as the best-so-far and let the next rung try to cover it.
            last_error = (
                f"max_hops={budget.max_hops}: required tables not observed: "
                f"{sorted(uncovered)}"
            )
            best_state = state

        # Coverage-driven escalation failed to observe every required table.
        # Delivering the deepest observed state is still better than failing
        # the intervention: the missing tables are recorded as unresolved.
        if best_state is None:
            return None, last_error
        return best_state, None

    def _hop_ladder(self) -> Tuple[QueryBudget, ...]:
        """Build this adapter's hop ladder, honouring an explicit override."""
        if self._budget is not None:
            return (self._budget,)
        return build_ladder(DEFAULT_HOP_LADDER)

    def _prompt_table_names(self) -> Tuple[str, ...]:
        """Prompt/tool-derived required tables for escalation (SEC-001 safe).

        Uses only prompt/tool reference data; verifier information is
        structurally unreachable here.
        """
        tables: set[str] = set()
        if hasattr(self._task, "reference_entities"):
            tables.update(str(value).lower() for value in (self._task.reference_entities or ()))
        if hasattr(self._task, "reference_rows"):
            tables.update(str(table).lower() for table, _row in (self._task.reference_rows or ()))
        tables.discard("")
        return tuple(sorted(tables))

    def _unified_requirements(self) -> Dict[str, Any]:
        """ONE source of task requirements for the whole CSM path (P0/A3).

        Unions the registry task-type tables/attributes/relations with the
        prompt/tool-derived mentions, so the query planner, the coverage-driven
        escalation loop, and the minimal selector all agree on the required
        structure. SEC-001: only prompt/tool data and the frozen registry are
        consulted - verifier information is structurally unreachable.

        When the GGQR environment is not attached (test doubles), the prompt-
        tool-derived set is used alone, preserving prior behaviour.
        """
        from .minimal_state import task_required_attributes

        ggqr = getattr(self._api, "_ggqr", None)
        req_registry = getattr(ggqr, "requirements", None)
        req = None
        if req_registry is not None:
            try:
                req = req_registry.requirements_for(self._task.ggqr_task_type)
            except Exception:
                req = None
        prompt_tables = set(self._prompt_table_names())
        prompt_attrs = tuple(task_required_attributes(self._task))
        if req is not None:
            tables = {t.lower() for t in req.required_tables} | prompt_tables
            relations = tuple(req.required_relations)
            attribute_keys = tuple(dict.fromkeys((*req.required_attributes, *prompt_attrs)))
        else:
            tables = prompt_tables
            relations = ()
            attribute_keys = prompt_attrs
        return {
            "tables": frozenset(tables),
            "relations": relations,
            "attribute_keys": attribute_keys,
        }

    def _minimal_max_tokens(self) -> int:
        """P4: source the token ceiling from the caller's budget when given.

        An explicit ``self._budget`` determines the route, so it also determines
        the minimal-selector token ceiling; otherwise the frozen fallback
        ``MAX_MINIMAL_TOKENS`` applies. (The ``QueryBudget.max_tokens`` default of
        20000 remains inert for broad runs, but a caller-supplied budget now
        actually bites.)
        """
        if self._budget is not None:
            return int(self._budget.max_tokens)
        return MAX_MINIMAL_TOKENS

    def _uncovered_required(self, state: Any) -> Tuple[str, ...]:
        """Required tables this state did not observe, when knowable.

        The state object records ``unresolved_details`` as
        ``(table, column, reason)``; those tables are reported so the Phase 2
        diagnostic table can distinguish "reached" from "reached and empty".
        """
        details = getattr(state, "unresolved_details", ()) or ()
        return tuple(sorted({str(table) for table, _column, _reason in details}))
