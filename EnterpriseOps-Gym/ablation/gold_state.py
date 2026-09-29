"""Evaluator-only final-state verifier target, not an Experiment-1 initial-state gold set.

Blueprint Phase 2. Delivery alone is not sufficient evidence that the state
model is useful: an intervention can deliver a large, well-formed state that
simply does not contain the facts the task needs. Measuring *what* was retrieved
is the missing layer.

``GoldState`` is the answer set for one task. It is derived mechanically from
verifier SQL - the predicates that decide whether the task succeeded - so it is
reproducible rather than hand-authored, and it is **evaluation-only**: the agent
never receives it. This module must stay unreachable from the Condition-B import
chain, the same boundary that already governs
:mod:`ablation.relevance`.

A gold fact is a ``(table, column, value)`` triple, optionally qualified by a
row key. Precision and recall are then computed at fact granularity:

.. math::

    P = \\frac{|R \\cap G|}{|R|}, \\quad R = \\frac{|R \\cap G|}{|G|}

Fact granularity is a deliberate change from the earlier table/row heuristics,
which could not distinguish "retrieved the right account" from "retrieved the
right account *and* its support level".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from csm_env.schema_spec.registry import SchemaRegistry

from .task_registry import resolve_task_config_path

PathLike = Union[str, Path]

#: ``FROM <table>`` / ``JOIN <table>`` references a verifier depends on.
_TABLE_RE = re.compile(
    r"\b(?:FROM|JOIN|UPDATE|INTO)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE
)

#: ``<table>.<column> = <literal>`` and ``<column> = <literal>``.
_PREDICATE_RE = re.compile(
    r"(?:([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(=|<>|!=|<|>)\s*('(?:[^']|'')*'|-?\d+(?:\.\d+)?)"
)

#: Aggregate and subquery columns that are task scaffolding, not task facts.
_NON_FACT_COLUMNS: Tuple[str, ...] = ("count", "row_count", "min", "max", "avg", "sum")


@dataclass(frozen=True)
class GoldFact:
    """One fact a correct state must contain.

    Attributes:
        table: Table the fact lives in.
        column: Column name.
        value: Normalised literal value the verifier constrains.
        row_pk: Primary-key value when the verifier pins one, else ``None``.
    """

    table: str
    column: str
    value: str
    row_pk: Optional[str] = None

    def key(self) -> Tuple[str, str, str, Optional[str]]:
        """Identity used for set intersection."""
        return (self.table, self.column, self.value, self.row_pk)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "column": self.column,
            "value": self.value,
            "row_pk": self.row_pk,
        }


@dataclass(frozen=True)
class GoldState:
    """The facts a correct state for one task would contain.

    Attributes:
        task_id: Task this gold state belongs to.
        facts: Distinct gold facts, sorted for determinism.
        tables: Tables the verifiers depend on.
        verifier_count: Number of verifiers inspected.
    """

    task_id: str
    facts: Tuple[GoldFact, ...]
    tables: Tuple[str, ...]
    verifier_count: int

    def __len__(self) -> int:
        return len(self.facts)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "fact_count": len(self.facts),
            "facts": [fact.as_dict() for fact in self.facts],
            "tables": list(self.tables),
            "verifier_count": self.verifier_count,
        }


#: ``FROM/JOIN <table> [AS] <alias>`` binds an alias to a real table.
_ALIAS_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
    re.IGNORECASE,
)

#: Words that may follow a table name but are not its alias.
_NON_ALIAS_TOKENS: Tuple[str, ...] = (
    "where", "join", "inner", "left", "right", "outer", "full", "cross",
    "group", "order", "having", "limit", "offset", "union", "on", "and",
    "or", "not", "select", "from", "as", "using", "set", "values", "and",
)


def _alias_map(sql: str, known_tables: Iterable[str]) -> Dict[str, str]:
    """Map every SQL alias in ``sql`` to the real table it refers to.

    Verifier SQL routinely abbreviates (``JOIN account a ON ...``), so a
    predicate such as ``a.name = 'x'`` carries an alias, not a table name.
    Without this map every aliased predicate is discarded and the task silently
    gets an empty gold set - which is exactly what happened to two of the eleven
    tasks before this was added.
    """
    known = set(known_tables)
    mapping: Dict[str, str] = {}
    for table, alias in _ALIAS_RE.findall(sql):
        table = table.lower()
        if table not in known:
            continue
        mapping[table] = table
        if alias:
            candidate = alias.lower()
            if candidate not in _NON_ALIAS_TOKENS and candidate not in known:
                mapping[candidate] = table
    return mapping


def _normalise_literal(raw: str) -> str:
    """Strip quotes and lowercase, so ``'open'`` and ``OPEN`` compare equal."""
    text = raw.strip()
    if text.startswith("'") and text.endswith("'") and len(text) >= 2:
        text = text[1:-1].replace("''", "'")
    return text.strip().lower()


def build_gold_state(
    task_config_path: PathLike,
    *,
    registry: Optional[SchemaRegistry] = None,
    task_id: Optional[str] = None,
) -> GoldState:
    """Derive a task's gold facts from its verifier SQL.

    Only tables and columns declared by the schema registry contribute, so
    alias noise (``c``, ``ip``, ``cc``) is discarded rather than compared.

    Args:
        task_config_path: Path to an EnterpriseOps-Gym task JSON.
        registry: Optional schema registry; defaults to the static CSM one.
        task_id: Override for the gold state's identifier.

    Returns:
        The derived :class:`GoldState`.

    Raises:
        ValueError: If ``verifiers`` is missing or not a list.
    """
    path = resolve_task_config_path(task_config_path)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    verifiers = payload.get("verifiers")
    if not isinstance(verifiers, list):
        raise ValueError(f"{path}: 'verifiers' must be a list")

    resolved_registry = registry or SchemaRegistry.from_static()
    tables: List[str] = []
    facts: Dict[Tuple[str, str, str, Optional[str]], GoldFact] = {}
    primary_keys: Dict[str, str] = {
        spec.table: spec.primary_key for spec in resolved_registry.manifest.tables
    }

    for verifier in verifiers:
        if not isinstance(verifier, Mapping):
            continue
        config = verifier.get("validation_config") or {}
        if not isinstance(config, Mapping):
            continue
        statements = [
            config.get(key) for key in ("query", "sql_query")
        ]
        for sql in (s for s in statements if isinstance(s, str)):
            sql_tables = [
                name.lower()
                for name in _TABLE_RE.findall(sql)
                if resolved_registry.table(name.lower()) is not None
            ]
            for table in sql_tables:
                if table not in tables:
                    tables.append(table)
            aliases = _alias_map(sql, sql_tables)

            for alias, column, _operator, raw_value in _PREDICATE_RE.findall(sql):
                qualifier = (alias or "").lower()
                if qualifier:
                    table = aliases.get(qualifier, "")
                    if not table:
                        # An unknown qualifier is either a subquery alias or a
                        # column alias; skipping is safer than guessing, because a
                        # wrong attribution silently corrupts precision.
                        continue
                else:
                    # Unqualified column: attribute it only when unambiguous.
                    unique = sorted(set(sql_tables))
                    if len(unique) != 1:
                        continue
                    table = unique[0]
                column = column.lower()
                if column in _NON_FACT_COLUMNS:
                    continue
                if column not in resolved_registry.require_table(table).column_names():
                    continue
                pk = primary_keys.get(table)
                value = _normalise_literal(raw_value)
                # A predicate on the table's own primary key pins the row.
                row_pk = value if (pk and column == pk.lower()) else None
                fact = GoldFact(table=table, column=column, value=value, row_pk=row_pk)
                facts.setdefault(fact.key(), fact)

    ordered = tuple(
        sorted(
            facts.values(),
            key=lambda f: (f.table, f.column, f.value, f.row_pk or ""),
        )
    )
    return GoldState(
        task_id=task_id or Path(path).stem,
        facts=ordered,
        tables=tuple(sorted(tables)),
        verifier_count=len(verifiers),
    )


@dataclass(frozen=True)
class FactComparison:
    """Fact-level retrieval quality for one task.

    Attributes:
        task_id: Task compared.
        gold_fact_count: Facts a correct state must contain.
        retrieved_fact_count: Facts the delivered state actually contained.
        matched_fact_count: Facts present in both.
        precision: ``matched / retrieved``.
        recall: ``matched / gold``.
        unresolved_facts: Gold facts absent from the delivered state.
        extra_facts: Delivered facts the verifiers did not ask for.
    """

    task_id: str
    gold_fact_count: int
    retrieved_fact_count: int
    matched_fact_count: int
    precision: Optional[float]
    recall: Optional[float]
    unresolved_facts: Tuple[Dict[str, Any], ...]
    extra_facts: Tuple[Dict[str, Any], ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "gold_fact_count": self.gold_fact_count,
            "retrieved_fact_count": self.retrieved_fact_count,
            "matched_fact_count": self.matched_fact_count,
            "precision": self.precision,
            "recall": self.recall,
            "unresolved_facts": [dict(f) for f in self.unresolved_facts],
            "extra_fact_count": len(self.extra_facts),
        }


def retrieved_fact_keys(state: Any) -> set:
    """Extract comparable fact keys from a :class:`GroundedState`.

    A retrieved fact matches a gold fact when the table, column and normalised
    value agree. ``row_pk`` is deliberately *not* required to match: the gold
    set is a projection of what the task needs, and demanding row identity
    would penalise a state that correctly carries the value from a different but
    equally valid row.
    """
    keys = set()
    for fact in getattr(state, "facts", ()) or ():
        provenance = getattr(fact, "provenance", None)
        if provenance is None:
            continue
        keys.add(
            (
                str(provenance.table).lower(),
                str(provenance.column).lower(),
                _normalise_literal(str(getattr(fact, "value", ""))),
            )
        )
    return keys


def compare_to_gold(gold: GoldState, state: Any) -> FactComparison:
    """Compare a delivered state against the gold state at fact granularity.

    Args:
        gold: The task's gold facts.
        state: A ``GroundedState`` (or any object exposing ``facts`` with
            ``provenance``), or ``None`` when nothing was delivered.

    Returns:
        The comparison, including which gold facts were missed and how many
        delivered facts the verifiers did not ask for.
    """
    gold_triples = {(f.table, f.column, f.value) for f in gold.facts}
    retrieved = retrieved_fact_keys(state) if state is not None else set()

    matched = gold_triples & retrieved
    missing = sorted(gold_triples - retrieved)
    extra = sorted(retrieved - gold_triples)

    precision = (len(matched) / len(retrieved)) if retrieved else None
    recall = (len(matched) / len(gold_triples)) if gold_triples else None

    return FactComparison(
        task_id=gold.task_id,
        gold_fact_count=len(gold_triples),
        retrieved_fact_count=len(retrieved),
        matched_fact_count=len(matched),
        precision=precision,
        recall=recall,
        unresolved_facts=tuple(
            {"table": t, "column": c, "value": v} for t, c, v in missing
        ),
        extra_facts=tuple(
            {"table": t, "column": c, "value": v} for t, c, v in extra
        ),
    )


def build_gold_state_map(
    task_config_paths: Sequence[PathLike],
    *,
    registry: Optional[SchemaRegistry] = None,
) -> Dict[str, GoldState]:
    """Build gold states for a whole evaluation set, keyed by task id."""
    return {
        Path(path).stem: build_gold_state(path, registry=registry)
        for path in task_config_paths
    }


