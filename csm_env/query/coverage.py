"""Fact-level coverage for the task-conditioned state (blueprint section 12).

``Coverage(S, F_T)`` is defined over *facts*, not tables:

    ``anchor``              - the resolved anchor row is present
    ``table:<t>``           - a required table was observed with rows
    ``attr:<t>.<c>``        - a required column of a required table is non-NULL
    ``relation:<a>:<b>``    - a required relation was realized between observed
                              rows (a non-NULL FK edge exists)

The vocabulary is deliberately identical to the ablation layer's selector, so
the retrieval-side gate and the delivery-side gate can never disagree about
what "covered" means.

The universe is *observable* facts: a required table that was never retrieved
cannot be "uncovered", because absence of evidence is not loss of evidence.
That is the change that makes ``tau`` meaningful -- under the old
table-granularity gate an attribute could be pruned from a kept row without
any gate noticing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Sequence, Set, Tuple

from .models import TaskRequirements
from .requirements import resolve_attribute_columns

ANCHOR_FACT = "anchor"
TABLE_PREFIX = "table:"
ATTRIBUTE_PREFIX = "attr:"
RELATION_PREFIX = "relation:"


def table_fact(table: str) -> str:
    return f"{TABLE_PREFIX}{str(table).lower()}"


def attribute_fact(table: str, column: str) -> str:
    return f"{ATTRIBUTE_PREFIX}{str(table).lower()}.{str(column).lower()}"


def relation_fact(source_table: str, target_table: str) -> str:
    return f"{RELATION_PREFIX}{str(source_table).lower()}:{str(target_table).lower()}"


@dataclass(frozen=True)
class CoverageReport:
    """One measured ``Coverage(S, F_T)``."""

    covered: FrozenSet[str] = frozenset()
    uncovered: FrozenSet[str] = frozenset()
    universe: FrozenSet[str] = frozenset()

    @property
    def ratio(self) -> float:
        if not self.universe:
            return 1.0
        return len(self.covered & self.universe) / len(self.universe)

    def meets(self, tau: float) -> bool:
        """Whether coverage satisfies the gate ``Coverage >= tau``.

        Coverage is compared over *observable* facts. A ``tau`` of 1.0 demands
        every fact the retrieval could observe was observed.
        """
        return self.ratio >= tau

    def missing_required(self) -> Tuple[str, ...]:
        return tuple(sorted(self.uncovered))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ratio": self.ratio,
            "covered": sorted(self.covered),
            "uncovered": sorted(self.uncovered),
            "universe": sorted(self.universe),
        }


def _lower_rows_by_table(rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Sequence[Mapping[str, Any]]]:
    return {str(table).lower(): rows for table, rows in (rows_by_table or {}).items()}


def realized_relation_pairs(
    registry,
    rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Set[Tuple[str, str]]:
    """``(source_table, target_table)`` FK pairs actually realized by rows.

    A pair is realized when at least one observed row carries a non-NULL value
    in the FK column that points at another table which was also retrieved.
    Only declared schema edges are considered; nothing is inferred.
    """
    observed = _lower_rows_by_table(rows_by_table)
    realized: Set[Tuple[str, str]] = set()
    for table, rows in observed.items():
        if not rows:
            continue
        for fk in registry.outgoing_fks(table):
            target = str(fk.target_table).lower()
            if target not in observed:
                continue
            for row in rows:
                if row.get(fk.source_column) not in (None, ""):
                    realized.add((table, target))
                    break
    return realized


def coverage_report(
    *,
    requirement: TaskRequirements,
    registry,
    rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]],
    anchor: Any = None,
    planned_tables: Optional[Iterable[str]] = None,
) -> CoverageReport:
    """Measure ``Coverage(S, F_T)`` over the facts currently retrieved.

    Args:
        requirement: The task's requirement object (tables/attributes/relations).
        registry: Schema registry, used for FK and column resolution.
        rows_by_table: Rows retrieved so far, keyed by table name.
        anchor: Optional ``(table, row_id)`` of the resolved anchor.
        planned_tables: Tables the plan intends to read. **Required for the
            frontier strategy**: the universe must be scoped to what the plan
            *should* observe, otherwise a required table that has not been
            fetched yet would silently disappear from the denominator and the
            loop would stop at the anchor. When ``None`` the scope degrades to
            the tables already retrieved, which is the historical behaviour.

    Returns:
        A :class:`CoverageReport` whose universe is limited to facts the
        retrieval was *capable* of observing.
    """
    observed = _lower_rows_by_table(rows_by_table)
    retrieved = {table for table in observed}
    if planned_tables is None:
        scope = set(retrieved)
    else:
        scope = {str(table).lower() for table in planned_tables}
    required_tables = {str(table).lower() for table in requirement.required_tables}

    universe: Set[str] = set()
    covered: Set[str] = set()

    # Anchor row.
    if anchor is not None:
        anchor_table, anchor_row = str(anchor[0]).lower(), str(anchor[1])
        if anchor_table in scope:
            universe.add(ANCHOR_FACT)
            if any(
                str(row.get(registry.require_table(anchor_table).primary_key)) == anchor_row
                for row in observed.get(anchor_table, ())
            ):
                covered.add(ANCHOR_FACT)

    # Required tables.
    for table in sorted(required_tables & scope):
        universe.add(table_fact(table))
        if observed.get(table):
            covered.add(table_fact(table))

    # Required attributes, resolved through the shared lexicon resolver so a
    # prompt shorthand ("assignment_group") matches the concrete FK column the
    # query layer projected ("assignment_group_id"). A required column of a
    # planned-but-not-yet-retrieved table stays in the universe as *uncovered*,
    # which is what keeps the frontier loop going instead of stopping early.
    #
    # A required attribute is covered when a retrieved row *carries the column*,
    # regardless of its value: a NULL read back from the database is a grounded
    # answer, not a missing fact. Requiring non-NULL would make tau unattainable
    # for any legitimately NULL column and the loop would never terminate.
    for table in sorted(required_tables & scope):
        rows = observed.get(table) or ()
        for column in resolve_attribute_columns(
            registry, (table,), tuple(requirement.required_attributes)
        ):
            column_l = str(column).lower()
            universe.add(attribute_fact(table, column_l))
            if any(column_l in row for row in rows):
                covered.add(attribute_fact(table, column_l))

    # Required relations.
    realized = realized_relation_pairs(registry, observed)
    for source, target in requirement.required_relations:
        source_l, target_l = str(source).lower(), str(target).lower()
        if source_l in scope and target_l in scope:
            universe.add(relation_fact(source_l, target_l))
            if (source_l, target_l) in realized or (target_l, source_l) in realized:
                covered.add(relation_fact(source_l, target_l))

    return CoverageReport(
        covered=frozenset(covered),
        uncovered=frozenset(universe - covered),
        universe=frozenset(universe),
    )


def uncovered_tables(report: CoverageReport) -> Tuple[str, ...]:
    """Required tables named by ``table:`` facts that remain uncovered."""
    return tuple(sorted(fact[len(TABLE_PREFIX):] for fact in report.uncovered if fact.startswith(TABLE_PREFIX)))


__all__ = [
    "ANCHOR_FACT",
    "ATTRIBUTE_PREFIX",
    "RELATION_PREFIX",
    "TABLE_PREFIX",
    "CoverageReport",
    "attribute_fact",
    "coverage_report",
    "realized_relation_pairs",
    "relation_fact",
    "table_fact",
    "uncovered_tables",
]
