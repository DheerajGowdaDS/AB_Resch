from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..state.models import GroundedState


@dataclass
class ComparisonReport:
    """End-to-end comparison artifact: pipeline state vs oracle state."""

    matches: int = 0
    missing_facts: List[Tuple[str, str, str]] = field(default_factory=list)
    extra_facts: List[Tuple[str, str, str]] = field(default_factory=list)
    value_mismatches: List[Tuple[str, str, str, str]] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)
    transport_errors: List[str] = field(default_factory=list)
    schema_errors: List[str] = field(default_factory=list)

    @property
    def equivalent(self) -> bool:
        return not (
            self.missing_facts
            or self.extra_facts
            or self.value_mismatches
            or self.transport_errors
            or self.schema_errors
        )

    def summary(self) -> str:
        return (
            f"matches={self.matches} missing={len(self.missing_facts)} "
            f"extra={len(self.extra_facts)} mismatches={len(self.value_mismatches)} "
            f"unresolved={len(self.unresolved)} errors={len(self.transport_errors) + len(self.schema_errors)}"
        )


def compare_states(pipeline_state: GroundedState, oracle_facts: Dict[Tuple[str, str, str], str]) -> ComparisonReport:
    """Compare pipeline facts against independently-derived oracle facts.

    `oracle_facts` maps (table, row_id, column) -> serialized value. The
    comparison is exact for values that are supposed to be equivalent.
    """
    report = ComparisonReport()
    pipeline_facts: Dict[Tuple[str, str, str], str] = {}
    for fact in pipeline_state.facts:
        key = (fact.provenance.table, fact.provenance.row_pk, fact.provenance.column)
        pipeline_facts[key] = _serialize(fact.value)

    for key, oracle_value in oracle_facts.items():
        if key not in pipeline_facts:
            report.missing_facts.append(key)
        elif pipeline_facts[key] != oracle_value:
            report.value_mismatches.append((key[0], key[1], key[2], oracle_value))
        else:
            report.matches += 1

    for key in pipeline_facts:
        if key not in oracle_facts:
            report.extra_facts.append(key)

    for table, column, reason in pipeline_state.unresolved_details:
        report.unresolved.append(f"{table}.{column or '-'} ({reason})")
    return report


def _serialize(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)
