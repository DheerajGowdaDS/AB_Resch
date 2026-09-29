from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from ..state.models import GroundedState
from .serializers import REPRESENTATION_VERSION, to_dict, to_json, to_text

# Fields that must survive every representation losslessly (the contract's
# "critical fields"). Presentation-only text formatting is excluded.
_CRITICAL_FACT_KEYS = ("state", "priority", "support_level", "status", "active", "name")


@dataclass(frozen=True)
class FidelityReport:
    representation_version: str
    json_pass: bool
    text_pass: bool
    failures: Tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.json_pass and self.text_pass


def verify_fidelity(state: GroundedState) -> FidelityReport:
    """S_t -> R(S_t) -> R^-1 approx -> S'_t; compare critical fields.

    JSON: full structural reconstruction must equal the source dict.
    Contradiction metadata is part of the lossless payload.
    Text: anchor/record identities and critical fact values must appear.
    """
    failures: List[str] = []

    # JSON round trip.
    source_dict = to_dict(state, redact_secrets=False)
    try:
        rebuilt = json.loads(to_json(state, redact_secrets=False))
    except json.JSONDecodeError as exc:
        rebuilt = None
        failures.append(f"json decode failed: {exc}")
    if rebuilt != source_dict:
        failures.append("json round trip lost information")

    # Text round trip: reconstruct the contract's critical fields from the
    # machine-readable footer rather than relying on substring presence.
    text = to_text(state, max_chars=0)
    text_pass = True
    try:
        marker = " ROUNDTRIP="
        if marker not in text:
            raise ValueError("roundtrip footer missing")
        payload = json.loads(text.split(marker, 1)[1])
        expected = _critical_payload(state)
        if payload != json.loads(json.dumps(expected)):
            text_pass = False
            failures.append("text reconstruction differs from source critical state")
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        text_pass = False
        failures.append(f"text round trip reconstruction failed: {exc}")

    return FidelityReport(
        representation_version=REPRESENTATION_VERSION,
        json_pass=rebuilt == source_dict,
        text_pass=text_pass,
        failures=tuple(failures),
    )


def _critical_payload(state: GroundedState) -> Dict[str, Any]:
    return {
        "state_id": state.state_id,
        "records": sorted((r.table, r.row_id) for r in state.records),
        "facts": sorted(
            [
                {
                    "table": fact.provenance.table,
                    "row_id": fact.provenance.row_pk,
                    "column": fact.provenance.column,
                    "value": fact.value,
                }
                for fact in state.facts
                if fact.provenance.column in _CRITICAL_FACT_KEYS
            ],
            key=lambda x: (x["table"], x["row_id"], x["column"]),
        ),
        "relations": sorted(
            (r.source_table, r.source_id, r.target_table, r.target_id, r.relation, r.edge_id)
            for r in state.relations
        ),
        "unresolved": sorted(state.unresolved_details),
    }


def _render_value(value: Any):
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return None
    return repr(value) if not isinstance(value, str) else repr(value)
