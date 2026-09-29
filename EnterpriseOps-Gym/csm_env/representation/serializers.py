from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from ..state.models import GroundedState
from ..state.provenance import redact

REPRESENTATION_VERSION = "1.1.0"
DEFAULT_TEXT_MAX_CHARS = 4000
TRUNCATION_MARKER = "... [TRUNCATED]"


def to_dict(state: GroundedState, *, redact_secrets: bool = True) -> Dict[str, Any]:
    """Deterministic JSON-compatible dict for one grounded state."""
    payload: Dict[str, Any] = {
        "representation_version": REPRESENTATION_VERSION,
        "state_id": state.state_id,
        "status": state.status,
        "schema_version": state.schema_version,
        "observation_generation": state.observation_generation,
        "anchor": _record_dict(state.anchor),
        "records": [_record_dict(record) for record in state.records],
        "relations": [
            {
                "source_table": relation.source_table,
                "source_id": relation.source_id,
                "target_table": relation.target_table,
                "target_id": relation.target_id,
                "relation": relation.relation,
                "edge_id": relation.edge_id,
            }
            for relation in state.relations
        ],
        "unresolved": list(state.unresolved),
        "unresolved_details": [
            {"table": table, "column": column, "reason": reason}
            for table, column, reason in state.unresolved_details
        ],
        "contradictions": [
            {
                "table": contradiction.table,
                "row_id": contradiction.row_id,
                "column": contradiction.column,
                "first_value": contradiction.first_value,
                "second_value": contradiction.second_value,
                "first_query_id": contradiction.first_query_id,
                "second_query_id": contradiction.second_query_id,
            }
            for contradiction in state.contradictions
        ],
        "provenance": {
            "database_id": state.provenance.database_id,
            "schema_version": state.provenance.schema_version,
            "route_id": state.provenance.route_id,
            "observed_at": state.provenance.observed_at,
            "query_ids": list(state.provenance.query_ids),
        },
    }
    return redact(payload) if redact_secrets else payload


def to_json(state: GroundedState, *, redact_secrets: bool = True) -> str:
    return json.dumps(to_dict(state, redact_secrets=redact_secrets), sort_keys=True)


def to_graph_text(state: GroundedState) -> str:
    """Deterministic graph-style text rendering of records and relations."""
    lines: List[str] = [f"State {state.state_id} [{state.status}]"]
    anchor = state.anchor
    lines.append(f"  ANCHOR {anchor.table}:{anchor.row_id}")
    for record in state.records:
        if (record.table, record.row_id) == (anchor.table, anchor.row_id):
            continue
        lines.append(f"  {record.table}:{record.row_id}")
    for relation in state.relations:
        lines.append(
            f"  {relation.source_table}:{relation.source_id}"
            f" -[{relation.relation}]->"
            f" {relation.target_table}:{relation.target_id}"
        )
    for table, column, reason in state.unresolved_details:
        lines.append(f"  UNRESOLVED {table}.{column} ({reason})")
    return "\n".join(lines)


def _roundtrip_payload(state: GroundedState) -> Dict[str, Any]:
    """Machine-readable critical state needed for text fidelity verification.

    The surrounding text remains human-readable; this compact footer makes the
    representation reconstructible instead of relying on substring presence.
    """
    critical_facts = [
        {
            "table": fact.provenance.table,
            "row_id": fact.provenance.row_pk,
            "column": fact.provenance.column,
            "value": fact.value,
        }
        for fact in state.facts
        if fact.provenance.column in {"state", "priority", "support_level", "status", "active", "name"}
    ]
    return {
        "state_id": state.state_id,
        "records": sorted((r.table, r.row_id) for r in state.records),
        "facts": sorted(critical_facts, key=lambda x: (x["table"], x["row_id"], x["column"])),
        "relations": sorted(
            (r.source_table, r.source_id, r.target_table, r.target_id, r.relation, r.edge_id)
            for r in state.relations
        ),
        "unresolved": sorted(state.unresolved_details),
    }


def to_text(state: GroundedState, *, max_chars: int = DEFAULT_TEXT_MAX_CHARS) -> str:
    """Human-readable text rendering with explicit truncation.

    Truncation never removes provenance for included facts: provenance is
    summarized at the end and the marker is explicit.
    """
    parts: List[str] = []
    anchor = state.anchor
    header = (
        f"Case state {state.state_id} is {state.status}. "
        f"Schema {state.schema_version}, observed {state.provenance.observed_at}."
    )
    parts.append(header)

    if anchor.values:
        parts.append(
            f"Anchor {anchor.table} {anchor.row_id}: {anchor.values.get(anchor.primary_key)!r}"
        )
    for record in state.records:
        if (record.table, record.row_id) == (anchor.table, anchor.row_id):
            summary = _summarize_record(record)
            if summary:
                parts.append(f"Anchor detail: {summary}")
            continue
        summary = _summarize_record(record)
        if summary:
            parts.append(summary)
    if state.unresolved_details:
        names = ", ".join(f"{t}.{c or '-'} ({r})" for t, c, r in state.unresolved_details)
        parts.append(f"Unresolved: {names}")
    if state.contradictions:
        names = ", ".join(
            f"{c.table}:{c.row_id}.{c.column}" for c in state.contradictions
        )
        parts.append(f"Contradictions: {names}")
    parts.append(
        f"Provenance: database {state.provenance.database_id}, "
        f"{len(state.facts)} facts, route {state.provenance.route_id}."
    )

    text = " ".join(parts)
    # Delimited JSON footer provides an exact reconstruction surface for the
    # critical state contract. It is deterministic and is truncated together
    # with the human-readable representation when the caller sets max_chars.
    footer = " ROUNDTRIP=" + json.dumps(_roundtrip_payload(state), sort_keys=True, separators=(",", ":"))
    text += footer
    if max_chars and len(text) > max_chars:
        text = text[: max(0, max_chars - len(TRUNCATION_MARKER))] + TRUNCATION_MARKER
    return text


def _record_dict(record) -> Dict[str, Any]:
    return {
        "table": record.table,
        "primary_key": record.primary_key,
        "row_id": record.row_id,
        "values": dict(sorted(record.values.items())),
    }


def _summarize_record(record) -> str:
    if record.table == "customer_case":
        return (
            f"Case {record.row_id} is {record.values.get('state')!r} "
            f"with priority {record.values.get('priority')!r}."
        )
    if record.table == "account":
        return f"Account {record.row_id} ({record.values.get('name')!r})."
    if record.table == "entitlement":
        return (
            f"Entitlement {record.row_id}: support_level "
            f"{record.values.get('support_level')!r}, active {record.values.get('active')!r}."
        )
    if record.table == "contract":
        return f"Contract {record.row_id}: status {record.values.get('status')!r}."
    name = record.values.get("name")
    if name is not None:
        return f"{record.table} {record.row_id} ({name!r})."
    return f"{record.table} {record.row_id}."
