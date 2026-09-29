#!/usr/bin/env python3
"""Live delivery probe: does the Condition-B intervention actually deliver?

This is the Phase 1.1 acceptance test, run against the **real** CSM server and
the **real** seed databases, but **without** an LLM. It isolates the expensive
and fragile part of the experiment - can the state model reliably resolve an
anchor, route to its required tables, and hand the agent usable state? - from
agent behaviour, which is the separation the blueprint insists on.

Why this exists
---------------
The 11-task pilot reported 7/11 delivery and therefore withheld the causal
effect. Three of those four failures were not routing faults at all (a required
table was simply empty), and one was a malformed-SQL 400. Both were fixed, but
neither fix can be verified offline: unit tests prove the *logic*, only this
probe proves the *behaviour against real data*.

The probe therefore replays, for every task:

1.  seed the real database from ``gym_dbs.zip``;
2.  fingerprint it (read-only) so a run is attributable to known state;
3.  resolve the anchor from the task prompt;
4.  build the grounded state through the normal GGQR pipeline, with hop
    escalation enabled;
5.  apply the *same* delivery rule the orchestrator enforces;
6.  delete the database it created.

Because the LLM never runs, this costs no tokens and is safe to re-run freely.

Usage (from ``EnterpriseOps-Gym``)::

    python validate_delivery_live.py
    python validate_delivery_live.py --task-ids task_20260101_122222_300_ad5a67e3_3757206c
    python validate_delivery_live.py --output-folder out/delivery_probe

Exit code is 0 only when every task delivered, so this doubles as a CI gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

GYM_ROOT = Path(__file__).resolve().parent
if str(GYM_ROOT) not in sys.path:
    sys.path.insert(0, str(GYM_ROOT))

from ablation.fingerprint import initial_state_fingerprint  # noqa: E402
from ablation.gold_state import build_gold_state_map, compare_to_gold  # noqa: E402
from ablation.route_policy import DEFAULT_HOP_LADDER  # noqa: E402
from ablation.seeds import SeedCache  # noqa: E402
from ablation.state_model import StateModelAdapter  # noqa: E402
from ablation.state_model_orchestrator import state_context_delivered  # noqa: E402
from ablation.task_registry import load_eval_set  # noqa: E402
from benchmark.mcp_client import (  # noqa: E402
    MCPClient,
    create_database_from_file,
    delete_database,
)
from csm_env import EnterpriseOpsSQLRunner, SchemaRegistry  # noqa: E402
from csm_env.transport.reader import EnterpriseOpsSQLRunnerReader  # noqa: E402
from csm_integration import build_csm_environment  # noqa: E402

DEFAULT_EVAL_SET = GYM_ROOT / "ablation" / "manifests" / "csm_eval_set.json"
DEFAULT_REQUIREMENTS = GYM_ROOT / "ablation" / "manifests" / "csm_requirements.json"
DEFAULT_OUTPUT = GYM_ROOT / "out" / "delivery_probe"
DEFAULT_ARCHIVE = GYM_ROOT / "gym_dbs.zip"


def _server_entry(task_config_path: Path) -> Dict[str, Any]:
    """Read the first ``gym_servers_config`` entry of a task file."""
    payload = json.loads(Path(task_config_path).read_text(encoding="utf-8"))
    servers = payload.get("gym_servers_config")
    if not isinstance(servers, list) or not servers:
        raise ValueError(f"{task_config_path}: missing gym_servers_config")
    return servers[0]


async def probe_task(
    record: Any,
    *,
    registry: SchemaRegistry,
    seed_cache: SeedCache,
    requirements_path: Path,
    gold_states: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Seed, probe, and clean up one task's database.

    Returns one report row. The database is always deleted, including when the
    probe raises, so a failure never leaks server-side state.
    """
    entry = _server_entry(Path(record.task_config_path))
    gym_url = entry["mcp_server_url"]
    seed_on_disk = seed_cache.materialize(entry["seed_database_file"])
    context = entry.get("context") or {}
    auth_config = entry.get("auth_config")

    row: Dict[str, Any] = {"task_id": record.task_id, "error": None}
    row["complexity_category"] = record.complexity_category
    row["ggqr_task_type"] = record.ggqr_task_type

    database_id: Optional[str] = None
    started = time.perf_counter()
    try:
        database_id = create_database_from_file(gym_url, str(seed_on_disk))
        if not database_id:
            row["error"] = "seed-database returned no database_id"
            return row

        client = MCPClient(
            base_url=gym_url,
            database_id=database_id,
            auth_config=auth_config,
            context=context,
        )
        row["database_id"] = database_id

        api = build_csm_environment(
            client, enable_ggqr=True, requirements_path=requirements_path
        )
        # The adapter's own reader is the live SQL runner, exactly as in a run.
        adapter = StateModelAdapter(api=api, task=record, registry=registry)

        runner = EnterpriseOpsSQLRunner(
            base_url=gym_url,
            database_id=database_id,
            auth_config=auth_config,
            context=context,
        )
        row["fingerprint"] = await initial_state_fingerprint(
            EnterpriseOpsSQLRunnerReader(runner), registry
        )

        ctx = await adapter.context_for()
        row.update(
            anchor_table=ctx.reference_type,
            anchor_row_id=ctx.reference_id,
            anchor_strategy=ctx.anchor.strategy,
            anchor_lookups=ctx.anchor.lookups,
            route_status=ctx.route_status,
            structurally_valid=ctx.structurally_valid,
            uncovered_required=list(ctx.uncovered_required),
            resolved_max_hops=ctx.resolved_max_hops,
            retrieved_tables=len(ctx.retrieved_tables),
            retrieved_records=ctx.retrieved_record_count,
            retrieved_facts=ctx.retrieved_fact_count,
            retrieved_tokens=ctx.retrieved_token_count,
            injected_chars=len(ctx.rendered_text),
            contradictions=list(ctx.contradictions),
            delivered=state_context_delivered(ctx),
            error=ctx.error,
        )

        # Fact-level retrieval quality against the evaluator-only gold state.
        gold = (gold_states or {}).get(record.task_id)
        if gold is not None:
            comparison = compare_to_gold(gold, adapter.last_state)
            row["gold_fact_count"] = comparison.gold_fact_count
            row["retrieved_fact_keys"] = comparison.retrieved_fact_count
            row["matched_fact_count"] = comparison.matched_fact_count
            row["precision"] = comparison.precision
            row["recall"] = comparison.recall
            row["unresolved_facts"] = [dict(f) for f in comparison.unresolved_facts]
    except Exception as exc:  # noqa: BLE001 - reported per task, never fatal
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["delivered"] = False
    finally:
        if database_id:
            try:
                delete_database(gym_url, database_id)
            except Exception:  # noqa: BLE001 - cleanup must not mask the result
                pass
        row["elapsed_s"] = round(time.perf_counter() - started, 2)

    return row


def _fmt(value: Optional[float], digits: int = 3) -> str:
    """Format a ratio for console output, tolerating ``None``."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_table(rows: Sequence[Dict[str, Any]]) -> str:
    """Render the probe results as the blueprint's Phase 2 diagnostic table."""
    header = (
        "| Task | Anchor | Route | Tables | Records | Delivery | Precision | Recall | Gold |\n"
        "|---|---|---|---:|---:|---|---:|---:|---:|\n"
    )
    body = []
    for row in rows:
        anchor = (
            f"{row.get('anchor_table')}:{row.get('anchor_row_id')}"
            if row.get("anchor_table")
            else "unresolved"
        )
        body.append(
            "| {task} | {anchor} | {route} | {tables} | {records} | {delivered} "
            "| {precision} | {recall} | {gold} |".format(
                task=str(row.get("task_id", ""))[-12:],
                anchor=anchor,
                route=row.get("route_status") or "n/a",
                tables=row.get("retrieved_tables") or 0,
                records=row.get("retrieved_records") or 0,
                delivered="yes" if row.get("delivered") else "NO",
                precision=_fmt(row.get("precision")),
                recall=_fmt(row.get("recall")),
                gold=row.get("gold_fact_count", "n/a"),
            )
        )
    return header + "\n".join(body)


async def run(args: argparse.Namespace) -> int:
    """Probe every selected task and write the delivery report."""
    registry = SchemaRegistry.from_static()
    records = list(load_eval_set(args.eval_set))
    if args.task_ids:
        wanted = {item.strip() for item in args.task_ids.split(",") if item.strip()}
        records = [record for record in records if record.task_id in wanted]
        if not records:
            raise SystemExit(f"no tasks matched {sorted(wanted)}")
    if args.limit:
        records = records[: args.limit]

    seed_cache = SeedCache(args.archive)
    gold_states = build_gold_state_map(
        [record.task_config_path for record in records], registry=registry
    )
    print(
        f"Probing {len(records)} task(s) against the live CSM server "
        f"(no LLM, hop ladder {list(DEFAULT_HOP_LADDER)}, fact-level gold scoring)\n"
    )

    rows: List[Dict[str, Any]] = []
    for record in records:
        row = await probe_task(
            record,
            registry=registry,
            seed_cache=seed_cache,
            requirements_path=Path(args.requirements_path),
            gold_states=gold_states,
        )
        rows.append(row)
        status = "DELIVERED" if row.get("delivered") else "FAILED   "
        detail = row.get("error") or (
            f"{row.get('route_status')} records={row.get('retrieved_records')} "
            f"P={_fmt(row.get('precision'))} R={_fmt(row.get('recall'))} "
            f"gold={row.get('gold_fact_count')}"
        )
        print(f"  [{status}] {record.task_id[-16:]}  {detail}  ({row['elapsed_s']}s)")

    delivered = sum(1 for row in rows if row.get("delivered"))
    rate = (delivered / len(rows)) if rows else 0.0
    precisions = [row["precision"] for row in rows if row.get("precision") is not None]
    recalls = [row["recall"] for row in rows if row.get("recall") is not None]
    mean_precision = sum(precisions) / len(precisions) if precisions else None
    mean_recall = sum(recalls) / len(recalls) if recalls else None

    print()
    print(render_table(rows))
    print()
    print(f"DeliveryRate   = {delivered}/{len(rows)} = {rate:.1%}   (target 100%)")
    print(f"Fact precision = {_fmt(mean_precision)}   (target >= 0.80)")
    print(f"Fact recall    = {_fmt(mean_recall)}   (target >= 0.80)")

    output = Path(args.output_folder)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "tasks": len(rows),
        "delivered": delivered,
        "delivery_rate": rate,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "hop_ladder": list(DEFAULT_HOP_LADDER),
        "targets": {
            "delivery": {"target": 1.0, "observed": rate, "met": rate >= 1.0},
            "precision": {
                "target": 0.80,
                "observed": mean_precision,
                "met": mean_precision is not None and mean_precision >= 0.80,
            },
            "recall": {
                "target": 0.80,
                "observed": mean_recall,
                "met": mean_recall is not None and mean_recall >= 0.80,
            },
        },
        "per_task": rows,
    }
    (output / "delivery_probe.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (output / "delivery_probe.md").write_text(
        "# Live delivery probe (no LLM)\n\n"
        + render_table(rows)
        + f"\n\nDeliveryRate   = {delivered}/{len(rows)} = {rate:.1%}\n"
        + f"Fact precision = {_fmt(mean_precision)}\n"
        + f"Fact recall    = {_fmt(mean_recall)}\n",
        encoding="utf-8",
    )
    print(f"Written: {output / 'delivery_probe.json'}")
    return 0 if rate >= 1.0 else 1


def build_parser() -> argparse.ArgumentParser:
    """Construct the probe CLI."""
    parser = argparse.ArgumentParser(
        prog="validate_delivery_live.py",
        description="Live, LLM-free validation of Condition-B state delivery.",
    )
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument("--requirements-path", default=str(DEFAULT_REQUIREMENTS))
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    parser.add_argument("--output-folder", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--task-ids", default="", help="Comma-separated subset")
    parser.add_argument("--limit", type=int, default=0, help="Probe at most N tasks")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
