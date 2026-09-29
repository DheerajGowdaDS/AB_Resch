"""Generate live_verification_report.json — source of truth for the live run.

Embeds the snapshots recorded during the verification session, captures the
CURRENT live system state for cross-check, pulls Docker container info, and
computes before/after diffs programmatically so the "what changed" section
is derived from data, not written by hand.

Run from the EnterpriseOps-Gym root with the csm-server container up.
Output: live_verification_report.json in the repo root.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx

BASE_URL = "http://localhost:8001"
DATABASE_ID = "db_1762254390925_u5icw4thh"
HEADERS = {"x-database-id": DATABASE_ID}
TABLES = ["account", "customer_case", "interaction", "knowledge", "product"]
OUT_FILE = "live_verification_report.json"

# --- Snapshots recorded verbatim during the verification session -----------
RECORDED_SNAPSHOTS: Dict[str, Dict[str, Any]] = {
    "snapshot_1_before_live_suite": {
        "captured_at_utc": "2026-09-24T15:19:15.431Z",
        "server_health": {"status_code": 200, "body": {"status": "healthy", "service": "sn-csm"}},
        "table_row_counts": {"account": 52, "customer_case": 1232, "interaction": 1232,
                             "knowledge": 369, "product": 240},
        "customer_case_1": {"case_id": 1, "number": "CS-0000001", "priority": "critical",
                            "state": "new", "assigned_to": 403, "account_id": 1,
                            "sys_updated_on": "2026-09-24 15:10:43"},
        "case_1_interactions": 1,
    },
    "snapshot_2_after_live_suite": {
        "captured_at_utc": "2026-09-24T15:19:35.400Z",
        "server_health": {"status_code": 200, "body": {"status": "healthy", "service": "sn-csm"}},
        "table_row_counts": {"account": 52, "customer_case": 1232, "interaction": 1232,
                             "knowledge": 369, "product": 240},
        "customer_case_1": {"case_id": 1, "number": "CS-0000001", "priority": "critical",
                            "state": "new", "assigned_to": 403, "account_id": 1,
                            "sys_updated_on": "2026-09-24 15:10:43"},
        "case_1_interactions": 1,
    },
    "snapshot_3_after_mutation_cycle": {
        "captured_at_utc": "2026-09-24T15:20:09.765Z",
        "server_health": {"status_code": 200, "body": {"status": "healthy", "service": "sn-csm"}},
        "table_row_counts": {"account": 52, "customer_case": 1232, "interaction": 1232,
                             "knowledge": 369, "product": 240},
        "customer_case_1": {"case_id": 1, "number": "CS-0000001", "priority": "critical",
                            "state": "new", "assigned_to": 403, "account_id": 1,
                            "sys_updated_on": "2026-09-24 15:19:47"},
        "case_1_interactions": 1,
    },
}

TEST_RUN = {
    "command": ("python -m pytest csm_env/tests tests/test_csm_wiring_smoke.py "
                "-p no:cacheprovider --asyncio-mode=auto"),
    "env_vars": {"CSM_LIVE_BASE_URL": BASE_URL, "CSM_LIVE_DATABASE_ID": DATABASE_ID},
    "note": ("--asyncio-mode=auto is a CLI-only harness flag; the 3 live opt-in "
             "tests lack @pytest.mark.asyncio markers, no source files were edited"),
    "results": {"total": 124, "passed": 124, "failed": 0, "skipped": 0, "duration_s": 3.68,
                "breakdown": {"csm_env_offline": 111, "csm_env_live_opt_in": 3,
                              "wiring_smoke": 10}},
    "live_opt_in_tests": [
        "csm_env/tests/verification/test_live_opt_in.py::test_live_schema_manifest_matches_database",
        "csm_env/tests/verification/test_live_opt_in.py::test_live_case_routing_round_trip",
        "csm_env/tests/verification/test_live_opt_in.py::test_live_graph_sql_equivalence",
    ],
}

MUTATION_CYCLE = {
    "purpose": "prove live writes reach the DB and csm_env fresh reads reflect them instantly",
    "write_path": "MCP tool update_case via benchmark.mcp_client.MCPClient "
                  "(sql-runner is read-only by design, 400 on UPDATE)",
    "target": "customer_case.case_id = 1, field priority",
    "phases": [
        {"step": "before", "priority": "critical"},
        {"step": "mutate", "action": "update_case(case_id=1, priority='low')"},
        {"step": "after_fresh_read", "priority": "low", "note": "cache bypassed, fresh DB read"},
        {"step": "revert", "action": "update_case(case_id=1, priority='critical')"},
        {"step": "restored", "priority": "critical"},
    ],
    "assertions_passed": True,
}


def _query(client: httpx.Client, sql: str) -> List[Dict[str, Any]]:
    response = client.post(
        f"{BASE_URL}/api/sql-runner",
        json={"database_id": DATABASE_ID, "query": sql},
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("data", [])


def capture_current_state() -> Dict[str, Any]:
    """Fresh live snapshot taken at report-generation time."""
    with httpx.Client() as client:
        health = client.get(f"{BASE_URL}/health", timeout=10)
        counts = {}
        for table in TABLES:
            rows = _query(client, f"SELECT COUNT(*) AS n FROM {table};")
            counts[table] = rows[0]["n"] if rows else None
        case = _query(
            client,
            "SELECT case_id, number, priority, state, assigned_to, account_id, "
            "sys_updated_on FROM customer_case WHERE case_id = 1;",
        )
        interactions = _query(
            client, "SELECT COUNT(*) AS n FROM interaction WHERE case_id = 1;"
        )
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "server_health": {"status_code": health.status_code, "body": health.json()},
        "table_row_counts": counts,
        "customer_case_1": case[0] if case else None,
        "case_1_interactions": interactions[0]["n"] if interactions else None,
    }


def docker_container_info() -> Dict[str, Any]:
    try:
        out = subprocess.run(
            ["docker", "ps", "--filter", "name=csm-server", "--format", "{{json .}}"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip()
        return json.loads(out) if out else {"error": "container not found"}
    except Exception as exc:  # docker missing/not running — record, don't fail
        return {"error": str(exc)}


def diff_snapshots(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Field-level diff; skips capture timestamps. Derived, never hand-written."""
    changed, unchanged = [], []

    def walk(prefix: str, a: Any, b: Any) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(f"{prefix}.{key}" if prefix else key, a.get(key), b.get(key))
        elif a == b:
            unchanged.append(prefix)
        else:
            changed.append({"field": prefix, "from": a, "to": b})

    for key in sorted(set(before) | set(after)):
        if key == "captured_at_utc":
            continue
        walk(key, before.get(key), after.get(key))
    return {"changed": changed, "unchanged_fields": unchanged,
            "verdict": "IDENTICAL" if not changed else f"{len(changed)} field(s) changed"}


def main() -> None:
    snap1 = RECORDED_SNAPSHOTS["snapshot_1_before_live_suite"]
    snap2 = RECORDED_SNAPSHOTS["snapshot_2_after_live_suite"]
    snap3 = RECORDED_SNAPSHOTS["snapshot_3_after_mutation_cycle"]
    current = capture_current_state()

    report = {
        "report": {
            "title": "EnterpriseOps-Gym x csm_env — live verification source of truth",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": ("Docker-hosted csm-server, seeded CSM database, full live test "
                      "suite + dynamic mutation cycle, with before/after snapshots"),
        },
        "system": {
            "docker_container": docker_container_info(),
            "server": {"base_url": BASE_URL, "health": current["server_health"]},
            "database": {
                "database_id": DATABASE_ID,
                "seed_source": ("gym_dbs.zip :: Domain Wise DBs and Task-DB Mappings/"
                                "csm/dbs/db_1762254390925_u5icw4thh.sql"),
                "seeded_via": "manual POST /api/seed-database",
                "note": ("manually seeded, so the executor's auto-cleanup does not "
                         "cover it; delete via /api/delete-database when done"),
            },
        },
        "test_run": TEST_RUN,
        "mutation_cycle": MUTATION_CYCLE,
        "snapshots_recorded_during_run": RECORDED_SNAPSHOTS,
        "current_state_at_generation": current,
        "diffs": {
            "suite_impact_snapshot1_vs_snapshot2": diff_snapshots(snap1, snap2),
            "overall_impact_snapshot1_vs_snapshot3": diff_snapshots(snap1, snap3),
            "snapshot3_vs_current_state": diff_snapshots(snap3, current),
        },
        "conclusions": [
            "124/124 tests passed (111 offline + 3 live opt-in + 10 wiring smoke).",
            "Live suite is fully read-only: snapshot_1 == snapshot_2 in every field.",
            "Dynamic write path proven: MCP update_case write was visible to fresh "
            "csm_env reads immediately; value reverted afterwards.",
            "Only residual system change: customer_case 1 sys_updated_on bumped by "
            "the server's audit trigger (15:10:43 -> 15:19:47); all values restored.",
        ],
        "artifacts": [
            "csm_env/ (wired package)", "csm_integration.py",
            "tests/test_csm_wiring_smoke.py", "live_snapshot.py",
            "live_wiring_demo.py", "live_freshness_demo.py",
            "generate_live_report.py", OUT_FILE,
        ],
    }

    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"written: {OUT_FILE}")
    for name, diff in report["diffs"].items():
        print(f"  {name}: {diff['verdict']}")
        for entry in diff["changed"]:
            print(f"    changed: {entry['field']}: {entry['from']} -> {entry['to']}")


if __name__ == "__main__":
    main()
