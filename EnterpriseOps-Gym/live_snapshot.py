"""Labeled system snapshot for live verification runs.

Captures the observable state of the Docker-hosted csm-server and the
seeded database so before/after comparisons prove what a step changed:

- server health (/health)
- database reachability + table row counts (fingerprint of the dataset)
- full customer_case 1 row including sys_updated_on (mutation fingerprint)

Usage: python live_snapshot.py <LABEL>
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import httpx

BASE_URL = "http://localhost:8001"
DATABASE_ID = "db_1762254390925_u5icw4thh"
HEADERS = {"x-database-id": DATABASE_ID}
TABLES = ["account", "customer_case", "interaction", "knowledge", "product"]


def _query(client: httpx.Client, sql: str) -> list[dict]:
    response = client.post(
        f"{BASE_URL}/api/sql-runner",
        json={"database_id": DATABASE_ID, "query": sql},
        headers=HEADERS,
        timeout=30,
    )
    if response.status_code != 200:
        print(f"  [query failed {response.status_code}] {sql} -> {response.text[:200]}")
        return []
    return response.json().get("data", [])


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "SNAPSHOT"
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    print(f"\n{'=' * 66}")
    print(f"  {label}  (utc {stamp})")
    print(f"{'=' * 66}")

    with httpx.Client() as client:
        health = client.get(f"{BASE_URL}/health", timeout=10)
        print("server health      :", health.status_code, health.json())

        counts = {}
        for table in TABLES:
            rows = _query(client, f"SELECT COUNT(*) AS n FROM {table};")
            counts[table] = rows[0]["n"] if rows else "?"
        print("table row counts   :", json.dumps(counts))

        case = _query(
            client,
            "SELECT case_id, number, priority, state, assigned_to, "
            "account_id, sys_updated_on FROM customer_case WHERE case_id = 1;",
        )
        print("customer_case 1    :", json.dumps(case[0] if case else None))

        notes = _query(
            client,
            "SELECT COUNT(*) AS n FROM interaction WHERE case_id = 1;",
        )
        print("case 1 interactions:", notes[0]["n"] if notes else "?")


if __name__ == "__main__":
    main()
