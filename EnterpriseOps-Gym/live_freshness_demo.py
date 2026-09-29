"""One-off live freshness demo: mutate via MCP tool, prove fresh reads, revert.

The sql-runner endpoint is read-only by design (400 on UPDATE), so the
dynamic write path is the server's own MCP tool surface — exactly how the
benchmark's agents mutate state. This script reversibly flips
customer_case 1 priority critical -> low -> critical through the gym's
MCPClient and asserts the wired csm_env ground-truth read reflects each
change (database is the source of truth; no stale cache).
"""

from __future__ import annotations

import asyncio

from benchmark.mcp_client import MCPClient

from csm_integration import build_csm_environment

DATABASE_ID = "db_1762254390925_u5icw4thh"
CONTEXT = {"x-user-email": "joanne.simpson@servicenow.com"}


async def _set_priority(client: MCPClient, priority: str) -> None:
    result = await client.call_tool(
        "update_case", {"case_id": 1, "priority": priority}
    )
    if not result.get("success"):
        raise RuntimeError(f"update_case failed: {result}")


async def main() -> None:
    client = MCPClient(
        base_url="http://localhost:8001", database_id=DATABASE_ID, context=CONTEXT
    )
    await client.initialize()
    api = build_csm_environment(client)

    before = await api.get_ground_truth("customer_case", 1)
    print("BEFORE  :", before["priority"])
    assert before["priority"] == "critical"

    await _set_priority(client, "low")
    print("MUTATE  : update_case priority -> low (via MCP tool)")
    after = await api.get_ground_truth("customer_case", 1)
    print("AFTER   :", after["priority"], "(fresh read, cache bypassed)")
    assert after["priority"] == "low"

    await _set_priority(client, "critical")
    print("REVERT  : update_case priority -> critical (via MCP tool)")
    restored = await api.get_ground_truth("customer_case", 1)
    print("RESTORED:", restored["priority"])
    assert restored["priority"] == "critical"

    print("DYNAMIC FRESHNESS: PROVEN")


if __name__ == "__main__":
    asyncio.run(main())

