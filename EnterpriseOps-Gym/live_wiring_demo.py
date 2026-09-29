"""One-off live wiring demo: gym task JSON -> MCPClient -> csm_env -> live DB.

Run from the EnterpriseOps-Gym root with the Docker csm-server on :8001 and
database db_1762254390925_u5icw4thh seeded. Not part of the test suite.
"""

from __future__ import annotations

import asyncio

from csm_integration import build_csm_environment_from_task

TASK = "data/revised/csm/task_20251205_153330_906_a8eea1c0_8c7a6205.json"
DATABASE_ID = "db_1762254390925_u5icw4thh"


async def main() -> None:
    api = build_csm_environment_from_task(TASK, database_id=DATABASE_ID)
    print("wired base_url:", api.environment.sql.base_url)
    print("wired database_id:", api.environment.sql.database_id)
    print("wired context:", api.environment.sql.context)

    row = await api.get_ground_truth("customer_case", 1)
    print(
        "GROUND TRUTH case 1:",
        {k: row.get(k) for k in ("case_id", "number", "priority", "state", "account_id")},
    )

    context = await api.get_case_context(1, hops=2)
    print("CASE CONTEXT keys:", sorted(context) if isinstance(context, dict) else type(context))
    if isinstance(context, dict):
        for key, value in context.items():
            if isinstance(value, (list, dict)):
                print(f"  {key}: {len(value)} entries")

    state = await api.build_state("resolve_case", 1, "customer_case")
    print("GGQR state:", state.status, "| facts:", len(state.facts))
    representations = api.represent_state(state)
    print("representations:", {k: len(v) for k, v in representations.items()})

    equivalence = await api.environment.ggqr.verify_case_equivalence(1)
    print(
        "GRAPH<->SQL EQUIVALENCE:",
        equivalence["equivalent"],
        "| state:",
        equivalence["state_status"],
        "| failures:",
        equivalence["failures"][:3],
    )


if __name__ == "__main__":
    asyncio.run(main())
