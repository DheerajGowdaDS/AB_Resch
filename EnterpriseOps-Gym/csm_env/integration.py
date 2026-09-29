from __future__ import annotations

from typing import Any, Dict, Optional

from .builder import CSMEnvironmentRepresentation
from .environment import GGQREnvironment
from .query.models import QueryBudget
from .sql_runner import EnterpriseOpsSQLRunner
from .transport.reader import EnterpriseOpsSQLRunnerReader


def from_enterpriseops_mcp_client(
    mcp_client: Any,
    *,
    enable_ggqr: bool = False,
    budget: Optional[QueryBudget] = None,
) -> CSMEnvironmentRepresentation:
    """Adapt the repository's benchmark.mcp_client.MCPClient into this layer.

    The function intentionally depends only on the MCPClient's public-ish
    configuration fields used by EnterpriseOps-Gym itself: base_url,
    database_id, auth_config, and context.

    When ``enable_ggqr=True`` the returned representation carries an attached
    :class:`GGQREnvironment` (accessible as ``representation.ggqr``), which
    the additive API methods use. GGQR remains read-only and deterministic.
    """
    sql = EnterpriseOpsSQLRunner(
        base_url=mcp_client.base_url,
        database_id=mcp_client.database_id,
        auth_config=getattr(mcp_client, "auth_config", None),
        context=getattr(mcp_client, "context", None),
    )
    representation = CSMEnvironmentRepresentation(sql)

    if enable_ggqr:
        reader = EnterpriseOpsSQLRunnerReader(sql)
        ggqr = GGQREnvironment(reader, database_id=mcp_client.database_id, budget=budget)
        representation.ggqr = ggqr  # additive attribute; legacy behavior intact

    return representation
