from __future__ import annotations

import json
from typing import Any, Dict, Optional, List

import httpx


class EnterpriseOpsSQLRunner:
    """Read the same live database used by EnterpriseOps-Gym verifiers.

    This follows the repository's VerifierEngine implementation: POST to
    <mcp_server_url>/api/sql-runner with query + database_id and x-database-id.
    """

    def __init__(
        self,
        base_url: str,
        database_id: str,
        auth_config: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.database_id = database_id
        self.auth_config = auth_config or {}
        self.context = context or {}
        self.timeout = timeout

    def _auth_headers(self) -> Dict[str, str]:
        auth_type = self.auth_config.get("type")
        token = self.auth_config.get("token")
        header_name = self.auth_config.get("header_name", "Authorization")
        if auth_type == "bearer":
            return {header_name: f"Bearer {token}"}
        if auth_type == "api_key":
            return {header_name: str(token)}
        return {}

    def _context_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        for key, value in self.context.items():
            header_key = key if key.lower().startswith("x-") else f"x-{key.lower().replace('_', '-')}"
            headers[header_key] = str(value)
        return headers

    async def execute(self, query: str) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "x-database-id": self.database_id,
        }
        headers.update(self._auth_headers())
        headers.update(self._context_headers())
        payload = {"query": query, "database_id": self.database_id}
        url = f"{self.base_url}/api/sql-runner"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    async def fetch_rows(self, query: str) -> List[Dict[str, Any]]:
        result = await self.execute(query)
        return extract_rows(result)


def extract_rows(result: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Normalize common EnterpriseOps-Gym sql-runner response shapes."""
    if not isinstance(result, dict):
        raise TypeError(f"Expected dict result, got {type(result).__name__}")

    candidate = result
    if "data" in candidate:
        candidate = candidate["data"]
    elif "rows" in candidate:
        candidate = candidate["rows"]
    elif "result" in candidate and isinstance(candidate["result"], dict):
        return extract_rows(candidate["result"])
    elif "content" in candidate:
        for item in candidate.get("content", []):
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                try:
                    decoded = json.loads(text)
                    return extract_rows(decoded)
                except json.JSONDecodeError:
                    continue
        return []

    if candidate is None:
        return []
    if not isinstance(candidate, list):
        return [candidate] if isinstance(candidate, dict) else []

    rows: list[Dict[str, Any]] = []
    for row in candidate:
        if isinstance(row, dict):
            rows.append(row)
        elif isinstance(row, list):
            rows.append({str(i): value for i, value in enumerate(row)})
    return rows
