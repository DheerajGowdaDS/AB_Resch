"""Index-keyed verifier execution, to recover collapsed verifier results.

``BenchmarkExecutor._run_verifiers`` stores results in a dict keyed by
``verifier.name``. When two verifiers share a name, the later overwrites the
earlier, so the official summary can silently under-count and even mask a
failure (upstream issue #23).

This module re-runs the *unchanged* ``VerifierEngine`` over the same verifier
list but keys results by position, then reports both the collapsed and the
index-keyed summary. The benchmark code itself is never edited.

Because ``BenchmarkExecutor.execute_benchmark`` deletes the seeded database in a
``finally`` block, the corrected pass must run inside the run, while the
database is still alive. :class:`IndexedVerifierExecutor` does exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from benchmark.executor import BenchmarkExecutor
from benchmark.models import VerifierConfig
from benchmark.verifier import VerifierEngine


def verifier_key(verifier: Dict[str, Any], index: int) -> str:
    """Positional key that stays unique even when names collide."""
    name = verifier.get("name")
    label = name if isinstance(name, str) and name else f"verifier_{index + 1}"
    return f"{index}:{label}"


def summarize(results: Dict[str, Any]) -> Dict[str, Any]:
    """Build a pass/total summary for a verifier result mapping."""
    total = len(results)
    passed = sum(1 for result in results.values() if result.get("passed", False))
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": (passed / total) if total else 0.0,
    }


@dataclass(frozen=True)
class IndexedVerificationReport:
    """Both views of one task's verifier outcomes."""

    collapsed_results: Dict[str, Any]
    indexed_results: Dict[str, Any]
    collapsed_summary: Dict[str, Any]
    indexed_summary: Dict[str, Any]

    @property
    def collapsed_count(self) -> int:
        return int(self.collapsed_summary.get("total", 0))

    @property
    def indexed_count(self) -> int:
        return int(self.indexed_summary.get("total", 0))

    @property
    def lost_to_collision(self) -> int:
        """How many verifier results the name collision destroyed."""
        return max(0, self.indexed_count - self.collapsed_count)

    @property
    def success_differs(self) -> bool:
        return self.collapsed_summary.get("passed") != self.indexed_summary.get("passed")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "collapsed_summary": dict(self.collapsed_summary),
            "indexed_summary": dict(self.indexed_summary),
            "lost_to_collision": self.lost_to_collision,
            "success_differs": self.success_differs,
        }


def select_database(
    verifier: VerifierConfig,
    gym_configs: Sequence[Dict[str, Any]],
    default_database_id: Optional[str],
    default_context: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool]:
    """Mirror the benchmark's gym -> database resolution.

    Returns ``(database_id, context, skip)``; ``skip`` is ``True`` when a named
    gym is not configured, which is how the benchmark drops the verifier.
    """
    if verifier.gym_name:
        for gym in gym_configs or ():
            if gym.get("mcp_server_name") == verifier.gym_name:
                return gym.get("database_id"), gym.get("context", {}), False
        return None, None, True
    return default_database_id, default_context, False


def build_model_response(task_result: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild the ``model_response`` payload the benchmark passes to verifiers."""
    return {
        "content": task_result.get("final_response", "") or "",
        "tool_calls": [
            {"name": item.get("tool_name"), "args": item.get("arguments")}
            for item in task_result.get("tool_results", []) or []
        ],
    }


async def run_verifiers_indexed(
    verifier_engine: VerifierEngine,
    verifier_configs: Sequence[Dict[str, Any]],
    task_result: Dict[str, Any],
    gym_configs: Sequence[Dict[str, Any]],
    default_database_id: Optional[str] = None,
    default_context: Optional[Dict[str, Any]] = None,
) -> IndexedVerificationReport:
    """Execute every verifier, keyed by position, and also collapse by name.

    The verifier engine is used unchanged; only the result key differs.
    """
    collapsed: Dict[str, Any] = {}
    indexed: Dict[str, Any] = {}
    model_response = build_model_response(task_result)

    for index, verifier_config in enumerate(verifier_configs or ()):
        verifier = VerifierConfig(**verifier_config)
        label = verifier.name or f"verifier_{index + 1}"
        database_id, context, skip = select_database(
            verifier, gym_configs, default_database_id, default_context
        )
        if skip:
            continue
        result = await verifier_engine.execute_verifier(
            verifier, model_response, database_id, context, gym_name=verifier.gym_name
        )
        collapsed[label] = result
        indexed[verifier_key(verifier_config, index)] = result

    return IndexedVerificationReport(
        collapsed_results=collapsed,
        indexed_results=indexed,
        collapsed_summary=summarize(collapsed),
        indexed_summary=summarize(indexed),
    )


class IndexedVerifierExecutor(BenchmarkExecutor):
    """``BenchmarkExecutor`` that also records index-keyed verifier results.

    The override runs inside ``execute_single_run``, i.e. after the agent has
    finished and *before* ``execute_benchmark``'s ``finally`` block deletes the
    seeded database. Nothing in ``benchmark/executor.py`` is modified.
    """

    async def execute_single_run(self, run_number: int) -> Dict[str, Any]:
        """Run one benchmark run and attach the corrected verifier view."""
        result = await super().execute_single_run(run_number)
        try:
            report = await run_verifiers_indexed(
                self.verifier_engine,
                self.config.verifiers or [],
                {
                    "final_response": result.get("model_response"),
                    "tool_results": result.get("tool_results", []) or [],
                },
                self.gym_configs or [],
                default_database_id=self.config.database_id,
                default_context=self.config.context,
            )
        except Exception as exc:  # noqa: BLE001 - never lose an otherwise valid run
            result["verification_indexed"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "collapsed_summary": {},
                "indexed_summary": {},
                "lost_to_collision": 0,
                "success_differs": False,
            }
            return result

        result["verification_indexed"] = {
            **report.as_dict(),
            "results_indexed": report.indexed_results,
        }
        return result
