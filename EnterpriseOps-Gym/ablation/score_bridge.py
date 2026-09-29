"""Bridge to the benchmark's official scorer.

``compute_score.py`` reads ``statistics.overall_success_rate`` and
``statistics.verifier_level_pass_rate`` from every ``*.json`` under a folder, and
treats each immediate subfolder as a "mode". This module reshapes the harness
records into exactly that layout, one folder per condition and one file per task,
so the official scorer can reproduce the headline numbers independently.

That reconciliation is the point: nothing is reported until the official tool
agrees with the harness (REQ-015, GUD-003).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from .results import TaskRunRecord, load_run_records
from .stats import agent_error_rate, tsr, vpr, verifier_level_pass_rate_pooled

PathLike = Union[str, Path]

#: Folder ``compute_score.py`` expects, since it scans ``run_*`` subfolders.
SCORE_RUN_DIRNAME = "run_1"


def file_statistics(records: Sequence[TaskRunRecord]) -> Dict[str, Any]:
    """Statistics block for a single ``results_*.json`` file.

    The two headline fields use the official definitions: the all-verifiers-passed
    indicator and the per-file verifier pass rate.
    """
    clean = [record for record in records if not record.agent_error]
    total_verifiers = sum(
        int((record.verifier_summary_collapsed or {}).get("total", 0) or 0) for record in clean
    )
    passed_verifiers = sum(
        int((record.verifier_summary_collapsed or {}).get("passed", 0) or 0) for record in clean
    )
    return {
        "total_runs": len(records),
        "successful_runs": sum(1 for record in clean if record.overall_success),
        "overall_success_rate": tsr(records),
        "verifier_level_pass_rate": vpr(records),
        "total_verifiers_checked": total_verifiers,
        "total_verifiers_passed": passed_verifiers,
        # Labelled companion values; compute_score.py ignores these keys.
        "harness_vpr_pooled_non_official": verifier_level_pass_rate_pooled(records),
        "harness_agent_error_rate": agent_error_rate(records),
    }


def run_payload(records: Sequence[TaskRunRecord]) -> Dict[str, Any]:
    """One ``results_*.json`` document: the ``runs`` array plus statistics."""
    return {
        "runs": [
            {
                "run_number": record.run_index,
                "overall_success": bool(record.overall_success),
                "error": record.error_message,
                "task_id": record.task_id,
                "condition": record.condition,
                "run_uid": record.run_uid,
                "execution_time_ms": record.latency_ms,
                "verification_summary": record.verifier_summary_collapsed,
                "verification_results": record.verifier_results_collapsed,
            }
            for record in records
        ],
        "statistics": file_statistics(records),
    }


def emit_compute_score_layout(
    records: Sequence[TaskRunRecord], output_dir: PathLike
) -> Dict[str, Path]:
    """Write a ``compute_score.py``-compatible tree and return the written paths.

    Args:
        records: Harness records, possibly spanning several conditions.
        output_dir: Destination root. One subfolder per condition is created,
            each holding ``run_1/results_<task_id>.json``.

    Returns:
        Mapping of condition name to the directory written for it.
    """
    root = Path(output_dir)
    by_condition: Dict[str, List[TaskRunRecord]] = {}
    for record in records:
        by_condition.setdefault(record.condition, []).append(record)

    written: Dict[str, Path] = {}
    for condition, subset in sorted(by_condition.items()):
        run_dir = root / condition / SCORE_RUN_DIRNAME
        run_dir.mkdir(parents=True, exist_ok=True)
        by_task: Dict[str, List[TaskRunRecord]] = {}
        for record in subset:
            by_task.setdefault(record.task_id, []).append(record)
        for task_id, task_records in sorted(by_task.items()):
            target = run_dir / f"results_{task_id}.json"
            target.write_text(
                json.dumps(run_payload(task_records), indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
        written[condition] = run_dir
    return written


def emit_from_directory(records_dir: PathLike, output_dir: PathLike) -> Dict[str, Path]:
    """Load records from a harness output folder and emit the score layout."""
    return emit_compute_score_layout(load_run_records(records_dir), output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m ablation.score_bridge --records-dir ... --output-dir ...``."""
    parser = argparse.ArgumentParser(
        prog="ablation.score_bridge",
        description="Reshape harness records into the compute_score.py layout.",
    )
    parser.add_argument("--records-dir", required=True, help="Harness output folder")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination; defaults to <records-dir>/compute_score",
    )
    args = parser.parse_args(argv)
    records_dir = Path(args.records_dir)
    output_dir = Path(args.output_dir) if args.output_dir else records_dir / "compute_score"
    records = load_run_records(records_dir)
    if not records:
        print(f"no records found in {records_dir}")
        return 1
    written = emit_from_directory(records_dir, output_dir)
    for condition, path in written.items():
        print(f"{condition}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
