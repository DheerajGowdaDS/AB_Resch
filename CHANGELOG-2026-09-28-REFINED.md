# Refined Experiment 1 package — 2026-09-28

This package is the refined pre-experiment baseline for the CSM state-model ablation.

## Research-validity fixes

- Experiment-1 retrieval-quality evaluation is now explicitly **initial-state** and is derived from the task user prompt plus the pinned seed SQL. It no longer compares pre-action state to final-state verifier predicates.
- Final-state verifier-derived `GoldState` remains available only for separate audit analysis.
- Retrieval diagnostics distinguish benchmark task success from state delivery and prompt-entity retrieval quality.
- The horizon recommender refuses to extrapolate from runs that all terminated at their own ceiling; no artificial `50`-step recommendation is produced from the old `5`-step pilot.
- The common experiment step budget is provisionally `15`; the primary ceiling is not considered validated until a fresh horizon-calibration run establishes an acceptable censoring rate.
- New run records are stamped `V1.2-STATE-MODEL-REFINED` and persist the version fields for auditability.

## Historical-artifact hygiene

The pre-fix 22-run pilot is retained only under `out/archive/V1-PREFIX-PILOT-2026-09-28/`.
`out/experiment_1/` is clean and reserved for the next post-fix calibration sweep.

## Validation

- `csm_env/tests`: 111 passed, 3 skipped in the source validation environment.
- Ablation harness tests: 200 passed in the validation environment using a lightweight local message stub because the container does not have the benchmark's external `langchain_core` dependency installed.
- No transition-model or Condition-C machinery was added.
