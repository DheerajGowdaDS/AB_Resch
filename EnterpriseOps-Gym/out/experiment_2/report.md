# Experiment 1 - csm_env state-model ablation (CSM, EnterpriseOps-Gym)

Generated: 2026-09-29T16:52:40.862056+00:00

## Headline


| Condition | Arm | Runs | Task Success Rate | Verifier Pass Rate | Error Rate |
|---|---|---|---|---|---|
| A_baseline | native agent | 3 | 66.7% | 66.7% | 0.0% |
| B1_state_model_broad | + csm_env broad state model | 3 | 0.0% | 33.3% | 0.0% |
| B2_state_model_minimal | + csm_env minimal state model | 1 | 100.0% | 100.0% | 0.0% |

- Baseline: A_baseline; headline treatment: B1_state_model_broad; arms: A_baseline, B1_state_model_broad, B2_state_model_minimal
- ΔTSR (treatment - baseline): -66.7%
- ΔVPR: -33.3%

## Pairwise comparisons

| Comparison | Pairs | ΔTSR | McNemar p | Delivery |
|---|---|---|---|---|
| A_baseline vs B1_state_model_broad | 3 | -66.7% | 0.5 | clean |
| A_baseline vs B2_state_model_minimal | 1 | 100.0% | 1 | clean |
| B1_state_model_broad vs B2_state_model_minimal | 1 | 100.0% | 1 | clean |

## Paired result

- Causal effect estimate ready: True
- Validity note: Every state-model run delivered a usable state intervention and paired analysis is eligible.
- Paired delta TSR: -66.7%
- 95% CI: [-100.0%, 0.0%] (seed 0, 10000 resamples, n=3)
- McNemar p-value: 0.5 (exact_binomial, 2 discordant pairs)
- Pairs: both pass 0, A only 2, treatment only 0, both fail 1, excluded 0

## Initial-state pairing gate (REQ-010)

- Pairs reviewed: 4
- Pairs matched: 4
- Pairs flagged and excluded: 0

## State injection delivery (B1_state_model_broad)

- Runs with the state block injected: 3/3 (100.0%)
- Delivery failures: 0
- Causal effect estimate ready: True
- Empty contexts: 0
- Runs without state telemetry: 0
- Unresolved anchors: 0
- Failed routes: 0
- Mean retrieved records: 32.0
- Mean retrieved tokens: 1000

## State injection delivery (B2_state_model_minimal)

- Runs with the state block injected: 1/1 (100.0%)
- Delivery failures: 0
- Causal effect estimate ready: True
- Empty contexts: 0
- Runs without state telemetry: 0
- Unresolved anchors: 0
- Failed routes: 0
- Mean retrieved records: 4.0
- Mean retrieved tokens: 388

## By state-dependence category (exploratory)

| Category | A_baseline n | A_baseline TSR | A_baseline VPR | B1_state_model_broad n | B1_state_model_broad TSR | B1_state_model_broad VPR | B2_state_model_minimal n | B2_state_model_minimal TSR | B2_state_model_minimal VPR |
|---|---|---|---|---|---|---|---|---|---|
| TYPE_1_SINGLE_ENTITY | 0 | n/a | n/a | 0 | n/a | n/a | 0 | n/a | n/a |
| TYPE_2_ONE_HOP | 0 | n/a | n/a | 0 | n/a | n/a | 0 | n/a | n/a |
| TYPE_3_MULTI_HOP | 0 | n/a | n/a | 0 | n/a | n/a | 0 | n/a | n/a |
| TYPE_4_CROSS_ENTITY | 3 | 66.7% | 66.7% | 3 | 0.0% | 33.3% | 1 | 100.0% | 100.0% |
| TYPE_5_MULTI_STEP | 0 | n/a | n/a | 0 | n/a | n/a | 0 | n/a | n/a |

## Verifier reconciliation (duplicate-name collision)

- Collapsed verifier results recorded: 21
- Index-keyed verifier results: 77
- Lost to name collision: 56
- Runs where the collision changed the pass count: 7

## State retrieval quality (B1_state_model_broad, post-hoc)

- Granularity: row
- Mean precision: 0.031
- Mean recall: 0.050
- Runs measured: 3 (excluded 0)

## Execution performance

- Goals: 99 total, 7 completed, 92 failed, 0 skipped
- Wall clock: 756.3 s at concurrency 1
- Throughput: 0.56 goals/min

| Stage | Mean ms | Total ms | Count |
|---|---|---|---|
| agent | 108043.6 | 756305.1 | 7 |

## Notes

- Primary outcome is task success (all verifiers pass), the benchmark's own definition.
- Stratified tables are exploratory; the pre-registered comparison is the paired one.
- Pooled verifier pass rate is a companion metric, not the official VPR.
- State-model arms are fail-closed: runs without a usable state intervention are execution errors, not baseline fallbacks.
- The causal effect is reported only when the state intervention is delivered on every analyzed state-model run.
- τ is pre-registered per task type in the design block. Measured recall (prompt-entity relevance) and required-fact coverage (τ) use different denominators: flat measured recall does not imply zero required-fact loss.
