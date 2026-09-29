# Experiment 1 - csm_env state-model ablation (CSM, EnterpriseOps-Gym)

Generated: 2026-09-28T16:30:07.472033+00:00

## Headline

| Condition | Arm | Runs | Task Success Rate | Verifier Pass Rate | Error Rate |
|---|---|---|---|---|---|
| A_baseline | native agent | 33 | 30.3% | 57.5% | 0.0% |
| B_state_model | + csm_env state model | 33 | 30.3% | 51.4% | 0.0% |
| Difference | B - A | - | 0.0% | -6.2% | 0.0% |

## Paired result

- Causal effect estimate ready: True
- Validity note: All Condition-B runs delivered a usable state intervention and paired analysis is eligible.
- Paired delta TSR: 0.0%
- 95% CI: [-12.1%, 12.1%] (seed 0, 10000 resamples, n=33)
- McNemar p-value: 1 (exact_binomial, 4 discordant pairs)
- Pairs: both pass 8, A only 2, B only 2, both fail 21, excluded 0

## Initial-state pairing gate (REQ-010)

- Pairs reviewed: 33
- Pairs matched: 33
- Pairs flagged and excluded: 0

## State injection delivery (treatment arm)

- Runs with the state block injected: 33/33 (100.0%)
- Delivery failures: 0
- Causal effect estimate ready: True
- Empty contexts: 0
- Runs without state telemetry: 0
- Unresolved anchors: 0
- Failed routes: 0
- Mean retrieved records: 28.3
- Mean retrieved tokens: 893

## By state-dependence category (exploratory)

| Category | A_baseline n | A_baseline TSR | A_baseline VPR | B_state_model n | B_state_model TSR | B_state_model VPR |
|---|---|---|---|---|---|---|
| TYPE_1_SINGLE_ENTITY | 3 | 66.7% | 83.3% | 3 | 66.7% | 91.7% |
| TYPE_2_ONE_HOP | 0 | n/a | n/a | 0 | n/a | n/a |
| TYPE_3_MULTI_HOP | 3 | 66.7% | 76.2% | 3 | 100.0% | 100.0% |
| TYPE_4_CROSS_ENTITY | 12 | 25.0% | 49.4% | 12 | 25.0% | 42.8% |
| TYPE_5_MULTI_STEP | 15 | 20.0% | 55.1% | 15 | 13.3% | 40.4% |

## Verifier reconciliation (duplicate-name collision)

- Collapsed verifier results recorded: 438
- Index-keyed verifier results: 576
- Lost to name collision: 138
- Runs where the collision changed the pass count: 18

## State retrieval quality (B_state_model, post-hoc)

- Granularity: row
- Mean precision: 0.088
- Mean recall: 0.184
- Runs measured: 33 (excluded 0)

## Execution performance

- Goals: 66 total, 66 completed, 0 failed, 0 skipped
- Wall clock: 6165.4 s at concurrency 1
- Throughput: 0.64 goals/min

| Stage | Mean ms | Total ms | Count |
|---|---|---|---|
| agent | 93414.0 | 6165325.6 | 66 |

## Notes

- Primary outcome is task success (all verifiers pass), the benchmark's own definition.
- Stratified tables are exploratory; the pre-registered comparison is the paired one.
- Pooled verifier pass rate is a companion metric, not the official VPR.
- Condition B is fail-closed: runs without a usable state intervention are execution errors, not baseline fallbacks.
- The Experiment-1 causal effect is reported only when the state intervention is delivered on every analyzed B run.
