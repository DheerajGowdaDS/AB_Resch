# Experiment 1 - csm_env state-model ablation (CSM, EnterpriseOps-Gym)

Generated: 2026-09-28T06:39:06.436214+00:00

## Headline

| Condition | Arm | Runs | Task Success Rate | Verifier Pass Rate | Error Rate |
|---|---|---|---|---|---|
| A_baseline | native agent | 11 | 0.0% | 39.8% | 0.0% |
| B_state_model | + csm_env state model | 11 | 0.0% | 34.6% | 36.4% |
| Difference | B - A | - | n/a | n/a | n/a |

## Paired result

- Causal effect estimate ready: False
- Validity note: Condition B was not delivered on every run; effect size is withheld.
- Paired delta TSR: 0.0%
- 95% CI: [0.0%, 0.0%] (seed 0, 10000 resamples, n=7)
- McNemar p-value: 1 (none, 0 discordant pairs)
- Pairs: both pass 0, A only 0, B only 0, both fail 7, excluded 4

## Initial-state pairing gate (REQ-010)

- Pairs reviewed: 11
- Pairs matched: 11
- Pairs flagged and excluded: 0

## State injection delivery (treatment arm)

- Runs with the state block injected: 7/11 (63.6%)
- Delivery failures: 4
- Causal effect estimate ready: False
- Empty contexts: 1
- Runs without state telemetry: 0
- Unresolved anchors: 1
- Failed routes: 1
- Mean retrieved records: 22.4
- Mean retrieved tokens: 907

## By state-dependence category (exploratory)

| Category | A_baseline n | A_baseline TSR | A_baseline VPR | B_state_model n | B_state_model TSR | B_state_model VPR |
|---|---|---|---|---|---|---|
| TYPE_1_SINGLE_ENTITY | 1 | 0.0% | 50.0% | 1 | 0.0% | 75.0% |
| TYPE_2_ONE_HOP | 0 | n/a | n/a | 0 | n/a | n/a |
| TYPE_3_MULTI_HOP | 1 | 0.0% | 57.1% | 1 | 0.0% | 57.1% |
| TYPE_4_CROSS_ENTITY | 4 | 0.0% | 34.4% | 3 | 0.0% | 20.0% |
| TYPE_5_MULTI_STEP | 5 | 0.0% | 38.7% | 2 | 0.0% | 25.0% |

## Verifier reconciliation (duplicate-name collision)

- Collapsed verifier results recorded: 114
- Index-keyed verifier results: 159
- Lost to name collision: 45
- Runs where the collision changed the pass count: 5

## State retrieval quality (B_state_model, post-hoc)

- Granularity: table
- Mean precision: 0.607
- Mean recall: 0.431
- Runs measured: 7 (excluded 4)

## Execution performance

- Goals: 22 total, 22 completed, 0 failed, 0 skipped
- Wall clock: 1326.7 s at concurrency 1
- Throughput: 0.99 goals/min

| Stage | Mean ms | Total ms | Count |
|---|---|---|---|
| agent | 60305.3 | 1326716.4 | 22 |

## Notes

- Primary outcome is task success (all verifiers pass), the benchmark's own definition.
- Stratified tables are exploratory; the pre-registered comparison is the paired one.
- Pooled verifier pass rate is a companion metric, not the official VPR.
- Condition B is fail-closed: runs without a usable state intervention are execution errors, not baseline fallbacks.
- The Experiment-1 causal effect is reported only when the state intervention is delivered on every analyzed B run.
