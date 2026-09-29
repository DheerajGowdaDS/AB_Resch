# Experiment 1 - csm_env state-model ablation (CSM, EnterpriseOps-Gym)

Generated: 2026-09-28T14:46:53.649358+00:00

## Headline

| Condition | Arm | Runs | Task Success Rate | Verifier Pass Rate | Error Rate |
|---|---|---|---|---|---|
| A_baseline | native agent | 1 | 100.0% | 100.0% | 0.0% |
| B_state_model | + csm_env state model | 1 | 100.0% | 100.0% | 0.0% |
| Difference | B - A | - | 0.0% | 0.0% | 0.0% |

## Paired result

- Causal effect estimate ready: True
- Validity note: All Condition-B runs delivered a usable state intervention and paired analysis is eligible.
- Paired delta TSR: 0.0%
- 95% CI: [0.0%, 0.0%] (seed 0, 10000 resamples, n=1)
- McNemar p-value: 1 (none, 0 discordant pairs)
- Pairs: both pass 1, A only 0, B only 0, both fail 0, excluded 0

## Initial-state pairing gate (REQ-010)

- Pairs reviewed: 1
- Pairs matched: 1
- Pairs flagged and excluded: 0

## State injection delivery (treatment arm)

- Runs with the state block injected: 1/1 (100.0%)
- Delivery failures: 0
- Causal effect estimate ready: True
- Empty contexts: 0
- Runs without state telemetry: 0
- Unresolved anchors: 0
- Failed routes: 0
- Mean retrieved records: 9.0
- Mean retrieved tokens: 619

## By state-dependence category (exploratory)

| Category | A_baseline n | A_baseline TSR | A_baseline VPR | B_state_model n | B_state_model TSR | B_state_model VPR |
|---|---|---|---|---|---|---|
| TYPE_1_SINGLE_ENTITY | 1 | 100.0% | 100.0% | 1 | 100.0% | 100.0% |
| TYPE_2_ONE_HOP | 0 | n/a | n/a | 0 | n/a | n/a |
| TYPE_3_MULTI_HOP | 0 | n/a | n/a | 0 | n/a | n/a |
| TYPE_4_CROSS_ENTITY | 0 | n/a | n/a | 0 | n/a | n/a |
| TYPE_5_MULTI_STEP | 0 | n/a | n/a | 0 | n/a | n/a |

## Verifier reconciliation (duplicate-name collision)

- Collapsed verifier results recorded: 8
- Index-keyed verifier results: 8
- Lost to name collision: 0
- Runs where the collision changed the pass count: 0

## State retrieval quality (B_state_model, post-hoc)

- Granularity: row
- Mean precision: 0.111
- Mean recall: 0.062
- Runs measured: 1 (excluded 0)

## Execution performance

- Goals: 2 total, 2 completed, 0 failed, 0 skipped
- Wall clock: 112.5 s at concurrency 1
- Throughput: 1.07 goals/min

| Stage | Mean ms | Total ms | Count |
|---|---|---|---|
| agent | 56258.8 | 112517.7 | 2 |

## Notes

- Primary outcome is task success (all verifiers pass), the benchmark's own definition.
- Stratified tables are exploratory; the pre-registered comparison is the paired one.
- Pooled verifier pass rate is a companion metric, not the official VPR.
- Condition B is fail-closed: runs without a usable state intervention are execution errors, not baseline fallbacks.
- The Experiment-1 causal effect is reported only when the state intervention is delivered on every analyzed B run.
