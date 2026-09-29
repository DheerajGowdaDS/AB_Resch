---
goal: Experiment 1 - paired ablation of the csm_env state model on EnterpriseOps-Gym CSM tasks
version: 1.3
date_created: 2026-09-26
last_updated: 2026-09-29
owner: Research engineering
status: 'Pre-registered (sweep not yet executed)'
implements: plan/feature-ablation-harness-1.md
---

# Experiment 1 protocol - does the graph-grounded state model help?

> **Revision 1.3 (State Model 2.0, cost objective).** The *frozen intervention*
> for arm `B2` changed: it now optimizes the literal objective
> `Cost(S) = λ_r·|rows| + λ_t·|tokens| + λ_q·|queries|` subject to
> `Coverage(S,F_T) ≥ τ` (blueprint §2). Concretely, `B2` routes with
> `RouteStrategy.FRONTIER`, which retrieves one relation at a time and **stops
> at the first prefix that meets coverage**, so `λ_q·|queries|` is genuinely
> minimized instead of being decorative. `B1` keeps the eager strategy, pinning
> its retrieval cost to State Model 1.0, so the two arms differ only in
> *selection*.
>
> The objective's weights are **engineering defaults, fixed in advance** and
> externalized in `csm_env.query.cost.CostWeights`
> (`λ_r = 1.0`, `λ_t = 0.05`, `λ_q = 8.0`). `τ` defaults to `1.0` and is exposed
> as `QueryBudget.coverage_threshold`. Every route records
> `diagnostics['cost']`, `diagnostics['cost_prefixes']` and
> `diagnostics['coverage']`, so the achieved coverage and the realized query
> count are auditable per run.
>
> No sweep has run under any registration, so no historical result is
> invalidated; archived `B_state_model` records keep their original broad
> meaning.

> **Revision 1.2 (State Model 2.0).** The condition set was re-registered from
> two arms to three before any sweep was executed: the original broad state
> model is retained as `B1`, and a task-conditioned *minimal* arm (`B2`) is
> added. Archived `B_state_model` records keep their original broad meaning.

This document is the pre-registration for Experiment 1. It is written **before**
any sweep is executed, so the hypothesis, constants, metrics, and analysis plan
are fixed in advance. Machine-readable revisions live in
`EnterpriseOps-Gym/ablation/manifests/pinned_revisions.json`.

## 1. Research question and hypothesis

> **Does the current graph-grounded environment/state model improve LLM-agent
> performance on EnterpriseOps-Gym CSM tasks?**

- `H1: TSR_stateModel > TSR_baseline` (directional; the experiment tests it, it is
  not assumed).
- `H2 (added in revision 1.3): TSR_B2 > TSR_B1` — i.e. optimizing the stated
  objective, rather than only pruning after the fact, is what improves the arm.
- The frozen intervention is the current `csm_env` pipeline:
  `SchemaGraph → GGQR → bounded retrieval → GroundedState → relations + facts + provenance`.
  No predicted future, no action simulation, no learned transition function, no
  planning search.
- **Experiment 2 is out of scope.** `C = B + transition model` may only be
  designed after this experiment yields a measured `ΔTSR`.

## 2. Conditions

| Condition | Arm | What the agent gets |
|---|---|---|
| `A_baseline` | native agent | task → LLM → native CSM MCP tools → environment |
| `B1_state_model_broad` | intervention (broad) | same, plus one labelled grounded-state block: the full graph-connected retrieval. `RouteStrategy.BROAD` — every planned query is executed, exactly as in State Model 1.0 |
| `B2_state_model_minimal` | intervention (minimal) | same, plus the task-conditioned *minimal* state: `RouteStrategy.FRONTIER` (stop at the first prefix meeting `Coverage ≥ τ`, minimizing `λ_q·|queries|`), then relevance ranking, greedy weighted set cover, attribute pruning, and coverage/token gates |

`B2_state_model_minimal` is the State Model 2.0 arm; `B1_state_model_broad`
preserves the original intervention so the B1-vs-B2 contrast isolates the
*selection* mechanism, and — since revision 1.3 — the **retrieval** mechanism as
well (`λ_q` is only optimizable if the query count is a decision). The
retrieval cost of `B1` is pinned to State Model 1.0 so it is never the
confound. The CLI accepts `A,B1,B2` (default), `A,B1`, `A,B2`,
and the historical names (`B`, `B_state_model` → B1). The CLI refuses
condition `C` by name (`ablation.conditions.FORBIDDEN_CONDITION_NAMES`).

Pairwise comparisons are pre-registered as A vs B1, A vs B2, and B1 vs B2.

## 3. Constants held fixed

| Variable | Value | How it is held fixed |
|---|---|---|
| Task | same CSM task file | both arms run from one `TaskRecord` |
| Initial database | same seed SQL | benchmark re-seeds per run; `initial_state_fingerprint` compared per pair |
| LLM / temperature | one `LLMConfig` per sweep | single config file, first entry used |
| Orchestrator | `react` | CLI rejects anything but `react` for Experiment 1 |
| Tool mode | `oracle` | the task's own `selected_tools`, used verbatim |
| Step budget | identical `max_iterations` | forwarded explicitly to both arms (`--max-steps`, default 15; provisional until empirical horizon calibration validates the common ceiling); identical by construction, not by default coincidence |
| Native tools | identical | neither arm adds, removes or rewrites a tool |
| System prompt | identical | the ReAct loop is the upstream implementation, unmodified |
| Verifiers | identical | the same `VerifierConfig` list runs in both arms |
| Repeats | `--num-runs` (default 3) | the harness owns repetition, not the task file |
| State context policy | `pre_action_once` | per-step refresh is deferred |

## 4. Fairness rule (SEC-001)

The Condition B adapter may read the database, the task's `user_prompt`, and the
task's `selected_tools`. It may **not** read `verifiers`, `validation_config`,
`query`, `expected_value`, `comparison_type`, or verifier names/descriptions.

Enforcement is structural, not documentary:

- `ablation.state_model.assert_no_verifier_metadata` rejects any payload carrying
  a forbidden key or attribute (instance **or** class level).
- The adapter's constructor takes a `TaskRecord`, never a task-config mapping.
- `ablation.relevance` (final-state verifier-derived audit analysis only) is unreachable from the Condition B import chain, asserted by
  `tests/test_ablation_boundaries.py::test_harness_does_not_import_verifier_derived_state`.

Anchors are resolved from prompt mentions against the live database
(`StateModelAdapter.resolve_anchor`), so the state model only ever sees
information obtainable from the environment.

## 5. Metrics

| Metric | Definition | Role |
|---|---|---|
| TSR | fraction of runs where **all** verifiers pass | primary, pre-registered |
| VPR | mean per-run verifier pass rate (`compute_score.py` semantics) | secondary |
| Agent error rate | fraction of runs with an agent error | reliability |
| ΔTSR | `TSR_B − TSR_A` | headline effect |
| 95% CI | paired percentile bootstrap, seeded | uncertainty |
| p-value | McNemar: exact binomial below 25 discordant pairs, else continuity-corrected χ² | significance |

Secondary telemetry: steps, tool calls, read calls, invalid actions, latency,
input/output tokens, and (Condition B) retrieved records, retrieved tokens, route
status, anchor status, and initial-state prompt/seed-derived retrieval precision/recall.

Stratification by state dependence (`TYPE_1_SINGLE_ENTITY` … `TYPE_5_MULTI_STEP`)
is **exploratory** and reported separately from the pre-registered comparison.

## 6. Evaluation set (as built)

Derived from the real dataset by `python run_ablation.py --init-manifests`.

- Source: `EnterpriseOps-Gym/data/revised/csm` (**12** task files).
- Published CSM task count: **186**. Recorded shortfall: **174**. A 12-task
  sweep is a **pilot**, not a confirmation of an effect size.
- Categories in the effective sweep: `TYPE_1_SINGLE_ENTITY` 1, `TYPE_2_ONE_HOP` 0,
  `TYPE_3_MULTI_HOP` 1, `TYPE_4_CROSS_ENTITY` 4, `TYPE_5_MULTI_STEP` 5.
- The source folder contains 12 task files; one is excluded because its seed SQL
  is absent from `gym_dbs.zip`, leaving 11 effective tasks. Categories are derived
  only from the prompt and tool list; `TYPE_2` is empty and is reported as `n/a`.
- One task (`task_20251209_132736_542_99ba2325_6705caf4`) references a seed SQL
  absent from `gym_dbs.zip`; it is excluded with an explicit reason unless
  `--allow-missing-seeds` is passed. Effective sweep size: **11 tasks**.

## 7. Known evaluator defect and its reconciliation

`BenchmarkExecutor._run_verifiers` keys results by `verifier.name`
(`benchmark/executor.py:474`), so verifiers sharing a name overwrite each other
(upstream issue #23). **4 of the 12 pinned tasks are affected** (3, 1, 5 and 9
duplicated names; 100 verifiers declared in total).

The harness does **not** patch the benchmark. It re-runs the unchanged
`VerifierEngine` with positional keys, inside the run, while the seeded database
is still alive, and reports both views:

- `verifier_summary_collapsed` - the official name-keyed summary (headline metric);
- `verifier_summary_indexed` - the collision-free summary (reconciliation).

Every report includes a reconciliation section stating how many verifier results
were lost to the collision and in how many runs that changed the pass count.

## 8. Pre-flight gate

A sweep may not start until every blocking check passes. The pairing gate is
enforced at report time by `ablation.runner.reconcile_fingerprint_pairs`: a pair
whose fingerprints are absent or unequal is flagged as an agent error on both
records and excluded from every metric, matching `run_pair`'s execution-time
behavior (REQ-010).

Blocking:
`csm_env_schema`, `csm_env_parity`, `eval_set`, `seed_archive`. Reported but
non-blocking: `verifier_sets` (reconciled rather than fatal), `llm_config` and
`csm_server` (supplied at run time).

`csm_env_parity` compares the two `csm_env` copies by SHA-256 and fails closed on
divergence, because running from the gym resolves the gym-local copy: a stale copy
would mean Condition B is not the frozen intervention.

Two fingerprint properties are worth recording. First, the digest proves
*structural* equality (tables, row counts, primary keys); it is not a full
content hash, which is acceptable because both arms seed from the same pinned
SQL file. Second, a systematically failing fingerprint probe is a correctness
risk: `--strict-fingerprint` makes probe failures raise instead of degrading to
`None`, and the report's pairing-gate section counts any pair left unverifiable.

## 9. Initial-state retrieval quality

Experiment 1 injects state before any agent action. Retrieval-quality precision
and recall therefore use an evaluator-only initial-state relevance set built from
the task user prompt and the pinned seed SQL. They do **not** compare pre-action
state against final-state verifier predicates. The verifier-derived ``GoldState``
module is retained only for separate final-state audit analysis and is never fed
into Condition B.

## 10. Reproduction

Every reported number is recomputed from persisted records only, with no live
server:

```powershell
Set-Location 'D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym'

# 1. Rebuild manifests from the real dataset and schema
python run_ablation.py --init-manifests

# 2. Offline readiness gate and run matrix (no server, no LLM, no writes)
python run_ablation.py --dry-run --num-runs 3 --allow-missing-seeds --output-folder out/experiment_1

# 3. Live anchor resolution and initial-state fingerprints (needs the CSM server)
python run_ablation.py --resolve-anchors --base-url http://localhost:8001

# 4. The sweep (needs the CSM server and an LLM config)
python run_ablation.py --conditions A,B --num-runs 3 --llm-config conf/llm/my-model.json --output-folder out/experiment_1

# 5. Regenerate the report from records only
python run_ablation.py --report-only --output-folder out/experiment_1

# 6. Reconcile against the official scorer
python -m ablation.score_bridge --records-dir out/experiment_1 --output-dir out/experiment_1/compute_score
python compute_score.py --results_folder out/experiment_1/compute_score
```

Records are one JSON file per `(task, condition, run)` plus an append-only
`index.jsonl`, so an interrupted sweep resumes without recomputation and
already-recorded goals are skipped.

## 11. Decision gate

1. If `ΔTSR` is measurable in either direction, Experiment 1 is complete and the
   Experiment 2 design may be drafted as a separate plan.
2. If any pair fails its `initial_state_fingerprint` check, that pair is marked
   unusable and must be re-run before any conclusion.
3. If the sweep is under-powered (as an 11-task pilot is), report it as a pilot
   and expand from a pinned Hugging Face revision before claiming an effect.

## 12. Current status

The harness is implemented and its offline suite passes. The REQ-010 pairing
gate now runs at report time (`reconcile_fingerprint_pairs`), the step budget is
forwarded explicitly to both arms, and the report carries fingerprint-pairing
and state-injection-delivery summaries. **No post-fix LLM calibration sweep has been executed in this workspace**: the historical pilot
records are archived separately and must not be pooled with the refined
intervention. A live post-fix sweep requires the Docker CSM server and an LLM credential.
