# Experiment 1 — CSM State-Model Ablation

## Purpose

Experiment 1 isolates whether the current graph-grounded environment/state model improves EnterpriseOps-Gym CSM task performance. It does **not** include a transition model, state prediction, or planning search.

## Conditions

- **A_baseline:** frozen EnterpriseOps-Gym ReAct agent with the native CSM MCP tools.
- **B1_state_model_broad:** the same agent and native MCP tools, plus one pre-action grounded-state block produced by `csm_env`/GGQR from the task prompt and live database (the full graph-connected retrieval), using the eager `RouteStrategy.BROAD` — every planned query executed, identical to State Model 1.0.
- **B2_state_model_minimal:** the same intervention with State Model 2.0 applied: `RouteStrategy.FRONTIER` retrieval that **stops at the first prefix meeting `Coverage(S,F_T) ≥ τ`**, then the task-conditioned **minimal** selector before injection — relevance-ranked candidates, greedy weighted set cover, attribute-level pruning, and explicit coverage/token gates.

The historical name `B_state_model` resolves to B1 so archived records keep their meaning. The hidden SQL verifier configuration is never passed to the state model.

### The objective (State Model 2.0, blueprint §2)

B2 minimizes

```text
Cost(S) = λ_r·|rows| + λ_t·|tokens| + λ_q·|queries|
```

subject to `Coverage(S,F_T) ≥ τ`, `Consistency(S,G_s) = 1` and `Provenance(S) = 1`. The weights are engineering defaults, fixed in advance in `csm_env.query.cost.CostWeights` (`λ_r = 1.0`, `λ_t = 0.05`, `λ_q = 8.0`), and `τ` defaults to `1.0` through `QueryBudget.coverage_threshold`.

`λ_q` is only optimizable if the query count is a *decision*, which is why B2 retrieves incrementally (frontier) instead of eagerly. Because `Cost(S)` is non-decreasing in queries, rows and tokens, the first prefix that meets coverage is the `argmin`; the realized saving is reported per run as `diagnostics['queries_saved']`, with the full prefix trace in `diagnostics['cost_prefixes']`. B1's retrieval cost is deliberately pinned to State Model 1.0 so it is never the confound.

### State Model 2.0 selection pipeline (B2)

B2 retrieves with the frontier strategy (above) and then reduces the retrieved
state to the minimal sufficient evidence for the task
(`ablation/minimal_state.py`):

1. every candidate row is scored `Score(r) = w_a·A(r) + w_e·E(r) + w_p·P(r) + w_d·D(r) − w_h·H(r)` with externalized weights, where `D(r)` is a dependency value (sole provider of a required table, or relational bridge carrying a required relation);
2. a greedy weighted set cover keeps the cheapest rows that cover every task requirement (the anchor, each required table, each required attribute, each required relation), so a required fact can no longer be silently dropped by a count cap;
3. kept rows are attribute-pruned to PK + required FKs + task-relevant + identity columns (an `update` intent additionally keeps every FK, as mutable context);
4. the selection must satisfy the coverage gate `Coverage(S, F_T) ≥ τ` and the token budget, or the broad state is delivered instead (fail closed).

Escalation is coverage-driven: the hop ladder continues while a prompt-required table is still unobserved, and stops as soon as coverage is complete.

## Primary and benchmark-native metrics

- **Task Success Rate (TSR):** a task succeeds only when all of its verifiers pass.
- **Verifier Pass Rate (VPR):** passed verifier evaluations divided by all verifier evaluations.
- **Agent Error Rate:** evaluation runs carrying an agent/execution error.

Secondary diagnostics include steps, tool calls, read queries, invalid actions, latency, retrieved records/tokens, and state-delivery status.

## Ablation validity requirements

A paired A/B task instance is eligible for the causal comparison only when:

1. A and B start from the same initial-state fingerprint.
2. Neither arm has an agent/execution error.
3. Condition B actually delivered grounded state: resolved anchor, COMPLETE route, non-empty rendered state, at least one retrieved record, and non-zero injected tokens.

A failed/unresolved B intervention is an execution failure, **not** a silent fallback to A.

The report withholds `delta_tsr` and `delta_vpr` until every analyzed B run satisfies these delivery requirements.

## Why this gate exists

The superseded 11-task pilot showed a higher raw B TSR, but state delivery succeeded on only part of the B runs and the single B-only success had no injected state. That run therefore cannot establish a state-model causal effect. This release fixes the delivery path and makes the statistical report refuse to present a causal effect when the intervention was not actually delivered.

## Evaluation-set hygiene

The persisted `ablation/manifests/csm_eval_set.json` contains only tasks whose seed SQL is present in the bundled `gym_dbs.zip`. This prevents a missing seed from becoming a false task failure in the primary experiment. Re-running `--init-manifests` applies the same filter and records the exclusion in the console.

## Reproduction

From the `EnterpriseOps-Gym` directory:

```bash
python run_ablation.py --dry-run --num-runs 1 --allow-missing-seeds
python run_ablation.py --conditions A,B1,B2 --num-runs 1 --llm-config conf/llm/<model>.json
python run_ablation.py --diagnose --output-folder out/experiment_1
python run_ablation.py --report-only --output-folder out/experiment_1
```

`--conditions A,B1` reproduces the original two-arm design; the default is the
three-arm State Model 2.0 design. `--diagnose` reports Phase 2 delivery,
precision and recall for the minimal arm when present (falling back to the only
state-model arm), and writes a companion report for every other state-model arm.

Pin the exact task files, benchmark files, seed archive, and ablation-code hashes in `ablation/manifests/pinned_revisions.json` for each reported experiment.

## Live delivery probe (`validate_delivery_live.py`)

The Phase 1.1 acceptance test runs against the **real** CSM server and the **real**
seed databases, but **without** an LLM. It isolates the expensive, fragile part of
the experiment — can the state model resolve an anchor, route to its required
tables, and hand the agent usable state? — from agent behaviour.

```bash
python validate_delivery_live.py                    # all tasks
python validate_delivery_live.py --task-ids <id>    # one task
```

It seeds each database, fingerprints it, resolves the anchor, builds the state
through the normal GGQR pipeline, applies the same delivery rule the orchestrator
enforces, and deletes every database it created. Exit code is 0 only at 100%
delivery, so it is safe to use as a CI gate. It costs no tokens.

### Verified result

`DeliveryRate = 11/11 = 100%` against the live server, up from 7/11 in the pilot.

Three defects were found and fixed by this probe that no offline test could have
surfaced, because they only appear against real data:

| Defect | Symptom | Fix |
|---|---|---|
| Unescaped apostrophe | `400 Bad Request`, anchor unresolved | `escape_like_pattern` doubles `'` |
| Empty required table treated as a failure | 3 runs scored undeliverable | structural validity separated from row coverage |
| Lookup budget exhausted by the exact tier | 2 anchors unresolved | per-tier budget reservation |

### Budget discipline (Phase 1.2)

The live trace showed the original resolver sweeping all 17 tables for candidate 1,
then all 17 for candidate 2 — `48 = 17 + 17 + 14` lookups, every one an *exact*
match, exhausting the budget before the semantic tier could run. Three rules
follow directly from that evidence:

* **Per-tier reservation.** Each tier gets a reserved share of the budget
  (`identifier` 25%, `name_match` 30%, `semantic_lookup` 45%). Without it, tasks
  whose entities must still be *created* could never resolve, because only a
  substring match could work.
* **Only probe tables that can match.** A name probe skips tables with no identity
  column (`entitlement`, `interaction`, `contact`) instead of querying and missing.
  Identity matching is token-based, so `entitlement_id` does not match `title`
  inside "en**title**ment".
* **Bounded fan-out.** At most `MAX_TABLES_PER_CANDIDATE` tables per candidate,
  set so every identity-capable CSM table fits, so budget is never the reason a
  resolvable anchor is missed.

Possessive compounds are decomposed (`Wayne Enterprises' Windows Server` →
`Wayne Enterprises` + `Windows Server`), because the whole phrase matches no stored
value while both halves match exactly. Multi-word names are deliberately *not*
shattered, since `Windows Server` is itself a useful substring of
`Windows Server 2022 Datacenter`.

## Validity gates (`--diagnose`)

`--diagnose` recomputes three gates from persisted records. It needs no server and no
LLM, and exits non-zero when any gate fails, so it is safe to run in CI.

| Phase | Gate | Artefact | Fails when |
|---|---|---|---|
| 2 | State model is actually delivered | `state_validation_report.json` | delivery < 100% |
| 3 | Evaluator is unambiguous | `verifier_validation_report.json` | a duplicate verifier name changed a task verdict |
| 4 | Step budget is not censoring runs | `horizon_report.json` | > 1/3 of runs hit the ceiling and failed |

Phase 2 reports the per-task delivery chain the blueprint asks for —
`Task | Anchor | Route | Tables | Records | Delivery | Precision | Recall` — so a weak
representation is distinguishable from a delivery bug. Precision and recall are
derived from verifier SQL and are strictly post-hoc; they never reach the agent.

Phase 4 exists because `max_steps = 5` made `TSR = 0/11`: the budget, not the agent,
was being measured. The default budget is now **15**, forwarded explicitly and
identically to both arms.

## Delivery and routing rules

A Condition-B run counts as delivered when the anchor resolved, the route did not
fail, and at least one record and token reached the agent. A required table that is
*reached but empty* no longer counts as a delivery failure — it is a true statement
about the database, recorded in `uncovered_required` instead. Only `FAILED` routes
fail the gate.

Routing escalates through a hop ladder (`2 -> 3 -> 4`) so the shortest valid schema
path is found rather than assumed; `location -> user -> user_group_member ->
user_group` is reachable when a task needs it. The planner still refuses to
fabricate a relationship: if no rung reaches a required table, routing fails closed.

Anchor resolution follows a deterministic ladder — explicit id, numeric identifier,
exact name, then constrained semantic lookup — and prompt-derived candidate mentions
are filtered of generic noise (`Product`, `Bind`, `There's`) so the lookup budget is
spent on mentions that can match a database value.

Every generated statement passes structural validation before it reaches the server:
single statement, read-only `SELECT`, explicit projection, balanced literals. A
proper noun containing an apostrophe (for example `Wayne Enterprises' Windows
Server`) is escaped rather than sent as malformed SQL, which previously produced an
HTTP 400 and silently cost a Condition-B run its delivery.
