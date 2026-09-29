---
goal: Build an EnterpriseOps-Gym-native paired A/B ablation harness that measures the task-success effect of the current csm_env graph-grounded environment/state model on CSM tasks
version: 1.0
date_created: 2026-09-26
last_updated: 2026-09-26
owner: Research engineering
status: 'Planned'
tags: [feature, experiment, ablation, benchmark, csm, evaluation, statistics]
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This plan implements **Experiment 1** of the frozen research design: a paired A/B ablation that asks
*"does the current graph-grounded environment/state model improve LLM-agent performance on
EnterpriseOps-Gym CSM tasks?"*

The plan is derived directly from the approved blueprint and executes it literally:

- **Condition A (baseline)** = native EnterpriseOps-Gym agent: task → LLM → CSM MCP tools → environment.
- **Condition B (intervention)** = the *current, frozen* `csm_env` model: task → LLM + `GroundedState`
  from GGQR → native MCP action → environment.
- Evaluation uses the benchmark's **own** tasks, execution path, SQL verifiers, and scoring metrics.
  No parallel success definition is invented.
- **The transition model, action simulation, learned transition function, and planning search are
  explicitly out of scope.** `C = B + transition model` is Experiment 2 and is gated on Experiment 1
  producing a measured result.

The first deliverable is therefore **not** another change to GGQR. It is a benchmark task runner, a
paired A/B evaluator, and a result schema, plus the statistics needed to report
`ΔTSR = TSR_B − TSR_A` with a paired bootstrap confidence interval and McNemar's test.

Grounding facts verified in this checkout (2026-09-26):

| Fact | Verified value | Source |
|---|---|---|
| CSM task files available locally | 12 (`data/revised/csm/task_*.json`) | directory listing |
| Verifiers across those 12 tasks | 100 total | JSON inspection |
| Tasks whose verifiers share duplicate `name` values | 4 of 12 (3, 1, 5, 9 duplicated names) | JSON inspection |
| Verifier result keying in the benchmark | keyed by `verifier.name` only → last-writer-wins | `benchmark/executor.py:420,474` |
| Task repeats declared in task JSON | `number_of_runs = 1`, `reset_database_between_runs = true` | JSON inspection |
| Seed database source | `gym_dbs.zip` → `Domain Wise DBs and Task-DB Mappings/csm/dbs/*.sql` (20 CSM entries) | archive listing |
| Executor scoring fields | `overall_success_rate`, `verifier_level_pass_rate`, `runs[].error` | `compute_score.py:12-39` |
| `conf/` and `config.json` | absent; only `conf.example/` exists | path probes |
| Python interpreter | 3.11.8 at `C:\Users\gowda\AppData\Local\Programs\Python\Python311\python.exe` | `python --version` |
| `csm_env` copies | root `csm_env/` **and** `EnterpriseOps-Gym/csm_env/` both present | directory listing |

The duplicate-verifier-name finding is not incidental: upstream EnterpriseOps-Gym issue #23 reports
that verifier results are silently dropped when verifiers share a name, and `benchmark/executor.py:474`
confirms the mechanism in this checkout (`verification_results[verifier_name] = result`). Because the
harness must reconcile against the official scorer, this plan **measures the defect and reports an
index-keyed corrected variant alongside the official collapsed number** rather than editing the
benchmark.

## 1. Requirements & Constraints

### 1.1 Functional requirements

- **REQ-001**: The harness must be strictly additive. It must not modify `csm_env/` (root or gym copy), `benchmark/`, `orchestrators/`, `evaluate.py`, `compute_score.py`, `benchmark_utils.py`, or any verifier logic.
- **REQ-002**: The unit of evaluation must be one EnterpriseOps-Gym CSM task, and task success must be the benchmark's own definition: all of that task's verifiers pass.
- **REQ-003**: The harness must report the official primary metric (Task Success Rate), the official secondary metric (Verifier-Level Pass Rate), and an agent-error rate matching `compute_score.py`'s "files with errors" semantics.
- **REQ-004**: Exactly two conditions must exist in Experiment 1, `A_baseline` and `B_state_model`, differing only in whether the frozen `csm_env` state model is available to the agent.
- **REQ-005**: The design must be paired: the identical task instance must be evaluated under A and B, and analysis must use paired methods (McNemar's test for the binary outcome, paired bootstrap CI for ΔTSR).
- **REQ-006**: Repeats must be preserved as explicit records. Each unit is `(task_id, run_index, condition)` and each carries a stable run identifier; no run-to-run determinism may be assumed.
- **REQ-007**: Each task in the evaluation set must carry `task_id`, seed database reference, `database_id`, `initial_state_fingerprint`, `complexity_category`, `reference_entities`, and `verifier_count`.
- **REQ-008**: The harness must collect the blueprint's secondary telemetry: agent steps, read/query calls, invalid actions, latency, input/output token usage, and the state-retrieval group (retrieval precision, retrieval recall, retrieved records, retrieved tokens).
- **REQ-009**: The harness must produce a stratified breakdown of TSR and VPR by state-dependence type (`TYPE_1_SINGLE_ENTITY` … `TYPE_5_MULTI_STEP`).
- **REQ-010**: The database must be reset between condition runs, and the reset must be proven by comparing `initial_state_fingerprint` values before each condition execution.
- **REQ-011**: Condition B must present the frozen pipeline only: `SchemaGraph → GGQR → relevant table/record retrieval → GroundedState → relations + facts + provenance → agent`. No predicted future, no action simulation, no learned transition function, no planning search.
- **REQ-012**: Experiment 1 must freeze one orchestrator (`react`) and one tool mode (`oracle`); other orchestrators and the `+5_tools`/`+10_tools`/`+15_tools` modes are later experiments.
- **REQ-013**: Every task's verifier set must be validated before any run, and the validation result must be recorded in the protocol document.
- **REQ-014**: Pinned revisions must be recorded for the benchmark code, the CSM dataset, the `csm_env` schema manifest, and the seed database archive.
- **REQ-015**: Harness-computed metrics must be reconcilable with `python compute_score.py --results_folder <dir>`; a bridge must exist that produces a `compute_score.py`-compatible layout.
- **REQ-016**: The harness must be offline-capable by default: unit tests and `--dry-run` must complete with no live CSM server and no LLM call.
- **REQ-017**: The harness must never feed verifier-only information into the state model used by Condition B, and must instead use verifier information only in post-hoc analysis.

### 1.2 Security requirements

- **SEC-001**: The Condition B state adapter must not have access to `verifiers`, `validation_config`, `query`, `expected_value`, `comparison_type`, `verifier.name`, or `verifier.description`. Any anchor/requirement metadata containing keys matching `verifier*`, `expected_value`, or `validation_config` must be rejected with an exception.
- **SEC-002**: Preflight, fingerprinting, manifest loading, and `--dry-run` must issue no mutating SQL and no mutating MCP call. Mutation occurs only through the agent's own native tools inside a scored run.
- **SEC-003**: LLM credentials must never be written into harness output. Any field named `llm_api_key`, `api_key`, or `token` nested under an LLM config object must be redacted before persistence.
- **SEC-004**: The SQL-runner path used for fingerprinting and verifier re-execution must remain read-only and must reuse the existing `EnterpriseOpsSQLRunner` / `SQLReader` transport rather than constructing SQL from untrusted input.

### 1.3 Constraints

- **CON-001**: No repository-local `pyproject.toml`, lockfile, CI config, or formatter config exists for the root `csm_env/` package; the harness must run with the plain interpreter and only the dependencies already present.
- **CON-002**: `conf/` and `config.json` do not exist in this checkout. They must be created by the operator from `conf.example/` and must never be committed with real credentials.
- **CON-003**: This checkout has no `.git` metadata, so revisions must be pinned by content hash (file SHA-256 list) and archive hash rather than by commit SHA.
- **CON-004**: The local CSM dataset is 12 task files, not the dataset card's 186 CSM tasks and not the README's headline totals. The evaluation set must be built from an explicitly pinned revision and the shortfall must be recorded in the protocol.
- **CON-005**: Local CSM tasks declare `number_of_runs = 1`; the harness owns repetition (`task × run × condition`) instead of relying on that field.
- **CON-006**: `gym_dbs.zip` is an input data asset and must not be modified, rewritten, or re-created by the harness.
- **CON-007**: No Docker CSM server is currently running. All execution tasks are gated on an operator-provided `sn-csm-server` reachable at the URL declared in the task JSON (observed: `http://localhost:8001`).
- **CON-008**: Local interpreter is Python 3.11.8; the harness must stay compatible with 3.11+.
- **CON-009**: The statistics implementation must not add a new third-party runtime dependency; paired tests must be implemented with the standard library (`math`, `random`, `statistics`).
- **CON-010**: `compute_score.py` defines TSR and VPR as **means over per-file rates**; the harness must not silently substitute a pooled ratio in the official numbers.

### 1.4 Architecture guidelines and patterns

- **GUD-001**: Keep the relational database as the single source of truth; all `csm_env` data remains derived and freshness-labeled.
- **GUD-002**: Compose, never fork. Reuse `BenchmarkExecutor`, `VerifierEngine`, `MCPClient`, `create_database_from_file`, `delete_database`, `LLMClient`, `ReactOrchestrator`, and `csm_integration` as-is.
- **GUD-003**: Mirror `compute_score.py` metric semantics exactly, and reconcile against it before reporting anything.
- **GUD-004**: Fail closed on unknown task ids, unknown tables, unknown complexity categories, unknown condition names, and unsupported verifier payloads; raise `KeyError` / `ValueError` rather than guessing.
- **GUD-005**: Persist one JSON record per `(task, condition, run)` plus an append-only JSONL index, mirroring `benchmark_utils.skip_sample` semantics so an interrupted sweep can resume without recomputation.
- **GUD-006**: Never fold agent-error runs into success metrics; report error rate separately as a reliability metric.
- **GUD-007**: Report both the official collapsed verifier summary and the index-keyed corrected summary, and label which one drives the headline table.
- **GUD-008**: Keep the intervention boundary explicit in code: `ablation/state_model.py` must not import or read `ablation/relevance.py` (the verifier-derived analysis module), and this must be asserted by a test.

- **PAT-001**: Use frozen dataclasses (`@dataclass(frozen=True)`) for condition, task, run-record, and statistics result types.
- **PAT-002**: Inject the SQL reader, clock, LLM client, and verifier engine so the whole harness is testable with local doubles, matching `csm_env`'s existing dependency-injection pattern.
- **PAT-003**: Implement the intervention by subclassing `orchestrators.base.AgentOrchestrator` / `ReactOrchestrator` instead of editing the ReAct loop.
- **PAT-004**: Match repository Python style: `from __future__ import annotations` in implementation modules, four-space indentation, predominantly double-quoted strings, explicit imports, no wildcard imports.
- **PAT-005**: Seed every stochastic analysis step (bootstrap resampling) with an explicit `seed` argument and record the seed in the report.
- **PAT-006**: Validate entity/table names against `csm_env`'s `SchemaRegistry` before any identifier is interpolated into SQL, and prefer structured inputs over string-built SQL.
- **PAT-007**: Write every offline test with local doubles only (no network), mirroring `EnterpriseOps-Gym/tests/test_csm_wiring_smoke.py` and `csm_env/tests/` conventions.

## 2. Implementation Steps

### Implementation Phase 1 — Protocol freeze, intervention freeze, and revision pinning

- GOAL-001: Pre-register Experiment 1 and freeze the intervention, constants, and pinned revisions before any code that could change GGQR is considered.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Create `docs/experiments/experiment-1-protocol.md`. Record: research question H1 (`TSR_stateModel > TSR_baseline`, direction not assumed); Condition A and Condition B definitions; the constants table (same task, initial DB, LLM, temperature, max steps, native tools, tool mode, orchestrator, verifiers, DB reset; only state-model availability differs); primary metric (TSR, all verifiers pass), secondary metric (VPR), reliability metric (agent error rate); the Type 1–5 stratification; and the analysis plan (paired ΔTSR, McNemar, paired bootstrap 95% CI). Include the fairness rule verbatim: the graph model may only see information legitimately obtainable from the environment, and the verifier is used only afterward to determine success. | | |
| TASK-002 | Record in the protocol that the transition model, action simulation, learned transition function, and planning search are **out of scope** for Experiment 1, and that `C = B + transition model` is Experiment 2 gated on Experiment 1 yielding a measured ΔTSR. Cite REQ-011 and REQ-017 as the enforcement clauses. | | |
| TASK-003 | Create `EnterpriseOps-Gym/ablation/manifests/pinned_revisions.json` containing: the pinned file-hash list (SHA-256) for `benchmark/*.py`, `orchestrators/*.py`, `evaluate.py`, `compute_score.py`, `benchmark_utils.py`, `csm_integration.py`; the pinned hashes of the 12 CSM task JSON files; the SHA-256 of `gym_dbs.zip`; the `csm_env` `SchemaManifest.schema_version`; the local CSM task count and the recorded shortfall versus the dataset card's 186 CSM tasks; and the interpreter version. Revisions are content hashes because this checkout has no `.git` metadata (CON-003). | | |
| TASK-004 | Add `EnterpriseOps-Gym/ablation/parity.py` with `compare_csm_env_copies(root_package, gym_package) -> ParityReport`, computing per-file SHA-256 for the two `csm_env` trees and reporting added, removed, and changed modules. Fail closed (`ValueError`) when a module present in both copies differs. Record the resulting parity hash list in `pinned_revisions.json`. | | |

**Completion gate:** the protocol exists, the intervention is frozen in writing, and `pinned_revisions.json` validates against `ablation/manifests/revisions.schema.json`.

### Implementation Phase 2 — Evaluation set, task registry, and verifier-set validation

- GOAL-002: Build the controlled, stratified, fingerprinted CSM evaluation set plus a pre-run verifier validator that detects the upstream duplicate-name collision.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-005 | Create `EnterpriseOps-Gym/ablation/task_registry.py` with frozen dataclasses `TaskRecord`, `EvalSetManifest`, the closed tuple `COMPLEXITY_CATEGORIES = ("TYPE_1_SINGLE_ENTITY", "TYPE_2_ONE_HOP", "TYPE_3_MULTI_HOP", "TYPE_4_CROSS_ENTITY", "TYPE_5_MULTI_STEP")`, and `load_eval_set(path) -> EvalSetManifest`. Unknown category, duplicate `task_id`, missing `seed_database_file`, or a `reference_entity` absent from `csm_env.SchemaRegistry` must raise `ValueError` / `KeyError`. | | |
| TASK-006 | Define `TaskRecord` fields exactly: `task_id`, `task_config_path`, `seed_database_file`, `database_id` (null until runtime), `initial_state_fingerprint`, `complexity_category`, `reference_entities` (tuple of tables), `reference_rows` (tuple of `(table, primary_key_value)` used only as the Condition B anchor), `verifier_count`, `selected_tools_count`, `number_of_runs`, `reset_database_between_runs`, and `category_rationale`. `reference_rows` is environment-obtainable metadata and must never be derived from verifier fields (SEC-001). | | |
| TASK-007 | Create `ablation/fingerprint.py` with `async def initial_state_fingerprint(reader, registry, *, max_rows_per_table=5000) -> str`. The fingerprint is SHA-256 over, for each table in sorted registry order: table name, `COUNT(*)`, and the ordered primary-key values truncated to `max_rows_per_table` with an explicit truncation marker. Queries must select explicit columns from the schema registry, must be read-only, and must not use `SELECT *` (SEC-004, PAT-006). | | |
| TASK-008 | Create `ablation/verify_pinning.py` with frozen `VerifierSetReport` and `validate_verifier_set(task_config_path) -> VerifierSetReport`, reporting `declared_count`, `unique_name_count`, `duplicate_names` (sorted tuple), `unsupported_verifier_types`, and `missing_gym_name_count`. Because `benchmark/executor.py:474` keys results by `verifier.name`, a non-empty `duplicate_names` means the official collapsed summary under-reports the verifier count. | | |
| TASK-009 | Generate `ablation/manifests/csm_eval_set.json` for the 12 pinned local CSM tasks: assign `complexity_category` by inspecting only the task's `user_prompt` and `selected_tools` (never `verifiers`), set `verifier_count` from the task file, derive `reference_entities` and `reference_rows` from the prompt plus environment lookups, and set `seed_database_file` from `gym_servers_config[*].seed_database_file`. Record `category_rationale` per assignment. | | |
| TASK-010 | Create `ablation/manifests/csm_requirements.json` mapping each evaluation-set task id to a GGQR `task_type` from the registered set (`resolve_case`, `case_overview`, `check_entitlement`, `find_knowledge`, `account_profile`) with a task-type → required-table mapping consumable by `RequirementRegistry.from_json` and `csm_integration._load_requirement_tables`. Unknown tables must fail closed inside `csm_env`. | | |
| TASK-011 | Extend the protocol document with verifier-validation results for all 12 pinned tasks, explicitly naming the 4 tasks with duplicate verifier names, and stating that the headline metric uses the official collapsed summary while the corrected index-keyed summary is reported alongside (GUD-007). | | |

**Completion gate:** `load_eval_set` and `validate_verifier_set` both succeed offline; every manifest task has a category, a recorded verifier report, and a requirements mapping.

### Implementation Phase 3 — Condition definitions and the Condition B state adapter

- GOAL-003: Define the two frozen conditions and implement the Condition B adapter entirely inside the fairness boundary.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-012 | Create `ablation/conditions.py` with frozen `Condition(name: str, state_model: bool)` and the singletons `CONDITION_A = Condition("A_baseline", False)` and `CONDITION_B = Condition("B_state_model", True)`, plus `CONDITIONS = (CONDITION_A, CONDITION_B)` and `condition_from_name(name)` raising `KeyError` for anything else (GUD-004). | | |
| TASK-013 | Create `ablation/state_model.py` with `StateModelAdapter`, constructed from the existing `csm_integration.build_csm_environment(mcp_client, enable_ggqr=True, budget=..., requirements_path=...)`. Expose `async def context_for(record: TaskRecord) -> StateContext` calling only `route_state` / `build_state` on the returned `CSMEnvironmentAPI` and rendering with `CSMEnvironmentAPI.represent_state` (the existing versioned serializer). | | |
| TASK-014 | Define frozen `StateContext` fields: `task_id`, `task_type`, `reference_type`, `reference_id`, `route_status`, `rendered_text`, `retrieved_tables` (tuple), `retrieved_rows` (tuple of `(table, row_pk)`), `retrieved_token_count`, `retrieval_latency_ms`, `unresolved` (tuple), `contradictions` (tuple), `schema_version`, `observed_at`. Token count is a deterministic whitespace/character-based estimate; record the estimator name in the field `token_estimator`. | | |
| TASK-015 | Enforce SEC-001 in `state_model.py`: validate the anchor metadata (`task_type`, `reference_type`, `reference_id`, `reference_rows`) against a forbidden-key pattern (`^verifier`, `^expected_value$`, `^validation_config$`, `^comparison_type$`, `^query$`) and raise `ValueError` on contact. The module must not import `ablation/relevance.py` and must not accept a task config dict as an input. | | |
| TASK-016 | Implement `StateContextPolicy` with the frozen modes `PRE_ACTION_ONCE` and `PRE_ACTION_PER_STEP`; freeze `PRE_ACTION_ONCE` for Experiment 1 and mark per-step refresh as deferred to Experiment 1b. Record the choice in the protocol. | | |
| TASK-017 | Create `ablation/relevance.py` (analysis-only module, GUD-008) with frozen `RelevanceSet` and `extract_relevant_entities(task_config_path) -> RelevanceSet`: extract table names from verifier SQL (`FROM` / `JOIN` clauses) and extract row-level anchors where a primary-key equality literal is parseable; carry `extraction_confidence` and an `unparsed` tuple with reasons. This module is used only for post-hoc precision/recall and must never be reachable from `state_model.py`. | | |

### Implementation Phase 4 — Additive agent-side integration

- GOAL-004: Inject the frozen state model into ReAct by subclassing, leaving native tools, prompts, and iteration limits untouched.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-018 | Create `ablation/state_model_orchestrator.py` defining `StateModelReactOrchestrator(ReactOrchestrator)`. Accept `state_context_provider` and `state_context_policy` keyword arguments. In `execute()`, await the provider for the task once, then prepend exactly one additional `HumanMessage` carrying `StateContext.rendered_text` after the existing system and user messages, and then delegate to `super().execute()`. Do not modify `available_tools`, `max_iterations`, or the tool-call loop. | | |
| TASK-019 | Ensure Condition A uses the unmodified `ReactOrchestrator` with no extra messages, so the only difference between conditions is the injected state context (REQ-004). Add a module-level constant `STATE_CONTEXT_HEADER` used to label the injected message so the experiment is auditable in `conversation_flow`. | | |
| TASK-020 | Implement `StateUsageTelemetry` collection inside `StateModelReactOrchestrator.get_result_metadata()` including `state_retrieval_count`, `retrieved_records`, `retrieved_tokens`, `context_tokens_injected`, and `state_route_status`, so the existing `BenchmarkExecutor.execute_single_run` result dict carries it without executor changes (the executor already merges `orchestrator.get_result_metadata()`). | | |
| TASK-021 | Create `ablation/verifier_bridge.py` with `async def run_verifiers_indexed(verifier_engine, verifier_configs, task_result, gym_configs, default_database_id, default_context) -> IndexedVerificationReport`. It reproduces `BenchmarkExecutor._run_verifiers` semantics exactly (same `VerifierConfig` construction, same gym/database selection, same `model_response` payload) but keys results as `f"{index}:{name}"` and returns both the collapsed summary and the index-keyed summary (GUD-007). Verifier execution itself must call `VerifierEngine.execute_verifier` unchanged. | | |

### Implementation Phase 5 — Paired A/B runner, result schema, and CLI

- GOAL-005: Execute matched task instances under both conditions and persist a complete, resumable experiment record.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-022 | Create `ablation/results.py` with frozen `TaskRunRecord` and `write_run_record(record, output_dir)`. Fields: `experiment="experiment_1"`, `run_uid` (uuid4), `condition`, `task_id`, `run_index`, `seed_database_file`, `runtime_database_id`, `initial_state_fingerprint`, `model_provider`, `model_name`, `temperature`, `max_steps`, `tool_mode`, `orchestrator`, `overall_success`, `verifier_summary_collapsed`, `verifier_summary_indexed`, `verifier_results_collapsed`, `verifier_results_indexed`, `agent_error`, `error_message`, `steps_taken`, `tool_call_count`, `read_query_count`, `invalid_action_count`, `latency_ms`, `input_tokens`, `output_tokens`, `state_retrieval` (nullable dict), `started_at`, `finished_at`. | | |
| TASK-023 | Implement resumption and indexing in `results.py`: one JSON file per `(task_id, condition, run_index)` named `run__{task_id}__{condition}__r{run_index}.json`, an append-only `index.jsonl`, and `is_recorded(task_id, condition, run_index) -> bool` mirroring `benchmark_utils.skip_sample` (GUD-005). Apply SEC-003 redaction of `api_key` / `llm_api_key` / `token` fields before writing. | | |
| TASK-024 | Create `ablation/runner.py` with `async def run_condition(condition, task_record, run_index, llm_config, output_dir, *, tool_mode="oracle", orchestrator="react", state_adapter=None) -> TaskRunRecord`. Condition A instantiates `BenchmarkExecutor` with `ReactOrchestrator`; Condition B instantiates the same executor with `StateModelReactOrchestrator` plus the state-context provider. Both paths must reuse the executor's own database creation and cleanup (`create_database_from_file` / `delete_database`) and must not bypass `execute_benchmark`. | | |
| TASK-025 | Implement `async def run_pair(task_record, run_index, ...) -> PairOutcome`. The pair executes A then B with identical LLM config, temperature, `max_steps`, tool mode, orchestrator family, and verifier set; it computes `initial_state_fingerprint` before each condition and sets `fingerprint_match: bool`. A mismatch must mark both records `agent_error=True` with an explicit reason and exclude the pair from the primary metric (REQ-010). | | |
| TASK-026 | Add per-run telemetry extraction: `steps_taken` from `conversation_flow` AI-message count, `tool_call_count` from `tool_results` length, `read_query_count` from read-only CSM tool names, `invalid_action_count` from tool results with `success=False`, `latency_ms` from `execution_time_ms`, and `input_tokens` / `output_tokens` from `usage_metadata` when present (absent usage must be recorded as null, never as zero). | | |
| TASK-027 | Create `run_ablation.py` in the `EnterpriseOps-Gym` root with arguments `--conditions` (default `A,B`), `--eval-set`, `--tool-mode` (default `oracle`), `--orchestrator` (default `react`), `--num-runs` (default 3), `--llm-config`, `--output-folder`, `--concurrency` (default 1), `--seed`, and `--dry-run`. `--dry-run` must load the eval set, run verifier validation, print the full `task × run × condition` matrix, and exit without touching the network or an LLM (REQ-016). | | |

**Completion gate:** `run_ablation.py --dry-run` prints the complete matrix offline and exits zero; a single paired live run writes two `run__*.json` records and two `index.jsonl` lines with matching fingerprints.

### Implementation Phase 6 — Official-metric computation, paired statistics, and reporting

- GOAL-006: Compute the benchmark's own metrics plus the paired statistical analysis, and emit the paper tables.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-028 | Create `ablation/stats.py` with `tsr(records)` (mean of per-record success, mirroring `compute_score.py`), `vpr(records)` (mean of per-record verifier pass rates, matching the "Avg Verifier Pass" semantics), `agent_error_rate(records)`, and `mean_extra_metrics(records)` for steps, tool calls, reads, invalid actions, latency, and tokens. Implement `verifier_level_pass_rate_pooled(records)` as a clearly labeled non-official companion metric (CON-010). | | |
| TASK-029 | Implement paired analysis in `ablation/stats.py`: `pair_records(records) -> PairedOutcomes` producing the four discordant cells (`A pass / B pass`, `A pass / B fail`, `A fail / B pass`, `A fail / B fail`), `mcnemar_test(pairs)` using the exact binomial test via `math.comb` when discordant pairs are below 25 and the continuity-corrected chi-square otherwise, and `paired_bootstrap_ci(pairs, n_resamples=10000, alpha=0.05, seed=0)` returning `ΔTSR`, CI bounds, and the seed used (CON-009, PAT-005). | | |
| TASK-030 | Implement `breakdown_by_complexity(records)` returning TSR and VPR per `TYPE_1_SINGLE_ENTITY` … `TYPE_5_MULTI_STEP` with per-cell `n`, and `breakdown_by_bucket(records, field, bounds)` for step-count and verifier-count buckets. Cells with `n = 0` must be reported as `n/a`, not `0`. | | |
| TASK-031 | Implement state-retrieval precision and recall in `ablation/stats.py` using `RelevanceSet` from `ablation/relevance.py`, computed over `(table, row_pk)` pairs with table-granularity fallback: `precision = retrieved ∩ relevant / retrieved`, `recall = retrieved ∩ relevant / relevant`. Records whose relevance set could not be parsed must be excluded and counted in `excluded_for_relevance`. | | |
| TASK-032 | Create `ablation/score_bridge.py` with `emit_compute_score_layout(records, output_dir)` writing `run_1/results_*.json` files containing the `statistics.overall_success_rate`, `statistics.verifier_level_pass_rate`, and `runs[].error` fields that `compute_score.py` reads, so `python compute_score.py --results_folder <dir>` reproduces the harness TSR and VPR within floating-point tolerance (REQ-015, GUD-003). | | |
| TASK-033 | Create `ablation/report.py` emitting `report.md` and `report.json` containing: the headline table (Condition, CSM Tasks, Task Success Rate, Verifier Pass Rate, Agent Error Rate); the difference row `ΔTSR`, `ΔVPR`, `ΔER`; the result block `TSR_A`, `TSR_B`, `Δ = TSR_B − TSR_A`, 95% CI, McNemar p-value, and the discordant table; the stratified table by complexity category; the secondary telemetry table; the collapsed-versus-indexed verifier reconciliation; and the pinned revisions plus seeds used. | | |

### Implementation Phase 7 — Preflight gate and offline verification

- GOAL-007: Prove the harness is correct, offline-repeatable, and ready before any scored sweep is started.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-034 | Create `ablation/preflight.py` with `async def preflight(...) -> PreflightReport` checking: CSM server reachability via `/health`; seed archive presence and the presence of each task's `seed_database_file` entry inside `gym_dbs.zip`; verifier validation status for every eval-set task; presence of an LLM config; presence of `conf/` and `config.json` created from `conf.example/`; and `csm_env` importability with `SchemaRegistry.from_static()` reporting 17 tables. Print PASS/FAIL per check and exit non-zero on any FAIL. This script performs no mutation (SEC-002). | | |
| TASK-035 | Implement `--dry-run` to share the same validation path as `preflight.py`, so the printed run matrix and the validation report always agree. | | |
| TASK-036 | Populate `initial_state_fingerprint` for all 12 eval-set tasks on first live preflight and record the values in `ablation/manifests/csm_eval_set.json` and the protocol document. | | |
| TASK-037 | Add metric-reproduction instructions to the protocol: the exact commands that regenerate every metric and table from raw run records only, with no live server, so an independent reader can recompute every reported number. | | |

### Implementation Phase 8 — Frozen sweep execution and analysis (gated on operator-provided infrastructure)

- GOAL-008: Execute the frozen Experiment 1 protocol and record the measured result and the Experiment 2 decision gate.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-038 | Operator action: start the Docker `sn-csm-server` and create `conf/llm/*.json` from `conf.example/llm/*.json`. Then run `python run_ablation.py --dry-run` and confirm PASS for every preflight check (CON-002, CON-007). | | |
| TASK-039 | Execute the evaluation sweep at the frozen configuration (`--conditions A,B --tool-mode oracle --orchestrator react --num-runs 3`, `--concurrency 1`), resuming from `index.jsonl` after any interruption. Record wall-clock start and end times and every seed used. | | |
| TASK-040 | Compute the primary and secondary metrics, the paired statistics, and the stratified breakdown; generate `report.md` and `report.json`; and verify that the `compute_score.py` bridge agrees with the harness numbers. | | |
| TASK-041 | Record the decision-gate outcome in the protocol: if ΔTSR is measurable (in either direction), Experiment 1 is complete and the Experiment 2 design (`C = B + transition model`) may be drafted as a separate plan; if the sweep is under-powered, or if the fingerprint check failed for any pair, repeat the affected pairs before drawing any conclusion. No transition-model code is written in this plan (REQ-011, TASK-002). | | |

**Completion gate (plan-level):** a `report.json` exists containing `TSR_A`, `TSR_B`, `ΔTSR` with a 95% CI and a McNemar p-value, plus the stratified table, the reconciliation against `compute_score.py`, and the pinned revisions and seeds.

## 3. Alternatives

- **ALT-001**: Build a standalone task runner and a new verifier instead of using EnterpriseOps-Gym's tasks, execution, and SQL verifiers. Rejected: the study must use the benchmark's own success definition, otherwise the result is not comparable to the leaderboard and the fairness argument weakens.
- **ALT-002**: Skip straight to `A → C` by adding the transition model now. Rejected: if task success increases, the improvement cannot be attributed to state modeling versus transition modeling. The blueprint requires `A → B → C` decomposition.
- **ALT-003**: Include `+5_tools`, `+10_tools`, and `+15_tools` modes in Experiment 1. Rejected as a separate research question; including them would add tool-discovery as a confound while measuring state modeling. Deferred.
- **ALT-004**: Redefine success as verifier-level pass rate alone, or as partial credit. Rejected: TSR (all verifiers pass) is the benchmark's primary metric; VPR is reported as a secondary metric, not a substitute.
- **ALT-005**: Treat A and B as unrelated samples and compare independent proportions. Rejected in favor of a paired design with McNemar's test and a paired bootstrap CI, which is more powerful and matches the matched-task design.
- **ALT-006**: Expose the GGQR state as an additional MCP tool callable by the agent, instead of injecting context into the prompt. Deferred: it changes the tool surface, and in `oracle` mode it would alter the fairness of the comparison. It would require its own ablation.
- **ALT-007**: Fix the upstream duplicate-verifier-name defect by editing `benchmark/executor.py`. Rejected: REQ-001 forbids modifying benchmark code, and an edited evaluator would no longer be the pinned official evaluator. Instead, the harness reports the official collapsed number and an index-keyed corrected number computed through the unchanged `VerifierEngine`.
- **ALT-008**: Refresh the GGQR state before every agent step (`PRE_ACTION_PER_STEP`) in Experiment 1. Deferred to Experiment 1b: per-step refresh changes the amount of state information available and would confound the "does state modeling help" question with a "how often does state refresh help" question.

## 4. Dependencies

- **DEP-001**: `EnterpriseOps-Gym/` checkout containing `benchmark/`, `orchestrators/`, `utils/`, `evaluate.py`, `compute_score.py`, `benchmark_utils.py`, and `csm_integration.py`. Present in this workspace.
- **DEP-002**: `csm_env` importable from the gym root (`EnterpriseOps-Gym/csm_env/`) and the working copy at `csm_env/`; both must pass the parity check in `ablation/parity.py`.
- **DEP-003**: A reachable Docker-hosted `sn-csm-server` (task JSON declares `http://localhost:8001`) with a `/health` endpoint. Not currently running; required only from Phase 8.
- **DEP-004**: An LLM configuration file in the `conf.example/llm/*.json` shape, providing `llm_provider`, `llm_model`, and `llm_api_key`, loadable by `benchmark_utils.load_llm_configs`.
- **DEP-005**: Seed database archives inside `gym_dbs.zip` at `Domain Wise DBs and Task-DB Mappings/csm/dbs/*.sql` (20 CSM entries present) as the source for `create_database_from_file`.
- **DEP-006**: The 12 pinned CSM task JSON files under `data/revised/csm/`.
- **DEP-007**: Python 3.11+ plus the already-present dependencies: `httpx`, `langchain_core`, `datasets`, `tqdm`, `tabulate`, `pytest`, `pytest-asyncio`. Verified importable on the local 3.11.8 interpreter.
- **DEP-008**: Standard-library statistics only for McNemar and the bootstrap (`math`, `random`, `statistics`); no new third-party runtime dependency (CON-009).
- **DEP-009**: The `csm_env` public surface used by the adapter: `from_enterpriseops_mcp_client`, `CSMEnvironmentAPI.route_state`, `CSMEnvironmentAPI.build_state`, `CSMEnvironmentAPI.represent_state`, `QueryBudget`, `SchemaRegistry.from_static`, and `GroundedState` field access.
- **DEP-010**: The evaluation-set manifest `ablation/manifests/csm_eval_set.json` and the requirement manifest `ablation/manifests/csm_requirements.json`, both produced in Phase 2 and consumed by Phases 3–8.

## 5. Files

### 5.1 New files (all additions; nothing existing is modified)

- **FILE-001**: `D:\Visual code projects\csm_ggqr_work\docs\experiments\experiment-1-protocol.md` — pre-registered protocol: hypothesis, conditions, constants, metrics, stratification, fairness rule, deferred Experiment 2, verifier-validation results, fingerprint values, seeds, and reproduction commands.
- **FILE-002**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\ablation\__init__.py` — package marker re-exporting the public harness names.
- **FILE-003**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\ablation\conditions.py` — frozen `Condition` type and the `A_baseline` / `B_state_model` singletons.
- **FILE-004**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\ablation\task_registry.py` — `TaskRecord`, `EvalSetManifest`, `COMPLEXITY_CATEGORIES`, `load_eval_set`.
- **FILE-005**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\ablation\fingerprint.py` — read-only `initial_state_fingerprint`.
- **FILE-006**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\ablation\verify_pinning.py` — `VerifierSetReport`, `validate_verifier_set`, duplicate-name detection.
- **FILE-007**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\ablation\parity.py` — `ParityReport`, `compare_csm_env_copies` for the two `csm_env` trees.
- **FILE-008**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\ablation\state_model.py` — `StateModelAdapter`, `StateContext`, `StateContextPolicy`, and the SEC-001 forbidden-key guard.
- **FILE-009**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\ablation\relevance.py` — analysis-only `RelevanceSet` and `extract_relevant_entities` (unreachable from `state_model.py`).
- **FILE-010**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\ablation\state_model_orchestrator.py` — `StateModelReactOrchestrator`, `STATE_CONTEXT_HEADER`, `StateUsageTelemetry`.
- **FILE-011**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\ablation\verifier_bridge.py` — `IndexedVerificationReport`, `run_verifiers_indexed`.
- **FILE-012**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\ablation\results.py` — `TaskRunRecord`, `write_run_record`, `is_recorded`, credential redaction.
- **FILE-013**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\ablation\runner.py` — `run_condition`, `run_pair`, `PairOutcome`, telemetry extraction.
- **FILE-014**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\ablation\stats.py` — `tsr`, `vpr`, `agent_error_rate`, `pair_records`, `mcnemar_test`, `paired_bootstrap_ci`, breakdowns, precision/recall.
- **FILE-015**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\ablation\score_bridge.py` — `emit_compute_score_layout` reconciling with `compute_score.py`.
- **FILE-016**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\ablation\report.py` — `report.md` / `report.json` generation.
- **FILE-017**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\ablation\preflight.py` — `PreflightReport`, `preflight`, shared validation path for `--dry-run`.
- **FILE-018**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\run_ablation.py` — CLI entry point (`--conditions`, `--eval-set`, `--tool-mode`, `--orchestrator`, `--num-runs`, `--llm-config`, `--output-folder`, `--concurrency`, `--seed`, `--dry-run`).
- **FILE-019**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\ablation\manifests\csm_eval_set.json` — the 12-task stratified evaluation set with fingerprints and reference entities.
- **FILE-020**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\ablation\manifests\csm_requirements.json` — task id to GGQR `task_type` plus task-type to required-table manifest.
- **FILE-021**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\ablation\manifests\pinned_revisions.json` — content-hash pins for benchmark code, dataset, archive, schema version, and interpreter.
- **FILE-022**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\ablation\manifests\revisions.schema.json` — JSON Schema validating `pinned_revisions.json`.
- **FILE-023**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\tests\test_ablation_conditions.py` — offline tests for conditions, task registry, and manifests.
- **FILE-024**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\tests\test_ablation_state_model.py` — offline tests for the state adapter, the fairness guard, and the orchestrator injection.
- **FILE-025**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\tests\test_ablation_runner.py` — offline tests for the runner, result schema, resumption, and redaction.
- **FILE-026**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\tests\test_ablation_stats.py` — offline tests for metrics, McNemar, bootstrap, breakdowns, and the score bridge.
- **FILE-027**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\tests\test_ablation_boundaries.py` — offline tests for fairness isolation, additivity, and interface parity.
- **FILE-028**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\tests\tests_support\ablation_doubles.py` — local doubles: `FakeSQLReader`, `FakeStateAdapter`, `FakeLLMClient`, `FakeVerifierEngine`.

### 5.2 Read-only inputs (must not be modified)

- **FILE-029**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\benchmark\executor.py`, `benchmark\verifier.py`, `benchmark\mcp_client.py`, `benchmark\models.py`, `benchmark\llm_client.py`, `orchestrators\react.py`, `orchestrators\base.py` — composed as-is by the harness.
- **FILE-030**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\compute_score.py`, `evaluate.py`, `benchmark_utils.py`, `csm_integration.py`, `utils\task_queue_worker.py` — official scorer, entry point, config loader, existing wiring, and worker.
- **FILE-031**: `D:\Visual code projects\csm_ggqr_work\csm_env\` and `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\csm_env\` — the frozen Condition B implementation, extended only by additive imports from `EnterpriseOps-Gym\ablation\`.
- **FILE-032**: `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\data\revised\csm\` task JSONs, `gym_dbs.zip`, and `conf.example\` — pinned task data, seed archive, and configuration templates.
- **FILE-033**: `D:\Visual code projects\csm_ggqr_work\plan\architecture-csm-environment-1.md` and `D:\Visual code projects\csm_ggqr_work\docs\architecture\csm-environment-contract.md` — the frozen contracts this plan depends on; referenced, not edited.

## 6. Testing

All tests below are offline: no CSM server, no LLM call, no network. Run the harness tests from the `EnterpriseOps-Gym` directory with `python -m pytest tests -v -p no:cacheprovider`, and keep the existing `csm_env` suite green with `python -m pytest csm_env/tests -v -p no:cacheprovider` from the repository root.

- **TEST-001**: `test_conditions_expose_only_a_and_b` — `CONDITIONS` contains exactly `A_baseline` and `B_state_model`; `condition_from_name("C_transition")` raises `KeyError`; `CONDITION_A.state_model is False` and `CONDITION_B.state_model is True`.
- **TEST-002**: `test_eval_set_manifest_loads_and_validates` — the 12-task manifest loads; every `complexity_category` is in `COMPLEXITY_CATEGORIES`; every `reference_entity` resolves in `SchemaRegistry.from_static()`; a duplicated `task_id`, an unknown category, and an unknown table each raise.
- **TEST-003**: `test_verifier_validation_detects_duplicate_names` — against the pinned task `task_20251205_153330_906_a8eea1c0_8c7a6205.json`, `validate_verifier_set` reports `declared_count == 11` with a non-empty `duplicate_names`; against a task with no duplicates, `declared_count == unique_name_count`.
- **TEST-004**: `test_eval_set_prompts_do_not_require_verifier_fields` — constructing `TaskRecord` values for the 12 pinned tasks succeeds while supplying only `user_prompt`, `selected_tools`, and `seed_database_file`, proving the intervention inputs never require verifier data.
- **TEST-005**: `test_fingerprint_is_deterministic_and_read_only` — with `FakeSQLReader`, two calls to `initial_state_fingerprint` are byte-identical; after the double is mutated to add a row, the fingerprint changes; the double records that no non-`SELECT` statement and no `SELECT *` was issued.
- **TEST-006**: `test_state_adapter_rejects_verifier_metadata` — passing a `TaskRecord`-like object carrying `verifier` / `expected_value` / `validation_config` keys raises `ValueError`; passing a raw task config dict raises `TypeError`.
- **TEST-007**: `test_state_model_context_contains_only_environment_entities` — with `FakeStateAdapter`, every key in `StateContext.retrieved_tables` is a registered `csm_env` table, and no rendered text contains a verifier `description` string taken from the corresponding task file.
- **TEST-008**: `test_intervention_is_additive_for_condition_a` — `ReactOrchestrator` produces a message list whose first two entries are the system and user prompt only; `StateModelReactOrchestrator` produces exactly one additional message bearing `STATE_CONTEXT_HEADER`; both report the same `max_iterations`.
- **TEST-009**: `test_run_record_schema_and_resumption` — `TaskRunRecord` requires all declared fields; `write_run_record` produces the expected file name; `is_recorded` returns `True` afterwards and a duplicate key is skipped; a missing required field raises.
- **TEST-010**: `test_credentials_are_redacted_in_persisted_records` — a record containing `llm_api_key`, `api_key`, and a nested `token` is written with those values replaced by the redaction marker, and the raw secret does not appear anywhere in the file bytes.
- **TEST-011**: `test_paired_statistics_match_hand_computed_cases` — for discordant counts `(0, 5)`, `(5, 0)`, `(3, 3)`, and `(12, 12)`, `mcnemar_test` reproduces hand-computed exact-binomial p-values within `1e-12`; `paired_bootstrap_ci` is reproducible under a fixed seed and returns `ΔTSR` equal to the observed difference; `breakdown_by_complexity` returns `n/a` for empty cells.
- **TEST-012**: `test_score_bridge_matches_compute_score_semantics` — output from `emit_compute_score_layout` re-read by `compute_score.process_mode` yields the same TSR and VPR as `ablation.stats.tsr` and `ablation.stats.vpr` on the same synthetic records, confirming CON-010 and REQ-015.
- **TEST-013**: `test_harness_does_not_import_verifier_derived_state` — static inspection asserts that `ablation/state_model.py` neither imports nor references `ablation.relevance`, and that no module in the Condition B import chain reads `validation_config` or `expected_value` (GUD-008, SEC-001).
- **TEST-014**: `test_existing_csm_env_and_gym_tests_still_pass` — `csm_env/tests` from the repository root and `EnterpriseOps-Gym/tests` from the gym root both pass unchanged, demonstrating REQ-001 additivity.
- **TEST-015**: `test_preflight_and_dry_run_agree_offline` — with the server unreachable, `preflight` reports the server check as FAIL while still evaluating every other check; `run_ablation.py --dry-run` exits without constructing an LLM client, performs no network call, and prints the same matrix the runner would execute.
- **TEST-016**: `test_verifier_bridge_recovers_collapsed_verifiers` — given a verifier list containing duplicate names and a `FakeVerifierEngine` that passes everything, the collapsed summary reports fewer entries than `declared_count` while the index-keyed summary reports exactly `declared_count` entries, reproducing the upstream collision on demand.
- **TEST-017**: `test_run_pair_flags_fingerprint_mismatch` — with a `FakeSQLReader` whose contents change between the two condition runs, `run_pair` returns `fingerprint_match=False`, marks both records `agent_error=True` with an explicit reason, and excludes the pair from `pair_records`.

**Execution commands (verification steps for the implementer):**

Step 1 — frozen Condition B suite must stay green, from the repository root:

```powershell
python -m pytest csm_env/tests -v -p no:cacheprovider
```

Step 2 — harness tests plus the existing wiring smoke tests, from `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym`:

```powershell
python -m pytest tests -v -p no:cacheprovider
```

Step 3 — offline readiness gate (no server, no LLM):

```powershell
python run_ablation.py --dry-run --eval-set ablation/manifests/csm_eval_set.json --llm-config conf/llm/my-model.json --output-folder out/experiment_1
```

Step 4 — sweep and official scorer reconciliation:

```powershell
python run_ablation.py --conditions A,B --num-runs 3 --llm-config conf/llm/my-model.json --output-folder out/experiment_1
python -m ablation.score_bridge --records-dir out/experiment_1 --output-dir out/experiment_1/compute_score
python compute_score.py --results_folder out/experiment_1/compute_score
```

## 7. Risks & Assumptions

- **RISK-001**: The local evaluation set is only 12 CSM tasks, far below the blueprint's 50–100 target and the dataset card's 186 CSM tasks. A 12-task paired set gives very low power and unstable category cells. Mitigation: report the shortfall explicitly, treat the 12-task run as a pipeline validation and effect-size pilot, and expand from a pinned Hugging Face revision before making any claim about ΔTSR magnitude. `--num-runs 3` on 12 tasks yields 36 pairs, which is a pilot rather than a confirmation.
- **RISK-002**: The anchor for Condition B must be derived from the prompt and the environment. Weak anchor selection produces empty or wrong `GroundedState`, understating Condition B for reasons unrelated to the research question. Mitigation: `reference_rows` are curated per task and validated against the live database during preflight, and the route status (`COMPLETE`, `UNRESOLVED`, `BUDGET_EXHAUSTED`) is recorded per run so retrieval failures are visible instead of being silently scored as model failure.
- **RISK-003**: The duplicate-verifier-name defect means the official collapsed summary can both under-count verifiers and mask a failing verifier behind a passing one with the same name. Mitigation: `ablation/verifier_bridge.py` reports an index-keyed summary through the unchanged `VerifierEngine`, and the report presents both numbers with an explicit diff (GUD-007, TEST-016).
- **RISK-004**: Two `csm_env` copies exist (repository root and inside `EnterpriseOps-Gym`). Running from the gym resolves the gym-local copy, so a stale copy would mean Condition B is not the frozen implementation. Mitigation: `ablation/parity.py` fails closed on divergence and the parity hash list is pinned (TASK-004).
- **RISK-005**: LLM cost and latency: 12 tasks × 2 conditions × 3 runs equals 72 agent runs, each up to the benchmark's step budget (CSM averages 12.10 steps per task). Mitigation: `--concurrency` defaults to 1 to keep database isolation clean, resumption via `index.jsonl` avoids recomputation, and `--dry-run` prevents accidental spend.
- **RISK-006**: Agent stochasticity even at temperature 0. Mitigation: preserve run indices and run identifiers, never assume determinism, and report run-to-run variation alongside the pooled metrics (REQ-006).
- **RISK-007**: Nondeterministic database state between conditions; a leaked mutation from Condition A would bias Condition B. Mitigation: the executor re-seeds the database per run and `run_pair` additionally compares `initial_state_fingerprint` before each condition, quarantining mismatched pairs (TASK-025, TEST-017).
- **RISK-008**: The injected state text could displace task context or exceed model context limits for large states. Mitigation: `StateContext.rendered_text` uses the existing truncating serializer, and both the retrieved token count and `context_tokens_injected` are recorded so an overloaded context is visible in the results.
- **RISK-009**: Type I error inflation from reporting the Type 1–5 stratified table and several secondary metrics. Mitigation: stratified analyses are labeled exploratory, the primary outcome is a single pre-registered comparison, and every reported p-value is accompanied by its CI and its `n`.
- **RISK-010**: Preflight or `--dry-run` could accidentally issue a mutating call. Mitigation: `FakeSQLReader` asserts read-only statements in tests, and preflight touches only `/health` plus read-only SQL (SEC-002, TEST-005, TEST-015).
- **RISK-011**: In-house statistics could silently diverge from standard definitions. Mitigation: `mcnemar_test` is verified against hand-computed exact-binomial values and `paired_bootstrap_ci` is verified for seed reproducibility (TEST-011).

- **ASSUMPTION-001**: The frozen Condition B is the currently implemented `csm_env` pipeline and requires no changes for Experiment 1, per the blueprint's freeze decision.
- **ASSUMPTION-002**: `oracle` tool mode is usable with the pinned tasks, because `selected_tools` is populated for all 12 local CSM tasks (8 to 19 tools each) and the executor filters tools by that field.
- **ASSUMPTION-003**: An operator will provide the Docker `sn-csm-server` and LLM credentials before Phase 8; no execution task starts before the preflight gate passes.
- **ASSUMPTION-004**: The 12 local CSM tasks are adequate for pipeline validation but are not assumed representative for any published effect size.
- **ASSUMPTION-005**: `create_database_from_file` fully re-seeds the task database for each run, so per-run state isolation holds without an additional reset mechanism.
- **ASSUMPTION-006**: Model identity, temperature, and step budget are held constant within a sweep by using one `LLMConfig`, and `max_iterations` is passed identically to both conditions.
- **ASSUMPTION-007**: Verifier `description` text and verifier SQL may be read by the analysis layer without contaminating the intervention, because `state_model.py` and `relevance.py` are structurally isolated and that isolation is enforced by TEST-013.
- **ASSUMPTION-008**: `compute_score.py` remains the scoring authority; any harness metric that cannot be reconciled with it is reported as a companion metric rather than a headline metric.

## 8. Related Specifications / Further Reading

- `D:\Visual code projects\csm_ggqr_work\plan\architecture-csm-environment-1.md` — the approved plan that froze Phases 0–8 of the `csm_env`/GGQR pipeline and deferred agent integration, learned routing, and predictive dynamics.
- `D:\Visual code projects\csm_ggqr_work\docs\architecture\csm-environment-contract.md` — approved contracts for `SchemaManifest`, `TaskRequest`, `QueryBudget`, `QueryPlan`, and `GroundedState`.
- `D:\Visual code projects\csm_ggqr_work\docs\architecture\csm-environment-fixtures.md` — fixture format and supported task types.
- `D:\Visual code projects\csm_ggqr_work\CHANGELOG-2026-09-24.md` — research-hardening record for the frozen intervention, including the 111-passed/3-skipped offline suite.
- `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\README.md` — benchmark description, scoring section, leaderboard context, and installation instructions.
- `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\csm_integration.py` — the existing additive wiring between the gym `MCPClient` and `csm_env`, reused by Condition B.
- `D:\Visual code projects\csm_ggqr_work\EnterpriseOps-Gym\tests\test_csm_wiring_smoke.py` — the offline wiring smoke tests whose style the harness tests mirror.
- EnterpriseOps-Gym dataset card (`ServiceNow-AI/EnterpriseOps-Gym`) — task fields, tool modes (`oracle`, `+5_tools`, `+10_tools`, `+15_tools`), and `number_of_runs` / `reset_database_between_runs` semantics.
- EnterpriseOps-Gym upstream issue #23 — duplicate verifier names silently dropping verifier results, which `ablation/verifier_bridge.py` measures and reconciles.
