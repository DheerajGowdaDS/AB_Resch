---
goal: Implement a schema-grounded CSM environment representation with GGQR, grounded state, representation, and verification
version: 1.1
date_created: 2026-09-24
last_updated: 2026-09-24
owner: Research engineering
status: 'Implemented (Phases 0-8)'
tags: [architecture, csm, schema-graph, ggqr, provenance, verification]
---

# Introduction

![Status: Implemented](https://img.shields.io/badge/status-Implemented-green)

This plan translates the three blueprint specifications into an executable extension of the active root `csm_env/` package. It preserves the blueprint flow:

`schema graph → deterministic query routing → database retrieval → grounded state → representation → verification`.

Phases 0–8 are implemented and verified against the deterministic offline suite (111 passed, 3 explicitly skipped live opt-ins). The clarification decisions in Section 1.4 are accepted; Phase 9 (agent integration, learned routing, predictive dynamics) remains deferred pending explicit owner approval.

The active implementation already contains a static 17-table schema registry (`csm_env/schema.py:7-116`), a record-level property graph (`csm_env/graph.py:7-125`), an HTTP SQL runner (`csm_env/sql_runner.py:9-100`), bounded hydration (`csm_env/builder.py:69-261`), a public facade (`csm_env/api.py:8-53`), and an MCP-client adapter (`csm_env/integration.py:9-22`). The plan extends these components rather than replacing them without compatibility review.

The schema graph and record graph are intentionally separate. The schema graph describes table structure and declared foreign keys. The record graph is a bounded, derived cache. The live relational database remains the only source of truth for record values.

## 1. Requirements & Constraints

### 1.1 Functional requirements

- **REQ-001**: Define a versioned machine-readable schema manifest containing all registered tables, primary keys, explicit columns, foreign keys, relation labels, and schema provenance.
- **REQ-002**: Build a structural schema graph containing table nodes and FK-backed relationship edges; structural metadata must be authoritative over semantic labels.
- **REQ-003**: Preserve the existing record-level graph as a separate derived view with deterministic node keys, edge deduplication, bounded traversal, and explicit invalidation behavior.
- **REQ-004**: Implement deterministic Graph-Guided Query Routing (GGQR) from a structured task request and reference identifier to a validated query plan and execution result.
- **REQ-005**: Implement anchor resolution, bounded FK propagation, direction-aware traversal, task-requirement filtering, path validation, independent-query parallel execution, and explicit stopping conditions.
- **REQ-006**: Implement table adapters that accept structured filters and explicit schema-derived columns; GGQR must not construct or accept arbitrary SQL.
- **REQ-007**: Construct a task-relevant `GroundedState` containing records, relationships, unresolved bindings, and provenance for every database-derived fact.
- **REQ-008**: Provide versioned graph, JSON, and text representations of the same state with stable ordering, redaction rules, and a fidelity check.
- **REQ-009**: Implement schema, relationship, routing, provenance, consistency, freshness, and graph-to-SQL equivalence verification.
- **REQ-010**: Preserve the current public import paths and existing API behavior through compatibility wrappers; add new capabilities through versioned, additive entry points.
- **REQ-011**: Keep the default test suite deterministic and offline; live-service checks must be explicit opt-in tests and must not be prerequisites for unit or integration tests.
- **REQ-012**: Organize new implementation components into purpose-specific subpackages so schema, routing, state, representation, transport, and verification responsibilities remain distinguishable.

### 1.2 Security requirements

- **SEC-001**: All database operations in the representation and GGQR paths are read-only; no SQL write, DDL, mutation, or environment-action execution is permitted in this plan.
- **SEC-002**: Table and column identifiers must come from the schema registry or an approved introspection result; user input must never be interpolated as an identifier.
- **SEC-003**: Provenance, route diagnostics, serialized state, and test artifacts must not contain auth tokens, credentials, raw authorization headers, or unrestricted context secrets.

### 1.3 Technical constraints and accepted defaults

- **CON-001**: The active source scope is the root `csm_env/` package, its tests, `README.md`, `AGENTS.md`, and the blueprint text supplied in the task; historical verification artifacts and archive copies are not implementation inputs.
- **CON-002**: The current package is flat and has no package-local build metadata; new subpackages must retain the existing flat-module import surface during migration.
- **CON-003**: The project targets Python 3.11 or newer; implementation must not rely on syntax or standard-library behavior older than that target.
- **CON-004**: `httpx` is the only approved runtime third-party dependency; tests use `pytest` and `pytest-asyncio`; no new dependency is added without approval.
- **CON-005**: No local CSM server, authoritative DDL, or confirmed live metadata endpoint is available in the active workspace; the first implementation must work with a static manifest and local test doubles.
- **CON-006**: Implementation may begin with Phase 0 after the accepted decisions in Section 1.4 are recorded; no later phase may start before its dependencies and completion gates pass.
- **CON-007**: The ZIP and `zip_extract/` copies are not modified; packaging changes require a separate request.
- **CON-008**: Raw SQL remains a legacy compatibility escape hatch only; GGQR and the planner never use it, and any public exposure requires a separate API decision.
- **CON-009**: The initial router is deterministic and does not call an LLM; an LLM task-requirement adapter is a later, separately measured experiment.
- **CON-010**: The first state implementation is case-local and task-relevant; it is not a full database snapshot and is not a replacement for fresh `get_ground_truth()` reads.
- **CON-011**: Missing values, null database values, unresolved FKs, missing rows, and unsupported metadata are distinct states and must not be conflated.
- **CON-012**: The implementation must use synthetic or redacted fixtures for deterministic tests and must not publish or commit raw CSM seed data.

### 1.4 Clarifications and approval record

The following decisions were confirmed before finalizing this plan. They are the implementation contract for Phases 0–8; Phase 9 remains deferred.

| Decision | Required question | Proposed default | Approval state |
|---|---|---|---|
| Scope | Implement the full Phase 0–8 representation pipeline first, or stop after schema graph and router? | Phase 0–8, with Phase 9 agent integration deferred | Accepted 2026-09-24 |
| GGQR input | Is the first input a structured `TaskRequest`, a natural-language task, or both? | Structured `TaskRequest`; optional text retained as metadata | Accepted 2026-09-24 |
| Anchor resolution | Must callers provide `reference_type`, or may GGQR infer it deterministically from aliases? | Require explicit type when ambiguous; deterministic alias fallback only for unique matches | Accepted 2026-09-24 |
| Schema authority | Is the static registry authoritative, or must runtime introspection be mandatory? | Static manifest for MVP; optional introspector with fail-closed mismatch reporting | Accepted 2026-09-24 |
| SQL safety | Does the SQL-runner endpoint support bound parameters? | Structured `QuerySpec` plus centralized safe literal rendering until bound parameters are confirmed | Accepted 2026-09-24 |
| State freshness | What freshness guarantee must a constructed state provide? | Fresh reads for each activated query; observation timestamps and optional refresh policy | Accepted 2026-09-24 |
| Provenance | Which provenance fields are mandatory? | `database_id`, schema version, table, column, row PK, query ID, route ID, and UTC observation time | Accepted 2026-09-24 |
| Task requirements | How are information requirements supplied? | Versioned deterministic requirement registry keyed by task type | Accepted 2026-09-24 |
| API compatibility | Must current return shapes remain unchanged? | Preserve current methods and add new methods/envelopes | Accepted 2026-09-24 |
| Budgets | What production limits should be configurable defaults? | `max_hops=2`, `max_tables=12`, `max_queries=32`, `max_rows_per_query=50`, `max_total_rows=500`, `max_parallel=8`; `max_total_rows` is a strict reservation cap across concurrent waves | Accepted 2026-09-24 |
| Test backend | May an external CSM dump or live endpoint be used in tests? | Synthetic fixture by default; live tests opt-in and excluded from normal suite | Accepted 2026-09-24 |
| Host integration | Which EnterpriseOps-Gym host version and `MCPClient` lifecycle are supported? | Keep current attribute-based adapter; defer version-specific extensions | Accepted 2026-09-24 |
| Research boundary | Is the work research-internal, read-only, and not intended for external publication of data/code? | Yes; no external publication and no upstream package import | Accepted 2026-09-24 |
| Agent integration | Should agent/planner/evaluator integration be part of this implementation? | Defer until schema, routing, state, and verification gates pass | Accepted 2026-09-24 |

### 1.5 Architecture guidelines and patterns

- **GUD-001**: Follow the blueprint dependency order: schema contract, schema graph, adapters, router, state, representation, verification, then agent integration.
- **GUD-002**: Keep the database as the source of truth and label all record-level graph data as derived or cached.
- **GUD-003**: Make the structural layer authoritative; every semantic relation must point to a structural FK or be explicitly marked as a derived annotation with lineage.
- **GUD-004**: Make every route decision explainable through an immutable `QueryPlan` containing anchor, edge IDs, filters, dependencies, and budgets.
- **GUD-005**: Make every acceptance gate executable with a focused test, a deterministic fixture, or a documented unsupported result.

- **PAT-001**: Use frozen dataclasses for schema specifications, schema nodes, query plans, provenance records, and verification results.
- **PAT-002**: Use dependency injection for the SQL reader, schema manifest, requirement registry, clock, and executor so tests do not require a live service.
- **PAT-003**: Execute independent query-plan steps in bounded asynchronous waves, then restore deterministic result ordering before aggregation.
- **PAT-004**: Add new subpackages behind compatibility wrappers; do not move or rename public flat modules until all callers and tests are migrated.
- **PAT-005**: Use a versioned envelope for state and representation output; never overload database row columns with route or provenance metadata.

## 2. Implementation Steps

### Implementation Phase 0 — Decision Freeze and Contract Drafting

- **GOAL-001**: Record the accepted decisions and freeze the contracts needed for deterministic implementation.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-001 | Record the accepted owner answers in Section 1.4, including scope, schema authority, SQL parameterization, freshness, provenance, budgets, compatibility, and research boundary. | Yes | 2026-09-24 |
| TASK-002 | Write the approved public contracts for `SchemaManifest`, `TaskRequest`, `QueryPlan`, `GroundedState`, `VerificationReport`, and compatibility behavior in `docs/architecture/csm-environment-contract.md`. | Yes | 2026-09-24 |
| TASK-003 | Define the deterministic fixture format, supported task types, anchor aliases, and the policy for unsupported live metadata without adding source code. | Yes | 2026-09-24 |
| TASK-004 | Freeze Phase 0–8 as the implementation scope and mark Phase 9 agent integration, learned routing, and predictive dynamics as deferred work. | Yes | 2026-09-24 |

**Completion gate:** All clarification rows have an explicit owner decision; no implementation task is started with an unresolved contract.

### Implementation Phase 1 — Versioned Schema Contract

- **GOAL-002**: Create a versioned, machine-readable schema contract while preserving the current flat schema API.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-005 | Create `csm_env/schema_spec/models.py` with frozen `ColumnSpec`, `TableSpec`, `ForeignKeySpec`, `SchemaManifest`, `SchemaSource`, and `SchemaCompatibility` models; require explicit table, primary-key, column, and FK metadata. | Yes | 2026-09-24 |
| TASK-006 | Create `csm_env/schema_spec/registry.py` with `SchemaRegistry.from_static()`, `get_table()`, `get_column()`, `get_outgoing_fk()`, `get_incoming_fk()`, and deterministic indexes; reject duplicate table, PK, or FK definitions. | Yes | 2026-09-24 |
| TASK-007 | Refactor `csm_env/schema.py:7-116` into a compatibility façade over `schema_spec`; preserve `ENTITIES`, `FKS`, `TABLE_COLUMNS`, and all existing import names. | Yes | 2026-09-24 |
| TASK-008 | Create `csm_env/schema_spec/introspector.py` with a `SchemaIntrospector` protocol and an optional `SQLSchemaIntrospector`; return an explicit unsupported result when the backend cannot provide metadata rather than guessing. | Yes | 2026-09-24 |
| TASK-009 | Create `csm_env/schema_spec/validator.py` with `validate_manifest()` checks for table uniqueness, PK presence, column existence, FK endpoint existence, duplicate relationships, and schema-version compatibility. | Yes | 2026-09-24 |
| TASK-010 | Add `csm_env/tests/schema/test_manifest.py` and `test_introspector.py` covering static manifest loading, invalid identifiers, duplicate metadata, and unsupported introspection behavior. | Yes | 2026-09-24 |

**Dependencies:** TASK-001 through TASK-004. **Completion gate:** The 17-table static manifest loads without changing existing imports; invalid metadata fails deterministically; introspection is optional and never silently invents types or constraints.

### Implementation Phase 2 — Structural Schema Graph

- **GOAL-003**: Implement a schema-only graph whose nodes and edges can be validated against the schema registry.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-011 | Create `csm_env/schema_graph/models.py` with frozen `TableNode`, `SchemaEdge`, `SemanticAnnotation`, and edge-direction models; store source/target table and column names on every structural edge. | Yes | 2026-09-24 |
| TASK-012 | Create `csm_env/schema_graph/graph.py` with `SchemaGraph` adjacency indexes, `outgoing()`, `incoming()`, `neighbors()`, `get_edge()`, and schema validation; reject endpoints or relations absent from the registry. | Yes | 2026-09-24 |
| TASK-013 | Create `csm_env/schema_graph/builder.py` with `build_schema_graph(manifest)`; generate structural FK edges first and attach semantic labels only as annotations that reference a structural edge. | Yes | 2026-09-24 |
| TASK-014 | Keep `csm_env/graph.py:7-125` as the record-level graph and document that it cannot be used as the structural schema graph; do not merge record IDs into table nodes. | Yes | 2026-09-24 |
| TASK-015 | Add `csm_env/tests/graph/test_schema_graph.py` covering all manifest tables, all declared FKs, directionality, duplicate-edge rejection, unsupported-edge rejection, and semantic lineage. | Yes | 2026-09-24 |

**Dependencies:** TASK-005 through TASK-009. **Completion gate:** The structural graph contains exactly the registry’s tables and FK-backed edges, has no fabricated relationship, and leaves the existing record graph tests passing.

### Implementation Phase 3 — Safe Transport and Table Adapters

- **GOAL-004**: Add structured, bounded, read-only table access without exposing arbitrary SQL to GGQR.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-016 | Create `csm_env/query/models.py` with frozen `Filter`, `QuerySpec`, `QueryBudget`, `QueryStep`, `QueryResult`, `TaskRequest`, `TaskRequirements`, and route status models. | Yes | 2026-09-24 |
| TASK-017 | Create `csm_env/transport/reader.py` with a `SQLReader` protocol and `EnterpriseOpsSQLRunnerReader`; isolate HTTP response, header, timeout, and error-envelope handling from routing logic. | Yes | 2026-09-24 |
| TASK-018 | Create `csm_env/query/adapters.py` with `TableAdapter.query_by_pk()`, `query_by_fk()`, and `query()`; select columns from the manifest, validate all identifiers, apply limits, and return raw row dictionaries plus query metadata. | Yes | 2026-09-24 |
| TASK-019 | Implement the approved SQL rendering strategy in `csm_env/query/adapters.py`: use bound parameters if the transport supports them; otherwise use one centralized, tested literal renderer that never renders identifiers from user input. | Yes | 2026-09-24 |
| TASK-020 | Preserve `csm_env/sql_runner.py:9-100` and `csm_env/builder.py:28-66` behavior through wrappers; add explicit transport errors for malformed responses and retain `raise_for_status()` semantics. | Yes | 2026-09-24 |
| TASK-021 | Add `csm_env/tests/routing/test_adapters.py` and `csm_env/tests/fixtures/csm_minimal.json`; test PK/FK queries, explicit columns, limits, null values, error envelopes, and rejection of unknown tables/columns. | Yes | 2026-09-24 |

**Dependencies:** TASK-005 through TASK-013 and TASK-001’s SQL-safety decision. **Completion gate:** Every adapter query is schema-validated, read-only, bounded, and testable without a live server; arbitrary SQL cannot enter a router plan.

### Implementation Phase 4 — Deterministic GGQR Router

- **GOAL-005**: Implement bounded Graph-Guided Query Routing with explainable plans and actual-row identifier propagation.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-022 | Create `csm_env/query/requirements.py` with `TaskRequirement`, `RequirementRegistry`, deterministic task-type lookup, and optional explicit caller-supplied requirements; do not call an LLM in the MVP. | Yes | 2026-09-24 |
| TASK-023 | Create `csm_env/query/router.py` with `AnchorResolver.resolve()` and `GraphQueryRouter.route()`; resolve unique aliases, reject ambiguous anchors, and validate the reference value before querying. | Yes | 2026-09-24 |
| TASK-024 | Create `csm_env/query/planner.py` with `QueryPlanner.plan()`; traverse only `SchemaGraph` edges, record edge IDs and direction, reject paths without a registry edge, and support outgoing and explicitly requested incoming traversal. | Yes | 2026-09-24 |
| TASK-025 | Implement identifier propagation in `csm_env/query/router.py`; derive every next binding only from a returned row, distinguish NULL from missing, mark NULL FKs unresolved, and cap one-to-many results. | Yes | 2026-09-24 |
| TASK-026 | Create `csm_env/query/executor.py` with `QueryExecutor.execute()`; execute dependency-ready steps concurrently in bounded waves, cap queries/rows/tables/hops, record latency and step status, and restore deterministic result order. | Yes | 2026-09-24 |
| TASK-027 | Implement router stopping and status rules in `csm_env/query/router.py`: `COMPLETE` when requirements are satisfied, `BUDGET_EXHAUSTED` at a configured limit, `UNRESOLVED` for a required missing binding, and `FAILED` for a transport or schema error. | Yes | 2026-09-24 |
| TASK-028 | Add `csm_env/tests/routing/test_anchor.py`, `test_propagation.py`, `test_budgets.py`, and `test_router.py`; test direct, downstream, upstream, NULL, cycle, ambiguity, requirement, and budget cases. | Yes | 2026-09-24 |

#### GGQR execution contract

1. Validate `TaskRequest`, anchor metadata, schema version, requirements, and budgets.
2. Resolve exactly one anchor table and primary key; reject ambiguity instead of guessing.
3. Execute one bounded anchor query and create the initial frontier from its returned row.
4. For each frontier candidate, select structural edges in this deterministic order: required target first, shortest hop, allowed direction, then stable edge ID.
5. Propagate a target binding only when the source value is non-null and the query result contains a matching row.
6. Create query steps with explicit table, PK/FK filters, selected columns, limit, dependency step ID, and provenance metadata.
7. Validate the complete plan against the structural graph before any non-anchor execution.
8. Execute independent ready steps concurrently, subject to `max_parallel` and the total query/row budgets.
9. Stop on satisfied requirements, exhausted frontier, budget exhaustion, unresolved required binding, or error. Status precedence is `FAILED` > `UNRESOLVED` > `BUDGET_EXHAUSTED` > `COMPLETE`; `COMPLETE` requires all required tables observed with no unresolved required bindings.
10. Return a `QueryPlan`, ordered `QueryResult` records, route diagnostics, and unresolved bindings; never return a value without a source row.

**Dependencies:** TASK-016 through TASK-021. **Completion gate:** A valid request produces a validated explainable plan; an invalid relationship is rejected before execution; budgets are enforced; every propagated identifier is observed in a database row.

### Implementation Phase 5 — Grounded State and Provenance

- **GOAL-006**: Aggregate activated query results into a task-relevant state with complete lineage and explicit unknowns.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-029 | Create `csm_env/state/models.py` with frozen `StateRecord`, `StateRelation`, `GroundedState`, `UnresolvedBinding`, `StateStatus`, and stable state identity fields. | Yes | 2026-09-24 |
| TASK-030 | Create `csm_env/state/provenance.py` with `FactProvenance`, `QueryProvenance`, and redaction helpers; require database ID, schema version, table, column, row PK, query ID, route ID, and UTC observation time for database facts. | Yes | 2026-09-24 |
| TASK-031 | Create `csm_env/state/builder.py` with `StateBuilder.build()`; aggregate anchor and activated records, preserve typed relationships, attach provenance, and never infer a value from an LLM or task text. | Yes | 2026-09-24 |
| TASK-032 | Create `csm_env/state/consistency.py` with checks for returned-row binding agreement, relation endpoint agreement, duplicate PK conflicts, and contradictions; report conflicts without silently resolving them. | Yes | 2026-09-24 |
| TASK-033 | Define freshness and invalidation behavior in `csm_env/state/models.py` and `builder.py` using observation generations, refresh timestamps, and explicit cache replacement; do not treat a cached state as current without a freshness check. | Yes | 2026-09-24 |
| TASK-034 | Add `csm_env/tests/state/test_builder.py`, `test_provenance.py`, and `test_consistency.py`; test NULL, missing, UNKNOWN, duplicate, contradictory, refreshed, and stale observations. | Yes | 2026-09-24 |

**State contract:** `GroundedState` is a task-relevant database-derived view, not the full database and not a self-verifying oracle. Database NULL remains a typed value; a missing or unsupported fact is marked `UNKNOWN`; an unresolved FK is recorded separately from both.

**Dependencies:** TASK-022 through TASK-028. **Completion gate:** Every included fact has valid provenance or an explicit UNKNOWN marker; every relationship points to observed rows; freshness and contradictions are visible to callers.

### Implementation Phase 6 — State Representation and Fidelity

- **GOAL-007**: Render the same grounded state as graph, JSON, and text without changing its facts or provenance.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-035 | Create `csm_env/representation/serializers.py` with versioned graph, JSON, and text serializers, stable ordering, explicit relationship direction, and no secret-bearing transport fields. | Yes | 2026-09-24 |
| TASK-036 | Create `csm_env/representation/fidelity.py` with `verify_fidelity()` for graph/JSON/text round trips and critical-field comparison; define which fields are lossless and which are presentation-only. | Yes | 2026-09-24 |
| TASK-037 | Define text-size and truncation policy in `csm_env/representation/serializers.py`; truncation must be explicit, marked, and must not remove provenance for included facts. | Yes | 2026-09-24 |
| TASK-038 | Add `csm_env/tests/representation/test_serializers.py`; test deterministic ordering, JSON schema, graph round trip, text fidelity, truncation markers, and redaction. | Yes | 2026-09-24 |

**Dependencies:** TASK-029 through TASK-034. **Completion gate:** All three representations reconstruct the same critical facts and preserve provenance; serialization does not mutate the source state.

### Implementation Phase 7 — Independent Oracle and Verification Subsystem

- **GOAL-008**: Verify schema, routing, state, freshness, and graph-to-SQL equivalence against an independent database path.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-039 | Create `csm_env/verification/oracle.py` with `DirectSqlOracle`; build explicit direct queries from approved table/column metadata without importing `SchemaGraph`, `GraphQueryRouter`, `StateBuilder`, or their helpers. | Yes | 2026-09-24 |
| TASK-040 | Create `csm_env/verification/schema_checks.py` implementing V1–V3: table completeness, column/PK completeness, FK endpoint/relation validation, and typed mismatch reports. | Yes | 2026-09-24 |
| TASK-041 | Create `csm_env/verification/relation_checks.py` implementing V4–V6: referential-integrity spot checks, graph-to-SQL equivalence, and pre-execution query-plan validation. | Yes | 2026-09-24 |
| TASK-042 | Create `csm_env/verification/grounding_checks.py` implementing V7–V10: identifier propagation, fact grounding, state consistency, and freshness/rebuild checks. | Yes | 2026-09-24 |
| TASK-043 | Create `csm_env/verification/integration_checks.py` with a comparison artifact that reports exact matches, missing facts, extra facts, unresolved bindings, and transport/schema errors separately. | Yes | 2026-09-24 |
| TASK-044 | Add `csm_env/tests/verification/test_oracle.py`, `test_schema_checks.py`, `test_relation_checks.py`, and `test_grounding_checks.py` with independent fake readers and fixed clocks. | Yes | 2026-09-24 |
| TASK-045 | Add `csm_env/tests/verification/test_live_opt_in.py` only if an approved metadata and SQL endpoint is supplied; skip it explicitly in the default offline suite. | Yes | 2026-09-24 |

**Dependencies:** TASK-005 through TASK-043. **Completion gate:** Verification never validates the router using the router’s own graph, and all V1–V10 checks return structured pass/fail evidence with a reason for every failure.

### Implementation Phase 8 — Public API Integration, Compatibility, and Documentation

- **GOAL-009**: Expose the new capabilities through additive APIs while preserving current package consumers.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-046 | Extend `csm_env/api.py:8-53` with explicit router/state/verification entry points while retaining `get_ground_truth()`, `get_case_context()`, `get_neighbors()`, and `search_cases()` behavior. | Yes | 2026-09-24 |
| TASK-047 | Update `csm_env/builder.py:17-261` only where needed to inject the schema registry, router, state builder, and reader; keep current record graph behavior isolated from the structural graph. | Yes | 2026-09-24 |
| TASK-048 | Update `csm_env/integration.py:9-22` to construct the new services through dependency injection while retaining the external `MCPClient` attribute boundary and avoiding imports from the host package. | Yes | 2026-09-24 |
| TASK-049 | Update `csm_env/__init__.py:1-19` and `__all__` with only approved public types and service factories; do not expose internal planner or transport helpers by default. | Yes | 2026-09-24 |
| TASK-050 | Update `README.md:1-86` with the schema/router/state flow, compatibility guarantees, freshness semantics, provenance contract, verification commands, and the distinction between record graph and schema graph. | Yes | 2026-09-24 |
| TASK-051 | Add `csm_env/tests/integration/test_csm_end_to_end.py` and compatibility tests for every existing public import and method return shape. | Yes | 2026-09-24 |
| TASK-052 | Run focused tests first, then the complete suite, import smoke test, and syntax check; record results in the implementation handoff. | Yes | 2026-09-24 |

**Dependencies:** TASK-005 through TASK-045. **Completion gate:** Existing tests and imports remain green; new APIs are additive; the README describes the implemented contracts without claiming deferred features.

### Implementation Phase 9 — Deferred Agent Integration and Experimental Extensions

- **GOAL-010**: Connect the verified state API to agents and experiments only after Phase 0–8 gates pass.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-053 | Define an agent-facing state contract and keep the agent implementation unchanged for the first experiment. | Deferred | - |
| TASK-054 | Add the proposed ablations: direct agent, flat retrieval, schema graph routing, and graph routing plus provenance, with query-count, latency, token-cost, and state-quality metrics. | Deferred | - |
| TASK-055 | Add action-to-state transition records and predictive dynamics only after transition verification is specified and approved. | Deferred | - |

**Dependencies:** TASK-052 and explicit owner approval to begin Phase 9. No task in this phase is part of the initial implementation release.

## 3. Alternatives

- **ALT-001**: Continue adding all responsibilities to the existing flat modules. Rejected because it would blur schema, graph, transport, routing, state, and verification boundaries and make replacement difficult.
- **ALT-002**: Use purpose-specific subpackages with flat compatibility wrappers. Selected because it follows the blueprint architecture while preserving current imports.
- **ALT-003**: Make runtime schema introspection mandatory before any routing. Rejected for the MVP because metadata availability is unconfirmed; selected design is static manifest first with optional fail-closed introspection.
- **ALT-004**: Use a record-level graph as the schema graph. Rejected because table structure and database records have different identities, lifecycles, and correctness rules.
- **ALT-005**: Use an LLM to select tables and identifiers. Rejected for the deterministic MVP; selected design separates optional task interpretation from graph-validated routing and database retrieval.
- **ALT-006**: Let GGQR accept generated SQL. Rejected because it permits fabricated relationships and bypasses schema grounding; selected design accepts structured `QuerySpec` objects only.
- **ALT-007**: Materialize the full CSM database into the record graph. Rejected for normal routing; selected design uses bounded case/task state and retains `snapshot_all()` only as a separately controlled compatibility operation.
- **ALT-008**: Modify the ZIP or `zip_extract/` copy to parallel new work. Rejected because it creates two active sources; selected work remains in root `csm_env/`.

## 4. Dependencies

- **DEP-001**: Python 3.11 or newer.
- **DEP-002**: `httpx` for the live EnterpriseOps SQL transport.
- **DEP-003**: `pytest` and `pytest-asyncio` for the test suite.
- **DEP-004**: The external EnterpriseOps-Gym `MCPClient` object contract, accessed only through `csm_env/integration.py`.
- **DEP-005**: An approved schema manifest or an authoritative metadata source for runtime introspection.
- **DEP-006**: A deterministic synthetic fixture; an external CSM dump is optional and must be supplied through an approved path.
- **DEP-007**: A deterministic task-requirement registry or an approved structured task-requirement provider.
- **DEP-008**: No additional package dependency is approved for the first release; standard-library modules are preferred for models, protocols, serialization, and verification.

Dependency order:

`TASK-001` → `TASK-005` → `TASK-011` → `TASK-016` → `TASK-022` → `TASK-029` → `TASK-035` → `TASK-039` → `TASK-046`.

The live endpoint and host MCP integration are optional dependencies for offline implementation and tests. They must not block the Phase 0–8 acceptance gates unless the owner changes the scope decision in Section 1.4.

## 5. Files

- **FILE-001**: `csm_env/schema.py` — preserve the static schema public API as a compatibility façade.
- **FILE-002**: `csm_env/schema_spec/__init__.py` — export approved schema-contract types.
- **FILE-003**: `csm_env/schema_spec/models.py` — define versioned table, column, FK, and manifest models.
- **FILE-004**: `csm_env/schema_spec/registry.py` — provide indexed schema lookup and validation.
- **FILE-005**: `csm_env/schema_spec/introspector.py` — define optional metadata discovery.
- **FILE-006**: `csm_env/schema_spec/validator.py` — implement manifest integrity checks.
- **FILE-007**: `csm_env/schema_graph/__init__.py` — export structural graph types.
- **FILE-008**: `csm_env/schema_graph/models.py` — define table nodes, FK edges, and semantic annotations.
- **FILE-009**: `csm_env/schema_graph/graph.py` — implement structural adjacency and validation.
- **FILE-010**: `csm_env/schema_graph/builder.py` — construct the graph from a manifest.
- **FILE-011**: `csm_env/graph.py` — retain the record-level graph and its existing public behavior.
- **FILE-012**: `csm_env/query/__init__.py` — export approved query-domain types.
- **FILE-013**: `csm_env/query/models.py` — define filters, budgets, plans, steps, results, and requests.
- **FILE-014**: `csm_env/query/adapters.py` — implement schema-bound table adapters.
- **FILE-015**: `csm_env/query/requirements.py` — define deterministic task requirements.
- **FILE-016**: `csm_env/query/router.py` — implement anchor resolution, propagation, and GGQR orchestration.
- **FILE-017**: `csm_env/query/planner.py` — build and validate structural query plans.
- **FILE-018**: `csm_env/query/executor.py` — execute bounded asynchronous query waves.
- **FILE-019**: `csm_env/transport/__init__.py` — export transport protocols.
- **FILE-020**: `csm_env/transport/reader.py` — isolate live and fake SQL readers.
- **FILE-021**: `csm_env/sql_runner.py` — preserve the existing HTTP adapter as a transport façade.
- **FILE-022**: `csm_env/state/__init__.py` — export approved state types.
- **FILE-023**: `csm_env/state/models.py` — define grounded state and freshness models.
- **FILE-024**: `csm_env/state/provenance.py` — define fact/query provenance and redaction.
- **FILE-025**: `csm_env/state/builder.py` — aggregate observed records into state.
- **FILE-026**: `csm_env/state/consistency.py` — validate state relationships and freshness.
- **FILE-027**: `csm_env/representation/__init__.py` — export serializers and fidelity APIs.
- **FILE-028**: `csm_env/representation/serializers.py` — implement graph, JSON, and text output.
- **FILE-029**: `csm_env/representation/fidelity.py` — implement representation round-trip checks.
- **FILE-030**: `csm_env/verification/__init__.py` — export verification result types.
- **FILE-031**: `csm_env/verification/oracle.py` — implement independent direct-SQL comparisons.
- **FILE-032**: `csm_env/verification/schema_checks.py` — implement V1–V3 checks.
- **FILE-033**: `csm_env/verification/relation_checks.py` — implement V4–V6 checks.
- **FILE-034**: `csm_env/verification/grounding_checks.py` — implement V7–V10 checks.
- **FILE-035**: `csm_env/verification/integration_checks.py` — implement end-to-end comparison reports.
- **FILE-036**: `csm_env/api.py` — add additive router, state, and verification entry points.
- **FILE-037**: `csm_env/builder.py` — inject new services without breaking existing hydration.
- **FILE-038**: `csm_env/integration.py` — preserve the host MCP boundary while wiring services.
- **FILE-039**: `csm_env/__init__.py` — update approved public exports and `__all__`.
- **FILE-040**: `csm_env/tests/conftest.py` — provide deterministic fake readers, clocks, and fixtures.
- **FILE-041**: `csm_env/tests/schema/test_manifest.py` — test schema contract loading and validation.
- **FILE-042**: `csm_env/tests/schema/test_introspector.py` — test metadata discovery and unsupported behavior.
- **FILE-043**: `csm_env/tests/graph/test_schema_graph.py` — test structural graph invariants.
- **FILE-044**: `csm_env/tests/routing/test_anchor.py` — test anchor resolution.
- **FILE-045**: `csm_env/tests/routing/test_propagation.py` — test identifier propagation and NULL handling.
- **FILE-046**: `csm_env/tests/routing/test_budgets.py` — test hop, query, row, and parallel budgets.
- **FILE-047**: `csm_env/tests/routing/test_router.py` — test deterministic end-to-end GGQR behavior.
- **FILE-048**: `csm_env/tests/state/test_builder.py` — test grounded state aggregation.
- **FILE-049**: `csm_env/tests/state/test_provenance.py` — test mandatory provenance and redaction.
- **FILE-050**: `csm_env/tests/state/test_consistency.py` — test consistency and freshness checks.
- **FILE-051**: `csm_env/tests/representation/test_serializers.py` — test representation fidelity and ordering.
- **FILE-052**: `csm_env/tests/verification/test_oracle.py` — test oracle independence and direct SQL results.
- **FILE-053**: `csm_env/tests/verification/test_checks.py` — test V1–V10 verification reports.
- **FILE-054**: `csm_env/tests/integration/test_csm_end_to_end.py` — test the complete offline pipeline.
- **FILE-055**: `csm_env/tests/fixtures/csm_minimal.json` — provide synthetic, non-sensitive table and relationship data.
- **FILE-056**: `README.md` — document the implemented architecture and compatibility behavior.
- **FILE-057**: `plan/architecture-csm-environment-1.md` — contain this versioned implementation plan.
- **FILE-058**: `pyproject-snippet.toml` — remain unchanged unless packaging scope is explicitly approved.
- **FILE-059**: `csm_env/tests/verification/test_live_opt_in.py` — house explicitly skipped live verification tests when an approved backend is available.
- **FILE-060**: `docs/architecture/csm-environment-contract.md` — record approved schema, routing, state, provenance, representation, and verification contracts.

## 6. Testing

### 6.1 Test layers

- **TEST-001**: Schema manifest tests verify all declared tables, primary keys, columns, FKs, duplicate detection, and version metadata.
- **TEST-002**: Schema graph tests verify table-node completeness, directionality, FK-backed edge completeness, semantic lineage, and rejection of fabricated edges.
- **TEST-003**: Adapter tests verify explicit columns, schema-bound identifiers, PK/FK filters, limits, NULL handling, error envelopes, and read-only behavior.
- **TEST-004**: Anchor tests verify explicit type resolution, unique aliases, ambiguous references, missing references, and invalid identifiers.
- **TEST-005**: Propagation tests verify outgoing and incoming FK traversal, cycles, one-to-many caps, NULL bindings, and refusal to use task-text-invented IDs.
- **TEST-006**: Plan tests verify that every hop is a structural edge and that invalid paths are rejected before execution.
- **TEST-007**: Budget tests verify `max_hops`, `max_tables`, `max_queries`, `max_rows_per_query`, `max_total_rows`, and `max_parallel` enforcement.
- **TEST-008**: Executor tests verify asynchronous independent execution, dependency ordering, deterministic result ordering, cancellation, and error status propagation.
- **TEST-009**: State tests verify record aggregation, relationship preservation, UNKNOWN semantics, unresolved bindings, duplicate detection, and contradiction reporting.
- **TEST-010**: Provenance tests verify every included fact has database/schema/table/column/row/query/route/time lineage and that secrets are redacted.
- **TEST-011**: Freshness tests verify observation generations, refresh behavior, stale-state detection, and rebuild-after-change behavior using a mutable fake reader.
- **TEST-012**: Representation tests verify deterministic graph/JSON/text output, critical-field fidelity, truncation markers, and round trips.
- **TEST-013**: Oracle tests verify direct SQL results are independent of router and state code and use explicit query contracts.
- **TEST-014**: Verification tests cover V1–V10 with pass, mismatch, missing-row, unsupported-metadata, and transport-error cases.
- **TEST-015**: End-to-end tests compare task request → validated plan → executed rows → grounded state → representation against direct SQL results.
- **TEST-016**: Compatibility tests verify every existing public import and existing API return shape remains valid.
- **TEST-017**: Optional live tests are marked and skipped unless an approved live endpoint, database ID, and test data are supplied.

### 6.2 Required commands

Run from the repository root:

```powershell
python -m pytest csm_env/tests -v -p no:cacheprovider
python -m pytest csm_env/tests/test_graph.py::test_graph_neighbors_and_subgraph -v -p no:cacheprovider
python -m pytest csm_env/tests/test_csm_representation.py::test_case_context_is_connected_and_grounded -v -p no:cacheprovider
python -c "from csm_env import EnterpriseOpsSQLRunner, CSMEnvironmentRepresentation, CSMEnvironmentAPI, from_enterpriseops_mcp_client; print('IMPORT OK')"
```

Run each new test module by node ID during its phase, then run the complete suite after every phase. Do not use `-p no:asyncio` for normal verification. No local lint or type-check command is configured; use the host repository’s configured commands only after integration.

### 6.3 Definition of done

A phase is complete only when its focused tests pass, the complete existing suite passes, public imports pass, and the phase’s completion gate is satisfied. A test that cannot run because an optional live dependency is unavailable must be skipped explicitly and must not be reported as a passing verification result.

## 7. Risks & Assumptions

### 7.1 Risks

- **RISK-001**: The static manifest may diverge from the target database schema, causing valid queries to fail or invalid columns to appear valid.
- **RISK-002**: The SQL-runner transport may not support bound parameters, making safe literal rendering a backend-specific security boundary.
- **RISK-003**: Incoming foreign keys and non-unique relationships can create fan-out, duplicate records, or ambiguous identifier propagation.
- **RISK-004**: NULL, missing rows, unsupported metadata, and transport failures may be conflated if status models are not explicit.
- **RISK-005**: Persistent record graphs can retain stale nodes and edges unless observation generations and invalidation are implemented.
- **RISK-006**: Concurrent query completion order can make state and diagnostics nondeterministic unless results are sorted by plan step ID.
- **RISK-007**: Backward-compatible wrappers may preserve unsafe raw SQL or private coupling unless the API boundary is reviewed.
- **RISK-008**: A verification oracle that imports router, planner, or state helpers can become circular and falsely validate the system.
- **RISK-009**: Missing or inconsistent table metadata can cause identifier-rendering errors even when values are escaped correctly.
- **RISK-010**: The absence of package metadata and a local host checkout makes dependency upgrades and live integration behavior difficult to reproduce.

### 7.2 Assumptions

- **ASSUMPTION-001**: The current static schema registry is the initial schema contract until the owner supplies authoritative DDL or metadata.
- **ASSUMPTION-002**: The existing `/api/sql-runner` transport accepts a query string; structured filters and a centralized literal renderer are the first safe implementation.
- **ASSUMPTION-003**: The first router uses a deterministic requirement registry and does not require an LLM at runtime.
- **ASSUMPTION-004**: `GroundedState` is a case-local, task-relevant subset of current database state and does not replace the full database.
- **ASSUMPTION-005**: The provisional budget defaults in Section 1.4 are safe starting values and remain configurable per request.
- **ASSUMPTION-006**: A synthetic fixture is sufficient for the first implementation gates; live verification is optional.
- **ASSUMPTION-007**: The project remains research-internal, read-only, and does not import the external host package directly.
- **ASSUMPTION-008**: Existing public API compatibility is required until an explicit versioned migration is approved.

## 8. Related Specifications / Further Reading

- `README.md:1-86` — current package purpose, integration model, and ground-truth/cache distinction.
- `AGENTS.md:1-164` — repository commands and coding conventions.
- `csm_env/schema.py:7-116` — current static entity, FK, and column registry.
- `csm_env/graph.py:7-125` — current record-level property graph.
- `csm_env/sql_runner.py:9-100` — current SQL-runner transport and response normalization.
- `csm_env/builder.py:17-261` — current hydration and record-graph construction flow.
- `csm_env/api.py:8-53` — current public API surface.
- `csm_env/integration.py:9-22` — current external MCP-client boundary.
- User-supplied Blueprint A — schema-grounded graph, adapters, GGQR routing, provenance, and V1–V10 verification.
- User-supplied Blueprint B — environment representation, state construction, state representation, independent oracle, and experimental layers.
- User-supplied Blueprint C — formal GGQR algorithm, identifier propagation, stopping criteria, budgets, and graph-to-SQL equivalence.
- No implementation task may treat agent integration, learned dynamics, or external publication as complete until its deferred phase and approval gate are satisfied.
