# CSM Environment Representation Layer

This package adds a **database-backed environment representation** for the
ServiceNow EnterpriseOps-Gym CSM domain, now including the schema-grounded
**GGQR** pipeline (Graph-Guided Query Routing) with grounded state,
provenance, representations, and an independent verification subsystem.

The design is deliberately aligned with the public EnterpriseOps-Gym codebase:

- the benchmark runs CSM through the `sn-csm-server` MCP server;
- each run uses a specific `database_id`;
- the benchmark's own verifier reads ground truth through
  `<mcp_server_url>/api/sql-runner` with the `x-database-id` header;
- this layer uses the same SQL-runner path, so the **database remains the
  source of truth** for record values.

## Architecture

```text
        Schema manifest (17 tables, PKs, columns, FKs)
                       |
                       v
        Structural SchemaGraph (TableNodes + FK SchemaEdges)
                       |            (semantic labels = annotations only)
                       v
        Deterministic GGQR router (anchor -> plan -> FK propagation)
                       |            (identifiers come only from returned rows)
                       v
        Bounded parallel table adapters -> parameter-shaped QuerySpecs
                       |
                       v
        GroundedState (records + typed relations + provenance + unresolved)
                       |
              graph / JSON / text (same S_t, reconstruction-fidelity-checked)
                       |
                       v
        Independent DirectSqlOracle verification (V1-V10)
```

Core distinction: the **schema graph** describes table structure and declared
foreign keys; the **record graph** (`EnvironmentGraph`) is a bounded, derived
cache. The two are never merged. Raw SQL is never constructed or accepted by
GGQR; adapters render structured `QuerySpec` objects through one centralized,
tested literal renderer, and identifiers always come from the schema registry.

## Layout

```text
csm_env/
  schema.py           compatibility facade (ENTITIES, FKS, TABLE_COLUMNS)
  schema_spec/        versioned manifest, registry, optional introspector, validator
  schema_graph/       structural graph: TableNode, SchemaEdge, annotations
  query/              QuerySpec/TaskRequest/plans, adapters, GGQR router, executor
  query/cost.py       Cost(S) = λ_r·rows + λ_t·tokens + λ_q·queries + argmin prefix
  query/coverage.py   fact-level Coverage(S, F_T) over anchor/table/attr/relation
  transport/          SQLReader protocol + EnterpriseOps wrapper (typed errors)
  state/              GroundedState, provenance, consistency, freshness
  representation/     graph/JSON/text serializers + fidelity checks
  verification/       independent oracle + V1-V10 checks
  environment.py      GGQREnvironment service composition
  builder.py          legacy record-graph hydration (unchanged behavior)
  api.py              stable facade + additive GGQR entry points
  integration.py      MCPClient adapter (optional enable_ggqr=True)
```

`query/cost.py` and `query/coverage.py` are what make the State Model 2.0
objective real rather than descriptive: the router retrieves incrementally
(`RouteStrategy.FRONTIER`), measures `Cost(S)` and `Coverage(S, F_T)` at every
prefix, and stops at the cheapest covering prefix. Condition B2 uses that
strategy; Condition B1 keeps the eager `BROAD` path unchanged.

## Quick start (legacy surface - unchanged)

```python
from csm_env import EnterpriseOpsSQLRunner, CSMEnvironmentRepresentation, CSMEnvironmentAPI

sql = EnterpriseOpsSQLRunner(base_url="http://localhost:8001", database_id="YOUR_DATABASE_ID")
env = CSMEnvironmentRepresentation(sql)
api = CSMEnvironmentAPI(env)

case = await api.get_ground_truth("customer_case", 1233)   # exact current row
context = await api.get_case_context(1233, hops=2)          # semantic neighborhood
```

## GGQR pipeline (additive)

```python
from csm_env import EnterpriseOpsSQLRunner, from_enterpriseops_mcp_client, GGQREnvironment, TaskRequest
from csm_env.transport.reader import EnterpriseOpsSQLRunnerReader

sql = EnterpriseOpsSQLRunner(base_url="http://localhost:8001", database_id="YOUR_DATABASE_ID")
ggqr = GGQREnvironment(EnterpriseOpsSQLRunnerReader(sql), database_id="YOUR_DATABASE_ID")

# Deterministic route: anchor -> validated plan -> bounded parallel queries
outcome = await ggqr.route(
    TaskRequest(task_type="resolve_case", reference_type="customer_case", reference_id=1233)
)

# Grounded state with per-fact provenance
state = await ggqr.build_state_for("resolve_case", 1233, "customer_case")
fact = state.facts[0]
print(fact.value, fact.provenance.table, fact.provenance.column, fact.provenance.row_pk)

# Same S_t as graph / JSON / text, fidelity-checked
rendered = ggqr.represent(state)          # {"json": ..., "graph": ..., "text": ...}
report = ggqr.fidelity(state)             # round-trip fidelity result

# Independent verification (direct SQL, never the router's own graph)
result = await ggqr.verify_case_equivalence(1233)
assert result["equivalent"]
```

Or enable it through the MCP adapter and the stable API:

```python
env = from_enterpriseops_mcp_client(mcp_client, enable_ggqr=True)
api = CSMEnvironmentAPI(env, ggqr_environment=env.ggqr)
outcome = await api.route_state("resolve_case", 1233, "customer_case")
state = await api.build_state("resolve_case", 1233, "case")
```

Task types registered by default: `resolve_case`, `case_overview`,
`check_entitlement`, `find_knowledge`, `account_profile`. Unknown task types
fail closed. Budgets (configurable per request): `max_hops=2`,
`max_tables=12`, `max_queries=32`, `max_rows_per_query=50`,
`max_total_rows=500`, `max_parallel=8`. `max_total_rows` is enforced as a strict reservation cap across concurrent queries; a query is never launched when its reserved row allowance would exceed the route budget.

## Provenance and freshness contracts

Every database-derived fact carries `database_id`, `schema_version`, `table`,
`column`, `row_pk`, `query_id`, `route_id`, and a timezone-aware UTC
`observed_at`. Facts without a source row are never included; a missing fact
is UNKNOWN, never guessed. Database NULL stays a typed value; NULL or dangling
FKs are recorded as `UnresolvedBinding`s, separate from both. States carry an
observation generation; a state rebuilt after a database change reflects the
new values, and cached states must not be treated as current without a
freshness check. `get_ground_truth()` remains the fresh-read API for
evaluators.

## Compatibility guarantees

- `ENTITIES`, `FKS`, `TABLE_COLUMNS`, `EnvironmentGraph`, `Node`, `Edge`,
  `EnterpriseOpsSQLRunner`, `CSMEnvironmentRepresentation`,
  `CSMEnvironmentAPI`, and `from_enterpriseops_mcp_client` keep their public
  shapes and behavior.
- `csm_env/schema.py` is now a facade over `schema_spec`; its data is unchanged.
- New capabilities are additive (`GGQREnvironment`, `build_state`,
  `route_state`, `represent_state`, verification entry points).

## Verification (V1-V10)

Verification is a separate subsystem using an independent
`DirectSqlOracle` that never imports router/planner/state code:

- V1-V3 schema checks (tables, columns/PKs, FK endpoints) against metadata;
- V4 referential-integrity spot checks, V5 graph-to-SQL equivalence,
  V6 pre-execution plan validation;
- V7 identifier-propagation grounding (`ID_next = DB(row, FK)`),
  V8 fact grounding/provenance completeness, V9 state consistency,
  V10 freshness/rebuild-after-change.

Offline tests run against a synthetic fixture; live-endpoint checks are
opt-in via `CSM_LIVE_BASE_URL` / `CSM_LIVE_DATABASE_ID` and are explicitly
skipped otherwise.

## Commands

```powershell
python -m pytest csm_env/tests -v -p no:cacheprovider
python -m pytest csm_env/tests/test_graph.py::test_graph_neighbors_and_subgraph -v -p no:cacheprovider
python -m pytest csm_env/tests/test_csm_representation.py::test_case_context_is_connected_and_grounded -v -p no:cacheprovider
python -c "from csm_env import EnterpriseOpsSQLRunner, CSMEnvironmentRepresentation, CSMEnvironmentAPI, from_enterpriseops_mcp_client; print('IMPORT OK')"
```

Do not use `-p no:asyncio` for normal verification. The approved contracts
live in `docs/architecture/csm-environment-contract.md` and
`docs/architecture/csm-environment-fixtures.md`; the implementation plan is
`plan/architecture-csm-environment-1.md`.

## 2026-09-24 research-hardening update

This revision fixes the issues identified during research review:

1. **Multi-ID FK fan-out**: propagated identifiers are rendered as a single
   `IN (...)` predicate rather than contradictory `column=a AND column=b`
   filters. This supports one-to-many relationships.
2. **Intermediate-table discovery**: GGQR now discovers shortest structural
   FK paths to required target tables and automatically inserts bridge tables.
   Callers do not need to enumerate join intermediates.
3. **Ambiguous path handling**: when equal-length paths exist, already-required
   tables are preferred as routing anchors so task-relevant fan-out paths are
   preserved deterministically.
4. **Task requirement loading**: requirements can be loaded from a benchmark
   JSON manifest with `RequirementRegistry.from_json(...)`, avoiding a hard
   dependency on the five demo task names.
5. **Contradiction safety**: duplicate observations of the same logical record
   are compared field-by-field; conflicts are recorded explicitly and mark the
   grounded state as `FAILED` rather than being silently overwritten.
6. **Runtime type verification**: row snapshots can be checked against the
   manifest with `check_v2_runtime_row_types(...)`. Identifier columns accept
   numeric DB-driver materialization as well as strings; `coverage_hours` is
   modeled as `string` because values such as `24x7`/`8x5` are non-numeric.
7. **Snapshot validation**: `check_v1_v2_snapshot(...)` validates table coverage
   and runtime row types for row-only snapshots. Structural V2 metadata is
   explicitly marked `skipped` when a snapshot does not carry column/PK
   metadata instead of claiming structural verification that was not possible.
8. **Research verification remains independent**: the SQL oracle is still a
   separate ground-truth branch; GGQR never verifies its own output using its
   schema graph.

The package still keeps the relational database as the source of truth. The
schema graph contains table/relationship metadata, while query results remain
live or snapshot-backed database data.
