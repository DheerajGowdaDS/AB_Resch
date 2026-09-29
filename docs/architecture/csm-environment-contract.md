# CSM Environment Contracts

Status: Approved (2026-09-24) per `plan/architecture-csm-environment-1.md` Section 1.4.
This document is the implementation contract for Phases 0-8. Phase 9 (agent
integration, learned routing, predictive dynamics) is deferred.

## 1. Schema contracts (`csm_env.schema_spec`)

### 1.1 `SchemaManifest`

A versioned, immutable, machine-readable description of the CSM relational
schema. Fields:

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | `str` | Version tag of this manifest, e.g. `"1.0.0"`. |
| `tables` | `tuple[TableSpec, ...]` | One entry per registered table, sorted by table name. |
| `foreign_keys` | `tuple[ForeignKeySpec, ...]` | One entry per declared FK, sorted by `(source_table, source_column, target_table)`. |
| `source` | `SchemaSource` | Where this manifest came from. |
| `derived_from` | `str` | Identifier of the registry/report the manifest was derived from. |

### 1.2 `TableSpec`

| Field | Type | Constraint |
|---|---|---|
| `table` | `str` | Lowercase SQL table name; unique in a manifest. |
| `node_type` | `str` | PascalCase graph label; unique in a manifest. |
| `primary_key` | `str` | Must appear in `columns`. |
| `columns` | `tuple[ColumnSpec, ...]` | Explicit, ordered, non-empty; unique column names. |

### 1.3 `ColumnSpec`

| Field | Type | Constraint |
|---|---|---|
| `name` | `str` | Lowercase column name; unique inside its table. |
| `type` | `str` | One of the portable type names: `"string"`, `"int"`, `"float"`, `"bool"`, `"datetime"`, `"text"`, `"unknown"`. |

Column type is descriptive metadata. Value interpretation at runtime is
permissive; the type must never be silently invented by the introspector.

### 1.4 `ForeignKeySpec`

| Field | Type | Constraint |
|---|---|---|
| `source_table`, `source_column` | `str` | Must exist in the manifest. |
| `target_table`, `target_column` | `str` | Must exist; the target column must be a primary key or a unique key (by convention, the PK). |
| `relation` | `str` | UPPERCASE_SNAKE semantic label; unique per `(source_table, source_column)` pair. |

### 1.5 `SchemaSource` and `SchemaCompatibility`

- `SchemaSource` is one of the frozen singletons: `STATIC_REGISTRY` (derived
  from `csm_env/schema.py`) or `RUNTIME_INTROSPECTION`.
- `SchemaCompatibility` records the schema version a consumer was built
  against, the manifest version it received, and a boolean `compatible`.
  Compatibility is major-version equality. It is never assumed silently:
  callers that request a specific version get an explicit
  `SchemaCompatibility`, and a mismatch fails closed in routing paths.

### 1.6 `SchemaRegistry`

- `SchemaRegistry.from_static()` builds the registry from the current static
  17-table registry in `csm_env/schema.py`.
- Lookup API: `get_table(name)`, `get_column(table, name)`,
  `get_outgoing_fk(table)`, `get_incoming_fk(table)`,
  `tables()`, `foreign_keys()`, `manifest`.
- Duplicated table names, primary keys, or FK definitions raise `ValueError`
  at construction time.
- Names are normalized to lowercase for lookup; IDs keep their source form.

## 2. Query contracts (`csm_env.query`, `csm_env.transport`)

### 2.1 `TaskRequest`

The approved GGQR input is a **structured** request:

| Field | Type | Meaning |
|---|---|---|
| `task_type` | `str` | Key into the deterministic requirement registry. |
| `reference_type` | `str \| None` | Table name or node type of the anchor. Required when the reference is ambiguous. |
| `reference_id` | `Any` | Anchor identifier value. |
| `task_text` | `str \| None` | Optional natural-language text, retained as metadata only. Never parsed for identifiers or requirements in the MVP. |
| `budget` | `QueryBudget \| None` | Optional per-request budget override. |

### 2.2 `QueryBudget`

Configurable defaults (accepted 2026-09-24): `max_hops=2`, `max_tables=12`,
`max_queries=32`, `max_rows_per_query=50`, `max_total_rows=500`,
`max_parallel=8`. All are enforced by the executor; a step that would exceed a
budget is not started and the route reports `BUDGET_EXHAUSTED`.

### 2.3 `QuerySpec` and SQL rendering

- A query is always a structured `QuerySpec`: explicit table, explicit column
  list from the manifest, structured `Filter` list, and a row limit.
- GGQR and the planner never construct, accept, or emit raw SQL strings.
- The transport renders SQL in one centralized, tested location
  (`csm_env/query/adapters.py` renderer helpers). Identifiers come only from
  the schema registry. Values are rendered through the shared literal
  escaper. If a future transport supports bound parameters, the renderer is
  replaced wholesale behind the same `QuerySpec` interface.
- Raw SQL remains available only through the legacy `CSMEnvironmentRepresentation.query()`
  escape hatch, which GGQR never uses.

### 2.4 `SQLReader` protocol

`csm_env/transport/reader.py` defines the reader protocol used by adapters:

```python
class SQLReader(Protocol):
    async def fetch_rows(self, query: str) -> list[dict[str, Any]]: ...
```

`EnterpriseOpsSQLRunnerReader` wraps the existing `EnterpriseOpsSQLRunner`;
malformed responses raise a typed `TransportError` instead of returning data
of the wrong shape. Tests inject deterministic fakes.

## 3. Route contracts (`csm_env.query.router`, `.planner`, `.executor`)

### 3.1 Route statuses

`RouteStatus`: `COMPLETE`, `BUDGET_EXHAUSTED`, `UNRESOLVED`, `FAILED`.

Precedence when several apply: `FAILED` > `UNRESOLVED` > `BUDGET_EXHAUSTED` >
`COMPLETE`. `COMPLETE` requires every required table of the task's
requirements to be observed with no unresolved required bindings.

### 3.2 Anchor resolution

- Resolution order: explicit `reference_type` first; if absent or ambiguous,
  deterministic alias lookup across table names and node types.
- An alias that matches multiple tables is rejected with `ValueError`
  (`ambiguous reference`); it is never guessed.
- The reference value must be non-empty; `None` and `""` are rejected.

### 3.3 Query plan and identifier propagation

- A `QueryPlan` is an immutable object listing anchor, ordered `QueryStep`
  records (each with table, filters, columns, limit, dependency step ID, and
  the edge ID it follows), budgets, and route metadata.
- Every hop in a plan must correspond to a structural edge in the
  `SchemaGraph`; plans containing paths with no registry edge are rejected
  before execution.
- Identifier propagation rule (`ID_next = DB(row, FK)`): a binding for a
  target table is created only when (a) the source row value for the FK
  column is non-null, and (b) the query result contains at least one matching
  row. NULL FK values produce an `UnresolvedBinding`; missing values are
  distinct from NULL and are also unresolved. One-to-many results are capped
  at `QueryBudget.max_rows_per_query` bindings per FK.
- No identifier may be taken from `task_text`, LLM output, or any source other
  than a returned database row.

### 3.4 Execution

- Dependency-ready steps run concurrently in bounded waves
  (`asyncio.gather`), capped at `max_parallel`; results are re-sorted by step
  ID so output is deterministic regardless of completion order.
- Per-step status (`ok`, `empty`, `error`, `skipped_budget`) and latency are
  recorded on each `QueryResult`.

## 4. State contracts (`csm_env.state`)

### 4.1 `GroundedState`

| Field | Type | Meaning |
|---|---|---|
| `state_id` | `str` | Deterministic identity: anchor table + id + schema version. |
| `anchor` | `StateRecord` | The anchor row record. |
| `records` | `tuple[StateRecord, ...]` | All observed rows, sorted by `(table, id)`. |
| `relations` | `tuple[StateRelation, ...]` | Typed relationships between observed records, each carrying the FK edge ID it followed. |
| `unresolved` | `tuple[UnresolvedBinding, ...]` | NULL/missing FK bindings and required-but-unobserved tables. |
| `provenance` | `StateProvenance` | Shared route/database/query lineage. |
| `status` | `StateStatus` | Aggregated route status. |
| `observed_at` / `schema_version` / `database_id` | | Freshness and binding metadata. |

### 4.2 Value semantics (CON-011)

- Database `NULL` is a typed value: it appears in records as `None` with
  provenance.
- A fact that is required but not observed is `UNKNOWN`; it is never guessed.
- An unresolved FK (NULL or dangling) is an `UnresolvedBinding`, recorded
  separately from both.
- Contradictions (e.g. duplicate PKs with different values, relation endpoint
  disagreement) are reported by `csm_env/state/consistency.py` and surfaced on
  the state; they are never silently resolved.

### 4.3 Provenance (mandatory fields)

Every database-derived fact carries: `database_id`, `schema_version`,
`table`, `column`, `row_pk`, `query_id`, `route_id`, and `observed_at` (UTC,
timezone-aware ISO-8601). Facts without complete provenance cannot be
included in a `GroundedState`; they become `UNKNOWN` instead. Redaction
helpers strip auth tokens, credentials, and raw authorization headers from
diagnostics and serialized output.

### 4.4 Freshness

Each state carries an observation generation counter and UTC timestamp. A
rebuild after a database change produces a new state; cached states older than
the latest observation are stale and must be labeled as such by callers. The
state is never treated as a substitute for `get_ground_truth()`.

## 5. Representation contracts (`csm_env.representation`)

- The same `GroundedState` is rendered as versioned graph, JSON, and text
  output. Serializers are deterministic (stable ordering, no timestamps in
  output other than those in provenance, no set/dict iteration order
  dependence) and never mutate the source state.
- Round-trip fidelity: `verify_fidelity()` reconstructs the critical fields
  (record identities, required fact values, relations, unresolved bindings)
  from each representation and compares them to the source state. Fields
  documented as presentation-only (text labels, formatting) are excluded.
- Text truncation is explicit and marked; truncation never removes provenance
  for included facts.

## 6. Verification contracts (`csm_env.verification`)

- `DirectSqlOracle` builds explicit direct queries from approved table/column
  metadata. It must not import `SchemaGraph`, `GraphQueryRouter`, `StateBuilder`,
  or their helpers: the oracle branch is independent by construction.
- Checks V1-V10 (accepted blueprint verification suite):

| ID | Check | Module |
|---|---|---|
| V1 | Table completeness (graph vs manifest) | `schema_checks.py` |
| V2 | Column/PK completeness and types | `schema_checks.py` |
| V3 | FK endpoint/relation validation | `schema_checks.py` |
| V4 | Referential-integrity spot checks | `relation_checks.py` |
| V5 | Graph-to-SQL equivalence | `relation_checks.py` |
| V6 | Pre-execution query-plan validation | `relation_checks.py` |
| V7 | Identifier-propagation grounding | `grounding_checks.py` |
| V8 | Fact grounding (provenance completeness) | `grounding_checks.py` |
| V9 | State consistency | `grounding_checks.py` |
| V10 | Freshness/rebuild | `grounding_checks.py` |

- Every check returns a typed `CheckResult` (`check_id`, `passed`,
  `failures[]`), with a reason for every failure. Live-endpoint checks are
  opt-in and explicitly skipped in the default offline suite.

## 7. Compatibility contract

- Existing public imports and behaviors are preserved unchanged:
  `csm_env.ENTITIES`, `FKS`, `TABLE_COLUMNS`, `EnvironmentGraph`, `Node`,
  `Edge`, `EnterpriseOpsSQLRunner`, `CSMEnvironmentRepresentation`
  (including `get_current_state`, `hydrate_entity`, `hydrate_case`,
  `snapshot_table`, `snapshot_all`, `get_case_context`, `query_local_graph`),
  `CSMEnvironmentAPI` (`get_ground_truth`, `get_case_context`, `get_neighbors`,
  `search_cases`), and `from_enterpriseops_mcp_client`.
- New capabilities are exposed as additive methods/types. Internals may be
  refactored behind the same public surface.
- `csm_env/schema.py` remains a compatibility facade: `ENTITIES`, `FKS`, and
  `TABLE_COLUMNS` keep their tuple/dict shapes, now derived from the
  `schema_spec` manifest.
