# Phase 0 Policy Record: Fixtures, Task Types, Anchor Aliases, Metadata Support

Status: Approved (2026-09-24). This record satisfies TASK-003 of
`plan/architecture-csm-environment-1.md` without adding source code. Phase 1+
modules and tests implement exactly what is written here.

## 1. Deterministic fixture format

Canonical fixture: `csm_env/tests/fixtures/csm_minimal.json` (created in
Phase 3). Format:

```json
{
  "fixture_version": "1.0.0",
  "schema_version": "1.0.0",
  "database_id": "fixture-db",
  "tables": {
    "customer_case": [ { "case_id": 1233, "account_id": 10, "...": "..." } ],
    "account":       [ { "account_id": 10, "name": "ACME", "...": "..." } ]
  }
}
```

Rules:

- Table keys are lowercase table names; each row is a flat dict of
  column -> JSON value. Rows contain only manifest columns.
- The fixture is a **table-keyed row store**, not SQL. The fake reader in
  tests interprets the same `QuerySpec`-rendered SELECT shape the real
  transport produces; it never parses arbitrary SQL.
- The fixture is synthetic and non-sensitive: no real customer data, tokens,
  URLs, or credentials (CON-012, SEC-003).
- Fixture data supports at minimum: a connected case (case -> account,
  contact, product, installed_product, user, user_group), entitlement and
  contract downstream of the account, a case SLA, knowledge linked through
  case_knowledge, a NULL FK somewhere, and a dangling FK somewhere, so every
  Phase 4-7 test case (NULL, unresolved, contradiction, freshness) is
  exercisable deterministically.

## 2. Supported task types (deterministic requirement registry)

Phase 4 seeds exactly these task types; each maps to required tables:

| task_type | anchor | required tables |
|---|---|---|
| `resolve_case` | customer_case | customer_case, account, entitlement, case_sla, product |
| `case_overview` | customer_case | customer_case, account, contact, product, installed_product, user, user_group |
| `check_entitlement` | customer_case | customer_case, account, entitlement, contract |
| `find_knowledge` | customer_case | customer_case, product, knowledge |
| `account_profile` | account | account, contract, entitlement, installed_product |

Unknown task types fail closed with `ValueError`; the router never invents
requirements. Caller-supplied explicit requirements are accepted and
validated against the schema registry (every required table must exist).

## 3. Anchor aliases

Resolution order (TASK-023 implements this):

1. Exact table name (`customer_case`).
2. Exact node type (`CustomerCase`).
3. Registered alias set per table, defined as: lowercase node type, plus the
   singular/plural forms of the table name (e.g. `case` -> `customer_case`,
   `cases` -> `customer_case`, `user_group` aliases include `group`).
4. If an alias resolves to multiple tables: reject with `ValueError`
   (ambiguous) — never guess.
5. If nothing matches: `KeyError` (unknown reference type).

`reference_id` must be non-empty; `None`, `""`, and whitespace-only values
are rejected before any query.

## 4. Unsupported live metadata policy (introspection)

- `SchemaIntrospector` is optional. When a backend cannot supply metadata
  (missing endpoint, error envelope, unknown shape), the introspector returns
  an explicit `UnsupportedSchemaMetadata` result — it never guesses tables,
  columns, types, or constraints, and never raises a bare exception for an
  expected unsupported condition.
- A mismatch between introspected metadata and the static manifest is
  **fail-closed**: routing refuses to proceed and the mismatch is reported
  structurally (missing tables, extra tables, missing columns, wrong PK).
- In the current workspace (CON-005) no live metadata endpoint is approved;
  all Phase 1-8 gates run against the static manifest and fixtures only.
  Live verification is opt-in (FILE-059) and skipped by default.
