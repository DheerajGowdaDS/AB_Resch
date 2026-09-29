
## Experiment 1 ablation-validity hardening (2026-09-28)

- Fixed brittle Condition-B anchor resolution observed in the 11-task pilot: anchor lookup is now batched per candidate table, full-name matching is supported from schema-derived columns, and the lookup budget is no longer spent once per column.
- Increased the default GGQR state-query hop budget from 2 to 3 so valid CSM paths such as `location -> user -> user_group_member -> user_group` are reachable.
- Made Condition B fail closed when grounded state is missing, unresolved, incomplete, or empty; it no longer silently falls back to the baseline agent.
- Preserved intervention errors and state-delivery telemetry when the benchmark executor raises before verifier execution.
- Excluded undelivered Condition-B runs from causal A/B effect estimates and withhold `delta_tsr` / `delta_vpr` until intervention delivery is complete.
- Removed ambiguous round-robin prompt-to-table anchor hints; only shape-unambiguous references such as `CS-<id>` receive a typed table hint.
- Made evaluation manifests portable by storing task-config paths relative to the `EnterpriseOps-Gym` root; readers can still recover legacy Windows paths.
- Pinned hashes now include the ablation package itself, in addition to benchmark and task files.
- Added regression coverage for all evidence-backed failures.

### Verification

- Full repository test suite: **221 passed, 3 skipped** (live opt-in).
- Experiment 1 dry-run readiness gate: **PASS**; verifier and LLM configuration warnings remain non-blocking by design.
- No new experimental result is claimed by this patch; the previous 11-task A/B result is treated as a superseded pilot because Condition-B delivery was incomplete.
# CSM GGQR Implementation — 2026-09-24 Update

## Research-hardening fixes

- Fixed one-to-many identifier fan-out by rendering set-valued filters as SQL `IN (...)`.
- Reworked GGQR planning to discover and materialize intermediate FK bridge tables automatically.
- Added deterministic tie-breaking that prefers already-required routing anchors for equal-length paths.
- Added `RequirementRegistry.from_json()` for external benchmark task-requirement manifests.
- Added explicit state contradiction objects and fail-closed handling.
- Added runtime value-type validation and corrected `entitlement.coverage_hours` to `string`.
- Added snapshot verification that distinguishes structural metadata checks from row-only runtime checks.
- Added regression tests for all fixes.

## Verification

Standalone suite from the unpacked archive:

- 111 passed
- 3 skipped (live opt-in tests)
- Python compilation: clean

## Post-audit hardening

- Restored executor dependency injection: `GraphQueryRouter` now uses the injected `QueryExecutor` for every execution wave while supplying the request-scoped runner required for bindings and budgets.
- Made `max_total_rows` strict under concurrency by reserving row capacity before launching each query and rolling back unused reservation after completion; query `LIMIT` is reduced to the reserved allowance.
- Replaced millisecond route IDs with UUID-based IDs to eliminate practical collision windows.
- Upgraded text fidelity from substring checks to deterministic reconstruction of record identities, critical fact values, relations, and unresolved bindings via a structured round-trip footer. Representation version is now `1.1.0`.
- Hardened duplicate-observation merging: later-only columns are preserved, and type-aware comparison avoids broad string coercion except for the documented string/integer materialization case.
- Added regression coverage for executor injection, strict concurrent row budgets, route-ID uniqueness, duplicate-column merging, and contradiction detection.

## Current verification

- Clean offline suite: **111 passed, 3 skipped** (explicit live opt-ins).
- Python compilation: clean.
- The external ServiceNow snapshot/task-manifest integrations remain explicit inputs: this package does not fabricate repository data when those external artifacts are unavailable.
