# AGENTS.md

## Repository scope

- This is a Python package for a database-backed CSM environment representation.
- Treat the database as the source of truth; `EnvironmentGraph` is a cache/view.
- The working source is the root `csm_env/` package; `zip_extract/` and the ZIP are
  extracted/delivered copies, not active source.
- Do not edit generated `.pytest_cache/`, `__pycache__/`, or other cache files.
- `CSM_ENV_VERIFICATION_REPORT.md` is historical, not project config, and
  `pyproject-snippet.toml` is a comments-only build note.
- There is no repository-local `pyproject.toml`, lockfile, CI config, or formatter config.

## Commands

Run commands from the repository root.

### Python and dependencies

- Target Python is 3.11 or newer; the package imports from the root without installation.
- The only runtime third-party import is `httpx`.
- Tests additionally require `pytest` and `pytest-asyncio`.

### Tests

Run the complete test directory with:

```powershell
python -m pytest csm_env/tests -v -p no:cacheprovider
```

Run one test function by node ID with:

```powershell
python -m pytest csm_env/tests/test_graph.py::test_graph_neighbors_and_subgraph -v -p no:cacheprovider
```

Run the async test by node ID with:

```powershell
python -m pytest csm_env/tests/test_csm_representation.py::test_case_context_is_connected_and_grounded -v -p no:cacheprovider
```

Keep `-p no:cacheprovider` when avoiding cache-file changes.

Do not use `-p no:asyncio` for normal verification; it intentionally disables
`pytest-asyncio` and makes the async test fail.

### Import check

A quick public-import smoke check is:

```powershell
python -c "from csm_env import EnterpriseOpsSQLRunner, CSMEnvironmentRepresentation, CSMEnvironmentAPI, from_enterpriseops_mcp_client; print('IMPORT OK')"
```

### Build, lint, and type checking

- No local build/install command or package metadata is defined; the intended
  integration is copying `csm_env/` into EnterpriseOps-Gym.
- No Ruff, Black, isort, Flake8, mypy, pyright, pylint, or pre-commit command is configured.
- Do not invent a formatter or type-check command for this standalone copy.
- If integrated into a host repository, use that repository’s configured lint,
  type-check, and test commands instead of adding duplicate configuration.
- No live CSM server or database checkout is present; do not pull images, start
  servers, or make network/database changes unless asked.

## Architecture

- `sql_runner.py` adapts the EnterpriseOps `/api/sql-runner` endpoint and sends
  the query, `database_id`, `x-database-id`, auth, and context headers.
- `schema.py` is the explicit table, column, entity, and foreign-key registry.
- `builder.py` validates entities, reads rows, hydrates nodes, and projects links.
- `graph.py` is dependency-free and stores nodes, edges, deduplication, and subgraphs.
- `api.py` is the stable facade for agents, planners, and evaluators.
- `integration.py` adapts an external `MCPClient` without importing that package.
- `get_ground_truth()` is for exact current database state; `get_case_context()`
  and `query_local_graph()` are relational representation views.
- Prefer bounded case hydration over materializing a full snapshot during agent steps;
  preserve the distinction between fresh database reads and cached graph data.

## Python style

- Follow the style of the file being edited; this project has no enforced formatter.
- Use four spaces for indentation and keep nesting shallow.
- Prefer readable lines, but do not reformat unrelated code while making a change.
- Use predominantly double-quoted strings, matching the existing implementation.
- Add `from __future__ import annotations` at the top of implementation modules
  unless the surrounding file intentionally differs.
- Order imports as future imports, standard library, third-party, then local imports.
- Keep imports explicit; avoid wildcard imports and unnecessary compatibility shims.
- Use relative imports for package internals and absolute `csm_env` imports in tests
  when the existing test style calls for them.
- Update `csm_env/__init__.py` and `__all__` when adding a public package export.
- Preserve public names and return shapes unless an intentional API change is needed.

## Types and data modeling

- Annotate public functions, methods, and meaningful local data.
- Match the typing vocabulary used by the surrounding module; the code mixes
  `typing.Dict`/`List` with built-in `list`/`set` generics.
- Prefer specific types at internal boundaries; reserve `Any` for external or
  heterogeneous database/API values.
- Use `Optional` for intentionally absent values and explicit return annotations
  for async methods.
- Use frozen dataclasses for static schema specifications.
- Use mutable dataclasses for graph nodes and edges, with `default_factory` for
  dictionary fields.
- Keep IDs string-normalized in graph keys, while accepting source IDs as `Any`.
- Do not weaken annotations merely to silence an unavailable type checker.

## Naming and API conventions

- Use `PascalCase` for classes and dataclass specifications.
- Use `snake_case` for functions, methods, variables, and keyword arguments.
- Use uppercase names for module constants such as `ENTITIES` and `TABLE_COLUMNS`.
- Use lowercase database table names and uppercase relation labels such as
  `BELONGS_TO` and `REPORTED_BY`.
- Keep public API names descriptive and stable; avoid abbreviations unless they
  match an existing external protocol.
- Use keyword-only arguments for optional public controls when the existing API
  already does so.
- Use async methods for database and HTTP operations.

## SQL and error handling

- Keep queries read-only and validate entity/table names before interpolation.
- Select explicit columns from `TABLE_COLUMNS`; do not introduce `SELECT *`.
- Use `LIMIT` for single-row lookups and bound user-facing search limits.
- Route values through `_sql_literal` or an equally explicit, tested escaping path.
- Never interpolate untrusted table, column, or identifier names.
- Let `httpx.Response.raise_for_status()` surface HTTP failures.
- Raise `KeyError` for unknown entities or missing requested rows.
- Raise `ValueError` for malformed rows or missing primary keys.
- Raise `TypeError` for unsupported response shapes.
- Do not silently turn error envelopes into successful data rows.
- Do not catch and discard exceptions unless the caller’s contract requires it.

## Graph and data-flow rules

- Keep graph nodes and edges deduplicated by their existing identity keys.
- Preserve incoming and outgoing foreign-key traversal.
- Keep semantic projections separate from database-faithful association nodes.
- Refresh or rehydrate data when freshness matters before making ground-truth claims.
- Avoid full-database snapshots in normal agent-time paths.
- Keep tests deterministic by using `FakeSQL` or another local double, not a live API.

## Tests and change discipline

- Keep tests in `csm_env/tests/` and name files `test_*.py`.
- Use plain pytest functions, direct assertions, and small focused fixtures/doubles.
- Mark async tests with `@pytest.mark.asyncio` and keep them awaitable.
- Test behavior and edge cases without requiring CSM server access.
- Run the focused test first, then the complete `csm_env/tests` command.
- Run import and syntax checks when public modules or exports change.
- Update README examples or integration notes when behavior or commands change.
- Do not modify the ZIP or `zip_extract/` copy unless packaging is explicitly requested.

## Existing agent/editor rules

- No `AGENTS.md` existed here before this file.
- No `.cursorrules` or `.cursor/rules/` files were found.
- No `.github/copilot-instructions.md` or related Copilot instructions were found.
- This document is therefore the repository-specific guidance for coding agents.
