"""The Cost(S) objective of State Model 2.0 (blueprint section 2).

    Cost(S) = lambda_r * |rows| + lambda_t * |tokens| + lambda_q * |queries|

The blueprint states the *shape* of this objective but deliberately leaves the
weights undefined ("Cost(S) can combine..."). Leaving them undefined is what
made the objective unimplementable: nothing could be minimized. This module
fixes that by supplying explicit, externalized, tunable defaults.

Why the query term matters
--------------------------
``lambda_q * |queries|`` is only optimizable if the number of queries is a
*decision* rather than a consequence of retrieval. That is exactly what the
``RouteStrategy.FRONTIER`` execution strategy provides: it retrieves
incrementally, evaluates the objective over each executed prefix, and stops as
soon as coverage is met, so ``S* = argmin Cost(S)`` subject to
``Coverage(S, F_T) >= tau`` is actually selectable. Under the eager ``BROAD``
strategy every planned query is spent before any selection happens, so
``lambda_q`` could not influence anything -- the defect this module closes.

Defaults
--------
The defaults are **engineering choices, not research constants**, and are
intentionally explicit so an experiment can sweep them:

``lambda_rows = 1.0``    - one row is the unit of cost.
``lambda_tokens = 0.05`` - 20 tokens ~= 1 row; tokens are a *pressure* rather
                           than the unit, because a row can be arbitrarily wide.
``lambda_queries = 8.0`` - a query costs server work, latency and connection
                           budget, so it is priced well above a single row.
                           This is what makes stopping early pay off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

#: Divider of the deterministic, dependency-free token estimate. Kept identical
#: to the ablation layer's ``chars_div_4`` estimator so the retrieval-phase and
#: delivery-phase measurements are on the same scale.
TOKEN_ESTIMATOR = "chars_div_4"

_TOKENS_PER_CHAR = 0.25


@dataclass(frozen=True)
class CostWeights:
    """Weights of ``Cost(S)``. Externalized so a sweep can vary them."""

    lambda_rows: float = 1.0
    lambda_tokens: float = 0.05
    lambda_queries: float = 8.0

    def validate(self) -> "CostWeights":
        for name, value in (
            ("lambda_rows", self.lambda_rows),
            ("lambda_tokens", self.lambda_tokens),
            ("lambda_queries", self.lambda_queries),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value!r}")
        if self.lambda_rows == 0 and self.lambda_tokens == 0 and self.lambda_queries == 0:
            raise ValueError("at least one Cost(S) weight must be positive")
        return self

    def as_dict(self) -> dict:
        return {
            "lambda_rows": self.lambda_rows,
            "lambda_tokens": self.lambda_tokens,
            "lambda_queries": self.lambda_queries,
        }


DEFAULT_COST_WEIGHTS = CostWeights()


def _cell_length(value: Any) -> int:
    if value is None:
        return 4  # len("None")
    if isinstance(value, str):
        return len(value)
    return len(str(value))


def estimate_tokens(rows: Iterable[Mapping[str, Any]]) -> int:
    """Deterministic token estimate for a row payload (``chars_div_4``).

    Deliberately dependency-free and deterministic: no tokenizer, no clock, no
    network. This is a *measurement*, never a truncation trigger.
    """
    chars = 0
    for row in rows:
        for column, value in row.items():
            chars += _cell_length(column) + _cell_length(value) + 2
    return int(math.ceil(chars * _TOKENS_PER_CHAR))


def estimate_tokens_of_results(results: Sequence[Any]) -> int:
    """Token estimate across ``QueryResult``-like objects that carry ``rows``."""
    total = 0
    for result in results:
        rows = getattr(result, "rows", None) or ()
        total += estimate_tokens(rows)
    return total


@dataclass(frozen=True)
class StateCost:
    """One measured ``Cost(S)`` for a candidate state."""

    rows: int
    tokens: int
    queries: int
    weights: CostWeights = DEFAULT_COST_WEIGHTS

    @property
    def total(self) -> float:
        return (
            self.weights.lambda_rows * self.rows
            + self.weights.lambda_tokens * self.tokens
            + self.weights.lambda_queries * self.queries
        )

    @property
    def query_cost(self) -> float:
        """The ``lambda_q * |queries|`` term, exposed for diagnostics."""
        return self.weights.lambda_queries * self.queries

    def as_dict(self) -> dict:
        return {
            "rows": self.rows,
            "tokens": self.tokens,
            "queries": self.queries,
            "total": self.total,
            "query_cost": self.query_cost,
            "weights": self.weights.as_dict(),
        }


def compute_cost(
    *,
    rows: int,
    tokens: int,
    queries: int,
    weights: CostWeights = DEFAULT_COST_WEIGHTS,
) -> StateCost:
    """Build a :class:`StateCost` from already-measured components."""
    if rows < 0 or tokens < 0 or queries < 0:
        raise ValueError("Cost(S) components must be non-negative")
    return StateCost(rows=rows, tokens=tokens, queries=queries, weights=weights.validate())


def cheapest_covering_prefix(
    candidates: Sequence[tuple],
) -> tuple:
    """``argmin Cost(S)`` over prefixes that meet coverage.

    Args:
        candidates: ``(index, cost, meets_coverage)`` triples in execution
            order.

    Returns:
        ``(index, cost)`` of the cheapest prefix satisfying coverage, or
        ``None`` when no prefix does. Ties break toward the *earliest* prefix so
        the result stays deterministic.

    This is the literal ``S* = argmin_S Cost(S)`` subject to
    ``Coverage(S, F_T) >= tau`` from blueprint section 2, evaluated over the
    prefixes the frontier strategy actually produced.
    """
    best = None
    for index, cost, meets in candidates:
        if not meets:
            continue
        if best is None or cost.total < best[1].total:
            best = (index, cost)
    return best


__all__ = [
    "DEFAULT_COST_WEIGHTS",
    "TOKEN_ESTIMATOR",
    "CostWeights",
    "StateCost",
    "cheapest_covering_prefix",
    "compute_cost",
    "estimate_tokens",
    "estimate_tokens_of_results",
]
