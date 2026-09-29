"""Pre-execution SQL validation for the read-only routing path.

Every statement produced by :mod:`csm_env.query.adapters` passes through
:func:`validate_read_only_select` before it can reach a live CSM server.

This module exists because of CON-012: a *malformed* query is not a bad
answer, it is a transport-level failure. The live Experiment-1 pilot returned
``HTTP 400 Bad Request`` for a task whose prompt contained the proper noun
``Wayne Enterprises' Windows Server``: the surrounding ``LIKE`` pattern was
built without escaping the embedded apostrophe, so the server received two
unbalanced string literals. The task was scored as an undeliverable
intervention when the real fault was ours, not the agent's.

The rules are intentionally strict and deliberately *structural*:

* exactly one statement, terminated by at most one ``;``;
* the statement is a ``SELECT`` (never a write);
* explicit projection - ``SELECT *`` is rejected (SEC-002 / CON-011);
* string literals and parentheses are balanced.

A violation raises :class:`SqlValidationError` so the caller can convert a
malformed statement into a typed, recorded failure instead of an opaque HTTP
error. Validation never rewrites SQL; it only accepts or rejects.
"""

from __future__ import annotations

import re
from typing import Tuple

#: Statement head the read-only path is allowed to emit.
READ_ONLY_HEAD = "SELECT"

#: ``SELECT`` optionally followed by ``DISTINCT``/``ALL``.
_SELECT_HEAD_RE = re.compile(r"^\s*SELECT\s+(?:(DISTINCT|ALL)\s+)?", re.IGNORECASE)

#: Rejects ``*`` and ``table.*`` projections, which the router never needs. The
#: pattern is applied to the text *after* the ``SELECT`` head has been removed,
#: so it is anchored on the projection list itself.
_STAR_PROJECTION_RE = re.compile(
    r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?\*\s*(?:,|\Z|;|\bFROM\b)",
    re.IGNORECASE,
)

#: Identifier used to quote an identifier when one must be inspected.
_QUOTED_IDENTIFIER_CHARS = ('"', "`", "[", "]")


class SqlValidationError(ValueError):
    """A statement violated the read-only structural contract.

    Subclasses :class:`ValueError` so existing ``except ValueError`` handling in
    the planning path keeps working, while callers that care can distinguish a
    malformed statement from any other bad value.
    """


def _mask_literals(query: str) -> Tuple[str, bool]:
    """Replace string-literal bodies with spaces, preserving offsets.

    Masking (rather than deleting) keeps character positions stable so later
    checks can report an accurate offset, and it lets structural analysis run on
    a statement whose literals legitimately contain ``FROM``, ``;`` or ``*``.

    A doubled ``''`` is SQL's escape for a literal apostrophe, so it is consumed
    as a pair rather than closing and reopening the literal.

    Returns:
        The masked text and whether the statement ended outside a literal.
    """
    masked = list(query)
    index = 0
    length = len(query)
    in_literal = False
    while index < length:
        char = query[index]
        if char == "'":
            if in_literal and index + 1 < length and query[index + 1] == "'":
                masked[index] = " "
                masked[index + 1] = " "
                index += 2
                continue
            masked[index] = " "
            in_literal = not in_literal
            index += 1
        elif in_literal:
            masked[index] = " "
            index += 1
        else:
            index += 1
    return "".join(masked), not in_literal


def split_statements(masked_query: str) -> Tuple[str, ...]:
    """Split a literal-masked statement on ``;`` separators."""
    return tuple(part.strip() for part in masked_query.split(";") if part.strip())


def validate_read_only_select(query: object) -> str:
    """Validate one read-only ``SELECT`` statement and return it unchanged.

    Args:
        query: The candidate SQL text.

    Returns:
        The same string, so the function can be used inline in an expression.

    Raises:
        TypeError: If ``query`` is not a string.
        SqlValidationError: If the statement is empty, is not a single
            ``SELECT``, uses ``SELECT *``, or has unbalanced quotes or
            parentheses.
    """
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}")

    statement = query.strip()
    if not statement:
        raise SqlValidationError("refusing to execute an empty statement")

    masked, balanced = _mask_literals(statement)
    if not balanced:
        raise SqlValidationError(
            "unbalanced string literal; escape embedded apostrophes by doubling them"
        )

    statements = split_statements(masked)
    if len(statements) != 1:
        raise SqlValidationError(
            f"expected exactly one statement, found {len(statements)}"
        )

    head = _SELECT_HEAD_RE.match(statement)
    if not head:
        leading = statement.split(None, 1)[0].upper() if statement.split() else "?"
        raise SqlValidationError(
            f"read-only path refuses statement starting with {leading!r}; "
            f"only {READ_ONLY_HEAD} is allowed"
        )

    # ``masked`` is aligned with ``statement``, so slicing it by the head width
    # keeps the offsets of the projection list correct.
    if _STAR_PROJECTION_RE.match(masked[head.end():]):
        raise SqlValidationError(
            "explicit column projection required; 'SELECT *' is not permitted"
        )

    if masked.count("(") != masked.count(")"):
        raise SqlValidationError(
            f"unbalanced parentheses: {masked.count('(')} '(' vs "
            f"{masked.count(')')} ')'"
        )

    if "--" in masked or "/*" in masked:
        raise SqlValidationError("SQL comments are not permitted in routed statements")

    return statement
