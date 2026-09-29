from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class UnresolvedBinding:
    """A binding that could not be established, with the reason why.

    NULL source values, missing rows, and required-but-unobserved tables are
    distinct conditions (CON-011) and are recorded separately.
    """

    table: str
    source_column: Optional[str]
    reason: str  # "null_fk" | "no_rows" | "required_table_unobserved"
    source_table: Optional[str] = None
    source_value: Any = None
