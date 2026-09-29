from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .schema_spec.csm_schema import RAW_ENTITIES, RAW_FKS, RAW_TABLE_COLUMNS


@dataclass(frozen=True)
class EntitySpec:
    table: str
    node_type: str
    primary_key: str


@dataclass(frozen=True)
class ForeignKeySpec:
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relation: str


ENTITIES: Tuple[EntitySpec, ...] = tuple(
    EntitySpec(entity.table, entity.node_type, entity.primary_key) for entity in RAW_ENTITIES
)

ENTITY_BY_TABLE: Dict[str, EntitySpec] = {entity.table: entity for entity in ENTITIES}
ENTITY_BY_TYPE: Dict[str, EntitySpec] = {entity.node_type: entity for entity in ENTITIES}


FKS: Tuple[ForeignKeySpec, ...] = tuple(
    ForeignKeySpec(
        source_table=fk.source_table,
        source_column=fk.source_column,
        target_table=fk.target_table,
        target_column=fk.target_column,
        relation=fk.relation,
    )
    for fk in RAW_FKS
)

FK_BY_SOURCE: Dict[str, List[ForeignKeySpec]] = {}
FK_BY_TARGET: Dict[str, List[ForeignKeySpec]] = {}
for fk in FKS:
    FK_BY_SOURCE.setdefault(fk.source_table, []).append(fk)
    FK_BY_TARGET.setdefault(fk.target_table, []).append(fk)


TABLE_COLUMNS: Dict[str, Tuple[str, ...]] = {
    table: tuple(columns) for table, columns in RAW_TABLE_COLUMNS.items()
}
