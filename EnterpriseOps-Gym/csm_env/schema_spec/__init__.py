from __future__ import annotations

from .csm_schema import EXPECTED_TABLE_COUNT, MANIFEST
from .introspector import (
    IntrospectionOutcome,
    MismatchReport,
    SQLSchemaIntrospector,
    SchemaIntrospector,
    UnsupportedSchemaMetadata,
    compare_with_manifest,
)
from .models import (
    COLUMN_TYPES,
    ColumnSpec,
    ColumnType,
    ForeignKeySpec,
    SchemaCompatibility,
    SchemaManifest,
    SchemaSource,
    SchemaSourceKind,
    TableSpec,
)
from .registry import SchemaRegistry, manifest_major
from .validator import ManifestValidationError, validate_manifest

__all__ = [
    "COLUMN_TYPES",
    "EXPECTED_TABLE_COUNT",
    "MANIFEST",
    "ColumnSpec",
    "ColumnType",
    "ForeignKeySpec",
    "IntrospectionOutcome",
    "ManifestValidationError",
    "MismatchReport",
    "SQLSchemaIntrospector",
    "SchemaCompatibility",
    "SchemaIntrospector",
    "SchemaManifest",
    "SchemaRegistry",
    "SchemaSource",
    "SchemaSourceKind",
    "TableSpec",
    "compare_with_manifest",
    "manifest_major",
    "validate_manifest",
]
