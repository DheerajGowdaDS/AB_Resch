from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .models import (
    ColumnSpec,
    ForeignKeySpec,
    SchemaManifest,
    SchemaSource,
    TableSpec,
)


@dataclass(frozen=True)
class RawEntity:
    table: str
    node_type: str
    primary_key: str


@dataclass(frozen=True)
class RawForeignKey:
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relation: str


# Canonical CSM schema manifest for this package, derived from the
# ServiceNow/EnterpriseOps-Gym CSM schema material available to the project.
# The mapping is deliberately explicit so the environment representation layer
# does not infer relationships at runtime. Runtime snapshot/DB checks can be
# applied independently through csm_env.verification.
RAW_ENTITIES: Tuple[RawEntity, ...] = (
    RawEntity("account", "Account", "account_id"),
    RawEntity("case_knowledge", "CaseKnowledge", "case_kb_id"),
    RawEntity("case_sla", "CaseSLA", "case_sla_id"),
    RawEntity("contact", "Contact", "contact_id"),
    RawEntity("contract", "Contract", "contract_id"),
    RawEntity("customer_case", "CustomerCase", "case_id"),
    RawEntity("entitlement", "Entitlement", "entitlement_id"),
    RawEntity("installed_product", "InstalledProduct", "installed_product_id"),
    RawEntity("interaction", "Interaction", "interaction_id"),
    RawEntity("knowledge", "Knowledge", "knowledge_id"),
    RawEntity("location", "Location", "location_id"),
    RawEntity("notification", "Notification", "notification_id"),
    RawEntity("product", "Product", "product_id"),
    RawEntity("sla_definition", "SLADefinition", "sla_def_id"),
    RawEntity("user", "User", "user_id"),
    RawEntity("user_group", "UserGroup", "group_id"),
    RawEntity("user_group_member", "UserGroupMember", "member_id"),
)

RAW_FKS: Tuple[RawForeignKey, ...] = (
    RawForeignKey("user", "location_id", "location", "location_id", "LOCATED_AT"),
    RawForeignKey("contact", "account_id", "account", "account_id", "BELONGS_TO"),
    RawForeignKey("contact", "portal_user_id", "user", "user_id", "PORTAL_USER"),
    RawForeignKey("installed_product", "account_id", "account", "account_id", "OWNED_BY"),
    RawForeignKey("installed_product", "product_id", "product", "product_id", "INSTANCE_OF"),
    RawForeignKey("installed_product", "location_id", "location", "location_id", "LOCATED_AT"),
    RawForeignKey("contract", "account_id", "account", "account_id", "BELONGS_TO"),
    RawForeignKey("entitlement", "account_id", "account", "account_id", "BELONGS_TO"),
    RawForeignKey("entitlement", "product_id", "product", "product_id", "COVERS_PRODUCT"),
    RawForeignKey("entitlement", "contract_id", "contract", "contract_id", "UNDER_CONTRACT"),
    RawForeignKey("user_group_member", "group_id", "user_group", "group_id", "MEMBER_OF_GROUP"),
    RawForeignKey("user_group_member", "user_id", "user", "user_id", "MEMBER"),
    RawForeignKey("customer_case", "account_id", "account", "account_id", "BELONGS_TO"),
    RawForeignKey("customer_case", "contact_id", "contact", "contact_id", "REPORTED_BY"),
    RawForeignKey("customer_case", "product_id", "product", "product_id", "CONCERNS_PRODUCT"),
    RawForeignKey("customer_case", "installed_product_id", "installed_product", "installed_product_id", "HAS_INSTALLED_PRODUCT"),
    RawForeignKey("customer_case", "assignment_group_id", "user_group", "group_id", "ASSIGNED_TO_GROUP"),
    RawForeignKey("customer_case", "assigned_to", "user", "user_id", "ASSIGNED_TO"),
    RawForeignKey("case_sla", "case_id", "customer_case", "case_id", "FOR_CASE"),
    RawForeignKey("case_sla", "sla_def_id", "sla_definition", "sla_def_id", "DEFINED_BY"),
    RawForeignKey("knowledge", "product_id", "product", "product_id", "FOR_PRODUCT"),
    RawForeignKey("knowledge", "owner_id", "user", "user_id", "OWNED_BY"),
    RawForeignKey("case_knowledge", "case_id", "customer_case", "case_id", "FOR_CASE"),
    RawForeignKey("case_knowledge", "knowledge_id", "knowledge", "knowledge_id", "KNOWLEDGE"),
    RawForeignKey("notification", "case_id", "customer_case", "case_id", "FOR_CASE"),
    RawForeignKey("notification", "knowledge_id", "knowledge", "knowledge_id", "REFERENCES_KNOWLEDGE"),
    RawForeignKey("interaction", "account_id", "account", "account_id", "WITH_ACCOUNT"),
    RawForeignKey("interaction", "contact_id", "contact", "contact_id", "WITH_CONTACT"),
    RawForeignKey("interaction", "case_id", "customer_case", "case_id", "FOR_CASE"),
)


# The report documents these concrete columns. We retain them as the
# canonical snapshot schema so queries never depend on SELECT * ordering.
RAW_TABLE_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "account": ("account_id", "name", "account_type", "email_domain", "active", "sys_created_on", "sys_updated_on"),
    "case_knowledge": ("case_kb_id", "case_id", "knowledge_id", "used_as"),
    "case_sla": ("case_sla_id", "case_id", "sla_def_id", "stage", "start_time", "breach_time", "has_breached", "completed_time"),
    "contact": ("contact_id", "account_id", "portal_user_id", "active", "is_primary", "sys_created_on", "sys_updated_on"),
    "contract": ("contract_id", "account_id", "contract_type", "status", "contract_price", "start_date", "end_date", "sys_created_on", "sys_updated_on"),
    "customer_case": ("case_id", "number", "account_id", "contact_id", "product_id", "installed_product_id", "channel", "priority", "state", "short_description", "assignment_group_id", "assigned_to", "escalation", "escalation_reason", "reopen_count", "sys_created_on", "sys_updated_on", "closed_on"),
    "entitlement": ("entitlement_id", "account_id", "product_id", "contract_id", "support_level", "coverage_hours", "max_cases_per_month", "active", "sys_created_on", "sys_updated_on"),
    "installed_product": ("installed_product_id", "account_id", "product_id", "location_id", "serial_number", "status", "warranty_end", "sys_created_on", "sys_updated_on"),
    # M0 fix (verification report D1): the real CSM `interaction` table has no
    # `sys_updated_on` column; the 11-column declaration caused HTTP 400 in
    # get_case_context()/hydrate_case() against every case.
    "interaction": ("interaction_id", "channel", "account_id", "contact_id", "case_id", "interacted_user", "status", "started_at", "ended_at", "sys_created_on"),
    # M0 fix (verification report D3): `body` exists in the real CSM table
    # (17/18 shipped snapshots) and was previously dropped silently.
    "knowledge": ("knowledge_id", "kb_number", "title", "state", "visibility", "product_id", "owner_id", "body", "sys_created_on", "sys_updated_on"),
    "location": ("location_id", "name", "plot_no", "street", "city", "country", "active", "sys_created_on", "sys_updated_on"),
    "notification": ("notification_id", "case_id", "knowledge_id", "email", "type", "status", "sys_created_on"),
    "product": ("product_id", "name", "category", "product_price", "lifecycle_state", "sys_created_on", "sys_updated_on"),
    "sla_definition": ("sla_def_id", "name", "metric", "target_mins", "pause_on_pending", "applies_to_priority", "active", "sys_created_on", "sys_updated_on"),
    "user": ("user_id", "first_name", "last_name", "email", "phone", "role", "location_id", "active", "sys_created_on", "sys_updated_on"),
    "user_group": ("group_id", "name", "type", "active", "sys_created_on", "sys_updated_on"),
    # M0 fix (verification report D2): the real `user_group_member` table has
    # no `sys_updated_on` column (only 4 columns).
    "user_group_member": ("member_id", "group_id", "user_id", "sys_created_on"),
}

_ID_LIKE_COLUMNS = {
    "assigned_to",
    "interacted_user",
    "owner_id",
    "portal_user_id",
}

_BOOL_COLUMNS = {"active", "is_primary", "pause_on_pending", "has_breached", "escalation"}

_INT_COLUMNS = {
    "product_price",
    "contract_price",
    "target_mins",
    "max_cases_per_month",
    "reopen_count",
}

_TEXT_COLUMNS = {"body", "short_description", "escalation_reason"}


def _column_type(table: str, column: str, primary_key: str) -> str:
    if column == primary_key:
        return "string"
    if column.endswith("_id") or column in _ID_LIKE_COLUMNS:
        return "string"
    if column in _BOOL_COLUMNS:
        return "bool"
    if column in _INT_COLUMNS:
        return "int"
    if column.endswith(("_on", "_time", "_at", "_date", "_end")):
        return "datetime"
    if column in _TEXT_COLUMNS:
        return "text"
    return "string"


def _static_manifest() -> SchemaManifest:
    tables: List[TableSpec] = []
    for entity in RAW_ENTITIES:
        columns = RAW_TABLE_COLUMNS[entity.table]
        tables.append(
            TableSpec(
                table=entity.table,
                node_type=entity.node_type,
                primary_key=entity.primary_key,
                columns=tuple(
                    ColumnSpec(name=column, type=_column_type(entity.table, column, entity.primary_key))
                    for column in columns
                ),
            )
        )

    fks = tuple(
        ForeignKeySpec(
            source_table=fk.source_table,
            source_column=fk.source_column,
            target_table=fk.target_table,
            target_column=fk.target_column,
            relation=fk.relation,
        )
        for fk in RAW_FKS
    )

    return SchemaManifest(
        schema_version="1.0.0",
        tables=tuple(sorted(tables, key=lambda t: t.table)),
        foreign_keys=tuple(
            sorted(fks, key=lambda f: (f.source_table, f.source_column, f.target_table))
        ),
        source=SchemaSource.STATIC_REGISTRY,
        derived_from="EnterpriseOps-Gym CSM schema report (static registry)",
    )


MANIFEST: SchemaManifest = _static_manifest()

EXPECTED_TABLE_COUNT = 17
