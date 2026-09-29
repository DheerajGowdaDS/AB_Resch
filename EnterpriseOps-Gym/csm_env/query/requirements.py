from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, Mapping, Optional, Tuple

from ..schema_spec.registry import SchemaRegistry
from .models import TaskRequirements


def _default_registry_tables() -> Dict[str, Tuple[str, ...]]:
    return {
        "resolve_case": (
            "customer_case", "account", "entitlement", "case_sla", "product",
        ),
        "case_overview": (
            "customer_case", "account", "contact", "product",
            "installed_product", "user", "user_group",
        ),
        "check_entitlement": (
            "customer_case", "account", "entitlement", "contract",
        ),
        "find_knowledge": (
            "customer_case", "product", "knowledge",
        ),
        "account_profile": (
            "account", "contract", "entitlement", "installed_product",
        ),
    }


def _default_registry_attributes() -> Dict[str, Tuple[str, ...]]:
    """Task-conditioned attribute *requests* (State Model 2.0, blueprint section 3).

    Values are attribute keys from :data:`ATTRIBUTE_LEXICON` (or concrete
    column names); they are resolved to concrete columns across the task's
    required tables at construction time via the shared
    :func:`resolve_attribute_columns`, so an unknown request fails closed
    rather than reaching the SQL layer.
    """
    return {
        "resolve_case": ("priority", "state", "assignment_group", "escalation"),
        "case_overview": ("state", "priority", "assignment_group", "assigned_to"),
        "check_entitlement": ("support_level", "active", "state"),
        "find_knowledge": ("state", "title"),
        "account_profile": ("support_level", "active", "state", "serial"),
    }


#: Attribute lexicon: an attribute key a task prompt plausibly names, mapped
#: to the concrete columns that can realize it. Keys absent from a task's
#: tables simply contribute nothing; a request matching nothing anywhere
#: fails closed.
#:
#: This is the *single* attribute-resolution source of truth (P0/A1). Both the
#: query layer (``project_columns``) and the requirement registry resolve
#: task attribute keys through it, so the two can never drift apart: the
#: prompt shorthand ``assignment_group`` resolves to the concrete
#: ``assignment_group_id`` FK in exactly the same way on both sides.
ATTRIBUTE_LEXICON: Dict[str, Tuple[str, ...]] = {
    "priority": ("priority",),
    "state": ("state", "status", "lifecycle_state"),
    "status": ("state", "status", "lifecycle_state"),
    "escalation": ("escalation", "escalation_reason"),
    "channel": ("channel",),
    "description": ("short_description",),
    "support_level": ("support_level", "coverage_hours"),
    "coverage": ("support_level", "coverage_hours"),
    "assignment_group": ("assignment_group_id",),
    "assigned_to": ("assigned_to",),
    "assignee": ("assigned_to",),
    "reopen": ("reopen_count",),
    "sla": ("stage", "has_breached", "breach_time"),
    "warranty": ("warranty_end",),
    "serial": ("serial_number",),
    "contract_type": ("contract_type",),
    "active": ("active",),
    "title": ("title",),
}


def resolve_attribute_columns(registry, tables, keys):
    """Resolve attribute keys to concrete columns on ``tables`` (P0/A1 shared resolver).

    This is the *single* attribute-resolution path used by BOTH the query
    planner (``project_columns``) and the requirement registry, so the two
    sides can never disagree on what a task attribute means. For each key the
    lexicon candidates (then the literal key) are kept only when they name a
    real column on one of ``tables`` (or its ``_`` suffix); a key with no
    match contributes nothing rather than fabricating a column.

    Args:
        registry: A :class:`~csm_env.schema_spec.registry.SchemaRegistry`.
        tables: Tables whose columns may realize the keys.
        keys: Attribute keys (lexicon keys or concrete column names).

    Returns:
        A stable, de-duplicated tuple of concrete column names.
    """
    tables_tuple = tuple(tables)
    columns: Tuple[str, ...] = ()
    for key in keys:
        key_l = str(key).lower()
        candidates = list(ATTRIBUTE_LEXICON.get(key_l, ()))
        if key_l not in candidates:
            candidates.append(key_l)
        for candidate in candidates:
            for table in tables_tuple:
                spec = registry.table(table)
                if spec is None:
                    continue
                column_names = spec.column_names()
                if candidate in column_names:
                    if candidate not in columns:
                        columns = columns + (candidate,)
                    break
                matched_suffix = False
                for column in column_names:
                    if column.endswith(f"_{candidate}"):
                        if column not in columns:
                            columns = columns + (column,)
                        matched_suffix = True
                        break
                if matched_suffix:
                    break
    return columns


def _default_registry_relations() -> Dict[str, Tuple[Tuple[str, str], ...]]:
    """Required relational structure per task type (blueprint section 3).

    These are the ``(source_table, target_table)`` pairs the task's reasoning
    needs, consumed by the planner's minimum-relevant-path search so the
    route connects exactly the entities the task asks about.
    """
    return {
        "resolve_case": (
            ("customer_case", "account"),
            ("customer_case", "user_group"),
        ),
        "case_overview": (
            ("customer_case", "account"),
            ("customer_case", "contact"),
            ("customer_case", "user_group"),
        ),
        "check_entitlement": (
            ("customer_case", "account"),
            ("entitlement", "contract"),
        ),
        "find_knowledge": (
            ("customer_case", "product"),
            ("knowledge", "product"),
        ),
        "account_profile": (
            ("account", "entitlement"),
            ("account", "contract"),
        ),
    }


def _default_registry_intent() -> Dict[str, Tuple[str, ...]]:
    """Coarse per-task intent, refined by the caller's TaskRequest when given."""
    return {
        "resolve_case": ("read", "update"),
        "case_overview": ("read",),
        "check_entitlement": ("read",),
        "find_knowledge": ("read",),
        "account_profile": ("read",),
    }


class TaskRequirement:
    """One registered task type with its tables, attributes and relations."""

    def __init__(
        self,
        task_type: str,
        required_tables: Tuple[str, ...],
        required_attributes: Tuple[str, ...] = (),
        required_relations: Tuple[Tuple[str, str], ...] = (),
        intent: Tuple[str, ...] = (),
    ) -> None:
        self.task_type = task_type
        self.required_tables = tuple(required_tables)
        self.required_attributes = tuple(required_attributes)
        self.required_relations = tuple(required_relations)
        self.intent = tuple(intent)

    def requirements(
        self,
        *,
        attribute_overrides: Optional[Tuple[str, ...]] = None,
        intent_overrides: Optional[Tuple[str, ...]] = None,
        relation_overrides: Optional[Tuple[Tuple[str, str], ...]] = None,
    ) -> TaskRequirements:
        """Build the frozen requirements view for one request.

        Request-level attribute/intent/relation overrides (State Model 2.0
        section 3) are *unioned* with the registered defaults: a caller can
        add task-specific attributes/relations without being able to silently
        drop the ones the task type needs for relation interpretation.
        """
        attributes = self.required_attributes
        if attribute_overrides:
            merged = list(dict.fromkeys((*attributes, *attribute_overrides)))
            attributes = tuple(merged)
        relations = self.required_relations
        if relation_overrides:
            relations = tuple(dict.fromkeys((*relations, *relation_overrides)))
        intent = self.intent
        if intent_overrides:
            intent = tuple(dict.fromkeys((*intent, *intent_overrides)))
        return TaskRequirements(
            task_type=self.task_type,
            required_tables=self.required_tables,
            required_attributes=attributes,
            required_relations=relations,
            intent=intent,
        )


class RequirementRegistry:
    """Deterministic task-type -> requirements lookup (Approach A).

    Unknown task types fail closed; the router never invents requirements.
    Caller-supplied explicit requirements are validated against the registry.

    State Model 2.0 (blueprint section 3) extends the tables-only registry
    with task-conditioned attributes, relations, and intent, while remaining
    backward compatible with ``{task_type: [table, ...]}`` manifests.
    """

    def __init__(
        self,
        registry: SchemaRegistry,
        tables_by_task: Optional[Dict[str, Tuple[str, ...]]] = None,
        *,
        attributes_by_task: Optional[Dict[str, Tuple[str, ...]]] = None,
        relations_by_task: Optional[Dict[str, Tuple[Tuple[str, str], ...]]] = None,
        intent_by_task: Optional[Dict[str, Tuple[str, ...]]] = None,
    ) -> None:
        self._schema = registry
        self._by_task: Dict[str, TaskRequirement] = {}
        source = tables_by_task if tables_by_task is not None else _default_registry_tables()
        attributes = (
            attributes_by_task
            if attributes_by_task is not None
            else _default_registry_attributes()
        )
        relations = (
            relations_by_task
            if relations_by_task is not None
            else _default_registry_relations()
        )
        intent = intent_by_task if intent_by_task is not None else _default_registry_intent()

        for task_type, tables in source.items():
            for table in tables:
                registry.require_table(table)
            # Validate every registered attribute against the real schema so a
            # typo can never reach SQL construction (fail closed).
            resolved_attributes: Tuple[str, ...] = ()
            for attribute in attributes.get(task_type, ()):
                resolved_attributes = resolved_attributes + tuple(
                    self._resolve_attribute(tables, attribute)
                )
            resolved_attributes = tuple(dict.fromkeys(resolved_attributes))
            resolved_relations: Tuple[Tuple[str, str], ...] = ()
            for pair in relations.get(task_type, ()):
                resolved_relations = resolved_relations + (
                    self._resolve_relation(pair),
                )
            self._by_task[task_type] = TaskRequirement(
                task_type,
                tables,
                required_attributes=resolved_attributes,
                required_relations=resolved_relations,
                intent=tuple(intent.get(task_type, ())),
            )

    def _resolve_attribute(self, tables: Tuple[str, ...], attribute: str) -> Tuple[str, ...]:
        """Resolve one attribute request against the task's own tables.

        Delegates to the shared :func:`resolve_attribute_columns` so the registry
        and the query planner use one resolver (P0/A1). An attribute request may
        be a lexicon key or a concrete column name; it must realize at least one
        column on a required table or the request fails closed.
        """
        resolved = resolve_attribute_columns(self._schema, tables, (attribute,))
        if not resolved:
            raise KeyError(
                f"Required attribute {attribute!r} matches no column of tables "
                f"{list(tables)}"
            )
        return resolved

    def _resolve_relation(self, pair: Tuple[str, str]) -> Tuple[str, str]:
        source, target = pair
        self._schema.require_table(source)
        self._schema.require_table(target)
        # Relation pairs describe required *connectivity* and are matched
        # direction-agnostically by the planner (an edge in either direction
        # realizes the pair), so both orientations are accepted here.
        fk_pairs = {
            (fk.source_table.lower(), fk.target_table.lower())
            for fk in self._schema.foreign_keys()
        }
        forward = (source.lower(), target.lower())
        backward = (target.lower(), source.lower())
        if forward not in fk_pairs and backward not in fk_pairs:
            raise KeyError(
                f"Required relation {source!r} -> {target!r} declares no foreign key"
            )
        return (source.lower(), target.lower())

    @classmethod
    def from_json(cls, registry: SchemaRegistry, path: str | Path) -> "RequirementRegistry":
        """Load task->required-table mappings from a benchmark/task manifest.

        This keeps the router independent of hand-coded benchmark task names.
        The JSON may be either {task_type: [table, ...]} or
        {"tasks": {task_type: [table, ...]}}. Unknown tables fail closed.

        Optional ``required_attributes`` / ``required_relations`` /
        ``intent`` blocks are consumed when present (State Model 2.0
        section 3); their absence keeps the tables-only behaviour.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        mapping = payload.get("tasks", payload) if isinstance(payload, Mapping) else None
        if not isinstance(mapping, Mapping):
            raise ValueError("requirements JSON must be an object mapping task types to tables")
        normalized: Dict[str, Tuple[str, ...]] = {}
        for task_type, tables in mapping.items():
            if not isinstance(task_type, str) or not isinstance(tables, (list, tuple)):
                raise ValueError("each task type must map to a list of table names")
            normalized[task_type] = tuple(str(table).lower() for table in tables)

        attributes: Dict[str, Tuple[str, ...]] = {}
        attribute_block = payload.get("required_attributes")
        if isinstance(attribute_block, Mapping):
            for task_type, names in attribute_block.items():
                if isinstance(names, (list, tuple)):
                    attributes[str(task_type)] = tuple(str(name) for name in names)

        relations: Dict[str, Tuple[Tuple[str, str], ...]] = {}
        relation_block = payload.get("required_relations")
        if isinstance(relation_block, Mapping):
            for task_type, pairs in relation_block.items():
                if isinstance(pairs, (list, tuple)):
                    resolved: Tuple[Tuple[str, str], ...] = ()
                    for pair in pairs:
                        if isinstance(pair, (list, tuple)) and len(pair) == 2:
                            resolved = resolved + ((str(pair[0]), str(pair[1])),)
                    relations[str(task_type)] = resolved

        intents: Dict[str, Tuple[str, ...]] = {}
        intent_block = payload.get("intent")
        if isinstance(intent_block, Mapping):
            for task_type, values in intent_block.items():
                if isinstance(values, (list, tuple)):
                    intents[str(task_type)] = tuple(str(value) for value in values)

        return cls(
            registry,
            normalized,
            attributes_by_task=attributes or None,
            relations_by_task=relations or None,
            intent_by_task=intents or None,
        )

    def has(self, task_type: str) -> bool:
        return task_type in self._by_task

    def get(self, task_type: str) -> TaskRequirement:
        if task_type not in self._by_task:
            raise KeyError(
                f"Unknown task type: {task_type!r}. "
                f"Registered: {sorted(self._by_task)}"
            )
        return self._by_task[task_type]

    def requirements_for(
        self,
        task_type: str,
        *,
        attribute_overrides: Tuple[str, ...] = (),
        intent_overrides: Tuple[str, ...] = (),
        relation_overrides: Tuple[Tuple[str, str], ...] = (),
    ) -> TaskRequirements:
        return self.get(task_type).requirements(
            attribute_overrides=attribute_overrides or None,
            intent_overrides=intent_overrides or None,
            relation_overrides=relation_overrides or None,
        )

    def task_types(self) -> Tuple[str, ...]:
        return tuple(sorted(self._by_task))
