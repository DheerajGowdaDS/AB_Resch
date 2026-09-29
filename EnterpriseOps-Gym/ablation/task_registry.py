"""Evaluation-set construction driven by the real dataset and real schema.

Nothing about the CSM domain is hard-coded here as an answer. Instead:

* the task list comes from the real ``data/revised/csm`` JSON files;
* the entity lexicon is validated against the real ``csm_env`` schema registry,
  so a table that does not exist can never enter the evaluation set;
* complexity categories, GGQR task types, and anchor candidates are *derived*
  from the task's ``user_prompt`` and ``selected_tools`` only.

Crucially, ``verifiers`` is never read while building an intervention input.
That is what keeps Condition B honest (see ``ablation.state_model``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from benchmark.models import BenchmarkConfig
from csm_env.query.requirements import RequirementRegistry
from csm_env.schema_spec.registry import SchemaRegistry

PathLike = Union[str, Path]

GYM_ROOT = Path(__file__).resolve().parents[1]

def resolve_task_config_path(task_config_path: PathLike) -> Path:
    """Resolve a task-config path from either an absolute path or a repo-relative path.

    Manifests are persisted with repository-relative paths so they remain portable
    across Windows, Linux, containers, and cloned workspaces.
    """
    raw = str(task_config_path)
    path = Path(raw)
    if path.is_absolute() and path.is_file():
        return path
    # A manifest generated on Windows can contain a drive-letter path.  On Linux
    # that string is not considered absolute, so recover the repository-relative
    # suffix instead of treating it as a literal filename.
    normalized = raw.replace("\\", "/")
    marker = "/EnterpriseOps-Gym/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    candidate = GYM_ROOT / normalized.lstrip("/")
    if candidate.is_file():
        return candidate
    candidate = GYM_ROOT / path
    return candidate if candidate.is_file() else path

#: State-dependence categories, ordered by increasing relational depth. These
#: are explanatory variables for stratified analysis; they never replace the
#: benchmark's own task definition.
COMPLEXITY_CATEGORIES: Tuple[str, ...] = (
    "TYPE_1_SINGLE_ENTITY",
    "TYPE_2_ONE_HOP",
    "TYPE_3_MULTI_HOP",
    "TYPE_4_CROSS_ENTITY",
    "TYPE_5_MULTI_STEP",
)

#: Surface phrases that indicate a prompt talks about a given CSM table. Keys are
#: validated against the schema registry at build time; the longest phrase wins
#: so that "installed product" is not also counted as "product".
ENTITY_PHRASES: Mapping[str, Tuple[str, ...]] = {
    "customer_case": ("customer case", "case", "incident", "ticket", "reopen", "reopened"),
    "account": ("account", "customer", "organization", "organisation", "company", "plc"),
    "contact": ("contact", "portal user", "go-to person", "primary contact"),
    "product": ("product", "device", "model", "sku", "box", "server", "appliance", "node"),
    "installed_product": ("installed product", "asset", "serial", "hardware"),
    "contract": ("contract", "renewal", "agreement"),
    "entitlement": ("entitlement", "support level", "support plan", "coverage", "24x7", "24/7"),
    "case_sla": ("sla", "service level", "stage", "target resolution"),
    "sla_definition": ("sla definition", "sla definitions", "sla policy", "response sla"),
    "knowledge": ("knowledge", "kb", "article", "documentation"),
    "case_knowledge": ("knowledge article", "attach knowledge", "kb article"),
    "user": ("user", "agent", "assigned to", "assign", "assignee", "investigate"),
    "user_group": ("group", "queue", "escalation management", "team"),
    "user_group_member": ("group membership", "membership", "add to the group"),
    "interaction": ("interaction", "call log", "note", "channel", "called in", "phone"),
    "location": ("location", "site", "facility", "office", "address", "lab", "plant"),
    "notification": ("notification", "notify", "email the", "alert"),
}

#: Coarse entity families, keyed by table, used to detect cross-entity tasks.
FAMILY_TABLES: Mapping[str, Tuple[str, ...]] = {
    "case": ("customer_case", "interaction", "case_sla", "case_knowledge"),
    "party": ("account", "contact", "user", "user_group", "user_group_member"),
    "commercial": ("contract", "entitlement", "notification"),
    "asset": ("product", "installed_product", "location"),
    "reference": ("knowledge", "sla_definition"),
}

TABLE_FAMILIES: Mapping[str, str] = {
    table: family for family, tables in FAMILY_TABLES.items() for table in tables
}

#: Verbs that mutate or create state. Their count drives the multi-step signal.
ACTION_VERBS: Tuple[str, ...] = (
    "register", "create", "log", "open", "raise", "update", "assign", "set",
    "change", "link", "attach", "remove", "escalate", "close", "cancel",
    "add", "deactivate", "activate", "apply", "ensure", "provide", "grant",
    "reflect", "migrate", "reassign", "overwrite", "adjust",
)

#: Verbs that genuinely inspect state. Kept deliberately narrow: copular and
#: possessive forms such as "have"/"see" appear in almost every prompt and would
#: collapse the stratification into a single category.
READ_VERBS: Tuple[str, ...] = (
    "check", "confirm", "verify", "retrieve", "review", "validate", "inspect",
    "assess", "determine", "investigate",
)

#: Cues that a prompt sequences operations rather than stating one goal.
SEQUENCE_CUES: Tuple[str, ...] = (
    "then", "after that", "followed by", "next,", "finally", "first,", "and then",
    "before that", "once ", "as part of", "step ",
)

#: Non-entity words that start a sentence or a clause, excluded from the
#: proper-noun candidate extraction.
_PROPER_NOUN_STOPWORDS = frozenset(
    """
    a an the this that these those and or but so if when while what which who whom whose
    register create update set link attach remove ensure provide log open raise assign
    assign add make use using via using for from with into about please kindly note
    there their they them it its we our us you your i me my do does did done have has had
    is are was were be been being will would can could should may might must shall
    new old current latest first second third also only just still already each every
    any all some none more most less least very much many few both either neither
    during after before between within without across around over under above below
    one two three four five six seven eight nine ten twelve fifteen twenty thirty
    urgent critical high medium low p1 p2 p3 minor major
    january february march april may june july august september october november december
    monday tuesday wednesday thursday friday saturday sunday
    """.split()
)

#: Capitalised words that survive sentence-initial capitalisation but are verbs,
#: field labels, or qualifiers rather than entity names. The proper-noun pattern
#: necessarily matches any capitalised run, so ``Bind``, ``Then`` and
#: ``Whoever's`` are indistinguishable from ``Larson PLC`` at the regex level.
#: Filtering them here is what keeps the anchor lookup budget (Phase 1.2) spent on
#: mentions that can actually match a database value.
_ANCHOR_NOISE_WORDS = frozenset(
    """
    product issue config configuration bind pass short document assigned assigned_to
    high-end high-level low-level mid-range who whoever whatever whenever then than
    there their theirs here this that these those they've we're it's there's here's
    what's who's let's please note kindly ensure make sure keep leave send show
    get got set sets create creates created update updates updated log logs logged
    attach link linkable record records report reports check checks verify verified
    team desk group queue board list lists page pages case cases number numbers
    detail details description descriptions summary summaries status statuses
    step steps item items order orders result results option options value values
    """.split()
)

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"\+\d[\d\s()-]{6,}\d")
# Quoted literals must not span an apostrophe: prompts use possessives such as
# "task's", and a permissive pattern would capture garbage between two of them.
_QUOTED_RE = re.compile(r"['\"]([A-Za-z0-9][A-Za-z0-9 ._+\-/@()#,]{1,60})['\"](?![A-Za-z])")
_SERIAL_RE = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]{2,}){1,4}\b")
_SUFFIX_RE = re.compile(
    r"\b(?:ends?|ending|wraps? up with|digits?|suffix(?: is)?)\D{0,12}?(\d{3,})\b", re.IGNORECASE
)
_PROPER_NOUN_RE = re.compile(
    r"\b[A-Z][\w&.+'-]*(?:\s+(?:of|and|de|for|the)?\s*[A-Z][\w&.+'-]*){0,3}\b"
)
_CLAUSE_SPLIT_RE = re.compile(r"[.;:\n\r!?]|(?<=\w),\s")
_WORD_RE = re.compile(r"[a-z0-9']+")
_CONNECTORS = frozenset({"of", "and", "de", "for", "the", "&"})


@dataclass(frozen=True)
class PromptSignals:
    """Everything derived from one task prompt, before touching the database.

    Attributes:
        task_id: Task identifier.
        user_prompt: The task's raw user prompt.
        matched_tables: CSM tables whose entity phrases appear, sorted.
        actions: Distinct action verbs observed, sorted.
        reads: Distinct read verbs observed, sorted.
        sequence_cues: Sequencing cues observed, sorted.
        literals: Candidate literal anchors (quoted text, emails, phones,
            serials) in first-seen order.
        proper_nouns: Candidate proper-noun anchors, longest-first.
        serial_suffixes: Trailing digit groups such as "ends 8853".
        tool_count: Number of oracle tools the task declares.
    """

    task_id: str
    user_prompt: str
    matched_tables: Tuple[str, ...]
    actions: Tuple[str, ...]
    reads: Tuple[str, ...]
    sequence_cues: Tuple[str, ...]
    literals: Tuple[str, ...]
    proper_nouns: Tuple[str, ...]
    serial_suffixes: Tuple[str, ...]
    tool_count: int

    @property
    def entity_count(self) -> int:
        return len(self.matched_tables)

    @property
    def family_span(self) -> Tuple[str, ...]:
        return tuple(
            sorted({TABLE_FAMILIES[t] for t in self.matched_tables if t in TABLE_FAMILIES})
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "matched_tables": list(self.matched_tables),
            "actions": list(self.actions),
            "reads": list(self.reads),
            "sequence_cues": list(self.sequence_cues),
            "literals": list(self.literals),
            "proper_nouns": list(self.proper_nouns),
            "serial_suffixes": list(self.serial_suffixes),
            "tool_count": self.tool_count,
            "entity_count": self.entity_count,
            "family_span": list(self.family_span),
        }


def _match_entities(text: str, registry: SchemaRegistry) -> Tuple[str, ...]:
    """Return CSM tables referenced by the text, longest phrase first."""
    unknown = [table for table in ENTITY_PHRASES if registry.table(table) is None]
    if unknown:
        raise KeyError(
            f"ENTITY_PHRASES references tables absent from the schema registry: {unknown}"
        )
    ordered: List[Tuple[str, str]] = [
        (phrase, table) for table, phrases in ENTITY_PHRASES.items() for phrase in phrases
    ]
    ordered.sort(key=lambda item: (-len(item[0]), item[0]))

    hits: List[str] = []
    for phrase, table in ordered:
        if re.search(rf"(?<![a-z0-9_]){re.escape(phrase)}(?![a-z0-9_])", text):
            if table not in hits:
                hits.append(table)
    return tuple(sorted(hits))


def _extract_literals(text: str) -> Tuple[str, ...]:
    """Collect candidate anchor literals in first-seen order.

    Precedence is deliberate: exact identifiers (emails, quoted values, serials)
    are more reliable anchors than free-form proper nouns, so they are offered
    to live resolution first.
    """
    found: List[str] = []

    def add(value: Optional[str]) -> None:
        if not value:
            return
        cleaned = value.strip().strip(".,;:")
        if len(cleaned) < 2 or cleaned in found:
            return
        found.append(cleaned)

    for match in _EMAIL_RE.finditer(text):
        add(match.group(0))
    for match in _QUOTED_RE.finditer(text):
        add(match.group(1))
    for match in _PHONE_RE.finditer(text):
        add(match.group(0).strip())
    for match in _SERIAL_RE.finditer(text):
        add(match.group(0))
    return tuple(found)


def _extract_proper_nouns(text: str) -> Tuple[str, ...]:
    """Collect proper-noun candidates, clause by clause.

    Working inside clauses keeps field labels such as ``Product:`` or
    ``Issue:`` from gluing onto the value that follows them, and the
    stopword/connector filter drops sentence openings like ``Register a new``.
    """
    found: List[str] = []
    for clause in _CLAUSE_SPLIT_RE.split(text):
        for match in _PROPER_NOUN_RE.finditer(clause):
            phrase = match.group(0).strip(" \t")
            if len(phrase) < 4:
                continue
            raw_words = phrase.split()
            if len(raw_words) > 4:
                continue
            lowered = [word.lower().strip(".,;:'\"") for word in raw_words]
            content = [
                word
                for word in lowered
                if word and word not in _CONNECTORS and word not in _PROPER_NOUN_STOPWORDS
            ]
            if not content:
                continue
            if any(len(word) == 1 for word in content):
                continue
            if not any(word[0].isalnum() for word in content if word):
                continue
            # Drop phrases whose content words are all generic verbs, field
            # labels, or qualifiers. ``Product`` and ``Then`` are capitalised only
            # because they open a clause, and spending anchor lookups on them
            # crowds out real entity names (Phase 1.2 budget discipline).
            if all(word in _ANCHOR_NOISE_WORDS for word in content):
                continue
            if phrase not in found:
                found.append(phrase)
    # Longer, more specific names first so live resolution prefers them.
    found.sort(key=lambda item: (-len(item), item))
    return tuple(found)


def _extract_serial_suffixes(text: str) -> Tuple[str, ...]:
    found: List[str] = []
    for match in _SUFFIX_RE.finditer(text):
        digits = match.group(1)
        if digits not in found:
            found.append(digits)
    return tuple(found)


def analyze_prompt(
    user_prompt: str,
    selected_tools: Sequence[str],
    *,
    task_id: str = "",
    registry: Optional[SchemaRegistry] = None,
) -> PromptSignals:
    """Derive every prompt signal for one task.

    Raises:
        ValueError: If ``user_prompt`` is empty.
    """
    if not user_prompt or not user_prompt.strip():
        raise ValueError(f"{task_id or '<task>'}: user_prompt is empty")
    resolved_registry = registry or SchemaRegistry.from_static()
    lowered = user_prompt.lower()

    actions = tuple(
        sorted({verb for verb in ACTION_VERBS if re.search(rf"\b{verb}", lowered)})
    )
    reads = tuple(sorted({verb for verb in READ_VERBS if re.search(rf"\b{verb}", lowered)}))
    cues = tuple(sorted({cue.strip() for cue in SEQUENCE_CUES if cue in lowered}))

    return PromptSignals(
        task_id=task_id,
        user_prompt=user_prompt,
        matched_tables=_match_entities(lowered, resolved_registry),
        actions=actions,
        reads=reads,
        sequence_cues=cues,
        literals=_extract_literals(user_prompt),
        proper_nouns=_extract_proper_nouns(user_prompt),
        serial_suffixes=_extract_serial_suffixes(user_prompt),
        tool_count=len(selected_tools or ()),
    )


def classify_complexity(signals: PromptSignals) -> Tuple[str, str]:
    """Assign a state-dependence category plus a human-readable rationale.

    Only prompt-derived signals are used, so a category can never leak verifier
    expectations into the experiment inputs.
    """
    entities = signals.entity_count
    families = signals.family_span
    actions = len(signals.actions)
    interleaved = bool(signals.reads) and actions >= 2
    sequenced = bool(signals.sequence_cues) and actions >= 3

    if entities <= 1:
        return (
            "TYPE_1_SINGLE_ENTITY",
            f"single entity touched ({entities} table(s), {actions} action verb(s))",
        )
    if interleaved or sequenced:
        return (
            "TYPE_5_MULTI_STEP",
            f"read+action interleaving (reads={list(signals.reads)}, actions={actions}, "
            f"cues={list(signals.sequence_cues)})",
        )
    if entities == 2:
        return (
            "TYPE_2_ONE_HOP",
            f"two related entities across families {list(families)}",
        )
    if "asset" in families and len(families) > 1:
        return (
            "TYPE_4_CROSS_ENTITY",
            f"asset path plus {len(families) - 1} other famil(ies): {list(families)}",
        )
    return ("TYPE_3_MULTI_HOP", f"{entities} entities across families {list(families)}")


def assign_task_type(signals: PromptSignals) -> str:
    """Map prompt signals onto a registered GGQR task type.

    The builder validates the result against the real ``RequirementRegistry``,
    so an unmapped domain concept fails closed.
    """
    tables = set(signals.matched_tables)
    if {"entitlement", "contract"} & tables:
        return "check_entitlement"
    if {"knowledge", "case_knowledge"} & tables:
        return "find_knowledge"
    if "installed_product" in tables and not ({"account", "contact"} & tables):
        return "account_profile"
    if {"account", "contact", "user", "user_group"} & tables:
        return "case_overview"
    return "resolve_case"


@dataclass(frozen=True)
class TaskRecord:
    """One evaluation-set task, fully described.

    ``verifier_count`` is the *declared* count. It is used for reporting and
    stratification only and is never passed to the state model.
    """

    task_id: str
    task_config_path: str
    seed_database_file: str
    database_id: Optional[str]
    initial_state_fingerprint: Optional[str]
    complexity_category: str
    category_rationale: str
    reference_entities: Tuple[str, ...]
    reference_rows: Tuple[Tuple[str, str], ...]
    reference_names: Tuple[Tuple[str, str], ...]
    verifier_count: int
    selected_tools_count: int
    number_of_runs: int
    reset_database_between_runs: bool
    ggqr_task_type: str
    signals: PromptSignals

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_config_path": self.task_config_path,
            "seed_database_file": self.seed_database_file,
            "database_id": self.database_id,
            "initial_state_fingerprint": self.initial_state_fingerprint,
            "complexity_category": self.complexity_category,
            "category_rationale": self.category_rationale,
            "reference_entities": list(self.reference_entities),
            "reference_rows": [list(row) for row in self.reference_rows],
            "reference_names": [list(item) for item in self.reference_names],
            "verifier_count": self.verifier_count,
            "selected_tools_count": self.selected_tools_count,
            "number_of_runs": self.number_of_runs,
            "reset_database_between_runs": self.reset_database_between_runs,
            "ggqr_task_type": self.ggqr_task_type,
            "signals": self.signals.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskRecord":
        signals_payload = payload.get("signals") or {}
        signals = PromptSignals(
            task_id=payload["task_id"],
            user_prompt=signals_payload.get("user_prompt", ""),
            matched_tables=tuple(signals_payload.get("matched_tables", ())),
            actions=tuple(signals_payload.get("actions", ())),
            reads=tuple(signals_payload.get("reads", ())),
            sequence_cues=tuple(signals_payload.get("sequence_cues", ())),
            literals=tuple(signals_payload.get("literals", ())),
            proper_nouns=tuple(signals_payload.get("proper_nouns", ())),
            serial_suffixes=tuple(signals_payload.get("serial_suffixes", ())),
            tool_count=int(signals_payload.get("tool_count", 0)),
        )
        return cls(
            task_id=payload["task_id"],
            task_config_path=payload["task_config_path"],
            seed_database_file=payload.get("seed_database_file", ""),
            database_id=payload.get("database_id"),
            initial_state_fingerprint=payload.get("initial_state_fingerprint"),
            complexity_category=payload["complexity_category"],
            category_rationale=payload.get("category_rationale", ""),
            reference_entities=tuple(payload.get("reference_entities", ())),
            reference_rows=tuple(tuple(row) for row in payload.get("reference_rows", ())),
            reference_names=tuple(tuple(item) for item in payload.get("reference_names", ())),
            verifier_count=int(payload.get("verifier_count", 0)),
            selected_tools_count=int(payload.get("selected_tools_count", 0)),
            number_of_runs=int(payload.get("number_of_runs", 1)),
            reset_database_between_runs=bool(payload.get("reset_database_between_runs", True)),
            ggqr_task_type=payload.get("ggqr_task_type", "resolve_case"),
            signals=signals,
        )


@dataclass(frozen=True)
class EvalSetManifest:
    """A pinned, stratified set of CSM tasks."""

    eval_set_id: str
    generated_from: str
    schema_version: str
    tasks: Tuple[TaskRecord, ...]

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def get(self, task_id: str) -> TaskRecord:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(f"Unknown task id: {task_id!r}")

    def category_counts(self) -> Dict[str, int]:
        counts = {category: 0 for category in COMPLEXITY_CATEGORIES}
        for task in self.tasks:
            counts[task.complexity_category] = counts.get(task.complexity_category, 0) + 1
        return counts

    def as_dict(self) -> Dict[str, Any]:
        return {
            "eval_set_id": self.eval_set_id,
            "generated_from": self.generated_from,
            "schema_version": self.schema_version,
            "task_count": len(self.tasks),
            "categories": self.category_counts(),
            "tasks": [task.as_dict() for task in self.tasks],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvalSetManifest":
        tasks = tuple(TaskRecord.from_dict(entry) for entry in payload.get("tasks", ()))
        return cls(
            eval_set_id=payload.get("eval_set_id", "csm-eval-set"),
            generated_from=payload.get("generated_from", ""),
            schema_version=payload.get("schema_version", ""),
            tasks=tasks,
        )


def load_task_config(task_config_path: PathLike) -> BenchmarkConfig:
    """Load one EnterpriseOps-Gym task JSON into a ``BenchmarkConfig``.

    Mirrors ``evaluate.load_config`` normalization (underscore-prefixed
    annotation keys are dropped) without importing ``datasets``.

    Raises:
        ValueError: If prompts or verifiers are missing or malformed.
    """
    resolved_path = resolve_task_config_path(task_config_path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{task_config_path}: task config must be a JSON object")
    cleaned = {key: value for key, value in payload.items() if not key.startswith("_")}
    for required in ("system_prompt", "user_prompt"):
        if not cleaned.get(required):
            raise ValueError(f"{task_config_path}: missing {required!r}")
    if "verifiers" not in cleaned:
        raise ValueError(f"{task_config_path}: missing 'verifiers'")
    verifiers = cleaned["verifiers"]
    if isinstance(verifiers, str):
        verifiers = json.loads(verifiers)
    cleaned["verifiers"] = [
        {k: v for k, v in verifier.items() if not k.startswith("_")} for verifier in verifiers
    ]
    if isinstance(cleaned.get("gym_servers_config"), str):
        cleaned["gym_servers_config"] = json.loads(cleaned["gym_servers_config"])
    cleaned.setdefault("number_of_runs", 1)
    return BenchmarkConfig(**cleaned)


def _first_seed_database_file(config: BenchmarkConfig) -> str:
    servers = config.gym_servers_config or []
    for server in servers:
        seed = server.get("seed_database_file")
        if seed:
            return str(seed)
    return ""


def _reference_names(signals: PromptSignals) -> Tuple[Tuple[str, str], ...]:
    """Attach a table hint only when the literal's shape makes it unambiguous.

    The previous implementation assigned literals to tables round-robin. That
    created false hints such as a product serial being searched first in
    ``contact`` and made anchor resolution dependent on candidate ordering.
    Ambiguous literals remain untyped and are searched through the schema-aware
    anchor order instead.
    """
    pairs: List[Tuple[str, str]] = []
    case_number = re.compile(r"^CS-\d+$", re.IGNORECASE)
    for literal in signals.literals:
        value = literal.strip()
        if case_number.fullmatch(value):
            pairs.append(("customer_case", value))
        elif "@" in value:
            # Email can belong to a user or contact in CSM, so it is intentionally
            # left untyped rather than forcing the wrong table.
            continue
    return tuple(pairs)


#: Tool modes this harness can honour. The local task files already carry the
#: oracle tool set in ``selected_tools``; the distractor modes only exist as
#: separate Hugging Face dataset configs and are not fabricated here.
SUPPORTED_TOOL_MODES: Tuple[str, ...] = ("oracle",)


def apply_tool_mode(config: BenchmarkConfig, mode: str) -> BenchmarkConfig:
    """Return a config with the requested tool mode applied.

    ``oracle`` keeps the task's own ``selected_tools`` verbatim, which is what the
    benchmark calls the oracle tool mode.

    Raises:
        ValueError: For any mode this checkout cannot honour honestly, or when
            the task carries no ``selected_tools`` at all.
    """
    normalized = (mode or "oracle").strip().lower()
    if normalized not in SUPPORTED_TOOL_MODES:
        raise ValueError(
            f"Unsupported tool mode {mode!r}. This checkout can only run "
            f"{list(SUPPORTED_TOOL_MODES)}: distractor modes require a dataset revision "
            "that ships separate tool sets, which is a separate experiment."
        )
    if not config.selected_tools:
        raise ValueError(
            "Tool mode 'oracle' requires a task file with a non-empty 'selected_tools' list"
        )
    return config


def with_anchor(
    record: TaskRecord,
    *,
    reference_rows: Iterable[Tuple[str, str]],
    initial_state_fingerprint: Optional[str] = None,
    database_id: Optional[str] = None,
) -> TaskRecord:
    """Return a copy of ``record`` with a live-resolved anchor attached.

    Anchor resolution happens in ``ablation.state_model`` against the seeded
    database; this helper only persists the result so the manifest is auditable.
    """
    rows = tuple((str(table), str(row_id)) for table, row_id in reference_rows)
    return TaskRecord(
        task_id=record.task_id,
        task_config_path=record.task_config_path,
        seed_database_file=record.seed_database_file,
        database_id=database_id if database_id is not None else record.database_id,
        initial_state_fingerprint=(
            initial_state_fingerprint
            if initial_state_fingerprint is not None
            else record.initial_state_fingerprint
        ),
        complexity_category=record.complexity_category,
        category_rationale=record.category_rationale,
        reference_entities=record.reference_entities,
        reference_rows=rows,
        reference_names=record.reference_names,
        verifier_count=record.verifier_count,
        selected_tools_count=record.selected_tools_count,
        number_of_runs=record.number_of_runs,
        reset_database_between_runs=record.reset_database_between_runs,
        ggqr_task_type=record.ggqr_task_type,
        signals=record.signals,
    )


def build_eval_set(
    tasks_dir: PathLike,
    *,
    registry: Optional[SchemaRegistry] = None,
    eval_set_id: str = "csm-eval-set",
    tool_mode: str = "oracle",
) -> EvalSetManifest:
    """Build an evaluation set from the real task JSON files in ``tasks_dir``.

    The only thing taken from ``verifiers`` is its length, used for
    stratification and reporting. No verifier name, SQL, or expected value is
    ever read while building an intervention input.

    Raises:
        ValueError: If a task is malformed, declares no seed database, or the
            directory yields no tasks.
        KeyError: On duplicate task ids or an unregistered GGQR task type.
    """
    resolved_registry = registry or SchemaRegistry.from_static()
    requirements = RequirementRegistry(resolved_registry)
    directory = Path(tasks_dir)
    if not directory.is_dir():
        raise ValueError(f"Task directory does not exist: {directory}")
    task_files = sorted(directory.glob("task_*.json"))
    if not task_files:
        raise ValueError(f"No task_*.json files found in {directory}")

    records: List[TaskRecord] = []
    seen: set[str] = set()
    for path in task_files:
        config = apply_tool_mode(load_task_config(path), tool_mode)
        task_id = path.stem
        if task_id in seen:
            raise ValueError(f"Duplicate task id: {task_id}")
        seen.add(task_id)
        seed_file = _first_seed_database_file(config)
        if not seed_file:
            raise ValueError(f"{task_id}: task declares no seed_database_file")
        signals = analyze_prompt(
            config.user_prompt,
            config.selected_tools or (),
            task_id=task_id,
            registry=resolved_registry,
        )
        category, rationale = classify_complexity(signals)
        task_type = assign_task_type(signals)
        if not requirements.has(task_type):
            raise KeyError(
                f"{task_id}: derived GGQR task type {task_type!r} is not registered; "
                f"available: {list(requirements.task_types())}"
            )
        for table in signals.matched_tables:
            resolved_registry.require_table(table)
        try:
            portable_path = path.resolve().relative_to(GYM_ROOT).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"{task_id}: task file {path} must live under the EnterpriseOps-Gym root "
                "so the evaluation manifest can remain portable"
            ) from exc
        records.append(
            TaskRecord(
                task_id=task_id,
                task_config_path=portable_path,
                seed_database_file=seed_file,
                database_id=None,
                initial_state_fingerprint=None,
                complexity_category=category,
                category_rationale=rationale,
                reference_entities=signals.matched_tables,
                reference_rows=(),
                reference_names=_reference_names(signals),
                verifier_count=len(config.verifiers),
                selected_tools_count=len(config.selected_tools or ()),
                number_of_runs=int(config.number_of_runs or 1),
                reset_database_between_runs=bool(config.reset_database_between_runs),
                ggqr_task_type=task_type,
                signals=signals,
            )
        )

    return EvalSetManifest(
        eval_set_id=eval_set_id,
        generated_from=str(directory.resolve().relative_to(GYM_ROOT)).replace("\\", "/"),
        schema_version=resolved_registry.manifest.schema_version,
        tasks=tuple(records),
    )


def build_requirements_manifest(
    eval_set: EvalSetManifest, *, registry: Optional[SchemaRegistry] = None
) -> Dict[str, Any]:
    """Derive the GGQR requirements manifest from the real requirement registry.

    The output is directly consumable by
    ``csm_integration.build_csm_environment(requirements_path=...)`` because the
    task-type -> required-table mapping lives under the ``tasks`` key, exactly as
    ``RequirementRegistry.from_json`` expects.
    """
    resolved_registry = registry or SchemaRegistry.from_static()
    requirements = RequirementRegistry(resolved_registry)
    task_types = requirements.task_types()
    assignment: Dict[str, str] = {}
    for record in eval_set:
        if record.ggqr_task_type not in task_types:
            raise KeyError(
                f"{record.task_id}: GGQR task type {record.ggqr_task_type!r} is not registered"
            )
        assignment[record.task_id] = record.ggqr_task_type
    # P0/A3: the manifest now carries the *full* requirement set - not just
    # tables, but also the task-conditioned attributes, relations, and intent -
    # so that ``RequirementRegistry.from_json`` (consumed by
    # ``csm_integration.build_csm_environment``) can load a single source of
    # truth for the whole CSM path.
    return {
        "schema_version": resolved_registry.manifest.schema_version,
        "derived_from": "csm_env.query.requirements.RequirementRegistry",
        "tasks": {
            task_type: list(requirements.requirements_for(task_type).required_tables)
            for task_type in task_types
        },
        "required_attributes": {
            task_type: list(requirements.requirements_for(task_type).required_attributes)
            for task_type in task_types
        },
        "required_relations": {
            task_type: [
                list(pair)
                for pair in requirements.requirements_for(task_type).required_relations
            ]
            for task_type in task_types
        },
        "intent": {
            task_type: list(requirements.requirements_for(task_type).intent)
            for task_type in task_types
        },
        "task_assignment": assignment,
    }


def save_json(payload: Any, path: PathLike) -> Path:
    """Write ``payload`` as pretty-printed, key-sorted JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return target


def save_eval_set(eval_set: EvalSetManifest, path: PathLike) -> Path:
    """Persist an evaluation set manifest."""
    return save_json(eval_set.as_dict(), path)


def load_eval_set(path: PathLike) -> EvalSetManifest:
    """Load and validate an evaluation set manifest.

    Raises:
        ValueError: If the manifest has no tasks, or a task carries an unknown
            complexity category.
        KeyError: If a task id appears twice.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: evaluation set must be a JSON object")
    eval_set = EvalSetManifest.from_dict(payload)
    if not eval_set.tasks:
        raise ValueError(f"{path}: evaluation set contains no tasks")
    seen: set[str] = set()
    for record in eval_set:
        if record.task_id in seen:
            raise KeyError(f"Duplicate task id in manifest: {record.task_id}")
        seen.add(record.task_id)
        if record.complexity_category not in COMPLEXITY_CATEGORIES:
            raise ValueError(
                f"{record.task_id}: unknown complexity category "
                f"{record.complexity_category!r}; expected one of {list(COMPLEXITY_CATEGORIES)}"
            )
    return eval_set
