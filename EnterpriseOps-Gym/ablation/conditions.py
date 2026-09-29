"""Frozen experimental conditions for Experiment 1.

Three conditions exist (State Model 2.0 split):

* ``A_baseline``          - the frozen ReAct agent with native CSM tools only;
* ``B1_state_model_broad`` - plus the full graph-connected grounded state;
* ``B2_state_model_minimal`` - plus the task-conditioned minimal grounded state
  (relevance-ranked rows, minimal-sufficient selection, attribute pruning).

They differ in exactly two binaries: whether the frozen ``csm_env``
graph-grounded state model is made available to the agent, and - when it is -
whether the task-conditioned minimal selector prunes the retrieved state.

Compatibility: ``B_state_model`` is the historical name of the *broad* arm.
It now resolves to ``B1_state_model_broad``, and :data:`CONDITION_B` keeps
that meaning so archived ``B_state_model`` records are never re-interpreted.
The minimal arm has its own distinct name.

``C = B + transition model`` is intentionally **not** defined here. Adding it
would require its own plan; see ``docs/experiments/experiment-1-protocol.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

EXPERIMENT_NAME = "experiment_1"


@dataclass(frozen=True)
class Condition:
    """One arm of the paired ablation.

    Attributes:
        name: Stable identifier used in file names, reports, and the run index.
        state_model: Whether the GGQR/``GroundedState`` representation is
            exposed to the agent.
        selection_mode: ``broad`` keeps the full retrieved state; ``minimal``
            applies task-conditioned pruning before the state is injected.
    """

    name: str
    state_model: bool
    selection_mode: str = "broad"

    def __post_init__(self) -> None:
        normalized = (self.selection_mode or "broad").strip().lower()
        if normalized not in {"broad", "minimal"}:
            raise ValueError(
                f"Unknown condition selection_mode {self.selection_mode!r}; "
                "expected 'broad' or 'minimal'"
            )
        object.__setattr__(self, "selection_mode", normalized)

    @property
    def minimal_state(self) -> bool:
        return self.selection_mode == "minimal"

    @property
    def label(self) -> str:
        if not self.state_model:
            return "native agent"
        if self.minimal_state:
            return "+ csm_env minimal state model"
        return "+ csm_env broad state model"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "state_model": self.state_model,
            "selection_mode": self.selection_mode,
            "label": self.label,
        }


CONDITION_A = Condition(name="A_baseline", state_model=False, selection_mode="broad")
CONDITION_B1 = Condition(name="B1_state_model_broad", state_model=True, selection_mode="broad")
CONDITION_B2 = Condition(name="B2_state_model_minimal", state_model=True, selection_mode="minimal")

#: Historical alias. ``B_state_model`` always meant the broad intervention; the
#: State Model 2.0 work split it into B1 (broad) and B2 (minimal), so the alias
#: resolves to B1 to keep archived records semantically stable. The canonical
#: export for new code is :data:`CONDITION_B1`/:data:`CONDITION_B2`.
CONDITION_B = CONDITION_B1

CONDITIONS: Tuple[Condition, ...] = (CONDITION_A, CONDITION_B1, CONDITION_B2)

_BY_NAME = {
    condition.name: condition
    for condition in CONDITIONS
}
_BY_NAME.update({
    "A": CONDITION_A,
    "B": CONDITION_B1,
    "B1": CONDITION_B1,
    "B2": CONDITION_B2,
    "B_state_model": CONDITION_B1,
    "B_state_model_broad": CONDITION_B1,
    "B_state_model_minimal": CONDITION_B2,
})

#: A transition-model arm is refused by name so that requesting it fails loudly
#: instead of silently degrading into Condition B.
FORBIDDEN_CONDITION_NAMES: Tuple[str, ...] = (
    "C_transition",
    "C",
    "transition",
    "C_state_and_transition",
)


def condition_from_name(name: str) -> Condition:
    """Resolve a condition by name, failing closed on anything else."""
    if name in FORBIDDEN_CONDITION_NAMES:
        raise KeyError(
            f"Condition {name!r} requires a transition model, which is Experiment 2 "
            "and is out of scope for Experiment 1."
        )
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"Unknown condition {name!r}. Available: {sorted(_BY_NAME)}"
        ) from None


def parse_conditions(raw: str) -> Tuple[Condition, ...]:
    """Parse a comma-separated ``--conditions`` argument.

    Args:
        raw: e.g. ``"A"``, ``"B1"``, ``"B2"``, ``"A,B1,B2"``. Bare letters are
            expanded to the canonical condition names (``B`` keeps its
            historical meaning of the broad arm, B1); full names are accepted
            as-is.

    Raises:
        ValueError: If the list is empty, resolves to nothing, or duplicates a
            condition.
    """
    if not raw or not raw.strip():
        raise ValueError("condition list is empty; expected e.g. 'A,B1,B2'")
    mapping = {
        "A": "A_baseline",
        "B": "B1_state_model_broad",
        "B1": "B1_state_model_broad",
        "B2": "B2_state_model_minimal",
    }
    resolved = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        candidate = token if "_" in token else mapping[token]
        resolved.append(condition_from_name(candidate))
    if not resolved:
        raise ValueError(f"condition list {raw!r} resolved to nothing")
    seen: set[str] = set()
    for condition in resolved:
        if condition.name in seen:
            raise ValueError(f"duplicate condition in {raw!r}: {condition.name}")
        seen.add(condition.name)
    return tuple(resolved)


def is_state_model_condition(condition: Condition) -> bool:
    """Whether a condition exposes the state model to the agent.

    This is the property-based key for every delivery gate and report section,
    replacing the old name-literal check that silently broke when the B arm
    was split into B1/B2.
    """
    return condition.state_model


#: Cache for :func:`condition_for`; conditions are frozen and immutable.
_CONDITION_CACHE: dict = {}


def condition_for(name: str) -> Condition:
    """Resolve a condition by name, tolerating archived unknown names.

    Records persisted by older builds can carry condition names this build no
    longer defines. Those resolve to a non-state-model placeholder so they are
    reported by name but never satisfy a state-model delivery gate.
    """
    if name in _CONDITION_CACHE:
        return _CONDITION_CACHE[name]
    try:
        condition = condition_from_name(name)
    except KeyError:
        condition = Condition(name=name, state_model=False)
    _CONDITION_CACHE[name] = condition
    return condition
