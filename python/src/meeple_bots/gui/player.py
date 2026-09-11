"""Shared graphical player configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..api import MctsAgent
from .._agent_config import UniformRandom, Greedy, EpsilonGreedy, Mast, ConditionalRollout, RolloutPolicy, ProgressiveBias
from .._mcts_profiles import _configured_rollout_policy, _configured_progressive_bias
from ..serialization import _evaluator_dict

PlayerKind = Literal["human", "random", "mcts"]


@dataclass(frozen=True, slots=True)
class GuiPlayer:
    """Configuration for one player shown in a graphical interface."""

    kind: PlayerKind
    iterations: int | None = 1_000
    time_budget: float | None = None
    exploration: float = 2.0**0.5
    rollout_depth: int = 256
    heuristic: int | None = None
    tree_reuse: bool = False
    transpositions: bool = False
    selection_policy: str = "uct"

    def __post_init__(self) -> None:
        if self.kind not in ("human", "random", "mcts"):
            raise ValueError("player kind must be human, random, or mcts")
        if self.kind == "mcts":
            MctsAgent(
                iterations=self.iterations,
                time_budget=self.time_budget,
                exploration=self.exploration,
                rollout_depth=self.rollout_depth,
                heuristic=self.heuristic,
                tree_reuse=self.tree_reuse,
                transpositions=self.transpositions,
                selection_policy=self.selection_policy,
            )

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-compatible representation used by GUI state."""

        return {
            "kind": self.kind,
            "iterations": self.iterations,
            "time_budget": self.time_budget,
            "exploration": self.exploration,
            "rollout_depth": self.rollout_depth,
            "heuristic": self.heuristic,
            "tree_reuse": self.tree_reuse,
            "transpositions": self.transpositions,
            "selection_policy": self.selection_policy,
        }


def _policy_dict(policy):
    if isinstance(policy, ConditionalRollout):
        return {"kind": "conditional", "condition": {"kind": "turn_phase", "phase": policy.condition.phase},
                "primary": _policy_dict(policy.primary), "fallback": _policy_dict(policy.fallback)}
    if isinstance(policy, UniformRandom):
        return {"kind": "uniform_random"}
    if isinstance(policy, Mast):
        return {"kind": "mast", "epsilon": policy.epsilon}
    result = {"kind": "greedy" if isinstance(policy, Greedy) else "epsilon_greedy", "evaluator": _evaluator_dict(policy.evaluator)}
    if isinstance(policy, EpsilonGreedy):
        result["epsilon"] = policy.epsilon
    return result


@dataclass(frozen=True, slots=True)
class ConfiguredGuiPlayer(GuiPlayer):
    """Opt-in GUI transport for the existing structured search policies."""

    rollout_policy: RolloutPolicy = field(default_factory=UniformRandom)
    progressive_bias: ProgressiveBias | None = None
    root_diagnostics: bool = False

    def __post_init__(self):
        GuiPlayer.__post_init__(self)
        if self.kind == "mcts":
            self.to_agent()

    def to_agent(self):
        return MctsAgent(**{k: v for k, v in GuiPlayer.as_dict(self).items() if k != "kind"},
                         rollout_policy=self.rollout_policy, progressive_bias=self.progressive_bias,
                         root_diagnostics=self.root_diagnostics)

    def as_dict(self):
        bias = self.progressive_bias
        return {**GuiPlayer.as_dict(self), "rollout_policy": _policy_dict(self.rollout_policy),
                "progressive_bias": None if bias is None else {
                    "weight": bias.weight, "evaluator": _evaluator_dict(bias.evaluator),
                    "condition": None if bias.condition is None else {"kind": "turn_phase", "phase": bias.condition.phase}},
                "root_diagnostics": self.root_diagnostics}


def parse_gui_player(
    raw: Any,
    name: str,
    *,
    default_rollout_depth: int,
    available_heuristics: tuple[int, ...] = (),
    default_mcts: GuiPlayer | None = None,
    with_policies: bool = False,
) -> GuiPlayer:
    """Validate one player configuration received from a browser."""

    if not isinstance(raw, dict):
        raise ValueError(f"{name} player configuration must be an object")
    if raw.get("kind") == "mcts" and default_mcts is not None:
        defaults = default_mcts.as_dict()
        # Supplying either budget selects that mode; never inherit a second budget.
        if "iterations" in raw or "time_budget" in raw:
            defaults["iterations"] = None
            defaults["time_budget"] = None
        raw = {**defaults, **raw}
    heuristic = raw.get("heuristic")
    if heuristic is not None:
        if isinstance(heuristic, bool) or not isinstance(heuristic, int):
            raise ValueError(f"{name} heuristic must be an integer or null")
        if heuristic not in available_heuristics:
            available = ", ".join(str(index) for index in available_heuristics) or "none"
            raise ValueError(f"{name} heuristic must be one of: {available}")
    player_type = ConfiguredGuiPlayer if with_policies else GuiPlayer
    extra = {} if not with_policies else {
        "rollout_policy": _configured_rollout_policy(raw, name),
        "progressive_bias": _configured_progressive_bias(raw, name),
        "root_diagnostics": raw.get("root_diagnostics", False),
    }
    return player_type(
        **extra,
        kind=raw.get("kind"),
        iterations=raw.get("iterations", 1_000),
        time_budget=raw.get("time_budget"),
        exploration=raw.get("exploration", 2.0**0.5),
        rollout_depth=raw.get("rollout_depth", default_rollout_depth),
        heuristic=heuristic,
        tree_reuse=raw.get("tree_reuse", False),
        transpositions=raw.get("transpositions", False),
        selection_policy=raw.get("selection_policy", "uct"),
    )
