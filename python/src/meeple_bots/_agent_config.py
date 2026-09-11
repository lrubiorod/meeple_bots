"""Agent configuration values and game-independent validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite, sqrt
from types import MappingProxyType
from typing import TypeAlias

_MAX_U32 = 2**32 - 1


def _positive_u32(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 1 <= value <= _MAX_U32:
        raise ValueError(f"{name} must be between 1 and {_MAX_U32}")


@dataclass(frozen=True, slots=True)
class RandomAgent:
    """An agent that chooses uniformly among legal actions."""


@dataclass(frozen=True, slots=True)
class NeutralEvaluator:
    """Assign zero utility to every non-terminal state."""


@dataclass(frozen=True, slots=True)
class GameHeuristic:
    """Evaluate states with one zero-based heuristic supplied by the game."""

    index: int
    params: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_negative_u32("heuristic index", self.index)
        if not isinstance(self.params, Mapping):
            raise TypeError("heuristic params must be a mapping")
        normalized: dict[str, float] = {}
        for name, value in self.params.items():
            if not isinstance(name, str) or not name:
                raise TypeError("heuristic parameter names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"heuristic parameter {name!r} must be a number")
            if not isfinite(value):
                raise ValueError(f"heuristic parameter {name!r} must be finite")
            normalized[name] = float(value)
        object.__setattr__(self, "params", MappingProxyType(normalized))

StateEvaluator: TypeAlias = NeutralEvaluator | GameHeuristic


def _validate_state_evaluator(name: str, evaluator: object) -> None:
    if not isinstance(evaluator, (NeutralEvaluator, GameHeuristic)):
        raise TypeError(f"{name} must be NeutralEvaluator or GameHeuristic")


@dataclass(frozen=True, slots=True)
class UniformRandom:
    """Select rollout actions uniformly without evaluating successors."""


@dataclass(frozen=True, slots=True)
class Greedy:
    """Always select a rollout action with the best evaluated successor."""

    evaluator: StateEvaluator

    def __post_init__(self) -> None:
        _validate_state_evaluator("rollout evaluator", self.evaluator)


@dataclass(frozen=True, slots=True)
class EpsilonGreedy:
    """Usually select the heuristic-best rollout action and sometimes explore."""

    epsilon: float
    evaluator: StateEvaluator

    def __post_init__(self) -> None:
        if isinstance(self.epsilon, bool) or not isinstance(self.epsilon, (int, float)):
            raise TypeError("rollout epsilon must be a number")
        if not isfinite(self.epsilon) or not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("rollout epsilon must be finite and between 0.0 and 1.0")
        _validate_state_evaluator("rollout evaluator", self.evaluator)


@dataclass(frozen=True, slots=True)
class Mast:
    """Learn player/action averages per decision and use epsilon-greedy rollouts."""

    epsilon: float = 0.1

    def __post_init__(self) -> None:
        if isinstance(self.epsilon, bool) or not isinstance(self.epsilon, (int, float)):
            raise TypeError("rollout epsilon must be a number")
        if not isfinite(self.epsilon) or not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("rollout epsilon must be finite and between 0.0 and 1.0")

BaseRolloutPolicy: TypeAlias = UniformRandom | Greedy | EpsilonGreedy | Mast


@dataclass(frozen=True, slots=True)
class TurnPhaseIs:
    """Match a named game turn phase while applying a search policy."""

    phase: str

    def __post_init__(self) -> None:
        if not isinstance(self.phase, str):
            raise TypeError("turn phase must be a string")
        normalized = self.phase.strip().lower().replace("-", "_")
        if normalized == "gemstones":
            normalized = "place_gemstone"
        if normalized not in {"collect", "place_gemstone", "choose", "continue"}:
            raise ValueError(
                "turn phase must be choose, continue, collect or place_gemstone"
            )
        object.__setattr__(self, "phase", normalized)


@dataclass(frozen=True, slots=True)
class ConditionalRollout:
    """Use one rollout policy when a condition matches and another otherwise."""

    condition: TurnPhaseIs
    primary: BaseRolloutPolicy
    fallback: BaseRolloutPolicy = field(default_factory=UniformRandom)

    def __post_init__(self) -> None:
        if not isinstance(self.condition, TurnPhaseIs):
            raise TypeError("rollout condition must be TurnPhaseIs")
        if not isinstance(self.primary, (UniformRandom, Greedy, EpsilonGreedy, Mast)):
            raise TypeError("primary rollout policy must be a base rollout policy")
        if not isinstance(self.fallback, (UniformRandom, Greedy, EpsilonGreedy, Mast)):
            raise TypeError("fallback rollout policy must be a base rollout policy")

RolloutPolicy: TypeAlias = BaseRolloutPolicy | ConditionalRollout


@dataclass(frozen=True, slots=True)
class ProgressiveBias:
    """Add a decaying heuristic prior to UCT tree selection."""

    weight: float
    evaluator: StateEvaluator
    condition: TurnPhaseIs | None = None

    def __post_init__(self) -> None:
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise TypeError("progressive bias weight must be a number")
        if not isfinite(self.weight) or self.weight < 0:
            raise ValueError(
                "progressive bias weight must be finite and non-negative"
            )
        _validate_state_evaluator("progressive bias evaluator", self.evaluator)
        if self.condition is not None and not isinstance(self.condition, TurnPhaseIs):
            raise TypeError("progressive bias condition must be TurnPhaseIs or None")


@dataclass(frozen=True, slots=True)
class MctsAgent:
    """Configuration for the Monte Carlo Tree Search agent."""

    iterations: int | None = None
    exploration: float = sqrt(2.0)
    rollout_depth: int = 256
    heuristic: int | None = None
    cutoff_evaluator: StateEvaluator | None = None
    rollout_policy: RolloutPolicy = field(default_factory=UniformRandom)
    time_budget: float | None = None
    progressive_bias: ProgressiveBias | None = None
    root_diagnostics: bool = False
    tree_reuse: bool = False
    transpositions: bool = False
    selection_policy: str = "uct"

    def __post_init__(self) -> None:
        if self.selection_policy not in ("uct", "ucb1_tuned"):
            raise ValueError("selection_policy must be uct or ucb1_tuned")
        if self.iterations is None and self.time_budget is None:
            object.__setattr__(self, "iterations", 1_000)
        elif self.iterations is not None and self.time_budget is not None:
            raise ValueError("iterations and time_budget are mutually exclusive")
        if self.iterations is not None:
            _positive_u32("iterations", self.iterations)
        if self.time_budget is not None:
            if isinstance(self.time_budget, bool) or not isinstance(
                self.time_budget, (int, float)
            ):
                raise TypeError("time_budget must be a number")
            if not isfinite(self.time_budget) or self.time_budget <= 0:
                raise ValueError("time_budget must be finite and greater than zero")
        _positive_u32("rollout_depth", self.rollout_depth)
        if isinstance(self.exploration, bool) or not isinstance(self.exploration, (int, float)):
            raise TypeError("exploration must be a number")
        if not isfinite(self.exploration) or self.exploration < 0:
            raise ValueError("exploration must be finite and non-negative")
        if self.heuristic is not None:
            _non_negative_u32("heuristic", self.heuristic)
        cutoff_evaluator = self.cutoff_evaluator
        if cutoff_evaluator is None:
            cutoff_evaluator = (
                NeutralEvaluator()
                if self.heuristic is None
                else GameHeuristic(self.heuristic)
            )
            object.__setattr__(self, "cutoff_evaluator", cutoff_evaluator)
        else:
            _validate_state_evaluator("cutoff_evaluator", cutoff_evaluator)
            evaluator_heuristic = (
                cutoff_evaluator.index
                if isinstance(cutoff_evaluator, GameHeuristic)
                else None
            )
            if self.heuristic is not None and self.heuristic != evaluator_heuristic:
                raise ValueError("heuristic and cutoff_evaluator configure different evaluators")
            object.__setattr__(self, "heuristic", evaluator_heuristic)
        if not isinstance(
            self.rollout_policy,
            (UniformRandom, Greedy, EpsilonGreedy, Mast, ConditionalRollout),
        ):
            raise TypeError(
                "rollout_policy must be UniformRandom, Greedy, EpsilonGreedy, Mast, "
                "or ConditionalRollout"
            )
        if self.progressive_bias is not None and not isinstance(
            self.progressive_bias, ProgressiveBias
        ):
            raise TypeError("progressive_bias must be ProgressiveBias or None")
        if not isinstance(self.root_diagnostics, bool):
            raise TypeError("root_diagnostics must be a boolean")
        if not isinstance(self.tree_reuse, bool):
            raise TypeError("tree_reuse must be a boolean")
        if not isinstance(self.transpositions, bool):
            raise TypeError("transpositions must be a boolean")


def _non_negative_u32(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= _MAX_U32:
        raise ValueError(f"{name} must be between 0 and {_MAX_U32}")
