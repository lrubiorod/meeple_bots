"""Typed Python facade over the private Rust extension."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from typing import TypeAlias

from . import _native

_MAX_U32 = 2**32 - 1
_MAX_U64 = 2**64 - 1


def _positive_u32(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 1 <= value <= _MAX_U32:
        raise ValueError(f"{name} must be between 1 and {_MAX_U32}")


@dataclass(frozen=True, slots=True)
class TicTacToe:
    """The standard 3x3 tic-tac-toe rules."""


@dataclass(frozen=True, slots=True)
class RandomAgent:
    """An agent that chooses uniformly among legal actions."""


@dataclass(frozen=True, slots=True)
class MctsAgent:
    """Configuration for the Monte Carlo Tree Search agent."""

    iterations: int = 1_000
    exploration: float = sqrt(2.0)
    rollout_depth: int = 256

    def __post_init__(self) -> None:
        _positive_u32("iterations", self.iterations)
        _positive_u32("rollout_depth", self.rollout_depth)
        if isinstance(self.exploration, bool) or not isinstance(self.exploration, (int, float)):
            raise TypeError("exploration must be a number")
        if not isfinite(self.exploration) or self.exploration < 0:
            raise ValueError("exploration must be finite and non-negative")


Agent: TypeAlias = RandomAgent | MctsAgent


@dataclass(frozen=True, slots=True)
class TicTacToeAction:
    """A zero-based row and column on the tic-tac-toe board."""

    row: int
    column: int


@dataclass(frozen=True, slots=True)
class Move:
    """One action selected by one player."""

    player: int
    action: TicTacToeAction


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Immutable summary and full action history of a completed match."""

    seed: int
    plies: int
    utilities: tuple[float, ...]
    winner: int | None
    moves: tuple[Move, ...]


@dataclass(frozen=True, slots=True)
class Match:
    """Configuration for one match executed by the Rust engine."""

    game: TicTacToe = field(default_factory=TicTacToe)
    first: Agent = field(default_factory=MctsAgent)
    second: Agent = field(default_factory=RandomAgent)
    seed: int = 0
    max_plies: int = 10_000

    def __post_init__(self) -> None:
        if not isinstance(self.game, TicTacToe):
            raise TypeError("game must be TicTacToe")
        if not isinstance(self.first, (RandomAgent, MctsAgent)):
            raise TypeError("first must be RandomAgent or MctsAgent")
        if not isinstance(self.second, (RandomAgent, MctsAgent)):
            raise TypeError("second must be RandomAgent or MctsAgent")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= self.seed <= _MAX_U64:
            raise ValueError(f"seed must be between 0 and {_MAX_U64}")
        _positive_u32("max_plies", self.max_plies)

    def run(self) -> MatchResult:
        """Execute the match and return its complete immutable report."""

        raw = _native.run_match(
            "tic_tac_toe",
            _native_agent(self.first),
            _native_agent(self.second),
            self.seed,
            self.max_plies,
        )
        moves = tuple(
            Move(
                player=item["player"],
                action=TicTacToeAction(
                    row=item["action"]["row"],
                    column=item["action"]["column"],
                ),
            )
            for item in raw["moves"]
        )
        return MatchResult(
            seed=raw["seed"],
            plies=raw["plies"],
            utilities=tuple(raw["utilities"]),
            winner=raw["winner"],
            moves=moves,
        )


def _native_agent(agent: Agent):
    if isinstance(agent, RandomAgent):
        return _native.AgentConfig.random()
    return _native.AgentConfig.mcts(
        agent.iterations,
        float(agent.exploration),
        agent.rollout_depth,
    )
