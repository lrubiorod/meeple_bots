"""Typed Python facade over the private Rust extension."""

from __future__ import annotations

import sys
from collections.abc import Callable
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


@dataclass(frozen=True, slots=True)
class TicTacToeAction:
    """A zero-based row and column on the tic-tac-toe board."""

    row: int
    column: int

    def __post_init__(self) -> None:
        for name, value in (("row", self.row), ("column", self.column)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= value < 3:
                raise ValueError(f"{name} must be between 0 and 2")


TicTacToeBoard: TypeAlias = tuple[
    tuple[int | None, int | None, int | None],
    tuple[int | None, int | None, int | None],
    tuple[int | None, int | None, int | None],
]


@dataclass(frozen=True, slots=True)
class HumanTurn:
    """Read-only position presented to a human move selector."""

    player: int
    board: TicTacToeBoard
    legal_actions: tuple[TicTacToeAction, ...]


MoveSelector: TypeAlias = Callable[[HumanTurn], TicTacToeAction]


@dataclass(frozen=True, slots=True)
class HumanAgent:
    """A player controlled by a Python function or an interactive terminal prompt."""

    select_action: MoveSelector = field(default=lambda turn: _prompt_human_action(turn))

    def __post_init__(self) -> None:
        if not callable(self.select_action):
            raise TypeError("select_action must be callable")


Agent: TypeAlias = RandomAgent | MctsAgent | HumanAgent


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
        if not isinstance(self.first, (RandomAgent, MctsAgent, HumanAgent)):
            raise TypeError("first must be RandomAgent, MctsAgent, or HumanAgent")
        if not isinstance(self.second, (RandomAgent, MctsAgent, HumanAgent)):
            raise TypeError("second must be RandomAgent, MctsAgent, or HumanAgent")
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
    if isinstance(agent, MctsAgent):
        return _native.AgentConfig.mcts(
            agent.iterations,
            float(agent.exploration),
            agent.rollout_depth,
        )
    return _native.AgentConfig.human(_human_selector(agent))


def _human_selector(agent: HumanAgent):
    def select(
        player: int,
        flat_board: list[int | None],
        legal_coordinates: list[tuple[int, int]],
    ) -> tuple[int, int]:
        board: TicTacToeBoard = (
            tuple(flat_board[0:3]),
            tuple(flat_board[3:6]),
            tuple(flat_board[6:9]),
        )
        turn = HumanTurn(
            player=player,
            board=board,
            legal_actions=tuple(
                TicTacToeAction(row=row, column=column)
                for row, column in legal_coordinates
            ),
        )
        action = agent.select_action(turn)
        if not isinstance(action, TicTacToeAction):
            raise TypeError("human select_action must return TicTacToeAction")
        if action not in turn.legal_actions:
            raise ValueError("the selected cell is not currently legal")
        return action.row, action.column

    return select


def _prompt_human_action(turn: HumanTurn) -> TicTacToeAction:
    symbols = {None: ".", 0: "X", 1: "O"}
    print(file=sys.stderr)
    print("    0 1 2", file=sys.stderr)
    for row, cells in enumerate(turn.board):
        rendered = " ".join(symbols[cell] for cell in cells)
        print(f"{row} | {rendered}", file=sys.stderr)

    while True:
        print(
            f"Player {turn.player}, enter row and column (for example, 1 2): ",
            end="",
            file=sys.stderr,
            flush=True,
        )
        try:
            parts = input().split()
            if len(parts) != 2:
                raise ValueError("enter exactly two numbers")
            action = TicTacToeAction(row=int(parts[0]), column=int(parts[1]))
            if action not in turn.legal_actions:
                raise ValueError("that cell is already occupied")
            return action
        except ValueError as error:
            print(f"Invalid move: {error}", file=sys.stderr)
