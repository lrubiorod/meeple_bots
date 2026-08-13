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
class ConnectFour:
    """The standard 6x7 Connect Four rules with gravity."""


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


@dataclass(frozen=True, slots=True)
class ConnectFourAction:
    """A zero-based column in which to drop a Connect Four piece."""

    column: int

    def __post_init__(self) -> None:
        if isinstance(self.column, bool) or not isinstance(self.column, int):
            raise TypeError("column must be an integer")
        if not 0 <= self.column < 7:
            raise ValueError("column must be between 0 and 6")


Game: TypeAlias = TicTacToe | ConnectFour
GameAction: TypeAlias = TicTacToeAction | ConnectFourAction
GameBoard: TypeAlias = tuple[tuple[int | None, ...], ...]


@dataclass(frozen=True, slots=True)
class HumanTurn:
    """Read-only position presented to a human move selector."""

    game: Game
    player: int
    board: GameBoard
    legal_actions: tuple[GameAction, ...]


MoveSelector: TypeAlias = Callable[[HumanTurn], GameAction]


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
    action: GameAction


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

    game: Game = field(default_factory=TicTacToe)
    first: Agent = field(default_factory=MctsAgent)
    second: Agent = field(default_factory=RandomAgent)
    seed: int = 0
    max_plies: int = 10_000

    def __post_init__(self) -> None:
        if not isinstance(self.game, (TicTacToe, ConnectFour)):
            raise TypeError("game must be TicTacToe or ConnectFour")
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
            _native_game(self.game),
            _native_agent(self.first, self.game),
            _native_agent(self.second, self.game),
            self.seed,
            self.max_plies,
        )
        moves = tuple(
            Move(
                player=item["player"],
                action=_action_from_native(item["action"]),
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


def _native_game(game: Game) -> str:
    return "tic_tac_toe" if isinstance(game, TicTacToe) else "connect_four"


def _action_from_native(raw: dict[str, object]) -> GameAction:
    if raw["type"] == "tic_tac_toe":
        return TicTacToeAction(row=raw["row"], column=raw["column"])
    return ConnectFourAction(column=raw["column"])


def _native_agent(agent: Agent, game: Game):
    if isinstance(agent, RandomAgent):
        return _native.AgentConfig.random()
    if isinstance(agent, MctsAgent):
        return _native.AgentConfig.mcts(
            agent.iterations,
            float(agent.exploration),
            agent.rollout_depth,
        )
    return _native.AgentConfig.human(_human_selector(agent, game))


def _human_selector(agent: HumanAgent, game: Game):
    def select(
        player: int,
        flat_board: list[int | None],
        native_legal_actions: list[tuple[int, int]] | list[int],
    ) -> tuple[int, int] | int:
        if isinstance(game, TicTacToe):
            board = _board_rows(flat_board, columns=3)
            legal_actions: tuple[GameAction, ...] = tuple(
                TicTacToeAction(row=row, column=column)
                for row, column in native_legal_actions
            )
        else:
            board = _board_rows(flat_board, columns=7)
            legal_actions = tuple(
                ConnectFourAction(column=column) for column in native_legal_actions
            )
        turn = HumanTurn(
            game=game,
            player=player,
            board=board,
            legal_actions=legal_actions,
        )
        action = agent.select_action(turn)
        expected_type = TicTacToeAction if isinstance(game, TicTacToe) else ConnectFourAction
        if not isinstance(action, expected_type):
            raise TypeError(f"human select_action must return {expected_type.__name__}")
        if action not in turn.legal_actions:
            raise ValueError("the selected action is not currently legal")
        if isinstance(action, TicTacToeAction):
            return action.row, action.column
        return action.column

    return select


def _board_rows(flat_board: list[int | None], columns: int) -> GameBoard:
    return tuple(
        tuple(flat_board[start : start + columns])
        for start in range(0, len(flat_board), columns)
    )


def _prompt_human_action(turn: HumanTurn) -> GameAction:
    symbols = {None: ".", 0: "X", 1: "O"}
    print(file=sys.stderr)
    print("    " + " ".join(str(column) for column in range(len(turn.board[0]))), file=sys.stderr)
    for row, cells in enumerate(turn.board):
        rendered = " ".join(symbols[cell] for cell in cells)
        print(f"{row} | {rendered}", file=sys.stderr)

    if isinstance(turn.game, ConnectFour):
        return _prompt_connect_four_action(turn)
    return _prompt_tic_tac_toe_action(turn)


def _prompt_connect_four_action(turn: HumanTurn) -> ConnectFourAction:
    while True:
        print(
            f"Player {turn.player}, enter column (0-6): ",
            end="",
            file=sys.stderr,
            flush=True,
        )
        try:
            action = ConnectFourAction(column=int(input()))
            if action not in turn.legal_actions:
                raise ValueError("that column is full")
            return action
        except ValueError as error:
            print(f"Invalid move: {error}", file=sys.stderr)


def _prompt_tic_tac_toe_action(turn: HumanTurn) -> TicTacToeAction:
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
