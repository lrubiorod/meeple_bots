"""Typed Python facade over the private Rust extension."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
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
class Boop:
    """The standard two-player rules for boop. on a 6x6 bed."""


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


class BoopPieceKind(str, Enum):
    """The two ranks of pieces in boop."""

    KITTEN = "kitten"
    CAT = "cat"


@dataclass(frozen=True, slots=True)
class BoopPosition:
    """A zero-based position on the 6x6 boop. board."""

    row: int
    column: int

    def __post_init__(self) -> None:
        for name, value in (("row", self.row), ("column", self.column)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= value < 6:
                raise ValueError(f"{name} must be between 0 and 5")


@dataclass(frozen=True, slots=True)
class BoopGraduateLine:
    """The exact line of three pieces selected for graduation."""

    positions: tuple[BoopPosition, BoopPosition, BoopPosition]

    def __post_init__(self) -> None:
        if not isinstance(self.positions, tuple) or len(self.positions) != 3:
            raise TypeError("positions must be a tuple containing exactly three positions")
        if not all(isinstance(position, BoopPosition) for position in self.positions):
            raise TypeError("every graduation position must be a BoopPosition")


@dataclass(frozen=True, slots=True)
class BoopRecoverPiece:
    """The piece selected for recovery when all eight pieces are on the board."""

    position: BoopPosition

    def __post_init__(self) -> None:
        if not isinstance(self.position, BoopPosition):
            raise TypeError("position must be a BoopPosition")


BoopResolution: TypeAlias = BoopGraduateLine | BoopRecoverPiece | None


@dataclass(frozen=True, slots=True)
class BoopAction:
    """A placement and any mandatory end-of-turn resolution in boop."""

    piece: BoopPieceKind
    row: int
    column: int
    resolution: BoopResolution = None

    def __post_init__(self) -> None:
        if not isinstance(self.piece, BoopPieceKind):
            raise TypeError("piece must be a BoopPieceKind")
        BoopPosition(self.row, self.column)


@dataclass(frozen=True, slots=True)
class BoopPiece:
    """A kitten or cat on the board and its owner."""

    player: int
    kind: BoopPieceKind

    def __post_init__(self) -> None:
        if self.player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        if not isinstance(self.kind, BoopPieceKind):
            raise TypeError("kind must be a BoopPieceKind")


@dataclass(frozen=True, slots=True)
class BoopPool:
    """The kittens and cats currently available to one player."""

    kittens: int
    cats: int

    def __post_init__(self) -> None:
        for name, value in (("kittens", self.kittens), ("cats", self.cats)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= value <= 8:
                raise ValueError(f"{name} must be between 0 and 8")


Game: TypeAlias = TicTacToe | ConnectFour | Boop
GameAction: TypeAlias = TicTacToeAction | ConnectFourAction | BoopAction
BoardCell: TypeAlias = int | BoopPiece | None
GameBoard: TypeAlias = tuple[tuple[BoardCell, ...], ...]


@dataclass(frozen=True, slots=True)
class HumanTurn:
    """Read-only position presented to a human move selector."""

    game: Game
    player: int
    board: GameBoard
    legal_actions: tuple[GameAction, ...]
    pools: tuple[BoopPool, BoopPool] | None = None


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
    final_board: GameBoard
    pools: tuple[BoopPool, BoopPool] | None


@dataclass(frozen=True, slots=True)
class Match:
    """Configuration for one match executed by the Rust engine."""

    game: Game = field(default_factory=TicTacToe)
    first: Agent = field(default_factory=MctsAgent)
    second: Agent = field(default_factory=RandomAgent)
    seed: int = 0
    max_plies: int = 10_000

    def __post_init__(self) -> None:
        if not isinstance(self.game, (TicTacToe, ConnectFour, Boop)):
            raise TypeError("game must be TicTacToe, ConnectFour, or Boop")
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
            final_board=_final_board_from_native(raw["final_board"], self.game),
            pools=_pools_from_native(raw["pools"]),
        )


def _native_game(game: Game) -> str:
    if isinstance(game, TicTacToe):
        return "tic_tac_toe"
    if isinstance(game, ConnectFour):
        return "connect_four"
    return "boop"


def _action_from_native(raw: dict[str, object]) -> GameAction:
    if raw["type"] == "tic_tac_toe":
        return TicTacToeAction(row=raw["row"], column=raw["column"])
    if raw["type"] == "connect_four":
        return ConnectFourAction(column=raw["column"])
    return BoopAction(
        piece=BoopPieceKind(raw["piece"]),
        row=raw["row"],
        column=raw["column"],
        resolution=_boop_resolution_from_native(raw["resolution"]),
    )


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
        flat_board,
        native_context,
        native_boop_actions=None,
    ) -> tuple[int, int] | int:
        if isinstance(game, TicTacToe):
            board = _board_rows(flat_board, columns=3)
            legal_actions: tuple[GameAction, ...] = tuple(
                TicTacToeAction(row=row, column=column)
                for row, column in native_context
            )
            pools = None
        elif isinstance(game, ConnectFour):
            board = _board_rows(flat_board, columns=7)
            legal_actions = tuple(
                ConnectFourAction(column=column) for column in native_context
            )
            pools = None
        else:
            board = _board_rows(
                [
                    None
                    if piece is None
                    else BoopPiece(player=piece[0], kind=BoopPieceKind(piece[1]))
                    for piece in flat_board
                ],
                columns=6,
            )
            pools = tuple(
                BoopPool(kittens=kittens, cats=cats)
                for kittens, cats in native_context
            )
            legal_actions = tuple(
                _boop_action_from_selector(action) for action in native_boop_actions
            )
        turn = HumanTurn(
            game=game,
            player=player,
            board=board,
            legal_actions=legal_actions,
            pools=pools,
        )
        action = agent.select_action(turn)
        if isinstance(game, TicTacToe):
            expected_type = TicTacToeAction
        elif isinstance(game, ConnectFour):
            expected_type = ConnectFourAction
        else:
            expected_type = BoopAction
        if not isinstance(action, expected_type):
            raise TypeError(f"human select_action must return {expected_type.__name__}")
        if action not in turn.legal_actions:
            raise ValueError("the selected action is not currently legal")
        if isinstance(action, TicTacToeAction):
            return action.row, action.column
        if isinstance(action, ConnectFourAction):
            return action.column
        return legal_actions.index(action)

    return select


def _board_rows(flat_board: list[BoardCell], columns: int) -> GameBoard:
    return tuple(
        tuple(flat_board[start : start + columns])
        for start in range(0, len(flat_board), columns)
    )


def _prompt_human_action(turn: HumanTurn) -> GameAction:
    print(file=sys.stderr)
    print("    " + " ".join(str(column) for column in range(len(turn.board[0]))), file=sys.stderr)
    for row, cells in enumerate(turn.board):
        rendered = " ".join(_board_symbol(cell) for cell in cells)
        print(f"{row} | {rendered}", file=sys.stderr)

    if turn.pools is not None:
        for player, pool in enumerate(turn.pools):
            print(
                f"Player {player} pool: {pool.kittens} kittens, {pool.cats} cats",
                file=sys.stderr,
            )

    if isinstance(turn.game, ConnectFour):
        return _prompt_connect_four_action(turn)
    if isinstance(turn.game, Boop):
        return _prompt_boop_action(turn)
    return _prompt_tic_tac_toe_action(turn)


def _prompt_boop_action(turn: HumanTurn) -> BoopAction:
    while True:
        print(
            f"Player {turn.player}, enter piece and position (for example, k 2 3): ",
            end="",
            file=sys.stderr,
            flush=True,
        )
        try:
            parts = input().lower().split()
            if len(parts) != 3:
                raise ValueError("enter k or c followed by two numbers")
            piece = {"k": BoopPieceKind.KITTEN, "c": BoopPieceKind.CAT}.get(parts[0])
            if piece is None:
                raise ValueError("piece must be k (kitten) or c (cat)")
            row, column = int(parts[1]), int(parts[2])
            candidates = [
                action
                for action in turn.legal_actions
                if isinstance(action, BoopAction)
                and action.piece == piece
                and action.row == row
                and action.column == column
            ]
            if not candidates:
                raise ValueError("that placement is not currently legal")
            if len(candidates) == 1:
                return candidates[0]
            return _prompt_boop_resolution(candidates)
        except ValueError as error:
            print(f"Invalid move: {error}", file=sys.stderr)


def _prompt_boop_resolution(candidates: list[BoopAction]) -> BoopAction:
    print("Choose the end-of-turn resolution:", file=sys.stderr)
    for index, action in enumerate(candidates):
        print(f"  {index}: {_resolution_description(action.resolution)}", file=sys.stderr)
    while True:
        print("Resolution number: ", end="", file=sys.stderr, flush=True)
        try:
            return candidates[int(input())]
        except (ValueError, IndexError):
            print("Invalid resolution number", file=sys.stderr)


def _resolution_description(resolution: BoopResolution) -> str:
    if isinstance(resolution, BoopGraduateLine):
        positions = ", ".join(
            f"({position.row}, {position.column})" for position in resolution.positions
        )
        return f"graduate line {positions}"
    if isinstance(resolution, BoopRecoverPiece):
        return f"recover ({resolution.position.row}, {resolution.position.column})"
    return "no resolution"


def _boop_action_from_selector(raw) -> BoopAction:
    piece, row, column, raw_resolution = raw
    resolution_type, positions = raw_resolution
    if resolution_type == "graduate":
        resolution: BoopResolution = BoopGraduateLine(
            tuple(BoopPosition(row, column) for row, column in positions)
        )
    elif resolution_type == "recover":
        resolution = BoopRecoverPiece(BoopPosition(*positions[0]))
    else:
        resolution = None
    return BoopAction(BoopPieceKind(piece), row, column, resolution)


def _boop_resolution_from_native(raw) -> BoopResolution:
    if raw["type"] == "graduate":
        return BoopGraduateLine(
            tuple(BoopPosition(row, column) for row, column in raw["positions"])
        )
    if raw["type"] == "recover":
        return BoopRecoverPiece(BoopPosition(raw["row"], raw["column"]))
    return None


def _final_board_from_native(flat_board, game: Game) -> GameBoard:
    cells: list[BoardCell] = []
    for piece in flat_board:
        if piece is None:
            cells.append(None)
        elif piece["kind"] == "token":
            cells.append(piece["player"])
        else:
            cells.append(
                BoopPiece(
                    player=piece["player"],
                    kind=BoopPieceKind(piece["kind"]),
                )
            )
    columns = 3 if isinstance(game, TicTacToe) else 7 if isinstance(game, ConnectFour) else 6
    return _board_rows(cells, columns)


def _pools_from_native(raw) -> tuple[BoopPool, BoopPool] | None:
    if raw is None:
        return None
    pools = tuple(BoopPool(kittens=pool["kittens"], cats=pool["cats"]) for pool in raw)
    return pools


def _board_symbol(cell: BoardCell) -> str:
    if cell is None:
        return "."
    if isinstance(cell, int):
        return "X" if cell == 0 else "O"
    if cell.player == 0:
        return "x" if cell.kind is BoopPieceKind.KITTEN else "X"
    return "o" if cell.kind is BoopPieceKind.KITTEN else "O"


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
