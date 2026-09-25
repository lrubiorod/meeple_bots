"""Human-agent callbacks and terminal interaction."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TypeAlias

from .. import _native
from .._agent_config import MctsAgent, RandomAgent, SoIsmctsAgent
from ..connect6 import Connect6, Connect6Action
from ..game_types import (
    BoardCell, Boop, BoopAction, BoopGraduateLine, BoopPiece, BoopPieceKind,
    BoopPool, BoopRecoverPiece, BoopResolution, ConnectFour,
    ConnectFourAction, EndSpiritCollection, Game, GameAction,
    HumanMoveObservation, HumanMoveObserver, HumanTurn, MoveSelector, MoveSpiritGemstone,
    PlaceSpiritGemstone, SkipSpiritGemstone, SpiritsOfTheForest,
    SpiritsOfTheForestAction, SpiritTile, TakeSpiritTile, TicTacToe, TicTacToeAction,
)
from ..native_bridge import (
    _board_rows, _boop_action_from_selector, _spirits_action_from_native,
    _spirits_state_from_native, native_agent_config,
)

@dataclass(frozen=True, slots=True)
class HumanAgent:
    """A player controlled by a Python function or an interactive terminal prompt."""

    select_action: MoveSelector = field(default=lambda turn: _prompt_human_action(turn))
    observe_action: HumanMoveObserver | None = None

    def __post_init__(self) -> None:
        if not callable(self.select_action):
            raise TypeError("select_action must be callable")
        if self.observe_action is not None and not callable(self.observe_action):
            raise TypeError("observe_action must be callable")


def _native_agent(agent: Agent, game: Game):
    """Retain the historical API seam, with human callbacks owned here."""
    if isinstance(agent, HumanAgent):
        return _native.AgentConfig.human(
            _human_selector(agent, game),
            _human_move_observer(agent, game),
        )
    return native_agent_config(agent)


def _human_selector(agent: HumanAgent, game: Game):
    if isinstance(game, SpiritsOfTheForest):
        def select_spirits(player: int, native_state, native_actions) -> int:
            board, collections, gems, phase, _active, scores = _spirits_state_from_native(
                native_state
            )
            legal_actions = tuple(
                _spirits_action_from_native(action) for action in native_actions
            )
            turn = HumanTurn(
                game=game,
                player=player,
                board=board,
                legal_actions=legal_actions,
                spirit_collections=collections,
                gemstone_pools=gems,
                scores=scores,
                phase=phase,
            )
            action = agent.select_action(turn)
            if not isinstance(
                action,
                (
                    TakeSpiritTile,
                    EndSpiritCollection,
                    PlaceSpiritGemstone,
                    MoveSpiritGemstone,
                    SkipSpiritGemstone,
                ),
            ):
                raise TypeError("human select_action must return a SpiritsOfTheForestAction")
            if action not in legal_actions:
                raise ValueError("the selected action is not currently legal")
            return legal_actions.index(action)

        return select_spirits

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
        elif isinstance(game, Connect6):
            board = _board_rows(flat_board, columns=game.board_size)
            legal_actions = tuple(Connect6Action(p) for p in native_context)
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
        elif isinstance(game, Connect6):
            expected_type = Connect6Action
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
        if isinstance(action, Connect6Action):
            return action.position
        if isinstance(action, ConnectFourAction):
            return action.column
        return legal_actions.index(action)

    return select


def _human_move_observer(agent: HumanAgent, game: Game):
    if agent.observe_action is None:
        return None

    if isinstance(game, SpiritsOfTheForest):
        def observe_spirits(player: int, native_state, native_action) -> None:
            board, collections, gems, phase, active, scores = _spirits_state_from_native(
                native_state
            )
            agent.observe_action(
                HumanMoveObservation(
                    game=game,
                    player=player,
                    action=_spirits_action_from_native(native_action),
                    board=board,
                    spirit_collections=collections,
                    gemstone_pools=gems,
                    scores=scores,
                    phase=phase,
                    active_player=active,
                )
            )

        return observe_spirits

    def observe(player: int, flat_board, native_pools, native_action) -> None:
        if isinstance(game, TicTacToe):
            board = _board_rows(flat_board, columns=3)
            pools = None
            action: GameAction = TicTacToeAction(
                row=native_action[0],
                column=native_action[1],
            )
        elif isinstance(game, Connect6):
            board = _board_rows(flat_board, columns=game.board_size)
            pools = None
            action = Connect6Action(native_action)
        elif isinstance(game, ConnectFour):
            board = _board_rows(flat_board, columns=7)
            pools = None
            action = ConnectFourAction(column=native_action)
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
                for kittens, cats in native_pools
            )
            action = _boop_action_from_selector(native_action)
        agent.observe_action(
            HumanMoveObservation(
                game=game,
                player=player,
                action=action,
                board=board,
                pools=pools,
            )
        )

    return observe


def _prompt_human_action(turn: HumanTurn) -> GameAction:
    print(file=sys.stderr)
    print(f"Current board before player {turn.player}'s move:", file=sys.stderr)
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

    if isinstance(turn.game, Connect6):
        while True:
            try:
                action = Connect6Action(int(input("Cell index (row * board_size + column): ")))
                if action in turn.legal_actions:
                    return action
            except ValueError:
                pass
            print("Choose an empty cell index.", file=sys.stderr)
    if isinstance(turn.game, ConnectFour):
        return _prompt_connect_four_action(turn)
    if isinstance(turn.game, Boop):
        return _prompt_boop_action(turn)
    if isinstance(turn.game, SpiritsOfTheForest):
        return _prompt_spirits_action(turn)
    return _prompt_tic_tac_toe_action(turn)


def _prompt_spirits_action(turn: HumanTurn) -> SpiritsOfTheForestAction:
    print(f"Phase: {turn.phase.value if turn.phase else 'unknown'}", file=sys.stderr)
    if turn.scores is not None:
        print(f"Provisional scores: {turn.scores[0]} / {turn.scores[1]}", file=sys.stderr)
    for index, action in enumerate(turn.legal_actions):
        print(f"  {index}: {_spirits_action_description(action)}", file=sys.stderr)
    while True:
        print("Choose a legal action number: ", end="", file=sys.stderr, flush=True)
        try:
            selected = int(input())
            return turn.legal_actions[selected]
        except (ValueError, IndexError):
            print("Enter one of the listed action numbers.", file=sys.stderr)


def _spirits_action_description(action: SpiritsOfTheForestAction) -> str:
    if isinstance(action, TakeSpiritTile):
        text = f"take ({action.position.row}, {action.position.column})"
        if action.sacrifice is not None:
            text += (
                " sacrificing an available gem"
                if action.sacrifice.source is None
                else " sacrificing gem at "
                f"({action.sacrifice.source.row}, {action.sacrifice.source.column})"
            )
        return text
    if isinstance(action, EndSpiritCollection):
        return "end collection"
    if isinstance(action, PlaceSpiritGemstone):
        return f"place gem at ({action.target.row}, {action.target.column})"
    if isinstance(action, MoveSpiritGemstone):
        return (
            f"move gem ({action.source.row}, {action.source.column}) to "
            f"({action.target.row}, {action.target.column})"
        )
    return "skip gemstone"


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


def _board_symbol(cell: BoardCell) -> str:
    if cell is None:
        return "."
    if isinstance(cell, int):
        return "X" if cell == 0 else "O"
    if isinstance(cell, SpiritTile):
        return cell.spirit.value[0].upper() + str(cell.spirit_symbols)
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


Agent: TypeAlias = RandomAgent | SoIsmctsAgent | MctsAgent | HumanAgent
