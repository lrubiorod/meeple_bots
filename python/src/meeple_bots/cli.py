"""Command-line interface for running observable matches."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .api import (
    ConnectFour,
    ConnectFourAction,
    HumanAgent,
    Match,
    MatchResult,
    MctsAgent,
    RandomAgent,
    TicTacToe,
    TicTacToeAction,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meeple-bots")
    commands = parser.add_subparsers(dest="command", required=True)
    match = commands.add_parser("match", help="run and display one match")
    match.add_argument(
        "--game", choices=["connect-four", "tic-tac-toe"], default="tic-tac-toe"
    )
    match.add_argument("--first", choices=["human", "mcts", "random"], default="mcts")
    match.add_argument("--second", choices=["human", "mcts", "random"], default="random")
    match.add_argument("--seed", type=int, default=0)
    match.add_argument("--max-plies", type=int, default=10_000)
    match.add_argument("--mcts-iterations", type=int, default=1_000)
    match.add_argument("--mcts-exploration", type=float, default=sqrt_two())
    match.add_argument("--mcts-rollout-depth", type=int, default=256)
    match.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser


def sqrt_two() -> float:
    return 2.0**0.5


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        mcts = MctsAgent(
            iterations=args.mcts_iterations,
            exploration=args.mcts_exploration,
            rollout_depth=args.mcts_rollout_depth,
        )
        game = ConnectFour() if args.game == "connect-four" else TicTacToe()
        result = Match(
            game=game,
            first=_agent(args.first, mcts),
            second=_agent(args.second, mcts),
            seed=args.seed,
            max_plies=args.max_plies,
        ).run()
    except (RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(_result_dict(result), indent=2))
    else:
        _print_result(result, game, args.first, args.second)
    return 0


def _agent(name: str, mcts: MctsAgent) -> HumanAgent | MctsAgent | RandomAgent:
    if name == "human":
        return HumanAgent()
    if name == "mcts":
        return mcts
    return RandomAgent()


def _result_dict(result: MatchResult) -> dict[str, object]:
    return {
        "seed": result.seed,
        "plies": result.plies,
        "utilities": list(result.utilities),
        "winner": result.winner,
        "moves": [
            {
                "ply": ply,
                "player": move.player,
                "action": _action_dict(move.action),
            }
            for ply, move in enumerate(result.moves, start=1)
        ],
    }


def _action_dict(action: TicTacToeAction | ConnectFourAction) -> dict[str, object]:
    if isinstance(action, TicTacToeAction):
        return {
            "type": "tic_tac_toe",
            "row": action.row,
            "column": action.column,
        }
    return {"type": "connect_four", "column": action.column}


def _print_result(
    result: MatchResult,
    game: TicTacToe | ConnectFour,
    first: str,
    second: str,
) -> None:
    print(f"Player 0: {first}")
    print(f"Player 1: {second}")
    print()
    for ply, move in enumerate(result.moves, start=1):
        if isinstance(move.action, TicTacToeAction):
            selected = f"row {move.action.row}, column {move.action.column}"
        else:
            selected = f"column {move.action.column}"
        print(f"{ply}. Player {move.player} -> {selected}")
    print()
    print("Final board:")
    _print_board(_replay_board(result, game))
    print()
    print("Result: draw" if result.winner is None else f"Winner: player {result.winner}")
    print(f"Utilities: {list(result.utilities)}")
    print(f"Plies: {result.plies}")
    print(f"Seed: {result.seed}")


def _replay_board(
    result: MatchResult, game: TicTacToe | ConnectFour
) -> list[list[int | None]]:
    rows, columns = (3, 3) if isinstance(game, TicTacToe) else (6, 7)
    board: list[list[int | None]] = [[None] * columns for _ in range(rows)]

    for move in result.moves:
        if isinstance(move.action, TicTacToeAction):
            board[move.action.row][move.action.column] = move.player
            continue

        row = next(
            row
            for row in range(rows - 1, -1, -1)
            if board[row][move.action.column] is None
        )
        board[row][move.action.column] = move.player

    return board


def _print_board(board: list[list[int | None]]) -> None:
    symbols = {None: ".", 0: "X", 1: "O"}
    print("    " + " ".join(str(column) for column in range(len(board[0]))))
    for row, cells in enumerate(board):
        print(f"{row} | " + " ".join(symbols[cell] for cell in cells))
