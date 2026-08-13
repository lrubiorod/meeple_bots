"""Command-line interface for running observable matches."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .api import Match, MatchResult, MctsAgent, RandomAgent, TicTacToe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meeple-bots")
    commands = parser.add_subparsers(dest="command", required=True)
    match = commands.add_parser("match", help="run and display one match")
    match.add_argument("--game", choices=["tic-tac-toe"], default="tic-tac-toe")
    match.add_argument("--first", choices=["mcts", "random"], default="mcts")
    match.add_argument("--second", choices=["mcts", "random"], default="random")
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
        result = Match(
            game=TicTacToe(),
            first=mcts if args.first == "mcts" else RandomAgent(),
            second=mcts if args.second == "mcts" else RandomAgent(),
            seed=args.seed,
            max_plies=args.max_plies,
        ).run()
    except (RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(_result_dict(result), indent=2))
    else:
        _print_result(result, args.first, args.second)
    return 0


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
                "action": {
                    "type": "tic_tac_toe",
                    "row": move.action.row,
                    "column": move.action.column,
                },
            }
            for ply, move in enumerate(result.moves, start=1)
        ],
    }


def _print_result(result: MatchResult, first: str, second: str) -> None:
    print(f"Player 0: {first}")
    print(f"Player 1: {second}")
    print()
    for ply, move in enumerate(result.moves, start=1):
        print(
            f"{ply}. Player {move.player} -> "
            f"row {move.action.row}, column {move.action.column}"
        )
    print()
    print("Result: draw" if result.winner is None else f"Winner: player {result.winner}")
    print(f"Utilities: {list(result.utilities)}")
    print(f"Plies: {result.plies}")
    print(f"Seed: {result.seed}")
