"""Command-line interface for running observable matches."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .api import (
    Boop,
    BoopAction,
    BoopGraduateLine,
    BoopPiece,
    BoopPieceKind,
    BoopRecoverPiece,
    ConnectFour,
    ConnectFourAction,
    GameEvaluationReport,
    HumanAgent,
    Match,
    MatchResult,
    MctsAgent,
    RandomAgent,
    TicTacToe,
    TicTacToeAction,
    evaluate_game,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meeple-bots")
    commands = parser.add_subparsers(dest="command", required=True)
    match = commands.add_parser("match", help="run and display one match")
    match.add_argument(
        "--game", choices=["boop", "connect-four", "tic-tac-toe"], default="tic-tac-toe"
    )
    match.add_argument("--first", choices=["human", "mcts", "random"], default="mcts")
    match.add_argument("--second", choices=["human", "mcts", "random"], default="random")
    match.add_argument("--seed", type=int, default=0)
    match.add_argument("--max-plies", type=int, default=10_000)
    match.add_argument("--mcts-iterations", type=int)
    match.add_argument("--mcts-exploration", type=float, default=sqrt_two())
    match.add_argument("--mcts-rollout-depth", type=int)
    match.add_argument(
        "--first-mcts-heuristic",
        nargs="?",
        const=0,
        type=int,
        help="use the selected game's heuristic (index 0 when no value is given)",
    )
    match.add_argument(
        "--second-mcts-heuristic",
        nargs="?",
        const=0,
        type=int,
        help="use the selected game's heuristic (index 0 when no value is given)",
    )
    match.add_argument("--json", action="store_true", help="print machine-readable JSON")

    analyze = commands.add_parser("analyze", help="measure game complexity and calibrate MCTS")
    analyze.add_argument(
        "--game", choices=["boop", "connect-four", "tic-tac-toe"], required=True
    )
    analyze.add_argument("--samples", type=int, default=128)
    analyze.add_argument("--max-depth", type=int, default=256)
    analyze.add_argument("--seed", type=int, default=0)
    analyze.add_argument("--json", action="store_true", help="print machine-readable JSON")

    return parser


def sqrt_two() -> float:
    return 2.0**0.5


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        game = _game(args.game)
        if args.command == "analyze":
            report = evaluate_game(
                game,
                samples=args.samples,
                max_depth=args.max_depth,
                seed=args.seed,
            )
            if args.json:
                print(json.dumps(_evaluation_dict(report), indent=2))
            else:
                _print_evaluation(report)
            return 0

        mcts = _mcts_configuration(args)
        first = _agent(
            args.first,
            mcts,
            args.first_mcts_heuristic,
            "--first-mcts-heuristic",
        )
        second = _agent(
            args.second,
            mcts,
            args.second_mcts_heuristic,
            "--second-mcts-heuristic",
        )
        result = Match(
            game=game,
            first=first,
            second=second,
            seed=args.seed,
            max_plies=args.max_plies,
        ).run()
    except (RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.json:
        payload = _result_dict(result)
        payload["players"] = [
            _agent_dict(args.first, first),
            _agent_dict(args.second, second),
        ]
        print(json.dumps(payload, indent=2))
    else:
        _print_result(result, args.first, args.second, first, second)
    return 0


def _game(name: str) -> TicTacToe | ConnectFour | Boop:
    if name == "boop":
        return Boop()
    if name == "connect-four":
        return ConnectFour()
    return TicTacToe()


def _mcts_configuration(args: argparse.Namespace) -> MctsAgent:
    return MctsAgent(
        iterations=1_000 if args.mcts_iterations is None else args.mcts_iterations,
        exploration=args.mcts_exploration,
        rollout_depth=(
            256 if args.mcts_rollout_depth is None else args.mcts_rollout_depth
        ),
    )


def _agent(
    name: str,
    mcts: MctsAgent,
    heuristic: int | None,
    option: str,
) -> HumanAgent | MctsAgent | RandomAgent:
    if heuristic is not None and name != "mcts":
        raise ValueError(f"{option} requires the corresponding player to be MCTS")
    if name == "human":
        return HumanAgent()
    if name == "mcts":
        return MctsAgent(
            iterations=mcts.iterations,
            exploration=mcts.exploration,
            rollout_depth=mcts.rollout_depth,
            heuristic=heuristic,
        )
    return RandomAgent()


def _agent_dict(name: str, agent) -> dict[str, object]:
    return {
        "type": name,
        "heuristic": agent.heuristic if isinstance(agent, MctsAgent) else None,
    }


def _heuristic_name(heuristic: int | None) -> str:
    return "none" if heuristic is None else str(heuristic)


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


def _action_dict(
    action: TicTacToeAction | ConnectFourAction | BoopAction,
) -> dict[str, object]:
    if isinstance(action, TicTacToeAction):
        return {
            "type": "tic_tac_toe",
            "row": action.row,
            "column": action.column,
        }
    if isinstance(action, ConnectFourAction):
        return {"type": "connect_four", "column": action.column}
    return {
        "type": "boop",
        "piece": action.piece.value,
        "row": action.row,
        "column": action.column,
        "resolution": _resolution_dict(action),
    }


def _print_result(
    result: MatchResult,
    first: str,
    second: str,
    first_agent,
    second_agent,
) -> None:
    print(f"Player 0: {_agent_name(first, first_agent)}")
    print(f"Player 1: {_agent_name(second, second_agent)}")
    print()
    for ply, move in enumerate(result.moves, start=1):
        if isinstance(move.action, TicTacToeAction):
            selected = f"row {move.action.row}, column {move.action.column}"
        elif isinstance(move.action, ConnectFourAction):
            selected = f"column {move.action.column}"
        else:
            selected = (
                f"{move.action.piece.value} at row {move.action.row}, "
                f"column {move.action.column}"
            )
            resolution = _resolution_text(move.action)
            if resolution:
                selected += f"; {resolution}"
        print(f"{ply}. Player {move.player} -> {selected}")
    print()
    print("Final board:")
    _print_board(result.final_board)
    if result.pools is not None:
        for player, pool in enumerate(result.pools):
            print(f"Player {player} pool: {pool.kittens} kittens, {pool.cats} cats")
    print()
    print("Result: draw" if result.winner is None else f"Winner: player {result.winner}")
    print(f"Utilities: {list(result.utilities)}")
    print(f"Plies: {result.plies}")
    print(f"Seed: {result.seed}")


def _agent_name(name: str, agent) -> str:
    if isinstance(agent, MctsAgent):
        return f"{name} (heuristic {_heuristic_name(agent.heuristic)})"
    return name


def _print_board(board) -> None:
    print("    " + " ".join(str(column) for column in range(len(board[0]))))
    for row, cells in enumerate(board):
        print(f"{row} | " + " ".join(_piece_symbol(cell) for cell in cells))


def _piece_symbol(piece) -> str:
    if piece is None:
        return "."
    if isinstance(piece, int):
        return "X" if piece == 0 else "O"
    if not isinstance(piece, BoopPiece):
        raise TypeError("unknown board piece")
    if piece.player == 0:
        return "x" if piece.kind is BoopPieceKind.KITTEN else "X"
    return "o" if piece.kind is BoopPieceKind.KITTEN else "O"


def _resolution_dict(action: BoopAction) -> dict[str, object]:
    if isinstance(action.resolution, BoopGraduateLine):
        return {
            "type": "graduate",
            "positions": [
                {"row": position.row, "column": position.column}
                for position in action.resolution.positions
            ],
        }
    if isinstance(action.resolution, BoopRecoverPiece):
        return {
            "type": "recover",
            "row": action.resolution.position.row,
            "column": action.resolution.position.column,
        }
    return {"type": "none"}


def _resolution_text(action: BoopAction) -> str:
    if isinstance(action.resolution, BoopGraduateLine):
        positions = ", ".join(
            f"({position.row}, {position.column})"
            for position in action.resolution.positions
        )
        return f"graduate {positions}"
    if isinstance(action.resolution, BoopRecoverPiece):
        position = action.resolution.position
        return f"recover ({position.row}, {position.column})"
    return ""


def _evaluation_dict(report: GameEvaluationReport) -> dict[str, object]:
    return {
        "game": _game_name(report.game),
        "samples": report.samples,
        "max_depth": report.max_depth,
        "terminal_rate": report.terminal_rate,
        "initial_legal_actions": report.initial_legal_actions,
        "effective_branching_factor": report.effective_branching_factor,
        "estimated_depth": report.estimated_depth,
        "depth_is_lower_bound": report.depth_is_lower_bound,
        "estimated_tree_log10": report.estimated_tree_log10,
        "recommended_rollout_depth": report.recommended_rollout_depth,
        "recommended_iterations": report.recommended_iterations,
        "iterations_capped": report.iterations_capped,
        "milliseconds_per_iteration": report.milliseconds_per_iteration,
        "estimated_decision_time_ms": report.estimated_decision_time_ms,
    }


def _print_evaluation(report: GameEvaluationReport) -> None:
    depth_note = " (lower bound)" if report.depth_is_lower_bound else ""
    iteration_note = (
        " (capped; structural estimate is higher)" if report.iterations_capped else ""
    )

    print(f"Game: {_game_name(report.game)}")
    print()
    print("Game structure:")
    print(f"  Initial legal actions: {report.initial_legal_actions}")
    print(f"  Effective branching factor: {report.effective_branching_factor:.2f}")
    print(f"  Estimated depth (p95): {report.estimated_depth}{depth_note}")
    print(f"  Terminal samples: {report.terminal_rate:.1%}")
    print(f"  Estimated tree size: 10^{report.estimated_tree_log10:.1f}")
    print()
    print("MCTS estimate for this machine:")
    print(f"  Rollout depth: {report.recommended_rollout_depth}")
    print(f"  Recommended iterations: {report.recommended_iterations:,}{iteration_note}")
    print(f"  Cost per iteration: ~{report.milliseconds_per_iteration:.6f} ms")
    print(f"  Estimated decision time: ~{report.estimated_decision_time_ms:.1f} ms")
    print()
    print("Interpretation:")
    if report.depth_is_lower_bound:
        print(
            "  - Some samples did not finish. Increase --max-depth or consider "
            "a heuristic for truncated rollouts."
        )
    else:
        print("  - Sampled full-depth rollouts usually reach a terminal result.")
    print(
        "  - If the estimated decision time is too high, reducing rollout depth "
        "usually makes a state heuristic more useful."
    )


def _game_name(game: TicTacToe | ConnectFour | Boop) -> str:
    if isinstance(game, TicTacToe):
        return "tic-tac-toe"
    if isinstance(game, ConnectFour):
        return "connect-four"
    return "boop"
