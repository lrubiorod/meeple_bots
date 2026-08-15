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
    HumanAgent,
    GameComplexityReport,
    MctsStrengthReport,
    MctsStrengthProgress,
    Match,
    MatchResult,
    MctsAgent,
    MctsLevel,
    MctsRecommendation,
    RandomAgent,
    OpponentResult,
    StrengthProgressStage,
    TicTacToe,
    TicTacToeAction,
    evaluate_game_complexity,
    evaluate_mcts_strength,
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
    match.add_argument("--mcts-level", choices=[level.value for level in MctsLevel])
    match.add_argument("--mcts-time-ms", type=int)
    match.add_argument("--json", action="store_true", help="print machine-readable JSON")

    analyze = commands.add_parser("analyze", help="measure game complexity and calibrate MCTS")
    analyze.add_argument(
        "--game", choices=["boop", "connect-four", "tic-tac-toe"], required=True
    )
    analyze.add_argument("--samples", type=int, default=128)
    analyze.add_argument("--max-depth", type=int, default=256)
    analyze.add_argument("--seed", type=int, default=0)
    analyze.add_argument("--json", action="store_true", help="print machine-readable JSON")

    assess = commands.add_parser(
        "assess", help="benchmark MCTS strength and inspect search behavior"
    )
    assess.add_argument(
        "--game", choices=["boop", "connect-four", "tic-tac-toe"], required=True
    )
    assess.add_argument("--seed", type=int, default=0)
    assess.add_argument("--samples", type=int, default=128)
    assess.add_argument("--max-depth", type=int, default=256)
    assess.add_argument("--matches", type=int, default=20)
    assess.add_argument("--max-plies", type=int, default=10_000)
    assess.add_argument("--reference-multiplier", type=int, default=4)
    assess.add_argument("--mcts-iterations", type=int)
    assess.add_argument("--mcts-exploration", type=float, default=sqrt_two())
    assess.add_argument("--mcts-rollout-depth", type=int)
    assess.add_argument(
        "--mcts-heuristic",
        nargs="?",
        const=0,
        type=int,
        help="use the selected game's heuristic (index 0 when no value is given)",
    )
    assess.add_argument("--mcts-level", choices=[level.value for level in MctsLevel])
    assess.add_argument("--mcts-time-ms", type=int)
    assess.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser


def sqrt_two() -> float:
    return 2.0**0.5


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        game = _game(args.game)
        if args.command == "analyze":
            report = evaluate_game_complexity(
                game,
                samples=args.samples,
                max_depth=args.max_depth,
                seed=args.seed,
            )
            if args.json:
                print(json.dumps(_complexity_dict(report), indent=2))
            else:
                _print_complexity(report)
            return 0

        if args.command == "assess":
            print("[setup] Analyzing game complexity and calibrating MCTS...", file=sys.stderr, flush=True)
            complexity = evaluate_game_complexity(
                game,
                samples=args.samples,
                max_depth=args.max_depth,
                seed=args.seed,
                heuristic=args.mcts_heuristic,
            )
            mcts = _assessment_mcts_configuration(args, complexity)
            print(
                f"[setup] Starting {args.matches * 2} benchmark matches "
                f"with {mcts.iterations} candidate iterations and "
                f"rollout depth {mcts.rollout_depth}, "
                f"heuristic {_heuristic_name(mcts.heuristic)}...",
                file=sys.stderr,
                flush=True,
            )
            report = evaluate_mcts_strength(
                game,
                mcts,
                matches_per_opponent=args.matches,
                reference_iterations_multiplier=args.reference_multiplier,
                max_plies=args.max_plies,
                seed=args.seed,
                complexity_report=complexity,
                progress=_print_strength_progress,
            )
            if args.json:
                print(json.dumps(_strength_dict(report), indent=2))
            else:
                _print_strength(report)
            return 0

        mcts, recommendation = _mcts_configuration(args, game)
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
        if recommendation is not None:
            payload["mcts_recommendation"] = _recommendation_dict(recommendation)
        print(json.dumps(payload, indent=2))
    else:
        _print_result(result, args.first, args.second, first, second, recommendation)
    return 0


def _game(name: str) -> TicTacToe | ConnectFour | Boop:
    if name == "boop":
        return Boop()
    if name == "connect-four":
        return ConnectFour()
    return TicTacToe()


def _mcts_configuration(
    args: argparse.Namespace,
    game: TicTacToe | ConnectFour | Boop,
) -> tuple[MctsAgent, MctsRecommendation | None]:
    if args.mcts_level is None:
        if args.mcts_time_ms is not None:
            raise ValueError("--mcts-time-ms requires --mcts-level")
        return (
            MctsAgent(
                iterations=(
                    1_000 if args.mcts_iterations is None else args.mcts_iterations
                ),
                exploration=args.mcts_exploration,
                rollout_depth=(
                    256 if args.mcts_rollout_depth is None else args.mcts_rollout_depth
                ),
            ),
            None,
        )
    if args.mcts_iterations is not None or args.mcts_rollout_depth is not None:
        raise ValueError(
            "--mcts-level cannot be combined with --mcts-iterations or "
            "--mcts-rollout-depth"
        )
    if "mcts" not in (args.first, args.second):
        raise ValueError("--mcts-level requires at least one MCTS player")

    report = evaluate_game_complexity(game, seed=args.seed)
    recommendation = report.recommend(
        MctsLevel(args.mcts_level),
        time_budget_ms=args.mcts_time_ms,
    )
    return (
        MctsAgent.from_recommendation(
            recommendation,
            exploration=args.mcts_exploration,
        ),
        recommendation,
    )


def _assessment_mcts_configuration(
    args: argparse.Namespace,
    complexity: GameComplexityReport,
) -> MctsAgent:
    manual = args.mcts_iterations is not None or args.mcts_rollout_depth is not None
    if manual:
        if args.mcts_level is not None:
            raise ValueError(
                "--mcts-level cannot be combined with --mcts-iterations or "
                "--mcts-rollout-depth"
            )
        if args.mcts_time_ms is not None:
            raise ValueError("--mcts-time-ms requires --mcts-level")
        return MctsAgent(
            iterations=1_000 if args.mcts_iterations is None else args.mcts_iterations,
            exploration=args.mcts_exploration,
            rollout_depth=(
                256
                if args.mcts_rollout_depth is None
                else args.mcts_rollout_depth
            ),
            heuristic=args.mcts_heuristic,
        )

    level = MctsLevel.FAST if args.mcts_level is None else MctsLevel(args.mcts_level)
    recommendation = complexity.recommend(level, time_budget_ms=args.mcts_time_ms)
    return MctsAgent.from_recommendation(
        recommendation,
        exploration=args.mcts_exploration,
        heuristic=args.mcts_heuristic,
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
    recommendation: MctsRecommendation | None = None,
) -> None:
    print(f"Player 0: {_agent_name(first, first_agent)}")
    print(f"Player 1: {_agent_name(second, second_agent)}")
    if recommendation is not None:
        print(
            "MCTS configuration: "
            f"{recommendation.level.value}, {recommendation.iterations} iterations, "
            f"depth {recommendation.rollout_depth}, "
            f"~{recommendation.estimated_time_ms:.1f} ms/decision"
        )
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


def _complexity_dict(report: GameComplexityReport) -> dict[str, object]:
    return {
        "game": _game_name(report.game),
        "samples": report.samples,
        "max_depth": report.max_depth,
        "completed_samples": report.completed_samples,
        "terminal_rate": report.terminal_rate,
        "initial_legal_actions": report.initial_legal_actions,
        "mean_branching_factor": report.mean_branching_factor,
        "effective_branching_factor": report.effective_branching_factor,
        "maximum_branching_factor": report.maximum_branching_factor,
        "p95_branching_factor": report.p95_branching_factor,
        "mean_plies": report.mean_plies,
        "median_plies": report.median_plies,
        "p75_plies": report.p75_plies,
        "p95_plies": report.p95_plies,
        "estimated_tree_log10": report.estimated_tree_log10,
        "estimate_is_lower_bound": report.estimate_is_lower_bound,
        "max_iterations": report.max_iterations,
        "recommendations": [
            _recommendation_dict(recommendation)
            for recommendation in report.recommendations
        ],
    }


def _recommendation_dict(recommendation: MctsRecommendation) -> dict[str, object]:
    return {
        "level": recommendation.level.value,
        "iterations": recommendation.iterations,
        "rollout_depth": recommendation.rollout_depth,
        "target_time_ms": recommendation.target_time_ms,
        "estimated_time_ms": recommendation.estimated_time_ms,
        "milliseconds_per_iteration": recommendation.milliseconds_per_iteration,
    }


def _strength_dict(report: MctsStrengthReport) -> dict[str, object]:
    return {
        "game": _game_name(report.game),
        "candidate": {
            "iterations": report.candidate.iterations,
            "exploration": report.candidate.exploration,
            "rollout_depth": report.candidate.rollout_depth,
            "heuristic": report.candidate.heuristic,
        },
        "reference": {
            "iterations": report.reference.iterations,
            "exploration": report.reference.exploration,
            "rollout_depth": report.reference.rollout_depth,
            "heuristic": report.reference.heuristic,
        },
        "matches_per_opponent": report.matches_per_opponent,
        "initial_expanded_nodes": report.initial_expanded_nodes,
        "tree_size_log10_gap": report.tree_size_log10_gap,
        "tree_size_estimate_is_lower_bound": (
            report.tree_size_estimate_is_lower_bound
        ),
        "search": {
            "decisions": report.search.decisions,
            "total_iterations": report.search.total_iterations,
            "mean_expanded_nodes": report.search.mean_expanded_nodes,
            "maximum_expanded_nodes": report.search.maximum_expanded_nodes,
            "mean_root_actions": report.search.mean_root_actions,
            "mean_iterations_per_root_action": (
                report.search.mean_iterations_per_root_action
            ),
            "mean_tree_revisit_rate": report.search.mean_tree_revisit_rate,
            "mean_tree_depth": report.search.mean_tree_depth,
            "maximum_tree_depth": report.search.maximum_tree_depth,
            "mean_simulation_depth": report.search.mean_simulation_depth,
            "maximum_simulation_depth": report.search.maximum_simulation_depth,
            "terminal_rollout_rate": report.search.terminal_rollout_rate,
            "truncated_rollout_rate": report.search.truncated_rollout_rate,
            "mean_selected_action_visit_share": (
                report.search.mean_selected_action_visit_share
            ),
        },
        "versus_random": _opponent_dict(report.versus_random),
        "versus_reference": _opponent_dict(report.versus_reference),
        "search_sufficiency": report.search_sufficiency.value,
        "benchmark_confidence": report.benchmark_confidence.value,
        "strength_estimate": report.strength_estimate.value,
        "cutoff_heuristic_evidence": report.cutoff_heuristic_evidence.value,
        "reasons": list(report.reasons),
    }


def _opponent_dict(result: OpponentResult) -> dict[str, object]:
    return {
        "matches": result.matches,
        "wins": result.wins,
        "draws": result.draws,
        "losses": result.losses,
        "score": result.score,
        "mean_utility": result.mean_utility,
        "utility_confidence_low": result.utility_confidence_low,
        "utility_confidence_high": result.utility_confidence_high,
    }


def _print_complexity(report: GameComplexityReport) -> None:
    lower_bound = (
        " (lower bound: some samples were truncated)"
        if report.estimate_is_lower_bound
        else ""
    )
    print(f"Game: {_game_name(report.game)}")
    print(f"Samples completed: {report.completed_samples}/{report.samples}")
    print(f"Terminal rate: {report.terminal_rate:.1%}")
    print(
        "Branching factor: "
        f"mean {report.mean_branching_factor:.2f}, "
        f"effective {report.effective_branching_factor:.2f}, "
        f"p95 {report.p95_branching_factor}, max {report.maximum_branching_factor}"
    )
    print(
        "Game length: "
        f"mean {report.mean_plies:.1f}, median {report.median_plies}, "
        f"p95 {report.p95_plies} plies"
    )
    print(f"Estimated tree size: 10^{report.estimated_tree_log10:.1f}{lower_bound}")
    print()
    print("MCTS recommendations for this machine:")
    for recommendation in report.recommendations:
        print(
            f"  {recommendation.level.value}: {recommendation.iterations} iterations, "
            f"depth {recommendation.rollout_depth}, "
            f"~{recommendation.estimated_time_ms:.1f} ms/decision"
        )


def _print_strength(report: MctsStrengthReport) -> None:
    print(f"Game: {_game_name(report.game)}")
    print(
        "Candidate: "
        f"{report.candidate.iterations} iterations, "
        f"depth {report.candidate.rollout_depth}, "
        f"heuristic {_heuristic_name(report.candidate.heuristic)}"
    )
    print(
        "Reference: "
        f"{report.reference.iterations} iterations, "
        f"depth {report.reference.rollout_depth}, "
        f"heuristic {_heuristic_name(report.reference.heuristic)}"
    )
    print()
    print("Assessment:")
    print(f"  Search sufficiency: {report.search_sufficiency.value}")
    print(
        "  Benchmark confidence: "
        f"{report.benchmark_confidence.value} "
        f"({report.matches_per_opponent} matches/opponent)"
    )
    print(f"  Relative strength: {report.strength_estimate.value}")
    print(
        "  Cutoff heuristic evidence: "
        f"{report.cutoff_heuristic_evidence.value}"
    )
    print(f"  Primary limitation: {_primary_limitation(report)}")
    print()
    print("Observed search:")
    print(
        f"  {report.search.decisions} decisions, "
        f"{report.search.mean_expanded_nodes:.1f} mean expanded nodes"
    )
    print(
        "  Root sampling: "
        f"{report.search.mean_root_actions:.1f} mean actions, "
        f"{report.search.mean_iterations_per_root_action:.2f} iterations/action"
    )
    print(f"  Existing-tree revisits: {report.search.mean_tree_revisit_rate:.1%}")
    print(
        "  Tree depth: "
        f"mean {report.search.mean_tree_depth:.1f}, "
        f"max {report.search.maximum_tree_depth}"
    )
    print(
        "  Simulation depth: "
        f"mean {report.search.mean_simulation_depth:.1f}, "
        f"max {report.search.maximum_simulation_depth}"
    )
    print(
        "  Rollouts: "
        f"{report.search.terminal_rollout_rate:.1%} terminal, "
        f"{report.search.truncated_rollout_rate:.1%} truncated"
    )
    print()
    _print_opponent_result("Random", report.versus_random)
    _print_opponent_result("higher-iteration MCTS", report.versus_reference)
    print()
    print(
        "Scale context (not a strength metric): initial search is approximately "
        f"10^{report.tree_size_log10_gap:.1f} below the sampled tree estimate"
    )
    print("Interpretation:")
    for reason in report.reasons:
        print(f"  - {reason}")


def _primary_limitation(report: MctsStrengthReport) -> str:
    if report.search_sufficiency.value == "insufficient":
        return "iteration budget; do not interpret playing strength yet"
    if report.benchmark_confidence.value == "low":
        return "match sample; run more matches before interpreting strength"
    if report.cutoff_heuristic_evidence.value in ("moderate", "high"):
        return "rollout cutoff; a non-terminal state heuristic may help"
    return "no single dominant limitation detected"


def _print_opponent_result(name: str, result: OpponentResult) -> None:
    print(
        f"Against {name}: {result.wins}W/{result.draws}D/{result.losses}L, "
        f"score {result.score:.1%}, mean utility {result.mean_utility:.3f} "
        f"(95% CI {result.utility_confidence_low:.3f} to "
        f"{result.utility_confidence_high:.3f})"
    )


def _print_strength_progress(progress: MctsStrengthProgress) -> None:
    opponent = "random" if progress.opponent.value == "random" else "reference MCTS"
    prefix = f"[{progress.match_number}/{progress.total_matches}]"
    if progress.stage is StrengthProgressStage.STARTED:
        print(
            f"{prefix} Starting against {opponent}; "
            f"candidate is player {progress.candidate_player}...",
            file=sys.stderr,
            flush=True,
        )
        return

    utility = progress.utility
    result = "win" if utility > 0 else "loss" if utility < 0 else "draw"
    print(
        f"{prefix} Completed in {progress.elapsed_seconds:.2f}s: "
        f"{result}, {progress.plies} plies",
        file=sys.stderr,
        flush=True,
    )


def _game_name(game: TicTacToe | ConnectFour | Boop) -> str:
    if isinstance(game, TicTacToe):
        return "tic-tac-toe"
    if isinstance(game, ConnectFour):
        return "connect-four"
    return "boop"
