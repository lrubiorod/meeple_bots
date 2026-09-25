"""Human-facing CLI output and terminal move presentation."""
from __future__ import annotations

import sys
from pathlib import Path

from .._concurrency import resolve_workers
from ..api import (
    Batch, BatchProgress, BatchProgressStatus, BatchResult, Boop, BoopAction,
    BoopGraduateLine, BoopPiece, BoopPieceKind, BoopRecoverPiece,
    ConditionalRollout, ConnectFour, ConnectFourAction, EndSpiritCollection,
    GameHeuristic, HumanMoveObservation, MatchResult, MctsAgent,
    MoveSpiritGemstone, NeutralEvaluator, PlaceSpiritGemstone, ProgressiveBias,
    RandomAgent, SkipSpiritGemstone, SpiritTile, SpiritsOfTheForest,
    TakeSpiritTile, TicTacToe, TicTacToeAction,
)
from ..connect6 import Connect6Action
from ..serialization import (
    game_name as _game_name, _base_rollout_policy_name, _rollout_policy_name,
    _rollout_policy_evaluator, _rollout_policy_epsilon,
)


def _print_tournament_start(header: dict[str, object]) -> None:
    print(
        f"Starting tournament: {header['game']}, {len(header['agents'])} agents, "
        f"{header['total_pairings']} pairings, {header['total_matches']} matches, {header['workers']} workers",
        file=sys.stderr, flush=True,
    )


def _print_tournament_progress(row: dict[str, object], total_matches: int) -> None:
    print(
        f"[{row['match_number']}/{total_matches}] {row['agent_a']} vs {row['agent_b']}: "
        f"winner={row['winner'] or 'draw'}, plies={row['result']['plies']}, "
        f"time={row['duration_seconds']:.3f}s",
        file=sys.stderr, flush=True,
    )

def _print_extraction_summary(summary: dict[str, object]) -> None:
    inputs = summary.get("inputs", [summary["input"]])
    print(f"Inputs ({len(inputs)}):")
    for input_path in inputs:
        print(f"  {input_path}")
    print(f"Output directory: {summary['output_dir']}")
    print(
        f"Matches: {summary['processed_matches']}/{summary['declared_matches']} "
        f"({'complete' if summary['complete'] else 'partial'})"
    )
    print(f"Rows: {summary['row_counts']}")


def _print_report_summary(summary: dict[str, object]) -> None:
    print(f"Input directory: {summary['input_dir']}")
    print(f"Output directory: {summary['output_dir']}")
    print(f"Game: {summary['game']}")
    print(f"Matches: {summary['matches']}")
    print(f"Study: {'complete' if summary['complete'] else 'partial'}")
    print(f"Artifacts: {summary['figures']} figures, {summary['tables']} tables")



def _print_tournament_summary(summary: dict[str, object]) -> None:
    print(f"Game: {summary['game']}")
    print(f"Agents: {summary['agents']}")
    print(f"Pairings: {summary['pairings']}")
    print(f"Seat mode: {summary['seat_mode']}")
    print(f"Matches: {summary['matches']}")
    print(f"Workers: {summary['workers']}")
    print(f"Total time: {summary['elapsed_seconds']:.3f}s")
    print(f"Trace output: {summary['output']}")
    print()
    print("Standings (self-play excluded):")
    standings = summary["standings"]
    if not isinstance(standings, dict):
        raise RuntimeError("tournament standings have an invalid shape")
    for name, result in standings.items():
        print(
            f"  {name}: {result['wins']}W {result['losses']}L {result['draws']}D "
            f"({result['games']} games, {result['self_play_games']} self-play)"
        )



def _print_batch_setup(
    batch: Batch,
    game: TicTacToe | ConnectFour | Boop | SpiritsOfTheForest,
    name_a: str,
    agent_a: RandomAgent | MctsAgent,
    name_b: str,
    agent_b: RandomAgent | MctsAgent,
) -> None:
    print(
        f"Starting batch: {_game_name(game)}, {batch.matches} matches, "
        f"alternate sides: {'yes' if batch.alternate_sides else 'no'}, "
        f"workers: {min(resolve_workers(batch.workers), batch.matches)}",
        file=sys.stderr,
    )
    print(f"Agent A: {_batch_agent_description(name_a, agent_a)}", file=sys.stderr)
    print(f"Agent B: {_batch_agent_description(name_b, agent_b)}", file=sys.stderr)
    print(file=sys.stderr, flush=True)


def _print_batch_progress(event: BatchProgress, name_a: str, name_b: str) -> None:
    prefix = f"[{event.match_number}/{event.total_matches}]"
    player_a = event.agent_a_player
    player_b = 1 - player_a
    if event.status is BatchProgressStatus.STARTED:
        print(
            f"{prefix} starting: A ({name_a})=P{player_a}, "
            f"B ({name_b})=P{player_b}, seed={event.seed}",
            file=sys.stderr,
            flush=True,
        )
        return

    result = event.result
    if result is None:
        raise RuntimeError("completed batch progress is missing its match result")
    winner = (
        "draw"
        if result.winner is None
        else f"A ({name_a})"
        if result.winner == 0
        else f"B ({name_b})"
    )
    print(
        f"{prefix} completed: winner={winner}, plies={result.plies}, "
        f"match={result.duration_seconds:.3f}s, elapsed={event.elapsed_seconds:.3f}s",
        file=sys.stderr,
        flush=True,
    )


def _batch_agent_description(name: str, agent: RandomAgent | MctsAgent) -> str:
    if isinstance(agent, RandomAgent):
        return name
    return (
        f"{name} ({_mcts_budget_description(agent)}, "
        f"rollout_depth={agent.rollout_depth}, "
        f"exploration={agent.exploration:.6f}, "
        f"selection_policy={agent.selection_policy}, "
        f"cutoff={_evaluator_name(agent.cutoff_evaluator)}, "
        f"rollout={_rollout_policy_description(agent)}, "
        f"progressive_bias={_progressive_bias_description(agent.progressive_bias)}, "
        f"tree_reuse={'yes' if agent.tree_reuse else 'no'}, "
        f"transpositions={'yes' if agent.transpositions else 'no'})"
    )


def _progressive_bias_description(bias: ProgressiveBias | None) -> str:
    if bias is None:
        return "none"
    condition = "always" if bias.condition is None else bias.condition.phase
    return (
        f"weight={bias.weight:g}/{_evaluator_name(bias.evaluator)}/"
        f"condition={condition}"
    )


def _mcts_budget_description(agent: MctsAgent) -> str:
    if agent.iterations is not None:
        return f"iterations={agent.iterations}"
    return f"time_budget={agent.time_budget:g}s"



def _print_batch_result(
    result: BatchResult,
    game: TicTacToe | ConnectFour | Boop | SpiritsOfTheForest,
    name_a: str,
    agent_a: RandomAgent | MctsAgent,
    name_b: str,
    agent_b: RandomAgent | MctsAgent,
    *,
    output: Path | None = None,
) -> None:
    print(f"Game: {_game_name(game)}")
    print(f"Matches: {result.matches}")
    print(f"Workers: {result.workers}")
    print(f"Alternate sides: {'yes' if result.alternate_sides else 'no'}")
    print(f"Agent A: {_batch_agent_description(name_a, agent_a)}")
    print(f"Agent B: {_batch_agent_description(name_b, agent_b)}")
    if output is not None:
        print(f"Trace: {output}")
    print()
    print("Results:")
    print(
        f"  Agent A wins: {result.agent_a_wins} "
        f"({result.agent_a_wins / result.matches:.1%})"
    )
    print(
        f"  Agent B wins: {result.agent_b_wins} "
        f"({result.agent_b_wins / result.matches:.1%})"
    )
    print(f"  Draws: {result.draws} ({result.draws / result.matches:.1%})")
    print(f"  Average plies: {result.average_plies:.1f}")
    print(f"  Total time: {result.elapsed_seconds:.3f}s")



def _heuristic_name(heuristic: int | None) -> str:
    return "none" if heuristic is None else str(heuristic)


def _evaluator_name(evaluator: NeutralEvaluator | GameHeuristic | None) -> str:
    if isinstance(evaluator, GameHeuristic):
        params = ""
        if evaluator.params:
            assignments = ",".join(
                f"{name}={value:g}" for name, value in evaluator.params.items()
            )
            params = f";{assignments}"
        return f"game_heuristic({evaluator.index}{params})"
    return "neutral" if evaluator is not None else "none"


def _serialized_evaluator_name(evaluator: object) -> str:
    if evaluator is None:
        return "none"
    if not isinstance(evaluator, dict):
        raise TypeError("serialized evaluator must be an object")
    kind = evaluator.get("kind")
    if kind == "game_heuristic":
        params = evaluator.get("params")
        suffix = ""
        if isinstance(params, dict) and params:
            assignments = ",".join(
                f"{name}={value}" for name, value in params.items()
            )
            suffix = f";{assignments}"
        return f"game_heuristic({evaluator.get('index')}{suffix})"
    return str(kind)


def _serialized_rollout_policy_description(fields: dict[str, object]) -> str:
    if fields["rollout_policy"] != "conditional":
        policy = str(fields["rollout_policy"])
        if fields["rollout_epsilon"] is not None:
            policy += f"(epsilon={fields['rollout_epsilon']})"
        evaluator = _serialized_evaluator_name(fields["rollout_evaluator"])
        return policy if evaluator == "none" else f"{policy}, evaluator={evaluator}"

    condition = fields["rollout_condition"]
    if not isinstance(condition, dict):
        raise TypeError("serialized conditional rollout requires a condition")
    primary = str(fields["rollout_primary_policy"])
    if fields["rollout_epsilon"] is not None:
        primary += f"(epsilon={fields['rollout_epsilon']})"
    primary_evaluator = _serialized_evaluator_name(fields["rollout_evaluator"])
    if primary_evaluator != "none":
        primary += f", evaluator={primary_evaluator}"
    fallback = str(fields["rollout_fallback_policy"])
    fallback_epsilon = fields["rollout_fallback_epsilon"]
    if fallback_epsilon is not None:
        fallback += f"(epsilon={fallback_epsilon})"
    fallback_evaluator = _serialized_evaluator_name(
        fields["rollout_fallback_evaluator"]
    )
    if fallback_evaluator != "none":
        fallback += f", evaluator={fallback_evaluator}"
    return f"conditional({condition.get('phase')} ? {primary} : {fallback})"


def _rollout_policy_description(agent: MctsAgent) -> str:
    if isinstance(agent.rollout_policy, ConditionalRollout):
        primary = _base_rollout_policy_name(agent.rollout_policy.primary)
        evaluator = _rollout_policy_evaluator(agent.rollout_policy.primary)
        if evaluator is not None:
            primary += f"/{_evaluator_name(evaluator)}"
        epsilon = _rollout_policy_epsilon(agent.rollout_policy.primary)
        if epsilon is not None:
            primary += f"/epsilon={epsilon}"
        fallback = _base_rollout_policy_name(agent.rollout_policy.fallback)
        return (
            f"conditional/{agent.rollout_policy.condition.phase}"
            f"?{primary}:{fallback}"
        )
    policy = _rollout_policy_name(agent)
    evaluator = _rollout_policy_evaluator(agent.rollout_policy)
    if evaluator is not None:
        policy += f"/{_evaluator_name(evaluator)}"
    epsilon = _rollout_policy_epsilon(agent.rollout_policy)
    if epsilon is not None:
        policy += f"/epsilon={epsilon}"
    return policy


def _print_result(
    result: MatchResult,
    first: str,
    second: str,
    first_agent,
    second_agent,
) -> None:
    from ..splendor import SplendorAction
    from ..lost_cities import LostCitiesAction
    print(f"Player 0: {_agent_name(first, first_agent)}")
    print(f"Player 1: {_agent_name(second, second_agent)}")
    print()
    for ply, move in enumerate(result.moves, start=1):
        if isinstance(move.action, (SplendorAction, LostCitiesAction)):
            selected = str(move.action.to_dict())
        elif isinstance(move.action, TicTacToeAction):
            selected = f"row {move.action.row}, column {move.action.column}"
        elif isinstance(move.action, Connect6Action):
            selected = f"cell {move.action.position}"
        elif isinstance(move.action, ConnectFourAction):
            selected = f"column {move.action.column}"
        elif isinstance(
            move.action,
            (
                TakeSpiritTile,
                EndSpiritCollection,
                PlaceSpiritGemstone,
                MoveSpiritGemstone,
                SkipSpiritGemstone,
            ),
        ):
            from ..api import _spirits_action_description

            selected = _spirits_action_description(move.action)
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
    if result.final_board:
        print("Final board:")
        _print_board(result.final_board)
    if result.pools is not None:
        for player, pool in enumerate(result.pools):
            print(f"Player {player} pool: {pool.kittens} kittens, {pool.cats} cats")
    if result.scores is not None:
        print(f"Scores: player 0 = {result.scores[0]}, player 1 = {result.scores[1]}")
    print()
    print("Result: draw" if result.winner is None else f"Winner: player {result.winner}")
    print(f"Utilities: {list(result.utilities)}")
    print(f"Plies: {result.plies}")
    print(f"Seed: {result.seed}")


def _agent_name(name: str, agent) -> str:
    if isinstance(agent, MctsAgent):
        return (
            f"{name} (cutoff {_evaluator_name(agent.cutoff_evaluator)}, "
            f"rollout {_rollout_policy_description(agent)}, "
            f"progressive bias {_progressive_bias_description(agent.progressive_bias)})"
        )
    return name


def _print_board(board) -> None:
    print("    " + " ".join(str(column) for column in range(len(board[0]))))
    for row, cells in enumerate(board):
        print(f"{row} | " + " ".join(_piece_symbol(cell) for cell in cells))


def _print_human_move(observation: HumanMoveObservation) -> None:
    print(file=sys.stderr)
    print(f"Board after player {observation.player}'s move:", file=sys.stderr)
    print(
        "    " + " ".join(str(column) for column in range(len(observation.board[0]))),
        file=sys.stderr,
    )
    for row, cells in enumerate(observation.board):
        rendered = " ".join(_piece_symbol(cell) for cell in cells)
        print(f"{row} | {rendered}", file=sys.stderr)
    if observation.pools is not None:
        for player, pool in enumerate(observation.pools):
            print(
                f"Player {player} pool: {pool.kittens} kittens, {pool.cats} cats",
                file=sys.stderr,
            )


def _piece_symbol(piece) -> str:
    if piece is None:
        return "."
    if isinstance(piece, int):
        return "X" if piece == 0 else "O"
    if isinstance(piece, SpiritTile):
        gemstone = "" if piece.gemstone is None else str(piece.gemstone)
        return f"{piece.spirit.value[:2].upper()}{piece.spirit_symbols}{gemstone}"
    if not isinstance(piece, BoopPiece):
        raise TypeError("unknown board piece")
    if piece.player == 0:
        return "x" if piece.kind is BoopPieceKind.KITTEN else "X"
    return "o" if piece.kind is BoopPieceKind.KITTEN else "O"


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

