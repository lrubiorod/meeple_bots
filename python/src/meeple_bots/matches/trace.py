"""Stable JSON serialization shared by CLI and graphical match traces."""

from __future__ import annotations

import json
from dataclasses import asdict
from ..connect6 import Connect6, Connect6Action
from ..game_config import game_parameters
from ..splendor import Splendor, SplendorAction
from ..lost_cities import LostCities, LostCitiesAction

from .._agent_config import (
    RandomAgent, SoIsmctsAgent, MctsAgent, ProgressiveBias, ConditionalRollout,
    UniformRandom, Greedy, EpsilonGreedy, Mast, NeutralEvaluator, GameHeuristic,
)
from ..game_types import (
    Boop, ConnectFour, SpiritsOfTheForest, TicTacToe, BoopAction,
    BoopGraduateLine, BoopRecoverPiece, ConnectFourAction, EndSpiritCollection,
    GameAction, MoveSpiritGemstone, PlaceSpiritGemstone, SkipSpiritGemstone,
    TakeSpiritTile, TicTacToeAction,
)
from .models import MatchResult


def match_result_dict(result: MatchResult) -> dict[str, object]:
    """Serialize one match result using the version-1 trace representation."""

    payload = {
        "unassigned_maintenance_seconds": list(result.unassigned_maintenance_seconds),
        "seed": result.seed,
        "plies": result.plies,
        "utilities": list(result.utilities),
        "winner": result.winner,
        "moves": [
            {
                "ply": ply,
                "player": move.player,
                "action": action_dict(move.action),
                "decision_seconds": move.decision_seconds,
                "selection_seconds": move.selection_seconds,
                "maintenance_seconds": move.maintenance_seconds,
                "search_iterations": move.search_iterations,
                "search_nodes": move.search_nodes,
                "terminal_simulations": move.terminal_simulations,
                "cutoff_simulations": move.cutoff_simulations,
                "root_actions": [
                    {
                        "action_index": root.action_index,
                        "visits": root.visits,
                        "mean_utility": root.mean_utility,
                        "heuristic_value": root.heuristic_value,
                        "progressive_bias": root.progressive_bias,
                        "selected": root.selected,
                    }
                    for root in move.root_actions
                ],
                "tree_reuse": (
                    None
                    if move.tree_reuse is None
                    else {
                        "transition_attempts": move.tree_reuse.transition_attempts,
                        "transition_hits": move.tree_reuse.transition_hits,
                        "transition_misses": move.tree_reuse.transition_misses,
                        "own_action_hits": move.tree_reuse.own_action_hits,
                        "opponent_action_hits": move.tree_reuse.opponent_action_hits,
                        "reused_root_visits": move.tree_reuse.reused_root_visits,
                        "reused_nodes": move.tree_reuse.reused_nodes,
                        "pruned_nodes": move.tree_reuse.pruned_nodes,
                        "resets": move.tree_reuse.resets,
                    }
                ),
            }
            for ply, move in enumerate(result.moves, start=1)
        ],
    }
    if result.scores is not None:
        payload["scores"] = list(result.scores)
        payload["collections"] = [
            {
                "spirit_symbols": list(collection.spirit_symbols),
                "power_sources": list(collection.power_sources),
                "tiles": collection.tiles,
            }
            for collection in result.spirit_collections or ()
        ]
        payload["gemstone_pools"] = [
            {
                "available": pool.available,
                "placed": pool.placed,
                "removed": pool.removed,
            }
            for pool in result.gemstone_pools or ()
        ]
    if result.game_params:
        payload["game_params"] = result.game_params
    if result.splendor_state is not None:
        state = asdict(result.splendor_state)
        state.pop('_position', None)
        payload['splendor_state'] = state
        payload['chance_events'] = [{'after_ply': e.after_ply, 'outcome': e.outcome.to_dict()} for e in result.chance_events]
    if result.lost_cities_state is not None:
        payload['lost_cities_state'] = result.lost_cities_state.to_dict()
        payload['chance_events'] = [{'after_ply': e.after_ply, 'outcome': e.outcome.to_dict()} for e in result.chance_events]
    return payload


def action_dict(action: GameAction) -> dict[str, object]:
    """Serialize one supported game action without game-specific callers."""
    if isinstance(action, Connect6Action):
        return {"type": "connect6", "position": action.position}

    if isinstance(action, (SplendorAction, LostCitiesAction)):
        return action.to_dict()
    if isinstance(action, TicTacToeAction):
        return {
            "type": "tic_tac_toe",
            "row": action.row,
            "column": action.column,
        }
    if isinstance(action, ConnectFourAction):
        return {"type": "connect_four", "column": action.column}
    if isinstance(action, TakeSpiritTile):
        sacrifice = None
        if action.sacrifice is not None:
            sacrifice = {
                "kind": "available" if action.sacrifice.source is None else "forest"
            }
            if action.sacrifice.source is not None:
                sacrifice.update(
                    row=action.sacrifice.source.row,
                    column=action.sacrifice.source.column,
                )
        return {
            "type": "spotf",
            "kind": "take_tile",
            "row": action.position.row,
            "column": action.position.column,
            "sacrifice": sacrifice,
        }
    if isinstance(action, EndSpiritCollection):
        return {"type": "spotf", "kind": "end_collection"}
    if isinstance(action, PlaceSpiritGemstone):
        return {
            "type": "spotf",
            "kind": "place_gemstone",
            "row": action.target.row,
            "column": action.target.column,
        }
    if isinstance(action, MoveSpiritGemstone):
        return {
            "type": "spotf",
            "kind": "move_gemstone",
            "source_row": action.source.row,
            "source_column": action.source.column,
            "target_row": action.target.row,
            "target_column": action.target.column,
        }
    if isinstance(action, SkipSpiritGemstone):
        return {"type": "spotf", "kind": "skip_gemstone"}
    if not isinstance(action, BoopAction):
        raise TypeError(f"unsupported action type: {type(action).__name__}")
    return {
        "type": "boop",
        "piece": action.piece.value,
        "row": action.row,
        "column": action.column,
        "resolution": _boop_resolution_dict(action),
    }


def _boop_resolution_dict(action: BoopAction) -> dict[str, object]:
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


def agent_dict(name: str, agent: RandomAgent | SoIsmctsAgent | MctsAgent) -> dict[str, object]:
    if isinstance(agent, RandomAgent):
        return {"name": name, "type": "random"}
    if isinstance(agent, SoIsmctsAgent):
        return {"name": name, "type": "so_ismcts", "iterations": agent.iterations, "time_budget": agent.time_budget,
                "exploration": agent.exploration, "selection_policy": agent.selection_policy, "tree_reuse": agent.tree_reuse, "rollout_policy": "uniform",
                "root_selection": "most_visited"}
    return {
        "name": name,
        "type": "mcts",
        "iterations": agent.iterations,
        "time_budget": agent.time_budget,
        "rollout_depth": agent.rollout_depth,
        "exploration": agent.exploration,
        "selection_policy": agent.selection_policy,
        **({"rave_equivalence": agent.rave_equivalence} if agent.selection_policy == "uct_rave" else {}),
        **({"progressive_widening": agent.progressive_widening, "progressive_widening_k": agent.progressive_widening_k, "progressive_widening_alpha": agent.progressive_widening_alpha, "progressive_widening_expansion": agent.progressive_widening_expansion} if getattr(agent, "progressive_widening", False) else {}),
        "heuristic": agent.heuristic,
        "cutoff_evaluator": _evaluator_dict(agent.cutoff_evaluator),
        "rollout_policy": _rollout_policy_name(agent),
        "rollout_evaluator": _evaluator_dict(
            _rollout_policy_evaluator(agent.rollout_policy)
        ),
        "rollout_epsilon": _rollout_policy_epsilon(agent.rollout_policy),
        **_conditional_rollout_fields(agent.rollout_policy),
        **_progressive_bias_fields(agent.progressive_bias),
        "root_diagnostics": agent.root_diagnostics,
        "tree_reuse": agent.tree_reuse,
        "transpositions": agent.transpositions,
    }


def _progressive_bias_fields(bias: ProgressiveBias | None) -> dict[str, object]:
    if bias is None:
        return {
            "progressive_bias_weight": None,
            "progressive_bias_evaluator": None,
            "progressive_bias_heuristic": None,
            "progressive_bias_condition": None,
            "progressive_bias_condition_phase": None,
        }
    return {
        "progressive_bias_weight": bias.weight,
        "progressive_bias_evaluator": _evaluator_dict(bias.evaluator),
        "progressive_bias_heuristic": _evaluator_heuristic_index(bias.evaluator),
        "progressive_bias_condition": (
            "turn_phase" if bias.condition is not None else None
        ),
        "progressive_bias_condition_phase": (
            bias.condition.phase if bias.condition is not None else None
        ),
    }


def _rollout_policy_name(agent: MctsAgent) -> str:
    if isinstance(agent.rollout_policy, ConditionalRollout):
        return "conditional"
    return _base_rollout_policy_name(agent.rollout_policy)


def _base_rollout_policy_name(
    policy: UniformRandom | Greedy | EpsilonGreedy | Mast,
) -> str:
    if isinstance(policy, Mast):
        return "mast"
    if isinstance(policy, EpsilonGreedy):
        return "epsilon_greedy"
    if isinstance(policy, Greedy):
        return "greedy"
    return "uniform_random"


def _rollout_policy_evaluator(
    policy: UniformRandom | Greedy | EpsilonGreedy | Mast | ConditionalRollout,
) -> NeutralEvaluator | GameHeuristic | None:
    if isinstance(policy, ConditionalRollout):
        return _rollout_policy_evaluator(policy.primary)
    return None if isinstance(policy, (UniformRandom, Mast)) else policy.evaluator


def _rollout_policy_epsilon(
    policy: UniformRandom | Greedy | EpsilonGreedy | Mast | ConditionalRollout,
) -> float | None:
    if isinstance(policy, ConditionalRollout):
        return _rollout_policy_epsilon(policy.primary)
    return policy.epsilon if isinstance(policy, (EpsilonGreedy, Mast)) else None


def _conditional_rollout_fields(
    policy: UniformRandom | Greedy | EpsilonGreedy | Mast | ConditionalRollout,
) -> dict[str, object]:
    if not isinstance(policy, ConditionalRollout):
        return {
            "rollout_condition": None,
            "rollout_primary_policy": None,
            "rollout_fallback_policy": None,
            "rollout_fallback_evaluator": None,
            "rollout_fallback_epsilon": None,
        }
    return {
        "rollout_condition": {
            "kind": "turn_phase",
            "phase": policy.condition.phase,
        },
        "rollout_primary_policy": _base_rollout_policy_name(policy.primary),
        "rollout_fallback_policy": _base_rollout_policy_name(policy.fallback),
        "rollout_fallback_evaluator": _evaluator_dict(
            _rollout_policy_evaluator(policy.fallback)
        ),
        "rollout_fallback_epsilon": _rollout_policy_epsilon(policy.fallback),
    }


def _evaluator_heuristic_index(
    evaluator: NeutralEvaluator | GameHeuristic | None,
) -> int | None:
    return evaluator.index if isinstance(evaluator, GameHeuristic) else None


def _evaluator_dict(
    evaluator: NeutralEvaluator | GameHeuristic | None,
) -> dict[str, object] | None:
    if evaluator is None:
        return None
    if isinstance(evaluator, GameHeuristic):
        serialized: dict[str, object] = {
            "kind": "game_heuristic",
            "index": evaluator.index,
        }
        if evaluator.params:
            serialized["params"] = dict(evaluator.params)
        return serialized
    return {"kind": "neutral"}


def game_name(game: TicTacToe | ConnectFour | Boop | SpiritsOfTheForest) -> str:
    if isinstance(game, Connect6):
        return "connect6"
    if isinstance(game, LostCities):
        return "lost_cities"
    if isinstance(game, Splendor):
        return "splendor"
    if isinstance(game, TicTacToe):
        return "tic-tac-toe"
    if isinstance(game, ConnectFour):
        return "connect-four"
    if isinstance(game, SpiritsOfTheForest):
        return "spotf"
    return "boop"


def trace_match_dict(
    *,
    result: MatchResult,
    match_number: int,
    pairing_number: int,
    pairing_match_number: int,
    agent_a: str,
    agent_b: str,
    self_play: bool,
    agent_a_player: int,
    winner: str | None,
    duration_seconds: float,
) -> dict[str, object]:
    players = [agent_a, agent_b] if agent_a_player == 0 else [agent_b, agent_a]
    return {
        "record_type": "match",
        "match_number": match_number,
        "pairing_number": pairing_number,
        "pairing_match_number": pairing_match_number,
        "agent_a": agent_a,
        "agent_b": agent_b,
        "self_play": self_play,
        "agent_a_player": agent_a_player,
        "players": players,
        "winner": winner,
        "duration_seconds": duration_seconds,
        "result": match_result_dict(result),
    }


def write_jsonl(output, value: dict[str, object]) -> None:
    output.write(json.dumps(value, separators=(",", ":")) + "\n")
    output.flush()


# Tournament trace validation is independent of tournament job scheduling.
import math
from pathlib import Path
from ..game_config import create_game

def match_record(job, outcome) -> dict[str, object]:
    """Serialize an outcome, translating physical seats into tournament roles."""
    winner = outcome.result.winner
    return trace_match_dict(
        result=outcome.result, match_number=job.match_number,
        pairing_number=job.pairing_number, pairing_match_number=job.pairing_match_number,
        agent_a=job.agent_a.name, agent_b=job.agent_b.name,
        self_play=job.self_play, agent_a_player=job.agent_a_player,
        winner=None if winner is None else "agent_a" if winner == job.agent_a_player else "agent_b",
        duration_seconds=outcome.duration_seconds,
    )


def _validate_splendor_result(result: dict) -> None:
    """Validate a public-chance result through Rust, independent of job scheduling."""
    def integer(value, label, minimum=0, maximum=2**64-1):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"Invalid {label}")
        return value
    seed = integer(result.get("seed"), "result seed")
    moves = result.get("moves")
    utilities = result.get("utilities")
    from types import SimpleNamespace
    from ..splendor import SplendorAction, SplendorState, SplendorChanceOutcome, ChanceEvent, replay_splendor
    if not isinstance(result.get("chance_events"), list) or not isinstance(result.get("splendor_state"), dict):
        raise ValueError("Splendor trace requires chance events and final state")
    try:
        parsed_moves = tuple(SimpleNamespace(player=m['player'], action=SplendorAction.from_dict(m['action'])) for m in moves)
        events = tuple(ChanceEvent(integer(e['after_ply'], 'chance after_ply', 1), SplendorChanceOutcome(integer(e['outcome']['card'], 'refill card', 0, 89))) for e in result['chance_events'])
        if any(e['outcome'].get('kind') != 'refill' or e['outcome'].get('type') != 'splendor' for e in result['chance_events']):
            raise ValueError("invalid Splendor chance outcome")
        state = replay_splendor(seed, parsed_moves, events)
        if state != SplendorState.from_dict(result['splendor_state']):
            raise ValueError("Splendor replay final state differs")
        if result.get('scores') != [player.prestige for player in state.players]:
            raise ValueError("Splendor replay scores differ")
        actual_utilities = state._native_position().utilities()
        winner = next((i for i, value in enumerate(actual_utilities) if value > 0), None)
        if result.get('winner') != winner:
            raise ValueError("Splendor replay winner differs")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in utilities):
            raise ValueError("Splendor utilities must be numeric")
        if tuple(actual_utilities) != tuple(utilities):
            raise ValueError("Splendor replay utilities differ")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("invalid Splendor trace") from error


def _validate_completed_record(record: dict, header: dict) -> None:
    """Validate persisted job identity and required result data without running games."""
    def integer(value, label, minimum=0, maximum=2**64 - 1):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"Invalid {label}")
        return value

    def number(value, label, minimum=0):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < minimum):
            raise ValueError(f"Invalid {label}")
        return value

    pairs = header.get("pairings")
    if pairs is None:
        if header.get("pairing_mode", "round_robin") != "round_robin":
            raise ValueError("Cannot verify legacy adjacent trace without an explicit pairing plan")
        agents = header["agents"]
        pairs = [[a["name"], b["name"]] for i, a in enumerate(agents) for b in agents[i+1:]]
        pairs.extend([a["name"], a["name"]] for a in agents if a.get("self_play", False))
    per_pair = integer(header.get("matches_per_pair"), "matches_per_pair", 1)
    pairing = integer(record.get("pairing_number"), "pairing_number", 1, len(pairs))
    within = integer(record.get("pairing_match_number"), "pairing_match_number", 1, per_pair)
    if record.get("match_number") != (pairing - 1) * per_pair + within:
        raise ValueError("Match number does not match its pairing")
    a, b = pairs[pairing - 1]
    seat = (within - 1) % 2
    players = [a, b] if seat == 0 else [b, a]
    if (record.get("agent_a") != a or record.get("agent_b") != b
            or type(record.get("self_play")) is not bool or record["self_play"] != (a == b)
            or type(record.get("agent_a_player")) is not int or record["agent_a_player"] != seat
            or record.get("players") != players):
        raise ValueError("Recorded agents or seats differ from the pairing plan")
    paired = header.get("seat_mode") == "paired"
    offset = sum(per_pair // 2 if paired and x != y else per_pair for x, y in pairs[:pairing-1])
    offset += (within - 1) // 2 if paired and a != b else within - 1
    seed = (header["seed"] + offset) & (2**64 - 1)
    result = record.get("result")
    if not isinstance(result, dict):
        raise ValueError("Missing match result")
    if integer(result.get("seed"), "result seed") != seed:
        raise ValueError("Recorded seed differs from the pairing plan")
    plies = integer(result.get("plies"), "plies", 1, header["max_plies"])
    moves = result.get("moves")
    if not isinstance(moves, list) or len(moves) != plies:
        raise ValueError("Move count differs from result plies")
    if "winner" not in result:
        raise ValueError("Missing result winner")
    winner = result["winner"]
    if winner is not None:
        integer(winner, "winner", 0, 1)
    expected_role = None if winner is None else "agent_a" if winner == seat else "agent_b"
    if "winner" not in record or record["winner"] != expected_role:
        raise ValueError("Winner role differs from result winner")
    utilities = result.get("utilities")
    if not isinstance(utilities, list) or len(utilities) != 2:
        raise ValueError("Utilities must contain two values")
    for utility in utilities:
        number(utility, "utility", -1)
        if utility > 1:
            raise ValueError("Utility is outside [-1, 1]")
    utility_winner = None if utilities[0] == 0 else 0 if utilities[0] > 0 else 1
    if utilities[0] != -utilities[1] or utility_winner != winner:
        raise ValueError("Utilities differ from winner")
    number(record.get("duration_seconds"), "duration_seconds")
    if header["game"] == "connect6":
        _validate_connect6_result(result, header.get("game_params"))
    if header["game"] == "lost_cities":
        from .. import _native
        from ..lost_cities import LostCitiesState
        try:
            position = _native.LostCitiesPosition.replay(result['moves'], result['chance_events'])
            state = LostCitiesState.from_dict(position.snapshot())
            if state != LostCitiesState.from_dict(result['lost_cities_state']):
                raise ValueError('Lost Cities final state differs from replay')
            if tuple(result['scores']) != state.scores:
                raise ValueError('Lost Cities scores differ from replay')
            expected = None if state.scores[0] == state.scores[1] else int(state.scores[1] > state.scores[0])
            if result['winner'] != expected:
                raise ValueError('Lost Cities winner differs from replay')
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('invalid Lost Cities trace') from error
    if header["game"] == "splendor":
        _validate_splendor_result(result)
    action_type = {
        "tic-tac-toe": "tic_tac_toe", "connect-four": "connect_four",
        "boop": "boop", "spotf": "spotf", "splendor": "splendor", "connect6": "connect6", "lost_cities": "lost_cities",
    }[header["game"]]
    if action_type == "spotf":
        for field in ("scores", "collections", "gemstone_pools"):
            if not isinstance(result.get(field), list) or len(result[field]) != 2:
                raise ValueError(f"Missing or invalid spotf {field}")
    for ply, move in enumerate(moves, 1):
        if not isinstance(move, dict) or integer(move.get("ply"), "move ply", 1) != ply:
            raise ValueError("Invalid move sequence")
        integer(move.get("player"), "move player", 0, 1)
        action = move.get("action")
        if not isinstance(action, dict) or action.get("type") != action_type:
            raise ValueError("Invalid move action")
        if action_type in {"tic_tac_toe", "connect_four", "boop"}:
            integer(action.get("column"), "action column", 0, {"tic_tac_toe": 2, "connect_four": 6, "boop": 5}[action_type])
            if action_type != "connect_four":
                integer(action.get("row"), "action row", 0, 2 if action_type == "tic_tac_toe" else 5)
        if action_type == "boop":
            resolution = action.get("resolution")
            if action.get("piece") not in {"kitten", "cat"} or not isinstance(resolution, dict):
                raise ValueError("Invalid boop action")
            kind = resolution.get("type")
            if kind not in {"none", "recover", "graduate"}:
                raise ValueError("Invalid boop resolution")
            positions = resolution.get("positions") if kind == "graduate" else [resolution] if kind == "recover" else []
            if not isinstance(positions, list) or (kind == "graduate" and len(positions) != 3):
                raise ValueError("Invalid graduation positions")
            for position in positions:
                if not isinstance(position, dict):
                    raise ValueError("Invalid resolution position")
                integer(position.get("row"), "resolution row")
                integer(position.get("column"), "resolution column")
        if action_type == "spotf":
            fields = {
                "take_tile": ("row", "column"), "end_collection": (),
                "place_gemstone": ("row", "column"),
                "move_gemstone": ("source_row", "source_column", "target_row", "target_column"),
                "skip_gemstone": (),
            }.get(action.get("kind"))
            if fields is None:
                raise ValueError("Invalid spotf action")
            for field in fields:
                integer(action.get(field), field)
            sacrifice = action.get("sacrifice")
            if sacrifice is not None:
                if not isinstance(sacrifice, dict) or sacrifice.get("kind") not in {"available", "forest"}:
                    raise ValueError("Invalid sacrifice")
                if sacrifice["kind"] == "forest":
                    integer(sacrifice.get("row"), "sacrifice row")
                    integer(sacrifice.get("column"), "sacrifice column")
        number(move.get("decision_seconds"), "decision_seconds")
        for field in ("search_iterations", "search_nodes", "terminal_simulations", "cutoff_simulations"):
            if move.get(field) is not None:
                integer(move[field], field)
        if header.get("decision_timing_scope") == "agent_total_v1":
            selection = number(move.get("selection_seconds"), "selection_seconds")
            maintenance = number(move.get("maintenance_seconds"), "maintenance_seconds")
            if not math.isclose(move["decision_seconds"], selection + maintenance, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("Decision time differs from component times")


class TournamentTrace:
    """Write and flush standard traces, optionally continuing a matching local plan.

    Resumption rejects conflicting headers, invalid match numbers, duplicate
    records, incomplete results, incorrect job identities and truncated lines.
    The same structural validator runs before writes; game legality is checked
    separately by extraction. It never silently discards recorded work.
    The caller skips completed jobs while retaining the original job identities.
    """

    def __init__(self, path: Path, header: dict[str, object], *, overwrite=False, resume=False):
        if overwrite and resume:
            raise ValueError("overwrite and resume are mutually exclusive")
        self.path = path
        self.header = header
        self.overwrite = overwrite
        self.resume = resume
        self.completed_match_numbers: set[int] = set()
        self._output = None

    def _check_number(self, record: dict) -> int:
        number = record.get("match_number")
        if type(number) is not int or not 1 <= number <= self.header["total_matches"]:
            raise ValueError(f"Invalid match number in {self.path}")
        if number in self.completed_match_numbers:
            raise ValueError(f"Duplicate match number in {self.path}: {number}")
        return number

    def __enter__(self):
        continuing = self.resume and self.path.exists()
        if continuing:
            with self.path.open(encoding="utf-8") as source:
                first_line = source.readline()
                if not first_line.endswith("\n"):
                    raise ValueError(f"Truncated trace header: {self.path}")
                if json.loads(first_line) != self.header:
                    raise ValueError(f"Trace header differs: {self.path}")
                for line in source:
                    if not line.endswith("\n"):
                        raise ValueError(f"Truncated trace line: {self.path}")
                    record = json.loads(line)
                    if not isinstance(record, dict) or record.get("record_type") != "match":
                        raise ValueError(f"Invalid match record in {self.path}")
                    number = self._check_number(record)
                    try:
                        _validate_completed_record(record, self.header)
                    except (ValueError, KeyError, TypeError) as error:
                        raise ValueError(f"Invalid match {number} in {self.path}: {error}") from error
                    self.completed_match_numbers.add(number)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._output = self.path.open(
            "a" if continuing else "w" if self.overwrite else "x", encoding="utf-8"
        )
        if not continuing:
            write_jsonl(self._output, self.header)
        return self

    def write(self, job, outcome) -> dict[str, object]:
        if self._output is None:
            raise RuntimeError("use TournamentTrace inside a with block")
        record = match_record(job, outcome)
        number = self._check_number(record)
        _validate_completed_record(record, self.header)
        write_jsonl(self._output, record)
        self.completed_match_numbers.add(number)
        return record

    def __exit__(self, *args):
        if self._output is not None:
            self._output.close()
            self._output = None


def _validate_connect6_result(result, parameters):
    from ..connect6 import Connect6Action
    game = create_game('connect6', parameters)
    if result.get('game_params') != game_parameters(game):
        raise ValueError('Connect6 result game_params differ from tournament configuration')
    state = game.initial_state()
    for move in result['moves']:
        if state.current_player != move['player'] or move['action'].get('type') != 'connect6':
            raise ValueError('Connect6 trace has invalid player or action type')
        state = state.apply_action(Connect6Action(move['action']['position']))
    if not state.terminal or state.winner != result['winner']:
        raise ValueError('Connect6 replay result differs from trace')
    if 'final_board' in result and [list(row) for row in state.board] != result['final_board']:
        raise ValueError('Connect6 replay board differs from trace')
