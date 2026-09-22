"""Stable JSON serialization shared by CLI and graphical match traces."""

from __future__ import annotations

import json
from dataclasses import asdict
from .connect6 import Connect6, Connect6Action
from .game_config import game_parameters
from .splendor import Splendor, SplendorAction
from .lost_cities import LostCities, LostCitiesAction

from .api import (
    Boop, ConnectFour, SpiritsOfTheForest, TicTacToe,
    RandomAgent, SoIsmctsAgent, MctsAgent, ProgressiveBias, ConditionalRollout,
    UniformRandom, Greedy, EpsilonGreedy, Mast, NeutralEvaluator, GameHeuristic,
    BoopAction,
    BoopGraduateLine,
    BoopRecoverPiece,
    ConnectFourAction,
    EndSpiritCollection,
    GameAction,
    MatchResult,
    MoveSpiritGemstone,
    PlaceSpiritGemstone,
    SkipSpiritGemstone,
    TakeSpiritTile,
    TicTacToeAction,
)


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
                "exploration": agent.exploration, "selection_policy": agent.selection_policy, "rollout_policy": "uniform",
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
