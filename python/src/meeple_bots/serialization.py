"""Stable JSON serialization shared by CLI and graphical match traces."""

from __future__ import annotations

from .api import (
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
                "search_iterations": move.search_iterations,
                "search_nodes": move.search_nodes,
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
    return payload


def action_dict(action: GameAction) -> dict[str, object]:
    """Serialize one supported game action without game-specific callers."""

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
