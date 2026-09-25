"""Spirits of the Forest trace replay projection and game-specific CSV tables."""

from __future__ import annotations

from ...game_types import (
    EndSpiritCollection, ForestPosition, MoveSpiritGemstone, PlaceSpiritGemstone,
    SkipSpiritGemstone, SpiritGemstoneSacrifice, SpiritsOfTheForest, TakeSpiritTile,
)
from ...matches.models import Move
from ...native_bridge import _analyze_trace
from ...extraction.schema import (
    _CsvWriter, _MatchContext, _integer_field, _trace_root_actions,
    _trace_tree_reuse, _tree_reuse_row, _game_quarter, _player_outcome,
    _TREE_REUSE_FIELDS,
)

_SPOTF_OUTPUT_FILES = {
    "spotf_matches": "spotf_matches.csv",
    "actions": "actions.csv",
    "player_turns": "player_turns.csv",
    "tile_takes": "tile_takes.csv",
    "gemstone_actions": "gemstone_actions.csv",
    "categories": "categories.csv",
    "root_actions": "root_actions.csv",
}

_SPOTF_TILE_COUNT = 48

_SPOTF_MATCH_FIELDS = (
    "match_number",
    "score_0",
    "score_1",
    "score_margin",
    "tiles_0",
    "tiles_1",
    "physical_turns",
    "actions_per_turn",
    "win_reason",
    "p0_gemstones_available",
    "p0_gemstones_placed",
    "p0_gemstones_removed",
    "p1_gemstones_available",
    "p1_gemstones_placed",
    "p1_gemstones_removed",
)

_SPOTF_ACTION_BASE_FIELDS = (
    "match_number",
    "ply",
    "physical_turn",
    "action_in_turn",
    "total_plies",
    "player",
    "agent",
    "outcome",
    "decision_seconds",
    "search_iterations",
    "search_nodes",
    *_TREE_REUSE_FIELDS,
    "progress_fraction",
    "game_quarter",
    "tile_progress_fraction",
    "tile_quarter",
    "phase_before",
    "phase_after",
    "action_kind",
    "legal_actions_before",
    "turn_completed_after",
    "terminal_after",
)

_SPOTF_ACTION_STATE_FIELDS = tuple(
    field
    for stage in ("before", "after")
    for field in (
        f"remaining_tiles_{stage}",
        f"completed_turns_{stage}",
        *(
            metric
            for player in (0, 1)
            for metric in (
                f"p{player}_score_{stage}",
                f"p{player}_reachable_score_{stage}",
                f"p{player}_categories_present_{stage}",
                f"p{player}_categories_reachable_{stage}",
                f"p{player}_categories_leading_{stage}",
                f"p{player}_tiles_{stage}",
                f"p{player}_gemstones_available_{stage}",
                f"p{player}_gemstones_placed_{stage}",
                f"p{player}_gemstones_removed_{stage}",
            )
        ),
    )
)

_SPOTF_ACTION_FIELDS = _SPOTF_ACTION_BASE_FIELDS + _SPOTF_ACTION_STATE_FIELDS

_ROOT_ACTION_FIELDS = (
    "match_number",
    "ply",
    "player",
    "agent",
    "phase",
    "action_index",
    "visits",
    "visit_share",
    "mean_utility",
    "heuristic_value",
    "progressive_bias",
    "selected",
)

_PLAYER_TURN_FIELDS = (
    "match_number",
    "physical_turn",
    "player",
    "agent",
    "outcome",
    "progress_fraction",
    "game_quarter",
    "tile_progress_fraction",
    "tile_quarter",
    "actions",
    "decision_seconds",
    "search_decisions",
    "search_iterations",
    "search_nodes",
    "tiles_collected",
    "symbols_collected",
    "used_end_collection",
    "gemstone_action",
    "remaining_tiles_after",
    "score_before",
    "score_after",
    "score_delta",
    "reachable_score_before",
    "reachable_score_after",
    "reachable_score_delta",
    "categories_present_after",
    "categories_reachable_after",
    "categories_leading_after",
    "gemstones_available_before",
    "gemstones_available_after",
    "gemstones_placed_before",
    "gemstones_placed_after",
    "gemstones_removed_before",
    "gemstones_removed_after",
    "gemstones_removed_delta",
    "gemstones_usable_after",
)

_TILE_TAKE_FIELDS = (
    "match_number",
    "ply",
    "physical_turn",
    "action_in_turn",
    "player",
    "agent",
    "outcome",
    "progress_fraction",
    "game_quarter",
    "tile_progress_fraction",
    "tile_quarter",
    "decision_seconds",
    "search_iterations",
    "search_nodes",
    "second_tile",
    "row",
    "column",
    "spirit",
    "spirit_symbols",
    "power_source",
    "reservation_owner",
    "reservation_agent",
    "reservation_relation",
    "sacrifice",
    "sacrifice_row",
    "sacrifice_column",
    "score_delta",
    "reachable_score_delta",
    "categories_reachable_delta",
    "categories_leading_delta",
    "gemstones_removed_delta",
)

_GEMSTONE_ACTION_FIELDS = (
    "match_number",
    "ply",
    "physical_turn",
    "player",
    "agent",
    "outcome",
    "progress_fraction",
    "game_quarter",
    "tile_progress_fraction",
    "tile_quarter",
    "decision_seconds",
    "search_iterations",
    "search_nodes",
    "action",
    "source_row",
    "source_column",
    "target_row",
    "target_column",
    "available_before",
    "placed_before",
    "removed_before",
    "available_after",
    "placed_after",
    "removed_after",
)

_CATEGORY_FIELDS = (
    "match_number",
    "player",
    "agent",
    "outcome",
    "category_type",
    "category",
    "count",
    "opponent_count",
    "count_gap",
    "points",
    "won_or_tied_majority",
    "absent_penalty",
    "lost_majority",
)

def _extract_spotf_match(
    context: _MatchContext,
    writers: dict[str, _CsvWriter],
    row_counts: dict[str, int],
    game: SpiritsOfTheForest,
) -> None:
    moves = tuple(
        _trace_spotf_move(raw, context.match_number, index)
        for index, raw in enumerate(context.raw_moves, 1)
    )
    analysis = _analyze_trace(game, moves, seed=context.seed)
    if analysis["winner"] != context.winner_player:
        raise ValueError(
            f"match {context.match_number} replay winner does not match its result"
        )
    raw_scores = context.raw_result.get("scores")
    final_scores = list(analysis["final_scores"])
    if raw_scores != final_scores:
        raise ValueError(
            f"match {context.match_number} replay scores do not match its result"
        )

    turns = analysis["turns"]
    final_state = turns[-1]["after"]
    physical_turns = turns[-1]["physical_turn"]
    final_players = final_state["players"]
    win_reason = (
        "draw"
        if context.winner_player is None
        else "score"
        if final_scores[0] != final_scores[1]
        else "fewer_tiles"
    )
    writers["spotf_matches"].writerow(
        {
            "match_number": context.match_number,
            "score_0": final_scores[0],
            "score_1": final_scores[1],
            "score_margin": abs(final_scores[0] - final_scores[1]),
            "tiles_0": final_players[0]["tiles"],
            "tiles_1": final_players[1]["tiles"],
            "physical_turns": physical_turns,
            "actions_per_turn": context.plies / physical_turns,
            "win_reason": win_reason,
            **{
                f"p{player}_gemstones_{metric}": final_players[player][
                    f"gemstones_{metric}"
                ]
                for player in (0, 1)
                for metric in ("available", "placed", "removed")
            },
        }
    )
    row_counts["spotf_matches"] += 1

    physical_turn: list[tuple[Move, dict[str, object]]] = []
    for move, turn in zip(moves, turns, strict=True):
        _write_spotf_action(
            context,
            writers,
            row_counts,
            move,
            turn,
        )
        total_root_visits = sum(root.visits for root in move.root_actions)
        for root in move.root_actions:
            writers["root_actions"].writerow(
                {
                    "match_number": context.match_number,
                    "ply": turn["ply"],
                    "player": move.player,
                    "agent": context.players[move.player],
                    "phase": turn["phase_before"],
                    "action_index": root.action_index,
                    "visits": root.visits,
                    "visit_share": (
                        root.visits / total_root_visits if total_root_visits else 0.0
                    ),
                    "mean_utility": root.mean_utility,
                    "heuristic_value": (
                        "" if root.heuristic_value is None else root.heuristic_value
                    ),
                    "progressive_bias": (
                        "" if root.progressive_bias is None else root.progressive_bias
                    ),
                    "selected": root.selected,
                }
            )
            row_counts["root_actions"] += 1
        physical_turn.append((move, turn))
        if turn["turn_completed_after"]:
            _write_spotf_player_turn(context, writers, row_counts, physical_turn)
            physical_turn = []
    if physical_turn:
        raise ValueError(f"match {context.match_number} ends inside a physical turn")

    for category in analysis["categories"]:
        counts = category["counts"]
        points = category["points"]
        for player in (0, 1):
            writers["categories"].writerow(
                {
                    "match_number": context.match_number,
                    "player": player,
                    "agent": context.players[player],
                    "outcome": _player_outcome(player, context.winner_player),
                    "category_type": category["category_type"],
                    "category": category["category"],
                    "count": counts[player],
                    "opponent_count": counts[1 - player],
                    "count_gap": counts[player] - counts[1 - player],
                    "points": points[player],
                    "won_or_tied_majority": (
                        counts[player] > 0 and counts[player] >= counts[1 - player]
                    ),
                    "absent_penalty": counts[player] == 0,
                    "lost_majority": (
                        counts[player] > 0 and counts[player] < counts[1 - player]
                    ),
                }
            )
            row_counts["categories"] += 1

def _write_spotf_action(
    context: _MatchContext,
    writers: dict[str, _CsvWriter],
    row_counts: dict[str, int],
    move: Move,
    turn: dict[str, object],
) -> None:
    if turn["player"] != move.player:
        raise ValueError(
            f"match {context.match_number} replay player differs at ply {turn['ply']}"
        )
    writers["actions"].writerow(
        {
            "match_number": context.match_number,
            "ply": turn["ply"],
            "physical_turn": turn["physical_turn"],
            "action_in_turn": turn["action_in_turn"],
            "total_plies": context.plies,
            "player": move.player,
            "agent": context.players[move.player],
            "outcome": _player_outcome(move.player, context.winner_player),
            "decision_seconds": move.decision_seconds,
            "search_iterations": move.search_iterations,
            "search_nodes": move.search_nodes,
            **_tree_reuse_row(move.tree_reuse),
            "progress_fraction": turn["ply"] / context.plies,
            "game_quarter": _game_quarter(turn["ply"], context.plies),
            "tile_progress_fraction": _tile_progress(turn["before"]),
            "tile_quarter": _tile_quarter(turn["before"]),
            "phase_before": turn["phase_before"],
            "phase_after": turn["phase_after"],
            "action_kind": _spotf_action_kind(move.action),
            "legal_actions_before": turn["legal_actions_before"],
            "turn_completed_after": turn["turn_completed_after"],
            "terminal_after": turn["terminal_after"],
            **_flatten_spotf_state("before", turn["before"]),
            **_flatten_spotf_state("after", turn["after"]),
        }
    )
    row_counts["actions"] += 1

    tile_take = turn["tile_take"]
    if tile_take is not None:
        reservation_owner = tile_take["reservation_owner"]
        sacrifice = tile_take["sacrifice"]
        before_player = turn["before"]["players"][move.player]
        after_player = turn["after"]["players"][move.player]
        writers["tile_takes"].writerow(
            {
                "match_number": context.match_number,
                "ply": turn["ply"],
                "physical_turn": turn["physical_turn"],
                "action_in_turn": turn["action_in_turn"],
                "player": move.player,
                "agent": context.players[move.player],
                "outcome": _player_outcome(move.player, context.winner_player),
                "progress_fraction": turn["ply"] / context.plies,
                "game_quarter": _game_quarter(turn["ply"], context.plies),
                "tile_progress_fraction": _tile_progress(turn["before"]),
                "tile_quarter": _tile_quarter(turn["before"]),
                "decision_seconds": move.decision_seconds,
                "search_iterations": move.search_iterations,
                "search_nodes": move.search_nodes,
                "second_tile": turn["action_in_turn"] > 1,
                "row": tile_take["row"],
                "column": tile_take["column"],
                "spirit": tile_take["spirit"],
                "spirit_symbols": tile_take["spirit_symbols"],
                "power_source": tile_take["power_source"] or "",
                "reservation_owner": (
                    "" if reservation_owner is None else reservation_owner
                ),
                "reservation_agent": (
                    "" if reservation_owner is None else context.players[reservation_owner]
                ),
                "reservation_relation": (
                    "none"
                    if reservation_owner is None
                    else "own"
                    if reservation_owner == move.player
                    else "opponent"
                ),
                "sacrifice": "none" if sacrifice is None else sacrifice["kind"],
                "sacrifice_row": (
                    "" if sacrifice is None else sacrifice.get("row", "")
                ),
                "sacrifice_column": (
                    "" if sacrifice is None else sacrifice.get("column", "")
                ),
                "score_delta": after_player["score"] - before_player["score"],
                "reachable_score_delta": (
                    after_player["reachable_score"] - before_player["reachable_score"]
                ),
                "categories_reachable_delta": (
                    after_player["categories_reachable"]
                    - before_player["categories_reachable"]
                ),
                "categories_leading_delta": (
                    after_player["categories_leading"]
                    - before_player["categories_leading"]
                ),
                "gemstones_removed_delta": (
                    after_player["gemstones_removed"]
                    - before_player["gemstones_removed"]
                ),
            }
        )
        row_counts["tile_takes"] += 1

    if isinstance(
        move.action,
        (PlaceSpiritGemstone, MoveSpiritGemstone, SkipSpiritGemstone),
    ):
        _write_gemstone_action(context, writers, row_counts, move, turn)

def _write_gemstone_action(
    context: _MatchContext,
    writers: dict[str, _CsvWriter],
    row_counts: dict[str, int],
    move: Move,
    turn: dict[str, object],
) -> None:
    action = move.action
    source = action.source if isinstance(action, MoveSpiritGemstone) else None
    target = (
        action.target
        if isinstance(action, (PlaceSpiritGemstone, MoveSpiritGemstone))
        else None
    )
    before = turn["before"]["players"][move.player]
    after = turn["after"]["players"][move.player]
    writers["gemstone_actions"].writerow(
        {
            "match_number": context.match_number,
            "ply": turn["ply"],
            "physical_turn": turn["physical_turn"],
            "player": move.player,
            "agent": context.players[move.player],
            "outcome": _player_outcome(move.player, context.winner_player),
            "progress_fraction": turn["ply"] / context.plies,
            "game_quarter": _game_quarter(turn["ply"], context.plies),
            "tile_progress_fraction": _tile_progress(turn["before"]),
            "tile_quarter": _tile_quarter(turn["before"]),
            "decision_seconds": move.decision_seconds,
            "search_iterations": move.search_iterations,
            "search_nodes": move.search_nodes,
            "action": _spotf_action_kind(action),
            "source_row": "" if source is None else source.row,
            "source_column": "" if source is None else source.column,
            "target_row": "" if target is None else target.row,
            "target_column": "" if target is None else target.column,
            **{
                f"{metric}_{stage}": state[f"gemstones_{metric}"]
                for stage, state in (("before", before), ("after", after))
                for metric in ("available", "placed", "removed")
            },
        }
    )
    row_counts["gemstone_actions"] += 1

def _write_spotf_player_turn(
    context: _MatchContext,
    writers: dict[str, _CsvWriter],
    row_counts: dict[str, int],
    actions: list[tuple[Move, dict[str, object]]],
) -> None:
    player = actions[0][0].player
    takes = [turn["tile_take"] for _, turn in actions if turn["tile_take"] is not None]
    gemstone_actions = [
        _spotf_action_kind(move.action)
        for move, _ in actions
        if isinstance(
            move.action,
            (PlaceSpiritGemstone, MoveSpiritGemstone, SkipSpiritGemstone),
        )
    ]
    final_turn = actions[-1][1]
    first_turn = actions[0][1]
    before = first_turn["before"]["players"][player]
    after = final_turn["after"]["players"][player]
    decision_seconds = [
        move.decision_seconds for move, _ in actions if move.decision_seconds is not None
    ]
    search_iterations = [
        move.search_iterations for move, _ in actions if move.search_iterations is not None
    ]
    search_nodes = [
        move.search_nodes for move, _ in actions if move.search_nodes is not None
    ]
    writers["player_turns"].writerow(
        {
            "match_number": context.match_number,
            "physical_turn": final_turn["physical_turn"],
            "player": player,
            "agent": context.players[player],
            "outcome": _player_outcome(player, context.winner_player),
            "progress_fraction": final_turn["ply"] / context.plies,
            "game_quarter": _game_quarter(final_turn["ply"], context.plies),
            "tile_progress_fraction": _tile_progress(final_turn["after"]),
            "tile_quarter": _tile_quarter(final_turn["after"]),
            "actions": len(actions),
            "decision_seconds": sum(decision_seconds),
            "search_decisions": len(search_iterations),
            "search_iterations": sum(search_iterations) if search_iterations else "",
            "search_nodes": sum(search_nodes) if search_nodes else "",
            "tiles_collected": len(takes),
            "symbols_collected": sum(take["spirit_symbols"] for take in takes),
            "used_end_collection": any(
                isinstance(move.action, EndSpiritCollection) for move, _ in actions
            ),
            "gemstone_action": gemstone_actions[-1] if gemstone_actions else "terminal",
            "remaining_tiles_after": final_turn["after"]["remaining_tiles"],
            "score_before": before["score"],
            "score_after": after["score"],
            "score_delta": after["score"] - before["score"],
            "reachable_score_before": before["reachable_score"],
            "reachable_score_after": after["reachable_score"],
            "reachable_score_delta": (
                after["reachable_score"] - before["reachable_score"]
            ),
            "categories_present_after": after["categories_present"],
            "categories_reachable_after": after["categories_reachable"],
            "categories_leading_after": after["categories_leading"],
            "gemstones_available_before": before["gemstones_available"],
            "gemstones_available_after": after["gemstones_available"],
            "gemstones_placed_before": before["gemstones_placed"],
            "gemstones_placed_after": after["gemstones_placed"],
            "gemstones_removed_before": before["gemstones_removed"],
            "gemstones_removed_after": after["gemstones_removed"],
            "gemstones_removed_delta": (
                after["gemstones_removed"] - before["gemstones_removed"]
            ),
            "gemstones_usable_after": (
                after["gemstones_available"] + after["gemstones_placed"]
            ),
        }
    )
    row_counts["player_turns"] += 1

def _flatten_spotf_state(stage: str, state: dict[str, object]) -> dict[str, object]:
    result = {
        f"remaining_tiles_{stage}": state["remaining_tiles"],
        f"completed_turns_{stage}": state["completed_turns"],
    }
    for player, metrics in enumerate(state["players"]):
        for metric in (
            "score",
            "reachable_score",
            "categories_present",
            "categories_reachable",
            "categories_leading",
            "tiles",
            "gemstones_available",
            "gemstones_placed",
            "gemstones_removed",
        ):
            result[f"p{player}_{metric}_{stage}"] = metrics[metric]
    return result

def _tile_progress(state: dict[str, object]) -> float:
    return (_SPOTF_TILE_COUNT - int(state["remaining_tiles"])) / _SPOTF_TILE_COUNT

def _tile_quarter(state: dict[str, object]) -> str:
    collected = _SPOTF_TILE_COUNT - int(state["remaining_tiles"])
    return f"q{min(3, collected * 4 // _SPOTF_TILE_COUNT) + 1}"

def _spotf_action_kind(action: object) -> str:
    if isinstance(action, TakeSpiritTile):
        return "take_tile"
    if isinstance(action, EndSpiritCollection):
        return "end_collection"
    if isinstance(action, PlaceSpiritGemstone):
        return "place_gemstone"
    if isinstance(action, MoveSpiritGemstone):
        return "move_gemstone"
    if isinstance(action, SkipSpiritGemstone):
        return "skip_gemstone"
    raise TypeError("expected a spotf action")

def _trace_spotf_move(raw: object, match_number: int, expected_ply: int) -> Move:
    if not isinstance(raw, dict):
        raise TypeError(f"match {match_number} ply {expected_ply} must be an object")
    ply = _integer_field(raw, "ply", f"match {match_number} move")
    if ply != expected_ply:
        raise ValueError(f"match {match_number} has unexpected ply {ply}, expected {expected_ply}")
    player = _integer_field(raw, "player", f"match {match_number} ply {ply}")
    action = raw.get("action")
    if not isinstance(action, dict) or action.get("type") not in {
        "spotf",
        "spirits_of_the_forest",
    }:
        raise TypeError(f"match {match_number} ply {ply} must contain a spotf action")
    kind = action.get("kind")
    if kind == "take_tile":
        sacrifice = action.get("sacrifice")
        parsed_sacrifice = None
        if sacrifice is not None:
            if not isinstance(sacrifice, dict):
                raise TypeError(f"match {match_number} ply {ply} sacrifice must be an object")
            sacrifice_kind = sacrifice.get("kind")
            if sacrifice_kind == "available":
                parsed_sacrifice = SpiritGemstoneSacrifice()
            elif sacrifice_kind == "forest":
                parsed_sacrifice = SpiritGemstoneSacrifice(
                    ForestPosition(
                        _integer_field(sacrifice, "row", "forest sacrifice"),
                        _integer_field(sacrifice, "column", "forest sacrifice"),
                    )
                )
            else:
                raise ValueError(f"match {match_number} ply {ply} has unknown sacrifice")
        parsed_action = TakeSpiritTile(
            ForestPosition(
                _integer_field(action, "row", "take tile action"),
                _integer_field(action, "column", "take tile action"),
            ),
            parsed_sacrifice,
        )
    elif kind == "end_collection":
        parsed_action = EndSpiritCollection()
    elif kind == "place_gemstone":
        parsed_action = PlaceSpiritGemstone(
            ForestPosition(
                _integer_field(action, "row", "place gemstone action"),
                _integer_field(action, "column", "place gemstone action"),
            )
        )
    elif kind == "move_gemstone":
        parsed_action = MoveSpiritGemstone(
            ForestPosition(
                _integer_field(action, "source_row", "move gemstone action"),
                _integer_field(action, "source_column", "move gemstone action"),
            ),
            ForestPosition(
                _integer_field(action, "target_row", "move gemstone action"),
                _integer_field(action, "target_column", "move gemstone action"),
            ),
        )
    elif kind == "skip_gemstone":
        parsed_action = SkipSpiritGemstone()
    else:
        raise ValueError(f"match {match_number} ply {ply} has unknown spotf action")
    return Move(
        player=player,
        action=parsed_action,
        decision_seconds=raw.get("decision_seconds", ""),
        search_iterations=raw.get("search_iterations", ""),
        search_nodes=raw.get("search_nodes", ""),
        root_actions=_trace_root_actions(
            raw.get("root_actions"), f"match {match_number} ply {ply}"
        ),
        tree_reuse=_trace_tree_reuse(
            raw.get("tree_reuse"), f"match {match_number} ply {ply}"
        ),
    )
