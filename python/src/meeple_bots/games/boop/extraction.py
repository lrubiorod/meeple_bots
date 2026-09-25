"""Boop trace replay projection and game-specific CSV tables."""

from __future__ import annotations

from ...game_types import Boop, BoopAction, BoopGraduateLine, BoopPieceKind, BoopPosition, BoopRecoverPiece
from ...matches.models import Move
from ...native_bridge import _analyze_trace
from ...extraction.schema import (
    _CsvWriter, _MatchContext, _integer_field, _string_field,
    _trace_root_actions, _trace_tree_reuse, _tree_reuse_row, _TREE_REUSE_FIELDS,
)

_BOOP_OUTPUT_FILES = {
    "boop_matches": "boop_matches.csv",
    "turns": "turns.csv",
    "boops": "boops.csv",
    "resolutions": "resolutions.csv",
    "winning_lines": "winning_lines.csv",
}

_BOOP_MATCH_FIELDS = (
    "match_number",
    "win_by_cat_line",
    "win_by_eight_cats",
    "first_graduation_ply",
    "first_graduation_player",
    "first_graduation_agent",
)

_TURN_BASE_FIELDS = (
    "match_number",
    "ply",
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
    "strategic_phase",
    "piece",
    "row",
    "column",
    "zone",
    "resolution",
    "kittens_promoted",
    "cats_recycled",
    "moved_own",
    "moved_opponent",
    "off_board_own",
    "off_board_opponent",
    "blocked_own",
    "blocked_opponent",
    "immune_own",
    "immune_opponent",
    "terminal_after",
)

_STATE_FIELDS = tuple(
    field
    for stage in ("before", "after")
    for field in (
        f"empty_center_{stage}",
        f"empty_middle_{stage}",
        f"empty_outer_{stage}",
        *(
            metric
            for player in (0, 1)
            for metric in (
                f"p{player}_pool_kittens_{stage}",
                f"p{player}_pool_cats_{stage}",
                f"p{player}_board_kittens_{stage}",
                f"p{player}_board_cats_{stage}",
                f"p{player}_total_cats_{stage}",
                f"p{player}_center_pieces_{stage}",
                f"p{player}_middle_pieces_{stage}",
                f"p{player}_outer_pieces_{stage}",
            )
        ),
    )
)

_TURN_FIELDS = _TURN_BASE_FIELDS + _STATE_FIELDS

_BOOP_FIELDS = (
    "match_number",
    "ply",
    "interaction_number",
    "actor_player",
    "actor_agent",
    "placed_piece",
    "placed_row",
    "placed_column",
    "target_player",
    "target_agent",
    "target_relation",
    "target_piece",
    "origin_row",
    "origin_column",
    "destination_row",
    "destination_column",
    "outcome",
)

_RESOLUTION_FIELDS = (
    "match_number",
    "ply",
    "player",
    "agent",
    "type",
    "kittens_promoted",
    "cats_recycled",
    "recovered_piece",
    "recovery_row",
    "recovery_column",
    "orientation",
    "line_row_1",
    "line_column_1",
    "line_row_2",
    "line_column_2",
    "line_row_3",
    "line_column_3",
)

_WINNING_LINE_FIELDS = (
    "match_number",
    "line_number",
    "player",
    "agent",
    "is_declared_winner",
    "orientation",
    "row_1",
    "column_1",
    "row_2",
    "column_2",
    "row_3",
    "column_3",
)

def _extract_boop_match(
    context: _MatchContext,
    writers: dict[str, _CsvWriter],
    row_counts: dict[str, int],
    game: Boop,
) -> None:
    moves = tuple(
        _trace_move(raw, context.match_number, index)
        for index, raw in enumerate(context.raw_moves, 1)
    )
    analysis = _analyze_trace(game, moves)
    winner_player = context.winner_player
    if winner_player is None or analysis["winner"] != winner_player:
        raise ValueError(
            f"match {context.match_number} replay winner does not match its result"
        )
    first_graduation = next(
        (
            turn
            for turn in analysis["turns"]
            if turn["resolution"] is not None
            and turn["resolution"]["type"] == "graduate"
        ),
        None,
    )
    writers["boop_matches"].writerow(
        {
            "match_number": context.match_number,
            "win_by_cat_line": analysis["winner_has_cat_line"],
            "win_by_eight_cats": analysis["winner_has_eight_cats"],
            "first_graduation_ply": "" if first_graduation is None else first_graduation["ply"],
            "first_graduation_player": (
                "" if first_graduation is None else first_graduation["player"]
            ),
            "first_graduation_agent": (
                ""
                if first_graduation is None
                else context.players[first_graduation["player"]]
            ),
        }
    )
    row_counts["boop_matches"] += 1

    for move, turn in zip(moves, analysis["turns"], strict=True):
        _write_turn(
            writers,
            row_counts,
            context.match_number,
            context.plies,
            context.players,
            winner_player,
            move,
            turn,
        )

    for line in analysis["winning_lines"]:
        positions = line["positions"]
        writers["winning_lines"].writerow(
            {
                "match_number": context.match_number,
                "line_number": line["line_number"],
                "player": line["player"],
                "agent": context.players[line["player"]],
                "is_declared_winner": line["player"] == winner_player,
                "orientation": line["orientation"],
                **_position_columns(positions, "row", "column"),
            }
        )
        row_counts["winning_lines"] += 1

def _write_turn(
    writers: dict[str, _CsvWriter],
    row_counts: dict[str, int],
    match_number: int,
    total_plies: int,
    players: list[str],
    winner_player: int,
    move: Move,
    turn: dict[str, object],
) -> None:
    action = move.action
    if not isinstance(action, BoopAction):
        raise TypeError(f"match {match_number} contains a non-boop action")
    if turn["player"] != move.player:
        raise ValueError(f"match {match_number} replay player differs at ply {turn['ply']}")
    relation_counts = {
        f"{outcome}_{relation}": 0
        for outcome in ("moved", "off_board", "blocked", "immune")
        for relation in ("own", "opponent")
    }
    for interaction in turn["interactions"]:
        relation = "own" if interaction["target_player"] == move.player else "opponent"
        relation_counts[f"{interaction['outcome']}_{relation}"] += 1
        writers["boops"].writerow(
            {
                "match_number": match_number,
                "ply": turn["ply"],
                "interaction_number": interaction["interaction_number"],
                "actor_player": move.player,
                "actor_agent": players[move.player],
                "placed_piece": action.piece.value,
                "placed_row": action.row,
                "placed_column": action.column,
                "target_player": interaction["target_player"],
                "target_agent": players[interaction["target_player"]],
                "target_relation": relation,
                "target_piece": interaction["target_piece"],
                "origin_row": interaction["origin_row"],
                "origin_column": interaction["origin_column"],
                "destination_row": interaction["destination_row"],
                "destination_column": interaction["destination_column"],
                "outcome": interaction["outcome"],
            }
        )
        row_counts["boops"] += 1

    resolution = turn["resolution"]
    resolution_type = "none" if resolution is None else resolution["type"]
    if resolution is not None:
        positions = resolution["positions"]
        row = {
            "match_number": match_number,
            "ply": turn["ply"],
            "player": move.player,
            "agent": players[move.player],
            "type": resolution_type,
            "kittens_promoted": resolution["kittens_promoted"],
            "cats_recycled": resolution["cats_recycled"],
            "recovered_piece": resolution["recovered_piece"] or "",
            "recovery_row": positions[0][0] if resolution_type == "recover" else "",
            "recovery_column": positions[0][1] if resolution_type == "recover" else "",
            "orientation": resolution["orientation"] or "",
            **_position_columns(
                positions if resolution_type == "graduate" else [],
                "line_row",
                "line_column",
            ),
        }
        writers["resolutions"].writerow(row)
        row_counts["resolutions"] += 1

    turn_row = {
        "match_number": match_number,
        "ply": turn["ply"],
        "total_plies": total_plies,
        "player": move.player,
        "agent": players[move.player],
        "outcome": "win" if move.player == winner_player else "loss",
        "decision_seconds": move.decision_seconds,
        "search_iterations": move.search_iterations,
        "search_nodes": move.search_nodes,
        **_tree_reuse_row(move.tree_reuse),
        "progress_fraction": turn["ply"] / total_plies,
        "game_quarter": f"q{min(3, ((turn['ply'] - 1) * 4) // total_plies) + 1}",
        "strategic_phase": turn["phase"],
        "piece": action.piece.value,
        "row": action.row,
        "column": action.column,
        "zone": turn["zone"],
        "resolution": resolution_type,
        "kittens_promoted": 0 if resolution is None else resolution["kittens_promoted"],
        "cats_recycled": 0 if resolution is None else resolution["cats_recycled"],
        "terminal_after": turn["terminal_after"],
        **relation_counts,
        **_flatten_state("before", turn["before"]),
        **_flatten_state("after", turn["after"]),
    }
    writers["turns"].writerow(turn_row)
    row_counts["turns"] += 1

def _flatten_state(stage: str, state: dict[str, object]) -> dict[str, object]:
    result = {
        f"empty_center_{stage}": state["empty_center"],
        f"empty_middle_{stage}": state["empty_middle"],
        f"empty_outer_{stage}": state["empty_outer"],
    }
    for player, metrics in enumerate(state["players"]):
        for source, target in (
            ("pool_kittens", "pool_kittens"),
            ("pool_cats", "pool_cats"),
            ("board_kittens", "board_kittens"),
            ("board_cats", "board_cats"),
            ("total_cats", "total_cats"),
            ("center_pieces", "center_pieces"),
            ("middle_pieces", "middle_pieces"),
            ("outer_pieces", "outer_pieces"),
        ):
            result[f"p{player}_{target}_{stage}"] = metrics[source]
    return result

def _trace_move(raw: object, match_number: int, expected_ply: int) -> Move:
    if not isinstance(raw, dict):
        raise TypeError(f"match {match_number} ply {expected_ply} must be an object")
    ply = _integer_field(raw, "ply", f"match {match_number} move")
    if ply != expected_ply:
        raise ValueError(f"match {match_number} has unexpected ply {ply}, expected {expected_ply}")
    player = _integer_field(raw, "player", f"match {match_number} ply {ply}")
    action = raw.get("action")
    if not isinstance(action, dict) or action.get("type") != "boop":
        raise TypeError(f"match {match_number} ply {ply} must contain a boop action")
    raw_resolution = action.get("resolution")
    if not isinstance(raw_resolution, dict):
        raise TypeError(f"match {match_number} ply {ply} resolution must be an object")
    resolution_type = raw_resolution.get("type")
    if resolution_type == "graduate":
        raw_positions = raw_resolution.get("positions")
        if not isinstance(raw_positions, list) or len(raw_positions) != 3:
            raise TypeError(f"match {match_number} ply {ply} graduation needs three positions")
        resolution = BoopGraduateLine(
            tuple(_trace_position(position) for position in raw_positions)
        )
    elif resolution_type == "recover":
        resolution = BoopRecoverPiece(
            BoopPosition(
                _integer_field(raw_resolution, "row", "recovery"),
                _integer_field(raw_resolution, "column", "recovery"),
            )
        )
    elif resolution_type == "none":
        resolution = None
    else:
        raise ValueError(f"match {match_number} ply {ply} has unknown resolution")
    return Move(
        player=player,
        decision_seconds=raw.get("decision_seconds", ""),
        search_iterations=raw.get("search_iterations", ""),
        search_nodes=raw.get("search_nodes", ""),
        root_actions=_trace_root_actions(
            raw.get("root_actions"), f"match {match_number} ply {ply}"
        ),
        tree_reuse=_trace_tree_reuse(
            raw.get("tree_reuse"), f"match {match_number} ply {ply}"
        ),
        action=BoopAction(
            piece=BoopPieceKind(_string_field(action, "piece", "boop action")),
            row=_integer_field(action, "row", "boop action"),
            column=_integer_field(action, "column", "boop action"),
            resolution=resolution,
        ),
    )

def _trace_position(raw: object) -> BoopPosition:
    if not isinstance(raw, dict):
        raise TypeError("graduation position must be an object")
    return BoopPosition(
        _integer_field(raw, "row", "graduation position"),
        _integer_field(raw, "column", "graduation position"),
    )

def _position_columns(
    positions: list[tuple[int, int]],
    row_prefix: str,
    column_prefix: str,
) -> dict[str, object]:
    result = {}
    for index in range(3):
        if index < len(positions):
            row, column = positions[index]
        else:
            row, column = "", ""
        result[f"{row_prefix}_{index + 1}"] = row
        result[f"{column_prefix}_{index + 1}"] = column
    return result
