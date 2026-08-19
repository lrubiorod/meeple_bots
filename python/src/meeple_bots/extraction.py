"""Streaming extraction of analysis-ready tables from tournament traces."""

from __future__ import annotations

import csv
import json
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from .api import (
    Boop,
    BoopAction,
    BoopGraduateLine,
    BoopPieceKind,
    BoopPosition,
    BoopRecoverPiece,
    ConnectFour,
    Move,
    TicTacToe,
    _analyze_trace,
)

_COMMON_OUTPUT_FILES = {
    "agents": "agents.csv",
    "matches": "matches.csv",
    "manifest": "manifest.json",
}

_BOOP_OUTPUT_FILES = {
    "boop_matches": "boop_matches.csv",
    "turns": "turns.csv",
    "boops": "boops.csv",
    "resolutions": "resolutions.csv",
    "winning_lines": "winning_lines.csv",
}

_AGENT_FIELDS = (
    "agent_name",
    "kind",
    "iterations",
    "rollout_depth",
    "exploration",
    "heuristic",
    "self_play",
)

_MATCH_FIELDS = (
    "match_number",
    "pairing_number",
    "pairing_match_number",
    "seed",
    "duration_seconds",
    "self_play",
    "agent_a",
    "agent_b",
    "agent_a_player",
    "player_0_agent",
    "player_1_agent",
    "winner_role",
    "winner_player",
    "winner_agent",
    "plies",
    "utility_0",
    "utility_1",
)

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


@dataclass(frozen=True, slots=True)
class _MatchContext:
    match_number: int
    players: list[str]
    winner_player: int | None
    plies: int
    raw_moves: list[object]


def extract_tournament(
    input_path: Path,
    output_dir: Path | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Dispatch a version-1 tournament trace to its game-specific extractor."""

    input_path = input_path.resolve()
    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}-data"
    else:
        output_dir = output_dir.resolve()
    with input_path.open(encoding="utf-8") as source:
        header_line = source.readline()
        if not header_line:
            raise ValueError("tournament trace is empty")
        header = _parse_json_record(header_line, 1)
        game_name = _validate_header(header)
        game = _trace_game(game_name)
        if isinstance(game, Boop):
            extract_match = _extract_boop_match
        else:
            _analyze_trace(game, ())
            raise AssertionError("unavailable analysis must return an error")
        output_files = _COMMON_OUTPUT_FILES | _BOOP_OUTPUT_FILES
        targets = {name: output_dir / filename for name, filename in output_files.items()}
        existing = sorted(str(path) for path in targets.values() if path.exists())
        if existing and not overwrite:
            raise FileExistsError(
                "extraction output already exists; use --overwrite to replace: "
                + ", ".join(existing)
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        row_counts = {name: 0 for name in output_files if name != "manifest"}
        processed_matches = 0
        truncated_last_line = False
        seen_matches: set[int] = set()
        declared_matches = _integer_field(header, "total_matches", "tournament header")
        raw_agents = header.get("agents")
        if not isinstance(raw_agents, list):
            raise TypeError("tournament header agents must be a list")
        agent_names = {
            _string_field(agent, "name", "tournament agent") for agent in raw_agents
        }

        with tempfile.TemporaryDirectory(prefix=".extract-", dir=output_dir) as temporary:
            temporary_dir = Path(temporary)
            with ExitStack() as stack:
                writers = {
                    "agents": _csv_writer(stack, temporary_dir / "agents.csv", _AGENT_FIELDS),
                    "matches": _csv_writer(stack, temporary_dir / "matches.csv", _MATCH_FIELDS),
                    "boop_matches": _csv_writer(
                        stack,
                        temporary_dir / "boop_matches.csv",
                        _BOOP_MATCH_FIELDS,
                    ),
                    "turns": _csv_writer(stack, temporary_dir / "turns.csv", _TURN_FIELDS),
                    "boops": _csv_writer(stack, temporary_dir / "boops.csv", _BOOP_FIELDS),
                    "resolutions": _csv_writer(
                        stack,
                        temporary_dir / "resolutions.csv",
                        _RESOLUTION_FIELDS,
                    ),
                    "winning_lines": _csv_writer(
                        stack,
                        temporary_dir / "winning_lines.csv",
                        _WINNING_LINE_FIELDS,
                    ),
                }
                for agent in raw_agents:
                    writers["agents"].writerow(_agent_row(agent))
                    row_counts["agents"] += 1

                for line_number, line in enumerate(source, start=2):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        if not line.endswith("\n"):
                            truncated_last_line = True
                            break
                        raise ValueError(
                            f"invalid JSON on tournament trace line {line_number}: {error.msg}"
                        ) from error
                    context = _extract_common_match(
                        record,
                        writers,
                        row_counts,
                        agent_names,
                        seen_matches,
                    )
                    extract_match(context, writers, row_counts, game)
                    processed_matches += 1

            complete = processed_matches == declared_matches and not truncated_last_line
            manifest = {
                "schema_version": 1,
                "source": str(input_path),
                "output_dir": str(output_dir),
                "game": game_name,
                "tournament_schema_version": header["schema_version"],
                "analysis_schema_version": 1,
                "declared_matches": declared_matches,
                "processed_matches": processed_matches,
                "complete": complete,
                "truncated_last_line": truncated_last_line,
                "row_counts": row_counts,
                "zones": {
                    "center": "rows 2-3 and columns 2-3 (4 cells)",
                    "middle": "remaining cells inside rows 1-4 and columns 1-4 (12 cells)",
                    "outer": "board perimeter (20 cells)",
                },
                "strategic_phases": {
                    "all_kittens": "neither player has acquired a cat",
                    "one_player_has_cats": "exactly one player has acquired at least one cat",
                    "both_players_have_cats": "both players have acquired at least one cat",
                },
                "tables": output_files,
            }
            (temporary_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            for name, target in targets.items():
                (temporary_dir / output_files[name]).replace(target)

    return {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "declared_matches": declared_matches,
        "processed_matches": processed_matches,
        "complete": complete,
        "truncated_last_line": truncated_last_line,
        "row_counts": row_counts,
    }


def _extract_common_match(
    record: object,
    writers: dict[str, csv.DictWriter],
    row_counts: dict[str, int],
    agent_names: set[str],
    seen_matches: set[int],
) -> _MatchContext:
    if not isinstance(record, dict) or record.get("record_type") != "match":
        raise ValueError("every tournament record after the header must be a match")
    match_number = _integer_field(record, "match_number", "match record")
    if match_number in seen_matches:
        raise ValueError(f"duplicate tournament match_number {match_number}")
    seen_matches.add(match_number)

    agent_a = _string_field(record, "agent_a", f"match {match_number}")
    agent_b = _string_field(record, "agent_b", f"match {match_number}")
    if agent_a not in agent_names or agent_b not in agent_names:
        raise ValueError(f"match {match_number} references an unknown agent")
    players = record.get("players")
    if (
        not isinstance(players, list)
        or len(players) != 2
        or not all(isinstance(player, str) for player in players)
    ):
        raise TypeError(f"match {match_number} players must contain two agent names")
    agent_a_player = _integer_field(record, "agent_a_player", f"match {match_number}")
    expected_players = [agent_a, agent_b] if agent_a_player == 0 else [agent_b, agent_a]
    if players != expected_players:
        raise ValueError(f"match {match_number} player ordering is inconsistent")

    raw_result = record.get("result")
    if not isinstance(raw_result, dict):
        raise TypeError(f"match {match_number} result must be an object")
    raw_moves = raw_result.get("moves")
    if not isinstance(raw_moves, list):
        raise TypeError(f"match {match_number} moves must be a list")
    plies = _integer_field(raw_result, "plies", f"match {match_number} result")
    if len(raw_moves) != plies:
        raise ValueError(
            f"match {match_number} has {len(raw_moves)} moves but reports {plies} plies"
        )
    winner_player = raw_result.get("winner")
    if winner_player is not None and (
        isinstance(winner_player, bool)
        or not isinstance(winner_player, int)
        or winner_player not in (0, 1)
    ):
        raise TypeError(f"match {match_number} winner must be player 0, player 1, or null")
    winner_agent = "" if winner_player is None else players[winner_player]
    winner_role = record.get("winner")
    expected_winner_role = (
        None
        if winner_player is None
        else "agent_a"
        if winner_player == agent_a_player
        else "agent_b"
    )
    if winner_role != expected_winner_role:
        raise ValueError(f"match {match_number} winner role is inconsistent")

    utilities = raw_result.get("utilities")
    if not isinstance(utilities, list) or len(utilities) != 2:
        raise TypeError(f"match {match_number} utilities must contain two values")
    writers["matches"].writerow(
        {
            "match_number": match_number,
            "pairing_number": _integer_field(record, "pairing_number", f"match {match_number}"),
            "pairing_match_number": _integer_field(
                record, "pairing_match_number", f"match {match_number}"
            ),
            "seed": _integer_field(raw_result, "seed", f"match {match_number} result"),
            "duration_seconds": record.get("duration_seconds"),
            "self_play": record.get("self_play"),
            "agent_a": agent_a,
            "agent_b": agent_b,
            "agent_a_player": agent_a_player,
            "player_0_agent": players[0],
            "player_1_agent": players[1],
            "winner_role": winner_role,
            "winner_player": "" if winner_player is None else winner_player,
            "winner_agent": winner_agent,
            "plies": plies,
            "utility_0": utilities[0],
            "utility_1": utilities[1],
        }
    )
    row_counts["matches"] += 1
    return _MatchContext(
        match_number=match_number,
        players=players,
        winner_player=winner_player,
        plies=plies,
        raw_moves=raw_moves,
    )


def _extract_boop_match(
    context: _MatchContext,
    writers: dict[str, csv.DictWriter],
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
    writers: dict[str, csv.DictWriter],
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


def _agent_row(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise TypeError("tournament agent must be an object")
    return {
        "agent_name": _string_field(raw, "name", "tournament agent"),
        "kind": _string_field(raw, "type", "tournament agent"),
        "iterations": raw.get("iterations", ""),
        "rollout_depth": raw.get("rollout_depth", ""),
        "exploration": raw.get("exploration", ""),
        "heuristic": "" if raw.get("heuristic") is None else raw["heuristic"],
        "self_play": raw.get("self_play", False),
    }


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


def _csv_writer(
    stack: ExitStack,
    path: Path,
    fields: tuple[str, ...],
) -> csv.DictWriter:
    output = stack.enter_context(path.open("w", encoding="utf-8", newline=""))
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    return writer


def _validate_header(header: object) -> str:
    if not isinstance(header, dict) or header.get("record_type") != "tournament":
        raise ValueError("first JSONL record must be a tournament header")
    if header.get("schema_version") != 1:
        raise ValueError("extract supports tournament schema_version 1")
    game = header.get("game")
    if game not in {"boop", "connect-four", "tic-tac-toe"}:
        raise ValueError(f"unknown tournament game: {game}")
    return game


def _trace_game(name: str) -> Boop | ConnectFour | TicTacToe:
    match name:
        case "boop":
            return Boop()
        case "connect-four":
            return ConnectFour()
        case "tic-tac-toe":
            return TicTacToe()
        case _:
            raise ValueError(f"unknown tournament game: {name}")


def _parse_json_record(line: str, line_number: int) -> object:
    try:
        return json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid JSON on tournament trace line {line_number}: {error.msg}"
        ) from error


def _integer_field(value: object, field: str, context: str) -> int:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    result = value.get(field)
    if isinstance(result, bool) or not isinstance(result, int):
        raise TypeError(f"{context} {field} must be an integer")
    return result


def _string_field(value: object, field: str, context: str) -> str:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    result = value.get(field)
    if not isinstance(result, str) or not result:
        raise TypeError(f"{context} {field} must be a non-empty string")
    return result
