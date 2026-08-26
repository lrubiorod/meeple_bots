"""Streaming extraction of analysis-ready tables from tournament and batch traces."""

from __future__ import annotations

import csv
import json
import tempfile
from collections.abc import Sequence
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
    EndSpiritCollection,
    ForestPosition,
    Move,
    MoveSpiritGemstone,
    PlaceSpiritGemstone,
    SkipSpiritGemstone,
    SpiritGemstoneSacrifice,
    SpiritsOfTheForest,
    TakeSpiritTile,
    TicTacToe,
    _analyze_trace,
)

_COMMON_OUTPUT_FILES = {
    "agents": "agents.csv",
    "studies": "studies.csv",
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

_SPOTF_OUTPUT_FILES = {
    "spotf_matches": "spotf_matches.csv",
    "actions": "actions.csv",
    "player_turns": "player_turns.csv",
    "tile_takes": "tile_takes.csv",
    "gemstone_actions": "gemstone_actions.csv",
    "categories": "categories.csv",
}

_SPOTF_TILE_COUNT = 48

_AGENT_FIELDS = (
    "agent_name",
    "kind",
    "iterations",
    "time_budget",
    "rollout_depth",
    "exploration",
    "heuristic",
    "cutoff_evaluator",
    "cutoff_heuristic",
    "rollout_policy",
    "rollout_evaluator",
    "rollout_heuristic",
    "rollout_epsilon",
    "self_play",
)

_STUDY_FIELDS = (
    "study_id",
    "study_type",
    "source",
    "tournament_schema_version",
    "declared_matches",
    "processed_matches",
    "complete",
    "truncated_last_line",
    "agents",
    "base_seed",
    "matches_per_pair",
)

_PROVENANCE_FIELDS = (
    "study_id",
    "source_match_number",
)

_MATCH_FIELDS = (
    "match_number",
    *_PROVENANCE_FIELDS,
    "pairing_number",
    "source_pairing_number",
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
    "decision_seconds",
    "search_iterations",
    "search_nodes",
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


@dataclass(frozen=True, slots=True)
class _MatchContext:
    match_number: int
    seed: int
    players: list[str]
    winner_player: int | None
    plies: int
    raw_moves: list[object]
    raw_result: dict[str, object]


@dataclass(frozen=True, slots=True)
class _StudySource:
    study_id: str
    study_type: str
    path: Path
    header: dict[str, object]
    game_name: str
    declared_matches: int
    raw_agents: list[object]
    agent_names: set[str]


class _CsvWriter:
    def __init__(self, writer: csv.DictWriter, *, with_provenance: bool = False) -> None:
        self._writer = writer
        self._with_provenance = with_provenance
        self._study_id: str | None = None
        self._source_match_number: int | None = None

    def select_match(self, study_id: str, source_match_number: int) -> None:
        self._study_id = study_id
        self._source_match_number = source_match_number

    def writerow(self, row: dict[str, object]) -> None:
        if not self._with_provenance:
            self._writer.writerow(row)
            return
        if self._study_id is None or self._source_match_number is None:
            raise RuntimeError("match provenance must be selected before writing rows")
        self._writer.writerow(
            {
                "study_id": self._study_id,
                "source_match_number": self._source_match_number,
                **row,
            }
        )


def _normalize_input_paths(input_paths: Path | Sequence[Path]) -> tuple[Path, ...]:
    if isinstance(input_paths, Path):
        normalized = (input_paths.resolve(),)
    else:
        normalized = tuple(path.resolve() for path in input_paths)
    if not normalized:
        raise ValueError("at least one tournament trace input is required")
    duplicates = sorted(
        str(path) for path in set(normalized) if normalized.count(path) > 1
    )
    if duplicates:
        raise ValueError("duplicate tournament trace inputs: " + ", ".join(duplicates))
    return normalized


def _load_study_sources(paths: tuple[Path, ...]) -> tuple[_StudySource, ...]:
    studies = []
    study_id_counts: dict[str, int] = {}
    expected_game: str | None = None
    for path in paths:
        with path.open(encoding="utf-8") as source:
            header_line = source.readline()
        if not header_line:
            raise ValueError(f"tournament trace is empty: {path}")
        raw_header = _parse_json_record(header_line, 1)
        game_name = _validate_header(raw_header)
        if not isinstance(raw_header, dict):
            raise AssertionError("validated tournament header must be an object")
        study_type = raw_header.get("study_type", "tournament")
        if not isinstance(study_type, str) or study_type not in {
            "batch",
            "tournament",
        }:
            raise ValueError(
                f"study_type must be batch or tournament in trace: {path}"
            )
        if expected_game is None:
            expected_game = game_name
        elif game_name != expected_game:
            raise ValueError(
                "all tournament traces must use the same game; "
                f"expected {expected_game}, found {game_name} in {path}"
            )

        raw_agents = raw_header.get("agents")
        if not isinstance(raw_agents, list):
            raise TypeError(f"tournament header agents must be a list: {path}")
        agent_names = [
            _string_field(agent, "name", f"tournament agent in {path}")
            for agent in raw_agents
        ]
        duplicate_agents = sorted(
            name for name in set(agent_names) if agent_names.count(name) > 1
        )
        if duplicate_agents:
            raise ValueError(
                f"tournament trace {path} contains duplicate agent names: "
                + ", ".join(duplicate_agents)
            )

        base_id = path.stem or "study"
        occurrence = study_id_counts.get(base_id, 0) + 1
        study_id_counts[base_id] = occurrence
        study_id = base_id if occurrence == 1 else f"{base_id}-{occurrence}"
        studies.append(
            _StudySource(
                study_id=study_id,
                study_type=study_type,
                path=path,
                header=raw_header,
                game_name=game_name,
                declared_matches=_integer_field(
                    raw_header, "total_matches", f"tournament header in {path}"
                ),
                raw_agents=raw_agents,
                agent_names=set(agent_names),
            )
        )
    return tuple(studies)


def _combined_agent_rows(studies: tuple[_StudySource, ...]) -> list[dict[str, object]]:
    combined: dict[str, dict[str, object]] = {}
    first_source: dict[str, Path] = {}
    for study in studies:
        for raw_agent in study.raw_agents:
            row = _agent_row(raw_agent)
            name = str(row["agent_name"])
            existing = combined.get(name)
            if existing is None:
                combined[name] = row
                first_source[name] = study.path
                continue
            signature = _agent_signature(row)
            existing_signature = _agent_signature(existing)
            if signature != existing_signature:
                raise ValueError(
                    f'agent "{name}" has conflicting configurations: '
                    f"{first_source[name]} defines {existing_signature}, "
                    f"but {study.path} defines {signature}; use distinct agent names"
                )
            existing["self_play"] = bool(existing["self_play"] or row["self_play"])
    return list(combined.values())


def _agent_signature(row: dict[str, object]) -> dict[str, object]:
    return {
        field: row[field]
        for field in (
            "kind",
            "iterations",
            "time_budget",
            "rollout_depth",
            "exploration",
            "heuristic",
            "cutoff_evaluator",
            "cutoff_heuristic",
            "rollout_policy",
            "rollout_evaluator",
            "rollout_heuristic",
            "rollout_epsilon",
        )
    }


def _fields_with_provenance(fields: tuple[str, ...]) -> tuple[str, ...]:
    if not fields or fields[0] != "match_number":
        raise ValueError("match analysis tables must begin with match_number")
    return (fields[0], *_PROVENANCE_FIELDS, *fields[1:])


def extract_tournament(
    input_paths: Path | Sequence[Path],
    output_dir: Path | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Combine compatible version-1 tournament or batch traces into analysis-ready tables."""

    paths = _normalize_input_paths(input_paths)
    studies = _load_study_sources(paths)
    agent_rows = _combined_agent_rows(studies)
    if output_dir is None:
        if len(paths) > 1:
            raise ValueError("--output-dir is required when extracting multiple studies")
        output_dir = paths[0].parent / paths[0].stem / "data"
    else:
        output_dir = output_dir.resolve()
    game_name = studies[0].game_name
    game = _trace_game(game_name)
    if isinstance(game, Boop):
        extract_match = _extract_boop_match
        game_output_files = _BOOP_OUTPUT_FILES
        game_writer_fields = {
            "boop_matches": _BOOP_MATCH_FIELDS,
            "turns": _TURN_FIELDS,
            "boops": _BOOP_FIELDS,
            "resolutions": _RESOLUTION_FIELDS,
            "winning_lines": _WINNING_LINE_FIELDS,
        }
        game_metadata = {
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
        }
    elif isinstance(game, SpiritsOfTheForest):
        extract_match = _extract_spotf_match
        game_output_files = _SPOTF_OUTPUT_FILES
        game_writer_fields = {
            "spotf_matches": _SPOTF_MATCH_FIELDS,
            "actions": _SPOTF_ACTION_FIELDS,
            "player_turns": _PLAYER_TURN_FIELDS,
            "tile_takes": _TILE_TAKE_FIELDS,
            "gemstone_actions": _GEMSTONE_ACTION_FIELDS,
            "categories": _CATEGORY_FIELDS,
        }
        game_metadata = {
            "turn_semantics": {
                "ply": "one internal game-tree action",
                "physical_turn": "consecutive actions by one player before control changes",
            },
            "scoring_categories": {
                "spirit": "nine spirit majorities",
                "power_source": "fire, moon, and sun majorities",
            },
        }
    else:
        _analyze_trace(game, ())
        raise AssertionError("unavailable analysis must return an error")

    output_files = _COMMON_OUTPUT_FILES | game_output_files
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
    declared_matches = sum(study.declared_matches for study in studies)
    pairing_offset = 0
    source_summaries = []

    with tempfile.TemporaryDirectory(prefix=".extract-", dir=output_dir) as temporary:
        temporary_dir = Path(temporary)
        with ExitStack() as stack:
            writers = {
                "agents": _csv_writer(stack, temporary_dir / "agents.csv", _AGENT_FIELDS),
                "studies": _csv_writer(stack, temporary_dir / "studies.csv", _STUDY_FIELDS),
                "matches": _csv_writer(
                    stack,
                    temporary_dir / "matches.csv",
                    _MATCH_FIELDS,
                    with_provenance=True,
                ),
                **{
                    name: _csv_writer(
                        stack,
                        temporary_dir / game_output_files[name],
                        _fields_with_provenance(fields),
                        with_provenance=True,
                    )
                    for name, fields in game_writer_fields.items()
                },
            }
            match_writers = [writers["matches"]] + [
                writers[name] for name in game_writer_fields
            ]
            for agent in agent_rows:
                writers["agents"].writerow(agent)
                row_counts["agents"] += 1

            for study in studies:
                study_processed = 0
                study_truncated = False
                study_pairing_max = 0
                seen_source_matches: set[int] = set()
                with study.path.open(encoding="utf-8") as source:
                    source.readline()
                    for line_number, line in enumerate(source, start=2):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as error:
                            if not line.endswith("\n"):
                                study_truncated = True
                                break
                            raise ValueError(
                                f"invalid JSON in {study.path} on line {line_number}: "
                                f"{error.msg}"
                            ) from error
                        source_match_number = _integer_field(
                            record, "match_number", f"match record in {study.path}"
                        )
                        for writer in match_writers:
                            writer.select_match(study.study_id, source_match_number)
                        processed_matches += 1
                        study_processed += 1
                        context = _extract_common_match(
                            record,
                            writers,
                            row_counts,
                            study.agent_names,
                            seen_source_matches,
                            global_match_number=processed_matches,
                            pairing_offset=pairing_offset,
                        )
                        source_pairing = _integer_field(
                            record,
                            "pairing_number",
                            f"match {source_match_number} in {study.path}",
                        )
                        study_pairing_max = max(study_pairing_max, source_pairing)
                        extract_match(context, writers, row_counts, game)

                study_complete = (
                    study_processed == study.declared_matches and not study_truncated
                )
                source_summary = {
                    "study_id": study.study_id,
                    "study_type": study.study_type,
                    "source": str(study.path),
                    "tournament_schema_version": study.header["schema_version"],
                    "declared_matches": study.declared_matches,
                    "processed_matches": study_processed,
                    "complete": study_complete,
                    "truncated_last_line": study_truncated,
                    "agents": len(study.raw_agents),
                    "base_seed": study.header.get("seed", ""),
                    "matches_per_pair": study.header.get("matches_per_pair", ""),
                }
                writers["studies"].writerow(source_summary)
                row_counts["studies"] += 1
                source_summaries.append(source_summary)
                declared_pairings = study.header.get("total_pairings", 0)
                if isinstance(declared_pairings, bool) or not isinstance(
                    declared_pairings, int
                ):
                    raise TypeError(
                        f"tournament header total_pairings must be an integer: {study.path}"
                    )
                pairing_offset += max(study_pairing_max, declared_pairings)

        complete = all(bool(source["complete"]) for source in source_summaries)
        truncated_last_line = any(
            bool(source["truncated_last_line"]) for source in source_summaries
        )
        manifest = {
            "schema_version": 1,
            "source": str(paths[0]) if len(paths) == 1 else None,
            "sources": source_summaries,
            "output_dir": str(output_dir),
            "game": game_name,
            "tournament_schema_version": 1,
            "analysis_schema_version": 4,
            "declared_matches": declared_matches,
            "processed_matches": processed_matches,
            "complete": complete,
            "truncated_last_line": truncated_last_line,
            "row_counts": row_counts,
            **game_metadata,
            "tables": output_files,
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        for name, target in targets.items():
            (temporary_dir / output_files[name]).replace(target)

    return {
        "input": str(paths[0]),
        "inputs": [str(path) for path in paths],
        "output_dir": str(output_dir),
        "declared_matches": declared_matches,
        "processed_matches": processed_matches,
        "complete": complete,
        "truncated_last_line": truncated_last_line,
        "row_counts": row_counts,
        "studies": source_summaries,
    }


def _extract_common_match(
    record: object,
    writers: dict[str, _CsvWriter],
    row_counts: dict[str, int],
    agent_names: set[str],
    seen_matches: set[int],
    *,
    global_match_number: int,
    pairing_offset: int,
) -> _MatchContext:
    if not isinstance(record, dict) or record.get("record_type") != "match":
        raise ValueError("every tournament record after the header must be a match")
    source_match_number = _integer_field(record, "match_number", "match record")
    if source_match_number in seen_matches:
        raise ValueError(f"duplicate tournament match_number {source_match_number}")
    seen_matches.add(source_match_number)
    match_number = global_match_number

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
    seed = _integer_field(raw_result, "seed", f"match {match_number} result")
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
            "pairing_number": pairing_offset
            + _integer_field(record, "pairing_number", f"match {source_match_number}"),
            "source_pairing_number": _integer_field(
                record, "pairing_number", f"match {source_match_number}"
            ),
            "pairing_match_number": _integer_field(
                record, "pairing_match_number", f"match {match_number}"
            ),
            "seed": seed,
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
        seed=seed,
        players=players,
        winner_player=winner_player,
        plies=plies,
        raw_moves=raw_moves,
        raw_result=raw_result,
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


def _game_quarter(ply: int, total_plies: int) -> str:
    return f"q{min(3, ((ply - 1) * 4) // total_plies) + 1}"


def _tile_progress(state: dict[str, object]) -> float:
    return (_SPOTF_TILE_COUNT - int(state["remaining_tiles"])) / _SPOTF_TILE_COUNT


def _tile_quarter(state: dict[str, object]) -> str:
    collected = _SPOTF_TILE_COUNT - int(state["remaining_tiles"])
    return f"q{min(3, collected * 4 // _SPOTF_TILE_COUNT) + 1}"


def _player_outcome(player: int, winner_player: int | None) -> str:
    if winner_player is None:
        return "draw"
    return "win" if player == winner_player else "loss"


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
        action=BoopAction(
            piece=BoopPieceKind(_string_field(action, "piece", "boop action")),
            row=_integer_field(action, "row", "boop action"),
            column=_integer_field(action, "column", "boop action"),
            resolution=resolution,
        ),
    )


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
    self_play = raw.get("self_play", False)
    if not isinstance(self_play, bool):
        raise TypeError("tournament agent self_play must be a boolean")
    kind = _string_field(raw, "type", "tournament agent")
    rollout_policy = raw.get("rollout_policy")
    if rollout_policy is None:
        rollout_policy = "uniform_random" if kind == "mcts" else ""
    if not isinstance(rollout_policy, str):
        raise TypeError("tournament agent rollout_policy must be a string")
    if rollout_policy == "epsilon_greedy_heuristic":
        rollout_policy = "epsilon_greedy"
    cutoff_kind, cutoff_heuristic = _serialized_evaluator(
        raw.get("cutoff_evaluator"),
        fallback_heuristic=raw.get("heuristic"),
        context="tournament cutoff evaluator",
        allow_missing=kind != "mcts",
    )
    rollout_raw = raw.get("rollout_evaluator")
    rollout_fallback = (
        raw.get("heuristic")
        if rollout_raw is None and rollout_policy in {"greedy", "epsilon_greedy"}
        else None
    )
    rollout_kind, rollout_heuristic = _serialized_evaluator(
        rollout_raw,
        fallback_heuristic=rollout_fallback,
        context="tournament rollout evaluator",
        allow_missing=True,
    )
    return {
        "agent_name": _string_field(raw, "name", "tournament agent"),
        "kind": kind,
        "iterations": raw.get("iterations", ""),
        "time_budget": raw.get("time_budget", ""),
        "rollout_depth": raw.get("rollout_depth", ""),
        "exploration": raw.get("exploration", ""),
        "heuristic": "" if raw.get("heuristic") is None else raw["heuristic"],
        "cutoff_evaluator": cutoff_kind,
        "cutoff_heuristic": cutoff_heuristic,
        "rollout_policy": rollout_policy,
        "rollout_evaluator": rollout_kind,
        "rollout_heuristic": rollout_heuristic,
        "rollout_epsilon": (
            "" if raw.get("rollout_epsilon") is None else raw["rollout_epsilon"]
        ),
        "self_play": self_play,
    }


def _serialized_evaluator(
    raw: object,
    *,
    fallback_heuristic: object,
    context: str,
    allow_missing: bool = False,
) -> tuple[str, object]:
    if raw is None:
        if fallback_heuristic is not None:
            return "game_heuristic", fallback_heuristic
        return ("", "") if allow_missing else ("neutral", "")
    if not isinstance(raw, dict):
        raise TypeError(f"{context} must be an object")
    kind = raw.get("kind")
    if kind == "neutral":
        if raw.get("index") is not None:
            raise ValueError(f"{context} neutral evaluator cannot have an index")
        return "neutral", ""
    if kind != "game_heuristic":
        raise ValueError(f"{context} kind must be neutral or game_heuristic")
    index = raw.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError(f"{context} game_heuristic requires an integer index")
    return "game_heuristic", index


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
    *,
    with_provenance: bool = False,
) -> _CsvWriter:
    output = stack.enter_context(path.open("w", encoding="utf-8", newline=""))
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    return _CsvWriter(writer, with_provenance=with_provenance)


def _validate_header(header: object) -> str:
    if not isinstance(header, dict) or header.get("record_type") != "tournament":
        raise ValueError("first JSONL record must be a tournament header")
    if header.get("schema_version") != 1:
        raise ValueError("extract supports tournament schema_version 1")
    game = header.get("game")
    if game not in {"boop", "connect-four", "spotf", "tic-tac-toe"}:
        raise ValueError(f"unknown tournament game: {game}")
    return game


def _trace_game(name: str) -> Boop | ConnectFour | SpiritsOfTheForest | TicTacToe:
    match name:
        case "boop":
            return Boop()
        case "connect-four":
            return ConnectFour()
        case "spotf":
            return SpiritsOfTheForest()
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
