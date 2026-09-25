"""Stream validated match traces into analysis tables and publish atomically."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from ..connect6 import Connect6, Connect6Action
from ..game_config import create_game, game_parameters
from ..game_types import Boop, ConnectFour, ConnectFourAction, SpiritsOfTheForest, TicTacToe, TicTacToeAction
from ..matches.models import Move
from ..native_bridge import _analyze_trace
from ..splendor import Splendor
from ..games.boop.extraction import (
    _BOOP_OUTPUT_FILES, _BOOP_MATCH_FIELDS, _TURN_FIELDS, _BOOP_FIELDS,
    _RESOLUTION_FIELDS, _WINNING_LINE_FIELDS, _extract_boop_match,
)
from ..games.spirits_of_the_forest.extraction import (
    _SPOTF_OUTPUT_FILES, _SPOTF_MATCH_FIELDS, _SPOTF_ACTION_FIELDS,
    _PLAYER_TURN_FIELDS, _TILE_TAKE_FIELDS, _GEMSTONE_ACTION_FIELDS,
    _CATEGORY_FIELDS, _ROOT_ACTION_FIELDS, _extract_spotf_match,
)
from .schema import (
    _COMMON_OUTPUT_FILES, _AGENT_FIELDS, _STUDY_FIELDS, _MATCH_FIELDS,
    _GENERIC_MOVE_FIELDS, _MatchContext, _CsvWriter, _agent_row,
    _csv_writer, _fields_with_provenance, _integer_field, _parse_json_record,
    _string_field, _trace_tree_reuse, _tree_reuse_row, _validate_header,
    _game_quarter, _player_outcome,
)

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
    used_study_ids: set[str] = set()
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
        study_id = base_id if occurrence == 1 else f"{base_id}-{occurrence}"
        while study_id in used_study_ids:
            occurrence += 1
            study_id = f"{base_id}-{occurrence}"
        study_id_counts[base_id] = occurrence
        used_study_ids.add(study_id)
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
    # Compare decoded values: object key order and 1 versus 1.0 do not change
    # a configuration. Do not infer missing defaults from the current API.
    return json.loads(row["config_json"])

def extract_tournament(
    input_paths: Path | Sequence[Path],
    output_dir: Path | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Combine compatible version-1 tournament or batch traces into analysis-ready tables."""

    paths = _normalize_input_paths(input_paths)
    studies = _load_study_sources(paths)
    timing_scopes = {
        study.header.get("decision_timing_scope", "selection_only_legacy")
        for study in studies
    }
    if len(timing_scopes) != 1:
        raise ValueError(
            "cannot combine selection-only and total-agent timing studies; extract them separately"
        )
    timing_scope = timing_scopes.pop()
    agent_rows = _combined_agent_rows(studies)
    if output_dir is None:
        if len(paths) > 1:
            raise ValueError("--output-dir is required when extracting multiple studies")
        output_dir = paths[0].parent / paths[0].stem / "data"
    else:
        output_dir = output_dir.resolve()
    game_name = studies[0].game_name
    game = create_game(game_name, studies[0].header.get("game_params"))
    for source in studies:
        if game_parameters(create_game(game_name, source.header.get("game_params"))) != game_parameters(game):
            raise ValueError("cannot combine studies with different game parameters")
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
    elif isinstance(game, Splendor):
        extract_match = _extract_chance_events
        game_output_files = {"chance_events": "chance_events.csv"}
        game_writer_fields = {
            "chance_events": ("match_number", "event_index", "after_ply", "outcome_json"),
        }
        game_metadata = {
            "analysis": "generic move-level extraction with explicit chance events",
            "chance_semantics": "after_ply counts player actions; ordered outcomes are replayed in Rust",
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
            "root_actions": _ROOT_ACTION_FIELDS,
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
        extract_match = None
        game_output_files = {}
        game_writer_fields = {}
        game_metadata = {
            "analysis": "generic move-level extraction",
        }

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
                "moves": _csv_writer(
                    stack,
                    temporary_dir / "moves.csv",
                    _fields_with_provenance(_GENERIC_MOVE_FIELDS),
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
            match_writers = [writers["matches"], writers["moves"]] + [
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
                        _extract_generic_match(context, writers, row_counts)
                        if extract_match is not None:
                            extract_match(context, writers, row_counts, game)
                        else:
                            _validate_generic_result(context, game)

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
                    "seat_mode": study.header.get(
                        "seat_mode",
                        "alternating" if study.study_type == "tournament" else "",
                    ),
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
            **({"game_params": game_parameters(game)} if game_parameters(game) else {}),
            "tournament_schema_version": 1,
            "analysis_schema_version": 9,
            "decision_timing_scope": timing_scope,
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
        "decision_timing_scope": timing_scope,
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

def _extract_generic_match(
    context: _MatchContext,
    writers: dict[str, _CsvWriter],
    row_counts: dict[str, int],
) -> None:
    """Normalize game-independent move and search metrics."""

    for ply, raw in enumerate(context.raw_moves, 1):
        move_context = f"match {context.match_number} ply {ply}"
        if not isinstance(raw, dict):
            raise TypeError(f"{move_context} must be an object")
        if _integer_field(raw, "ply", move_context) != ply:
            raise ValueError(f"{move_context} reports an inconsistent ply")
        player = _integer_field(raw, "player", move_context)
        if player not in (0, 1):
            raise ValueError(f"{move_context} player must be 0 or 1")
        action = raw.get("action")
        if not isinstance(action, dict):
            raise TypeError(f"{move_context} action must be an object")
        action_type = _string_field(action, "type", f"{move_context} action")
        action_kind = action.get("kind", action_type)
        if not isinstance(action_kind, str):
            raise TypeError(f"{move_context} action kind must be a string")
        root_actions = raw.get("root_actions")
        if root_actions is None:
            root_actions = []
        if not isinstance(root_actions, list):
            raise TypeError(f"{move_context} root_actions must be a list")
        tree_reuse = _trace_tree_reuse(raw.get("tree_reuse"), move_context)
        writers["moves"].writerow(
            {
                "match_number": context.match_number,
                "ply": ply,
                "total_plies": context.plies,
                "player": player,
                "agent": context.players[player],
                "outcome": _player_outcome(player, context.winner_player),
                "decision_seconds": raw.get("decision_seconds", ""),
                "selection_seconds": raw.get("selection_seconds", ""),
                "maintenance_seconds": raw.get("maintenance_seconds", ""),
                "search_iterations": raw.get("search_iterations", ""),
                "search_nodes": raw.get("search_nodes", ""),
                "terminal_simulations": raw.get("terminal_simulations", ""),
                "cutoff_simulations": raw.get("cutoff_simulations", ""),
                **_tree_reuse_row(tree_reuse),
                "progress_fraction": ply / context.plies,
                "game_quarter": _game_quarter(ply, context.plies),
                "action_type": action_type,
                "action_kind": action_kind,
                "action_json": json.dumps(
                    action,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "root_action_count": len(root_actions),
                "root_actions_json": json.dumps(
                    root_actions,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "terminal_after": ply == context.plies,
            }
        )
        row_counts["moves"] += 1

def _extract_chance_events(context, writers, row_counts, game):
    """Preserve explicit environment outcomes separately from player move metrics."""
    from ..matches.trace import _validate_splendor_result
    _validate_splendor_result(context.raw_result)
    for index, event in enumerate(context.raw_result["chance_events"], 1):
        writers["chance_events"].writerow({
            "match_number": context.match_number,
            "event_index": index,
            "after_ply": event["after_ply"],
            "outcome_json": json.dumps(event["outcome"], sort_keys=True, separators=(",", ":")),
        })
        row_counts["chance_events"] += 1

def _validate_generic_result(context: _MatchContext, game: ConnectFour | TicTacToe) -> None:
    """Convert trace data; Rust owns legality, turn order and terminal outcomes."""
    if isinstance(game, Connect6) and context.raw_result.get("game_params") != game_parameters(game):
        raise ValueError("Connect6 result game_params differ from tournament configuration")
    moves = []
    for ply, raw in enumerate(context.raw_moves, 1):
        where = f"match {context.match_number} ply {ply}"
        action = raw["action"]  # Structure and player were checked by generic extraction.
        try:
            expected_type = "connect6" if isinstance(game, Connect6) else "connect_four" if isinstance(game, ConnectFour) else "tic_tac_toe"
            if action["type"] != expected_type:
                raise ValueError(f"expected a {expected_type} action")
            if isinstance(game, Connect6):
                moves.append(Move(raw["player"], Connect6Action(_integer_field(action, "position", where))))
                continue
            column = _integer_field(action, "column", where)
            typed_action = (
                ConnectFourAction(column)
                if isinstance(game, ConnectFour)
                else TicTacToeAction(_integer_field(action, "row", where), column)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{where}: {error}") from error
        moves.append(Move(player=raw["player"], action=typed_action))
    try:
        analysis = _analyze_trace(game, tuple(moves), seed=context.seed)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"match {context.match_number}: {error}") from error
    if analysis["winner"] != context.winner_player:
        raise ValueError(f"match {context.match_number} replay winner does not match its result")
    utilities = context.raw_result["utilities"]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in utilities):
        raise ValueError(f"match {context.match_number} utilities must be numeric")
    if list(analysis["utilities"]) != utilities:
        raise ValueError(f"match {context.match_number} replay utilities do not match its result")
