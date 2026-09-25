"""Versioned trace and analysis-table schema, codecs, and CSV writer."""

from __future__ import annotations

import csv
import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from ..matches.models import RootActionDiagnostic, TreeReuseDiagnostic

_COMMON_OUTPUT_FILES = {
    "agents": "agents.csv",
    "studies": "studies.csv",
    "matches": "matches.csv",
    "moves": "moves.csv",
    "manifest": "manifest.json",
}

_AGENT_FIELDS = (
    "agent_name",
    "kind",
    "config_json",
    "iterations",
    "time_budget",
    "rollout_depth",
    "exploration",
    "selection_policy",
    "rave_equivalence",
    "progressive_widening",
    "progressive_widening_k",
    "progressive_widening_alpha",
    "progressive_widening_expansion",
    "heuristic",
    "cutoff_evaluator",
    "cutoff_heuristic",
    "rollout_policy",
    "rollout_evaluator",
    "rollout_heuristic",
    "rollout_epsilon",
    "rollout_condition",
    "rollout_condition_phase",
    "rollout_primary_policy",
    "rollout_fallback_policy",
    "rollout_fallback_evaluator",
    "rollout_fallback_heuristic",
    "rollout_fallback_epsilon",
    "progressive_bias_weight",
    "progressive_bias_evaluator",
    "progressive_bias_heuristic",
    "progressive_bias_condition",
    "progressive_bias_condition_phase",
    "root_diagnostics",
    "tree_reuse",
    "transpositions",
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
    "seat_mode",
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

_TREE_REUSE_FIELDS = (
    "reuse_transition_attempts",
    "reuse_transition_hits",
    "reuse_transition_misses",
    "reuse_own_action_hits",
    "reuse_opponent_action_hits",
    "reused_root_visits",
    "reused_nodes",
    "pruned_nodes",
    "reuse_resets",
)

_GENERIC_MOVE_FIELDS = (
    "match_number",
    "ply",
    "total_plies",
    "player",
    "agent",
    "outcome",
    "decision_seconds",
    "selection_seconds",
    "maintenance_seconds",
    "search_iterations",
    "search_nodes",
    "terminal_simulations",
    "cutoff_simulations",
    *_TREE_REUSE_FIELDS,
    "progress_fraction",
    "game_quarter",
    "action_type",
    "action_kind",
    "action_json",
    "root_action_count",
    "root_actions_json",
    "terminal_after",
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

def _fields_with_provenance(fields: tuple[str, ...]) -> tuple[str, ...]:
    if not fields or fields[0] != "match_number":
        raise ValueError("match analysis tables must begin with match_number")
    return (fields[0], *_PROVENANCE_FIELDS, *fields[1:])

def _game_quarter(ply: int, total_plies: int) -> str:
    return f"q{min(3, ((ply - 1) * 4) // total_plies) + 1}"

def _player_outcome(player: int, winner_player: int | None) -> str:
    if winner_player is None:
        return "draw"
    return "win" if player == winner_player else "loss"

def _tree_reuse_row(diagnostic: TreeReuseDiagnostic | None) -> dict[str, object]:
    if diagnostic is None:
        return {field: "" for field in _TREE_REUSE_FIELDS}
    return {
        "reuse_transition_attempts": diagnostic.transition_attempts,
        "reuse_transition_hits": diagnostic.transition_hits,
        "reuse_transition_misses": diagnostic.transition_misses,
        "reuse_own_action_hits": diagnostic.own_action_hits,
        "reuse_opponent_action_hits": diagnostic.opponent_action_hits,
        "reused_root_visits": diagnostic.reused_root_visits,
        "reused_nodes": diagnostic.reused_nodes,
        "pruned_nodes": diagnostic.pruned_nodes,
        "reuse_resets": diagnostic.resets,
    }

def _trace_tree_reuse(raw: object, context: str) -> TreeReuseDiagnostic | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError(f"{context} tree_reuse must be an object or null")
    return TreeReuseDiagnostic(
        transition_attempts=_integer_field(raw, "transition_attempts", context),
        transition_hits=_integer_field(raw, "transition_hits", context),
        transition_misses=_integer_field(raw, "transition_misses", context),
        own_action_hits=_integer_field(raw, "own_action_hits", context),
        opponent_action_hits=_integer_field(raw, "opponent_action_hits", context),
        reused_root_visits=_integer_field(raw, "reused_root_visits", context),
        reused_nodes=_integer_field(raw, "reused_nodes", context),
        pruned_nodes=_integer_field(raw, "pruned_nodes", context),
        resets=_integer_field(raw, "resets", context),
    )

def _trace_root_actions(raw: object, context: str) -> tuple[RootActionDiagnostic, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise TypeError(f"{context} root_actions must be a list")
    diagnostics = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"{context} root action {index} must be an object")
        diagnostics.append(
            RootActionDiagnostic(
                action_index=_integer_field(item, "action_index", context),
                visits=_integer_field(item, "visits", context),
                mean_utility=float(item["mean_utility"]),
                heuristic_value=(
                    None
                    if item.get("heuristic_value") is None
                    else float(item["heuristic_value"])
                ),
                progressive_bias=(
                    None
                    if item.get("progressive_bias") is None
                    else float(item["progressive_bias"])
                ),
                selected=bool(item.get("selected", False)),
            )
        )
    return tuple(diagnostics)

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
    condition_kind = ""
    condition_phase = ""
    condition = raw.get("rollout_condition")
    if condition is not None:
        if not isinstance(condition, dict):
            raise TypeError("tournament agent rollout_condition must be an object")
        condition_kind = _string_field(
            condition, "kind", "tournament rollout condition"
        )
        condition_phase = _string_field(
            condition, "phase", "tournament rollout condition"
        )
    fallback_kind, fallback_heuristic = _serialized_evaluator(
        raw.get("rollout_fallback_evaluator"),
        fallback_heuristic=None,
        context="tournament fallback rollout evaluator",
        allow_missing=True,
    )
    bias_kind, bias_heuristic = _serialized_evaluator(
        raw.get("progressive_bias_evaluator"),
        fallback_heuristic=raw.get("progressive_bias_heuristic"),
        context="tournament progressive bias evaluator",
        allow_missing=True,
    )
    return {
        "agent_name": _string_field(raw, "name", "tournament agent"),
        "kind": kind,
        "config_json": json.dumps(
            {key: value for key, value in raw.items() if key not in {"name", "self_play"}},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        "iterations": raw.get("iterations", ""),
        "time_budget": raw.get("time_budget", ""),
        "rollout_depth": raw.get("rollout_depth", ""),
        "exploration": raw.get("exploration", ""),
        "selection_policy": raw.get("selection_policy", "uct") if raw.get("type") == "mcts" else "",
        "rave_equivalence": raw.get("rave_equivalence", 1000) if raw.get("selection_policy") == "uct_rave" else "",
        "progressive_widening": raw.get("progressive_widening", False) if raw.get("type") == "mcts" else "",
        "progressive_widening_k": raw.get("progressive_widening_k", 1.5) if raw.get("progressive_widening") else "",
        "progressive_widening_alpha": raw.get("progressive_widening_alpha", 0.5) if raw.get("progressive_widening") else "",
        "progressive_widening_expansion": raw.get("progressive_widening_expansion", "random") if raw.get("progressive_widening") else "",
        "heuristic": "" if raw.get("heuristic") is None else raw["heuristic"],
        "cutoff_evaluator": cutoff_kind,
        "cutoff_heuristic": cutoff_heuristic,
        "rollout_policy": rollout_policy,
        "rollout_evaluator": rollout_kind,
        "rollout_heuristic": rollout_heuristic,
        "rollout_epsilon": (
            "" if raw.get("rollout_epsilon") is None else raw["rollout_epsilon"]
        ),
        "rollout_condition": condition_kind,
        "rollout_condition_phase": condition_phase,
        "rollout_primary_policy": raw.get("rollout_primary_policy") or "",
        "rollout_fallback_policy": raw.get("rollout_fallback_policy") or "",
        "rollout_fallback_evaluator": fallback_kind,
        "rollout_fallback_heuristic": fallback_heuristic,
        "rollout_fallback_epsilon": (
            ""
            if raw.get("rollout_fallback_epsilon") is None
            else raw["rollout_fallback_epsilon"]
        ),
        "progressive_bias_weight": (
            ""
            if raw.get("progressive_bias_weight") is None
            else raw["progressive_bias_weight"]
        ),
        "progressive_bias_evaluator": bias_kind,
        "progressive_bias_heuristic": bias_heuristic,
        "progressive_bias_condition": raw.get("progressive_bias_condition") or "",
        "progressive_bias_condition_phase": (
            raw.get("progressive_bias_condition_phase") or ""
        ),
        "root_diagnostics": bool(raw.get("root_diagnostics", False)),
        "tree_reuse": bool(raw.get("tree_reuse", False)),
        "transpositions": bool(raw.get("transpositions", False)),
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
    if game not in {"boop", "connect-four", "spotf", "tic-tac-toe", "splendor", "connect6"}:
        raise ValueError(f"unknown tournament game: {game}")
    return game

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
