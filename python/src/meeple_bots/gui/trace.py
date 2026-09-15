"""Extractable JSONL traces shared by graphical interfaces."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ..api import MatchResult
from ..serialization import match_result_dict
from .player import GuiPlayer


def write_gui_trace(
    output_dir: Path,
    *,
    game: str,
    max_plies: int,
    result: MatchResult,
    players: tuple[GuiPlayer, GuiPlayer],
    duration_seconds: float,
) -> Path:
    """Atomically write one completed GUI match in the tournament trace schema."""

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = output_dir / f"{game}-gui-{timestamp}-{uuid4().hex[:8]}.jsonl"
    temporary = target.with_suffix(".jsonl.tmp")
    names = tuple(_player_name(index, player) for index, player in enumerate(players))
    winner = None if result.winner is None else (
        "agent_a" if result.winner == 0 else "agent_b"
    )
    header = {
        "record_type": "tournament",
        "schema_version": 1,
        "decision_timing_scope": "agent_total_v1",
        "study_type": "batch",
        "game": game,
        **({"game_params": result.game_params} if result.game_params else {}),
        "output": str(target),
        "matches_per_pair": 1,
        "seed": result.seed,
        "max_plies": max_plies,
        "workers": 1,
        "total_pairings": 1,
        "total_matches": 1,
        "agents": [
            _agent_dict(name, player)
            for name, player in zip(names, players, strict=True)
        ],
    }
    match = {
        "record_type": "match",
        "match_number": 1,
        "pairing_number": 1,
        "pairing_match_number": 1,
        "agent_a": names[0],
        "agent_b": names[1],
        "self_play": False,
        "agent_a_player": 0,
        "players": list(names),
        "winner": winner,
        "duration_seconds": duration_seconds,
        "result": match_result_dict(result),
    }
    try:
        with temporary.open("x", encoding="utf-8") as output:
            for record in (header, match):
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _player_name(index: int, player: GuiPlayer) -> str:
    if player.kind != "mcts":
        return f"gui-player-{index + 1}-{player.kind}"
    budget = (
        f"t{player.time_budget:g}"
        if player.time_budget is not None
        else f"i{player.iterations}"
    )
    heuristic = "none" if player.heuristic is None else str(player.heuristic)
    reuse = "reuse" if player.tree_reuse else "fresh"
    return (
        f"gui-player-{index + 1}-mcts-{budget}-d{player.rollout_depth}"
        f"-c{player.exploration:g}-h{heuristic}-{reuse}-{player.selection_policy}"
    )


def _agent_dict(name: str, player: GuiPlayer) -> dict[str, object]:
    if player.kind != "mcts":
        return {"name": name, "type": player.kind, "self_play": False}
    evaluator = (
        {"kind": "neutral"}
        if player.heuristic is None
        else {"kind": "game_heuristic", "index": player.heuristic}
    )
    return {
        "name": name,
        "type": "mcts",
        "iterations": player.iterations,
        "time_budget": player.time_budget,
        "rollout_depth": player.rollout_depth,
        "exploration": player.exploration,
        "selection_policy": player.selection_policy,
        "progressive_widening": player.progressive_widening,
        "progressive_widening_k": player.progressive_widening_k,
        "progressive_widening_alpha": player.progressive_widening_alpha,
        "progressive_widening_expansion": player.progressive_widening_expansion,

        **({"rave_equivalence": player.rave_equivalence} if player.selection_policy == "uct_rave" else {}),
        "heuristic": player.heuristic,
        "cutoff_evaluator": evaluator,
        "rollout_policy": "uniform_random",
        "rollout_evaluator": None,
        "rollout_epsilon": None,
        "root_diagnostics": False,
        "tree_reuse": player.tree_reuse,
        "self_play": False,
    }
