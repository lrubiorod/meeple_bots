"""HTTP-facing application adapter for the Boop GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ....gui.player import parse_gui_player
from ...._capabilities import heuristic_indices
from ....gui.application import GuiApplication
from .controller import BoopGui


class BoopApplication(GuiApplication):
    """Own the current Boop match and validate browser commands."""

    def __init__(self, trace_dir: Path = Path("results/gui/boop")) -> None:
        super().__init__(BoopGui, trace_dir)

    def start(self, payload: dict[str, Any]) -> dict[str, object]:
        first = parse_gui_player(
            payload.get("first"),
            "first",
            default_rollout_depth=15,
            available_heuristics=heuristic_indices("boop"),
        )
        second = parse_gui_player(
            payload.get("second"),
            "second",
            default_rollout_depth=15,
            available_heuristics=heuristic_indices("boop"),
        )
        seed = payload.get("seed", 0)
        delay = payload.get("minimum_move_seconds", 0.6)
        return self._start_match(
            first,
            second,
            seed=seed,
            minimum_move_seconds=delay,
            save_trace=payload.get("save_trace", False),
        )

    def move(self, payload: dict[str, Any]) -> dict[str, object]:
        action = payload.get("action")
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError("action must be an integer")
        with self._lock:
            game = self._game
        game.submit_move(action)
        return game.snapshot()
