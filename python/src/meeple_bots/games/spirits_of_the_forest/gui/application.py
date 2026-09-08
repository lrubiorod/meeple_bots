"""HTTP-facing adapter for the Spirits of the Forest GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ....gui.baselines import SPOTF_BASELINE
from ....gui.player import parse_gui_player
from ...._capabilities import heuristic_indices
from ....gui.application import GuiApplication
from .controller import SpiritsOfTheForestGui


class SpiritsOfTheForestApplication(GuiApplication):
    def __init__(self, trace_dir: Path = Path("results/gui/spotf")) -> None:
        super().__init__(SpiritsOfTheForestGui, trace_dir)

    def start(self, payload: dict[str, Any]) -> dict[str, object]:
        first = parse_gui_player(
            payload.get("first"),
            "first",
            default_rollout_depth=64,
            default_mcts=SPOTF_BASELINE,
            available_heuristics=heuristic_indices("spotf"),
        )
        second = parse_gui_player(
            payload.get("second"),
            "second",
            default_rollout_depth=64,
            default_mcts=SPOTF_BASELINE,
            available_heuristics=heuristic_indices("spotf"),
        )
        return self._start_match(
            first,
            second,
            seed=payload.get("seed", 0),
            minimum_move_seconds=payload.get("minimum_move_seconds", 0.4),
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
