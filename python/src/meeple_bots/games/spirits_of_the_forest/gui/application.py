"""HTTP-facing adapter for the Spirits of the Forest GUI."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from ....gui.player import parse_gui_player
from .controller import SpiritsOfTheForestGui


class SpiritsOfTheForestApplication:
    def __init__(self, trace_dir: Path = Path("results/gui/spotf")) -> None:
        self._lock = threading.Lock()
        self._trace_dir = trace_dir
        self._game = SpiritsOfTheForestGui(trace_dir=trace_dir)

    def start(self, payload: dict[str, Any]) -> dict[str, object]:
        first = parse_gui_player(
            payload.get("first"),
            "first",
            default_rollout_depth=64,
            available_heuristics=(0,),
        )
        second = parse_gui_player(
            payload.get("second"),
            "second",
            default_rollout_depth=64,
            available_heuristics=(0,),
        )
        with self._lock:
            previous = self._game
            candidate = SpiritsOfTheForestGui(trace_dir=self._trace_dir)
            candidate.start(
                first,
                second,
                seed=payload.get("seed", 0),
                minimum_move_seconds=payload.get("minimum_move_seconds", 0.4),
                save_trace=payload.get("save_trace", False),
            )
            previous.cancel()
            self._game = candidate
            return candidate.snapshot()

    def move(self, payload: dict[str, Any]) -> dict[str, object]:
        action = payload.get("action")
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError("action must be an integer")
        with self._lock:
            game = self._game
        game.submit_move(action)
        return game.snapshot()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            game = self._game
        return game.snapshot()

    def cancel(self) -> None:
        with self._lock:
            self._game.cancel()
