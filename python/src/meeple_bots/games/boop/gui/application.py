"""HTTP-facing application adapter for the Boop GUI."""

from __future__ import annotations

import threading
from typing import Any

from ....gui.player import parse_gui_player
from .controller import BoopGui


class BoopApplication:
    """Own the current Boop match and validate browser commands."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._game = BoopGui()

    def start(self, payload: dict[str, Any]) -> dict[str, object]:
        first = parse_gui_player(
            payload.get("first"),
            "first",
            default_rollout_depth=15,
            available_heuristics=(0, 1),
        )
        second = parse_gui_player(
            payload.get("second"),
            "second",
            default_rollout_depth=15,
            available_heuristics=(0, 1),
        )
        seed = payload.get("seed", 0)
        delay = payload.get("minimum_move_seconds", 0.6)
        with self._lock:
            previous = self._game
            self._game = BoopGui()
            previous.cancel()
            self._game.start(
                first,
                second,
                seed=seed,
                minimum_move_seconds=delay,
            )
            return self._game.snapshot()

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
