"""HTTP-facing application adapter for the Connect Four GUI."""

from __future__ import annotations

import threading
from typing import Any

from ....gui.player import parse_gui_player
from .controller import ConnectFourGui


class ConnectFourApplication:
    """Own the current Connect Four match and validate browser commands."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._game = ConnectFourGui()

    def start(self, payload: dict[str, Any]) -> dict[str, object]:
        first = parse_gui_player(
            payload.get("first"), "first", default_rollout_depth=64
        )
        second = parse_gui_player(
            payload.get("second"), "second", default_rollout_depth=64
        )
        seed = payload.get("seed", 0)
        delay = payload.get("minimum_move_seconds", 0.6)
        with self._lock:
            previous = self._game
            self._game = ConnectFourGui()
            previous.cancel()
            self._game.start(
                first,
                second,
                seed=seed,
                minimum_move_seconds=delay,
            )
            return self._game.snapshot()

    def move(self, payload: dict[str, Any]) -> dict[str, object]:
        column = payload.get("column")
        if isinstance(column, bool) or not isinstance(column, int):
            raise ValueError("column must be an integer")
        with self._lock:
            game = self._game
        game.submit_move(column)
        return game.snapshot()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            game = self._game
        return game.snapshot()

    def cancel(self) -> None:
        with self._lock:
            self._game.cancel()
