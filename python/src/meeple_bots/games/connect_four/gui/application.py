"""HTTP-facing application adapter for the Connect Four GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ....gui.player import parse_gui_player
from ....gui.application import GuiApplication
from .controller import ConnectFourGui


class ConnectFourApplication(GuiApplication):
    """Own the current Connect Four match and validate browser commands."""

    def __init__(self, trace_dir: Path = Path("results/gui/connect-four")) -> None:
        super().__init__(ConnectFourGui, trace_dir)

    def start(self, payload: dict[str, Any]) -> dict[str, object]:
        first = parse_gui_player(
            payload.get("first"), "first", default_rollout_depth=64
        )
        second = parse_gui_player(
            payload.get("second"), "second", default_rollout_depth=64
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
        column = payload.get("column")
        if isinstance(column, bool) or not isinstance(column, int):
            raise ValueError("column must be an integer")
        with self._lock:
            game = self._game
        game.submit_move(column)
        return game.snapshot()
