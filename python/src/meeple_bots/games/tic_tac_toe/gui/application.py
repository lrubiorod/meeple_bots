"""HTTP-facing application adapter for the tic-tac-toe GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ....gui.player import parse_gui_player
from ...._capabilities import heuristic_indices
from ....gui.application import GuiApplication
from .controller import TicTacToeGui


class TicTacToeApplication(GuiApplication):
    """Own the current tic-tac-toe match and validate browser commands."""

    def __init__(self, trace_dir: Path = Path("results/gui/tic-tac-toe")) -> None:
        super().__init__(TicTacToeGui, trace_dir)

    def start(self, payload: dict[str, Any]) -> dict[str, object]:
        first = parse_gui_player(
            payload.get("first"), "first", default_rollout_depth=9,
            available_heuristics=heuristic_indices("tic-tac-toe"),
        )
        second = parse_gui_player(
            payload.get("second"), "second", default_rollout_depth=9,
            available_heuristics=heuristic_indices("tic-tac-toe"),
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
        row = payload.get("row")
        column = payload.get("column")
        if isinstance(row, bool) or not isinstance(row, int):
            raise ValueError("row must be an integer")
        if isinstance(column, bool) or not isinstance(column, int):
            raise ValueError("column must be an integer")
        with self._lock:
            game = self._game
        game.submit_move(row, column)
        return game.snapshot()
