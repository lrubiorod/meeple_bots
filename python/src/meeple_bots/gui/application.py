"""Atomic replacement of the active browser controller."""

import threading
from pathlib import Path

from .controller import GuiController
from .player import GuiPlayer


class GuiApplication:
    def __init__(self, controller: type[GuiController], trace_dir: Path) -> None:
        self._lock = threading.Lock()
        self._controller = controller
        self._trace_dir = trace_dir
        self._game = controller(trace_dir=trace_dir)

    def _start_match(
        self, first: GuiPlayer, second: GuiPlayer, *, seed: int,
        minimum_move_seconds: float, save_trace: bool,
    ) -> dict[str, object]:
        with self._lock:
            candidate = self._controller(trace_dir=self._trace_dir)
            candidate.start(first, second, seed=seed,
                            minimum_move_seconds=minimum_move_seconds, save_trace=save_trace)
            self._game.cancel()
            self._game = candidate
            return candidate.snapshot()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            game = self._game
        return game.snapshot()

    def cancel(self) -> None:
        with self._lock:
            self._game.cancel()
