"""Splendor browser interface."""
from ....gui.application import GuiApplication
from ....gui.player import parse_gui_player
from ....gui.player import GuiPlayer
from .controller import SplendorGui
from .page import PAGE
from pathlib import Path


class SplendorApplication(GuiApplication):
    def __init__(self, trace_dir=Path("results/gui/splendor")):
        super().__init__(SplendorGui, trace_dir)

    def start(self, payload):
        players = [parse_gui_player(payload.get(name), name, default_rollout_depth=64,
                                   available_heuristics=(), default_mcts=GuiPlayer("mcts", iterations=256, rollout_depth=64), with_policies=True)
                   for name in ("first", "second")]
        return self._start_match(*players, seed=payload.get("seed", 0),
                                 minimum_move_seconds=payload.get("minimum_move_seconds", 0.4),
                                 save_trace=payload.get("save_trace", False))

    def move(self, payload):
        with self._lock:
            game = self._game
        game.submit_move(payload.get("action"), payload.get("turn"), payload.get("session_id"))
        return game.snapshot()
