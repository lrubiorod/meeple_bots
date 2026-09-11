"""Can't Stop browser interface."""
from ....gui.application import GuiApplication
from ....gui.player import parse_gui_player
from ....gui.baselines import CANT_STOP_BASELINE
from .controller import CantStopGui
from .page import PAGE
from pathlib import Path


class CantStopApplication(GuiApplication):
    def __init__(self, trace_dir=Path("results/gui/cant-stop")):
        super().__init__(CantStopGui, trace_dir)

    def start(self, payload):
        players = [parse_gui_player(payload.get(name), name, default_rollout_depth=100,
                                   available_heuristics=(0,), default_mcts=CANT_STOP_BASELINE, with_policies=True)
                   for name in ("first", "second")]
        return self._start_match(*players, seed=payload.get("seed", 0),
                                 minimum_move_seconds=payload.get("minimum_move_seconds", 0.4),
                                 save_trace=payload.get("save_trace", False))

    def move(self, payload):
        with self._lock:
            game = self._game
        game.submit_move(payload.get("action"), payload.get("turn"))
        return game.snapshot()
