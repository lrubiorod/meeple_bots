"""Connect6 browser application."""
from pathlib import Path

from ....gui.application import GuiApplication
from ....gui.baselines import CONNECT6_BASELINE
from ....gui.player import parse_gui_player
from .controller import Connect6Gui
from .page import PAGE


class Connect6Application(GuiApplication):
    def __init__(self, trace_dir=Path('results/gui/connect6')):
        super().__init__(Connect6Gui, trace_dir)

    def start(self, payload):
        players = [parse_gui_player(payload.get(key), key,
                                   default_rollout_depth=CONNECT6_BASELINE.rollout_depth,
                                   default_mcts=CONNECT6_BASELINE, available_heuristics=())
                   for key in ('first','second')]
        with self._lock:
            candidate = Connect6Gui(self._trace_dir, payload.get('board_size'))
            candidate.start(*players, seed=payload.get('seed',0),
                            minimum_move_seconds=payload.get('minimum_move_seconds',0.4),
                            save_trace=payload.get('save_trace',False))
            self._game.cancel()
            self._game = candidate
            return candidate.snapshot()

    def move(self, payload):
        with self._lock:
            game = self._game
        game.submit_move(payload.get('position'), payload.get('turn'), payload.get('session_id'))
        return game.snapshot()
