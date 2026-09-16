"""Open-hand Lost Cities debugging interface."""
from pathlib import Path
from ....gui.application import GuiApplication
from ....gui.player import GuiPlayer
from .controller import LostCitiesGui
from .page import PAGE


class LostCitiesApplication(GuiApplication):
    def __init__(self, trace_dir=Path('results/gui/lost_cities')):
        super().__init__(LostCitiesGui, trace_dir)

    def start(self, payload):
        players = []
        for name, default in (('first', 'human'), ('second', 'random')):
            raw = payload.get(name, {'kind': default})
            if not isinstance(raw, dict) or raw.get('kind') not in ('human', 'random'):
                raise ValueError('Lost Cities GUI supports human and random players only')
            players.append(GuiPlayer(raw['kind']))
        return self._start_match(*players, seed=payload.get('seed', 0),
            minimum_move_seconds=payload.get('minimum_move_seconds', .4),
            save_trace=payload.get('save_trace', False))

    def move(self, payload):
        with self._lock:
            game = self._game
        game.submit_move(payload.get('action'), payload.get('turn'), payload.get('session_id'))
        return game.snapshot()
