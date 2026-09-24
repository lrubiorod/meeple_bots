"""Executable perfect-information SPOTF positions using authoritative Rust rules."""
from dataclasses import dataclass, field
from . import _native
from .api import (SpiritsOfTheForest, _spirits_action_from_native,
                  _spirits_state_from_native, _native_agent)


@dataclass(frozen=True)
class SpotfPosition:
    seed: int
    action_indices: tuple[int, ...] = ()
    _position: object = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if self._position is None:
            native = _native.SpotfPosition(self.seed)
            for index in self.action_indices:
                native = native.apply(index)
            object.__setattr__(self, '_position', native)

    @property
    def state(self):
        return _spirits_state_from_native(self._position.snapshot())

    @property
    def root_player(self):
        return self.state[4]

    def legal_actions(self):
        return tuple(_spirits_action_from_native(a) for a in self._position.legal_actions())

    def apply(self, action):
        index = self.legal_actions().index(action)
        return SpotfPosition(self.seed, (*self.action_indices, index), self._position.apply(index))

    def to_dict(self):
        # The full position is legitimate information in this perfect-information game.
        return {'seed': self.seed, 'action_indices': self.action_indices,
                'state': self._position.snapshot()}

    def search(self, agent, *, seed):
        raw = self._position.search(_native_agent(agent, SpiritsOfTheForest()), seed)
        legal = self.legal_actions()
        edges = {e['action_index']: e for e in raw['root_actions']}
        return {'action': _spirits_action_from_native(raw['action']),
                'root_visits': raw['completed_iterations'],
                'root_actions': [{'action': action, **edges.get(i, {'visits': 0, 'q': 0., 'heuristic_value': None, 'progressive_bias': None})} for i, action in enumerate(legal)],
                'diagnostics': {k: v for k, v in raw.items() if k not in ('action', 'root_actions')}}
