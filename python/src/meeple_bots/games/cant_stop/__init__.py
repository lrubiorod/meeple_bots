"""Public-chance Can't Stop sessions backed by authoritative Rust rules."""

from ... import _native
from ...api import MctsAgent, RandomAgent, _native_agent


class CantStopSession:
    """Advance one player action or dice event at a time; None seats are human.

    Snapshots contain only public state. Agent search and actual dice use separate RNGs.
    Human actions are indices into the current snapshot's legal_actions list.
    """

    def __init__(self, *, seed=0, first=None, second=None):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        def native(agent):
            if agent is None:
                return None
            if not isinstance(agent, (MctsAgent, RandomAgent)):
                raise TypeError("session seats must be None, MctsAgent or RandomAgent")
            # Native construction validates Can't Stop capabilities; conversion itself is generic.
            return _native_agent(agent, game=None)
        self._session = _native.CantStopSession(seed, native(first), native(second))

    def snapshot(self):
        return self._session.snapshot()

    def step(self, action=None):
        if action is not None and (isinstance(action, bool) or not isinstance(action, int) or action < 0):
            raise ValueError("action must be a non-negative integer")
        return self._session.step(action)
