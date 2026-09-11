"""Public Splendor types; all transitions are validated by Rust."""
from dataclasses import dataclass, field
from enum import Enum

from . import _native


class SplendorMoveKind(str, Enum):
    PASS = "pass"
    TAKE_DIFFERENT = "take_different"
    TAKE_SAME = "take_same"
    RESERVE_VISIBLE = "reserve_visible"
    BUY_VISIBLE = "buy_visible"
    BUY_RESERVED = "buy_reserved"


@dataclass(frozen=True, slots=True)
class SplendorAction:
    """A complete player decision; colors uses a W/B/G/R/K bit mask."""

    kind: SplendorMoveKind
    returned: tuple[int, ...] = (0,) * 6
    payment: tuple[int, ...] = (0,) * 6
    noble: int | None = None
    colors: int | None = None
    color: int | None = None
    tier: int | None = None
    slot: int | None = None
    index: int | None = None

    def to_dict(self):
        return {
            "type": "splendor", "kind": self.kind.value,
            "returned": list(self.returned), "payment": list(self.payment),
            "noble": self.noble,
            **{key: getattr(self, key) for key in ("colors", "color", "tier", "slot", "index")
               if getattr(self, key) is not None},
        }

    @classmethod
    def from_dict(cls, raw):
        return cls(
            kind=SplendorMoveKind(raw["kind"]), returned=tuple(raw["returned"]),
            payment=tuple(raw["payment"]), noble=raw["noble"],
            **{key: raw[key] for key in ("colors", "color", "tier", "slot", "index") if key in raw},
        )


@dataclass(frozen=True, slots=True)
class SplendorChanceOutcome:
    card: int

    def to_dict(self):
        return {"type": "splendor", "kind": "refill", "card": self.card}


@dataclass(frozen=True, slots=True)
class ChanceEvent:
    after_ply: int
    outcome: SplendorChanceOutcome


@dataclass(frozen=True, slots=True)
class SplendorPlayerState:
    tokens: tuple[int, ...]
    bonuses: tuple[int, ...]
    purchased: tuple[int, ...]
    reserved: tuple[int, ...]
    nobles: tuple[int, ...]
    prestige: int


@dataclass(frozen=True, slots=True)
class SplendorState:
    bank: tuple[int, ...]
    market: tuple[tuple[int | None, ...], ...]
    remaining: tuple[tuple[int, ...], ...]
    nobles: tuple[int, ...]
    players: tuple[SplendorPlayerState, ...]
    active_player: int
    pending_refill: tuple[int, int] | None
    finished: bool
    final_round: bool
    consecutive_passes: int
    status: str
    _position: object = field(default=None, repr=False, compare=False)

    @classmethod
    def from_dict(cls, raw, position=None):
        return cls(
            bank=tuple(raw["bank"]), market=tuple(map(tuple, raw["market"])),
            remaining=tuple(map(tuple, raw["remaining"])), nobles=tuple(raw["nobles"]),
            players=tuple(SplendorPlayerState(**{
                key: value if key == "prestige" else tuple(value) for key, value in player.items()
            }) for player in raw["players"]),
            active_player=raw["active_player"],
            pending_refill=None if raw["pending_refill"] is None else tuple(raw["pending_refill"]),
            finished=raw["finished"], final_round=raw["final_round"],
            consecutive_passes=raw["consecutive_passes"],
            status=raw["status"], _position=position,
        )

    def _native_position(self):
        if self._position is None:
            raise ValueError("this is a summary snapshot; use replay_splendor to recover a position")
        return self._position


@dataclass(frozen=True, slots=True)
class Splendor:
    """Two players, public visible reservations, no blind reservation."""

    def initial_state(self, seed=0):
        position = _native.SplendorPosition(seed)
        return SplendorState.from_dict(position.snapshot(), position)

    def legal_actions(self, state):
        return tuple(SplendorAction.from_dict(action)
                     for action in state._native_position().legal_actions())

    def chance_outcomes(self, state):
        return tuple((SplendorChanceOutcome(action["card"]), probability)
                     for action, probability in state._native_position().chance_outcomes())

    def apply_action(self, state, action):
        if not isinstance(action, SplendorAction):
            raise TypeError("expected a player action")
        return self._apply(state, action)

    def apply_chance_outcome(self, state, outcome):
        if not isinstance(outcome, SplendorChanceOutcome):
            raise TypeError("expected a chance outcome")
        return self._apply(state, outcome)

    def _apply(self, state, transition):
        position = state._native_position().apply(transition.to_dict())
        return SplendorState.from_dict(position.snapshot(), position)


def replay_splendor(seed, moves, chance_events):
    """Replay recorded events, never resample; reject missing or extra outcomes."""
    game = Splendor()
    state = game.initial_state(seed)
    events = iter(chance_events)
    event = next(events, None)
    for ply, move in enumerate(moves, 1):
        if state.active_player != move.player or state.status != "player_turn":
            raise ValueError("invalid Splendor player sequence")
        state = game.apply_action(state, move.action)
        while event is not None and event.after_ply == ply:
            state = game.apply_chance_outcome(state, event.outcome)
            event = next(events, None)
    if event is not None or state.status != "terminal":
        raise ValueError("incomplete Splendor replay or extra chance events")
    return state


class SplendorSession:
    """Interactive Rust session. None seats are human; chance is always automatic."""

    def __init__(self, *, seed=0, first=None, second=None):
        from .api import MctsAgent, RandomAgent, _native_agent
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")

        def native(agent):
            if agent is None:
                return None
            if not isinstance(agent, (MctsAgent, RandomAgent)):
                raise TypeError("session seats must be None, MctsAgent or RandomAgent")
            return _native_agent(agent, game=None)

        self._session = _native.SplendorSession(seed, native(first), native(second))

    def snapshot(self):
        return self._session.snapshot()

    def step(self, action=None):
        if action is not None and (isinstance(action, bool) or not isinstance(action, int) or action < 0):
            raise ValueError("action must be a non-negative integer")
        return self._session.step(action)
