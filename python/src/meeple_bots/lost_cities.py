"""Lost Cities single-round experimental variant. Rust owns rules and hidden sampling.

State/position and match traces are authoritative administrative data, never agent
observations. Observation is a frozen, serializable value without a position handle.
Cards are (color index, value), with value 0 for wagers; colors red/green/blue/yellow/white.
"""
from dataclasses import dataclass, asdict, field
from . import _native


def _so_ismcts_search(
    observation, legal_actions, iterations, exploration, seed,
    time_budget, selection_policy, tree_reuse,
):
    """Adapt an observation-only Lost Cities search request to the native API."""
    if not isinstance(observation, LostCitiesObservation):
        raise TypeError("search requires a LostCitiesObservation")
    result = _native.lost_cities_so_ismcts_search(
        observation.to_dict(), [action.to_dict() for action in legal_actions],
        iterations, exploration, seed, time_budget, selection_policy, tree_reuse,
    )
    result["action"] = LostCitiesAction.from_dict(result["action"])
    return result


@dataclass(frozen=True, slots=True)
class LostCitiesAction:
    kind: str
    card: tuple[int, int] | None = None
    color: int | None = None

    def to_dict(self):
        return {'type': 'lost_cities', 'kind': self.kind,
                **({'card': list(self.card)} if self.card is not None else {}),
                **({'color': self.color} if self.color is not None else {})}

    @classmethod
    def from_dict(cls, raw):
        if raw['type'] != 'lost_cities':
            raise ValueError('expected Lost Cities action')
        return cls(raw['kind'], tuple(raw['card']) if 'card' in raw else None, raw.get('color'))


@dataclass(frozen=True, slots=True)
class LostCitiesObservation:
    observer: int
    current_player: int
    phase: str
    dealt: int | None
    hand: tuple
    opponent_hand_size: int
    deck_size: int
    expeditions: tuple
    discards: tuple
    blocked_discard: int | None

    @classmethod
    def from_dict(cls, raw):
        return cls(**{**raw, 'hand': tuple(map(tuple, raw['hand'])),
            'expeditions': tuple(tuple(tuple(map(tuple, c)) for c in p) for p in raw['expeditions']),
            'discards': tuple(tuple(map(tuple, c)) for c in raw['discards'])})

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LostCitiesState:
    current_player: int
    phase: str
    dealt: int | None
    hands: tuple
    deck: tuple
    expeditions: tuple
    discards: tuple
    blocked_discard: int | None
    scores: tuple[int, int]
    status: str
    _position: object = field(default=None, repr=False, compare=False)

    @classmethod
    def from_dict(cls, raw, position=None):
        return cls(**{**raw, 'hands': tuple(tuple(map(tuple, h)) for h in raw['hands']),
            'deck': tuple(map(tuple, raw['deck'])), 'scores': tuple(raw['scores']),
            'expeditions': tuple(tuple(tuple(map(tuple, c)) for c in p) for p in raw['expeditions']),
            'discards': tuple(tuple(map(tuple, c)) for c in raw['discards'])}, _position=position)

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if name != '_position'}

    def _native_position(self):
        if self._position is None:
            raise ValueError('summary state has no executable position')
        return self._position


@dataclass(frozen=True, slots=True)
class LostCitiesChanceEvent:
    after_ply: int
    outcome: LostCitiesAction


@dataclass(frozen=True, slots=True)
class LostCities:
    def initial_state(self, seed=0):
        position = _native.LostCitiesPosition(seed)
        return LostCitiesState.from_dict(position.snapshot(), position)

    def observation(self, state, observer):
        return LostCitiesObservation.from_dict(state._native_position().observation(observer))

    def sample_determinization(self, observation, observer, seed=0):
        position = _native.LostCitiesPosition.determinize(observation.to_dict(), observer, seed)
        return LostCitiesSimulationWorld._from_native(position)

    def legal_actions(self, state):
        return tuple(LostCitiesAction.from_dict(a) for a in state._native_position().legal_actions())

    def apply_action(self, state, action):
        position = state._native_position().apply_action(action.to_dict())
        return LostCitiesState.from_dict(position.snapshot(), position)

    def apply_chance_outcome(self, state, outcome):
        position = state._native_position().apply_chance_outcome(outcome.to_dict())
        return LostCitiesState.from_dict(position.snapshot(), position)

    def chance_outcomes(self, state):
        return tuple((LostCitiesAction.from_dict(a), p) for a, p in state._native_position().chance_outcomes())

    def sample_chance(self, state, seed=0):
        return LostCitiesAction.from_dict(state._native_position().sample_chance(seed))

    def terminal_utility(self, state, player):
        if player not in (0, 1):
            raise ValueError('player must be 0 or 1')
        if state.status != 'terminal':
            return None
        a, b = state.scores[player], state.scores[1-player]
        return float((a > b) - (a < b))


@dataclass(frozen=True, slots=True)
class LostCitiesSimulationWorld:
    """Temporary complete world; never an information-set key or an agent observation.

    state is an administrative snapshot without an executable environment handle.
    deck_order[0] is the next card. Simulation transitions never resample draws.
    """
    state: LostCitiesState
    deck_order: tuple
    _world: object = field(repr=False, compare=False)

    @classmethod
    def _from_native(cls, world):
        return cls(LostCitiesState.from_dict(world.snapshot()), tuple(map(tuple, world.deck_order())), world)

    def observation(self, observer):
        return LostCitiesObservation.from_dict(self._world.observation(observer))

    def legal_actions(self):
        return tuple(LostCitiesAction.from_dict(a) for a in self._world.legal_actions())

    def apply_action(self, action):
        return self._from_native(self._world.apply_action(action.to_dict()))

    def resolve_pending_draws(self):
        return self._from_native(self._world.resolve_pending_draws())

    def validate(self):
        self._world.validate()
