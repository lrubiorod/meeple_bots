"""Connect6 public types; all rules and default configuration live in Rust."""

from __future__ import annotations

from dataclasses import dataclass

from . import _native


@dataclass(frozen=True, slots=True)
class Connect6Action:
    """Place one stone at a zero-based row-major cell index."""

    position: int

    def __post_init__(self):
        if type(self.position) is not int or self.position < 0:
            raise ValueError('position must be a non-negative integer')


@dataclass(frozen=True, slots=True, init=False)
class Connect6:
    board_size: int

    def __init__(self, board_size: int | None = None):
        from .game_config import normalize_game_parameters

        params = {} if board_size is None else {'board_size': board_size}
        object.__setattr__(self, 'board_size', normalize_game_parameters('connect6', params)['board_size'])

    def initial_state(self) -> Connect6State:
        return Connect6State(_native.Connect6Position(self.board_size))

    def legal_actions(self, state: Connect6State) -> tuple[Connect6Action, ...]:
        self._check(state)
        return state.legal_actions()

    def apply_action(self, state: Connect6State, action: Connect6Action) -> Connect6State:
        self._check(state)
        return state.apply_action(action)

    def _check(self, state):
        if not isinstance(state, Connect6State) or state.board_size != self.board_size:
            raise ValueError('state belongs to a different game configuration')


@dataclass(frozen=True, slots=True)
class Connect6State:
    _native_position: object

    @property
    def board_size(self) -> int:
        return self._native_position.board_size

    @property
    def board(self) -> tuple[tuple[int | None, ...], ...]:
        flat = self._native_position.board
        n = self.board_size
        return tuple(tuple(flat[i:i+n]) for i in range(0, len(flat), n))

    @property
    def current_player(self) -> int | None:
        return self._native_position.current_player

    @property
    def placements_remaining(self) -> int:
        return self._native_position.placements_remaining

    @property
    def terminal(self) -> bool:
        return self._native_position.terminal

    @property
    def winner(self) -> int | None:
        return self._native_position.winner

    def legal_actions(self) -> tuple[Connect6Action, ...]:
        return tuple(Connect6Action(p) for p in self._native_position.legal_actions())

    def apply_action(self, action: Connect6Action) -> Connect6State:
        if not isinstance(action, Connect6Action):
            raise TypeError('expected Connect6Action')
        return Connect6State(self._native_position.apply(action.position))

    def __reduce__(self):
        # Workers carry game configurations. Snapshots need an explicit native format;
        # reconstructing cells in an arbitrary order would change turns and terminality.
        raise TypeError('serialize the game configuration and replay ordered actions instead')
