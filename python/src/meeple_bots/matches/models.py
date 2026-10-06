"""Immutable match/batch results and callback payloads."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from ..game_types import GameAction, GameBoard, BoopPool, SpiritCollection, SpiritGemstonePool
from ..lost_cities import LostCitiesChanceEvent, LostCitiesState
from ..splendor import ChanceEvent, SplendorState

@dataclass(frozen=True, slots=True)
class RootActionDiagnostic:
    """Cached root-child statistics from one MCTS decision."""

    action_index: int
    visits: int
    mean_utility: float
    heuristic_value: float | None
    progressive_bias: float | None
    selected: bool

@dataclass(frozen=True, slots=True)
class TreeReuseDiagnostic:
    """Subtree reuse observed before one MCTS decision."""

    transition_attempts: int
    transition_hits: int
    transition_misses: int
    own_action_hits: int
    opponent_action_hits: int
    reused_root_visits: int
    reused_nodes: int
    pruned_nodes: int
    resets: int

@dataclass(frozen=True, slots=True)
class Move:
    """One action; decision_seconds includes selection and attributed lifecycle work.

    Optional component times distinguish current measurements from legacy traces.
    """

    player: int
    action: GameAction
    decision_seconds: float = field(default=0.0, compare=False)
    search_iterations: int | None = field(default=None, compare=False)
    search_nodes: int | None = field(default=None, compare=False)
    root_actions: tuple[RootActionDiagnostic, ...] = field(
        default=(), compare=False
    )
    tree_reuse: TreeReuseDiagnostic | None = field(default=None, compare=False)
    selection_seconds: float | None = field(default=None, compare=False)
    maintenance_seconds: float | None = field(default=None, compare=False)
    terminal_simulations: int | None = field(default=None, compare=False)
    cutoff_simulations: int | None = field(default=None, compare=False)

class MatchTermination(str, Enum):
    """Authoritative game completion versus administrative adjudication."""

    GAME_TERMINAL = "game_terminal"
    PLY_LIMIT = "ply_limit"

    @classmethod
    def from_result(cls, result: dict) -> MatchTermination:
        """Read explicit reasons and legacy version-1 traces without guessing outcomes."""
        legacy = result.get("ply_limit_reached", False)
        if type(legacy) is not bool:
            raise ValueError("ply_limit_reached must be a boolean")
        reason = cls(result.get("termination", "ply_limit" if legacy else "game_terminal"))
        if "ply_limit_reached" in result and legacy != (reason is cls.PLY_LIMIT):
            raise ValueError("conflicting match termination metadata")
        return reason

    def validate_endpoint(self, *, terminal: bool, resolved: bool = True) -> None:
        """Replay validates rules; this validates the match completion policy."""
        if self is self.GAME_TERMINAL and not terminal:
            raise ValueError("trace does not end in a terminal position")
        if self is self.PLY_LIMIT and (terminal or not resolved):
            raise ValueError("ply-limit trace must end at a resolved nonterminal position")


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Immutable summary and full action history of a completed match."""

    unassigned_maintenance_seconds: tuple[float, float] = field(
        default=(0.0, 0.0), compare=False, kw_only=True
    )
    termination: MatchTermination = field(default=MatchTermination.GAME_TERMINAL, kw_only=True)

    seed: int
    plies: int
    utilities: tuple[float, ...]
    winner: int | None
    moves: tuple[Move, ...]
    final_board: GameBoard
    pools: tuple[BoopPool, BoopPool] | None
    spirit_collections: tuple[SpiritCollection, SpiritCollection] | None = None
    gemstone_pools: tuple[SpiritGemstonePool, SpiritGemstonePool] | None = None
    scores: tuple[int, int] | None = None

    chance_events: tuple[ChanceEvent | LostCitiesChanceEvent, ...] = ()
    lost_cities_state: LostCitiesState | None = None
    splendor_state: SplendorState | None = None
    game_params: dict[str, int] = field(default_factory=dict)

    @property
    def ply_limit_reached(self) -> bool:
        return self.termination is MatchTermination.PLY_LIMIT


class BatchProgressStatus(str, Enum):
    """Stage reported by a batch progress event."""

    STARTED = "started"
    COMPLETED = "completed"

@dataclass(frozen=True, slots=True)
class BatchMatchResult:
    """Participant-oriented result for one match in a batch."""

    match_number: int
    seed: int
    agent_a_player: int
    winner: int | None
    plies: int
    utilities: tuple[float, float]
    duration_seconds: float

@dataclass(frozen=True, slots=True)
class BatchProgress:
    """Ordered batch progress, including the full trace for a completed match."""

    status: BatchProgressStatus
    match_number: int
    total_matches: int
    seed: int
    agent_a_player: int
    elapsed_seconds: float
    result: BatchMatchResult | None = None
    match_result: MatchResult | None = None

BatchProgressCallback: TypeAlias = Callable[[BatchProgress], None]

@dataclass(frozen=True, slots=True)
class BatchResult:
    """Aggregate and per-match results for a completed simulation batch."""

    seed: int
    matches: int
    workers: int
    alternate_sides: bool
    agent_a_wins: int
    agent_b_wins: int
    draws: int
    total_plies: int
    average_plies: float
    elapsed_seconds: float
    games: tuple[BatchMatchResult, ...]

@dataclass(frozen=True, slots=True)
class _BatchJob:
    match_number: int
    seed: int
    agent_a_player: int

@dataclass(frozen=True, slots=True)
class _BatchMatchOutcome:
    summary: BatchMatchResult
    match_result: MatchResult

