"""Typed Python facade over the private Rust extension."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, sqrt
from time import perf_counter
from typing import TypeAlias

from . import _native
from ._concurrency import WorkerSetting, ordered_parallel_map, resolve_workers

_MAX_U32 = 2**32 - 1
_MAX_U64 = 2**64 - 1


def _positive_u32(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 1 <= value <= _MAX_U32:
        raise ValueError(f"{name} must be between 1 and {_MAX_U32}")


@dataclass(frozen=True, slots=True)
class TicTacToe:
    """The standard 3x3 tic-tac-toe rules."""


@dataclass(frozen=True, slots=True)
class ConnectFour:
    """The standard 6x7 Connect Four rules with gravity."""


@dataclass(frozen=True, slots=True)
class Boop:
    """The standard two-player rules for boop. on a 6x6 bed."""


@dataclass(frozen=True, slots=True)
class SpiritsOfTheForest:
    """Two-player Spirits of the Forest without favor tokens."""


@dataclass(frozen=True, slots=True)
class RandomAgent:
    """An agent that chooses uniformly among legal actions."""


@dataclass(frozen=True, slots=True)
class MctsAgent:
    """Configuration for the Monte Carlo Tree Search agent."""

    iterations: int = 1_000
    exploration: float = sqrt(2.0)
    rollout_depth: int = 256
    heuristic: int | None = None

    def __post_init__(self) -> None:
        _positive_u32("iterations", self.iterations)
        _positive_u32("rollout_depth", self.rollout_depth)
        if isinstance(self.exploration, bool) or not isinstance(self.exploration, (int, float)):
            raise TypeError("exploration must be a number")
        if not isfinite(self.exploration) or self.exploration < 0:
            raise ValueError("exploration must be finite and non-negative")
        if self.heuristic is not None:
            _non_negative_u32("heuristic", self.heuristic)


@dataclass(frozen=True, slots=True)
class IterationBudgetEstimate:
    """Approximate iterations that fit in one time budget."""

    seconds: int
    iterations: int


@dataclass(frozen=True, slots=True)
class RolloutCostEstimate:
    """Measured MCTS cost for one candidate rollout horizon."""

    rollout_depth: int
    approximate_player_turns: float
    milliseconds_per_iteration: float
    iteration_budgets: tuple[IterationBudgetEstimate, ...]


@dataclass(frozen=True, slots=True)
class SuggestedMctsExperiment:
    """A starting point for comparative MCTS benchmarks, not a strength claim."""

    label: str
    iterations: int
    iterations_capped: bool
    rollout_depth: int
    approximate_player_turns: float
    estimated_decision_time_ms: float


@dataclass(frozen=True, slots=True)
class GameEvaluationReport:
    """Structural metrics and practical, locally measured MCTS starting points."""

    game: Game
    samples: int
    max_depth: int
    terminal_rate: float
    initial_legal_actions: int
    effective_branching_factor: float
    player_turn_choice_product_log10: float
    depth_p50: int
    estimated_depth: int
    player_turn_depth_p50: int
    player_turn_depth_p95: int
    player_changes_p50: int
    player_changes_p95: int
    actions_per_player_turn_mean: float
    actions_per_player_turn_p95: int
    actions_per_player_turn_max: int
    depth_is_lower_bound: bool
    estimated_tree_log10: float
    calibration_positions: int
    target_time_seconds: float
    rollout_costs: tuple[RolloutCostEstimate, ...]
    suggested_experiments: tuple[SuggestedMctsExperiment, ...]
    # Compatibility aliases for the Balanced experiment.
    recommended_rollout_depth: int
    recommended_iterations: int
    iterations_capped: bool
    milliseconds_per_iteration: float
    estimated_decision_time_ms: float


@dataclass(frozen=True, slots=True)
class TicTacToeAction:
    """A zero-based row and column on the tic-tac-toe board."""

    row: int
    column: int

    def __post_init__(self) -> None:
        for name, value in (("row", self.row), ("column", self.column)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= value < 3:
                raise ValueError(f"{name} must be between 0 and 2")


@dataclass(frozen=True, slots=True)
class ConnectFourAction:
    """A zero-based column in which to drop a Connect Four piece."""

    column: int

    def __post_init__(self) -> None:
        if isinstance(self.column, bool) or not isinstance(self.column, int):
            raise TypeError("column must be an integer")
        if not 0 <= self.column < 7:
            raise ValueError("column must be between 0 and 6")


class BoopPieceKind(str, Enum):
    """The two ranks of pieces in boop."""

    KITTEN = "kitten"
    CAT = "cat"


@dataclass(frozen=True, slots=True)
class BoopPosition:
    """A zero-based position on the 6x6 boop. board."""

    row: int
    column: int

    def __post_init__(self) -> None:
        for name, value in (("row", self.row), ("column", self.column)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= value < 6:
                raise ValueError(f"{name} must be between 0 and 5")


@dataclass(frozen=True, slots=True)
class BoopGraduateLine:
    """The exact line of three pieces selected for graduation."""

    positions: tuple[BoopPosition, BoopPosition, BoopPosition]

    def __post_init__(self) -> None:
        if not isinstance(self.positions, tuple) or len(self.positions) != 3:
            raise TypeError("positions must be a tuple containing exactly three positions")
        if not all(isinstance(position, BoopPosition) for position in self.positions):
            raise TypeError("every graduation position must be a BoopPosition")


@dataclass(frozen=True, slots=True)
class BoopRecoverPiece:
    """The piece selected for recovery when all eight pieces are on the board."""

    position: BoopPosition

    def __post_init__(self) -> None:
        if not isinstance(self.position, BoopPosition):
            raise TypeError("position must be a BoopPosition")


BoopResolution: TypeAlias = BoopGraduateLine | BoopRecoverPiece | None


@dataclass(frozen=True, slots=True)
class BoopAction:
    """A placement and any mandatory end-of-turn resolution in boop."""

    piece: BoopPieceKind
    row: int
    column: int
    resolution: BoopResolution = None

    def __post_init__(self) -> None:
        if not isinstance(self.piece, BoopPieceKind):
            raise TypeError("piece must be a BoopPieceKind")
        BoopPosition(self.row, self.column)


@dataclass(frozen=True, slots=True)
class BoopPiece:
    """A kitten or cat on the board and its owner."""

    player: int
    kind: BoopPieceKind

    def __post_init__(self) -> None:
        if self.player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        if not isinstance(self.kind, BoopPieceKind):
            raise TypeError("kind must be a BoopPieceKind")


@dataclass(frozen=True, slots=True)
class BoopPool:
    """The kittens and cats currently available to one player."""

    kittens: int
    cats: int

    def __post_init__(self) -> None:
        for name, value in (("kittens", self.kittens), ("cats", self.cats)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= value <= 8:
                raise ValueError(f"{name} must be between 0 and 8")


class Spirit(str, Enum):
    MOSS = "moss"
    FLOWERS = "flowers"
    FRUITS = "fruits"
    MUSHROOMS = "mushrooms"
    WATER = "water"
    VINES = "vines"
    BRANCHES = "branches"
    LEAVES = "leaves"
    WEBS = "webs"


class PowerSource(str, Enum):
    FIRE = "fire"
    MOON = "moon"
    SUN = "sun"


class SpiritsTurnPhase(str, Enum):
    COLLECT = "collect"
    PLACE_GEMSTONE = "place_gemstone"


@dataclass(frozen=True, slots=True)
class ForestPosition:
    row: int
    column: int

    def __post_init__(self) -> None:
        for name, value, upper in (("row", self.row, 4), ("column", self.column, 12)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= value < upper:
                raise ValueError(f"{name} must be between 0 and {upper - 1}")


@dataclass(frozen=True, slots=True)
class SpiritGemstoneSacrifice:
    """A sacrificed gem; a missing source means a gem from the available supply."""

    source: ForestPosition | None = None

    def __post_init__(self) -> None:
        if self.source is not None and not isinstance(self.source, ForestPosition):
            raise TypeError("source must be a ForestPosition or None")


@dataclass(frozen=True, slots=True)
class TakeSpiritTile:
    position: ForestPosition
    sacrifice: SpiritGemstoneSacrifice | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.position, ForestPosition):
            raise TypeError("position must be a ForestPosition")
        if self.sacrifice is not None and not isinstance(
            self.sacrifice, SpiritGemstoneSacrifice
        ):
            raise TypeError("sacrifice must be a SpiritGemstoneSacrifice or None")


@dataclass(frozen=True, slots=True)
class EndSpiritCollection:
    pass


@dataclass(frozen=True, slots=True)
class PlaceSpiritGemstone:
    target: ForestPosition

    def __post_init__(self) -> None:
        if not isinstance(self.target, ForestPosition):
            raise TypeError("target must be a ForestPosition")


@dataclass(frozen=True, slots=True)
class MoveSpiritGemstone:
    source: ForestPosition
    target: ForestPosition

    def __post_init__(self) -> None:
        if not isinstance(self.source, ForestPosition) or not isinstance(
            self.target, ForestPosition
        ):
            raise TypeError("source and target must be ForestPosition values")


@dataclass(frozen=True, slots=True)
class SkipSpiritGemstone:
    pass


SpiritsOfTheForestAction: TypeAlias = (
    TakeSpiritTile
    | EndSpiritCollection
    | PlaceSpiritGemstone
    | MoveSpiritGemstone
    | SkipSpiritGemstone
)


@dataclass(frozen=True, slots=True)
class SpiritTile:
    spirit: Spirit
    spirit_symbols: int
    power_source: PowerSource | None
    gemstone: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.spirit, Spirit):
            raise TypeError("spirit must be a Spirit")
        if self.spirit_symbols not in (1, 2):
            raise ValueError("spirit_symbols must be 1 or 2")
        if self.power_source is not None and not isinstance(
            self.power_source, PowerSource
        ):
            raise TypeError("power_source must be a PowerSource or None")
        if self.gemstone not in (None, 0, 1):
            raise ValueError("gemstone must be player 0, player 1, or None")


@dataclass(frozen=True, slots=True)
class SpiritCollection:
    spirit_symbols: tuple[int, ...]
    power_sources: tuple[int, ...]
    tiles: int

    def __post_init__(self) -> None:
        if len(self.spirit_symbols) != 9 or len(self.power_sources) != 3:
            raise ValueError("a collection needs 9 spirit and 3 power-source counts")
        values = (*self.spirit_symbols, *self.power_sources, self.tiles)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("collection counts must be non-negative integers")


@dataclass(frozen=True, slots=True)
class SpiritGemstonePool:
    available: int
    placed: int
    removed: int

    def __post_init__(self) -> None:
        values = (self.available, self.placed, self.removed)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("gemstone counts must be non-negative integers")
        if sum(values) != 3:
            raise ValueError("each player must account for exactly three gemstones")


Game: TypeAlias = TicTacToe | ConnectFour | Boop | SpiritsOfTheForest
GameAction: TypeAlias = (
    TicTacToeAction | ConnectFourAction | BoopAction | SpiritsOfTheForestAction
)
BoardCell: TypeAlias = int | BoopPiece | SpiritTile | None
GameBoard: TypeAlias = tuple[tuple[BoardCell, ...], ...]


@dataclass(frozen=True, slots=True)
class HumanTurn:
    """Read-only position presented to a human move selector."""

    game: Game
    player: int
    board: GameBoard
    legal_actions: tuple[GameAction, ...]
    pools: tuple[BoopPool, BoopPool] | None = None
    spirit_collections: tuple[SpiritCollection, SpiritCollection] | None = None
    gemstone_pools: tuple[SpiritGemstonePool, SpiritGemstonePool] | None = None
    scores: tuple[int, int] | None = None
    phase: SpiritsTurnPhase | None = None


MoveSelector: TypeAlias = Callable[[HumanTurn], GameAction]


@dataclass(frozen=True, slots=True)
class HumanMoveObservation:
    """State immediately after an accepted human action."""

    game: Game
    player: int
    action: GameAction
    board: GameBoard
    pools: tuple[BoopPool, BoopPool] | None = None
    spirit_collections: tuple[SpiritCollection, SpiritCollection] | None = None
    gemstone_pools: tuple[SpiritGemstonePool, SpiritGemstonePool] | None = None
    scores: tuple[int, int] | None = None
    phase: SpiritsTurnPhase | None = None
    active_player: int | None = None


HumanMoveObserver: TypeAlias = Callable[[HumanMoveObservation], None]


@dataclass(frozen=True, slots=True)
class MatchMoveObservation:
    """State after any accepted action, including the agent's thinking time."""

    game: Game
    player: int
    action: GameAction
    board: GameBoard
    decision_seconds: float
    pools: tuple[BoopPool, BoopPool] | None = None
    spirit_collections: tuple[SpiritCollection, SpiritCollection] | None = None
    gemstone_pools: tuple[SpiritGemstonePool, SpiritGemstonePool] | None = None
    scores: tuple[int, int] | None = None
    phase: SpiritsTurnPhase | None = None
    active_player: int | None = None


MatchMoveObserver: TypeAlias = Callable[[MatchMoveObservation], None]


@dataclass(frozen=True, slots=True)
class HumanAgent:
    """A player controlled by a Python function or an interactive terminal prompt."""

    select_action: MoveSelector = field(default=lambda turn: _prompt_human_action(turn))
    observe_action: HumanMoveObserver | None = None

    def __post_init__(self) -> None:
        if not callable(self.select_action):
            raise TypeError("select_action must be callable")
        if self.observe_action is not None and not callable(self.observe_action):
            raise TypeError("observe_action must be callable")


Agent: TypeAlias = RandomAgent | MctsAgent | HumanAgent


@dataclass(frozen=True, slots=True)
class Move:
    """One action selected by one player."""

    player: int
    action: GameAction


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Immutable summary and full action history of a completed match."""

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


@dataclass(frozen=True, slots=True)
class Match:
    """Configuration for one match executed by the Rust engine."""

    game: Game = field(default_factory=TicTacToe)
    first: Agent = field(default_factory=MctsAgent)
    second: Agent = field(default_factory=RandomAgent)
    seed: int = 0
    max_plies: int = 10_000
    observe_move: MatchMoveObserver | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.game, (TicTacToe, ConnectFour, Boop, SpiritsOfTheForest)):
            raise TypeError(
                "game must be TicTacToe, ConnectFour, Boop, or SpiritsOfTheForest"
            )
        if not isinstance(self.first, (RandomAgent, MctsAgent, HumanAgent)):
            raise TypeError("first must be RandomAgent, MctsAgent, or HumanAgent")
        if not isinstance(self.second, (RandomAgent, MctsAgent, HumanAgent)):
            raise TypeError("second must be RandomAgent, MctsAgent, or HumanAgent")
        _validate_agent_heuristic(self.game, self.first)
        _validate_agent_heuristic(self.game, self.second)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= self.seed <= _MAX_U64:
            raise ValueError(f"seed must be between 0 and {_MAX_U64}")
        _positive_u32("max_plies", self.max_plies)
        if self.observe_move is not None:
            if not callable(self.observe_move):
                raise TypeError("observe_move must be callable")

    def run(self) -> MatchResult:
        """Execute the match and return its complete immutable report."""

        raw = _native.run_match(
            _native_game(self.game),
            _native_agent(self.first, self.game),
            _native_agent(self.second, self.game),
            self.seed,
            self.max_plies,
            _match_move_observer(self.observe_move, self.game),
        )
        moves = tuple(
            Move(
                player=item["player"],
                action=_action_from_native(item["action"]),
            )
            for item in raw["moves"]
        )
        return MatchResult(
            seed=raw["seed"],
            plies=raw["plies"],
            utilities=tuple(raw["utilities"]),
            winner=raw["winner"],
            moves=moves,
            final_board=_final_board_from_native(
                raw["spirit_forest"]
                if isinstance(self.game, SpiritsOfTheForest)
                else raw["final_board"],
                self.game,
            ),
            pools=_pools_from_native(raw["pools"]),
            spirit_collections=_spirit_collections_from_native(
                raw["spirit_collections"]
            ),
            gemstone_pools=_gemstone_pools_from_native(raw["gemstone_pools"]),
            scores=None if raw["scores"] is None else tuple(raw["scores"]),
        )


@dataclass(frozen=True, slots=True)
class Batch:
    """A reproducible series of automated matches between two participants."""

    game: Game = field(default_factory=TicTacToe)
    agent_a: RandomAgent | MctsAgent = field(default_factory=RandomAgent)
    agent_b: RandomAgent | MctsAgent = field(default_factory=MctsAgent)
    matches: int = 20
    seed: int = 0
    max_plies: int = 10_000
    alternate_sides: bool = True
    workers: WorkerSetting = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.game, (TicTacToe, ConnectFour, Boop, SpiritsOfTheForest)):
            raise TypeError(
                "game must be TicTacToe, ConnectFour, Boop, or SpiritsOfTheForest"
            )
        for name, agent in (("agent_a", self.agent_a), ("agent_b", self.agent_b)):
            if not isinstance(agent, (RandomAgent, MctsAgent)):
                raise TypeError(f"{name} must be RandomAgent or MctsAgent")
            _validate_agent_heuristic(self.game, agent)
        _positive_u32("matches", self.matches)
        _positive_u32("max_plies", self.max_plies)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= self.seed <= _MAX_U64:
            raise ValueError(f"seed must be between 0 and {_MAX_U64}")
        if not isinstance(self.alternate_sides, bool):
            raise TypeError("alternate_sides must be a boolean")
        resolve_workers(self.workers)

    def run(
        self,
        progress: BatchProgressCallback | None = None,
    ) -> BatchResult:
        """Run matches concurrently and optionally report submission and completion."""

        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable")

        batch_started = perf_counter()
        games = []
        agent_a_wins = 0
        agent_b_wins = 0
        draws = 0
        worker_count = min(resolve_workers(self.workers), self.matches)
        jobs = (
            _BatchJob(
                match_number=offset + 1,
                seed=(self.seed + offset) & _MAX_U64,
                agent_a_player=offset % 2 if self.alternate_sides else 0,
            )
            for offset in range(self.matches)
        )

        def notify_started(job: _BatchJob) -> None:
            if progress is not None:
                progress(
                    BatchProgress(
                        status=BatchProgressStatus.STARTED,
                        match_number=job.match_number,
                        total_matches=self.matches,
                        seed=job.seed,
                        agent_a_player=job.agent_a_player,
                        elapsed_seconds=perf_counter() - batch_started,
                    )
                )

        for job, outcome in ordered_parallel_map(
            self._run_job,
            jobs,
            worker_count,
            notify_started,
        ):
            game_result = outcome.summary
            if game_result.winner is None:
                draws += 1
            elif game_result.winner == 0:
                agent_a_wins += 1
            else:
                agent_b_wins += 1
            games.append(game_result)
            if progress is not None:
                progress(
                    BatchProgress(
                        status=BatchProgressStatus.COMPLETED,
                        match_number=job.match_number,
                        total_matches=self.matches,
                        seed=job.seed,
                        agent_a_player=job.agent_a_player,
                        elapsed_seconds=perf_counter() - batch_started,
                        result=game_result,
                        match_result=outcome.match_result,
                    )
                )

        elapsed_seconds = perf_counter() - batch_started
        total_plies = sum(game.plies for game in games)
        return BatchResult(
            seed=self.seed,
            matches=self.matches,
            workers=worker_count,
            alternate_sides=self.alternate_sides,
            agent_a_wins=agent_a_wins,
            agent_b_wins=agent_b_wins,
            draws=draws,
            total_plies=total_plies,
            average_plies=total_plies / self.matches,
            elapsed_seconds=elapsed_seconds,
            games=tuple(games),
        )

    def _run_job(self, job: _BatchJob) -> _BatchMatchOutcome:
        match_started = perf_counter()
        first, second = (
            (self.agent_a, self.agent_b)
            if job.agent_a_player == 0
            else (self.agent_b, self.agent_a)
        )
        match = Match(
            game=self.game,
            first=first,
            second=second,
            seed=job.seed,
            max_plies=self.max_plies,
        ).run()
        if match.winner is None:
            winner = None
        elif match.winner == job.agent_a_player:
            winner = 0
        else:
            winner = 1
        return _BatchMatchOutcome(
            summary=BatchMatchResult(
                match_number=job.match_number,
                seed=job.seed,
                agent_a_player=job.agent_a_player,
                winner=winner,
                plies=match.plies,
                utilities=(
                    match.utilities[job.agent_a_player],
                    match.utilities[1 - job.agent_a_player],
                ),
                duration_seconds=perf_counter() - match_started,
            ),
            match_result=match,
        )


def evaluate_game(
    game: Game,
    samples: int = 128,
    max_depth: int = 256,
    seed: int = 0,
    target_time: float = 5.0,
) -> GameEvaluationReport:
    """Measure game structure and produce practical local MCTS starting points."""

    if not isinstance(game, (TicTacToe, ConnectFour, Boop, SpiritsOfTheForest)):
        raise TypeError(
            "game must be TicTacToe, ConnectFour, Boop, or SpiritsOfTheForest"
        )
    _positive_u32("samples", samples)
    _positive_u32("max_depth", max_depth)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= _MAX_U64:
        raise ValueError(f"seed must be between 0 and {_MAX_U64}")
    if isinstance(target_time, bool) or not isinstance(target_time, (int, float)):
        raise TypeError("target_time must be a number")
    if not isfinite(target_time) or not 0 < target_time <= 3_600:
        raise ValueError(
            "target_time must be finite, greater than zero, and at most 3600 seconds"
        )

    raw = _native.evaluate_game(
        _native_game(game),
        samples,
        max_depth,
        seed,
        target_time,
    )
    return GameEvaluationReport(
        game=game,
        samples=raw["samples"],
        max_depth=raw["max_depth"],
        terminal_rate=raw["terminal_rate"],
        initial_legal_actions=raw["initial_legal_actions"],
        effective_branching_factor=raw["effective_branching_factor"],
        player_turn_choice_product_log10=raw["player_turn_choice_product_log10"],
        depth_p50=raw["depth_p50"],
        estimated_depth=raw["estimated_depth"],
        player_turn_depth_p50=raw["player_turn_depth_p50"],
        player_turn_depth_p95=raw["player_turn_depth_p95"],
        player_changes_p50=raw["player_changes_p50"],
        player_changes_p95=raw["player_changes_p95"],
        actions_per_player_turn_mean=raw["actions_per_player_turn_mean"],
        actions_per_player_turn_p95=raw["actions_per_player_turn_p95"],
        actions_per_player_turn_max=raw["actions_per_player_turn_max"],
        depth_is_lower_bound=raw["depth_is_lower_bound"],
        estimated_tree_log10=raw["estimated_tree_log10"],
        calibration_positions=raw["calibration_positions"],
        target_time_seconds=raw["target_time_seconds"],
        rollout_costs=tuple(
            RolloutCostEstimate(
                rollout_depth=cost["rollout_depth"],
                approximate_player_turns=cost["approximate_player_turns"],
                milliseconds_per_iteration=cost["milliseconds_per_iteration"],
                iteration_budgets=tuple(
                    IterationBudgetEstimate(
                        seconds=budget["seconds"],
                        iterations=budget["iterations"],
                    )
                    for budget in cost["iteration_budgets"]
                ),
            )
            for cost in raw["rollout_costs"]
        ),
        suggested_experiments=tuple(
            SuggestedMctsExperiment(
                label=experiment["label"],
                iterations=experiment["iterations"],
                iterations_capped=experiment["iterations_capped"],
                rollout_depth=experiment["rollout_depth"],
                approximate_player_turns=experiment["approximate_player_turns"],
                estimated_decision_time_ms=experiment["estimated_decision_time_ms"],
            )
            for experiment in raw["suggested_experiments"]
        ),
        recommended_rollout_depth=raw["recommended_rollout_depth"],
        recommended_iterations=raw["recommended_iterations"],
        iterations_capped=raw["iterations_capped"],
        milliseconds_per_iteration=raw["milliseconds_per_iteration"],
        estimated_decision_time_ms=raw["estimated_decision_time_ms"],
    )


def _native_game(game: Game) -> str:
    if isinstance(game, TicTacToe):
        return "tic_tac_toe"
    if isinstance(game, ConnectFour):
        return "connect_four"
    if isinstance(game, SpiritsOfTheForest):
        return "spotf"
    return "boop"


def _action_from_native(raw: dict[str, object]) -> GameAction:
    if raw["type"] == "tic_tac_toe":
        return TicTacToeAction(row=raw["row"], column=raw["column"])
    if raw["type"] == "connect_four":
        return ConnectFourAction(column=raw["column"])
    if raw["type"] in ("spotf", "spirits_of_the_forest"):
        return _spirits_action_from_mapping(raw)
    return BoopAction(
        piece=BoopPieceKind(raw["piece"]),
        row=raw["row"],
        column=raw["column"],
        resolution=_boop_resolution_from_native(raw["resolution"]),
    )


def _analyze_trace(game: Game, moves: tuple[Move, ...], *, seed: int = 0):
    """Dispatch a completed trace to the selected game's native analyzer."""

    native_moves = []
    if isinstance(game, Boop):
        for move in moves:
            if not isinstance(move.action, BoopAction):
                raise TypeError("boop trace contains a non-boop action")
            resolution = move.action.resolution
            if isinstance(resolution, BoopGraduateLine):
                native_resolution = (
                    "graduate",
                    [(position.row, position.column) for position in resolution.positions],
                )
            elif isinstance(resolution, BoopRecoverPiece):
                native_resolution = (
                    "recover",
                    [(resolution.position.row, resolution.position.column)],
                )
            else:
                native_resolution = ("none", [])
            native_moves.append(
                (
                    move.player,
                    (
                        move.action.piece.value,
                        move.action.row,
                        move.action.column,
                        native_resolution,
                    ),
                )
            )
    elif isinstance(game, SpiritsOfTheForest):
        for move in moves:
            action = move.action
            if isinstance(action, TakeSpiritTile):
                sacrifice = None
                if action.sacrifice is not None:
                    source = action.sacrifice.source
                    sacrifice = (
                        "available" if source is None else "forest",
                        None if source is None else (source.row, source.column),
                    )
                native_action = (
                    "take_tile",
                    [(action.position.row, action.position.column)],
                    sacrifice,
                )
            elif isinstance(action, EndSpiritCollection):
                native_action = ("end_collection", [], None)
            elif isinstance(action, PlaceSpiritGemstone):
                native_action = (
                    "place_gemstone",
                    [(action.target.row, action.target.column)],
                    None,
                )
            elif isinstance(action, MoveSpiritGemstone):
                native_action = (
                    "move_gemstone",
                    [
                        (action.source.row, action.source.column),
                        (action.target.row, action.target.column),
                    ],
                    None,
                )
            elif isinstance(action, SkipSpiritGemstone):
                native_action = ("skip_gemstone", [], None)
            else:
                raise TypeError("spotf trace contains a non-spotf action")
            native_moves.append((move.player, native_action))
    return _native.analyze_trace(_native_game(game), native_moves, seed)


def _native_agent(agent: Agent, game: Game):
    if isinstance(agent, RandomAgent):
        return _native.AgentConfig.random()
    if isinstance(agent, MctsAgent):
        return _native.AgentConfig.mcts(
            agent.iterations,
            float(agent.exploration),
            agent.rollout_depth,
            agent.heuristic,
        )
    return _native.AgentConfig.human(
        _human_selector(agent, game),
        _human_move_observer(agent, game),
    )


def _non_negative_u32(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= _MAX_U32:
        raise ValueError(f"{name} must be between 0 and {_MAX_U32}")


def _validate_agent_heuristic(game: Game, agent: Agent) -> None:
    if isinstance(agent, MctsAgent):
        _validate_game_heuristic(game, agent.heuristic)


def _validate_game_heuristic(game: Game, heuristic: int | None) -> None:
    if heuristic is None:
        return
    _non_negative_u32("heuristic", heuristic)
    if isinstance(game, SpiritsOfTheForest):
        if heuristic != 0:
            raise ValueError(
                "spotf does not provide MCTS heuristic "
                f"{heuristic}; available indices: 0..0"
            )
        return
    if not isinstance(game, Boop):
        raise ValueError(f"{_game_display_name(game)} does not provide MCTS heuristics")
    if heuristic not in (0, 1):
        raise ValueError(
            f"boop does not provide MCTS heuristic {heuristic}; available indices: 0..1"
        )


def _game_display_name(game: Game) -> str:
    if isinstance(game, TicTacToe):
        return "tic-tac-toe"
    if isinstance(game, ConnectFour):
        return "connect-four"
    if isinstance(game, SpiritsOfTheForest):
        return "spotf"
    return "boop"


def _human_selector(agent: HumanAgent, game: Game):
    if isinstance(game, SpiritsOfTheForest):
        def select_spirits(player: int, native_state, native_actions) -> int:
            board, collections, gems, phase, _active, scores = _spirits_state_from_native(
                native_state
            )
            legal_actions = tuple(
                _spirits_action_from_native(action) for action in native_actions
            )
            turn = HumanTurn(
                game=game,
                player=player,
                board=board,
                legal_actions=legal_actions,
                spirit_collections=collections,
                gemstone_pools=gems,
                scores=scores,
                phase=phase,
            )
            action = agent.select_action(turn)
            if not isinstance(
                action,
                (
                    TakeSpiritTile,
                    EndSpiritCollection,
                    PlaceSpiritGemstone,
                    MoveSpiritGemstone,
                    SkipSpiritGemstone,
                ),
            ):
                raise TypeError("human select_action must return a SpiritsOfTheForestAction")
            if action not in legal_actions:
                raise ValueError("the selected action is not currently legal")
            return legal_actions.index(action)

        return select_spirits

    def select(
        player: int,
        flat_board,
        native_context,
        native_boop_actions=None,
    ) -> tuple[int, int] | int:
        if isinstance(game, TicTacToe):
            board = _board_rows(flat_board, columns=3)
            legal_actions: tuple[GameAction, ...] = tuple(
                TicTacToeAction(row=row, column=column)
                for row, column in native_context
            )
            pools = None
        elif isinstance(game, ConnectFour):
            board = _board_rows(flat_board, columns=7)
            legal_actions = tuple(
                ConnectFourAction(column=column) for column in native_context
            )
            pools = None
        else:
            board = _board_rows(
                [
                    None
                    if piece is None
                    else BoopPiece(player=piece[0], kind=BoopPieceKind(piece[1]))
                    for piece in flat_board
                ],
                columns=6,
            )
            pools = tuple(
                BoopPool(kittens=kittens, cats=cats)
                for kittens, cats in native_context
            )
            legal_actions = tuple(
                _boop_action_from_selector(action) for action in native_boop_actions
            )
        turn = HumanTurn(
            game=game,
            player=player,
            board=board,
            legal_actions=legal_actions,
            pools=pools,
        )
        action = agent.select_action(turn)
        if isinstance(game, TicTacToe):
            expected_type = TicTacToeAction
        elif isinstance(game, ConnectFour):
            expected_type = ConnectFourAction
        else:
            expected_type = BoopAction
        if not isinstance(action, expected_type):
            raise TypeError(f"human select_action must return {expected_type.__name__}")
        if action not in turn.legal_actions:
            raise ValueError("the selected action is not currently legal")
        if isinstance(action, TicTacToeAction):
            return action.row, action.column
        if isinstance(action, ConnectFourAction):
            return action.column
        return legal_actions.index(action)

    return select


def _human_move_observer(agent: HumanAgent, game: Game):
    if agent.observe_action is None:
        return None

    if isinstance(game, SpiritsOfTheForest):
        def observe_spirits(player: int, native_state, native_action) -> None:
            board, collections, gems, phase, active, scores = _spirits_state_from_native(
                native_state
            )
            agent.observe_action(
                HumanMoveObservation(
                    game=game,
                    player=player,
                    action=_spirits_action_from_native(native_action),
                    board=board,
                    spirit_collections=collections,
                    gemstone_pools=gems,
                    scores=scores,
                    phase=phase,
                    active_player=active,
                )
            )

        return observe_spirits

    def observe(player: int, flat_board, native_pools, native_action) -> None:
        if isinstance(game, TicTacToe):
            board = _board_rows(flat_board, columns=3)
            pools = None
            action: GameAction = TicTacToeAction(
                row=native_action[0],
                column=native_action[1],
            )
        elif isinstance(game, ConnectFour):
            board = _board_rows(flat_board, columns=7)
            pools = None
            action = ConnectFourAction(column=native_action)
        else:
            board = _board_rows(
                [
                    None
                    if piece is None
                    else BoopPiece(player=piece[0], kind=BoopPieceKind(piece[1]))
                    for piece in flat_board
                ],
                columns=6,
            )
            pools = tuple(
                BoopPool(kittens=kittens, cats=cats)
                for kittens, cats in native_pools
            )
            action = _boop_action_from_selector(native_action)
        agent.observe_action(
            HumanMoveObservation(
                game=game,
                player=player,
                action=action,
                board=board,
                pools=pools,
            )
        )

    return observe


def _match_move_observer(observer: MatchMoveObserver | None, game: Game):
    if observer is None:
        return None

    if isinstance(game, Boop):
        def observe_boop(
            player: int,
            flat_board,
            native_pools,
            native_action,
            decision_seconds: float,
        ) -> None:
            board = _board_rows(
                [
                    None
                    if piece is None
                    else BoopPiece(player=piece[0], kind=BoopPieceKind(piece[1]))
                    for piece in flat_board
                ],
                columns=6,
            )
            pools = tuple(
                BoopPool(kittens=kittens, cats=cats)
                for kittens, cats in native_pools
            )
            observer(
                MatchMoveObservation(
                    game=game,
                    player=player,
                    action=_boop_action_from_selector(native_action),
                    board=board,
                    decision_seconds=decision_seconds,
                    pools=pools,
                )
            )

        return observe_boop

    if isinstance(game, SpiritsOfTheForest):
        def observe_spirits(
            player: int,
            native_state,
            native_action,
            decision_seconds: float,
        ) -> None:
            board, collections, gems, phase, active, scores = _spirits_state_from_native(
                native_state
            )
            observer(
                MatchMoveObservation(
                    game=game,
                    player=player,
                    action=_spirits_action_from_native(native_action),
                    board=board,
                    decision_seconds=decision_seconds,
                    spirit_collections=collections,
                    gemstone_pools=gems,
                    scores=scores,
                    phase=phase,
                    active_player=active,
                )
            )

        return observe_spirits

    def observe(
        player: int,
        flat_board,
        native_action,
        decision_seconds: float,
    ) -> None:
        if isinstance(game, TicTacToe):
            action: GameAction = TicTacToeAction(
                row=native_action[0],
                column=native_action[1],
            )
            board = _board_rows(flat_board, columns=3)
        elif isinstance(game, ConnectFour):
            action = ConnectFourAction(column=native_action)
            board = _board_rows(flat_board, columns=7)
        observer(
            MatchMoveObservation(
                game=game,
                player=player,
                action=action,
                board=board,
                decision_seconds=decision_seconds,
            )
        )

    return observe


def _board_rows(flat_board: list[BoardCell], columns: int) -> GameBoard:
    return tuple(
        tuple(flat_board[start : start + columns])
        for start in range(0, len(flat_board), columns)
    )


def _prompt_human_action(turn: HumanTurn) -> GameAction:
    print(file=sys.stderr)
    print(f"Current board before player {turn.player}'s move:", file=sys.stderr)
    print("    " + " ".join(str(column) for column in range(len(turn.board[0]))), file=sys.stderr)
    for row, cells in enumerate(turn.board):
        rendered = " ".join(_board_symbol(cell) for cell in cells)
        print(f"{row} | {rendered}", file=sys.stderr)

    if turn.pools is not None:
        for player, pool in enumerate(turn.pools):
            print(
                f"Player {player} pool: {pool.kittens} kittens, {pool.cats} cats",
                file=sys.stderr,
            )

    if isinstance(turn.game, ConnectFour):
        return _prompt_connect_four_action(turn)
    if isinstance(turn.game, Boop):
        return _prompt_boop_action(turn)
    if isinstance(turn.game, SpiritsOfTheForest):
        return _prompt_spirits_action(turn)
    return _prompt_tic_tac_toe_action(turn)


def _prompt_spirits_action(turn: HumanTurn) -> SpiritsOfTheForestAction:
    print(f"Phase: {turn.phase.value if turn.phase else 'unknown'}", file=sys.stderr)
    if turn.scores is not None:
        print(f"Provisional scores: {turn.scores[0]} / {turn.scores[1]}", file=sys.stderr)
    for index, action in enumerate(turn.legal_actions):
        print(f"  {index}: {_spirits_action_description(action)}", file=sys.stderr)
    while True:
        print("Choose a legal action number: ", end="", file=sys.stderr, flush=True)
        try:
            selected = int(input())
            return turn.legal_actions[selected]
        except (ValueError, IndexError):
            print("Enter one of the listed action numbers.", file=sys.stderr)


def _spirits_action_description(action: SpiritsOfTheForestAction) -> str:
    if isinstance(action, TakeSpiritTile):
        text = f"take ({action.position.row}, {action.position.column})"
        if action.sacrifice is not None:
            text += (
                " sacrificing an available gem"
                if action.sacrifice.source is None
                else " sacrificing gem at "
                f"({action.sacrifice.source.row}, {action.sacrifice.source.column})"
            )
        return text
    if isinstance(action, EndSpiritCollection):
        return "end collection"
    if isinstance(action, PlaceSpiritGemstone):
        return f"place gem at ({action.target.row}, {action.target.column})"
    if isinstance(action, MoveSpiritGemstone):
        return (
            f"move gem ({action.source.row}, {action.source.column}) to "
            f"({action.target.row}, {action.target.column})"
        )
    return "skip gemstone"


def _prompt_boop_action(turn: HumanTurn) -> BoopAction:
    while True:
        print(
            f"Player {turn.player}, enter piece and position (for example, k 2 3): ",
            end="",
            file=sys.stderr,
            flush=True,
        )
        try:
            parts = input().lower().split()
            if len(parts) != 3:
                raise ValueError("enter k or c followed by two numbers")
            piece = {"k": BoopPieceKind.KITTEN, "c": BoopPieceKind.CAT}.get(parts[0])
            if piece is None:
                raise ValueError("piece must be k (kitten) or c (cat)")
            row, column = int(parts[1]), int(parts[2])
            candidates = [
                action
                for action in turn.legal_actions
                if isinstance(action, BoopAction)
                and action.piece == piece
                and action.row == row
                and action.column == column
            ]
            if not candidates:
                raise ValueError("that placement is not currently legal")
            if len(candidates) == 1:
                return candidates[0]
            return _prompt_boop_resolution(candidates)
        except ValueError as error:
            print(f"Invalid move: {error}", file=sys.stderr)


def _prompt_boop_resolution(candidates: list[BoopAction]) -> BoopAction:
    print("Choose the end-of-turn resolution:", file=sys.stderr)
    for index, action in enumerate(candidates):
        print(f"  {index}: {_resolution_description(action.resolution)}", file=sys.stderr)
    while True:
        print("Resolution number: ", end="", file=sys.stderr, flush=True)
        try:
            return candidates[int(input())]
        except (ValueError, IndexError):
            print("Invalid resolution number", file=sys.stderr)


def _resolution_description(resolution: BoopResolution) -> str:
    if isinstance(resolution, BoopGraduateLine):
        positions = ", ".join(
            f"({position.row}, {position.column})" for position in resolution.positions
        )
        return f"graduate line {positions}"
    if isinstance(resolution, BoopRecoverPiece):
        return f"recover ({resolution.position.row}, {resolution.position.column})"
    return "no resolution"


def _boop_action_from_selector(raw) -> BoopAction:
    piece, row, column, raw_resolution = raw
    resolution_type, positions = raw_resolution
    if resolution_type == "graduate":
        resolution: BoopResolution = BoopGraduateLine(
            tuple(BoopPosition(row, column) for row, column in positions)
        )
    elif resolution_type == "recover":
        resolution = BoopRecoverPiece(BoopPosition(*positions[0]))
    else:
        resolution = None
    return BoopAction(BoopPieceKind(piece), row, column, resolution)


def _boop_resolution_from_native(raw) -> BoopResolution:
    if raw["type"] == "graduate":
        return BoopGraduateLine(
            tuple(BoopPosition(row, column) for row, column in raw["positions"])
        )
    if raw["type"] == "recover":
        return BoopRecoverPiece(BoopPosition(raw["row"], raw["column"]))
    return None


def _spirits_action_from_native(raw) -> SpiritsOfTheForestAction:
    kind, positions, sacrifice = raw
    if kind == "take_tile":
        row, column = positions[0]
        parsed_sacrifice = None
        if sacrifice is not None:
            sacrifice_kind, source = sacrifice
            parsed_sacrifice = SpiritGemstoneSacrifice(
                None if sacrifice_kind == "available" else ForestPosition(*source)
            )
        return TakeSpiritTile(ForestPosition(row, column), parsed_sacrifice)
    if kind == "end_collection":
        return EndSpiritCollection()
    if kind == "place_gemstone":
        return PlaceSpiritGemstone(ForestPosition(*positions[0]))
    if kind == "move_gemstone":
        return MoveSpiritGemstone(
            ForestPosition(*positions[0]), ForestPosition(*positions[1])
        )
    if kind == "skip_gemstone":
        return SkipSpiritGemstone()
    raise ValueError(f"unknown Spirits of the Forest action: {kind}")


def _spirits_action_from_mapping(raw: dict[str, object]) -> SpiritsOfTheForestAction:
    kind = raw["kind"]
    if kind == "take_tile":
        sacrifice = raw["sacrifice"]
        parsed = None
        if sacrifice is not None:
            parsed = SpiritGemstoneSacrifice(
                None
                if sacrifice["kind"] == "available"
                else ForestPosition(sacrifice["row"], sacrifice["column"])
            )
        return TakeSpiritTile(
            ForestPosition(raw["row"], raw["column"]),
            parsed,
        )
    if kind == "end_collection":
        return EndSpiritCollection()
    if kind == "place_gemstone":
        return PlaceSpiritGemstone(ForestPosition(raw["row"], raw["column"]))
    if kind == "move_gemstone":
        return MoveSpiritGemstone(
            ForestPosition(raw["source_row"], raw["source_column"]),
            ForestPosition(raw["target_row"], raw["target_column"]),
        )
    return SkipSpiritGemstone()


def _spirits_state_from_native(raw):
    flat_forest, raw_collections, raw_gems, raw_phase, active, raw_scores = raw
    board = _board_rows(
        [
            None
            if tile is None
            else SpiritTile(
                spirit=Spirit(tile[0]),
                spirit_symbols=tile[1],
                power_source=None if tile[2] is None else PowerSource(tile[2]),
                gemstone=tile[3],
            )
            for tile in flat_forest
        ],
        columns=12,
    )
    collections = tuple(
        SpiritCollection(tuple(spirits), tuple(sources), tiles)
        for spirits, sources, tiles in raw_collections
    )
    gems = tuple(
        SpiritGemstonePool(available, placed, removed)
        for available, placed, removed in raw_gems
    )
    return (
        board,
        collections,
        gems,
        SpiritsTurnPhase(raw_phase),
        active,
        tuple(raw_scores),
    )


def _initial_spirits_state(seed: int):
    return _spirits_state_from_native(_native.spirits_initial_state(seed))


def _final_board_from_native(flat_board, game: Game) -> GameBoard:
    if isinstance(game, SpiritsOfTheForest):
        return _board_rows(
            [
                None
                if tile is None
                else SpiritTile(
                    spirit=Spirit(tile["spirit"]),
                    spirit_symbols=tile["spirit_symbols"],
                    power_source=(
                        None
                        if tile["power_source"] is None
                        else PowerSource(tile["power_source"])
                    ),
                    gemstone=tile["gemstone"],
                )
                for tile in flat_board
            ],
            columns=12,
        )
    cells: list[BoardCell] = []
    for piece in flat_board:
        if piece is None:
            cells.append(None)
        elif piece["kind"] == "token":
            cells.append(piece["player"])
        else:
            cells.append(
                BoopPiece(
                    player=piece["player"],
                    kind=BoopPieceKind(piece["kind"]),
                )
            )
    columns = 3 if isinstance(game, TicTacToe) else 7 if isinstance(game, ConnectFour) else 6
    return _board_rows(cells, columns)


def _pools_from_native(raw) -> tuple[BoopPool, BoopPool] | None:
    if raw is None:
        return None
    pools = tuple(BoopPool(kittens=pool["kittens"], cats=pool["cats"]) for pool in raw)
    return pools


def _spirit_collections_from_native(raw):
    if raw is None:
        return None
    return tuple(
        SpiritCollection(
            tuple(collection["spirit_symbols"]),
            tuple(collection["power_sources"]),
            collection["tiles"],
        )
        for collection in raw
    )


def _gemstone_pools_from_native(raw):
    if raw is None:
        return None
    return tuple(
        SpiritGemstonePool(pool["available"], pool["placed"], pool["removed"])
        for pool in raw
    )


def _board_symbol(cell: BoardCell) -> str:
    if cell is None:
        return "."
    if isinstance(cell, int):
        return "X" if cell == 0 else "O"
    if isinstance(cell, SpiritTile):
        return cell.spirit.value[0].upper() + str(cell.spirit_symbols)
    if cell.player == 0:
        return "x" if cell.kind is BoopPieceKind.KITTEN else "X"
    return "o" if cell.kind is BoopPieceKind.KITTEN else "O"


def _prompt_connect_four_action(turn: HumanTurn) -> ConnectFourAction:
    while True:
        print(
            f"Player {turn.player}, enter column (0-6): ",
            end="",
            file=sys.stderr,
            flush=True,
        )
        try:
            action = ConnectFourAction(column=int(input()))
            if action not in turn.legal_actions:
                raise ValueError("that column is full")
            return action
        except ValueError as error:
            print(f"Invalid move: {error}", file=sys.stderr)


def _prompt_tic_tac_toe_action(turn: HumanTurn) -> TicTacToeAction:
    while True:
        print(
            f"Player {turn.player}, enter row and column (for example, 1 2): ",
            end="",
            file=sys.stderr,
            flush=True,
        )
        try:
            parts = input().split()
            if len(parts) != 2:
                raise ValueError("enter exactly two numbers")
            action = TicTacToeAction(row=int(parts[0]), column=int(parts[1]))
            if action not in turn.legal_actions:
                raise ValueError("that cell is already occupied")
            return action
        except ValueError as error:
            print(f"Invalid move: {error}", file=sys.stderr)
