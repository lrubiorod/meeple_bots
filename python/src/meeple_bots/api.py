"""Typed Python facade over the private Rust extension."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, sqrt
from typing import TypeAlias

from . import _native

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

    @classmethod
    def from_recommendation(
        cls,
        recommendation: MctsRecommendation,
        exploration: float = sqrt(2.0),
        heuristic: int | None = None,
    ) -> MctsAgent:
        """Build an agent from a hardware-calibrated recommendation."""

        if not isinstance(recommendation, MctsRecommendation):
            raise TypeError("recommendation must be an MctsRecommendation")
        return cls(
            iterations=recommendation.iterations,
            exploration=exploration,
            rollout_depth=recommendation.rollout_depth,
            heuristic=heuristic,
        )


class MctsLevel(str, Enum):
    """Named MCTS compute budgets, not guaranteed cross-game strength levels."""

    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"


@dataclass(frozen=True, slots=True)
class MctsRecommendation:
    """Fixed MCTS parameters calibrated for one game and machine."""

    level: MctsLevel
    iterations: int
    rollout_depth: int
    target_time_ms: int
    estimated_time_ms: float
    milliseconds_per_iteration: float


@dataclass(frozen=True, slots=True)
class GameComplexityReport:
    """Empirical structural metrics and hardware-aware MCTS recommendations."""

    game: Game
    samples: int
    max_depth: int
    completed_samples: int
    terminal_rate: float
    initial_legal_actions: int
    mean_branching_factor: float
    effective_branching_factor: float
    maximum_branching_factor: int
    p95_branching_factor: int
    mean_plies: float
    median_plies: int
    p75_plies: int
    p95_plies: int
    estimated_tree_log10: float
    estimate_is_lower_bound: bool
    max_iterations: int
    recommendations: tuple[
        MctsRecommendation, MctsRecommendation, MctsRecommendation
    ]

    def recommend(
        self,
        level: MctsLevel = MctsLevel.BALANCED,
        time_budget_ms: int | None = None,
    ) -> MctsRecommendation:
        """Return a level recommendation, optionally rescaled to another time budget."""

        if not isinstance(level, MctsLevel):
            raise TypeError("level must be an MctsLevel")
        base = next(item for item in self.recommendations if item.level is level)
        if time_budget_ms is None:
            return base
        _positive_u32("time_budget_ms", time_budget_ms)
        iterations = int(time_budget_ms / base.milliseconds_per_iteration)
        iterations = min(
            self.max_iterations,
            max(self.initial_legal_actions, iterations),
        )
        return MctsRecommendation(
            level=level,
            iterations=iterations,
            rollout_depth=base.rollout_depth,
            target_time_ms=time_budget_ms,
            estimated_time_ms=iterations * base.milliseconds_per_iteration,
            milliseconds_per_iteration=base.milliseconds_per_iteration,
        )


class StrengthEstimate(str, Enum):
    """Relative MCTS strength inferred from confidence intervals."""

    INCONCLUSIVE = "inconclusive"
    UNPROVEN_AGAINST_RANDOM = "unproven_against_random"
    BEATS_RANDOM_BELOW_REFERENCE = "beats_random_below_reference"
    BEATS_RANDOM_NO_DETECTED_REFERENCE_GAP = (
        "beats_random_no_detected_reference_gap"
    )


class CutoffHeuristicEvidence(str, Enum):
    """Evidence that depth-limited rollouts would benefit from a state heuristic."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class SearchSufficiency(str, Enum):
    """Whether MCTS revisits branches often enough to compare them."""

    INSUFFICIENT = "insufficient"
    LIMITED = "limited"
    ADEQUATE = "adequate"


class BenchmarkConfidence(str, Enum):
    """Confidence supported by the number of matches per opponent."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class StrengthOpponent(str, Enum):
    """Opponent currently used by the strength benchmark."""

    RANDOM = "random"
    REFERENCE = "reference"


class StrengthProgressStage(str, Enum):
    """Whether a benchmark match is about to start or has completed."""

    STARTED = "started"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class MctsStrengthProgress:
    """Synchronous progress event emitted around one benchmark match."""

    stage: StrengthProgressStage
    match_number: int
    total_matches: int
    opponent: StrengthOpponent
    candidate_player: int
    plies: int | None
    utility: float | None
    elapsed_seconds: float | None


@dataclass(frozen=True, slots=True)
class OpponentResult:
    """Candidate results against one opponent, always from the candidate's view."""

    matches: int
    wins: int
    draws: int
    losses: int
    score: float
    mean_utility: float
    utility_confidence_low: float
    utility_confidence_high: float


@dataclass(frozen=True, slots=True)
class MctsSearchSummary:
    """Search telemetry aggregated over every candidate decision in the benchmark."""

    decisions: int
    total_iterations: int
    mean_expanded_nodes: float
    maximum_expanded_nodes: int
    mean_root_actions: float
    mean_iterations_per_root_action: float
    mean_tree_revisit_rate: float
    mean_tree_depth: float
    maximum_tree_depth: int
    mean_simulation_depth: float
    maximum_simulation_depth: int
    terminal_rollout_rate: float
    truncated_rollout_rate: float
    mean_selected_action_visit_share: float


@dataclass(frozen=True, slots=True)
class MctsStrengthReport:
    """Relative playing results and evidence about whether MCTS needs a heuristic."""

    game: Game
    candidate: MctsAgent
    reference: MctsAgent
    matches_per_opponent: int
    initial_expanded_nodes: int
    tree_size_log10_gap: float
    tree_size_estimate_is_lower_bound: bool
    search: MctsSearchSummary
    versus_random: OpponentResult
    versus_reference: OpponentResult
    search_sufficiency: SearchSufficiency
    benchmark_confidence: BenchmarkConfidence
    strength_estimate: StrengthEstimate
    cutoff_heuristic_evidence: CutoffHeuristicEvidence
    reasons: tuple[str, ...]


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


Game: TypeAlias = TicTacToe | ConnectFour | Boop
GameAction: TypeAlias = TicTacToeAction | ConnectFourAction | BoopAction
BoardCell: TypeAlias = int | BoopPiece | None
GameBoard: TypeAlias = tuple[tuple[BoardCell, ...], ...]


@dataclass(frozen=True, slots=True)
class HumanTurn:
    """Read-only position presented to a human move selector."""

    game: Game
    player: int
    board: GameBoard
    legal_actions: tuple[GameAction, ...]
    pools: tuple[BoopPool, BoopPool] | None = None


MoveSelector: TypeAlias = Callable[[HumanTurn], GameAction]


@dataclass(frozen=True, slots=True)
class HumanAgent:
    """A player controlled by a Python function or an interactive terminal prompt."""

    select_action: MoveSelector = field(default=lambda turn: _prompt_human_action(turn))

    def __post_init__(self) -> None:
        if not callable(self.select_action):
            raise TypeError("select_action must be callable")


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


@dataclass(frozen=True, slots=True)
class Match:
    """Configuration for one match executed by the Rust engine."""

    game: Game = field(default_factory=TicTacToe)
    first: Agent = field(default_factory=MctsAgent)
    second: Agent = field(default_factory=RandomAgent)
    seed: int = 0
    max_plies: int = 10_000

    def __post_init__(self) -> None:
        if not isinstance(self.game, (TicTacToe, ConnectFour, Boop)):
            raise TypeError("game must be TicTacToe, ConnectFour, or Boop")
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

    def run(self) -> MatchResult:
        """Execute the match and return its complete immutable report."""

        raw = _native.run_match(
            _native_game(self.game),
            _native_agent(self.first, self.game),
            _native_agent(self.second, self.game),
            self.seed,
            self.max_plies,
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
            final_board=_final_board_from_native(raw["final_board"], self.game),
            pools=_pools_from_native(raw["pools"]),
        )


def evaluate_game_complexity(
    game: Game,
    samples: int = 128,
    max_depth: int = 256,
    seed: int = 0,
    heuristic: int | None = None,
) -> GameComplexityReport:
    """Sample a game tree and calibrate MCTS recommendations on this machine."""

    if not isinstance(game, (TicTacToe, ConnectFour, Boop)):
        raise TypeError("game must be TicTacToe, ConnectFour, or Boop")
    _positive_u32("samples", samples)
    _positive_u32("max_depth", max_depth)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= _MAX_U64:
        raise ValueError(f"seed must be between 0 and {_MAX_U64}")
    _validate_game_heuristic(game, heuristic)

    raw = _native.evaluate_game_complexity(
        _native_game(game),
        samples,
        max_depth,
        seed,
        heuristic,
    )
    recommendations = tuple(
        MctsRecommendation(
            level=MctsLevel(item["level"]),
            iterations=item["iterations"],
            rollout_depth=item["rollout_depth"],
            target_time_ms=item["target_time_ms"],
            estimated_time_ms=item["estimated_time_ms"],
            milliseconds_per_iteration=item["milliseconds_per_iteration"],
        )
        for item in raw["recommendations"]
    )
    return GameComplexityReport(
        game=game,
        samples=raw["samples"],
        max_depth=raw["max_depth"],
        completed_samples=raw["completed_samples"],
        terminal_rate=raw["terminal_rate"],
        initial_legal_actions=raw["initial_legal_actions"],
        mean_branching_factor=raw["mean_branching_factor"],
        effective_branching_factor=raw["effective_branching_factor"],
        maximum_branching_factor=raw["maximum_branching_factor"],
        p95_branching_factor=raw["p95_branching_factor"],
        mean_plies=raw["mean_plies"],
        median_plies=raw["median_plies"],
        p75_plies=raw["p75_plies"],
        p95_plies=raw["p95_plies"],
        estimated_tree_log10=raw["estimated_tree_log10"],
        estimate_is_lower_bound=raw["estimate_is_lower_bound"],
        max_iterations=raw["max_iterations"],
        recommendations=recommendations,
    )


def evaluate_mcts_strength(
    game: Game,
    agent: MctsAgent,
    matches_per_opponent: int = 20,
    reference_iterations_multiplier: int = 4,
    max_plies: int = 10_000,
    seed: int = 0,
    complexity_report: GameComplexityReport | None = None,
    progress: Callable[[MctsStrengthProgress], None] | None = None,
) -> MctsStrengthReport:
    """Benchmark one MCTS configuration and inspect its real search behavior."""

    if not isinstance(game, (TicTacToe, ConnectFour, Boop)):
        raise TypeError("game must be TicTacToe, ConnectFour, or Boop")
    if not isinstance(agent, MctsAgent):
        raise TypeError("agent must be an MctsAgent")
    _validate_game_heuristic(game, agent.heuristic)
    _positive_u32("matches_per_opponent", matches_per_opponent)
    if matches_per_opponent % 2 != 0:
        raise ValueError("matches_per_opponent must be even")
    _positive_u32(
        "reference_iterations_multiplier", reference_iterations_multiplier
    )
    if reference_iterations_multiplier < 2:
        raise ValueError("reference_iterations_multiplier must be at least 2")
    if agent.iterations >= 1_000_000:
        raise ValueError("agent.iterations must be less than 1000000")
    _positive_u32("max_plies", max_plies)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= _MAX_U64:
        raise ValueError(f"seed must be between 0 and {_MAX_U64}")
    if complexity_report is None:
        complexity_report = evaluate_game_complexity(
            game,
            seed=seed,
            heuristic=agent.heuristic,
        )
    elif not isinstance(complexity_report, GameComplexityReport):
        raise TypeError("complexity_report must be a GameComplexityReport")
    elif complexity_report.game != game:
        raise ValueError("complexity_report must describe the selected game")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")

    def notify(raw: dict[str, object]) -> None:
        if progress is not None:
            progress(_strength_progress_from_native(raw))

    raw = _native.evaluate_mcts_strength(
        _native_game(game),
        agent.iterations,
        agent.exploration,
        agent.rollout_depth,
        complexity_report.estimated_tree_log10,
        complexity_report.estimate_is_lower_bound,
        agent.heuristic,
        matches_per_opponent,
        reference_iterations_multiplier,
        max_plies,
        seed,
        notify if progress is not None else None,
    )
    candidate = MctsAgent(
        iterations=raw["candidate_iterations"],
        exploration=raw["candidate_exploration"],
        rollout_depth=raw["candidate_rollout_depth"],
        heuristic=agent.heuristic,
    )
    reference = MctsAgent(
        iterations=raw["reference_iterations"],
        exploration=raw["reference_exploration"],
        rollout_depth=raw["reference_rollout_depth"],
        heuristic=agent.heuristic,
    )
    search = raw["search"]
    return MctsStrengthReport(
        game=game,
        candidate=candidate,
        reference=reference,
        matches_per_opponent=raw["matches_per_opponent"],
        initial_expanded_nodes=raw["initial_expanded_nodes"],
        tree_size_log10_gap=raw["tree_size_log10_gap"],
        tree_size_estimate_is_lower_bound=raw[
            "tree_size_estimate_is_lower_bound"
        ],
        search=MctsSearchSummary(
            decisions=search["decisions"],
            total_iterations=search["total_iterations"],
            mean_expanded_nodes=search["mean_expanded_nodes"],
            maximum_expanded_nodes=search["maximum_expanded_nodes"],
            mean_root_actions=search["mean_root_actions"],
            mean_iterations_per_root_action=search[
                "mean_iterations_per_root_action"
            ],
            mean_tree_revisit_rate=search["mean_tree_revisit_rate"],
            mean_tree_depth=search["mean_tree_depth"],
            maximum_tree_depth=search["maximum_tree_depth"],
            mean_simulation_depth=search["mean_simulation_depth"],
            maximum_simulation_depth=search["maximum_simulation_depth"],
            terminal_rollout_rate=search["terminal_rollout_rate"],
            truncated_rollout_rate=search["truncated_rollout_rate"],
            mean_selected_action_visit_share=search[
                "mean_selected_action_visit_share"
            ],
        ),
        versus_random=_opponent_result_from_native(raw["versus_random"]),
        versus_reference=_opponent_result_from_native(raw["versus_reference"]),
        search_sufficiency=SearchSufficiency(raw["search_sufficiency"]),
        benchmark_confidence=BenchmarkConfidence(raw["benchmark_confidence"]),
        strength_estimate=StrengthEstimate(raw["strength_estimate"]),
        cutoff_heuristic_evidence=CutoffHeuristicEvidence(
            raw["cutoff_heuristic_evidence"]
        ),
        reasons=tuple(raw["reasons"]),
    )


def _strength_progress_from_native(raw: dict[str, object]) -> MctsStrengthProgress:
    return MctsStrengthProgress(
        stage=StrengthProgressStage(raw["stage"]),
        match_number=raw["match_number"],
        total_matches=raw["total_matches"],
        opponent=StrengthOpponent(raw["opponent"]),
        candidate_player=raw["candidate_player"],
        plies=raw["plies"],
        utility=raw["utility"],
        elapsed_seconds=raw["elapsed_seconds"],
    )


def _opponent_result_from_native(raw: dict[str, object]) -> OpponentResult:
    return OpponentResult(
        matches=raw["matches"],
        wins=raw["wins"],
        draws=raw["draws"],
        losses=raw["losses"],
        score=raw["score"],
        mean_utility=raw["mean_utility"],
        utility_confidence_low=raw["utility_confidence_low"],
        utility_confidence_high=raw["utility_confidence_high"],
    )


def _native_game(game: Game) -> str:
    if isinstance(game, TicTacToe):
        return "tic_tac_toe"
    if isinstance(game, ConnectFour):
        return "connect_four"
    return "boop"


def _action_from_native(raw: dict[str, object]) -> GameAction:
    if raw["type"] == "tic_tac_toe":
        return TicTacToeAction(row=raw["row"], column=raw["column"])
    if raw["type"] == "connect_four":
        return ConnectFourAction(column=raw["column"])
    return BoopAction(
        piece=BoopPieceKind(raw["piece"]),
        row=raw["row"],
        column=raw["column"],
        resolution=_boop_resolution_from_native(raw["resolution"]),
    )


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
    return _native.AgentConfig.human(_human_selector(agent, game))


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
    if not isinstance(game, Boop):
        raise ValueError(f"{_game_display_name(game)} does not provide MCTS heuristics")
    if heuristic != 0:
        raise ValueError(
            f"boop does not provide MCTS heuristic {heuristic}; available indices: 0"
        )


def _game_display_name(game: Game) -> str:
    if isinstance(game, TicTacToe):
        return "tic-tac-toe"
    if isinstance(game, ConnectFour):
        return "connect-four"
    return "boop"


def _human_selector(agent: HumanAgent, game: Game):
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


def _board_rows(flat_board: list[BoardCell], columns: int) -> GameBoard:
    return tuple(
        tuple(flat_board[start : start + columns])
        for start in range(0, len(flat_board), columns)
    )


def _prompt_human_action(turn: HumanTurn) -> GameAction:
    print(file=sys.stderr)
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
    return _prompt_tic_tac_toe_action(turn)


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


def _final_board_from_native(flat_board, game: Game) -> GameBoard:
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


def _board_symbol(cell: BoardCell) -> str:
    if cell is None:
        return "."
    if isinstance(cell, int):
        return "X" if cell == 0 else "O"
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
