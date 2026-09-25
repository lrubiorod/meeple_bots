"""Typed Python facade over the private Rust extension."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ._agent_config import (
    BaseRolloutPolicy,
    ConditionalRollout,
    EpsilonGreedy,
    GameHeuristic,
    Greedy,
    Mast,
    MctsAgent,
    NeutralEvaluator,
    ProgressiveBias,
    RandomAgent,
    SoIsmctsAgent,
    RolloutPolicy,
    StateEvaluator,
    TurnPhaseIs,
    UniformRandom,
    _MAX_U32,
    _non_negative_u32,
    _positive_u32,
    _validate_state_evaluator,
)

from . import _native
from .connect6 import Connect6, Connect6Action, Connect6State
from .game_config import game_parameters
from .lost_cities import LostCities, LostCitiesAction, LostCitiesState, LostCitiesChanceEvent
from .splendor import Splendor, SplendorAction, SplendorState, SplendorChanceOutcome, ChanceEvent
from .game_types import (
    BoardCell, Boop, BoopAction,
    BoopGraduateLine, BoopPiece, BoopPieceKind,
    BoopPool, BoopPosition, BoopRecoverPiece,
    BoopResolution, ConnectFour, ConnectFourAction,
    EndSpiritCollection, ForestPosition, Game,
    GameAction, GameBoard, HumanMoveObservation,
    HumanMoveObserver, HumanTurn, MatchMoveObservation,
    MatchMoveObserver, MoveSelector, MoveSpiritGemstone,
    PlaceSpiritGemstone, PowerSource, SkipSpiritGemstone,
    Spirit, SpiritCollection, SpiritGemstonePool,
    SpiritGemstoneSacrifice, SpiritTile, SpiritsOfTheForest,
    SpiritsOfTheForestAction, SpiritsTurnPhase, TakeSpiritTile,
    TicTacToe, TicTacToeAction,
)
from .matches.models import (
    BatchMatchResult, BatchProgress, BatchProgressCallback,
    BatchProgressStatus, BatchResult, MatchResult,
    Move, RootActionDiagnostic, TreeReuseDiagnostic,
    _BatchJob, _BatchMatchOutcome,
)
from .native_bridge import (
    _native_game, _analyze_trace, _native_evaluator, _native_progressive_bias,
    _native_rollout_policy, _native_base_rollout_policy,
    _action_from_native, _board_rows, _boop_action_from_selector,
    _boop_resolution_from_native, _final_board_from_native, _game_display_name,
    _gemstone_pools_from_native, _initial_spirits_state, _pools_from_native,
    _rollout_evaluator, _rollout_evaluators, _spirit_collections_from_native,
    _spirits_action_from_mapping, _spirits_action_from_native, _spirits_state_from_native,
    _validate_agent_evaluators, _validate_game_evaluator, _validate_game_heuristic,
    _validate_game_heuristic_params,
)
from .matches.human import (
    Agent, HumanAgent, _board_symbol,
    _human_move_observer, _human_selector, _native_agent,
    _prompt_boop_action, _prompt_boop_resolution, _prompt_connect_four_action,
    _prompt_human_action, _prompt_spirits_action, _prompt_tic_tac_toe_action,
    _resolution_description, _spirits_action_description,
)
from .matches.execution import (
    Batch, Match, _MAX_U64,
    _match_move_observer,
)


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
class SampledDecisionTiming:
    """Measured MCTS decision latency at one reproducibly sampled ply."""

    sampled_ply: int
    milliseconds: float
    iterations: int
    nodes: int
    legal_actions: int = 0
    terminal_simulations: int | None = None
    cutoff_simulations: int | None = None
    root_visits: tuple[int, ...] = ()
    root_expansion: tuple[int, int, int] | None = None
    widening_expansions: tuple[int, int, int, int] | None = None
    phase: str | None = None


@dataclass(frozen=True, slots=True)
class MctsAgentBenchmark:
    """Isolated decision-latency measurements for one exact MCTS configuration."""

    game: Game
    agent: MctsAgent
    sampled_positions: int
    decision_time_mean_ms: float
    decision_time_p50_ms: float
    decision_time_p95_ms: float
    decision_time_max_ms: float
    milliseconds_per_iteration: float
    position_timings: tuple[SampledDecisionTiming, ...]
    maximum_decision_horizon: int | None = None


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
    # Deprecated compatibility presets; analyze renders operating points instead.
    suggested_experiments: tuple[SuggestedMctsExperiment, ...]
    # Compatibility aliases for the Balanced experiment.
    recommended_rollout_depth: int
    recommended_iterations: int
    iterations_capped: bool
    milliseconds_per_iteration: float
    estimated_decision_time_ms: float
    structural: dict | None = None


def evaluate_game(
    game: Game,
    samples: int = 128,
    max_depth: int = 256,
    seed: int = 0,
    target_time: float = 5.0,
) -> GameEvaluationReport:
    """Measure game structure and produce practical local MCTS starting points."""

    if not isinstance(game, (TicTacToe, ConnectFour, Boop, SpiritsOfTheForest, Splendor, LostCities, Connect6)):
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
        game_params=game_parameters(game),
    )
    return GameEvaluationReport(
        game=game,
        structural=raw.get("structural"),
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


def benchmark_mcts_agent(
    game: Game,
    agent: MctsAgent,
    median_depth: int,
    seed: int = 0,
) -> MctsAgentBenchmark:
    """Measure one exact MCTS configuration on shared early, middle, and late states."""

    if not isinstance(game, (TicTacToe, ConnectFour, Boop, SpiritsOfTheForest, Splendor, LostCities, Connect6)):
        raise TypeError(
            "game must be TicTacToe, ConnectFour, Connect6, Boop, SpiritsOfTheForest, Splendor, or LostCities"
        )
    if not isinstance(agent, MctsAgent):
        raise TypeError("agent must be an MctsAgent")
    _non_negative_u32("median_depth", median_depth)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= _MAX_U64:
        raise ValueError(f"seed must be between 0 and {_MAX_U64}")
    _validate_agent_evaluators(game, agent)

    (
        rollout_policy,
        rollout_evaluator,
        rollout_heuristic,
        rollout_params,
        rollout_epsilon,
        rollout_condition_phase,
        fallback_rollout_policy,
        fallback_rollout_evaluator,
        fallback_rollout_heuristic,
        fallback_rollout_params,
        fallback_rollout_epsilon,
    ) = _native_rollout_policy(agent.rollout_policy)
    raw = _native.benchmark_mcts_agent(
        _native_game(game),
        agent.iterations,
        agent.time_budget,
        float(agent.exploration),
        agent.rollout_depth,
        *_native_evaluator(agent.cutoff_evaluator),
        rollout_policy,
        rollout_evaluator,
        rollout_heuristic,
        rollout_params,
        rollout_epsilon,
        median_depth,
        seed,
        rollout_condition_phase,
        fallback_rollout_policy,
        fallback_rollout_evaluator,
        fallback_rollout_heuristic,
        fallback_rollout_params,
        fallback_rollout_epsilon,
        *_native_progressive_bias(agent.progressive_bias),
        agent.root_diagnostics,
        agent.tree_reuse,
        agent.transpositions,
        agent.selection_policy,
        game_params=game_parameters(game),
        rave_equivalence=agent.rave_equivalence,
        progressive_widening=agent.progressive_widening,
        progressive_widening_k=agent.progressive_widening_k,
        progressive_widening_alpha=agent.progressive_widening_alpha,
        progressive_widening_expansion=agent.progressive_widening_expansion,
    )
    return MctsAgentBenchmark(
        game=game,
        agent=agent,
        maximum_decision_horizon=raw["maximum_decision_horizon"],
        sampled_positions=raw["sampled_positions"],
        decision_time_mean_ms=raw["decision_time_mean_ms"],
        decision_time_p50_ms=raw["decision_time_p50_ms"],
        decision_time_p95_ms=raw["decision_time_p95_ms"],
        decision_time_max_ms=raw["decision_time_max_ms"],
        milliseconds_per_iteration=raw["milliseconds_per_iteration"],
        position_timings=tuple(
            SampledDecisionTiming(
                sampled_ply=timing["sampled_ply"],
                phase=timing.get("phase"),
                milliseconds=timing["milliseconds"],
                iterations=timing["iterations"],
                nodes=timing["nodes"],
                legal_actions=timing["legal_actions"],
                terminal_simulations=timing["terminal_simulations"],
                cutoff_simulations=timing["cutoff_simulations"],
                root_visits=tuple(timing["root_visits"]),
                widening_expansions=tuple(timing["widening_expansions"]) if timing["widening_expansions"] is not None else None,
                root_expansion=tuple(timing["root_expansion"]) if timing["root_expansion"] is not None else None,
            )
            for timing in raw["position_timings"]
        ),
    )
