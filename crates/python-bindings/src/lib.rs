//! Private PyO3 boundary for the public Python package.

mod cant_stop;
mod connect6;
mod evaluation;
mod lost_cities;
mod splendor;

use meeple_bots_catalog::{
    configured_connect6_mcts, run_connect6_match_with_observer, run_connect6_match_with_trace,
};
use meeple_bots_connect6::{Connect6, Connect6Action};
use std::{collections::BTreeMap, num::NonZeroU32, time::Duration};

use meeple_bots_boop::{
    BoardZone, Boop, BoopAction, BoopInteractionOutcome, BoopReplayAnalysis, BoopStateMetrics,
    GraduateLine, LineOrientation, PieceKind as BoopPieceKind, Resolution, StrategicPhase,
};
use meeple_bots_catalog::{
    AgentConfig, CatalogAction, CatalogBoopPieceKind, CatalogBoopResolution, CatalogError,
    CatalogGemstoneSacrifice, CatalogMatchReport, CatalogPieceKind, CatalogPowerSource,
    CatalogSpirit, CatalogSpiritsAction, CatalogTraceAnalysis, CatalogTurnPhase, ConfiguredAgent,
    ConfiguredRolloutPolicy, ConfiguredSelectionBias, EvaluationConfig, EvaluatorConfig, GameId,
    MatchConfig, MctsAgentConfig, MctsConfig, RecordedMove, RolloutConditionConfig,
    RolloutPolicyConfig, SearchBudget, analyze_seeded_trace, benchmark_mcts_agent,
    configured_boop_mcts, configured_connect_four_mcts, configured_spirits_of_the_forest_mcts,
    configured_tic_tac_toe_mcts, evaluate_game, run_boop_match_with_observer,
    run_boop_match_with_trace, run_connect_four_match_with_observer,
    run_connect_four_match_with_trace, run_spirits_of_the_forest_match_with_observer,
    run_spirits_of_the_forest_match_with_trace, run_tic_tac_toe_match_with_observer,
    run_tic_tac_toe_match_with_trace, spirits_of_the_forest_game,
};
use meeple_bots_connect_four::{ConnectFour, ConnectFourAction};
use meeple_bots_core::{
    Agent, AgentDecisionStats, AgentError, DecisionContext, Game, PlayerId, RandomSource,
};
use meeple_bots_simulation::{DecisionTiming, MatchObserver};
use meeple_bots_spirits_of_the_forest::{
    ForestPosition, GemstoneSacrifice, PowerSource, ScoringCategory, Spirit, SpiritsOfTheForest,
    SpiritsOfTheForestAction, SpiritsOfTheForestState, SpiritsReplayAnalysis, SpiritsStateMetrics,
    TurnPhase,
};
use meeple_bots_tic_tac_toe::{TicTacToe, TicTacToeAction};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule};

enum PythonAgentConfig {
    Automated(AgentConfig),
    Human {
        selector: Py<PyAny>,
        observer: Option<Py<PyAny>>,
    },
}

#[pyclass(name = "AgentConfig", frozen)]
struct PyAgentConfig {
    inner: PythonAgentConfig,
}

#[pymethods]
impl PyAgentConfig {
    #[staticmethod]
    #[pyo3(signature = (iterations=None, exploration=std::f64::consts::SQRT_2, time_budget=None, selection_policy="uct", tree_reuse=false))]
    fn so_ismcts(
        iterations: Option<u32>,
        exploration: f64,
        time_budget: Option<f64>,
        selection_policy: &str,
        tree_reuse: bool,
    ) -> PyResult<Self> {
        let config = meeple_bots_so_ismcts::SoIsmctsConfig {
            budget: parse_search_budget(iterations, time_budget)?,
            exploration,
            tree_reuse,
            selection_policy: parse_bandit_policy(selection_policy)?,
        };
        config
            .validate()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: PythonAgentConfig::Automated(AgentConfig::SoIsmcts(config)),
        })
    }
    #[staticmethod]
    fn random() -> Self {
        Self {
            inner: PythonAgentConfig::Automated(AgentConfig::Random),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (
        iterations=None,
        exploration=std::f64::consts::SQRT_2,
        rollout_depth=256,
        cutoff_evaluator="neutral",
        cutoff_heuristic=None,
        cutoff_params=None,
        rollout_policy="uniform_random",
        rollout_evaluator=None,
        rollout_heuristic=None,
        rollout_params=None,
        rollout_epsilon=None,
        time_budget=None,
        rollout_condition_phase=None,
        fallback_rollout_policy=None,
        fallback_rollout_evaluator=None,
        fallback_rollout_heuristic=None,
        fallback_rollout_params=None,
        fallback_rollout_epsilon=None,
        progressive_bias_weight=None,
        progressive_bias_evaluator=None,
        progressive_bias_heuristic=None,
        progressive_bias_params=None,
        progressive_bias_condition_phase=None,
        root_diagnostics=false,
        tree_reuse=false,
        transpositions=false,
        selection_policy="uct",
        rave_equivalence=1000,
        progressive_widening=false,
        progressive_widening_k=1.5,
        progressive_widening_alpha=0.5,
        progressive_widening_expansion="random",
    ))]
    // Preserve the Python keyword-argument interface.
    #[allow(clippy::too_many_arguments)]
    fn mcts(
        iterations: Option<u32>,
        exploration: f64,
        rollout_depth: u32,
        cutoff_evaluator: &str,
        cutoff_heuristic: Option<u32>,
        cutoff_params: Option<BTreeMap<String, f64>>,
        rollout_policy: &str,
        rollout_evaluator: Option<&str>,
        rollout_heuristic: Option<u32>,
        rollout_params: Option<BTreeMap<String, f64>>,
        rollout_epsilon: Option<f64>,
        time_budget: Option<f64>,
        rollout_condition_phase: Option<&str>,
        fallback_rollout_policy: Option<&str>,
        fallback_rollout_evaluator: Option<&str>,
        fallback_rollout_heuristic: Option<u32>,
        fallback_rollout_params: Option<BTreeMap<String, f64>>,
        fallback_rollout_epsilon: Option<f64>,
        progressive_bias_weight: Option<f64>,
        progressive_bias_evaluator: Option<&str>,
        progressive_bias_heuristic: Option<u32>,
        progressive_bias_params: Option<BTreeMap<String, f64>>,
        progressive_bias_condition_phase: Option<&str>,
        root_diagnostics: bool,
        tree_reuse: bool,
        transpositions: bool,
        selection_policy: &str,
        rave_equivalence: u32,
        progressive_widening: bool,
        progressive_widening_k: f64,
        progressive_widening_alpha: f64,
        progressive_widening_expansion: &str,
    ) -> PyResult<Self> {
        if !exploration.is_finite() || exploration < 0.0 {
            return Err(PyValueError::new_err(
                "exploration must be finite and non-negative",
            ));
        }
        if rollout_depth == 0 {
            return Err(PyValueError::new_err(
                "rollout_depth must be greater than zero",
            ));
        }

        Ok(Self {
            inner: PythonAgentConfig::Automated(AgentConfig::Mcts(MctsAgentConfig {
                search: MctsConfig {
                    progressive_widening: parse_progressive_widening(
                        progressive_widening,
                        progressive_widening_k,
                        progressive_widening_alpha,
                        progressive_widening_expansion,
                    )?,
                    selection_policy: parse_selection_policy(selection_policy, rave_equivalence)?,
                    budget: parse_search_budget(iterations, time_budget)?,
                    exploration,
                    rollout_depth,
                    rollout_policy: parse_configured_rollout_policy(
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
                    )?,
                },
                cutoff_evaluator: parse_evaluator(
                    cutoff_evaluator,
                    cutoff_heuristic,
                    cutoff_params,
                )?,
                progressive_bias: parse_selection_bias(
                    progressive_bias_weight,
                    progressive_bias_evaluator,
                    progressive_bias_heuristic,
                    progressive_bias_params,
                    progressive_bias_condition_phase,
                )?,
                root_diagnostics,
                tree_reuse,
                transpositions,
            })),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (selector, observer=None))]
    fn human(py: Python<'_>, selector: Py<PyAny>, observer: Option<Py<PyAny>>) -> PyResult<Self> {
        if !selector.bind(py).is_callable() {
            return Err(PyValueError::new_err("human selector must be callable"));
        }
        if observer
            .as_ref()
            .is_some_and(|callback| !callback.bind(py).is_callable())
        {
            return Err(PyValueError::new_err("human observer must be callable"));
        }
        Ok(Self {
            inner: PythonAgentConfig::Human { selector, observer },
        })
    }
}

struct PythonHumanAgent<'a> {
    selector: &'a Py<PyAny>,
    observer: Option<&'a Py<PyAny>>,
}

struct PythonTicTacToeMatchObserver<'a> {
    callback: &'a Py<PyAny>,
    error: Option<String>,
}

struct PythonConnectFourMatchObserver<'a> {
    callback: &'a Py<PyAny>,
    error: Option<String>,
}

struct PythonBoopMatchObserver<'a> {
    callback: &'a Py<PyAny>,
    error: Option<String>,
}

struct PythonSpiritsMatchObserver<'a> {
    callback: &'a Py<PyAny>,
    error: Option<String>,
}

enum PythonParticipant<'a, M> {
    Human(PythonHumanAgent<'a>),
    Automated(ConfiguredAgent<M>),
}

impl<'a, G: Game, M: Agent<G>> Agent<G> for PythonParticipant<'a, M>
where
    PythonHumanAgent<'a>: Agent<G>,
{
    fn on_match_start(&mut self, game: &G, state: &G::State, player: PlayerId) {
        match self {
            Self::Human(agent) => {
                <PythonHumanAgent<'_> as Agent<G>>::on_match_start(agent, game, state, player)
            }
            Self::Automated(agent) => agent.on_match_start(game, state, player),
        }
    }

    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, G>,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        match self {
            Self::Human(agent) => agent.select_action(decision, rng),
            Self::Automated(agent) => agent.select_action(decision, rng),
        }
    }

    fn last_decision_stats(&self) -> AgentDecisionStats {
        match self {
            Self::Human(agent) => <PythonHumanAgent<'_> as Agent<G>>::last_decision_stats(agent),
            Self::Automated(agent) => agent.last_decision_stats(),
        }
    }

    fn on_action_applied(
        &mut self,
        game: &G,
        state: &G::State,
        player: PlayerId,
        action: &G::Action,
    ) {
        match self {
            Self::Human(agent) => <PythonHumanAgent<'_> as Agent<G>>::on_action_applied(
                agent, game, state, player, action,
            ),
            Self::Automated(agent) => agent.on_action_applied(game, state, player, action),
        }
    }

    fn on_match_end(&mut self, game: &G, state: &G::State) {
        match self {
            Self::Human(agent) => {
                <PythonHumanAgent<'_> as Agent<G>>::on_match_end(agent, game, state)
            }
            Self::Automated(agent) => agent.on_match_end(game, state),
        }
    }
}

impl MatchObserver<TicTacToe> for PythonTicTacToeMatchObserver<'_> {
    fn measures_decision_time(&self) -> bool {
        true
    }

    fn on_action(
        &mut self,
        _game: &TicTacToe,
        state: &<TicTacToe as Game>::State,
        player: PlayerId,
        action: &TicTacToeAction,
        decision_time: DecisionTiming,
        decision_stats: AgentDecisionStats,
    ) {
        if self.error.is_some() {
            return;
        }
        let board: Vec<_> = state
            .board()
            .iter()
            .map(|cell| cell.map(|occupant| occupant.index()))
            .collect();
        if let Err(error) = Python::attach(|py| -> PyResult<()> {
            self.callback.bind(py).call1((
                player.index(),
                board,
                (action.row(), action.column()),
                decision_time.total().as_secs_f64(),
                decision_stats.search_iterations,
                decision_stats.search_nodes,
            ))?;
            Ok(())
        }) {
            self.error = Some(error.to_string());
        }
    }
}

struct PythonConnect6MatchObserver<'a> {
    callback: &'a Py<PyAny>,
    error: Option<String>,
}
impl MatchObserver<Connect6> for PythonConnect6MatchObserver<'_> {
    fn measures_decision_time(&self) -> bool {
        true
    }

    fn on_action(
        &mut self,
        _game: &Connect6,
        state: &<Connect6 as Game>::State,
        player: PlayerId,
        action: &Connect6Action,
        decision_time: DecisionTiming,
        decision_stats: AgentDecisionStats,
    ) {
        if self.error.is_some() {
            return;
        }
        let board: Vec<_> = state
            .board()
            .iter()
            .map(|cell| {
                if *cell == 0 {
                    None
                } else {
                    Some((*cell - 1) as usize)
                }
            })
            .collect();
        if let Err(error) = Python::attach(|py| -> PyResult<()> {
            self.callback.bind(py).call1((
                player.index(),
                board,
                match action {
                    Connect6Action::Place(p) => *p,
                },
                decision_time.total().as_secs_f64(),
                decision_stats.search_iterations,
                decision_stats.search_nodes,
            ))?;
            Ok(())
        }) {
            self.error = Some(error.to_string());
        }
    }
}

impl MatchObserver<ConnectFour> for PythonConnectFourMatchObserver<'_> {
    fn measures_decision_time(&self) -> bool {
        true
    }

    fn on_action(
        &mut self,
        _game: &ConnectFour,
        state: &<ConnectFour as Game>::State,
        player: PlayerId,
        action: &ConnectFourAction,
        decision_time: DecisionTiming,
        decision_stats: AgentDecisionStats,
    ) {
        if self.error.is_some() {
            return;
        }
        let board: Vec<_> = state
            .board()
            .iter()
            .map(|cell| cell.map(|occupant| occupant.index()))
            .collect();
        if let Err(error) = Python::attach(|py| -> PyResult<()> {
            self.callback.bind(py).call1((
                player.index(),
                board,
                action.column(),
                decision_time.total().as_secs_f64(),
                decision_stats.search_iterations,
                decision_stats.search_nodes,
            ))?;
            Ok(())
        }) {
            self.error = Some(error.to_string());
        }
    }
}

impl MatchObserver<Boop> for PythonBoopMatchObserver<'_> {
    fn measures_decision_time(&self) -> bool {
        true
    }

    fn on_action(
        &mut self,
        _game: &Boop,
        state: &<Boop as Game>::State,
        player: PlayerId,
        action: &BoopAction,
        decision_time: DecisionTiming,
        decision_stats: AgentDecisionStats,
    ) {
        if self.error.is_some() {
            return;
        }
        let board: Vec<_> = state
            .board()
            .iter()
            .map(|cell| {
                cell.map(|piece| {
                    (
                        piece.owner().index(),
                        boop_piece_name(piece.kind()).to_owned(),
                    )
                })
            })
            .collect();
        let pools: Vec<_> = state
            .pools()
            .iter()
            .map(|pool| (pool.kittens(), pool.cats()))
            .collect();
        if let Err(error) = Python::attach(|py| -> PyResult<()> {
            self.callback.bind(py).call1((
                player.index(),
                board,
                pools,
                native_boop_action(action),
                decision_time.total().as_secs_f64(),
                decision_stats.search_iterations,
                decision_stats.search_nodes,
            ))?;
            Ok(())
        }) {
            self.error = Some(error.to_string());
        }
    }
}

impl MatchObserver<SpiritsOfTheForest> for PythonSpiritsMatchObserver<'_> {
    fn measures_decision_time(&self) -> bool {
        true
    }

    fn on_action(
        &mut self,
        game: &SpiritsOfTheForest,
        state: &SpiritsOfTheForestState,
        player: PlayerId,
        action: &SpiritsOfTheForestAction,
        decision_time: DecisionTiming,
        decision_stats: AgentDecisionStats,
    ) {
        if self.error.is_some() {
            return;
        }
        if let Err(error) = Python::attach(|py| -> PyResult<()> {
            self.callback.bind(py).call1((
                player.index(),
                native_spirits_state(game, state),
                native_spirits_action(*action),
                decision_time.total().as_secs_f64(),
                decision_stats.search_iterations,
                decision_stats.search_nodes,
            ))?;
            Ok(())
        }) {
            self.error = Some(error.to_string());
        }
    }
}

#[pyfunction(name = "evaluate_game")]
#[pyo3(signature = (game, samples=128, max_depth=256, seed=0, target_time=5.0, game_params=None))]
fn py_evaluate_game(
    py: Python<'_>,
    game: &str,
    samples: u32,
    max_depth: u32,
    seed: u64,
    target_time: f64,
    game_params: Option<BTreeMap<String, i64>>,
) -> PyResult<Py<PyDict>> {
    let game = parse_configured_game(game, game_params)?;
    let samples = NonZeroU32::new(samples)
        .ok_or_else(|| PyValueError::new_err("samples must be greater than zero"))?;
    let max_depth = NonZeroU32::new(max_depth)
        .ok_or_else(|| PyValueError::new_err("max_depth must be greater than zero"))?;
    if !target_time.is_finite() || target_time <= 0.0 || target_time > 3_600.0 {
        return Err(PyValueError::new_err(
            "target_time must be finite, greater than zero, and at most 3600 seconds",
        ));
    }
    let report = evaluate_game(
        game,
        EvaluationConfig {
            samples,
            max_depth,
            seed,
            target_time: Duration::from_secs_f64(target_time),
        },
    )
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

    let serialized = PyDict::new(py);
    serialized.set_item(
        "structural",
        evaluation::structural_dict(py, &report.structural)?,
    )?;
    serialized.set_item("samples", report.samples)?;
    serialized.set_item("max_depth", report.max_depth)?;
    serialized.set_item("terminal_rate", report.terminal_rate)?;
    serialized.set_item("initial_legal_actions", report.initial_legal_actions)?;
    serialized.set_item(
        "effective_branching_factor",
        report.effective_branching_factor,
    )?;
    serialized.set_item(
        "player_turn_choice_product_log10",
        report.player_turn_choice_product_log10,
    )?;
    serialized.set_item("depth_p50", report.depth_p50)?;
    serialized.set_item("estimated_depth", report.estimated_depth)?;
    serialized.set_item("player_turn_depth_p50", report.player_turn_depth_p50)?;
    serialized.set_item("player_turn_depth_p95", report.player_turn_depth_p95)?;
    serialized.set_item("player_changes_p50", report.player_changes_p50)?;
    serialized.set_item("player_changes_p95", report.player_changes_p95)?;
    serialized.set_item(
        "actions_per_player_turn_mean",
        report.actions_per_player_turn_mean,
    )?;
    serialized.set_item(
        "actions_per_player_turn_p95",
        report.actions_per_player_turn_p95,
    )?;
    serialized.set_item(
        "actions_per_player_turn_max",
        report.actions_per_player_turn_max,
    )?;
    serialized.set_item("depth_is_lower_bound", report.depth_is_lower_bound)?;
    serialized.set_item("estimated_tree_log10", report.estimated_tree_log10)?;
    serialized.set_item("calibration_positions", report.calibration_positions)?;
    serialized.set_item("target_time_seconds", report.target_time_seconds)?;

    let costs = PyList::empty(py);
    for cost in &report.rollout_costs {
        let item = PyDict::new(py);
        item.set_item("rollout_depth", cost.rollout_depth)?;
        item.set_item("approximate_player_turns", cost.approximate_player_turns)?;
        item.set_item(
            "milliseconds_per_iteration",
            cost.milliseconds_per_iteration,
        )?;
        let budgets = PyList::empty(py);
        for budget in &cost.iteration_budgets {
            let entry = PyDict::new(py);
            entry.set_item("seconds", budget.seconds)?;
            entry.set_item("iterations", budget.iterations)?;
            budgets.append(entry)?;
        }
        item.set_item("iteration_budgets", budgets)?;
        costs.append(item)?;
    }
    serialized.set_item("rollout_costs", costs)?;

    let experiments = PyList::empty(py);
    for experiment in &report.suggested_experiments {
        let item = PyDict::new(py);
        item.set_item("label", &experiment.label)?;
        item.set_item("iterations", experiment.iterations)?;
        item.set_item("iterations_capped", experiment.iterations_capped)?;
        item.set_item("rollout_depth", experiment.rollout_depth)?;
        item.set_item(
            "approximate_player_turns",
            experiment.approximate_player_turns,
        )?;
        item.set_item(
            "estimated_decision_time_ms",
            experiment.estimated_decision_time_ms,
        )?;
        experiments.append(item)?;
    }
    serialized.set_item("suggested_experiments", experiments)?;
    serialized.set_item(
        "recommended_rollout_depth",
        report.recommended_rollout_depth,
    )?;
    serialized.set_item("recommended_iterations", report.recommended_iterations)?;
    serialized.set_item("iterations_capped", report.iterations_capped)?;
    serialized.set_item(
        "milliseconds_per_iteration",
        report.milliseconds_per_iteration,
    )?;
    serialized.set_item(
        "estimated_decision_time_ms",
        report.estimated_decision_time_ms,
    )?;
    Ok(serialized.unbind())
}

#[pyfunction(name = "benchmark_mcts_agent")]
#[pyo3(signature = (
    game,
    iterations,
    time_budget,
    exploration,
    rollout_depth,
    cutoff_evaluator,
    cutoff_heuristic,
    cutoff_params,
    rollout_policy,
    rollout_evaluator,
    rollout_heuristic,
    rollout_params,
    rollout_epsilon,
    median_depth,
    seed=0,
    rollout_condition_phase=None,
    fallback_rollout_policy=None,
    fallback_rollout_evaluator=None,
    fallback_rollout_heuristic=None,
    fallback_rollout_params=None,
    fallback_rollout_epsilon=None,
    progressive_bias_weight=None,
    progressive_bias_evaluator=None,
    progressive_bias_heuristic=None,
    progressive_bias_params=None,
    progressive_bias_condition_phase=None,
    root_diagnostics=false,
    tree_reuse=false,
    transpositions=false,
    selection_policy="uct",
    game_params=None,
    rave_equivalence=1000,
        progressive_widening=false,
        progressive_widening_k=1.5,
        progressive_widening_alpha=0.5,
        progressive_widening_expansion="random",
))]
// Preserve the Python keyword-argument interface.
#[allow(clippy::too_many_arguments)]
fn py_benchmark_mcts_agent(
    py: Python<'_>,
    game: &str,
    iterations: Option<u32>,
    time_budget: Option<f64>,
    exploration: f64,
    rollout_depth: u32,
    cutoff_evaluator: &str,
    cutoff_heuristic: Option<u32>,
    cutoff_params: Option<BTreeMap<String, f64>>,
    rollout_policy: &str,
    rollout_evaluator: Option<&str>,
    rollout_heuristic: Option<u32>,
    rollout_params: Option<BTreeMap<String, f64>>,
    rollout_epsilon: Option<f64>,
    median_depth: u32,
    seed: u64,
    rollout_condition_phase: Option<&str>,
    fallback_rollout_policy: Option<&str>,
    fallback_rollout_evaluator: Option<&str>,
    fallback_rollout_heuristic: Option<u32>,
    fallback_rollout_params: Option<BTreeMap<String, f64>>,
    fallback_rollout_epsilon: Option<f64>,
    progressive_bias_weight: Option<f64>,
    progressive_bias_evaluator: Option<&str>,
    progressive_bias_heuristic: Option<u32>,
    progressive_bias_params: Option<BTreeMap<String, f64>>,
    progressive_bias_condition_phase: Option<&str>,
    root_diagnostics: bool,
    tree_reuse: bool,
    transpositions: bool,
    selection_policy: &str,
    game_params: Option<BTreeMap<String, i64>>,
    rave_equivalence: u32,
    progressive_widening: bool,
    progressive_widening_k: f64,
    progressive_widening_alpha: f64,
    progressive_widening_expansion: &str,
) -> PyResult<Py<PyDict>> {
    let game = parse_configured_game(game, game_params)?;
    if !exploration.is_finite() || exploration < 0.0 {
        return Err(PyValueError::new_err(
            "exploration must be finite and non-negative",
        ));
    }
    if rollout_depth == 0 {
        return Err(PyValueError::new_err(
            "rollout_depth must be greater than zero",
        ));
    }
    let report = benchmark_mcts_agent(
        game,
        MctsAgentConfig {
            search: MctsConfig {
                progressive_widening: parse_progressive_widening(
                    progressive_widening,
                    progressive_widening_k,
                    progressive_widening_alpha,
                    progressive_widening_expansion,
                )?,
                selection_policy: parse_selection_policy(selection_policy, rave_equivalence)?,
                budget: parse_search_budget(iterations, time_budget)?,
                exploration,
                rollout_depth,
                rollout_policy: parse_configured_rollout_policy(
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
                )?,
            },
            cutoff_evaluator: parse_evaluator(cutoff_evaluator, cutoff_heuristic, cutoff_params)?,
            progressive_bias: parse_selection_bias(
                progressive_bias_weight,
                progressive_bias_evaluator,
                progressive_bias_heuristic,
                progressive_bias_params,
                progressive_bias_condition_phase,
            )?,
            root_diagnostics,
            tree_reuse,
            transpositions,
        },
        median_depth,
        seed,
    )
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

    let serialized = PyDict::new(py);
    serialized.set_item("sampled_positions", report.sampled_positions)?;
    serialized.set_item("decision_time_mean_ms", report.decision_time_mean_ms)?;
    serialized.set_item("decision_time_p50_ms", report.decision_time_p50_ms)?;
    serialized.set_item("decision_time_p95_ms", report.decision_time_p95_ms)?;
    serialized.set_item("decision_time_max_ms", report.decision_time_max_ms)?;
    serialized.set_item(
        "milliseconds_per_iteration",
        report.milliseconds_per_iteration,
    )?;
    serialized.set_item("maximum_decision_horizon", report.maximum_decision_horizon)?;
    let position_timings = PyList::empty(py);
    for timing in report.position_timings {
        let item = PyDict::new(py);
        item.set_item("sampled_ply", timing.sampled_ply)?;
        item.set_item("milliseconds", timing.milliseconds)?;
        item.set_item("iterations", timing.iterations)?;
        item.set_item("nodes", timing.nodes)?;
        item.set_item("legal_actions", timing.legal_actions)?;
        item.set_item("terminal_simulations", timing.terminal_simulations)?;
        item.set_item("cutoff_simulations", timing.cutoff_simulations)?;
        item.set_item("root_expansion", timing.root_expansion)?;
        item.set_item("widening_expansions", timing.widening_expansions)?;
        item.set_item("root_visits", timing.root_visits)?;
        position_timings.append(item)?;
    }
    serialized.set_item("position_timings", position_timings)?;
    Ok(serialized.unbind())
}

fn parse_search_budget(
    iterations: Option<u32>,
    time_budget: Option<f64>,
) -> PyResult<SearchBudget> {
    match (iterations, time_budget) {
        (Some(_), Some(_)) => Err(PyValueError::new_err(
            "iterations and time_budget are mutually exclusive",
        )),
        (Some(iterations), None) => NonZeroU32::new(iterations)
            .map(SearchBudget::Iterations)
            .ok_or_else(|| PyValueError::new_err("iterations must be greater than zero")),
        (None, Some(seconds)) => {
            if !seconds.is_finite() || seconds <= 0.0 {
                return Err(PyValueError::new_err(
                    "time_budget must be finite and greater than zero",
                ));
            }
            Duration::try_from_secs_f64(seconds)
                .map(SearchBudget::Time)
                .map_err(|_| PyValueError::new_err("time_budget is too large"))
        }
        (None, None) => Ok(SearchBudget::default()),
    }
}

fn parse_evaluator(
    kind: &str,
    heuristic: Option<u32>,
    parameters: Option<BTreeMap<String, f64>>,
) -> PyResult<EvaluatorConfig> {
    match kind {
        "neutral" => {
            if heuristic.is_some() || parameters.is_some() {
                return Err(PyValueError::new_err(
                    "neutral evaluator does not accept a heuristic index or params",
                ));
            }
            Ok(EvaluatorConfig::Neutral)
        }
        "game_heuristic" => heuristic
            .map(|index| EvaluatorConfig::GameHeuristic {
                index,
                parameters: parameters.unwrap_or_default(),
            })
            .ok_or_else(|| {
                PyValueError::new_err("game_heuristic evaluator requires a heuristic index")
            }),
        _ => Err(PyValueError::new_err(format!(
            "unknown evaluator {kind}; expected neutral or game_heuristic"
        ))),
    }
}

fn parse_selection_bias(
    weight: Option<f64>,
    evaluator: Option<&str>,
    heuristic: Option<u32>,
    parameters: Option<BTreeMap<String, f64>>,
    condition_phase: Option<&str>,
) -> PyResult<ConfiguredSelectionBias> {
    if weight.is_none()
        && evaluator.is_none()
        && heuristic.is_none()
        && parameters.is_none()
        && condition_phase.is_none()
    {
        return Ok(ConfiguredSelectionBias::None);
    }
    let weight =
        weight.ok_or_else(|| PyValueError::new_err("progressive bias requires a weight"))?;
    if !weight.is_finite() || weight < 0.0 {
        return Err(PyValueError::new_err(
            "progressive bias weight must be finite and non-negative",
        ));
    }
    let evaluator =
        evaluator.ok_or_else(|| PyValueError::new_err("progressive bias requires an evaluator"))?;
    let condition = condition_phase
        .map(parse_turn_phase)
        .transpose()?
        .map(RolloutConditionConfig::TurnPhase);
    Ok(ConfiguredSelectionBias::Progressive {
        weight,
        evaluator: parse_evaluator(evaluator, heuristic, parameters)?,
        condition,
    })
}

fn parse_turn_phase(phase: &str) -> PyResult<CatalogTurnPhase> {
    match phase {
        "choose" => Ok(CatalogTurnPhase::Choose),
        "continue" => Ok(CatalogTurnPhase::Continue),
        "collect" => Ok(CatalogTurnPhase::Collect),
        "place_gemstone" | "gemstones" => Ok(CatalogTurnPhase::PlaceGemstone),
        _ => Err(PyValueError::new_err(format!(
            "unknown turn phase {phase}; expected choose, continue, collect or place_gemstone"
        ))),
    }
}

// Mirrors the primary/fallback fields of the native Python entry points.
#[allow(clippy::too_many_arguments)]
fn parse_configured_rollout_policy(
    policy: &str,
    evaluator: Option<&str>,
    heuristic: Option<u32>,
    parameters: Option<BTreeMap<String, f64>>,
    epsilon: Option<f64>,
    condition_phase: Option<&str>,
    fallback_policy: Option<&str>,
    fallback_evaluator: Option<&str>,
    fallback_heuristic: Option<u32>,
    fallback_parameters: Option<BTreeMap<String, f64>>,
    fallback_epsilon: Option<f64>,
) -> PyResult<ConfiguredRolloutPolicy> {
    let primary = parse_rollout_policy(policy, evaluator, heuristic, parameters, epsilon)?;
    let Some(phase) = condition_phase else {
        if fallback_policy.is_some()
            || fallback_evaluator.is_some()
            || fallback_heuristic.is_some()
            || fallback_parameters.is_some()
            || fallback_epsilon.is_some()
        {
            return Err(PyValueError::new_err(
                "fallback rollout fields require a rollout condition",
            ));
        }
        return Ok(ConfiguredRolloutPolicy::Standard(primary));
    };
    let phase = parse_turn_phase(phase)?;
    let fallback_policy = fallback_policy.ok_or_else(|| {
        PyValueError::new_err("conditional rollout requires a fallback rollout policy")
    })?;
    let fallback = parse_rollout_policy(
        fallback_policy,
        fallback_evaluator,
        fallback_heuristic,
        fallback_parameters,
        fallback_epsilon,
    )?;
    Ok(ConfiguredRolloutPolicy::Conditional {
        condition: RolloutConditionConfig::TurnPhase(phase),
        primary,
        fallback,
    })
}

fn parse_rollout_policy(
    policy: &str,
    evaluator: Option<&str>,
    heuristic: Option<u32>,
    parameters: Option<BTreeMap<String, f64>>,
    epsilon: Option<f64>,
) -> PyResult<RolloutPolicyConfig<EvaluatorConfig>> {
    match policy {
        "uniform_random" => {
            if evaluator.is_some()
                || heuristic.is_some()
                || parameters.is_some()
                || epsilon.is_some()
            {
                return Err(PyValueError::new_err(
                    "uniform_random rollout does not accept an evaluator or epsilon",
                ));
            }
            Ok(RolloutPolicyConfig::UniformRandom)
        }
        "mast" => {
            if evaluator.is_some() || heuristic.is_some() || parameters.is_some() {
                return Err(PyValueError::new_err(
                    "mast rollout does not accept an evaluator",
                ));
            }
            let epsilon = epsilon.unwrap_or(0.1);
            if !epsilon.is_finite() || !(0.0..=1.0).contains(&epsilon) {
                return Err(PyValueError::new_err(
                    "rollout_epsilon must be finite and between 0.0 and 1.0",
                ));
            }
            Ok(RolloutPolicyConfig::Mast { epsilon })
        }
        "greedy" => {
            if epsilon.is_some() {
                return Err(PyValueError::new_err(
                    "greedy rollout does not accept epsilon",
                ));
            }
            let evaluator = evaluator
                .ok_or_else(|| PyValueError::new_err("greedy rollout requires an evaluator"))?;
            Ok(RolloutPolicyConfig::Greedy {
                evaluator: parse_evaluator(evaluator, heuristic, parameters)?,
            })
        }
        "epsilon_greedy" | "epsilon_greedy_heuristic" => {
            let evaluator = evaluator.ok_or_else(|| {
                PyValueError::new_err("epsilon_greedy rollout requires an evaluator")
            })?;
            let epsilon = epsilon
                .ok_or_else(|| PyValueError::new_err("epsilon_greedy requires rollout_epsilon"))?;
            if !epsilon.is_finite() || !(0.0..=1.0).contains(&epsilon) {
                return Err(PyValueError::new_err(
                    "rollout_epsilon must be finite and between 0.0 and 1.0",
                ));
            }
            Ok(RolloutPolicyConfig::EpsilonGreedy {
                epsilon,
                evaluator: parse_evaluator(evaluator, heuristic, parameters)?,
            })
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown rollout policy {policy}; expected uniform_random, greedy, epsilon_greedy, or mast"
        ))),
    }
}

impl Agent<TicTacToe> for PythonHumanAgent<'_> {
    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, TicTacToe>,
        _rng: &mut R,
    ) -> Result<TicTacToeAction, AgentError> {
        let player = decision.player().index();
        let board: Vec<_> = decision
            .state()
            .board()
            .iter()
            .map(|cell| cell.map(|occupant| occupant.index()))
            .collect();
        let legal_actions: Vec<_> = decision.legal_actions().collect();
        let legal_coordinates: Vec<_> = legal_actions
            .iter()
            .map(|action| (action.row(), action.column()))
            .collect();

        let (row, column): (u8, u8) = Python::attach(|py| {
            self.selector
                .bind(py)
                .call1((player, board, legal_coordinates))?
                .extract()
        })
        .map_err(|error| AgentError::message(format!("human selector failed: {error}")))?;
        let action = TicTacToeAction::new(row, column)
            .ok_or_else(|| AgentError::message("human selected a cell outside the board"))?;
        if !legal_actions.contains(&action) {
            return Err(AgentError::message(
                "human selected a cell that is not currently legal",
            ));
        }
        if let Some(observer) = self.observer {
            let mut state = decision.state().clone();
            decision
                .game()
                .apply_action(&mut state, &action)
                .map_err(|error| {
                    AgentError::message(format!("failed to preview human action: {error}"))
                })?;
            let board: Vec<_> = state
                .board()
                .iter()
                .map(|cell| cell.map(|occupant| occupant.index()))
                .collect();
            Python::attach(|py| -> PyResult<()> {
                observer.bind(py).call1((
                    player,
                    board,
                    py.None(),
                    (action.row(), action.column()),
                ))?;
                Ok(())
            })
            .map_err(|error| AgentError::message(format!("human observer failed: {error}")))?;
        }
        Ok(action)
    }
}

impl Agent<Connect6> for PythonHumanAgent<'_> {
    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, Connect6>,
        _rng: &mut R,
    ) -> Result<Connect6Action, AgentError> {
        let player = decision.player().index();
        let board: Vec<_> = decision
            .state()
            .board()
            .iter()
            .map(|cell| {
                if *cell == 0 {
                    None
                } else {
                    Some((*cell - 1) as usize)
                }
            })
            .collect();
        let legal_actions: Vec<_> = decision.legal_actions().collect();
        let legal_columns: Vec<_> = legal_actions
            .iter()
            .map(|action| match action {
                Connect6Action::Place(p) => *p,
            })
            .collect();

        let column: usize = Python::attach(|py| {
            self.selector
                .bind(py)
                .call1((player, board, legal_columns))?
                .extract()
        })
        .map_err(|error| AgentError::message(format!("human selector failed: {error}")))?;
        let action = Connect6Action::Place(column);
        if !legal_actions.contains(&action) {
            return Err(AgentError::message(
                "human selected a column that is not currently legal",
            ));
        }
        if let Some(observer) = self.observer {
            let mut state = decision.state().clone();
            decision
                .game()
                .apply_action(&mut state, &action)
                .map_err(|error| {
                    AgentError::message(format!("failed to preview human action: {error}"))
                })?;
            let board: Vec<_> = state
                .board()
                .iter()
                .map(|cell| {
                    if *cell == 0 {
                        None
                    } else {
                        Some((*cell - 1) as usize)
                    }
                })
                .collect();
            Python::attach(|py| -> PyResult<()> {
                observer.bind(py).call1((
                    player,
                    board,
                    py.None(),
                    match action {
                        Connect6Action::Place(p) => p,
                    },
                ))?;
                Ok(())
            })
            .map_err(|error| AgentError::message(format!("human observer failed: {error}")))?;
        }
        Ok(action)
    }
}

impl Agent<ConnectFour> for PythonHumanAgent<'_> {
    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, ConnectFour>,
        _rng: &mut R,
    ) -> Result<ConnectFourAction, AgentError> {
        let player = decision.player().index();
        let board: Vec<_> = decision
            .state()
            .board()
            .iter()
            .map(|cell| cell.map(|occupant| occupant.index()))
            .collect();
        let legal_actions: Vec<_> = decision.legal_actions().collect();
        let legal_columns: Vec<_> = legal_actions.iter().map(|action| action.column()).collect();

        let column: u8 = Python::attach(|py| {
            self.selector
                .bind(py)
                .call1((player, board, legal_columns))?
                .extract()
        })
        .map_err(|error| AgentError::message(format!("human selector failed: {error}")))?;
        let action = ConnectFourAction::new(column)
            .ok_or_else(|| AgentError::message("human selected a column outside the board"))?;
        if !legal_actions.contains(&action) {
            return Err(AgentError::message(
                "human selected a column that is not currently legal",
            ));
        }
        if let Some(observer) = self.observer {
            let mut state = decision.state().clone();
            decision
                .game()
                .apply_action(&mut state, &action)
                .map_err(|error| {
                    AgentError::message(format!("failed to preview human action: {error}"))
                })?;
            let board: Vec<_> = state
                .board()
                .iter()
                .map(|cell| cell.map(|occupant| occupant.index()))
                .collect();
            Python::attach(|py| -> PyResult<()> {
                observer
                    .bind(py)
                    .call1((player, board, py.None(), action.column()))?;
                Ok(())
            })
            .map_err(|error| AgentError::message(format!("human observer failed: {error}")))?;
        }
        Ok(action)
    }
}

impl Agent<Boop> for PythonHumanAgent<'_> {
    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, Boop>,
        _rng: &mut R,
    ) -> Result<BoopAction, AgentError> {
        let player = decision.player().index();
        let board: Vec<_> = decision
            .state()
            .board()
            .iter()
            .map(|cell| {
                cell.map(|piece| {
                    (
                        piece.owner().index(),
                        boop_piece_name(piece.kind()).to_owned(),
                    )
                })
            })
            .collect();
        let pools: Vec<_> = decision
            .state()
            .pools()
            .iter()
            .map(|pool| (pool.kittens(), pool.cats()))
            .collect();
        let legal_actions: Vec<_> = decision.legal_actions().collect();
        let native_actions: Vec<_> = legal_actions.iter().map(native_boop_action).collect();

        let selected: usize = Python::attach(|py| {
            self.selector
                .bind(py)
                .call1((player, board, pools, native_actions))?
                .extract()
        })
        .map_err(|error| AgentError::message(format!("human selector failed: {error}")))?;
        let action = legal_actions.get(selected).copied().ok_or_else(|| {
            AgentError::message("human selected an action index that is not currently legal")
        })?;
        if let Some(observer) = self.observer {
            let mut state = decision.state().clone();
            decision
                .game()
                .apply_action(&mut state, &action)
                .map_err(|error| {
                    AgentError::message(format!("failed to preview human action: {error}"))
                })?;
            let board: Vec<_> = state
                .board()
                .iter()
                .map(|cell| {
                    cell.map(|piece| {
                        (
                            piece.owner().index(),
                            boop_piece_name(piece.kind()).to_owned(),
                        )
                    })
                })
                .collect();
            let pools: Vec<_> = state
                .pools()
                .iter()
                .map(|pool| (pool.kittens(), pool.cats()))
                .collect();
            Python::attach(|py| -> PyResult<()> {
                observer
                    .bind(py)
                    .call1((player, board, pools, native_boop_action(&action)))?;
                Ok(())
            })
            .map_err(|error| AgentError::message(format!("human observer failed: {error}")))?;
        }
        Ok(action)
    }
}

impl Agent<SpiritsOfTheForest> for PythonHumanAgent<'_> {
    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, SpiritsOfTheForest>,
        _rng: &mut R,
    ) -> Result<SpiritsOfTheForestAction, AgentError> {
        let player = decision.player().index();
        let legal_actions: Vec<_> = decision.legal_actions().collect();
        let native_actions: Vec<_> = legal_actions
            .iter()
            .copied()
            .map(native_spirits_action)
            .collect();
        let selected: usize = Python::attach(|py| {
            self.selector
                .bind(py)
                .call1((
                    player,
                    native_spirits_state(decision.game(), decision.state()),
                    native_actions,
                ))?
                .extract()
        })
        .map_err(|error| AgentError::message(format!("human selector failed: {error}")))?;
        let action = legal_actions.get(selected).copied().ok_or_else(|| {
            AgentError::message("human selected an action index that is not currently legal")
        })?;
        if let Some(observer) = self.observer {
            let mut state = decision.state().clone();
            decision
                .game()
                .apply_action(&mut state, &action)
                .map_err(|error| {
                    AgentError::message(format!("failed to preview human action: {error}"))
                })?;
            Python::attach(|py| -> PyResult<()> {
                observer.bind(py).call1((
                    player,
                    native_spirits_state(decision.game(), &state),
                    native_spirits_action(action),
                ))?;
                Ok(())
            })
            .map_err(|error| AgentError::message(format!("human observer failed: {error}")))?;
        }
        Ok(action)
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction(name = "run_match")]
#[pyo3(signature = (game, first, second, seed=0, max_plies=10_000, observer=None, game_params=None))]
fn py_run_match(
    py: Python<'_>,
    game: &str,
    first: PyRef<'_, PyAgentConfig>,
    second: PyRef<'_, PyAgentConfig>,
    seed: u64,
    max_plies: u32,
    observer: Option<Py<PyAny>>,
    game_params: Option<BTreeMap<String, i64>>,
) -> PyResult<Py<PyDict>> {
    let game = parse_configured_game(game, game_params)?;
    let max_plies = NonZeroU32::new(max_plies)
        .ok_or_else(|| PyValueError::new_err("max_plies must be greater than zero"))?;
    let config = MatchConfig::new(seed, max_plies);
    if let Some(observer) = observer.as_ref()
        && !observer.bind(py).is_callable()
    {
        return Err(PyValueError::new_err("match observer must be callable"));
    }
    let first = clone_python_agent_config(py, &first.inner);
    let second = clone_python_agent_config(py, &second.inner);
    let observer = observer.map(|callback| callback.clone_ref(py));
    let report =
        py.detach(move || run_python_match(game, &first, &second, observer.as_ref(), config))?;

    let result = PyDict::new(py);
    result.set_item("seed", report.seed)?;
    result.set_item("plies", report.plies)?;
    result.set_item("utilities", PyList::new(py, report.utilities)?)?;
    result.set_item("winner", report.winner)?;

    let final_board = PyList::empty(py);
    for piece in report.final_board {
        match piece {
            None => final_board.append(py.None())?,
            Some(piece) => {
                let serialized = PyDict::new(py);
                serialized.set_item("player", piece.player)?;
                serialized.set_item(
                    "kind",
                    match piece.kind {
                        CatalogPieceKind::Token => "token",
                        CatalogPieceKind::Kitten => "kitten",
                        CatalogPieceKind::Cat => "cat",
                    },
                )?;
                final_board.append(serialized)?;
            }
        }
    }
    result.set_item("final_board", final_board)?;
    match report.pools {
        None => result.set_item("pools", py.None())?,
        Some(pools) => {
            let serialized = PyList::empty(py);
            for pool in pools {
                let item = PyDict::new(py);
                item.set_item("kittens", pool.kittens)?;
                item.set_item("cats", pool.cats)?;
                serialized.append(item)?;
            }
            result.set_item("pools", serialized)?;
        }
    }
    match report.spirit_forest {
        None => result.set_item("spirit_forest", py.None())?,
        Some(forest) => {
            let serialized = PyList::empty(py);
            for tile in forest {
                match tile {
                    None => serialized.append(py.None())?,
                    Some(tile) => {
                        let item = PyDict::new(py);
                        item.set_item("spirit", catalog_spirit_name(tile.spirit))?;
                        item.set_item("spirit_symbols", tile.spirit_symbols)?;
                        item.set_item(
                            "power_source",
                            tile.power_source.map(catalog_power_source_name),
                        )?;
                        item.set_item("gemstone", tile.gemstone)?;
                        serialized.append(item)?;
                    }
                }
            }
            result.set_item("spirit_forest", serialized)?;
        }
    }
    match report.spirit_collections {
        None => result.set_item("spirit_collections", py.None())?,
        Some(collections) => {
            let serialized = PyList::empty(py);
            for collection in collections {
                let item = PyDict::new(py);
                item.set_item("spirit_symbols", collection.spirit_symbols)?;
                item.set_item("power_sources", collection.power_sources)?;
                item.set_item("tiles", collection.tiles)?;
                serialized.append(item)?;
            }
            result.set_item("spirit_collections", serialized)?;
        }
    }
    match report.gemstone_pools {
        None => result.set_item("gemstone_pools", py.None())?,
        Some(pools) => {
            let serialized = PyList::empty(py);
            for pool in pools {
                let item = PyDict::new(py);
                item.set_item("available", pool.available)?;
                item.set_item("placed", pool.placed)?;
                item.set_item("removed", pool.removed)?;
                serialized.append(item)?;
            }
            result.set_item("gemstone_pools", serialized)?;
        }
    }
    result.set_item("scores", report.scores)?;
    if let Some(state) = &report.splendor_state {
        result.set_item(
            "splendor_state",
            splendor::snapshot(py, &meeple_bots_catalog::splendor::game(report.seed), state)?,
        )?;
        let events = PyList::empty(py);
        for event in &report.chance_events {
            let CatalogAction::Splendor(outcome) = event.event else {
                return Err(PyRuntimeError::new_err("unexpected chance event game"));
            };
            let item = PyDict::new(py);
            item.set_item("after_ply", event.after_ply)?;
            item.set_item("outcome", splendor::action_dict(py, outcome)?)?;
            events.append(item)?;
        }
        result.set_item("chance_events", events)?;
    }
    if let Some(state) = &report.lost_cities_state {
        result.set_item("lost_cities_state", lost_cities::snapshot(py, state)?)?;
        let events = PyList::empty(py);
        for e in &report.chance_events {
            let CatalogAction::LostCities(a) = e.event else {
                return Err(PyRuntimeError::new_err("unexpected chance event"));
            };
            let item = PyDict::new(py);
            item.set_item("after_ply", e.after_ply)?;
            item.set_item("outcome", lost_cities::action_dict(py, a)?)?;
            events.append(item)?;
        }
        result.set_item("chance_events", events)?;
    }
    result.set_item(
        "unassigned_maintenance_seconds",
        report.unassigned_maintenance_seconds,
    )?;

    let moves = PyList::empty(py);
    for recorded in report.moves {
        let action = if let CatalogAction::LostCities(a) = recorded.action {
            lost_cities::action_dict(py, a)?.into_bound(py)
        } else if let CatalogAction::Splendor(a) = recorded.action {
            splendor::action_dict(py, a)?.into_bound(py)
        } else {
            PyDict::new(py)
        };
        match recorded.action {
            CatalogAction::Splendor(_) | CatalogAction::LostCities(_) => {}
            CatalogAction::Boop {
                piece,
                row,
                column,
                resolution,
            } => {
                action.set_item("type", "boop")?;
                action.set_item("piece", catalog_boop_piece_name(piece))?;
                action.set_item("row", row)?;
                action.set_item("column", column)?;
                let serialized_resolution = PyDict::new(py);
                match resolution {
                    CatalogBoopResolution::None => {
                        serialized_resolution.set_item("type", "none")?;
                    }
                    CatalogBoopResolution::Graduate { positions } => {
                        serialized_resolution.set_item("type", "graduate")?;
                        serialized_resolution.set_item("positions", positions)?;
                    }
                    CatalogBoopResolution::Recover { row, column } => {
                        serialized_resolution.set_item("type", "recover")?;
                        serialized_resolution.set_item("row", row)?;
                        serialized_resolution.set_item("column", column)?;
                    }
                }
                action.set_item("resolution", serialized_resolution)?;
            }
            CatalogAction::Connect6 { position } => {
                action.set_item("type", "connect6")?;
                action.set_item("position", position)?;
            }
            CatalogAction::ConnectFour { column } => {
                action.set_item("type", "connect_four")?;
                action.set_item("column", column)?;
            }
            CatalogAction::SpiritsOfTheForest(spirits_action) => {
                serialize_catalog_spirits_action(&action, spirits_action)?;
            }
            CatalogAction::TicTacToe { row, column } => {
                action.set_item("type", "tic_tac_toe")?;
                action.set_item("row", row)?;
                action.set_item("column", column)?;
            }
        }

        let movement = PyDict::new(py);
        movement.set_item("player", recorded.player)?;
        movement.set_item("action", action)?;
        movement.set_item("decision_seconds", recorded.decision_seconds)?;
        movement.set_item("selection_seconds", recorded.selection_seconds)?;
        movement.set_item("maintenance_seconds", recorded.maintenance_seconds)?;
        movement.set_item("search_iterations", recorded.search_iterations)?;
        movement.set_item("search_nodes", recorded.search_nodes)?;
        movement.set_item("terminal_simulations", recorded.terminal_simulations)?;
        movement.set_item("cutoff_simulations", recorded.cutoff_simulations)?;
        let root_actions = PyList::empty(py);
        for root_action in &recorded.root_actions {
            let item = PyDict::new(py);
            item.set_item("action_index", root_action.action_index)?;
            item.set_item("visits", root_action.visits)?;
            item.set_item("mean_utility", root_action.mean_utility)?;
            item.set_item("heuristic_value", root_action.heuristic_value)?;
            item.set_item("progressive_bias", root_action.progressive_bias)?;
            item.set_item("selected", root_action.selected)?;
            root_actions.append(item)?;
        }
        movement.set_item("root_actions", root_actions)?;
        if let Some(tree_reuse) = recorded.tree_reuse {
            let reuse = PyDict::new(py);
            reuse.set_item("transition_attempts", tree_reuse.transition_attempts)?;
            reuse.set_item("transition_hits", tree_reuse.transition_hits)?;
            reuse.set_item("transition_misses", tree_reuse.transition_misses)?;
            reuse.set_item("own_action_hits", tree_reuse.own_action_hits)?;
            reuse.set_item("opponent_action_hits", tree_reuse.opponent_action_hits)?;
            reuse.set_item("reused_root_visits", tree_reuse.reused_root_visits)?;
            reuse.set_item("reused_nodes", tree_reuse.reused_nodes)?;
            reuse.set_item("pruned_nodes", tree_reuse.pruned_nodes)?;
            reuse.set_item("resets", tree_reuse.resets)?;
            movement.set_item("tree_reuse", reuse)?;
        } else {
            movement.set_item("tree_reuse", py.None())?;
        }
        moves.append(movement)?;
    }
    result.set_item("moves", moves)?;

    Ok(result.unbind())
}

fn clone_python_agent_config(py: Python<'_>, configured: &PythonAgentConfig) -> PythonAgentConfig {
    match configured {
        PythonAgentConfig::Automated(agent) => PythonAgentConfig::Automated(agent.clone()),
        PythonAgentConfig::Human { selector, observer } => PythonAgentConfig::Human {
            selector: selector.clone_ref(py),
            observer: observer.as_ref().map(|callback| callback.clone_ref(py)),
        },
    }
}

fn run_python_match(
    game: GameId,
    first: &PythonAgentConfig,
    second: &PythonAgentConfig,
    observer: Option<&Py<PyAny>>,
    config: MatchConfig,
) -> PyResult<CatalogMatchReport> {
    if matches!(game, GameId::Splendor | GameId::LostCities) {
        if observer.is_some() {
            return Err(PyValueError::new_err(
                "Live observers are not supported for this game",
            ));
        }
        let (PythonAgentConfig::Automated(first), PythonAgentConfig::Automated(second)) =
            (first, second)
        else {
            return Err(PyValueError::new_err(
                "This game currently supports automated agents only",
            ));
        };
        return meeple_bots_catalog::run_match_with_trace(
            game,
            first.clone(),
            second.clone(),
            config,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()));
    }
    if let Some(observer) = observer {
        return match game {
            GameId::Connect6(size) => {
                run_observed_connect6_match(size, first, second, observer, config)
            }
            GameId::Splendor | GameId::LostCities => unreachable!("handled above"),
            GameId::Boop => run_observed_boop_match(first, second, observer, config),
            GameId::ConnectFour => run_observed_connect_four_match(first, second, observer, config),
            GameId::SpiritsOfTheForest => {
                run_observed_spirits_match(first, second, observer, config)
            }
            GameId::TicTacToe => run_observed_tic_tac_toe_match(first, second, observer, config),
        };
    }

    match game {
        GameId::Splendor | GameId::LostCities => unreachable!("handled above"),
        GameId::Connect6(size) => run_connect6_match_with_trace(
            size,
            &mut python_participant(first, configured_connect6_mcts)?,
            &mut python_participant(second, configured_connect6_mcts)?,
            config,
        ),
        GameId::Boop => run_boop_match_with_trace(
            &mut python_participant(first, configured_boop_mcts)?,
            &mut python_participant(second, configured_boop_mcts)?,
            config,
        ),
        GameId::ConnectFour => run_connect_four_match_with_trace(
            &mut python_participant(first, configured_connect_four_mcts)?,
            &mut python_participant(second, configured_connect_four_mcts)?,
            config,
        ),
        GameId::TicTacToe => run_tic_tac_toe_match_with_trace(
            &mut python_participant(first, configured_tic_tac_toe_mcts)?,
            &mut python_participant(second, configured_tic_tac_toe_mcts)?,
            config,
        ),
        GameId::SpiritsOfTheForest => run_spirits_of_the_forest_match_with_trace(
            &mut python_participant(first, configured_spirits_of_the_forest_mcts)?,
            &mut python_participant(second, configured_spirits_of_the_forest_mcts)?,
            config,
        ),
    }
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pyfunction(name = "analyze_trace", signature = (game, moves, seed=0, game_params=None))]
fn py_analyze_trace(
    py: Python<'_>,
    game: &str,
    moves: &Bound<'_, PyAny>,
    seed: u64,
    game_params: Option<BTreeMap<String, i64>>,
) -> PyResult<Py<PyDict>> {
    let game = parse_configured_game(game, game_params)?;
    let recorded = match game {
        GameId::Connect6(_) => moves
            .extract::<Vec<(u8, usize)>>()?
            .into_iter()
            .map(|(player, position)| replay_record(player, CatalogAction::Connect6 { position }))
            .collect(),
        GameId::Splendor | GameId::LostCities => {
            return Err(PyValueError::new_err(
                "Splendor replay requires chance events; use replay_splendor",
            ));
        }
        GameId::Boop => moves
            .extract::<Vec<(u8, NativeBoopAction)>>()?
            .into_iter()
            .map(|(player, action)| {
                Ok(replay_record(
                    player,
                    parse_native_catalog_boop_action(action)?,
                ))
            })
            .collect::<PyResult<Vec<_>>>()?,
        GameId::SpiritsOfTheForest => moves
            .extract::<Vec<(u8, NativeSpiritsAction)>>()?
            .into_iter()
            .map(|(player, action)| {
                Ok(replay_record(
                    player,
                    parse_native_catalog_spirits_action(action)?,
                ))
            })
            .collect::<PyResult<Vec<_>>>()?,
        GameId::ConnectFour => moves
            .extract::<Vec<(u8, u8)>>()?
            .into_iter()
            .map(|(player, column)| replay_record(player, CatalogAction::ConnectFour { column }))
            .collect(),
        GameId::TicTacToe => moves
            .extract::<Vec<(u8, (u8, u8))>>()?
            .into_iter()
            .map(|(player, (row, column))| {
                replay_record(player, CatalogAction::TicTacToe { row, column })
            })
            .collect(),
    };
    let analysis = analyze_seeded_trace(game, &recorded, seed)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    match analysis {
        CatalogTraceAnalysis::Generic { utilities } => {
            let result = PyDict::new(py);
            result.set_item("utilities", utilities)?;
            result.set_item(
                "winner",
                utilities.iter().position(|utility| *utility > 0.0),
            )?;
            Ok(result.unbind())
        }
        CatalogTraceAnalysis::Boop(analysis) => serialize_boop_analysis(py, analysis),
        CatalogTraceAnalysis::SpiritsOfTheForest(analysis) => {
            serialize_spirits_analysis(py, analysis)
        }
    }
}

fn replay_record(player: u8, action: CatalogAction) -> RecordedMove {
    RecordedMove {
        player: usize::from(player),
        action,
        decision_seconds: 0.0,
        selection_seconds: 0.0,
        maintenance_seconds: 0.0,
        search_iterations: None,
        search_nodes: None,
        terminal_simulations: None,
        cutoff_simulations: None,
        root_actions: Vec::new(),
        tree_reuse: None,
    }
}

fn serialize_boop_analysis(py: Python<'_>, analysis: BoopReplayAnalysis) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);
    result.set_item("winner", analysis.winner.index())?;
    result.set_item("winner_has_cat_line", analysis.winner_has_cat_line)?;
    result.set_item("winner_has_eight_cats", analysis.winner_has_eight_cats)?;

    let turns = PyList::empty(py);
    for turn in analysis.turns {
        let item = PyDict::new(py);
        item.set_item("ply", turn.ply)?;
        item.set_item("player", turn.player.index())?;
        item.set_item("zone", board_zone_name(turn.zone))?;
        item.set_item("phase", strategic_phase_name(turn.phase))?;
        item.set_item("before", serialize_boop_state_metrics(py, turn.before)?)?;
        item.set_item("after", serialize_boop_state_metrics(py, turn.after)?)?;
        item.set_item("terminal_after", turn.terminal_after)?;

        let interactions = PyList::empty(py);
        for (interaction_index, interaction) in turn.interactions.into_iter().enumerate() {
            let serialized = PyDict::new(py);
            serialized.set_item("interaction_number", interaction_index + 1)?;
            serialized.set_item("target_player", interaction.target.owner().index())?;
            serialized.set_item("target_piece", boop_piece_name(interaction.target.kind()))?;
            serialized.set_item("origin_row", interaction.origin.row())?;
            serialized.set_item("origin_column", interaction.origin.column())?;
            serialized.set_item("destination_row", interaction.destination_row)?;
            serialized.set_item("destination_column", interaction.destination_column)?;
            serialized.set_item("outcome", interaction_outcome_name(interaction.outcome))?;
            interactions.append(serialized)?;
        }
        item.set_item("interactions", interactions)?;

        match turn.resolution {
            None => item.set_item("resolution", py.None())?,
            Some(resolution) => {
                let serialized = PyDict::new(py);
                serialized.set_item("kittens_promoted", resolution.kittens_promoted)?;
                serialized.set_item("cats_recycled", resolution.cats_recycled)?;
                serialized.set_item(
                    "recovered_piece",
                    resolution.recovered_piece.map(boop_piece_name),
                )?;
                serialized.set_item(
                    "orientation",
                    resolution.orientation.map(line_orientation_name),
                )?;
                match resolution.resolution {
                    Resolution::Graduate(line) => {
                        serialized.set_item("type", "graduate")?;
                        serialized.set_item("positions", native_line_positions(line))?;
                    }
                    Resolution::Recover(position) => {
                        serialized.set_item("type", "recover")?;
                        serialized
                            .set_item("positions", vec![(position.row(), position.column())])?;
                    }
                    Resolution::None => unreachable!("analyzed resolution is not none"),
                }
                item.set_item("resolution", serialized)?;
            }
        }
        turns.append(item)?;
    }
    result.set_item("turns", turns)?;

    let winning_lines = PyList::empty(py);
    for (line_index, winning_line) in analysis.winning_lines.into_iter().enumerate() {
        let item = PyDict::new(py);
        item.set_item("line_number", line_index + 1)?;
        item.set_item("player", winning_line.player.index())?;
        item.set_item(
            "orientation",
            line_orientation_name(winning_line.orientation),
        )?;
        item.set_item("positions", native_line_positions(winning_line.line))?;
        winning_lines.append(item)?;
    }
    result.set_item("winning_lines", winning_lines)?;

    Ok(result.unbind())
}

fn serialize_spirits_analysis(
    py: Python<'_>,
    analysis: SpiritsReplayAnalysis,
) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);
    result.set_item("winner", analysis.winner.map(PlayerId::index))?;
    result.set_item("final_scores", analysis.final_scores)?;

    let turns = PyList::empty(py);
    for turn in analysis.turns {
        let item = PyDict::new(py);
        item.set_item("ply", turn.ply)?;
        item.set_item("physical_turn", turn.physical_turn)?;
        item.set_item("action_in_turn", turn.action_in_turn)?;
        item.set_item("player", turn.player.index())?;
        item.set_item("phase_before", spirits_phase_name(turn.phase_before))?;
        item.set_item("phase_after", spirits_phase_name(turn.phase_after))?;
        item.set_item("legal_actions_before", turn.legal_actions_before)?;
        item.set_item("before", serialize_spirits_state_metrics(py, turn.before)?)?;
        item.set_item("after", serialize_spirits_state_metrics(py, turn.after)?)?;
        item.set_item("turn_completed_after", turn.turn_completed_after)?;
        item.set_item("terminal_after", turn.terminal_after)?;
        match turn.tile_take {
            None => item.set_item("tile_take", py.None())?,
            Some(take) => {
                let serialized = PyDict::new(py);
                serialized.set_item("row", take.position.row())?;
                serialized.set_item("column", take.position.column())?;
                serialized.set_item("spirit", spirit_name(take.tile.spirit()))?;
                serialized.set_item("spirit_symbols", take.tile.spirit_symbols())?;
                serialized.set_item(
                    "power_source",
                    take.tile.power_source().map(power_source_name),
                )?;
                serialized.set_item(
                    "reservation_owner",
                    take.reservation_owner.map(PlayerId::index),
                )?;
                match take.sacrifice {
                    None => serialized.set_item("sacrifice", py.None())?,
                    Some(GemstoneSacrifice::Available) => {
                        let sacrifice = PyDict::new(py);
                        sacrifice.set_item("kind", "available")?;
                        serialized.set_item("sacrifice", sacrifice)?;
                    }
                    Some(GemstoneSacrifice::Forest(position)) => {
                        let sacrifice = PyDict::new(py);
                        sacrifice.set_item("kind", "forest")?;
                        sacrifice.set_item("row", position.row())?;
                        sacrifice.set_item("column", position.column())?;
                        serialized.set_item("sacrifice", sacrifice)?;
                    }
                }
                item.set_item("tile_take", serialized)?;
            }
        }
        turns.append(item)?;
    }
    result.set_item("turns", turns)?;

    let categories = PyList::empty(py);
    for (index, category) in analysis.categories.into_iter().enumerate() {
        let item = PyDict::new(py);
        item.set_item("category_number", index + 1)?;
        match category.category {
            ScoringCategory::Spirit(spirit) => {
                item.set_item("category_type", "spirit")?;
                item.set_item("category", spirit_name(spirit))?;
            }
            ScoringCategory::PowerSource(source) => {
                item.set_item("category_type", "power_source")?;
                item.set_item("category", power_source_name(source))?;
            }
        }
        item.set_item("counts", category.counts)?;
        item.set_item("points", category.points)?;
        categories.append(item)?;
    }
    result.set_item("categories", categories)?;
    Ok(result.unbind())
}

fn serialize_spirits_state_metrics(
    py: Python<'_>,
    metrics: SpiritsStateMetrics,
) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);
    result.set_item("remaining_tiles", metrics.remaining_tiles)?;
    result.set_item("completed_turns", metrics.completed_turns)?;
    let players = PyList::empty(py);
    for player in metrics.players {
        let item = PyDict::new(py);
        item.set_item("spirit_symbols", player.spirit_symbols)?;
        item.set_item("power_sources", player.power_sources)?;
        item.set_item("tiles", player.tiles)?;
        item.set_item("score", player.score)?;
        item.set_item("reachable_score", player.reachable_score)?;
        item.set_item("categories_present", player.categories_present)?;
        item.set_item("categories_reachable", player.categories_reachable)?;
        item.set_item("categories_leading", player.categories_leading)?;
        item.set_item("gemstones_available", player.gemstones_available)?;
        item.set_item("gemstones_placed", player.gemstones_placed)?;
        item.set_item("gemstones_removed", player.gemstones_removed)?;
        players.append(item)?;
    }
    result.set_item("players", players)?;
    Ok(result.unbind())
}

fn parse_native_catalog_spirits_action(action: NativeSpiritsAction) -> PyResult<CatalogAction> {
    let (kind, positions, sacrifice) = action;
    let action = match kind.as_str() {
        "take_tile" if positions.len() == 1 => {
            let (row, column) = positions[0];
            let sacrifice = match sacrifice {
                None => None,
                Some((kind, None)) if kind == "available" => {
                    Some(CatalogGemstoneSacrifice::Available)
                }
                Some((kind, Some((row, column)))) if kind == "forest" => {
                    Some(CatalogGemstoneSacrifice::Forest { row, column })
                }
                _ => {
                    return Err(PyValueError::new_err("invalid Spirits gemstone sacrifice"));
                }
            };
            CatalogSpiritsAction::TakeTile {
                row,
                column,
                sacrifice,
            }
        }
        "end_collection" if positions.is_empty() && sacrifice.is_none() => {
            CatalogSpiritsAction::EndCollection
        }
        "place_gemstone" if positions.len() == 1 && sacrifice.is_none() => {
            let (row, column) = positions[0];
            CatalogSpiritsAction::PlaceGemstone { row, column }
        }
        "move_gemstone" if positions.len() == 2 && sacrifice.is_none() => {
            let (source_row, source_column) = positions[0];
            let (target_row, target_column) = positions[1];
            CatalogSpiritsAction::MoveGemstone {
                source_row,
                source_column,
                target_row,
                target_column,
            }
        }
        "skip_gemstone" if positions.is_empty() && sacrifice.is_none() => {
            CatalogSpiritsAction::SkipGemstone
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "invalid Spirits action representation: {kind}"
            )));
        }
    };
    Ok(CatalogAction::SpiritsOfTheForest(action))
}

fn parse_native_catalog_boop_action(action: NativeBoopAction) -> PyResult<CatalogAction> {
    let (piece, row, column, (resolution_type, positions)) = action;
    let piece = match piece.as_str() {
        "kitten" => CatalogBoopPieceKind::Kitten,
        "cat" => CatalogBoopPieceKind::Cat,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown boop piece: {piece}"
            )));
        }
    };
    let resolution = match resolution_type.as_str() {
        "none" if positions.is_empty() => CatalogBoopResolution::None,
        "graduate" if positions.len() == 3 => CatalogBoopResolution::Graduate {
            positions: positions.try_into().expect("graduation length was checked"),
        },
        "recover" if positions.len() == 1 => {
            let (row, column) = positions[0];
            CatalogBoopResolution::Recover { row, column }
        }
        "none" => {
            return Err(PyValueError::new_err(
                "none resolution must not contain positions",
            ));
        }
        "graduate" => {
            return Err(PyValueError::new_err(
                "graduate resolution must contain three positions",
            ));
        }
        "recover" => {
            return Err(PyValueError::new_err(
                "recover resolution must contain one position",
            ));
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown boop resolution: {resolution_type}"
            )));
        }
    };
    Ok(CatalogAction::Boop {
        piece,
        row,
        column,
        resolution,
    })
}

fn serialize_boop_state_metrics(py: Python<'_>, metrics: BoopStateMetrics) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);
    let players = PyList::empty(py);
    for player in metrics.players {
        let item = PyDict::new(py);
        item.set_item("pool_kittens", player.pool_kittens)?;
        item.set_item("pool_cats", player.pool_cats)?;
        item.set_item("board_kittens", player.board_kittens)?;
        item.set_item("board_cats", player.board_cats)?;
        item.set_item("total_cats", player.total_cats())?;
        item.set_item("center_pieces", player.center_pieces)?;
        item.set_item("middle_pieces", player.middle_pieces)?;
        item.set_item("outer_pieces", player.outer_pieces)?;
        players.append(item)?;
    }
    result.set_item("players", players)?;
    result.set_item("empty_center", metrics.empty_center)?;
    result.set_item("empty_middle", metrics.empty_middle)?;
    result.set_item("empty_outer", metrics.empty_outer)?;
    Ok(result.unbind())
}

fn native_line_positions(line: GraduateLine) -> Vec<(u8, u8)> {
    line.positions()
        .into_iter()
        .map(|position| (position.row(), position.column()))
        .collect()
}

const fn board_zone_name(zone: BoardZone) -> &'static str {
    match zone {
        BoardZone::Center => "center",
        BoardZone::Middle => "middle",
        BoardZone::Outer => "outer",
    }
}

const fn strategic_phase_name(phase: StrategicPhase) -> &'static str {
    match phase {
        StrategicPhase::AllKittens => "all_kittens",
        StrategicPhase::OnePlayerHasCats => "one_player_has_cats",
        StrategicPhase::BothPlayersHaveCats => "both_players_have_cats",
    }
}

const fn spirits_phase_name(phase: TurnPhase) -> &'static str {
    match phase {
        TurnPhase::Collect => "collect",
        TurnPhase::PlaceGemstone => "place_gemstone",
    }
}

const fn line_orientation_name(orientation: LineOrientation) -> &'static str {
    match orientation {
        LineOrientation::Horizontal => "horizontal",
        LineOrientation::Vertical => "vertical",
        LineOrientation::DiagonalDown => "diagonal_down",
        LineOrientation::DiagonalUp => "diagonal_up",
    }
}

const fn interaction_outcome_name(outcome: BoopInteractionOutcome) -> &'static str {
    match outcome {
        BoopInteractionOutcome::Moved => "moved",
        BoopInteractionOutcome::OffBoard => "off_board",
        BoopInteractionOutcome::Blocked => "blocked",
        BoopInteractionOutcome::Immune => "immune",
    }
}

fn run_observed_tic_tac_toe_match(
    first: &PythonAgentConfig,
    second: &PythonAgentConfig,
    callback: &Py<PyAny>,
    config: MatchConfig,
) -> PyResult<CatalogMatchReport> {
    let mut first = python_participant(first, configured_tic_tac_toe_mcts)?;
    let mut second = python_participant(second, configured_tic_tac_toe_mcts)?;
    let mut observer = PythonTicTacToeMatchObserver {
        callback,
        error: None,
    };
    let report =
        run_tic_tac_toe_match_with_observer(&mut first, &mut second, config, &mut observer)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

    if let Some(error) = observer.error {
        return Err(PyRuntimeError::new_err(format!(
            "match observer failed: {error}"
        )));
    }
    Ok(report)
}

fn run_observed_boop_match(
    first: &PythonAgentConfig,
    second: &PythonAgentConfig,
    callback: &Py<PyAny>,
    config: MatchConfig,
) -> PyResult<CatalogMatchReport> {
    let mut first = python_participant(first, configured_boop_mcts)?;
    let mut second = python_participant(second, configured_boop_mcts)?;
    let mut observer = PythonBoopMatchObserver {
        callback,
        error: None,
    };
    let report = run_boop_match_with_observer(&mut first, &mut second, config, &mut observer)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

    if let Some(error) = observer.error {
        return Err(PyRuntimeError::new_err(format!(
            "match observer failed: {error}"
        )));
    }
    Ok(report)
}

fn run_observed_connect6_match(
    size: usize,
    first: &PythonAgentConfig,
    second: &PythonAgentConfig,
    callback: &Py<PyAny>,
    config: MatchConfig,
) -> PyResult<CatalogMatchReport> {
    let mut first = python_participant(first, configured_connect6_mcts)?;
    let mut second = python_participant(second, configured_connect6_mcts)?;
    let mut observer = PythonConnect6MatchObserver {
        callback,
        error: None,
    };
    let report =
        run_connect6_match_with_observer(size, &mut first, &mut second, config, &mut observer)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

    if let Some(error) = observer.error {
        return Err(PyRuntimeError::new_err(format!(
            "match observer failed: {error}"
        )));
    }
    Ok(report)
}

fn run_observed_connect_four_match(
    first: &PythonAgentConfig,
    second: &PythonAgentConfig,
    callback: &Py<PyAny>,
    config: MatchConfig,
) -> PyResult<CatalogMatchReport> {
    let mut first = python_participant(first, configured_connect_four_mcts)?;
    let mut second = python_participant(second, configured_connect_four_mcts)?;
    let mut observer = PythonConnectFourMatchObserver {
        callback,
        error: None,
    };
    let report =
        run_connect_four_match_with_observer(&mut first, &mut second, config, &mut observer)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

    if let Some(error) = observer.error {
        return Err(PyRuntimeError::new_err(format!(
            "match observer failed: {error}"
        )));
    }
    Ok(report)
}

fn run_observed_spirits_match(
    first: &PythonAgentConfig,
    second: &PythonAgentConfig,
    callback: &Py<PyAny>,
    config: MatchConfig,
) -> PyResult<CatalogMatchReport> {
    let mut first = python_participant(first, configured_spirits_of_the_forest_mcts)?;
    let mut second = python_participant(second, configured_spirits_of_the_forest_mcts)?;
    let mut observer = PythonSpiritsMatchObserver {
        callback,
        error: None,
    };
    let report = run_spirits_of_the_forest_match_with_observer(
        &mut first,
        &mut second,
        config,
        &mut observer,
    )
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    if let Some(error) = observer.error {
        return Err(PyRuntimeError::new_err(format!(
            "match observer failed: {error}"
        )));
    }
    Ok(report)
}

fn python_participant<'a, M>(
    configured: &'a PythonAgentConfig,
    mcts: impl FnOnce(MctsAgentConfig) -> Result<M, CatalogError>,
) -> PyResult<PythonParticipant<'a, M>> {
    match configured {
        PythonAgentConfig::Automated(config) => ConfiguredAgent::new(config.clone(), mcts)
            .map(PythonParticipant::Automated)
            .map_err(|error| PyRuntimeError::new_err(error.to_string())),
        PythonAgentConfig::Human { selector, observer } => {
            Ok(PythonParticipant::Human(PythonHumanAgent {
                selector,
                observer: observer.as_ref(),
            }))
        }
    }
}

type NativeBoopAction = (String, u8, u8, (String, Vec<(u8, u8)>));

type NativeSpiritsAction = (String, Vec<(u8, u8)>, Option<(String, Option<(u8, u8)>)>);
type NativeSpiritTile = (String, u8, Option<String>, Option<usize>);
type NativeSpiritCollection = (Vec<u8>, Vec<u8>, u8);
type NativeGemstonePool = (u8, u8, u8);
type NativeSpiritsState = (
    Vec<Option<NativeSpiritTile>>,
    Vec<NativeSpiritCollection>,
    Vec<NativeGemstonePool>,
    String,
    usize,
    Vec<i16>,
);

fn native_spirits_action(action: SpiritsOfTheForestAction) -> NativeSpiritsAction {
    match action {
        SpiritsOfTheForestAction::TakeTile {
            position,
            sacrifice,
        } => (
            "take_tile".to_owned(),
            vec![(position.row(), position.column())],
            sacrifice.map(|sacrifice| match sacrifice {
                GemstoneSacrifice::Available => ("available".to_owned(), None),
                GemstoneSacrifice::Forest(position) => (
                    "forest".to_owned(),
                    Some((position.row(), position.column())),
                ),
            }),
        ),
        SpiritsOfTheForestAction::EndCollection => ("end_collection".to_owned(), Vec::new(), None),
        SpiritsOfTheForestAction::PlaceGemstone { target } => (
            "place_gemstone".to_owned(),
            vec![(target.row(), target.column())],
            None,
        ),
        SpiritsOfTheForestAction::MoveGemstone { source, target } => (
            "move_gemstone".to_owned(),
            vec![
                (source.row(), source.column()),
                (target.row(), target.column()),
            ],
            None,
        ),
        SpiritsOfTheForestAction::SkipGemstone => ("skip_gemstone".to_owned(), Vec::new(), None),
    }
}

fn serialize_catalog_spirits_action(
    action: &Bound<'_, PyDict>,
    spirits_action: CatalogSpiritsAction,
) -> PyResult<()> {
    action.set_item("type", "spotf")?;
    match spirits_action {
        CatalogSpiritsAction::TakeTile {
            row,
            column,
            sacrifice,
        } => {
            action.set_item("kind", "take_tile")?;
            action.set_item("row", row)?;
            action.set_item("column", column)?;
            match sacrifice {
                None => action.set_item("sacrifice", action.py().None())?,
                Some(CatalogGemstoneSacrifice::Available) => {
                    let item = PyDict::new(action.py());
                    item.set_item("kind", "available")?;
                    action.set_item("sacrifice", item)?;
                }
                Some(CatalogGemstoneSacrifice::Forest { row, column }) => {
                    let item = PyDict::new(action.py());
                    item.set_item("kind", "forest")?;
                    item.set_item("row", row)?;
                    item.set_item("column", column)?;
                    action.set_item("sacrifice", item)?;
                }
            }
        }
        CatalogSpiritsAction::EndCollection => action.set_item("kind", "end_collection")?,
        CatalogSpiritsAction::PlaceGemstone { row, column } => {
            action.set_item("kind", "place_gemstone")?;
            action.set_item("row", row)?;
            action.set_item("column", column)?;
        }
        CatalogSpiritsAction::MoveGemstone {
            source_row,
            source_column,
            target_row,
            target_column,
        } => {
            action.set_item("kind", "move_gemstone")?;
            action.set_item("source_row", source_row)?;
            action.set_item("source_column", source_column)?;
            action.set_item("target_row", target_row)?;
            action.set_item("target_column", target_column)?;
        }
        CatalogSpiritsAction::SkipGemstone => action.set_item("kind", "skip_gemstone")?,
    }
    Ok(())
}

fn native_spirits_state(
    game: &SpiritsOfTheForest,
    state: &SpiritsOfTheForestState,
) -> NativeSpiritsState {
    let forest = state
        .remaining()
        .iter()
        .enumerate()
        .map(|(index, remaining)| {
            remaining.then(|| {
                let position = ForestPosition::new(
                    (index / meeple_bots_spirits_of_the_forest::COLUMNS) as u8,
                    (index % meeple_bots_spirits_of_the_forest::COLUMNS) as u8,
                )
                .expect("forest index is valid");
                let tile = game.tile(position);
                (
                    spirit_name(tile.spirit()).to_owned(),
                    tile.spirit_symbols(),
                    tile.power_source()
                        .map(|source| power_source_name(source).to_owned()),
                    state.gemstones()[index].map(PlayerId::index),
                )
            })
        })
        .collect();
    let collections = state
        .collections()
        .iter()
        .map(|collection| {
            (
                collection.spirit_symbols().to_vec(),
                collection.power_sources().to_vec(),
                collection.tiles(),
            )
        })
        .collect();
    let pools = state
        .gemstone_pools()
        .iter()
        .map(|pool| (pool.available(), pool.placed(), pool.removed()))
        .collect();
    (
        forest,
        collections,
        pools,
        match state.phase() {
            TurnPhase::Collect => "collect",
            TurnPhase::PlaceGemstone => "place_gemstone",
        }
        .to_owned(),
        state.next_player().index(),
        game.scores(state).to_vec(),
    )
}

const fn spirit_name(spirit: Spirit) -> &'static str {
    match spirit {
        Spirit::Moss => "moss",
        Spirit::Flowers => "flowers",
        Spirit::Fruits => "fruits",
        Spirit::Mushrooms => "mushrooms",
        Spirit::Water => "water",
        Spirit::Vines => "vines",
        Spirit::Branches => "branches",
        Spirit::Leaves => "leaves",
        Spirit::Webs => "webs",
    }
}

const fn power_source_name(source: PowerSource) -> &'static str {
    match source {
        PowerSource::Fire => "fire",
        PowerSource::Moon => "moon",
        PowerSource::Sun => "sun",
    }
}

const fn catalog_spirit_name(spirit: CatalogSpirit) -> &'static str {
    match spirit {
        CatalogSpirit::Moss => "moss",
        CatalogSpirit::Flowers => "flowers",
        CatalogSpirit::Fruits => "fruits",
        CatalogSpirit::Mushrooms => "mushrooms",
        CatalogSpirit::Water => "water",
        CatalogSpirit::Vines => "vines",
        CatalogSpirit::Branches => "branches",
        CatalogSpirit::Leaves => "leaves",
        CatalogSpirit::Webs => "webs",
    }
}

const fn catalog_power_source_name(source: CatalogPowerSource) -> &'static str {
    match source {
        CatalogPowerSource::Fire => "fire",
        CatalogPowerSource::Moon => "moon",
        CatalogPowerSource::Sun => "sun",
    }
}

fn native_boop_action(action: &BoopAction) -> NativeBoopAction {
    let resolution = match action.resolution() {
        Resolution::None => ("none".to_owned(), Vec::new()),
        Resolution::Graduate(line) => (
            "graduate".to_owned(),
            line.positions()
                .into_iter()
                .map(|position| (position.row(), position.column()))
                .collect(),
        ),
        Resolution::Recover(position) => (
            "recover".to_owned(),
            vec![(position.row(), position.column())],
        ),
    };
    (
        boop_piece_name(action.piece()).to_owned(),
        action.position().row(),
        action.position().column(),
        resolution,
    )
}

fn boop_piece_name(piece: BoopPieceKind) -> &'static str {
    match piece {
        BoopPieceKind::Kitten => "kitten",
        BoopPieceKind::Cat => "cat",
    }
}

fn catalog_boop_piece_name(piece: CatalogBoopPieceKind) -> &'static str {
    match piece {
        CatalogBoopPieceKind::Kitten => "kitten",
        CatalogBoopPieceKind::Cat => "cat",
    }
}

fn parse_configured_game(
    game: &str,
    parameters: Option<BTreeMap<String, i64>>,
) -> PyResult<GameId> {
    meeple_bots_catalog::configure_game(parse_game(game)?, &parameters.unwrap_or_default())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}
#[pyfunction(signature = (game, game_params=None))]
fn normalize_game_parameters(
    game: &str,
    game_params: Option<BTreeMap<String, i64>>,
) -> PyResult<BTreeMap<String, i64>> {
    Ok(meeple_bots_catalog::game_parameters(parse_configured_game(
        game,
        game_params,
    )?))
}

fn parse_game(game: &str) -> PyResult<GameId> {
    match game {
        "connect6" => Ok(GameId::Connect6(meeple_bots_connect6::DEFAULT_BOARD_SIZE)),
        "boop" => Ok(GameId::Boop),
        "splendor" => Ok(GameId::Splendor),
        "lost_cities" => Ok(GameId::LostCities),
        "connect_four" => Ok(GameId::ConnectFour),
        "spotf" | "spirits_of_the_forest" => Ok(GameId::SpiritsOfTheForest),
        "tic_tac_toe" => Ok(GameId::TicTacToe),
        other => Err(PyValueError::new_err(format!("unknown game: {other}"))),
    }
}

#[pyfunction(name = "spirits_initial_state")]
fn py_spirits_initial_state(seed: u64) -> NativeSpiritsState {
    let game = spirits_of_the_forest_game(seed);
    let state = game.initial_state();
    native_spirits_state(&game, &state)
}

#[pyfunction(name = "game_search_capabilities")]
fn py_game_search_capabilities(py: Python<'_>, game: &str) -> PyResult<Py<PyDict>> {
    let capabilities = meeple_bots_catalog::game_search_capabilities(parse_game(game)?);
    let result = PyDict::new(py);
    result.set_item("search_agents", capabilities.search_agents)?;
    result.set_item("imperfect_information", capabilities.imperfect_information)?;
    result.set_item("stochastic", capabilities.stochastic)?;
    result.set_item("players", capabilities.players)?;
    result.set_item("turn_phase_conditions", capabilities.turn_phase_conditions)?;
    result.set_item("selection_policies", capabilities.selection_policies)?;
    let heuristics = PyDict::new(py);
    for heuristic in capabilities.heuristics {
        let parameters = PyDict::new(py);
        for spec in heuristic.parameters {
            let parameter = PyDict::new(py);
            parameter.set_item("default", spec.default)?;
            parameter.set_item("minimum", spec.minimum)?;
            parameter.set_item("maximum", spec.maximum)?;
            parameters.set_item(spec.name, parameter)?;
        }
        heuristics.set_item(heuristic.index, parameters)?;
    }
    result.set_item("heuristics", heuristics)?;
    Ok(result.unbind())
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAgentConfig>()?;
    module.add_class::<cant_stop::PyCantStopSession>()?;
    module.add_class::<connect6::Position>()?;
    module.add_function(wrap_pyfunction!(normalize_game_parameters, module)?)?;
    module.add_class::<splendor::PySplendorPosition>()?;
    module.add_class::<lost_cities::PyLostCitiesPosition>()?;
    module.add_class::<lost_cities::PyLostCitiesWorld>()?;
    module.add_function(wrap_pyfunction!(
        lost_cities::lost_cities_so_ismcts_search,
        module
    )?)?;
    module.add_class::<splendor::PySplendorSession>()?;
    module.add_function(wrap_pyfunction!(py_game_search_capabilities, module)?)?;
    module.add_function(wrap_pyfunction!(py_evaluate_game, module)?)?;
    module.add_function(wrap_pyfunction!(evaluation::analyze_structure, module)?)?;
    module.add_function(wrap_pyfunction!(evaluation::benchmark_so_ismcts, module)?)?;
    module.add_function(wrap_pyfunction!(py_benchmark_mcts_agent, module)?)?;
    module.add_function(wrap_pyfunction!(py_run_match, module)?)?;
    module.add_function(wrap_pyfunction!(py_analyze_trace, module)?)?;
    module.add_function(wrap_pyfunction!(py_spirits_initial_state, module)?)?;
    Ok(())
}

fn parse_selection_policy(
    value: &str,
    rave_equivalence: u32,
) -> PyResult<meeple_bots_catalog::SelectionPolicy> {
    if rave_equivalence == 0 {
        return Err(PyValueError::new_err(
            "rave_equivalence must be greater than zero",
        ));
    }
    match value {
        "uct_rave" => Ok(meeple_bots_catalog::SelectionPolicy::UctRave { rave_equivalence }),
        "uct" => Ok(meeple_bots_catalog::SelectionPolicy::Uct),
        "ucb1_tuned" => Ok(meeple_bots_catalog::SelectionPolicy::Ucb1Tuned),
        _ => Err(PyValueError::new_err(
            "selection_policy must be uct, ucb1_tuned or uct_rave",
        )),
    }
}

fn parse_progressive_widening(
    enabled: bool,
    k: f64,
    alpha: f64,
    expansion: &str,
) -> PyResult<Option<meeple_bots_catalog::ProgressiveWidening>> {
    let expansion = match expansion {
        "random" => meeple_bots_catalog::WideningExpansionPolicy::Random,
        "rave" if enabled => meeple_bots_catalog::WideningExpansionPolicy::Rave,
        "rave" => {
            return Err(PyValueError::new_err(
                "rave expansion requires progressive_widening=true",
            ));
        }
        _ => {
            return Err(PyValueError::new_err(
                "progressive_widening_expansion must be random or rave",
            ));
        }
    };
    let pw = meeple_bots_catalog::ProgressiveWidening {
        k,
        alpha,
        expansion,
    };
    pw.validate().map_err(PyValueError::new_err)?;
    Ok(enabled.then_some(pw))
}

fn parse_bandit_policy(value: &str) -> PyResult<meeple_bots_core::BanditPolicy> {
    match value {
        "uct" => Ok(meeple_bots_core::BanditPolicy::Uct),
        "ucb1_tuned" => Ok(meeple_bots_core::BanditPolicy::Ucb1Tuned),
        _ => Err(PyValueError::new_err(
            "SO-ISMCTS selection_policy must be uct or ucb1_tuned",
        )),
    }
}
