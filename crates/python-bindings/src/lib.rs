//! Private PyO3 boundary for the public Python package.

use std::{num::NonZeroU32, time::Duration};

use meeple_bots_boop::{
    BoardZone, Boop, BoopAction, BoopInteractionOutcome, BoopStateMetrics, GraduateLine,
    LineOrientation, PieceKind as BoopPieceKind, Resolution, StrategicPhase,
};
use meeple_bots_catalog::{
    AgentConfig, CatalogAction, CatalogBoopPieceKind, CatalogBoopResolution, CatalogError,
    CatalogMatchReport, CatalogPieceKind, CatalogTraceAnalysis, EvaluationConfig, GameId,
    MatchConfig, MctsAgentConfig, MctsConfig, RecordedMove, analyze_trace, configured_boop_mcts,
    configured_connect_four_mcts, configured_tic_tac_toe_mcts, evaluate_game,
    run_boop_match_with_observer, run_boop_match_with_trace, run_connect_four_match_with_observer,
    run_connect_four_match_with_trace, run_match_with_trace, run_tic_tac_toe_match_with_observer,
    run_tic_tac_toe_match_with_trace,
};
use meeple_bots_connect_four::{ConnectFour, ConnectFourAction};
use meeple_bots_core::{Agent, AgentError, DecisionContext, Game, PlayerId, RandomSource};
use meeple_bots_mcts_agent::MctsAgent;
use meeple_bots_random_agent::RandomAgent;
use meeple_bots_simulation::MatchObserver;
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
    fn random() -> Self {
        Self {
            inner: PythonAgentConfig::Automated(AgentConfig::Random),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (
        iterations=1_000,
        exploration=std::f64::consts::SQRT_2,
        rollout_depth=256,
        heuristic=None,
    ))]
    fn mcts(
        iterations: u32,
        exploration: f64,
        rollout_depth: u32,
        heuristic: Option<u32>,
    ) -> PyResult<Self> {
        let iterations = NonZeroU32::new(iterations)
            .ok_or_else(|| PyValueError::new_err("iterations must be greater than zero"))?;
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
                    iterations,
                    exploration,
                    rollout_depth,
                },
                heuristic,
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

enum PythonObservedAgent<'a> {
    Human(PythonHumanAgent<'a>),
    Mcts(MctsAgent),
    Random(RandomAgent),
}

enum PythonObservedBoopAgent<'a> {
    Human(PythonHumanAgent<'a>),
    Mcts(meeple_bots_catalog::BoopMctsAgent),
    Random(RandomAgent),
}

impl Agent<Boop> for PythonObservedBoopAgent<'_> {
    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, Boop>,
        rng: &mut R,
    ) -> Result<BoopAction, AgentError> {
        match self {
            Self::Human(agent) => agent.select_action(decision, rng),
            Self::Mcts(agent) => agent.select_action(decision, rng),
            Self::Random(agent) => agent.select_action(decision, rng),
        }
    }
}

impl Agent<TicTacToe> for PythonObservedAgent<'_> {
    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, TicTacToe>,
        rng: &mut R,
    ) -> Result<TicTacToeAction, AgentError> {
        match self {
            Self::Human(agent) => agent.select_action(decision, rng),
            Self::Mcts(agent) => agent.select_action(decision, rng),
            Self::Random(agent) => agent.select_action(decision, rng),
        }
    }
}

impl Agent<ConnectFour> for PythonObservedAgent<'_> {
    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, ConnectFour>,
        rng: &mut R,
    ) -> Result<ConnectFourAction, AgentError> {
        match self {
            Self::Human(agent) => agent.select_action(decision, rng),
            Self::Mcts(agent) => agent.select_action(decision, rng),
            Self::Random(agent) => agent.select_action(decision, rng),
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
        decision_time: Duration,
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
                decision_time.as_secs_f64(),
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
        decision_time: Duration,
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
                decision_time.as_secs_f64(),
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
        decision_time: Duration,
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
                decision_time.as_secs_f64(),
            ))?;
            Ok(())
        }) {
            self.error = Some(error.to_string());
        }
    }
}

#[pyfunction(name = "evaluate_game")]
#[pyo3(signature = (game, samples=128, max_depth=256, seed=0))]
fn py_evaluate_game(
    py: Python<'_>,
    game: &str,
    samples: u32,
    max_depth: u32,
    seed: u64,
) -> PyResult<Py<PyDict>> {
    let game = parse_game(game)?;
    let samples = NonZeroU32::new(samples)
        .ok_or_else(|| PyValueError::new_err("samples must be greater than zero"))?;
    let max_depth = NonZeroU32::new(max_depth)
        .ok_or_else(|| PyValueError::new_err("max_depth must be greater than zero"))?;
    let report = evaluate_game(
        game,
        EvaluationConfig {
            samples,
            max_depth,
            seed,
        },
    )
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

    let serialized = PyDict::new(py);
    serialized.set_item("samples", report.samples)?;
    serialized.set_item("max_depth", report.max_depth)?;
    serialized.set_item("terminal_rate", report.terminal_rate)?;
    serialized.set_item("initial_legal_actions", report.initial_legal_actions)?;
    serialized.set_item(
        "effective_branching_factor",
        report.effective_branching_factor,
    )?;
    serialized.set_item("estimated_depth", report.estimated_depth)?;
    serialized.set_item("depth_is_lower_bound", report.depth_is_lower_bound)?;
    serialized.set_item("estimated_tree_log10", report.estimated_tree_log10)?;
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

#[pyfunction(name = "run_match")]
#[pyo3(signature = (game, first, second, seed=0, max_plies=10_000, observer=None))]
fn py_run_match(
    py: Python<'_>,
    game: &str,
    first: PyRef<'_, PyAgentConfig>,
    second: PyRef<'_, PyAgentConfig>,
    seed: u64,
    max_plies: u32,
    observer: Option<Py<PyAny>>,
) -> PyResult<Py<PyDict>> {
    let game = parse_game(game)?;
    let max_plies = NonZeroU32::new(max_plies)
        .ok_or_else(|| PyValueError::new_err("max_plies must be greater than zero"))?;
    let config = MatchConfig::new(seed, max_plies);
    if let Some(observer) = observer.as_ref() {
        if !observer.bind(py).is_callable() {
            return Err(PyValueError::new_err("match observer must be callable"));
        }
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

    let moves = PyList::empty(py);
    for recorded in report.moves {
        let action = PyDict::new(py);
        match recorded.action {
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
            CatalogAction::ConnectFour { column } => {
                action.set_item("type", "connect_four")?;
                action.set_item("column", column)?;
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
        moves.append(movement)?;
    }
    result.set_item("moves", moves)?;

    Ok(result.unbind())
}

fn clone_python_agent_config(py: Python<'_>, configured: &PythonAgentConfig) -> PythonAgentConfig {
    match configured {
        PythonAgentConfig::Automated(agent) => PythonAgentConfig::Automated(*agent),
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
    if let Some(observer) = observer {
        return match game {
            GameId::Boop => run_observed_boop_match(first, second, observer, config),
            GameId::ConnectFour => run_observed_connect_four_match(first, second, observer, config),
            GameId::TicTacToe => run_observed_tic_tac_toe_match(first, second, observer, config),
        };
    }

    match (first, second) {
        (PythonAgentConfig::Automated(first), PythonAgentConfig::Automated(second)) => {
            run_match_with_trace(game, *first, *second, config)
        }
        (PythonAgentConfig::Human { selector, observer }, PythonAgentConfig::Automated(second)) => {
            run_with_human_first(game, selector, observer.as_ref(), *second, config)
        }
        (PythonAgentConfig::Automated(first), PythonAgentConfig::Human { selector, observer }) => {
            run_with_human_second(game, *first, selector, observer.as_ref(), config)
        }
        (
            PythonAgentConfig::Human {
                selector: first,
                observer: first_observer,
            },
            PythonAgentConfig::Human {
                selector: second,
                observer: second_observer,
            },
        ) => run_with_two_humans(
            game,
            first,
            first_observer.as_ref(),
            second,
            second_observer.as_ref(),
            config,
        ),
    }
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pyfunction(name = "analyze_trace")]
fn py_analyze_trace(py: Python<'_>, game: &str, moves: &Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    let game = parse_game(game)?;
    let recorded = match game {
        GameId::Boop => moves
            .extract::<Vec<(u8, NativeBoopAction)>>()?
            .into_iter()
            .map(|(player, action)| {
                Ok(RecordedMove {
                    player: usize::from(player),
                    action: parse_native_catalog_boop_action(action)?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?,
        GameId::ConnectFour | GameId::TicTacToe => {
            return Err(PyValueError::new_err(
                analyze_trace(game, &[])
                    .expect_err("games without analysis return an error")
                    .to_string(),
            ));
        }
    };
    let analysis =
        analyze_trace(game, &recorded).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let CatalogTraceAnalysis::Boop(analysis) = analysis;

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
    let mut first = python_tic_tac_toe_agent(first)?;
    let mut second = python_tic_tac_toe_agent(second)?;
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
    let mut first = python_boop_agent(first)?;
    let mut second = python_boop_agent(second)?;
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

fn run_observed_connect_four_match(
    first: &PythonAgentConfig,
    second: &PythonAgentConfig,
    callback: &Py<PyAny>,
    config: MatchConfig,
) -> PyResult<CatalogMatchReport> {
    let mut first = python_connect_four_agent(first)?;
    let mut second = python_connect_four_agent(second)?;
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

fn configured_tic_tac_toe_mcts_for_python(config: MctsAgentConfig) -> PyResult<MctsAgent> {
    configured_tic_tac_toe_mcts(config).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

fn python_tic_tac_toe_agent(configured: &PythonAgentConfig) -> PyResult<PythonObservedAgent<'_>> {
    match configured {
        PythonAgentConfig::Automated(AgentConfig::Random) => {
            Ok(PythonObservedAgent::Random(RandomAgent))
        }
        PythonAgentConfig::Automated(AgentConfig::Mcts(config)) => Ok(PythonObservedAgent::Mcts(
            configured_tic_tac_toe_mcts_for_python(*config)?,
        )),
        PythonAgentConfig::Human { selector, observer } => {
            Ok(PythonObservedAgent::Human(PythonHumanAgent {
                selector,
                observer: observer.as_ref(),
            }))
        }
    }
}

fn python_connect_four_agent(configured: &PythonAgentConfig) -> PyResult<PythonObservedAgent<'_>> {
    match configured {
        PythonAgentConfig::Automated(AgentConfig::Random) => {
            Ok(PythonObservedAgent::Random(RandomAgent))
        }
        PythonAgentConfig::Automated(AgentConfig::Mcts(config)) => Ok(PythonObservedAgent::Mcts(
            configured_connect_four_mcts(*config)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
        )),
        PythonAgentConfig::Human { selector, observer } => {
            Ok(PythonObservedAgent::Human(PythonHumanAgent {
                selector,
                observer: observer.as_ref(),
            }))
        }
    }
}

fn python_boop_agent(configured: &PythonAgentConfig) -> PyResult<PythonObservedBoopAgent<'_>> {
    match configured {
        PythonAgentConfig::Automated(AgentConfig::Random) => {
            Ok(PythonObservedBoopAgent::Random(RandomAgent))
        }
        PythonAgentConfig::Automated(AgentConfig::Mcts(config)) => {
            Ok(PythonObservedBoopAgent::Mcts(
                configured_boop_mcts(*config)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
            ))
        }
        PythonAgentConfig::Human { selector, observer } => {
            Ok(PythonObservedBoopAgent::Human(PythonHumanAgent {
                selector,
                observer: observer.as_ref(),
            }))
        }
    }
}

fn run_with_human_first(
    game: GameId,
    first: &Py<PyAny>,
    observer: Option<&Py<PyAny>>,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError> {
    let mut first = PythonHumanAgent {
        selector: first,
        observer,
    };
    match (game, second) {
        (GameId::Boop, AgentConfig::Random) => {
            run_boop_match_with_trace(&mut first, &mut RandomAgent, config)
        }
        (GameId::Boop, AgentConfig::Mcts(configured)) => {
            run_boop_match_with_trace(&mut first, &mut configured_boop_mcts(configured)?, config)
        }
        (GameId::ConnectFour, AgentConfig::Random) => {
            run_connect_four_match_with_trace(&mut first, &mut RandomAgent, config)
        }
        (GameId::ConnectFour, AgentConfig::Mcts(configured)) => run_connect_four_match_with_trace(
            &mut first,
            &mut configured_connect_four_mcts(configured)?,
            config,
        ),
        (GameId::TicTacToe, AgentConfig::Random) => {
            run_tic_tac_toe_match_with_trace(&mut first, &mut RandomAgent, config)
        }
        (GameId::TicTacToe, AgentConfig::Mcts(configured)) => run_tic_tac_toe_match_with_trace(
            &mut first,
            &mut configured_tic_tac_toe_mcts(configured)?,
            config,
        ),
    }
}

fn run_with_human_second(
    game: GameId,
    first: AgentConfig,
    second: &Py<PyAny>,
    observer: Option<&Py<PyAny>>,
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError> {
    let mut second = PythonHumanAgent {
        selector: second,
        observer,
    };
    match (game, first) {
        (GameId::Boop, AgentConfig::Random) => {
            run_boop_match_with_trace(&mut RandomAgent, &mut second, config)
        }
        (GameId::Boop, AgentConfig::Mcts(configured)) => {
            run_boop_match_with_trace(&mut configured_boop_mcts(configured)?, &mut second, config)
        }
        (GameId::ConnectFour, AgentConfig::Random) => {
            run_connect_four_match_with_trace(&mut RandomAgent, &mut second, config)
        }
        (GameId::ConnectFour, AgentConfig::Mcts(configured)) => run_connect_four_match_with_trace(
            &mut configured_connect_four_mcts(configured)?,
            &mut second,
            config,
        ),
        (GameId::TicTacToe, AgentConfig::Random) => {
            run_tic_tac_toe_match_with_trace(&mut RandomAgent, &mut second, config)
        }
        (GameId::TicTacToe, AgentConfig::Mcts(configured)) => run_tic_tac_toe_match_with_trace(
            &mut configured_tic_tac_toe_mcts(configured)?,
            &mut second,
            config,
        ),
    }
}

fn run_with_two_humans(
    game: GameId,
    first: &Py<PyAny>,
    first_observer: Option<&Py<PyAny>>,
    second: &Py<PyAny>,
    second_observer: Option<&Py<PyAny>>,
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError> {
    match game {
        GameId::Boop => run_boop_match_with_trace(
            &mut PythonHumanAgent {
                selector: first,
                observer: first_observer,
            },
            &mut PythonHumanAgent {
                selector: second,
                observer: second_observer,
            },
            config,
        ),
        GameId::ConnectFour => run_connect_four_match_with_trace(
            &mut PythonHumanAgent {
                selector: first,
                observer: first_observer,
            },
            &mut PythonHumanAgent {
                selector: second,
                observer: second_observer,
            },
            config,
        ),
        GameId::TicTacToe => run_tic_tac_toe_match_with_trace(
            &mut PythonHumanAgent {
                selector: first,
                observer: first_observer,
            },
            &mut PythonHumanAgent {
                selector: second,
                observer: second_observer,
            },
            config,
        ),
    }
}

type NativeBoopAction = (String, u8, u8, (String, Vec<(u8, u8)>));

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

fn parse_game(game: &str) -> PyResult<GameId> {
    match game {
        "boop" => Ok(GameId::Boop),
        "connect_four" => Ok(GameId::ConnectFour),
        "tic_tac_toe" => Ok(GameId::TicTacToe),
        other => Err(PyValueError::new_err(format!("unknown game: {other}"))),
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAgentConfig>()?;
    module.add_function(wrap_pyfunction!(py_evaluate_game, module)?)?;
    module.add_function(wrap_pyfunction!(py_run_match, module)?)?;
    module.add_function(wrap_pyfunction!(py_analyze_trace, module)?)?;
    Ok(())
}
