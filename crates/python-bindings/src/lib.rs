//! Private PyO3 boundary for the public Python package.

use std::num::NonZeroU32;

use meeple_bots_boop::{Boop, BoopAction, PieceKind as BoopPieceKind, Resolution};
use meeple_bots_catalog::{
    AgentConfig, CatalogAction, CatalogBoopPieceKind, CatalogBoopResolution, CatalogError,
    CatalogMatchReport, CatalogPieceKind, EvaluationConfig, GameId, MatchConfig, MctsAgentConfig,
    MctsConfig, configured_boop_mcts, configured_connect_four_mcts, configured_tic_tac_toe_mcts,
    evaluate_game, run_boop_match_with_trace, run_connect_four_match_with_trace,
    run_match_with_trace, run_tic_tac_toe_match_with_trace,
};
use meeple_bots_connect_four::{ConnectFour, ConnectFourAction};
use meeple_bots_core::{Agent, AgentError, DecisionContext, RandomSource};
use meeple_bots_random_agent::RandomAgent;
use meeple_bots_tic_tac_toe::{TicTacToe, TicTacToeAction};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule};

enum PythonAgentConfig {
    Automated(AgentConfig),
    Human(Py<PyAny>),
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
    fn human(py: Python<'_>, selector: Py<PyAny>) -> PyResult<Self> {
        if !selector.bind(py).is_callable() {
            return Err(PyValueError::new_err("human selector must be callable"));
        }
        Ok(Self {
            inner: PythonAgentConfig::Human(selector),
        })
    }
}

struct PythonHumanAgent<'a> {
    selector: &'a Py<PyAny>,
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
        legal_actions.get(selected).copied().ok_or_else(|| {
            AgentError::message("human selected an action index that is not currently legal")
        })
    }
}

#[pyfunction(name = "run_match")]
#[pyo3(signature = (game, first, second, seed=0, max_plies=10_000))]
fn py_run_match(
    py: Python<'_>,
    game: &str,
    first: PyRef<'_, PyAgentConfig>,
    second: PyRef<'_, PyAgentConfig>,
    seed: u64,
    max_plies: u32,
) -> PyResult<Py<PyDict>> {
    let game = parse_game(game)?;
    let max_plies = NonZeroU32::new(max_plies)
        .ok_or_else(|| PyValueError::new_err("max_plies must be greater than zero"))?;
    let config = MatchConfig::new(seed, max_plies);
    let report = match (&first.inner, &second.inner) {
        (PythonAgentConfig::Automated(first), PythonAgentConfig::Automated(second)) => {
            run_match_with_trace(game, *first, *second, config)
        }
        (PythonAgentConfig::Human(first), PythonAgentConfig::Automated(second)) => {
            run_with_human_first(game, first, *second, config)
        }
        (PythonAgentConfig::Automated(first), PythonAgentConfig::Human(second)) => {
            run_with_human_second(game, *first, second, config)
        }
        (PythonAgentConfig::Human(first), PythonAgentConfig::Human(second)) => {
            run_with_two_humans(game, first, second, config)
        }
    }
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

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

fn run_with_human_first(
    game: GameId,
    first: &Py<PyAny>,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError> {
    let mut first = PythonHumanAgent { selector: first };
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
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError> {
    let mut second = PythonHumanAgent { selector: second };
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
    second: &Py<PyAny>,
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError> {
    match game {
        GameId::Boop => run_boop_match_with_trace(
            &mut PythonHumanAgent { selector: first },
            &mut PythonHumanAgent { selector: second },
            config,
        ),
        GameId::ConnectFour => run_connect_four_match_with_trace(
            &mut PythonHumanAgent { selector: first },
            &mut PythonHumanAgent { selector: second },
            config,
        ),
        GameId::TicTacToe => run_tic_tac_toe_match_with_trace(
            &mut PythonHumanAgent { selector: first },
            &mut PythonHumanAgent { selector: second },
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
    Ok(())
}
