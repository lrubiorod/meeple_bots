//! Private PyO3 boundary for the public Python package.

use std::num::NonZeroU32;

use meeple_bots_catalog::{
    AgentConfig, CatalogAction, CatalogError, CatalogMatchReport, GameId, MatchConfig, MctsConfig,
    run_connect_four_match_with_trace, run_match_with_trace, run_tic_tac_toe_match_with_trace,
};
use meeple_bots_connect_four::{ConnectFour, ConnectFourAction};
use meeple_bots_core::{Agent, AgentError, DecisionContext, RandomSource};
use meeple_bots_mcts_agent::MctsAgent;
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
    ))]
    fn mcts(iterations: u32, exploration: f64, rollout_depth: u32) -> PyResult<Self> {
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
            inner: PythonAgentConfig::Automated(AgentConfig::Mcts(MctsConfig {
                iterations,
                exploration,
                rollout_depth,
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
    let game = match game {
        "connect_four" => GameId::ConnectFour,
        "tic_tac_toe" => GameId::TicTacToe,
        other => return Err(PyValueError::new_err(format!("unknown game: {other}"))),
    };
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

    let moves = PyList::empty(py);
    for recorded in report.moves {
        let action = PyDict::new(py);
        match recorded.action {
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
        (GameId::ConnectFour, AgentConfig::Random) => {
            run_connect_four_match_with_trace(&mut first, &mut RandomAgent, config)
        }
        (GameId::ConnectFour, AgentConfig::Mcts(configured)) => {
            run_connect_four_match_with_trace(&mut first, &mut MctsAgent::new(configured), config)
        }
        (GameId::TicTacToe, AgentConfig::Random) => {
            run_tic_tac_toe_match_with_trace(&mut first, &mut RandomAgent, config)
        }
        (GameId::TicTacToe, AgentConfig::Mcts(configured)) => {
            run_tic_tac_toe_match_with_trace(&mut first, &mut MctsAgent::new(configured), config)
        }
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
        (GameId::ConnectFour, AgentConfig::Random) => {
            run_connect_four_match_with_trace(&mut RandomAgent, &mut second, config)
        }
        (GameId::ConnectFour, AgentConfig::Mcts(configured)) => {
            run_connect_four_match_with_trace(&mut MctsAgent::new(configured), &mut second, config)
        }
        (GameId::TicTacToe, AgentConfig::Random) => {
            run_tic_tac_toe_match_with_trace(&mut RandomAgent, &mut second, config)
        }
        (GameId::TicTacToe, AgentConfig::Mcts(configured)) => {
            run_tic_tac_toe_match_with_trace(&mut MctsAgent::new(configured), &mut second, config)
        }
    }
}

fn run_with_two_humans(
    game: GameId,
    first: &Py<PyAny>,
    second: &Py<PyAny>,
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError> {
    match game {
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

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAgentConfig>()?;
    module.add_function(wrap_pyfunction!(py_run_match, module)?)?;
    Ok(())
}
