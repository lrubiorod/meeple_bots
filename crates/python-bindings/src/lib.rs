//! Private PyO3 boundary for the public Python package.

use std::num::NonZeroU32;

use meeple_bots_catalog::{
    AgentConfig, CatalogAction, GameId, MatchConfig, MctsConfig, run_match_with_trace,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

#[pyclass(name = "AgentConfig", frozen)]
struct PyAgentConfig {
    inner: AgentConfig,
}

#[pymethods]
impl PyAgentConfig {
    #[staticmethod]
    fn random() -> Self {
        Self {
            inner: AgentConfig::Random,
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
            inner: AgentConfig::Mcts(MctsConfig {
                iterations,
                exploration,
                rollout_depth,
            }),
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
    let game = match game {
        "tic_tac_toe" => GameId::TicTacToe,
        other => return Err(PyValueError::new_err(format!("unknown game: {other}"))),
    };
    let max_plies = NonZeroU32::new(max_plies)
        .ok_or_else(|| PyValueError::new_err("max_plies must be greater than zero"))?;
    let report = run_match_with_trace(
        game,
        first.inner,
        second.inner,
        MatchConfig::new(seed, max_plies),
    )
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

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAgentConfig>()?;
    module.add_function(wrap_pyfunction!(py_run_match, module)?)?;
    Ok(())
}
