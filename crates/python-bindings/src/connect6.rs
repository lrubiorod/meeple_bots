//! Typed native Connect6 positions. Python never computes legal moves or turns.
use meeple_bots_connect6::{Connect6, Connect6Action, Connect6State};
use meeple_bots_core::{Game, PositionStatus};
use pyo3::{exceptions::PyValueError, prelude::*};
#[pyclass(name = "Connect6Position", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct Position {
    game: Connect6,
    state: Connect6State,
}
#[pymethods]
impl Position {
    #[new]
    #[pyo3(signature = (board_size=None))]
    fn new(board_size: Option<usize>) -> PyResult<Self> {
        let game = Connect6::new(board_size.unwrap_or(meeple_bots_connect6::DEFAULT_BOARD_SIZE))
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            state: game.initial_state(),
            game,
        })
    }
    #[getter]
    fn board_size(&self) -> usize {
        self.game.board_size()
    }
    #[getter]
    fn board(&self) -> Vec<Option<usize>> {
        self.state
            .board()
            .iter()
            .map(|&v| {
                if v == 0 {
                    None
                } else {
                    Some(usize::from(v - 1))
                }
            })
            .collect()
    }
    #[getter]
    fn current_player(&self) -> Option<usize> {
        match self.game.status(&self.state) {
            PositionStatus::PlayerTurn(p) => Some(p.index()),
            _ => None,
        }
    }
    #[getter]
    fn placements_remaining(&self) -> u8 {
        self.state.placements_remaining()
    }
    #[getter]
    fn winner(&self) -> Option<usize> {
        self.state.winner().map(|p| p.index())
    }
    #[getter]
    fn terminal(&self) -> bool {
        matches!(self.game.status(&self.state), PositionStatus::Terminal)
    }
    fn legal_actions(&self) -> Vec<usize> {
        self.game
            .legal_actions(&self.state)
            .map(|Connect6Action::Place(p)| p)
            .collect()
    }
    fn apply(&self, position: usize) -> PyResult<Self> {
        let mut next = self.clone();
        next.game
            .apply_action(&mut next.state, &Connect6Action::Place(position))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(next)
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.game == other.game && self.state == other.state
    }
}
