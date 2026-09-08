//! Python boundary for interactive stochastic sessions; all game rules remain in Rust.
use super::{PyAgentConfig, PythonAgentConfig};
use meeple_bots_cant_stop::{CantStop, CantStopAction, HEIGHTS, Phase};
use meeple_bots_catalog::{AgentConfig, cant_stop::CantStopSession};
use meeple_bots_core::Game;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

#[pyclass(name = "CantStopSession")]
pub struct PyCantStopSession {
    inner: CantStopSession,
}

fn automated(config: Option<&PyAgentConfig>) -> PyResult<Option<AgentConfig>> {
    match config.map(|c| &c.inner) {
        None => Ok(None),
        Some(PythonAgentConfig::Automated(a)) => Ok(Some(a.clone())),
        _ => Err(PyValueError::new_err("use None for a human session seat")),
    }
}
fn action_dict(py: Python<'_>, action: &CantStopAction) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    match action {
        CantStopAction::Advance(a, b) => {
            d.set_item("kind", "advance")?;
            d.set_item(
                "columns",
                if *b == 0 {
                    vec![u32::from(*a)]
                } else {
                    vec![u32::from(*a), u32::from(*b)]
                },
            )?;
        }
        CantStopAction::RollAgain => d.set_item("kind", "roll")?,
        CantStopAction::Stop => d.set_item("kind", "stop")?,
        CantStopAction::Dice(dice) => {
            d.set_item("kind", "dice")?;
            d.set_item("dice", dice.map(u32::from).to_vec())?;
        }
    }
    Ok(d.unbind())
}
#[pymethods]
impl PyCantStopSession {
    #[new]
    #[pyo3(signature=(seed=0,first=None,second=None))]
    fn new(
        seed: u64,
        first: Option<&PyAgentConfig>,
        second: Option<&PyAgentConfig>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CantStopSession::new(seed, automated(first)?, automated(second)?)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }
    #[pyo3(signature=(action=None))]
    fn step(&mut self, py: Python<'_>, action: Option<usize>) -> PyResult<Py<PyDict>> {
        py.detach(|| self.inner.step(action))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.snapshot(py)
    }
    fn snapshot(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let s = &self.inner.state;
        let d = PyDict::new(py);
        d.set_item("game", "cant-stop")?;
        d.set_item("heights", HEIGHTS.map(u32::from).to_vec())?;
        d.set_item(
            "progress",
            s.progress.map(|p| p.map(u32::from).to_vec()).to_vec(),
        )?;
        d.set_item("runners", s.runners.map(u32::from).to_vec())?;
        d.set_item("claimed", s.claimed.map(|p| p.map(|p| p.index())).to_vec())?;
        d.set_item("dice", s.dice.map(u32::from).to_vec())?;
        d.set_item("active_player", s.active.index())?;
        d.set_item("winner", s.winner.map(|p| p.index()))?;
        d.set_item(
            "phase",
            match s.phase {
                Phase::Roll => "roll",
                Phase::Choose => "choose",
                Phase::Continue => "continue",
                Phase::Finished => "finished",
            },
        )?;
        d.set_item("waiting_human", self.inner.waiting_human())?;
        d.set_item("last_bust", s.last_bust)?;
        let actions = CantStop
            .legal_actions(s)
            .map(|a| action_dict(py, &a))
            .collect::<PyResult<Vec<_>>>()?;
        d.set_item("legal_actions", actions)?;
        let events = self
            .inner
            .events
            .iter()
            .map(|event| {
                let e = PyDict::new(py);
                e.set_item("player", event.player.map(|p| p.index()))?;
                e.set_item("action", action_dict(py, &event.action)?)?;
                e.set_item("decision_seconds", event.seconds.as_secs_f64())?;
                e.set_item("search_iterations", event.stats.search_iterations)?;
                e.set_item("bust", event.bust)?;
                Ok(e.unbind())
            })
            .collect::<PyResult<Vec<_>>>()?;
        d.set_item("events", events)?;
        Ok(d.unbind())
    }
}
