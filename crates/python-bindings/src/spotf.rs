//! Executable public SPOTF positions; search uses the normal catalog agent.
use super::*;
use meeple_bots_core::{HeuristicGame, PositionStatus};
use meeple_bots_simulation::SplitMix64;

#[pyclass(name = "SpotfPosition", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct Position {
    game: SpiritsOfTheForest,
    state: SpiritsOfTheForestState,
}

#[pymethods]
impl Position {
    #[new]
    fn new(seed: u64) -> Self {
        let game = spirits_of_the_forest_game(seed);
        let state = game.initial_state();
        Self { game, state }
    }

    fn snapshot(&self) -> NativeSpiritsState {
        native_spirits_state(&self.game, &self.state)
    }

    fn legal_actions(&self) -> Vec<NativeSpiritsAction> {
        self.game
            .legal_actions(&self.state)
            .map(native_spirits_action)
            .collect()
    }

    fn apply(&self, index: usize) -> PyResult<Self> {
        let action = self
            .game
            .legal_actions(&self.state)
            .nth(index)
            .ok_or_else(|| PyValueError::new_err("illegal action index"))?;
        let mut next = self.clone();
        next.game
            .apply_action(&mut next.state, &action)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(next)
    }

    fn heuristic(&self, player: usize) -> PyResult<f32> {
        let player = match player {
            0 => PlayerId::FIRST,
            1 => PlayerId::SECOND,
            _ => return Err(PyValueError::new_err("player must be 0 or 1")),
        };
        self.game
            .heuristic_utility(0, &self.state, player)
            .ok_or_else(|| PyValueError::new_err("H0 unavailable"))
    }

    fn search(&self, py: Python<'_>, config: &PyAgentConfig, seed: u64) -> PyResult<Py<PyDict>> {
        let PythonAgentConfig::Automated(config) = &config.inner else {
            return Err(PyValueError::new_err(
                "probe requires an automated MCTS config",
            ));
        };
        let AgentConfig::Mcts(config) = config.as_ref() else {
            return Err(PyValueError::new_err("SPOTF position search requires MCTS"));
        };
        let mut config = config.clone();
        config.root_diagnostics = true;
        let (action, stats, seconds) = py.detach(|| {
            let PositionStatus::PlayerTurn(player) = self.game.status(&self.state) else {
                return Err(PyValueError::new_err("search requires a player turn"));
            };
            let mut agent = configured_spirits_of_the_forest_mcts(config)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            agent.on_match_start(&self.game, &self.state, player);
            let started = std::time::Instant::now();
            let action = agent
                .select_action(
                    DecisionContext::new(&self.game, &self.state, player),
                    &mut SplitMix64::new(seed),
                )
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let seconds = started.elapsed().as_secs_f64();
            Ok::<_, PyErr>((action, agent.last_decision_stats(), seconds))
        })?;
        let out = PyDict::new(py);
        out.set_item("action", native_spirits_action(action))?;
        out.set_item("search_seconds", seconds)?;
        out.set_item("completed_iterations", stats.search_iterations)?;
        out.set_item("terminal_simulations", stats.terminal_simulations)?;
        out.set_item("cutoff_simulations", stats.cutoff_simulations)?;
        let edges = PyList::empty(py);
        for edge in stats.root_actions {
            let item = PyDict::new(py);
            item.set_item("action_index", edge.action_index)?;
            item.set_item("visits", edge.visits)?;
            item.set_item("q", edge.mean_utility)?;
            item.set_item("heuristic_value", edge.heuristic_value)?;
            item.set_item("progressive_bias", edge.progressive_bias)?;
            edges.append(item)?;
        }
        out.set_item("root_actions", edges)?;
        Ok(out.unbind())
    }
}
