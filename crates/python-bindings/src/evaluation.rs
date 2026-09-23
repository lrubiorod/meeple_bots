//! Analysis bindings keep structural sampling separate from search calibration.
use super::*;

pub(crate) fn structural_dict(
    py: Python<'_>,
    report: &meeple_bots_evaluation::StructuralReport,
) -> PyResult<Py<PyDict>> {
    let value = PyDict::new(py);
    let phases = PyList::empty(py);
    for phase in &report.phases {
        let row = PyDict::new(py);
        row.set_item("label", phase.label)?;
        row.set_item("samples", phase.samples)?;
        row.set_item("legal_actions_mean", phase.legal_actions_mean)?;
        row.set_item("legal_actions_p50", phase.legal_actions_p50)?;
        row.set_item("legal_actions_p95", phase.legal_actions_p95)?;
        row.set_item("legal_actions_min", phase.legal_actions_min)?;
        row.set_item("legal_actions_max", phase.legal_actions_max)?;
        row.set_item(
            "effective_branching_factor",
            phase.effective_branching_factor,
        )?;
        phases.append(row)?;
    }
    value.set_item("phases", phases)?;
    value.set_item("physical_turns_p50", report.physical_turns_p50)?;
    value.set_item("physical_turns_p95", report.physical_turns_p95)?;
    value.set_item("samples", report.samples)?;
    value.set_item("max_depth", report.max_depth)?;
    value.set_item("terminal_rate", report.terminal_rate)?;
    value.set_item("initial_legal_actions", report.initial_legal_actions)?;
    value.set_item(
        "initial_legal_actions_mean",
        report.initial_legal_actions_mean,
    )?;
    value.set_item(
        "initial_legal_actions_p50",
        report.initial_legal_actions_p50,
    )?;
    value.set_item(
        "initial_legal_actions_p95",
        report.initial_legal_actions_p95,
    )?;
    value.set_item(
        "initial_legal_actions_min",
        report.initial_legal_actions_min,
    )?;
    value.set_item(
        "initial_legal_actions_max",
        report.initial_legal_actions_max,
    )?;
    value.set_item("chance_events_mean", report.chance_events_mean)?;
    value.set_item("chance_events_p50", report.chance_events_p50)?;
    value.set_item("chance_events_p95", report.chance_events_p95)?;
    value.set_item("decisions_mean", report.decisions_mean)?;
    value.set_item(
        "effective_branching_factor",
        report.effective_branching_factor,
    )?;
    value.set_item(
        "player_turn_choice_product_log10",
        report.player_turn_choice_product_log10,
    )?;
    value.set_item("depth_p50", report.depth_p50)?;
    value.set_item("estimated_depth", report.estimated_depth)?;
    value.set_item("player_turn_depth_p50", report.player_turn_depth_p50)?;
    value.set_item("player_turn_depth_p95", report.player_turn_depth_p95)?;
    value.set_item("player_changes_p50", report.player_changes_p50)?;
    value.set_item("player_changes_p95", report.player_changes_p95)?;
    value.set_item(
        "actions_per_player_turn_mean",
        report.actions_per_player_turn_mean,
    )?;
    value.set_item(
        "actions_per_player_turn_p95",
        report.actions_per_player_turn_p95,
    )?;
    value.set_item(
        "actions_per_player_turn_max",
        report.actions_per_player_turn_max,
    )?;
    value.set_item("depth_is_lower_bound", report.depth_is_lower_bound)?;
    value.set_item("estimated_tree_log10", report.estimated_tree_log10)?;
    Ok(value.unbind())
}

#[pyfunction]
#[pyo3(signature = (game, samples=128, max_depth=256, seed=0, game_params=None))]
pub fn analyze_structure(
    py: Python<'_>,
    game: &str,
    samples: u32,
    max_depth: u32,
    seed: u64,
    game_params: Option<BTreeMap<String, i64>>,
) -> PyResult<Py<PyDict>> {
    let config = EvaluationConfig {
        samples: NonZeroU32::new(samples)
            .ok_or_else(|| PyValueError::new_err("samples must be positive"))?,
        max_depth: NonZeroU32::new(max_depth)
            .ok_or_else(|| PyValueError::new_err("max_depth must be positive"))?,
        seed,
        ..Default::default()
    };
    let report = if game == "cant_stop" {
        if game_params.as_ref().is_some_and(|p| !p.is_empty()) {
            return Err(PyValueError::new_err(
                "cant_stop does not accept game parameters",
            ));
        }
        py.detach(|| {
            meeple_bots_evaluation::analyze_structure(&meeple_bots_cant_stop::CantStop, config)
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    } else {
        let game = parse_configured_game(game, game_params)?;
        py.detach(|| meeple_bots_catalog::analyze_structure(game, config))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    };
    structural_dict(py, &report)
}

#[pyfunction]
#[pyo3(signature = (game, median_depth, seed=0, iterations=None, exploration=std::f64::consts::SQRT_2, time_budget=None, selection_policy="uct", tree_reuse=false))]
#[allow(clippy::too_many_arguments)] // Python API keeps existing positional parameters compatible.
pub fn benchmark_so_ismcts(
    py: Python<'_>,
    game: &str,
    median_depth: u32,
    seed: u64,
    iterations: Option<u32>,
    exploration: f64,
    time_budget: Option<f64>,
    selection_policy: &str,
    tree_reuse: bool,
) -> PyResult<Py<PyList>> {
    let game = parse_configured_game(game, None)?;
    if !meeple_bots_catalog::game_search_capabilities(game)
        .search_agents
        .contains(&"so_ismcts")
    {
        return Err(PyValueError::new_err("game does not support SO-ISMCTS"));
    }
    let config = meeple_bots_so_ismcts::SoIsmctsConfig {
        budget: parse_search_budget(iterations, time_budget)?,
        exploration,
        tree_reuse,
        selection_policy: super::parse_bandit_policy(selection_policy)?,
    };
    let timings = py
        .detach(|| meeple_bots_catalog::benchmark_so_ismcts(game, config, median_depth, seed))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let rows = PyList::empty(py);
    for t in timings {
        let row = PyDict::new(py);
        row.set_item("sampled_ply", t.sampled_ply)?;
        row.set_item("phase", t.phase)?;
        row.set_item("milliseconds", t.milliseconds)?;
        row.set_item("iterations", t.iterations)?;
        row.set_item("determinizations", t.determinizations)?;
        row.set_item("nodes", t.nodes)?;
        row.set_item("action_edges", t.action_edges)?;
        row.set_item("legal_actions", t.legal_actions)?;
        row.set_item("root_visits", t.root_visits)?;
        row.set_item("mean_availability", t.mean_availability)?;
        row.set_item("mean_availability_ratio", t.mean_availability_ratio)?;
        row.set_item(
            "determinization_milliseconds",
            t.determinization_milliseconds,
        )?;
        row.set_item("terminal_simulations", t.terminal_simulations)?;
        row.set_item("cutoff_simulations", t.cutoff_simulations)?;
        rows.append(row)?;
    }
    Ok(rows.unbind())
}
