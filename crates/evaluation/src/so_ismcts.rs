//! Observation-only calibration; sampled worlds never enter a report.
use super::{EvaluationError, sample_calibration_states};
use meeple_bots_core::{
    DeterminizedWorld, Game, ImperfectInformationGame, PositionStatus, TwoPlayerZeroSumGame,
};
use meeple_bots_simulation::SplitMix64;
use meeple_bots_so_ismcts::{SoIsmctsAgent, SoIsmctsConfig};
use std::time::Instant;

#[derive(Clone, Debug, PartialEq)]
pub struct SoIsmctsTiming {
    pub phase: Option<&'static str>,
    pub sampled_ply: u32,
    pub milliseconds: f64,
    pub iterations: u64,
    pub determinizations: u64,
    pub nodes: usize,
    pub action_edges: usize,
    pub legal_actions: usize,
    pub root_visits: Vec<u64>,
    pub mean_availability: f64,
    pub mean_availability_ratio: f64,
    pub determinization_milliseconds: f64,
    pub terminal_simulations: u64,
    pub cutoff_simulations: u64,
}

pub fn benchmark<G, W, O>(
    game: &G,
    config: SoIsmctsConfig,
    median_depth: u32,
    seed: u64,
) -> Result<Vec<SoIsmctsTiming>, EvaluationError>
where
    G: TwoPlayerZeroSumGame
        + ImperfectInformationGame<Determinization = W>
        + for<'a> Game<Observation<'a> = O>
        + 'static,
    G::State: Clone,
    G::Action: Clone + Eq,
    W: DeterminizedWorld<Action = G::Action, Observation = O>,
    O: Clone + Eq,
{
    let agent = SoIsmctsAgent { config };
    let states = sample_calibration_states(game, median_depth, seed ^ 0xD1B5_4A32_D192_ED03)?;
    let mut timings = Vec::new();
    for (i, (ply, state)) in states.iter().enumerate() {
        let PositionStatus::PlayerTurn(observer) = game.status(state) else {
            continue;
        };
        // Construct the restricted input outside the search boundary. The agent gets no State.
        let observation = game.observation(state, observer);
        let legal: Vec<_> = game.legal_actions(state).collect();
        let mut search_rng = SplitMix64::new(seed.wrapping_add(i as u64) ^ 0x94D0_49BB_1331_11EB);
        let started = Instant::now();
        let result = agent
            .search(game, &observation, observer, &legal, &mut search_rng)
            .map_err(EvaluationError::Agent)?;
        let milliseconds = started.elapsed().as_secs_f64() * 1000.;
        // Independent microbenchmark stream: does not change search samples or environment.
        let mut sample_rng = SplitMix64::new(seed.wrapping_add(i as u64) ^ 0xA076_1D64_78BD_642F);
        let started = Instant::now();
        for _ in 0..256 {
            std::hint::black_box(
                game.sample_determinization(&observation, observer, &mut sample_rng)
                    .map_err(EvaluationError::IllegalAction)?,
            );
        }
        let determinization_milliseconds = started.elapsed().as_secs_f64() * 1000. / 256.;
        let edges = result.diagnostics.action_edges.max(1) as f64;
        timings.push(SoIsmctsTiming {
            phase: game.diagnostic_phase(state),
            sampled_ply: *ply,
            milliseconds,
            iterations: result.diagnostics.completed_iterations,
            determinizations: result.diagnostics.determinizations_sampled,
            nodes: result.diagnostics.tree_nodes,
            action_edges: result.diagnostics.action_edges,
            legal_actions: legal.len(),
            root_visits: result.nodes[0].edges.iter().map(|e| e.visits).collect(),
            mean_availability: result
                .nodes
                .iter()
                .flat_map(|n| &n.edges)
                .map(|e| e.availability as f64)
                .sum::<f64>()
                / edges,
            mean_availability_ratio: result
                .nodes
                .iter()
                .flat_map(|n| {
                    n.edges
                        .iter()
                        .map(move |e| e.availability as f64 / n.visits.max(1) as f64)
                })
                .sum::<f64>()
                / edges,
            determinization_milliseconds,
            terminal_simulations: result.diagnostics.terminal_simulations,
            cutoff_simulations: result.diagnostics.cutoff_simulations,
        });
    }
    if timings.is_empty() {
        return Err(EvaluationError::NoLegalActions);
    }
    Ok(timings)
}
