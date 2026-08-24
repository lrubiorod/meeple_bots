//! Structural game analysis and local MCTS cost estimation.

use std::{
    error::Error,
    fmt,
    num::NonZeroU32,
    time::{Duration, Instant},
};

use meeple_bots_core::{
    Agent, AgentError, DecisionContext, DeterministicGame, IllegalAction, PerfectInformationGame,
    PositionStatus, RandomSource, TwoPlayerZeroSumGame,
};
use meeple_bots_mcts_agent::{MctsAgent, MctsConfig};
use meeple_bots_simulation::SplitMix64;

const CALIBRATION_PROBE_ITERATIONS: u32 = 8;
const CALIBRATION_MIN_ITERATIONS: u32 = 16;
const CALIBRATION_MAX_ITERATIONS: u32 = 4_096;
const CALIBRATION_TARGET_MILLISECONDS: f64 = 25.0;
const STANDARD_TIME_BUDGETS_SECONDS: [u32; 5] = [1, 2, 5, 10, 20];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvaluationConfig {
    pub samples: NonZeroU32,
    pub max_depth: NonZeroU32,
    pub seed: u64,
    pub target_time: Duration,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            samples: NonZeroU32::new(128).expect("constant is non-zero"),
            max_depth: NonZeroU32::new(256).expect("constant is non-zero"),
            seed: 0,
            target_time: Duration::from_secs(5),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IterationBudgetEstimate {
    pub seconds: u32,
    pub iterations: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RolloutCostEstimate {
    pub rollout_depth: u32,
    pub approximate_player_turns: f64,
    pub milliseconds_per_iteration: f64,
    pub iteration_budgets: Vec<IterationBudgetEstimate>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SuggestedMctsExperiment {
    pub label: String,
    pub iterations: u32,
    pub iterations_capped: bool,
    pub rollout_depth: u32,
    pub approximate_player_turns: f64,
    pub estimated_decision_time_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampledDecisionTiming {
    pub sampled_ply: u32,
    pub milliseconds: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MctsAgentBenchmark {
    pub sampled_positions: u32,
    pub decision_time_mean_ms: f64,
    pub decision_time_p50_ms: f64,
    pub decision_time_p95_ms: f64,
    pub decision_time_max_ms: f64,
    pub milliseconds_per_iteration: f64,
    pub position_timings: Vec<SampledDecisionTiming>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GameEvaluationReport {
    pub samples: u32,
    pub max_depth: u32,
    pub terminal_rate: f64,
    pub initial_legal_actions: u32,
    pub effective_branching_factor: f64,
    pub player_turn_choice_product_log10: f64,
    pub depth_p50: u32,
    pub estimated_depth: u32,
    pub player_turn_depth_p50: u32,
    pub player_turn_depth_p95: u32,
    pub player_changes_p50: u32,
    pub player_changes_p95: u32,
    pub actions_per_player_turn_mean: f64,
    pub actions_per_player_turn_p95: u32,
    pub actions_per_player_turn_max: u32,
    pub depth_is_lower_bound: bool,
    pub estimated_tree_log10: f64,
    pub calibration_positions: u32,
    pub target_time_seconds: f64,
    pub rollout_costs: Vec<RolloutCostEstimate>,
    pub suggested_experiments: Vec<SuggestedMctsExperiment>,
    /// Compatibility alias for the Balanced experiment's rollout depth.
    pub recommended_rollout_depth: u32,
    /// Compatibility alias for the Balanced experiment's iteration count.
    pub recommended_iterations: u32,
    /// Whether the compatibility iteration count reached the MCTS `u32` limit.
    pub iterations_capped: bool,
    /// Compatibility alias for the Balanced experiment's measured iteration cost.
    pub milliseconds_per_iteration: f64,
    /// Compatibility alias for the Balanced experiment's estimated decision time.
    pub estimated_decision_time_ms: f64,
}

#[derive(Debug)]
pub enum EvaluationError {
    UnexpectedChance,
    NoLegalActions,
    IllegalAction(IllegalAction),
    Agent(AgentError),
    InvalidTargetTime,
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedChance => {
                formatter.write_str("game evaluation does not support chance transitions")
            }
            Self::NoLegalActions => {
                formatter.write_str("a non-terminal position has no legal actions")
            }
            Self::IllegalAction(error) => write!(formatter, "sampled action was rejected: {error}"),
            Self::Agent(error) => write!(formatter, "MCTS calibration failed: {error}"),
            Self::InvalidTargetTime => formatter.write_str("target time must be greater than zero"),
        }
    }
}

impl Error for EvaluationError {}

pub fn evaluate_game<G>(
    game: &G,
    config: EvaluationConfig,
) -> Result<GameEvaluationReport, EvaluationError>
where
    G: DeterministicGame + PerfectInformationGame + TwoPlayerZeroSumGame,
    G::State: Clone,
    G::Action: Clone,
{
    if config.target_time.is_zero() {
        return Err(EvaluationError::InvalidTargetTime);
    }
    let initial_state = game.initial_state();
    match game.status(&initial_state) {
        PositionStatus::PlayerTurn(_) => {}
        PositionStatus::Chance => return Err(EvaluationError::UnexpectedChance),
        PositionStatus::Terminal => return Err(EvaluationError::NoLegalActions),
        _ => return Err(EvaluationError::UnexpectedChance),
    }
    let initial_legal_actions = game.legal_actions(&initial_state).count() as u32;
    if initial_legal_actions == 0 {
        return Err(EvaluationError::NoLegalActions);
    }

    let sampled = sample_game_tree(game, config)?;
    let estimated_depth = sampled.p95_depth.max(1);
    let calibration_states =
        sample_calibration_states(game, sampled.p50_depth, config.seed ^ 0xD1B5_4A32_D192_ED03)?;
    let mut rollout_costs = Vec::new();
    for rollout_depth in candidate_rollout_depths(estimated_depth) {
        let milliseconds_per_iteration = calibrate_iteration_cost(
            game,
            &calibration_states,
            rollout_depth,
            config.seed ^ u64::from(rollout_depth),
        )?;
        rollout_costs.push(RolloutCostEstimate {
            rollout_depth,
            approximate_player_turns: f64::from(rollout_depth)
                / sampled.actions_per_player_turn_mean,
            milliseconds_per_iteration,
            iteration_budgets: STANDARD_TIME_BUDGETS_SECONDS
                .into_iter()
                .map(|seconds| IterationBudgetEstimate {
                    seconds,
                    iterations: iterations_for_time(milliseconds_per_iteration, f64::from(seconds)),
                })
                .collect(),
        });
    }
    let suggested_experiments =
        build_suggested_experiments(&rollout_costs, config.target_time.as_secs_f64());
    let balanced = suggested_experiments
        .iter()
        .find(|experiment| experiment.label == "Balanced")
        .expect("evaluation always creates a Balanced experiment");
    let balanced_cost = rollout_costs
        .iter()
        .find(|cost| cost.rollout_depth == balanced.rollout_depth)
        .expect("Balanced uses a calibrated rollout depth");
    let recommended_rollout_depth = balanced.rollout_depth;
    let recommended_iterations = balanced.iterations;
    let iterations_capped = balanced.iterations_capped;
    let milliseconds_per_iteration = balanced_cost.milliseconds_per_iteration;
    let estimated_decision_time_ms = balanced.estimated_decision_time_ms;
    let estimated_tree_log10 = f64::from(initial_legal_actions).log10()
        + f64::from(estimated_depth.saturating_sub(1)) * sampled.effective_branching_factor.log10();

    Ok(GameEvaluationReport {
        samples: config.samples.get(),
        max_depth: config.max_depth.get(),
        terminal_rate: f64::from(sampled.completed_samples) / f64::from(config.samples.get()),
        initial_legal_actions,
        effective_branching_factor: sampled.effective_branching_factor,
        player_turn_choice_product_log10: sampled.player_turn_choice_product_log10,
        depth_p50: sampled.p50_depth,
        estimated_depth,
        player_turn_depth_p50: sampled.p50_player_turn_depth,
        player_turn_depth_p95: sampled.p95_player_turn_depth,
        player_changes_p50: sampled.p50_player_changes,
        player_changes_p95: sampled.p95_player_changes,
        actions_per_player_turn_mean: sampled.actions_per_player_turn_mean,
        actions_per_player_turn_p95: sampled.p95_actions_per_player_turn,
        actions_per_player_turn_max: sampled.max_actions_per_player_turn,
        depth_is_lower_bound: sampled.completed_samples < config.samples.get(),
        estimated_tree_log10,
        calibration_positions: calibration_states.len() as u32,
        target_time_seconds: config.target_time.as_secs_f64(),
        rollout_costs,
        suggested_experiments,
        recommended_rollout_depth,
        recommended_iterations,
        iterations_capped,
        milliseconds_per_iteration,
        estimated_decision_time_ms,
    })
}

pub fn benchmark_mcts_agent<G, A>(
    game: &G,
    agent: &mut A,
    iterations: NonZeroU32,
    median_depth: u32,
    seed: u64,
) -> Result<MctsAgentBenchmark, EvaluationError>
where
    G: DeterministicGame + PerfectInformationGame + TwoPlayerZeroSumGame,
    G::State: Clone,
    G::Action: Clone,
    A: Agent<G>,
{
    let states = sample_calibration_states(game, median_depth, seed ^ 0xD1B5_4A32_D192_ED03)?;
    let mut position_timings = Vec::with_capacity(states.len());
    for (position_index, (sampled_ply, state)) in states.iter().enumerate() {
        let player = match game.status(state) {
            PositionStatus::PlayerTurn(player) => player,
            PositionStatus::Chance => return Err(EvaluationError::UnexpectedChance),
            PositionStatus::Terminal => continue,
            _ => return Err(EvaluationError::UnexpectedChance),
        };
        let mut rng =
            SplitMix64::new(seed.wrapping_add(position_index as u64) ^ 0x94D0_49BB_1331_11EB);
        let started = Instant::now();
        agent
            .select_action(DecisionContext::new(game, state, player), &mut rng)
            .map_err(EvaluationError::Agent)?;
        position_timings.push(SampledDecisionTiming {
            sampled_ply: *sampled_ply,
            milliseconds: (started.elapsed().as_secs_f64() * 1_000.0).max(f64::EPSILON),
        });
    }
    if position_timings.is_empty() {
        return Err(EvaluationError::NoLegalActions);
    }

    let decision_time_mean_ms = position_timings
        .iter()
        .map(|timing| timing.milliseconds)
        .sum::<f64>()
        / position_timings.len() as f64;
    let mut sorted_timings: Vec<_> = position_timings
        .iter()
        .map(|timing| timing.milliseconds)
        .collect();
    sorted_timings.sort_by(f64::total_cmp);

    Ok(MctsAgentBenchmark {
        sampled_positions: position_timings.len() as u32,
        decision_time_mean_ms,
        decision_time_p50_ms: percentile_f64(&sorted_timings, 50),
        decision_time_p95_ms: percentile_f64(&sorted_timings, 95),
        decision_time_max_ms: *sorted_timings
            .last()
            .expect("non-empty timings have a maximum"),
        milliseconds_per_iteration: decision_time_mean_ms / f64::from(iterations.get()),
        position_timings,
    })
}

struct SampledMetrics {
    completed_samples: u32,
    effective_branching_factor: f64,
    player_turn_choice_product_log10: f64,
    p50_depth: u32,
    p95_depth: u32,
    p50_player_turn_depth: u32,
    p95_player_turn_depth: u32,
    p50_player_changes: u32,
    p95_player_changes: u32,
    actions_per_player_turn_mean: f64,
    p95_actions_per_player_turn: u32,
    max_actions_per_player_turn: u32,
}

fn sample_game_tree<G>(
    game: &G,
    config: EvaluationConfig,
) -> Result<SampledMetrics, EvaluationError>
where
    G: DeterministicGame,
{
    let mut rng = SplitMix64::new(config.seed);
    let mut branch_counts = Vec::new();
    let mut depths = Vec::with_capacity(config.samples.get() as usize);
    let mut player_turn_depths = Vec::with_capacity(config.samples.get() as usize);
    let mut player_changes = Vec::with_capacity(config.samples.get() as usize);
    let mut actions_per_player_turn = Vec::new();
    let mut player_turn_choice_logs = Vec::new();
    let mut completed_samples = 0;

    for _ in 0..config.samples.get() {
        let mut state = game.initial_state();
        let mut depth = 0;
        let mut current_player = None;
        let mut current_turn_actions = 0;
        let mut current_turn_choice_log = 0.0;
        let mut turn_count = 0;
        loop {
            match game.status(&state) {
                PositionStatus::Terminal => {
                    completed_samples += 1;
                    finish_player_turn(
                        &mut current_turn_actions,
                        &mut current_turn_choice_log,
                        &mut turn_count,
                        &mut actions_per_player_turn,
                        &mut player_turn_choice_logs,
                    );
                    record_path_metrics(
                        depth,
                        turn_count,
                        &mut depths,
                        &mut player_turn_depths,
                        &mut player_changes,
                    );
                    break;
                }
                PositionStatus::PlayerTurn(player) => {
                    if depth >= config.max_depth.get() {
                        finish_player_turn(
                            &mut current_turn_actions,
                            &mut current_turn_choice_log,
                            &mut turn_count,
                            &mut actions_per_player_turn,
                            &mut player_turn_choice_logs,
                        );
                        record_path_metrics(
                            depth,
                            turn_count,
                            &mut depths,
                            &mut player_turn_depths,
                            &mut player_changes,
                        );
                        break;
                    }
                    if current_player.is_some_and(|active| active != player) {
                        finish_player_turn(
                            &mut current_turn_actions,
                            &mut current_turn_choice_log,
                            &mut turn_count,
                            &mut actions_per_player_turn,
                            &mut player_turn_choice_logs,
                        );
                    }
                    current_player = Some(player);
                    let actions: Vec<_> = game.legal_actions(&state).collect();
                    let action_index = rng
                        .index(actions.len())
                        .ok_or(EvaluationError::NoLegalActions)?;
                    branch_counts.push(actions.len() as u32);
                    current_turn_actions += 1;
                    current_turn_choice_log += (actions.len() as f64).ln();
                    game.apply_action(&mut state, &actions[action_index])
                        .map_err(EvaluationError::IllegalAction)?;
                    depth += 1;
                }
                PositionStatus::Chance => return Err(EvaluationError::UnexpectedChance),
                _ => return Err(EvaluationError::UnexpectedChance),
            }
        }
    }

    depths.sort_unstable();
    player_turn_depths.sort_unstable();
    player_changes.sort_unstable();
    actions_per_player_turn.sort_unstable();
    let branch_samples = branch_counts.len() as f64;
    let log_branch_sum: f64 = branch_counts
        .iter()
        .map(|value| f64::from(*value).ln())
        .sum();
    let player_turn_count = actions_per_player_turn.len() as f64;

    Ok(SampledMetrics {
        completed_samples,
        effective_branching_factor: (log_branch_sum / branch_samples).exp(),
        player_turn_choice_product_log10: player_turn_choice_logs.iter().sum::<f64>()
            / player_turn_count
            / 10_f64.ln(),
        p50_depth: percentile(&depths, 50),
        p95_depth: percentile(&depths, 95),
        p50_player_turn_depth: percentile(&player_turn_depths, 50),
        p95_player_turn_depth: percentile(&player_turn_depths, 95),
        p50_player_changes: percentile(&player_changes, 50),
        p95_player_changes: percentile(&player_changes, 95),
        actions_per_player_turn_mean: actions_per_player_turn
            .iter()
            .map(|value| f64::from(*value))
            .sum::<f64>()
            / player_turn_count,
        p95_actions_per_player_turn: percentile(&actions_per_player_turn, 95),
        max_actions_per_player_turn: actions_per_player_turn.last().copied().unwrap_or(0),
    })
}

fn finish_player_turn(
    actions: &mut u32,
    choice_log: &mut f64,
    turn_count: &mut u32,
    actions_per_player_turn: &mut Vec<u32>,
    player_turn_choice_logs: &mut Vec<f64>,
) {
    if *actions == 0 {
        return;
    }
    actions_per_player_turn.push(*actions);
    player_turn_choice_logs.push(*choice_log);
    *turn_count += 1;
    *actions = 0;
    *choice_log = 0.0;
}

fn record_path_metrics(
    depth: u32,
    turn_count: u32,
    depths: &mut Vec<u32>,
    player_turn_depths: &mut Vec<u32>,
    player_changes: &mut Vec<u32>,
) {
    depths.push(depth);
    player_turn_depths.push(turn_count);
    player_changes.push(turn_count.saturating_sub(1));
}

fn sample_calibration_states<G>(
    game: &G,
    median_depth: u32,
    seed: u64,
) -> Result<Vec<(u32, G::State)>, EvaluationError>
where
    G: DeterministicGame,
    G::State: Clone,
    G::Action: Clone,
{
    let mut states = vec![(0, game.initial_state())];
    let mut target_depths = [median_depth / 3, median_depth.saturating_mul(2) / 3];
    target_depths.sort_unstable();

    for target_depth in target_depths {
        if target_depth == 0 {
            continue;
        }
        let mut state = game.initial_state();
        let mut rng = SplitMix64::new(seed ^ u64::from(target_depth));
        let mut reached = true;
        for _ in 0..target_depth {
            match game.status(&state) {
                PositionStatus::PlayerTurn(_) => {
                    let actions: Vec<_> = game.legal_actions(&state).collect();
                    let index = rng
                        .index(actions.len())
                        .ok_or(EvaluationError::NoLegalActions)?;
                    game.apply_action(&mut state, &actions[index])
                        .map_err(EvaluationError::IllegalAction)?;
                }
                PositionStatus::Terminal => {
                    reached = false;
                    break;
                }
                PositionStatus::Chance => return Err(EvaluationError::UnexpectedChance),
                _ => return Err(EvaluationError::UnexpectedChance),
            }
        }
        if reached && matches!(game.status(&state), PositionStatus::PlayerTurn(_)) {
            states.push((target_depth, state));
        }
    }

    Ok(states)
}

fn candidate_rollout_depths(p95_depth: u32) -> Vec<u32> {
    let full_depth = p95_depth.max(1);
    let mut depths = vec![full_depth];
    for denominator in [6, 3] {
        depths.push(nearest_power_of_two(full_depth.div_ceil(denominator)));
    }
    let two_thirds = (u64::from(full_depth) * 2).div_ceil(3) as u32;
    depths.push(nearest_power_of_two(two_thirds));
    for depth in &mut depths {
        *depth = (*depth).clamp(1, full_depth);
    }
    depths.sort_unstable();
    depths.dedup();
    depths
}

fn nearest_power_of_two(value: u32) -> u32 {
    if value <= 1 {
        return 1;
    }
    let upper = u64::from(value).next_power_of_two();
    let lower = upper / 2;
    let value = u64::from(value);
    if value - lower < upper - value {
        lower as u32
    } else {
        upper.min(u64::from(u32::MAX)) as u32
    }
}

fn calibrate_iteration_cost<G>(
    game: &G,
    states: &[(u32, G::State)],
    rollout_depth: u32,
    seed: u64,
) -> Result<f64, EvaluationError>
where
    G: DeterministicGame + PerfectInformationGame + TwoPlayerZeroSumGame,
    G::State: Clone,
    G::Action: Clone,
{
    let mut timings = Vec::with_capacity(states.len());
    for (position_index, (_, state)) in states.iter().enumerate() {
        let player = match game.status(state) {
            PositionStatus::PlayerTurn(player) => player,
            PositionStatus::Chance => return Err(EvaluationError::UnexpectedChance),
            PositionStatus::Terminal => continue,
            _ => return Err(EvaluationError::UnexpectedChance),
        };
        let position_seed = seed.wrapping_add(position_index as u64);
        let probe_cost = measure_mcts_cost(
            game,
            state,
            player,
            rollout_depth,
            CALIBRATION_PROBE_ITERATIONS,
            position_seed,
        )?;
        let measured_iterations = (CALIBRATION_TARGET_MILLISECONDS / probe_cost)
            .round()
            .clamp(
                f64::from(CALIBRATION_MIN_ITERATIONS),
                f64::from(CALIBRATION_MAX_ITERATIONS),
            ) as u32;
        timings.push(measure_mcts_cost(
            game,
            state,
            player,
            rollout_depth,
            measured_iterations,
            position_seed ^ 0x94D0_49BB_1331_11EB,
        )?);
    }

    if timings.is_empty() {
        return Err(EvaluationError::NoLegalActions);
    }
    timings.sort_by(f64::total_cmp);
    Ok(timings[timings.len() / 2].max(f64::EPSILON))
}

fn measure_mcts_cost<G>(
    game: &G,
    state: &G::State,
    player: meeple_bots_core::PlayerId,
    rollout_depth: u32,
    iterations: u32,
    seed: u64,
) -> Result<f64, EvaluationError>
where
    G: DeterministicGame + PerfectInformationGame + TwoPlayerZeroSumGame,
    G::State: Clone,
    G::Action: Clone,
{
    let iterations = NonZeroU32::new(iterations).expect("calibration count is non-zero");
    let mut agent = MctsAgent::new(MctsConfig {
        iterations,
        exploration: std::f64::consts::SQRT_2,
        rollout_depth,
    });
    let mut rng = SplitMix64::new(seed);
    let started = Instant::now();
    agent
        .select_action(DecisionContext::new(game, state, player), &mut rng)
        .map_err(EvaluationError::Agent)?;
    Ok((started.elapsed().as_secs_f64() * 1_000.0 / f64::from(iterations.get())).max(f64::EPSILON))
}

fn iterations_for_time(milliseconds_per_iteration: f64, seconds: f64) -> u64 {
    let raw_iterations = seconds * 1_000.0 / milliseconds_per_iteration;
    round_down_two_significant(raw_iterations).max(1)
}

fn round_down_two_significant(value: f64) -> u64 {
    if !value.is_finite() || value >= u64::MAX as f64 {
        return u64::MAX;
    }
    if value <= 1.0 {
        return 1;
    }
    let magnitude = 10_f64.powf(value.log10().floor() - 1.0).max(1.0);
    (value / magnitude).floor() as u64 * magnitude as u64
}

fn build_suggested_experiments(
    costs: &[RolloutCostEstimate],
    target_seconds: f64,
) -> Vec<SuggestedMctsExperiment> {
    let balanced_index = (costs.len() - 1) / 2;
    let balanced_cost = &costs[balanced_index];
    let mut experiments = vec![
        make_experiment("Fast", balanced_cost, (target_seconds / 2.0).max(0.1)),
        make_experiment("Balanced", balanced_cost, target_seconds),
        make_experiment("Wide", balanced_cost, target_seconds * 2.0),
    ];
    if let Some(deep_cost) = costs.get(balanced_index + 1) {
        experiments.push(make_experiment("Deep", deep_cost, target_seconds));
    }
    experiments
}

fn make_experiment(
    label: &str,
    cost: &RolloutCostEstimate,
    seconds: f64,
) -> SuggestedMctsExperiment {
    let raw_iterations = iterations_for_time(cost.milliseconds_per_iteration, seconds);
    let iterations_capped = raw_iterations > u64::from(u32::MAX);
    let iterations = raw_iterations.min(u64::from(u32::MAX)) as u32;
    SuggestedMctsExperiment {
        label: label.to_owned(),
        iterations,
        iterations_capped,
        rollout_depth: cost.rollout_depth,
        approximate_player_turns: cost.approximate_player_turns,
        estimated_decision_time_ms: f64::from(iterations) * cost.milliseconds_per_iteration,
    }
}

fn percentile(sorted: &[u32], percentage: usize) -> u32 {
    let rank = (percentage * sorted.len()).div_ceil(100).max(1);
    sorted[rank - 1]
}

fn percentile_f64(sorted: &[f64], percentage: usize) -> f64 {
    let rank = (percentage * sorted.len()).div_ceil(100).max(1);
    sorted[rank - 1]
}

#[cfg(test)]
mod tests {
    use meeple_bots_connect_four::ConnectFour;
    use meeple_bots_core::{Game, PlayerId};
    use meeple_bots_tic_tac_toe::TicTacToe;

    use super::*;

    fn small_config() -> EvaluationConfig {
        EvaluationConfig {
            samples: NonZeroU32::new(16).unwrap(),
            ..EvaluationConfig::default()
        }
    }

    #[test]
    fn tic_tac_toe_report_has_known_bounds() {
        let report = evaluate_game(&TicTacToe, small_config()).unwrap();

        assert_eq!(report.initial_legal_actions, 9);
        assert_eq!(report.terminal_rate, 1.0);
        assert!(report.estimated_depth <= 9);
        assert!(!report.depth_is_lower_bound);
        assert!(report.depth_p50 <= report.estimated_depth);
        assert_eq!(report.player_turn_depth_p95, report.estimated_depth);
        assert_eq!(report.actions_per_player_turn_mean, 1.0);
        assert!(!report.rollout_costs.is_empty());
        assert!(report.rollout_costs.iter().all(|cost| {
            cost.iteration_budgets
                .windows(2)
                .all(|pair| pair[0].iterations <= pair[1].iterations)
        }));
        assert!(
            report
                .suggested_experiments
                .iter()
                .any(|experiment| experiment.label == "Balanced")
        );
        assert!(report.milliseconds_per_iteration.is_finite());
        assert!(report.milliseconds_per_iteration > 0.0);
        assert_eq!(
            report.estimated_decision_time_ms,
            f64::from(report.recommended_iterations) * report.milliseconds_per_iteration
        );
    }

    #[test]
    fn connect_four_report_has_known_bounds() {
        let report = evaluate_game(&ConnectFour, small_config()).unwrap();

        assert_eq!(report.initial_legal_actions, 7);
        assert!(report.estimated_depth <= 42);
    }

    #[test]
    fn configured_agent_benchmark_times_exact_search_on_shared_positions() {
        let iterations = NonZeroU32::new(4).unwrap();
        let mut agent = MctsAgent::new(MctsConfig {
            iterations,
            rollout_depth: 4,
            ..MctsConfig::default()
        });

        let benchmark = benchmark_mcts_agent(&TicTacToe, &mut agent, iterations, 6, 42).unwrap();

        assert_eq!(benchmark.sampled_positions, 3);
        assert_eq!(
            benchmark
                .position_timings
                .iter()
                .map(|timing| timing.sampled_ply)
                .collect::<Vec<_>>(),
            vec![0, 2, 4]
        );
        assert!(benchmark.decision_time_mean_ms > 0.0);
        assert!(benchmark.decision_time_p50_ms <= benchmark.decision_time_p95_ms);
        assert_eq!(
            benchmark.decision_time_p95_ms,
            benchmark.decision_time_max_ms
        );
        assert_eq!(
            benchmark.milliseconds_per_iteration,
            benchmark.decision_time_mean_ms / f64::from(iterations.get())
        );
    }

    #[test]
    fn structural_metrics_are_reproducible() {
        let first = evaluate_game(&TicTacToe, small_config()).unwrap();
        let repeated = evaluate_game(&TicTacToe, small_config()).unwrap();

        assert_eq!(first.terminal_rate, repeated.terminal_rate);
        assert_eq!(
            first.effective_branching_factor,
            repeated.effective_branching_factor
        );
        assert_eq!(first.estimated_depth, repeated.estimated_depth);
        assert_eq!(first.estimated_tree_log10, repeated.estimated_tree_log10);
        assert_eq!(first.player_turn_depth_p95, repeated.player_turn_depth_p95);
        assert_eq!(
            first.actions_per_player_turn_mean,
            repeated.actions_per_player_turn_mean
        );
    }

    #[test]
    fn depth_limit_is_reported_as_a_lower_bound() {
        let report = evaluate_game(
            &TicTacToe,
            EvaluationConfig {
                max_depth: NonZeroU32::new(1).unwrap(),
                ..small_config()
            },
        )
        .unwrap();

        assert_eq!(report.terminal_rate, 0.0);
        assert_eq!(report.estimated_depth, 1);
        assert!(report.depth_is_lower_bound);
    }

    #[test]
    fn candidate_depths_span_short_to_full_horizons() {
        assert_eq!(candidate_rollout_depths(96), vec![16, 32, 64, 96]);
        assert_eq!(candidate_rollout_depths(1), vec![1]);
        assert_eq!(candidate_rollout_depths(u32::MAX).last(), Some(&u32::MAX));
    }

    #[test]
    fn iteration_budget_is_derived_from_measured_cost() {
        assert_eq!(iterations_for_time(0.02, 1.0), 50_000);
        assert_eq!(iterations_for_time(0.02, 5.0), 250_000);
    }

    #[test]
    fn experiments_compare_width_and_depth_around_the_target_time() {
        let costs: Vec<_> = [16, 32, 64]
            .into_iter()
            .map(|rollout_depth| RolloutCostEstimate {
                rollout_depth,
                approximate_player_turns: f64::from(rollout_depth),
                milliseconds_per_iteration: 1.0,
                iteration_budgets: Vec::new(),
            })
            .collect();

        let experiments = build_suggested_experiments(&costs, 5.0);

        assert_eq!(experiments[0].iterations, 2_500);
        assert_eq!(experiments[1].iterations, 5_000);
        assert_eq!(experiments[2].iterations, 10_000);
        assert_eq!(experiments[0].rollout_depth, 32);
        assert_eq!(experiments[3].rollout_depth, 64);
        assert_eq!(experiments[3].iterations, 5_000);
    }

    #[derive(Clone, Copy)]
    struct PhasedGame;

    impl Game for PhasedGame {
        type State = u8;
        type Action = ();
        type Observation<'a> = &'a u8;
        type LegalActions<'a> = std::option::IntoIter<()>;

        fn player_count(&self) -> u8 {
            2
        }

        fn initial_state(&self) -> Self::State {
            0
        }

        fn status(&self, state: &Self::State) -> PositionStatus {
            match *state {
                0..=2 | 5 => PositionStatus::PlayerTurn(PlayerId::FIRST),
                3..=4 => PositionStatus::PlayerTurn(PlayerId::SECOND),
                _ => PositionStatus::Terminal,
            }
        }

        fn legal_actions<'a>(&'a self, state: &'a Self::State) -> Self::LegalActions<'a> {
            (*state < 6).then_some(()).into_iter()
        }

        fn apply_action(
            &self,
            state: &mut Self::State,
            _action: &Self::Action,
        ) -> Result<(), IllegalAction> {
            if *state >= 6 {
                return Err(IllegalAction::new("game is over"));
            }
            *state += 1;
            Ok(())
        }

        fn observation<'a>(
            &'a self,
            state: &'a Self::State,
            _player: PlayerId,
        ) -> Self::Observation<'a> {
            state
        }

        fn terminal_utility(&self, state: &Self::State, _player: PlayerId) -> Option<f32> {
            (*state >= 6).then_some(0.0)
        }
    }

    impl DeterministicGame for PhasedGame {}

    #[test]
    fn consecutive_actions_are_grouped_into_player_turns() {
        let metrics = sample_game_tree(
            &PhasedGame,
            EvaluationConfig {
                samples: NonZeroU32::new(4).unwrap(),
                max_depth: NonZeroU32::new(10).unwrap(),
                ..EvaluationConfig::default()
            },
        )
        .unwrap();

        assert_eq!(metrics.p50_depth, 6);
        assert_eq!(metrics.p95_depth, 6);
        assert_eq!(metrics.p50_player_turn_depth, 3);
        assert_eq!(metrics.p95_player_turn_depth, 3);
        assert_eq!(metrics.p50_player_changes, 2);
        assert_eq!(metrics.p95_player_changes, 2);
        assert_eq!(metrics.actions_per_player_turn_mean, 2.0);
        assert_eq!(metrics.p95_actions_per_player_turn, 3);
        assert_eq!(metrics.max_actions_per_player_turn, 3);
        assert_eq!(metrics.effective_branching_factor, 1.0);
        assert_eq!(metrics.player_turn_choice_product_log10, 0.0);
    }
}
