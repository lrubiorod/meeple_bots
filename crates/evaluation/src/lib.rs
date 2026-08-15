//! Structural game analysis and local MCTS cost estimation.

use std::{error::Error, fmt, num::NonZeroU32, time::Instant};

use meeple_bots_core::{
    Agent, AgentError, DecisionContext, DeterministicGame, IllegalAction, PerfectInformationGame,
    PositionStatus, RandomSource, TwoPlayerZeroSumGame,
};
use meeple_bots_mcts_agent::{MctsAgent, MctsConfig};
use meeple_bots_simulation::SplitMix64;

const CALIBRATION_ITERATIONS: u32 = 100;
const CALIBRATION_REPEATS: usize = 3;
const MAX_RECOMMENDED_ITERATIONS: u32 = 1_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvaluationConfig {
    pub samples: NonZeroU32,
    pub max_depth: NonZeroU32,
    pub seed: u64,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            samples: NonZeroU32::new(128).expect("constant is non-zero"),
            max_depth: NonZeroU32::new(256).expect("constant is non-zero"),
            seed: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GameEvaluationReport {
    pub samples: u32,
    pub max_depth: u32,
    pub terminal_rate: f64,
    pub initial_legal_actions: u32,
    pub effective_branching_factor: f64,
    pub estimated_depth: u32,
    pub depth_is_lower_bound: bool,
    pub estimated_tree_log10: f64,
    pub recommended_rollout_depth: u32,
    pub recommended_iterations: u32,
    pub iterations_capped: bool,
    pub milliseconds_per_iteration: f64,
    pub estimated_decision_time_ms: f64,
}

#[derive(Debug)]
pub enum EvaluationError {
    UnexpectedChance,
    NoLegalActions,
    IllegalAction(IllegalAction),
    Agent(AgentError),
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
    let initial_state = game.initial_state();
    let initial_legal_actions = game.legal_actions(&initial_state).count() as u32;
    if initial_legal_actions == 0 {
        return Err(EvaluationError::NoLegalActions);
    }

    let sampled = sample_game_tree(game, config)?;
    let estimated_depth = sampled.p95_depth.max(1);
    let estimated_iterations = f64::from(initial_legal_actions)
        * sampled.effective_branching_factor
        * f64::from(estimated_depth).powi(2);
    let rounded_iterations = round_up_nice(estimated_iterations);
    let iterations_capped = rounded_iterations > u64::from(MAX_RECOMMENDED_ITERATIONS);
    let recommended_iterations = rounded_iterations
        .min(u64::from(MAX_RECOMMENDED_ITERATIONS))
        .max(u64::from(initial_legal_actions)) as u32;
    let milliseconds_per_iteration = calibrate_iteration_cost(game, estimated_depth, config.seed)?;
    let estimated_tree_log10 = f64::from(initial_legal_actions).log10()
        + f64::from(estimated_depth.saturating_sub(1)) * sampled.effective_branching_factor.log10();

    Ok(GameEvaluationReport {
        samples: config.samples.get(),
        max_depth: config.max_depth.get(),
        terminal_rate: f64::from(sampled.completed_samples) / f64::from(config.samples.get()),
        initial_legal_actions,
        effective_branching_factor: sampled.effective_branching_factor,
        estimated_depth,
        depth_is_lower_bound: sampled.completed_samples < config.samples.get(),
        estimated_tree_log10,
        recommended_rollout_depth: estimated_depth,
        recommended_iterations,
        iterations_capped,
        milliseconds_per_iteration,
        estimated_decision_time_ms: f64::from(recommended_iterations) * milliseconds_per_iteration,
    })
}

struct SampledMetrics {
    completed_samples: u32,
    effective_branching_factor: f64,
    p95_depth: u32,
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
    let mut completed_samples = 0;

    for _ in 0..config.samples.get() {
        let mut state = game.initial_state();
        let mut depth = 0;
        loop {
            match game.status(&state) {
                PositionStatus::Terminal => {
                    completed_samples += 1;
                    depths.push(depth);
                    break;
                }
                PositionStatus::PlayerTurn(_) => {
                    if depth >= config.max_depth.get() {
                        depths.push(depth);
                        break;
                    }
                    let actions: Vec<_> = game.legal_actions(&state).collect();
                    let action_index = rng
                        .index(actions.len())
                        .ok_or(EvaluationError::NoLegalActions)?;
                    branch_counts.push(actions.len() as u32);
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
    let branch_samples = branch_counts.len() as f64;
    let log_branch_sum: f64 = branch_counts
        .iter()
        .map(|value| f64::from(*value).ln())
        .sum();

    Ok(SampledMetrics {
        completed_samples,
        effective_branching_factor: (log_branch_sum / branch_samples).exp(),
        p95_depth: percentile(&depths, 95),
    })
}

fn calibrate_iteration_cost<G>(
    game: &G,
    rollout_depth: u32,
    seed: u64,
) -> Result<f64, EvaluationError>
where
    G: DeterministicGame + PerfectInformationGame + TwoPlayerZeroSumGame,
    G::State: Clone,
    G::Action: Clone,
{
    let state = game.initial_state();
    let player = match game.status(&state) {
        PositionStatus::PlayerTurn(player) => player,
        PositionStatus::Chance => return Err(EvaluationError::UnexpectedChance),
        PositionStatus::Terminal => return Err(EvaluationError::NoLegalActions),
        _ => return Err(EvaluationError::UnexpectedChance),
    };
    let iterations =
        NonZeroU32::new(CALIBRATION_ITERATIONS).expect("calibration count is non-zero");
    let mut timings = [0.0; CALIBRATION_REPEATS];

    for (repeat, timing) in timings.iter_mut().enumerate() {
        let mut agent = MctsAgent::new(MctsConfig {
            iterations,
            exploration: std::f64::consts::SQRT_2,
            rollout_depth,
        });
        let mut rng = SplitMix64::new(seed.wrapping_add(repeat as u64));
        let started = Instant::now();
        agent
            .select_action(DecisionContext::new(game, &state, player), &mut rng)
            .map_err(EvaluationError::Agent)?;
        *timing = started.elapsed().as_secs_f64() * 1_000.0 / f64::from(iterations.get());
    }

    timings.sort_by(f64::total_cmp);
    Ok(timings[CALIBRATION_REPEATS / 2].max(f64::EPSILON))
}

fn round_up_nice(value: f64) -> u64 {
    if !value.is_finite() || value >= u64::MAX as f64 {
        return u64::MAX;
    }
    if value <= 1.0 {
        return 1;
    }

    let magnitude = 10_f64.powf(value.log10().floor());
    let normalized = value / magnitude;
    let multiplier = if normalized <= 1.0 {
        1.0
    } else if normalized <= 2.0 {
        2.0
    } else if normalized <= 5.0 {
        5.0
    } else {
        10.0
    };
    (multiplier * magnitude).ceil() as u64
}

fn percentile(sorted: &[u32], percentage: usize) -> u32 {
    let rank = (percentage * sorted.len()).div_ceil(100).max(1);
    sorted[rank - 1]
}

#[cfg(test)]
mod tests {
    use meeple_bots_connect_four::ConnectFour;
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
        assert_eq!(report.recommended_rollout_depth, report.estimated_depth);
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
        assert_eq!(
            first.recommended_iterations,
            repeated.recommended_iterations
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
    fn recommendation_uses_nice_structural_budgets() {
        assert_eq!(round_up_nice(9.0 * 4.0 * 9.0_f64.powi(2)), 5_000);
        assert_eq!(round_up_nice(7.0 * 4.0 * 42.0_f64.powi(2)), 50_000);
    }
}
