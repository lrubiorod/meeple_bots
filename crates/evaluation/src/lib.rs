//! Game-independent structural sampling and search-family cost calibration.

pub mod so_ismcts;

use std::{
    collections::BTreeMap,
    error::Error,
    fmt,
    num::NonZeroU32,
    time::{Duration, Instant},
};

use meeple_bots_core::{
    Agent, AgentError, DecisionContext, DeterministicGame, Game, IllegalAction,
    PerfectInformationGame, PositionStatus, RandomSource, TwoPlayerZeroSumGame,
};
use meeple_bots_mcts_agent::{MctsAgent, MctsConfig, SearchBudget, UniformRandom};
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
    /// Configured soft decision limit, not actual simulated action count.
    pub rollout_depth: u32,
    /// Nominal same-player-block estimate; excludes turn-completion overshoot.
    pub approximate_player_turns: f64,
    pub milliseconds_per_iteration: f64,
    /// Deprecated compatibility grid; human analysis uses shared operating points.
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

#[derive(Clone, Debug, PartialEq)]
pub struct SampledDecisionTiming {
    pub phase: Option<&'static str>,
    pub sampled_ply: u32,
    pub milliseconds: f64,
    pub iterations: u64,
    pub nodes: u64,
    pub legal_actions: usize,
    pub terminal_simulations: Option<u64>,
    pub cutoff_simulations: Option<u64>,
    pub widening_expansions: Option<(u64, u64, u64, u64)>,
    pub root_expansion: Option<(usize, usize, usize)>,
    pub root_visits: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MctsAgentBenchmark {
    pub maximum_decision_horizon: Option<u32>,
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
    pub structural: StructuralReport,
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
    /// Deprecated compatibility presets; never rendered as recommendations by analyze.
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
    MissingSearchStats,
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
            Self::Agent(error) => write!(formatter, "search calibration failed: {error}"),
            Self::MissingSearchStats => {
                formatter.write_str("configured MCTS agent did not report search statistics")
            }
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
    let structural = structural_report(&sampled, config);
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
        structural,
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
    median_depth: u32,
    seed: u64,
) -> Result<MctsAgentBenchmark, EvaluationError>
where
    G: PerfectInformationGame + TwoPlayerZeroSumGame,
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
        let milliseconds = (started.elapsed().as_secs_f64() * 1_000.0).max(f64::EPSILON);
        let stats = agent.last_decision_stats();
        position_timings.push(SampledDecisionTiming {
            phase: game.diagnostic_phase(state),
            sampled_ply: *sampled_ply,
            legal_actions: game.legal_actions(state).count(),
            terminal_simulations: stats.terminal_simulations,
            cutoff_simulations: stats.cutoff_simulations,
            root_expansion: stats.root_expansion,
            widening_expansions: stats.widening_expansions,
            root_visits: stats.root_actions.iter().map(|a| a.visits).collect(),
            milliseconds,
            iterations: stats
                .search_iterations
                .ok_or(EvaluationError::MissingSearchStats)?,
            nodes: stats
                .search_nodes
                .ok_or(EvaluationError::MissingSearchStats)?,
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
        maximum_decision_horizon: game.maximum_decision_horizon(),
        sampled_positions: position_timings.len() as u32,
        decision_time_mean_ms,
        decision_time_p50_ms: percentile_f64(&sorted_timings, 50),
        decision_time_p95_ms: percentile_f64(&sorted_timings, 95),
        decision_time_max_ms: *sorted_timings
            .last()
            .expect("non-empty timings have a maximum"),
        milliseconds_per_iteration: position_timings
            .iter()
            .map(|timing| timing.milliseconds)
            .sum::<f64>()
            / position_timings
                .iter()
                .map(|timing| timing.iterations as f64)
                .sum::<f64>(),
        position_timings,
    })
}

/// Player-decision branching grouped by a game-provided diagnostic label.
#[derive(Clone, Debug, PartialEq)]
pub struct PhaseStructure {
    pub label: &'static str,
    pub samples: u64,
    pub legal_actions_mean: f64,
    pub legal_actions_p50: u32,
    pub legal_actions_p95: u32,
    pub legal_actions_min: u32,
    pub legal_actions_max: u32,
    pub effective_branching_factor: f64,
}

fn phase_structure(label: &'static str, counts: &BTreeMap<u32, u64>) -> PhaseStructure {
    let samples: u64 = counts.values().sum();
    let percentile = |percent: u64| {
        let rank = (samples * percent).div_ceil(100).max(1);
        let mut cumulative = 0;
        for (&actions, &count) in counts {
            cumulative += count;
            if cumulative >= rank {
                return actions;
            }
        }
        0
    };
    PhaseStructure {
        label,
        samples,
        legal_actions_mean: counts
            .iter()
            .map(|(&a, &n)| f64::from(a) * n as f64)
            .sum::<f64>()
            / samples as f64,
        legal_actions_p50: percentile(50),
        legal_actions_p95: percentile(95),
        legal_actions_min: *counts.first_key_value().unwrap().0,
        legal_actions_max: *counts.last_key_value().unwrap().0,
        effective_branching_factor: (counts
            .iter()
            .map(|(&a, &n)| f64::from(a).ln() * n as f64)
            .sum::<f64>()
            / samples as f64)
            .exp(),
    }
}

/// Random-policy physical decision structure; never an information-set tree estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralReport {
    pub phases: Vec<PhaseStructure>,
    pub physical_turns_p50: u32,
    pub physical_turns_p95: u32,
    pub samples: u32,
    pub max_depth: u32,
    pub terminal_rate: f64,
    pub initial_legal_actions: u32,
    pub initial_legal_actions_mean: f64,
    pub initial_legal_actions_p50: u32,
    pub initial_legal_actions_p95: u32,
    pub initial_legal_actions_min: u32,
    pub initial_legal_actions_max: u32,
    pub chance_events_mean: f64,
    pub chance_events_p50: u32,
    pub chance_events_p95: u32,
    pub decisions_mean: f64,
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
}

pub fn analyze_structure<G: Game>(
    game: &G,
    config: EvaluationConfig,
) -> Result<StructuralReport, EvaluationError> {
    let s = sample_game_tree(game, config)?;
    Ok(structural_report(&s, config))
}

fn structural_report(s: &SampledMetrics, config: EvaluationConfig) -> StructuralReport {
    let initial = s.initial_branches.first().copied().unwrap_or(0);
    let mean = |values: &[u32]| {
        values.iter().map(|&x| f64::from(x)).sum::<f64>() / values.len().max(1) as f64
    };
    StructuralReport {
        phases: s
            .phase_counts
            .iter()
            .map(|(&label, counts)| phase_structure(label, counts))
            .collect(),
        physical_turns_p50: percentile(&s.physical_turns, 50),
        physical_turns_p95: percentile(&s.physical_turns, 95),
        samples: config.samples.get(),
        max_depth: config.max_depth.get(),
        terminal_rate: f64::from(s.completed_samples) / f64::from(config.samples.get()),
        initial_legal_actions: percentile(&s.initial_branches, 50),
        initial_legal_actions_mean: mean(&s.initial_branches),
        initial_legal_actions_p50: percentile(&s.initial_branches, 50),
        initial_legal_actions_p95: percentile(&s.initial_branches, 95),
        initial_legal_actions_min: initial,
        initial_legal_actions_max: s.initial_branches.last().copied().unwrap_or(0),
        chance_events_mean: mean(&s.chance_events),
        chance_events_p50: percentile(&s.chance_events, 50),
        chance_events_p95: percentile(&s.chance_events, 95),
        decisions_mean: s.mean_depth,
        effective_branching_factor: s.effective_branching_factor,
        player_turn_choice_product_log10: s.player_turn_choice_product_log10,
        depth_p50: s.p50_depth,
        estimated_depth: s.p95_depth,
        player_turn_depth_p50: s.p50_player_turn_depth,
        player_turn_depth_p95: s.p95_player_turn_depth,
        player_changes_p50: s.p50_player_changes,
        player_changes_p95: s.p95_player_changes,
        actions_per_player_turn_mean: s.actions_per_player_turn_mean,
        actions_per_player_turn_p95: s.p95_actions_per_player_turn,
        actions_per_player_turn_max: s.max_actions_per_player_turn,
        depth_is_lower_bound: s.completed_samples < config.samples.get(),
        estimated_tree_log10: s
            .initial_branches
            .iter()
            .map(|&x| f64::from(x.max(1)).log10())
            .sum::<f64>()
            / s.initial_branches.len().max(1) as f64
            + f64::from(s.p95_depth.saturating_sub(1)) * s.effective_branching_factor.log10(),
    }
}

struct SampledMetrics {
    phase_counts: BTreeMap<&'static str, BTreeMap<u32, u64>>,
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
    initial_branches: Vec<u32>,
    chance_events: Vec<u32>,
    mean_depth: f64,
    physical_turns: Vec<u32>,
}

fn sample_game_tree<G>(
    game: &G,
    config: EvaluationConfig,
) -> Result<SampledMetrics, EvaluationError>
where
    G: Game,
{
    let mut rng = SplitMix64::new(config.seed);
    let mut environment = SplitMix64::new(config.seed ^ 0x8EBC_6AF0_9C88_C6E3);
    let mut physical_turns = Vec::new();
    let mut initial_branches = Vec::new();
    let mut chance_events = Vec::new();
    let mut branch_counts = Vec::new();
    let mut phase_counts: BTreeMap<_, BTreeMap<u32, u64>> = BTreeMap::new();
    let mut depths = Vec::with_capacity(config.samples.get() as usize);
    let mut player_turn_depths = Vec::with_capacity(config.samples.get() as usize);
    let mut player_changes = Vec::with_capacity(config.samples.get() as usize);
    let mut actions_per_player_turn = Vec::new();
    let mut player_turn_choice_logs = Vec::new();
    let mut completed_samples = 0;

    for _ in 0..config.samples.get() {
        let mut state = game.initial_state();
        let mut physical_turn_count = 0;
        let mut events = 0;
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
                    if depth == 0 || game.is_turn_boundary(&state) {
                        physical_turn_count += 1;
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
                    if depth == 0 {
                        initial_branches.push(actions.len() as u32);
                    }
                    branch_counts.push(actions.len() as u32);
                    if let Some(label) = game.diagnostic_phase(&state) {
                        *phase_counts
                            .entry(label)
                            .or_default()
                            .entry(actions.len() as u32)
                            .or_default() += 1;
                    }
                    current_turn_actions += 1;
                    current_turn_choice_log += (actions.len() as f64).ln();
                    game.apply_action(&mut state, &actions[action_index])
                        .map_err(EvaluationError::IllegalAction)?;
                    depth += 1;
                }
                PositionStatus::Chance => {
                    events += resolve_calibration_chance(game, &mut state, &mut environment)?;
                }
                _ => return Err(EvaluationError::UnexpectedChance),
            }
        }
        chance_events.push(events);
        physical_turns.push(physical_turn_count);
    }

    physical_turns.sort_unstable();
    initial_branches.sort_unstable();
    chance_events.sort_unstable();
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
        phase_counts,
        completed_samples,
        initial_branches,
        physical_turns,
        chance_events,
        mean_depth: depths.iter().map(|&n| f64::from(n)).sum::<f64>() / depths.len() as f64,
        effective_branching_factor: (log_branch_sum / branch_samples.max(1.0)).exp(),
        player_turn_choice_product_log10: player_turn_choice_logs.iter().sum::<f64>()
            / player_turn_count.max(1.0)
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
            / player_turn_count.max(1.0),
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
    G: Game,
    G::State: Clone,
    G::Action: Clone,
{
    let mut initial = game.initial_state();
    resolve_calibration_chance(
        game,
        &mut initial,
        &mut SplitMix64::new(seed ^ 0x8EBC_6AF0_9C88_C6E3),
    )?;
    let mut states = vec![(0, initial)];
    let mut phase_examples: BTreeMap<&'static str, (u32, G::State)> = BTreeMap::new();
    let mut target_depths = [median_depth / 3, median_depth.saturating_mul(2) / 3];
    target_depths.sort_unstable();

    for target_depth in target_depths {
        if target_depth == 0 {
            continue;
        }
        let mut state = game.initial_state();
        let mut rng = SplitMix64::new(seed ^ u64::from(target_depth));
        let mut chance_rng =
            SplitMix64::new(seed ^ u64::from(target_depth) ^ 0x8EBC_6AF0_9C88_C6E3);
        resolve_calibration_chance(game, &mut state, &mut chance_rng)?;
        let mut reached = true;
        for ply in 0..target_depth {
            match game.status(&state) {
                PositionStatus::PlayerTurn(_) => {
                    if let Some(label) = game.diagnostic_phase(&state)
                        && phase_examples.len() < 4
                        && !phase_examples.contains_key(label)
                    {
                        phase_examples.insert(label, (ply, state.clone()));
                    }
                    let actions: Vec<_> = game.legal_actions(&state).collect();
                    let index = rng
                        .index(actions.len())
                        .ok_or(EvaluationError::NoLegalActions)?;
                    game.apply_action(&mut state, &actions[index])
                        .map_err(EvaluationError::IllegalAction)?;
                    resolve_calibration_chance(game, &mut state, &mut chance_rng)?;
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

    // Keep the original early/mid/late probes first and unchanged. Add only missing
    // decision kinds seen along those paths; never benchmark chance/terminal states.
    for (label, example) in phase_examples {
        if !states
            .iter()
            .any(|(_, state)| game.diagnostic_phase(state) == Some(label))
        {
            states.push(example);
        }
    }
    Ok(states)
}

/// Environment events do not consume sampled player plies or the decision RNG.
fn resolve_calibration_chance<G: Game>(
    game: &G,
    state: &mut G::State,
    rng: &mut SplitMix64,
) -> Result<u32, EvaluationError> {
    let mut events = 0;
    while game.status(state) == PositionStatus::Chance {
        let outcome = game
            .sample_chance(state, rng)
            .map_err(EvaluationError::IllegalAction)?;
        game.apply_chance_outcome(state, &outcome)
            .map_err(EvaluationError::IllegalAction)?;
        events += 1;
    }
    Ok(events)
}

/// Sampled tail margin, not a proven bound. Saturate at the config's u32 limit.
fn sampled_full_horizon(p95_depth: u32) -> u32 {
    (u64::from(p95_depth) * 3)
        .div_ceil(2)
        .clamp(1, u64::from(u32::MAX)) as u32
}

fn candidate_rollout_depths(p95_depth: u32) -> Vec<u32> {
    let full_depth = p95_depth.max(1);
    let mut depths = Vec::new();
    for denominator in [6, 3] {
        depths.push(nearest_power_of_two(full_depth.div_ceil(denominator)));
    }
    let two_thirds = (u64::from(full_depth) * 2).div_ceil(3) as u32;
    depths.push(nearest_power_of_two(two_thirds));
    for depth in &mut depths {
        *depth = (*depth).clamp(1, full_depth);
    }
    depths.push(sampled_full_horizon(p95_depth));
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
        progressive_widening: None,
        selection_policy: Default::default(),
        budget: SearchBudget::Iterations(iterations),
        exploration: std::f64::consts::SQRT_2,
        rollout_depth,
        rollout_policy: UniformRandom,
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
    if sorted.is_empty() {
        return 0;
    }
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
            selection_policy: Default::default(),
            budget: SearchBudget::Iterations(iterations),
            rollout_depth: 4,
            ..MctsConfig::default()
        });

        let benchmark = benchmark_mcts_agent(&TicTacToe, &mut agent, 6, 42).unwrap();

        assert_eq!(benchmark.maximum_decision_horizon, Some(9));
        assert_eq!(benchmark.position_timings[0].legal_actions, 9);
        for timing in &benchmark.position_timings {
            assert_eq!(
                timing.terminal_simulations.unwrap() + timing.cutoff_simulations.unwrap(),
                4
            );
        }
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
        for (p95, expected) in [(100, 150), (101, 152), (124, 186)] {
            assert_eq!(sampled_full_horizon(p95), expected);
            assert_eq!(candidate_rollout_depths(p95).last(), Some(&expected));
        }
        assert_eq!(candidate_rollout_depths(96), vec![16, 32, 64, 144]);
        assert_eq!(candidate_rollout_depths(1), vec![1, 2]);
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

#[cfg(test)]
mod structural_tests;
