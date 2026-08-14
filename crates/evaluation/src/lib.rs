//! Empirical game-tree analysis and hardware-aware MCTS recommendations.

use std::{error::Error, fmt, num::NonZeroU32, time::Instant};

use meeple_bots_core::{
    Agent, AgentError, DecisionContext, DeterministicGame, IllegalAction, PerfectInformationGame,
    PositionStatus, RandomSource, TwoPlayerZeroSumGame,
};
use meeple_bots_mcts_agent::{MctsAgent, MctsConfig, MctsSearchStats};
use meeple_bots_random_agent::RandomAgent;
use meeple_bots_simulation::{MatchConfig, MatchError, SplitMix64, play_match};

const CALIBRATION_REPEATS: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComplexityConfig {
    pub samples: NonZeroU32,
    pub max_depth: NonZeroU32,
    pub seed: u64,
    pub calibration_iterations: NonZeroU32,
    pub max_iterations: NonZeroU32,
}

impl Default for ComplexityConfig {
    fn default() -> Self {
        Self {
            samples: NonZeroU32::new(128).expect("constant is non-zero"),
            max_depth: NonZeroU32::new(256).expect("constant is non-zero"),
            seed: 0,
            calibration_iterations: NonZeroU32::new(8).expect("constant is non-zero"),
            max_iterations: NonZeroU32::new(1_000_000).expect("constant is non-zero"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MctsLevel {
    Fast,
    Balanced,
    Thorough,
}

impl MctsLevel {
    pub const ALL: [Self; 3] = [Self::Fast, Self::Balanced, Self::Thorough];

    pub const fn target_time_ms(self) -> u32 {
        match self {
            Self::Fast => 100,
            Self::Balanced => 500,
            Self::Thorough => 2_000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MctsRecommendation {
    pub level: MctsLevel,
    pub iterations: u32,
    pub rollout_depth: u32,
    pub target_time_ms: u32,
    pub estimated_time_ms: f64,
    pub milliseconds_per_iteration: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GameComplexityReport {
    pub samples: u32,
    pub max_depth: u32,
    pub completed_samples: u32,
    pub terminal_rate: f64,
    pub initial_legal_actions: u32,
    pub mean_branching_factor: f64,
    pub effective_branching_factor: f64,
    pub maximum_branching_factor: u32,
    pub p95_branching_factor: u32,
    pub mean_plies: f64,
    pub median_plies: u32,
    pub p75_plies: u32,
    pub p95_plies: u32,
    pub estimated_tree_log10: f64,
    pub estimate_is_lower_bound: bool,
    pub max_iterations: u32,
    pub recommendations: [MctsRecommendation; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrengthConfig {
    pub candidate: MctsConfig,
    pub matches_per_opponent: NonZeroU32,
    pub reference_iterations_multiplier: NonZeroU32,
    pub max_reference_iterations: NonZeroU32,
    pub max_plies: NonZeroU32,
    pub seed: u64,
}

impl Default for StrengthConfig {
    fn default() -> Self {
        Self {
            candidate: MctsConfig::default(),
            matches_per_opponent: NonZeroU32::new(20).expect("constant is non-zero"),
            reference_iterations_multiplier: NonZeroU32::new(4).expect("constant is non-zero"),
            max_reference_iterations: NonZeroU32::new(1_000_000).expect("constant is non-zero"),
            max_plies: NonZeroU32::new(10_000).expect("constant is non-zero"),
            seed: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StrengthEstimate {
    Inconclusive,
    UnprovenAgainstRandom,
    BeatsRandomBelowReference,
    BeatsRandomNoDetectedReferenceGap,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CutoffHeuristicEvidence {
    Low,
    Moderate,
    High,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SearchSufficiency {
    Insufficient,
    Limited,
    Adequate,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BenchmarkConfidence {
    Low,
    Moderate,
    High,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StrengthOpponent {
    Random,
    Reference,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StrengthProgressStage {
    Started,
    Completed,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrengthProgress {
    pub stage: StrengthProgressStage,
    pub match_number: u32,
    pub total_matches: u32,
    pub opponent: StrengthOpponent,
    pub candidate_player: u8,
    pub plies: Option<u32>,
    pub utility: Option<f32>,
    pub elapsed_seconds: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpponentResult {
    pub matches: u32,
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
    pub score: f64,
    pub mean_utility: f64,
    pub utility_confidence_low: f64,
    pub utility_confidence_high: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AggregatedSearchStats {
    pub decisions: u32,
    pub total_iterations: u64,
    pub mean_expanded_nodes: f64,
    pub maximum_expanded_nodes: u32,
    pub mean_root_actions: f64,
    pub mean_iterations_per_root_action: f64,
    pub mean_tree_revisit_rate: f64,
    pub mean_tree_depth: f64,
    pub maximum_tree_depth: u32,
    pub mean_simulation_depth: f64,
    pub maximum_simulation_depth: u32,
    pub terminal_rollout_rate: f64,
    pub truncated_rollout_rate: f64,
    pub mean_selected_action_visit_share: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MctsStrengthReport {
    pub candidate: MctsConfig,
    pub reference: MctsConfig,
    pub matches_per_opponent: u32,
    pub initial_expanded_nodes: u32,
    pub tree_size_log10_gap: f64,
    pub tree_size_estimate_is_lower_bound: bool,
    pub search: AggregatedSearchStats,
    pub versus_random: OpponentResult,
    pub versus_reference: OpponentResult,
    pub search_sufficiency: SearchSufficiency,
    pub benchmark_confidence: BenchmarkConfidence,
    pub strength_estimate: StrengthEstimate,
    pub cutoff_heuristic_evidence: CutoffHeuristicEvidence,
    pub reasons: Vec<String>,
}

impl GameComplexityReport {
    pub fn recommend(
        &self,
        level: MctsLevel,
        target_time_ms: Option<NonZeroU32>,
    ) -> MctsRecommendation {
        let base = self.recommendations[level_index(level)];
        let target_time_ms = target_time_ms
            .map(NonZeroU32::get)
            .unwrap_or_else(|| level.target_time_ms());
        recommendation(
            level,
            base.rollout_depth,
            target_time_ms,
            base.milliseconds_per_iteration,
            self.initial_legal_actions,
            self.max_iterations,
        )
    }
}

#[derive(Debug)]
pub enum EvaluationError {
    UnexpectedChance,
    NoLegalActions,
    OddMatchCount(u32),
    ReferenceNotStronger,
    Progress(String),
    IllegalAction(IllegalAction),
    Agent(AgentError),
    Match(MatchError),
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedChance => formatter
                .write_str("game complexity evaluation does not support chance transitions"),
            Self::NoLegalActions => {
                formatter.write_str("a non-terminal position has no legal actions")
            }
            Self::OddMatchCount(matches) => write!(
                formatter,
                "matches per opponent must be even to swap player positions, got {matches}"
            ),
            Self::ReferenceNotStronger => formatter.write_str(
                "the reference iteration limit must allow more iterations than the candidate",
            ),
            Self::Progress(message) => {
                write!(formatter, "strength progress callback failed: {message}")
            }
            Self::IllegalAction(error) => write!(formatter, "sampled action was rejected: {error}"),
            Self::Agent(error) => write!(formatter, "MCTS calibration failed: {error}"),
            Self::Match(error) => write!(formatter, "MCTS strength match failed: {error}"),
        }
    }
}

impl Error for EvaluationError {}

impl From<MatchError> for EvaluationError {
    fn from(error: MatchError) -> Self {
        Self::Match(error)
    }
}

pub fn evaluate_game<G>(
    game: &G,
    config: ComplexityConfig,
) -> Result<GameComplexityReport, EvaluationError>
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
    let fast_depth = sampled
        .median_plies
        .clamp(1, config.max_depth.get().min(64));
    let balanced_depth = sampled.p75_plies.clamp(1, config.max_depth.get().min(128));
    let thorough_depth = sampled.p95_plies.clamp(1, config.max_depth.get().min(256));
    let depths = [fast_depth, balanced_depth, thorough_depth];
    let mut recommendations = [MctsRecommendation {
        level: MctsLevel::Fast,
        iterations: 1,
        rollout_depth: 1,
        target_time_ms: 1,
        estimated_time_ms: 0.0,
        milliseconds_per_iteration: 0.0,
    }; 3];

    for (index, level) in MctsLevel::ALL.into_iter().enumerate() {
        let milliseconds_per_iteration = calibrate_iteration_cost(
            game,
            depths[index],
            config.calibration_iterations.get(),
            config.seed.wrapping_add(index as u64),
        )?;
        recommendations[index] = recommendation(
            level,
            depths[index],
            level.target_time_ms(),
            milliseconds_per_iteration,
            initial_legal_actions,
            config.max_iterations.get(),
        );
    }

    Ok(GameComplexityReport {
        samples: config.samples.get(),
        max_depth: config.max_depth.get(),
        completed_samples: sampled.completed_samples,
        terminal_rate: f64::from(sampled.completed_samples) / f64::from(config.samples.get()),
        initial_legal_actions,
        mean_branching_factor: sampled.mean_branching_factor,
        effective_branching_factor: sampled.effective_branching_factor,
        maximum_branching_factor: sampled.maximum_branching_factor,
        p95_branching_factor: sampled.p95_branching_factor,
        mean_plies: sampled.mean_plies,
        median_plies: sampled.median_plies,
        p75_plies: sampled.p75_plies,
        p95_plies: sampled.p95_plies,
        estimated_tree_log10: f64::from(sampled.p95_plies)
            * sampled.effective_branching_factor.log10(),
        estimate_is_lower_bound: sampled.completed_samples < config.samples.get(),
        max_iterations: config.max_iterations.get(),
        recommendations,
    })
}

pub fn evaluate_mcts_strength<G>(
    game: &G,
    estimated_tree_log10: f64,
    tree_size_estimate_is_lower_bound: bool,
    config: StrengthConfig,
) -> Result<MctsStrengthReport, EvaluationError>
where
    G: DeterministicGame + PerfectInformationGame + TwoPlayerZeroSumGame,
    G::State: Clone,
    G::Action: Clone,
{
    evaluate_mcts_strength_with_progress(
        game,
        estimated_tree_log10,
        tree_size_estimate_is_lower_bound,
        config,
        |_| Ok(()),
    )
}

pub fn evaluate_mcts_strength_with_progress<G, F>(
    game: &G,
    estimated_tree_log10: f64,
    tree_size_estimate_is_lower_bound: bool,
    config: StrengthConfig,
    mut progress: F,
) -> Result<MctsStrengthReport, EvaluationError>
where
    G: DeterministicGame + PerfectInformationGame + TwoPlayerZeroSumGame,
    G::State: Clone,
    G::Action: Clone,
    F: FnMut(StrengthProgress) -> Result<(), EvaluationError>,
{
    if !config.matches_per_opponent.get().is_multiple_of(2) {
        return Err(EvaluationError::OddMatchCount(
            config.matches_per_opponent.get(),
        ));
    }

    let reference_iterations = u64::from(config.candidate.iterations.get())
        .saturating_mul(u64::from(config.reference_iterations_multiplier.get()))
        .min(u64::from(config.max_reference_iterations.get()))
        as u32;
    if reference_iterations <= config.candidate.iterations.get() {
        return Err(EvaluationError::ReferenceNotStronger);
    }
    let reference = MctsConfig {
        iterations: NonZeroU32::new(reference_iterations).expect("reference is non-zero"),
        ..config.candidate
    };

    let initial_state = game.initial_state();
    let initial_player = match game.status(&initial_state) {
        PositionStatus::PlayerTurn(player) => player,
        PositionStatus::Chance => return Err(EvaluationError::UnexpectedChance),
        PositionStatus::Terminal => return Err(EvaluationError::NoLegalActions),
        _ => return Err(EvaluationError::UnexpectedChance),
    };
    let mut initial_agent = MctsAgent::new(config.candidate);
    let mut initial_rng = SplitMix64::new(config.seed ^ 0xD1B5_4A32_D192_ED03);
    let initial_search = initial_agent
        .select_action_with_stats(
            DecisionContext::new(game, &initial_state, initial_player),
            &mut initial_rng,
        )
        .map_err(EvaluationError::Agent)?
        .stats;

    let mut observed = ObservedMcts::new(config.candidate);
    let mut seed_stream = SplitMix64::new(config.seed ^ 0x8CB9_2BA7_2F3D_8DD7);
    let versus_random = play_paired_against_random(
        game,
        &mut observed,
        config.matches_per_opponent.get(),
        config.max_plies,
        &mut seed_stream,
        &mut progress,
    )?;
    let versus_reference = play_paired_against_reference(
        game,
        &mut observed,
        reference,
        config.matches_per_opponent.get(),
        config.max_plies,
        &mut seed_stream,
        &mut progress,
    )?;
    let search = aggregate_search_stats(&observed.stats).ok_or(EvaluationError::NoLegalActions)?;

    let search_sufficiency = if search.mean_iterations_per_root_action < 2.0
        || search.mean_tree_revisit_rate < 0.1
    {
        SearchSufficiency::Insufficient
    } else if search.mean_iterations_per_root_action < 10.0 || search.mean_tree_revisit_rate < 0.5 {
        SearchSufficiency::Limited
    } else {
        SearchSufficiency::Adequate
    };
    let benchmark_confidence = if config.matches_per_opponent.get() < 20 {
        BenchmarkConfidence::Low
    } else if config.matches_per_opponent.get() < 100 {
        BenchmarkConfidence::Moderate
    } else {
        BenchmarkConfidence::High
    };
    let beats_random = versus_random.utility_confidence_low > 0.0;
    let below_reference = versus_reference.utility_confidence_high < 0.0;
    let strength_estimate = if benchmark_confidence == BenchmarkConfidence::Low {
        StrengthEstimate::Inconclusive
    } else if !beats_random {
        StrengthEstimate::UnprovenAgainstRandom
    } else if below_reference {
        StrengthEstimate::BeatsRandomBelowReference
    } else {
        StrengthEstimate::BeatsRandomNoDetectedReferenceGap
    };
    let cutoff_heuristic_evidence = if search.truncated_rollout_rate >= 0.5 {
        CutoffHeuristicEvidence::High
    } else if search.truncated_rollout_rate >= 0.1 {
        CutoffHeuristicEvidence::Moderate
    } else {
        CutoffHeuristicEvidence::Low
    };

    let explored_log10 = f64::from(initial_search.expanded_nodes.saturating_add(1)).log10();
    let tree_size_log10_gap = (estimated_tree_log10 - explored_log10).max(0.0);
    let mut reasons = Vec::new();
    if search_sufficiency != SearchSufficiency::Adequate {
        reasons.push(format!(
            "the candidate averages {:.2} iterations per legal root action and revisits the existing tree on only {:.1}% of iterations",
            search.mean_iterations_per_root_action,
            search.mean_tree_revisit_rate * 100.0
        ));
    }
    if benchmark_confidence == BenchmarkConfidence::Low {
        reasons.push(format!(
            "only {} matches were played per opponent, so strength results are preliminary",
            config.matches_per_opponent
        ));
    }
    reasons.push(format!(
        "{:.1}% of candidate rollouts reached the depth limit and received neutral utility",
        search.truncated_rollout_rate * 100.0
    ));
    if benchmark_confidence != BenchmarkConfidence::Low && below_reference {
        reasons.push("the higher-iteration reference performed significantly better".to_owned());
    } else if benchmark_confidence != BenchmarkConfidence::Low {
        reasons.push(
            "the benchmark did not detect a significant gap against the stronger reference"
                .to_owned(),
        );
    }

    Ok(MctsStrengthReport {
        candidate: config.candidate,
        reference,
        matches_per_opponent: config.matches_per_opponent.get(),
        initial_expanded_nodes: initial_search.expanded_nodes,
        tree_size_log10_gap,
        tree_size_estimate_is_lower_bound,
        search,
        versus_random,
        versus_reference,
        search_sufficiency,
        benchmark_confidence,
        strength_estimate,
        cutoff_heuristic_evidence,
        reasons,
    })
}

struct ObservedMcts {
    agent: MctsAgent,
    stats: Vec<MctsSearchStats>,
}

impl ObservedMcts {
    fn new(config: MctsConfig) -> Self {
        Self {
            agent: MctsAgent::new(config),
            stats: Vec::new(),
        }
    }
}

impl<G> Agent<G> for ObservedMcts
where
    G: DeterministicGame + PerfectInformationGame + TwoPlayerZeroSumGame,
    G::State: Clone,
    G::Action: Clone,
{
    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, G>,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        let decision = self.agent.select_action_with_stats(decision, rng)?;
        self.stats.push(decision.stats);
        Ok(decision.action)
    }
}

#[derive(Default)]
struct Outcomes {
    utilities: Vec<f64>,
    wins: u32,
    draws: u32,
    losses: u32,
}

impl Outcomes {
    fn record(&mut self, utility: f32) {
        self.utilities.push(f64::from(utility));
        if utility > 0.0 {
            self.wins += 1;
        } else if utility < 0.0 {
            self.losses += 1;
        } else {
            self.draws += 1;
        }
    }

    fn report(self) -> OpponentResult {
        let matches = self.utilities.len() as u32;
        let mean_utility = self.utilities.iter().sum::<f64>() / f64::from(matches);
        let score = (f64::from(self.wins) + 0.5 * f64::from(self.draws)) / f64::from(matches);
        let (score_confidence_low, score_confidence_high) = wilson_interval(score, matches);
        OpponentResult {
            matches,
            wins: self.wins,
            draws: self.draws,
            losses: self.losses,
            score,
            mean_utility,
            utility_confidence_low: 2.0 * score_confidence_low - 1.0,
            utility_confidence_high: 2.0 * score_confidence_high - 1.0,
        }
    }
}

fn wilson_interval(score: f64, samples: u32) -> (f64, f64) {
    const Z: f64 = 1.96;
    let samples = f64::from(samples);
    let z_squared = Z * Z;
    let denominator = 1.0 + z_squared / samples;
    let center = (score + z_squared / (2.0 * samples)) / denominator;
    let margin = Z
        * (score * (1.0 - score) / samples + z_squared / (4.0 * samples * samples)).sqrt()
        / denominator;
    ((center - margin).max(0.0), (center + margin).min(1.0))
}

fn play_paired_against_random<G, F>(
    game: &G,
    candidate: &mut ObservedMcts,
    matches: u32,
    max_plies: NonZeroU32,
    seed_stream: &mut SplitMix64,
    progress: &mut F,
) -> Result<OpponentResult, EvaluationError>
where
    G: DeterministicGame + PerfectInformationGame + TwoPlayerZeroSumGame,
    G::State: Clone,
    G::Action: Clone,
    F: FnMut(StrengthProgress) -> Result<(), EvaluationError>,
{
    let mut outcomes = Outcomes::default();
    for pair in 0..matches / 2 {
        let seed = seed_stream.next_u64();
        let first_number = pair * 2 + 1;
        progress(started_progress(
            first_number,
            matches * 2,
            StrengthOpponent::Random,
            0,
        ))?;
        let started = Instant::now();
        let first = play_match(
            game,
            candidate,
            &mut RandomAgent,
            MatchConfig::new(seed, max_plies),
        )?;
        progress(completed_progress(
            first_number,
            matches * 2,
            StrengthOpponent::Random,
            0,
            &first,
            started.elapsed().as_secs_f64(),
        ))?;
        outcomes.record(first.utilities[0]);

        let second_number = first_number + 1;
        progress(started_progress(
            second_number,
            matches * 2,
            StrengthOpponent::Random,
            1,
        ))?;
        let started = Instant::now();
        let second = play_match(
            game,
            &mut RandomAgent,
            candidate,
            MatchConfig::new(seed, max_plies),
        )?;
        progress(completed_progress(
            second_number,
            matches * 2,
            StrengthOpponent::Random,
            1,
            &second,
            started.elapsed().as_secs_f64(),
        ))?;
        outcomes.record(second.utilities[1]);
    }
    Ok(outcomes.report())
}

fn play_paired_against_reference<G, F>(
    game: &G,
    candidate: &mut ObservedMcts,
    reference_config: MctsConfig,
    matches: u32,
    max_plies: NonZeroU32,
    seed_stream: &mut SplitMix64,
    progress: &mut F,
) -> Result<OpponentResult, EvaluationError>
where
    G: DeterministicGame + PerfectInformationGame + TwoPlayerZeroSumGame,
    G::State: Clone,
    G::Action: Clone,
    F: FnMut(StrengthProgress) -> Result<(), EvaluationError>,
{
    let mut reference = MctsAgent::new(reference_config);
    let mut outcomes = Outcomes::default();
    for pair in 0..matches / 2 {
        let seed = seed_stream.next_u64();
        let first_number = matches + pair * 2 + 1;
        progress(started_progress(
            first_number,
            matches * 2,
            StrengthOpponent::Reference,
            0,
        ))?;
        let started = Instant::now();
        let first = play_match(
            game,
            candidate,
            &mut reference,
            MatchConfig::new(seed, max_plies),
        )?;
        progress(completed_progress(
            first_number,
            matches * 2,
            StrengthOpponent::Reference,
            0,
            &first,
            started.elapsed().as_secs_f64(),
        ))?;
        outcomes.record(first.utilities[0]);

        let second_number = first_number + 1;
        progress(started_progress(
            second_number,
            matches * 2,
            StrengthOpponent::Reference,
            1,
        ))?;
        let started = Instant::now();
        let second = play_match(
            game,
            &mut reference,
            candidate,
            MatchConfig::new(seed, max_plies),
        )?;
        progress(completed_progress(
            second_number,
            matches * 2,
            StrengthOpponent::Reference,
            1,
            &second,
            started.elapsed().as_secs_f64(),
        ))?;
        outcomes.record(second.utilities[1]);
    }
    Ok(outcomes.report())
}

fn started_progress(
    match_number: u32,
    total_matches: u32,
    opponent: StrengthOpponent,
    candidate_player: u8,
) -> StrengthProgress {
    StrengthProgress {
        stage: StrengthProgressStage::Started,
        match_number,
        total_matches,
        opponent,
        candidate_player,
        plies: None,
        utility: None,
        elapsed_seconds: None,
    }
}

fn completed_progress(
    match_number: u32,
    total_matches: u32,
    opponent: StrengthOpponent,
    candidate_player: u8,
    result: &meeple_bots_simulation::MatchResult,
    elapsed_seconds: f64,
) -> StrengthProgress {
    StrengthProgress {
        stage: StrengthProgressStage::Completed,
        match_number,
        total_matches,
        opponent,
        candidate_player,
        plies: Some(result.plies),
        utility: Some(result.utilities[usize::from(candidate_player)]),
        elapsed_seconds: Some(elapsed_seconds),
    }
}

fn aggregate_search_stats(stats: &[MctsSearchStats]) -> Option<AggregatedSearchStats> {
    if stats.is_empty() {
        return None;
    }
    let decisions = stats.len() as u32;
    let total_iterations: u64 = stats.iter().map(|stats| u64::from(stats.iterations)).sum();
    let terminal_rollouts: u64 = stats
        .iter()
        .map(|stats| u64::from(stats.terminal_rollouts))
        .sum();
    let truncated_rollouts: u64 = stats
        .iter()
        .map(|stats| u64::from(stats.truncated_rollouts))
        .sum();
    Some(AggregatedSearchStats {
        decisions,
        total_iterations,
        mean_expanded_nodes: stats
            .iter()
            .map(|stats| f64::from(stats.expanded_nodes))
            .sum::<f64>()
            / f64::from(decisions),
        maximum_expanded_nodes: stats
            .iter()
            .map(|stats| stats.expanded_nodes)
            .max()
            .expect("stats are non-empty"),
        mean_root_actions: stats
            .iter()
            .map(|stats| f64::from(stats.root_actions))
            .sum::<f64>()
            / f64::from(decisions),
        mean_iterations_per_root_action: stats
            .iter()
            .map(|stats| f64::from(stats.iterations) / f64::from(stats.root_actions))
            .sum::<f64>()
            / f64::from(decisions),
        mean_tree_revisit_rate: stats
            .iter()
            .map(|stats| {
                f64::from(stats.iterations - stats.expanded_nodes) / f64::from(stats.iterations)
            })
            .sum::<f64>()
            / f64::from(decisions),
        mean_tree_depth: stats
            .iter()
            .map(|stats| stats.mean_tree_depth * f64::from(stats.iterations))
            .sum::<f64>()
            / total_iterations as f64,
        maximum_tree_depth: stats
            .iter()
            .map(|stats| stats.maximum_tree_depth)
            .max()
            .expect("stats are non-empty"),
        mean_simulation_depth: stats
            .iter()
            .map(|stats| stats.mean_simulation_depth * f64::from(stats.iterations))
            .sum::<f64>()
            / total_iterations as f64,
        maximum_simulation_depth: stats
            .iter()
            .map(|stats| stats.maximum_simulation_depth)
            .max()
            .expect("stats are non-empty"),
        terminal_rollout_rate: terminal_rollouts as f64 / total_iterations as f64,
        truncated_rollout_rate: truncated_rollouts as f64 / total_iterations as f64,
        mean_selected_action_visit_share: stats
            .iter()
            .map(|stats| stats.selected_action_visit_share)
            .sum::<f64>()
            / f64::from(decisions),
    })
}

struct SampledMetrics {
    completed_samples: u32,
    mean_branching_factor: f64,
    effective_branching_factor: f64,
    maximum_branching_factor: u32,
    p95_branching_factor: u32,
    mean_plies: f64,
    median_plies: u32,
    p75_plies: u32,
    p95_plies: u32,
}

fn sample_game_tree<G>(
    game: &G,
    config: ComplexityConfig,
) -> Result<SampledMetrics, EvaluationError>
where
    G: DeterministicGame,
{
    let mut rng = SplitMix64::new(config.seed);
    let mut branch_counts = Vec::new();
    let mut lengths = Vec::with_capacity(config.samples.get() as usize);
    let mut completed_samples = 0;

    for _ in 0..config.samples.get() {
        let mut state = game.initial_state();
        let mut depth = 0;
        loop {
            match game.status(&state) {
                PositionStatus::Terminal => {
                    completed_samples += 1;
                    lengths.push(depth);
                    break;
                }
                PositionStatus::PlayerTurn(_) => {
                    if depth >= config.max_depth.get() {
                        lengths.push(depth);
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

    branch_counts.sort_unstable();
    lengths.sort_unstable();
    let branch_sum: u64 = branch_counts.iter().map(|value| u64::from(*value)).sum();
    let log_branch_sum: f64 = branch_counts
        .iter()
        .map(|value| f64::from(*value).ln())
        .sum();
    let branch_samples = branch_counts.len() as f64;
    let length_sum: u64 = lengths.iter().map(|value| u64::from(*value)).sum();

    Ok(SampledMetrics {
        completed_samples,
        mean_branching_factor: branch_sum as f64 / branch_samples,
        effective_branching_factor: (log_branch_sum / branch_samples).exp(),
        maximum_branching_factor: *branch_counts
            .last()
            .ok_or(EvaluationError::NoLegalActions)?,
        p95_branching_factor: percentile(&branch_counts, 95),
        mean_plies: length_sum as f64 / lengths.len() as f64,
        median_plies: percentile(&lengths, 50),
        p75_plies: percentile(&lengths, 75),
        p95_plies: percentile(&lengths, 95),
    })
}

fn calibrate_iteration_cost<G>(
    game: &G,
    rollout_depth: u32,
    iterations: u32,
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
    let iterations = NonZeroU32::new(iterations).expect("calibration iterations are non-zero");
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

fn recommendation(
    level: MctsLevel,
    rollout_depth: u32,
    target_time_ms: u32,
    milliseconds_per_iteration: f64,
    minimum_iterations: u32,
    maximum_iterations: u32,
) -> MctsRecommendation {
    let estimated_iterations = (f64::from(target_time_ms) / milliseconds_per_iteration).floor();
    let iterations = estimated_iterations
        .clamp(f64::from(minimum_iterations), f64::from(maximum_iterations))
        as u32;
    MctsRecommendation {
        level,
        iterations,
        rollout_depth,
        target_time_ms,
        estimated_time_ms: f64::from(iterations) * milliseconds_per_iteration,
        milliseconds_per_iteration,
    }
}

fn percentile(sorted: &[u32], percentage: usize) -> u32 {
    let rank = (percentage * sorted.len()).div_ceil(100).max(1);
    sorted[rank - 1]
}

const fn level_index(level: MctsLevel) -> usize {
    match level {
        MctsLevel::Fast => 0,
        MctsLevel::Balanced => 1,
        MctsLevel::Thorough => 2,
    }
}

#[cfg(test)]
mod tests {
    use meeple_bots_connect_four::ConnectFour;
    use meeple_bots_tic_tac_toe::TicTacToe;

    use super::*;

    fn small_config() -> ComplexityConfig {
        ComplexityConfig {
            samples: NonZeroU32::new(16).unwrap(),
            calibration_iterations: NonZeroU32::new(9).unwrap(),
            ..ComplexityConfig::default()
        }
    }

    fn small_strength_config() -> StrengthConfig {
        StrengthConfig {
            candidate: MctsConfig {
                iterations: NonZeroU32::new(16).unwrap(),
                rollout_depth: 1,
                ..MctsConfig::default()
            },
            matches_per_opponent: NonZeroU32::new(2).unwrap(),
            max_reference_iterations: NonZeroU32::new(1_000).unwrap(),
            max_plies: NonZeroU32::new(32).unwrap(),
            seed: 7,
            ..StrengthConfig::default()
        }
    }

    #[test]
    fn tic_tac_toe_metrics_respect_known_bounds() {
        let report = evaluate_game(&TicTacToe, small_config()).unwrap();

        assert_eq!(report.initial_legal_actions, 9);
        assert_eq!(report.completed_samples, report.samples);
        assert_eq!(report.terminal_rate, 1.0);
        assert!(report.p95_plies <= 9);
        assert!(report.maximum_branching_factor <= 9);
        assert!(!report.estimate_is_lower_bound);
    }

    #[test]
    fn structural_metrics_are_reproducible() {
        let first = evaluate_game(&TicTacToe, small_config()).unwrap();
        let repeated = evaluate_game(&TicTacToe, small_config()).unwrap();

        assert_eq!(first.mean_branching_factor, repeated.mean_branching_factor);
        assert_eq!(
            first.effective_branching_factor,
            repeated.effective_branching_factor
        );
        assert_eq!(first.median_plies, repeated.median_plies);
        assert_eq!(first.estimated_tree_log10, repeated.estimated_tree_log10);
    }

    #[test]
    fn custom_budget_respects_iteration_limits() {
        let report = evaluate_game(&TicTacToe, small_config()).unwrap();
        let recommendation = report.recommend(MctsLevel::Fast, Some(NonZeroU32::new(1).unwrap()));

        assert!(recommendation.iterations >= report.initial_legal_actions);
        assert!(recommendation.iterations <= report.max_iterations);
        assert_eq!(recommendation.target_time_ms, 1);
    }

    #[test]
    fn connect_four_samples_respect_known_bounds() {
        let report = evaluate_game(&ConnectFour, small_config()).unwrap();

        assert_eq!(report.initial_legal_actions, 7);
        assert!(report.maximum_branching_factor <= 7);
        assert!(report.p95_plies <= 42);
    }

    #[test]
    fn depth_limit_marks_the_tree_estimate_as_a_lower_bound() {
        let report = evaluate_game(
            &TicTacToe,
            ComplexityConfig {
                max_depth: NonZeroU32::new(1).unwrap(),
                ..small_config()
            },
        )
        .unwrap();

        assert_eq!(report.completed_samples, 0);
        assert_eq!(report.terminal_rate, 0.0);
        assert!(report.estimate_is_lower_bound);
        assert!(
            report
                .recommendations
                .iter()
                .all(|recommendation| recommendation.rollout_depth == 1)
        );
    }

    #[test]
    fn strength_report_combines_search_and_paired_results() {
        let complexity = evaluate_game(&TicTacToe, small_config()).unwrap();
        let mut progress = Vec::new();
        let report = evaluate_mcts_strength_with_progress(
            &TicTacToe,
            complexity.estimated_tree_log10,
            complexity.estimate_is_lower_bound,
            small_strength_config(),
            |event| {
                progress.push(event);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(report.reference.iterations.get(), 64);
        assert_eq!(report.versus_random.matches, 2);
        assert_eq!(report.versus_reference.matches, 2);
        assert_eq!(
            report.versus_random.wins + report.versus_random.draws + report.versus_random.losses,
            2
        );
        assert!(report.search.decisions > 0);
        assert_eq!(
            report.search.terminal_rollout_rate + report.search.truncated_rollout_rate,
            1.0
        );
        assert!(report.initial_expanded_nodes > 0);
        assert!(!report.reasons.is_empty());
        assert_ne!(report.search_sufficiency, SearchSufficiency::Adequate);
        assert_eq!(report.benchmark_confidence, BenchmarkConfidence::Low);
        assert_eq!(report.strength_estimate, StrengthEstimate::Inconclusive);
        assert!(report.search.mean_root_actions > 0.0);
        assert!(report.search.mean_iterations_per_root_action > 0.0);
        assert_eq!(progress.len(), 8);
        assert_eq!(progress[0].stage, StrengthProgressStage::Started);
        assert_eq!(progress[1].stage, StrengthProgressStage::Completed);
        assert_eq!(progress[0].match_number, 1);
        assert_eq!(progress[7].match_number, 4);
        assert!(progress[7].elapsed_seconds.is_some());
    }

    #[test]
    fn strength_evaluation_requires_paired_matches() {
        let error = evaluate_mcts_strength(
            &TicTacToe,
            4.0,
            false,
            StrengthConfig {
                matches_per_opponent: NonZeroU32::new(3).unwrap(),
                ..small_strength_config()
            },
        )
        .unwrap_err();

        assert!(matches!(error, EvaluationError::OddMatchCount(3)));
    }

    #[test]
    fn wilson_interval_does_not_claim_certainty_from_two_wins() {
        let mut outcomes = Outcomes::default();
        outcomes.record(1.0);
        outcomes.record(1.0);

        let report = outcomes.report();

        assert_eq!(report.mean_utility, 1.0);
        assert!(report.utility_confidence_low < 0.0);
        assert_eq!(report.utility_confidence_high, 1.0);
    }
}
