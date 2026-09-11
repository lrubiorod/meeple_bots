//! Monte Carlo Tree Search for deterministic, perfect-information games.

mod stochastic;
mod stochastic_reuse;
pub use stochastic::StochasticMctsAgent;
pub use stochastic_reuse::ReusableStochasticMctsAgent;

use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt,
    hash::Hash,
    num::NonZeroU32,
    time::{Duration, Instant},
};

use meeple_bots_core::{
    Agent, AgentDecisionStats, AgentError, DecisionContext, DeterministicGame, HeuristicGame,
    PerfectInformationGame, PlayerId, PositionStatus, RandomSource, RootActionStats,
    TreeReuseStats, TwoPlayerZeroSumGame,
};

/// Tree selection rule. UCB1-Tuned uses its canonical variance bound and ignores `exploration`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SelectionPolicy {
    #[default]
    Uct,
    Ucb1Tuned,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MctsConfig<P = RolloutPolicyConfig<NeutralEvaluator>> {
    pub budget: SearchBudget,
    pub exploration: f64,
    pub selection_policy: SelectionPolicy,
    pub rollout_depth: u32,
    pub rollout_policy: P,
}

impl<P> MctsConfig<P> {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.exploration.is_finite() || self.exploration < 0.0 {
            return Err("MCTS exploration must be finite and non-negative");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchBudget {
    Iterations(NonZeroU32),
    Time(Duration),
}

impl Default for SearchBudget {
    fn default() -> Self {
        Self::Iterations(NonZeroU32::new(1_000).expect("constant is non-zero"))
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MctsSearchStats {
    pub iterations: u64,
    pub nodes: u64,
    pub elapsed: Duration,
}

impl Default for MctsConfig<RolloutPolicyConfig<NeutralEvaluator>> {
    fn default() -> Self {
        Self {
            budget: SearchBudget::default(),
            exploration: std::f64::consts::SQRT_2,
            selection_policy: SelectionPolicy::Uct,
            rollout_depth: 256,
            rollout_policy: RolloutPolicyConfig::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum RolloutPolicyConfig<E = NeutralEvaluator> {
    #[default]
    UniformRandom,
    Greedy {
        evaluator: E,
    },
    EpsilonGreedy {
        epsilon: f64,
        evaluator: E,
    },
    /// Learn action values during each search and use epsilon-greedy rollouts.
    Mast {
        epsilon: f64,
    },
}

/// Statistics shared by simulations within one decision, never across decisions.
pub struct RolloutMemory<A> {
    values: HashMap<(PlayerId, A), MoveAverage>,
    trajectory: Vec<(PlayerId, A)>,
}

impl<A> Default for RolloutMemory<A> {
    fn default() -> Self {
        Self {
            values: HashMap::new(),
            trajectory: Vec::new(),
        }
    }
}

#[derive(Default)]
struct MoveAverage {
    visits: u64,
    total: f64,
}

impl<A: Eq + Hash> RolloutMemory<A> {
    fn finish_simulation(&mut self, root_player: PlayerId, utility: f64) {
        // Every occurrence in selection, expansion, and rollout receives credit.
        for (player, action) in self.trajectory.drain(..) {
            let value = self.values.entry((player, action)).or_default();
            value.visits += 1;
            value.total += if player == root_player {
                utility
            } else {
                -utility
            };
        }
    }
}

pub trait StateEvaluator<G: meeple_bots_core::Game> {
    fn evaluate(
        &self,
        game: &G,
        state: &G::State,
        root_player: PlayerId,
    ) -> Result<f64, AgentError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NeutralEvaluator;

impl<G: meeple_bots_core::Game> StateEvaluator<G> for NeutralEvaluator {
    fn evaluate(
        &self,
        _game: &G,
        _state: &G::State,
        _root_player: PlayerId,
    ) -> Result<f64, AgentError> {
        Ok(0.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GameHeuristic {
    pub index: u32,
}

impl GameHeuristic {
    pub const fn new(index: u32) -> Self {
        Self { index }
    }
}

impl<G> StateEvaluator<G> for GameHeuristic
where
    G: HeuristicGame,
{
    fn evaluate(
        &self,
        game: &G,
        state: &G::State,
        root_player: PlayerId,
    ) -> Result<f64, AgentError> {
        game.heuristic_utility(self.index, state, root_player)
            .map(f64::from)
            .ok_or_else(|| {
                AgentError::message(format!(
                    "heuristic index {} is not available; the game provides {} heuristics",
                    self.index,
                    game.heuristic_count()
                ))
            })
    }
}

/// Selects player actions; the search resolves public chance before calling the policy.
/// Implementations that evaluate successors must explicitly support chance states or
/// retain a `DeterministicGame` bound.
pub trait RolloutPolicy<G>
where
    G: meeple_bots_core::Game,
    G::State: Clone,
{
    /// Validates configuration before a search starts.
    fn validate(&self) -> Result<(), AgentError> {
        Ok(())
    }

    /// Select with statistics accumulated by earlier simulations in this search.
    fn select_action_with_memory<R: RandomSource + ?Sized>(
        &self,
        game: &G,
        state: &G::State,
        active_player: PlayerId,
        root_player: PlayerId,
        memory: &RolloutMemory<G::Action>,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        let _ = memory;
        self.select_action(game, state, active_player, root_player, rng)
    }

    /// Record tree and rollout actions only for policies that learn from results.
    fn action_for_learning(&self, _action: &G::Action) -> Option<G::Action> {
        None
    }

    fn finish_simulation(
        &self,
        _memory: &mut RolloutMemory<G::Action>,
        _root_player: PlayerId,
        _utility: f64,
    ) {
    }

    fn select_action<R: RandomSource + ?Sized>(
        &self,
        game: &G,
        state: &G::State,
        active_player: PlayerId,
        root_player: PlayerId,
        rng: &mut R,
    ) -> Result<G::Action, AgentError>;
}

/// Matches the player decision state before an action is applied.
pub trait PolicyCondition<G>
where
    G: meeple_bots_core::Game,
{
    fn matches(
        &self,
        game: &G,
        state: &G::State,
        active_player: PlayerId,
        root_player: PlayerId,
    ) -> bool;
}

pub use PolicyCondition as RolloutCondition;

#[derive(Clone, Copy, Debug, Default)]
pub struct Always;

impl<G: meeple_bots_core::Game> PolicyCondition<G> for Always {
    fn matches(
        &self,
        _game: &G,
        _state: &G::State,
        _active_player: PlayerId,
        _root_player: PlayerId,
    ) -> bool {
        true
    }
}

/// A prior on player actions, never a policy for choosing chance outcomes.
/// Implementations define which successor states their evaluator supports.
pub trait SelectionBias<G>
where
    G: meeple_bots_core::Game,
{
    fn weight(&self) -> f64;

    fn applies(
        &self,
        game: &G,
        state: &G::State,
        active_player: PlayerId,
        root_player: PlayerId,
    ) -> bool;

    fn evaluate_child(
        &self,
        game: &G,
        state: &G::State,
        root_player: PlayerId,
    ) -> Result<f64, AgentError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoSelectionBias;

impl<G: meeple_bots_core::Game> SelectionBias<G> for NoSelectionBias {
    fn weight(&self) -> f64 {
        0.0
    }

    fn applies(
        &self,
        _game: &G,
        _state: &G::State,
        _active_player: PlayerId,
        _root_player: PlayerId,
    ) -> bool {
        false
    }

    fn evaluate_child(
        &self,
        _game: &G,
        _state: &G::State,
        _root_player: PlayerId,
    ) -> Result<f64, AgentError> {
        Ok(0.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProgressiveBias<C, E> {
    pub weight: f64,
    pub condition: C,
    pub evaluator: E,
}

impl<C, E> ProgressiveBias<C, E> {
    pub const fn new(weight: f64, condition: C, evaluator: E) -> Self {
        Self {
            weight,
            condition,
            evaluator,
        }
    }
}

impl<G, C, E> SelectionBias<G> for ProgressiveBias<C, E>
where
    G: meeple_bots_core::Game,
    C: PolicyCondition<G>,
    E: StateEvaluator<G>,
{
    fn weight(&self) -> f64 {
        self.weight
    }

    fn applies(
        &self,
        game: &G,
        state: &G::State,
        active_player: PlayerId,
        root_player: PlayerId,
    ) -> bool {
        self.condition
            .matches(game, state, active_player, root_player)
    }

    fn evaluate_child(
        &self,
        game: &G,
        state: &G::State,
        root_player: PlayerId,
    ) -> Result<f64, AgentError> {
        evaluate_state(game, state, root_player, &self.evaluator)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConditionalRollout<C, P, F> {
    pub condition: C,
    pub primary: P,
    pub fallback: F,
}

impl<C, P, F> ConditionalRollout<C, P, F> {
    pub const fn new(condition: C, primary: P, fallback: F) -> Self {
        Self {
            condition,
            primary,
            fallback,
        }
    }
}

impl<G, C, P, F> RolloutPolicy<G> for ConditionalRollout<C, P, F>
where
    G: meeple_bots_core::Game,
    G::State: Clone,
    C: PolicyCondition<G>,
    P: RolloutPolicy<G>,
    F: RolloutPolicy<G>,
{
    fn validate(&self) -> Result<(), AgentError> {
        self.primary.validate()?;
        self.fallback.validate()
    }

    fn action_for_learning(&self, action: &G::Action) -> Option<G::Action> {
        self.primary
            .action_for_learning(action)
            .or_else(|| self.fallback.action_for_learning(action))
    }

    fn finish_simulation(
        &self,
        memory: &mut RolloutMemory<G::Action>,
        root_player: PlayerId,
        utility: f64,
    ) {
        self.primary.finish_simulation(memory, root_player, utility);
        self.fallback
            .finish_simulation(memory, root_player, utility);
    }

    fn select_action_with_memory<R: RandomSource + ?Sized>(
        &self,
        game: &G,
        state: &G::State,
        active_player: PlayerId,
        root_player: PlayerId,
        memory: &RolloutMemory<G::Action>,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        if self
            .condition
            .matches(game, state, active_player, root_player)
        {
            self.primary.select_action_with_memory(
                game,
                state,
                active_player,
                root_player,
                memory,
                rng,
            )
        } else {
            self.fallback.select_action_with_memory(
                game,
                state,
                active_player,
                root_player,
                memory,
                rng,
            )
        }
    }

    fn select_action<R: RandomSource + ?Sized>(
        &self,
        game: &G,
        state: &G::State,
        active_player: PlayerId,
        root_player: PlayerId,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        if self
            .condition
            .matches(game, state, active_player, root_player)
        {
            self.primary
                .select_action(game, state, active_player, root_player, rng)
        } else {
            self.fallback
                .select_action(game, state, active_player, root_player, rng)
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct UniformRandom;

impl<G> RolloutPolicy<G> for UniformRandom
where
    G: meeple_bots_core::Game,
    G::State: Clone,
{
    fn select_action<R: RandomSource + ?Sized>(
        &self,
        game: &G,
        state: &G::State,
        _active_player: PlayerId,
        _root_player: PlayerId,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        let mut actions: Vec<_> = game.legal_actions(state).collect();
        let index = rng.index(actions.len()).ok_or(AgentError::NoLegalActions)?;
        Ok(actions.swap_remove(index))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Greedy<E> {
    pub evaluator: E,
}

impl<E> Greedy<E> {
    pub const fn new(evaluator: E) -> Self {
        Self { evaluator }
    }
}

impl<G, E> RolloutPolicy<G> for Greedy<E>
where
    G: meeple_bots_core::Game,
    G::State: Clone,
    E: StateEvaluator<G>,
{
    fn select_action<R: RandomSource + ?Sized>(
        &self,
        game: &G,
        state: &G::State,
        active_player: PlayerId,
        root_player: PlayerId,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        select_greedy_action(
            game,
            state,
            active_player,
            root_player,
            &self.evaluator,
            rng,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpsilonGreedy<E> {
    pub epsilon: f64,
    pub evaluator: E,
}

impl<E> EpsilonGreedy<E> {
    pub const fn new(epsilon: f64, evaluator: E) -> Self {
        Self { epsilon, evaluator }
    }
}

impl<G, E> RolloutPolicy<G> for EpsilonGreedy<E>
where
    G: meeple_bots_core::Game,
    G::State: Clone,
    E: StateEvaluator<G>,
{
    fn validate(&self) -> Result<(), AgentError> {
        validate_rollout_epsilon(self.epsilon)
    }

    fn select_action<R: RandomSource + ?Sized>(
        &self,
        game: &G,
        state: &G::State,
        active_player: PlayerId,
        root_player: PlayerId,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        select_epsilon_greedy_action(
            game,
            state,
            active_player,
            root_player,
            self.epsilon,
            &self.evaluator,
            rng,
        )
    }
}

impl<G, E> RolloutPolicy<G> for RolloutPolicyConfig<E>
where
    G: meeple_bots_core::Game,
    G::State: Clone,
    G::Action: Clone + Eq + Hash,
    E: StateEvaluator<G>,
{
    fn validate(&self) -> Result<(), AgentError> {
        match self {
            Self::EpsilonGreedy { epsilon, .. } | Self::Mast { epsilon } => {
                validate_rollout_epsilon(*epsilon)
            }
            Self::UniformRandom | Self::Greedy { .. } => Ok(()),
        }
    }

    fn action_for_learning(&self, action: &G::Action) -> Option<G::Action> {
        matches!(self, Self::Mast { .. }).then(|| action.clone())
    }

    fn finish_simulation(
        &self,
        memory: &mut RolloutMemory<G::Action>,
        root_player: PlayerId,
        utility: f64,
    ) {
        memory.finish_simulation(root_player, utility);
    }

    fn select_action_with_memory<R: RandomSource + ?Sized>(
        &self,
        game: &G,
        state: &G::State,
        active_player: PlayerId,
        root_player: PlayerId,
        memory: &RolloutMemory<G::Action>,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        let Self::Mast { epsilon } = self else {
            return self.select_action(game, state, active_player, root_player, rng);
        };
        if *epsilon == 1.0 || (*epsilon > 0.0 && rng.unit_f64() < *epsilon) {
            return UniformRandom.select_action(game, state, active_player, root_player, rng);
        }
        let mut best = f64::NEG_INFINITY;
        let mut candidates = Vec::new();
        for action in game.legal_actions(state) {
            let mean = memory
                .values
                .get(&(active_player, action.clone()))
                .map_or(0.0, |value| value.total / value.visits as f64);
            match mean.total_cmp(&best) {
                Ordering::Greater => {
                    best = mean;
                    candidates.clear();
                    candidates.push(action);
                }
                Ordering::Equal => candidates.push(action),
                Ordering::Less => {}
            }
        }
        let index = rng
            .index(candidates.len())
            .ok_or(AgentError::NoLegalActions)?;
        Ok(candidates.swap_remove(index))
    }

    fn select_action<R: RandomSource + ?Sized>(
        &self,
        game: &G,
        state: &G::State,
        active_player: PlayerId,
        root_player: PlayerId,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        match self {
            // A standalone selection has no previous simulations to learn from.
            Self::UniformRandom | Self::Mast { .. } => {
                UniformRandom.select_action(game, state, active_player, root_player, rng)
            }
            Self::Greedy { evaluator } => {
                select_greedy_action(game, state, active_player, root_player, evaluator, rng)
            }
            Self::EpsilonGreedy { epsilon, evaluator } => select_epsilon_greedy_action(
                game,
                state,
                active_player,
                root_player,
                *epsilon,
                evaluator,
                rng,
            ),
        }
    }
}

fn select_epsilon_greedy_action<G, E, R>(
    game: &G,
    state: &G::State,
    active_player: PlayerId,
    root_player: PlayerId,
    epsilon: f64,
    evaluator: &E,
    rng: &mut R,
) -> Result<G::Action, AgentError>
where
    G: meeple_bots_core::Game,
    G::State: Clone,
    E: StateEvaluator<G>,
    R: RandomSource + ?Sized,
{
    if epsilon == 1.0 || (epsilon > 0.0 && rng.unit_f64() < epsilon) {
        UniformRandom.select_action(game, state, active_player, root_player, rng)
    } else {
        select_greedy_action(game, state, active_player, root_player, evaluator, rng)
    }
}

fn validate_rollout_epsilon(epsilon: f64) -> Result<(), AgentError> {
    if !epsilon.is_finite() || !(0.0..=1.0).contains(&epsilon) {
        return Err(AgentError::message(
            "MCTS rollout epsilon must be finite and between 0.0 and 1.0",
        ));
    }
    Ok(())
}

fn select_greedy_action<G, E, R>(
    game: &G,
    state: &G::State,
    active_player: PlayerId,
    root_player: PlayerId,
    evaluator: &E,
    rng: &mut R,
) -> Result<G::Action, AgentError>
where
    G: meeple_bots_core::Game,
    G::State: Clone,
    E: StateEvaluator<G>,
    R: RandomSource + ?Sized,
{
    let maximizing = active_player == root_player;
    let mut best_score = None;
    let mut best_actions = Vec::new();
    for action in game.legal_actions(state) {
        let mut successor = state.clone();
        game.apply_action(&mut successor, &action)
            .map_err(|error| AgentError::message(error.to_string()))?;
        // Candidate probes are discarded. The selected action is applied and chance
        // sampled independently by the rollout, avoiding selection-conditioned outcomes.
        stochastic::resolve_chance(game, &mut successor, rng)?;
        let score = evaluate_state(game, &successor, root_player, evaluator)?;
        let ordering = best_score.map(|best: f64| score.total_cmp(&best));
        let is_better = matches!(
            (maximizing, ordering),
            (true, Some(Ordering::Greater)) | (false, Some(Ordering::Less))
        );
        if best_score.is_none() || is_better {
            best_score = Some(score);
            best_actions.clear();
            best_actions.push(action);
        } else if ordering == Some(Ordering::Equal) {
            best_actions.push(action);
        }
    }

    let tie_index = rng
        .index(best_actions.len())
        .ok_or(AgentError::NoLegalActions)?;
    Ok(best_actions.swap_remove(tie_index))
}

/// Tracks agent work only: opponent search and GUI delays never consume this budget.
#[derive(Debug)]
struct DecisionBudget {
    pending_maintenance: Duration,
    tail_reserve: Duration,
    started: Option<Instant>,
    allowance: Duration,
    search_finished: Option<Instant>,
}

impl Clone for DecisionBudget {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl DecisionBudget {
    const fn new() -> Self {
        Self {
            pending_maintenance: Duration::ZERO,
            tail_reserve: Duration::ZERO,
            started: None,
            allowance: Duration::ZERO,
            search_finished: None,
        }
    }

    fn begin(&mut self, budget: SearchBudget) {
        self.search_finished = None;
        if let SearchBudget::Time(limit) = budget {
            self.started = Some(Instant::now());
            self.allowance = limit
                .saturating_sub(self.pending_maintenance)
                .saturating_sub(self.tail_reserve);
        } else {
            self.started = None;
        }
        self.pending_maintenance = Duration::ZERO;
    }

    fn exhausted(&self) -> bool {
        self.started
            .is_some_and(|start| start.elapsed() >= self.allowance)
    }

    fn finish_search(&mut self) {
        self.search_finished = self.started.map(|_| Instant::now());
    }

    fn finish(&mut self) {
        if let Some(start) = self.search_finished.take() {
            self.tail_reserve = start.elapsed();
        }
        self.started = None;
    }
}

#[derive(Clone, Debug)]
pub struct MctsAgent<
    C = NeutralEvaluator,
    P = RolloutPolicyConfig<NeutralEvaluator>,
    B = NoSelectionBias,
> {
    pub config: MctsConfig<P>,
    pub cutoff_evaluator: C,
    pub selection_bias: B,
    pub root_diagnostics: bool,
    last_search_stats: Option<MctsSearchStats>,
    last_root_actions: Vec<RootActionStats>,
    decision_budget: DecisionBudget,
}

impl Default for MctsAgent<NeutralEvaluator, RolloutPolicyConfig<NeutralEvaluator>> {
    fn default() -> Self {
        Self::new(MctsConfig::default())
    }
}

impl<P> MctsAgent<NeutralEvaluator, P> {
    pub const fn new(config: MctsConfig<P>) -> Self {
        Self {
            config,
            cutoff_evaluator: NeutralEvaluator,
            selection_bias: NoSelectionBias,
            root_diagnostics: false,
            last_search_stats: None,
            last_root_actions: Vec::new(),
            decision_budget: DecisionBudget::new(),
        }
    }
}

impl<C, P> MctsAgent<C, P> {
    pub const fn with_cutoff_evaluator(config: MctsConfig<P>, cutoff_evaluator: C) -> Self {
        Self {
            config,
            cutoff_evaluator,
            selection_bias: NoSelectionBias,
            root_diagnostics: false,
            last_search_stats: None,
            last_root_actions: Vec::new(),
            decision_budget: DecisionBudget::new(),
        }
    }
}

impl<C, P, B> MctsAgent<C, P, B> {
    pub const fn with_progressive_bias(
        config: MctsConfig<P>,
        cutoff_evaluator: C,
        selection_bias: B,
    ) -> Self {
        Self {
            config,
            cutoff_evaluator,
            selection_bias,
            root_diagnostics: false,
            last_search_stats: None,
            last_root_actions: Vec::new(),
            decision_budget: DecisionBudget::new(),
        }
    }

    pub fn with_root_diagnostics(mut self, enabled: bool) -> Self {
        self.root_diagnostics = enabled;
        self
    }

    pub const fn last_search_stats(&self) -> Option<MctsSearchStats> {
        self.last_search_stats
    }

    pub fn decision_stats(&self) -> AgentDecisionStats {
        self.last_search_stats
            .as_ref()
            .map_or_else(AgentDecisionStats::default, |stats| AgentDecisionStats {
                search_iterations: Some(stats.iterations),
                search_nodes: Some(stats.nodes),
                root_actions: self.last_root_actions.clone(),
                tree_reuse: None,
            })
    }

    fn choose_action<G, R>(
        &mut self,
        decision: DecisionContext<'_, G>,
        rng: &mut R,
    ) -> Result<G::Action, AgentError>
    where
        G: DeterministicGame + PerfectInformationGame + TwoPlayerZeroSumGame,
        G::State: Clone,
        C: StateEvaluator<G>,
        P: RolloutPolicy<G>,
        B: SelectionBias<G>,
        R: RandomSource + ?Sized,
    {
        self.last_search_stats = None;
        self.last_root_actions.clear();

        self.config.validate().map_err(AgentError::message)?;
        self.config.rollout_policy.validate()?;

        let game = decision.game();
        let root_state = decision.state();
        let root_player = decision.player();

        if game.status(root_state) != PositionStatus::PlayerTurn(root_player) {
            return Err(AgentError::message(
                "MCTS received a position for the wrong player",
            ));
        }

        let root = Node::new(None, 0.0, game.legal_actions(root_state));
        if root.unexpanded.is_empty() {
            return Err(AgentError::NoLegalActions);
        }
        let root_unexpanded_action_indices = if self.root_diagnostics {
            let root_action_count = u32::try_from(root.unexpanded.len())
                .map_err(|_| AgentError::message("MCTS root has too many legal actions"))?;
            (0..root_action_count).collect()
        } else {
            Vec::new()
        };
        let root_action_indices = if self.root_diagnostics {
            Some(RootActionIndices {
                unexpanded: root_unexpanded_action_indices,
                children: Vec::with_capacity(root.unexpanded.len()),
            })
        } else {
            None
        };
        let mut nodes = vec![root];
        let selected_index = self.search_tree(
            game,
            root_state,
            root_player,
            &mut nodes,
            root_action_indices,
            true,
            rng,
        )?;

        nodes[selected_index]
            .action
            .take()
            .ok_or(AgentError::NoLegalActions)
    }

    // Keep the borrowed search inputs explicit; grouping them would only wrap this call.
    #[allow(clippy::too_many_arguments)]
    fn search_tree<G, R>(
        &mut self,
        game: &G,
        root_state: &G::State,
        root_player: PlayerId,
        nodes: &mut Vec<Node<G::Action>>,
        mut root_action_indices: Option<RootActionIndices>,
        count_existing_nodes: bool,
        rng: &mut R,
    ) -> Result<usize, AgentError>
    where
        G: DeterministicGame + PerfectInformationGame + TwoPlayerZeroSumGame,
        G::State: Clone,
        C: StateEvaluator<G>,
        P: RolloutPolicy<G>,
        B: SelectionBias<G>,
        R: RandomSource + ?Sized,
    {
        let nodes_before_search = nodes.len();
        let bias_weight = self.selection_bias.weight();
        if !bias_weight.is_finite() || bias_weight < 0.0 {
            return Err(AgentError::message(
                "MCTS progressive bias weight must be finite and non-negative",
            ));
        }
        let bias_enabled = bias_weight > 0.0;

        let search_started = Instant::now();
        let mut rollout_memory = RolloutMemory::default();
        let mut completed_iterations = 0_u64;
        let mut path = Vec::new();
        loop {
            let budget_exhausted = match self.config.budget {
                SearchBudget::Iterations(iterations) => {
                    completed_iterations >= u64::from(iterations.get())
                }
                SearchBudget::Time(_) => {
                    completed_iterations > 0 && self.decision_budget.exhausted()
                }
            };
            if budget_exhausted {
                break;
            }

            let mut state = root_state.clone();
            let mut node_index = 0;
            path.clear();
            path.push(0);

            loop {
                let status = game.status(&state);
                if matches!(status, PositionStatus::Terminal) {
                    break;
                }

                let active_player = match status {
                    PositionStatus::PlayerTurn(player) => player,
                    PositionStatus::Chance => {
                        return Err(AgentError::message(
                            "MCTS does not support chance transitions",
                        ));
                    }
                    PositionStatus::Terminal => unreachable!(),
                    _ => return Err(AgentError::message("unsupported position status")),
                };

                if !nodes[node_index].unexpanded.is_empty() {
                    let bias_applies = bias_enabled
                        && self
                            .selection_bias
                            .applies(game, &state, active_player, root_player);
                    let unexpanded_slot = rng
                        .index(nodes[node_index].unexpanded.len())
                        .expect("non-empty actions");
                    let unexpanded = nodes[node_index].unexpanded.swap_remove(unexpanded_slot);
                    let root_action_index = (node_index == 0)
                        .then(|| {
                            root_action_indices
                                .as_mut()
                                .map(|indices| indices.unexpanded.swap_remove(unexpanded_slot))
                        })
                        .flatten();
                    if let Some(recorded) =
                        self.config.rollout_policy.action_for_learning(&unexpanded)
                    {
                        rollout_memory.trajectory.push((active_player, recorded));
                    }
                    game.apply_action(&mut state, &unexpanded)
                        .map_err(|error| AgentError::message(error.to_string()))?;

                    let heuristic_value = if bias_applies {
                        self.selection_bias
                            .evaluate_child(game, &state, root_player)?
                    } else {
                        0.0
                    };

                    let child_index = nodes.len();
                    let child = if matches!(game.status(&state), PositionStatus::PlayerTurn(_)) {
                        Node::new(
                            Some(unexpanded),
                            heuristic_value,
                            game.legal_actions(&state),
                        )
                    } else {
                        Node::new(Some(unexpanded), heuristic_value, std::iter::empty())
                    };
                    nodes.push(child);
                    nodes[node_index].children.push(child_index);
                    if let Some(action_index) = root_action_index {
                        root_action_indices
                            .as_mut()
                            .expect("root action indices exist")
                            .children
                            .push(action_index);
                    }
                    node_index = child_index;
                    path.push(node_index);
                    break;
                }

                if nodes[node_index].children.is_empty() {
                    return Err(AgentError::message(
                        "non-terminal MCTS node has no legal actions",
                    ));
                }

                let maximizing = active_player == root_player;
                let selected = best_child(
                    nodes,
                    node_index,
                    maximizing,
                    self.config.exploration,
                    self.config.selection_policy,
                    bias_weight,
                );
                let action = nodes[selected]
                    .action
                    .as_ref()
                    .expect("child has an action");
                if let Some(recorded) = self.config.rollout_policy.action_for_learning(action) {
                    rollout_memory.trajectory.push((active_player, recorded));
                }
                game.apply_action(&mut state, action)
                    .map_err(|error| AgentError::message(error.to_string()))?;
                node_index = selected;
                path.push(node_index);
            }

            let utility = rollout_with_memory(
                game,
                &mut state,
                root_player,
                self.config.rollout_depth,
                &self.config.rollout_policy,
                &self.cutoff_evaluator,
                &mut rollout_memory,
                rng,
            )?;
            self.config
                .rollout_policy
                .finish_simulation(&mut rollout_memory, root_player, utility);
            for &visited in &path {
                nodes[visited].visits += 1;
                nodes[visited].total_utility += utility;
                if self.config.selection_policy == SelectionPolicy::Ucb1Tuned {
                    nodes[visited].total_squared_utility += utility * utility;
                }
            }
            completed_iterations += 1;
        }

        self.decision_budget.finish_search();
        let selected_index = nodes[0]
            .children
            .iter()
            .copied()
            .max_by(|left, right| {
                nodes[*left]
                    .visits
                    .cmp(&nodes[*right].visits)
                    .then_with(|| {
                        nodes[*left]
                            .mean_utility()
                            .total_cmp(&nodes[*right].mean_utility())
                    })
            })
            .ok_or(AgentError::NoLegalActions)?;

        self.last_root_actions = if self.root_diagnostics {
            let indices = root_action_indices
                .as_ref()
                .expect("root diagnostics indices exist");
            nodes[0]
                .children
                .iter()
                .copied()
                .zip(indices.children.iter().copied())
                .map(|(child_index, action_index)| {
                    let child = &nodes[child_index];
                    RootActionStats {
                        action_index,
                        visits: child.visits,
                        mean_utility: child.mean_utility(),
                        heuristic_value: bias_enabled.then_some(child.heuristic_value),
                        progressive_bias: bias_enabled
                            .then(|| progressive_bias_term(child, true, bias_weight)),
                        selected: child_index == selected_index,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        self.last_search_stats = Some(MctsSearchStats {
            iterations: completed_iterations,
            nodes: if count_existing_nodes {
                nodes.len() as u64
            } else {
                nodes.len().saturating_sub(nodes_before_search) as u64
            },
            elapsed: search_started.elapsed(),
        });
        Ok(selected_index)
    }

    fn search_graph<G, R>(
        &mut self,
        game: &G,
        root_player: PlayerId,
        graph: &mut ReusableGraph<G>,
        mut root_action_indices: Option<RootActionIndices>,
        count_existing_nodes: bool,
        rng: &mut R,
    ) -> Result<usize, AgentError>
    where
        G: DeterministicGame + PerfectInformationGame + TwoPlayerZeroSumGame,
        G::State: Clone + Eq + Hash,
        G::Action: Clone,
        C: StateEvaluator<G>,
        P: RolloutPolicy<G>,
        B: SelectionBias<G>,
        R: RandomSource + ?Sized,
    {
        let nodes_before_search = graph.nodes.len();
        let bias_weight = self.selection_bias.weight();
        if !bias_weight.is_finite() || bias_weight < 0.0 {
            return Err(AgentError::message(
                "MCTS progressive bias weight must be finite and non-negative",
            ));
        }
        let bias_enabled = bias_weight > 0.0;

        let search_started = Instant::now();
        let mut rollout_memory = RolloutMemory::default();
        let mut completed_iterations = 0_u64;
        let mut path_nodes = Vec::new();
        let mut path_edges = Vec::new();
        loop {
            let budget_exhausted = match self.config.budget {
                SearchBudget::Iterations(iterations) => {
                    completed_iterations >= u64::from(iterations.get())
                }
                SearchBudget::Time(_) => {
                    completed_iterations > 0 && self.decision_budget.exhausted()
                }
            };
            if budget_exhausted {
                break;
            }

            let mut state = graph.nodes[0].state.clone();
            let mut node_index = 0;
            path_nodes.clear();
            path_edges.clear();
            path_nodes.push(0);

            loop {
                let status = game.status(&state);
                if matches!(status, PositionStatus::Terminal) {
                    break;
                }

                let active_player = match status {
                    PositionStatus::PlayerTurn(player) => player,
                    PositionStatus::Chance => {
                        return Err(AgentError::message(
                            "MCTS does not support chance transitions",
                        ));
                    }
                    PositionStatus::Terminal => unreachable!(),
                    _ => return Err(AgentError::message("unsupported position status")),
                };

                if !graph.nodes[node_index].unexpanded.is_empty() {
                    let bias_applies = bias_enabled
                        && self
                            .selection_bias
                            .applies(game, &state, active_player, root_player);
                    let unexpanded_slot = rng
                        .index(graph.nodes[node_index].unexpanded.len())
                        .expect("non-empty actions");
                    let action = graph.nodes[node_index]
                        .unexpanded
                        .swap_remove(unexpanded_slot);
                    let root_action_index = (node_index == 0)
                        .then(|| {
                            root_action_indices
                                .as_mut()
                                .map(|indices| indices.unexpanded.swap_remove(unexpanded_slot))
                        })
                        .flatten();
                    if let Some(recorded) = self.config.rollout_policy.action_for_learning(&action)
                    {
                        rollout_memory.trajectory.push((active_player, recorded));
                    }
                    game.apply_action(&mut state, &action)
                        .map_err(|error| AgentError::message(error.to_string()))?;

                    let heuristic_value = if bias_applies {
                        self.selection_bias
                            .evaluate_child(game, &state, root_player)?
                    } else {
                        0.0
                    };
                    let child_index = if let Some(existing) = graph.state_indices.get(&state) {
                        *existing
                    } else {
                        let child_index = graph.nodes.len();
                        let child = if matches!(game.status(&state), PositionStatus::PlayerTurn(_))
                        {
                            GraphNode::new(state.clone(), game.legal_actions(&state))
                        } else {
                            GraphNode::new(state.clone(), std::iter::empty())
                        };
                        graph.nodes.push(child);
                        graph.state_indices.insert(state.clone(), child_index);
                        child_index
                    };
                    let edge_index = graph.nodes[node_index].edges.len();
                    graph.nodes[node_index].edges.push(GraphEdge {
                        action,
                        child: child_index,
                        visits: 0,
                        total_utility: 0.0,
                        total_squared_utility: 0.0,
                        heuristic_value,
                    });
                    if let Some(action_index) = root_action_index {
                        root_action_indices
                            .as_mut()
                            .expect("root action indices exist")
                            .children
                            .push(action_index);
                    }
                    path_edges.push((node_index, edge_index));
                    if !path_nodes.contains(&child_index) {
                        path_nodes.push(child_index);
                    }
                    break;
                }

                if graph.nodes[node_index].edges.is_empty() {
                    return Err(AgentError::message(
                        "non-terminal MCTS node has no legal actions",
                    ));
                }

                let maximizing = active_player == root_player;
                let edge_index = best_graph_edge(
                    &graph.nodes,
                    node_index,
                    maximizing,
                    self.config.exploration,
                    self.config.selection_policy,
                    bias_weight,
                );
                let action = graph.nodes[node_index].edges[edge_index].action.clone();
                let child_index = graph.nodes[node_index].edges[edge_index].child;
                if let Some(recorded) = self.config.rollout_policy.action_for_learning(&action) {
                    rollout_memory.trajectory.push((active_player, recorded));
                }
                game.apply_action(&mut state, &action)
                    .map_err(|error| AgentError::message(error.to_string()))?;
                path_edges.push((node_index, edge_index));
                if path_nodes.contains(&child_index) {
                    break;
                }
                node_index = child_index;
                path_nodes.push(node_index);
            }

            let utility = rollout_with_memory(
                game,
                &mut state,
                root_player,
                self.config.rollout_depth,
                &self.config.rollout_policy,
                &self.cutoff_evaluator,
                &mut rollout_memory,
                rng,
            )?;
            self.config
                .rollout_policy
                .finish_simulation(&mut rollout_memory, root_player, utility);
            for &visited in &path_nodes {
                graph.nodes[visited].visits += 1;
                graph.nodes[visited].total_utility += utility;
            }
            for &(parent, edge) in &path_edges {
                let edge = &mut graph.nodes[parent].edges[edge];
                edge.visits += 1;
                if self.config.selection_policy == SelectionPolicy::Ucb1Tuned {
                    edge.total_utility += utility;
                    edge.total_squared_utility += utility * utility;
                }
            }
            completed_iterations += 1;
        }

        self.decision_budget.finish_search();
        let selected_edge = graph.nodes[0]
            .edges
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                left.visits.cmp(&right.visits).then_with(|| {
                    graph_edge_mean(&graph.nodes, left, self.config.selection_policy).total_cmp(
                        &graph_edge_mean(&graph.nodes, right, self.config.selection_policy),
                    )
                })
            })
            .map(|(index, _)| index)
            .ok_or(AgentError::NoLegalActions)?;

        self.last_root_actions = if self.root_diagnostics {
            let indices = root_action_indices
                .as_ref()
                .expect("root diagnostics indices exist");
            graph.nodes[0]
                .edges
                .iter()
                .enumerate()
                .zip(indices.children.iter().copied())
                .map(|((edge_index, edge), action_index)| RootActionStats {
                    action_index,
                    visits: edge.visits,
                    mean_utility: graph_edge_mean(&graph.nodes, edge, self.config.selection_policy),
                    heuristic_value: bias_enabled.then_some(edge.heuristic_value),
                    progressive_bias: bias_enabled
                        .then(|| graph_progressive_bias_term(edge, true, bias_weight)),
                    selected: edge_index == selected_edge,
                })
                .collect()
        } else {
            Vec::new()
        };
        self.last_search_stats = Some(MctsSearchStats {
            iterations: completed_iterations,
            nodes: if count_existing_nodes {
                graph.nodes.len() as u64
            } else {
                graph.nodes.len().saturating_sub(nodes_before_search) as u64
            },
            elapsed: search_started.elapsed(),
        });
        Ok(selected_edge)
    }
}

struct RootActionIndices {
    unexpanded: Vec<u32>,
    children: Vec<u32>,
}

/// Opt-in MCTS wrapper that retains the reachable subtree between match decisions.
pub struct TreeReuseMctsAgent<
    G: meeple_bots_core::Game,
    C = NeutralEvaluator,
    P = RolloutPolicyConfig<NeutralEvaluator>,
    B = NoSelectionBias,
> {
    inner: MctsAgent<C, P, B>,
    enabled: bool,
    owner: Option<PlayerId>,
    tree: Option<ReusableTree<G>>,
    pending_reuse_stats: TreeReuseStats,
    last_reuse_stats: Option<TreeReuseStats>,
}

impl<G, C, P, B> TreeReuseMctsAgent<G, C, P, B>
where
    G: meeple_bots_core::Game,
{
    pub const fn new(inner: MctsAgent<C, P, B>, enabled: bool) -> Self {
        Self {
            inner,
            enabled,
            owner: None,
            tree: None,
            pending_reuse_stats: TreeReuseStats {
                transition_attempts: 0,
                transition_hits: 0,
                transition_misses: 0,
                own_action_hits: 0,
                opponent_action_hits: 0,
                reused_root_visits: 0,
                reused_nodes: 0,
                pruned_nodes: 0,
                resets: 0,
            },
            last_reuse_stats: None,
        }
    }

    pub const fn tree_reuse_enabled(&self) -> bool {
        self.enabled
    }

    pub const fn inner(&self) -> &MctsAgent<C, P, B> {
        &self.inner
    }

    pub const fn last_search_stats(&self) -> Option<MctsSearchStats> {
        self.inner.last_search_stats()
    }

    pub fn decision_stats(&self) -> AgentDecisionStats {
        let mut stats = self.inner.decision_stats();
        if self.enabled {
            stats.tree_reuse = self.last_reuse_stats;
        }
        stats
    }

    /// Mutable access invalidates retained search data because configuration may change.
    pub fn inner_mut(&mut self) -> &mut MctsAgent<C, P, B> {
        self.reset_tree();
        &mut self.inner
    }

    fn reset_tree(&mut self) {
        self.tree = None;
        self.owner = None;
        self.pending_reuse_stats = TreeReuseStats::default();
        self.last_reuse_stats = None;
    }

    fn record_transition_miss(&mut self) {
        self.tree = None;
        self.pending_reuse_stats.transition_misses += 1;
        self.pending_reuse_stats.resets += 1;
    }
}

impl<G, C, P, B> Clone for TreeReuseMctsAgent<G, C, P, B>
where
    G: meeple_bots_core::Game,
    MctsAgent<C, P, B>: Clone,
{
    fn clone(&self) -> Self {
        Self::new(self.inner.clone(), self.enabled)
    }
}

impl<G, C, P, B> fmt::Debug for TreeReuseMctsAgent<G, C, P, B>
where
    G: meeple_bots_core::Game,
    MctsAgent<C, P, B>: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TreeReuseMctsAgent")
            .field("inner", &self.inner)
            .field("enabled", &self.enabled)
            .field("owner", &self.owner)
            .field("has_tree", &self.tree.is_some())
            .field("pending_reuse_stats", &self.pending_reuse_stats)
            .field("last_reuse_stats", &self.last_reuse_stats)
            .finish()
    }
}

struct ReusableTree<G: meeple_bots_core::Game> {
    game: G,
    state: G::State,
    nodes: Vec<Node<G::Action>>,
}

struct ReusableGraph<G: meeple_bots_core::Game> {
    game: G,
    nodes: Vec<GraphNode<G::State, G::Action>>,
    state_indices: HashMap<G::State, usize>,
}

struct GraphNode<S, A> {
    state: S,
    edges: Vec<GraphEdge<A>>,
    unexpanded: Vec<A>,
    visits: u32,
    total_utility: f64,
}

struct GraphEdge<A> {
    action: A,
    child: usize,
    visits: u32,
    heuristic_value: f64,
    total_utility: f64,
    total_squared_utility: f64,
}

struct Node<A> {
    action: Option<A>,
    heuristic_value: f64,
    children: Vec<usize>,
    unexpanded: Vec<A>,
    visits: u32,
    total_utility: f64,
    total_squared_utility: f64,
}

impl<G, C, P, B> Agent<G> for TreeReuseMctsAgent<G, C, P, B>
where
    G: DeterministicGame + PerfectInformationGame + TwoPlayerZeroSumGame + Clone + PartialEq,
    G::State: Clone + PartialEq,
    G::Action: Clone + PartialEq,
    C: StateEvaluator<G>,
    P: RolloutPolicy<G>,
    B: SelectionBias<G>,
{
    fn on_match_start(&mut self, _game: &G, _state: &G::State, player: PlayerId) {
        self.inner.decision_budget = DecisionBudget::new();
        let timed = matches!(self.inner.config.budget, SearchBudget::Time(_));
        let started = timed.then(Instant::now);
        self.reset_tree();
        self.owner = Some(player);
        if let Some(start) = started {
            self.inner.decision_budget.pending_maintenance += start.elapsed();
        }
    }

    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, G>,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        if !self.enabled {
            return self.inner.select_action(decision, rng);
        }
        self.inner.decision_budget.begin(self.inner.config.budget);
        let result = (|| {
            self.inner.last_search_stats = None;
            self.inner.last_root_actions.clear();
            self.inner.config.validate().map_err(AgentError::message)?;
            self.inner.config.rollout_policy.validate()?;

            let game = decision.game();
            let root_state = decision.state();
            let root_player = decision.player();
            if game.status(root_state) != PositionStatus::PlayerTurn(root_player) {
                return Err(AgentError::message(
                    "MCTS received a position for the wrong player",
                ));
            }
            if self.owner.is_none() {
                self.owner = Some(root_player);
            }
            if self.owner != Some(root_player) {
                self.pending_reuse_stats.resets += 1;
                self.tree = None;
                self.owner = Some(root_player);
            }

            let tree_matches = self
                .tree
                .as_ref()
                .is_some_and(|tree| tree.game == *game && tree.state == *root_state);
            if self.tree.is_some() && !tree_matches {
                self.record_transition_miss();
            }

            let fresh_tree = self.tree.is_none();
            if fresh_tree {
                let root = Node::new(None, 0.0, game.legal_actions(root_state));
                if root.unexpanded.is_empty() {
                    return Err(AgentError::NoLegalActions);
                }
                self.tree = Some(ReusableTree {
                    game: game.clone(),
                    state: root_state.clone(),
                    nodes: vec![root],
                });
            }

            let mut reuse_stats = std::mem::take(&mut self.pending_reuse_stats);
            if !fresh_tree {
                let tree = self.tree.as_ref().expect("tree was initialized");
                reuse_stats.reused_root_visits = tree.nodes[0].visits;
                reuse_stats.reused_nodes = tree.nodes.len() as u64;
            }

            let root_action_indices = if self.inner.root_diagnostics {
                let tree = self.tree.as_ref().expect("tree was initialized");
                match map_root_action_indices(game, root_state, &tree.nodes) {
                    Ok(indices) => Some(indices),
                    Err(error) => {
                        self.tree = None;
                        reuse_stats.resets += 1;
                        self.last_reuse_stats = Some(reuse_stats);
                        return Err(error);
                    }
                }
            } else {
                None
            };

            let tree = self.tree.as_mut().expect("tree was initialized");
            let selected = self.inner.search_tree(
                game,
                root_state,
                root_player,
                &mut tree.nodes,
                root_action_indices,
                fresh_tree,
                rng,
            );
            let selected_index = match selected {
                Ok(index) => index,
                Err(error) => {
                    self.tree = None;
                    reuse_stats.resets += 1;
                    self.last_reuse_stats = Some(reuse_stats);
                    return Err(error);
                }
            };
            let action = tree.nodes[selected_index]
                .action
                .as_ref()
                .cloned()
                .ok_or(AgentError::NoLegalActions)?;
            self.last_reuse_stats = Some(reuse_stats);
            Ok(action)
        })();
        self.inner.decision_budget.finish();
        result
    }

    fn last_decision_stats(&self) -> AgentDecisionStats {
        self.decision_stats()
    }

    fn on_action_applied(
        &mut self,
        game: &G,
        state: &G::State,
        player: PlayerId,
        action: &G::Action,
    ) {
        let timed = matches!(self.inner.config.budget, SearchBudget::Time(_));
        let started = timed.then(Instant::now);
        (|| {
            if !self.enabled || self.tree.is_none() {
                return;
            }
            self.pending_reuse_stats.transition_attempts += 1;

            let transition = self.tree.as_mut().and_then(|tree| {
                if tree.game != *game {
                    return None;
                }
                let mut expected_state = tree.state.clone();
                if game.apply_action(&mut expected_state, action).is_err()
                    || expected_state != *state
                {
                    return None;
                }
                let child_index = tree.nodes[0]
                    .children
                    .iter()
                    .copied()
                    .find(|index| tree.nodes[*index].action.as_ref() == Some(action))?;
                let (retained, pruned) = compact_to_subtree(&mut tree.nodes, child_index);
                tree.state = state.clone();
                Some((retained, pruned))
            });

            if let Some((retained, pruned)) = transition {
                self.pending_reuse_stats.transition_hits += 1;
                self.pending_reuse_stats.reused_nodes = retained as u64;
                self.pending_reuse_stats.pruned_nodes += pruned as u64;
                if self.owner == Some(player) {
                    self.pending_reuse_stats.own_action_hits += 1;
                } else {
                    self.pending_reuse_stats.opponent_action_hits += 1;
                }
            } else {
                self.record_transition_miss();
            }
        })();
        if let Some(start) = started {
            self.inner.decision_budget.pending_maintenance += start.elapsed();
        }
    }

    fn on_match_end(&mut self, _game: &G, _state: &G::State) {
        self.tree = None;
        self.owner = None;
    }
}

/// Opt-in MCTS backend that merges nodes with exactly equal game states.
pub struct TranspositionMctsAgent<
    G: meeple_bots_core::Game,
    C = NeutralEvaluator,
    P = RolloutPolicyConfig<NeutralEvaluator>,
    B = NoSelectionBias,
> {
    basic: TreeReuseMctsAgent<G, C, P, B>,
    enabled: bool,
    owner: Option<PlayerId>,
    graph: Option<ReusableGraph<G>>,
    pending_reuse_stats: TreeReuseStats,
    last_reuse_stats: Option<TreeReuseStats>,
}

impl<G, C, P, B> TranspositionMctsAgent<G, C, P, B>
where
    G: meeple_bots_core::Game,
{
    pub const fn new(inner: MctsAgent<C, P, B>, tree_reuse: bool, transpositions: bool) -> Self {
        Self {
            basic: TreeReuseMctsAgent::new(inner, tree_reuse),
            enabled: transpositions,
            owner: None,
            graph: None,
            pending_reuse_stats: TreeReuseStats {
                transition_attempts: 0,
                transition_hits: 0,
                transition_misses: 0,
                own_action_hits: 0,
                opponent_action_hits: 0,
                reused_root_visits: 0,
                reused_nodes: 0,
                pruned_nodes: 0,
                resets: 0,
            },
            last_reuse_stats: None,
        }
    }

    pub const fn transpositions_enabled(&self) -> bool {
        self.enabled
    }

    pub const fn tree_reuse_enabled(&self) -> bool {
        self.basic.tree_reuse_enabled()
    }

    pub const fn inner(&self) -> &MctsAgent<C, P, B> {
        self.basic.inner()
    }

    pub const fn last_search_stats(&self) -> Option<MctsSearchStats> {
        self.basic.last_search_stats()
    }

    pub fn decision_stats(&self) -> AgentDecisionStats {
        if !self.enabled {
            return self.basic.decision_stats();
        }
        let mut stats = self.basic.inner.decision_stats();
        if self.basic.tree_reuse_enabled() {
            stats.tree_reuse = self.last_reuse_stats;
        }
        stats
    }

    /// Mutable access invalidates retained search data because configuration may change.
    pub fn inner_mut(&mut self) -> &mut MctsAgent<C, P, B> {
        self.reset_graph();
        self.basic.inner_mut()
    }

    fn reset_graph(&mut self) {
        self.graph = None;
        self.owner = None;
        self.pending_reuse_stats = TreeReuseStats::default();
        self.last_reuse_stats = None;
    }

    fn record_transition_miss(&mut self) {
        self.graph = None;
        self.pending_reuse_stats.transition_misses += 1;
        self.pending_reuse_stats.resets += 1;
    }
}

impl<G, C, P, B> Clone for TranspositionMctsAgent<G, C, P, B>
where
    G: meeple_bots_core::Game,
    MctsAgent<C, P, B>: Clone,
{
    fn clone(&self) -> Self {
        Self::new(
            self.basic.inner.clone(),
            self.basic.tree_reuse_enabled(),
            self.enabled,
        )
    }
}

impl<G, C, P, B> fmt::Debug for TranspositionMctsAgent<G, C, P, B>
where
    G: meeple_bots_core::Game,
    MctsAgent<C, P, B>: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TranspositionMctsAgent")
            .field("inner", &self.basic.inner)
            .field("tree_reuse", &self.basic.tree_reuse_enabled())
            .field("enabled", &self.enabled)
            .field("owner", &self.owner)
            .field("has_graph", &self.graph.is_some())
            .field("pending_reuse_stats", &self.pending_reuse_stats)
            .field("last_reuse_stats", &self.last_reuse_stats)
            .finish()
    }
}

impl<G, C, P, B> Agent<G> for TranspositionMctsAgent<G, C, P, B>
where
    G: DeterministicGame + PerfectInformationGame + TwoPlayerZeroSumGame + Clone + PartialEq,
    G::State: Clone + Eq + Hash,
    G::Action: Clone + PartialEq,
    C: StateEvaluator<G>,
    P: RolloutPolicy<G>,
    B: SelectionBias<G>,
{
    fn on_match_start(&mut self, game: &G, state: &G::State, player: PlayerId) {
        if !self.enabled {
            self.basic.on_match_start(game, state, player);
            return;
        }
        self.basic.inner.decision_budget = DecisionBudget::new();
        let timed = matches!(self.basic.inner.config.budget, SearchBudget::Time(_));
        let started = timed.then(Instant::now);
        self.reset_graph();
        self.owner = Some(player);
        if let Some(start) = started {
            self.basic.inner.decision_budget.pending_maintenance += start.elapsed();
        }
    }

    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, G>,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        if !self.enabled {
            return self.basic.select_action(decision, rng);
        }
        self.basic
            .inner
            .decision_budget
            .begin(self.basic.inner.config.budget);
        let result = (|| {
            self.basic.inner.last_search_stats = None;
            self.basic.inner.last_root_actions.clear();
            self.basic
                .inner
                .config
                .validate()
                .map_err(AgentError::message)?;
            self.basic.inner.config.rollout_policy.validate()?;

            let game = decision.game();
            let root_state = decision.state();
            let root_player = decision.player();
            if game.status(root_state) != PositionStatus::PlayerTurn(root_player) {
                return Err(AgentError::message(
                    "MCTS received a position for the wrong player",
                ));
            }
            if self.owner.is_none() {
                self.owner = Some(root_player);
            }
            if self.owner != Some(root_player) {
                self.pending_reuse_stats.resets += 1;
                self.graph = None;
                self.owner = Some(root_player);
            }

            let graph_matches = self
                .graph
                .as_ref()
                .is_some_and(|graph| graph.game == *game && graph.nodes[0].state == *root_state);
            if self.graph.is_some() && !graph_matches {
                self.record_transition_miss();
            }

            let fresh_graph = self.graph.is_none();
            if fresh_graph {
                let root = GraphNode::new(root_state.clone(), game.legal_actions(root_state));
                if root.unexpanded.is_empty() {
                    return Err(AgentError::NoLegalActions);
                }
                self.graph = Some(ReusableGraph {
                    game: game.clone(),
                    nodes: vec![root],
                    state_indices: HashMap::from([(root_state.clone(), 0)]),
                });
            }

            let mut reuse_stats = std::mem::take(&mut self.pending_reuse_stats);
            if self.basic.tree_reuse_enabled() && !fresh_graph {
                let graph = self.graph.as_ref().expect("graph was initialized");
                reuse_stats.reused_root_visits = graph.nodes[0].visits;
                reuse_stats.reused_nodes = graph.nodes.len() as u64;
            }

            let root_action_indices = if self.basic.inner.root_diagnostics {
                let graph = self.graph.as_ref().expect("graph was initialized");
                match map_graph_root_action_indices(game, root_state, &graph.nodes) {
                    Ok(indices) => Some(indices),
                    Err(error) => {
                        self.graph = None;
                        reuse_stats.resets += 1;
                        self.last_reuse_stats = Some(reuse_stats);
                        return Err(error);
                    }
                }
            } else {
                None
            };

            let graph = self.graph.as_mut().expect("graph was initialized");
            let selected = self.basic.inner.search_graph(
                game,
                root_player,
                graph,
                root_action_indices,
                fresh_graph,
                rng,
            );
            let selected_edge = match selected {
                Ok(index) => index,
                Err(error) => {
                    self.graph = None;
                    reuse_stats.resets += 1;
                    self.last_reuse_stats = Some(reuse_stats);
                    return Err(error);
                }
            };
            let action = graph.nodes[0].edges[selected_edge].action.clone();
            if self.basic.tree_reuse_enabled() {
                self.last_reuse_stats = Some(reuse_stats);
            } else {
                self.graph = None;
                self.last_reuse_stats = None;
            }
            Ok(action)
        })();
        self.basic.inner.decision_budget.finish();
        result
    }

    fn last_decision_stats(&self) -> AgentDecisionStats {
        self.decision_stats()
    }

    fn on_action_applied(
        &mut self,
        game: &G,
        state: &G::State,
        player: PlayerId,
        action: &G::Action,
    ) {
        if !self.enabled {
            self.basic.on_action_applied(game, state, player, action);
            return;
        }
        let timed = matches!(self.basic.inner.config.budget, SearchBudget::Time(_));
        let started = timed.then(Instant::now);
        (|| {
            if !self.basic.tree_reuse_enabled() || self.graph.is_none() {
                return;
            }
            self.pending_reuse_stats.transition_attempts += 1;

            let transition = self.graph.as_mut().and_then(|graph| {
                if graph.game != *game {
                    return None;
                }
                let mut expected_state = graph.nodes[0].state.clone();
                if game.apply_action(&mut expected_state, action).is_err()
                    || expected_state != *state
                {
                    return None;
                }
                let child_index = graph.nodes[0]
                    .edges
                    .iter()
                    .find(|edge| &edge.action == action)
                    .map(|edge| edge.child)?;
                let (retained, pruned) = compact_graph(&mut graph.nodes, child_index);
                graph.nodes[0].state = state.clone();
                graph.state_indices.clear();
                graph.state_indices.extend(
                    graph
                        .nodes
                        .iter()
                        .enumerate()
                        .map(|(index, node)| (node.state.clone(), index)),
                );
                Some((retained, pruned))
            });

            if let Some((retained, pruned)) = transition {
                self.pending_reuse_stats.transition_hits += 1;
                self.pending_reuse_stats.reused_nodes = retained as u64;
                self.pending_reuse_stats.pruned_nodes += pruned as u64;
                if self.owner == Some(player) {
                    self.pending_reuse_stats.own_action_hits += 1;
                } else {
                    self.pending_reuse_stats.opponent_action_hits += 1;
                }
            } else {
                self.record_transition_miss();
            }
        })();
        if let Some(start) = started {
            self.basic.inner.decision_budget.pending_maintenance += start.elapsed();
        }
    }

    fn on_match_end(&mut self, game: &G, state: &G::State) {
        if !self.enabled {
            self.basic.on_match_end(game, state);
            return;
        }
        self.graph = None;
        self.owner = None;
    }
}

fn map_root_action_indices<G>(
    game: &G,
    state: &G::State,
    nodes: &[Node<G::Action>],
) -> Result<RootActionIndices, AgentError>
where
    G: DeterministicGame,
    G::Action: PartialEq,
{
    let legal_actions: Vec<_> = game.legal_actions(state).collect();
    let action_index = |action: &G::Action| {
        legal_actions
            .iter()
            .position(|candidate| candidate == action)
            .and_then(|index| u32::try_from(index).ok())
            .ok_or_else(|| AgentError::message("reused MCTS root does not match legal actions"))
    };
    let unexpanded = nodes[0]
        .unexpanded
        .iter()
        .map(&action_index)
        .collect::<Result<Vec<_>, _>>()?;
    let children = nodes[0]
        .children
        .iter()
        .map(|index| {
            nodes[*index]
                .action
                .as_ref()
                .ok_or_else(|| AgentError::message("reused MCTS child has no action"))
                .and_then(&action_index)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RootActionIndices {
        unexpanded,
        children,
    })
}

fn map_graph_root_action_indices<G>(
    game: &G,
    state: &G::State,
    nodes: &[GraphNode<G::State, G::Action>],
) -> Result<RootActionIndices, AgentError>
where
    G: DeterministicGame,
    G::Action: PartialEq,
{
    let legal_actions: Vec<_> = game.legal_actions(state).collect();
    let action_index = |action: &G::Action| {
        legal_actions
            .iter()
            .position(|candidate| candidate == action)
            .and_then(|index| u32::try_from(index).ok())
            .ok_or_else(|| AgentError::message("reused MCTS root does not match legal actions"))
    };
    let unexpanded = nodes[0]
        .unexpanded
        .iter()
        .map(&action_index)
        .collect::<Result<Vec<_>, _>>()?;
    let children = nodes[0]
        .edges
        .iter()
        .map(|edge| action_index(&edge.action))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RootActionIndices {
        unexpanded,
        children,
    })
}

fn compact_to_subtree<A>(nodes: &mut Vec<Node<A>>, new_root: usize) -> (usize, usize) {
    let old_len = nodes.len();
    let mut reachable = Vec::new();
    let mut stack = vec![new_root];
    while let Some(index) = stack.pop() {
        reachable.push(index);
        stack.extend(nodes[index].children.iter().rev().copied());
    }

    let mut remap = vec![usize::MAX; old_len];
    for (new_index, old_index) in reachable.iter().copied().enumerate() {
        remap[old_index] = new_index;
    }
    let mut old_nodes: Vec<_> = std::mem::take(nodes).into_iter().map(Some).collect();
    nodes.reserve(reachable.len());
    for old_index in reachable.iter().copied() {
        let mut node = old_nodes[old_index]
            .take()
            .expect("reachable node is unique");
        for child in &mut node.children {
            *child = remap[*child];
        }
        nodes.push(node);
    }
    nodes[0].action = None;
    nodes[0].heuristic_value = 0.0;
    (nodes.len(), old_len.saturating_sub(nodes.len()))
}

fn compact_graph<S, A>(nodes: &mut Vec<GraphNode<S, A>>, new_root: usize) -> (usize, usize) {
    let old_len = nodes.len();
    let mut reachable = Vec::new();
    let mut visited = vec![false; old_len];
    let mut stack = vec![new_root];
    while let Some(index) = stack.pop() {
        if visited[index] {
            continue;
        }
        visited[index] = true;
        reachable.push(index);
        stack.extend(nodes[index].edges.iter().rev().map(|edge| edge.child));
    }

    let mut remap = vec![usize::MAX; old_len];
    for (new_index, old_index) in reachable.iter().copied().enumerate() {
        remap[old_index] = new_index;
    }
    let mut old_nodes: Vec<_> = std::mem::take(nodes).into_iter().map(Some).collect();
    nodes.reserve(reachable.len());
    for old_index in reachable.iter().copied() {
        let mut node = old_nodes[old_index]
            .take()
            .expect("reachable graph node is unique");
        for edge in &mut node.edges {
            edge.child = remap[edge.child];
        }
        nodes.push(node);
    }
    (nodes.len(), old_len.saturating_sub(nodes.len()))
}

impl<A> Node<A> {
    fn new<I>(action: Option<A>, heuristic_value: f64, unexpanded: I) -> Self
    where
        I: IntoIterator<Item = A>,
    {
        Self {
            action,
            heuristic_value,
            children: Vec::new(),
            unexpanded: unexpanded.into_iter().collect(),
            visits: 0,
            total_utility: 0.0,
            total_squared_utility: 0.0,
        }
    }

    fn mean_utility(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_utility / f64::from(self.visits)
        }
    }
}

impl<S, A> GraphNode<S, A> {
    fn new<I>(state: S, unexpanded: I) -> Self
    where
        I: IntoIterator<Item = A>,
    {
        Self {
            state,
            edges: Vec::new(),
            unexpanded: unexpanded.into_iter().collect(),
            visits: 0,
            total_utility: 0.0,
        }
    }

    fn mean_utility(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_utility / f64::from(self.visits)
        }
    }
}

impl<G, C, P, B> Agent<G> for MctsAgent<C, P, B>
where
    G: DeterministicGame + PerfectInformationGame + TwoPlayerZeroSumGame,
    G::State: Clone,
    C: StateEvaluator<G>,
    P: RolloutPolicy<G>,
    B: SelectionBias<G>,
{
    fn on_match_start(&mut self, _game: &G, _state: &G::State, _player: PlayerId) {
        self.decision_budget = DecisionBudget::new();
    }

    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, G>,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        self.decision_budget.begin(self.config.budget);
        let result = self.choose_action(decision, rng);
        self.decision_budget.finish();
        result
    }

    fn last_decision_stats(&self) -> AgentDecisionStats {
        self.decision_stats()
    }
}

fn best_child<A>(
    nodes: &[Node<A>],
    parent: usize,
    maximizing: bool,
    exploration: f64,
    selection_policy: SelectionPolicy,
    bias_weight: f64,
) -> usize {
    let parent_visits = f64::from(nodes[parent].visits.max(1));
    nodes[parent]
        .children
        .iter()
        .copied()
        .max_by(|left, right| {
            selection_score(
                &nodes[*left],
                parent_visits,
                maximizing,
                exploration,
                selection_policy,
                bias_weight,
            )
            .total_cmp(&selection_score(
                &nodes[*right],
                parent_visits,
                maximizing,
                exploration,
                selection_policy,
                bias_weight,
            ))
        })
        .expect("parent has children")
}

fn best_graph_edge<S, A>(
    nodes: &[GraphNode<S, A>],
    parent: usize,
    maximizing: bool,
    exploration: f64,
    selection_policy: SelectionPolicy,
    bias_weight: f64,
) -> usize {
    let parent_visits = f64::from(nodes[parent].visits.max(1));
    nodes[parent]
        .edges
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            graph_selection_score(
                nodes,
                left,
                parent_visits,
                maximizing,
                exploration,
                selection_policy,
                bias_weight,
            )
            .total_cmp(&graph_selection_score(
                nodes,
                right,
                parent_visits,
                maximizing,
                exploration,
                selection_policy,
                bias_weight,
            ))
        })
        .map(|(index, _)| index)
        .expect("parent has edges")
}

fn graph_edge_mean<S, A>(
    nodes: &[GraphNode<S, A>],
    edge: &GraphEdge<A>,
    policy: SelectionPolicy,
) -> f64 {
    if policy == SelectionPolicy::Ucb1Tuned {
        if edge.visits == 0 {
            0.0
        } else {
            edge.total_utility / f64::from(edge.visits)
        }
    } else {
        nodes[edge.child].mean_utility()
    }
}

fn graph_selection_score<S, A>(
    nodes: &[GraphNode<S, A>],
    edge: &GraphEdge<A>,
    parent_visits: f64,
    maximizing: bool,
    exploration: f64,
    selection_policy: SelectionPolicy,
    bias_weight: f64,
) -> f64 {
    if edge.visits == 0 {
        return f64::INFINITY;
    }
    if selection_policy == SelectionPolicy::Ucb1Tuned {
        return tuned_score(
            edge.visits,
            edge.total_utility,
            edge.total_squared_utility,
            parent_visits,
            maximizing,
        ) + graph_progressive_bias_term(edge, maximizing, bias_weight);
    }
    let child_mean = nodes[edge.child].mean_utility();
    let exploitation = if maximizing { child_mean } else { -child_mean };
    let exploration_term = exploration * (parent_visits.ln() / f64::from(edge.visits)).sqrt();
    let bias = if bias_weight == 0.0 {
        0.0
    } else {
        graph_progressive_bias_term(edge, maximizing, bias_weight)
    };
    exploitation + exploration_term + bias
}

fn graph_progressive_bias_term<A>(edge: &GraphEdge<A>, maximizing: bool, weight: f64) -> f64 {
    let signed_heuristic = if maximizing {
        edge.heuristic_value
    } else {
        -edge.heuristic_value
    };
    weight * signed_heuristic / (f64::from(edge.visits) + 1.0)
}

fn selection_score<A>(
    node: &Node<A>,
    parent_visits: f64,
    maximizing: bool,
    exploration: f64,
    selection_policy: SelectionPolicy,
    bias_weight: f64,
) -> f64 {
    let uct = match selection_policy {
        SelectionPolicy::Uct => uct_score(node, parent_visits, maximizing, exploration),
        SelectionPolicy::Ucb1Tuned => tuned_score(
            node.visits,
            node.total_utility,
            node.total_squared_utility,
            parent_visits,
            maximizing,
        ),
    };
    if bias_weight == 0.0 {
        uct
    } else {
        uct + progressive_bias_term(node, maximizing, bias_weight)
    }
}

// Equivalent to 2 * UCB1-Tuned([0, 1]) - 1. Keeping scores on the utility
// scale preserves the existing progressive-bias scale. Variance is unchanged
// by reversing the player perspective; converting [-1, 1] to [0, 1] divides it by 4.
fn tuned_score(visits: u32, total: f64, squared: f64, parent_visits: f64, maximizing: bool) -> f64 {
    if visits == 0 {
        return f64::INFINITY;
    }
    let count = f64::from(visits);
    let mean = total / count;
    let variance = ((squared / count - mean * mean) / 4.0).clamp(0.0, 0.25);
    let log_ratio = parent_visits.max(1.0).ln() / count;
    let bound = (variance + (2.0 * log_ratio).sqrt()).min(0.25);
    let signed_mean = if maximizing { mean } else { -mean };
    signed_mean + 2.0 * (log_ratio * bound).sqrt()
}

fn progressive_bias_term<A>(node: &Node<A>, maximizing: bool, weight: f64) -> f64 {
    let signed_heuristic = if maximizing {
        node.heuristic_value
    } else {
        -node.heuristic_value
    };
    weight * signed_heuristic / (f64::from(node.visits) + 1.0)
}

fn uct_score<A>(node: &Node<A>, parent_visits: f64, maximizing: bool, exploration: f64) -> f64 {
    if node.visits == 0 {
        return f64::INFINITY;
    }

    let exploitation = if maximizing {
        node.mean_utility()
    } else {
        -node.mean_utility()
    };
    exploitation + exploration * (parent_visits.ln() / f64::from(node.visits)).sqrt()
}

#[cfg(test)]
fn rollout<G, P, C, R>(
    game: &G,
    state: &mut G::State,
    root_player: PlayerId,
    max_depth: u32,
    policy: &P,
    cutoff_evaluator: &C,
    rng: &mut R,
) -> Result<f64, AgentError>
where
    G: DeterministicGame,
    G::State: Clone,
    P: RolloutPolicy<G>,
    C: StateEvaluator<G>,
    R: RandomSource + ?Sized,
{
    rollout_with_memory(
        game,
        state,
        root_player,
        max_depth,
        policy,
        cutoff_evaluator,
        &mut RolloutMemory::default(),
        rng,
    )
}

// Policy, evaluator and per-search memory have independent ownership and lifetimes.
#[allow(clippy::too_many_arguments)]
fn rollout_with_memory<G, P, C, R>(
    game: &G,
    state: &mut G::State,
    root_player: PlayerId,
    max_depth: u32,
    policy: &P,
    cutoff_evaluator: &C,
    memory: &mut RolloutMemory<G::Action>,
    rng: &mut R,
) -> Result<f64, AgentError>
where
    G: DeterministicGame,
    G::State: Clone,
    P: RolloutPolicy<G>,
    C: StateEvaluator<G>,
    R: RandomSource + ?Sized,
{
    for _ in 0..max_depth {
        match game.status(state) {
            PositionStatus::Terminal => {
                return terminal_utility(game, state, root_player);
            }
            PositionStatus::PlayerTurn(active_player) => {
                let action = policy.select_action_with_memory(
                    game,
                    state,
                    active_player,
                    root_player,
                    memory,
                    rng,
                )?;
                if let Some(recorded) = policy.action_for_learning(&action) {
                    memory.trajectory.push((active_player, recorded));
                }
                game.apply_action(state, &action)
                    .map_err(|error| AgentError::message(error.to_string()))?;
            }
            PositionStatus::Chance => {
                return Err(AgentError::message(
                    "MCTS rollout does not support chance transitions",
                ));
            }
            _ => return Err(AgentError::message("unsupported position status")),
        }
    }

    match game.status(state) {
        PositionStatus::Terminal => terminal_utility(game, state, root_player),
        _ => evaluate_state(game, state, root_player, cutoff_evaluator),
    }
}

fn terminal_utility<G>(game: &G, state: &G::State, root_player: PlayerId) -> Result<f64, AgentError>
where
    G: meeple_bots_core::Game,
{
    let utility = game
        .terminal_utility(state, root_player)
        .map(f64::from)
        .ok_or_else(|| AgentError::message("terminal utility is missing"))?;
    validate_utility(utility, "terminal")
}

fn evaluate_state<G, E>(
    game: &G,
    state: &G::State,
    root_player: PlayerId,
    evaluator: &E,
) -> Result<f64, AgentError>
where
    G: meeple_bots_core::Game,
    E: StateEvaluator<G>,
{
    if game.status(state) == PositionStatus::Terminal {
        return terminal_utility(game, state, root_player);
    }

    let utility = evaluator.evaluate(game, state, root_player)?;
    validate_utility(utility, "heuristic")
}

fn validate_utility(utility: f64, source: &str) -> Result<f64, AgentError> {
    if !utility.is_finite() || !(-1.0..=1.0).contains(&utility) {
        return Err(AgentError::message(format!(
            "MCTS {source} utility must be finite and between -1.0 and 1.0"
        )));
    }
    Ok(utility)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use meeple_bots_core::{Game, IllegalAction};
    use meeple_bots_random_agent::RandomAgent;
    use meeple_bots_simulation::{MatchConfig, SplitMix64, play_match};
    use meeple_bots_tic_tac_toe::{TicTacToe, TicTacToeAction};

    use super::*;

    #[test]
    fn tuned_matches_normalized_formula_and_handles_player_perspective() {
        let n = 1000;
        let parent = 1000.0_f64;
        let mean = 0.4;
        let squared_mean = 0.3;
        let normalized_mean = (mean + 1.0) / 2.0;
        let normalized_second = (squared_mean + 2.0 * mean + 1.0) / 4.0;
        let variance = normalized_second - normalized_mean * normalized_mean;
        let ratio = parent.ln() / f64::from(n);
        let bonus = (ratio * (variance + (2.0 * ratio).sqrt()).min(0.25)).sqrt();
        for maximizing in [false, true] {
            let oriented_mean = if maximizing {
                normalized_mean
            } else {
                1.0 - normalized_mean
            };
            let expected = 2.0 * (oriented_mean + bonus) - 1.0;
            assert!((tuned_score(n, 400.0, 300.0, parent, maximizing) - expected).abs() < 1e-12);
        }
        assert!(tuned_score(n, 0.0, 1000.0, parent, true) > tuned_score(n, 0.0, 0.0, parent, true));
        assert_eq!(tuned_score(0, 0.0, 0.0, 1.0, true), f64::INFINITY);
        assert_eq!(tuned_score(1, 1.0, 1.0, 1.0, false), -1.0);
    }

    #[test]
    fn tuned_graph_statistics_belong_to_the_edge_not_the_shared_child() {
        let nodes = vec![GraphNode::new((), std::iter::empty::<u8>())];
        let edge = GraphEdge {
            action: 0,
            child: 0,
            visits: 1000,
            total_utility: 400.0,
            total_squared_utility: 300.0,
            heuristic_value: 0.0,
        };
        let score = graph_selection_score(
            &nodes,
            &edge,
            1000.0,
            true,
            99.0,
            SelectionPolicy::Ucb1Tuned,
            0.0,
        );
        assert_eq!(score, tuned_score(1000, 400.0, 300.0, 1000.0, true));
        assert_eq!(
            graph_edge_mean(&nodes, &edge, SelectionPolicy::Ucb1Tuned),
            0.4
        );
        assert_eq!(graph_edge_mean(&nodes, &edge, SelectionPolicy::Uct), 0.0);
    }

    #[test]
    fn tuned_backpropagates_moments_and_preserves_them_on_reuse() {
        let game = TicTacToe;
        let state = game.initial_state();
        for graph in [false, true] {
            let inner = MctsAgent::with_cutoff_evaluator(
                MctsConfig {
                    selection_policy: SelectionPolicy::Ucb1Tuned,
                    budget: SearchBudget::Iterations(NonZeroU32::new(1).unwrap()),
                    rollout_depth: 0,
                    ..MctsConfig::default()
                },
                FixedEvaluator(0.25),
            );
            let mut agent = TranspositionMctsAgent::new(inner, true, graph);
            let chosen = agent
                .select_action(
                    DecisionContext::new(&game, &state, PlayerId::FIRST),
                    &mut SplitMix64::new(4),
                )
                .unwrap();
            if graph {
                let edge = &agent.graph.as_ref().unwrap().nodes[0].edges[0];
                assert_eq!(
                    (edge.visits, edge.total_utility, edge.total_squared_utility),
                    (1, 0.25, 0.0625)
                );
            } else {
                let node = &agent.basic.tree.as_ref().unwrap().nodes[1];
                assert_eq!(
                    (node.visits, node.total_utility, node.total_squared_utility),
                    (1, 0.25, 0.0625)
                );
            }
            let mut next = state.clone();
            game.apply_action(&mut next, &chosen).unwrap();
            agent.on_action_applied(&game, &next, PlayerId::FIRST, &chosen);
            if !graph {
                let node = &agent.basic.tree.as_ref().unwrap().nodes[0];
                assert_eq!(node.total_squared_utility, 0.0625);
            }
        }
    }

    #[test]
    fn total_budget_deducts_maintenance_and_reserves_finalization() {
        let mut clock = DecisionBudget::new();
        clock.pending_maintenance = Duration::from_millis(30);
        clock.tail_reserve = Duration::from_millis(10);
        clock.begin(SearchBudget::Time(Duration::from_millis(100)));
        assert_eq!(clock.allowance, Duration::from_millis(60));
        assert_eq!(clock.pending_maintenance, Duration::ZERO);
        // Preparation before the simulation loop counts against the allowance.
        clock.started = Some(Instant::now() - Duration::from_millis(100));
        assert!(clock.exhausted());
        clock.finish_search();
        clock.search_finished = Some(Instant::now() - Duration::from_millis(5));
        clock.finish();
        assert!(clock.tail_reserve >= Duration::from_millis(5));
        assert_eq!(clock.clone().tail_reserve, Duration::ZERO);
    }

    #[test]
    fn maintenance_debt_limits_all_backends_but_not_iteration_budgets() {
        let game = TicTacToe;
        let state = game.initial_state();
        for transpositions in [false, true] {
            for reuse in [false, true] {
                for timed in [false, true] {
                    let mut agent = TranspositionMctsAgent::new(
                        MctsAgent::new(MctsConfig {
                            selection_policy: Default::default(),
                            budget: if timed {
                                SearchBudget::Time(Duration::from_secs(1))
                            } else {
                                SearchBudget::Iterations(NonZeroU32::new(8).unwrap())
                            },
                            ..MctsConfig::default()
                        }),
                        reuse,
                        transpositions,
                    );
                    agent.basic.inner.decision_budget.pending_maintenance = Duration::from_secs(2);
                    let chosen = agent
                        .select_action(
                            DecisionContext::new(&game, &state, PlayerId::FIRST),
                            &mut SplitMix64::new(42),
                        )
                        .unwrap();
                    assert!(game.legal_actions(&state).any(|a| a == chosen));
                    assert_eq!(
                        agent.last_search_stats().unwrap().iterations,
                        if timed { 1 } else { 8 }
                    );
                    let mut next = state.clone();
                    game.apply_action(&mut next, &chosen).unwrap();
                    agent.on_action_applied(&game, &next, PlayerId::FIRST, &chosen);
                    if timed && reuse {
                        assert!(
                            agent.basic.inner.decision_budget.pending_maintenance > Duration::ZERO
                        );
                    }
                    agent.on_match_start(&game, &state, PlayerId::FIRST);
                    assert!(
                        agent.basic.inner.decision_budget.pending_maintenance
                            < Duration::from_secs(2)
                    );
                    assert_eq!(
                        agent.basic.inner.decision_budget.tail_reserve,
                        Duration::ZERO
                    );
                }
            }
        }
    }

    #[test]
    fn mast_averages_credit_each_occurrence_from_the_acting_players_perspective() {
        let mut memory = RolloutMemory::<u8> {
            trajectory: vec![
                (PlayerId::FIRST, 0),
                (PlayerId::SECOND, 0),
                (PlayerId::FIRST, 0),
            ],
            ..Default::default()
        };
        memory.finish_simulation(PlayerId::SECOND, 0.5);
        memory.trajectory.push((PlayerId::FIRST, 0));
        memory.finish_simulation(PlayerId::SECOND, -1.0);
        let first = &memory.values[&(PlayerId::FIRST, 0)];
        assert_eq!(first.visits, 3);
        assert_eq!(first.total, 0.0);
        assert_eq!(memory.values[&(PlayerId::SECOND, 0)].total, 0.5);
        assert!(memory.trajectory.is_empty());
    }

    #[test]
    fn mast_uses_player_averages_and_only_legal_actions() {
        let game = TicTacToe;
        let mut state = game.initial_state();
        let policy = RolloutPolicyConfig::<NeutralEvaluator>::Mast { epsilon: 0.0 };
        let mut memory = RolloutMemory::default();
        for (player, chosen) in [(PlayerId::FIRST, action(0)), (PlayerId::SECOND, action(1))] {
            memory.values.insert(
                (player, chosen),
                MoveAverage {
                    visits: 2,
                    total: 1.0,
                },
            );
            let selected = policy
                .select_action_with_memory(
                    &game,
                    &state,
                    player,
                    PlayerId::FIRST,
                    &memory,
                    &mut SplitMix64::new(42),
                )
                .unwrap();
            assert_eq!(selected, chosen);
        }
        game.apply_action(&mut state, &action(0)).unwrap();
        for seed in 0..16 {
            let selected = policy
                .select_action_with_memory(
                    &game,
                    &state,
                    PlayerId::FIRST,
                    PlayerId::FIRST,
                    &memory,
                    &mut SplitMix64::new(seed),
                )
                .unwrap();
            assert_ne!(selected, action(0));
        }
        // Unseen actions have neutral value and can beat a known losing action.
        memory.values.insert(
            (PlayerId::SECOND, action(1)),
            MoveAverage {
                visits: 1,
                total: -1.0,
            },
        );
        let selected = policy
            .select_action_with_memory(
                &game,
                &state,
                PlayerId::SECOND,
                PlayerId::FIRST,
                &memory,
                &mut SplitMix64::new(42),
            )
            .unwrap();
        assert_ne!(selected, action(1));
    }

    #[test]
    fn mast_epsilon_one_matches_uniform_with_reuse_and_transpositions() {
        let game = TicTacToe;
        for reuse in [false, true] {
            for transpositions in [false, true] {
                for seed in 0..8 {
                    let make_agent = |rollout_policy| {
                        TranspositionMctsAgent::new(
                            MctsAgent::new(MctsConfig {
                                selection_policy: Default::default(),
                                budget: SearchBudget::Iterations(NonZeroU32::new(64).unwrap()),
                                exploration: 1.0,
                                rollout_depth: 9,
                                rollout_policy,
                            })
                            .with_root_diagnostics(true),
                            reuse,
                            transpositions,
                        )
                    };
                    let mut uniform =
                        make_agent(RolloutPolicyConfig::<NeutralEvaluator>::UniformRandom);
                    let mut mast = make_agent(RolloutPolicyConfig::Mast { epsilon: 1.0 });
                    let mut state = game.initial_state();
                    let mut uniform_rng = SplitMix64::new(seed);
                    let mut mast_rng = SplitMix64::new(seed);
                    uniform.on_match_start(&game, &state, PlayerId::FIRST);
                    mast.on_match_start(&game, &state, PlayerId::FIRST);
                    while let PositionStatus::PlayerTurn(player) = game.status(&state) {
                        let chosen = if player == PlayerId::FIRST {
                            let first = uniform
                                .select_action(
                                    DecisionContext::new(&game, &state, player),
                                    &mut uniform_rng,
                                )
                                .unwrap();
                            let second = mast
                                .select_action(
                                    DecisionContext::new(&game, &state, player),
                                    &mut mast_rng,
                                )
                                .unwrap();
                            assert_eq!(first, second);
                            assert_eq!(uniform.last_decision_stats(), mast.last_decision_stats());
                            first
                        } else {
                            game.legal_actions(&state).next().unwrap()
                        };
                        game.apply_action(&mut state, &chosen).unwrap();
                        uniform.on_action_applied(&game, &state, player, &chosen);
                        mast.on_action_applied(&game, &state, player, &chosen);
                    }
                }
            }
        }
    }

    #[test]
    fn mast_restarts_learning_each_decision() {
        let game = TicTacToe;
        let state = game.initial_state();
        let mut agent = MctsAgent::new(MctsConfig {
            selection_policy: Default::default(),
            budget: SearchBudget::Iterations(NonZeroU32::new(128).unwrap()),
            exploration: 1.0,
            rollout_depth: 9,
            rollout_policy: RolloutPolicyConfig::<NeutralEvaluator>::Mast { epsilon: 0.1 },
        })
        .with_root_diagnostics(true);
        let chosen = agent
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut SplitMix64::new(42),
            )
            .unwrap();
        let stats = agent.decision_stats();
        let repeated = agent
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut SplitMix64::new(42),
            )
            .unwrap();
        assert_eq!(chosen, repeated);
        assert_eq!(stats, agent.decision_stats());
    }

    #[test]
    fn mast_records_tree_actions_and_learns_cutoff_values_in_both_search_engines() {
        struct InspectMast {
            calls: std::rc::Rc<Cell<u32>>,
            expected_actions: usize,
        }
        impl RolloutPolicy<TicTacToe> for InspectMast {
            fn select_action<R: RandomSource + ?Sized>(
                &self,
                game: &TicTacToe,
                state: &<TicTacToe as Game>::State,
                active: PlayerId,
                root: PlayerId,
                rng: &mut R,
            ) -> Result<TicTacToeAction, AgentError> {
                UniformRandom.select_action(game, state, active, root, rng)
            }
            fn action_for_learning(&self, action: &TicTacToeAction) -> Option<TicTacToeAction> {
                Some(*action)
            }
            fn finish_simulation(
                &self,
                memory: &mut RolloutMemory<TicTacToeAction>,
                root: PlayerId,
                utility: f64,
            ) {
                assert_eq!(memory.trajectory.len(), self.expected_actions);
                assert_eq!(
                    memory
                        .values
                        .values()
                        .map(|value| value.visits)
                        .sum::<u64>(),
                    u64::from(self.calls.get()) * self.expected_actions as u64
                );
                for (index, (player, _)) in memory.trajectory.iter().enumerate() {
                    assert_eq!(
                        *player,
                        if index % 2 == 0 {
                            PlayerId::FIRST
                        } else {
                            PlayerId::SECOND
                        }
                    );
                }
                assert_eq!(utility, 0.4);
                memory.finish_simulation(root, utility);
                for ((player, _), value) in &memory.values {
                    let expected = if *player == root { 0.4 } else { -0.4 };
                    assert!((value.total / value.visits as f64 - expected).abs() < 1e-12);
                }
                self.calls.set(self.calls.get() + 1);
            }
        }
        for transpositions in [false, true] {
            for rollout_depth in [0, 2] {
                let calls = std::rc::Rc::new(Cell::new(0));
                let mut agent = TranspositionMctsAgent::new(
                    MctsAgent::with_cutoff_evaluator(
                        MctsConfig {
                            selection_policy: Default::default(),
                            budget: SearchBudget::Iterations(NonZeroU32::new(2).unwrap()),
                            exploration: 1.0,
                            rollout_depth,
                            rollout_policy: InspectMast {
                                calls: calls.clone(),
                                expected_actions: 1 + rollout_depth as usize,
                            },
                        },
                        FixedEvaluator(0.4),
                    ),
                    true,
                    transpositions,
                );
                let game = TicTacToe;
                agent
                    .select_action(
                        DecisionContext::new(&game, &game.initial_state(), PlayerId::FIRST),
                        &mut SplitMix64::new(42),
                    )
                    .unwrap();
                assert_eq!(calls.get(), 2);
            }
        }
    }

    #[test]
    fn mast_rejects_invalid_epsilon() {
        for epsilon in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            let policy = RolloutPolicyConfig::<NeutralEvaluator>::Mast { epsilon };
            assert!(<RolloutPolicyConfig as RolloutPolicy<TicTacToe>>::validate(&policy).is_err());
        }
    }

    #[derive(Clone, Copy)]
    struct FixedEvaluator(f64);

    impl StateEvaluator<TicTacToe> for FixedEvaluator {
        fn evaluate(
            &self,
            _game: &TicTacToe,
            _state: &<TicTacToe as Game>::State,
            _root_player: PlayerId,
        ) -> Result<f64, AgentError> {
            Ok(self.0)
        }
    }

    #[derive(Clone, Copy)]
    struct FailingEvaluator;

    impl StateEvaluator<TicTacToe> for FailingEvaluator {
        fn evaluate(
            &self,
            _game: &TicTacToe,
            _state: &<TicTacToe as Game>::State,
            _root_player: PlayerId,
        ) -> Result<f64, AgentError> {
            Err(AgentError::message("evaluator should not be called"))
        }
    }

    #[derive(Clone, Copy)]
    struct FixedTerminalUtilityGame(Option<f32>);

    impl Game for FixedTerminalUtilityGame {
        type State = ();
        type Action = ();
        type Observation<'a> = &'a ();
        type LegalActions<'a> = std::iter::Empty<()>;

        fn player_count(&self) -> u8 {
            2
        }

        fn initial_state(&self) -> Self::State {}

        fn status(&self, _state: &Self::State) -> PositionStatus {
            PositionStatus::Terminal
        }

        fn legal_actions<'a>(&'a self, _state: &'a Self::State) -> Self::LegalActions<'a> {
            std::iter::empty()
        }

        fn apply_action(
            &self,
            _state: &mut Self::State,
            _action: &Self::Action,
        ) -> Result<(), IllegalAction> {
            Err(IllegalAction::new("terminal game has no legal actions"))
        }

        fn observation<'a>(
            &'a self,
            state: &'a Self::State,
            _player: PlayerId,
        ) -> Self::Observation<'a> {
            state
        }

        fn terminal_utility(&self, _state: &Self::State, _player: PlayerId) -> Option<f32> {
            self.0
        }
    }

    impl DeterministicGame for FixedTerminalUtilityGame {}

    #[derive(Debug, Eq, PartialEq)]
    struct NonCloneAction;

    #[derive(Clone, Copy)]
    struct NonCloneActionGame;

    impl Game for NonCloneActionGame {
        type State = bool;
        type Action = NonCloneAction;
        type Observation<'a> = &'a bool;
        type LegalActions<'a> = std::option::IntoIter<NonCloneAction>;

        fn player_count(&self) -> u8 {
            2
        }

        fn initial_state(&self) -> Self::State {
            false
        }

        fn status(&self, state: &Self::State) -> PositionStatus {
            if *state {
                PositionStatus::Terminal
            } else {
                PositionStatus::PlayerTurn(PlayerId::FIRST)
            }
        }

        fn legal_actions<'a>(&'a self, state: &'a Self::State) -> Self::LegalActions<'a> {
            (!*state).then_some(NonCloneAction).into_iter()
        }

        fn apply_action(
            &self,
            state: &mut Self::State,
            _action: &Self::Action,
        ) -> Result<(), IllegalAction> {
            if *state {
                return Err(IllegalAction::new("game is already terminal"));
            }
            *state = true;
            Ok(())
        }

        fn observation<'a>(
            &'a self,
            state: &'a Self::State,
            _player: PlayerId,
        ) -> Self::Observation<'a> {
            state
        }

        fn terminal_utility(&self, state: &Self::State, player: PlayerId) -> Option<f32> {
            state.then_some(if player == PlayerId::FIRST { 1.0 } else { -1.0 })
        }
    }

    impl DeterministicGame for NonCloneActionGame {}
    impl PerfectInformationGame for NonCloneActionGame {}
    impl TwoPlayerZeroSumGame for NonCloneActionGame {}

    #[derive(Clone, Copy)]
    struct FixedCondition(bool);

    impl RolloutCondition<TicTacToe> for FixedCondition {
        fn matches(
            &self,
            _game: &TicTacToe,
            _state: &<TicTacToe as Game>::State,
            _active_player: PlayerId,
            _root_player: PlayerId,
        ) -> bool {
            self.0
        }
    }

    #[derive(Clone, Copy)]
    struct FixedActionPolicy(u8);

    impl RolloutPolicy<TicTacToe> for FixedActionPolicy {
        fn select_action<R: RandomSource + ?Sized>(
            &self,
            _game: &TicTacToe,
            _state: &<TicTacToe as Game>::State,
            _active_player: PlayerId,
            _root_player: PlayerId,
            _rng: &mut R,
        ) -> Result<TicTacToeAction, AgentError> {
            Ok(action(self.0))
        }
    }

    #[derive(Default)]
    struct CountingRolloutPolicy {
        validations: Cell<u32>,
        selections: Cell<u32>,
    }

    impl RolloutPolicy<TicTacToe> for CountingRolloutPolicy {
        fn validate(&self) -> Result<(), AgentError> {
            self.validations.set(self.validations.get() + 1);
            Ok(())
        }

        fn select_action<R: RandomSource + ?Sized>(
            &self,
            game: &TicTacToe,
            state: &<TicTacToe as Game>::State,
            active_player: PlayerId,
            root_player: PlayerId,
            rng: &mut R,
        ) -> Result<TicTacToeAction, AgentError> {
            self.selections.set(self.selections.get() + 1);
            UniformRandom.select_action(game, state, active_player, root_player, rng)
        }
    }

    fn action(index: u8) -> TicTacToeAction {
        TicTacToeAction::from_index(index).unwrap()
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum ChainedAction {
        Risk,
        Draw,
        Win,
        Lose,
    }

    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    enum ChainedState {
        Root,
        Continuation,
        Win,
        Loss,
        Draw,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct ChainedDecisionGame {
        continuation_player: PlayerId,
    }

    impl Game for ChainedDecisionGame {
        type State = ChainedState;
        type Action = ChainedAction;
        type Observation<'a> = &'a ChainedState;
        type LegalActions<'a> = std::vec::IntoIter<ChainedAction>;

        fn player_count(&self) -> u8 {
            2
        }

        fn initial_state(&self) -> Self::State {
            ChainedState::Root
        }

        fn status(&self, state: &Self::State) -> PositionStatus {
            match state {
                ChainedState::Root => PositionStatus::PlayerTurn(PlayerId::FIRST),
                ChainedState::Continuation => PositionStatus::PlayerTurn(self.continuation_player),
                ChainedState::Win | ChainedState::Loss | ChainedState::Draw => {
                    PositionStatus::Terminal
                }
            }
        }

        fn legal_actions<'a>(&'a self, state: &'a Self::State) -> Self::LegalActions<'a> {
            match state {
                ChainedState::Root => vec![ChainedAction::Risk, ChainedAction::Draw],
                ChainedState::Continuation => vec![ChainedAction::Win, ChainedAction::Lose],
                ChainedState::Win | ChainedState::Loss | ChainedState::Draw => Vec::new(),
            }
            .into_iter()
        }

        fn apply_action(
            &self,
            state: &mut Self::State,
            action: &Self::Action,
        ) -> Result<(), IllegalAction> {
            *state = match (*state, *action) {
                (ChainedState::Root, ChainedAction::Risk) => ChainedState::Continuation,
                (ChainedState::Root, ChainedAction::Draw) => ChainedState::Draw,
                (ChainedState::Continuation, ChainedAction::Win) => ChainedState::Win,
                (ChainedState::Continuation, ChainedAction::Lose) => ChainedState::Loss,
                _ => return Err(IllegalAction::new("action is not legal")),
            };
            Ok(())
        }

        fn observation<'a>(
            &'a self,
            state: &'a Self::State,
            _player: PlayerId,
        ) -> Self::Observation<'a> {
            state
        }

        fn terminal_utility(&self, state: &Self::State, player: PlayerId) -> Option<f32> {
            let first_utility = match state {
                ChainedState::Win => 1.0,
                ChainedState::Loss => -1.0,
                ChainedState::Draw => 0.0,
                ChainedState::Root | ChainedState::Continuation => return None,
            };
            Some(if player == PlayerId::FIRST {
                first_utility
            } else {
                -first_utility
            })
        }
    }

    impl DeterministicGame for ChainedDecisionGame {}
    impl PerfectInformationGame for ChainedDecisionGame {}
    impl TwoPlayerZeroSumGame for ChainedDecisionGame {}

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum DiamondAction {
        Left,
        Right,
        Merge,
        Finish,
    }

    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    enum DiamondState {
        Root,
        Left,
        Right,
        Merged,
        Terminal,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct DiamondGame;

    impl Game for DiamondGame {
        type State = DiamondState;
        type Action = DiamondAction;
        type Observation<'a> = &'a DiamondState;
        type LegalActions<'a> = std::vec::IntoIter<DiamondAction>;

        fn player_count(&self) -> u8 {
            2
        }

        fn initial_state(&self) -> Self::State {
            DiamondState::Root
        }

        fn status(&self, state: &Self::State) -> PositionStatus {
            match state {
                DiamondState::Terminal => PositionStatus::Terminal,
                DiamondState::Root
                | DiamondState::Left
                | DiamondState::Right
                | DiamondState::Merged => PositionStatus::PlayerTurn(PlayerId::FIRST),
            }
        }

        fn legal_actions<'a>(&'a self, state: &'a Self::State) -> Self::LegalActions<'a> {
            match state {
                DiamondState::Root => vec![DiamondAction::Left, DiamondAction::Right],
                DiamondState::Left | DiamondState::Right => vec![DiamondAction::Merge],
                DiamondState::Merged => vec![DiamondAction::Finish],
                DiamondState::Terminal => Vec::new(),
            }
            .into_iter()
        }

        fn apply_action(
            &self,
            state: &mut Self::State,
            action: &Self::Action,
        ) -> Result<(), IllegalAction> {
            *state = match (*state, *action) {
                (DiamondState::Root, DiamondAction::Left) => DiamondState::Left,
                (DiamondState::Root, DiamondAction::Right) => DiamondState::Right,
                (DiamondState::Left | DiamondState::Right, DiamondAction::Merge) => {
                    DiamondState::Merged
                }
                (DiamondState::Merged, DiamondAction::Finish) => DiamondState::Terminal,
                _ => return Err(IllegalAction::new("action is not legal")),
            };
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
            (*state == DiamondState::Terminal).then_some(0.0)
        }
    }

    impl DeterministicGame for DiamondGame {}
    impl PerfectInformationGame for DiamondGame {}
    impl TwoPlayerZeroSumGame for DiamondGame {}

    fn select_chained_action(continuation_player: PlayerId, seed: u64) -> ChainedAction {
        let game = ChainedDecisionGame {
            continuation_player,
        };
        let mut agent = MctsAgent::new(MctsConfig {
            selection_policy: Default::default(),
            budget: SearchBudget::Iterations(NonZeroU32::new(2_000).unwrap()),
            exploration: std::f64::consts::SQRT_2,
            rollout_depth: 4,
            rollout_policy: UniformRandom,
        });
        agent
            .select_action(
                DecisionContext::new(&game, &game.initial_state(), PlayerId::FIRST),
                &mut SplitMix64::new(seed),
            )
            .unwrap()
    }

    #[test]
    fn chooses_immediate_win() {
        let game = TicTacToe;
        let mut state = game.initial_state();
        for index in [0, 3, 1, 4] {
            game.apply_action(&mut state, &action(index)).unwrap();
        }

        let mut agent = MctsAgent::default();
        let selected = agent
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut SplitMix64::new(11),
            )
            .unwrap();

        assert_eq!(selected, action(2));
    }

    #[test]
    fn blocks_an_immediate_loss() {
        let game = TicTacToe;
        let mut state = game.initial_state();
        for index in [0, 4, 1] {
            game.apply_action(&mut state, &action(index)).unwrap();
        }

        let mut agent = MctsAgent::default();
        let selected = agent
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::SECOND),
                &mut SplitMix64::new(17),
            )
            .unwrap();

        assert_eq!(selected, action(2));
    }

    #[test]
    fn uses_the_active_player_instead_of_alternating_max_and_min_by_depth() {
        assert_eq!(
            select_chained_action(PlayerId::FIRST, 23),
            ChainedAction::Risk,
            "the root player should maximize again on a consecutive decision"
        );
        assert_eq!(
            select_chained_action(PlayerId::SECOND, 29),
            ChainedAction::Draw,
            "the opponent should minimize after control changes"
        );
    }

    #[test]
    fn supports_actions_that_are_not_cloneable() {
        let game = NonCloneActionGame;
        let mut agent = MctsAgent::new(MctsConfig {
            selection_policy: Default::default(),
            budget: SearchBudget::Iterations(NonZeroU32::new(1).unwrap()),
            exploration: std::f64::consts::SQRT_2,
            rollout_depth: 0,
            rollout_policy: UniformRandom,
        });

        let selected = agent
            .select_action(
                DecisionContext::new(&game, &game.initial_state(), PlayerId::FIRST),
                &mut SplitMix64::new(7),
            )
            .unwrap();

        assert_eq!(selected, NonCloneAction);
    }

    #[test]
    fn has_positive_utility_against_random_over_fixed_seeds() {
        let game = TicTacToe;
        let mut total_utility = 0.0;

        for seed in 0..24 {
            let result = play_match(
                &game,
                &mut MctsAgent::default(),
                &mut RandomAgent,
                MatchConfig {
                    seed,
                    ..MatchConfig::default()
                },
            )
            .unwrap();
            total_utility += result.utilities[PlayerId::FIRST.index()];
        }

        assert!(total_utility > 0.0);
    }
    #[test]
    fn result_is_reproducible() {
        let game = TicTacToe;
        let state = game.initial_state();
        let mut first = MctsAgent::default();
        let mut repeated = MctsAgent::default();

        let a = first
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut SplitMix64::new(99),
            )
            .unwrap();
        let b = repeated
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut SplitMix64::new(99),
            )
            .unwrap();

        assert_eq!(a, b);
        assert_eq!(first.last_search_stats().unwrap().iterations, 1_000);
        assert_eq!(repeated.last_search_stats().unwrap().iterations, 1_000);
    }

    #[test]
    fn time_budget_finishes_at_least_one_complete_iteration() {
        let game = TicTacToe;
        let state = game.initial_state();
        let mut agent = MctsAgent::new(MctsConfig {
            selection_policy: Default::default(),
            budget: SearchBudget::Time(Duration::ZERO),
            exploration: std::f64::consts::SQRT_2,
            rollout_depth: 2,
            rollout_policy: UniformRandom,
        });

        agent
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut SplitMix64::new(99),
            )
            .unwrap();

        let stats = agent.last_search_stats().unwrap();
        assert_eq!(stats.iterations, 1);
        assert!(stats.nodes >= 2);
    }

    #[test]
    fn a_concrete_rollout_policy_can_be_injected_without_runtime_dispatch() {
        let game = TicTacToe;
        let state = game.initial_state();
        let mut agent = MctsAgent::new(MctsConfig {
            selection_policy: Default::default(),
            budget: SearchBudget::Iterations(NonZeroU32::new(4).unwrap()),
            exploration: std::f64::consts::SQRT_2,
            rollout_depth: 2,
            rollout_policy: UniformRandom,
        });

        let selected = agent
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut SplitMix64::new(5),
            )
            .unwrap();

        assert!(game.legal_actions(&state).any(|action| action == selected));
    }

    #[test]
    fn cutoff_uses_the_configured_evaluator() {
        let game = TicTacToe;
        let mut state = game.initial_state();
        let result = rollout(
            &game,
            &mut state,
            PlayerId::FIRST,
            0,
            &UniformRandom,
            &FixedEvaluator(0.25),
            &mut SplitMix64::new(3),
        )
        .unwrap();

        assert_eq!(result, 0.25);
    }

    #[test]
    fn terminal_utility_does_not_call_the_evaluator() {
        let game = TicTacToe;
        let mut state = game.initial_state();
        for index in [0, 3, 1, 4, 2] {
            game.apply_action(&mut state, &action(index)).unwrap();
        }

        let result = rollout(
            &game,
            &mut state,
            PlayerId::FIRST,
            0,
            &UniformRandom,
            &FailingEvaluator,
            &mut SplitMix64::new(3),
        )
        .unwrap();

        assert_eq!(result, 1.0);
    }

    #[test]
    fn rejects_invalid_heuristic_utilities() {
        let game = TicTacToe;
        for invalid in [f64::NAN, f64::INFINITY, -1.1, 1.1] {
            let error = rollout(
                &game,
                &mut game.initial_state(),
                PlayerId::FIRST,
                0,
                &UniformRandom,
                &FixedEvaluator(invalid),
                &mut SplitMix64::new(3),
            )
            .unwrap_err();

            assert!(error.to_string().contains("between -1.0 and 1.0"));
        }
    }

    #[test]
    fn rejects_invalid_terminal_utilities() {
        for invalid in [f32::NAN, f32::INFINITY, -1.1, 1.1] {
            let game = FixedTerminalUtilityGame(Some(invalid));
            let error = rollout(
                &game,
                &mut game.initial_state(),
                PlayerId::FIRST,
                1,
                &UniformRandom,
                &NeutralEvaluator,
                &mut SplitMix64::new(3),
            )
            .unwrap_err();

            assert!(error.to_string().contains("MCTS terminal utility"));
        }
    }

    #[test]
    fn rejects_missing_terminal_utilities() {
        let game = FixedTerminalUtilityGame(None);
        let error = rollout(
            &game,
            &mut game.initial_state(),
            PlayerId::FIRST,
            0,
            &UniformRandom,
            &NeutralEvaluator,
            &mut SplitMix64::new(3),
        )
        .unwrap_err();

        assert_eq!(error.to_string(), "terminal utility is missing");
    }

    #[test]
    fn rejects_invalid_exploration_values() {
        let game = TicTacToe;
        for invalid in [f64::NAN, f64::INFINITY, -0.1] {
            let mut agent = MctsAgent::new(MctsConfig {
                selection_policy: Default::default(),
                budget: SearchBudget::default(),
                exploration: invalid,
                rollout_depth: 1,
                rollout_policy: UniformRandom,
            });
            let error = agent
                .select_action(
                    DecisionContext::new(&game, &game.initial_state(), PlayerId::FIRST),
                    &mut SplitMix64::new(3),
                )
                .unwrap_err();

            assert_eq!(
                error.to_string(),
                "MCTS exploration must be finite and non-negative"
            );
        }
    }

    #[test]
    fn failed_selection_clears_previous_decision_stats() {
        let game = TicTacToe;
        let state = game.initial_state();
        let mut agent = MctsAgent::new(MctsConfig {
            selection_policy: Default::default(),
            budget: SearchBudget::Iterations(NonZeroU32::new(1).unwrap()),
            exploration: std::f64::consts::SQRT_2,
            rollout_depth: 1,
            rollout_policy: UniformRandom,
        });
        let mut rng = SplitMix64::new(3);

        agent
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut rng,
            )
            .unwrap();
        assert!(agent.last_search_stats().is_some());

        agent
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::SECOND),
                &mut rng,
            )
            .unwrap_err();

        assert_eq!(agent.last_search_stats(), None);
        assert_eq!(agent.decision_stats(), AgentDecisionStats::default());
    }

    #[test]
    fn epsilon_greedy_uses_the_active_players_perspective() {
        let policy = EpsilonGreedy::new(0.0, NeutralEvaluator);

        for (active_player, expected) in [
            (PlayerId::FIRST, ChainedAction::Win),
            (PlayerId::SECOND, ChainedAction::Lose),
        ] {
            let game = ChainedDecisionGame {
                continuation_player: active_player,
            };
            let selected = policy
                .select_action(
                    &game,
                    &ChainedState::Continuation,
                    active_player,
                    PlayerId::FIRST,
                    &mut SplitMix64::new(7),
                )
                .unwrap();
            assert_eq!(selected, expected);
        }
    }

    #[test]
    fn epsilon_one_matches_uniform_random_selection() {
        let game = TicTacToe;
        let state = game.initial_state();
        let mut uniform_rng = SplitMix64::new(13);
        let mut epsilon_rng = SplitMix64::new(13);

        let uniform = UniformRandom
            .select_action(
                &game,
                &state,
                PlayerId::FIRST,
                PlayerId::FIRST,
                &mut uniform_rng,
            )
            .unwrap();
        let epsilon = EpsilonGreedy::new(1.0, FixedEvaluator(0.0))
            .select_action(
                &game,
                &state,
                PlayerId::FIRST,
                PlayerId::FIRST,
                &mut epsilon_rng,
            )
            .unwrap();

        assert_eq!(epsilon, uniform);
        assert_eq!(epsilon_rng.next_u64(), uniform_rng.next_u64());
    }

    #[test]
    fn greedy_ties_match_uniform_random_selection() {
        let game = TicTacToe;
        let state = game.initial_state();

        for seed in [0, 13, 42, 99] {
            let mut uniform_rng = SplitMix64::new(seed);
            let mut greedy_rng = SplitMix64::new(seed);
            let uniform = UniformRandom
                .select_action(
                    &game,
                    &state,
                    PlayerId::FIRST,
                    PlayerId::FIRST,
                    &mut uniform_rng,
                )
                .unwrap();
            let greedy = Greedy::new(FixedEvaluator(0.0))
                .select_action(
                    &game,
                    &state,
                    PlayerId::FIRST,
                    PlayerId::FIRST,
                    &mut greedy_rng,
                )
                .unwrap();

            assert_eq!(greedy, uniform, "seed {seed}");
            assert_eq!(greedy_rng.next_u64(), uniform_rng.next_u64(), "seed {seed}");
        }
    }

    #[test]
    fn conditional_rollout_routes_without_consuming_randomness() {
        let game = TicTacToe;
        let state = game.initial_state();

        for (condition, expected) in [(true, action(0)), (false, action(1))] {
            let mut rng = SplitMix64::new(13);
            let mut untouched_rng = SplitMix64::new(13);
            let selected = ConditionalRollout::new(
                FixedCondition(condition),
                FixedActionPolicy(0),
                FixedActionPolicy(1),
            )
            .select_action(&game, &state, PlayerId::FIRST, PlayerId::FIRST, &mut rng)
            .unwrap();

            assert_eq!(selected, expected);
            assert_eq!(rng.next_u64(), untouched_rng.next_u64());
        }
    }

    #[test]
    fn conditional_fallback_does_not_evaluate_the_primary_policy() {
        let game = TicTacToe;
        let state = game.initial_state();
        let mut conditional_rng = SplitMix64::new(31);
        let mut uniform_rng = SplitMix64::new(31);
        let policy = ConditionalRollout::new(
            FixedCondition(false),
            EpsilonGreedy::new(0.0, FailingEvaluator),
            UniformRandom,
        );

        let selected = policy
            .select_action(
                &game,
                &state,
                PlayerId::FIRST,
                PlayerId::FIRST,
                &mut conditional_rng,
            )
            .unwrap();
        let uniform = UniformRandom
            .select_action(
                &game,
                &state,
                PlayerId::FIRST,
                PlayerId::FIRST,
                &mut uniform_rng,
            )
            .unwrap();

        assert_eq!(selected, uniform);
        assert_eq!(conditional_rng.next_u64(), uniform_rng.next_u64());
    }

    #[test]
    fn epsilon_one_matches_uniform_random_mcts_search() {
        let game = TicTacToe;
        let state = game.initial_state();
        let budget = SearchBudget::Iterations(NonZeroU32::new(512).unwrap());

        for seed in [0, 13, 42, 99] {
            let mut uniform = MctsAgent::new(MctsConfig {
                selection_policy: SelectionPolicy::Uct,
                budget,
                exploration: std::f64::consts::SQRT_2,
                rollout_depth: 9,
                rollout_policy: RolloutPolicyConfig::<FailingEvaluator>::UniformRandom,
            });
            let mut epsilon = MctsAgent::new(MctsConfig {
                selection_policy: SelectionPolicy::Uct,
                budget,
                exploration: std::f64::consts::SQRT_2,
                rollout_depth: 9,
                rollout_policy: RolloutPolicyConfig::EpsilonGreedy {
                    epsilon: 1.0,
                    evaluator: FailingEvaluator,
                },
            });
            let mut uniform_rng = SplitMix64::new(seed);
            let mut epsilon_rng = SplitMix64::new(seed);

            let uniform_action = uniform
                .select_action(
                    DecisionContext::new(&game, &state, PlayerId::FIRST),
                    &mut uniform_rng,
                )
                .unwrap();
            let epsilon_action = epsilon
                .select_action(
                    DecisionContext::new(&game, &state, PlayerId::FIRST),
                    &mut epsilon_rng,
                )
                .unwrap();
            let uniform_stats = uniform.last_search_stats().unwrap();
            let epsilon_stats = epsilon.last_search_stats().unwrap();

            assert_eq!(epsilon_action, uniform_action, "seed {seed}");
            assert_eq!(
                epsilon_stats.iterations, uniform_stats.iterations,
                "seed {seed}"
            );
            assert_eq!(epsilon_stats.nodes, uniform_stats.nodes, "seed {seed}");
            assert_eq!(
                epsilon.decision_stats(),
                uniform.decision_stats(),
                "seed {seed}"
            );
            assert_eq!(
                epsilon_rng.next_u64(),
                uniform_rng.next_u64(),
                "seed {seed}"
            );
        }
    }

    #[test]
    fn conditional_epsilon_one_matches_uniform_random_mcts_search() {
        let game = TicTacToe;
        let state = game.initial_state();
        let budget = SearchBudget::Iterations(NonZeroU32::new(512).unwrap());

        for seed in [0, 13, 42, 99] {
            let mut uniform = MctsAgent::new(MctsConfig {
                selection_policy: SelectionPolicy::Uct,
                budget,
                exploration: std::f64::consts::SQRT_2,
                rollout_depth: 9,
                rollout_policy: UniformRandom,
            });
            let mut conditional = MctsAgent::new(MctsConfig {
                selection_policy: SelectionPolicy::Uct,
                budget,
                exploration: std::f64::consts::SQRT_2,
                rollout_depth: 9,
                rollout_policy: ConditionalRollout::new(
                    FixedCondition(true),
                    EpsilonGreedy::new(1.0, FailingEvaluator),
                    UniformRandom,
                ),
            });
            let mut uniform_rng = SplitMix64::new(seed);
            let mut conditional_rng = SplitMix64::new(seed);

            let uniform_action = uniform
                .select_action(
                    DecisionContext::new(&game, &state, PlayerId::FIRST),
                    &mut uniform_rng,
                )
                .unwrap();
            let conditional_action = conditional
                .select_action(
                    DecisionContext::new(&game, &state, PlayerId::FIRST),
                    &mut conditional_rng,
                )
                .unwrap();

            assert_eq!(conditional_action, uniform_action, "seed {seed}");
            assert_eq!(
                conditional.decision_stats(),
                uniform.decision_stats(),
                "seed {seed}"
            );
            assert_eq!(
                conditional_rng.next_u64(),
                uniform_rng.next_u64(),
                "seed {seed}"
            );
        }
    }

    #[test]
    fn zero_progressive_bias_matches_uct_baseline_exactly() {
        let game = TicTacToe;
        let state = game.initial_state();
        let config = MctsConfig {
            selection_policy: Default::default(),
            budget: SearchBudget::Iterations(NonZeroU32::new(256).unwrap()),
            exploration: std::f64::consts::SQRT_2,
            rollout_depth: 9,
            rollout_policy: UniformRandom,
        };
        for seed in [0, 17, 42] {
            let mut baseline = MctsAgent::new(config).with_root_diagnostics(true);
            let mut biased = MctsAgent::with_progressive_bias(
                config,
                NeutralEvaluator,
                ProgressiveBias::new(0.0, FixedCondition(true), FailingEvaluator),
            )
            .with_root_diagnostics(true);
            let mut baseline_rng = SplitMix64::new(seed);
            let mut biased_rng = SplitMix64::new(seed);

            let baseline_action = baseline
                .select_action(
                    DecisionContext::new(&game, &state, PlayerId::FIRST),
                    &mut baseline_rng,
                )
                .unwrap();
            let biased_action = biased
                .select_action(
                    DecisionContext::new(&game, &state, PlayerId::FIRST),
                    &mut biased_rng,
                )
                .unwrap();

            assert_eq!(biased_action, baseline_action);
            assert_eq!(biased.decision_stats(), baseline.decision_stats());
            assert_eq!(biased_rng.next_u64(), baseline_rng.next_u64());
        }
    }

    #[test]
    fn root_diagnostics_preserve_original_action_indices() {
        let game = TicTacToe;
        let state = game.initial_state();
        let mut agent = MctsAgent::new(MctsConfig {
            selection_policy: Default::default(),
            budget: SearchBudget::Iterations(NonZeroU32::new(9).unwrap()),
            exploration: std::f64::consts::SQRT_2,
            rollout_depth: 1,
            rollout_policy: UniformRandom,
        })
        .with_root_diagnostics(true);

        let selected = agent
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut SplitMix64::new(5),
            )
            .unwrap();
        let stats = agent.decision_stats();
        let selected_stats = stats
            .root_actions
            .iter()
            .find(|action| action.selected)
            .expect("one root action is selected");
        let mut action_indices: Vec<_> = stats
            .root_actions
            .iter()
            .map(|action| action.action_index)
            .collect();
        action_indices.sort_unstable();

        assert_eq!(action_indices, (0..9).collect::<Vec<_>>());
        assert_eq!(selected, action(selected_stats.action_index as u8));
    }

    #[test]
    fn disabled_root_diagnostics_do_not_report_action_stats() {
        let game = TicTacToe;
        let mut agent = MctsAgent::new(MctsConfig {
            selection_policy: Default::default(),
            budget: SearchBudget::Iterations(NonZeroU32::new(9).unwrap()),
            exploration: std::f64::consts::SQRT_2,
            rollout_depth: 1,
            rollout_policy: UniformRandom,
        });

        agent
            .select_action(
                DecisionContext::new(&game, &game.initial_state(), PlayerId::FIRST),
                &mut SplitMix64::new(5),
            )
            .unwrap();

        assert!(agent.decision_stats().root_actions.is_empty());
    }

    #[test]
    fn progressive_bias_favors_better_equal_statistics_and_decays() {
        let mut lower = Node::new(Some(0_u8), -0.5, Vec::new());
        lower.visits = 4;
        let mut higher = Node::new(Some(1_u8), 0.5, Vec::new());
        higher.visits = 4;
        assert!(
            selection_score(&higher, 20.0, true, 0.0, SelectionPolicy::Uct, 1.0)
                > selection_score(&lower, 20.0, true, 0.0, SelectionPolicy::Uct, 1.0)
        );

        let early = progressive_bias_term(&higher, true, 1.0);
        higher.visits = 40;
        let late = progressive_bias_term(&higher, true, 1.0);
        assert!(late.abs() < early.abs());
    }

    #[test]
    fn conditional_progressive_bias_skips_evaluator_outside_condition() {
        let game = TicTacToe;
        let state = game.initial_state();
        let config = MctsConfig {
            selection_policy: Default::default(),
            budget: SearchBudget::Iterations(NonZeroU32::new(32).unwrap()),
            exploration: std::f64::consts::SQRT_2,
            rollout_depth: 2,
            rollout_policy: UniformRandom,
        };
        let mut agent = MctsAgent::with_progressive_bias(
            config,
            NeutralEvaluator,
            ProgressiveBias::new(1.0, FixedCondition(false), FailingEvaluator),
        );

        agent
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut SplitMix64::new(9),
            )
            .unwrap();
    }

    #[test]
    fn progressive_bias_sign_follows_active_player_perspective() {
        let mut parent = Node::new(None, 0.0, Vec::<u8>::new());
        parent.visits = 20;
        parent.children = vec![1, 2];
        let mut lower = Node::new(Some(0), -0.5, Vec::new());
        lower.visits = 4;
        let mut higher = Node::new(Some(1), 0.5, Vec::new());
        higher.visits = 4;
        let nodes = vec![parent, lower, higher];

        assert_eq!(
            best_child(&nodes, 0, true, 0.0, SelectionPolicy::Uct, 1.0),
            2
        );
        assert_eq!(
            best_child(&nodes, 0, false, 0.0, SelectionPolicy::Uct, 1.0),
            1
        );
    }

    #[test]
    fn rejects_invalid_rollout_epsilon_before_simulation() {
        let game = TicTacToe;
        for epsilon in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            let mut agent = MctsAgent::new(MctsConfig {
                selection_policy: Default::default(),
                budget: SearchBudget::Iterations(NonZeroU32::new(1).unwrap()),
                exploration: std::f64::consts::SQRT_2,
                rollout_depth: 0,
                rollout_policy: EpsilonGreedy::new(epsilon, FixedEvaluator(0.0)),
            });
            let error = agent
                .select_action(
                    DecisionContext::new(&game, &game.initial_state(), PlayerId::FIRST),
                    &mut SplitMix64::new(3),
                )
                .unwrap_err();

            assert!(error.to_string().contains("between 0.0 and 1.0"));
        }
    }

    #[test]
    fn conditional_rollout_validates_both_branches() {
        let game = TicTacToe;
        let mut agent = MctsAgent::new(MctsConfig {
            selection_policy: Default::default(),
            budget: SearchBudget::Iterations(NonZeroU32::new(1).unwrap()),
            exploration: std::f64::consts::SQRT_2,
            rollout_depth: 0,
            rollout_policy: ConditionalRollout::new(
                FixedCondition(false),
                EpsilonGreedy::new(f64::NAN, FixedEvaluator(0.0)),
                UniformRandom,
            ),
        });

        let error = agent
            .select_action(
                DecisionContext::new(&game, &game.initial_state(), PlayerId::FIRST),
                &mut SplitMix64::new(3),
            )
            .unwrap_err();

        assert!(error.to_string().contains("between 0.0 and 1.0"));
    }

    #[test]
    fn rollout_policy_is_validated_once_per_search() {
        let game = TicTacToe;
        let mut agent = MctsAgent::new(MctsConfig {
            selection_policy: Default::default(),
            budget: SearchBudget::Iterations(NonZeroU32::new(32).unwrap()),
            exploration: std::f64::consts::SQRT_2,
            rollout_depth: 2,
            rollout_policy: CountingRolloutPolicy::default(),
        });

        agent
            .select_action(
                DecisionContext::new(&game, &game.initial_state(), PlayerId::FIRST),
                &mut SplitMix64::new(3),
            )
            .unwrap();

        assert_eq!(agent.config.rollout_policy.validations.get(), 1);
        assert!(agent.config.rollout_policy.selections.get() > 1);
    }

    #[test]
    fn disabled_tree_reuse_matches_the_baseline_search() {
        let game = TicTacToe;
        let state = game.initial_state();
        let config = MctsConfig {
            selection_policy: Default::default(),
            budget: SearchBudget::Iterations(NonZeroU32::new(64).unwrap()),
            exploration: std::f64::consts::SQRT_2,
            rollout_depth: 4,
            rollout_policy: RolloutPolicyConfig::<NeutralEvaluator>::UniformRandom,
        };
        let mut baseline = MctsAgent::new(config);
        let mut wrapped = TreeReuseMctsAgent::<TicTacToe>::new(MctsAgent::new(config), false);
        let mut baseline_rng = SplitMix64::new(91);
        let mut wrapped_rng = SplitMix64::new(91);

        let baseline_action = baseline
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut baseline_rng,
            )
            .unwrap();
        let wrapped_action = wrapped
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut wrapped_rng,
            )
            .unwrap();

        assert_eq!(wrapped_action, baseline_action);
        assert_eq!(wrapped.decision_stats(), baseline.decision_stats());
        assert!(wrapped.tree.is_none());
    }

    #[test]
    fn disabled_transpositions_match_the_baseline_search() {
        let game = TicTacToe;
        let state = game.initial_state();
        let config = MctsConfig {
            selection_policy: Default::default(),
            budget: SearchBudget::Iterations(NonZeroU32::new(64).unwrap()),
            exploration: std::f64::consts::SQRT_2,
            rollout_depth: 4,
            rollout_policy: RolloutPolicyConfig::<NeutralEvaluator>::UniformRandom,
        };
        let mut baseline = MctsAgent::new(config);
        let mut wrapped =
            TranspositionMctsAgent::<TicTacToe>::new(MctsAgent::new(config), false, false);
        let mut baseline_rng = SplitMix64::new(91);
        let mut wrapped_rng = SplitMix64::new(91);

        let baseline_action = baseline
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut baseline_rng,
            )
            .unwrap();
        let wrapped_action = wrapped
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut wrapped_rng,
            )
            .unwrap();

        assert_eq!(wrapped_action, baseline_action);
        assert_eq!(wrapped.decision_stats(), baseline.decision_stats());
        assert!(wrapped.graph.is_none());
    }

    #[test]
    fn merges_transpositions_and_reuses_the_resulting_graph() {
        let game = DiamondGame;
        let mut state = game.initial_state();
        let config = MctsConfig {
            selection_policy: Default::default(),
            budget: SearchBudget::Iterations(NonZeroU32::new(4).unwrap()),
            exploration: std::f64::consts::SQRT_2,
            rollout_depth: 4,
            rollout_policy: UniformRandom,
        };
        let mut agent = TranspositionMctsAgent::new(MctsAgent::new(config), true, true);
        let mut rng = SplitMix64::new(17);
        agent.on_match_start(&game, &state, PlayerId::FIRST);

        let selected = agent
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut rng,
            )
            .unwrap();

        let graph = agent.graph.as_ref().unwrap();
        assert_eq!(graph.nodes.len(), 4);
        assert_eq!(agent.decision_stats().search_nodes, Some(4));
        let branch_children: Vec<_> = graph.nodes[0].edges.iter().map(|edge| edge.child).collect();
        let merged_children: Vec<_> = branch_children
            .iter()
            .map(|branch| graph.nodes[*branch].edges[0].child)
            .collect();
        assert_eq!(merged_children.len(), 2);
        assert_eq!(merged_children[0], merged_children[1]);

        game.apply_action(&mut state, &selected).unwrap();
        agent.on_action_applied(&game, &state, PlayerId::FIRST, &selected);
        agent
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut rng,
            )
            .unwrap();
        let reuse = agent.decision_stats().tree_reuse.unwrap();
        assert_eq!(reuse.transition_attempts, 1);
        assert_eq!(reuse.transition_hits, 1);
        assert!(reuse.reused_root_visits > 0);
    }

    #[test]
    fn reuses_tree_across_consecutive_decisions_by_the_same_player() {
        let game = ChainedDecisionGame {
            continuation_player: PlayerId::FIRST,
        };
        let mut state = game.initial_state();
        let config = MctsConfig {
            selection_policy: Default::default(),
            budget: SearchBudget::Iterations(NonZeroU32::new(128).unwrap()),
            exploration: std::f64::consts::SQRT_2,
            rollout_depth: 4,
            rollout_policy: UniformRandom,
        };
        let inner = MctsAgent::new(config).with_root_diagnostics(true);
        let mut agent = TreeReuseMctsAgent::new(inner, true);
        let mut rng = SplitMix64::new(7);
        agent.on_match_start(&game, &state, PlayerId::FIRST);

        let first = agent
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut rng,
            )
            .unwrap();
        assert_eq!(first, ChainedAction::Risk);
        game.apply_action(&mut state, &first).unwrap();
        agent.on_action_applied(&game, &state, PlayerId::FIRST, &first);

        let second = agent
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut rng,
            )
            .unwrap();
        let reuse = agent.decision_stats().tree_reuse.unwrap();

        assert_eq!(second, ChainedAction::Win);
        assert_eq!(reuse.transition_attempts, 1);
        assert_eq!(reuse.transition_hits, 1);
        assert_eq!(reuse.own_action_hits, 1);
        assert!(reuse.reused_root_visits > 0);
        assert!(reuse.reused_nodes > 0);
        assert_eq!(agent.decision_stats().root_actions.len(), 2);
    }

    #[test]
    fn follows_expanded_opponent_actions_before_reusing_the_tree() {
        let game = TicTacToe;
        let mut state = game.initial_state();
        let config = MctsConfig {
            selection_policy: Default::default(),
            budget: SearchBudget::Iterations(NonZeroU32::new(512).unwrap()),
            exploration: std::f64::consts::SQRT_2,
            rollout_depth: 9,
            rollout_policy: RolloutPolicyConfig::<NeutralEvaluator>::UniformRandom,
        };
        let mut agent = TreeReuseMctsAgent::<TicTacToe>::new(MctsAgent::new(config), true);
        let mut rng = SplitMix64::new(12);
        agent.on_match_start(&game, &state, PlayerId::FIRST);

        let own_action = agent
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut rng,
            )
            .unwrap();
        game.apply_action(&mut state, &own_action).unwrap();
        agent.on_action_applied(&game, &state, PlayerId::FIRST, &own_action);

        let opponent_action = agent.tree.as_ref().unwrap().nodes[0].children[0];
        let opponent_action = agent.tree.as_ref().unwrap().nodes[opponent_action]
            .action
            .unwrap();
        game.apply_action(&mut state, &opponent_action).unwrap();
        agent.on_action_applied(&game, &state, PlayerId::SECOND, &opponent_action);

        agent
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut rng,
            )
            .unwrap();
        let reuse = agent.decision_stats().tree_reuse.unwrap();
        assert_eq!(reuse.transition_attempts, 2);
        assert_eq!(reuse.transition_hits, 2);
        assert_eq!(reuse.own_action_hits, 1);
        assert_eq!(reuse.opponent_action_hits, 1);
        assert!(reuse.reused_root_visits > 0);
    }

    #[test]
    fn resets_when_an_observed_action_was_not_expanded() {
        let game = TicTacToe;
        let mut state = game.initial_state();
        let config = MctsConfig {
            selection_policy: Default::default(),
            budget: SearchBudget::Iterations(NonZeroU32::new(1).unwrap()),
            exploration: std::f64::consts::SQRT_2,
            rollout_depth: 1,
            rollout_policy: RolloutPolicyConfig::<NeutralEvaluator>::UniformRandom,
        };
        let mut agent = TreeReuseMctsAgent::<TicTacToe>::new(MctsAgent::new(config), true);
        agent.on_match_start(&game, &state, PlayerId::FIRST);
        agent
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut SplitMix64::new(3),
            )
            .unwrap();
        let expanded = agent.tree.as_ref().unwrap().nodes[0].children[0];
        let expanded = agent.tree.as_ref().unwrap().nodes[expanded].action.unwrap();
        let unexpanded = game
            .legal_actions(&state)
            .find(|action| *action != expanded)
            .unwrap();
        game.apply_action(&mut state, &unexpanded).unwrap();

        agent.on_action_applied(&game, &state, PlayerId::FIRST, &unexpanded);

        assert!(agent.tree.is_none());
        assert_eq!(agent.pending_reuse_stats.transition_attempts, 1);
        assert_eq!(agent.pending_reuse_stats.transition_hits, 0);
        assert_eq!(agent.pending_reuse_stats.transition_misses, 1);
        assert_eq!(agent.pending_reuse_stats.resets, 1);
    }

    #[test]
    fn subtree_compaction_remaps_children_and_drops_unreachable_nodes() {
        let mut root = Node::new(None, 0.0, Vec::<u8>::new());
        root.children = vec![1, 2];
        let mut retained = Node::new(Some(10), 0.5, Vec::new());
        retained.children = vec![3];
        let discarded = Node::new(Some(20), -0.5, Vec::new());
        let leaf = Node::new(Some(30), 0.25, Vec::new());
        let mut nodes = vec![root, retained, discarded, leaf];

        let (retained_count, pruned_count) = compact_to_subtree(&mut nodes, 1);

        assert_eq!((retained_count, pruned_count), (2, 2));
        assert_eq!(nodes[0].action, None);
        assert_eq!(nodes[0].children, vec![1]);
        assert_eq!(nodes[1].action, Some(30));
    }

    #[test]
    fn starting_a_new_match_discards_the_retained_tree() {
        let game = TicTacToe;
        let state = game.initial_state();
        let mut agent = TreeReuseMctsAgent::<TicTacToe>::new(MctsAgent::default(), true);
        agent
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut SplitMix64::new(5),
            )
            .unwrap();
        assert!(agent.tree.is_some());

        agent.on_match_start(&game, &state, PlayerId::SECOND);

        assert!(agent.tree.is_none());
        assert_eq!(agent.owner, Some(PlayerId::SECOND));
        assert_eq!(agent.pending_reuse_stats, TreeReuseStats::default());
    }
}
