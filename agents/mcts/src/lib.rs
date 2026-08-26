//! Monte Carlo Tree Search for deterministic, perfect-information games.

use std::{
    cmp::Ordering,
    num::NonZeroU32,
    time::{Duration, Instant},
};

use meeple_bots_core::{
    Agent, AgentDecisionStats, AgentError, DecisionContext, DeterministicGame, HeuristicGame,
    PerfectInformationGame, PlayerId, PositionStatus, RandomSource, RootActionStats,
    TwoPlayerZeroSumGame,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MctsConfig<P = RolloutPolicyConfig<NeutralEvaluator>> {
    pub budget: SearchBudget,
    pub exploration: f64,
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
}

pub trait StateEvaluator<G: DeterministicGame> {
    fn evaluate(
        &self,
        game: &G,
        state: &G::State,
        root_player: PlayerId,
    ) -> Result<f64, AgentError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NeutralEvaluator;

impl<G: DeterministicGame> StateEvaluator<G> for NeutralEvaluator {
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
    G: DeterministicGame + HeuristicGame,
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

pub trait RolloutPolicy<G>
where
    G: DeterministicGame,
    G::State: Clone,
{
    fn select_action<R: RandomSource + ?Sized>(
        &self,
        game: &G,
        state: &G::State,
        active_player: PlayerId,
        root_player: PlayerId,
        rng: &mut R,
    ) -> Result<G::Action, AgentError>;
}

pub trait PolicyCondition<G>
where
    G: DeterministicGame,
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

impl<G: DeterministicGame> PolicyCondition<G> for Always {
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

pub trait SelectionBias<G>
where
    G: DeterministicGame,
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

impl<G: DeterministicGame> SelectionBias<G> for NoSelectionBias {
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
    G: DeterministicGame,
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
    G: DeterministicGame,
    G::State: Clone,
    C: PolicyCondition<G>,
    P: RolloutPolicy<G>,
    F: RolloutPolicy<G>,
{
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
    G: DeterministicGame,
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
    G: DeterministicGame,
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
    G: DeterministicGame,
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
    G: DeterministicGame,
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
        match self {
            Self::UniformRandom => {
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
    G: DeterministicGame,
    G::State: Clone,
    E: StateEvaluator<G>,
    R: RandomSource + ?Sized,
{
    if !epsilon.is_finite() || !(0.0..=1.0).contains(&epsilon) {
        return Err(AgentError::message(
            "MCTS rollout epsilon must be finite and between 0.0 and 1.0",
        ));
    }

    if epsilon == 1.0 || (epsilon > 0.0 && rng.unit_f64() < epsilon) {
        UniformRandom.select_action(game, state, active_player, root_player, rng)
    } else {
        select_greedy_action(game, state, active_player, root_player, evaluator, rng)
    }
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
    G: DeterministicGame,
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
        G::Action: Clone,
        C: StateEvaluator<G>,
        P: RolloutPolicy<G>,
        B: SelectionBias<G>,
        R: RandomSource + ?Sized,
    {
        self.last_search_stats = None;
        self.last_root_actions.clear();

        self.config.validate().map_err(AgentError::message)?;

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
        let root_action_count = u32::try_from(root.unexpanded.len())
            .map_err(|_| AgentError::message("MCTS root has too many legal actions"))?;
        let mut root_unexpanded_action_indices: Vec<_> = (0..root_action_count).collect();
        let mut root_child_action_indices = Vec::with_capacity(root.unexpanded.len());
        let bias_weight = self.selection_bias.weight();
        if !bias_weight.is_finite() || bias_weight < 0.0 {
            return Err(AgentError::message(
                "MCTS progressive bias weight must be finite and non-negative",
            ));
        }
        let bias_enabled = bias_weight > 0.0;
        let mut nodes = vec![root];

        let search_started = Instant::now();
        let mut completed_iterations = 0_u64;
        let mut path = Vec::new();
        loop {
            let budget_exhausted = match self.config.budget {
                SearchBudget::Iterations(iterations) => {
                    completed_iterations >= u64::from(iterations.get())
                }
                SearchBudget::Time(duration) => {
                    completed_iterations > 0 && search_started.elapsed() >= duration
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
                        .then(|| root_unexpanded_action_indices.swap_remove(unexpanded_slot));
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
                        root_child_action_indices.push(action_index);
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
                    &nodes,
                    node_index,
                    maximizing,
                    self.config.exploration,
                    bias_weight,
                );
                let action = nodes[selected]
                    .action
                    .as_ref()
                    .expect("child has an action")
                    .clone();
                game.apply_action(&mut state, &action)
                    .map_err(|error| AgentError::message(error.to_string()))?;
                node_index = selected;
                path.push(node_index);
            }

            let utility = rollout(
                game,
                &mut state,
                root_player,
                self.config.rollout_depth,
                &self.config.rollout_policy,
                &self.cutoff_evaluator,
                rng,
            )?;
            for &visited in &path {
                nodes[visited].visits += 1;
                nodes[visited].total_utility += utility;
            }
            completed_iterations += 1;
        }

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
            nodes[0]
                .children
                .iter()
                .copied()
                .zip(root_child_action_indices.iter().copied())
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
            nodes: nodes.len() as u64,
            elapsed: search_started.elapsed(),
        });

        nodes[selected_index]
            .action
            .clone()
            .ok_or(AgentError::NoLegalActions)
    }
}

struct Node<A> {
    action: Option<A>,
    heuristic_value: f64,
    children: Vec<usize>,
    unexpanded: Vec<A>,
    visits: u32,
    total_utility: f64,
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
    G::Action: Clone,
    C: StateEvaluator<G>,
    P: RolloutPolicy<G>,
    B: SelectionBias<G>,
{
    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, G>,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        self.choose_action(decision, rng)
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
                bias_weight,
            )
            .total_cmp(&selection_score(
                &nodes[*right],
                parent_visits,
                maximizing,
                exploration,
                bias_weight,
            ))
        })
        .expect("parent has children")
}

fn selection_score<A>(
    node: &Node<A>,
    parent_visits: f64,
    maximizing: bool,
    exploration: f64,
    bias_weight: f64,
) -> f64 {
    let uct = uct_score(node, parent_visits, maximizing, exploration);
    if bias_weight == 0.0 {
        uct
    } else {
        uct + progressive_bias_term(node, maximizing, bias_weight)
    }
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
    for _ in 0..max_depth {
        match game.status(state) {
            PositionStatus::Terminal => {
                return terminal_utility(game, state, root_player);
            }
            PositionStatus::PlayerTurn(active_player) => {
                let action = policy.select_action(game, state, active_player, root_player, rng)?;
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
    G: DeterministicGame,
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
    G: DeterministicGame,
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
    use meeple_bots_core::{Game, IllegalAction};
    use meeple_bots_random_agent::RandomAgent;
    use meeple_bots_simulation::{MatchConfig, SplitMix64, play_match};
    use meeple_bots_tic_tac_toe::{TicTacToe, TicTacToeAction};

    use super::*;

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

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum ChainedState {
        Root,
        Continuation,
        Win,
        Loss,
        Draw,
    }

    #[derive(Clone, Copy)]
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

    fn select_chained_action(continuation_player: PlayerId, seed: u64) -> ChainedAction {
        let game = ChainedDecisionGame {
            continuation_player,
        };
        let mut agent = MctsAgent::new(MctsConfig {
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
                budget,
                exploration: std::f64::consts::SQRT_2,
                rollout_depth: 9,
                rollout_policy: RolloutPolicyConfig::<FailingEvaluator>::UniformRandom,
            });
            let mut epsilon = MctsAgent::new(MctsConfig {
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
                budget,
                exploration: std::f64::consts::SQRT_2,
                rollout_depth: 9,
                rollout_policy: UniformRandom,
            });
            let mut conditional = MctsAgent::new(MctsConfig {
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
    fn progressive_bias_favors_better_equal_statistics_and_decays() {
        let mut lower = Node::new(Some(0_u8), -0.5, Vec::new());
        lower.visits = 4;
        let mut higher = Node::new(Some(1_u8), 0.5, Vec::new());
        higher.visits = 4;
        assert!(
            selection_score(&higher, 20.0, true, 0.0, 1.0)
                > selection_score(&lower, 20.0, true, 0.0, 1.0)
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

        assert_eq!(best_child(&nodes, 0, true, 0.0, 1.0), 2);
        assert_eq!(best_child(&nodes, 0, false, 0.0, 1.0), 1);
    }

    #[test]
    fn rejects_invalid_rollout_epsilon() {
        let game = TicTacToe;
        for epsilon in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            let error = EpsilonGreedy::new(epsilon, FixedEvaluator(0.0))
                .select_action(
                    &game,
                    &game.initial_state(),
                    PlayerId::FIRST,
                    PlayerId::FIRST,
                    &mut SplitMix64::new(3),
                )
                .unwrap_err();

            assert!(error.to_string().contains("between 0.0 and 1.0"));
        }
    }
}
