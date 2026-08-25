//! Monte Carlo Tree Search for deterministic, perfect-information games.

use std::{cmp::Ordering, num::NonZeroU32};

use meeple_bots_core::{
    Agent, AgentError, DecisionContext, DeterministicGame, HeuristicGame, PerfectInformationGame,
    PlayerId, PositionStatus, RandomSource, TwoPlayerZeroSumGame,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MctsConfig<P = RolloutPolicyConfig<NeutralEvaluator>> {
    pub iterations: NonZeroU32,
    pub exploration: f64,
    pub rollout_depth: u32,
    pub rollout_policy: P,
}

impl Default for MctsConfig<RolloutPolicyConfig<NeutralEvaluator>> {
    fn default() -> Self {
        Self {
            iterations: NonZeroU32::new(1_000).expect("constant is non-zero"),
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
    G::Action: Clone,
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

#[derive(Clone, Copy, Debug, Default)]
pub struct UniformRandom;

impl<G> RolloutPolicy<G> for UniformRandom
where
    G: DeterministicGame,
    G::State: Clone,
    G::Action: Clone,
{
    fn select_action<R: RandomSource + ?Sized>(
        &self,
        game: &G,
        state: &G::State,
        _active_player: PlayerId,
        _root_player: PlayerId,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        let actions: Vec<_> = game.legal_actions(state).collect();
        let index = rng.index(actions.len()).ok_or(AgentError::NoLegalActions)?;
        Ok(actions[index].clone())
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
    G::Action: Clone,
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
    G::Action: Clone,
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
        if !self.epsilon.is_finite() || !(0.0..=1.0).contains(&self.epsilon) {
            return Err(AgentError::message(
                "MCTS rollout epsilon must be finite and between 0.0 and 1.0",
            ));
        }

        if self.epsilon == 1.0 || (self.epsilon > 0.0 && rng.unit_f64() < self.epsilon) {
            return UniformRandom.select_action(game, state, active_player, root_player, rng);
        }

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

impl<G, E> RolloutPolicy<G> for RolloutPolicyConfig<E>
where
    G: DeterministicGame,
    G::State: Clone,
    G::Action: Clone,
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
            Self::EpsilonGreedy { epsilon, evaluator } => {
                if !epsilon.is_finite() || !(0.0..=1.0).contains(epsilon) {
                    return Err(AgentError::message(
                        "MCTS rollout epsilon must be finite and between 0.0 and 1.0",
                    ));
                }
                if *epsilon == 1.0 || (*epsilon > 0.0 && rng.unit_f64() < *epsilon) {
                    UniformRandom.select_action(game, state, active_player, root_player, rng)
                } else {
                    select_greedy_action(game, state, active_player, root_player, evaluator, rng)
                }
            }
        }
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
    G::Action: Clone,
    E: StateEvaluator<G>,
    R: RandomSource + ?Sized,
{
    let actions: Vec<_> = game.legal_actions(state).collect();
    if actions.is_empty() {
        return Err(AgentError::NoLegalActions);
    }
    let maximizing = active_player == root_player;
    let mut best_score = None;
    let mut best_indices = Vec::new();
    for (index, action) in actions.iter().enumerate() {
        let mut successor = state.clone();
        game.apply_action(&mut successor, action)
            .map_err(|error| AgentError::message(error.to_string()))?;
        let score = evaluate_state(game, &successor, root_player, evaluator)?;
        let ordering = best_score.map(|best: f64| score.total_cmp(&best));
        let is_better = matches!(
            (maximizing, ordering),
            (true, Some(Ordering::Greater)) | (false, Some(Ordering::Less))
        );
        if best_score.is_none() || is_better {
            best_score = Some(score);
            best_indices.clear();
            best_indices.push(index);
        } else if ordering == Some(Ordering::Equal) {
            best_indices.push(index);
        }
    }

    let tie_index = rng
        .index(best_indices.len())
        .expect("at least one action has the best score");
    Ok(actions[best_indices[tie_index]].clone())
}

#[derive(Clone, Copy, Debug)]
pub struct MctsAgent<C = NeutralEvaluator, P = RolloutPolicyConfig<NeutralEvaluator>> {
    pub config: MctsConfig<P>,
    pub cutoff_evaluator: C,
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
        }
    }
}

impl<C, P> MctsAgent<C, P> {
    pub const fn with_cutoff_evaluator(config: MctsConfig<P>, cutoff_evaluator: C) -> Self {
        Self {
            config,
            cutoff_evaluator,
        }
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
        R: RandomSource + ?Sized,
    {
        let game = decision.game();
        let root_state = decision.state();
        let root_player = decision.player();

        if game.status(root_state) != PositionStatus::PlayerTurn(root_player) {
            return Err(AgentError::message(
                "MCTS received a position for the wrong player",
            ));
        }

        let root_actions: Vec<_> = game.legal_actions(root_state).collect();
        if root_actions.is_empty() {
            return Err(AgentError::NoLegalActions);
        }
        let mut nodes = vec![Node::new(None, root_actions)];

        for _ in 0..self.config.iterations.get() {
            let mut state = root_state.clone();
            let mut node_index = 0;
            let mut path = vec![0];

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
                    let action_index = rng
                        .index(nodes[node_index].unexpanded.len())
                        .expect("non-empty actions");
                    let action = nodes[node_index].unexpanded.swap_remove(action_index);
                    game.apply_action(&mut state, &action)
                        .map_err(|error| AgentError::message(error.to_string()))?;

                    let child_actions =
                        if matches!(game.status(&state), PositionStatus::PlayerTurn(_)) {
                            game.legal_actions(&state).collect()
                        } else {
                            Vec::new()
                        };
                    let child_index = nodes.len();
                    nodes.push(Node::new(Some(action), child_actions));
                    nodes[node_index].children.push(child_index);
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
                let selected = best_child(&nodes, node_index, maximizing, self.config.exploration);
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
            for visited in path {
                nodes[visited].visits += 1;
                nodes[visited].total_utility += utility;
            }
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
        nodes[selected_index]
            .action
            .clone()
            .ok_or(AgentError::NoLegalActions)
    }
}

struct Node<A> {
    action: Option<A>,
    children: Vec<usize>,
    unexpanded: Vec<A>,
    visits: u32,
    total_utility: f64,
}

impl<A> Node<A> {
    fn new(action: Option<A>, unexpanded: Vec<A>) -> Self {
        Self {
            action,
            children: Vec::new(),
            unexpanded,
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

impl<G, C, P> Agent<G> for MctsAgent<C, P>
where
    G: DeterministicGame + PerfectInformationGame + TwoPlayerZeroSumGame,
    G::State: Clone,
    G::Action: Clone,
    C: StateEvaluator<G>,
    P: RolloutPolicy<G>,
{
    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, G>,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        self.choose_action(decision, rng)
    }
}

fn best_child<A>(nodes: &[Node<A>], parent: usize, maximizing: bool, exploration: f64) -> usize {
    let parent_visits = f64::from(nodes[parent].visits.max(1));
    nodes[parent]
        .children
        .iter()
        .copied()
        .max_by(|left, right| {
            uct_score(&nodes[*left], parent_visits, maximizing, exploration).total_cmp(&uct_score(
                &nodes[*right],
                parent_visits,
                maximizing,
                exploration,
            ))
        })
        .expect("parent has children")
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
    G::Action: Clone,
    P: RolloutPolicy<G>,
    C: StateEvaluator<G>,
    R: RandomSource + ?Sized,
{
    for _ in 0..max_depth {
        match game.status(state) {
            PositionStatus::Terminal => {
                return game
                    .terminal_utility(state, root_player)
                    .map(f64::from)
                    .ok_or_else(|| AgentError::message("terminal utility is missing"));
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
        PositionStatus::Terminal => game
            .terminal_utility(state, root_player)
            .map(f64::from)
            .ok_or_else(|| AgentError::message("terminal utility is missing")),
        _ => evaluate_state(game, state, root_player, cutoff_evaluator),
    }
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
    let utility = match game.status(state) {
        PositionStatus::Terminal => game
            .terminal_utility(state, root_player)
            .map(f64::from)
            .ok_or_else(|| AgentError::message("terminal utility is missing"))?,
        _ => evaluator.evaluate(game, state, root_player)?,
    };
    if !utility.is_finite() || !(-1.0..=1.0).contains(&utility) {
        return Err(AgentError::message(
            "MCTS heuristic utility must be finite and between -1.0 and 1.0",
        ));
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
            iterations: NonZeroU32::new(2_000).unwrap(),
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
    }

    #[test]
    fn a_concrete_rollout_policy_can_be_injected_without_runtime_dispatch() {
        let game = TicTacToe;
        let state = game.initial_state();
        let mut agent = MctsAgent::new(MctsConfig {
            iterations: NonZeroU32::new(4).unwrap(),
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
