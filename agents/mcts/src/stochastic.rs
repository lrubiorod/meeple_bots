//! MCTS with sampled public chance outcomes. Decision edges aggregate all outcomes;
//! chance probabilities are sampled by the game and never optimized with UCB.
use super::{
    DecisionBudget, MctsConfig, NeutralEvaluator, NoSelectionBias, RolloutMemory, RolloutPolicy,
    SearchBudget, SelectionBias, SelectionPolicy, StateEvaluator, UniformRandom, terminal_utility,
    tuned_score, validate_utility,
};
use meeple_bots_core::{
    Agent, AgentDecisionStats, AgentError, DecisionContext, Game, PerfectInformationGame, PlayerId,
    PositionStatus, RandomSource, RootActionStats, TwoPlayerZeroSumGame,
};
use std::{collections::HashMap, hash::Hash};

#[derive(Clone, Debug)]
pub struct StochasticMctsAgent<C = NeutralEvaluator, P = UniformRandom, B = NoSelectionBias> {
    pub config: MctsConfig<P>,
    pub cutoff_evaluator: C,
    pub selection_bias: B,
    pub root_diagnostics: bool,
    pub(super) stats: AgentDecisionStats,
    pub(super) budget: DecisionBudget,
}

pub(super) struct Edge<S, A> {
    pub(super) action: A,
    pub(super) visits: u32,
    total: f64,
    squared: f64,
    pub(super) outcomes: HashMap<S, usize>,
    heuristic_total: f64,
    heuristic_samples: u32,
}
pub(super) struct Node<S, A> {
    pub(super) visits: u32,
    pub(super) state: S,
    pub(super) edges: Vec<Edge<S, A>>,
}

impl<C, P> StochasticMctsAgent<C, P> {
    pub fn new(config: MctsConfig<P>, cutoff_evaluator: C) -> Self {
        Self::with_progressive_bias(config, cutoff_evaluator, NoSelectionBias)
    }
}
impl<C, P, B> StochasticMctsAgent<C, P, B> {
    pub fn with_progressive_bias(
        config: MctsConfig<P>,
        cutoff_evaluator: C,
        selection_bias: B,
    ) -> Self {
        Self {
            config,
            cutoff_evaluator,
            selection_bias,
            root_diagnostics: false,
            stats: AgentDecisionStats::default(),
            budget: DecisionBudget::new(),
        }
    }
}
fn edge_bias<S, A>(edge: &Edge<S, A>, weight: f64, maximizing: bool) -> f64 {
    if edge.heuristic_samples == 0 {
        return 0.0;
    }
    let mean = edge.heuristic_total / f64::from(edge.heuristic_samples);
    weight * (if maximizing { mean } else { -mean }) / (f64::from(edge.visits) + 1.0)
}

pub(super) fn resolve_chance<G: Game, R: RandomSource + ?Sized>(
    game: &G,
    state: &mut G::State,
    rng: &mut R,
) -> Result<(), AgentError> {
    // Bound pathological chains of chance-only transitions.
    for _ in 0..10_000 {
        if game.status(state) != PositionStatus::Chance {
            return Ok(());
        }
        let event = game
            .sample_chance(state, rng)
            .map_err(|e| AgentError::message(e.to_string()))?;
        game.apply_chance_outcome(state, &event)
            .map_err(|e| AgentError::message(e.to_string()))?;
    }
    Err(AgentError::message("too many consecutive chance events"))
}

fn node<G: Game>(game: &G, state: &G::State) -> Node<G::State, G::Action>
where
    G::State: Clone,
{
    Node {
        state: state.clone(),
        visits: 0,
        edges: game
            .legal_actions(state)
            .map(|action| Edge {
                action,
                visits: 0,
                total: 0.0,
                squared: 0.0,
                outcomes: HashMap::new(),
                heuristic_total: 0.0,
                heuristic_samples: 0,
            })
            .collect(),
    }
}

impl<G, C, P, B> Agent<G> for StochasticMctsAgent<C, P, B>
where
    G: Game + PerfectInformationGame + TwoPlayerZeroSumGame,
    G::State: Clone + Eq + Hash,
    G::Action: Clone,
    C: StateEvaluator<G>,
    P: RolloutPolicy<G>,
    B: SelectionBias<G>,
{
    fn on_match_start(&mut self, _: &G, _: &G::State, _: PlayerId) {
        self.budget = DecisionBudget::new();
    }
    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, G>,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        let mut nodes = Vec::new();
        let result = self.search(decision, rng, &mut nodes, false);
        // Include destruction of the temporary tree in the finalization reserve.
        drop(nodes);
        self.budget.finish();
        result
    }

    fn last_decision_stats(&self) -> AgentDecisionStats {
        self.stats.clone()
    }
}

impl<C, P, B> StochasticMctsAgent<C, P, B> {
    pub(super) fn search<G, R>(
        &mut self,
        decision: DecisionContext<'_, G>,
        rng: &mut R,
        nodes: &mut Vec<Node<G::State, G::Action>>,
        transpositions: bool,
    ) -> Result<G::Action, AgentError>
    where
        G: Game + PerfectInformationGame + TwoPlayerZeroSumGame,
        G::State: Clone + Eq + Hash,
        G::Action: Clone,
        C: StateEvaluator<G>,
        P: RolloutPolicy<G>,
        B: SelectionBias<G>,
        R: RandomSource + ?Sized,
    {
        self.budget.begin(self.config.budget);
        self.stats = AgentDecisionStats::default();
        self.config.validate().map_err(AgentError::message)?;
        if self.config.progressive_widening.is_some() {
            return Err(AgentError::message(
                "Progressive Widening is only supported by deterministic MCTS",
            ));
        }
        if matches!(
            self.config.selection_policy,
            SelectionPolicy::UctRave { .. }
        ) {
            return Err(AgentError::message(
                "UCT-RAVE is only supported by deterministic MCTS",
            ));
        }
        self.config.rollout_policy.validate()?;
        let bias_weight = self.selection_bias.weight();
        if !bias_weight.is_finite() || bias_weight < 0.0 {
            return Err(AgentError::message(
                "progressive bias weight must be finite and non-negative",
            ));
        }
        let game = decision.game();
        let root = decision.state();
        let owner = decision.player();
        if game.status(root) != PositionStatus::PlayerTurn(owner) {
            return Err(AgentError::message("expected a player decision"));
        }
        if nodes.is_empty() {
            nodes.push(node(game, root));
        }
        if nodes[0].edges.is_empty() {
            return Err(AgentError::NoLegalActions);
        }
        let mut states: HashMap<G::State, usize> = HashMap::new();
        if transpositions {
            states.extend(nodes.iter().enumerate().map(|(i, n)| (n.state.clone(), i)));
        }
        let mut rollout_memory = RolloutMemory::default();
        let mut iterations = 0;
        let mut terminal_simulations = 0_u64;
        loop {
            let exhausted = match self.config.budget {
                SearchBudget::Iterations(n) => iterations >= n.get(),
                SearchBudget::Time(_) => iterations > 0 && self.budget.exhausted(),
            };
            if exhausted {
                break;
            }
            rollout_memory.trajectory.clear();
            let mut state = root.clone();
            let mut index = 0;
            let mut path = Vec::new();
            let mut depth = 0;
            while let PositionStatus::PlayerTurn(active) = game.status(&state) {
                if depth >= self.config.rollout_depth.max(1) {
                    break;
                }
                let n = &nodes[index];
                if n.edges.is_empty() {
                    return Err(AgentError::NoLegalActions);
                }
                let unvisited: Vec<_> = n
                    .edges
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.visits == 0)
                    .map(|(i, _)| i)
                    .collect();
                let chosen = if let Some(i) = rng.index(unvisited.len()) {
                    unvisited[i]
                } else {
                    let score = |e: &Edge<G::State, G::Action>| {
                        let base = if self.config.selection_policy == SelectionPolicy::Ucb1Tuned {
                            tuned_score(
                                e.visits,
                                e.total,
                                e.squared,
                                f64::from(n.visits),
                                active == owner,
                            )
                        } else {
                            let mean = e.total / f64::from(e.visits);
                            (if active == owner { mean } else { -mean })
                                + self.config.exploration
                                    * (f64::from(n.visits.max(1)).ln() / f64::from(e.visits)).sqrt()
                        };
                        base + edge_bias(e, bias_weight, active == owner)
                    };
                    n.edges
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| score(a).total_cmp(&score(b)))
                        .unwrap()
                        .0
                };
                let bias_applies =
                    bias_weight > 0.0 && self.selection_bias.applies(game, &state, active, owner);
                let action = &nodes[index].edges[chosen].action;
                if let Some(recorded) = self.config.rollout_policy.action_for_learning(action) {
                    rollout_memory.trajectory.push((active, recorded));
                }
                game.apply_action(&mut state, action)
                    .map_err(|e| AgentError::message(e.to_string()))?;
                depth += 1;
                path.push((index, chosen));
                resolve_chance(game, &mut state, rng)?;
                if bias_applies {
                    let value = if game.status(&state) == PositionStatus::Terminal {
                        terminal_utility(game, &state, owner)?
                    } else {
                        validate_utility(
                            self.selection_bias.evaluate_child(game, &state, owner)?,
                            "bias",
                        )?
                    };
                    let edge = &mut nodes[index].edges[chosen];
                    edge.heuristic_total += value;
                    edge.heuristic_samples += 1;
                }
                if game.status(&state) == PositionStatus::Terminal {
                    break;
                }
                let existing = nodes[index].edges[chosen]
                    .outcomes
                    .get(&state)
                    .copied()
                    .or_else(|| {
                        if transpositions {
                            states.get(&state).copied()
                        } else {
                            None
                        }
                    });
                if let Some(child) = existing {
                    nodes[index].edges[chosen]
                        .outcomes
                        .insert(state.clone(), child);
                    // A repeated node ends tree traversal. The rollout still obeys
                    // the remaining player-action horizon, without double credit.
                    if path.iter().any(|(visited, _)| *visited == child) {
                        break;
                    }
                    index = child;
                } else {
                    let child = nodes.len();
                    nodes.push(node(game, &state));
                    nodes[index].edges[chosen]
                        .outcomes
                        .insert(state.clone(), child);
                    if transpositions {
                        states.insert(state.clone(), child);
                    }
                    break;
                }
            }
            // Policies see player decisions only; depth excludes sampled chance events.
            loop {
                resolve_chance(game, &mut state, rng)?;
                if super::rollout_cutoff_reached(game, &state, depth, self.config.rollout_depth) {
                    break;
                }
                let active = match game.status(&state) {
                    PositionStatus::Terminal => break,
                    PositionStatus::PlayerTurn(active) => active,
                    _ => return Err(AgentError::message("expected a player decision")),
                };
                let action = self.config.rollout_policy.select_action_with_memory(
                    game,
                    &state,
                    active,
                    owner,
                    &rollout_memory,
                    rng,
                )?;
                if let Some(recorded) = self.config.rollout_policy.action_for_learning(&action) {
                    rollout_memory.trajectory.push((active, recorded));
                }
                game.apply_action(&mut state, &action)
                    .map_err(|e| AgentError::message(e.to_string()))?;
                depth = depth.saturating_add(1);
            }
            resolve_chance(game, &mut state, rng)?;
            terminal_simulations += u64::from(game.status(&state) == PositionStatus::Terminal);
            let utility = if game.status(&state) == PositionStatus::Terminal {
                f64::from(
                    game.terminal_utility(&state, owner)
                        .ok_or_else(|| AgentError::message("missing terminal utility"))?,
                )
            } else {
                self.cutoff_evaluator.evaluate(game, &state, owner)?
            };
            if !utility.is_finite() || !(-1.0..=1.0).contains(&utility) {
                return Err(AgentError::message("utility must be finite and in [-1, 1]"));
            }
            self.config
                .rollout_policy
                .finish_simulation(&mut rollout_memory, owner, utility);
            for (i, e) in path {
                nodes[i].visits += 1;
                let edge = &mut nodes[i].edges[e];
                edge.visits += 1;
                edge.total += utility;
                edge.squared += utility * utility;
            }
            iterations += 1;
        }
        self.budget.finish_search();
        let chosen = nodes[0]
            .edges
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.visits.cmp(&b.visits).then_with(|| {
                    (a.total / f64::from(a.visits.max(1)))
                        .total_cmp(&(b.total / f64::from(b.visits.max(1))))
                })
            })
            .unwrap()
            .0;
        self.stats = AgentDecisionStats {
            root_expansion: None,
            search_iterations: Some(u64::from(iterations)),
            search_nodes: Some(nodes.len() as u64),
            terminal_simulations: Some(terminal_simulations),
            cutoff_simulations: Some(u64::from(iterations) - terminal_simulations),
            root_actions: if self.root_diagnostics {
                nodes[0]
                    .edges
                    .iter()
                    .enumerate()
                    .map(|(i, e)| RootActionStats {
                        action_index: i as u32,
                        visits: e.visits,
                        mean_utility: e.total / f64::from(e.visits.max(1)),
                        heuristic_value: (e.heuristic_samples > 0)
                            .then(|| e.heuristic_total / f64::from(e.heuristic_samples)),
                        progressive_bias: (e.heuristic_samples > 0)
                            .then(|| edge_bias(e, bias_weight, true)),
                        selected: i == chosen,
                    })
                    .collect()
            } else {
                Vec::new()
            },
            tree_reuse: None,
        };
        Ok(nodes[0].edges[chosen].action.clone())
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::{Always, ConditionalRollout, NoSelectionBias, PolicyCondition, SelectionBias};
    use meeple_bots_core::IllegalAction;
    use meeple_bots_simulation::SplitMix64;
    use std::cell::Cell;
    use std::num::NonZeroU32;

    #[test]
    fn temporary_tree_cleanup_is_reserved_for_next_timed_decision() {
        use std::{
            rc::Rc,
            time::{Duration, Instant},
        };

        #[derive(Clone)]
        struct Action(Rc<Cell<Duration>>);
        impl Drop for Action {
            fn drop(&mut self) {
                let started = Instant::now();
                std::thread::sleep(Duration::from_millis(10));
                self.0.set(started.elapsed());
            }
        }
        struct CleanupGame(Rc<Cell<Duration>>);
        impl Game for CleanupGame {
            type State = bool;
            type Action = Action;
            type Observation<'a> = bool;
            type LegalActions<'a> = std::option::IntoIter<Action>;
            fn player_count(&self) -> u8 {
                2
            }
            fn initial_state(&self) -> bool {
                false
            }
            fn status(&self, finished: &bool) -> PositionStatus {
                if *finished {
                    PositionStatus::Terminal
                } else {
                    PositionStatus::PlayerTurn(PlayerId::FIRST)
                }
            }
            fn legal_actions(&self, finished: &bool) -> Self::LegalActions<'_> {
                (!finished).then(|| Action(self.0.clone())).into_iter()
            }
            fn apply_action(&self, finished: &mut bool, _: &Action) -> Result<(), IllegalAction> {
                *finished = true;
                Ok(())
            }
            fn observation(&self, finished: &bool, _: PlayerId) -> bool {
                *finished
            }
            fn terminal_utility(&self, finished: &bool, _: PlayerId) -> Option<f32> {
                finished.then_some(0.0)
            }
        }
        impl PerfectInformationGame for CleanupGame {}
        impl TwoPlayerZeroSumGame for CleanupGame {}

        let cleanup = Rc::new(Cell::new(Duration::ZERO));
        let game = CleanupGame(cleanup.clone());
        let mut agent = StochasticMctsAgent::new(
            MctsConfig {
                progressive_widening: None,
                budget: SearchBudget::Time(Duration::ZERO),
                exploration: 1.4,
                selection_policy: SelectionPolicy::Uct,
                rollout_depth: 1,
                rollout_policy: UniformRandom,
            },
            NeutralEvaluator,
        );
        // Keep the returned action alive so only the tree's action has been dropped.
        let _action = agent
            .select_action(
                DecisionContext::new(&game, &false, PlayerId::FIRST),
                &mut SplitMix64::new(42),
            )
            .unwrap();
        assert!(cleanup.get() >= Duration::from_millis(10));
        assert!(
            agent.budget.tail_reserve >= cleanup.get(),
            "tree destruction must be inside finalization"
        );
        agent.budget.begin(SearchBudget::Time(cleanup.get()));
        assert_eq!(agent.budget.allowance, Duration::ZERO);
    }

    // Three player decisions, with a chance chain after the first and a chance
    // event between two consecutive decisions by the second player.
    #[derive(Clone, PartialEq)]
    pub(crate) struct ChanceChain;
    impl Game for ChanceChain {
        type State = u8;
        type Action = u8;
        type Observation<'a> = u8;
        type LegalActions<'a> = std::vec::IntoIter<u8>;

        fn player_count(&self) -> u8 {
            2
        }
        fn initial_state(&self) -> u8 {
            0
        }
        fn status(&self, state: &u8) -> PositionStatus {
            match state {
                0 => PositionStatus::PlayerTurn(PlayerId::FIRST),
                1 | 2 | 4 => PositionStatus::Chance,
                3 | 5 => PositionStatus::PlayerTurn(PlayerId::SECOND),
                6 => PositionStatus::Terminal,
                _ => panic!("invalid test state"),
            }
        }
        fn legal_actions(&self, state: &u8) -> Self::LegalActions<'_> {
            match self.status(state) {
                PositionStatus::PlayerTurn(_) => vec![10, 11],
                _ => vec![],
            }
            .into_iter()
        }
        fn apply_action(&self, state: &mut u8, action: &u8) -> Result<(), IllegalAction> {
            let legal = match self.status(state) {
                PositionStatus::PlayerTurn(_) => matches!(action, 10 | 11),
                PositionStatus::Chance => matches!(action, 100 | 101),
                _ => false,
            };
            if !legal {
                return Err(IllegalAction::new("invalid test action"));
            }
            *state += 1;
            Ok(())
        }
        fn sample_chance<R: RandomSource + ?Sized>(
            &self,
            state: &u8,
            rng: &mut R,
        ) -> Result<u8, IllegalAction> {
            assert_eq!(self.status(state), PositionStatus::Chance);
            Ok(100 + rng.index(2).unwrap() as u8)
        }
        fn observation(&self, state: &u8, _: PlayerId) -> u8 {
            *state
        }
        fn terminal_utility(&self, state: &u8, player: PlayerId) -> Option<f32> {
            (*state == 6).then_some(if player == PlayerId::FIRST { 1.0 } else { -1.0 })
        }
    }
    impl PerfectInformationGame for ChanceChain {}
    impl TwoPlayerZeroSumGame for ChanceChain {}

    #[derive(Default)]
    struct InspectPolicy {
        validations: Cell<u32>,
        selections: Cell<u32>,
        completions: Cell<u32>,
        reject: bool,
    }
    impl RolloutPolicy<ChanceChain> for InspectPolicy {
        fn validate(&self) -> Result<(), AgentError> {
            self.validations.set(self.validations.get() + 1);
            if self.reject {
                return Err(AgentError::message("rejected test policy"));
            }
            Ok(())
        }
        fn select_action<R: RandomSource + ?Sized>(
            &self,
            game: &ChanceChain,
            state: &u8,
            active: PlayerId,
            root: PlayerId,
            rng: &mut R,
        ) -> Result<u8, AgentError> {
            assert_eq!(game.status(state), PositionStatus::PlayerTurn(active));
            assert_eq!(active, PlayerId::SECOND);
            assert_eq!(root, PlayerId::FIRST);
            self.selections.set(self.selections.get() + 1);
            // Preserve the original inline uniform rollout's draw and action order.
            let actions: Vec<_> = game.legal_actions(state).collect();
            Ok(actions[rng.index(actions.len()).unwrap()])
        }
        fn action_for_learning(&self, action: &u8) -> Option<u8> {
            Some(*action)
        }
        fn finish_simulation(&self, memory: &mut RolloutMemory<u8>, root: PlayerId, utility: f64) {
            assert_eq!(root, PlayerId::FIRST);
            assert_eq!(utility, 1.0);
            assert_eq!(memory.trajectory.len(), 3);
            for ((player, action), expected) in
                memory
                    .trajectory
                    .iter()
                    .zip([PlayerId::FIRST, PlayerId::SECOND, PlayerId::SECOND])
            {
                assert_eq!(*player, expected);
                assert!(
                    matches!(action, 10 | 11),
                    "chance must not enter policy memory"
                );
            }
            self.completions.set(self.completions.get() + 1);
            // Leave the trajectory intact: the search must delimit simulations.
        }
    }

    fn chain_agent<P>(
        policy: P,
        selection_policy: SelectionPolicy,
    ) -> StochasticMctsAgent<NeutralEvaluator, P> {
        let mut agent = StochasticMctsAgent::new(
            MctsConfig {
                progressive_widening: None,
                budget: SearchBudget::Iterations(NonZeroU32::new(16).unwrap()),
                exploration: 1.4,
                selection_policy,
                rollout_depth: 3,
                rollout_policy: policy,
            },
            NeutralEvaluator,
        );
        agent.root_diagnostics = true;
        agent
    }

    #[test]
    fn generic_policy_preserves_uniform_search_and_rng_through_chance_chains() {
        for selection in [SelectionPolicy::Uct, SelectionPolicy::Ucb1Tuned] {
            for seed in [0, 11, 42] {
                let mut uniform = chain_agent(UniformRandom, selection);
                let mut custom = chain_agent(InspectPolicy::default(), selection);
                let mut uniform_rng = SplitMix64::new(seed);
                let mut custom_rng = SplitMix64::new(seed);
                // Repeat decisions to verify that policy memory does not survive a search.
                for _ in 0..2 {
                    assert_eq!(
                        uniform
                            .select_action(
                                DecisionContext::new(&ChanceChain, &0, PlayerId::FIRST),
                                &mut uniform_rng
                            )
                            .unwrap(),
                        custom
                            .select_action(
                                DecisionContext::new(&ChanceChain, &0, PlayerId::FIRST),
                                &mut custom_rng
                            )
                            .unwrap(),
                    );
                    assert_eq!(uniform.stats, custom.stats);
                    assert_eq!(uniform_rng.next_u64(), custom_rng.next_u64());
                }
                assert!(custom.config.rollout_policy.selections.get() > 0);
                assert_eq!(custom.config.rollout_policy.validations.get(), 2);
                assert_eq!(custom.config.rollout_policy.completions.get(), 32);
            }
        }
    }

    #[test]
    fn invalid_policy_fails_before_simulation_or_rng_consumption() {
        let mut agent = chain_agent(
            InspectPolicy {
                reject: true,
                ..Default::default()
            },
            SelectionPolicy::Uct,
        );
        let mut rng = SplitMix64::new(42);
        assert!(
            agent
                .select_action(
                    DecisionContext::new(&ChanceChain, &0, PlayerId::FIRST),
                    &mut rng
                )
                .is_err()
        );
        assert_eq!(rng.next_u64(), SplitMix64::new(42).next_u64());
        assert_eq!(agent.config.rollout_policy.selections.get(), 0);
        assert_eq!(agent.config.rollout_policy.completions.get(), 0);
    }

    #[test]
    fn shared_conditions_and_no_bias_support_public_chance_games() {
        assert!(Always.matches(&ChanceChain, &0, PlayerId::FIRST, PlayerId::FIRST));
        assert!(!NoSelectionBias.applies(&ChanceChain, &0, PlayerId::FIRST, PlayerId::FIRST));
        let policy = ConditionalRollout::new(Always, UniformRandom, UniformRandom);
        let mut conditional = chain_agent(policy, SelectionPolicy::Uct);
        let mut uniform = chain_agent(UniformRandom, SelectionPolicy::Uct);
        let decision = || DecisionContext::new(&ChanceChain, &0, PlayerId::FIRST);
        assert_eq!(
            conditional
                .select_action(decision(), &mut SplitMix64::new(42))
                .unwrap(),
            uniform
                .select_action(decision(), &mut SplitMix64::new(42))
                .unwrap(),
        );
        assert_eq!(conditional.stats, uniform.stats);
    }
    struct Lottery;
    #[derive(Clone, Debug, Eq, PartialEq, Hash)]
    enum State {
        Start,
        Chance,
        End(i8),
    }
    #[derive(Clone, Debug, Eq, PartialEq)]
    enum Action {
        Safe,
        Risky,
        Outcome(bool),
    }
    impl Game for Lottery {
        type State = State;
        type Action = Action;
        type Observation<'a> = &'a State;
        type LegalActions<'a> = std::vec::IntoIter<Action>;
        fn player_count(&self) -> u8 {
            2
        }
        fn initial_state(&self) -> State {
            State::Start
        }
        fn status(&self, s: &State) -> PositionStatus {
            match s {
                State::Start => PositionStatus::PlayerTurn(PlayerId::FIRST),
                State::Chance => PositionStatus::Chance,
                State::End(_) => PositionStatus::Terminal,
            }
        }
        fn legal_actions<'a>(&'a self, s: &'a State) -> Self::LegalActions<'a> {
            if *s == State::Start {
                vec![Action::Safe, Action::Risky]
            } else {
                vec![]
            }
            .into_iter()
        }
        fn apply_action(&self, s: &mut State, a: &Action) -> Result<(), IllegalAction> {
            *s = match (&*s, a) {
                (State::Start, Action::Safe) => State::End(2),
                (State::Start, Action::Risky) => State::Chance,
                (State::Chance, Action::Outcome(win)) => State::End(if *win { 10 } else { -10 }),
                _ => return Err(IllegalAction::new("illegal lottery action")),
            };
            Ok(())
        }
        fn sample_chance<R: RandomSource + ?Sized>(
            &self,
            s: &State,
            r: &mut R,
        ) -> Result<Action, IllegalAction> {
            if *s != State::Chance {
                return Err(IllegalAction::new("not chance"));
            }
            Ok(Action::Outcome(r.index(10) == Some(0)))
        }
        fn observation<'a>(&'a self, s: &'a State, _: PlayerId) -> &'a State {
            s
        }
        fn terminal_utility(&self, s: &State, p: PlayerId) -> Option<f32> {
            if let State::End(v) = s {
                Some(f32::from(*v) / 10.0 * if p == PlayerId::FIRST { 1.0 } else { -1.0 })
            } else {
                None
            }
        }
    }
    impl PerfectInformationGame for Lottery {}
    impl TwoPlayerZeroSumGame for Lottery {}
    #[test]
    fn informed_chance_probes_do_not_commit_the_selected_outcome() {
        struct Scripted(std::vec::IntoIter<u64>);
        impl RandomSource for Scripted {
            fn next_u64(&mut self) -> u64 {
                self.0.next().expect("unexpected RNG draw")
            }
        }
        // 10 samples the rare win, 0 breaks the action tie, 9 samples a loss.
        let mut rng = Scripted(vec![10, 0, 9].into_iter());
        let action = crate::Greedy::new(NeutralEvaluator)
            .select_action(
                &Lottery,
                &State::Start,
                PlayerId::FIRST,
                PlayerId::FIRST,
                &mut rng,
            )
            .unwrap();
        assert_eq!(action, Action::Risky);
        let mut state = State::Start;
        Lottery.apply_action(&mut state, &action).unwrap();
        resolve_chance(&Lottery, &mut state, &mut rng).unwrap();
        assert_eq!(state, State::End(-10));
        assert!(rng.0.next().is_none());
    }

    #[test]
    fn epsilon_one_preserves_uniform_rng_on_a_chance_action() {
        for seed in [0, 11, 42] {
            let mut a = SplitMix64::new(seed);
            let mut b = SplitMix64::new(seed);
            assert_eq!(
                UniformRandom
                    .select_action(
                        &Lottery,
                        &State::Start,
                        PlayerId::FIRST,
                        PlayerId::FIRST,
                        &mut a
                    )
                    .unwrap(),
                crate::EpsilonGreedy::new(1.0, NeutralEvaluator)
                    .select_action(
                        &Lottery,
                        &State::Start,
                        PlayerId::FIRST,
                        PlayerId::FIRST,
                        &mut b
                    )
                    .unwrap(),
            );
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn stochastic_mast_resets_and_epsilon_one_matches_uniform() {
        for selection in [SelectionPolicy::Uct, SelectionPolicy::Ucb1Tuned] {
            let mut uniform = chain_agent(UniformRandom, selection);
            let mut mast = chain_agent(
                crate::RolloutPolicyConfig::<NeutralEvaluator>::Mast { epsilon: 1.0 },
                selection,
            );
            let mut a = SplitMix64::new(42);
            let mut b = SplitMix64::new(42);
            let decision = || DecisionContext::new(&ChanceChain, &0, PlayerId::FIRST);
            assert_eq!(
                uniform.select_action(decision(), &mut a).unwrap(),
                mast.select_action(decision(), &mut b).unwrap()
            );
            assert_eq!(uniform.stats, mast.stats);
            assert_eq!(a.next_u64(), b.next_u64());
            mast.config.rollout_policy = crate::RolloutPolicyConfig::Mast { epsilon: 0.1 };
            mast.select_action(decision(), &mut SplitMix64::new(1))
                .unwrap();
            let mut fresh = chain_agent(
                crate::RolloutPolicyConfig::<NeutralEvaluator>::Mast { epsilon: 0.1 },
                selection,
            );
            assert_eq!(
                mast.select_action(decision(), &mut SplitMix64::new(7))
                    .unwrap(),
                fresh
                    .select_action(decision(), &mut SplitMix64::new(7))
                    .unwrap()
            );
            assert_eq!(mast.stats, fresh.stats);
        }
    }

    #[test]
    fn stochastic_bias_averages_outcomes_and_decays() {
        let mut agent = StochasticMctsAgent::with_progressive_bias(
            MctsConfig {
                progressive_widening: None,
                budget: SearchBudget::Iterations(NonZeroU32::new(500).unwrap()),
                exploration: 2.0,
                selection_policy: SelectionPolicy::Uct,
                rollout_depth: 10,
                rollout_policy: UniformRandom,
            },
            NeutralEvaluator,
            crate::ProgressiveBias::new(0.25, Always, NeutralEvaluator),
        );
        agent.root_diagnostics = true;
        assert_eq!(
            agent
                .select_action(
                    DecisionContext::new(&Lottery, &State::Start, PlayerId::FIRST),
                    &mut SplitMix64::new(11)
                )
                .unwrap(),
            Action::Safe
        );
        let risky = &agent.stats.root_actions[1];
        assert!(risky.visits > 1);
        assert!(risky.heuristic_value.unwrap() < 0.0);
        assert_eq!(risky.heuristic_value.unwrap(), risky.mean_utility);
        assert_eq!(
            risky.progressive_bias.unwrap(),
            0.25 * risky.mean_utility / (f64::from(risky.visits) + 1.0)
        );
    }

    #[test]
    fn disabled_stochastic_bias_skips_evaluation_and_preserves_baseline() {
        struct Fail;
        impl StateEvaluator<ChanceChain> for Fail {
            fn evaluate(&self, _: &ChanceChain, _: &u8, _: PlayerId) -> Result<f64, AgentError> {
                panic!("disabled bias evaluator");
            }
        }
        struct Condition(bool);
        impl PolicyCondition<ChanceChain> for Condition {
            fn matches(&self, _: &ChanceChain, _: &u8, _: PlayerId, _: PlayerId) -> bool {
                self.0
            }
        }
        for (weight, condition) in [(0.0, true), (1.0, false)] {
            let mut baseline = chain_agent(UniformRandom, SelectionPolicy::Uct);
            let mut agent = StochasticMctsAgent::with_progressive_bias(
                baseline.config,
                NeutralEvaluator,
                crate::ProgressiveBias::new(weight, Condition(condition), Fail),
            );
            agent.root_diagnostics = true;
            let mut a = SplitMix64::new(42);
            let mut b = SplitMix64::new(42);
            let decision = || DecisionContext::new(&ChanceChain, &0, PlayerId::FIRST);
            assert_eq!(
                baseline.select_action(decision(), &mut a).unwrap(),
                agent.select_action(decision(), &mut b).unwrap()
            );
            assert_eq!(baseline.stats, agent.stats);
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
    #[test]
    fn chance_is_sampled_not_maximized_and_search_does_not_mutate_state() {
        for policy in [SelectionPolicy::Uct, SelectionPolicy::Ucb1Tuned] {
            let mut a = StochasticMctsAgent::new(
                MctsConfig {
                    progressive_widening: None,
                    budget: SearchBudget::Iterations(NonZeroU32::new(500).unwrap()),
                    exploration: 2.0,
                    selection_policy: policy,
                    rollout_depth: 10,
                    rollout_policy: UniformRandom,
                },
                NeutralEvaluator,
            );
            a.root_diagnostics = true;
            let s = Lottery.initial_state();
            let chosen = a
                .select_action(
                    DecisionContext::new(&Lottery, &s, PlayerId::FIRST),
                    &mut SplitMix64::new(11),
                )
                .unwrap();
            assert_eq!(chosen, Action::Safe);
            assert_eq!(s, State::Start);
            let stats = <StochasticMctsAgent as Agent<Lottery>>::last_decision_stats(&a);
            assert_eq!(stats.search_iterations, Some(500));
            assert_eq!(
                stats.root_actions.iter().map(|a| a.visits).sum::<u32>(),
                500
            );
            assert!(stats.root_actions[1].mean_utility < 0.0);
        }
    }
    #[test]
    fn zero_time_budget_still_returns_a_legal_move() {
        let mut a = StochasticMctsAgent::new(
            MctsConfig {
                progressive_widening: None,
                budget: SearchBudget::Time(std::time::Duration::ZERO),
                exploration: 1.0,
                selection_policy: SelectionPolicy::Uct,
                rollout_depth: 4,
                rollout_policy: UniformRandom,
            },
            NeutralEvaluator,
        );
        let chosen = a
            .select_action(
                DecisionContext::new(&Lottery, &State::Start, PlayerId::FIRST),
                &mut SplitMix64::new(0),
            )
            .unwrap();
        assert!(Lottery.legal_actions(&State::Start).any(|x| x == chosen));
        assert_eq!(
            <StochasticMctsAgent as Agent<Lottery>>::last_decision_stats(&a).search_iterations,
            Some(1)
        );
    }
}
