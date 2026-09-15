//! Retained public-chance search follows accepted actions and every observed event.
use super::stochastic::Node;
use super::{
    DecisionBudget, NeutralEvaluator, NoSelectionBias, RolloutPolicy, SearchBudget, SelectionBias,
    StateEvaluator, StochasticMctsAgent, UniformRandom,
};
use meeple_bots_core::{
    Agent, AgentDecisionStats, AgentError, DecisionContext, Game, PerfectInformationGame, PlayerId,
    PositionStatus, RandomSource, TreeReuseStats, TwoPlayerZeroSumGame,
};
use std::{hash::Hash, time::Instant};

pub struct ReusableStochasticMctsAgent<
    G: Game,
    C = NeutralEvaluator,
    P = UniformRandom,
    B = NoSelectionBias,
> {
    inner: StochasticMctsAgent<C, P, B>,
    enabled: bool,
    transpositions: bool,
    game: Option<G>,
    owner: Option<PlayerId>,
    nodes: Vec<Node<G::State, G::Action>>,
    // Root edge, acting player and the last verified intermediate chance state.
    pending: Option<(usize, PlayerId, G::State)>,
    reuse: TreeReuseStats,
}
impl<G: Game, C, P, B> ReusableStochasticMctsAgent<G, C, P, B> {
    pub fn new(
        inner: StochasticMctsAgent<C, P, B>,
        tree_reuse: bool,
        transpositions: bool,
    ) -> Self {
        Self {
            inner,
            enabled: tree_reuse,
            transpositions,
            game: None,
            owner: None,
            nodes: Vec::new(),
            pending: None,
            reuse: TreeReuseStats::default(),
        }
    }
    fn clear(&mut self) {
        self.nodes.clear();
        self.pending = None;
        self.game = None;
    }
    fn miss(&mut self) {
        self.reuse.transition_misses += 1;
        self.reuse.resets += 1;
        self.reuse.pruned_nodes += self.nodes.len() as u64;
        self.clear();
    }
}
impl<G: Game, C: Clone, P: Clone, B: Clone> Clone for ReusableStochasticMctsAgent<G, C, P, B> {
    fn clone(&self) -> Self {
        let mut inner = self.inner.clone();
        inner.stats = AgentDecisionStats::default();
        Self::new(inner, self.enabled, self.transpositions)
    }
}
impl<G: Game, C, P, B> std::fmt::Debug for ReusableStochasticMctsAgent<G, C, P, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReusableStochasticMctsAgent")
            .field("tree_reuse", &self.enabled)
            .field("transpositions", &self.transpositions)
            .finish_non_exhaustive()
    }
}

// Remap the reachable component, keeping its root at index zero. Also supports cycles.
fn compact<S, A>(nodes: &mut Vec<Node<S, A>>, root: usize) -> usize {
    let before = nodes.len();
    let mut order = Vec::new();
    let mut mapping = vec![None; before];
    let mut stack = vec![root];
    while let Some(index) = stack.pop() {
        if mapping[index].is_some() {
            continue;
        }
        mapping[index] = Some(order.len());
        order.push(index);
        for edge in &nodes[index].edges {
            stack.extend(edge.outcomes.values().copied());
        }
    }
    let mut old: Vec<_> = std::mem::take(nodes).into_iter().map(Some).collect();
    for index in order {
        let mut node = old[index].take().unwrap();
        for edge in &mut node.edges {
            for child in edge.outcomes.values_mut() {
                *child = mapping[*child].unwrap();
            }
        }
        nodes.push(node);
    }
    before - nodes.len()
}

impl<G, C, P, B> ReusableStochasticMctsAgent<G, C, P, B>
where
    G: Game + PartialEq,
    G::State: Clone + Eq + Hash,
    G::Action: Clone + Eq,
{
    fn settle(&mut self, game: &G, state: &G::State) {
        if game.status(state) == PositionStatus::Chance {
            return;
        }
        let Some((edge, player, _)) = self.pending.take() else {
            self.miss();
            return;
        };
        if game.status(state) == PositionStatus::Terminal {
            self.clear();
            return;
        }
        let Some(&child) = self.nodes[0].edges[edge].outcomes.get(state) else {
            self.miss();
            return;
        };
        self.reuse.pruned_nodes += compact(&mut self.nodes, child) as u64;
        self.reuse.transition_hits += 1;
        self.reuse.reused_nodes = self.nodes.len() as u64;
        if self.owner == Some(player) {
            self.reuse.own_action_hits += 1;
        } else {
            self.reuse.opponent_action_hits += 1;
        }
    }
    fn observe(
        &mut self,
        game: &G,
        state: &G::State,
        player: Option<PlayerId>,
        action: &G::Action,
    ) {
        if !self.enabled || self.nodes.is_empty() {
            return;
        }
        if self.game.as_ref() != Some(game) {
            self.miss();
            return;
        }
        let mut expected = if let Some(player) = player {
            self.reuse.transition_attempts += 1;
            if self.pending.is_some()
                || game.status(&self.nodes[0].state) != PositionStatus::PlayerTurn(player)
            {
                self.miss();
                return;
            }
            let Some(edge) = self.nodes[0].edges.iter().position(|e| e.action == *action) else {
                self.miss();
                return;
            };
            self.pending = Some((edge, player, self.nodes[0].state.clone()));
            self.nodes[0].state.clone()
        } else {
            let Some((_, _, expected)) = &self.pending else {
                self.miss();
                return;
            };
            if game.status(expected) != PositionStatus::Chance {
                self.miss();
                return;
            }
            expected.clone()
        };
        let applied = if player.is_some() {
            game.apply_action(&mut expected, action)
        } else {
            game.apply_chance_outcome(&mut expected, action)
        };
        if applied.is_err() || expected != *state {
            self.miss();
            return;
        }
        self.pending.as_mut().unwrap().2 = expected;
        self.settle(game, state);
    }
}

impl<G, C, P, B> Agent<G> for ReusableStochasticMctsAgent<G, C, P, B>
where
    G: Game + PerfectInformationGame + TwoPlayerZeroSumGame + Clone + PartialEq,
    G::State: Clone + Eq + Hash,
    G::Action: Clone + Eq,
    C: StateEvaluator<G>,
    P: RolloutPolicy<G>,
    B: SelectionBias<G>,
{
    fn on_match_start(&mut self, _: &G, _: &G::State, player: PlayerId) {
        self.inner.budget = DecisionBudget::new();
        let started = Instant::now();
        self.clear();
        self.owner = Some(player);
        self.reuse = TreeReuseStats::default();
        self.inner.stats = AgentDecisionStats::default();
        self.inner.budget.pending_maintenance += started.elapsed();
    }
    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, G>,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        if !self.enabled && !self.transpositions {
            return self.inner.select_action(decision, rng);
        }
        let started = Instant::now();
        if !self.nodes.is_empty()
            && (self.game.as_ref() != Some(decision.game())
                || self.owner != Some(decision.player())
                || self.pending.is_some()
                || self.nodes[0].state != *decision.state())
        {
            self.miss();
        }
        self.game = Some(decision.game().clone());
        self.owner = Some(decision.player());
        self.reuse.reused_root_visits = self.nodes.first().map_or(0, |n| n.visits);
        self.reuse.reused_nodes = self.nodes.len() as u64;
        self.inner.budget.pending_maintenance += started.elapsed();
        let result = self
            .inner
            .search(decision, rng, &mut self.nodes, self.transpositions);
        if result.is_err() || !self.enabled {
            self.clear();
        }
        self.inner.stats.tree_reuse = self.enabled.then(|| std::mem::take(&mut self.reuse));
        self.inner.budget.finish();
        result
    }
    fn last_decision_stats(&self) -> AgentDecisionStats {
        self.inner.stats.clone()
    }
    fn on_action_applied(
        &mut self,
        game: &G,
        state: &G::State,
        player: PlayerId,
        action: &G::Action,
    ) {
        let started = Instant::now();
        self.observe(game, state, Some(player), action);
        if matches!(self.inner.config.budget, SearchBudget::Time(_)) {
            self.inner.budget.pending_maintenance += started.elapsed();
        }
    }
    fn on_chance_applied(&mut self, game: &G, state: &G::State, event: &G::Action) {
        let started = Instant::now();
        self.observe(game, state, None, event);
        if matches!(self.inner.config.budget, SearchBudget::Time(_)) {
            self.inner.budget.pending_maintenance += started.elapsed();
        }
    }
    fn on_match_end(&mut self, _: &G, _: &G::State) {
        let started = Instant::now();
        self.clear();
        self.owner = None;
        self.inner.budget.pending_maintenance += started.elapsed();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stochastic::tests::ChanceChain;
    use crate::{MctsConfig, SearchBudget, SelectionPolicy};
    use meeple_bots_simulation::SplitMix64;
    use std::num::NonZeroU32;

    fn agent(enabled: bool, iterations: u32) -> ReusableStochasticMctsAgent<ChanceChain> {
        let mut inner = StochasticMctsAgent::new(
            MctsConfig {
                progressive_widening: None,
                budget: SearchBudget::Iterations(NonZeroU32::new(iterations).unwrap()),
                exploration: 1.4,
                selection_policy: SelectionPolicy::Uct,
                rollout_depth: 3,
                rollout_policy: UniformRandom,
            },
            NeutralEvaluator,
        );
        inner.root_diagnostics = true;
        ReusableStochasticMctsAgent::new(inner, enabled, false)
    }
    fn decide(a: &mut ReusableStochasticMctsAgent<ChanceChain>, state: u8) -> u8 {
        let PositionStatus::PlayerTurn(player) = ChanceChain.status(&state) else {
            panic!()
        };
        a.select_action(
            DecisionContext::new(&ChanceChain, &state, player),
            &mut SplitMix64::new(42),
        )
        .unwrap()
    }
    #[test]
    fn reuses_after_chance_between_consecutive_decisions_and_clears_lifecycle() {
        let mut a = agent(true, 32);
        let action = decide(&mut a, 3);
        a.on_action_applied(&ChanceChain, &4, PlayerId::SECOND, &action);
        assert!(a.pending.is_some());
        assert_eq!(a.nodes[0].state, 3);
        a.on_chance_applied(&ChanceChain, &5, &100);
        assert!(a.pending.is_none());
        assert_eq!(a.nodes[0].state, 5);
        decide(&mut a, 5);
        let stats = a.last_decision_stats().tree_reuse.unwrap();
        assert_eq!(stats.transition_hits, 1);
        assert_eq!(stats.own_action_hits, 1);
        assert!(stats.reused_root_visits > 0);
        assert!(a.clone().nodes.is_empty());
        a.on_match_end(&ChanceChain, &6);
        assert!(a.nodes.is_empty());
        decide(&mut a, 3);
        a.on_match_start(&ChanceChain, &0, PlayerId::FIRST);
        assert!(a.nodes.is_empty());
        assert!(a.pending.is_none());
    }
    #[test]
    fn waits_for_entire_chance_chain_and_follows_opponent() {
        let mut a = agent(true, 32);
        let action = decide(&mut a, 0);
        a.on_action_applied(&ChanceChain, &1, PlayerId::FIRST, &action);
        a.on_chance_applied(&ChanceChain, &2, &100);
        assert_eq!(a.nodes[0].state, 0);
        assert!(a.pending.is_some());
        a.on_chance_applied(&ChanceChain, &3, &101);
        assert_eq!(a.nodes[0].state, 3);
        a.on_action_applied(&ChanceChain, &4, PlayerId::SECOND, &10);
        a.on_chance_applied(&ChanceChain, &5, &100);
        assert_eq!(a.nodes[0].state, 5);
        assert_eq!(a.reuse.opponent_action_hits, 1);
    }
    #[test]
    fn unexplored_or_inconsistent_outcomes_reset_safely() {
        let mut a = agent(true, 1);
        let selected = decide(&mut a, 3);
        let other = if selected == 10 { 11 } else { 10 };
        a.on_action_applied(&ChanceChain, &4, PlayerId::SECOND, &other);
        a.on_chance_applied(&ChanceChain, &5, &100);
        assert!(a.nodes.is_empty());
        assert_eq!(a.reuse.transition_misses, 1);
        let mut a = agent(true, 32);
        let action = decide(&mut a, 3);
        a.on_action_applied(&ChanceChain, &5, PlayerId::SECOND, &action);
        assert!(a.nodes.is_empty());
        decide(&mut a, 3);
        assert_eq!(
            a.last_decision_stats()
                .tree_reuse
                .unwrap()
                .reused_root_visits,
            0
        );
    }
    #[test]
    fn disabled_wrapper_preserves_search_and_rng() {
        let mut a = agent(false, 32);
        let mut b = a.inner.clone();
        let mut ar = SplitMix64::new(42);
        let mut br = SplitMix64::new(42);
        let decision = || DecisionContext::new(&ChanceChain, &0, PlayerId::FIRST);
        assert_eq!(
            a.select_action(decision(), &mut ar).unwrap(),
            b.select_action(decision(), &mut br).unwrap()
        );
        assert_eq!(a.last_decision_stats(), b.stats);
        assert_eq!(ar.next_u64(), br.next_u64());
        assert!(a.nodes.is_empty());
    }

    #[test]
    fn transpositions_merge_states_but_keep_incoming_visits_separate() {
        for selection in [SelectionPolicy::Uct, SelectionPolicy::Ucb1Tuned] {
            let mut tree = agent(true, 32);
            tree.inner.config.selection_policy = selection;
            decide(&mut tree, 0);
            let mut graph = agent(true, 32);
            graph.inner.config.selection_policy = selection;
            graph.transpositions = true;
            let action = decide(&mut graph, 0);
            assert_eq!(graph.nodes.len(), 3);
            assert!(tree.nodes.len() > graph.nodes.len());
            let root = &graph.nodes[0];
            assert_eq!(root.edges.iter().map(|e| e.visits).sum::<u32>(), 32);
            let shared = root.edges[0].outcomes[&3];
            assert_eq!(root.edges[1].outcomes[&3], shared);
            assert!(
                root.edges
                    .iter()
                    .all(|e| e.visits < graph.nodes[shared].visits)
            );
            graph.on_action_applied(&ChanceChain, &1, PlayerId::FIRST, &action);
            graph.on_chance_applied(&ChanceChain, &2, &100);
            graph.on_chance_applied(&ChanceChain, &3, &101);
            assert_eq!(graph.nodes.len(), 2);
            assert_eq!(graph.nodes[0].state, 3);
            assert!(graph.nodes[0].edges.iter().all(|e| e.outcomes[&5] == 1));
        }
    }

    #[derive(Clone, PartialEq)]
    struct Loop;
    impl Game for Loop {
        type State = u8;
        type Action = u8;
        type Observation<'a> = u8;
        type LegalActions<'a> = std::vec::IntoIter<u8>;
        fn initial_state(&self) -> u8 {
            0
        }
        fn player_count(&self) -> u8 {
            2
        }
        fn status(&self, state: &u8) -> PositionStatus {
            if *state == 1 {
                PositionStatus::Chance
            } else {
                PositionStatus::PlayerTurn(PlayerId::FIRST)
            }
        }
        fn legal_actions(&self, state: &u8) -> Self::LegalActions<'_> {
            if *state == 1 { vec![] } else { vec![10] }.into_iter()
        }
        fn apply_action(
            &self,
            state: &mut u8,
            action: &u8,
        ) -> Result<(), meeple_bots_core::IllegalAction> {
            *state = match (*state, *action) {
                (0 | 2, 10) => 1,
                (1, 100) => 0,
                (1, 101) => 2,
                _ => return Err(meeple_bots_core::IllegalAction::new("illegal loop action")),
            };
            Ok(())
        }
        fn sample_chance<R: RandomSource + ?Sized>(
            &self,
            state: &u8,
            rng: &mut R,
        ) -> Result<u8, meeple_bots_core::IllegalAction> {
            assert_eq!(*state, 1);
            Ok(100 + rng.index(2).unwrap() as u8)
        }
        fn terminal_utility(&self, _: &u8, _: PlayerId) -> Option<f32> {
            None
        }
        fn observation(&self, state: &u8, _: PlayerId) -> u8 {
            *state
        }
    }
    impl PerfectInformationGame for Loop {}
    impl TwoPlayerZeroSumGame for Loop {}

    #[test]
    fn cyclic_chance_graphs_bound_traversal_and_survive_compaction() {
        for selection in [SelectionPolicy::Uct, SelectionPolicy::Ucb1Tuned] {
            let mut inner = agent(false, 64).inner;
            inner.config.selection_policy = selection;
            let mut graph = ReusableStochasticMctsAgent::<Loop>::new(inner, true, true);
            let decision = || DecisionContext::new(&Loop, &0, PlayerId::FIRST);
            graph
                .select_action(decision(), &mut SplitMix64::new(42))
                .unwrap();
            assert_eq!(graph.nodes.len(), 2);
            assert_eq!(graph.nodes[0].visits, 64);
            assert_eq!(graph.nodes[0].edges[0].visits, 64);
            assert_eq!(graph.nodes[0].edges[0].outcomes.len(), 2);
            graph.on_action_applied(&Loop, &1, PlayerId::FIRST, &10);
            graph.on_chance_applied(&Loop, &0, &100);
            graph
                .select_action(decision(), &mut SplitMix64::new(11))
                .unwrap();
            assert_eq!(graph.nodes[0].visits, 128);
            assert_eq!(graph.nodes.len(), 2);
            graph.inner.config.budget = SearchBudget::Time(std::time::Duration::ZERO);
            graph
                .select_action(decision(), &mut SplitMix64::new(11))
                .unwrap();
            assert_eq!(graph.inner.stats.search_iterations, Some(1));
        }
    }

    #[test]
    fn transpositions_without_reuse_start_fresh_each_decision() {
        let mut graph = agent(false, 32);
        graph.transpositions = true;
        let first = decide(&mut graph, 0);
        let stats = graph.last_decision_stats();
        assert_eq!(stats.search_nodes, Some(3));
        assert!(stats.tree_reuse.is_none());
        assert!(graph.nodes.is_empty());
        assert_eq!(decide(&mut graph, 0), first);
        assert_eq!(graph.last_decision_stats(), stats);
    }
}
