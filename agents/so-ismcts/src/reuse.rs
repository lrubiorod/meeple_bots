//! Realized-path reuse only. No authoritative state or determinization is retained.
use super::*;
use meeple_bots_core::TreeReuseStats;
use std::time::Duration;

/// Same accounting as MCTS: pending lifecycle debt and previous finalization reserve.
#[derive(Debug, Default)]
pub(super) struct ReuseTiming {
    pending: Duration,
    reserve: Duration,
    started: Option<Instant>,
    allowance: Duration,
    search_finished: Option<Instant>,
}
impl ReuseTiming {
    fn begin(&mut self, budget: SearchBudget) {
        self.search_finished = None;
        self.started = if let SearchBudget::Time(limit) = budget {
            self.allowance = limit
                .saturating_sub(self.pending)
                .saturating_sub(self.reserve);
            Some(Instant::now())
        } else {
            None
        };
        self.pending = Duration::ZERO;
    }
    pub(super) fn exhausted(&self) -> bool {
        self.started.is_some_and(|t| t.elapsed() >= self.allowance)
    }
    pub(super) fn finish_search(&mut self) {
        self.search_finished = self.started.map(|_| Instant::now());
    }
    fn finish(&mut self) {
        if let Some(t) = self.search_finished {
            self.reserve = t.elapsed();
        }
        self.started = None;
    }
}

#[derive(Debug)]
pub struct ReuseSearchResult<A, O> {
    pub search: SearchResult<A, O>,
    pub reuse: TreeReuseStats,
    pub new_nodes: u64,
}

/// One owner's current history subtree. G is rules/configuration, never G::State.
/// Start/end delimit a match; cloning carries configuration only.
/// Standalone SoIsmctsAgent::search remains fresh, even with tree_reuse=true.
pub struct ReusableSoIsmcts<G: Game, O> {
    agent: SoIsmctsAgent,
    game: Option<G>,
    owner: Option<PlayerId>,
    observation: Option<O>,
    retained: Option<ReuseSearchResult<G::Action, O>>,
    pending: TreeReuseStats,
    timing: ReuseTiming,
}
impl<G: Game, O> Clone for ReusableSoIsmcts<G, O> {
    fn clone(&self) -> Self {
        Self::new(self.agent.config.clone())
    }
}
impl<G: Game, O> ReusableSoIsmcts<G, O> {
    pub fn new(config: SoIsmctsConfig) -> Self {
        Self {
            agent: SoIsmctsAgent { config },
            game: None,
            owner: None,
            observation: None,
            retained: None,
            pending: TreeReuseStats::default(),
            timing: ReuseTiming::default(),
        }
    }
    pub fn enabled(&self) -> bool {
        self.agent.config.tree_reuse
    }
    pub fn owner(&self) -> Option<PlayerId> {
        self.owner
    }
    pub fn reset(&mut self) {
        self.pending.pruned_nodes += self
            .retained
            .as_ref()
            .map_or(0, |r| r.search.nodes.len() as u64);
        self.retained = None;
        self.observation = None;
        self.pending.reused_nodes = 0;
        self.pending.reused_root_visits = 0;
        self.pending.resets += 1;
    }
    pub fn end_match(&mut self) {
        self.retained = None;
        self.observation = None;
        self.owner = None;
        self.game = None;
        self.pending = TreeReuseStats::default();
        self.timing = ReuseTiming::default();
    }
    /// Call before constructing the decision observation, to include preparation.
    pub fn begin_decision(&mut self) {
        self.timing.begin(self.agent.config.budget);
    }
    /// May be called again after adapting diagnostics, reserving that tail next time.
    pub fn finish_decision(&mut self) {
        self.timing.finish();
    }
    pub fn record_maintenance(&mut self, duration: Duration) {
        if matches!(self.agent.config.budget, SearchBudget::Time(_)) {
            self.timing.pending += duration;
        }
    }
}
impl<G: Game + Clone + Eq, O: Clone + Eq> ReusableSoIsmcts<G, O>
where
    G::Action: Clone + Eq,
{
    pub fn start_match(&mut self, game: &G, owner: PlayerId) {
        self.end_match();
        self.game = Some(game.clone());
        self.owner = Some(owner);
    }
    /// Follow only an immediate action edge, then one of its observable outcomes.
    pub fn advance_real_transition(
        &mut self,
        game: &G,
        owner: PlayerId,
        actor: PlayerId,
        action: &G::Action,
        observation: &O,
    ) {
        if !self.enabled() {
            return;
        }
        if self.owner != Some(owner) || self.game.as_ref() != Some(game) || actor.index() >= 2 {
            self.reset();
            return;
        }
        let Some(retained) = self.retained.as_mut() else {
            return;
        };
        self.pending.transition_attempts += 1;
        let child = (|| {
            let root = retained.search.nodes.first()?;
            let mut edges = root.edges.iter().filter(|e| &e.action == action);
            let edge = edges.next()?;
            if edges.next().is_some() {
                return None;
            }
            let mut outcomes = edge.outcomes.iter().filter(|(o, _)| o == observation);
            let child = outcomes.next()?.1;
            if outcomes.next().is_some() {
                return None;
            }
            Some(child)
        })();
        let counts = child.and_then(|child| compact(&mut retained.search.nodes, child));
        if let Some((kept, pruned)) = counts {
            self.pending.transition_hits += 1;
            if actor == owner {
                self.pending.own_action_hits += 1;
            } else {
                self.pending.opponent_action_hits += 1;
            }
            self.pending.reused_nodes = kept as u64;
            self.pending.pruned_nodes += pruned as u64;
            self.observation = Some(observation.clone());
        } else {
            self.pending.transition_misses += 1;
            self.reset();
        }
    }
    pub fn search<W, R>(
        &mut self,
        game: &G,
        observation: &O,
        observer: PlayerId,
        legal: &[G::Action],
        rng: &mut R,
    ) -> Result<&ReuseSearchResult<G::Action, O>, AgentError>
    where
        G: TwoPlayerZeroSumGame
            + ImperfectInformationGame<Determinization = W>
            + for<'a> Game<Observation<'a> = O>
            + 'static,
        W: DeterminizedWorld<Action = G::Action, Observation = O>,
        R: RandomSource + ?Sized,
    {
        if self.timing.started.is_none() {
            self.begin_decision();
        }
        if !self.enabled() {
            self.retained = None;
            self.observation = None;
            self.pending = TreeReuseStats::default();
        }
        if self.owner != Some(observer) || self.game.as_ref() != Some(game) {
            self.reset();
            self.owner = Some(observer);
            self.game = Some(game.clone());
        }
        if self.retained.is_some()
            && (self.observation.as_ref() != Some(observation)
                || self
                    .retained
                    .as_ref()
                    .unwrap()
                    .search
                    .nodes
                    .first()
                    .is_none_or(|n| n.edges.iter().any(|e| !legal.contains(&e.action))))
        {
            self.reset();
        }
        let nodes = self
            .retained
            .take()
            .map(|r| r.search.nodes)
            .unwrap_or_default();
        let retained_count = nodes.len();
        self.pending.reused_root_visits = nodes
            .first()
            .map_or(0, |n| n.visits.min(u32::MAX as u64) as u32);
        self.pending.reused_nodes = retained_count as u64;
        let nodes = if nodes.is_empty() {
            vec![Node {
                visits: 0,
                edges: vec![],
            }]
        } else {
            nodes
        };
        let result = self.agent.search_with_nodes(
            game,
            observation,
            observer,
            legal,
            rng,
            nodes,
            Some(&mut self.timing),
        );
        match result {
            Ok(search) => {
                let new_nodes = (search.nodes.len() - retained_count) as u64;
                let reuse = std::mem::take(&mut self.pending);
                self.observation = Some(observation.clone());
                self.retained = Some(ReuseSearchResult {
                    search,
                    reuse,
                    new_nodes,
                });
                self.finish_decision();
                Ok(self.retained.as_ref().unwrap())
            }
            Err(e) => {
                self.reset();
                self.finish_decision();
                Err(e)
            }
        }
    }
}

/// Validate a tree (no cycles/shared children), then move only the selected subtree.
/// This indexes node IDs for remapping, never observations for transpositions.
fn compact<A: Eq, O: Eq>(nodes: &mut Vec<Node<A, O>>, root: usize) -> Option<(usize, usize)> {
    if root == 0 || root >= nodes.len() {
        return None;
    }
    let mut parents = vec![0usize; nodes.len()];
    for (index, node) in nodes.iter().enumerate() {
        for (i, edge) in node.edges.iter().enumerate() {
            if node.edges[..i].iter().any(|e| e.action == edge.action)
                || edge.visits > edge.availability
            {
                return None;
            }
            for (j, (o, child)) in edge.outcomes.iter().enumerate() {
                if *child <= index
                    || *child >= nodes.len()
                    || edge.outcomes[..j].iter().any(|(other, _)| other == o)
                {
                    return None;
                }
                parents[*child] += 1;
                if parents[*child] != 1 {
                    return None;
                }
            }
        }
    }
    if parents.iter().skip(1).any(|&p| p != 1) {
        return None;
    }
    let mut order = vec![root];
    let mut cursor = 0;
    while cursor < order.len() {
        for edge in &nodes[order[cursor]].edges {
            order.extend(edge.outcomes.iter().map(|(_, child)| *child));
        }
        cursor += 1;
    }
    let old_count = nodes.len();
    let mut remap = vec![usize::MAX; old_count];
    for (new, &old) in order.iter().enumerate() {
        remap[old] = new;
    }
    let mut old: Vec<_> = std::mem::take(nodes).into_iter().map(Some).collect();
    for i in order {
        let mut node = old[i].take()?;
        for edge in &mut node.edges {
            for (_, child) in &mut edge.outcomes {
                *child = remap[*child];
            }
        }
        nodes.push(node);
    }
    Some((nodes.len(), old_count - nodes.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use meeple_bots_lost_cities::{LostCities, LostCitiesAction as A, LostCitiesObservation};
    use meeple_bots_simulation::SplitMix64;
    use std::num::NonZeroU32;
    fn ready() -> meeple_bots_lost_cities::LostCitiesState {
        super::super::tests::ready()
    }
    fn wrapper(policy: BanditPolicy) -> ReusableSoIsmcts<LostCities, LostCitiesObservation> {
        ReusableSoIsmcts::new(SoIsmctsConfig {
            tree_reuse: true,
            selection_policy: policy,
            budget: SearchBudget::Iterations(NonZeroU32::new(128).unwrap()),
            ..Default::default()
        })
    }
    #[test]
    fn disabled_reuse_stays_fresh_even_for_repeated_observations() {
        let state = ready();
        let owner = PlayerId::FIRST;
        let observation = LostCities.observation(&state, owner);
        let legal: Vec<_> = LostCities.legal_actions(&state).collect();
        for policy in [BanditPolicy::Uct, BanditPolicy::Ucb1Tuned] {
            let mut w = wrapper(policy);
            w.agent.config.tree_reuse = false;
            w.start_match(&LostCities, owner);
            let fresh = w
                .agent
                .search(
                    &LostCities,
                    &observation,
                    owner,
                    &legal,
                    &mut SplitMix64::new(7),
                )
                .unwrap();
            for _ in 0..2 {
                let result = w
                    .search(
                        &LostCities,
                        &observation,
                        owner,
                        &legal,
                        &mut SplitMix64::new(7),
                    )
                    .unwrap();
                assert_eq!(result.search.action, fresh.action);
                assert_eq!(result.search.nodes, fresh.nodes);
                assert_eq!(result.reuse.reused_nodes, 0);
                assert_eq!(result.reuse.reused_root_visits, 0);
                assert_eq!(result.new_nodes, fresh.nodes.len() as u64);
            }
        }
    }
    #[test]
    fn reset_clears_retained_metrics_after_a_hit() {
        let mut w = fixture();
        let obs = w.observation.clone().unwrap();
        w.advance_real_transition(
            &LostCities,
            PlayerId::FIRST,
            PlayerId::FIRST,
            &A::DrawDeck,
            &obs,
        );
        assert_eq!(w.pending.reused_nodes, 1);
        w.reset();
        assert_eq!(w.pending.reused_nodes, 0);
        assert_eq!(w.pending.reused_root_visits, 0);
        assert_eq!(w.pending.pruned_nodes, 3);
    }
    #[test]
    fn own_microaction_preserves_subtree_and_adds_only_new_work() {
        for policy in [BanditPolicy::Uct, BanditPolicy::Ucb1Tuned] {
            let mut w = wrapper(policy);
            let mut state = ready();
            let owner = PlayerId::FIRST;
            w.start_match(&LostCities, owner);
            let obs = LostCities.observation(&state, owner);
            let legal: Vec<_> = LostCities.legal_actions(&state).collect();
            let r = w
                .search(&LostCities, &obs, owner, &legal, &mut SplitMix64::new(1))
                .unwrap();
            let action = r.search.action;
            let child = r.search.nodes[0]
                .edges
                .iter()
                .find(|e| e.action == action)
                .unwrap()
                .outcomes[0]
                .1;
            let expected = r.search.nodes[child].clone();
            let old_nodes = r.search.nodes.len();
            LostCities.apply_action(&mut state, &action).unwrap();
            assert_eq!(LostCities.status(&state), PositionStatus::PlayerTurn(owner));
            let obs = LostCities.observation(&state, owner);
            w.advance_real_transition(&LostCities, owner, owner, &action, &obs);
            assert_eq!(w.pending.own_action_hits, 1);
            let retained = &w.retained.as_ref().unwrap().search.nodes;
            assert_eq!(retained[0].visits, expected.visits);
            for (a, b) in retained[0].edges.iter().zip(&expected.edges) {
                assert_eq!(
                    (
                        a.visits,
                        a.availability,
                        a.total_utility,
                        a.total_squared_utility
                    ),
                    (
                        b.visits,
                        b.availability,
                        b.total_utility,
                        b.total_squared_utility
                    )
                );
            }
            assert_eq!(w.pending.pruned_nodes as usize + retained.len(), old_nodes);
            let legal: Vec<_> = LostCities.legal_actions(&state).collect();
            let r = w
                .search(&LostCities, &obs, owner, &legal, &mut SplitMix64::new(2))
                .unwrap();
            assert_eq!(r.search.diagnostics.completed_iterations, 128);
            assert_eq!(r.search.diagnostics.determinizations_sampled, 128);
            assert_eq!(r.search.nodes[0].visits, expected.visits + 128);
            assert_eq!(r.reuse.reused_root_visits as u64, expected.visits);
            assert_eq!(
                r.new_nodes + r.reuse.reused_nodes,
                r.search.nodes.len() as u64
            );
            for n in &r.search.nodes {
                for e in &n.edges {
                    assert!(e.visits <= e.availability);
                    assert!(
                        e.outcomes
                            .iter()
                            .all(|(_, child)| *child < r.search.nodes.len())
                    );
                }
            }
        }
    }
    fn fixture() -> ReusableSoIsmcts<LostCities, LostCitiesObservation> {
        let mut w = wrapper(BanditPolicy::Uct);
        let obs = LostCities.observation(&ready(), PlayerId::FIRST);
        w.start_match(&LostCities, PlayerId::FIRST);
        w.observation = Some(obs.clone());
        let mut other = obs.clone();
        other.deck_size -= 1;
        let node = |visits| Node {
            visits,
            edges: vec![],
        };
        w.retained = Some(ReuseSearchResult {
            new_nodes: 4,
            reuse: TreeReuseStats::default(),
            search: SearchResult {
                action: A::DrawDeck,
                diagnostics: Diagnostics::default(),
                nodes: vec![
                    Node {
                        visits: 12,
                        edges: vec![Edge {
                            action: A::DrawDeck,
                            visits: 12,
                            availability: 12,
                            total_utility: 2.,
                            total_squared_utility: 8.,
                            outcomes: vec![(obs, 1), (other, 2)],
                        }],
                    },
                    node(5),
                    node(7),
                ],
            },
        });
        w
    }
    #[test]
    fn outcomes_are_matched_only_beneath_actual_action() {
        let mut w = fixture();
        let obs = w.retained.as_ref().unwrap().search.nodes[0].edges[0].outcomes[1]
            .0
            .clone();
        w.advance_real_transition(
            &LostCities,
            PlayerId::FIRST,
            PlayerId::SECOND,
            &A::DrawDeck,
            &obs,
        );
        assert_eq!(w.retained.as_ref().unwrap().search.nodes[0].visits, 7);
        assert_eq!(w.pending.opponent_action_hits, 1);
        assert_eq!(w.pending.pruned_nodes, 2);
        let mut w = fixture();
        let mut obs = obs;
        obs.deck_size -= 1;
        w.advance_real_transition(
            &LostCities,
            PlayerId::FIRST,
            PlayerId::FIRST,
            &A::DrawDeck,
            &obs,
        );
        assert!(w.retained.is_none());
        assert_eq!(w.pending.transition_misses, 1);
        let mut w = fixture();
        let obs = w.observation.clone().unwrap();
        let action = LostCities.legal_actions(&ready()).next().unwrap();
        w.advance_real_transition(&LostCities, PlayerId::FIRST, PlayerId::FIRST, &action, &obs);
        assert!(w.retained.is_none()); // Matching observation under another action is not enough.
    }
    #[test]
    fn owner_observation_lifecycle_and_clone_reset() {
        let mut w = fixture();
        assert!(w.clone().retained.is_none());
        assert!(w.clone().owner.is_none());
        let obs = w.observation.clone().unwrap();
        w.advance_real_transition(
            &LostCities,
            PlayerId::SECOND,
            PlayerId::FIRST,
            &A::DrawDeck,
            &obs,
        );
        assert!(w.retained.is_none());
        let mut w = fixture();
        w.start_match(&LostCities, PlayerId::FIRST);
        assert!(w.retained.is_none());
        let mut w = fixture();
        w.end_match();
        assert!(w.retained.is_none());
        assert!(w.owner.is_none());
        let mut w = fixture();
        let state = ready();
        let mut obs = w.observation.clone().unwrap();
        obs.deck_size -= 1;
        // Reset before attempting to sample an invalid observation.
        let _ = w.search(
            &LostCities,
            &obs,
            PlayerId::FIRST,
            &LostCities.legal_actions(&state).collect::<Vec<_>>(),
            &mut SplitMix64::new(1),
        );
        assert!(w.retained.is_none());
    }
    #[test]
    fn malformed_outcomes_reset_and_no_unreachable_data_survives() {
        for mode in 0..3 {
            let mut w = fixture();
            let obs = w.observation.clone().unwrap();
            let edges = &mut w.retained.as_mut().unwrap().search.nodes[0].edges;
            if mode == 0 {
                edges[0].outcomes.push((obs.clone(), 1));
            }
            if mode == 1 {
                edges[0].outcomes[0].1 = 99;
            }
            if mode == 2 {
                edges[0].outcomes[0].1 = 0;
            }
            w.advance_real_transition(
                &LostCities,
                PlayerId::FIRST,
                PlayerId::FIRST,
                &A::DrawDeck,
                &obs,
            );
            assert!(w.retained.is_none());
        }
    }
    #[test]
    fn real_draw_observations_split_own_and_merge_hidden_outcomes() {
        for actor in [PlayerId::FIRST, PlayerId::SECOND] {
            let owner = PlayerId::FIRST;
            let mut state = ready();
            state.current_player = actor;
            let action = LostCities.legal_actions(&state).next().unwrap();
            LostCities.apply_action(&mut state, &action).unwrap();
            let before = LostCities.observation(&state, owner);
            let mut nodes = vec![Node {
                visits: 0,
                edges: vec![],
            }];
            available(&mut nodes[0], &[A::DrawDeck]);
            let mut observations = Vec::new();
            for seed in 0..20 {
                let mut real = state.clone();
                LostCities.apply_action(&mut real, &A::DrawDeck).unwrap();
                let event = LostCities
                    .sample_chance(&real, &mut SplitMix64::new(seed))
                    .unwrap();
                LostCities.apply_chance_outcome(&mut real, &event).unwrap();
                let obs = LostCities.observation(&real, owner);
                let (child, _) = outcome(&mut nodes, 0, 0, obs.clone());
                nodes[child].visits = child as u64;
                observations.push((obs, child));
            }
            if actor == owner {
                assert!(nodes[0].edges[0].outcomes.len() > 1);
            } else {
                assert_eq!(nodes[0].edges[0].outcomes.len(), 1);
            }
            for (obs, child) in observations {
                let mut w = wrapper(BanditPolicy::Uct);
                w.start_match(&LostCities, owner);
                w.observation = Some(before.clone());
                w.retained = Some(ReuseSearchResult {
                    new_nodes: 0,
                    reuse: TreeReuseStats::default(),
                    search: SearchResult {
                        action: A::DrawDeck,
                        diagnostics: Diagnostics::default(),
                        nodes: nodes.clone(),
                    },
                });
                w.advance_real_transition(&LostCities, owner, actor, &A::DrawDeck, &obs);
                assert_eq!(w.pending.transition_hits, 1);
                assert_eq!(
                    w.retained.as_ref().unwrap().search.nodes[0].visits,
                    child as u64
                );
            }
        }
    }
    #[test]
    fn timed_budget_deducts_maintenance_and_reserve() {
        let mut w = wrapper(BanditPolicy::Uct);
        w.agent.config.budget = SearchBudget::Time(Duration::from_millis(50));
        w.timing.pending = Duration::from_millis(20);
        w.timing.reserve = Duration::from_millis(10);
        w.begin_decision();
        assert_eq!(w.timing.allowance, Duration::from_millis(20));
        w.timing.pending = Duration::from_secs(2);
        w.begin_decision();
        let state = ready();
        let obs = LostCities.observation(&state, PlayerId::FIRST);
        let r = w
            .search(
                &LostCities,
                &obs,
                PlayerId::FIRST,
                &LostCities.legal_actions(&state).collect::<Vec<_>>(),
                &mut SplitMix64::new(1),
            )
            .unwrap();
        assert_eq!(r.search.diagnostics.completed_iterations, 1);
    }
}
