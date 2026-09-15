//! Classic state-local AMAF. Independent of rollout-policy learning memory.
use super::*;

pub const DEFAULT_RAVE_EQUIVALENCE: u32 = 1000;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct AmafStats {
    pub(super) visits: u64,
    pub(super) total: f64,
}
impl AmafStats {
    pub(super) fn mean(self) -> Option<f64> {
        (self.visits > 0).then(|| self.total / self.visits as f64)
    }
    pub(super) fn update(&mut self, utility: f64) {
        self.visits += 1;
        self.total += utility;
    }
}
pub(super) struct AmafEdge<A> {
    pub(super) action: A,
    pub(super) stats: AmafStats,
}

// The existing typed reuse adapter supplies these operations without adding Clone/Eq
// requirements to the basic MctsAgent, which also supports non-cloneable actions.
pub(super) struct ActionOps<A> {
    clone: fn(&A) -> A,
    equal: fn(&A, &A) -> bool,
}
impl<A: Clone + PartialEq> ActionOps<A> {
    pub(super) fn typed() -> Self {
        Self {
            clone: A::clone,
            equal: A::eq,
        }
    }
}
pub(super) struct RaveTrace<A> {
    ops: Option<ActionOps<A>>,
    actions: Vec<(PlayerId, A)>,
    pub(super) visited: Vec<(usize, PlayerId, usize)>,
}
impl<A> RaveTrace<A> {
    pub(super) fn new(ops: Option<ActionOps<A>>) -> Self {
        Self {
            ops,
            actions: Vec::new(),
            visited: Vec::new(),
        }
    }
    pub(super) fn clear(&mut self) {
        self.actions.clear();
        self.visited.clear();
    }
    pub(super) fn record(&mut self, player: PlayerId, action: &A) {
        if let Some(ops) = &self.ops {
            self.actions.push((player, (ops.clone)(action)));
        }
    }
    pub(super) fn visit<G: meeple_bots_core::Game<Action = A>>(
        &mut self,
        index: usize,
        game: &G,
        state: &G::State,
        table: &mut Option<Vec<AmafEdge<A>>>,
    ) {
        if self.ops.is_none() || self.visited.iter().any(|(n, _, _)| *n == index) {
            return;
        }
        if let PositionStatus::PlayerTurn(player) = game.status(state) {
            // Includes unexpanded legal edges. Allocate once, never regenerate at backup.
            table.get_or_insert_with(|| {
                game.legal_actions(state)
                    .map(|action| AmafEdge {
                        action,
                        stats: AmafStats::default(),
                    })
                    .collect()
            });
            self.visited.push((index, player, self.actions.len()));
        }
    }
    pub(super) fn update(
        &self,
        table: &mut [AmafEdge<A>],
        player: PlayerId,
        start: usize,
        utility: f64,
    ) {
        let ops = self.ops.as_ref().expect("RAVE is enabled");
        for edge in table {
            // any() gives at most one credit per state/action/simulation, including
            // the direct action, without allocating a set for every visited node.
            if self.actions[start..]
                .iter()
                .any(|(p, a)| *p == player && (ops.equal)(&edge.action, a))
            {
                edge.stats.update(utility);
            }
        }
    }
    pub(super) fn stats(&self, table: &Option<Vec<AmafEdge<A>>>, action: &A) -> AmafStats {
        let ops = self.ops.as_ref().expect("RAVE is enabled");
        table
            .as_ref()
            .and_then(|entries| entries.iter().find(|e| (ops.equal)(&e.action, action)))
            .map(|e| e.stats)
            .unwrap_or_default()
    }
}
pub(super) fn rave_beta(parent_visits: f64, k: f64) -> f64 {
    (k / (3.0 * parent_visits + k)).sqrt()
}
pub(super) fn rave_mean(normal: f64, amaf: AmafStats, parent_visits: f64, k: f64) -> f64 {
    match amaf.mean() {
        None => normal,
        Some(mean) => {
            let beta = rave_beta(parent_visits, k);
            (1.0 - beta) * normal + beta * mean
        }
    }
}
pub(super) fn rave_score(
    visits: u32,
    normal: f64,
    amaf: AmafStats,
    parent: f64,
    maximizing: bool,
    c: f64,
    k: f64,
) -> f64 {
    if visits == 0 {
        return f64::INFINITY;
    }
    let mean = rave_mean(normal, amaf, parent, k);
    (if maximizing { mean } else { -mean }) + c * (parent.max(1.0).ln() / f64::from(visits)).sqrt()
}
pub(super) fn best_rave_child<A>(
    nodes: &[Node<A>],
    parent: usize,
    maximizing: bool,
    c: f64,
    k: f64,
    bias: f64,
    trace: &RaveTrace<A>,
) -> usize {
    let n = f64::from(nodes[parent].visits);
    let score = |i: usize| {
        let edge = &nodes[i];
        rave_score(
            edge.visits,
            edge.mean_utility(),
            trace.stats(&nodes[parent].amaf, edge.action.as_ref().unwrap()),
            n,
            maximizing,
            c,
            k,
        ) + progressive_bias_term(edge, maximizing, bias)
    };
    *nodes[parent]
        .children
        .iter()
        .max_by(|a, b| score(**a).total_cmp(&score(**b)))
        .unwrap()
}
pub(super) fn best_rave_graph_edge<S, A>(
    nodes: &[GraphNode<S, A>],
    parent: usize,
    maximizing: bool,
    c: f64,
    k: f64,
    bias: f64,
    trace: &RaveTrace<A>,
) -> usize {
    let n = f64::from(nodes[parent].visits);
    let score = |edge: &GraphEdge<A>| {
        rave_score(
            edge.visits,
            if edge.visits == 0 {
                0.0
            } else {
                edge.total_utility / f64::from(edge.visits)
            },
            trace.stats(&nodes[parent].amaf, &edge.action),
            n,
            maximizing,
            c,
            k,
        ) + graph_progressive_bias_term(edge, maximizing, bias)
    };
    nodes[parent]
        .edges
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| score(a).total_cmp(&score(b)))
        .unwrap()
        .0
}

#[cfg(test)]
mod tests {
    use super::*;
    use meeple_bots_connect6::{Connect6, Connect6Action};
    use meeple_bots_core::Game;
    use meeple_bots_simulation::SplitMix64;

    fn policy() -> SelectionPolicy {
        SelectionPolicy::UctRave {
            rave_equivalence: 1000,
        }
    }
    fn config(iterations: u32) -> MctsConfig<UniformRandom> {
        MctsConfig {
            progressive_widening: None,
            budget: SearchBudget::Iterations(NonZeroU32::new(iterations).unwrap()),
            exploration: 1.4,
            selection_policy: policy(),
            rollout_depth: 2,
            rollout_policy: UniformRandom,
        }
    }
    #[test]
    fn beta_blend_missing_samples_and_real_exploration() {
        let k = 1000.;
        assert!(rave_beta(1., k) > 0.);
        assert!(rave_beta(1., k) > rave_beta(1000., k));
        assert!(rave_beta(1e15, k) < 1e-5);
        assert_eq!(rave_mean(0.4, AmafStats::default(), 10., k), 0.4);
        let stats = AmafStats {
            visits: 4,
            total: 4.,
        };
        let beta = rave_beta(1000., k);
        assert!((beta - 0.5).abs() < 1e-12);
        assert!((rave_mean(-0.2, stats, 1000., k) - 0.4).abs() < 1e-12);
        assert!(rave_score(0, 0., stats, 1000., true, 1., k).is_infinite());
        let score = rave_score(10, -0.2, stats, 1000., true, 2., k);
        assert!((score - (0.4 + 2. * (1000_f64.ln() / 10.).sqrt())).abs() < 1e-12);
        assert!((rave_score(10, -0.2, stats, 1000., false, 0., k) + 0.4).abs() < 1e-12);
        assert!(
            MctsConfig {
                selection_policy: SelectionPolicy::UctRave {
                    rave_equivalence: 0
                },
                ..config(1)
            }
            .validate()
            .is_err()
        );
    }
    #[test]
    fn connect6_amaf_credits_actual_player_direct_action_and_deduplicates() {
        let game = Connect6::new(6).unwrap();
        let mut state = game.initial_state();
        game.apply_action(&mut state, &Connect6Action::Place(0))
            .unwrap();
        let mut trace = RaveTrace::new(Some(ActionOps::typed()));
        let mut table = None;
        trace.visit(0, &game, &state, &mut table);
        for pos in [1, 2, 3, 4] {
            let PositionStatus::PlayerTurn(player) = game.status(&state) else {
                panic!()
            };
            let action = Connect6Action::Place(pos);
            trace.record(player, &action);
            game.apply_action(&mut state, &action).unwrap();
        }
        // Duplicate later occurrence and action not legal in the original state.
        trace.record(PlayerId::SECOND, &Connect6Action::Place(1));
        trace.record(PlayerId::SECOND, &Connect6Action::Place(0));
        trace.update(table.as_mut().unwrap(), PlayerId::SECOND, 0, 1.);
        for pos in [1, 2] {
            assert_eq!(
                trace.stats(&table, &Connect6Action::Place(pos)),
                AmafStats {
                    visits: 1,
                    total: 1.
                }
            );
        }
        for pos in [0, 3, 4, 5] {
            assert_eq!(
                trace.stats(&table, &Connect6Action::Place(pos)),
                AmafStats::default()
            );
        }
        // Suffix scope excludes an action which only occurred before node arrival.
        let mut later = None;
        trace.visit(1, &game, &state, &mut later);
        trace.record(PlayerId::SECOND, &Connect6Action::Place(5));
        trace.update(
            later.as_mut().unwrap(),
            PlayerId::SECOND,
            trace.visited[1].2,
            1.,
        );
        assert_eq!(trace.stats(&later, &Connect6Action::Place(5)).visits, 1);
        assert_eq!(trace.stats(&later, &Connect6Action::Place(6)).visits, 0);
    }
    #[test]
    fn tree_search_records_rollout_overshoot_and_preserves_amaf_on_reroot() {
        let game = Connect6::new(6).unwrap();
        let state = game.initial_state();
        let mut agent = TreeReuseMctsAgent::new(MctsAgent::new(config(1)), true);
        agent.on_match_start(&game, &state, PlayerId::FIRST);
        let action = agent
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut SplitMix64::new(42),
            )
            .unwrap();
        let tree = agent.tree.as_mut().unwrap();
        let child = tree.nodes[0].children[0];
        assert_eq!(tree.nodes[child].visits, 1);
        assert_eq!(tree.nodes[0].visits, 1);
        let root = tree.nodes[0].amaf.as_ref().unwrap();
        assert_eq!(
            root.iter()
                .find(|e| e.action == action)
                .unwrap()
                .stats
                .visits,
            1
        );
        // Opening expansion followed by White's two rollout placements.
        assert_eq!(root.iter().filter(|e| e.stats.visits > 0).count(), 1);
        let child_table = tree.nodes[child].amaf.as_ref().unwrap();
        assert_eq!(child_table.iter().filter(|e| e.stats.visits > 0).count(), 2);
        let saved: Vec<_> = child_table.iter().map(|e| (e.action, e.stats)).collect();
        let mut next = state.clone();
        game.apply_action(&mut next, &action).unwrap();
        agent.on_action_applied(&game, &next, PlayerId::FIRST, &action);
        let retained = agent.tree.as_ref().unwrap();
        assert_eq!(
            retained.nodes[0]
                .amaf
                .as_ref()
                .unwrap()
                .iter()
                .map(|e| (e.action, e.stats))
                .collect::<Vec<_>>(),
            saved
        );
        assert_eq!(retained.nodes[0].visits, 1);

        // Nominal one rollout action must include the second White placement too.
        let mut agent = TreeReuseMctsAgent::new(
            MctsAgent::new(MctsConfig {
                rollout_depth: 1,
                ..config(1)
            }),
            true,
        );
        agent
            .select_action(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut SplitMix64::new(42),
            )
            .unwrap();
        let tree = agent.tree.as_ref().unwrap();
        assert_eq!(
            tree.nodes[1]
                .amaf
                .as_ref()
                .unwrap()
                .iter()
                .filter(|e| e.stats.visits > 0)
                .count(),
            2
        );
    }
    #[test]
    fn shared_state_amaf_is_updated_once_and_graph_compaction_preserves_it() {
        let game = Connect6::new(6).unwrap();
        let mut start = game.initial_state();
        game.apply_action(&mut start, &Connect6Action::Place(0))
            .unwrap();
        let mut ab = start.clone();
        let mut ba = start.clone();
        for pos in [1, 2] {
            game.apply_action(&mut ab, &Connect6Action::Place(pos))
                .unwrap();
        }
        for pos in [2, 1] {
            game.apply_action(&mut ba, &Connect6Action::Place(pos))
                .unwrap();
        }
        assert_eq!(ab, ba);
        let mut a = start.clone();
        let mut b = start.clone();
        game.apply_action(&mut a, &Connect6Action::Place(1))
            .unwrap();
        game.apply_action(&mut b, &Connect6Action::Place(2))
            .unwrap();
        let mut graph = vec![
            GraphNode::new(start.clone(), game.legal_actions(&start)),
            GraphNode::new(a.clone(), game.legal_actions(&a)),
            GraphNode::new(b.clone(), game.legal_actions(&b)),
            GraphNode::new(ab.clone(), game.legal_actions(&ab)),
        ];
        for (parent, pos, child) in [(0, 1, 1), (0, 2, 2), (1, 2, 3), (2, 1, 3)] {
            graph[parent].edges.push(GraphEdge {
                action: Connect6Action::Place(pos),
                child,
                visits: 1,
                heuristic_value: 0.,
                total_utility: 1.,
                total_squared_utility: 1.,
            });
        }
        let mut trace = RaveTrace::new(Some(ActionOps::typed()));
        trace.visit(3, &game, &ab, &mut graph[3].amaf);
        trace.record(PlayerId::FIRST, &Connect6Action::Place(3));
        trace.visit(3, &game, &ba, &mut graph[3].amaf);
        assert_eq!(trace.visited.len(), 1);
        for &(i, p, offset) in &trace.visited {
            trace.update(graph[i].amaf.as_mut().unwrap(), p, offset, 1.);
        }
        assert_eq!(
            trace
                .stats(&graph[3].amaf, &Connect6Action::Place(3))
                .visits,
            1
        );
        compact_graph(&mut graph, 1);
        assert_eq!(graph.len(), 2);
        assert_eq!(
            trace
                .stats(&graph[1].amaf, &Connect6Action::Place(3))
                .visits,
            1
        );
    }
    #[test]
    fn root_choice_and_unvisited_expansion_use_only_normal_statistics() {
        let game = Connect6::new(6).unwrap();
        let state = game.initial_state();
        let mut nodes = vec![Node::new(None, 0., game.legal_actions(&state))];
        let mut agent = MctsAgent::new(config(1));
        agent
            .search_tree(
                &game,
                &state,
                PlayerId::FIRST,
                &mut nodes,
                None,
                true,
                Some(ActionOps::typed()),
                &mut SplitMix64::new(4),
            )
            .unwrap();
        let first = nodes[0].children[0];
        let action = nodes[first].action.unwrap();
        // Force an absurd AMAF advantage for this edge; unexpanded actions still expand.
        nodes[0]
            .amaf
            .as_mut()
            .unwrap()
            .iter_mut()
            .find(|e| e.action == action)
            .unwrap()
            .stats = AmafStats {
            visits: 1_000_000,
            total: 1_000_000.,
        };
        agent
            .search_tree(
                &game,
                &state,
                PlayerId::FIRST,
                &mut nodes,
                None,
                true,
                Some(ActionOps::typed()),
                &mut SplitMix64::new(4),
            )
            .unwrap();
        assert_eq!(nodes[0].children.len(), 2);
        assert_eq!(nodes[first].visits, 1);
        nodes[first].visits = 100;
        nodes[first].total_utility = -100.;
        let chosen = agent
            .search_tree(
                &game,
                &state,
                PlayerId::FIRST,
                &mut nodes,
                None,
                true,
                Some(ActionOps::typed()),
                &mut SplitMix64::new(4),
            )
            .unwrap();
        assert_eq!(chosen, first);
        assert_eq!(nodes[first].visits, 100);
    }
    #[test]
    fn both_backends_run_with_and_without_reuse_and_reject_stochastic_rave() {
        let game = Connect6::new(6).unwrap();
        let state = game.initial_state();
        for reuse in [false, true] {
            for transpositions in [false, true] {
                let mut agent =
                    TranspositionMctsAgent::new(MctsAgent::new(config(80)), reuse, transpositions);
                let a = agent
                    .select_action(
                        DecisionContext::new(&game, &state, PlayerId::FIRST),
                        &mut SplitMix64::new(4),
                    )
                    .unwrap();
                assert!(game.legal_actions(&state).any(|legal| legal == a));
                if transpositions && reuse {
                    assert!(
                        agent.graph.as_ref().unwrap().nodes[0]
                            .amaf
                            .as_ref()
                            .unwrap()
                            .iter()
                            .any(|e| e.stats.visits > 0)
                    );
                }
            }
        }
        let mut stochastic = StochasticMctsAgent::new(config(1), NeutralEvaluator);
        assert!(
            stochastic
                .select_action(
                    DecisionContext::new(&game, &state, PlayerId::FIRST),
                    &mut SplitMix64::new(4)
                )
                .unwrap_err()
                .to_string()
                .contains("deterministic")
        );
    }
}
