//! Deterministic action admission, independent of the widening limit and selection.
use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WideningExpansionPolicy {
    #[default]
    Random,
    Rave,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WideningStats {
    pub expansions_total: u64,
    pub expansions_random: u64,
    pub expansions_rave_guided: u64,
    pub rave_fallbacks_no_amaf: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProgressiveWidening {
    pub k: f64,
    pub alpha: f64,
    pub expansion: WideningExpansionPolicy,
}

impl Default for ProgressiveWidening {
    fn default() -> Self {
        Self {
            k: 1.5,
            alpha: 0.5,
            expansion: WideningExpansionPolicy::Random,
        }
    }
}
impl ProgressiveWidening {
    pub fn validate(self) -> Result<(), &'static str> {
        if !self.k.is_finite() || self.k <= 0.0 {
            return Err("progressive_widening_k must be finite and greater than zero");
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 || self.alpha > 1.0 {
            return Err("progressive_widening_alpha must be finite and in (0, 1]");
        }
        Ok(())
    }
    /// N is the node's completed normal visits before this simulation's backup.
    /// Zero legal actions gives zero; otherwise N=0 permits exactly one child.
    pub fn limit(self, visits: u32, legal: usize) -> usize {
        ((self.k * f64::from(visits).powf(self.alpha)).floor() as usize)
            .max(1)
            .min(legal)
    }
}
pub(super) fn can_expand(
    pw: Option<ProgressiveWidening>,
    visits: u32,
    active: usize,
    pending: usize,
) -> bool {
    pending > 0 && pw.is_none_or(|pw| active < pw.limit(visits, active + pending))
}

/// Called only after the PW gate allows admission. AMAF totals use root utility,
/// so opponent nodes minimize the stored mean, just like RAVE selection.
pub(super) fn choose_action<A, R: RandomSource + ?Sized>(
    pw: Option<ProgressiveWidening>,
    pending: &[A],
    table: &Option<Vec<AmafEdge<A>>>,
    trace: &RaveTrace<A>,
    maximizing: bool,
    rng: &mut R,
    stats: &mut WideningStats,
) -> usize {
    if let Some(pw) = pw {
        stats.expansions_total += 1;
        if pw.expansion == WideningExpansionPolicy::Rave {
            let mut best = f64::NEG_INFINITY;
            let mut ties = Vec::new();
            for (index, action) in pending.iter().enumerate() {
                if let Some(mean) = trace.stats(table, action).mean() {
                    let score = if maximizing { mean } else { -mean };
                    if score > best {
                        best = score;
                        ties.clear();
                    }
                    if score == best {
                        ties.push(index);
                    }
                }
            }
            if !ties.is_empty() {
                stats.expansions_rave_guided += 1;
                return ties[rng.index(ties.len()).expect("nonempty ties")];
            }
            stats.rave_fallbacks_no_amaf += 1;
        }
        stats.expansions_random += 1;
    }
    rng.index(pending.len()).expect("nonempty pending actions")
}

#[cfg(test)]
mod tests {
    use super::*;
    fn guided() -> Option<ProgressiveWidening> {
        Some(ProgressiveWidening {
            expansion: WideningExpansionPolicy::Rave,
            ..Default::default()
        })
    }
    fn table() -> Option<Vec<AmafEdge<u8>>> {
        Some(vec![
            AmafEdge {
                action: 0,
                stats: AmafStats {
                    visits: 1,
                    total: 99.,
                },
            },
            AmafEdge {
                action: 1,
                stats: AmafStats {
                    visits: 20,
                    total: -10.,
                },
            },
            AmafEdge {
                action: 2,
                stats: AmafStats {
                    visits: 5,
                    total: -1.,
                },
            },
            AmafEdge {
                action: 3,
                stats: AmafStats::default(),
            },
        ])
    }
    #[test]
    fn guided_best_mean_partial_negative_samples_and_player_perspective() {
        let trace = RaveTrace::new(Some(ActionOps::typed()));
        let mut stats = WideningStats::default();
        let pending = [1, 2, 3]; // action 0 is already expanded, cannot enter again
        for seed in 0..10 {
            assert_eq!(
                choose_action(
                    guided(),
                    &pending,
                    &table(),
                    &trace,
                    true,
                    &mut SplitMix64::new(seed),
                    &mut stats
                ),
                1
            );
            assert_eq!(
                choose_action(
                    guided(),
                    &pending,
                    &table(),
                    &trace,
                    false,
                    &mut SplitMix64::new(seed),
                    &mut stats
                ),
                0
            );
        }
        assert_eq!(stats.expansions_rave_guided, 20);
        assert_eq!(stats.expansions_random, 0);
    }
    #[test]
    fn guided_ties_are_seeded_random_and_node_local_fallback_matches_random() {
        let trace = RaveTrace::new(Some(ActionOps::typed()));
        let tied = Some(vec![
            AmafEdge {
                action: 1u8,
                stats: AmafStats {
                    visits: 2,
                    total: 1.,
                },
            },
            AmafEdge {
                action: 2,
                stats: AmafStats {
                    visits: 4,
                    total: 2.,
                },
            },
        ]);
        let choose = |seed| {
            choose_action(
                guided(),
                &[1, 2, 3],
                &tied,
                &trace,
                true,
                &mut SplitMix64::new(seed),
                &mut WideningStats::default(),
            )
        };
        let choices: Vec<_> = (0..30).map(choose).collect();
        assert_eq!(choices, (0..30).map(choose).collect::<Vec<_>>());
        assert!(choices.contains(&0) && choices.contains(&1));
        assert!(!choices.contains(&2));
        for seed in 0..10 {
            let mut stats = WideningStats::default();
            let fallback = choose_action(
                guided(),
                &[1, 2, 3],
                &None,
                &trace,
                true,
                &mut SplitMix64::new(seed),
                &mut stats,
            );
            let random = choose_action(
                Some(ProgressiveWidening::default()),
                &[1, 2, 3],
                &tied,
                &trace,
                true,
                &mut SplitMix64::new(seed),
                &mut WideningStats::default(),
            );
            assert_eq!(fallback, random);
            assert_eq!(stats.rave_fallbacks_no_amaf, 1);
            assert_eq!(stats.expansions_random, 1);
            assert_eq!(stats.expansions_total, 1);
        }
    }
    #[test]
    fn formula_and_validation() {
        let pw = ProgressiveWidening {
            k: 2.0,
            alpha: 0.5,
            ..Default::default()
        };
        assert_eq!([0, 1, 4, 9, 16].map(|n| pw.limit(n, 100)), [1, 2, 4, 6, 8]);
        assert_eq!(pw.limit(100, 3), 3);
        assert_eq!(pw.limit(0, 0), 0);
        for n in 0..1000 {
            assert!(pw.limit(n, 100) <= pw.limit(n + 1, 100));
        }
        for k in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(ProgressiveWidening { k, ..pw }.validate().is_err());
        }
        for alpha in [0.0, -1.0, 1.1, f64::NAN, f64::INFINITY] {
            assert!(ProgressiveWidening { alpha, ..pw }.validate().is_err());
        }
        assert!(ProgressiveWidening { alpha: 1.0, ..pw }.validate().is_ok());
    }
    #[test]
    fn admission_boundary_and_disabled() {
        let pw = Some(ProgressiveWidening {
            k: 1.0,
            alpha: 0.5,
            ..Default::default()
        });
        assert!(can_expand(pw, 0, 0, 361));
        assert!(!can_expand(pw, 3, 1, 360));
        assert!(can_expand(pw, 4, 1, 360));
        assert!(!can_expand(pw, 4, 2, 359));
        assert!(can_expand(None, 4, 2, 359));
        assert!(!can_expand(None, 4, 2, 0));
    }
    use meeple_bots_connect6::{Connect6, Connect6Action};
    use meeple_bots_core::Game;
    use meeple_bots_simulation::SplitMix64;

    fn config(selection_policy: SelectionPolicy) -> MctsConfig<UniformRandom> {
        MctsConfig {
            budget: SearchBudget::Iterations(NonZeroU32::new(1).unwrap()),
            selection_policy,
            progressive_widening: Some(ProgressiveWidening {
                k: 1.,
                alpha: 0.5,
                ..Default::default()
            }),
            exploration: 1.4,
            rollout_depth: 2,
            rollout_policy: UniformRandom,
        }
    }
    #[test]
    fn connect6_all_selectors_admit_one_at_a_time_and_select_when_full() {
        let game = Connect6::default();
        let state = game.initial_state();
        for expansion in [
            WideningExpansionPolicy::Random,
            WideningExpansionPolicy::Rave,
        ] {
            for policy in [
                SelectionPolicy::Uct,
                SelectionPolicy::Ucb1Tuned,
                SelectionPolicy::UctRave {
                    rave_equivalence: 1000,
                },
            ] {
                let mut nodes = vec![Node::new(None, 0., game.legal_actions(&state))];
                let mut agent = MctsAgent::new(config(policy));
                agent
                    .config
                    .progressive_widening
                    .as_mut()
                    .unwrap()
                    .expansion = expansion;
                let mut rng = SplitMix64::new(42);
                for _ in 0..25 {
                    let before = nodes[0].children.len();
                    let admit = can_expand(
                        agent.config.progressive_widening,
                        nodes[0].visits,
                        before,
                        nodes[0].unexpanded.len(),
                    );
                    let chosen = agent
                        .search_tree(
                            &game,
                            &state,
                            PlayerId::FIRST,
                            &mut nodes,
                            None,
                            true,
                            agent.config.collects_amaf().then(ActionOps::typed),
                            &mut rng,
                        )
                        .unwrap();
                    assert_eq!(nodes[0].children.len(), before + usize::from(admit));
                    assert!(nodes[0].children.contains(&chosen));
                    assert!(
                        nodes[0].children.len()
                            <= agent
                                .config
                                .progressive_widening
                                .unwrap()
                                .limit(nodes[0].visits, 361)
                    );
                }
                assert_eq!(nodes[0].children.len(), 4); // opening fifth slot uses N=25 on next visit
                assert_eq!(nodes[0].unexpanded.len(), 357);
                assert_eq!(
                    nodes[0]
                        .children
                        .iter()
                        .map(|i| nodes[*i].visits)
                        .sum::<u32>(),
                    25
                );
            }
        }
    }
    #[test]
    fn pending_amaf_does_not_admit_or_win_and_disabled_expands() {
        let game = Connect6::new(6).unwrap();
        let mut state = game.initial_state();
        game.apply_action(&mut state, &Connect6Action::Place(0))
            .unwrap();
        let mut nodes = vec![Node::new(None, 0., game.legal_actions(&state))];
        let mut agent = MctsAgent::new(config(SelectionPolicy::UctRave {
            rave_equivalence: 1000,
        }));
        agent
            .config
            .progressive_widening
            .as_mut()
            .unwrap()
            .expansion = WideningExpansionPolicy::Rave;
        let mut rng = SplitMix64::new(2);
        let chosen = agent
            .search_tree(
                &game,
                &state,
                PlayerId::SECOND,
                &mut nodes,
                None,
                true,
                Some(ActionOps::typed()),
                &mut rng,
            )
            .unwrap();
        let pending = nodes[0]
            .unexpanded
            .iter()
            .find(|a| {
                nodes[0]
                    .amaf
                    .as_ref()
                    .unwrap()
                    .iter()
                    .any(|e| &e.action == *a && e.stats.visits > 0)
            })
            .copied()
            .unwrap();
        nodes[0]
            .amaf
            .as_mut()
            .unwrap()
            .iter_mut()
            .find(|e| e.action == pending)
            .unwrap()
            .stats = AmafStats {
            visits: 1_000_000,
            total: 1_000_000.,
        };
        let selected = agent
            .search_tree(
                &game,
                &state,
                PlayerId::SECOND,
                &mut nodes,
                None,
                true,
                Some(ActionOps::typed()),
                &mut rng,
            )
            .unwrap();
        assert_eq!(selected, chosen);
        assert!(nodes[0].unexpanded.contains(&pending));
        assert_eq!(nodes[0].children.len(), 1);
        assert_eq!(nodes[chosen].visits, 2);
        agent.config.progressive_widening = None;
        agent
            .search_tree(
                &game,
                &state,
                PlayerId::SECOND,
                &mut nodes,
                None,
                true,
                Some(ActionOps::typed()),
                &mut rng,
            )
            .unwrap();
        assert_eq!(nodes[0].children.len(), 2);
    }
    #[test]
    fn random_admission_is_reproducible_and_not_first_legal() {
        let game = Connect6::default();
        let state = game.initial_state();
        let run = |seed| {
            MctsAgent::new(config(SelectionPolicy::Uct))
                .select_action(
                    DecisionContext::new(&game, &state, PlayerId::FIRST),
                    &mut SplitMix64::new(seed),
                )
                .unwrap()
        };
        assert_eq!(run(42), run(42));
        let actions: Vec<_> = (0..8).map(run).collect();
        assert!(actions.iter().any(|a| *a != actions[0]));
        assert!(
            actions
                .iter()
                .any(|a| *a != game.legal_actions(&state).next().unwrap())
        );
    }
    #[test]
    fn reroot_preserves_admission_and_graph_nodes_share_progression() {
        let game = Connect6::new(6).unwrap();
        let state = game.initial_state();
        for (transpositions, expansion) in [
            (false, WideningExpansionPolicy::Random),
            (true, WideningExpansionPolicy::Random),
            (false, WideningExpansionPolicy::Rave),
            (true, WideningExpansionPolicy::Rave),
        ] {
            let mut cfg = config(SelectionPolicy::UctRave {
                rave_equivalence: 1000,
            });
            cfg.progressive_widening.as_mut().unwrap().expansion = expansion;
            cfg.budget = SearchBudget::Iterations(NonZeroU32::new(100).unwrap());
            let mut agent = TranspositionMctsAgent::new(MctsAgent::new(cfg), true, transpositions);
            let action = agent
                .select_action(
                    DecisionContext::new(&game, &state, PlayerId::FIRST),
                    &mut SplitMix64::new(42),
                )
                .unwrap();
            let saved = if transpositions {
                let g = agent.graph.as_ref().unwrap();
                let e = g.nodes[0]
                    .edges
                    .iter()
                    .find(|e| e.action == action)
                    .unwrap();
                let n = &g.nodes[e.child];
                (
                    n.visits,
                    n.edges.iter().map(|e| e.action).collect::<Vec<_>>(),
                    n.unexpanded.clone(),
                    n.amaf
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|e| (e.action, e.stats))
                        .collect::<Vec<_>>(),
                )
            } else {
                let t = agent.basic.tree.as_ref().unwrap();
                let i = *t.nodes[0]
                    .children
                    .iter()
                    .find(|i| t.nodes[**i].action == Some(action))
                    .unwrap();
                let n = &t.nodes[i];
                (
                    n.visits,
                    n.children
                        .iter()
                        .map(|i| t.nodes[*i].action.unwrap())
                        .collect(),
                    n.unexpanded.clone(),
                    n.amaf
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|e| (e.action, e.stats))
                        .collect::<Vec<_>>(),
                )
            };
            assert!(saved.0 > 1 && !saved.1.is_empty() && !saved.2.is_empty());
            let mut next = state.clone();
            game.apply_action(&mut next, &action).unwrap();
            agent.on_action_applied(&game, &next, PlayerId::FIRST, &action);
            let retained = if transpositions {
                let g = agent.graph.as_ref().unwrap();
                let n = &g.nodes[0];
                (
                    n.visits,
                    n.edges.iter().map(|e| e.action).collect::<Vec<_>>(),
                    n.unexpanded.clone(),
                    n.amaf
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|e| (e.action, e.stats))
                        .collect::<Vec<_>>(),
                )
            } else {
                let t = agent.basic.tree.as_ref().unwrap();
                let n = &t.nodes[0];
                (
                    n.visits,
                    n.children
                        .iter()
                        .map(|i| t.nodes[*i].action.unwrap())
                        .collect(),
                    n.unexpanded.clone(),
                    n.amaf
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|e| (e.action, e.stats))
                        .collect::<Vec<_>>(),
                )
            };
            assert_eq!(saved, retained);
        }
    }
}
