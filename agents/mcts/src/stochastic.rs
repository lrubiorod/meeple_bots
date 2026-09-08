//! MCTS with sampled public chance outcomes. Decision edges aggregate all outcomes;
//! chance probabilities are sampled by the game and never optimized with UCB.
use super::{
    DecisionBudget, MctsConfig, NeutralEvaluator, SearchBudget, SelectionPolicy, StateEvaluator,
    UniformRandom, tuned_score,
};
use meeple_bots_core::{
    Agent, AgentDecisionStats, AgentError, DecisionContext, Game, PerfectInformationGame, PlayerId,
    PositionStatus, RandomSource, RootActionStats, TwoPlayerZeroSumGame,
};
use std::{collections::HashMap, hash::Hash};

#[derive(Clone, Debug)]
pub struct StochasticMctsAgent<C = NeutralEvaluator> {
    pub config: MctsConfig<UniformRandom>,
    pub cutoff_evaluator: C,
    pub root_diagnostics: bool,
    stats: AgentDecisionStats,
    budget: DecisionBudget,
}

struct Edge<S, A> {
    action: A,
    visits: u32,
    total: f64,
    squared: f64,
    outcomes: HashMap<S, usize>,
}
struct Node<S, A> {
    visits: u32,
    edges: Vec<Edge<S, A>>,
}

impl<C> StochasticMctsAgent<C> {
    pub fn new(config: MctsConfig<UniformRandom>, cutoff_evaluator: C) -> Self {
        Self {
            config,
            cutoff_evaluator,
            root_diagnostics: false,
            stats: AgentDecisionStats::default(),
            budget: DecisionBudget::new(),
        }
    }
}

fn resolve_chance<G: Game, R: RandomSource + ?Sized>(
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
        game.apply_action(state, &event)
            .map_err(|e| AgentError::message(e.to_string()))?;
    }
    Err(AgentError::message("too many consecutive chance events"))
}

fn node<G: Game>(game: &G, state: &G::State) -> Node<G::State, G::Action> {
    Node {
        visits: 0,
        edges: game
            .legal_actions(state)
            .map(|action| Edge {
                action,
                visits: 0,
                total: 0.0,
                squared: 0.0,
                outcomes: HashMap::new(),
            })
            .collect(),
    }
}

impl<G, C> Agent<G> for StochasticMctsAgent<C>
where
    G: Game + PerfectInformationGame + TwoPlayerZeroSumGame,
    G::State: Clone + Eq + Hash,
    G::Action: Clone,
    C: StateEvaluator<G>,
{
    fn on_match_start(&mut self, _: &G, _: &G::State, _: PlayerId) {
        self.budget = DecisionBudget::new();
    }
    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, G>,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        self.budget.begin(self.config.budget);
        self.stats = AgentDecisionStats::default();
        let result = (|| {
            self.config.validate().map_err(AgentError::message)?;
            let game = decision.game();
            let root = decision.state();
            let owner = decision.player();
            if game.status(root) != PositionStatus::PlayerTurn(owner) {
                return Err(AgentError::message("expected a player decision"));
            }
            let mut nodes = vec![node(game, root)];
            if nodes[0].edges.is_empty() {
                return Err(AgentError::NoLegalActions);
            }
            let mut iterations = 0;
            loop {
                let exhausted = match self.config.budget {
                    SearchBudget::Iterations(n) => iterations >= n.get(),
                    SearchBudget::Time(_) => iterations > 0 && self.budget.exhausted(),
                };
                if exhausted {
                    break;
                }
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
                            if self.config.selection_policy == SelectionPolicy::Ucb1Tuned {
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
                                        * (f64::from(n.visits.max(1)).ln() / f64::from(e.visits))
                                            .sqrt()
                            }
                        };
                        n.edges
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| score(a).total_cmp(&score(b)))
                            .unwrap()
                            .0
                    };
                    game.apply_action(&mut state, &nodes[index].edges[chosen].action)
                        .map_err(|e| AgentError::message(e.to_string()))?;
                    depth += 1;
                    path.push((index, chosen));
                    resolve_chance(game, &mut state, rng)?;
                    if game.status(&state) == PositionStatus::Terminal {
                        break;
                    }
                    if let Some(&child) = nodes[index].edges[chosen].outcomes.get(&state) {
                        index = child;
                    } else {
                        let child = nodes.len();
                        nodes.push(node(game, &state));
                        nodes[index].edges[chosen]
                            .outcomes
                            .insert(state.clone(), child);
                        break;
                    }
                }
                // Uniform rollouts; depth counts player actions, not sampled dice events.
                while depth < self.config.rollout_depth
                    && game.status(&state) != PositionStatus::Terminal
                {
                    resolve_chance(game, &mut state, rng)?;
                    if game.status(&state) == PositionStatus::Terminal {
                        break;
                    }
                    let actions: Vec<_> = game.legal_actions(&state).collect();
                    let action =
                        &actions[rng.index(actions.len()).ok_or(AgentError::NoLegalActions)?];
                    game.apply_action(&mut state, action)
                        .map_err(|e| AgentError::message(e.to_string()))?;
                    depth += 1;
                }
                resolve_chance(game, &mut state, rng)?;
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
                search_iterations: Some(u64::from(iterations)),
                search_nodes: Some(nodes.len() as u64),
                root_actions: if self.root_diagnostics {
                    nodes[0]
                        .edges
                        .iter()
                        .enumerate()
                        .map(|(i, e)| RootActionStats {
                            action_index: i as u32,
                            visits: e.visits,
                            mean_utility: e.total / f64::from(e.visits.max(1)),
                            heuristic_value: None,
                            progressive_bias: None,
                            selected: i == chosen,
                        })
                        .collect()
                } else {
                    Vec::new()
                },
                tree_reuse: None,
            };
            Ok(nodes[0].edges[chosen].action.clone())
        })();
        self.budget.finish();
        result
    }
    fn last_decision_stats(&self) -> AgentDecisionStats {
        self.stats.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use meeple_bots_core::IllegalAction;
    use meeple_bots_simulation::SplitMix64;
    use std::num::NonZeroU32;
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
    fn chance_is_sampled_not_maximized_and_search_does_not_mutate_state() {
        for policy in [SelectionPolicy::Uct, SelectionPolicy::Ucb1Tuned] {
            let mut a = StochasticMctsAgent::new(
                MctsConfig {
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
