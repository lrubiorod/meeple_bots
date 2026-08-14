//! Monte Carlo Tree Search for deterministic, perfect-information games.

use std::num::NonZeroU32;

use meeple_bots_core::{
    Agent, AgentError, DecisionContext, DeterministicGame, PerfectInformationGame, PlayerId,
    PositionStatus, RandomSource, TwoPlayerZeroSumGame,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MctsConfig {
    pub iterations: NonZeroU32,
    pub exploration: f64,
    pub rollout_depth: u32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            iterations: NonZeroU32::new(1_000).expect("constant is non-zero"),
            exploration: std::f64::consts::SQRT_2,
            rollout_depth: 256,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MctsAgent {
    pub config: MctsConfig,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MctsSearchStats {
    pub iterations: u32,
    pub expanded_nodes: u32,
    pub root_actions: u32,
    pub maximum_tree_depth: u32,
    pub mean_tree_depth: f64,
    pub maximum_simulation_depth: u32,
    pub mean_simulation_depth: f64,
    pub terminal_rollouts: u32,
    pub truncated_rollouts: u32,
    pub selected_action_visits: u32,
    pub selected_action_visit_share: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MctsDecision<A> {
    pub action: A,
    pub stats: MctsSearchStats,
}

impl MctsAgent {
    pub const fn new(config: MctsConfig) -> Self {
        Self { config }
    }

    pub fn select_action_with_stats<G, R>(
        &mut self,
        decision: DecisionContext<'_, G>,
        rng: &mut R,
    ) -> Result<MctsDecision<G::Action>, AgentError>
    where
        G: DeterministicGame + PerfectInformationGame + TwoPlayerZeroSumGame,
        G::State: Clone,
        G::Action: Clone,
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
        let root_action_count = root_actions.len() as u32;

        let mut nodes = vec![Node::new(None, root_actions)];
        let mut tree_depth_sum = 0_u64;
        let mut simulation_depth_sum = 0_u64;
        let mut maximum_tree_depth = 0;
        let mut maximum_simulation_depth = 0;
        let mut terminal_rollouts = 0;
        let mut truncated_rollouts = 0;

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

            let tree_depth = (path.len() - 1) as u32;
            let rollout = rollout(
                game,
                &mut state,
                root_player,
                self.config.rollout_depth,
                rng,
            )?;
            let simulation_depth = tree_depth.saturating_add(rollout.plies);
            tree_depth_sum += u64::from(tree_depth);
            simulation_depth_sum += u64::from(simulation_depth);
            maximum_tree_depth = maximum_tree_depth.max(tree_depth);
            maximum_simulation_depth = maximum_simulation_depth.max(simulation_depth);
            if rollout.terminal {
                terminal_rollouts += 1;
            } else {
                truncated_rollouts += 1;
            }
            for visited in path {
                nodes[visited].visits += 1;
                nodes[visited].total_utility += rollout.utility;
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
        let iterations = self.config.iterations.get();
        let selected_action_visits = nodes[selected_index].visits;

        Ok(MctsDecision {
            action: nodes[selected_index]
                .action
                .clone()
                .ok_or(AgentError::NoLegalActions)?,
            stats: MctsSearchStats {
                iterations,
                expanded_nodes: (nodes.len() - 1) as u32,
                root_actions: root_action_count,
                maximum_tree_depth,
                mean_tree_depth: tree_depth_sum as f64 / f64::from(iterations),
                maximum_simulation_depth,
                mean_simulation_depth: simulation_depth_sum as f64 / f64::from(iterations),
                terminal_rollouts,
                truncated_rollouts,
                selected_action_visits,
                selected_action_visit_share: f64::from(selected_action_visits)
                    / f64::from(iterations),
            },
        })
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

impl<G> Agent<G> for MctsAgent
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
        self.select_action_with_stats(decision, rng)
            .map(|decision| decision.action)
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

fn rollout<G, R>(
    game: &G,
    state: &mut G::State,
    root_player: PlayerId,
    max_depth: u32,
    rng: &mut R,
) -> Result<RolloutResult, AgentError>
where
    G: DeterministicGame,
    G::Action: Clone,
    R: RandomSource + ?Sized,
{
    for plies in 0..max_depth {
        match game.status(state) {
            PositionStatus::Terminal => {
                return game
                    .terminal_utility(state, root_player)
                    .map(|utility| RolloutResult {
                        utility: f64::from(utility),
                        plies,
                        terminal: true,
                    })
                    .ok_or_else(|| AgentError::message("terminal utility is missing"));
            }
            PositionStatus::PlayerTurn(_) => {
                let actions: Vec<_> = game.legal_actions(state).collect();
                let index = rng.index(actions.len()).ok_or(AgentError::NoLegalActions)?;
                game.apply_action(state, &actions[index])
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
            .map(|utility| RolloutResult {
                utility: f64::from(utility),
                plies: max_depth,
                terminal: true,
            })
            .ok_or_else(|| AgentError::message("terminal utility is missing")),
        _ => Ok(RolloutResult {
            utility: 0.0,
            plies: max_depth,
            terminal: false,
        }),
    }
}

struct RolloutResult {
    utility: f64,
    plies: u32,
    terminal: bool,
}

#[cfg(test)]
mod tests {
    use meeple_bots_core::Game;
    use meeple_bots_random_agent::RandomAgent;
    use meeple_bots_simulation::{MatchConfig, SplitMix64, play_match};
    use meeple_bots_tic_tac_toe::{TicTacToe, TicTacToeAction};

    use super::*;

    fn action(index: u8) -> TicTacToeAction {
        TicTacToeAction::from_index(index).unwrap()
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
    fn reports_search_work_and_rollout_outcomes() {
        let game = TicTacToe;
        let state = game.initial_state();
        let mut agent = MctsAgent::new(MctsConfig {
            iterations: NonZeroU32::new(32).unwrap(),
            rollout_depth: 1,
            ..MctsConfig::default()
        });
        let decision = agent
            .select_action_with_stats(
                DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut SplitMix64::new(5),
            )
            .unwrap();

        assert_eq!(decision.stats.iterations, 32);
        assert_eq!(decision.stats.root_actions, 9);
        assert!(decision.stats.expanded_nodes <= 32);
        assert_eq!(
            decision.stats.terminal_rollouts + decision.stats.truncated_rollouts,
            decision.stats.iterations
        );
        assert!(decision.stats.maximum_tree_depth >= 1);
        assert!(decision.stats.maximum_simulation_depth >= 2);
        assert!((0.0..=1.0).contains(&decision.stats.selected_action_visit_share));
    }
}
