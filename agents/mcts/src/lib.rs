//! Monte Carlo Tree Search for deterministic, perfect-information games.

use std::num::NonZeroU32;

use meeple_bots_core::{
    Agent, AgentError, DecisionContext, DeterministicGame, HeuristicGame, PerfectInformationGame,
    PlayerId, PositionStatus, RandomSource, TwoPlayerZeroSumGame,
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

pub trait CutoffEvaluator<G: DeterministicGame> {
    fn evaluate(
        &self,
        game: &G,
        state: &G::State,
        root_player: PlayerId,
    ) -> Result<f64, AgentError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NeutralEvaluator;

impl<G: DeterministicGame> CutoffEvaluator<G> for NeutralEvaluator {
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

impl<G> CutoffEvaluator<G> for GameHeuristic
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

#[derive(Clone, Copy, Debug)]
pub struct MctsAgent<E = NeutralEvaluator> {
    pub config: MctsConfig,
    pub evaluator: E,
}

impl Default for MctsAgent<NeutralEvaluator> {
    fn default() -> Self {
        Self::new(MctsConfig::default())
    }
}

impl MctsAgent<NeutralEvaluator> {
    pub const fn new(config: MctsConfig) -> Self {
        Self {
            config,
            evaluator: NeutralEvaluator,
        }
    }
}

impl<E> MctsAgent<E> {
    pub const fn with_evaluator(config: MctsConfig, evaluator: E) -> Self {
        Self { config, evaluator }
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
        E: CutoffEvaluator<G>,
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
                &self.evaluator,
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

impl<G, E> Agent<G> for MctsAgent<E>
where
    G: DeterministicGame + PerfectInformationGame + TwoPlayerZeroSumGame,
    G::State: Clone,
    G::Action: Clone,
    E: CutoffEvaluator<G>,
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

fn rollout<G, E, R>(
    game: &G,
    state: &mut G::State,
    root_player: PlayerId,
    max_depth: u32,
    evaluator: &E,
    rng: &mut R,
) -> Result<f64, AgentError>
where
    G: DeterministicGame,
    G::Action: Clone,
    E: CutoffEvaluator<G>,
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
            .map(f64::from)
            .ok_or_else(|| AgentError::message("terminal utility is missing")),
        _ => {
            let utility = evaluator.evaluate(game, state, root_player)?;
            if !utility.is_finite() || !(-1.0..=1.0).contains(&utility) {
                return Err(AgentError::message(
                    "MCTS heuristic utility must be finite and between -1.0 and 1.0",
                ));
            }
            Ok(utility)
        }
    }
}

#[cfg(test)]
mod tests {
    use meeple_bots_core::Game;
    use meeple_bots_random_agent::RandomAgent;
    use meeple_bots_simulation::{MatchConfig, SplitMix64, play_match};
    use meeple_bots_tic_tac_toe::{TicTacToe, TicTacToeAction};

    use super::*;

    #[derive(Clone, Copy)]
    struct FixedEvaluator(f64);

    impl CutoffEvaluator<TicTacToe> for FixedEvaluator {
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

    impl CutoffEvaluator<TicTacToe> for FailingEvaluator {
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
    fn cutoff_uses_the_configured_evaluator() {
        let game = TicTacToe;
        let mut state = game.initial_state();
        let result = rollout(
            &game,
            &mut state,
            PlayerId::FIRST,
            0,
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
                &FixedEvaluator(invalid),
                &mut SplitMix64::new(3),
            )
            .unwrap_err();

            assert!(error.to_string().contains("between -1.0 and 1.0"));
        }
    }
}
