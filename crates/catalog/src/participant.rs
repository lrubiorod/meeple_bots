//! Independently configured participants; dispatch occurs at agent boundaries, not inside search.

use crate::{AgentConfig, CatalogError, MctsAgentConfig};
use meeple_bots_core::{
    Agent, AgentDecisionStats, AgentError, DecisionContext, Game, PlayerId, RandomSource,
};
use meeple_bots_random_agent::RandomAgent;

/// An automated participant with a statically typed search implementation.
#[derive(Clone, Debug)]
pub enum ConfiguredAgent<M> {
    Random(RandomAgent),
    Mcts(M),
}

impl<M> ConfiguredAgent<M> {
    pub fn new(
        config: AgentConfig,
        mcts: impl FnOnce(MctsAgentConfig) -> Result<M, CatalogError>,
    ) -> Result<Self, CatalogError> {
        match config {
            AgentConfig::SoIsmcts(_) => Err(CatalogError::InvalidMctsConfig(
                "SO-ISMCTS is only supported by Lost Cities",
            )),
            AgentConfig::Random => Ok(Self::Random(RandomAgent)),
            AgentConfig::Mcts(config) => mcts(config).map(Self::Mcts),
        }
    }
}

impl<G: Game, M: Agent<G>> Agent<G> for ConfiguredAgent<M> {
    fn on_match_start(&mut self, game: &G, state: &G::State, player: PlayerId) {
        match self {
            Self::Random(agent) => {
                <RandomAgent as Agent<G>>::on_match_start(agent, game, state, player)
            }
            Self::Mcts(agent) => agent.on_match_start(game, state, player),
        }
    }

    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, G>,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        match self {
            Self::Random(agent) => agent.select_action(decision, rng),
            Self::Mcts(agent) => agent.select_action(decision, rng),
        }
    }

    fn last_decision_stats(&self) -> AgentDecisionStats {
        match self {
            Self::Random(agent) => <RandomAgent as Agent<G>>::last_decision_stats(agent),
            Self::Mcts(agent) => agent.last_decision_stats(),
        }
    }

    fn on_action_applied(
        &mut self,
        game: &G,
        state: &G::State,
        player: PlayerId,
        action: &G::Action,
    ) {
        match self {
            Self::Random(agent) => {
                <RandomAgent as Agent<G>>::on_action_applied(agent, game, state, player, action)
            }
            Self::Mcts(agent) => agent.on_action_applied(game, state, player, action),
        }
    }

    fn on_chance_applied(&mut self, game: &G, state: &G::State, event: &G::Action) {
        match self {
            Self::Random(agent) => agent.on_chance_applied(game, state, event),
            Self::Mcts(agent) => agent.on_chance_applied(game, state, event),
        }
    }

    fn on_match_end(&mut self, game: &G, state: &G::State) {
        match self {
            Self::Random(agent) => <RandomAgent as Agent<G>>::on_match_end(agent, game, state),
            Self::Mcts(agent) => agent.on_match_end(game, state),
        }
    }
}
