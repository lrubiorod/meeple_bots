//! Runtime configuration boundary for the statically dispatched engine.

use std::{error::Error, fmt, num::NonZeroU32};

use meeple_bots_mcts_agent::{MctsAgent, MctsConfig};
use meeple_bots_random_agent::RandomAgent;
use meeple_bots_simulation::{
    BatchConfig, MatchConfig, MatchError, MatchResult, play_batch, play_match,
};
use meeple_bots_tic_tac_toe::TicTacToe;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GameId {
    TicTacToe,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AgentConfig {
    Random,
    Mcts(MctsConfig),
}

#[derive(Debug)]
pub struct CatalogError(MatchError);

impl fmt::Display for CatalogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl Error for CatalogError {}

impl From<MatchError> for CatalogError {
    fn from(error: MatchError) -> Self {
        Self(error)
    }
}

pub fn run_match(
    game: GameId,
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<MatchResult, CatalogError> {
    match game {
        GameId::TicTacToe => run_tic_tac_toe(first, second, config),
    }
}

fn run_tic_tac_toe(
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<MatchResult, CatalogError> {
    let game = TicTacToe;
    let result = match (first, second) {
        (AgentConfig::Random, AgentConfig::Random) => {
            play_match(&game, &mut RandomAgent, &mut RandomAgent, config)
        }
        (AgentConfig::Random, AgentConfig::Mcts(second)) => {
            play_match(&game, &mut RandomAgent, &mut MctsAgent::new(second), config)
        }
        (AgentConfig::Mcts(first), AgentConfig::Random) => {
            play_match(&game, &mut MctsAgent::new(first), &mut RandomAgent, config)
        }
        (AgentConfig::Mcts(first), AgentConfig::Mcts(second)) => play_match(
            &game,
            &mut MctsAgent::new(first),
            &mut MctsAgent::new(second),
            config,
        ),
    }?;
    Ok(result)
}

pub fn run_batch(
    game: GameId,
    first: AgentConfig,
    second: AgentConfig,
    seed: u64,
    matches: NonZeroU32,
    max_plies: NonZeroU32,
) -> Result<Vec<MatchResult>, CatalogError> {
    let config = BatchConfig {
        seed,
        matches,
        max_plies,
    };
    let game = match game {
        GameId::TicTacToe => TicTacToe,
    };

    let results = match (first, second) {
        (AgentConfig::Random, AgentConfig::Random) => {
            play_batch(&game, config, || RandomAgent, || RandomAgent)
        }
        (AgentConfig::Random, AgentConfig::Mcts(second)) => {
            play_batch(&game, config, || RandomAgent, || MctsAgent::new(second))
        }
        (AgentConfig::Mcts(first), AgentConfig::Random) => {
            play_batch(&game, config, || MctsAgent::new(first), || RandomAgent)
        }
        (AgentConfig::Mcts(first), AgentConfig::Mcts(second)) => play_batch(
            &game,
            config,
            || MctsAgent::new(first),
            || MctsAgent::new(second),
        ),
    }?;
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_catalog_dispatches_outside_the_match_loop() {
        let result = run_match(
            GameId::TicTacToe,
            AgentConfig::Random,
            AgentConfig::Random,
            MatchConfig::default(),
        )
        .unwrap();

        assert_eq!(result.utilities.len(), 2);
        assert!((5..=9).contains(&result.plies));
    }
}
