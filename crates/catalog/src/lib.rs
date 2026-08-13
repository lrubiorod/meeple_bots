//! Runtime configuration boundary for the statically dispatched engine.

use std::{error::Error, fmt, num::NonZeroU32};

use meeple_bots_connect_four::{ConnectFour, ConnectFourAction};
use meeple_bots_core::Agent;
use meeple_bots_mcts_agent::MctsAgent;
pub use meeple_bots_mcts_agent::MctsConfig;
use meeple_bots_random_agent::RandomAgent;
use meeple_bots_simulation::{
    BatchConfig, MatchError, TracedMatchResult, play_batch, play_match,
    play_match_with_trace as play_typed_match_with_trace,
};
pub use meeple_bots_simulation::{MatchConfig, MatchResult};
use meeple_bots_tic_tac_toe::{TicTacToe, TicTacToeAction};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GameId {
    ConnectFour,
    TicTacToe,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AgentConfig {
    Random,
    Mcts(MctsConfig),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CatalogAction {
    ConnectFour { column: u8 },
    TicTacToe { row: u8, column: u8 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecordedMove {
    pub player: usize,
    pub action: CatalogAction,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CatalogMatchReport {
    pub seed: u64,
    pub plies: u32,
    pub utilities: Vec<f32>,
    pub winner: Option<usize>,
    pub moves: Vec<RecordedMove>,
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
        GameId::ConnectFour => run_connect_four(first, second, config),
        GameId::TicTacToe => run_tic_tac_toe(first, second, config),
    }
}

pub fn run_match_with_trace(
    game: GameId,
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError> {
    match game {
        GameId::ConnectFour => run_connect_four_with_trace(first, second, config),
        GameId::TicTacToe => run_tic_tac_toe_with_trace(first, second, config),
    }
}

pub fn run_connect_four_match_with_trace<A, B>(
    first: &mut A,
    second: &mut B,
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError>
where
    A: Agent<ConnectFour>,
    B: Agent<ConnectFour>,
{
    let traced = play_typed_match_with_trace(&ConnectFour, first, second, config)?;
    Ok(connect_four_report(traced))
}

pub fn run_tic_tac_toe_match_with_trace<A, B>(
    first: &mut A,
    second: &mut B,
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError>
where
    A: Agent<TicTacToe>,
    B: Agent<TicTacToe>,
{
    let traced = play_typed_match_with_trace(&TicTacToe, first, second, config)?;
    Ok(tic_tac_toe_report(traced))
}

fn run_connect_four(
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<MatchResult, CatalogError> {
    let game = ConnectFour;
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

fn run_connect_four_with_trace(
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError> {
    match (first, second) {
        (AgentConfig::Random, AgentConfig::Random) => {
            run_connect_four_match_with_trace(&mut RandomAgent, &mut RandomAgent, config)
        }
        (AgentConfig::Random, AgentConfig::Mcts(second)) => {
            run_connect_four_match_with_trace(&mut RandomAgent, &mut MctsAgent::new(second), config)
        }
        (AgentConfig::Mcts(first), AgentConfig::Random) => {
            run_connect_four_match_with_trace(&mut MctsAgent::new(first), &mut RandomAgent, config)
        }
        (AgentConfig::Mcts(first), AgentConfig::Mcts(second)) => run_connect_four_match_with_trace(
            &mut MctsAgent::new(first),
            &mut MctsAgent::new(second),
            config,
        ),
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

fn run_tic_tac_toe_with_trace(
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError> {
    match (first, second) {
        (AgentConfig::Random, AgentConfig::Random) => {
            run_tic_tac_toe_match_with_trace(&mut RandomAgent, &mut RandomAgent, config)
        }
        (AgentConfig::Random, AgentConfig::Mcts(second)) => {
            run_tic_tac_toe_match_with_trace(&mut RandomAgent, &mut MctsAgent::new(second), config)
        }
        (AgentConfig::Mcts(first), AgentConfig::Random) => {
            run_tic_tac_toe_match_with_trace(&mut MctsAgent::new(first), &mut RandomAgent, config)
        }
        (AgentConfig::Mcts(first), AgentConfig::Mcts(second)) => run_tic_tac_toe_match_with_trace(
            &mut MctsAgent::new(first),
            &mut MctsAgent::new(second),
            config,
        ),
    }
}

fn connect_four_report(traced: TracedMatchResult<ConnectFourAction>) -> CatalogMatchReport {
    let winner = winner_from_utilities(&traced.result.utilities);
    let moves = traced
        .actions
        .into_iter()
        .map(|(player, action)| RecordedMove {
            player: player.index(),
            action: CatalogAction::ConnectFour {
                column: action.column(),
            },
        })
        .collect();

    CatalogMatchReport {
        seed: traced.result.seed,
        plies: traced.result.plies,
        utilities: traced.result.utilities,
        winner,
        moves,
    }
}

fn tic_tac_toe_report(traced: TracedMatchResult<TicTacToeAction>) -> CatalogMatchReport {
    let winner = winner_from_utilities(&traced.result.utilities);
    let moves = traced
        .actions
        .into_iter()
        .map(|(player, action)| RecordedMove {
            player: player.index(),
            action: CatalogAction::TicTacToe {
                row: action.row(),
                column: action.column(),
            },
        })
        .collect();

    CatalogMatchReport {
        seed: traced.result.seed,
        plies: traced.result.plies,
        utilities: traced.result.utilities,
        winner,
        moves,
    }
}

fn winner_from_utilities(utilities: &[f32]) -> Option<usize> {
    match utilities {
        [first, second] if first > second => Some(0),
        [first, second] if second > first => Some(1),
        _ => None,
    }
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
    match game {
        GameId::ConnectFour => run_connect_four_batch(first, second, config),
        GameId::TicTacToe => run_tic_tac_toe_batch(first, second, config),
    }
}

fn run_connect_four_batch(
    first: AgentConfig,
    second: AgentConfig,
    config: BatchConfig,
) -> Result<Vec<MatchResult>, CatalogError> {
    let results = match (first, second) {
        (AgentConfig::Random, AgentConfig::Random) => {
            play_batch(&ConnectFour, config, || RandomAgent, || RandomAgent)
        }
        (AgentConfig::Random, AgentConfig::Mcts(second)) => play_batch(
            &ConnectFour,
            config,
            || RandomAgent,
            || MctsAgent::new(second),
        ),
        (AgentConfig::Mcts(first), AgentConfig::Random) => play_batch(
            &ConnectFour,
            config,
            || MctsAgent::new(first),
            || RandomAgent,
        ),
        (AgentConfig::Mcts(first), AgentConfig::Mcts(second)) => play_batch(
            &ConnectFour,
            config,
            || MctsAgent::new(first),
            || MctsAgent::new(second),
        ),
    }?;
    Ok(results)
}

fn run_tic_tac_toe_batch(
    first: AgentConfig,
    second: AgentConfig,
    config: BatchConfig,
) -> Result<Vec<MatchResult>, CatalogError> {
    let results = match (first, second) {
        (AgentConfig::Random, AgentConfig::Random) => {
            play_batch(&TicTacToe, config, || RandomAgent, || RandomAgent)
        }
        (AgentConfig::Random, AgentConfig::Mcts(second)) => play_batch(
            &TicTacToe,
            config,
            || RandomAgent,
            || MctsAgent::new(second),
        ),
        (AgentConfig::Mcts(first), AgentConfig::Random) => {
            play_batch(&TicTacToe, config, || MctsAgent::new(first), || RandomAgent)
        }
        (AgentConfig::Mcts(first), AgentConfig::Mcts(second)) => play_batch(
            &TicTacToe,
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

    #[test]
    fn traced_match_contains_every_typed_move() {
        let report = run_match_with_trace(
            GameId::TicTacToe,
            AgentConfig::Random,
            AgentConfig::Random,
            MatchConfig::default(),
        )
        .unwrap();

        assert_eq!(report.moves.len(), report.plies as usize);
        for (ply, recorded) in report.moves.iter().enumerate() {
            assert_eq!(recorded.player, ply % 2);
            match recorded.action {
                CatalogAction::TicTacToe { row, column } => {
                    assert!(row < 3);
                    assert!(column < 3);
                }
                CatalogAction::ConnectFour { .. } => panic!("unexpected Connect Four action"),
            }
        }
    }

    #[test]
    fn connect_four_trace_contains_legal_columns() {
        let report = run_match_with_trace(
            GameId::ConnectFour,
            AgentConfig::Random,
            AgentConfig::Random,
            MatchConfig::default(),
        )
        .unwrap();

        assert_eq!(report.moves.len(), report.plies as usize);
        assert!((7..=42).contains(&report.plies));
        for recorded in report.moves {
            match recorded.action {
                CatalogAction::ConnectFour { column } => assert!(column < 7),
                CatalogAction::TicTacToe { .. } => panic!("unexpected tic-tac-toe action"),
            }
        }
    }
}
