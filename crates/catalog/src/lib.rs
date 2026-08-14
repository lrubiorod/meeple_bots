//! Runtime configuration boundary for the statically dispatched engine.

use std::{error::Error, fmt, num::NonZeroU32};

use meeple_bots_boop::{
    Boop, BoopAction, PieceKind as BoopPieceKind, Resolution as BoopResolution,
};
use meeple_bots_connect_four::{ConnectFour, ConnectFourAction};
use meeple_bots_core::{Agent, Game};
pub use meeple_bots_evaluation::{
    AggregatedSearchStats, BenchmarkConfidence, ComplexityConfig, CutoffHeuristicEvidence,
    EvaluationError, GameComplexityReport, MctsLevel, MctsRecommendation, MctsStrengthReport,
    OpponentResult, SearchSufficiency, StrengthConfig, StrengthEstimate, StrengthOpponent,
    StrengthProgress, StrengthProgressStage,
};
use meeple_bots_evaluation::{
    evaluate_game, evaluate_mcts_strength_with_progress as evaluate_typed_mcts_strength,
};
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
    Boop,
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
    Boop {
        piece: CatalogBoopPieceKind,
        row: u8,
        column: u8,
        resolution: CatalogBoopResolution,
    },
    ConnectFour {
        column: u8,
    },
    TicTacToe {
        row: u8,
        column: u8,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CatalogBoopPieceKind {
    Kitten,
    Cat,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CatalogBoopResolution {
    None,
    Graduate { positions: [(u8, u8); 3] },
    Recover { row: u8, column: u8 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CatalogPieceKind {
    Token,
    Kitten,
    Cat,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CatalogPiece {
    pub player: usize,
    pub kind: CatalogPieceKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CatalogPool {
    pub kittens: u8,
    pub cats: u8,
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
    pub final_board: Vec<Option<CatalogPiece>>,
    pub pools: Option<[CatalogPool; 2]>,
}

#[derive(Debug)]
pub enum CatalogError {
    Match(MatchError),
    Evaluation(EvaluationError),
}

impl fmt::Display for CatalogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Match(error) => error.fmt(formatter),
            Self::Evaluation(error) => error.fmt(formatter),
        }
    }
}

impl Error for CatalogError {}

impl From<MatchError> for CatalogError {
    fn from(error: MatchError) -> Self {
        Self::Match(error)
    }
}

impl From<EvaluationError> for CatalogError {
    fn from(error: EvaluationError) -> Self {
        Self::Evaluation(error)
    }
}

pub fn evaluate_game_complexity(
    game: GameId,
    config: ComplexityConfig,
) -> Result<GameComplexityReport, CatalogError> {
    let report = match game {
        GameId::Boop => evaluate_game(&Boop, config),
        GameId::ConnectFour => evaluate_game(&ConnectFour, config),
        GameId::TicTacToe => evaluate_game(&TicTacToe, config),
    }?;
    Ok(report)
}

pub fn evaluate_mcts_strength(
    game: GameId,
    estimated_tree_log10: f64,
    tree_size_estimate_is_lower_bound: bool,
    config: StrengthConfig,
) -> Result<MctsStrengthReport, CatalogError> {
    evaluate_mcts_strength_with_progress(
        game,
        estimated_tree_log10,
        tree_size_estimate_is_lower_bound,
        config,
        |_| Ok(()),
    )
}

pub fn evaluate_mcts_strength_with_progress<F>(
    game: GameId,
    estimated_tree_log10: f64,
    tree_size_estimate_is_lower_bound: bool,
    config: StrengthConfig,
    mut progress: F,
) -> Result<MctsStrengthReport, CatalogError>
where
    F: FnMut(StrengthProgress) -> Result<(), EvaluationError>,
{
    let report = match game {
        GameId::Boop => evaluate_typed_mcts_strength(
            &Boop,
            estimated_tree_log10,
            tree_size_estimate_is_lower_bound,
            config,
            &mut progress,
        ),
        GameId::ConnectFour => evaluate_typed_mcts_strength(
            &ConnectFour,
            estimated_tree_log10,
            tree_size_estimate_is_lower_bound,
            config,
            &mut progress,
        ),
        GameId::TicTacToe => evaluate_typed_mcts_strength(
            &TicTacToe,
            estimated_tree_log10,
            tree_size_estimate_is_lower_bound,
            config,
            &mut progress,
        ),
    }?;
    Ok(report)
}

pub fn run_match(
    game: GameId,
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<MatchResult, CatalogError> {
    match game {
        GameId::Boop => run_boop(first, second, config),
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
        GameId::Boop => run_boop_with_trace(first, second, config),
        GameId::ConnectFour => run_connect_four_with_trace(first, second, config),
        GameId::TicTacToe => run_tic_tac_toe_with_trace(first, second, config),
    }
}

pub fn run_boop_match_with_trace<A, B>(
    first: &mut A,
    second: &mut B,
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError>
where
    A: Agent<Boop>,
    B: Agent<Boop>,
{
    let traced = play_typed_match_with_trace(&Boop, first, second, config)?;
    Ok(boop_report(traced))
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

fn run_boop(
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<MatchResult, CatalogError> {
    let game = Boop;
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

fn run_boop_with_trace(
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError> {
    match (first, second) {
        (AgentConfig::Random, AgentConfig::Random) => {
            run_boop_match_with_trace(&mut RandomAgent, &mut RandomAgent, config)
        }
        (AgentConfig::Random, AgentConfig::Mcts(second)) => {
            run_boop_match_with_trace(&mut RandomAgent, &mut MctsAgent::new(second), config)
        }
        (AgentConfig::Mcts(first), AgentConfig::Random) => {
            run_boop_match_with_trace(&mut MctsAgent::new(first), &mut RandomAgent, config)
        }
        (AgentConfig::Mcts(first), AgentConfig::Mcts(second)) => run_boop_match_with_trace(
            &mut MctsAgent::new(first),
            &mut MctsAgent::new(second),
            config,
        ),
    }
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
    let game = ConnectFour;
    let mut state = game.initial_state();
    for (_, action) in &traced.actions {
        game.apply_action(&mut state, action)
            .expect("trace contains actions accepted by the game");
    }
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
        final_board: state
            .board()
            .iter()
            .map(|piece| {
                piece.map(|player| CatalogPiece {
                    player: player.index(),
                    kind: CatalogPieceKind::Token,
                })
            })
            .collect(),
        pools: None,
    }
}

fn tic_tac_toe_report(traced: TracedMatchResult<TicTacToeAction>) -> CatalogMatchReport {
    let game = TicTacToe;
    let mut state = game.initial_state();
    for (_, action) in &traced.actions {
        game.apply_action(&mut state, action)
            .expect("trace contains actions accepted by the game");
    }
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
        final_board: state
            .board()
            .iter()
            .map(|piece| {
                piece.map(|player| CatalogPiece {
                    player: player.index(),
                    kind: CatalogPieceKind::Token,
                })
            })
            .collect(),
        pools: None,
    }
}

fn boop_report(traced: TracedMatchResult<BoopAction>) -> CatalogMatchReport {
    let game = Boop;
    let mut state = game.initial_state();
    for (_, action) in &traced.actions {
        game.apply_action(&mut state, action)
            .expect("trace contains actions accepted by the game");
    }
    let winner = winner_from_utilities(&traced.result.utilities);
    let moves = traced
        .actions
        .into_iter()
        .map(|(player, action)| RecordedMove {
            player: player.index(),
            action: CatalogAction::Boop {
                piece: catalog_boop_piece(action.piece()),
                row: action.position().row(),
                column: action.position().column(),
                resolution: match action.resolution() {
                    BoopResolution::None => CatalogBoopResolution::None,
                    BoopResolution::Graduate(line) => CatalogBoopResolution::Graduate {
                        positions: line
                            .positions()
                            .map(|position| (position.row(), position.column())),
                    },
                    BoopResolution::Recover(position) => CatalogBoopResolution::Recover {
                        row: position.row(),
                        column: position.column(),
                    },
                },
            },
        })
        .collect();
    let pools = state.pools().map(|pool| CatalogPool {
        kittens: pool.kittens(),
        cats: pool.cats(),
    });

    CatalogMatchReport {
        seed: traced.result.seed,
        plies: traced.result.plies,
        utilities: traced.result.utilities,
        winner,
        moves,
        final_board: state
            .board()
            .iter()
            .map(|piece| {
                piece.map(|piece| CatalogPiece {
                    player: piece.owner().index(),
                    kind: match piece.kind() {
                        BoopPieceKind::Kitten => CatalogPieceKind::Kitten,
                        BoopPieceKind::Cat => CatalogPieceKind::Cat,
                    },
                })
            })
            .collect(),
        pools: Some(pools),
    }
}

fn catalog_boop_piece(piece: BoopPieceKind) -> CatalogBoopPieceKind {
    match piece {
        BoopPieceKind::Kitten => CatalogBoopPieceKind::Kitten,
        BoopPieceKind::Cat => CatalogBoopPieceKind::Cat,
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
        GameId::Boop => run_boop_batch(first, second, config),
        GameId::ConnectFour => run_connect_four_batch(first, second, config),
        GameId::TicTacToe => run_tic_tac_toe_batch(first, second, config),
    }
}

fn run_boop_batch(
    first: AgentConfig,
    second: AgentConfig,
    config: BatchConfig,
) -> Result<Vec<MatchResult>, CatalogError> {
    let results = match (first, second) {
        (AgentConfig::Random, AgentConfig::Random) => {
            play_batch(&Boop, config, || RandomAgent, || RandomAgent)
        }
        (AgentConfig::Random, AgentConfig::Mcts(second)) => {
            play_batch(&Boop, config, || RandomAgent, || MctsAgent::new(second))
        }
        (AgentConfig::Mcts(first), AgentConfig::Random) => {
            play_batch(&Boop, config, || MctsAgent::new(first), || RandomAgent)
        }
        (AgentConfig::Mcts(first), AgentConfig::Mcts(second)) => play_batch(
            &Boop,
            config,
            || MctsAgent::new(first),
            || MctsAgent::new(second),
        ),
    }?;
    Ok(results)
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
    use std::num::NonZeroU32;

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
                CatalogAction::Boop { .. } => panic!("unexpected boop action"),
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
                CatalogAction::Boop { .. } => panic!("unexpected boop action"),
            }
        }
    }

    #[test]
    fn boop_trace_contains_actions_and_an_authoritative_board() {
        let report = run_match_with_trace(
            GameId::Boop,
            AgentConfig::Random,
            AgentConfig::Random,
            MatchConfig::default(),
        )
        .unwrap();

        assert_eq!(report.moves.len(), report.plies as usize);
        assert_eq!(report.final_board.len(), 36);
        assert!(report.pools.is_some());
        assert!(report.moves.iter().all(|movement| matches!(
            movement.action,
            CatalogAction::Boop { row, column, .. } if row < 6 && column < 6
        )));
    }

    #[test]
    fn boop_samples_as_more_complex_than_tic_tac_toe() {
        let config = ComplexityConfig {
            samples: NonZeroU32::new(16).unwrap(),
            calibration_iterations: NonZeroU32::new(1).unwrap(),
            ..ComplexityConfig::default()
        };
        let tic_tac_toe = evaluate_game_complexity(GameId::TicTacToe, config).unwrap();
        let boop = evaluate_game_complexity(GameId::Boop, config).unwrap();

        assert!(boop.initial_legal_actions > tic_tac_toe.initial_legal_actions);
        assert!(boop.mean_branching_factor > tic_tac_toe.mean_branching_factor);
        assert!(boop.median_plies > tic_tac_toe.median_plies);
        assert!(boop.estimated_tree_log10 > tic_tac_toe.estimated_tree_log10);
    }
}
