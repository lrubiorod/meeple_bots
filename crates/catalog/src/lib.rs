//! Runtime configuration and independent participants for the typed engine.

pub mod cant_stop;
pub mod splendor;

mod configuration;
pub use configuration::{
    AgentConfig, BoopMctsAgent, ConfiguredRolloutPolicy, ConfiguredSelectionBias,
    Connect6MctsAgent, ConnectFourMctsAgent, EvaluatorConfig, MctsAgentConfig,
    RolloutConditionConfig, SpiritsOfTheForestMctsAgent, TicTacToeMctsAgent, configured_boop_mcts,
    configured_connect_four_mcts, configured_connect6_mcts, configured_spirits_of_the_forest_mcts,
    configured_tic_tac_toe_mcts,
};

mod participant;
pub use participant::ConfiguredAgent;

use std::{error::Error, fmt, num::NonZeroU32};

use meeple_bots_boop::{
    Boop, BoopAction, BoopReplayAnalysis, GraduateLine, PieceKind as BoopPieceKind,
    Position as BoopPosition, Resolution as BoopResolution, analyze_replay as analyze_boop_replay,
};
use meeple_bots_connect_four::{ConnectFour, ConnectFourAction};
use meeple_bots_connect6::{Connect6, Connect6Action};
use meeple_bots_core::{
    Agent, DeterministicGame, Game, HeuristicGame, PlayerId, PositionStatus, RandomSource,
    RootActionStats, TreeReuseStats,
};
pub use meeple_bots_evaluation::{
    EvaluationConfig, EvaluationError, GameEvaluationReport, IterationBudgetEstimate,
    MctsAgentBenchmark, RolloutCostEstimate, SampledDecisionTiming, SuggestedMctsExperiment,
};
use meeple_bots_evaluation::{
    benchmark_mcts_agent as benchmark_typed_mcts_agent, evaluate_game as evaluate_typed_game,
};
pub use meeple_bots_mcts_agent::{
    MctsConfig, RolloutPolicyConfig, SearchBudget, SelectionPolicy, UniformRandom,
};
use meeple_bots_simulation::{
    BatchConfig, MatchError, MatchObserver, SplitMix64, TracedMatchResult, play_batch, play_match,
    play_match_with_trace as play_typed_match_with_trace, play_match_with_trace_and_observer,
};
pub use meeple_bots_simulation::{MatchConfig, MatchResult};
use meeple_bots_spirits_of_the_forest::{
    ForestPosition, GemstoneSacrifice, PowerSource, Spirit, SpiritsOfTheForest,
    SpiritsOfTheForestAction, SpiritsReplayAnalysis, analyze_replay as analyze_spirits_replay,
};
use meeple_bots_tic_tac_toe::{TicTacToe, TicTacToeAction};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GameId {
    Connect6(usize),
    Splendor,
    Boop,
    ConnectFour,
    SpiritsOfTheForest,
    TicTacToe,
}

/// Integer initialization parameters, validated before any game allocation.
pub type GameParameters = std::collections::BTreeMap<String, i64>;

pub fn configure_game(game: GameId, parameters: &GameParameters) -> Result<GameId, CatalogError> {
    for key in parameters.keys() {
        if !matches!(game, GameId::Connect6(_)) || key != "board_size" {
            return Err(CatalogError::InvalidGameParameters(format!(
                "{} does not accept game parameter {key:?}",
                game_name(game)
            )));
        }
    }
    if let GameId::Connect6(default_size) = game {
        let size = match parameters.get("board_size") {
            Some(value) => usize::try_from(*value).map_err(|_| {
                CatalogError::InvalidGameParameters(
                    "board_size must be a non-negative integer".into(),
                )
            })?,
            None => default_size,
        };
        connect6_game(size)?;
        return Ok(GameId::Connect6(size));
    }
    Ok(game)
}

pub fn game_parameters(game: GameId) -> GameParameters {
    match game {
        GameId::Connect6(size) => [("board_size".into(), size as i64)].into(),
        _ => GameParameters::new(),
    }
}
fn connect6_game(size: usize) -> Result<Connect6, CatalogError> {
    Connect6::new(size).map_err(|e| CatalogError::InvalidGameParameters(e.into()))
}

/// Search options exposed by the registered game integration.
#[derive(Clone, Debug, PartialEq)]
pub struct GameSearchCapabilities {
    pub heuristics: Vec<HeuristicDescriptor>,
    pub turn_phase_conditions: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HeuristicDescriptor {
    pub index: u32,
    pub parameters: &'static [meeple_bots_core::HeuristicParameterSpec],
}

/// Reuse the game-owned heuristic schemas used by native validation.
pub fn game_search_capabilities(game: GameId) -> GameSearchCapabilities {
    fn heuristics<G: HeuristicGame>(game: &G) -> Vec<HeuristicDescriptor> {
        (0..game.heuristic_count())
            .map(|index| HeuristicDescriptor {
                index,
                parameters: game
                    .heuristic_parameter_specs(index)
                    .expect("registered heuristic must provide its parameter schema"),
            })
            .collect()
    }
    GameSearchCapabilities {
        heuristics: match game {
            GameId::Boop => heuristics(&Boop),
            GameId::SpiritsOfTheForest => heuristics(&spirits_of_the_forest_game(0)),
            GameId::Splendor => heuristics(&splendor::game(0)),
            GameId::Connect6(_) | GameId::ConnectFour | GameId::TicTacToe => Vec::new(),
        },
        turn_phase_conditions: supports_turn_phase_conditions(game),
    }
}

pub const fn game_name(game: GameId) -> &'static str {
    match game {
        GameId::Connect6(_) => "connect6",
        GameId::Splendor => "splendor",
        GameId::Boop => "boop",
        GameId::ConnectFour => "connect-four",
        GameId::SpiritsOfTheForest => "spotf",
        GameId::TicTacToe => "tic-tac-toe",
    }
}

fn supports_turn_phase_conditions(game: GameId) -> bool {
    matches!(game, GameId::SpiritsOfTheForest)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CatalogAction {
    Connect6 {
        position: usize,
    },
    Splendor(meeple_bots_splendor::SplendorAction),
    Boop {
        piece: CatalogBoopPieceKind,
        row: u8,
        column: u8,
        resolution: CatalogBoopResolution,
    },
    ConnectFour {
        column: u8,
    },
    SpiritsOfTheForest(CatalogSpiritsAction),
    TicTacToe {
        row: u8,
        column: u8,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CatalogSpiritsAction {
    TakeTile {
        row: u8,
        column: u8,
        sacrifice: Option<CatalogGemstoneSacrifice>,
    },
    EndCollection,
    PlaceGemstone {
        row: u8,
        column: u8,
    },
    MoveGemstone {
        source_row: u8,
        source_column: u8,
        target_row: u8,
        target_column: u8,
    },
    SkipGemstone,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CatalogGemstoneSacrifice {
    Available,
    Forest { row: u8, column: u8 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CatalogSpirit {
    Moss,
    Flowers,
    Fruits,
    Mushrooms,
    Water,
    Vines,
    Branches,
    Leaves,
    Webs,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CatalogPowerSource {
    Fire,
    Moon,
    Sun,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CatalogSpiritTile {
    pub spirit: CatalogSpirit,
    pub spirit_symbols: u8,
    pub power_source: Option<CatalogPowerSource>,
    pub gemstone: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CatalogSpiritCollection {
    pub spirit_symbols: [u8; 9],
    pub power_sources: [u8; 3],
    pub tiles: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CatalogGemstonePool {
    pub available: u8,
    pub placed: u8,
    pub removed: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CatalogTurnPhase {
    Choose,
    Continue,
    Collect,
    PlaceGemstone,
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

#[derive(Clone, Debug, PartialEq)]
pub struct RecordedMove {
    pub player: usize,
    pub action: CatalogAction,
    pub decision_seconds: f64,
    pub selection_seconds: f64,
    pub maintenance_seconds: f64,
    pub search_iterations: Option<u64>,
    pub search_nodes: Option<u64>,
    pub terminal_simulations: Option<u64>,
    pub cutoff_simulations: Option<u64>,
    pub root_actions: Vec<RootActionStats>,
    pub tree_reuse: Option<TreeReuseStats>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CatalogMatchReport {
    pub chance_events: Vec<meeple_bots_simulation::TracedChance<CatalogAction>>,
    pub splendor_state: Option<meeple_bots_splendor::SplendorState>,
    pub seed: u64,
    pub plies: u32,
    pub utilities: Vec<f32>,
    pub winner: Option<usize>,
    pub moves: Vec<RecordedMove>,
    pub unassigned_maintenance_seconds: [f64; 2],
    pub final_board: Vec<Option<CatalogPiece>>,
    pub pools: Option<[CatalogPool; 2]>,
    pub spirit_forest: Option<Vec<Option<CatalogSpiritTile>>>,
    pub spirit_collections: Option<[CatalogSpiritCollection; 2]>,
    pub gemstone_pools: Option<[CatalogGemstonePool; 2]>,
    pub scores: Option<[i16; 2]>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CatalogTraceAnalysis {
    Generic { utilities: [f32; 2] },
    Boop(BoopReplayAnalysis),
    SpiritsOfTheForest(SpiritsReplayAnalysis),
}

#[derive(Debug)]
pub enum CatalogError {
    InvalidGameParameters(String),
    Match(MatchError),
    Evaluation(EvaluationError),
    UnsupportedHeuristic {
        game: GameId,
        index: u32,
        available: u32,
    },
    InvalidMctsConfig(&'static str),
    InvalidHeuristicParameter {
        game: GameId,
        index: u32,
        message: String,
    },
    AnalysisUnavailable(GameId),
    InvalidTrace {
        game: GameId,
        message: String,
    },
}

impl fmt::Display for CatalogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGameParameters(message) => write!(formatter, "{message}"),
            Self::Match(error) => error.fmt(formatter),
            Self::Evaluation(error) => error.fmt(formatter),
            Self::UnsupportedHeuristic {
                game,
                index,
                available,
            } => {
                let name = game_name(*game);
                if *available == 0 {
                    write!(formatter, "{name} does not provide MCTS heuristics")
                } else if *available == 1 {
                    write!(
                        formatter,
                        "{name} does not provide MCTS heuristic {index}; available index: 0"
                    )
                } else {
                    write!(
                        formatter,
                        "{name} does not provide MCTS heuristic {index}; available indices: 0..{}",
                        available - 1
                    )
                }
            }
            Self::InvalidMctsConfig(message) => formatter.write_str(message),
            Self::InvalidHeuristicParameter {
                game,
                index,
                message,
            } => write!(
                formatter,
                "{} heuristic {index} {message}",
                game_name(*game)
            ),
            Self::AnalysisUnavailable(game) => {
                write!(
                    formatter,
                    "tournament analysis is not available for {}",
                    game_name(*game)
                )
            }
            Self::InvalidTrace { game, message } => {
                write!(
                    formatter,
                    "invalid {} tournament trace: {message}",
                    game_name(*game)
                )
            }
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

pub fn evaluate_game(
    game: GameId,
    config: EvaluationConfig,
) -> Result<GameEvaluationReport, CatalogError> {
    let report = match game {
        GameId::Splendor => return Err(CatalogError::AnalysisUnavailable(game)),
        GameId::Boop => evaluate_typed_game(&Boop, config),
        GameId::Connect6(size) => evaluate_typed_game(&connect6_game(size)?, config),
        GameId::ConnectFour => evaluate_typed_game(&ConnectFour, config),
        GameId::SpiritsOfTheForest => {
            let game = spirits_of_the_forest_game(config.seed);
            evaluate_typed_game(&game, config)
        }
        GameId::TicTacToe => evaluate_typed_game(&TicTacToe, config),
    }?;
    Ok(report)
}

pub fn benchmark_mcts_agent(
    game: GameId,
    config: MctsAgentConfig,
    median_depth: u32,
    seed: u64,
) -> Result<MctsAgentBenchmark, CatalogError> {
    let benchmark = match game {
        GameId::Connect6(size) => {
            let mut agent = configured_connect6_mcts(config)?;
            benchmark_typed_mcts_agent(&connect6_game(size)?, &mut agent, median_depth, seed)
        }
        GameId::Splendor => {
            let mut agent = splendor::configured_agent(AgentConfig::Mcts(config))
                .map_err(EvaluationError::Agent)?;
            benchmark_typed_mcts_agent(&splendor::game(seed), &mut agent, median_depth, seed)
        }
        GameId::Boop => {
            let mut agent = configured_boop_mcts(config)?;
            benchmark_typed_mcts_agent(&Boop, &mut agent, median_depth, seed)
        }
        GameId::ConnectFour => {
            let mut agent = configured_connect_four_mcts(config)?;
            benchmark_typed_mcts_agent(&ConnectFour, &mut agent, median_depth, seed)
        }
        GameId::SpiritsOfTheForest => {
            let game = spirits_of_the_forest_game(seed);
            let mut agent = configured_spirits_of_the_forest_mcts(config)?;
            benchmark_typed_mcts_agent(&game, &mut agent, median_depth, seed)
        }
        GameId::TicTacToe => {
            let mut agent = configured_tic_tac_toe_mcts(config)?;
            benchmark_typed_mcts_agent(&TicTacToe, &mut agent, median_depth, seed)
        }
    }?;
    Ok(benchmark)
}

pub fn analyze_trace(
    game: GameId,
    moves: &[RecordedMove],
) -> Result<CatalogTraceAnalysis, CatalogError> {
    analyze_seeded_trace(game, moves, 0)
}

pub fn analyze_seeded_trace(
    game: GameId,
    moves: &[RecordedMove],
    seed: u64,
) -> Result<CatalogTraceAnalysis, CatalogError> {
    match game {
        GameId::Splendor => Err(CatalogError::InvalidTrace {
            game,
            message: "Splendor replay requires recorded chance events; use splendor::replay".into(),
        }),
        GameId::Boop => {
            let actions = moves
                .iter()
                .enumerate()
                .map(|(index, movement)| {
                    let player = u8::try_from(movement.player)
                        .ok()
                        .filter(|player| *player < 2)
                        .map(PlayerId::new)
                        .ok_or_else(|| invalid_trace(game, index, "player must be 0 or 1"))?;
                    let action = catalog_boop_action(&movement.action)
                        .map_err(|message| invalid_trace(game, index, message))?;
                    Ok((player, action))
                })
                .collect::<Result<Vec<_>, CatalogError>>()?;
            analyze_boop_replay(&actions)
                .map(CatalogTraceAnalysis::Boop)
                .map_err(|error| CatalogError::InvalidTrace {
                    game,
                    message: error.to_string(),
                })
        }
        GameId::SpiritsOfTheForest => {
            let actions = moves
                .iter()
                .enumerate()
                .map(|(index, movement)| {
                    let player = u8::try_from(movement.player)
                        .ok()
                        .filter(|player| *player < 2)
                        .map(PlayerId::new)
                        .ok_or_else(|| invalid_trace(game, index, "player must be 0 or 1"))?;
                    let action = catalog_spirits_replay_action(&movement.action)
                        .map_err(|message| invalid_trace(game, index, message))?;
                    Ok((player, action))
                })
                .collect::<Result<Vec<_>, CatalogError>>()?;
            let configured_game = spirits_of_the_forest_game(seed);
            analyze_spirits_replay(&configured_game, &actions)
                .map(CatalogTraceAnalysis::SpiritsOfTheForest)
                .map_err(|error| CatalogError::InvalidTrace {
                    game,
                    message: error.to_string(),
                })
        }
        GameId::Connect6(size) => {
            analyze_generic_replay(game, &connect6_game(size)?, moves, |action| {
                let CatalogAction::Connect6 { position } = action else {
                    return Err("expected Connect6 action");
                };
                Ok(Connect6Action::Place(*position))
            })
        }
        GameId::ConnectFour => analyze_generic_replay(game, &ConnectFour, moves, |action| {
            let CatalogAction::ConnectFour { column } = action else {
                return Err("expected a connect-four action");
            };
            ConnectFourAction::new(*column).ok_or("column is outside the board")
        }),
        GameId::TicTacToe => analyze_generic_replay(game, &TicTacToe, moves, |action| {
            let CatalogAction::TicTacToe { row, column } = action else {
                return Err("expected a tic-tac-toe action");
            };
            TicTacToeAction::new(*row, *column).ok_or("position is outside the board")
        }),
    }
}

/// Validate a completed trace using the authoritative game contract, without
/// depending on agents or reimplementing board rules in an extractor.
fn analyze_generic_replay<G: DeterministicGame>(
    id: GameId,
    game: &G,
    moves: &[RecordedMove],
    action_from_catalog: impl Fn(&CatalogAction) -> Result<G::Action, &'static str>,
) -> Result<CatalogTraceAnalysis, CatalogError> {
    let mut state = game.initial_state();
    for (index, movement) in moves.iter().enumerate() {
        let PositionStatus::PlayerTurn(player) = game.status(&state) else {
            return Err(invalid_trace(id, index, "action after the end of the game"));
        };
        if movement.player != player.index() {
            return Err(invalid_trace(
                id,
                index,
                "recorded player is not the active player",
            ));
        }
        let action = action_from_catalog(&movement.action)
            .map_err(|message| invalid_trace(id, index, message))?;
        game.apply_action(&mut state, &action)
            .map_err(|error| invalid_trace(id, index, error))?;
    }
    if game.status(&state) != PositionStatus::Terminal {
        return Err(CatalogError::InvalidTrace {
            game: id,
            message: "trace does not end in a terminal position".to_owned(),
        });
    }
    let mut utilities = [0.0; 2];
    for player in [PlayerId::FIRST, PlayerId::SECOND] {
        utilities[player.index()] =
            game.terminal_utility(&state, player)
                .ok_or_else(|| CatalogError::InvalidTrace {
                    game: id,
                    message: format!("terminal utility is missing for player {player}"),
                })?;
    }
    Ok(CatalogTraceAnalysis::Generic { utilities })
}

fn catalog_spirits_replay_action(
    action: &CatalogAction,
) -> Result<SpiritsOfTheForestAction, &'static str> {
    let CatalogAction::SpiritsOfTheForest(action) = action else {
        return Err("action belongs to a different game");
    };
    Ok(match *action {
        CatalogSpiritsAction::TakeTile {
            row,
            column,
            sacrifice,
        } => SpiritsOfTheForestAction::TakeTile {
            position: replay_forest_position(row, column)?,
            sacrifice: match sacrifice {
                None => None,
                Some(CatalogGemstoneSacrifice::Available) => Some(GemstoneSacrifice::Available),
                Some(CatalogGemstoneSacrifice::Forest { row, column }) => Some(
                    GemstoneSacrifice::Forest(replay_forest_position(row, column)?),
                ),
            },
        },
        CatalogSpiritsAction::EndCollection => SpiritsOfTheForestAction::EndCollection,
        CatalogSpiritsAction::PlaceGemstone { row, column } => {
            SpiritsOfTheForestAction::PlaceGemstone {
                target: replay_forest_position(row, column)?,
            }
        }
        CatalogSpiritsAction::MoveGemstone {
            source_row,
            source_column,
            target_row,
            target_column,
        } => SpiritsOfTheForestAction::MoveGemstone {
            source: replay_forest_position(source_row, source_column)?,
            target: replay_forest_position(target_row, target_column)?,
        },
        CatalogSpiritsAction::SkipGemstone => SpiritsOfTheForestAction::SkipGemstone,
    })
}

fn replay_forest_position(row: u8, column: u8) -> Result<ForestPosition, &'static str> {
    ForestPosition::new(row, column).ok_or("forest position is outside the 4x12 board")
}

fn invalid_trace(game: GameId, index: usize, message: impl fmt::Display) -> CatalogError {
    CatalogError::InvalidTrace {
        game,
        message: format!("ply {}: {message}", index + 1),
    }
}

fn catalog_boop_action(action: &CatalogAction) -> Result<BoopAction, &'static str> {
    let CatalogAction::Boop {
        piece,
        row,
        column,
        resolution,
    } = action
    else {
        return Err("expected a boop action");
    };
    let piece = match piece {
        CatalogBoopPieceKind::Kitten => BoopPieceKind::Kitten,
        CatalogBoopPieceKind::Cat => BoopPieceKind::Cat,
    };
    let position = BoopPosition::new(*row, *column).ok_or("placement is outside the board")?;
    let resolution = match resolution {
        CatalogBoopResolution::None => BoopResolution::None,
        CatalogBoopResolution::Graduate { positions } => {
            let positions = positions.map(|(row, column)| BoopPosition::new(row, column));
            let [Some(first), Some(second), Some(third)] = positions else {
                return Err("graduation contains a position outside the board");
            };
            BoopResolution::Graduate(
                GraduateLine::new([first, second, third])
                    .ok_or("graduation positions do not form a valid line")?,
            )
        }
        CatalogBoopResolution::Recover { row, column } => BoopResolution::Recover(
            BoopPosition::new(*row, *column).ok_or("recovery is outside the board")?,
        ),
    };
    Ok(BoopAction::new(piece, position, resolution))
}

pub fn run_match(
    game: GameId,
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<MatchResult, CatalogError> {
    match game {
        GameId::Splendor => Ok(splendor::run(first, second, config)?.result),
        GameId::Boop => run_boop(first, second, config),
        GameId::Connect6(size) => {
            let mut a = ConfiguredAgent::new(first, configured_connect6_mcts)?;
            let mut b = ConfiguredAgent::new(second, configured_connect6_mcts)?;
            Ok(play_match(&connect6_game(size)?, &mut a, &mut b, config)?)
        }
        GameId::ConnectFour => run_connect_four(first, second, config),
        GameId::SpiritsOfTheForest => run_spirits_of_the_forest(first, second, config),
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
        GameId::Splendor => splendor::report(splendor::run(first, second, config)?),
        GameId::Boop => run_boop_with_trace(first, second, config),
        GameId::Connect6(size) => {
            let mut a = ConfiguredAgent::new(first, configured_connect6_mcts)?;
            let mut b = ConfiguredAgent::new(second, configured_connect6_mcts)?;
            run_connect6_match_with_trace(size, &mut a, &mut b, config)
        }
        GameId::ConnectFour => run_connect_four_with_trace(first, second, config),
        GameId::SpiritsOfTheForest => run_spirits_of_the_forest_with_trace(first, second, config),
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

pub fn run_boop_match_with_observer<A, B, O>(
    first: &mut A,
    second: &mut B,
    config: MatchConfig,
    observer: &mut O,
) -> Result<CatalogMatchReport, CatalogError>
where
    A: Agent<Boop>,
    B: Agent<Boop>,
    O: MatchObserver<Boop>,
{
    let traced = play_match_with_trace_and_observer(&Boop, first, second, config, observer)?;
    Ok(boop_report(traced))
}

pub fn spirits_of_the_forest_game(seed: u64) -> SpiritsOfTheForest {
    let mut setup_rng = SplitMix64::new(seed ^ 0x8EBC_6AF0_9C88_C6E3);
    SpiritsOfTheForest::shuffled(&mut setup_rng)
}

pub fn run_spirits_of_the_forest_match_with_trace<A, B>(
    first: &mut A,
    second: &mut B,
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError>
where
    A: Agent<SpiritsOfTheForest>,
    B: Agent<SpiritsOfTheForest>,
{
    let game = spirits_of_the_forest_game(config.seed);
    let traced = play_typed_match_with_trace(&game, first, second, config)?;
    Ok(spirits_of_the_forest_report(&game, traced))
}

pub fn run_spirits_of_the_forest_match_with_observer<A, B, O>(
    first: &mut A,
    second: &mut B,
    config: MatchConfig,
    observer: &mut O,
) -> Result<CatalogMatchReport, CatalogError>
where
    A: Agent<SpiritsOfTheForest>,
    B: Agent<SpiritsOfTheForest>,
    O: MatchObserver<SpiritsOfTheForest>,
{
    let game = spirits_of_the_forest_game(config.seed);
    let traced = play_match_with_trace_and_observer(&game, first, second, config, observer)?;
    Ok(spirits_of_the_forest_report(&game, traced))
}

pub fn run_connect6_match_with_trace<A: Agent<Connect6>, B: Agent<Connect6>>(
    size: usize,
    first: &mut A,
    second: &mut B,
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError> {
    let game = connect6_game(size)?;
    Ok(connect6_report(
        game,
        play_typed_match_with_trace(&game, first, second, config)?,
    ))
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

pub fn run_connect_four_match_with_observer<A, B, O>(
    first: &mut A,
    second: &mut B,
    config: MatchConfig,
    observer: &mut O,
) -> Result<CatalogMatchReport, CatalogError>
where
    A: Agent<ConnectFour>,
    B: Agent<ConnectFour>,
    O: MatchObserver<ConnectFour>,
{
    let traced = play_match_with_trace_and_observer(&ConnectFour, first, second, config, observer)?;
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

pub fn run_tic_tac_toe_match_with_observer<A, B, O>(
    first: &mut A,
    second: &mut B,
    config: MatchConfig,
    observer: &mut O,
) -> Result<CatalogMatchReport, CatalogError>
where
    A: Agent<TicTacToe>,
    B: Agent<TicTacToe>,
    O: MatchObserver<TicTacToe>,
{
    let traced = play_match_with_trace_and_observer(&TicTacToe, first, second, config, observer)?;
    Ok(tic_tac_toe_report(traced))
}

fn run_connect_four(
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<MatchResult, CatalogError> {
    let mut first = ConfiguredAgent::new(first, configured_connect_four_mcts)?;
    let mut second = ConfiguredAgent::new(second, configured_connect_four_mcts)?;
    Ok(play_match(&ConnectFour, &mut first, &mut second, config)?)
}

fn run_boop(
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<MatchResult, CatalogError> {
    let mut first = ConfiguredAgent::new(first, configured_boop_mcts)?;
    let mut second = ConfiguredAgent::new(second, configured_boop_mcts)?;
    Ok(play_match(&Boop, &mut first, &mut second, config)?)
}

fn run_spirits_of_the_forest(
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<MatchResult, CatalogError> {
    let mut first = ConfiguredAgent::new(first, configured_spirits_of_the_forest_mcts)?;
    let mut second = ConfiguredAgent::new(second, configured_spirits_of_the_forest_mcts)?;
    Ok(play_match(
        &spirits_of_the_forest_game(config.seed),
        &mut first,
        &mut second,
        config,
    )?)
}

fn run_boop_with_trace(
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError> {
    let mut first = ConfiguredAgent::new(first, configured_boop_mcts)?;
    let mut second = ConfiguredAgent::new(second, configured_boop_mcts)?;
    run_boop_match_with_trace(&mut first, &mut second, config)
}

fn run_spirits_of_the_forest_with_trace(
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError> {
    let mut first = ConfiguredAgent::new(first, configured_spirits_of_the_forest_mcts)?;
    let mut second = ConfiguredAgent::new(second, configured_spirits_of_the_forest_mcts)?;
    run_spirits_of_the_forest_match_with_trace(&mut first, &mut second, config)
}

fn run_connect_four_with_trace(
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError> {
    let mut first = ConfiguredAgent::new(first, configured_connect_four_mcts)?;
    let mut second = ConfiguredAgent::new(second, configured_connect_four_mcts)?;
    run_connect_four_match_with_trace(&mut first, &mut second, config)
}

fn run_tic_tac_toe(
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<MatchResult, CatalogError> {
    let mut first = ConfiguredAgent::new(first, configured_tic_tac_toe_mcts)?;
    let mut second = ConfiguredAgent::new(second, configured_tic_tac_toe_mcts)?;
    Ok(play_match(&TicTacToe, &mut first, &mut second, config)?)
}

fn run_tic_tac_toe_with_trace(
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError> {
    let mut first = ConfiguredAgent::new(first, configured_tic_tac_toe_mcts)?;
    let mut second = ConfiguredAgent::new(second, configured_tic_tac_toe_mcts)?;
    run_tic_tac_toe_match_with_trace(&mut first, &mut second, config)
}

fn connect6_report(
    game: Connect6,
    traced: TracedMatchResult<Connect6Action>,
) -> CatalogMatchReport {
    let mut state = game.initial_state();
    for traced_action in &traced.actions {
        game.apply_action(&mut state, &traced_action.action)
            .expect("trace contains actions accepted by the game");
    }
    let winner = winner_from_utilities(&traced.result.utilities);
    let moves = traced
        .actions
        .into_iter()
        .map(|traced_action| RecordedMove {
            player: traced_action.player.index(),
            action: CatalogAction::Connect6 {
                position: match traced_action.action {
                    Connect6Action::Place(position) => position,
                },
            },
            decision_seconds: traced_action.decision_time.as_secs_f64(),
            selection_seconds: traced_action.selection_time.as_secs_f64(),
            maintenance_seconds: traced_action.maintenance_time.as_secs_f64(),
            search_iterations: traced_action.decision_stats.search_iterations,
            search_nodes: traced_action.decision_stats.search_nodes,
            terminal_simulations: traced_action.decision_stats.terminal_simulations,
            cutoff_simulations: traced_action.decision_stats.cutoff_simulations,
            root_actions: traced_action.decision_stats.root_actions,
            tree_reuse: traced_action.decision_stats.tree_reuse,
        })
        .collect();

    CatalogMatchReport {
        chance_events: Vec::new(),
        splendor_state: None,
        unassigned_maintenance_seconds: traced
            .unassigned_maintenance_time
            .map(|time| time.as_secs_f64()),
        seed: traced.result.seed,
        plies: traced.result.plies,
        utilities: traced.result.utilities,
        winner,
        moves,
        final_board: state
            .board()
            .iter()
            .map(|piece| {
                (*piece != 0).then(|| CatalogPiece {
                    player: usize::from(*piece - 1),
                    kind: CatalogPieceKind::Token,
                })
            })
            .collect(),
        pools: None,
        spirit_forest: None,
        spirit_collections: None,
        gemstone_pools: None,
        scores: None,
    }
}

fn connect_four_report(traced: TracedMatchResult<ConnectFourAction>) -> CatalogMatchReport {
    let game = ConnectFour;
    let mut state = game.initial_state();
    for traced_action in &traced.actions {
        game.apply_action(&mut state, &traced_action.action)
            .expect("trace contains actions accepted by the game");
    }
    let winner = winner_from_utilities(&traced.result.utilities);
    let moves = traced
        .actions
        .into_iter()
        .map(|traced_action| RecordedMove {
            player: traced_action.player.index(),
            action: CatalogAction::ConnectFour {
                column: traced_action.action.column(),
            },
            decision_seconds: traced_action.decision_time.as_secs_f64(),
            selection_seconds: traced_action.selection_time.as_secs_f64(),
            maintenance_seconds: traced_action.maintenance_time.as_secs_f64(),
            search_iterations: traced_action.decision_stats.search_iterations,
            search_nodes: traced_action.decision_stats.search_nodes,
            terminal_simulations: traced_action.decision_stats.terminal_simulations,
            cutoff_simulations: traced_action.decision_stats.cutoff_simulations,
            root_actions: traced_action.decision_stats.root_actions,
            tree_reuse: traced_action.decision_stats.tree_reuse,
        })
        .collect();

    CatalogMatchReport {
        chance_events: Vec::new(),
        splendor_state: None,
        unassigned_maintenance_seconds: traced
            .unassigned_maintenance_time
            .map(|time| time.as_secs_f64()),
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
        spirit_forest: None,
        spirit_collections: None,
        gemstone_pools: None,
        scores: None,
    }
}

fn tic_tac_toe_report(traced: TracedMatchResult<TicTacToeAction>) -> CatalogMatchReport {
    let game = TicTacToe;
    let mut state = game.initial_state();
    for traced_action in &traced.actions {
        game.apply_action(&mut state, &traced_action.action)
            .expect("trace contains actions accepted by the game");
    }
    let winner = winner_from_utilities(&traced.result.utilities);
    let moves = traced
        .actions
        .into_iter()
        .map(|traced_action| RecordedMove {
            player: traced_action.player.index(),
            action: CatalogAction::TicTacToe {
                row: traced_action.action.row(),
                column: traced_action.action.column(),
            },
            decision_seconds: traced_action.decision_time.as_secs_f64(),
            selection_seconds: traced_action.selection_time.as_secs_f64(),
            maintenance_seconds: traced_action.maintenance_time.as_secs_f64(),
            search_iterations: traced_action.decision_stats.search_iterations,
            search_nodes: traced_action.decision_stats.search_nodes,
            terminal_simulations: traced_action.decision_stats.terminal_simulations,
            cutoff_simulations: traced_action.decision_stats.cutoff_simulations,
            root_actions: traced_action.decision_stats.root_actions,
            tree_reuse: traced_action.decision_stats.tree_reuse,
        })
        .collect();

    CatalogMatchReport {
        chance_events: Vec::new(),
        splendor_state: None,
        unassigned_maintenance_seconds: traced
            .unassigned_maintenance_time
            .map(|time| time.as_secs_f64()),
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
        spirit_forest: None,
        spirit_collections: None,
        gemstone_pools: None,
        scores: None,
    }
}

fn boop_report(traced: TracedMatchResult<BoopAction>) -> CatalogMatchReport {
    let game = Boop;
    let mut state = game.initial_state();
    for traced_action in &traced.actions {
        game.apply_action(&mut state, &traced_action.action)
            .expect("trace contains actions accepted by the game");
    }
    let winner = winner_from_utilities(&traced.result.utilities);
    let moves = traced
        .actions
        .into_iter()
        .map(|traced_action| RecordedMove {
            player: traced_action.player.index(),
            action: CatalogAction::Boop {
                piece: catalog_boop_piece(traced_action.action.piece()),
                row: traced_action.action.position().row(),
                column: traced_action.action.position().column(),
                resolution: match traced_action.action.resolution() {
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
            decision_seconds: traced_action.decision_time.as_secs_f64(),
            selection_seconds: traced_action.selection_time.as_secs_f64(),
            maintenance_seconds: traced_action.maintenance_time.as_secs_f64(),
            search_iterations: traced_action.decision_stats.search_iterations,
            search_nodes: traced_action.decision_stats.search_nodes,
            terminal_simulations: traced_action.decision_stats.terminal_simulations,
            cutoff_simulations: traced_action.decision_stats.cutoff_simulations,
            root_actions: traced_action.decision_stats.root_actions,
            tree_reuse: traced_action.decision_stats.tree_reuse,
        })
        .collect();
    let pools = state.pools().map(|pool| CatalogPool {
        kittens: pool.kittens(),
        cats: pool.cats(),
    });

    CatalogMatchReport {
        chance_events: Vec::new(),
        splendor_state: None,
        unassigned_maintenance_seconds: traced
            .unassigned_maintenance_time
            .map(|time| time.as_secs_f64()),
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
        spirit_forest: None,
        spirit_collections: None,
        gemstone_pools: None,
        scores: None,
    }
}

fn spirits_of_the_forest_report(
    game: &SpiritsOfTheForest,
    traced: TracedMatchResult<SpiritsOfTheForestAction>,
) -> CatalogMatchReport {
    let mut state = game.initial_state();
    for traced_action in &traced.actions {
        game.apply_action(&mut state, &traced_action.action)
            .expect("trace contains actions accepted by the game");
    }
    let winner = winner_from_utilities(&traced.result.utilities);
    let moves = traced
        .actions
        .into_iter()
        .map(|traced_action| RecordedMove {
            player: traced_action.player.index(),
            action: CatalogAction::SpiritsOfTheForest(catalog_spirits_action(traced_action.action)),
            decision_seconds: traced_action.decision_time.as_secs_f64(),
            selection_seconds: traced_action.selection_time.as_secs_f64(),
            maintenance_seconds: traced_action.maintenance_time.as_secs_f64(),
            search_iterations: traced_action.decision_stats.search_iterations,
            search_nodes: traced_action.decision_stats.search_nodes,
            terminal_simulations: traced_action.decision_stats.terminal_simulations,
            cutoff_simulations: traced_action.decision_stats.cutoff_simulations,
            root_actions: traced_action.decision_stats.root_actions,
            tree_reuse: traced_action.decision_stats.tree_reuse,
        })
        .collect();
    let spirit_forest = (0..meeple_bots_spirits_of_the_forest::TILE_COUNT)
        .map(|index| {
            state.remaining()[index].then(|| {
                let position = ForestPosition::new(
                    (index / meeple_bots_spirits_of_the_forest::COLUMNS) as u8,
                    (index % meeple_bots_spirits_of_the_forest::COLUMNS) as u8,
                )
                .expect("catalog index is inside the forest");
                let tile = game.tile(position);
                CatalogSpiritTile {
                    spirit: catalog_spirit(tile.spirit()),
                    spirit_symbols: tile.spirit_symbols(),
                    power_source: tile.power_source().map(catalog_power_source),
                    gemstone: state.gemstones()[index].map(PlayerId::index),
                }
            })
        })
        .collect();
    let spirit_collections = state
        .collections()
        .map(|collection| CatalogSpiritCollection {
            spirit_symbols: *collection.spirit_symbols(),
            power_sources: *collection.power_sources(),
            tiles: collection.tiles(),
        });
    let gemstone_pools = state.gemstone_pools().map(|pool| CatalogGemstonePool {
        available: pool.available(),
        placed: pool.placed(),
        removed: pool.removed(),
    });

    CatalogMatchReport {
        chance_events: Vec::new(),
        splendor_state: None,
        unassigned_maintenance_seconds: traced
            .unassigned_maintenance_time
            .map(|time| time.as_secs_f64()),
        seed: traced.result.seed,
        plies: traced.result.plies,
        utilities: traced.result.utilities,
        winner,
        moves,
        final_board: Vec::new(),
        pools: None,
        spirit_forest: Some(spirit_forest),
        spirit_collections: Some(spirit_collections),
        gemstone_pools: Some(gemstone_pools),
        scores: Some(game.scores(&state)),
    }
}

pub fn catalog_spirits_action(action: SpiritsOfTheForestAction) -> CatalogSpiritsAction {
    match action {
        SpiritsOfTheForestAction::TakeTile {
            position,
            sacrifice,
        } => CatalogSpiritsAction::TakeTile {
            row: position.row(),
            column: position.column(),
            sacrifice: sacrifice.map(|sacrifice| match sacrifice {
                GemstoneSacrifice::Available => CatalogGemstoneSacrifice::Available,
                GemstoneSacrifice::Forest(position) => CatalogGemstoneSacrifice::Forest {
                    row: position.row(),
                    column: position.column(),
                },
            }),
        },
        SpiritsOfTheForestAction::EndCollection => CatalogSpiritsAction::EndCollection,
        SpiritsOfTheForestAction::PlaceGemstone { target } => CatalogSpiritsAction::PlaceGemstone {
            row: target.row(),
            column: target.column(),
        },
        SpiritsOfTheForestAction::MoveGemstone { source, target } => {
            CatalogSpiritsAction::MoveGemstone {
                source_row: source.row(),
                source_column: source.column(),
                target_row: target.row(),
                target_column: target.column(),
            }
        }
        SpiritsOfTheForestAction::SkipGemstone => CatalogSpiritsAction::SkipGemstone,
    }
}

pub const fn catalog_spirit(spirit: Spirit) -> CatalogSpirit {
    match spirit {
        Spirit::Moss => CatalogSpirit::Moss,
        Spirit::Flowers => CatalogSpirit::Flowers,
        Spirit::Fruits => CatalogSpirit::Fruits,
        Spirit::Mushrooms => CatalogSpirit::Mushrooms,
        Spirit::Water => CatalogSpirit::Water,
        Spirit::Vines => CatalogSpirit::Vines,
        Spirit::Branches => CatalogSpirit::Branches,
        Spirit::Leaves => CatalogSpirit::Leaves,
        Spirit::Webs => CatalogSpirit::Webs,
    }
}

pub const fn catalog_power_source(source: PowerSource) -> CatalogPowerSource {
    match source {
        PowerSource::Fire => CatalogPowerSource::Fire,
        PowerSource::Moon => CatalogPowerSource::Moon,
        PowerSource::Sun => CatalogPowerSource::Sun,
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
        GameId::Connect6(_) | GameId::Splendor => (0..matches.get())
            .map(|i| {
                run_match(
                    game,
                    first.clone(),
                    second.clone(),
                    MatchConfig::new(seed.wrapping_add(u64::from(i)), max_plies),
                )
            })
            .collect(),
        GameId::Boop => run_boop_batch(first, second, config),
        GameId::ConnectFour => run_connect_four_batch(first, second, config),
        GameId::SpiritsOfTheForest => run_spirits_of_the_forest_batch(first, second, config),
        GameId::TicTacToe => run_tic_tac_toe_batch(first, second, config),
    }
}

fn run_spirits_of_the_forest_batch(
    first: AgentConfig,
    second: AgentConfig,
    config: BatchConfig,
) -> Result<Vec<MatchResult>, CatalogError> {
    let mut seed_stream = SplitMix64::new(config.seed);
    let mut results = Vec::with_capacity(config.matches.get() as usize);
    for _ in 0..config.matches.get() {
        let match_config = MatchConfig::new(seed_stream.next_u64(), config.max_plies);
        results.push(run_spirits_of_the_forest(
            first.clone(),
            second.clone(),
            match_config,
        )?);
    }
    Ok(results)
}

fn run_boop_batch(
    first: AgentConfig,
    second: AgentConfig,
    config: BatchConfig,
) -> Result<Vec<MatchResult>, CatalogError> {
    let first = ConfiguredAgent::new(first, configured_boop_mcts)?;
    let second = ConfiguredAgent::new(second, configured_boop_mcts)?;
    Ok(play_batch(
        &Boop,
        config,
        || first.clone(),
        || second.clone(),
    )?)
}

fn run_connect_four_batch(
    first: AgentConfig,
    second: AgentConfig,
    config: BatchConfig,
) -> Result<Vec<MatchResult>, CatalogError> {
    let first = ConfiguredAgent::new(first, configured_connect_four_mcts)?;
    let second = ConfiguredAgent::new(second, configured_connect_four_mcts)?;
    Ok(play_batch(
        &ConnectFour,
        config,
        || first.clone(),
        || second.clone(),
    )?)
}

fn run_tic_tac_toe_batch(
    first: AgentConfig,
    second: AgentConfig,
    config: BatchConfig,
) -> Result<Vec<MatchResult>, CatalogError> {
    let first = ConfiguredAgent::new(first, configured_tic_tac_toe_mcts)?;
    let second = ConfiguredAgent::new(second, configured_tic_tac_toe_mcts)?;
    Ok(play_batch(
        &TicTacToe,
        config,
        || first.clone(),
        || second.clone(),
    )?)
}

#[cfg(test)]
mod tests {
    #[test]
    fn game_parameters_are_validated_by_the_catalog() {
        use super::*;
        let default = GameId::Connect6(meeple_bots_connect6::DEFAULT_BOARD_SIZE);
        assert_eq!(
            configure_game(default, &GameParameters::new()).unwrap(),
            default
        );
        assert_eq!(
            configure_game(default, &[("board_size".into(), 11)].into()).unwrap(),
            GameId::Connect6(11)
        );
        for value in [-1, 0, 5, i64::MAX] {
            assert!(configure_game(default, &[("board_size".into(), value)].into()).is_err());
        }
        assert!(configure_game(default, &[("unknown".into(), 11)].into()).is_err());
        for game in [
            GameId::Boop,
            GameId::ConnectFour,
            GameId::TicTacToe,
            GameId::SpiritsOfTheForest,
            GameId::Splendor,
        ] {
            assert_eq!(configure_game(game, &GameParameters::new()).unwrap(), game);
            assert!(configure_game(game, &[("board_size".into(), 11)].into()).is_err());
        }
    }
    use meeple_bots_core::HeuristicParameters;
    use meeple_bots_mcts_agent::{MctsAgent, RolloutPolicy};
    use meeple_bots_spirits_of_the_forest::TurnPhase;
    use std::num::NonZeroU32;

    use super::*;

    fn mcts(heuristic: Option<u32>) -> AgentConfig {
        AgentConfig::Mcts(MctsAgentConfig {
            search: MctsConfig {
                selection_policy: Default::default(),
                budget: SearchBudget::Iterations(NonZeroU32::new(4).unwrap()),
                exploration: std::f64::consts::SQRT_2,
                rollout_depth: 1,
                rollout_policy: RolloutPolicyConfig::UniformRandom.into(),
            },
            cutoff_evaluator: heuristic
                .map_or(EvaluatorConfig::Neutral, EvaluatorConfig::game_heuristic),
            progressive_bias: ConfiguredSelectionBias::None,
            root_diagnostics: false,
            tree_reuse: false,
            transpositions: false,
        })
    }

    #[test]
    fn configured_participants_preserve_typed_traces_and_lifecycle() {
        fn compare<G, M>(game: &G, agent: M)
        where
            G: DeterministicGame,
            G::Action: Clone + PartialEq + std::fmt::Debug,
            M: Agent<G> + Clone,
        {
            for seed in [3, 17] {
                let config = MatchConfig {
                    seed,
                    ..MatchConfig::default()
                };
                let direct = play_typed_match_with_trace(
                    game,
                    &mut agent.clone(),
                    &mut agent.clone(),
                    config,
                )
                .unwrap();
                let wrapped = play_typed_match_with_trace(
                    game,
                    &mut ConfiguredAgent::Mcts(agent.clone()),
                    &mut ConfiguredAgent::Mcts(agent.clone()),
                    config,
                )
                .unwrap();
                assert_eq!(direct.result, wrapped.result);
                assert_eq!(direct.actions.len(), wrapped.actions.len());
                for (a, b) in direct.actions.iter().zip(&wrapped.actions) {
                    assert_eq!(
                        (a.player, &a.action, &a.decision_stats),
                        (b.player, &b.action, &b.decision_stats)
                    );
                }
            }
        }
        let AgentConfig::Mcts(mut config) = mcts(None) else {
            unreachable!()
        };
        config.root_diagnostics = true;
        config.tree_reuse = true;
        for transpositions in [false, true] {
            config.transpositions = transpositions;
            compare(
                &TicTacToe,
                configured_tic_tac_toe_mcts(config.clone()).unwrap(),
            );
            compare(
                &ConnectFour,
                configured_connect_four_mcts(config.clone()).unwrap(),
            );
            compare(&Boop, configured_boop_mcts(config.clone()).unwrap());
            compare(
                &spirits_of_the_forest_game(17),
                configured_spirits_of_the_forest_mcts(config.clone()).unwrap(),
            );
        }
    }

    #[test]
    fn independent_participants_preserve_all_pairings_and_batch_seeds() {
        for game in [
            GameId::Boop,
            GameId::ConnectFour,
            GameId::TicTacToe,
            GameId::SpiritsOfTheForest,
        ] {
            for first in [AgentConfig::Random, mcts(None)] {
                for second in [AgentConfig::Random, mcts(None)] {
                    let batch_config = BatchConfig {
                        matches: NonZeroU32::new(2).unwrap(),
                        seed: 7,
                        max_plies: MatchConfig::default().max_plies,
                    };
                    let batch = run_batch(
                        game,
                        first.clone(),
                        second.clone(),
                        batch_config.seed,
                        batch_config.matches,
                        batch_config.max_plies,
                    )
                    .unwrap();
                    let mut seeds = SplitMix64::new(batch_config.seed);
                    for result in batch {
                        let config = MatchConfig::new(seeds.next_u64(), batch_config.max_plies);
                        assert_eq!(
                            result,
                            run_match(game, first.clone(), second.clone(), config).unwrap()
                        );
                        let trace =
                            run_match_with_trace(game, first.clone(), second.clone(), config)
                                .unwrap();
                        assert_eq!(result.seed, trace.seed);
                        assert_eq!(result.plies, trace.plies);
                        assert_eq!(result.utilities, trace.utilities);
                    }
                }
            }
        }
    }

    #[test]
    fn runtime_catalog_runs_independently_configured_participants() {
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
    fn root_diagnostics_are_optional_for_every_game_and_search_mode() {
        for game in [
            GameId::Boop,
            GameId::ConnectFour,
            GameId::SpiritsOfTheForest,
            GameId::TicTacToe,
        ] {
            for transpositions in [false, true] {
                for enabled in [false, true] {
                    let AgentConfig::Mcts(mut config) = mcts(None) else {
                        unreachable!()
                    };
                    config.root_diagnostics = enabled;
                    config.transpositions = transpositions;
                    let report = run_match_with_trace(
                        game,
                        AgentConfig::Mcts(config),
                        AgentConfig::Random,
                        MatchConfig::default(),
                    )
                    .unwrap();
                    assert!(!report.moves.is_empty());
                    let roots = &report.moves[0].root_actions;
                    assert_eq!(
                        !roots.is_empty(),
                        enabled,
                        "{game:?}, transpositions={transpositions}"
                    );
                    if enabled {
                        assert_eq!(roots.iter().filter(|root| root.selected).count(), 1);
                    }
                }
            }
        }
    }

    #[test]
    fn connect_four_mast_epsilon_one_preserves_uniform_search() {
        use meeple_bots_core::{DecisionContext, PositionStatus};
        let AgentConfig::Mcts(mut config) = mcts(None) else {
            unreachable!()
        };
        config.search.budget = SearchBudget::Iterations(NonZeroU32::new(64).unwrap());
        config.search.rollout_depth = 42;
        config.root_diagnostics = true;
        let mut uniform = configured_connect_four_mcts(config.clone()).unwrap();
        config.search.rollout_policy = RolloutPolicyConfig::Mast { epsilon: 1.0 }.into();
        let mut mast = configured_connect_four_mcts(config).unwrap();
        let game = ConnectFour;
        let mut state = game.initial_state();
        let mut uniform_rng = SplitMix64::new(42);
        let mut mast_rng = SplitMix64::new(42);
        while let PositionStatus::PlayerTurn(player) = game.status(&state) {
            let a = uniform
                .select_action(
                    DecisionContext::new(&game, &state, player),
                    &mut uniform_rng,
                )
                .unwrap();
            let b = mast
                .select_action(DecisionContext::new(&game, &state, player), &mut mast_rng)
                .unwrap();
            assert_eq!(a, b);
            assert_eq!(uniform.last_decision_stats(), mast.last_decision_stats());
            assert!(!mast.last_decision_stats().root_actions.is_empty());
            game.apply_action(&mut state, &a).unwrap();
        }
    }

    #[test]
    fn connect_four_mast_validates_epsilon_and_rejects_heuristics() {
        let AgentConfig::Mcts(mut config) = mcts(None) else {
            unreachable!()
        };
        for epsilon in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            config.search.rollout_policy = RolloutPolicyConfig::Mast { epsilon }.into();
            assert!(configured_connect_four_mcts(config.clone()).is_err());
        }
        config.search.rollout_policy = RolloutPolicyConfig::Mast { epsilon: 0.25 }.into();
        assert!(configured_connect_four_mcts(config.clone()).is_ok());
        config.cutoff_evaluator = EvaluatorConfig::game_heuristic(0);
        assert!(configured_connect_four_mcts(config).is_err());
    }

    #[test]
    fn runtime_catalog_benchmarks_a_concrete_mcts_configuration() {
        let AgentConfig::Mcts(config) = mcts(None) else {
            unreachable!();
        };

        let benchmark = benchmark_mcts_agent(GameId::TicTacToe, config, 6, 42).unwrap();

        assert_eq!(benchmark.sampled_positions, 3);
        assert!(benchmark.decision_time_mean_ms > 0.0);
        assert!(benchmark.milliseconds_per_iteration > 0.0);
    }

    #[test]
    fn rejects_unknown_or_unsupported_heuristics() {
        let unknown = configured_boop_mcts(match mcts(Some(2)) {
            AgentConfig::Mcts(config) => config,
            AgentConfig::Random => unreachable!(),
        })
        .unwrap_err();
        assert!(unknown.to_string().contains("available indices: 0..1"));

        let unsupported = run_match(
            GameId::TicTacToe,
            mcts(Some(0)),
            AgentConfig::Random,
            MatchConfig::default(),
        )
        .unwrap_err();
        assert_eq!(
            unsupported.to_string(),
            "tic-tac-toe does not provide MCTS heuristics"
        );

        let AgentConfig::Mcts(mut informed_without_cutoff_heuristic) = mcts(None) else {
            unreachable!();
        };
        informed_without_cutoff_heuristic.search.rollout_policy =
            RolloutPolicyConfig::EpsilonGreedy {
                epsilon: 0.1,
                evaluator: EvaluatorConfig::game_heuristic(0),
            }
            .into();
        configured_boop_mcts(informed_without_cutoff_heuristic).unwrap();

        let AgentConfig::Mcts(mut invalid_rollout_heuristic) = mcts(None) else {
            unreachable!();
        };
        invalid_rollout_heuristic.search.rollout_policy = RolloutPolicyConfig::Greedy {
            evaluator: EvaluatorConfig::game_heuristic(2),
        }
        .into();
        let error = configured_boop_mcts(invalid_rollout_heuristic).unwrap_err();
        assert!(error.to_string().contains("available indices: 0..1"));

        let AgentConfig::Mcts(spirits_h0) = mcts(Some(0)) else {
            unreachable!();
        };
        configured_spirits_of_the_forest_mcts(spirits_h0).unwrap();

        let AgentConfig::Mcts(spirits_h1) = mcts(Some(1)) else {
            unreachable!();
        };
        let error = configured_spirits_of_the_forest_mcts(spirits_h1).unwrap_err();
        assert!(error.to_string().contains("available index: 0"));
    }

    #[test]
    fn validates_named_heuristic_parameters_against_the_game_schema() {
        let configured = |name: &str, value: f64| {
            let AgentConfig::Mcts(mut config) = mcts(None) else {
                unreachable!();
            };
            config.cutoff_evaluator = EvaluatorConfig::GameHeuristic {
                index: 0,
                parameters: HeuristicParameters::from([(name.to_owned(), value)]),
            };
            config
        };

        configured_spirits_of_the_forest_mcts(configured("gemstone_early_bonus", 2.0)).unwrap();

        let unknown =
            configured_spirits_of_the_forest_mcts(configured("unknown", 2.0)).unwrap_err();
        assert_eq!(
            unknown.to_string(),
            "spotf heuristic 0 does not accept parameter \"unknown\""
        );

        let negative =
            configured_spirits_of_the_forest_mcts(configured("gemstone_early_bonus", -1.0))
                .unwrap_err();
        assert_eq!(
            negative.to_string(),
            "spotf heuristic 0 parameter \"gemstone_early_bonus\" must be at least 0"
        );

        let unsupported =
            configured_boop_mcts(configured("gemstone_early_bonus", 2.0)).unwrap_err();
        assert_eq!(
            unsupported.to_string(),
            "boop heuristic 0 does not accept parameter \"gemstone_early_bonus\""
        );
    }

    #[test]
    fn rejects_invalid_search_exploration() {
        for invalid in [f64::NAN, f64::INFINITY, -0.1] {
            let AgentConfig::Mcts(mut config) = mcts(None) else {
                unreachable!();
            };
            config.search.exploration = invalid;

            let error = configured_tic_tac_toe_mcts(config).unwrap_err();
            assert_eq!(
                error.to_string(),
                "MCTS exploration must be finite and non-negative"
            );
        }
    }

    #[test]
    fn spotf_conditional_rollout_only_uses_its_primary_in_the_matching_phase() {
        let game = spirits_of_the_forest_game(7);
        let mut state = game.initial_state();
        let policy = ConfiguredRolloutPolicy::Conditional {
            condition: RolloutConditionConfig::TurnPhase(CatalogTurnPhase::Collect),
            primary: RolloutPolicyConfig::EpsilonGreedy {
                epsilon: 0.0,
                evaluator: EvaluatorConfig::game_heuristic(99),
            },
            fallback: RolloutPolicyConfig::UniformRandom,
        };

        let collect_error = policy
            .select_action(
                &game,
                &state,
                PlayerId::FIRST,
                PlayerId::FIRST,
                &mut SplitMix64::new(11),
            )
            .unwrap_err();
        assert!(collect_error.to_string().contains("heuristic index 99"));

        let take = game
            .legal_actions(&state)
            .next()
            .expect("the initial forest has a tile to take");
        game.apply_action(&mut state, &take).unwrap();
        assert_eq!(state.phase(), TurnPhase::PlaceGemstone);
        let mut conditional_rng = SplitMix64::new(13);
        let mut uniform_rng = SplitMix64::new(13);
        let conditional_action = policy
            .select_action(
                &game,
                &state,
                PlayerId::FIRST,
                PlayerId::FIRST,
                &mut conditional_rng,
            )
            .unwrap();
        let uniform_action = RolloutPolicyConfig::<EvaluatorConfig>::UniformRandom
            .select_action(
                &game,
                &state,
                PlayerId::FIRST,
                PlayerId::FIRST,
                &mut uniform_rng,
            )
            .unwrap();

        assert_eq!(conditional_action, uniform_action);
        assert_eq!(conditional_rng.next_u64(), uniform_rng.next_u64());
    }

    #[test]
    fn spotf_conditional_epsilon_one_matches_uniform_random_search() {
        let game = spirits_of_the_forest_game(17);
        let state = game.initial_state();
        let base = MctsConfig {
            selection_policy: Default::default(),
            budget: SearchBudget::Iterations(NonZeroU32::new(128).unwrap()),
            exploration: std::f64::consts::SQRT_2,
            rollout_depth: 32,
            rollout_policy: ConfiguredRolloutPolicy::Standard(RolloutPolicyConfig::UniformRandom),
        };
        let mut uniform = MctsAgent::with_cutoff_evaluator(base.clone(), EvaluatorConfig::Neutral);
        let mut conditional = MctsAgent::with_cutoff_evaluator(
            MctsConfig {
                rollout_policy: ConfiguredRolloutPolicy::Conditional {
                    condition: RolloutConditionConfig::TurnPhase(CatalogTurnPhase::Collect),
                    primary: RolloutPolicyConfig::EpsilonGreedy {
                        epsilon: 1.0,
                        evaluator: EvaluatorConfig::game_heuristic(0),
                    },
                    fallback: RolloutPolicyConfig::UniformRandom,
                },
                ..base
            },
            EvaluatorConfig::Neutral,
        );
        let mut uniform_rng = SplitMix64::new(23);
        let mut conditional_rng = SplitMix64::new(23);

        let uniform_action = uniform
            .select_action(
                meeple_bots_core::DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut uniform_rng,
            )
            .unwrap();
        let conditional_action = conditional
            .select_action(
                meeple_bots_core::DecisionContext::new(&game, &state, PlayerId::FIRST),
                &mut conditional_rng,
            )
            .unwrap();

        assert_eq!(conditional_action, uniform_action);
        assert_eq!(conditional.decision_stats(), uniform.decision_stats());
        assert_eq!(conditional_rng.next_u64(), uniform_rng.next_u64());
    }

    #[test]
    fn turn_phase_rollout_condition_is_rejected_for_other_games() {
        let AgentConfig::Mcts(mut config) = mcts(None) else {
            unreachable!();
        };
        config.search.rollout_policy = ConfiguredRolloutPolicy::Conditional {
            condition: RolloutConditionConfig::TurnPhase(CatalogTurnPhase::Collect),
            primary: RolloutPolicyConfig::UniformRandom,
            fallback: RolloutPolicyConfig::UniformRandom,
        };

        let error = configured_boop_mcts(config).unwrap_err();
        assert_eq!(
            error.to_string(),
            "turn-phase rollout conditions are only supported by spotf"
        );
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
                CatalogAction::Connect6 { .. } => panic!("unexpected Connect6 action"),
                CatalogAction::ConnectFour { .. } => panic!("unexpected Connect Four action"),
                CatalogAction::Boop { .. } => panic!("unexpected boop action"),
                CatalogAction::Splendor(_) => panic!("unexpected Splendor action"),
                CatalogAction::SpiritsOfTheForest(_) => {
                    panic!("unexpected Spirits of the Forest action")
                }
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
                CatalogAction::Connect6 { .. } => panic!("unexpected Connect6 action"),
                CatalogAction::ConnectFour { column } => assert!(column < 7),
                CatalogAction::TicTacToe { .. } => panic!("unexpected tic-tac-toe action"),
                CatalogAction::Boop { .. } => panic!("unexpected boop action"),
                CatalogAction::Splendor(_) => panic!("unexpected Splendor action"),
                CatalogAction::SpiritsOfTheForest(_) => {
                    panic!("unexpected Spirits of the Forest action")
                }
            }
        }
    }

    #[test]
    fn spirits_trace_finishes_with_scores_and_collections() {
        let report = run_match_with_trace(
            GameId::SpiritsOfTheForest,
            AgentConfig::Random,
            AgentConfig::Random,
            MatchConfig::default(),
        )
        .unwrap();

        assert_eq!(report.moves.len(), report.plies as usize);
        assert!(
            report
                .moves
                .iter()
                .all(|movement| matches!(movement.action, CatalogAction::SpiritsOfTheForest(_)))
        );
        assert_eq!(report.spirit_forest.as_ref().unwrap().len(), 48);
        assert!(
            report
                .spirit_forest
                .as_ref()
                .unwrap()
                .iter()
                .all(Option::is_none)
        );
        assert_eq!(
            report
                .spirit_collections
                .as_ref()
                .unwrap()
                .iter()
                .map(|collection| usize::from(collection.tiles))
                .sum::<usize>(),
            48
        );
        assert!(report.scores.is_some());
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
    fn trace_analysis_dispatches_to_all_supported_games() {
        let report = run_match_with_trace(
            GameId::Boop,
            AgentConfig::Random,
            AgentConfig::Random,
            MatchConfig::default(),
        )
        .unwrap();
        let expected_winner = report.winner.unwrap();
        let CatalogTraceAnalysis::Boop(analysis) =
            analyze_trace(GameId::Boop, &report.moves).unwrap()
        else {
            panic!("boop trace returned the wrong analysis type");
        };
        assert_eq!(analysis.winner.index(), expected_winner);

        let spirits_report = run_match_with_trace(
            GameId::SpiritsOfTheForest,
            AgentConfig::Random,
            AgentConfig::Random,
            MatchConfig::default(),
        )
        .unwrap();
        let CatalogTraceAnalysis::SpiritsOfTheForest(spirits_analysis) = analyze_seeded_trace(
            GameId::SpiritsOfTheForest,
            &spirits_report.moves,
            spirits_report.seed,
        )
        .unwrap() else {
            panic!("spotf trace returned the wrong analysis type");
        };
        assert_eq!(
            spirits_analysis.winner.map(PlayerId::index),
            spirits_report.winner
        );
        assert_eq!(
            spirits_analysis.final_scores,
            spirits_report.scores.unwrap()
        );

        for game in [GameId::ConnectFour, GameId::TicTacToe] {
            let report = run_match_with_trace(
                game,
                AgentConfig::Random,
                AgentConfig::Random,
                MatchConfig::default(),
            )
            .unwrap();
            let CatalogTraceAnalysis::Generic { utilities } =
                analyze_trace(game, &report.moves).unwrap()
            else {
                panic!("expected generic replay analysis");
            };
            assert_eq!(utilities.as_slice(), report.utilities);
            assert!(analyze_trace(game, &[]).is_err());
            assert!(analyze_trace(game, &report.moves[..report.moves.len() - 1]).is_err());
            let mut invalid = report.moves.clone();
            invalid[0].player = 1;
            assert!(analyze_trace(game, &invalid).is_err());
            invalid = report.moves.clone();
            invalid.push(invalid[0].clone());
            assert!(analyze_trace(game, &invalid).is_err());
        }
    }

    #[test]
    fn boop_evaluation_is_structurally_larger_than_tic_tac_toe() {
        let config = EvaluationConfig {
            samples: NonZeroU32::new(16).unwrap(),
            ..EvaluationConfig::default()
        };
        let tic_tac_toe = evaluate_game(GameId::TicTacToe, config).unwrap();
        let boop = evaluate_game(GameId::Boop, config).unwrap();

        assert!(boop.initial_legal_actions > tic_tac_toe.initial_legal_actions);
        assert!(boop.effective_branching_factor > tic_tac_toe.effective_branching_factor);
        assert!(boop.estimated_depth > tic_tac_toe.estimated_depth);
        assert!(boop.estimated_tree_log10 > tic_tac_toe.estimated_tree_log10);
    }
}
