//! Runtime configuration boundary for the statically dispatched engine.

use std::{error::Error, fmt, num::NonZeroU32};

use meeple_bots_boop::{
    Boop, BoopAction, BoopReplayAnalysis, GraduateLine, PieceKind as BoopPieceKind,
    Position as BoopPosition, Resolution as BoopResolution, analyze_replay as analyze_boop_replay,
};
use meeple_bots_connect_four::{ConnectFour, ConnectFourAction};
use meeple_bots_core::{
    Agent, AgentError, DeterministicGame, Game, HeuristicGame, HeuristicParameters, PlayerId,
    RandomSource, RootActionStats, TreeReuseStats,
};
pub use meeple_bots_evaluation::{
    EvaluationConfig, EvaluationError, GameEvaluationReport, IterationBudgetEstimate,
    MctsAgentBenchmark, RolloutCostEstimate, SampledDecisionTiming, SuggestedMctsExperiment,
};
use meeple_bots_evaluation::{
    benchmark_mcts_agent as benchmark_typed_mcts_agent, evaluate_game as evaluate_typed_game,
};
use meeple_bots_mcts_agent::{
    MctsAgent, PolicyCondition, RolloutPolicy, SelectionBias, StateEvaluator, TreeReuseMctsAgent,
};
pub use meeple_bots_mcts_agent::{MctsConfig, RolloutPolicyConfig, SearchBudget, UniformRandom};
use meeple_bots_random_agent::RandomAgent;
use meeple_bots_simulation::{
    BatchConfig, MatchError, MatchObserver, SplitMix64, TracedMatchResult, play_batch, play_match,
    play_match_with_trace as play_typed_match_with_trace, play_match_with_trace_and_observer,
};
pub use meeple_bots_simulation::{MatchConfig, MatchResult};
use meeple_bots_spirits_of_the_forest::{
    ForestPosition, GemstoneSacrifice, PowerSource, Spirit, SpiritsOfTheForest,
    SpiritsOfTheForestAction, SpiritsReplayAnalysis, TurnPhase,
    analyze_replay as analyze_spirits_replay,
};
use meeple_bots_tic_tac_toe::{TicTacToe, TicTacToeAction};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GameId {
    Boop,
    ConnectFour,
    SpiritsOfTheForest,
    TicTacToe,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AgentConfig {
    Random,
    Mcts(MctsAgentConfig),
}

#[derive(Clone, Debug, PartialEq)]
pub struct MctsAgentConfig {
    pub search: MctsConfig<ConfiguredRolloutPolicy>,
    pub cutoff_evaluator: EvaluatorConfig,
    pub progressive_bias: ConfiguredSelectionBias,
    pub root_diagnostics: bool,
    pub tree_reuse: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum ConfiguredSelectionBias {
    #[default]
    None,
    Progressive {
        weight: f64,
        evaluator: EvaluatorConfig,
        condition: Option<RolloutConditionConfig>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RolloutConditionConfig {
    TurnPhase(CatalogTurnPhase),
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConfiguredRolloutPolicy {
    Standard(RolloutPolicyConfig<EvaluatorConfig>),
    Conditional {
        condition: RolloutConditionConfig,
        primary: RolloutPolicyConfig<EvaluatorConfig>,
        fallback: RolloutPolicyConfig<EvaluatorConfig>,
    },
}

impl Default for ConfiguredRolloutPolicy {
    fn default() -> Self {
        Self::Standard(RolloutPolicyConfig::UniformRandom)
    }
}

impl From<RolloutPolicyConfig<EvaluatorConfig>> for ConfiguredRolloutPolicy {
    fn from(policy: RolloutPolicyConfig<EvaluatorConfig>) -> Self {
        Self::Standard(policy)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum EvaluatorConfig {
    #[default]
    Neutral,
    GameHeuristic {
        index: u32,
        parameters: HeuristicParameters,
    },
}

impl EvaluatorConfig {
    pub fn game_heuristic(index: u32) -> Self {
        Self::GameHeuristic {
            index,
            parameters: HeuristicParameters::new(),
        }
    }
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
    pub search_iterations: Option<u64>,
    pub search_nodes: Option<u64>,
    pub root_actions: Vec<RootActionStats>,
    pub tree_reuse: Option<TreeReuseStats>,
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
    pub spirit_forest: Option<Vec<Option<CatalogSpiritTile>>>,
    pub spirit_collections: Option<[CatalogSpiritCollection; 2]>,
    pub gemstone_pools: Option<[CatalogGemstonePool; 2]>,
    pub scores: Option<[i16; 2]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CatalogTraceAnalysis {
    Boop(BoopReplayAnalysis),
    SpiritsOfTheForest(SpiritsReplayAnalysis),
}

#[derive(Debug)]
pub enum CatalogError {
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
        GameId::Boop => evaluate_typed_game(&Boop, config),
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
        GameId::ConnectFour | GameId::TicTacToe => Err(CatalogError::AnalysisUnavailable(game)),
    }
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

pub type BoopMctsAgent = TreeReuseMctsAgent<
    Boop,
    EvaluatorConfig,
    RolloutPolicyConfig<EvaluatorConfig>,
    ConfiguredSelectionBias,
>;
pub type SpiritsOfTheForestMctsAgent = TreeReuseMctsAgent<
    SpiritsOfTheForest,
    EvaluatorConfig,
    ConfiguredRolloutPolicy,
    ConfiguredSelectionBias,
>;
pub type ConnectFourMctsAgent =
    TreeReuseMctsAgent<ConnectFour, meeple_bots_mcts_agent::NeutralEvaluator, UniformRandom>;
pub type TicTacToeMctsAgent =
    TreeReuseMctsAgent<TicTacToe, meeple_bots_mcts_agent::NeutralEvaluator, UniformRandom>;

#[derive(Clone, Copy, Debug)]
struct SpiritsTurnPhaseCondition(CatalogTurnPhase);

impl PolicyCondition<SpiritsOfTheForest> for SpiritsTurnPhaseCondition {
    fn matches(
        &self,
        _game: &SpiritsOfTheForest,
        state: &<SpiritsOfTheForest as Game>::State,
        _active_player: PlayerId,
        _root_player: PlayerId,
    ) -> bool {
        matches!(
            (self.0, state.phase()),
            (CatalogTurnPhase::Collect, TurnPhase::Collect)
                | (CatalogTurnPhase::PlaceGemstone, TurnPhase::PlaceGemstone)
        )
    }
}

impl SelectionBias<SpiritsOfTheForest> for ConfiguredSelectionBias {
    fn weight(&self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Progressive { weight, .. } => *weight,
        }
    }

    fn applies(
        &self,
        game: &SpiritsOfTheForest,
        state: &<SpiritsOfTheForest as Game>::State,
        active_player: PlayerId,
        root_player: PlayerId,
    ) -> bool {
        match self {
            Self::None => false,
            Self::Progressive {
                condition: None, ..
            } => true,
            Self::Progressive {
                condition: Some(RolloutConditionConfig::TurnPhase(phase)),
                ..
            } => SpiritsTurnPhaseCondition(*phase).matches(game, state, active_player, root_player),
        }
    }

    fn evaluate_child(
        &self,
        game: &SpiritsOfTheForest,
        state: &<SpiritsOfTheForest as Game>::State,
        root_player: PlayerId,
    ) -> Result<f64, AgentError> {
        match self {
            Self::None => Ok(0.0),
            Self::Progressive { evaluator, .. } => evaluator.evaluate(game, state, root_player),
        }
    }
}

impl SelectionBias<Boop> for ConfiguredSelectionBias {
    fn weight(&self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Progressive { weight, .. } => *weight,
        }
    }

    fn applies(
        &self,
        _game: &Boop,
        _state: &<Boop as Game>::State,
        _active_player: PlayerId,
        _root_player: PlayerId,
    ) -> bool {
        matches!(
            self,
            Self::Progressive {
                condition: None,
                ..
            }
        )
    }

    fn evaluate_child(
        &self,
        game: &Boop,
        state: &<Boop as Game>::State,
        root_player: PlayerId,
    ) -> Result<f64, AgentError> {
        match self {
            Self::None => Ok(0.0),
            Self::Progressive { evaluator, .. } => evaluator.evaluate(game, state, root_player),
        }
    }
}

impl RolloutPolicy<SpiritsOfTheForest> for ConfiguredRolloutPolicy {
    fn select_action<R: RandomSource + ?Sized>(
        &self,
        game: &SpiritsOfTheForest,
        state: &<SpiritsOfTheForest as Game>::State,
        active_player: PlayerId,
        root_player: PlayerId,
        rng: &mut R,
    ) -> Result<SpiritsOfTheForestAction, AgentError> {
        match self {
            Self::Standard(policy) => {
                policy.select_action(game, state, active_player, root_player, rng)
            }
            Self::Conditional {
                condition: RolloutConditionConfig::TurnPhase(phase),
                primary,
                fallback,
            } => {
                let policy = if SpiritsTurnPhaseCondition(*phase).matches(
                    game,
                    state,
                    active_player,
                    root_player,
                ) {
                    primary
                } else {
                    fallback
                };
                policy.select_action(game, state, active_player, root_player, rng)
            }
        }
    }
}

impl<G> StateEvaluator<G> for EvaluatorConfig
where
    G: DeterministicGame + HeuristicGame,
{
    fn evaluate(
        &self,
        game: &G,
        state: &G::State,
        perspective: PlayerId,
    ) -> Result<f64, AgentError> {
        match self {
            Self::Neutral => Ok(0.0),
            Self::GameHeuristic { index, parameters } => game
                .heuristic_utility_with_parameters(*index, parameters, state, perspective)
                .map(f64::from)
                .ok_or_else(|| {
                    AgentError::message(format!(
                        "heuristic index {index} is not available; the game provides {} heuristics",
                        game.heuristic_count()
                    ))
                }),
        }
    }
}

pub fn configured_boop_mcts(config: MctsAgentConfig) -> Result<BoopMctsAgent, CatalogError> {
    validate_agent_evaluators(GameId::Boop, &Boop, &config)?;
    Ok(TreeReuseMctsAgent::new(
        MctsAgent::with_progressive_bias(
            standard_search_config(config.search)?,
            config.cutoff_evaluator,
            config.progressive_bias,
        )
        .with_root_diagnostics(config.root_diagnostics),
        config.tree_reuse,
    ))
}

pub fn configured_spirits_of_the_forest_mcts(
    config: MctsAgentConfig,
) -> Result<SpiritsOfTheForestMctsAgent, CatalogError> {
    let game = spirits_of_the_forest_game(0);
    validate_agent_evaluators(GameId::SpiritsOfTheForest, &game, &config)?;
    Ok(TreeReuseMctsAgent::new(
        MctsAgent::with_progressive_bias(
            config.search,
            config.cutoff_evaluator,
            config.progressive_bias,
        )
        .with_root_diagnostics(config.root_diagnostics),
        config.tree_reuse,
    ))
}

pub fn configured_connect_four_mcts(
    config: MctsAgentConfig,
) -> Result<ConnectFourMctsAgent, CatalogError> {
    validate_uninformed_agent(GameId::ConnectFour, &config)?;
    Ok(TreeReuseMctsAgent::new(
        MctsAgent::new(uniform_search_config(config.search)),
        config.tree_reuse,
    ))
}

pub fn configured_tic_tac_toe_mcts(
    config: MctsAgentConfig,
) -> Result<TicTacToeMctsAgent, CatalogError> {
    validate_uninformed_agent(GameId::TicTacToe, &config)?;
    Ok(TreeReuseMctsAgent::new(
        MctsAgent::new(uniform_search_config(config.search)),
        config.tree_reuse,
    ))
}

fn validate_agent_evaluators<G: HeuristicGame>(
    game_id: GameId,
    game: &G,
    config: &MctsAgentConfig,
) -> Result<(), CatalogError> {
    validate_search_config(&config.search)?;
    validate_evaluator(game_id, game, &config.cutoff_evaluator)?;
    validate_selection_bias(game_id, game, &config.progressive_bias)?;
    match &config.search.rollout_policy {
        ConfiguredRolloutPolicy::Standard(policy) => {
            validate_base_rollout_policy(game_id, game, policy)?;
        }
        ConfiguredRolloutPolicy::Conditional {
            condition,
            primary,
            fallback,
        } => {
            validate_rollout_condition(game_id, *condition)?;
            validate_base_rollout_policy(game_id, game, primary)?;
            validate_base_rollout_policy(game_id, game, fallback)?;
        }
    }
    Ok(())
}

fn validate_selection_bias<G: HeuristicGame>(
    game_id: GameId,
    game: &G,
    bias: &ConfiguredSelectionBias,
) -> Result<(), CatalogError> {
    let ConfiguredSelectionBias::Progressive {
        weight,
        evaluator,
        condition,
    } = bias
    else {
        return Ok(());
    };
    if !weight.is_finite() || *weight < 0.0 {
        return Err(CatalogError::InvalidMctsConfig(
            "MCTS progressive bias weight must be finite and non-negative",
        ));
    }
    validate_evaluator(game_id, game, evaluator)?;
    if let Some(condition) = condition {
        validate_selection_condition(game_id, *condition)?;
    }
    Ok(())
}

fn validate_selection_condition(
    game: GameId,
    condition: RolloutConditionConfig,
) -> Result<(), CatalogError> {
    match (game, condition) {
        (GameId::SpiritsOfTheForest, RolloutConditionConfig::TurnPhase(_)) => Ok(()),
        (_, RolloutConditionConfig::TurnPhase(_)) => Err(CatalogError::InvalidMctsConfig(
            "turn-phase progressive bias conditions are only supported by spotf",
        )),
    }
}

fn validate_base_rollout_policy<G: HeuristicGame>(
    game_id: GameId,
    game: &G,
    policy: &RolloutPolicyConfig<EvaluatorConfig>,
) -> Result<(), CatalogError> {
    match policy {
        RolloutPolicyConfig::UniformRandom => Ok(()),
        RolloutPolicyConfig::Greedy { evaluator } => validate_evaluator(game_id, game, evaluator),
        RolloutPolicyConfig::EpsilonGreedy { epsilon, evaluator } => {
            if !epsilon.is_finite() || !(0.0..=1.0).contains(epsilon) {
                return Err(CatalogError::InvalidMctsConfig(
                    "MCTS rollout epsilon must be finite and between 0.0 and 1.0",
                ));
            }
            validate_evaluator(game_id, game, evaluator)
        }
    }
}

fn validate_rollout_condition(
    game: GameId,
    condition: RolloutConditionConfig,
) -> Result<(), CatalogError> {
    match (game, condition) {
        (GameId::SpiritsOfTheForest, RolloutConditionConfig::TurnPhase(_)) => Ok(()),
        (_, RolloutConditionConfig::TurnPhase(_)) => Err(CatalogError::InvalidMctsConfig(
            "turn-phase rollout conditions are only supported by spotf",
        )),
    }
}

fn validate_evaluator<G: HeuristicGame>(
    game_id: GameId,
    game: &G,
    evaluator: &EvaluatorConfig,
) -> Result<(), CatalogError> {
    match evaluator {
        EvaluatorConfig::Neutral => Ok(()),
        EvaluatorConfig::GameHeuristic { index, parameters } => {
            validate_heuristic(game_id, game, *index)?;
            validate_heuristic_parameters(game_id, game, *index, parameters)
        }
    }
}

fn validate_heuristic_parameters<G: HeuristicGame>(
    game_id: GameId,
    game: &G,
    index: u32,
    parameters: &HeuristicParameters,
) -> Result<(), CatalogError> {
    let specs = game
        .heuristic_parameter_specs(index)
        .expect("validated heuristic index provides a parameter schema");
    for (name, value) in parameters {
        let Some(spec) = specs.iter().find(|spec| spec.name == name) else {
            return Err(CatalogError::InvalidHeuristicParameter {
                game: game_id,
                index,
                message: format!("does not accept parameter {name:?}"),
            });
        };
        if !value.is_finite() {
            return Err(CatalogError::InvalidHeuristicParameter {
                game: game_id,
                index,
                message: format!("parameter {name:?} must be finite"),
            });
        }
        if spec.minimum.is_some_and(|minimum| *value < minimum) {
            return Err(CatalogError::InvalidHeuristicParameter {
                game: game_id,
                index,
                message: format!(
                    "parameter {name:?} must be at least {}",
                    spec.minimum.unwrap()
                ),
            });
        }
        if spec.maximum.is_some_and(|maximum| *value > maximum) {
            return Err(CatalogError::InvalidHeuristicParameter {
                game: game_id,
                index,
                message: format!(
                    "parameter {name:?} must be at most {}",
                    spec.maximum.unwrap()
                ),
            });
        }
    }
    Ok(())
}

fn validate_uninformed_agent(game: GameId, config: &MctsAgentConfig) -> Result<(), CatalogError> {
    validate_search_config(&config.search)?;
    if let EvaluatorConfig::GameHeuristic { index, .. } = &config.cutoff_evaluator {
        return Err(unsupported_heuristic(game, *index, 0));
    }
    if !matches!(
        config.search.rollout_policy,
        ConfiguredRolloutPolicy::Standard(RolloutPolicyConfig::UniformRandom)
    ) {
        return Err(CatalogError::InvalidMctsConfig(
            "informed rollout requires a game with MCTS evaluators",
        ));
    }
    if !matches!(config.progressive_bias, ConfiguredSelectionBias::None) {
        return Err(CatalogError::InvalidMctsConfig(
            "progressive bias requires a game with MCTS evaluators",
        ));
    }
    Ok(())
}

fn validate_search_config<P>(config: &MctsConfig<P>) -> Result<(), CatalogError> {
    config.validate().map_err(CatalogError::InvalidMctsConfig)?;
    if matches!(config.budget, SearchBudget::Time(duration) if duration.is_zero()) {
        return Err(CatalogError::InvalidMctsConfig(
            "MCTS time budget must be greater than zero",
        ));
    }
    Ok(())
}

fn uniform_search_config(config: MctsConfig<ConfiguredRolloutPolicy>) -> MctsConfig<UniformRandom> {
    MctsConfig {
        budget: config.budget,
        exploration: config.exploration,
        rollout_depth: config.rollout_depth,
        rollout_policy: UniformRandom,
    }
}

fn standard_search_config(
    config: MctsConfig<ConfiguredRolloutPolicy>,
) -> Result<MctsConfig<RolloutPolicyConfig<EvaluatorConfig>>, CatalogError> {
    let ConfiguredRolloutPolicy::Standard(rollout_policy) = config.rollout_policy else {
        return Err(CatalogError::InvalidMctsConfig(
            "conditional rollout is not supported by this game",
        ));
    };
    Ok(MctsConfig {
        budget: config.budget,
        exploration: config.exploration,
        rollout_depth: config.rollout_depth,
        rollout_policy,
    })
}

fn validate_heuristic<G: HeuristicGame>(
    game_id: GameId,
    game: &G,
    index: u32,
) -> Result<(), CatalogError> {
    let available = game.heuristic_count();
    if index < available {
        Ok(())
    } else {
        Err(unsupported_heuristic(game_id, index, available))
    }
}

const fn unsupported_heuristic(game: GameId, index: u32, available: u32) -> CatalogError {
    CatalogError::UnsupportedHeuristic {
        game,
        index,
        available,
    }
}

const fn game_name(game: GameId) -> &'static str {
    match game {
        GameId::Boop => "boop",
        GameId::ConnectFour => "connect-four",
        GameId::SpiritsOfTheForest => "spotf",
        GameId::TicTacToe => "tic-tac-toe",
    }
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
        GameId::Boop => run_boop_with_trace(first, second, config),
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
    let game = ConnectFour;
    let result = match (first, second) {
        (AgentConfig::Random, AgentConfig::Random) => {
            play_match(&game, &mut RandomAgent, &mut RandomAgent, config)
        }
        (AgentConfig::Random, AgentConfig::Mcts(second)) => play_match(
            &game,
            &mut RandomAgent,
            &mut configured_connect_four_mcts(second)?,
            config,
        ),
        (AgentConfig::Mcts(first), AgentConfig::Random) => play_match(
            &game,
            &mut configured_connect_four_mcts(first)?,
            &mut RandomAgent,
            config,
        ),
        (AgentConfig::Mcts(first), AgentConfig::Mcts(second)) => play_match(
            &game,
            &mut configured_connect_four_mcts(first)?,
            &mut configured_connect_four_mcts(second)?,
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
        (AgentConfig::Random, AgentConfig::Mcts(second)) => play_match(
            &game,
            &mut RandomAgent,
            &mut configured_boop_mcts(second)?,
            config,
        ),
        (AgentConfig::Mcts(first), AgentConfig::Random) => play_match(
            &game,
            &mut configured_boop_mcts(first)?,
            &mut RandomAgent,
            config,
        ),
        (AgentConfig::Mcts(first), AgentConfig::Mcts(second)) => play_match(
            &game,
            &mut configured_boop_mcts(first)?,
            &mut configured_boop_mcts(second)?,
            config,
        ),
    }?;
    Ok(result)
}

fn run_spirits_of_the_forest(
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<MatchResult, CatalogError> {
    let game = spirits_of_the_forest_game(config.seed);
    let result = match (first, second) {
        (AgentConfig::Random, AgentConfig::Random) => {
            play_match(&game, &mut RandomAgent, &mut RandomAgent, config)
        }
        (AgentConfig::Random, AgentConfig::Mcts(second)) => play_match(
            &game,
            &mut RandomAgent,
            &mut configured_spirits_of_the_forest_mcts(second)?,
            config,
        ),
        (AgentConfig::Mcts(first), AgentConfig::Random) => play_match(
            &game,
            &mut configured_spirits_of_the_forest_mcts(first)?,
            &mut RandomAgent,
            config,
        ),
        (AgentConfig::Mcts(first), AgentConfig::Mcts(second)) => play_match(
            &game,
            &mut configured_spirits_of_the_forest_mcts(first)?,
            &mut configured_spirits_of_the_forest_mcts(second)?,
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
            run_boop_match_with_trace(&mut RandomAgent, &mut configured_boop_mcts(second)?, config)
        }
        (AgentConfig::Mcts(first), AgentConfig::Random) => {
            run_boop_match_with_trace(&mut configured_boop_mcts(first)?, &mut RandomAgent, config)
        }
        (AgentConfig::Mcts(first), AgentConfig::Mcts(second)) => run_boop_match_with_trace(
            &mut configured_boop_mcts(first)?,
            &mut configured_boop_mcts(second)?,
            config,
        ),
    }
}

fn run_spirits_of_the_forest_with_trace(
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<CatalogMatchReport, CatalogError> {
    match (first, second) {
        (AgentConfig::Random, AgentConfig::Random) => {
            run_spirits_of_the_forest_match_with_trace(&mut RandomAgent, &mut RandomAgent, config)
        }
        (AgentConfig::Random, AgentConfig::Mcts(second)) => {
            run_spirits_of_the_forest_match_with_trace(
                &mut RandomAgent,
                &mut configured_spirits_of_the_forest_mcts(second)?,
                config,
            )
        }
        (AgentConfig::Mcts(first), AgentConfig::Random) => {
            run_spirits_of_the_forest_match_with_trace(
                &mut configured_spirits_of_the_forest_mcts(first)?,
                &mut RandomAgent,
                config,
            )
        }
        (AgentConfig::Mcts(first), AgentConfig::Mcts(second)) => {
            run_spirits_of_the_forest_match_with_trace(
                &mut configured_spirits_of_the_forest_mcts(first)?,
                &mut configured_spirits_of_the_forest_mcts(second)?,
                config,
            )
        }
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
        (AgentConfig::Random, AgentConfig::Mcts(second)) => run_connect_four_match_with_trace(
            &mut RandomAgent,
            &mut configured_connect_four_mcts(second)?,
            config,
        ),
        (AgentConfig::Mcts(first), AgentConfig::Random) => run_connect_four_match_with_trace(
            &mut configured_connect_four_mcts(first)?,
            &mut RandomAgent,
            config,
        ),
        (AgentConfig::Mcts(first), AgentConfig::Mcts(second)) => run_connect_four_match_with_trace(
            &mut configured_connect_four_mcts(first)?,
            &mut configured_connect_four_mcts(second)?,
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
        (AgentConfig::Random, AgentConfig::Mcts(second)) => play_match(
            &game,
            &mut RandomAgent,
            &mut configured_tic_tac_toe_mcts(second)?,
            config,
        ),
        (AgentConfig::Mcts(first), AgentConfig::Random) => play_match(
            &game,
            &mut configured_tic_tac_toe_mcts(first)?,
            &mut RandomAgent,
            config,
        ),
        (AgentConfig::Mcts(first), AgentConfig::Mcts(second)) => play_match(
            &game,
            &mut configured_tic_tac_toe_mcts(first)?,
            &mut configured_tic_tac_toe_mcts(second)?,
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
        (AgentConfig::Random, AgentConfig::Mcts(second)) => run_tic_tac_toe_match_with_trace(
            &mut RandomAgent,
            &mut configured_tic_tac_toe_mcts(second)?,
            config,
        ),
        (AgentConfig::Mcts(first), AgentConfig::Random) => run_tic_tac_toe_match_with_trace(
            &mut configured_tic_tac_toe_mcts(first)?,
            &mut RandomAgent,
            config,
        ),
        (AgentConfig::Mcts(first), AgentConfig::Mcts(second)) => run_tic_tac_toe_match_with_trace(
            &mut configured_tic_tac_toe_mcts(first)?,
            &mut configured_tic_tac_toe_mcts(second)?,
            config,
        ),
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
            search_iterations: traced_action.decision_stats.search_iterations,
            search_nodes: traced_action.decision_stats.search_nodes,
            root_actions: traced_action.decision_stats.root_actions,
            tree_reuse: traced_action.decision_stats.tree_reuse,
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
            search_iterations: traced_action.decision_stats.search_iterations,
            search_nodes: traced_action.decision_stats.search_nodes,
            root_actions: traced_action.decision_stats.root_actions,
            tree_reuse: traced_action.decision_stats.tree_reuse,
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
            search_iterations: traced_action.decision_stats.search_iterations,
            search_nodes: traced_action.decision_stats.search_nodes,
            root_actions: traced_action.decision_stats.root_actions,
            tree_reuse: traced_action.decision_stats.tree_reuse,
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
            search_iterations: traced_action.decision_stats.search_iterations,
            search_nodes: traced_action.decision_stats.search_nodes,
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
    let results = match (first, second) {
        (AgentConfig::Random, AgentConfig::Random) => {
            play_batch(&Boop, config, || RandomAgent, || RandomAgent)
        }
        (AgentConfig::Random, AgentConfig::Mcts(second)) => {
            let second = configured_boop_mcts(second)?;
            play_batch(&Boop, config, || RandomAgent, || second.clone())
        }
        (AgentConfig::Mcts(first), AgentConfig::Random) => {
            let first = configured_boop_mcts(first)?;
            play_batch(&Boop, config, || first.clone(), || RandomAgent)
        }
        (AgentConfig::Mcts(first), AgentConfig::Mcts(second)) => {
            let first = configured_boop_mcts(first)?;
            let second = configured_boop_mcts(second)?;
            play_batch(&Boop, config, || first.clone(), || second.clone())
        }
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
        (AgentConfig::Random, AgentConfig::Mcts(second)) => {
            let second = configured_connect_four_mcts(second)?;
            play_batch(&ConnectFour, config, || RandomAgent, || second.clone())
        }
        (AgentConfig::Mcts(first), AgentConfig::Random) => {
            let first = configured_connect_four_mcts(first)?;
            play_batch(&ConnectFour, config, || first.clone(), || RandomAgent)
        }
        (AgentConfig::Mcts(first), AgentConfig::Mcts(second)) => {
            let first = configured_connect_four_mcts(first)?;
            let second = configured_connect_four_mcts(second)?;
            play_batch(&ConnectFour, config, || first.clone(), || second.clone())
        }
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
        (AgentConfig::Random, AgentConfig::Mcts(second)) => {
            let second = configured_tic_tac_toe_mcts(second)?;
            play_batch(&TicTacToe, config, || RandomAgent, || second.clone())
        }
        (AgentConfig::Mcts(first), AgentConfig::Random) => {
            let first = configured_tic_tac_toe_mcts(first)?;
            play_batch(&TicTacToe, config, || first.clone(), || RandomAgent)
        }
        (AgentConfig::Mcts(first), AgentConfig::Mcts(second)) => {
            let first = configured_tic_tac_toe_mcts(first)?;
            let second = configured_tic_tac_toe_mcts(second)?;
            play_batch(&TicTacToe, config, || first.clone(), || second.clone())
        }
    }?;
    Ok(results)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;

    fn mcts(heuristic: Option<u32>) -> AgentConfig {
        AgentConfig::Mcts(MctsAgentConfig {
            search: MctsConfig {
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
        })
    }

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
                CatalogAction::ConnectFour { .. } => panic!("unexpected Connect Four action"),
                CatalogAction::Boop { .. } => panic!("unexpected boop action"),
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
                CatalogAction::ConnectFour { column } => assert!(column < 7),
                CatalogAction::TicTacToe { .. } => panic!("unexpected tic-tac-toe action"),
                CatalogAction::Boop { .. } => panic!("unexpected boop action"),
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
    fn trace_analysis_dispatches_to_supported_games_and_rejects_unimplemented_games() {
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

        let unavailable = analyze_trace(GameId::ConnectFour, &[]).unwrap_err();
        assert_eq!(
            unavailable.to_string(),
            "tournament analysis is not available for connect-four"
        );
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
