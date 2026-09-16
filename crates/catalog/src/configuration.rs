//! Configured MCTS policies, game-specific construction and validation.

use super::{
    CatalogError, CatalogTurnPhase, GameId, spirits_of_the_forest_game,
    supports_turn_phase_conditions,
};
use meeple_bots_boop::Boop;
use meeple_bots_connect_four::ConnectFour;
use meeple_bots_connect6::Connect6;
use meeple_bots_core::{
    AgentError, Game, HeuristicGame, HeuristicParameters, PlayerId, RandomSource,
};
use meeple_bots_mcts_agent::{
    MctsAgent, MctsConfig, PolicyCondition, RolloutMemory, RolloutPolicy, RolloutPolicyConfig,
    SearchBudget, SelectionBias, StateEvaluator, TranspositionMctsAgent, UniformRandom,
};
use meeple_bots_spirits_of_the_forest::{SpiritsOfTheForest, SpiritsOfTheForestAction, TurnPhase};
use meeple_bots_tic_tac_toe::TicTacToe;

// Configuration is constructed outside search; keep the existing value-based public API.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum AgentConfig {
    SoIsmcts(meeple_bots_so_ismcts::SoIsmctsConfig),
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
    pub transpositions: bool,
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

pub type BoopMctsAgent = TranspositionMctsAgent<
    Boop,
    EvaluatorConfig,
    RolloutPolicyConfig<EvaluatorConfig>,
    ConfiguredSelectionBias,
>;
pub type SpiritsOfTheForestMctsAgent = TranspositionMctsAgent<
    SpiritsOfTheForest,
    EvaluatorConfig,
    ConfiguredRolloutPolicy,
    ConfiguredSelectionBias,
>;
pub type ConnectFourMctsAgent = TranspositionMctsAgent<
    ConnectFour,
    meeple_bots_mcts_agent::NeutralEvaluator,
    RolloutPolicyConfig,
>;
pub type Connect6MctsAgent =
    TranspositionMctsAgent<Connect6, meeple_bots_mcts_agent::NeutralEvaluator, RolloutPolicyConfig>;
pub type TicTacToeMctsAgent =
    TranspositionMctsAgent<TicTacToe, meeple_bots_mcts_agent::NeutralEvaluator, UniformRandom>;

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
    fn validate(&self) -> Result<(), AgentError> {
        match self {
            Self::Standard(policy) => <RolloutPolicyConfig<EvaluatorConfig> as RolloutPolicy<
                SpiritsOfTheForest,
            >>::validate(policy),
            Self::Conditional {
                primary, fallback, ..
            } => {
                <RolloutPolicyConfig<EvaluatorConfig> as RolloutPolicy<SpiritsOfTheForest>>::validate(primary)?;
                <RolloutPolicyConfig<EvaluatorConfig> as RolloutPolicy<SpiritsOfTheForest>>::validate(fallback)
            }
        }
    }

    fn action_for_learning(
        &self,
        action: &SpiritsOfTheForestAction,
    ) -> Option<SpiritsOfTheForestAction> {
        let uses_mast = match self {
            Self::Standard(policy) => matches!(policy, RolloutPolicyConfig::Mast { .. }),
            Self::Conditional {
                primary, fallback, ..
            } => {
                matches!(primary, RolloutPolicyConfig::Mast { .. })
                    || matches!(fallback, RolloutPolicyConfig::Mast { .. })
            }
        };
        uses_mast.then_some(*action)
    }

    fn finish_simulation(
        &self,
        memory: &mut RolloutMemory<SpiritsOfTheForestAction>,
        root_player: PlayerId,
        utility: f64,
    ) {
        let policy = match self {
            Self::Standard(policy) => policy,
            Self::Conditional { primary, .. } => primary,
        };
        <RolloutPolicyConfig<EvaluatorConfig> as RolloutPolicy<SpiritsOfTheForest>>::finish_simulation(policy, memory, root_player, utility);
    }

    fn select_action_with_memory<R: RandomSource + ?Sized>(
        &self,
        game: &SpiritsOfTheForest,
        state: &<SpiritsOfTheForest as Game>::State,
        active_player: PlayerId,
        root_player: PlayerId,
        memory: &RolloutMemory<SpiritsOfTheForestAction>,
        rng: &mut R,
    ) -> Result<SpiritsOfTheForestAction, AgentError> {
        let policy = match self {
            Self::Standard(policy) => policy,
            Self::Conditional {
                condition: RolloutConditionConfig::TurnPhase(phase),
                primary,
                fallback,
            } => {
                if SpiritsTurnPhaseCondition(*phase).matches(
                    game,
                    state,
                    active_player,
                    root_player,
                ) {
                    primary
                } else {
                    fallback
                }
            }
        };
        policy.select_action_with_memory(game, state, active_player, root_player, memory, rng)
    }

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
    G: HeuristicGame,
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
    Ok(TranspositionMctsAgent::new(
        MctsAgent::with_progressive_bias(
            standard_search_config(config.search)?,
            config.cutoff_evaluator,
            config.progressive_bias,
        )
        .with_root_diagnostics(config.root_diagnostics),
        config.tree_reuse,
        config.transpositions,
    ))
}

pub fn configured_spirits_of_the_forest_mcts(
    config: MctsAgentConfig,
) -> Result<SpiritsOfTheForestMctsAgent, CatalogError> {
    let game = spirits_of_the_forest_game(0);
    validate_agent_evaluators(GameId::SpiritsOfTheForest, &game, &config)?;
    Ok(TranspositionMctsAgent::new(
        MctsAgent::with_progressive_bias(
            config.search,
            config.cutoff_evaluator,
            config.progressive_bias,
        )
        .with_root_diagnostics(config.root_diagnostics),
        config.tree_reuse,
        config.transpositions,
    ))
}

pub fn configured_connect_four_mcts(
    config: MctsAgentConfig,
) -> Result<ConnectFourMctsAgent, CatalogError> {
    validate_uninformed_agent(GameId::ConnectFour, &config)?;
    let rollout_policy = match config.search.rollout_policy {
        ConfiguredRolloutPolicy::Standard(RolloutPolicyConfig::UniformRandom) => {
            RolloutPolicyConfig::UniformRandom
        }
        ConfiguredRolloutPolicy::Standard(RolloutPolicyConfig::Mast { epsilon }) => {
            RolloutPolicyConfig::Mast { epsilon }
        }
        _ => unreachable!("Connect Four rollout policy was validated"),
    };
    Ok(TranspositionMctsAgent::new(
        MctsAgent::new(MctsConfig {
            progressive_widening: config.search.progressive_widening,
            selection_policy: config.search.selection_policy,
            budget: config.search.budget,
            exploration: config.search.exploration,
            rollout_depth: config.search.rollout_depth,
            rollout_policy,
        })
        .with_root_diagnostics(config.root_diagnostics),
        config.tree_reuse,
        config.transpositions,
    ))
}

pub fn configured_connect6_mcts(
    config: MctsAgentConfig,
) -> Result<Connect6MctsAgent, CatalogError> {
    validate_uninformed_agent(
        GameId::Connect6(meeple_bots_connect6::DEFAULT_BOARD_SIZE),
        &config,
    )?;
    let rollout_policy = match config.search.rollout_policy {
        ConfiguredRolloutPolicy::Standard(RolloutPolicyConfig::UniformRandom) => {
            RolloutPolicyConfig::UniformRandom
        }
        ConfiguredRolloutPolicy::Standard(RolloutPolicyConfig::Mast { epsilon }) => {
            RolloutPolicyConfig::Mast { epsilon }
        }
        _ => unreachable!("Connect6 rollout policy was validated"),
    };
    Ok(TranspositionMctsAgent::new(
        MctsAgent::new(MctsConfig {
            progressive_widening: config.search.progressive_widening,
            selection_policy: config.search.selection_policy,
            budget: config.search.budget,
            exploration: config.search.exploration,
            rollout_depth: config.search.rollout_depth,
            rollout_policy,
        })
        .with_root_diagnostics(config.root_diagnostics),
        config.tree_reuse,
        config.transpositions,
    ))
}

pub fn configured_tic_tac_toe_mcts(
    config: MctsAgentConfig,
) -> Result<TicTacToeMctsAgent, CatalogError> {
    validate_uninformed_agent(GameId::TicTacToe, &config)?;
    Ok(TranspositionMctsAgent::new(
        MctsAgent::new(uniform_search_config(config.search))
            .with_root_diagnostics(config.root_diagnostics),
        config.tree_reuse,
        config.transpositions,
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
        (
            _,
            RolloutConditionConfig::TurnPhase(
                CatalogTurnPhase::Choose | CatalogTurnPhase::Continue,
            ),
        ) => Err(CatalogError::InvalidMctsConfig(
            "choose and continue phases are only supported by Can't Stop",
        )),
        (
            game,
            RolloutConditionConfig::TurnPhase(
                CatalogTurnPhase::Collect | CatalogTurnPhase::PlaceGemstone,
            ),
        ) if supports_turn_phase_conditions(game) => Ok(()),
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
        RolloutPolicyConfig::Mast { epsilon } => {
            if !epsilon.is_finite() || !(0.0..=1.0).contains(epsilon) {
                return Err(CatalogError::InvalidMctsConfig(
                    "MCTS rollout epsilon must be finite and between 0.0 and 1.0",
                ));
            }
            Ok(())
        }
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
        (
            _,
            RolloutConditionConfig::TurnPhase(
                CatalogTurnPhase::Choose | CatalogTurnPhase::Continue,
            ),
        ) => Err(CatalogError::InvalidMctsConfig(
            "choose and continue phases are only supported by Can't Stop",
        )),
        (
            game,
            RolloutConditionConfig::TurnPhase(
                CatalogTurnPhase::Collect | CatalogTurnPhase::PlaceGemstone,
            ),
        ) if supports_turn_phase_conditions(game) => Ok(()),
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
    let connect_four_mast = if let (
        GameId::ConnectFour,
        ConfiguredRolloutPolicy::Standard(RolloutPolicyConfig::Mast { epsilon }),
    ) = (game, &config.search.rollout_policy)
    {
        if !epsilon.is_finite() || !(0.0..=1.0).contains(epsilon) {
            return Err(CatalogError::InvalidMctsConfig(
                "MCTS rollout epsilon must be finite and between 0.0 and 1.0",
            ));
        }
        true
    } else {
        false
    };
    if !connect_four_mast
        && !matches!(
            config.search.rollout_policy,
            ConfiguredRolloutPolicy::Standard(RolloutPolicyConfig::UniformRandom)
        )
    {
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
        progressive_widening: config.progressive_widening,
        selection_policy: config.selection_policy,
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
        progressive_widening: config.progressive_widening,
        selection_policy: config.selection_policy,
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
