//! Typed public-chance integration. The session advances one decision or dice event at a time.
use crate::{
    AgentConfig, CatalogTurnPhase, ConfiguredAgent, ConfiguredRolloutPolicy,
    ConfiguredSelectionBias, EvaluatorConfig, MctsAgentConfig, RolloutConditionConfig,
    RolloutPolicyConfig,
};
use meeple_bots_cant_stop::{CantStop, CantStopAction, CantStopState, Phase};
use meeple_bots_core::RandomSource;
use meeple_bots_core::{
    Agent, AgentDecisionStats, AgentError, DecisionContext, Game, PlayerId, PositionStatus,
};
use meeple_bots_mcts_agent::{
    PolicyCondition, ReusableStochasticMctsAgent, RolloutMemory, RolloutPolicy, SelectionBias,
    StateEvaluator, StochasticMctsAgent,
};
use meeple_bots_simulation::SplitMix64;
use std::time::{Duration, Instant};

pub type CantStopAgent = ConfiguredAgent<
    ReusableStochasticMctsAgent<
        CantStop,
        EvaluatorConfig,
        ConfiguredRolloutPolicy,
        ConfiguredSelectionBias,
    >,
>;

pub fn configured_cant_stop_agent(config: AgentConfig) -> Result<CantStopAgent, AgentError> {
    match config {
        AgentConfig::Random => Ok(ConfiguredAgent::Random(
            meeple_bots_random_agent::RandomAgent,
        )),
        AgentConfig::Mcts(MctsAgentConfig {
            search,
            cutoff_evaluator,
            progressive_bias,
            tree_reuse,
            transpositions,
            root_diagnostics,
        }) => {
            validate_evaluator(&cutoff_evaluator)?;
            validate_rollout(&search.rollout_policy)?;
            search.validate().map_err(AgentError::message)?;
            if let ConfiguredSelectionBias::Progressive {
                weight,
                evaluator,
                condition,
            } = &progressive_bias
            {
                if !weight.is_finite() || *weight < 0.0 {
                    return Err(AgentError::message(
                        "progressive bias weight must be finite and non-negative",
                    ));
                }
                validate_evaluator(evaluator)?;
                if let Some(condition) = condition {
                    validate_condition(condition)?;
                }
            }
            let mut agent = StochasticMctsAgent::with_progressive_bias(
                search,
                cutoff_evaluator,
                progressive_bias,
            );
            agent.root_diagnostics = root_diagnostics;
            Ok(ConfiguredAgent::Mcts(ReusableStochasticMctsAgent::new(
                agent,
                tree_reuse,
                transpositions,
            )))
        }
    }
}

#[derive(Clone, Debug)]
pub struct CantStopEvent {
    pub player: Option<PlayerId>,
    pub action: CantStopAction,
    pub seconds: Duration,
    pub stats: AgentDecisionStats,
    pub bust: bool,
}

pub struct CantStopSession {
    pub state: CantStopState,
    pub events: Vec<CantStopEvent>,
    agents: [Option<CantStopAgent>; 2],
    chance_rng: SplitMix64,
    agent_rngs: [SplitMix64; 2],
}
impl CantStopSession {
    pub fn new(
        seed: u64,
        first: Option<AgentConfig>,
        second: Option<AgentConfig>,
    ) -> Result<Self, AgentError> {
        let mut agents = [
            first.map(configured_cant_stop_agent).transpose()?,
            second.map(configured_cant_stop_agent).transpose()?,
        ];
        let state = CantStop.initial_state();
        for (i, a) in agents.iter_mut().enumerate() {
            if let Some(a) = a {
                a.on_match_start(
                    &CantStop,
                    &state,
                    if i == 0 {
                        PlayerId::FIRST
                    } else {
                        PlayerId::SECOND
                    },
                );
            }
        }
        Ok(Self {
            state,
            events: Vec::new(),
            agents,
            chance_rng: SplitMix64::new(seed ^ 0x8EBC_6AF0_9C88_C6E3),
            agent_rngs: [
                SplitMix64::new(seed ^ 0xA076_1D64_78BD_642F),
                SplitMix64::new(seed ^ 0xE703_7ED1_A0B4_28DB),
            ],
        })
    }
    pub fn waiting_human(&self) -> bool {
        matches!(CantStop.status(&self.state),PositionStatus::PlayerTurn(p) if self.agents[p.index()].is_none())
    }
    pub fn step(&mut self, chosen: Option<usize>) -> Result<(), AgentError> {
        if self.events.len() >= 20_000 {
            return Err(AgentError::message(
                "Can't Stop session exceeded 20000 events",
            ));
        }
        let (player, action, seconds, stats) = match CantStop.status(&self.state) {
            PositionStatus::Chance => {
                if chosen.is_some() {
                    return Err(AgentError::message("players cannot choose dice outcomes"));
                }
                (
                    None,
                    CantStop
                        .sample_chance(&self.state, &mut self.chance_rng)
                        .map_err(|e| AgentError::message(e.to_string()))?,
                    Duration::ZERO,
                    AgentDecisionStats::default(),
                )
            }
            PositionStatus::PlayerTurn(p) => {
                if let Some(agent) = &mut self.agents[p.index()] {
                    if chosen.is_some() {
                        return Err(AgentError::message("this seat is automated"));
                    }
                    let start = Instant::now();
                    let a = agent.select_action(
                        DecisionContext::new(&CantStop, &self.state, p),
                        &mut self.agent_rngs[p.index()],
                    )?;
                    let stats = <CantStopAgent as Agent<CantStop>>::last_decision_stats(agent);
                    (Some(p), a, start.elapsed(), stats)
                } else {
                    let a = chosen
                        .and_then(|i| CantStop.legal_actions(&self.state).nth(i))
                        .ok_or_else(|| AgentError::message("select a legal action index"))?;
                    (Some(p), a, Duration::ZERO, AgentDecisionStats::default())
                }
            }
            _ => return Err(AgentError::message("the game has finished")),
        };
        CantStop
            .apply_action(&mut self.state, &action)
            .map_err(|e| AgentError::message(e.to_string()))?;
        for a in self.agents.iter_mut().flatten() {
            if let Some(p) = player {
                a.on_action_applied(&CantStop, &self.state, p, &action);
            } else {
                a.on_chance_applied(&CantStop, &self.state, &action);
            }
            if CantStop.status(&self.state) == PositionStatus::Terminal {
                a.on_match_end(&CantStop, &self.state);
            }
        }
        self.events.push(CantStopEvent {
            player,
            action,
            seconds,
            stats,
            bust: self.state.last_bust,
        });
        Ok(())
    }
}

struct CantStopPhaseCondition(CatalogTurnPhase);
impl PolicyCondition<CantStop> for CantStopPhaseCondition {
    fn matches(&self, _: &CantStop, state: &CantStopState, _: PlayerId, _: PlayerId) -> bool {
        matches!(
            (self.0, state.phase),
            (CatalogTurnPhase::Choose, Phase::Choose)
                | (CatalogTurnPhase::Continue, Phase::Continue)
        )
    }
}
fn validate_evaluator(evaluator: &EvaluatorConfig) -> Result<(), AgentError> {
    match evaluator {
        EvaluatorConfig::Neutral => Ok(()),
        EvaluatorConfig::GameHeuristic {
            index: 0,
            parameters,
        } if parameters.is_empty() => Ok(()),
        _ => Err(AgentError::message(
            "Can't Stop supports neutral evaluation or heuristic 0 without parameters",
        )),
    }
}
fn validate_condition(condition: &RolloutConditionConfig) -> Result<(), AgentError> {
    match condition {
        RolloutConditionConfig::TurnPhase(
            CatalogTurnPhase::Choose | CatalogTurnPhase::Continue,
        ) => Ok(()),
        _ => Err(AgentError::message(
            "Can't Stop turn phase must be choose or continue",
        )),
    }
}
fn validate_rollout(policy: &ConfiguredRolloutPolicy) -> Result<(), AgentError> {
    <ConfiguredRolloutPolicy as RolloutPolicy<CantStop>>::validate(policy)?;
    let validate = |p: &RolloutPolicyConfig<EvaluatorConfig>| match p {
        RolloutPolicyConfig::UniformRandom => Ok(()),
        RolloutPolicyConfig::Mast { .. } => Ok(()),
        RolloutPolicyConfig::Greedy { evaluator }
        | RolloutPolicyConfig::EpsilonGreedy { evaluator, .. } => validate_evaluator(evaluator),
    };
    match policy {
        ConfiguredRolloutPolicy::Standard(p) => validate(p),
        ConfiguredRolloutPolicy::Conditional {
            condition,
            primary,
            fallback,
        } => {
            validate_condition(condition)?;
            validate(primary)?;
            validate(fallback)
        }
    }
}
impl RolloutPolicy<CantStop> for ConfiguredRolloutPolicy {
    fn validate(&self) -> Result<(), AgentError> {
        match self {
            Self::Standard(policy) => {
                <RolloutPolicyConfig<EvaluatorConfig> as RolloutPolicy<CantStop>>::validate(policy)
            }
            Self::Conditional {
                primary, fallback, ..
            } => {
                <RolloutPolicyConfig<EvaluatorConfig> as RolloutPolicy<CantStop>>::validate(
                    primary,
                )?;
                <RolloutPolicyConfig<EvaluatorConfig> as RolloutPolicy<CantStop>>::validate(
                    fallback,
                )
            }
        }
    }

    fn action_for_learning(&self, action: &CantStopAction) -> Option<CantStopAction> {
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
        memory: &mut RolloutMemory<CantStopAction>,
        root_player: PlayerId,
        utility: f64,
    ) {
        let policy = match self {
            Self::Standard(policy) => policy,
            Self::Conditional { primary, .. } => primary,
        };
        <RolloutPolicyConfig<EvaluatorConfig> as RolloutPolicy<CantStop>>::finish_simulation(
            policy,
            memory,
            root_player,
            utility,
        );
    }

    fn select_action_with_memory<R: RandomSource + ?Sized>(
        &self,
        game: &CantStop,
        state: &<CantStop as Game>::State,
        active_player: PlayerId,
        root_player: PlayerId,
        memory: &RolloutMemory<CantStopAction>,
        rng: &mut R,
    ) -> Result<CantStopAction, AgentError> {
        let policy = match self {
            Self::Standard(policy) => policy,
            Self::Conditional {
                condition: RolloutConditionConfig::TurnPhase(phase),
                primary,
                fallback,
            } => {
                if CantStopPhaseCondition(*phase).matches(game, state, active_player, root_player) {
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
        game: &CantStop,
        state: &<CantStop as Game>::State,
        active_player: PlayerId,
        root_player: PlayerId,
        rng: &mut R,
    ) -> Result<CantStopAction, AgentError> {
        match self {
            Self::Standard(policy) => {
                policy.select_action(game, state, active_player, root_player, rng)
            }
            Self::Conditional {
                condition: RolloutConditionConfig::TurnPhase(phase),
                primary,
                fallback,
            } => {
                let policy = if CantStopPhaseCondition(*phase).matches(
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

impl SelectionBias<CantStop> for ConfiguredSelectionBias {
    fn weight(&self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Progressive { weight, .. } => *weight,
        }
    }
    fn applies(
        &self,
        game: &CantStop,
        state: &CantStopState,
        active: PlayerId,
        root: PlayerId,
    ) -> bool {
        match self {
            Self::None => false,
            Self::Progressive {
                condition: None, ..
            } => true,
            Self::Progressive {
                condition: Some(RolloutConditionConfig::TurnPhase(phase)),
                ..
            } => CantStopPhaseCondition(*phase).matches(game, state, active, root),
        }
    }
    fn evaluate_child(
        &self,
        game: &CantStop,
        state: &CantStopState,
        root: PlayerId,
    ) -> Result<f64, AgentError> {
        match self {
            Self::None => Ok(0.0),
            Self::Progressive { evaluator, .. } => evaluator.evaluate(game, state, root),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn automated_session_finishes_and_preserves_public_events() {
        let mut s =
            CantStopSession::new(42, Some(AgentConfig::Random), Some(AgentConfig::Random)).unwrap();
        while s.state.winner.is_none() {
            s.step(None).unwrap();
        }
        assert!(
            s.events
                .iter()
                .any(|e| matches!(e.action, CantStopAction::Dice(_)))
        );
        assert!(
            s.events
                .iter()
                .all(|e| e.player.is_none() == matches!(e.action, CantStopAction::Dice(_)))
        );
    }
    #[test]
    fn human_cannot_pick_dice_and_bad_indices_leave_state_unchanged() {
        let mut s = CantStopSession::new(1, None, None).unwrap();
        let before = s.state.clone();
        assert!(s.step(Some(0)).is_err());
        assert_eq!(s.state, before);
        s.step(None).unwrap();
        assert!(s.waiting_human());
        let before = s.state.clone();
        assert!(s.step(Some(100)).is_err());
        assert_eq!(s.state, before);
    }
}
