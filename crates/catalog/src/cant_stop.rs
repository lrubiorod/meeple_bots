//! Typed public-chance integration. The session advances one decision or dice event at a time.
use crate::{
    AgentConfig, ConfiguredAgent, ConfiguredRolloutPolicy, ConfiguredSelectionBias,
    EvaluatorConfig, MctsAgentConfig, RolloutPolicyConfig,
};
use meeple_bots_cant_stop::{CantStop, CantStopAction, CantStopState};
use meeple_bots_core::{
    Agent, AgentDecisionStats, AgentError, DecisionContext, Game, PlayerId, PositionStatus,
};
use meeple_bots_mcts_agent::{MctsConfig, StochasticMctsAgent, UniformRandom};
use meeple_bots_simulation::SplitMix64;
use std::time::{Duration, Instant};

pub type CantStopAgent = ConfiguredAgent<StochasticMctsAgent<EvaluatorConfig>>;

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
            if tree_reuse
                || transpositions
                || progressive_bias != ConfiguredSelectionBias::None
                || search.rollout_policy
                    != ConfiguredRolloutPolicy::Standard(RolloutPolicyConfig::UniformRandom)
            {
                return Err(AgentError::message(
                    "Can't Stop currently supports uniform rollouts without tree reuse, transpositions or progressive bias",
                ));
            }
            match &cutoff_evaluator {
                EvaluatorConfig::Neutral => (),
                EvaluatorConfig::GameHeuristic {
                    index: 0,
                    parameters,
                } if parameters.is_empty() => (),
                _ => {
                    return Err(AgentError::message(
                        "Can't Stop supports neutral evaluation or heuristic 0 without parameters",
                    ));
                }
            }
            search.validate().map_err(AgentError::message)?;
            let mut agent = StochasticMctsAgent::new(
                MctsConfig {
                    budget: search.budget,
                    exploration: search.exploration,
                    selection_policy: search.selection_policy,
                    rollout_depth: search.rollout_depth,
                    rollout_policy: UniformRandom,
                },
                cutoff_evaluator,
            );
            agent.root_diagnostics = root_diagnostics;
            Ok(ConfiguredAgent::Mcts(agent))
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
