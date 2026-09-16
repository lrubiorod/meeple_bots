//! Registration of public-chance Splendor with configurable cutoff evaluation and uniform stochastic MCTS.
use crate::{
    AgentConfig, ConfiguredAgent, ConfiguredRolloutPolicy, ConfiguredSelectionBias, EvaluatorConfig,
};
use crate::{CatalogAction, CatalogError, CatalogMatchReport, RecordedMove};
use meeple_bots_core::{
    Agent, AgentDecisionStats, AgentError, DecisionContext, Game, PlayerId, PositionStatus,
};
use meeple_bots_mcts_agent::{
    MctsConfig, ReusableStochasticMctsAgent, RolloutPolicyConfig, StochasticMctsAgent,
    UniformRandom,
};
use meeple_bots_simulation::{
    MatchConfig, MatchError, SplitMix64, TracedMatchResult, play_match_with_trace,
};
use meeple_bots_splendor::SplendorState;
use meeple_bots_splendor::{Splendor, SplendorAction};
use std::time::{Duration, Instant};
pub type SplendorAgent = ConfiguredAgent<ReusableStochasticMctsAgent<Splendor, EvaluatorConfig>>;
pub fn game(seed: u64) -> Splendor {
    Splendor::new(&mut SplitMix64::new(seed ^ 0xD1B5_4A32_D192_ED03))
}
pub fn configured_agent(config: AgentConfig) -> Result<SplendorAgent, AgentError> {
    match config {
        AgentConfig::SoIsmcts(_) => Err(AgentError::message(
            "SO-ISMCTS is only supported by Lost Cities",
        )),
        AgentConfig::Random => Ok(ConfiguredAgent::Random(
            meeple_bots_random_agent::RandomAgent,
        )),
        AgentConfig::Mcts(c) => {
            if c.search.progressive_widening.is_some() {
                return Err(AgentError::message(
                    "Progressive Widening is only supported by deterministic MCTS",
                ));
            }
            if matches!(
                c.search.selection_policy,
                meeple_bots_mcts_agent::SelectionPolicy::UctRave { .. }
            ) {
                return Err(AgentError::message(
                    "UCT-RAVE is only supported by deterministic MCTS",
                ));
            }

            let valid_cutoff = match &c.cutoff_evaluator {
                EvaluatorConfig::Neutral => true,
                EvaluatorConfig::GameHeuristic { index, parameters } => {
                    matches!(*index, 0 | 1) && parameters.is_empty()
                }
            };
            if !valid_cutoff
                || c.progressive_bias != ConfiguredSelectionBias::None
                || c.search.rollout_policy
                    != ConfiguredRolloutPolicy::Standard(RolloutPolicyConfig::UniformRandom)
            {
                return Err(AgentError::message(
                    "Splendor supports neutral or heuristics 0/1 (no parameters), uniform rollouts and no progressive bias",
                ));
            }
            let config = MctsConfig {
                progressive_widening: c.search.progressive_widening,
                budget: c.search.budget,
                exploration: c.search.exploration,
                rollout_depth: c.search.rollout_depth,
                selection_policy: c.search.selection_policy,
                rollout_policy: UniformRandom,
            };
            config.validate().map_err(AgentError::message)?;
            let mut inner = StochasticMctsAgent::new(config, c.cutoff_evaluator);
            inner.root_diagnostics = c.root_diagnostics;
            Ok(ConfiguredAgent::Mcts(ReusableStochasticMctsAgent::new(
                inner,
                c.tree_reuse,
                c.transpositions,
            )))
        }
    }
}
pub fn run(
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<TracedMatchResult<SplendorAction>, CatalogError> {
    let mut first = configured_agent(first).map_err(|source| MatchError::Agent {
        player: PlayerId::FIRST,
        source,
    })?;
    let mut second = configured_agent(second).map_err(|source| MatchError::Agent {
        player: PlayerId::SECOND,
        source,
    })?;
    play_match_with_trace(&game(config.seed), &mut first, &mut second, config).map_err(Into::into)
}

/// Replay all player decisions and explicit chance events, checking order and terminality.
pub fn replay(
    seed: u64,
    moves: &[RecordedMove],
    events: &[meeple_bots_simulation::TracedChance<CatalogAction>],
) -> Result<meeple_bots_splendor::SplendorState, CatalogError> {
    let invalid = |message: String| CatalogError::InvalidTrace {
        game: crate::GameId::Splendor,
        message,
    };
    let g = game(seed);
    let mut state = g.initial_state();
    let mut pending = events.iter().peekable();
    for (i, m) in moves.iter().enumerate() {
        if g.status(&state)
            != PositionStatus::PlayerTurn(if m.player == 0 {
                PlayerId::FIRST
            } else {
                PlayerId::SECOND
            })
            || m.player > 1
        {
            return Err(invalid("incorrect player or transition order".into()));
        }
        let CatalogAction::Splendor(action) = m.action else {
            return Err(invalid("expected Splendor action".into()));
        };
        g.apply_action(&mut state, &action)
            .map_err(|e| invalid(e.to_string()))?;
        while pending.peek().is_some_and(|e| e.after_ply == i + 1) {
            let CatalogAction::Splendor(outcome) = pending.next().unwrap().event else {
                return Err(invalid("expected Splendor outcome".into()));
            };
            g.apply_chance_outcome(&mut state, &outcome)
                .map_err(|e| invalid(e.to_string()))?;
        }
    }
    if pending.next().is_some() || g.status(&state) != PositionStatus::Terminal {
        return Err(invalid("missing or extra transitions".into()));
    }
    Ok(state)
}
pub fn report(
    traced: TracedMatchResult<SplendorAction>,
) -> Result<CatalogMatchReport, CatalogError> {
    let moves: Vec<_> = traced
        .actions
        .into_iter()
        .map(|a| RecordedMove {
            player: a.player.index(),
            action: CatalogAction::Splendor(a.action),
            decision_seconds: a.decision_time.as_secs_f64(),
            selection_seconds: a.selection_time.as_secs_f64(),
            maintenance_seconds: a.maintenance_time.as_secs_f64(),
            search_iterations: a.decision_stats.search_iterations,
            search_nodes: a.decision_stats.search_nodes,
            terminal_simulations: a.decision_stats.terminal_simulations,
            cutoff_simulations: a.decision_stats.cutoff_simulations,
            root_actions: a.decision_stats.root_actions,
            tree_reuse: a.decision_stats.tree_reuse,
        })
        .collect();
    let chance_events: Vec<_> = traced
        .chance_events
        .into_iter()
        .map(|e| meeple_bots_simulation::TracedChance {
            after_ply: e.after_ply,
            event: CatalogAction::Splendor(e.event),
        })
        .collect();
    let state = replay(traced.result.seed, &moves, &chance_events)?;
    let winner = traced.result.utilities.iter().position(|&v| v > 0.0);
    let scores = state.players.each_ref().map(|p| i16::from(p.prestige));
    Ok(CatalogMatchReport {
        seed: traced.result.seed,
        plies: traced.result.plies,
        utilities: traced.result.utilities,
        winner,
        moves,
        chance_events,
        lost_cities_state: None,
        splendor_state: Some(state),
        unassigned_maintenance_seconds: traced.unassigned_maintenance_time.map(|d| d.as_secs_f64()),
        final_board: Vec::new(),
        pools: None,
        spirit_forest: None,
        spirit_collections: None,
        gemstone_pools: None,
        scores: Some(scores),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GameId, game_search_capabilities, run_batch, run_match, run_match_with_trace};
    use std::num::NonZeroU32;

    #[test]
    fn interactive_session_replays_catalog_actions_and_rejects_player_chance() {
        let report = run_match_with_trace(
            GameId::Splendor,
            AgentConfig::Random,
            AgentConfig::Random,
            MatchConfig::new(42, NonZeroU32::new(10000).unwrap()),
        )
        .unwrap();
        let mut session = SplendorSession::new(42, None, None).unwrap();
        let before = session.state.clone();
        assert!(session.step(Some(usize::MAX)).is_err());
        assert_eq!(session.state, before);
        for m in &report.moves {
            assert!(session.waiting_human());
            let CatalogAction::Splendor(action) = m.action else {
                panic!("wrong game")
            };
            let index = session
                .game
                .legal_actions(&session.state)
                .position(|a| a == action)
                .unwrap();
            session.step(Some(index)).unwrap();
            if session.game.status(&session.state) == PositionStatus::Chance {
                let before = session.state.clone();
                assert!(session.step(Some(0)).is_err());
                assert_eq!(session.state, before);
                session.step(None).unwrap();
            }
        }
        assert_eq!(Some(session.state.clone()), report.splendor_state);
        assert!(session.step(None).is_err());
        assert_eq!(
            session.events.iter().filter(|e| e.player.is_none()).count(),
            report.chance_events.len()
        );
    }

    #[test]
    fn catalog_registers_seeded_matches_batch_and_explicit_replay() {
        let config = MatchConfig::new(42, NonZeroU32::new(10000).unwrap());
        let report = run_match_with_trace(
            GameId::Splendor,
            AgentConfig::Random,
            AgentConfig::Random,
            config,
        )
        .unwrap();
        assert!(!report.chance_events.is_empty());
        assert_eq!(
            replay(42, &report.moves, &report.chance_events).unwrap(),
            report.splendor_state.clone().unwrap()
        );
        assert!(replay(42, &report.moves, &[]).is_err());
        let mut wrong = report.chance_events.clone();
        wrong[0].after_ply = 0;
        assert!(replay(42, &report.moves, &wrong).is_err());
        let direct = run_match(
            GameId::Splendor,
            AgentConfig::Random,
            AgentConfig::Random,
            config,
        )
        .unwrap();
        assert_eq!(direct.utilities, report.utilities);
        assert_eq!(direct.plies, report.plies);
        let batch = run_batch(
            GameId::Splendor,
            AgentConfig::Random,
            AgentConfig::Random,
            42,
            NonZeroU32::new(2).unwrap(),
            config.max_plies,
        )
        .unwrap();
        assert_eq!(batch[0], direct);
        assert_eq!(batch[1].seed, 43);
        let caps = game_search_capabilities(GameId::Splendor);
        assert_eq!(caps.heuristics.len(), 2);
        assert!(!caps.turn_phase_conditions);
    }
}

#[derive(Clone, Debug)]
pub struct SplendorEvent {
    pub player: Option<PlayerId>,
    pub action: SplendorAction,
    pub seconds: Duration,
    pub stats: AgentDecisionStats,
}

pub struct SplendorSession {
    pub game: Splendor,
    pub state: SplendorState,
    pub events: Vec<SplendorEvent>,
    agents: [Option<SplendorAgent>; 2],
    chance_rng: SplitMix64,
    agent_rngs: [SplitMix64; 2],
}
impl SplendorSession {
    pub fn new(
        seed: u64,
        first: Option<AgentConfig>,
        second: Option<AgentConfig>,
    ) -> Result<Self, AgentError> {
        let mut agents = [
            first.map(configured_agent).transpose()?,
            second.map(configured_agent).transpose()?,
        ];
        let game = game(seed);
        let state = game.initial_state();
        for (i, a) in agents.iter_mut().enumerate() {
            if let Some(a) = a {
                a.on_match_start(
                    &game,
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
            game,
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
        matches!(self.game.status(&self.state),PositionStatus::PlayerTurn(p) if self.agents[p.index()].is_none())
    }
    pub fn step(&mut self, chosen: Option<usize>) -> Result<(), AgentError> {
        if self.events.len() >= 20_000 {
            return Err(AgentError::message(
                "Splendor session exceeded 20000 events",
            ));
        }
        let (player, action, seconds, stats) = match self.game.status(&self.state) {
            PositionStatus::Chance => {
                if chosen.is_some() {
                    return Err(AgentError::message("players cannot choose chance outcomes"));
                }
                (
                    None,
                    self.game
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
                        DecisionContext::new(&self.game, &self.state, p),
                        &mut self.agent_rngs[p.index()],
                    )?;
                    let stats = <SplendorAgent as Agent<Splendor>>::last_decision_stats(agent);
                    (Some(p), a, start.elapsed(), stats)
                } else {
                    let a = chosen
                        .and_then(|i| self.game.legal_actions(&self.state).nth(i))
                        .ok_or_else(|| AgentError::message("select a legal action index"))?;
                    (Some(p), a, Duration::ZERO, AgentDecisionStats::default())
                }
            }
            _ => return Err(AgentError::message("the game has finished")),
        };
        if player.is_some() {
            self.game.apply_action(&mut self.state, &action)
        } else {
            self.game.apply_chance_outcome(&mut self.state, &action)
        }
        .map_err(|e| AgentError::message(e.to_string()))?;
        for a in self.agents.iter_mut().flatten() {
            if let Some(p) = player {
                a.on_action_applied(&self.game, &self.state, p, &action);
            } else {
                a.on_chance_applied(&self.game, &self.state, &action);
            }
            if self.game.status(&self.state) == PositionStatus::Terminal {
                a.on_match_end(&self.game, &self.state);
            }
        }
        self.events.push(SplendorEvent {
            player,
            action,
            seconds,
            stats,
        });
        Ok(())
    }
}
