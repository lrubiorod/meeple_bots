//! Only audited RandomAgent is registered. Authoritative replay is an engine/admin API.
use crate::{AgentConfig, CatalogAction, CatalogError, CatalogMatchReport, RecordedMove};
use meeple_bots_core::{Game, PlayerId, PositionStatus};
use meeple_bots_lost_cities::{LostCities, LostCitiesAction, LostCitiesState};
use meeple_bots_random_agent::RandomAgent;
use meeple_bots_simulation::{MatchConfig, TracedChance, TracedMatchResult, play_match_with_trace};
pub const SEARCH_UNAVAILABLE: &str = "Lost Cities has imperfect information: standard MCTS is not compatible; no searchable agent yet (SO-ISMCTS is not implemented)";
/// Perfect-information searches cannot implement Agent<LostCities>.
/// ```compile_fail
/// use meeple_bots_core::Agent;
/// use meeple_bots_lost_cities::LostCities;
/// fn compatible<A: Agent<LostCities>>(_: A) {}
/// compatible(meeple_bots_mcts_agent::MctsAgent::default());
/// ```
/// ```compile_fail
/// use meeple_bots_core::Agent;
/// use meeple_bots_lost_cities::LostCities;
/// use meeple_bots_mcts_agent::{StochasticMctsAgent, MctsConfig, NeutralEvaluator};
/// fn compatible<A: Agent<LostCities>>(_: A) {}
/// compatible(StochasticMctsAgent::new(MctsConfig::default(), NeutralEvaluator));
/// ```
pub fn run(
    first: AgentConfig,
    second: AgentConfig,
    config: MatchConfig,
) -> Result<TracedMatchResult<LostCitiesAction>, CatalogError> {
    if !matches!(first, AgentConfig::Random) || !matches!(second, AgentConfig::Random) {
        return Err(CatalogError::InvalidMctsConfig(SEARCH_UNAVAILABLE));
    }
    // RandomAgent uses only legal actions; its lifecycle callbacks are no-ops.
    play_match_with_trace(&LostCities, &mut RandomAgent, &mut RandomAgent, config)
        .map_err(Into::into)
}
pub fn replay(
    moves: &[RecordedMove],
    events: &[TracedChance<CatalogAction>],
) -> Result<LostCitiesState, CatalogError> {
    let invalid = |message: String| CatalogError::InvalidTrace {
        game: crate::GameId::LostCities,
        message,
    };
    let mut state = LostCities.initial_state();
    let mut pending = events.iter().peekable();
    for i in 0..=moves.len() {
        while pending.peek().is_some_and(|e| e.after_ply == i) {
            let CatalogAction::LostCities(event) = pending.next().unwrap().event else {
                return Err(invalid("expected Lost Cities event".into()));
            };
            LostCities
                .apply_chance_outcome(&mut state, &event)
                .map_err(|e| invalid(e.to_string()))?;
        }
        if let Some(m) = moves.get(i) {
            if m.player > 1
                || LostCities.status(&state)
                    != PositionStatus::PlayerTurn(if m.player == 0 {
                        PlayerId::FIRST
                    } else {
                        PlayerId::SECOND
                    })
            {
                return Err(invalid("incorrect player or transition order".into()));
            }
            let CatalogAction::LostCities(action) = m.action else {
                return Err(invalid("expected Lost Cities action".into()));
            };
            LostCities
                .apply_action(&mut state, &action)
                .map_err(|e| invalid(e.to_string()))?;
        }
    }
    if pending.next().is_some() || LostCities.status(&state) != PositionStatus::Terminal {
        return Err(invalid("missing or extra transitions".into()));
    }
    LostCities
        .validate_state(&state)
        .map_err(|e| invalid(e.to_string()))?;
    Ok(state)
}
pub fn report(
    traced: TracedMatchResult<LostCitiesAction>,
) -> Result<CatalogMatchReport, CatalogError> {
    let moves: Vec<_> = traced
        .actions
        .into_iter()
        .map(|a| RecordedMove {
            player: a.player.index(),
            action: CatalogAction::LostCities(a.action),
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
        .map(|e| TracedChance {
            after_ply: e.after_ply,
            event: CatalogAction::LostCities(e.event),
        })
        .collect();
    let state = replay(&moves, &chance_events)?;
    let scores = state.scores();
    Ok(CatalogMatchReport {
        seed: traced.result.seed,
        plies: traced.result.plies,
        winner: traced.result.utilities.iter().position(|&v| v > 0.),
        utilities: traced.result.utilities,
        moves,
        chance_events,
        lost_cities_state: Some(state),
        splendor_state: None,
        unassigned_maintenance_seconds: traced.unassigned_maintenance_time.map(|d| d.as_secs_f64()),
        final_board: vec![],
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
    use crate::{GameId, game_search_capabilities, run_match_with_trace};
    #[test]
    fn registration_replay_and_search_rejection() {
        let caps = game_search_capabilities(GameId::LostCities);
        assert!(caps.imperfect_information && caps.stochastic);
        assert!(caps.selection_policies.is_empty());
        let report = run_match_with_trace(
            GameId::LostCities,
            AgentConfig::Random,
            AgentConfig::Random,
            MatchConfig::default(),
        )
        .unwrap();
        let state = replay(&report.moves, &report.chance_events).unwrap();
        assert_eq!(report.scores, Some(state.scores()));
        assert_eq!(report.lost_cities_state, Some(state));
        assert!(replay(&report.moves, &report.chance_events[1..]).is_err());
        let config = crate::MctsAgentConfig {
            search: crate::MctsConfig {
                progressive_widening: None,
                selection_policy: Default::default(),
                budget: Default::default(),
                exploration: 1.4,
                rollout_depth: 1,
                rollout_policy: crate::RolloutPolicyConfig::UniformRandom.into(),
            },
            cutoff_evaluator: crate::EvaluatorConfig::Neutral,
            progressive_bias: Default::default(),
            root_diagnostics: false,
            tree_reuse: false,
            transpositions: false,
        };
        assert!(
            run(
                AgentConfig::Mcts(config),
                AgentConfig::Random,
                MatchConfig::default()
            )
            .unwrap_err()
            .to_string()
            .contains("imperfect information")
        );
    }
}
