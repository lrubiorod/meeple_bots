//! Uniform random policy for every game implementing the core contract.

use meeple_bots_core::{Agent, AgentError, DecisionContext, Game, RandomSource};

#[derive(Clone, Copy, Debug, Default)]
pub struct RandomAgent;

impl<G: Game> Agent<G> for RandomAgent {
    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, G>,
        rng: &mut R,
    ) -> Result<G::Action, AgentError> {
        let mut selected = None;
        for (index, action) in decision.legal_actions().enumerate() {
            if rng.index(index + 1) == Some(0) {
                selected = Some(action);
            }
        }
        selected.ok_or(AgentError::NoLegalActions)
    }
}

#[cfg(test)]
mod tests {
    use meeple_bots_simulation::{ActionTrace, MatchConfig, SplitMix64, play_match};
    use meeple_bots_tic_tac_toe::TicTacToe;

    use super::*;

    #[test]
    fn same_seed_selects_same_action() {
        let game = TicTacToe;
        let state = game.initial_state();
        let mut first_rng = SplitMix64::new(7);
        let mut repeated_rng = SplitMix64::new(7);

        let first = RandomAgent
            .select_action(
                DecisionContext::new(&game, &state, meeple_bots_core::PlayerId::FIRST),
                &mut first_rng,
            )
            .unwrap();
        let repeated = RandomAgent
            .select_action(
                DecisionContext::new(&game, &state, meeple_bots_core::PlayerId::FIRST),
                &mut repeated_rng,
            )
            .unwrap();

        assert_eq!(first, repeated);
    }

    #[test]
    fn random_agents_finish_a_legal_match() {
        let game = TicTacToe;
        let result = play_match(
            &game,
            &mut RandomAgent,
            &mut RandomAgent,
            MatchConfig::default(),
        )
        .unwrap();

        assert!((5..=9).contains(&result.plies));
        let _trace: ActionTrace<meeple_bots_tic_tac_toe::TicTacToeAction> = ActionTrace::default();
    }
}
