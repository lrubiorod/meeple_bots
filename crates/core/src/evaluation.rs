use crate::{AgentError, Game, PlayerId};

/// Scalar state evaluation from an explicitly requested player's perspective.
///
/// The value represents finite normalized utility in [-1, 1]. Search consumers decide
/// when to evaluate and validate the result at their own boundary; authoritative
/// terminal utility remains part of [`Game`].
pub trait StateEvaluator<G: Game> {
    fn evaluate(
        &self,
        game: &G,
        state: &G::State,
        root_player: PlayerId,
    ) -> Result<f64, AgentError>;
}

/// Scalar evaluation of legitimate player information only.
///
/// Unlike StateEvaluator, implementations receive no game, state or sampled world.
/// Values are finite utilities in [-1, 1], from the explicitly requested observer.
pub trait ObservationEvaluator<O> {
    fn evaluate(&self, observation: &O, root_player: PlayerId) -> Result<f64, AgentError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NeutralObservationEvaluator;

impl<O> ObservationEvaluator<O> for NeutralObservationEvaluator {
    fn evaluate(&self, _: &O, _: PlayerId) -> Result<f64, AgentError> {
        Ok(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn neutral_observation_value_is_zero_for_either_perspective() {
        for player in [PlayerId::FIRST, PlayerId::SECOND] {
            assert_eq!(
                NeutralObservationEvaluator.evaluate(&(), player).unwrap(),
                0.0
            );
        }
    }
}
