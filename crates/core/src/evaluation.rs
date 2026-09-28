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
