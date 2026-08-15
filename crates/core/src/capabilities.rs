use crate::{Game, PlayerId};

/// Contract: status never returns PositionStatus::Chance.
pub trait DeterministicGame: Game {}

/// Contract: every player can observe the complete authoritative state.
pub trait PerfectInformationGame: Game {}

/// Optional contract for games that provide state evaluations to search agents.
pub trait HeuristicGame: Game {
    /// Number of heuristic variants exposed by this game.
    fn heuristic_count(&self) -> u32;

    /// Returns a normalized utility for `player`, or `None` for an unknown index.
    fn heuristic_utility(&self, index: u32, state: &Self::State, player: PlayerId) -> Option<f32>;
}

/// Contract used by the first MCTS implementation.
pub trait TwoPlayerZeroSumGame: Game {
    fn opponent(player: PlayerId) -> Option<PlayerId> {
        match player {
            PlayerId::FIRST => Some(PlayerId::SECOND),
            PlayerId::SECOND => Some(PlayerId::FIRST),
            _ => None,
        }
    }
}
