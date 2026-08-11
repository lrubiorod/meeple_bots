use crate::{Game, PlayerId};

/// Contract: status never returns PositionStatus::Chance.
pub trait DeterministicGame: Game {}

/// Contract: every player can observe the complete authoritative state.
pub trait PerfectInformationGame: Game {}

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
