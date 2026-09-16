use std::collections::BTreeMap;

use crate::{Game, PlayerId};

pub type HeuristicParameters = BTreeMap<String, f64>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeuristicParameterSpec {
    pub name: &'static str,
    pub default: f64,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
}

impl HeuristicParameterSpec {
    pub fn resolve(&self, parameters: &HeuristicParameters) -> f64 {
        parameters.get(self.name).copied().unwrap_or(self.default)
    }
}

/// Contract: status never returns PositionStatus::Chance.
pub trait DeterministicGame: Game {}

/// Contract: every player can observe the complete authoritative state.
pub trait PerfectInformationGame: Game {}

/// Optional hidden-information contract. Observations must contain only the observer's
/// information, with no authoritative-state handle. Eq/Hash/serialization must not leak
/// hidden assignments. Sampling uses only this observation and the supplied RNG, and
/// observing the result again must reproduce the input. A temporary simulation world
/// may also fix future randomness; it must not resample the same uncertainty.
pub trait ImperfectInformationGame: Game {
    type Determinization;
    fn sample_determinization<R: crate::RandomSource + ?Sized>(
        &self,
        observation: &Self::Observation<'_>,
        observer: PlayerId,
        rng: &mut R,
    ) -> Result<Self::Determinization, crate::IllegalAction>;
}

/// Optional contract for games that provide state evaluations to search agents.
pub trait HeuristicGame: Game {
    /// Number of heuristic variants exposed by this game.
    fn heuristic_count(&self) -> u32;

    /// Returns a normalized utility for `player`, or `None` for an unknown index.
    fn heuristic_utility(&self, index: u32, state: &Self::State, player: PlayerId) -> Option<f32>;

    /// Describes the named numeric parameters accepted by one heuristic.
    fn heuristic_parameter_specs(&self, index: u32) -> Option<&'static [HeuristicParameterSpec]> {
        (index < self.heuristic_count()).then_some(&[])
    }

    /// Evaluates a state with named parameters validated against `heuristic_parameter_specs`.
    fn heuristic_utility_with_parameters(
        &self,
        index: u32,
        parameters: &HeuristicParameters,
        state: &Self::State,
        player: PlayerId,
    ) -> Option<f32> {
        parameters
            .is_empty()
            .then(|| self.heuristic_utility(index, state, player))
            .flatten()
    }
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
