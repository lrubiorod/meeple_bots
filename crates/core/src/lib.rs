//! Strongly typed contracts shared by games, agents, and simulations.

mod agent;
mod capabilities;
mod error;
mod game;
mod player;
mod random;

pub use agent::{Agent, AgentDecisionStats, DecisionContext, RootActionStats, TreeReuseStats};
pub use capabilities::{
    DeterministicGame, DeterminizedWorld, HeuristicGame, HeuristicParameterSpec,
    HeuristicParameters, ImperfectInformationGame, PerfectInformationGame, TwoPlayerZeroSumGame,
};
pub use error::{AgentError, IllegalAction};
pub use game::{Game, PositionStatus, validate_chance_probabilities};
pub use player::PlayerId;
pub use random::RandomSource;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchBudget {
    Iterations(std::num::NonZeroU32),
    Time(std::time::Duration),
}

impl Default for SearchBudget {
    fn default() -> Self {
        Self::Iterations(std::num::NonZeroU32::new(1_000).expect("constant is non-zero"))
    }
}
