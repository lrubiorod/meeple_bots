//! Strongly typed contracts shared by games, agents, and simulations.

mod agent;
mod capabilities;
mod error;
mod game;
mod player;
mod random;

pub use agent::{Agent, DecisionContext};
pub use capabilities::{DeterministicGame, PerfectInformationGame, TwoPlayerZeroSumGame};
pub use error::{AgentError, IllegalAction};
pub use game::{Game, PositionStatus};
pub use player::PlayerId;
pub use random::RandomSource;
