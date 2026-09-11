//! Strongly typed contracts shared by games, agents, and simulations.

mod agent;
mod capabilities;
mod error;
mod game;
mod player;
mod random;

pub use agent::{Agent, AgentDecisionStats, DecisionContext, RootActionStats, TreeReuseStats};
pub use capabilities::{
    DeterministicGame, HeuristicGame, HeuristicParameterSpec, HeuristicParameters,
    PerfectInformationGame, TwoPlayerZeroSumGame,
};
pub use error::{AgentError, IllegalAction};
pub use game::{Game, PositionStatus, validate_chance_probabilities};
pub use player::PlayerId;
pub use random::RandomSource;
