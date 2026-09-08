use crate::{IllegalAction, PlayerId};

/// Whose decision is required at a position, or whether the game has ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PositionStatus {
    PlayerTurn(PlayerId),
    /// Reserved for games with stochastic transitions.
    /// The current deterministic runner rejects this status; it does not sample
    /// or apply chance outcomes.
    Chance,
    Terminal,
}

/// Authoritative rules and state transitions for a turn-based game.
///
/// State and action representations remain owned by the concrete game. The
/// observation GAT can represent a player-filtered view. It does not by itself
/// provide hidden-information isolation: current agent lifecycle callbacks and
/// simulation observers still receive the complete authoritative state.
pub trait Game {
    type State: 'static;
    type Action: 'static;
    type Observation<'a>
    where
        Self: 'a;
    type LegalActions<'a>: Iterator<Item = Self::Action> + 'a
    where
        Self: 'a;

    fn player_count(&self) -> u8;
    fn initial_state(&self) -> Self::State;
    fn status(&self, state: &Self::State) -> PositionStatus;

    fn legal_actions<'a>(&'a self, state: &'a Self::State) -> Self::LegalActions<'a>;

    fn apply_action(
        &self,
        state: &mut Self::State,
        action: &Self::Action,
    ) -> Result<(), IllegalAction>;

    fn observation<'a>(&'a self, state: &'a Self::State, player: PlayerId)
    -> Self::Observation<'a>;

    /// Returns a player's normalized terminal utility, or None while ongoing.
    fn terminal_utility(&self, state: &Self::State, player: PlayerId) -> Option<f32>;
}
