use crate::{IllegalAction, PlayerId, RandomSource};

/// Whose decision is required at a position, or whether the game has ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PositionStatus {
    PlayerTurn(PlayerId),
    /// A stochastic event sampled by the environment, never by a player.
    /// Its outcome may be private; observations control what each player sees.
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

    /// Rule-proven upper bound on player decisions from setup to terminal.
    /// Chance events are excluded. None means no useful bound is provided.
    fn maximum_decision_horizon(&self) -> Option<u32> {
        None
    }
    fn status(&self, state: &Self::State) -> PositionStatus;

    /// Optional public, human-readable decision kind for analysis only.
    /// Return None for games without meaningful phases and for non-decision states.
    /// Labels must not contain hidden information or influence agent behavior.
    fn diagnostic_phase(&self, _state: &Self::State) -> Option<&'static str> {
        None
    }

    /// Whether this position is between physical turns, before the next player decision.
    /// The initial position is also a boundary. This is independent of player identity:
    /// the same player may take consecutive turns. Mandatory chance preparation may
    /// leave a boundary in place, but callers must resolve Chance before evaluating.
    ///
    /// The default describes one-decision turns (possibly followed by chance).
    /// The rollout policy must eventually reach another boundary or terminal position.
    /// Games with microactions must override this using their authoritative phase or
    /// progress state, returning false until the current physical turn is complete.
    fn is_turn_boundary(&self, state: &Self::State) -> bool {
        !matches!(self.status(state), PositionStatus::Chance)
    }

    fn legal_actions<'a>(&'a self, state: &'a Self::State) -> Self::LegalActions<'a>;

    fn apply_action(
        &self,
        state: &mut Self::State,
        action: &Self::Action,
    ) -> Result<(), IllegalAction>;

    /// Enumerated authoritative events. Probabilities must be finite, positive and sum to one.
    /// Event values share the transition transport type, but are never player legal actions.
    fn chance_outcomes(
        &self,
        _state: &Self::State,
    ) -> Result<Vec<(Self::Action, f64)>, IllegalAction> {
        Err(IllegalAction::new(
            "this game does not enumerate chance events",
        ))
    }

    /// Apply an environment event, separately from a player's choice.
    fn apply_chance_outcome(
        &self,
        state: &mut Self::State,
        outcome: &Self::Action,
    ) -> Result<(), IllegalAction> {
        if self.status(state) != PositionStatus::Chance {
            return Err(IllegalAction::new("not a chance position"));
        }
        self.apply_action(state, outcome)
    }

    /// Sample an environment event according to its probability distribution. Apply it with
    /// `apply_chance_outcome`; event actions must not appear in `legal_actions`. Sampling must
    /// only use the supplied RNG, never a hidden seed stored in the authoritative state.
    /// Deterministic games keep the default implementation.
    fn sample_chance<R: RandomSource + ?Sized>(
        &self,
        state: &Self::State,
        rng: &mut R,
    ) -> Result<Self::Action, IllegalAction> {
        if self.status(state) != PositionStatus::Chance {
            return Err(IllegalAction::new("not a chance position"));
        }
        let outcomes = self.chance_outcomes(state)?;
        validate_chance_probabilities(outcomes.iter().map(|(_, p)| *p))?;
        let draw = rng.unit_f64();
        let mut total = 0.0;
        let last = outcomes.len() - 1;
        for (index, (outcome, probability)) in outcomes.into_iter().enumerate() {
            total += probability;
            if draw < total || index == last {
                return Ok(outcome);
            }
        }
        unreachable!("validated nonempty distribution")
    }

    fn observation<'a>(&'a self, state: &'a Self::State, player: PlayerId)
    -> Self::Observation<'a>;

    /// Returns a player's normalized terminal utility, or None while ongoing.
    fn terminal_utility(&self, state: &Self::State, player: PlayerId) -> Option<f32>;
}

/// Reject malformed distributions before sampling; do not silently normalize game bugs.
pub fn validate_chance_probabilities(
    probabilities: impl IntoIterator<Item = f64>,
) -> Result<(), IllegalAction> {
    let mut sum = 0.0;
    for p in probabilities {
        if !p.is_finite() || p <= 0.0 {
            return Err(IllegalAction::new(
                "chance probabilities must be finite and positive",
            ));
        }
        sum += p;
    }
    if (sum - 1.0).abs() > 1e-10 {
        return Err(IllegalAction::new("chance probabilities must sum to one"));
    }
    Ok(())
}

#[cfg(test)]
mod probability_tests {
    use super::*;
    #[test]
    fn validates_normalized_positive_finite_distributions() {
        assert!(validate_chance_probabilities([0.25, 0.75]).is_ok());
        for values in [
            vec![],
            vec![0.5],
            vec![0.0, 1.0],
            vec![-1.0, 2.0],
            vec![f64::NAN],
            vec![f64::INFINITY],
        ] {
            assert!(validate_chance_probabilities(values).is_err());
        }
    }
}
