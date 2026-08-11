use crate::{AgentError, Game, PerfectInformationGame, PlayerId, RandomSource};

/// Read-only decision boundary handed to an agent by the simulation.
pub struct DecisionContext<'a, G: Game> {
    game: &'a G,
    state: &'a G::State,
    player: PlayerId,
}

impl<'a, G: Game> DecisionContext<'a, G> {
    pub fn new(game: &'a G, state: &'a G::State, player: PlayerId) -> Self {
        Self {
            game,
            state,
            player,
        }
    }

    pub fn game(&self) -> &'a G {
        self.game
    }

    pub const fn player(&self) -> PlayerId {
        self.player
    }

    pub fn observation(&self) -> G::Observation<'a> {
        self.game.observation(self.state, self.player)
    }

    pub fn legal_actions(&self) -> G::LegalActions<'a> {
        self.game.legal_actions(self.state)
    }
}

impl<'a, G: PerfectInformationGame> DecisionContext<'a, G> {
    /// Full state access is deliberately available only for perfect-information games.
    pub fn state(&self) -> &'a G::State {
        self.state
    }
}

/// A policy that selects strongly typed actions for G.
pub trait Agent<G: Game> {
    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, G>,
        rng: &mut R,
    ) -> Result<G::Action, AgentError>;
}
