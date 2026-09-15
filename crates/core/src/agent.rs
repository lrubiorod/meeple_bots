use crate::{AgentError, Game, PerfectInformationGame, PlayerId, RandomSource};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AgentDecisionStats {
    pub search_iterations: Option<u64>,
    pub search_nodes: Option<u64>,
    pub terminal_simulations: Option<u64>,
    pub cutoff_simulations: Option<u64>,
    /// (legal actions, admitted children, current admission limit), when available.
    pub root_expansion: Option<(usize, usize, usize)>,
    pub root_actions: Vec<RootActionStats>,
    pub tree_reuse: Option<TreeReuseStats>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TreeReuseStats {
    pub transition_attempts: u32,
    pub transition_hits: u32,
    pub transition_misses: u32,
    pub own_action_hits: u32,
    pub opponent_action_hits: u32,
    pub reused_root_visits: u32,
    pub reused_nodes: u64,
    pub pruned_nodes: u64,
    pub resets: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootActionStats {
    pub action_index: u32,
    pub visits: u32,
    pub mean_utility: f64,
    pub heuristic_value: Option<f64>,
    pub progressive_bias: Option<f64>,
    pub selected: bool,
}

/// Read-only decision boundary handed to an agent by the simulation.
///
/// Restricting `state()` does not isolate hidden information across the entire
/// agent lifecycle: [`Agent`] callbacks currently receive the authoritative state.
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
    /// Full state access through this decision context requires perfect information.
    pub fn state(&self) -> &'a G::State {
        self.state
    }
}

/// A policy that selects strongly typed actions for G.
///
/// Lifecycle callbacks receive the complete authoritative state without requiring
/// [`PerfectInformationGame`]. This contract is not a hidden-information boundary.
/// Before supporting hidden-information games, redesign these callbacks to supply
/// player-filtered observations or use a separate contract; `DecisionContext::state`
/// alone cannot prevent an agent from retaining private information from callbacks.
pub trait Agent<G: Game> {
    /// Starts a new match and assigns the seat controlled by this agent instance.
    fn on_match_start(&mut self, _game: &G, _state: &G::State, _player: PlayerId) {}

    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        decision: DecisionContext<'_, G>,
        rng: &mut R,
    ) -> Result<G::Action, AgentError>;

    fn last_decision_stats(&self) -> AgentDecisionStats {
        AgentDecisionStats::default()
    }

    /// Observes an action after the game has accepted it and updated the state.
    fn on_action_applied(
        &mut self,
        _game: &G,
        _state: &G::State,
        _player: PlayerId,
        _action: &G::Action,
    ) {
    }

    /// Observe a public chance event after it has been applied.
    fn on_chance_applied(&mut self, _game: &G, _state: &G::State, _event: &G::Action) {}

    fn on_match_end(&mut self, _game: &G, _state: &G::State) {}
}
