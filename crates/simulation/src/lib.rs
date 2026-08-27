//! Generic, reproducible match execution.

use std::{
    error::Error,
    fmt,
    num::NonZeroU32,
    time::{Duration, Instant},
};

use meeple_bots_core::{
    Agent, AgentDecisionStats, AgentError, DecisionContext, DeterministicGame, Game, IllegalAction,
    PlayerId, PositionStatus, RandomSource,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatchConfig {
    pub seed: u64,
    pub max_plies: NonZeroU32,
}

impl MatchConfig {
    pub const fn new(seed: u64, max_plies: NonZeroU32) -> Self {
        Self { seed, max_plies }
    }
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            max_plies: NonZeroU32::new(10_000).expect("constant is non-zero"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatchResult {
    pub seed: u64,
    pub plies: u32,
    pub utilities: Vec<f32>,
}

#[derive(Debug)]
pub enum MatchError {
    UnsupportedPlayerCount(u8),
    InvalidPlayer(PlayerId),
    UnexpectedChance,
    PlyLimitExceeded(u32),
    Agent {
        player: PlayerId,
        source: AgentError,
    },
    IllegalAction {
        player: PlayerId,
        source: IllegalAction,
    },
    MissingTerminalUtility(PlayerId),
}

impl fmt::Display for MatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlayerCount(count) => {
                write!(formatter, "the two-player runner received {count} players")
            }
            Self::InvalidPlayer(player) => write!(formatter, "invalid player {player}"),
            Self::UnexpectedChance => {
                formatter.write_str("a deterministic game requested a chance transition")
            }
            Self::PlyLimitExceeded(limit) => {
                write!(formatter, "the match exceeded its {limit}-ply limit")
            }
            Self::Agent { player, source } => {
                write!(formatter, "agent {player} failed: {source}")
            }
            Self::IllegalAction { player, source } => {
                write!(
                    formatter,
                    "agent {player} selected an illegal action: {source}"
                )
            }
            Self::MissingTerminalUtility(player) => {
                write!(formatter, "terminal utility is missing for player {player}")
            }
        }
    }
}

impl Error for MatchError {}

/// Compile-time observer: NoopObserver is optimized away when traces are disabled.
pub trait MatchObserver<G: Game> {
    fn measures_decision_time(&self) -> bool {
        false
    }

    fn on_start(&mut self, _game: &G, _state: &G::State) {}

    fn on_action(
        &mut self,
        _game: &G,
        _state: &G::State,
        _player: PlayerId,
        _action: &G::Action,
        _decision_time: Duration,
        _decision_stats: AgentDecisionStats,
    ) {
    }

    fn on_finish(&mut self, _game: &G, _state: &G::State, _result: &MatchResult) {}
}

#[derive(Default)]
pub struct NoopObserver;

impl<G: Game> MatchObserver<G> for NoopObserver {}

#[derive(Clone, Debug, PartialEq)]
pub struct ActionTrace<A> {
    pub actions: Vec<TracedAction<A>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TracedAction<A> {
    pub player: PlayerId,
    pub action: A,
    pub decision_time: Duration,
    pub decision_stats: AgentDecisionStats,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TracedMatchResult<A> {
    pub result: MatchResult,
    pub actions: Vec<TracedAction<A>>,
}

impl<A> Default for ActionTrace<A> {
    fn default() -> Self {
        Self {
            actions: Vec::new(),
        }
    }
}

impl<G, A> MatchObserver<G> for ActionTrace<A>
where
    G: Game<Action = A>,
    A: Clone,
{
    fn measures_decision_time(&self) -> bool {
        true
    }

    fn on_action(
        &mut self,
        _game: &G,
        _state: &G::State,
        player: PlayerId,
        action: &A,
        decision_time: Duration,
        decision_stats: AgentDecisionStats,
    ) {
        self.actions.push(TracedAction {
            player,
            action: action.clone(),
            decision_time,
            decision_stats,
        });
    }
}

struct TracingObserver<'a, O, A> {
    trace: ActionTrace<A>,
    observer: &'a mut O,
}

impl<G, O, A> MatchObserver<G> for TracingObserver<'_, O, A>
where
    G: Game<Action = A>,
    O: MatchObserver<G>,
    A: Clone,
{
    fn measures_decision_time(&self) -> bool {
        true
    }

    fn on_start(&mut self, game: &G, state: &G::State) {
        self.observer.on_start(game, state);
    }

    fn on_action(
        &mut self,
        game: &G,
        state: &G::State,
        player: PlayerId,
        action: &A,
        decision_time: Duration,
        decision_stats: AgentDecisionStats,
    ) {
        self.trace.on_action(
            game,
            state,
            player,
            action,
            decision_time,
            decision_stats.clone(),
        );
        self.observer
            .on_action(game, state, player, action, decision_time, decision_stats);
    }

    fn on_finish(&mut self, game: &G, state: &G::State, result: &MatchResult) {
        self.observer.on_finish(game, state, result);
    }
}

/// SplitMix64 is fast, deterministic, and sufficient for simulation stream derivation.
#[derive(Clone, Copy, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl RandomSource for SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
}

pub fn play_match<G, A, B>(
    game: &G,
    first: &mut A,
    second: &mut B,
    config: MatchConfig,
) -> Result<MatchResult, MatchError>
where
    G: DeterministicGame,
    A: Agent<G>,
    B: Agent<G>,
{
    play_match_with_observer(game, first, second, config, &mut NoopObserver)
}

pub fn play_match_with_trace<G, A, B>(
    game: &G,
    first: &mut A,
    second: &mut B,
    config: MatchConfig,
) -> Result<TracedMatchResult<G::Action>, MatchError>
where
    G: DeterministicGame,
    G::Action: Clone,
    A: Agent<G>,
    B: Agent<G>,
{
    let mut trace = ActionTrace::default();
    let result = play_match_with_observer(game, first, second, config, &mut trace)?;

    Ok(TracedMatchResult {
        result,
        actions: trace.actions,
    })
}

/// Run a match while retaining its trace and forwarding live observer events.
pub fn play_match_with_trace_and_observer<G, A, B, O>(
    game: &G,
    first: &mut A,
    second: &mut B,
    config: MatchConfig,
    observer: &mut O,
) -> Result<TracedMatchResult<G::Action>, MatchError>
where
    G: DeterministicGame,
    G::Action: Clone,
    A: Agent<G>,
    B: Agent<G>,
    O: MatchObserver<G>,
{
    let mut tracing_observer = TracingObserver {
        trace: ActionTrace::default(),
        observer,
    };
    let result = play_match_with_observer(game, first, second, config, &mut tracing_observer)?;

    Ok(TracedMatchResult {
        result,
        actions: tracing_observer.trace.actions,
    })
}

pub fn play_match_with_observer<G, A, B, O>(
    game: &G,
    first: &mut A,
    second: &mut B,
    config: MatchConfig,
    observer: &mut O,
) -> Result<MatchResult, MatchError>
where
    G: DeterministicGame,
    A: Agent<G>,
    B: Agent<G>,
    O: MatchObserver<G>,
{
    if game.player_count() != 2 {
        return Err(MatchError::UnsupportedPlayerCount(game.player_count()));
    }

    let mut state = game.initial_state();
    let mut first_rng = SplitMix64::new(config.seed ^ 0xA076_1D64_78BD_642F);
    let mut second_rng = SplitMix64::new(config.seed ^ 0xE703_7ED1_A0B4_28DB);
    let mut plies = 0;
    first.on_match_start(game, &state, PlayerId::FIRST);
    second.on_match_start(game, &state, PlayerId::SECOND);
    observer.on_start(game, &state);

    loop {
        match game.status(&state) {
            PositionStatus::Terminal => {
                let utilities = [PlayerId::FIRST, PlayerId::SECOND]
                    .into_iter()
                    .map(|player| {
                        game.terminal_utility(&state, player)
                            .ok_or(MatchError::MissingTerminalUtility(player))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let result = MatchResult {
                    seed: config.seed,
                    plies,
                    utilities,
                };
                first.on_match_end(game, &state);
                second.on_match_end(game, &state);
                observer.on_finish(game, &state, &result);
                return Ok(result);
            }
            PositionStatus::PlayerTurn(player) => {
                if plies >= config.max_plies.get() {
                    return Err(MatchError::PlyLimitExceeded(config.max_plies.get()));
                }

                let decision = DecisionContext::new(game, &state, player);
                let decision_started = observer.measures_decision_time().then(Instant::now);
                let (action, decision_stats) = match player {
                    PlayerId::FIRST => {
                        let action = first
                            .select_action(decision, &mut first_rng)
                            .map_err(|source| MatchError::Agent { player, source })?;
                        (action, first.last_decision_stats())
                    }
                    PlayerId::SECOND => {
                        let action = second
                            .select_action(decision, &mut second_rng)
                            .map_err(|source| MatchError::Agent { player, source })?;
                        (action, second.last_decision_stats())
                    }
                    _ => return Err(MatchError::InvalidPlayer(player)),
                };
                let decision_time =
                    decision_started.map_or(Duration::ZERO, |start| start.elapsed());

                game.apply_action(&mut state, &action)
                    .map_err(|source| MatchError::IllegalAction { player, source })?;
                plies += 1;
                first.on_action_applied(game, &state, player, &action);
                second.on_action_applied(game, &state, player, &action);
                observer.on_action(game, &state, player, &action, decision_time, decision_stats);
            }
            PositionStatus::Chance => return Err(MatchError::UnexpectedChance),
            _ => return Err(MatchError::UnexpectedChance),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchConfig {
    pub seed: u64,
    pub matches: NonZeroU32,
    pub max_plies: NonZeroU32,
}

pub fn play_batch<G, A, B, FA, FB>(
    game: &G,
    config: BatchConfig,
    mut make_first: FA,
    mut make_second: FB,
) -> Result<Vec<MatchResult>, MatchError>
where
    G: DeterministicGame,
    A: Agent<G>,
    B: Agent<G>,
    FA: FnMut() -> A,
    FB: FnMut() -> B,
{
    let mut seed_stream = SplitMix64::new(config.seed);
    let mut results = Vec::with_capacity(config.matches.get() as usize);

    for _ in 0..config.matches.get() {
        let mut first = make_first();
        let mut second = make_second();
        results.push(play_match(
            game,
            &mut first,
            &mut second,
            MatchConfig::new(seed_stream.next_u64(), config.max_plies),
        )?);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct LifecycleGame;

    impl Game for LifecycleGame {
        type State = u8;
        type Action = ();
        type Observation<'a> = &'a u8;
        type LegalActions<'a> = std::option::IntoIter<()>;

        fn player_count(&self) -> u8 {
            2
        }

        fn initial_state(&self) -> Self::State {
            0
        }

        fn status(&self, state: &Self::State) -> PositionStatus {
            match *state {
                0 => PositionStatus::PlayerTurn(PlayerId::FIRST),
                1 => PositionStatus::PlayerTurn(PlayerId::SECOND),
                _ => PositionStatus::Terminal,
            }
        }

        fn legal_actions<'a>(&'a self, state: &'a Self::State) -> Self::LegalActions<'a> {
            (*state < 2).then_some(()).into_iter()
        }

        fn apply_action(
            &self,
            state: &mut Self::State,
            _action: &Self::Action,
        ) -> Result<(), IllegalAction> {
            if *state >= 2 {
                return Err(IllegalAction::new("game is terminal"));
            }
            *state += 1;
            Ok(())
        }

        fn observation<'a>(
            &'a self,
            state: &'a Self::State,
            _player: PlayerId,
        ) -> Self::Observation<'a> {
            state
        }

        fn terminal_utility(&self, state: &Self::State, _player: PlayerId) -> Option<f32> {
            (*state == 2).then_some(0.0)
        }
    }

    impl DeterministicGame for LifecycleGame {}

    #[derive(Default)]
    struct LifecycleAgent {
        seat: Option<PlayerId>,
        observed_players: Vec<PlayerId>,
        ended: bool,
    }

    impl Agent<LifecycleGame> for LifecycleAgent {
        fn on_match_start(&mut self, _game: &LifecycleGame, _state: &u8, player: PlayerId) {
            self.seat = Some(player);
        }

        fn select_action<R: RandomSource + ?Sized>(
            &mut self,
            _decision: DecisionContext<'_, LifecycleGame>,
            _rng: &mut R,
        ) -> Result<(), AgentError> {
            Ok(())
        }

        fn on_action_applied(
            &mut self,
            _game: &LifecycleGame,
            _state: &u8,
            player: PlayerId,
            _action: &(),
        ) {
            self.observed_players.push(player);
        }

        fn on_match_end(&mut self, _game: &LifecycleGame, _state: &u8) {
            self.ended = true;
        }
    }

    #[test]
    fn split_mix_is_reproducible_and_streams_differ() {
        let mut first = SplitMix64::new(42);
        let mut repeated = SplitMix64::new(42);
        let mut other = SplitMix64::new(43);

        assert_eq!(first.next_u64(), repeated.next_u64());
        assert_ne!(first.next_u64(), other.next_u64());
    }

    #[test]
    fn match_lifecycle_is_delivered_to_both_agents() {
        let mut first = LifecycleAgent::default();
        let mut second = LifecycleAgent::default();

        let result = play_match(
            &LifecycleGame,
            &mut first,
            &mut second,
            MatchConfig::new(4, NonZeroU32::new(2).unwrap()),
        )
        .unwrap();

        assert_eq!(result.plies, 2);
        assert_eq!(first.seat, Some(PlayerId::FIRST));
        assert_eq!(second.seat, Some(PlayerId::SECOND));
        assert_eq!(
            first.observed_players,
            vec![PlayerId::FIRST, PlayerId::SECOND]
        );
        assert_eq!(second.observed_players, first.observed_players);
        assert!(first.ended);
        assert!(second.ended);
    }
}
