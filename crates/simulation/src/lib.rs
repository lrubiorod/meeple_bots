//! Generic, reproducible match execution.

use std::{
    error::Error,
    fmt,
    num::NonZeroU32,
    time::{Duration, Instant},
};

use meeple_bots_core::{
    Agent, AgentError, DecisionContext, DeterministicGame, Game, IllegalAction, PlayerId,
    PositionStatus, RandomSource,
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
    ) {
    }

    fn on_finish(&mut self, _game: &G, _state: &G::State, _result: &MatchResult) {}
}

#[derive(Default)]
pub struct NoopObserver;

impl<G: Game> MatchObserver<G> for NoopObserver {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionTrace<A> {
    pub actions: Vec<(PlayerId, A)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TracedMatchResult<A> {
    pub result: MatchResult,
    pub actions: Vec<(PlayerId, A)>,
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
    fn on_action(
        &mut self,
        _game: &G,
        _state: &G::State,
        player: PlayerId,
        action: &A,
        _decision_time: Duration,
    ) {
        self.actions.push((player, action.clone()));
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
        self.observer.measures_decision_time()
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
    ) {
        self.trace
            .on_action(game, state, player, action, decision_time);
        self.observer
            .on_action(game, state, player, action, decision_time);
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
                observer.on_finish(game, &state, &result);
                return Ok(result);
            }
            PositionStatus::PlayerTurn(player) => {
                if plies >= config.max_plies.get() {
                    return Err(MatchError::PlyLimitExceeded(config.max_plies.get()));
                }

                let decision = DecisionContext::new(game, &state, player);
                let decision_started = observer.measures_decision_time().then(Instant::now);
                let action = match player {
                    PlayerId::FIRST => first
                        .select_action(decision, &mut first_rng)
                        .map_err(|source| MatchError::Agent { player, source })?,
                    PlayerId::SECOND => second
                        .select_action(decision, &mut second_rng)
                        .map_err(|source| MatchError::Agent { player, source })?,
                    _ => return Err(MatchError::InvalidPlayer(player)),
                };
                let decision_time =
                    decision_started.map_or(Duration::ZERO, |start| start.elapsed());

                game.apply_action(&mut state, &action)
                    .map_err(|source| MatchError::IllegalAction { player, source })?;
                plies += 1;
                observer.on_action(game, &state, player, &action, decision_time);
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

    #[test]
    fn split_mix_is_reproducible_and_streams_differ() {
        let mut first = SplitMix64::new(42);
        let mut repeated = SplitMix64::new(42);
        let mut other = SplitMix64::new(43);

        assert_eq!(first.next_u64(), repeated.next_u64());
        assert_ne!(first.next_u64(), other.next_u64());
    }
}
