//! Two-player Can't Stop, basic rules: public dice and no private information.
use meeple_bots_core::{
    Game, IllegalAction, PerfectInformationGame, PlayerId, PositionStatus, RandomSource,
    TwoPlayerZeroSumGame,
};

pub const HEIGHTS: [u8; 11] = [3, 5, 7, 9, 11, 13, 11, 9, 7, 5, 3];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Phase {
    Roll,
    Choose,
    Continue,
    Finished,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CantStopState {
    pub progress: [[u8; 11]; 2],
    /// Zero means no temporary runner in this column.
    pub runners: [u8; 11],
    pub claimed: [Option<PlayerId>; 11],
    pub dice: [u8; 4],
    pub active: PlayerId,
    pub phase: Phase,
    pub winner: Option<PlayerId>,
    pub last_bust: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum CantStopAction {
    /// Sorted column sums. Zero in the second slot denotes a single advance.
    Advance(u8, u8),
    RollAgain,
    Stop,
    /// Public chance event; never returned by legal_actions.
    Dice([u8; 4]),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CantStop;

fn advance(state: &mut CantStopState, column: u8) -> bool {
    let i = usize::from(column - 2);
    if state.claimed[i].is_some() || state.runners[i] == HEIGHTS[i] {
        return false;
    }
    if state.runners[i] == 0 {
        if state.runners.iter().filter(|&&p| p != 0).count() == 3 {
            return false;
        }
        state.runners[i] = state.progress[state.active.index()][i];
    }
    state.runners[i] += 1;
    true
}

fn next_turn(state: &mut CantStopState) {
    state.runners = [0; 11];
    state.active = CantStop::opponent(state.active).expect("two players");
    state.phase = Phase::Roll;
}

impl CantStop {
    fn advances(&self, state: &CantStopState) -> Vec<CantStopAction> {
        let [a, b, c, d] = state.dice;
        let mut moves = Vec::new();
        for (x, y) in [(a + b, c + d), (a + c, b + d), (a + d, b + c)] {
            // Either order matters when only one unused runner remains.
            for (first, second) in [(x, y), (y, x)] {
                let mut trial = state.clone();
                let one = advance(&mut trial, first);
                let two = advance(&mut trial, second);
                let action = match (one, two) {
                    (true, true) => CantStopAction::Advance(first.min(second), first.max(second)),
                    (true, false) => CantStopAction::Advance(first, 0),
                    (false, true) => CantStopAction::Advance(second, 0),
                    (false, false) => continue,
                };
                if !moves.contains(&action) {
                    moves.push(action);
                }
            }
        }
        moves.sort();
        moves
    }
}

impl Game for CantStop {
    type State = CantStopState;
    type Action = CantStopAction;
    type Observation<'a> = &'a CantStopState;
    type LegalActions<'a> = std::vec::IntoIter<CantStopAction>;
    fn player_count(&self) -> u8 {
        2
    }
    fn initial_state(&self) -> CantStopState {
        CantStopState {
            progress: [[0; 11]; 2],
            runners: [0; 11],
            claimed: [None; 11],
            dice: [0; 4],
            active: PlayerId::FIRST,
            phase: Phase::Roll,
            winner: None,
            last_bust: false,
        }
    }
    fn status(&self, state: &CantStopState) -> PositionStatus {
        match state.phase {
            Phase::Roll => PositionStatus::Chance,
            Phase::Finished => PositionStatus::Terminal,
            _ => PositionStatus::PlayerTurn(state.active),
        }
    }
    fn is_turn_boundary(&self, state: &Self::State) -> bool {
        state.runners == [0; 11]
    }
    fn legal_actions<'a>(&'a self, state: &'a CantStopState) -> Self::LegalActions<'a> {
        match state.phase {
            Phase::Choose => self.advances(state),
            Phase::Continue => vec![CantStopAction::RollAgain, CantStopAction::Stop],
            _ => vec![],
        }
        .into_iter()
    }
    fn sample_chance<R: RandomSource + ?Sized>(
        &self,
        state: &CantStopState,
        rng: &mut R,
    ) -> Result<CantStopAction, IllegalAction> {
        if state.phase != Phase::Roll {
            return Err(IllegalAction::new("not waiting for dice"));
        }
        Ok(CantStopAction::Dice(std::array::from_fn(|_| {
            rng.index(6).unwrap() as u8 + 1
        })))
    }
    fn apply_action(
        &self,
        state: &mut CantStopState,
        action: &CantStopAction,
    ) -> Result<(), IllegalAction> {
        if let CantStopAction::Dice(dice) = action {
            if state.phase != Phase::Roll || dice.iter().any(|d| !(1..=6).contains(d)) {
                return Err(IllegalAction::new("invalid dice event"));
            }
            state.dice = *dice;
            state.last_bust = false;
            state.phase = Phase::Choose;
            if self.advances(state).is_empty() {
                state.last_bust = true;
                next_turn(state);
            }
            return Ok(());
        }
        if !self.legal_actions(state).any(|a| a == *action) {
            return Err(IllegalAction::new("illegal Can't Stop action"));
        }
        match *action {
            CantStopAction::Advance(a, b) => {
                advance(state, a);
                if b != 0 {
                    advance(state, b);
                }
                state.phase = Phase::Continue;
            }
            CantStopAction::RollAgain => state.phase = Phase::Roll,
            CantStopAction::Stop => {
                for (i, &height) in HEIGHTS.iter().enumerate() {
                    if state.runners[i] == 0 {
                        continue;
                    }
                    state.progress[state.active.index()][i] = state.runners[i];
                    if state.runners[i] == height {
                        state.claimed[i] = Some(state.active);
                        state.progress[CantStop::opponent(state.active).unwrap().index()][i] = 0;
                    }
                }
                if state
                    .claimed
                    .iter()
                    .filter(|&&p| p == Some(state.active))
                    .count()
                    >= 3
                {
                    state.winner = Some(state.active);
                    state.runners = [0; 11];
                    state.phase = Phase::Finished;
                } else {
                    next_turn(state);
                }
            }
            CantStopAction::Dice(_) => unreachable!(),
        }
        Ok(())
    }
    fn observation<'a>(&'a self, state: &'a CantStopState, _: PlayerId) -> &'a CantStopState {
        state
    }
    fn terminal_utility(&self, state: &CantStopState, player: PlayerId) -> Option<f32> {
        state
            .winner
            .map(|winner| if winner == player { 1.0 } else { -1.0 })
    }
}
impl PerfectInformationGame for CantStop {}
impl TwoPlayerZeroSumGame for CantStop {}

impl meeple_bots_core::HeuristicGame for CantStop {
    fn heuristic_count(&self) -> u32 {
        1
    }
    fn heuristic_utility(
        &self,
        index: u32,
        state: &CantStopState,
        player: PlayerId,
    ) -> Option<f32> {
        if index != 0 {
            return None;
        }
        if let Some(value) = self.terminal_utility(state, player) {
            return Some(value);
        }
        let score = |p: PlayerId| {
            HEIGHTS
                .iter()
                .enumerate()
                .map(|(i, &h)| {
                    if state.claimed[i] == Some(p) {
                        1.0
                    } else if state.claimed[i].is_some() {
                        0.0
                    } else {
                        let secured = f32::from(state.progress[p.index()][i]);
                        let provisional = if p == state.active {
                            f32::from(state.runners[i]).max(secured) - secured
                        } else {
                            0.0
                        };
                        0.25 * (secured + 0.5 * provisional) / f32::from(h)
                    }
                })
                .sum::<f32>()
        };
        Some(((score(player) - score(Self::opponent(player)?)) / 3.0).clamp(-1.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use meeple_bots_core::Agent;
    use meeple_bots_random_agent::RandomAgent;
    use meeple_bots_simulation::{MatchConfig, SplitMix64, play_match_with_trace};

    #[test]
    fn turn_boundary_survives_initial_dice_but_not_roll_again() {
        let game = CantStop;
        let mut state = game.initial_state();
        game.apply_chance_outcome(&mut state, &CantStopAction::Dice([1, 2, 3, 4]))
            .unwrap();
        assert!(game.is_turn_boundary(&state));
        let advance = game.legal_actions(&state).next().unwrap();
        game.apply_action(&mut state, &advance).unwrap();
        assert!(!game.is_turn_boundary(&state));
        game.apply_action(&mut state, &CantStopAction::RollAgain)
            .unwrap();
        assert!(!game.is_turn_boundary(&state));
        game.apply_chance_outcome(&mut state, &CantStopAction::Dice([1, 2, 3, 4]))
            .unwrap();
        assert!(!game.is_turn_boundary(&state));
        let advance = game.legal_actions(&state).next().unwrap();
        game.apply_action(&mut state, &advance).unwrap();
        game.apply_action(&mut state, &CantStopAction::Stop)
            .unwrap();
        assert!(game.is_turn_boundary(&state));
    }

    #[test]
    fn dice_are_public_events_and_invalid_actions_are_atomic() {
        let g = CantStop;
        let mut s = g.initial_state();
        let before = s.clone();
        assert_eq!(g.status(&s), PositionStatus::Chance);
        assert_eq!(g.legal_actions(&s).count(), 0);
        for a in [CantStopAction::Stop, CantStopAction::Dice([0, 1, 1, 1])] {
            assert!(g.apply_action(&mut s, &a).is_err());
            assert_eq!(s, before);
        }
        g.apply_action(&mut s, &CantStopAction::Dice([1, 2, 3, 4]))
            .unwrap();
        assert_eq!(
            g.legal_actions(&s).collect::<Vec<_>>(),
            vec![
                CantStopAction::Advance(3, 7),
                CantStopAction::Advance(4, 6),
                CantStopAction::Advance(5, 5)
            ]
        );
        let before = s.clone();
        assert!(
            g.apply_action(&mut s, &CantStopAction::Advance(2, 12))
                .is_err()
        );
        assert_eq!(s, before);
    }
    #[test]
    fn single_runner_slot_allows_either_column_but_not_skipping_a_usable_die() {
        let g = CantStop;
        let mut s = g.initial_state();
        s.runners[0] = 1;
        s.runners[1] = 1;
        g.apply_action(&mut s, &CantStopAction::Dice([2, 3, 4, 5]))
            .unwrap();
        let a: Vec<_> = g.legal_actions(&s).collect();
        assert!(
            a.contains(&CantStopAction::Advance(5, 0))
                && a.contains(&CantStopAction::Advance(9, 0))
        );
        assert!(a.contains(&CantStopAction::Advance(7, 7)));
        assert!(!a.contains(&CantStopAction::Advance(7, 0)));
    }
    #[test]
    fn double_at_summit_advances_once_and_requires_banking() {
        let g = CantStop;
        let mut s = g.initial_state();
        s.progress[0][0] = 2;
        g.apply_action(&mut s, &CantStopAction::Dice([1, 1, 1, 1]))
            .unwrap();
        assert_eq!(
            g.legal_actions(&s).collect::<Vec<_>>(),
            vec![CantStopAction::Advance(2, 0)]
        );
        g.apply_action(&mut s, &CantStopAction::Advance(2, 0))
            .unwrap();
        assert_eq!(s.claimed[0], None);
        g.apply_action(&mut s, &CantStopAction::Stop).unwrap();
        assert_eq!(s.claimed[0], Some(PlayerId::FIRST));
    }
    #[test]
    fn bust_discards_only_temporary_progress_and_passes_turn() {
        let g = CantStop;
        let mut s = g.initial_state();
        assert!(g.is_turn_boundary(&s));
        s.progress[0][0] = 1;
        s.runners[0] = 2;
        s.runners[1] = 1;
        s.runners[2] = 1;
        assert!(!g.is_turn_boundary(&s));
        g.apply_action(&mut s, &CantStopAction::Dice([6, 6, 6, 6]))
            .unwrap();
        assert!(s.last_bust);
        assert!(g.is_turn_boundary(&s));
        assert_eq!(s.progress[0][0], 1);
        assert_eq!(s.runners, [0; 11]);
        assert_eq!(s.active, PlayerId::SECOND);
        assert_eq!(s.phase, Phase::Roll);
    }
    #[test]
    fn closed_columns_cannot_be_reused_and_three_claims_win_only_when_banked() {
        let g = CantStop;
        let mut s = g.initial_state();
        s.claimed[0] = Some(PlayerId::FIRST);
        s.claimed[1] = Some(PlayerId::FIRST);
        s.runners[2] = HEIGHTS[2];
        s.progress[1][2] = 4;
        s.phase = Phase::Continue;
        assert!(g.terminal_utility(&s, PlayerId::FIRST).is_none());
        g.apply_action(&mut s, &CantStopAction::Stop).unwrap();
        assert_eq!(s.progress[1][2], 0);
        assert_eq!(g.status(&s), PositionStatus::Terminal);
        assert_eq!(g.terminal_utility(&s, PlayerId::FIRST), Some(1.0));
        assert!(g.apply_action(&mut s, &CantStopAction::RollAgain).is_err());
    }
    #[test]
    fn seeded_random_matches_replay_dice_and_decisions() {
        for seed in 0..8 {
            let g = CantStop;
            let config = MatchConfig {
                seed,
                ..MatchConfig::default()
            };
            let trace =
                play_match_with_trace(&g, &mut RandomAgent, &mut RandomAgent, config).unwrap();
            let again =
                play_match_with_trace(&g, &mut RandomAgent, &mut RandomAgent, config).unwrap();
            assert_eq!(trace.chance_events, again.chance_events);
            assert_eq!(
                trace.actions.iter().map(|a| a.action).collect::<Vec<_>>(),
                again.actions.iter().map(|a| a.action).collect::<Vec<_>>()
            );
            let mut s = g.initial_state();
            let mut events = trace.chance_events.iter().peekable();
            for ply in 0..=trace.actions.len() {
                while events.peek().is_some_and(|e| e.after_ply == ply) {
                    g.apply_action(&mut s, &events.next().unwrap().event)
                        .unwrap();
                }
                if let Some(a) = trace.actions.get(ply) {
                    assert_eq!(g.status(&s), PositionStatus::PlayerTurn(a.player));
                    g.apply_action(&mut s, &a.action).unwrap();
                }
            }
            assert_eq!(g.status(&s), PositionStatus::Terminal);
        }
    }
    #[test]
    fn agent_rng_consumption_does_not_change_real_dice() {
        struct Burner;
        impl Agent<CantStop> for Burner {
            fn select_action<R: RandomSource + ?Sized>(
                &mut self,
                d: meeple_bots_core::DecisionContext<'_, CantStop>,
                rng: &mut R,
            ) -> Result<CantStopAction, meeple_bots_core::AgentError> {
                for _ in 0..100 {
                    rng.next_u64();
                }
                Ok(d.legal_actions().last().unwrap())
            }
        }
        struct Last;
        impl Agent<CantStop> for Last {
            fn select_action<R: RandomSource + ?Sized>(
                &mut self,
                d: meeple_bots_core::DecisionContext<'_, CantStop>,
                _: &mut R,
            ) -> Result<CantStopAction, meeple_bots_core::AgentError> {
                Ok(d.legal_actions().last().unwrap())
            }
        }
        let a = play_match_with_trace(&CantStop, &mut Burner, &mut Last, MatchConfig::default())
            .unwrap();
        let b =
            play_match_with_trace(&CantStop, &mut Last, &mut Last, MatchConfig::default()).unwrap();
        assert_eq!(a.chance_events, b.chance_events);
        let mut rng = SplitMix64::new(10);
        let event = CantStop
            .sample_chance(&CantStop.initial_state(), &mut rng)
            .unwrap();
        assert!(matches!(event, CantStopAction::Dice(_)));
    }
}
