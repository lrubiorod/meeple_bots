//! Connect6: each stone is a decision, including the two decisions of a normal turn.
use meeple_bots_core::{
    DeterministicGame, Game, HeuristicGame, IllegalAction, PerfectInformationGame, PlayerId,
    PositionStatus, TwoPlayerZeroSumGame,
};

pub const DEFAULT_BOARD_SIZE: usize = 19;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Connect6 {
    board_size: usize,
}
impl Default for Connect6 {
    fn default() -> Self {
        Self::new(DEFAULT_BOARD_SIZE).expect("valid default")
    }
}
impl Connect6 {
    pub fn new(board_size: usize) -> Result<Self, &'static str> {
        if board_size < 6 {
            return Err("board_size must be at least 6");
        }
        // Match/search counters use u32; board indexing must also fit Rust allocations.
        if board_size
            .checked_mul(board_size)
            .is_none_or(|n| n > u32::MAX as usize || n > isize::MAX as usize)
        {
            return Err("board_size squared must fit u32 match counters and allocation indexing");
        }
        Ok(Self { board_size })
    }
    pub const fn board_size(&self) -> usize {
        self.board_size
    }
    fn line(&self, state: &Connect6State, cell: usize, player: u8) -> bool {
        let n = self.board_size as isize;
        let row = (cell / self.board_size) as isize;
        let col = (cell % self.board_size) as isize;
        [(0, 1), (1, 0), (1, 1), (1, -1)]
            .into_iter()
            .any(|(dr, dc)| {
                let mut count = 1;
                for sign in [-1, 1] {
                    let (mut r, mut c) = (row + sign * dr, col + sign * dc);
                    while r >= 0
                        && c >= 0
                        && r < n
                        && c < n
                        && state.board[(r * n + c) as usize] == player
                    {
                        count += 1;
                        if count >= 6 {
                            return true;
                        }
                        r += sign * dr;
                        c += sign * dc;
                    }
                }
                false
            })
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Connect6State {
    // 0 empty, 1 black, 2 white. No last-move history: placement order may transpose.
    board: Box<[u8]>,
    current_player: PlayerId,
    placements_remaining: u8,
    winner: Option<PlayerId>,
}
impl Connect6State {
    pub fn board(&self) -> &[u8] {
        &self.board
    }
    pub fn current_player(&self) -> PlayerId {
        self.current_player
    }
    pub fn placements_remaining(&self) -> u8 {
        self.placements_remaining
    }
    pub fn winner(&self) -> Option<PlayerId> {
        self.winner
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Connect6Action {
    Place(usize),
}
impl Game for Connect6 {
    type State = Connect6State;
    type Action = Connect6Action;
    type Observation<'a> = &'a Connect6State;
    type LegalActions<'a> = std::vec::IntoIter<Connect6Action>;
    fn maximum_decision_horizon(&self) -> Option<u32> {
        Some((self.board_size * self.board_size) as u32)
    }

    fn player_count(&self) -> u8 {
        2
    }
    fn initial_state(&self) -> Self::State {
        Connect6State {
            board: vec![0; self.board_size * self.board_size].into_boxed_slice(),
            current_player: PlayerId::FIRST,
            placements_remaining: 1,
            winner: None,
        }
    }
    fn status(&self, state: &Self::State) -> PositionStatus {
        if state.winner.is_some() || !state.board.contains(&0) {
            PositionStatus::Terminal
        } else {
            PositionStatus::PlayerTurn(state.current_player)
        }
    }
    fn is_turn_boundary(&self, state: &Self::State) -> bool {
        state.placements_remaining == 2 || state.board.iter().all(|&cell| cell == 0)
    }
    fn legal_actions<'a>(&'a self, state: &'a Self::State) -> Self::LegalActions<'a> {
        if matches!(self.status(state), PositionStatus::Terminal) {
            return Vec::new().into_iter();
        }
        state
            .board
            .iter()
            .enumerate()
            .filter(|(_, v)| **v == 0)
            .map(|(i, _)| Connect6Action::Place(i))
            .collect::<Vec<_>>()
            .into_iter()
    }
    fn apply_action(
        &self,
        state: &mut Self::State,
        action: &Self::Action,
    ) -> Result<(), IllegalAction> {
        let Connect6Action::Place(cell) = *action;
        if state.board.len() != self.board_size * self.board_size {
            return Err(IllegalAction::new(
                "state belongs to a different board size",
            ));
        }
        if matches!(self.status(state), PositionStatus::Terminal) {
            return Err(IllegalAction::new("game is terminal"));
        }
        if state.board.get(cell) != Some(&0) {
            return Err(IllegalAction::new(
                "cell must be empty and inside the board",
            ));
        }
        let player = state.current_player;
        state.board[cell] = player.index() as u8 + 1;
        state.placements_remaining -= 1;
        if self.line(state, cell, player.index() as u8 + 1) {
            state.winner = Some(player);
        }
        if !matches!(self.status(state), PositionStatus::Terminal)
            && state.placements_remaining == 0
        {
            state.current_player = Self::opponent(player).expect("two players");
            state.placements_remaining = 2;
        }
        Ok(())
    }
    fn observation<'a>(&'a self, state: &'a Self::State, _: PlayerId) -> Self::Observation<'a> {
        state
    }
    fn terminal_utility(&self, state: &Self::State, player: PlayerId) -> Option<f32> {
        if player.index() >= 2 || !matches!(self.status(state), PositionStatus::Terminal) {
            return None;
        }
        Some(match state.winner {
            Some(w) if w == player => 1.,
            Some(_) => -1.,
            None => 0.,
        })
    }
}
impl PerfectInformationGame for Connect6 {}
impl DeterministicGame for Connect6 {}
impl TwoPlayerZeroSumGame for Connect6 {}
impl HeuristicGame for Connect6 {
    fn heuristic_count(&self) -> u32 {
        0
    }
    fn heuristic_utility(&self, _: u32, _: &Connect6State, _: PlayerId) -> Option<f32> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{DefaultHasher, Hash, Hasher};
    #[test]
    fn sizes_and_turns() {
        assert_eq!(Connect6::default().board_size(), 19);
        assert!(Connect6::new(5).is_err());
        assert!(Connect6::new(usize::MAX).is_err());
        for n in [6, 9, 11, 19] {
            let g = Connect6::new(n).unwrap();
            let mut s = g.initial_state();
            assert_eq!(g.legal_actions(&s).count(), n * n);
            assert_eq!(s.current_player(), PlayerId::FIRST);
            assert_eq!(s.placements_remaining(), 1);
            assert!(g.is_turn_boundary(&s));
            for (i, p, left) in [
                (0, PlayerId::SECOND, 2),
                (1, PlayerId::SECOND, 1),
                (2, PlayerId::FIRST, 2),
            ] {
                g.apply_action(&mut s, &Connect6Action::Place(i)).unwrap();
                assert_eq!(g.status(&s), PositionStatus::PlayerTurn(p));
                assert_eq!(s.placements_remaining(), left);
                assert_eq!(g.is_turn_boundary(&s), left == 2);
                assert_eq!(g.legal_actions(&s).count(), n * n - i - 1);
            }
            let before = s.clone();
            assert!(g.apply_action(&mut s, &Connect6Action::Place(0)).is_err());
            assert!(
                g.apply_action(&mut s, &Connect6Action::Place(n * n))
                    .is_err()
            );
            assert_eq!(s, before);
        }
    }
    #[test]
    fn lines_and_immediate_win() {
        let g = Connect6::new(9).unwrap();
        for (start, step) in [(9, 1), (1, 9), (0, 10), (8, 8)] {
            let mut s = g.initial_state();
            s.placements_remaining = 2;
            for i in 0..5 {
                s.board[start + i * step] = 1;
            }
            g.apply_action(&mut s, &Connect6Action::Place(start + 5 * step))
                .unwrap();
            assert_eq!(s.winner(), Some(PlayerId::FIRST));
            assert_eq!(s.placements_remaining(), 1);
            assert_eq!(g.terminal_utility(&s, PlayerId::FIRST), Some(1.));
            assert_eq!(g.terminal_utility(&s, PlayerId::SECOND), Some(-1.));
            assert_eq!(g.legal_actions(&s).count(), 0);
            assert!(g.apply_action(&mut s, &Connect6Action::Place(80)).is_err());
        }
        let mut s = g.initial_state();
        for i in [0, 1, 2, 4, 5, 6] {
            s.board[i] = 1;
        }
        g.apply_action(&mut s, &Connect6Action::Place(3)).unwrap();
        assert!(s.winner().is_some());
    }
    #[test]
    fn full_board_draw() {
        let g = Connect6::new(6).unwrap();
        let mut s = g.initial_state();
        for r in 0..6 {
            for c in 0..6 {
                s.board[r * 6 + c] = 1 + ((r + c / 2) % 2) as u8;
            }
        }
        for i in 0..36 {
            assert!(!g.line(&s, i, s.board[i]));
        }
        s.board[35] = 0;
        s.current_player = PlayerId::SECOND;
        s.placements_remaining = 2;
        g.apply_action(&mut s, &Connect6Action::Place(35)).unwrap();
        assert_eq!(g.status(&s), PositionStatus::Terminal);
        assert_eq!(s.winner(), None);
        assert_eq!(g.terminal_utility(&s, PlayerId::FIRST), Some(0.));
    }
    #[test]
    fn two_placements_commute_in_equality_and_hash() {
        let g = Connect6::new(9).unwrap();
        let mut a = g.initial_state();
        g.apply_action(&mut a, &Connect6Action::Place(0)).unwrap();
        let mut b = a.clone();
        for i in [1, 2] {
            g.apply_action(&mut a, &Connect6Action::Place(i)).unwrap();
        }
        for i in [2, 1] {
            g.apply_action(&mut b, &Connect6Action::Place(i)).unwrap();
        }
        assert_eq!(a, b);
        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }
}
