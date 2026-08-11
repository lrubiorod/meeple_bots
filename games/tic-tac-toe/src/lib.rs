//! Allocation-free, strongly typed tic-tac-toe implementation.

use meeple_bots_core::{
    DeterministicGame, Game, IllegalAction, PerfectInformationGame, PlayerId, PositionStatus,
    TwoPlayerZeroSumGame,
};

const WINNING_LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
];

#[derive(Clone, Copy, Debug, Default)]
pub struct TicTacToe;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TicTacToeState {
    board: [Option<PlayerId>; 9],
    next_player: PlayerId,
    moves: u8,
}

impl TicTacToeState {
    pub fn board(&self) -> &[Option<PlayerId>; 9] {
        &self.board
    }

    pub const fn next_player(&self) -> PlayerId {
        self.next_player
    }

    pub const fn moves(&self) -> u8 {
        self.moves
    }

    pub fn winner(&self) -> Option<PlayerId> {
        WINNING_LINES.iter().find_map(|line| {
            let player = self.board[line[0]]?;
            (self.board[line[1]] == Some(player) && self.board[line[2]] == Some(player))
                .then_some(player)
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TicTacToeAction(u8);

impl TicTacToeAction {
    pub const fn new(row: u8, column: u8) -> Option<Self> {
        if row < 3 && column < 3 {
            Some(Self(row * 3 + column))
        } else {
            None
        }
    }

    pub const fn from_index(index: u8) -> Option<Self> {
        if index < 9 { Some(Self(index)) } else { None }
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn row(self) -> u8 {
        self.0 / 3
    }

    pub const fn column(self) -> u8 {
        self.0 % 3
    }
}

pub struct LegalActions<'a> {
    state: &'a TicTacToeState,
    next: u8,
    enabled: bool,
}

impl Iterator for LegalActions<'_> {
    type Item = TicTacToeAction;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.enabled {
            return None;
        }

        while self.next < 9 {
            let index = self.next;
            self.next += 1;
            if self.state.board[index as usize].is_none() {
                return Some(TicTacToeAction(index));
            }
        }
        None
    }
}

impl Game for TicTacToe {
    type State = TicTacToeState;
    type Action = TicTacToeAction;
    type Observation<'a> = &'a TicTacToeState;
    type LegalActions<'a> = LegalActions<'a>;

    fn player_count(&self) -> u8 {
        2
    }

    fn initial_state(&self) -> Self::State {
        TicTacToeState {
            board: [None; 9],
            next_player: PlayerId::FIRST,
            moves: 0,
        }
    }

    fn status(&self, state: &Self::State) -> PositionStatus {
        if state.winner().is_some() || state.moves == 9 {
            PositionStatus::Terminal
        } else {
            PositionStatus::PlayerTurn(state.next_player)
        }
    }

    fn legal_actions<'a>(&'a self, state: &'a Self::State) -> Self::LegalActions<'a> {
        LegalActions {
            state,
            next: 0,
            enabled: !matches!(self.status(state), PositionStatus::Terminal),
        }
    }

    fn apply_action(
        &self,
        state: &mut Self::State,
        action: &Self::Action,
    ) -> Result<(), IllegalAction> {
        if matches!(self.status(state), PositionStatus::Terminal) {
            return Err(IllegalAction::new("the game is already terminal"));
        }

        let cell = state
            .board
            .get_mut(action.index())
            .ok_or_else(|| IllegalAction::new("cell is outside the board"))?;
        if cell.is_some() {
            return Err(IllegalAction::new("cell is already occupied"));
        }

        *cell = Some(state.next_player);
        state.moves += 1;
        state.next_player =
            <Self as TwoPlayerZeroSumGame>::opponent(state.next_player).expect("valid player");
        Ok(())
    }

    fn observation<'a>(
        &'a self,
        state: &'a Self::State,
        _player: PlayerId,
    ) -> Self::Observation<'a> {
        state
    }

    fn terminal_utility(&self, state: &Self::State, player: PlayerId) -> Option<f32> {
        if !matches!(self.status(state), PositionStatus::Terminal) || player.index() >= 2 {
            return None;
        }

        Some(match state.winner() {
            Some(winner) if winner == player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        })
    }
}

impl DeterministicGame for TicTacToe {}
impl PerfectInformationGame for TicTacToe {}
impl TwoPlayerZeroSumGame for TicTacToe {}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(index: u8) -> TicTacToeAction {
        TicTacToeAction::from_index(index).unwrap()
    }

    #[test]
    fn initial_state_has_nine_actions() {
        let game = TicTacToe;
        let state = game.initial_state();

        assert_eq!(game.legal_actions(&state).count(), 9);
        assert_eq!(
            game.status(&state),
            PositionStatus::PlayerTurn(PlayerId::FIRST)
        );
    }

    #[test]
    fn rejects_occupied_cells() {
        let game = TicTacToe;
        let mut state = game.initial_state();
        game.apply_action(&mut state, &action(0)).unwrap();

        assert!(game.apply_action(&mut state, &action(0)).is_err());
    }

    #[test]
    fn reports_win_from_both_perspectives() {
        let game = TicTacToe;
        let mut state = game.initial_state();
        for index in [0, 3, 1, 4, 2] {
            game.apply_action(&mut state, &action(index)).unwrap();
        }

        assert_eq!(state.winner(), Some(PlayerId::FIRST));
        assert_eq!(game.terminal_utility(&state, PlayerId::FIRST), Some(1.0));
        assert_eq!(game.terminal_utility(&state, PlayerId::SECOND), Some(-1.0));
        assert_eq!(game.legal_actions(&state).count(), 0);
    }

    #[test]
    fn reports_draw() {
        let game = TicTacToe;
        let mut state = game.initial_state();
        for index in [0, 1, 2, 4, 3, 5, 7, 6, 8] {
            game.apply_action(&mut state, &action(index)).unwrap();
        }

        assert_eq!(state.winner(), None);
        assert_eq!(game.terminal_utility(&state, PlayerId::FIRST), Some(0.0));
    }
}
