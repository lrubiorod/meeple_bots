//! Allocation-free implementation of the standard 6x7 Connect Four game.

use meeple_bots_core::{
    DeterministicGame, Game, IllegalAction, PerfectInformationGame, PlayerId, PositionStatus,
    TwoPlayerZeroSumGame,
};

pub const ROWS: usize = 6;
pub const COLUMNS: usize = 7;
const CELL_COUNT: usize = ROWS * COLUMNS;
const CONNECTED: usize = 4;

#[derive(Clone, Copy, Debug, Default)]
pub struct ConnectFour;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConnectFourState {
    board: [Option<PlayerId>; CELL_COUNT],
    next_player: PlayerId,
    moves: u8,
}

impl ConnectFourState {
    pub fn board(&self) -> &[Option<PlayerId>; CELL_COUNT] {
        &self.board
    }

    pub const fn next_player(&self) -> PlayerId {
        self.next_player
    }

    pub const fn moves(&self) -> u8 {
        self.moves
    }

    pub fn winner(&self) -> Option<PlayerId> {
        for row in 0..ROWS {
            for column in 0..COLUMNS {
                let Some(player) = self.board[cell_index(row, column)] else {
                    continue;
                };
                for (row_step, column_step) in [(0, 1), (1, 0), (1, 1), (1, -1)] {
                    if line_belongs_to(&self.board, player, row, column, row_step, column_step) {
                        return Some(player);
                    }
                }
            }
        }
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ConnectFourAction(u8);

impl ConnectFourAction {
    pub const fn new(column: u8) -> Option<Self> {
        if column < COLUMNS as u8 {
            Some(Self(column))
        } else {
            None
        }
    }

    pub const fn column(self) -> u8 {
        self.0
    }
}

pub struct LegalActions<'a> {
    state: &'a ConnectFourState,
    next_column: u8,
    enabled: bool,
}

impl Iterator for LegalActions<'_> {
    type Item = ConnectFourAction;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.enabled {
            return None;
        }

        while usize::from(self.next_column) < COLUMNS {
            let column = self.next_column;
            self.next_column += 1;
            if self.state.board[cell_index(0, usize::from(column))].is_none() {
                return Some(ConnectFourAction(column));
            }
        }
        None
    }
}

impl Game for ConnectFour {
    type State = ConnectFourState;
    type Action = ConnectFourAction;
    type Observation<'a> = &'a ConnectFourState;
    type LegalActions<'a> = LegalActions<'a>;

    fn player_count(&self) -> u8 {
        2
    }

    fn initial_state(&self) -> Self::State {
        ConnectFourState {
            board: [None; CELL_COUNT],
            next_player: PlayerId::FIRST,
            moves: 0,
        }
    }

    fn status(&self, state: &Self::State) -> PositionStatus {
        if state.winner().is_some() || usize::from(state.moves) == CELL_COUNT {
            PositionStatus::Terminal
        } else {
            PositionStatus::PlayerTurn(state.next_player)
        }
    }

    fn legal_actions<'a>(&'a self, state: &'a Self::State) -> Self::LegalActions<'a> {
        LegalActions {
            state,
            next_column: 0,
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

        let column = usize::from(action.column());
        let row = (0..ROWS)
            .rev()
            .find(|row| state.board[cell_index(*row, column)].is_none())
            .ok_or_else(|| IllegalAction::new("column is full"))?;
        state.board[cell_index(row, column)] = Some(state.next_player);
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

impl DeterministicGame for ConnectFour {}
impl PerfectInformationGame for ConnectFour {}
impl TwoPlayerZeroSumGame for ConnectFour {}

const fn cell_index(row: usize, column: usize) -> usize {
    row * COLUMNS + column
}

fn line_belongs_to(
    board: &[Option<PlayerId>; CELL_COUNT],
    player: PlayerId,
    row: usize,
    column: usize,
    row_step: isize,
    column_step: isize,
) -> bool {
    (1..CONNECTED).all(|distance| {
        let checked_row = row as isize + row_step * distance as isize;
        let checked_column = column as isize + column_step * distance as isize;
        checked_row >= 0
            && checked_row < ROWS as isize
            && checked_column >= 0
            && checked_column < COLUMNS as isize
            && board[cell_index(checked_row as usize, checked_column as usize)] == Some(player)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(column: u8) -> ConnectFourAction {
        ConnectFourAction::new(column).unwrap()
    }

    fn play(columns: &[u8]) -> ConnectFourState {
        let game = ConnectFour;
        let mut state = game.initial_state();
        for column in columns {
            game.apply_action(&mut state, &action(*column)).unwrap();
        }
        state
    }

    #[test]
    fn initial_state_has_seven_actions() {
        let game = ConnectFour;
        let state = game.initial_state();

        assert_eq!(game.legal_actions(&state).count(), COLUMNS);
        assert_eq!(
            game.status(&state),
            PositionStatus::PlayerTurn(PlayerId::FIRST)
        );
    }

    #[test]
    fn pieces_fall_and_full_columns_are_rejected() {
        let game = ConnectFour;
        let mut state = game.initial_state();
        for _ in 0..ROWS {
            game.apply_action(&mut state, &action(3)).unwrap();
        }

        assert!(state.board[cell_index(0, 3)].is_some());
        assert!(
            !game
                .legal_actions(&state)
                .any(|candidate| candidate == action(3))
        );
        assert!(game.apply_action(&mut state, &action(3)).is_err());
    }

    #[test]
    fn reports_horizontal_win() {
        let game = ConnectFour;
        let state = play(&[0, 0, 1, 1, 2, 2, 3]);

        assert_eq!(state.winner(), Some(PlayerId::FIRST));
        assert_eq!(game.terminal_utility(&state, PlayerId::FIRST), Some(1.0));
        assert_eq!(game.terminal_utility(&state, PlayerId::SECOND), Some(-1.0));
    }

    #[test]
    fn reports_vertical_win() {
        let state = play(&[0, 1, 0, 1, 0, 1, 0]);

        assert_eq!(state.winner(), Some(PlayerId::FIRST));
    }

    #[test]
    fn reports_both_diagonal_directions() {
        let mut state = ConnectFour.initial_state();
        for (row, column) in [(5, 0), (4, 1), (3, 2), (2, 3)] {
            state.board[cell_index(row, column)] = Some(PlayerId::FIRST);
        }
        assert_eq!(state.winner(), Some(PlayerId::FIRST));

        let mut state = ConnectFour.initial_state();
        for (row, column) in [(2, 0), (3, 1), (4, 2), (5, 3)] {
            state.board[cell_index(row, column)] = Some(PlayerId::SECOND);
        }
        assert_eq!(state.winner(), Some(PlayerId::SECOND));
    }
}
