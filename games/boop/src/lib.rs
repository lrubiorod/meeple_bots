//! Rules for the standard two-player game of boop.

use meeple_bots_core::{
    DeterministicGame, Game, HeuristicGame, IllegalAction, PerfectInformationGame, PlayerId,
    PositionStatus, TwoPlayerZeroSumGame,
};

pub const ROWS: usize = 6;
pub const COLUMNS: usize = 6;
const CELL_COUNT: usize = ROWS * COLUMNS;
const PIECES_PER_PLAYER: u8 = 8;
const LINE_COUNT: usize = 80;

#[derive(Clone, Copy, Debug, Default)]
pub struct Boop;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PieceKind {
    Kitten,
    Cat,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Piece {
    owner: PlayerId,
    kind: PieceKind,
}

impl Piece {
    pub const fn new(owner: PlayerId, kind: PieceKind) -> Self {
        Self { owner, kind }
    }

    pub const fn owner(self) -> PlayerId {
        self.owner
    }

    pub const fn kind(self) -> PieceKind {
        self.kind
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Position(u8);

impl Position {
    pub const fn new(row: u8, column: u8) -> Option<Self> {
        if row < ROWS as u8 && column < COLUMNS as u8 {
            Some(Self(row * COLUMNS as u8 + column))
        } else {
            None
        }
    }

    pub const fn row(self) -> u8 {
        self.0 / COLUMNS as u8
    }

    pub const fn column(self) -> u8 {
        self.0 % COLUMNS as u8
    }

    const fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraduateLine {
    positions: [Position; 3],
}

impl GraduateLine {
    pub fn new(mut positions: [Position; 3]) -> Option<Self> {
        positions.sort_by_key(|position| position.0);
        is_straight_consecutive_line(positions).then_some(Self { positions })
    }

    pub const fn positions(self) -> [Position; 3] {
        self.positions
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Resolution {
    None,
    Graduate(GraduateLine),
    Recover(Position),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BoopAction {
    piece: PieceKind,
    position: Position,
    resolution: Resolution,
}

impl BoopAction {
    pub const fn new(piece: PieceKind, position: Position, resolution: Resolution) -> Self {
        Self {
            piece,
            position,
            resolution,
        }
    }

    pub const fn piece(self) -> PieceKind {
        self.piece
    }

    pub const fn position(self) -> Position {
        self.position
    }

    pub const fn resolution(self) -> Resolution {
        self.resolution
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Pool {
    kittens: u8,
    cats: u8,
}

impl Pool {
    pub const fn kittens(self) -> u8 {
        self.kittens
    }

    pub const fn cats(self) -> u8 {
        self.cats
    }

    fn count(self, kind: PieceKind) -> u8 {
        match kind {
            PieceKind::Kitten => self.kittens,
            PieceKind::Cat => self.cats,
        }
    }

    fn add(&mut self, kind: PieceKind) {
        match kind {
            PieceKind::Kitten => self.kittens += 1,
            PieceKind::Cat => self.cats += 1,
        }
    }

    fn remove(&mut self, kind: PieceKind) {
        match kind {
            PieceKind::Kitten => self.kittens -= 1,
            PieceKind::Cat => self.cats -= 1,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoopState {
    board: [Option<Piece>; CELL_COUNT],
    pools: [Pool; 2],
    next_player: PlayerId,
    winner: Option<PlayerId>,
    moves: u32,
}

impl BoopState {
    pub fn board(&self) -> &[Option<Piece>; CELL_COUNT] {
        &self.board
    }

    pub const fn pools(&self) -> &[Pool; 2] {
        &self.pools
    }

    pub const fn next_player(&self) -> PlayerId {
        self.next_player
    }

    pub const fn winner(&self) -> Option<PlayerId> {
        self.winner
    }

    pub const fn moves(&self) -> u32 {
        self.moves
    }
}

impl Game for Boop {
    type State = BoopState;
    type Action = BoopAction;
    type Observation<'a> = &'a BoopState;
    type LegalActions<'a> = std::vec::IntoIter<BoopAction>;

    fn player_count(&self) -> u8 {
        2
    }

    fn initial_state(&self) -> Self::State {
        BoopState {
            board: [None; CELL_COUNT],
            pools: [
                Pool {
                    kittens: PIECES_PER_PLAYER,
                    cats: 0,
                },
                Pool {
                    kittens: PIECES_PER_PLAYER,
                    cats: 0,
                },
            ],
            next_player: PlayerId::FIRST,
            winner: None,
            moves: 0,
        }
    }

    fn status(&self, state: &Self::State) -> PositionStatus {
        if state.winner.is_some() {
            PositionStatus::Terminal
        } else {
            PositionStatus::PlayerTurn(state.next_player)
        }
    }

    fn legal_actions<'a>(&'a self, state: &'a Self::State) -> Self::LegalActions<'a> {
        if state.winner.is_some() {
            return Vec::new().into_iter();
        }

        let player = state.next_player;
        let mut actions = Vec::new();
        for kind in [PieceKind::Kitten, PieceKind::Cat] {
            if state.pools[player.index()].count(kind) == 0 {
                continue;
            }
            for index in 0..CELL_COUNT {
                if state.board[index].is_some() {
                    continue;
                }
                let position = Position(index as u8);
                let mut after_boop = state.clone();
                place_and_boop(&mut after_boop, player, kind, position);
                let winner = winner_after_boop(&after_boop, player);
                if winner.is_some() {
                    actions.push(BoopAction::new(kind, position, Resolution::None));
                    continue;
                }

                let resolutions = available_resolutions(&after_boop, player);
                if resolutions.is_empty() {
                    actions.push(BoopAction::new(kind, position, Resolution::None));
                } else {
                    actions.extend(
                        resolutions
                            .into_iter()
                            .map(|resolution| BoopAction::new(kind, position, resolution)),
                    );
                }
            }
        }
        actions.into_iter()
    }

    fn apply_action(
        &self,
        state: &mut Self::State,
        action: &Self::Action,
    ) -> Result<(), IllegalAction> {
        if state.winner.is_some() {
            return Err(IllegalAction::new("the game is already terminal"));
        }
        let player = state.next_player;
        if state.pools[player.index()].count(action.piece) == 0
            || state.board[action.position.index()].is_some()
        {
            return Err(IllegalAction::new(
                "action is not legal in the current position",
            ));
        }

        let mut next_state = state.clone();
        place_and_boop(&mut next_state, player, action.piece, action.position);
        next_state.winner = winner_after_boop(&next_state, player);
        let resolution_is_legal = if next_state.winner.is_some() {
            action.resolution == Resolution::None
        } else {
            let resolutions = available_resolutions(&next_state, player);
            if resolutions.is_empty() {
                action.resolution == Resolution::None
            } else {
                resolutions.contains(&action.resolution)
            }
        };
        if !resolution_is_legal {
            return Err(IllegalAction::new(
                "action has an invalid end-of-turn resolution",
            ));
        }

        if next_state.winner.is_none() {
            apply_resolution(&mut next_state, player, action.resolution);
            next_state.next_player =
                <Self as TwoPlayerZeroSumGame>::opponent(player).expect("valid player");
        }
        next_state.moves += 1;
        *state = next_state;
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
        let winner = state.winner?;
        if player.index() >= 2 {
            return None;
        }
        Some(if winner == player { 1.0 } else { -1.0 })
    }
}

impl HeuristicGame for Boop {
    fn heuristic_count(&self) -> u32 {
        2
    }

    fn heuristic_utility(&self, index: u32, state: &Self::State, player: PlayerId) -> Option<f32> {
        match index {
            0 => {
                let opponent = <Self as TwoPlayerZeroSumGame>::opponent(player)?;
                let player_cats = pieces_on_board(state, player, Some(PieceKind::Cat));
                let opponent_cats = pieces_on_board(state, opponent, Some(PieceKind::Cat));
                Some((f32::from(player_cats) - f32::from(opponent_cats)) * 0.1)
            }
            1 => strategic_heuristic_utility(state, player),
            _ => None,
        }
    }
}

impl DeterministicGame for Boop {}
impl PerfectInformationGame for Boop {}
impl TwoPlayerZeroSumGame for Boop {}

fn place_and_boop(state: &mut BoopState, player: PlayerId, kind: PieceKind, placed: Position) {
    state.pools[player.index()].remove(kind);
    state.board[placed.index()] = Some(Piece::new(player, kind));
    let before_boops = state.board;

    for row_delta in -1..=1 {
        for column_delta in -1..=1 {
            if row_delta == 0 && column_delta == 0 {
                continue;
            }
            let adjacent_row = i16::from(placed.row()) + row_delta;
            let adjacent_column = i16::from(placed.column()) + column_delta;
            let Some(adjacent) = checked_position(adjacent_row, adjacent_column) else {
                continue;
            };
            let Some(target) = before_boops[adjacent.index()] else {
                continue;
            };
            if kind == PieceKind::Kitten && target.kind == PieceKind::Cat {
                continue;
            }

            let destination =
                checked_position(adjacent_row + row_delta, adjacent_column + column_delta);
            match destination {
                None => {
                    state.board[adjacent.index()] = None;
                    state.pools[target.owner.index()].add(target.kind);
                }
                Some(destination) if before_boops[destination.index()].is_none() => {
                    state.board[adjacent.index()] = None;
                    state.board[destination.index()] = Some(target);
                }
                Some(_) => {}
            }
        }
    }
}

fn winner_after_boop(state: &BoopState, active: PlayerId) -> Option<PlayerId> {
    let opponent = <Boop as TwoPlayerZeroSumGame>::opponent(active).expect("valid player");
    [active, opponent].into_iter().find(|player| {
        cat_lines(state, *player).next().is_some()
            || pieces_on_board(state, *player, Some(PieceKind::Cat)) == PIECES_PER_PLAYER
    })
}

fn available_resolutions(state: &BoopState, player: PlayerId) -> Vec<Resolution> {
    let mut resolutions: Vec<_> = owned_lines(state, player)
        .map(Resolution::Graduate)
        .collect();
    if pieces_on_board(state, player, None) == PIECES_PER_PLAYER {
        resolutions.extend(state.board.iter().enumerate().filter_map(|(index, piece)| {
            piece
                .filter(|piece| piece.owner == player)
                .map(|_| Resolution::Recover(Position(index as u8)))
        }));
    }
    resolutions
}

fn apply_resolution(state: &mut BoopState, player: PlayerId, resolution: Resolution) {
    let positions: Vec<Position> = match resolution {
        Resolution::None => return,
        Resolution::Graduate(line) => line.positions.into_iter().collect(),
        Resolution::Recover(position) => vec![position],
    };

    for position in positions {
        let piece = state.board[position.index()]
            .take()
            .expect("legal resolution points to a piece");
        debug_assert_eq!(piece.owner, player);
        match piece.kind {
            PieceKind::Kitten => state.pools[player.index()].add(PieceKind::Cat),
            PieceKind::Cat => state.pools[player.index()].add(PieceKind::Cat),
        }
    }
}

fn owned_lines(state: &BoopState, player: PlayerId) -> impl Iterator<Item = GraduateLine> + '_ {
    all_lines().filter(move |line| {
        line.positions.iter().all(|position| {
            state.board[position.index()].is_some_and(|piece| piece.owner == player)
        })
    })
}

fn cat_lines(state: &BoopState, player: PlayerId) -> impl Iterator<Item = GraduateLine> + '_ {
    all_lines().filter(move |line| {
        line.positions.iter().all(|position| {
            state.board[position.index()]
                == Some(Piece {
                    owner: player,
                    kind: PieceKind::Cat,
                })
        })
    })
}

fn all_lines() -> impl Iterator<Item = GraduateLine> {
    ALL_LINES.into_iter()
}

const ALL_LINES: [GraduateLine; LINE_COUNT] = make_all_lines();

const fn make_all_lines() -> [GraduateLine; LINE_COUNT] {
    let empty = GraduateLine {
        positions: [Position(0); 3],
    };
    let mut lines = [empty; LINE_COUNT];
    let steps = [(0_i16, 1_i16), (1, 0), (1, 1), (1, -1)];
    let mut line_index = 0;
    let mut row = 0;
    while row < ROWS as i16 {
        let mut column = 0;
        while column < COLUMNS as i16 {
            let mut step_index = 0;
            while step_index < steps.len() {
                let (row_step, column_step) = steps[step_index];
                let end_row = row + 2 * row_step;
                let end_column = column + 2 * column_step;
                if end_row >= 0
                    && end_row < ROWS as i16
                    && end_column >= 0
                    && end_column < COLUMNS as i16
                {
                    lines[line_index] = GraduateLine {
                        positions: [
                            Position((row * COLUMNS as i16 + column) as u8),
                            Position(
                                ((row + row_step) * COLUMNS as i16 + column + column_step) as u8,
                            ),
                            Position((end_row * COLUMNS as i16 + end_column) as u8),
                        ],
                    };
                    line_index += 1;
                }
                step_index += 1;
            }
            column += 1;
        }
        row += 1;
    }
    assert!(line_index == LINE_COUNT);
    lines
}

fn pieces_on_board(state: &BoopState, player: PlayerId, kind: Option<PieceKind>) -> u8 {
    state
        .board
        .iter()
        .filter(|piece| {
            piece.is_some_and(|piece| {
                piece.owner == player && kind.is_none_or(|kind| piece.kind == kind)
            })
        })
        .count() as u8
}

fn strategic_heuristic_utility(state: &BoopState, player: PlayerId) -> Option<f32> {
    const NON_TERMINAL_LIMIT: f32 = 0.95;
    const IMMEDIATE_WIN: f32 = 0.9;

    let opponent = <Boop as TwoPlayerZeroSumGame>::opponent(player)?;
    let active = state.next_player;
    if winning_cat_pairs(state, active) > 0 {
        return Some(if active == player {
            IMMEDIATE_WIN
        } else {
            -IMMEDIATE_WIN
        });
    }

    let cat_progress =
        normalized_total_cats(state, player) - normalized_total_cats(state, opponent);
    let winning_threats =
        normalized_winning_threats(state, player) - normalized_winning_threats(state, opponent);
    let graduation_progress = normalized_graduation_potential(state, player)
        - normalized_graduation_potential(state, opponent);
    let safe_presence =
        normalized_safe_presence(state, player) - normalized_safe_presence(state, opponent);

    Some(
        (0.8 * cat_progress
            + 0.5 * winning_threats
            + 0.3 * graduation_progress
            + 0.1 * safe_presence)
            .clamp(-NON_TERMINAL_LIMIT, NON_TERMINAL_LIMIT),
    )
}

fn normalized_total_cats(state: &BoopState, player: PlayerId) -> f32 {
    let total =
        state.pools[player.index()].cats + pieces_on_board(state, player, Some(PieceKind::Cat));
    f32::from(total) / f32::from(PIECES_PER_PLAYER)
}

fn normalized_winning_threats(state: &BoopState, player: PlayerId) -> f32 {
    let threats = winning_cat_pairs(state, player).min(2);
    let tempo = if state.next_player == player {
        1.0
    } else {
        0.5
    };
    tempo * f32::from(threats) / 2.0
}

fn winning_cat_pairs(state: &BoopState, player: PlayerId) -> u8 {
    if state.pools[player.index()].cats == 0 {
        return 0;
    }

    all_lines()
        .filter(|line| {
            completable_endpoint_pair(state, *line, player)
                .is_some_and(|pieces| pieces.iter().all(|piece| piece.kind == PieceKind::Cat))
        })
        .count() as u8
}

fn normalized_graduation_potential(state: &BoopState, player: PlayerId) -> f32 {
    let tempo = if state.next_player == player {
        1.0
    } else {
        0.5
    };
    tempo * f32::from(graduation_potential(state, player).min(6)) / 6.0
}

fn graduation_potential(state: &BoopState, player: PlayerId) -> u8 {
    let pool = state.pools[player.index()];
    if pool.kittens + pool.cats == 0 {
        return 0;
    }
    let placed_kitten = u8::from(pool.kittens > 0);

    all_lines()
        .filter_map(|line| completable_endpoint_pair(state, line, player))
        .map(|pieces| {
            pieces
                .iter()
                .filter(|piece| piece.kind == PieceKind::Kitten)
                .count() as u8
                + placed_kitten
        })
        .sum()
}

fn completable_endpoint_pair(
    state: &BoopState,
    line: GraduateLine,
    player: PlayerId,
) -> Option<[Piece; 2]> {
    let [first, middle, last] = line.positions.map(|position| state.board[position.index()]);
    let pair = match (first, middle, last) {
        (Some(first), Some(middle), None) => [first, middle],
        (None, Some(middle), Some(last)) => [middle, last],
        _ => return None,
    };
    pair.iter()
        .all(|piece| piece.owner == player)
        .then_some(pair)
}

fn normalized_safe_presence(state: &BoopState, player: PlayerId) -> f32 {
    let score: u8 = state
        .board
        .iter()
        .enumerate()
        .filter_map(|(index, piece)| piece.filter(|piece| piece.owner == player).map(|_| index))
        .map(|index| position_safety(Position(index as u8)))
        .sum();
    f32::from(score) / (f32::from(PIECES_PER_PLAYER) * 3.0)
}

fn position_safety(position: Position) -> u8 {
    let row = position.row();
    let column = position.column();
    let edge_distance = row
        .min(ROWS as u8 - 1 - row)
        .min(column.min(COLUMNS as u8 - 1 - column));
    match edge_distance {
        0 => 1,
        1 => 2,
        _ => 3,
    }
}

fn checked_position(row: i16, column: i16) -> Option<Position> {
    if (0..ROWS as i16).contains(&row) && (0..COLUMNS as i16).contains(&column) {
        Position::new(row as u8, column as u8)
    } else {
        None
    }
}

fn is_straight_consecutive_line(positions: [Position; 3]) -> bool {
    let rows = positions.map(|position| i16::from(position.row()));
    let columns = positions.map(|position| i16::from(position.column()));
    let first_step = (rows[1] - rows[0], columns[1] - columns[0]);
    let second_step = (rows[2] - rows[1], columns[2] - columns[1]);
    first_step == second_step && matches!(first_step, (0, 1) | (1, 0) | (1, 1) | (1, -1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn position(row: u8, column: u8) -> Position {
        Position::new(row, column).unwrap()
    }

    fn simple_action(kind: PieceKind, row: u8, column: u8) -> BoopAction {
        BoopAction::new(kind, position(row, column), Resolution::None)
    }

    #[test]
    fn starts_with_eight_kittens_and_36_placements() {
        let game = Boop;
        let state = game.initial_state();

        assert_eq!(state.pools[0].kittens(), 8);
        assert_eq!(state.pools[0].cats(), 0);
        assert_eq!(game.legal_actions(&state).count(), 36);
    }

    #[test]
    fn cat_balance_heuristic_uses_the_requested_player_perspective() {
        let game = Boop;
        let mut state = game.initial_state();
        state.board[position(0, 0).index()] = Some(Piece::new(PlayerId::FIRST, PieceKind::Cat));
        state.board[position(1, 1).index()] = Some(Piece::new(PlayerId::FIRST, PieceKind::Cat));
        state.board[position(2, 2).index()] = Some(Piece::new(PlayerId::SECOND, PieceKind::Cat));
        state.board[position(3, 3).index()] = Some(Piece::new(PlayerId::SECOND, PieceKind::Kitten));

        assert_eq!(game.heuristic_count(), 2);
        assert_eq!(
            game.heuristic_utility(0, &state, PlayerId::FIRST),
            Some(0.1)
        );
        assert_eq!(
            game.heuristic_utility(0, &state, PlayerId::SECOND),
            Some(-0.1)
        );
        assert!(game.heuristic_utility(1, &state, PlayerId::FIRST).is_some());
        assert_eq!(game.heuristic_utility(2, &state, PlayerId::FIRST), None);
        assert_eq!(game.heuristic_utility(0, &state, PlayerId::new(2)), None);
    }

    #[test]
    fn cat_balance_heuristic_is_neutral_on_the_initial_board() {
        let game = Boop;
        let state = game.initial_state();

        assert_eq!(
            game.heuristic_utility(0, &state, PlayerId::FIRST),
            Some(0.0)
        );
        assert_eq!(
            game.heuristic_utility(0, &state, PlayerId::SECOND),
            Some(0.0)
        );
    }

    #[test]
    fn strategic_heuristic_values_cats_in_the_pool_and_is_symmetric() {
        let game = Boop;
        let mut state = game.initial_state();
        state.pools[0] = Pool {
            kittens: 7,
            cats: 1,
        };

        let first = game.heuristic_utility(1, &state, PlayerId::FIRST).unwrap();
        let second = game.heuristic_utility(1, &state, PlayerId::SECOND).unwrap();

        assert!(first > 0.0);
        assert_eq!(first, -second);
    }

    #[test]
    fn strategic_heuristic_preserves_cat_progress_between_pool_and_board() {
        let mut pool_state = Boop.initial_state();
        pool_state.pools[0] = Pool {
            kittens: 7,
            cats: 1,
        };
        let mut board_state = Boop.initial_state();
        board_state.pools[0] = Pool {
            kittens: 7,
            cats: 0,
        };
        board_state.board[position(2, 2).index()] =
            Some(Piece::new(PlayerId::FIRST, PieceKind::Cat));

        assert_eq!(
            normalized_total_cats(&pool_state, PlayerId::FIRST),
            normalized_total_cats(&board_state, PlayerId::FIRST)
        );
        assert!(
            strategic_heuristic_utility(&board_state, PlayerId::FIRST)
                > strategic_heuristic_utility(&pool_state, PlayerId::FIRST)
        );
    }

    #[test]
    fn strategic_heuristic_detects_an_immediate_cat_win_only_with_a_pool_cat() {
        let mut state = Boop.initial_state();
        state.board[position(2, 1).index()] = Some(Piece::new(PlayerId::FIRST, PieceKind::Cat));
        state.board[position(2, 2).index()] = Some(Piece::new(PlayerId::FIRST, PieceKind::Cat));
        state.pools[0] = Pool {
            kittens: 5,
            cats: 1,
        };

        assert_eq!(winning_cat_pairs(&state, PlayerId::FIRST), 2);
        assert_eq!(
            strategic_heuristic_utility(&state, PlayerId::FIRST),
            Some(0.9)
        );
        assert_eq!(
            strategic_heuristic_utility(&state, PlayerId::SECOND),
            Some(-0.9)
        );

        state.pools[0].cats = 0;
        state.pools[0].kittens = 6;
        assert_eq!(winning_cat_pairs(&state, PlayerId::FIRST), 0);
        assert!(strategic_heuristic_utility(&state, PlayerId::FIRST).unwrap() < 0.9);
    }

    #[test]
    fn strategic_heuristic_ranks_graduations_by_promoted_kittens() {
        fn potential(first: PieceKind, second: PieceKind) -> u8 {
            let mut state = Boop.initial_state();
            state.board[position(2, 0).index()] =
                Some(Piece::new(PlayerId::SECOND, PieceKind::Cat));
            state.board[position(2, 1).index()] = Some(Piece::new(PlayerId::FIRST, first));
            state.board[position(2, 2).index()] = Some(Piece::new(PlayerId::FIRST, second));
            graduation_potential(&state, PlayerId::FIRST)
        }

        let kittens = potential(PieceKind::Kitten, PieceKind::Kitten);
        let mixed = potential(PieceKind::Kitten, PieceKind::Cat);
        let cats = potential(PieceKind::Cat, PieceKind::Cat);

        assert!(kittens > mixed);
        assert!(mixed > cats);
    }

    #[test]
    fn strategic_heuristic_prefers_safe_central_positions() {
        let mut edge = Boop.initial_state();
        edge.board[position(0, 0).index()] = Some(Piece::new(PlayerId::FIRST, PieceKind::Kitten));
        edge.pools[0].kittens -= 1;
        let mut center = Boop.initial_state();
        center.board[position(2, 2).index()] = Some(Piece::new(PlayerId::FIRST, PieceKind::Kitten));
        center.pools[0].kittens -= 1;

        assert!(
            normalized_safe_presence(&center, PlayerId::FIRST)
                > normalized_safe_presence(&edge, PlayerId::FIRST)
        );
        assert!(
            strategic_heuristic_utility(&center, PlayerId::FIRST)
                > strategic_heuristic_utility(&edge, PlayerId::FIRST)
        );
    }

    #[test]
    fn strategic_heuristic_is_bounded() {
        let mut state = Boop.initial_state();
        state.pools[0] = Pool {
            kittens: 0,
            cats: 8,
        };

        for player in [PlayerId::FIRST, PlayerId::SECOND] {
            let utility = strategic_heuristic_utility(&state, player).unwrap();
            assert!(utility.is_finite());
            assert!((-0.95..=0.95).contains(&utility));
        }
    }

    #[test]
    fn placed_kitten_boops_adjacent_pieces_without_chain_reactions() {
        let game = Boop;
        let mut state = game.initial_state();
        state.board[position(2, 2).index()] = Some(Piece::new(PlayerId::SECOND, PieceKind::Kitten));
        state.board[position(2, 4).index()] = Some(Piece::new(PlayerId::SECOND, PieceKind::Kitten));

        game.apply_action(&mut state, &simple_action(PieceKind::Kitten, 2, 3))
            .unwrap();

        assert_eq!(
            state.board[position(2, 1).index()].unwrap().owner(),
            PlayerId::SECOND
        );
        assert_eq!(
            state.board[position(2, 5).index()].unwrap().owner(),
            PlayerId::SECOND
        );
    }

    #[test]
    fn edge_boops_return_pieces_to_their_owner() {
        let game = Boop;
        let mut state = game.initial_state();
        state.board[position(0, 0).index()] = Some(Piece::new(PlayerId::SECOND, PieceKind::Kitten));
        state.pools[1].kittens -= 1;

        game.apply_action(&mut state, &simple_action(PieceKind::Kitten, 1, 1))
            .unwrap();

        assert!(state.board[position(0, 0).index()].is_none());
        assert_eq!(state.pools[1].kittens(), 8);
    }

    #[test]
    fn kittens_cannot_boop_cats_but_cats_can() {
        let game = Boop;
        let mut kitten_state = game.initial_state();
        kitten_state.board[position(2, 2).index()] =
            Some(Piece::new(PlayerId::SECOND, PieceKind::Cat));
        game.apply_action(&mut kitten_state, &simple_action(PieceKind::Kitten, 2, 3))
            .unwrap();
        assert!(kitten_state.board[position(2, 2).index()].is_some());

        let mut cat_state = game.initial_state();
        cat_state.board[position(2, 2).index()] =
            Some(Piece::new(PlayerId::SECOND, PieceKind::Cat));
        cat_state.pools[0] = Pool {
            kittens: 7,
            cats: 1,
        };
        game.apply_action(&mut cat_state, &simple_action(PieceKind::Cat, 2, 3))
            .unwrap();
        assert!(cat_state.board[position(2, 1).index()].is_some());
    }

    #[test]
    fn a_kitten_line_requires_a_graduation_choice() {
        let game = Boop;
        let mut state = game.initial_state();
        for column in 0..2 {
            state.board[position(3, column).index()] =
                Some(Piece::new(PlayerId::FIRST, PieceKind::Kitten));
            state.pools[0].kittens -= 1;
        }

        let actions: Vec<_> = game
            .legal_actions(&state)
            .filter(|action| action.piece == PieceKind::Kitten && action.position == position(3, 2))
            .collect();
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0].resolution, Resolution::Graduate(_)));

        game.apply_action(&mut state, &actions[0]).unwrap();
        assert_eq!(state.pools[0].cats(), 3);
        assert!(state.board[position(3, 0).index()].is_none());
    }

    #[test]
    fn three_cats_win_before_graduation() {
        let game = Boop;
        let mut state = game.initial_state();
        for column in 0..2 {
            state.board[position(4, column).index()] =
                Some(Piece::new(PlayerId::FIRST, PieceKind::Cat));
        }
        state.pools[0] = Pool {
            kittens: 5,
            cats: 1,
        };

        game.apply_action(&mut state, &simple_action(PieceKind::Cat, 4, 2))
            .unwrap();

        assert_eq!(state.winner(), Some(PlayerId::FIRST));
        assert_eq!(game.terminal_utility(&state, PlayerId::FIRST), Some(1.0));
    }

    #[test]
    fn blocked_pieces_do_not_boop() {
        let game = Boop;
        let mut state = game.initial_state();
        state.board[position(2, 2).index()] = Some(Piece::new(PlayerId::SECOND, PieceKind::Kitten));
        state.board[position(2, 1).index()] = Some(Piece::new(PlayerId::FIRST, PieceKind::Kitten));

        game.apply_action(&mut state, &simple_action(PieceKind::Kitten, 2, 3))
            .unwrap();

        assert!(state.board[position(2, 2).index()].is_some());
        assert!(state.board[position(2, 1).index()].is_some());
    }

    #[test]
    fn placing_the_eighth_piece_requires_one_recovery() {
        let game = Boop;
        let mut state = game.initial_state();
        for (row, column) in [(0, 0), (0, 2), (0, 4), (2, 0), (2, 2), (2, 4), (4, 0)] {
            state.board[position(row, column).index()] =
                Some(Piece::new(PlayerId::FIRST, PieceKind::Kitten));
        }
        state.pools[0] = Pool {
            kittens: 1,
            cats: 0,
        };

        let actions: Vec<_> = game
            .legal_actions(&state)
            .filter(|action| action.position == position(5, 5))
            .collect();
        assert_eq!(actions.len(), 8);
        assert!(
            actions
                .iter()
                .all(|action| matches!(action.resolution, Resolution::Recover(_)))
        );

        game.apply_action(&mut state, &actions[0]).unwrap();
        assert_eq!(pieces_on_board(&state, PlayerId::FIRST, None), 7);
        assert_eq!(state.pools[0].cats(), 1);
    }

    #[test]
    fn all_eight_cats_on_the_board_win() {
        let game = Boop;
        let mut state = game.initial_state();
        for (row, column) in [(0, 0), (0, 2), (0, 4), (2, 0), (2, 2), (2, 4), (4, 0)] {
            state.board[position(row, column).index()] =
                Some(Piece::new(PlayerId::FIRST, PieceKind::Cat));
        }
        state.pools[0] = Pool {
            kittens: 0,
            cats: 1,
        };

        game.apply_action(&mut state, &simple_action(PieceKind::Cat, 5, 5))
            .unwrap();

        assert_eq!(state.winner(), Some(PlayerId::FIRST));
    }
}
