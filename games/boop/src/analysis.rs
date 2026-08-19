use std::{error::Error, fmt};

use meeple_bots_core::{Game, PlayerId, PositionStatus};

use super::{
    Boop, BoopAction, BoopState, ClassifiedBoop, GraduateLine, PIECES_PER_PLAYER, Piece, PieceKind,
    Position, Resolution, cat_lines, checked_position, classify_boop_target, pieces_on_board,
    place_and_boop,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoardZone {
    Center,
    Middle,
    Outer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StrategicPhase {
    AllKittens,
    OnePlayerHasCats,
    BothPlayersHaveCats,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LineOrientation {
    Horizontal,
    Vertical,
    DiagonalDown,
    DiagonalUp,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoopInteractionOutcome {
    Moved,
    OffBoard,
    Blocked,
    Immune,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlayerStateMetrics {
    pub pool_kittens: u8,
    pub pool_cats: u8,
    pub board_kittens: u8,
    pub board_cats: u8,
    pub center_pieces: u8,
    pub middle_pieces: u8,
    pub outer_pieces: u8,
}

impl PlayerStateMetrics {
    pub const fn total_cats(self) -> u8 {
        self.pool_cats + self.board_cats
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BoopStateMetrics {
    pub players: [PlayerStateMetrics; 2],
    pub empty_center: u8,
    pub empty_middle: u8,
    pub empty_outer: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BoopInteraction {
    pub target: Piece,
    pub origin: Position,
    pub destination_row: i16,
    pub destination_column: i16,
    pub outcome: BoopInteractionOutcome,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BoopResolutionAnalysis {
    pub resolution: Resolution,
    pub kittens_promoted: u8,
    pub cats_recycled: u8,
    pub recovered_piece: Option<PieceKind>,
    pub orientation: Option<LineOrientation>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoopTurnAnalysis {
    pub ply: u32,
    pub player: PlayerId,
    pub action: BoopAction,
    pub zone: BoardZone,
    pub phase: StrategicPhase,
    pub before: BoopStateMetrics,
    pub after: BoopStateMetrics,
    pub interactions: Vec<BoopInteraction>,
    pub resolution: Option<BoopResolutionAnalysis>,
    pub terminal_after: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WinningLineAnalysis {
    pub player: PlayerId,
    pub line: GraduateLine,
    pub orientation: LineOrientation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoopReplayAnalysis {
    pub turns: Vec<BoopTurnAnalysis>,
    pub winner: PlayerId,
    pub winner_has_cat_line: bool,
    pub winner_has_eight_cats: bool,
    pub winning_lines: Vec<WinningLineAnalysis>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BoopReplayError {
    ActionAfterTerminal {
        ply: u32,
    },
    UnexpectedPlayer {
        ply: u32,
        expected: PlayerId,
        recorded: PlayerId,
    },
    IllegalAction {
        ply: u32,
        message: String,
    },
    NonTerminalTrace,
}

impl fmt::Display for BoopReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ActionAfterTerminal { ply } => {
                write!(
                    formatter,
                    "trace contains an action after the game ended at ply {ply}"
                )
            }
            Self::UnexpectedPlayer {
                ply,
                expected,
                recorded,
            } => write!(
                formatter,
                "trace ply {ply} belongs to player {recorded}, expected player {expected}"
            ),
            Self::IllegalAction { ply, message } => {
                write!(
                    formatter,
                    "trace contains an illegal action at ply {ply}: {message}"
                )
            }
            Self::NonTerminalTrace => formatter.write_str("trace ends before the game is terminal"),
        }
    }
}

impl Error for BoopReplayError {}

pub fn analyze_replay(
    recorded_actions: &[(PlayerId, BoopAction)],
) -> Result<BoopReplayAnalysis, BoopReplayError> {
    let game = Boop;
    let mut state = game.initial_state();
    let mut turns = Vec::with_capacity(recorded_actions.len());

    for (index, (recorded_player, action)) in recorded_actions.iter().enumerate() {
        let ply = index as u32 + 1;
        let expected = match game.status(&state) {
            PositionStatus::PlayerTurn(player) => player,
            PositionStatus::Terminal => return Err(BoopReplayError::ActionAfterTerminal { ply }),
            _ => return Err(BoopReplayError::ActionAfterTerminal { ply }),
        };
        if *recorded_player != expected {
            return Err(BoopReplayError::UnexpectedPlayer {
                ply,
                expected,
                recorded: *recorded_player,
            });
        }

        let before = state_metrics(&state);
        let phase = strategic_phase(before);
        let interactions = boop_interactions(&state, *action);

        let mut next_state = state.clone();
        game.apply_action(&mut next_state, action)
            .map_err(|error| BoopReplayError::IllegalAction {
                ply,
                message: error.to_string(),
            })?;

        let mut after_boop = state.clone();
        place_and_boop(
            &mut after_boop,
            *recorded_player,
            action.piece(),
            action.position(),
        );
        let resolution = resolution_analysis(&after_boop, *action);
        let terminal_after = matches!(game.status(&next_state), PositionStatus::Terminal);
        let after = state_metrics(&next_state);
        turns.push(BoopTurnAnalysis {
            ply,
            player: *recorded_player,
            action: *action,
            zone: board_zone(action.position()),
            phase,
            before,
            after,
            interactions,
            resolution,
            terminal_after,
        });
        state = next_state;
    }

    if !matches!(game.status(&state), PositionStatus::Terminal) {
        return Err(BoopReplayError::NonTerminalTrace);
    }
    let winner = state.winner().expect("terminal boop state has a winner");
    let (winner_has_cat_line, winner_has_eight_cats, winning_lines) = winner_facts(&state, winner);

    Ok(BoopReplayAnalysis {
        turns,
        winner,
        winner_has_cat_line,
        winner_has_eight_cats,
        winning_lines,
    })
}

fn winner_facts(state: &BoopState, winner: PlayerId) -> (bool, bool, Vec<WinningLineAnalysis>) {
    let winning_lines: Vec<_> = [PlayerId::FIRST, PlayerId::SECOND]
        .into_iter()
        .flat_map(|player| {
            cat_lines(&state, player).map(move |line| WinningLineAnalysis {
                player,
                line,
                orientation: line_orientation(line),
            })
        })
        .collect();
    (
        winning_lines.iter().any(|line| line.player == winner),
        pieces_on_board(state, winner, Some(PieceKind::Cat)) == PIECES_PER_PLAYER,
        winning_lines,
    )
}

pub const fn board_zone(position: Position) -> BoardZone {
    let row = position.row();
    let column = position.column();
    if row >= 2 && row <= 3 && column >= 2 && column <= 3 {
        BoardZone::Center
    } else if row >= 1 && row <= 4 && column >= 1 && column <= 4 {
        BoardZone::Middle
    } else {
        BoardZone::Outer
    }
}

fn strategic_phase(metrics: BoopStateMetrics) -> StrategicPhase {
    match (
        metrics.players[0].total_cats() > 0,
        metrics.players[1].total_cats() > 0,
    ) {
        (false, false) => StrategicPhase::AllKittens,
        (true, true) => StrategicPhase::BothPlayersHaveCats,
        _ => StrategicPhase::OnePlayerHasCats,
    }
}

fn state_metrics(state: &BoopState) -> BoopStateMetrics {
    let mut players = [PlayerStateMetrics {
        pool_kittens: 0,
        pool_cats: 0,
        board_kittens: 0,
        board_cats: 0,
        center_pieces: 0,
        middle_pieces: 0,
        outer_pieces: 0,
    }; 2];
    for (index, pool) in state.pools().iter().enumerate() {
        players[index].pool_kittens = pool.kittens();
        players[index].pool_cats = pool.cats();
    }
    let mut empty_by_zone = [0_u8; 3];
    for (index, cell) in state.board().iter().enumerate() {
        let position = Position(index as u8);
        let zone_index = zone_index(board_zone(position));
        match cell {
            None => empty_by_zone[zone_index] += 1,
            Some(piece) => {
                let metrics = &mut players[piece.owner().index()];
                match piece.kind() {
                    PieceKind::Kitten => metrics.board_kittens += 1,
                    PieceKind::Cat => metrics.board_cats += 1,
                }
                match board_zone(position) {
                    BoardZone::Center => metrics.center_pieces += 1,
                    BoardZone::Middle => metrics.middle_pieces += 1,
                    BoardZone::Outer => metrics.outer_pieces += 1,
                }
            }
        }
    }

    BoopStateMetrics {
        players,
        empty_center: empty_by_zone[0],
        empty_middle: empty_by_zone[1],
        empty_outer: empty_by_zone[2],
    }
}

const fn zone_index(zone: BoardZone) -> usize {
    match zone {
        BoardZone::Center => 0,
        BoardZone::Middle => 1,
        BoardZone::Outer => 2,
    }
}

fn boop_interactions(state: &BoopState, action: BoopAction) -> Vec<BoopInteraction> {
    let mut interactions = Vec::with_capacity(8);
    let placed = action.position();
    for row_delta in -1..=1 {
        for column_delta in -1..=1 {
            if row_delta == 0 && column_delta == 0 {
                continue;
            }
            let adjacent_row = i16::from(placed.row()) + row_delta;
            let adjacent_column = i16::from(placed.column()) + column_delta;
            let Some(origin) = checked_position(adjacent_row, adjacent_column) else {
                continue;
            };
            let Some(target) = state.board()[origin.index()] else {
                continue;
            };
            let classified = classify_boop_target(
                state.board(),
                action.piece(),
                target,
                adjacent_row,
                adjacent_column,
                row_delta,
                column_delta,
            );
            let (destination_row, destination_column, outcome) = match classified {
                ClassifiedBoop::Immune => (
                    adjacent_row + row_delta,
                    adjacent_column + column_delta,
                    BoopInteractionOutcome::Immune,
                ),
                ClassifiedBoop::Moved(destination) => (
                    i16::from(destination.row()),
                    i16::from(destination.column()),
                    BoopInteractionOutcome::Moved,
                ),
                ClassifiedBoop::OffBoard { row, column } => {
                    (row, column, BoopInteractionOutcome::OffBoard)
                }
                ClassifiedBoop::Blocked(destination) => (
                    i16::from(destination.row()),
                    i16::from(destination.column()),
                    BoopInteractionOutcome::Blocked,
                ),
            };
            interactions.push(BoopInteraction {
                target,
                origin,
                destination_row,
                destination_column,
                outcome,
            });
        }
    }
    interactions
}

fn resolution_analysis(
    after_boop: &BoopState,
    action: BoopAction,
) -> Option<BoopResolutionAnalysis> {
    let (positions, orientation) = match action.resolution() {
        Resolution::None => return None,
        Resolution::Graduate(line) => (line.positions().to_vec(), Some(line_orientation(line))),
        Resolution::Recover(position) => (vec![position], None),
    };
    let pieces: Vec<_> = positions
        .iter()
        .filter_map(|position| after_boop.board()[position.index()])
        .collect();
    let kittens_promoted = pieces
        .iter()
        .filter(|piece| piece.kind() == PieceKind::Kitten)
        .count() as u8;
    let cats_recycled = pieces.len() as u8 - kittens_promoted;
    let recovered_piece = match action.resolution() {
        Resolution::Recover(_) => pieces.first().map(|piece| piece.kind()),
        _ => None,
    };

    Some(BoopResolutionAnalysis {
        resolution: action.resolution(),
        kittens_promoted,
        cats_recycled,
        recovered_piece,
        orientation,
    })
}

pub fn line_orientation(line: GraduateLine) -> LineOrientation {
    let [first, second, _] = line.positions();
    let row_step = i16::from(second.row()) - i16::from(first.row());
    let column_step = i16::from(second.column()) - i16::from(first.column());
    match (row_step, column_step) {
        (0, 1) => LineOrientation::Horizontal,
        (1, 0) => LineOrientation::Vertical,
        (1, 1) => LineOrientation::DiagonalDown,
        (1, -1) => LineOrientation::DiagonalUp,
        _ => unreachable!("GraduateLine always contains a straight consecutive line"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{COLUMNS, ROWS};

    fn position(row: u8, column: u8) -> Position {
        Position::new(row, column).unwrap()
    }

    fn state_with(pieces: &[(u8, u8, PlayerId, PieceKind)]) -> BoopState {
        let mut state = Boop.initial_state();
        for (row, column, player, kind) in pieces {
            state.board[position(*row, *column).index()] = Some(Piece::new(*player, *kind));
        }
        state
    }

    #[test]
    fn zones_partition_the_board_as_four_twelve_and_twenty_cells() {
        let mut counts = [0; 3];
        for row in 0..ROWS as u8 {
            for column in 0..COLUMNS as u8 {
                counts[zone_index(board_zone(position(row, column)))] += 1;
            }
        }
        assert_eq!(counts, [4, 12, 20]);
    }

    #[test]
    fn interactions_distinguish_move_off_board_block_and_immunity() {
        let moved = state_with(&[(2, 3, PlayerId::SECOND, PieceKind::Kitten)]);
        let moved = boop_interactions(
            &moved,
            BoopAction::new(PieceKind::Kitten, position(2, 2), Resolution::None),
        );
        assert_eq!(moved[0].outcome, BoopInteractionOutcome::Moved);
        assert_eq!(
            (moved[0].destination_row, moved[0].destination_column),
            (2, 4)
        );

        let off_board = state_with(&[(0, 0, PlayerId::SECOND, PieceKind::Kitten)]);
        let off_board = boop_interactions(
            &off_board,
            BoopAction::new(PieceKind::Kitten, position(0, 1), Resolution::None),
        );
        assert_eq!(off_board[0].outcome, BoopInteractionOutcome::OffBoard);

        let blocked = state_with(&[
            (2, 3, PlayerId::SECOND, PieceKind::Kitten),
            (2, 4, PlayerId::FIRST, PieceKind::Kitten),
        ]);
        let blocked = boop_interactions(
            &blocked,
            BoopAction::new(PieceKind::Kitten, position(2, 2), Resolution::None),
        );
        assert_eq!(blocked[0].outcome, BoopInteractionOutcome::Blocked);

        let immune = state_with(&[(2, 3, PlayerId::SECOND, PieceKind::Cat)]);
        let immune = boop_interactions(
            &immune,
            BoopAction::new(PieceKind::Kitten, position(2, 2), Resolution::None),
        );
        assert_eq!(immune[0].outcome, BoopInteractionOutcome::Immune);
    }

    #[test]
    fn mixed_graduation_counts_promoted_kittens_and_recycled_cats() {
        let state = state_with(&[
            (2, 1, PlayerId::FIRST, PieceKind::Kitten),
            (2, 2, PlayerId::FIRST, PieceKind::Cat),
            (2, 3, PlayerId::FIRST, PieceKind::Kitten),
        ]);
        let line = GraduateLine::new([position(2, 1), position(2, 2), position(2, 3)]).unwrap();
        let resolution = resolution_analysis(
            &state,
            BoopAction::new(
                PieceKind::Kitten,
                position(0, 0),
                Resolution::Graduate(line),
            ),
        )
        .unwrap();

        assert_eq!(resolution.kittens_promoted, 2);
        assert_eq!(resolution.cats_recycled, 1);
        assert_eq!(resolution.orientation, Some(LineOrientation::Horizontal));
    }

    #[test]
    fn recovery_distinguishes_a_promoted_kitten_from_a_recycled_cat() {
        for (kind, promoted, recycled) in [(PieceKind::Kitten, 1, 0), (PieceKind::Cat, 0, 1)] {
            let state = state_with(&[(4, 1, PlayerId::FIRST, kind)]);
            let resolution = resolution_analysis(
                &state,
                BoopAction::new(
                    PieceKind::Kitten,
                    position(0, 0),
                    Resolution::Recover(position(4, 1)),
                ),
            )
            .unwrap();

            assert_eq!(resolution.kittens_promoted, promoted);
            assert_eq!(resolution.cats_recycled, recycled);
            assert_eq!(resolution.recovered_piece, Some(kind));
        }
    }

    #[test]
    fn winner_facts_distinguish_cat_lines_and_eight_cats() {
        let line_state = state_with(&[
            (2, 1, PlayerId::FIRST, PieceKind::Cat),
            (2, 2, PlayerId::FIRST, PieceKind::Cat),
            (2, 3, PlayerId::FIRST, PieceKind::Cat),
        ]);
        let (has_line, has_eight, lines) = winner_facts(&line_state, PlayerId::FIRST);
        assert!(has_line);
        assert!(!has_eight);
        assert_eq!(lines.len(), 1);

        let pieces = [
            (0, 0, PlayerId::FIRST, PieceKind::Cat),
            (0, 2, PlayerId::FIRST, PieceKind::Cat),
            (0, 4, PlayerId::FIRST, PieceKind::Cat),
            (2, 0, PlayerId::FIRST, PieceKind::Cat),
            (2, 2, PlayerId::FIRST, PieceKind::Cat),
            (4, 0, PlayerId::FIRST, PieceKind::Cat),
            (4, 3, PlayerId::FIRST, PieceKind::Cat),
            (5, 5, PlayerId::FIRST, PieceKind::Cat),
        ];
        let eight_state = state_with(&pieces);
        let (has_line, has_eight, _) = winner_facts(&eight_state, PlayerId::FIRST);
        assert!(!has_line);
        assert!(has_eight);
    }

    #[test]
    fn every_line_orientation_is_identified() {
        let cases = [
            ([(1, 1), (1, 2), (1, 3)], LineOrientation::Horizontal),
            ([(1, 1), (2, 1), (3, 1)], LineOrientation::Vertical),
            ([(1, 1), (2, 2), (3, 3)], LineOrientation::DiagonalDown),
            ([(1, 3), (2, 2), (3, 1)], LineOrientation::DiagonalUp),
        ];
        for (positions, expected) in cases {
            let line =
                GraduateLine::new(positions.map(|(row, column)| position(row, column))).unwrap();
            assert_eq!(line_orientation(line), expected);
        }
    }

    #[test]
    fn replay_requires_a_complete_legal_trace() {
        assert_eq!(analyze_replay(&[]), Err(BoopReplayError::NonTerminalTrace));
        let action = BoopAction::new(PieceKind::Kitten, position(2, 2), Resolution::None);
        assert!(matches!(
            analyze_replay(&[(PlayerId::SECOND, action)]),
            Err(BoopReplayError::UnexpectedPlayer { ply: 1, .. })
        ));
    }
}
