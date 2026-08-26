use std::{error::Error, fmt};

use meeple_bots_core::{Game, PlayerId, PositionStatus};

use super::{
    ForestPosition, GemstoneSacrifice, PowerSource, Spirit, SpiritTile, SpiritsOfTheForest,
    SpiritsOfTheForestAction, SpiritsOfTheForestState, TurnPhase,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpiritsPlayerStateMetrics {
    pub spirit_symbols: [u8; 9],
    pub power_sources: [u8; 3],
    pub tiles: u8,
    pub score: i16,
    pub reachable_score: i16,
    pub categories_present: u8,
    pub categories_reachable: u8,
    pub categories_leading: u8,
    pub gemstones_available: u8,
    pub gemstones_placed: u8,
    pub gemstones_removed: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpiritsStateMetrics {
    pub players: [SpiritsPlayerStateMetrics; 2],
    pub remaining_tiles: u8,
    pub completed_turns: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TileTakeAnalysis {
    pub position: ForestPosition,
    pub tile: SpiritTile,
    pub reservation_owner: Option<PlayerId>,
    pub sacrifice: Option<GemstoneSacrifice>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpiritsTurnAnalysis {
    pub ply: u32,
    pub physical_turn: u32,
    pub action_in_turn: u8,
    pub player: PlayerId,
    pub action: SpiritsOfTheForestAction,
    pub phase_before: TurnPhase,
    pub phase_after: TurnPhase,
    pub legal_actions_before: u32,
    pub before: SpiritsStateMetrics,
    pub after: SpiritsStateMetrics,
    pub tile_take: Option<TileTakeAnalysis>,
    pub turn_completed_after: bool,
    pub terminal_after: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScoringCategory {
    Spirit(Spirit),
    PowerSource(PowerSource),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CategoryScoreAnalysis {
    pub category: ScoringCategory,
    pub counts: [u8; 2],
    pub points: [i16; 2],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpiritsReplayAnalysis {
    pub turns: Vec<SpiritsTurnAnalysis>,
    pub winner: Option<PlayerId>,
    pub final_scores: [i16; 2],
    pub categories: Vec<CategoryScoreAnalysis>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SpiritsReplayError {
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

impl fmt::Display for SpiritsReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ActionAfterTerminal { ply } => write!(
                formatter,
                "trace contains an action after the game ended at ply {ply}"
            ),
            Self::UnexpectedPlayer {
                ply,
                expected,
                recorded,
            } => write!(
                formatter,
                "trace ply {ply} belongs to player {recorded}, expected player {expected}"
            ),
            Self::IllegalAction { ply, message } => write!(
                formatter,
                "trace contains an illegal action at ply {ply}: {message}"
            ),
            Self::NonTerminalTrace => formatter.write_str("trace ends before the game is terminal"),
        }
    }
}

impl Error for SpiritsReplayError {}

pub fn analyze_replay(
    game: &SpiritsOfTheForest,
    recorded_actions: &[(PlayerId, SpiritsOfTheForestAction)],
) -> Result<SpiritsReplayAnalysis, SpiritsReplayError> {
    let mut state = game.initial_state();
    let mut turns = Vec::with_capacity(recorded_actions.len());
    let mut previous_physical_turn = 0;
    let mut action_in_turn = 0;

    for (index, (recorded_player, action)) in recorded_actions.iter().enumerate() {
        let ply = index as u32 + 1;
        let expected = match game.status(&state) {
            PositionStatus::PlayerTurn(player) => player,
            PositionStatus::Terminal => {
                return Err(SpiritsReplayError::ActionAfterTerminal { ply });
            }
            _ => return Err(SpiritsReplayError::ActionAfterTerminal { ply }),
        };
        if *recorded_player != expected {
            return Err(SpiritsReplayError::UnexpectedPlayer {
                ply,
                expected,
                recorded: *recorded_player,
            });
        }

        let physical_turn = u32::from(state.completed_turns) + 1;
        if physical_turn == previous_physical_turn {
            action_in_turn += 1;
        } else {
            previous_physical_turn = physical_turn;
            action_in_turn = 1;
        }
        let before = state_metrics(game, &state);
        let phase_before = state.phase;
        let legal_actions_before = game.legal_actions(&state).count() as u32;
        let tile_take = tile_take_analysis(game, &state, *action);

        let mut next_state = state.clone();
        game.apply_action(&mut next_state, action)
            .map_err(|error| SpiritsReplayError::IllegalAction {
                ply,
                message: error.to_string(),
            })?;

        let terminal_after = matches!(game.status(&next_state), PositionStatus::Terminal);
        let turn_completed_after =
            terminal_after || next_state.completed_turns > state.completed_turns;
        turns.push(SpiritsTurnAnalysis {
            ply,
            physical_turn,
            action_in_turn,
            player: *recorded_player,
            action: *action,
            phase_before,
            phase_after: next_state.phase,
            legal_actions_before,
            before,
            after: state_metrics(game, &next_state),
            tile_take,
            turn_completed_after,
            terminal_after,
        });
        state = next_state;
    }

    if !matches!(game.status(&state), PositionStatus::Terminal) {
        return Err(SpiritsReplayError::NonTerminalTrace);
    }

    Ok(SpiritsReplayAnalysis {
        final_scores: game.scores(&state),
        winner: game.winner(&state),
        categories: category_scores(&state),
        turns,
    })
}

fn state_metrics(
    game: &SpiritsOfTheForest,
    state: &SpiritsOfTheForestState,
) -> SpiritsStateMetrics {
    let scores = game.scores(state);
    let reachable_scores = game.reachable_progress_scores(state);
    let category_statuses = category_statuses(game, state);
    let players = std::array::from_fn(|index| {
        let collection = state.collections[index];
        let gemstones = state.gemstone_pools[index];
        SpiritsPlayerStateMetrics {
            spirit_symbols: collection.spirit_symbols,
            power_sources: collection.power_sources,
            tiles: collection.tiles,
            score: scores[index],
            reachable_score: reachable_scores[index],
            categories_present: category_statuses[index].present,
            categories_reachable: category_statuses[index].reachable,
            categories_leading: category_statuses[index].leading,
            gemstones_available: gemstones.available,
            gemstones_placed: gemstones.placed(),
            gemstones_removed: gemstones.removed,
        }
    });
    SpiritsStateMetrics {
        players,
        remaining_tiles: state.remaining_tiles() as u8,
        completed_turns: state.completed_turns,
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct CategoryStatusMetrics {
    present: u8,
    reachable: u8,
    leading: u8,
}

fn category_statuses(
    game: &SpiritsOfTheForest,
    state: &SpiritsOfTheForestState,
) -> [CategoryStatusMetrics; 2] {
    let (remaining_spirits, remaining_sources) = game.remaining_category_symbols(state);
    let mut statuses = [CategoryStatusMetrics::default(); 2];
    for spirit in Spirit::ALL {
        update_category_statuses(
            &mut statuses,
            [
                state.collections[0].spirit_symbols[spirit.index()],
                state.collections[1].spirit_symbols[spirit.index()],
            ],
            remaining_spirits[spirit.index()],
        );
    }
    for source in PowerSource::ALL {
        update_category_statuses(
            &mut statuses,
            [
                state.collections[0].power_sources[source.index()],
                state.collections[1].power_sources[source.index()],
            ],
            remaining_sources[source.index()],
        );
    }
    statuses
}

fn update_category_statuses(
    statuses: &mut [CategoryStatusMetrics; 2],
    collected: [u8; 2],
    remaining: u8,
) {
    let threshold = (collected[0] + collected[1] + remaining).div_ceil(2);
    for player in 0..2 {
        if collected[player] > 0 {
            statuses[player].present += 1;
        }
        if collected[player] + remaining >= threshold {
            statuses[player].reachable += 1;
        }
        if collected[player] > 0 && collected[player] >= collected[1 - player] {
            statuses[player].leading += 1;
        }
    }
}

fn tile_take_analysis(
    game: &SpiritsOfTheForest,
    state: &SpiritsOfTheForestState,
    action: SpiritsOfTheForestAction,
) -> Option<TileTakeAnalysis> {
    let SpiritsOfTheForestAction::TakeTile {
        position,
        sacrifice,
    } = action
    else {
        return None;
    };
    Some(TileTakeAnalysis {
        position,
        tile: game.tile(position),
        reservation_owner: state.gemstones[position.index()],
        sacrifice,
    })
}

fn category_scores(state: &SpiritsOfTheForestState) -> Vec<CategoryScoreAnalysis> {
    let mut categories = Vec::with_capacity(Spirit::ALL.len() + PowerSource::ALL.len());
    for spirit in Spirit::ALL {
        let counts = [
            state.collections[0].spirit_symbols[spirit.index()],
            state.collections[1].spirit_symbols[spirit.index()],
        ];
        categories.push(CategoryScoreAnalysis {
            category: ScoringCategory::Spirit(spirit),
            counts,
            points: category_points(counts),
        });
    }
    for source in PowerSource::ALL {
        let counts = [
            state.collections[0].power_sources[source.index()],
            state.collections[1].power_sources[source.index()],
        ];
        categories.push(CategoryScoreAnalysis {
            category: ScoringCategory::PowerSource(source),
            counts,
            points: category_points(counts),
        });
    }
    categories
}

fn category_points(counts: [u8; 2]) -> [i16; 2] {
    std::array::from_fn(|player| {
        let opponent = 1 - player;
        if counts[player] == 0 {
            -3
        } else if counts[player] >= counts[opponent] {
            i16::from(counts[player])
        } else {
            0
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SPIRIT_TILES;

    fn complete_replay(game: &SpiritsOfTheForest) -> Vec<(PlayerId, SpiritsOfTheForestAction)> {
        let mut state = game.initial_state();
        let mut recorded = Vec::new();
        while let PositionStatus::PlayerTurn(player) = game.status(&state) {
            let action = game
                .legal_actions(&state)
                .next()
                .expect("non-terminal Spirits state has a legal action");
            game.apply_action(&mut state, &action).unwrap();
            recorded.push((player, action));
        }
        recorded
    }

    #[test]
    fn replay_groups_internal_actions_into_physical_turns() {
        let game = SpiritsOfTheForest::from_tiles(SPIRIT_TILES);
        let recorded = complete_replay(&game);
        let analysis = analyze_replay(&game, &recorded).unwrap();

        assert_eq!(analysis.turns.len(), recorded.len());
        assert!(analysis.turns.len() > analysis.turns.last().unwrap().physical_turn as usize);
        for pair in analysis.turns.windows(2) {
            if pair[0].physical_turn == pair[1].physical_turn {
                assert_eq!(pair[1].action_in_turn, pair[0].action_in_turn + 1);
                assert_eq!(pair[1].player, pair[0].player);
            } else {
                assert_eq!(pair[1].physical_turn, pair[0].physical_turn + 1);
                assert_eq!(pair[1].action_in_turn, 1);
            }
        }
        assert!(analysis.turns.last().unwrap().terminal_after);
    }

    #[test]
    fn category_breakdown_reconstructs_final_scores() {
        let game = SpiritsOfTheForest::from_tiles(SPIRIT_TILES);
        let analysis = analyze_replay(&game, &complete_replay(&game)).unwrap();
        let reconstructed = analysis
            .categories
            .iter()
            .fold([0, 0], |mut total, category| {
                total[0] += category.points[0];
                total[1] += category.points[1];
                total
            });

        assert_eq!(analysis.categories.len(), 12);
        assert_eq!(reconstructed, analysis.final_scores);
    }

    #[test]
    fn replay_tracks_category_viability_and_reachable_progress() {
        let game = SpiritsOfTheForest::from_tiles(SPIRIT_TILES);
        let analysis = analyze_replay(&game, &complete_replay(&game)).unwrap();
        let initial = analysis.turns[0].before;
        let final_state = analysis.turns.last().unwrap().after;

        for player in 0..2 {
            assert_eq!(initial.players[player].reachable_score, 0);
            assert_eq!(initial.players[player].categories_present, 0);
            assert_eq!(initial.players[player].categories_reachable, 12);
            assert_eq!(initial.players[player].categories_leading, 0);
            assert_eq!(
                final_state.players[player].reachable_score,
                final_state.players[player].score
            );
            assert!(final_state.players[player].categories_present <= 12);
            assert!(final_state.players[player].categories_reachable <= 12);
            assert!(final_state.players[player].categories_leading <= 12);
        }
    }

    #[test]
    fn replay_rejects_wrong_players_and_incomplete_traces() {
        let game = SpiritsOfTheForest::from_tiles(SPIRIT_TILES);
        assert_eq!(
            analyze_replay(&game, &[]),
            Err(SpiritsReplayError::NonTerminalTrace)
        );
        let action = game.legal_actions(&game.initial_state()).next().unwrap();
        assert!(matches!(
            analyze_replay(&game, &[(PlayerId::SECOND, action)]),
            Err(SpiritsReplayError::UnexpectedPlayer { ply: 1, .. })
        ));
    }
}
