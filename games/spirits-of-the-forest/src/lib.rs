//! Two-player Spirits of the Forest without favor tokens.

use meeple_bots_core::{
    DeterministicGame, Game, HeuristicGame, HeuristicParameterSpec, HeuristicParameters,
    IllegalAction, PerfectInformationGame, PlayerId, PositionStatus, RandomSource,
    TwoPlayerZeroSumGame,
};

mod analysis;

pub use analysis::{
    CategoryScoreAnalysis, ScoringCategory, SpiritsPlayerStateMetrics, SpiritsReplayAnalysis,
    SpiritsReplayError, SpiritsStateMetrics, SpiritsTurnAnalysis, TileTakeAnalysis, analyze_replay,
};

pub const ROWS: usize = 4;
pub const COLUMNS: usize = 12;
pub const TILE_COUNT: usize = ROWS * COLUMNS;
pub const GEMSTONES_PER_PLAYER: u8 = 3;
pub const DEFAULT_GEMSTONE_EARLY_BONUS: f64 = 4.0;

const H0_PARAMETERS: [HeuristicParameterSpec; 1] = [HeuristicParameterSpec {
    name: "gemstone_early_bonus",
    default: DEFAULT_GEMSTONE_EARLY_BONUS,
    minimum: Some(0.0),
    maximum: None,
}];

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum Spirit {
    Moss,
    Flowers,
    Fruits,
    Mushrooms,
    Water,
    Vines,
    Branches,
    Leaves,
    Webs,
}

impl Spirit {
    pub const ALL: [Self; 9] = [
        Self::Moss,
        Self::Flowers,
        Self::Fruits,
        Self::Mushrooms,
        Self::Water,
        Self::Vines,
        Self::Branches,
        Self::Leaves,
        Self::Webs,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum PowerSource {
    Fire,
    Moon,
    Sun,
}

impl PowerSource {
    pub const ALL: [Self; 3] = [Self::Fire, Self::Moon, Self::Sun];

    pub const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SpiritTile {
    spirit: Spirit,
    spirit_symbols: u8,
    power_source: Option<PowerSource>,
}

impl SpiritTile {
    pub const fn new(
        spirit: Spirit,
        spirit_symbols: u8,
        power_source: Option<PowerSource>,
    ) -> Self {
        Self {
            spirit,
            spirit_symbols,
            power_source,
        }
    }

    pub const fn spirit(self) -> Spirit {
        self.spirit
    }

    pub const fn spirit_symbols(self) -> u8 {
        self.spirit_symbols
    }

    pub const fn power_source(self) -> Option<PowerSource> {
        self.power_source
    }
}

const fn double(spirit: Spirit) -> SpiritTile {
    SpiritTile::new(spirit, 2, None)
}

const fn powered(spirit: Spirit, source: PowerSource) -> SpiritTile {
    SpiritTile::new(spirit, 1, Some(source))
}

const fn single(spirit: Spirit) -> SpiritTile {
    SpiritTile::new(spirit, 1, None)
}

pub const SPIRIT_TILES: [SpiritTile; TILE_COUNT] = [
    double(Spirit::Moss),
    powered(Spirit::Moss, PowerSource::Moon),
    powered(Spirit::Moss, PowerSource::Sun),
    powered(Spirit::Moss, PowerSource::Fire),
    double(Spirit::Flowers),
    double(Spirit::Flowers),
    powered(Spirit::Flowers, PowerSource::Sun),
    powered(Spirit::Flowers, PowerSource::Moon),
    double(Spirit::Fruits),
    powered(Spirit::Fruits, PowerSource::Sun),
    powered(Spirit::Fruits, PowerSource::Moon),
    powered(Spirit::Fruits, PowerSource::Fire),
    powered(Spirit::Fruits, PowerSource::Fire),
    double(Spirit::Mushrooms),
    double(Spirit::Mushrooms),
    powered(Spirit::Mushrooms, PowerSource::Moon),
    powered(Spirit::Mushrooms, PowerSource::Fire),
    powered(Spirit::Mushrooms, PowerSource::Sun),
    double(Spirit::Water),
    double(Spirit::Water),
    powered(Spirit::Water, PowerSource::Moon),
    powered(Spirit::Water, PowerSource::Fire),
    powered(Spirit::Water, PowerSource::Sun),
    double(Spirit::Vines),
    double(Spirit::Vines),
    powered(Spirit::Vines, PowerSource::Moon),
    powered(Spirit::Vines, PowerSource::Fire),
    powered(Spirit::Vines, PowerSource::Sun),
    single(Spirit::Vines),
    double(Spirit::Branches),
    double(Spirit::Branches),
    powered(Spirit::Branches, PowerSource::Moon),
    powered(Spirit::Branches, PowerSource::Fire),
    powered(Spirit::Branches, PowerSource::Sun),
    single(Spirit::Branches),
    double(Spirit::Leaves),
    double(Spirit::Leaves),
    powered(Spirit::Leaves, PowerSource::Moon),
    powered(Spirit::Leaves, PowerSource::Fire),
    powered(Spirit::Leaves, PowerSource::Sun),
    single(Spirit::Leaves),
    double(Spirit::Webs),
    double(Spirit::Webs),
    double(Spirit::Webs),
    powered(Spirit::Webs, PowerSource::Moon),
    powered(Spirit::Webs, PowerSource::Fire),
    powered(Spirit::Webs, PowerSource::Sun),
    single(Spirit::Webs),
];

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ForestPosition(u8);

impl ForestPosition {
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

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GemstoneSacrifice {
    Available,
    Forest(ForestPosition),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SpiritsOfTheForestAction {
    TakeTile {
        position: ForestPosition,
        sacrifice: Option<GemstoneSacrifice>,
    },
    EndCollection,
    PlaceGemstone {
        target: ForestPosition,
    },
    MoveGemstone {
        source: ForestPosition,
        target: ForestPosition,
    },
    SkipGemstone,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TurnPhase {
    Collect,
    PlaceGemstone,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PlayerCollection {
    spirit_symbols: [u8; 9],
    power_sources: [u8; 3],
    tiles: u8,
}

impl PlayerCollection {
    const EMPTY: Self = Self {
        spirit_symbols: [0; 9],
        power_sources: [0; 3],
        tiles: 0,
    };

    pub const fn spirit_symbols(&self) -> &[u8; 9] {
        &self.spirit_symbols
    }

    pub const fn power_sources(&self) -> &[u8; 3] {
        &self.power_sources
    }

    pub const fn tiles(&self) -> u8 {
        self.tiles
    }

    fn add(&mut self, tile: SpiritTile) {
        self.spirit_symbols[tile.spirit.index()] += tile.spirit_symbols;
        if let Some(source) = tile.power_source {
            self.power_sources[source.index()] += 1;
        }
        self.tiles += 1;
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GemstonePool {
    available: u8,
    removed: u8,
}

impl GemstonePool {
    pub const fn available(self) -> u8 {
        self.available
    }

    pub const fn removed(self) -> u8 {
        self.removed
    }

    pub const fn placed(self) -> u8 {
        GEMSTONES_PER_PLAYER - self.available - self.removed
    }

    pub const fn usable(self) -> u8 {
        GEMSTONES_PER_PLAYER - self.removed
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SpiritsOfTheForestState {
    remaining: [bool; TILE_COUNT],
    gemstones: [Option<PlayerId>; TILE_COUNT],
    collections: [PlayerCollection; 2],
    gemstone_pools: [GemstonePool; 2],
    next_player: PlayerId,
    phase: TurnPhase,
    collected_this_turn: u8,
    collection_spirit: Option<Spirit>,
    completed_turns: u8,
}

impl SpiritsOfTheForestState {
    pub const fn remaining(&self) -> &[bool; TILE_COUNT] {
        &self.remaining
    }

    pub const fn gemstones(&self) -> &[Option<PlayerId>; TILE_COUNT] {
        &self.gemstones
    }

    pub const fn collections(&self) -> &[PlayerCollection; 2] {
        &self.collections
    }

    pub const fn gemstone_pools(&self) -> &[GemstonePool; 2] {
        &self.gemstone_pools
    }

    pub const fn next_player(&self) -> PlayerId {
        self.next_player
    }

    pub const fn phase(&self) -> TurnPhase {
        self.phase
    }

    pub const fn completed_turns(&self) -> u8 {
        self.completed_turns
    }

    pub fn remaining_tiles(&self) -> usize {
        self.remaining
            .iter()
            .filter(|remaining| **remaining)
            .count()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpiritsOfTheForest {
    tiles: [SpiritTile; TILE_COUNT],
}

impl SpiritsOfTheForest {
    pub fn shuffled<R: RandomSource + ?Sized>(rng: &mut R) -> Self {
        let mut tiles = SPIRIT_TILES;
        for upper in (1..tiles.len()).rev() {
            let selected = rng.index(upper + 1).expect("shuffle range is non-empty");
            tiles.swap(upper, selected);
        }
        Self { tiles }
    }

    pub const fn from_tiles(tiles: [SpiritTile; TILE_COUNT]) -> Self {
        Self { tiles }
    }

    pub const fn tiles(&self) -> &[SpiritTile; TILE_COUNT] {
        &self.tiles
    }

    pub const fn tile(&self, position: ForestPosition) -> SpiritTile {
        self.tiles[position.index()]
    }

    pub fn scores(&self, state: &SpiritsOfTheForestState) -> [i16; 2] {
        let mut scores = [0_i16; 2];
        for spirit in Spirit::ALL {
            score_category(
                &mut scores,
                state.collections[0].spirit_symbols[spirit.index()],
                state.collections[1].spirit_symbols[spirit.index()],
            );
        }
        for source in PowerSource::ALL {
            score_category(
                &mut scores,
                state.collections[0].power_sources[source.index()],
                state.collections[1].power_sources[source.index()],
            );
        }
        scores
    }

    fn reachable_progress_scores(&self, state: &SpiritsOfTheForestState) -> [i16; 2] {
        let (remaining_spirits, remaining_sources) = self.remaining_category_symbols(state);

        let mut scores = [0_i16; 2];
        for spirit in Spirit::ALL {
            score_reachable_category(
                &mut scores,
                [
                    state.collections[0].spirit_symbols[spirit.index()],
                    state.collections[1].spirit_symbols[spirit.index()],
                ],
                remaining_spirits[spirit.index()],
            );
        }
        for source in PowerSource::ALL {
            score_reachable_category(
                &mut scores,
                [
                    state.collections[0].power_sources[source.index()],
                    state.collections[1].power_sources[source.index()],
                ],
                remaining_sources[source.index()],
            );
        }
        scores
    }

    fn remaining_category_symbols(&self, state: &SpiritsOfTheForestState) -> ([u8; 9], [u8; 3]) {
        let mut remaining_spirits = [0_u8; 9];
        let mut remaining_sources = [0_u8; 3];
        for (tile, remaining) in self.tiles.iter().zip(state.remaining) {
            if !remaining {
                continue;
            }
            remaining_spirits[tile.spirit.index()] += tile.spirit_symbols;
            if let Some(source) = tile.power_source {
                remaining_sources[source.index()] += 1;
            }
        }
        (remaining_spirits, remaining_sources)
    }

    pub fn winner(&self, state: &SpiritsOfTheForestState) -> Option<PlayerId> {
        if state.remaining_tiles() != 0 {
            return None;
        }
        let scores = self.scores(state);
        match scores[0].cmp(&scores[1]) {
            std::cmp::Ordering::Greater => Some(PlayerId::FIRST),
            std::cmp::Ordering::Less => Some(PlayerId::SECOND),
            std::cmp::Ordering::Equal => {
                match state.collections[0].tiles.cmp(&state.collections[1].tiles) {
                    std::cmp::Ordering::Less => Some(PlayerId::FIRST),
                    std::cmp::Ordering::Greater => Some(PlayerId::SECOND),
                    std::cmp::Ordering::Equal => None,
                }
            }
        }
    }

    fn end_positions(&self, state: &SpiritsOfTheForestState) -> Vec<ForestPosition> {
        let mut positions = Vec::with_capacity(ROWS * 2);
        for row in 0..ROWS {
            let start = row * COLUMNS;
            let end = start + COLUMNS;
            let Some(left) = (start..end).find(|index| state.remaining[*index]) else {
                continue;
            };
            let right = (start..end)
                .rev()
                .find(|index| state.remaining[*index])
                .expect("non-empty row has a right edge");
            positions.push(ForestPosition(left as u8));
            if right != left {
                positions.push(ForestPosition(right as u8));
            }
        }
        positions
    }

    fn take_actions(&self, state: &SpiritsOfTheForestState) -> Vec<SpiritsOfTheForestAction> {
        let player = state.next_player;
        let mut actions = Vec::new();
        for position in self.end_positions(state) {
            let tile = self.tile(position);
            if state.collected_this_turn == 1
                && (tile.spirit_symbols != 1 || Some(tile.spirit) != state.collection_spirit)
            {
                continue;
            }
            match state.gemstones[position.index()] {
                Some(owner) if owner != player => {
                    let pool = state.gemstone_pools[player.index()];
                    if pool.available > 0 {
                        actions.push(SpiritsOfTheForestAction::TakeTile {
                            position,
                            sacrifice: Some(GemstoneSacrifice::Available),
                        });
                    }
                    for (index, gemstone) in state.gemstones.iter().enumerate() {
                        if *gemstone == Some(player) {
                            actions.push(SpiritsOfTheForestAction::TakeTile {
                                position,
                                sacrifice: Some(GemstoneSacrifice::Forest(ForestPosition(
                                    index as u8,
                                ))),
                            });
                        }
                    }
                }
                _ => actions.push(SpiritsOfTheForestAction::TakeTile {
                    position,
                    sacrifice: None,
                }),
            }
        }
        actions
    }

    fn finish_turn(&self, state: &mut SpiritsOfTheForestState) {
        state.completed_turns += 1;
        state.next_player = <Self as TwoPlayerZeroSumGame>::opponent(state.next_player)
            .expect("two-player state contains a valid player");
        state.phase = TurnPhase::Collect;
        state.collected_this_turn = 0;
        state.collection_spirit = None;
    }
}

impl Game for SpiritsOfTheForest {
    type State = SpiritsOfTheForestState;
    type Action = SpiritsOfTheForestAction;
    type Observation<'a> = &'a SpiritsOfTheForestState;
    type LegalActions<'a> = std::vec::IntoIter<SpiritsOfTheForestAction>;

    fn player_count(&self) -> u8 {
        2
    }

    fn initial_state(&self) -> Self::State {
        SpiritsOfTheForestState {
            remaining: [true; TILE_COUNT],
            gemstones: [None; TILE_COUNT],
            collections: [PlayerCollection::EMPTY; 2],
            gemstone_pools: [
                GemstonePool {
                    available: GEMSTONES_PER_PLAYER,
                    removed: 0,
                },
                GemstonePool {
                    available: GEMSTONES_PER_PLAYER,
                    removed: 0,
                },
            ],
            next_player: PlayerId::FIRST,
            phase: TurnPhase::Collect,
            collected_this_turn: 0,
            collection_spirit: None,
            completed_turns: 0,
        }
    }

    fn status(&self, state: &Self::State) -> PositionStatus {
        if state.remaining_tiles() == 0 {
            PositionStatus::Terminal
        } else {
            PositionStatus::PlayerTurn(state.next_player)
        }
    }

    fn is_turn_boundary(&self, state: &Self::State) -> bool {
        state.phase == TurnPhase::Collect && state.collected_this_turn == 0
    }
    fn legal_actions<'a>(&'a self, state: &'a Self::State) -> Self::LegalActions<'a> {
        if matches!(self.status(state), PositionStatus::Terminal) {
            return Vec::new().into_iter();
        }

        let actions = match state.phase {
            TurnPhase::Collect => {
                let mut takes = self.take_actions(state);
                if state.collected_this_turn == 1 || takes.is_empty() {
                    takes.push(SpiritsOfTheForestAction::EndCollection);
                }
                takes
            }
            TurnPhase::PlaceGemstone => {
                let player = state.next_player;
                let pool = state.gemstone_pools[player.index()];
                let mut choices = vec![SpiritsOfTheForestAction::SkipGemstone];
                if pool.available > 0 {
                    for index in 0..TILE_COUNT {
                        if state.remaining[index] && state.gemstones[index].is_none() {
                            choices.push(SpiritsOfTheForestAction::PlaceGemstone {
                                target: ForestPosition(index as u8),
                            });
                        }
                    }
                } else if pool.placed() > 0 {
                    for source in 0..TILE_COUNT {
                        if state.gemstones[source] != Some(player) {
                            continue;
                        }
                        for target in 0..TILE_COUNT {
                            if state.remaining[target] && state.gemstones[target].is_none() {
                                choices.push(SpiritsOfTheForestAction::MoveGemstone {
                                    source: ForestPosition(source as u8),
                                    target: ForestPosition(target as u8),
                                });
                            }
                        }
                    }
                }
                choices
            }
        };
        actions.into_iter()
    }

    fn apply_action(
        &self,
        state: &mut Self::State,
        action: &Self::Action,
    ) -> Result<(), IllegalAction> {
        if !self
            .legal_actions(state)
            .any(|candidate| candidate == *action)
        {
            return Err(IllegalAction::new(
                "action is not legal in the current Spirits of the Forest position",
            ));
        }

        let player = state.next_player;
        match *action {
            SpiritsOfTheForestAction::TakeTile {
                position,
                sacrifice,
            } => {
                if let Some(owner) = state.gemstones[position.index()] {
                    state.gemstones[position.index()] = None;
                    if owner == player {
                        state.gemstone_pools[player.index()].available += 1;
                    } else {
                        match sacrifice.expect("opponent reservation requires a sacrifice") {
                            GemstoneSacrifice::Available => {
                                state.gemstone_pools[player.index()].available -= 1;
                            }
                            GemstoneSacrifice::Forest(source) => {
                                state.gemstones[source.index()] = None;
                            }
                        }
                        state.gemstone_pools[player.index()].removed += 1;
                        state.gemstone_pools[owner.index()].available += 1;
                    }
                }

                let tile = self.tile(position);
                state.remaining[position.index()] = false;
                state.collections[player.index()].add(tile);
                state.collected_this_turn += 1;
                state.collection_spirit = Some(tile.spirit);

                if state.remaining_tiles() == 0 {
                    state.completed_turns += 1;
                } else {
                    let first_turn = state.completed_turns == 0;
                    if first_turn || tile.spirit_symbols == 2 || state.collected_this_turn == 2 {
                        state.phase = TurnPhase::PlaceGemstone;
                    }
                }
            }
            SpiritsOfTheForestAction::EndCollection => {
                state.phase = TurnPhase::PlaceGemstone;
            }
            SpiritsOfTheForestAction::PlaceGemstone { target } => {
                state.gemstone_pools[player.index()].available -= 1;
                state.gemstones[target.index()] = Some(player);
                self.finish_turn(state);
            }
            SpiritsOfTheForestAction::MoveGemstone { source, target } => {
                state.gemstones[source.index()] = None;
                state.gemstones[target.index()] = Some(player);
                self.finish_turn(state);
            }
            SpiritsOfTheForestAction::SkipGemstone => self.finish_turn(state),
        }
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
        if state.remaining_tiles() != 0 || player.index() >= 2 {
            return None;
        }
        Some(match self.winner(state) {
            Some(winner) if winner == player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        })
    }
}

impl HeuristicGame for SpiritsOfTheForest {
    fn heuristic_count(&self) -> u32 {
        1
    }

    fn heuristic_utility(&self, index: u32, state: &Self::State, player: PlayerId) -> Option<f32> {
        self.heuristic_utility_with_parameters(index, &HeuristicParameters::new(), state, player)
    }

    fn heuristic_parameter_specs(&self, index: u32) -> Option<&'static [HeuristicParameterSpec]> {
        (index == 0).then_some(&H0_PARAMETERS)
    }

    fn heuristic_utility_with_parameters(
        &self,
        index: u32,
        parameters: &HeuristicParameters,
        state: &Self::State,
        player: PlayerId,
    ) -> Option<f32> {
        if index != 0 || player.index() >= 2 {
            return None;
        }
        if let Some(utility) = self.terminal_utility(state, player) {
            return Some(utility);
        }
        let remaining_fraction = state.remaining_tiles() as f32 / TILE_COUNT as f32;
        let gemstone_early_bonus = H0_PARAMETERS[0].resolve(parameters) as f32;
        let gemstone_weight = 0.5 + gemstone_early_bonus * remaining_fraction * remaining_fraction;
        let opponent = <Self as TwoPlayerZeroSumGame>::opponent(player)?;
        let scores = self.reachable_progress_scores(state);
        let player_pool = state.gemstone_pools[player.index()];
        let opponent_pool = state.gemstone_pools[opponent.index()];
        let raw = f32::from(scores[player.index()] - scores[opponent.index()])
            + gemstone_weight * f32::from(player_pool.usable())
            - gemstone_weight * f32::from(opponent_pool.usable())
            + 0.25 * f32::from(player_pool.placed())
            - 0.25 * f32::from(opponent_pool.placed());
        Some((raw / 20.0).tanh())
    }
}

impl DeterministicGame for SpiritsOfTheForest {}
impl PerfectInformationGame for SpiritsOfTheForest {}
impl TwoPlayerZeroSumGame for SpiritsOfTheForest {}

fn score_category(scores: &mut [i16; 2], first: u8, second: u8) {
    if first == 0 {
        scores[0] -= 3;
    } else if first >= second {
        scores[0] += i16::from(first);
    }
    if second == 0 {
        scores[1] -= 3;
    } else if second >= first {
        scores[1] += i16::from(second);
    }
}

fn score_reachable_category(scores: &mut [i16; 2], collected: [u8; 2], remaining: u8) {
    let total = collected[0] + collected[1] + remaining;
    let threshold = total.div_ceil(2);
    for player in 0..2 {
        if collected[player] == 0 && remaining == 0 {
            scores[player] -= 3;
        } else if collected[player] + remaining >= threshold {
            scores[player] += i16::from(collected[player]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CounterRng(u64);

    impl RandomSource for CounterRng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.0
        }
    }

    fn position(row: u8, column: u8) -> ForestPosition {
        ForestPosition::new(row, column).unwrap()
    }

    fn take(position: ForestPosition) -> SpiritsOfTheForestAction {
        SpiritsOfTheForestAction::TakeTile {
            position,
            sacrifice: None,
        }
    }

    fn assert_state_invariants(game: &SpiritsOfTheForest, state: &SpiritsOfTheForestState) {
        let collected_tiles: usize = state
            .collections
            .iter()
            .map(|collection| usize::from(collection.tiles))
            .sum();
        assert_eq!(collected_tiles + state.remaining_tiles(), TILE_COUNT);

        for player in [PlayerId::FIRST, PlayerId::SECOND] {
            let forest_gemstones = state
                .gemstones
                .iter()
                .filter(|owner| **owner == Some(player))
                .count();
            let pool = state.gemstone_pools[player.index()];
            assert_eq!(forest_gemstones, usize::from(pool.placed()));
            assert_eq!(
                usize::from(pool.available()) + forest_gemstones + usize::from(pool.removed()),
                usize::from(GEMSTONES_PER_PLAYER)
            );
        }

        for (remaining, gemstone) in state.remaining.iter().zip(&state.gemstones) {
            assert!(*remaining || gemstone.is_none());
        }

        let mut spirit_symbols = [0_u8; 9];
        let mut power_sources = [0_u8; 3];
        for collection in &state.collections {
            for (total, collected) in spirit_symbols.iter_mut().zip(collection.spirit_symbols) {
                *total += collected;
            }
            for (total, collected) in power_sources.iter_mut().zip(collection.power_sources) {
                *total += collected;
            }
        }
        for (index, remaining) in state.remaining.iter().enumerate() {
            if !remaining {
                continue;
            }
            let tile = game.tiles[index];
            spirit_symbols[tile.spirit.index()] += tile.spirit_symbols;
            if let Some(source) = tile.power_source {
                power_sources[source.index()] += 1;
            }
        }
        assert_eq!(spirit_symbols, [5, 6, 6, 7, 7, 8, 8, 8, 10]);
        assert_eq!(power_sources, [9, 9, 9]);
    }

    #[test]
    fn supplied_tile_catalog_has_exact_composition() {
        assert_eq!(SPIRIT_TILES.len(), 48);
        let mut spirits = [0_u8; 9];
        let mut sources = [0_u8; 3];
        let mut composition = [[0_u8; 5]; 9];
        for tile in SPIRIT_TILES {
            spirits[tile.spirit.index()] += tile.spirit_symbols;
            if let Some(source) = tile.power_source {
                sources[source.index()] += 1;
            }
            let kind = match (tile.spirit_symbols, tile.power_source) {
                (1, None) => 0,
                (2, None) => 1,
                (1, Some(PowerSource::Fire)) => 2,
                (1, Some(PowerSource::Moon)) => 3,
                (1, Some(PowerSource::Sun)) => 4,
                _ => panic!("unsupported tile in the supplied catalog: {tile:?}"),
            };
            composition[tile.spirit.index()][kind] += 1;
        }
        assert_eq!(spirits, [5, 6, 6, 7, 7, 8, 8, 8, 10]);
        assert_eq!(sources, [9, 9, 9]);
        assert_eq!(
            composition,
            [
                [0, 1, 1, 1, 1],
                [0, 2, 0, 1, 1],
                [0, 1, 2, 1, 1],
                [0, 2, 1, 1, 1],
                [0, 2, 1, 1, 1],
                [1, 2, 1, 1, 1],
                [1, 2, 1, 1, 1],
                [1, 2, 1, 1, 1],
                [1, 3, 1, 1, 1],
            ]
        );
    }

    #[test]
    fn shuffle_is_reproducible() {
        let first = SpiritsOfTheForest::shuffled(&mut CounterRng(42));
        let repeated = SpiritsOfTheForest::shuffled(&mut CounterRng(42));
        let different = SpiritsOfTheForest::shuffled(&mut CounterRng(43));
        assert_eq!(first.tiles(), repeated.tiles());
        assert_ne!(first.tiles(), different.tiles());
    }

    #[test]
    fn only_row_ends_can_be_taken_and_first_turn_stops_after_one_tile() {
        let game = SpiritsOfTheForest::from_tiles(SPIRIT_TILES);
        let mut state = game.initial_state();
        let actions: Vec<_> = game.legal_actions(&state).collect();
        assert_eq!(actions.len(), 8);
        assert!(actions.contains(&take(position(0, 0))));
        assert!(actions.contains(&take(position(0, 11))));
        assert!(!actions.contains(&take(position(0, 1))));

        game.apply_action(&mut state, &take(position(0, 0)))
            .unwrap();
        assert_eq!(state.phase(), TurnPhase::PlaceGemstone);
        assert!(
            game.legal_actions(&state)
                .any(|action| action == SpiritsOfTheForestAction::SkipGemstone)
        );
    }

    #[test]
    fn two_matching_single_tiles_can_be_collected_sequentially() {
        let mut tiles = SPIRIT_TILES;
        tiles[0] = single(Spirit::Leaves);
        tiles[1] = single(Spirit::Leaves);
        let game = SpiritsOfTheForest::from_tiles(tiles);
        let mut state = game.initial_state();
        state.completed_turns = 1;

        game.apply_action(&mut state, &take(position(0, 0)))
            .unwrap();
        assert_eq!(state.phase(), TurnPhase::Collect);
        assert!(
            game.legal_actions(&state)
                .any(|action| action == take(position(0, 1)))
        );
        game.apply_action(&mut state, &take(position(0, 1)))
            .unwrap();
        assert_eq!(state.phase(), TurnPhase::PlaceGemstone);
        assert_eq!(state.collections[0].tiles(), 2);
    }

    #[test]
    fn second_matching_single_tile_can_come_from_another_row() {
        let mut tiles = SPIRIT_TILES;
        tiles[position(0, 0).index()] = single(Spirit::Leaves);
        tiles[position(1, 0).index()] = single(Spirit::Leaves);
        let game = SpiritsOfTheForest::from_tiles(tiles);
        let mut state = game.initial_state();
        state.completed_turns = 1;

        game.apply_action(&mut state, &take(position(0, 0)))
            .unwrap();
        assert!(
            game.legal_actions(&state)
                .any(|action| action == take(position(1, 0)))
        );
        game.apply_action(&mut state, &take(position(1, 0)))
            .unwrap();

        assert_eq!(state.phase(), TurnPhase::PlaceGemstone);
        assert_eq!(state.collections[0].tiles(), 2);
    }

    #[test]
    fn collection_can_end_while_a_second_matching_tile_is_available() {
        let mut tiles = SPIRIT_TILES;
        tiles[position(0, 0).index()] = single(Spirit::Leaves);
        tiles[position(1, 0).index()] = single(Spirit::Leaves);
        let game = SpiritsOfTheForest::from_tiles(tiles);
        let mut state = game.initial_state();
        state.completed_turns = 1;

        game.apply_action(&mut state, &take(position(0, 0)))
            .unwrap();
        let actions: Vec<_> = game.legal_actions(&state).collect();
        assert!(actions.contains(&take(position(1, 0))));
        assert!(actions.contains(&SpiritsOfTheForestAction::EndCollection));

        game.apply_action(&mut state, &SpiritsOfTheForestAction::EndCollection)
            .unwrap();
        assert_eq!(state.phase(), TurnPhase::PlaceGemstone);
        assert_eq!(state.collections[0].tiles(), 1);
    }

    #[test]
    fn a_physical_turn_keeps_the_player_until_the_gemstone_decision_finishes() {
        let mut tiles = SPIRIT_TILES;
        tiles[0] = single(Spirit::Leaves);
        tiles[1] = single(Spirit::Leaves);
        let game = SpiritsOfTheForest::from_tiles(tiles);
        let mut state = game.initial_state();
        state.completed_turns = 1;
        assert!(game.is_turn_boundary(&state));

        game.apply_action(&mut state, &take(position(0, 0)))
            .unwrap();
        assert_eq!(state.next_player(), PlayerId::FIRST);
        assert!(!game.is_turn_boundary(&state));
        assert_eq!(state.phase(), TurnPhase::Collect);

        game.apply_action(&mut state, &take(position(0, 1)))
            .unwrap();
        assert_eq!(state.next_player(), PlayerId::FIRST);
        assert!(!game.is_turn_boundary(&state));
        assert_eq!(state.phase(), TurnPhase::PlaceGemstone);

        game.apply_action(&mut state, &SpiritsOfTheForestAction::SkipGemstone)
            .unwrap();
        assert_eq!(state.next_player(), PlayerId::SECOND);
        assert_eq!(state.phase(), TurnPhase::Collect);
        assert_eq!(state.completed_turns(), 2);
        assert!(game.is_turn_boundary(&state));
    }

    #[test]
    fn collecting_an_opponents_reservation_removes_a_chosen_gem() {
        let game = SpiritsOfTheForest::from_tiles(SPIRIT_TILES);
        let mut state = game.initial_state();
        state.completed_turns = 1;
        state.gemstones[position(0, 0).index()] = Some(PlayerId::SECOND);
        state.gemstone_pools[1].available -= 1;
        let action = SpiritsOfTheForestAction::TakeTile {
            position: position(0, 0),
            sacrifice: Some(GemstoneSacrifice::Available),
        };

        game.apply_action(&mut state, &action).unwrap();
        assert_eq!(state.gemstone_pools[0].available(), 2);
        assert_eq!(state.gemstone_pools[0].removed(), 1);
        assert_eq!(state.gemstone_pools[1].available(), 3);
    }

    #[test]
    fn an_opponents_reservation_offers_only_owned_gems_as_sacrifices() {
        let game = SpiritsOfTheForest::from_tiles(SPIRIT_TILES);
        let mut state = game.initial_state();
        state.completed_turns = 1;
        let target = position(0, 0);
        let first_owned = position(1, 0);
        let second_owned = position(2, 0);
        state.gemstones[target.index()] = Some(PlayerId::SECOND);
        state.gemstones[first_owned.index()] = Some(PlayerId::FIRST);
        state.gemstones[second_owned.index()] = Some(PlayerId::FIRST);
        state.gemstone_pools[0].available = 1;
        state.gemstone_pools[1].available = 2;

        let target_actions: Vec<_> = game
            .legal_actions(&state)
            .filter(|action| {
                matches!(
                    action,
                    SpiritsOfTheForestAction::TakeTile { position, .. }
                        if *position == target
                )
            })
            .collect();

        assert_eq!(target_actions.len(), 3);
        assert!(
            target_actions.contains(&SpiritsOfTheForestAction::TakeTile {
                position: target,
                sacrifice: Some(GemstoneSacrifice::Available),
            })
        );
        for source in [first_owned, second_owned] {
            assert!(
                target_actions.contains(&SpiritsOfTheForestAction::TakeTile {
                    position: target,
                    sacrifice: Some(GemstoneSacrifice::Forest(source)),
                })
            );
        }
        assert!(
            !target_actions.contains(&SpiritsOfTheForestAction::TakeTile {
                position: target,
                sacrifice: Some(GemstoneSacrifice::Forest(position(0, 11))),
            })
        );
    }

    #[test]
    fn sacrificing_a_forest_gem_updates_both_players_pools() {
        let game = SpiritsOfTheForest::from_tiles(SPIRIT_TILES);
        let mut state = game.initial_state();
        state.completed_turns = 1;
        let target = position(0, 0);
        let sacrificed = position(1, 0);
        let retained = position(2, 0);
        state.gemstones[target.index()] = Some(PlayerId::SECOND);
        state.gemstones[sacrificed.index()] = Some(PlayerId::FIRST);
        state.gemstones[retained.index()] = Some(PlayerId::FIRST);
        state.gemstone_pools[0].available = 1;
        state.gemstone_pools[1].available = 2;

        game.apply_action(
            &mut state,
            &SpiritsOfTheForestAction::TakeTile {
                position: target,
                sacrifice: Some(GemstoneSacrifice::Forest(sacrificed)),
            },
        )
        .unwrap();

        assert_eq!(state.gemstones[target.index()], None);
        assert_eq!(state.gemstones[sacrificed.index()], None);
        assert_eq!(state.gemstones[retained.index()], Some(PlayerId::FIRST));
        assert_eq!(state.gemstone_pools[0].available(), 1);
        assert_eq!(state.gemstone_pools[0].placed(), 1);
        assert_eq!(state.gemstone_pools[0].removed(), 1);
        assert_eq!(state.gemstone_pools[1].available(), 3);
        assert_eq!(state.gemstone_pools[1].placed(), 0);
    }

    #[test]
    fn collecting_an_own_reservation_returns_and_reuses_the_gem() {
        let game = SpiritsOfTheForest::from_tiles(SPIRIT_TILES);
        let mut state = game.initial_state();
        state.completed_turns = 1;
        state.gemstones[position(0, 0).index()] = Some(PlayerId::FIRST);
        state.gemstone_pools[0].available = 2;

        game.apply_action(&mut state, &take(position(0, 0)))
            .unwrap();
        assert_eq!(state.gemstone_pools[0].available(), 3);
        assert!(
            game.legal_actions(&state)
                .any(|action| matches!(action, SpiritsOfTheForestAction::PlaceGemstone { .. }))
        );
    }

    #[test]
    fn a_reserved_tile_cannot_be_stolen_after_all_gems_are_removed() {
        let game = SpiritsOfTheForest::from_tiles(SPIRIT_TILES);
        let mut state = game.initial_state();
        state.completed_turns = 1;
        state.gemstones[position(0, 0).index()] = Some(PlayerId::SECOND);
        state.gemstone_pools[1].available = 2;
        state.gemstone_pools[0].available = 0;
        state.gemstone_pools[0].removed = 3;

        assert!(!game.legal_actions(&state).any(|action| matches!(
            action,
            SpiritsOfTheForestAction::TakeTile { position: target, .. }
                if target == position(0, 0)
        )));
    }

    #[test]
    fn placed_gems_can_move_only_when_the_supply_is_empty() {
        let game = SpiritsOfTheForest::from_tiles(SPIRIT_TILES);
        let mut state = game.initial_state();
        state.phase = TurnPhase::PlaceGemstone;
        state.gemstone_pools[0].available = 0;
        for column in 0..3 {
            state.gemstones[position(0, column).index()] = Some(PlayerId::FIRST);
        }

        assert!(game.legal_actions(&state).any(|action| matches!(
            action,
            SpiritsOfTheForestAction::MoveGemstone { source, target }
                if source == position(0, 0) && target == position(0, 3)
        )));
        assert!(
            !game
                .legal_actions(&state)
                .any(|action| matches!(action, SpiritsOfTheForestAction::PlaceGemstone { .. }))
        );
    }

    #[test]
    fn invalid_actions_leave_state_unchanged() {
        let game = SpiritsOfTheForest::from_tiles(SPIRIT_TILES);
        let mut state = game.initial_state();
        let before = state.clone();
        assert!(
            game.apply_action(&mut state, &take(position(0, 4)))
                .is_err()
        );
        assert_eq!(state, before);
    }

    #[test]
    fn taking_the_last_tile_ends_without_an_extra_gemstone_decision() {
        let game = SpiritsOfTheForest::from_tiles(SPIRIT_TILES);
        let mut state = game.initial_state();
        state.remaining = [false; TILE_COUNT];
        state.remaining[position(0, 0).index()] = true;
        state.completed_turns = 1;

        game.apply_action(&mut state, &take(position(0, 0)))
            .unwrap();

        assert_eq!(game.status(&state), PositionStatus::Terminal);
        assert_eq!(state.next_player(), PlayerId::FIRST);
        assert_eq!(state.remaining_tiles(), 0);
        assert_eq!(state.completed_turns(), 2);
        assert_eq!(game.legal_actions(&state).count(), 0);
    }

    #[test]
    fn reachable_states_preserve_game_invariants() {
        for seed in 0..128 {
            let mut setup_rng = CounterRng(seed);
            let game = SpiritsOfTheForest::shuffled(&mut setup_rng);
            let mut action_rng = CounterRng(seed ^ 0xa076_1d64_78bd_642f);
            let mut state = game.initial_state();

            loop {
                assert_state_invariants(&game, &state);
                if matches!(game.status(&state), PositionStatus::Terminal) {
                    assert_eq!(game.legal_actions(&state).count(), 0);
                    break;
                }

                let actions: Vec<_> = game.legal_actions(&state).collect();
                assert!(!actions.is_empty());
                for action in &actions {
                    let mut successor = state.clone();
                    game.apply_action(&mut successor, action).unwrap();
                    assert_state_invariants(&game, &successor);
                }

                let selected = action_rng.index(actions.len()).unwrap();
                game.apply_action(&mut state, &actions[selected]).unwrap();
            }
        }
    }

    #[test]
    fn scoring_handles_majorities_absence_and_ties() {
        let game = SpiritsOfTheForest::from_tiles(SPIRIT_TILES);
        let mut state = game.initial_state();
        state.collections[0].spirit_symbols = [2, 1, 0, 1, 1, 1, 1, 1, 1];
        state.collections[1].spirit_symbols = [1, 1, 1, 1, 1, 1, 1, 1, 1];
        state.collections[0].power_sources = [1, 0, 1];
        state.collections[1].power_sources = [0, 1, 1];
        assert_eq!(game.scores(&state), [5, 7]);
    }

    #[test]
    fn fewer_collected_tiles_break_an_equal_score() {
        let game = SpiritsOfTheForest::from_tiles(SPIRIT_TILES);
        let mut state = game.initial_state();
        state.remaining = [false; TILE_COUNT];
        state.collections[0].spirit_symbols = [1; 9];
        state.collections[1].spirit_symbols = [1; 9];
        state.collections[0].power_sources = [1; 3];
        state.collections[1].power_sources = [1; 3];
        state.collections[0].tiles = 23;
        state.collections[1].tiles = 25;
        assert_eq!(game.winner(&state), Some(PlayerId::FIRST));

        state.collections[0].tiles = 24;
        state.collections[1].tiles = 24;
        assert_eq!(game.winner(&state), None);
        assert_eq!(game.terminal_utility(&state, PlayerId::FIRST), Some(0.0));
    }

    #[test]
    fn heuristics_are_bounded_and_zero_sum() {
        let game = SpiritsOfTheForest::from_tiles(SPIRIT_TILES);
        let state = game.initial_state();
        assert_eq!(game.heuristic_count(), 1);
        for index in 0..game.heuristic_count() {
            let first = game
                .heuristic_utility(index, &state, PlayerId::FIRST)
                .unwrap();
            let second = game
                .heuristic_utility(index, &state, PlayerId::SECOND)
                .unwrap();
            assert!((-1.0..=1.0).contains(&first));
            assert_eq!(first, -second);
        }
        assert_eq!(game.heuristic_utility(1, &state, PlayerId::FIRST), None);
        assert_eq!(game.heuristic_utility(2, &state, PlayerId::FIRST), None);
    }

    #[test]
    fn reachable_progress_scores_only_categories_that_can_still_reach_half() {
        let mut scores = [0, 0];
        score_reachable_category(&mut scores, [2, 1], 2);
        assert_eq!(scores, [2, 1]);

        let mut scores = [0, 0];
        score_reachable_category(&mut scores, [2, 3], 0);
        assert_eq!(scores, [0, 3]);

        let mut scores = [0, 0];
        score_reachable_category(&mut scores, [0, 5], 0);
        assert_eq!(scores, [-3, 5]);
    }

    #[test]
    fn heuristic_zero_uses_reachable_progress() {
        let game = SpiritsOfTheForest::from_tiles(SPIRIT_TILES);
        let mut state = game.initial_state();
        state.remaining[0] = false;
        state.collections[0].add(game.tiles[0]);

        let reachable = game.heuristic_utility(0, &state, PlayerId::FIRST).unwrap();

        assert!(reachable > 0.0);
    }

    #[test]
    fn heuristic_zero_penalizes_early_gemstone_sacrifices_more() {
        let game = SpiritsOfTheForest::from_tiles(SPIRIT_TILES);
        let mut early = game.initial_state();
        early.gemstone_pools[0].available = 2;
        early.gemstone_pools[0].removed = 1;

        let early_conservation = game.heuristic_utility(0, &early, PlayerId::FIRST).unwrap();

        let mut late = early.clone();
        late.remaining[..TILE_COUNT - ROWS].fill(false);
        let late_conservation = game.heuristic_utility(0, &late, PlayerId::FIRST).unwrap();
        assert!(early_conservation < late_conservation);
    }

    #[test]
    fn heuristic_zero_exposes_a_defaulted_early_gemstone_bonus() {
        let game = SpiritsOfTheForest::from_tiles(SPIRIT_TILES);
        assert_eq!(
            game.heuristic_parameter_specs(0),
            Some(H0_PARAMETERS.as_slice())
        );
        assert_eq!(game.heuristic_parameter_specs(1), None);

        let mut state = game.initial_state();
        state.gemstone_pools[0].available = 2;
        state.gemstone_pools[0].removed = 1;
        let default = game.heuristic_utility(0, &state, PlayerId::FIRST).unwrap();
        let implicit = game
            .heuristic_utility_with_parameters(
                0,
                &HeuristicParameters::new(),
                &state,
                PlayerId::FIRST,
            )
            .unwrap();
        assert_eq!(default, implicit);

        let explicit_default =
            HeuristicParameters::from([("gemstone_early_bonus".to_owned(), 4.0)]);
        assert_eq!(
            game.heuristic_utility_with_parameters(0, &explicit_default, &state, PlayerId::FIRST,),
            Some(default)
        );

        let parameters = HeuristicParameters::from([("gemstone_early_bonus".to_owned(), 0.0)]);
        let constant_weight = game
            .heuristic_utility_with_parameters(0, &parameters, &state, PlayerId::FIRST)
            .unwrap();
        assert!(constant_weight > default);
    }
}
