//! Lost Cities single-round experimental variant: five colors, 60 cards, two players.
//! State is authoritative. Only Observation is suitable as input to a hidden-information policy.
use meeple_bots_core::{
    Game, IllegalAction, ImperfectInformationGame, PlayerId, PositionStatus, RandomSource,
    TwoPlayerZeroSumGame,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[repr(u8)]
pub enum LostCitiesColor {
    Red,
    Green,
    Blue,
    Yellow,
    White,
}
impl LostCitiesColor {
    pub const ALL: [Self; 5] = [
        Self::Red,
        Self::Green,
        Self::Blue,
        Self::Yellow,
        Self::White,
    ];
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum LostCitiesCard {
    Wager(LostCitiesColor),
    Number(LostCitiesColor, u8),
}
impl LostCitiesCard {
    pub fn color(self) -> LostCitiesColor {
        match self {
            Self::Wager(c) | Self::Number(c, _) => c,
        }
    }
    pub fn value(self) -> u8 {
        match self {
            Self::Wager(_) => 0,
            Self::Number(_, n) => n,
        }
    }
    pub fn valid(self) -> bool {
        matches!(self, Self::Wager(_)) || (2..=10).contains(&self.value())
    }
}
pub fn full_deck() -> Vec<LostCitiesCard> {
    let mut cards = Vec::with_capacity(60);
    for color in LostCitiesColor::ALL {
        cards.extend([LostCitiesCard::Wager(color); 3]);
        cards.extend((2..=10).map(|n| LostCitiesCard::Number(color, n)));
    }
    cards.sort();
    cards
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Phase {
    Deal(u8),
    Play,
    Draw,
    DrawChance,
    Finished,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum LostCitiesAction {
    Play(LostCitiesCard),
    Discard(LostCitiesCard),
    DrawDeck,
    DrawDiscard(LostCitiesColor),
    /// Private environment event, never a player action. Authoritative traces are not observations.
    DealCard(LostCitiesCard),
}
pub type Expeditions = [[Vec<LostCitiesCard>; 5]; 2];
pub type Discards = [Vec<LostCitiesCard>; 5];
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct LostCitiesState {
    pub current_player: PlayerId,
    pub phase: Phase,
    pub hands: [Vec<LostCitiesCard>; 2],
    pub expeditions: Expeditions,
    pub discards: Discards,
    /// Canonical multiset, NOT a future draw order.
    pub deck: Vec<LostCitiesCard>,
    pub blocked_discard: Option<LostCitiesColor>,
}
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct LostCitiesObservation {
    pub observer: PlayerId,
    pub current_player: PlayerId,
    pub phase: Phase,
    pub hand: Vec<LostCitiesCard>,
    pub opponent_hand_size: usize,
    pub deck_size: usize,
    pub expeditions: Expeditions,
    pub discards: Discards,
    pub blocked_discard: Option<LostCitiesColor>,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LostCities;
impl TwoPlayerZeroSumGame for LostCities {}

pub fn expedition_score(cards: &[LostCitiesCard]) -> i16 {
    if cards.is_empty() {
        return 0;
    }
    let sum: i16 = cards.iter().map(|c| i16::from(c.value())).sum();
    let wagers = cards
        .iter()
        .filter(|c| matches!(c, LostCitiesCard::Wager(_)))
        .count() as i16;
    (sum - 20) * (1 + wagers) + if cards.len() >= 8 { 20 } else { 0 }
}
impl LostCitiesState {
    pub fn scores(&self) -> [i16; 2] {
        self.expeditions
            .each_ref()
            .map(|e| e.iter().map(|c| expedition_score(c)).sum())
    }
}
fn accepts(column: &[LostCitiesCard], card: LostCitiesCard) -> bool {
    match column.last() {
        None | Some(LostCitiesCard::Wager(_)) => true,
        Some(LostCitiesCard::Number(_, n)) => card.value() > *n,
    }
}
fn remove(cards: &mut Vec<LostCitiesCard>, card: LostCitiesCard) -> Result<(), IllegalAction> {
    let index = cards
        .iter()
        .position(|c| *c == card)
        .ok_or_else(|| IllegalAction::new("card absent from pool"))?;
    cards.remove(index);
    Ok(())
}
fn finish_draw(s: &mut LostCitiesState) {
    s.blocked_discard = None;
    if s.deck.is_empty() {
        s.phase = Phase::Finished;
    } else {
        s.current_player = LostCities::opponent(s.current_player).unwrap();
        s.phase = Phase::Play;
    }
}
impl Game for LostCities {
    type State = LostCitiesState;
    type Action = LostCitiesAction;
    type Observation<'a> = LostCitiesObservation;
    type LegalActions<'a> = std::vec::IntoIter<LostCitiesAction>;
    fn player_count(&self) -> u8 {
        2
    }
    fn initial_state(&self) -> Self::State {
        LostCitiesState {
            current_player: PlayerId::FIRST,
            phase: Phase::Deal(0),
            hands: Default::default(),
            expeditions: Default::default(),
            discards: Default::default(),
            deck: full_deck(),
            blocked_discard: None,
        }
    }
    fn status(&self, s: &Self::State) -> PositionStatus {
        match s.phase {
            Phase::Deal(_) | Phase::DrawChance => PositionStatus::Chance,
            Phase::Finished => PositionStatus::Terminal,
            _ => PositionStatus::PlayerTurn(s.current_player),
        }
    }
    fn is_turn_boundary(&self, s: &Self::State) -> bool {
        matches!(s.phase, Phase::Deal(_) | Phase::Play | Phase::Finished)
    }
    fn legal_actions<'a>(&'a self, s: &'a Self::State) -> Self::LegalActions<'a> {
        let mut actions = Vec::new();
        match s.phase {
            Phase::Play => {
                let mut hand = s.hands[s.current_player.index()].clone();
                hand.sort();
                hand.dedup();
                for card in hand {
                    if accepts(
                        &s.expeditions[s.current_player.index()][card.color() as usize],
                        card,
                    ) {
                        actions.push(LostCitiesAction::Play(card));
                    }
                    actions.push(LostCitiesAction::Discard(card));
                }
            }
            Phase::Draw => {
                if !s.deck.is_empty() {
                    actions.push(LostCitiesAction::DrawDeck);
                }
                for color in LostCitiesColor::ALL {
                    if s.blocked_discard != Some(color) && !s.discards[color as usize].is_empty() {
                        actions.push(LostCitiesAction::DrawDiscard(color));
                    }
                }
            }
            _ => {}
        }
        actions.into_iter()
    }
    fn apply_action(&self, s: &mut Self::State, a: &Self::Action) -> Result<(), IllegalAction> {
        if !self.legal_actions(s).any(|legal| legal == *a) {
            return Err(IllegalAction::new("illegal Lost Cities player action"));
        }
        match *a {
            LostCitiesAction::Play(card) | LostCitiesAction::Discard(card) => {
                remove(&mut s.hands[s.current_player.index()], card)?;
                if matches!(a, LostCitiesAction::Play(_)) {
                    s.expeditions[s.current_player.index()][card.color() as usize].push(card);
                } else {
                    s.discards[card.color() as usize].push(card);
                    s.blocked_discard = Some(card.color());
                }
                s.phase = Phase::Draw;
            }
            LostCitiesAction::DrawDeck => s.phase = Phase::DrawChance,
            LostCitiesAction::DrawDiscard(color) => {
                let card = s.discards[color as usize].pop().unwrap();
                s.hands[s.current_player.index()].push(card);
                s.hands[s.current_player.index()].sort();
                finish_draw(s);
            }
            LostCitiesAction::DealCard(_) => unreachable!(),
        }
        Ok(())
    }
    fn chance_outcomes(&self, s: &Self::State) -> Result<Vec<(Self::Action, f64)>, IllegalAction> {
        if self.status(s) != PositionStatus::Chance || s.deck.is_empty() {
            return Err(IllegalAction::new("not a drawable chance position"));
        }
        let mut counts = std::collections::BTreeMap::new();
        for card in &s.deck {
            *counts.entry(*card).or_insert(0u32) += 1;
        }
        Ok(counts
            .into_iter()
            .map(|(card, n)| {
                (
                    LostCitiesAction::DealCard(card),
                    f64::from(n) / s.deck.len() as f64,
                )
            })
            .collect())
    }
    fn sample_chance<R: RandomSource + ?Sized>(
        &self,
        s: &Self::State,
        rng: &mut R,
    ) -> Result<Self::Action, IllegalAction> {
        if self.status(s) != PositionStatus::Chance {
            return Err(IllegalAction::new("not a chance position"));
        }
        let index = rng
            .index(s.deck.len())
            .ok_or_else(|| IllegalAction::new("empty draw pool"))?;
        Ok(LostCitiesAction::DealCard(s.deck[index]))
    }
    fn apply_chance_outcome(
        &self,
        s: &mut Self::State,
        a: &Self::Action,
    ) -> Result<(), IllegalAction> {
        if self.status(s) != PositionStatus::Chance {
            return Err(IllegalAction::new("not a chance position"));
        }
        let LostCitiesAction::DealCard(card) = *a else {
            return Err(IllegalAction::new("expected a dealt card"));
        };
        remove(&mut s.deck, card)?;
        let recipient = if let Phase::Deal(n) = s.phase {
            usize::from(n % 2)
        } else {
            s.current_player.index()
        };
        s.hands[recipient].push(card);
        s.hands[recipient].sort();
        if let Phase::Deal(n) = s.phase {
            s.phase = if n == 15 {
                Phase::Play
            } else {
                Phase::Deal(n + 1)
            };
        } else {
            finish_draw(s);
        }
        Ok(())
    }
    fn observation<'a>(&'a self, s: &'a Self::State, observer: PlayerId) -> LostCitiesObservation {
        assert!(observer.index() < 2, "Lost Cities has two players");
        LostCitiesObservation {
            observer,
            current_player: s.current_player,
            phase: s.phase,
            hand: s.hands[observer.index()].clone(),
            opponent_hand_size: s.hands[1 - observer.index()].len(),
            deck_size: s.deck.len(),
            expeditions: s.expeditions.clone(),
            discards: s.discards.clone(),
            blocked_discard: s.blocked_discard,
        }
    }
    fn terminal_utility(&self, s: &Self::State, player: PlayerId) -> Option<f32> {
        if s.phase != Phase::Finished || player.index() >= 2 {
            return None;
        }
        let scores = s.scores();
        Some(
            match scores[player.index()].cmp(&scores[1 - player.index()]) {
                std::cmp::Ordering::Greater => 1.,
                std::cmp::Ordering::Less => -1.,
                std::cmp::Ordering::Equal => 0.,
            },
        )
    }
}
impl ImperfectInformationGame for LostCities {
    fn sample_determinization<R: RandomSource + ?Sized>(
        &self,
        o: &LostCitiesObservation,
        observer: PlayerId,
        rng: &mut R,
    ) -> Result<LostCitiesState, IllegalAction> {
        if observer.index() >= 2 || observer != o.observer {
            return Err(IllegalAction::new(
                "observation belongs to a different player",
            ));
        }
        let mut unknown = full_deck();
        for card in o
            .hand
            .iter()
            .chain(o.expeditions.iter().flatten().flatten())
            .chain(o.discards.iter().flatten())
        {
            remove(&mut unknown, *card)?;
        }
        if o.opponent_hand_size > unknown.len()
            || unknown.len() - o.opponent_hand_size != o.deck_size
        {
            return Err(IllegalAction::new("inconsistent hidden card counts"));
        }
        let mut hands: [Vec<LostCitiesCard>; 2] = Default::default();
        hands[observer.index()] = o.hand.clone();
        for _ in 0..o.opponent_hand_size {
            let index = rng.index(unknown.len()).unwrap();
            hands[1 - observer.index()].push(unknown.remove(index));
        }
        hands[1 - observer.index()].sort();
        let state = LostCitiesState {
            current_player: o.current_player,
            phase: o.phase,
            hands,
            expeditions: o.expeditions.clone(),
            discards: o.discards.clone(),
            deck: unknown,
            blocked_discard: o.blocked_discard,
        };
        self.validate_state(&state)?;
        Ok(state)
    }
}
impl LostCities {
    /// Validate externally reconstructed states as well as conservation in tests.
    pub fn validate_state(&self, s: &LostCitiesState) -> Result<(), IllegalAction> {
        let bad = || IllegalAction::new("invalid Lost Cities state");
        if s.current_player.index() >= 2 {
            return Err(bad());
        }
        let mut cards: Vec<_> = s
            .hands
            .iter()
            .flatten()
            .chain(s.expeditions.iter().flatten().flatten())
            .chain(s.discards.iter().flatten())
            .chain(&s.deck)
            .copied()
            .collect();
        cards.sort();
        if cards != full_deck() {
            return Err(bad());
        }
        for (color, pile) in s.discards.iter().enumerate() {
            if pile.iter().any(|c| c.color() as usize != color) {
                return Err(bad());
            }
        }
        for columns in &s.expeditions {
            for (color, column) in columns.iter().enumerate() {
                for (i, card) in column.iter().enumerate() {
                    if card.color() as usize != color || !accepts(&column[..i], *card) {
                        return Err(bad());
                    }
                }
            }
        }
        let sizes = s.hands.each_ref().map(Vec::len);
        match s.phase {
            Phase::Deal(n) => {
                if n >= 16
                    || sizes != [usize::from(n.div_ceil(2)), usize::from(n / 2)]
                    || s.deck.len() != 60 - usize::from(n)
                    || s.current_player != PlayerId::FIRST
                {
                    return Err(bad());
                }
            }
            Phase::Play | Phase::Finished => {
                if sizes != [8, 8] {
                    return Err(bad());
                }
            }
            Phase::Draw | Phase::DrawChance => {
                if sizes[s.current_player.index()] != 7 || sizes[1 - s.current_player.index()] != 8
                {
                    return Err(bad());
                }
            }
        }
        if (s.phase == Phase::Finished) != s.deck.is_empty() {
            return Err(bad());
        }
        if let Some(color) = s.blocked_discard {
            if !matches!(s.phase, Phase::Draw | Phase::DrawChance)
                || s.discards[color as usize].is_empty()
            {
                return Err(bad());
            }
        }
        if s.hands
            .iter()
            .chain(std::iter::once(&s.deck))
            .any(|c| !c.is_sorted())
        {
            return Err(bad());
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests;
