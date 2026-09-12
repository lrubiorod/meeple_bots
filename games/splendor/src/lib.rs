//! Public-information, two-player base Splendor without blind reservation.
use meeple_bots_core::{
    Game, IllegalAction, PerfectInformationGame, PlayerId, PositionStatus, RandomSource,
    TwoPlayerZeroSumGame,
};
pub mod data;
use data::{CARDS, NOBLES};
pub type CardId = u8;
pub type NobleId = u8;
/// White, blue, green, red, black; gold is index 5 in token arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Card {
    pub id: CardId,
    pub tier: u8,
    pub bonus: u8,
    pub points: u8,
    pub cost: [u8; 5],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Noble {
    pub id: NobleId,
    pub requirements: [u8; 5],
    pub points: u8,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Player {
    pub tokens: [u8; 6],
    pub bonuses: [u8; 5],
    pub purchased: Vec<CardId>,
    pub reserved: Vec<CardId>,
    pub nobles: Vec<NobleId>,
    pub prestige: u8,
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SplendorState {
    pub bank: [u8; 6],
    pub players: [Player; 2],
    pub market: [[Option<CardId>; 4]; 3],
    /// Sorted sets, never a shuffled future sequence.
    pub remaining: [Vec<CardId>; 3],
    pub nobles: Vec<NobleId>,
    pub active: PlayerId,
    pub pending_refill: Option<(u8, u8)>,
    pub finished: bool,
    pub final_round: bool,
    /// Forced passes only; two consecutive blocked turns draw unless the final round ends.
    pub consecutive_passes: u8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Move {
    /// Legal only when no ordinary action is available.
    Pass,
    TakeDifferent {
        colors: u8,
    },
    TakeSame {
        color: u8,
    },
    ReserveVisible {
        tier: u8,
        slot: u8,
    },
    BuyVisible {
        tier: u8,
        slot: u8,
    },
    BuyReserved {
        index: u8,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SplendorAction {
    /// Complete player decision. Payment includes gold; returned includes newly taken tokens.
    Play {
        decision: Move,
        returned: [u8; 6],
        payment: [u8; 6],
        noble: Option<NobleId>,
    },
    /// Environment event only: excluded from legal_actions and rejected by apply_action.
    Refill { card: CardId },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Splendor {
    initial: SplendorState,
}
impl Splendor {
    pub fn new<R: RandomSource + ?Sized>(rng: &mut R) -> Self {
        let mut s = SplendorState {
            bank: [4, 4, 4, 4, 4, 5],
            players: Default::default(),
            market: [[None; 4]; 3],
            remaining: std::array::from_fn(|t| {
                CARDS
                    .iter()
                    .filter(|c| c.tier as usize == t + 1)
                    .map(|c| c.id)
                    .collect()
            }),
            nobles: Vec::new(),
            active: PlayerId::FIRST,
            pending_refill: None,
            finished: false,
            final_round: false,
            consecutive_passes: 0,
        };
        for t in 0..3 {
            for slot in 0..4 {
                let i = rng.index(s.remaining[t].len()).unwrap();
                s.market[t][slot] = Some(s.remaining[t].remove(i));
            }
        }
        let mut nobles: Vec<u8> = (0..10).collect();
        for _ in 0..3 {
            let i = rng.index(nobles.len()).unwrap();
            s.nobles.push(nobles.remove(i));
        }
        s.nobles.sort();
        Self { initial: s }
    }
    fn finish_turn(s: &mut SplendorState) {
        s.final_round |= s.players.iter().any(|p| p.prestige >= 15);
        s.finished = (s.final_round && s.active == PlayerId::SECOND) || s.consecutive_passes >= 2;
        s.active = Self::opponent(s.active).unwrap();
    }
    fn noble_choices(s: &SplendorState, bonuses: &[u8; 5]) -> Vec<Option<u8>> {
        let eligible: Vec<_> = s
            .nobles
            .iter()
            .copied()
            .filter(|&id| (0..5).all(|c| bonuses[c] >= NOBLES[id as usize].requirements[c]))
            .map(Some)
            .collect();
        if eligible.is_empty() {
            vec![None]
        } else {
            eligible
        }
    }
    fn payment_options(p: &Player, card: &Card) -> Vec<[u8; 6]> {
        fn visit(
            i: usize,
            need: &[u8; 5],
            tokens: &[u8; 6],
            pay: &mut [u8; 6],
            out: &mut Vec<[u8; 6]>,
        ) {
            if i == 5 {
                out.push(*pay);
                return;
            }
            for colored in 0..=need[i].min(tokens[i]) {
                let gold = need[i] - colored;
                if pay[5] + gold <= tokens[5] {
                    pay[i] = colored;
                    pay[5] += gold;
                    visit(i + 1, need, tokens, pay, out);
                    pay[5] -= gold;
                }
            }
        }
        let need = std::array::from_fn(|i| card.cost[i].saturating_sub(p.bonuses[i]));
        let mut out = Vec::new();
        visit(0, &need, &p.tokens, &mut [0; 6], &mut out);
        out
    }
    fn returns(tokens: [u8; 6]) -> Vec<[u8; 6]> {
        fn visit(i: usize, n: u8, t: &[u8; 6], r: &mut [u8; 6], out: &mut Vec<[u8; 6]>) {
            if i == 6 {
                if n == 0 {
                    out.push(*r);
                }
                return;
            }
            for v in 0..=n.min(t[i]) {
                r[i] = v;
                visit(i + 1, n - v, t, r, out);
            }
        }
        let mut out = Vec::new();
        visit(
            0,
            tokens.iter().sum::<u8>().saturating_sub(10),
            &tokens,
            &mut [0; 6],
            &mut out,
        );
        out
    }
    fn actions(s: &SplendorState) -> Vec<SplendorAction> {
        if s.finished || s.pending_refill.is_some() {
            return Vec::new();
        }
        let p = &s.players[s.active.index()];
        let mut out = Vec::new();
        let mut add_take = |decision: Move, taken: [u8; 6]| {
            let tokens = std::array::from_fn(|i| p.tokens[i] + taken[i]);
            for returned in Self::returns(tokens) {
                for noble in Self::noble_choices(s, &p.bonuses) {
                    out.push(SplendorAction::Play {
                        decision,
                        returned,
                        payment: [0; 6],
                        noble,
                    });
                }
            }
        };
        let available = (0..5).filter(|&i| s.bank[i] > 0).count().min(3) as u32;
        for mask in 1u8..32 {
            if mask.count_ones() == available
                && (0..5).all(|i| mask & (1 << i) == 0 || s.bank[i] > 0)
            {
                add_take(
                    Move::TakeDifferent { colors: mask },
                    std::array::from_fn(|i| u8::from(i < 5 && mask & (1 << i) != 0)),
                );
            }
        }
        for color in 0..5 {
            if s.bank[color] >= 4 {
                let mut taken = [0; 6];
                taken[color] = 2;
                add_take(Move::TakeSame { color: color as u8 }, taken);
            }
        }
        if p.reserved.len() < 3 {
            for tier in 0..3 {
                for slot in 0..4 {
                    if s.market[tier][slot].is_some() {
                        let mut taken = [0; 6];
                        taken[5] = u8::from(s.bank[5] > 0);
                        add_take(
                            Move::ReserveVisible {
                                tier: tier as u8,
                                slot: slot as u8,
                            },
                            taken,
                        );
                    }
                }
            }
        }
        let mut add_buy = |decision: Move, id: CardId| {
            let card = &CARDS[id as usize];
            let mut bonuses = p.bonuses;
            bonuses[card.bonus as usize] += 1;
            for payment in Self::payment_options(p, card) {
                for noble in Self::noble_choices(s, &bonuses) {
                    out.push(SplendorAction::Play {
                        decision,
                        returned: [0; 6],
                        payment,
                        noble,
                    });
                }
            }
        };
        for tier in 0..3 {
            for slot in 0..4 {
                if let Some(id) = s.market[tier][slot] {
                    add_buy(
                        Move::BuyVisible {
                            tier: tier as u8,
                            slot: slot as u8,
                        },
                        id,
                    );
                }
            }
        }
        for (index, &id) in p.reserved.iter().enumerate() {
            add_buy(Move::BuyReserved { index: index as u8 }, id);
        }
        if out.is_empty() {
            for noble in Self::noble_choices(s, &p.bonuses) {
                out.push(SplendorAction::Play {
                    decision: Move::Pass,
                    returned: [0; 6],
                    payment: [0; 6],
                    noble,
                });
            }
        }
        out
    }
}
impl Game for Splendor {
    type State = SplendorState;
    type Action = SplendorAction;
    type Observation<'a> = &'a SplendorState;
    type LegalActions<'a> = std::vec::IntoIter<SplendorAction>;
    fn player_count(&self) -> u8 {
        2
    }
    fn initial_state(&self) -> SplendorState {
        self.initial.clone()
    }
    fn status(&self, s: &SplendorState) -> PositionStatus {
        if s.pending_refill.is_some() {
            PositionStatus::Chance
        } else if s.finished {
            PositionStatus::Terminal
        } else {
            PositionStatus::PlayerTurn(s.active)
        }
    }
    fn legal_actions<'a>(&'a self, s: &'a SplendorState) -> Self::LegalActions<'a> {
        Self::actions(s).into_iter()
    }
    fn apply_action(&self, s: &mut SplendorState, a: &SplendorAction) -> Result<(), IllegalAction> {
        if !Self::actions(s).contains(a) {
            return Err(IllegalAction::new("illegal Splendor player action"));
        }
        let SplendorAction::Play {
            decision,
            returned,
            payment,
            noble,
        } = *a
        else {
            unreachable!()
        };
        s.consecutive_passes = if decision == Move::Pass {
            s.consecutive_passes + 1
        } else {
            0
        };
        let p = &mut s.players[s.active.index()];
        let mut taken = [0; 6];
        let mut bought = None;
        match decision {
            Move::Pass => {}
            Move::TakeDifferent { colors } => {
                for (i, v) in taken.iter_mut().enumerate().take(5) {
                    *v = u8::from(colors & (1 << i) != 0);
                }
            }
            Move::TakeSame { color } => taken[color as usize] = 2,
            Move::ReserveVisible { tier, slot } | Move::BuyVisible { tier, slot } => {
                let id = s.market[tier as usize][slot as usize].take().unwrap();
                if matches!(decision, Move::ReserveVisible { .. }) {
                    p.reserved.push(id);
                    taken[5] = u8::from(s.bank[5] > 0);
                } else {
                    bought = Some(id);
                }
                if !s.remaining[tier as usize].is_empty() {
                    s.pending_refill = Some((tier, slot));
                }
            }
            Move::BuyReserved { index } => bought = Some(p.reserved.remove(index as usize)),
        }
        for i in 0..6 {
            p.tokens[i] = p.tokens[i] + taken[i] - returned[i] - payment[i];
            s.bank[i] = s.bank[i] - taken[i] + returned[i] + payment[i];
        }
        if let Some(id) = bought {
            let c = CARDS[id as usize];
            p.purchased.push(id);
            p.purchased.sort();
            p.bonuses[c.bonus as usize] += 1;
            p.prestige += c.points;
        }
        if let Some(id) = noble {
            s.nobles.retain(|&n| n != id);
            p.nobles.push(id);
            p.nobles.sort();
            p.prestige += 3;
        }
        if s.pending_refill.is_none() {
            Self::finish_turn(s);
        }
        Ok(())
    }
    fn chance_outcomes(
        &self,
        s: &SplendorState,
    ) -> Result<Vec<(SplendorAction, f64)>, IllegalAction> {
        let (tier, _) = s
            .pending_refill
            .ok_or_else(|| IllegalAction::new("no pending refill"))?;
        let pool = &s.remaining[tier as usize];
        let probability = 1.0 / pool.len() as f64;
        Ok(pool
            .iter()
            .map(|&card| (SplendorAction::Refill { card }, probability))
            .collect())
    }
    fn apply_chance_outcome(
        &self,
        s: &mut SplendorState,
        a: &SplendorAction,
    ) -> Result<(), IllegalAction> {
        let (tier, slot) = s
            .pending_refill
            .ok_or_else(|| IllegalAction::new("no pending refill"))?;
        let SplendorAction::Refill { card } = *a else {
            return Err(IllegalAction::new("expected refill outcome"));
        };
        let i = s.remaining[tier as usize]
            .iter()
            .position(|&c| c == card)
            .ok_or_else(|| IllegalAction::new("card not in remaining tier"))?;
        s.remaining[tier as usize].remove(i);
        s.market[tier as usize][slot as usize] = Some(card);
        s.pending_refill = None;
        Self::finish_turn(s);
        Ok(())
    }
    fn observation<'a>(&'a self, s: &'a SplendorState, _: PlayerId) -> &'a SplendorState {
        s
    }
    fn terminal_utility(&self, s: &SplendorState, player: PlayerId) -> Option<f32> {
        if self.status(s) != PositionStatus::Terminal {
            return None;
        }
        Self::opponent(player)?;
        if s.consecutive_passes >= 2 && !s.final_round {
            return Some(0.0);
        }
        let a = &s.players[player.index()];
        let b = &s.players[Self::opponent(player)?.index()];
        Some(
            match a
                .prestige
                .cmp(&b.prestige)
                .then_with(|| b.purchased.len().cmp(&a.purchased.len()))
            {
                std::cmp::Ordering::Greater => 1.0,
                std::cmp::Ordering::Less => -1.0,
                std::cmp::Ordering::Equal => 0.0,
            },
        )
    }
}
impl PerfectInformationGame for Splendor {}
impl TwoPlayerZeroSumGame for Splendor {}

#[cfg(test)]
mod tests {
    use super::*;
    use meeple_bots_random_agent::RandomAgent;
    use meeple_bots_simulation::{MatchConfig, SplitMix64, play_match_with_trace};
    use std::num::NonZeroU32;
    fn setup() -> (Splendor, SplendorState) {
        let g = Splendor::new(&mut SplitMix64::new(42));
        let s = g.initial_state();
        (g, s)
    }
    fn play(decision: Move) -> SplendorAction {
        SplendorAction::Play {
            decision,
            returned: [0; 6],
            payment: [0; 6],
            noble: None,
        }
    }
    fn blocked_state() -> (Splendor, SplendorState) {
        let (g, mut s) = setup();
        s.bank = [0, 0, 0, 0, 0, 5];
        s.players[0].tokens = [4, 4, 2, 0, 0, 0];
        s.players[1].tokens = [0, 0, 2, 4, 4, 0];
        s.players[0].reserved = vec![72, 73, 76];
        s.players[1].reserved = vec![77, 80, 81];
        s.market = [
            [Some(0), Some(8), Some(16), Some(24)],
            [Some(44), Some(50), Some(56), Some(62)],
            [Some(71), Some(75), Some(79), Some(83)],
        ];
        s.remaining = std::array::from_fn(|tier| {
            CARDS
                .iter()
                .filter(|c| {
                    c.tier as usize == tier + 1
                        && !s.market.iter().flatten().any(|id| *id == Some(c.id))
                        && !s.players.iter().any(|p| p.reserved.contains(&c.id))
                })
                .map(|c| c.id)
                .collect()
        });
        (g, s)
    }

    #[test]
    fn two_consecutive_blocked_turns_draw_and_pass_is_forced() {
        let (g, mut s) = blocked_state();
        let blocked_game = Splendor { initial: s.clone() };
        let trace = play_match_with_trace(
            &blocked_game,
            &mut RandomAgent,
            &mut RandomAgent,
            MatchConfig::new(42, NonZeroU32::new(2).unwrap()),
        )
        .unwrap();
        assert_eq!(trace.result.utilities, vec![0.0, 0.0]);
        assert_eq!(trace.actions.len(), 2);
        assert!(trace.chance_events.is_empty());
        assert!(trace.actions.iter().all(|a| a.action == play(Move::Pass)));
        let pass = play(Move::Pass);
        assert_eq!(g.legal_actions(&s).collect::<Vec<_>>(), vec![pass]);
        g.apply_action(&mut s, &pass).unwrap();
        assert_eq!(s.consecutive_passes, 1);
        assert_eq!(g.status(&s), PositionStatus::PlayerTurn(PlayerId::SECOND));
        assert_eq!(g.legal_actions(&s).collect::<Vec<_>>(), vec![pass]);
        g.apply_action(&mut s, &pass).unwrap();
        assert_eq!(s.consecutive_passes, 2);
        assert_eq!(g.status(&s), PositionStatus::Terminal);
        assert_eq!(g.legal_actions(&s).count(), 0);
        // Blocking is a draw even when scores differ below the endgame threshold.
        s.players[0].prestige = 3;
        for p in [PlayerId::FIRST, PlayerId::SECOND] {
            assert_eq!(g.terminal_utility(&s, p), Some(0.0));
        }
        let (g, mut s) = setup();
        let original = s.clone();
        assert!(g.apply_action(&mut s, &pass).is_err());
        assert_eq!(s, original);
    }

    #[test]
    fn ordinary_action_resets_blockage_and_final_round_keeps_normal_scoring() {
        let (g, mut s) = blocked_state();
        g.apply_action(&mut s, &play(Move::Pass)).unwrap();
        s.players[0].tokens[1] -= 1;
        s.bank[1] += 1;
        let action = g.legal_actions(&s).next().unwrap();
        assert!(!matches!(
            action,
            SplendorAction::Play {
                decision: Move::Pass,
                ..
            }
        ));
        g.apply_action(&mut s, &action).unwrap();
        assert_eq!(s.consecutive_passes, 0);
        assert!(!s.finished);
        let (g, mut s) = blocked_state();
        s.players[0].prestige = 15;
        g.apply_action(&mut s, &play(Move::Pass)).unwrap();
        g.apply_action(&mut s, &play(Move::Pass)).unwrap();
        assert_eq!(g.terminal_utility(&s, PlayerId::FIRST), Some(1.0));
    }

    #[test]
    fn data_and_setup_invariants() {
        assert_eq!(CARDS.len(), 90);
        assert_eq!(NOBLES.len(), 10);
        for tier in 1..=3 {
            assert_eq!(
                CARDS.iter().filter(|c| c.tier == tier).count(),
                [40, 30, 20][tier as usize - 1]
            );
        }
        for (i, c) in CARDS.iter().enumerate() {
            assert_eq!(c.id as usize, i);
            assert!(c.bonus < 5);
            assert!(c.cost.iter().all(|&n| n <= 7));
        }
        for (i, n) in NOBLES.iter().enumerate() {
            assert_eq!(n.id as usize, i);
            assert_eq!(n.points, 3);
            assert!(n.requirements.iter().all(|&v| matches!(v, 0 | 3 | 4)));
        }
        let (g, s) = setup();
        assert_eq!(s, setup().1);
        assert_ne!(s, Splendor::new(&mut SplitMix64::new(43)).initial_state());
        assert_eq!(s.bank, [4, 4, 4, 4, 4, 5]);
        assert_eq!(s.nobles.len(), 3);
        assert_eq!(
            s.remaining.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![36, 26, 16]
        );
        assert_eq!(
            s.market.iter().flatten().filter(|c| c.is_some()).count(),
            12
        );
        let ids: std::collections::HashSet<_> = s
            .remaining
            .iter()
            .flatten()
            .copied()
            .chain(s.market.iter().flatten().flatten().copied())
            .collect();
        assert_eq!(ids.len(), 90);
        assert_eq!(g.status(&s), PositionStatus::PlayerTurn(PlayerId::FIRST));
    }
    #[test]
    fn takes_and_returns() {
        let (g, mut s) = setup();
        assert!(
            g.legal_actions(&s)
                .any(|a| a == play(Move::TakeSame { color: 0 }))
        );
        s.bank[0] = 3;
        assert!(
            !g.legal_actions(&s)
                .any(|a| a == play(Move::TakeSame { color: 0 }))
        );
        assert!(
            g.apply_action(&mut s, &play(Move::TakeDifferent { colors: 3 }))
                .is_err()
        );
        s.bank = [1, 1, 0, 0, 0, 5];
        assert!(
            g.legal_actions(&s)
                .any(|a| a == play(Move::TakeDifferent { colors: 3 }))
        );
        s.bank = [4; 6];
        s.players[0].tokens = [2, 2, 2, 2, 2, 0];
        let actions: Vec<_> = g
            .legal_actions(&s)
            .filter(|a| {
                matches!(
                    a,
                    SplendorAction::Play {
                        decision: Move::TakeDifferent { .. },
                        ..
                    }
                )
            })
            .collect();
        assert!(!actions.is_empty());
        for a in actions {
            let mut t = s.clone();
            g.apply_action(&mut t, &a).unwrap();
            assert_eq!(t.players[0].tokens.iter().sum::<u8>(), 10);
        }
    }
    #[test]
    fn reserve_refill_and_empty_pool() {
        let (g, mut s) = setup();
        let a = play(Move::ReserveVisible { tier: 0, slot: 0 });
        let card = s.market[0][0].unwrap();
        g.apply_action(&mut s, &a).unwrap();
        assert_eq!(s.players[0].reserved, vec![card]);
        assert_eq!(s.players[0].tokens[5], 1);
        assert_eq!(g.status(&s), PositionStatus::Chance);
        assert_eq!(g.legal_actions(&s).count(), 0);
        let outcomes = g.chance_outcomes(&s).unwrap();
        assert_eq!(outcomes.len(), 36);
        assert!(outcomes.iter().all(|(_, p)| *p == 1.0 / 36.0));
        let event = g.sample_chance(&s, &mut SplitMix64::new(8)).unwrap();
        assert_eq!(event, g.sample_chance(&s, &mut SplitMix64::new(8)).unwrap());
        assert!(g.apply_action(&mut s, &event).is_err());
        g.apply_chance_outcome(&mut s, &event).unwrap();
        assert_eq!(s.remaining[0].len(), 35);
        assert_eq!(s.active, PlayerId::SECOND);
        assert!(g.apply_chance_outcome(&mut s, &event).is_err());
        s.remaining[0].clear();
        s.bank[5] = 0;
        g.apply_action(&mut s, &a).unwrap();
        assert!(s.market[0][0].is_none());
        assert!(s.pending_refill.is_none());
        assert_eq!(s.players[1].tokens[5], 0);
        s.players[0].reserved = vec![0, 1, 2];
        assert!(!g.legal_actions(&s).any(|a| matches!(
            a,
            SplendorAction::Play {
                decision: Move::ReserveVisible { .. },
                ..
            }
        )));
    }
    #[test]
    fn discounts_gold_and_nobles() {
        let (g, mut s) = setup();
        s.players[0].reserved = vec![0]; // cost W1 B1 G1 R1
        s.players[0].bonuses = [1, 0, 0, 0, 0];
        s.players[0].tokens = [0, 1, 1, 0, 0, 1];
        let a = g
            .legal_actions(&s)
            .find(|a| {
                matches!(
                    a,
                    SplendorAction::Play {
                        decision: Move::BuyReserved { .. },
                        ..
                    }
                )
            })
            .unwrap();
        g.apply_action(&mut s, &a).unwrap();
        assert_eq!(s.players[0].tokens, [0; 6]);
        assert_eq!(s.players[0].bonuses, [1, 0, 0, 0, 1]);
        assert!(s.pending_refill.is_none());
        s.active = PlayerId::FIRST;
        s.players[0].bonuses = [4; 5];
        s.nobles = vec![0, 1];
        let choices: Vec<_> = g
            .legal_actions(&s)
            .filter(|a| {
                matches!(
                    a,
                    SplendorAction::Play {
                        decision: Move::TakeSame { color: 0 },
                        ..
                    }
                )
            })
            .collect();
        assert_eq!(choices.len(), 2);
        g.apply_action(&mut s, &choices[1]).unwrap();
        assert_eq!(s.players[0].nobles, vec![1]);
        assert_eq!(s.nobles, vec![0]);
    }
    #[test]
    fn end_round_and_tiebreak() {
        let (g, mut s) = setup();
        s.players[0].prestige = 15;
        g.apply_action(&mut s, &play(Move::TakeSame { color: 0 }))
            .unwrap();
        assert!(!s.finished);
        g.apply_action(&mut s, &play(Move::TakeSame { color: 1 }))
            .unwrap();
        assert!(s.finished);
        assert_eq!(g.terminal_utility(&s, PlayerId::FIRST), Some(1.0));
        s.players[1].prestige = 15;
        assert_eq!(g.terminal_utility(&s, PlayerId::FIRST), Some(0.0));
        s.players[0].purchased.push(1);
        assert_eq!(g.terminal_utility(&s, PlayerId::FIRST), Some(-1.0));
    }
    #[test]
    fn visible_purchase_refills_and_preserves_payment_choices() {
        let (g, mut s) = setup();
        s.market[0][0] = Some(0);
        s.remaining[0].retain(|&id| id != 0);
        s.players[0].tokens = [1, 1, 1, 1, 0, 1];
        let actions: Vec<_> = g
            .legal_actions(&s)
            .filter(|a| {
                matches!(
                    a,
                    SplendorAction::Play {
                        decision: Move::BuyVisible { tier: 0, slot: 0 },
                        ..
                    }
                )
            })
            .collect();
        assert_eq!(actions.len(), 5); // colored-only, or gold replacing any one of four colors
        let a = actions
            .iter()
            .find(
                |a| matches!(a,SplendorAction::Play{payment,..} if payment[5]==1 && payment[0]==0),
            )
            .unwrap();
        g.apply_action(&mut s, a).unwrap();
        assert_eq!(s.players[0].tokens[0], 1);
        assert_eq!(s.players[0].tokens[5], 0);
        assert_eq!(g.status(&s), PositionStatus::Chance);
        assert_eq!(s.players[0].bonuses[4], 1);
    }
    #[test]
    fn final_refill_precedes_terminal_and_single_noble_is_mandatory() {
        let (g, mut s) = setup();
        s.active = PlayerId::SECOND;
        s.players[1].prestige = 12;
        s.players[1].bonuses = [4; 5];
        s.nobles = vec![0];
        let action = g
            .legal_actions(&s)
            .find(|a| {
                matches!(
                    a,
                    SplendorAction::Play {
                        decision: Move::ReserveVisible { tier: 0, slot: 0 },
                        noble: Some(0),
                        ..
                    }
                )
            })
            .unwrap();
        let illegal = play(Move::ReserveVisible { tier: 0, slot: 0 });
        assert!(g.apply_action(&mut s, &illegal).is_err());
        g.apply_action(&mut s, &action).unwrap();
        assert_eq!(s.players[1].prestige, 15);
        assert_eq!(g.status(&s), PositionStatus::Chance);
        let event = g.sample_chance(&s, &mut SplitMix64::new(2)).unwrap();
        g.apply_chance_outcome(&mut s, &event).unwrap();
        assert_eq!(g.status(&s), PositionStatus::Terminal);
    }
    #[test]
    fn full_random_match_reproducible() {
        let (g, _) = setup();
        let config = MatchConfig {
            seed: 42,
            max_plies: NonZeroU32::new(10000).unwrap(),
        };
        let a = play_match_with_trace(&g, &mut RandomAgent, &mut RandomAgent, config).unwrap();
        let b = play_match_with_trace(&g, &mut RandomAgent, &mut RandomAgent, config).unwrap();
        assert_eq!(a.result, b.result);
        assert_eq!(
            a.actions.iter().map(|a| a.action).collect::<Vec<_>>(),
            b.actions.iter().map(|a| a.action).collect::<Vec<_>>()
        );
        assert!(!a.chance_events.is_empty());
        assert_eq!(
            a.chance_events.iter().map(|e| e.event).collect::<Vec<_>>(),
            b.chance_events.iter().map(|e| e.event).collect::<Vec<_>>()
        );
        let mut s = g.initial_state();
        let mut events = a.chance_events.iter().peekable();
        for (i, a) in a.actions.iter().enumerate() {
            g.apply_action(&mut s, &a.action).unwrap();
            while events.peek().is_some_and(|e| e.after_ply == i + 1) {
                g.apply_chance_outcome(&mut s, &events.next().unwrap().event)
                    .unwrap();
            }
        }
        assert!(s.finished);
    }
}

#[cfg(test)]
mod search_tests {
    use super::*;
    use meeple_bots_core::{Agent, AgentError, DecisionContext};
    use meeple_bots_mcts_agent::{
        MctsConfig, NeutralEvaluator, ReusableStochasticMctsAgent, SearchBudget, SelectionPolicy,
        StochasticMctsAgent, UniformRandom,
    };
    use meeple_bots_random_agent::RandomAgent;
    use meeple_bots_simulation::{MatchConfig, SplitMix64, play_match_with_trace};
    use std::num::NonZeroU32;
    #[test]
    fn complete_matches_with_both_selectors_reuse_and_transpositions() {
        for selection_policy in [SelectionPolicy::Uct, SelectionPolicy::Ucb1Tuned] {
            for (reuse, transpositions) in
                [(false, false), (false, true), (true, false), (true, true)]
            {
                let game = Splendor::new(&mut SplitMix64::new(42));
                let config = MctsConfig {
                    budget: SearchBudget::Iterations(NonZeroU32::new(8).unwrap()),
                    exploration: 1.0,
                    selection_policy,
                    rollout_depth: 20,
                    rollout_policy: UniformRandom,
                };
                let mut agent = ReusableStochasticMctsAgent::new(
                    StochasticMctsAgent::new(config, NeutralEvaluator),
                    reuse,
                    transpositions,
                );
                let result = play_match_with_trace(
                    &game,
                    &mut agent,
                    &mut RandomAgent,
                    MatchConfig {
                        seed: 42,
                        max_plies: NonZeroU32::new(3000).unwrap(),
                    },
                )
                .unwrap();
                assert!(!result.chance_events.is_empty());
                assert_eq!(result.result.utilities.len(), 2);
            }
        }
    }
    struct Burn {
        own: SplitMix64,
        count: usize,
    }
    impl Agent<Splendor> for Burn {
        fn select_action<R: RandomSource + ?Sized>(
            &mut self,
            d: DecisionContext<'_, Splendor>,
            rng: &mut R,
        ) -> Result<SplendorAction, AgentError> {
            for _ in 0..self.count {
                rng.next_u64();
            }
            RandomAgent.select_action(d, &mut self.own)
        }
    }
    #[test]
    fn agent_rng_consumption_does_not_change_environment_chance() {
        let g = Splendor::new(&mut SplitMix64::new(42));
        let cfg = MatchConfig {
            seed: 42,
            max_plies: NonZeroU32::new(10000).unwrap(),
        };
        let run = |count| {
            play_match_with_trace(
                &g,
                &mut Burn {
                    own: SplitMix64::new(123),
                    count,
                },
                &mut Burn {
                    own: SplitMix64::new(456),
                    count,
                },
                cfg,
            )
            .unwrap()
        };
        let a = run(0);
        let b = run(17);
        assert_eq!(a.chance_events, b.chance_events);
        assert_eq!(a.result, b.result);
        assert_eq!(
            a.actions.iter().map(|a| a.action).collect::<Vec<_>>(),
            b.actions.iter().map(|a| a.action).collect::<Vec<_>>()
        );
    }
}

/// H0: prestige only. H1: prestige, bounded discounts and visible noble proximity.
impl meeple_bots_core::HeuristicGame for Splendor {
    fn heuristic_count(&self) -> u32 {
        2
    }
    fn heuristic_utility(&self, index: u32, state: &Self::State, player: PlayerId) -> Option<f32> {
        if index >= self.heuristic_count() {
            return None;
        }
        let opponent = Self::opponent(player)?;
        if let Some(value) = self.terminal_utility(state, player) {
            return Some(value);
        }
        let score = |player: &Player| {
            let prestige = f32::from(player.prestige);
            if index == 0 {
                return prestige;
            }
            // Saturation limits engine value and discourages collecting one color forever.
            let discounts: f32 = player.bonuses.iter().map(|&n| f32::from(n.min(4))).sum();
            // Only the closest remaining noble contributes: overlapping requirements
            // should not multiply the reward for the same development cards.
            let noble_potential = state
                .nobles
                .iter()
                .map(|&id| {
                    let noble = &NOBLES[id as usize];
                    let missing: u16 = noble
                        .requirements
                        .iter()
                        .zip(player.bonuses)
                        .map(|(&required, bonus)| u16::from(required.saturating_sub(bonus)))
                        .sum();
                    f32::from(noble.points) / (1.0 + f32::from(missing))
                })
                .fold(0.0_f32, f32::max);
            prestige + 0.15 * discounts + noble_potential
        };
        let difference =
            score(&state.players[player.index()]) - score(&state.players[opponent.index()]);
        Some(difference / (15.0 + difference.abs()))
    }
}

#[cfg(test)]
mod prestige_tests {
    use super::*;
    use meeple_bots_core::HeuristicGame;
    use meeple_bots_simulation::SplitMix64;
    #[test]
    fn prestige_is_symmetric_bounded_and_terminal_aware() {
        let game = Splendor::new(&mut SplitMix64::new(42));
        let mut state = game.initial_state();
        assert_eq!(
            game.heuristic_utility(0, &state, PlayerId::FIRST),
            Some(0.0)
        );
        state.players[0].prestige = 10;
        state.players[1].prestige = 5;
        assert_eq!(
            game.heuristic_utility(0, &state, PlayerId::FIRST),
            Some(0.25)
        );
        assert_eq!(
            game.heuristic_utility(0, &state, PlayerId::SECOND),
            Some(-0.25)
        );
        state.players[0].prestige = 255;
        assert!(game.heuristic_utility(0, &state, PlayerId::FIRST).unwrap() < 1.0);
        assert_eq!(game.heuristic_utility(2, &state, PlayerId::FIRST), None);
        state.finished = true;
        state.final_round = true;
        assert_eq!(
            game.heuristic_utility(0, &state, PlayerId::FIRST),
            Some(1.0)
        );
        state.players[1].prestige = 255;
        assert_eq!(
            game.heuristic_utility(0, &state, PlayerId::FIRST),
            Some(0.0)
        );
    }

    #[test]
    fn h1_prefers_progress_toward_visible_nobles() {
        let game = Splendor::new(&mut SplitMix64::new(42));
        let mut state = game.initial_state();
        state.nobles = vec![0];
        let requirements = NOBLES[0].requirements;
        let relevant = requirements.iter().position(|&n| n > 0).unwrap();
        let irrelevant = requirements.iter().position(|&n| n == 0).unwrap();
        state.players[0].bonuses[relevant] = 1;
        state.players[1].bonuses[irrelevant] = 1;
        let value = game.heuristic_utility(1, &state, PlayerId::FIRST).unwrap();
        assert!(value > 0.0);
        assert_eq!(
            game.heuristic_utility(1, &state, PlayerId::SECOND),
            Some(-value)
        );
        assert_eq!(
            game.heuristic_utility(0, &state, PlayerId::FIRST),
            Some(0.0)
        );
        // An unavailable noble must no longer influence color preferences.
        state.nobles.clear();
        assert_eq!(
            game.heuristic_utility(1, &state, PlayerId::FIRST),
            Some(0.0)
        );
    }

    #[test]
    fn h1_rewards_getting_close_without_counting_tokens_or_reserves() {
        let game = Splendor::new(&mut SplitMix64::new(42));
        let mut state = game.initial_state();
        state.nobles = vec![0];
        let baseline = game.heuristic_utility(1, &state, PlayerId::FIRST);
        state.players[0].tokens = [1, 1, 1, 1, 1, 5];
        state.players[0]
            .reserved
            .push(state.market[0][0].take().unwrap());
        assert_eq!(game.heuristic_utility(1, &state, PlayerId::FIRST), baseline);
        state.players[0].bonuses = NOBLES[0].requirements;
        let color = NOBLES[0].requirements.iter().position(|&n| n >= 3).unwrap();
        let mut values = Vec::new();
        for missing in (1..=3).rev() {
            state.players[0].bonuses[color] = NOBLES[0].requirements[color] - missing;
            values.push(game.heuristic_utility(1, &state, PlayerId::FIRST).unwrap());
        }
        assert!(values[1] > values[0]);
        assert!(values[2] - values[1] > values[1] - values[0]);
    }

    #[test]
    fn h1_is_bounded_and_respects_terminal_tiebreaks() {
        let game = Splendor::new(&mut SplitMix64::new(42));
        let mut state = game.initial_state();
        assert_eq!(
            game.heuristic_utility(1, &state, PlayerId::FIRST),
            Some(0.0)
        );
        state.players[0].bonuses = [255; 5];
        state.players[0].prestige = 255;
        let value = game.heuristic_utility(1, &state, PlayerId::FIRST).unwrap();
        assert!(value > 0.0 && value < 1.0);
        assert_eq!(
            game.heuristic_utility(1, &state, PlayerId::SECOND),
            Some(-value)
        );
        state.finished = true;
        state.final_round = true;
        assert_eq!(
            game.heuristic_utility(1, &state, PlayerId::FIRST),
            Some(1.0)
        );
        state.players[1].prestige = 255;
        assert_eq!(
            game.heuristic_utility(1, &state, PlayerId::FIRST),
            Some(0.0)
        );
        state.players[0].purchased.push(0);
        assert_eq!(
            game.heuristic_utility(1, &state, PlayerId::FIRST),
            Some(-1.0)
        );
    }
}
