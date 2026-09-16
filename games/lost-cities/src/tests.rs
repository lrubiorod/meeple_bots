use super::*;
use LostCitiesAction::*;
use LostCitiesCard::{Number as N, Wager as W};
use LostCitiesColor::*;
use meeple_bots_random_agent::RandomAgent;
use meeple_bots_simulation::{MatchConfig, SplitMix64, play_match_with_trace};
use std::{
    collections::HashSet,
    hash::{Hash, Hasher},
};
fn ready(seed: u64) -> LostCitiesState {
    let mut s = LostCities.initial_state();
    let mut rng = SplitMix64::new(seed);
    while LostCities.status(&s) == PositionStatus::Chance {
        let a = LostCities.sample_chance(&s, &mut rng).unwrap();
        LostCities.apply_chance_outcome(&mut s, &a).unwrap();
        LostCities.validate_state(&s).unwrap();
    }
    s
}
fn hash(v: &impl Hash) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}
#[test]
fn composition_and_setup() {
    let deck = full_deck();
    assert_eq!(deck.len(), 60);
    for c in LostCitiesColor::ALL {
        assert_eq!(deck.iter().filter(|&&x| x == W(c)).count(), 3);
        for n in 2..=10 {
            assert_eq!(deck.iter().filter(|&&x| x == N(c, n)).count(), 1);
        }
    }
    let s = ready(42);
    assert_eq!(s.hands.each_ref().map(Vec::len), [8, 8]);
    assert_eq!(s.deck.len(), 44);
    assert_eq!(ready(42), s);
    assert_eq!(LostCities.maximum_decision_horizon(), None);
}
#[test]
fn expedition_order_and_unique_actions() {
    let mut s = ready(1);
    s.hands[0] = vec![W(Red), W(Red), N(Red, 2), N(Red, 4), N(Red, 6)];
    for column in [vec![], vec![W(Red)], vec![W(Red), W(Red)]] {
        s.expeditions[0][0] = column;
        let a: Vec<_> = LostCities.legal_actions(&s).collect();
        assert!(a.contains(&Play(W(Red))));
        assert!(a.contains(&Play(N(Red, 2))));
        assert_eq!(a.iter().filter(|&&x| x == Play(W(Red))).count(), 1);
        assert_eq!(a.iter().filter(|&&x| x == Discard(W(Red))).count(), 1);
    }
    s.expeditions[0][0] = vec![N(Red, 4)];
    let a: Vec<_> = LostCities.legal_actions(&s).collect();
    assert!(a.contains(&Play(N(Red, 6))));
    for card in [W(Red), N(Red, 2), N(Red, 4)] {
        assert!(!a.contains(&Play(card)));
        assert!(a.contains(&Discard(card)));
    }
    let before = s.clone();
    assert!(LostCities.apply_action(&mut s, &Play(W(Red))).is_err());
    assert_eq!(before, s);
}
#[test]
fn discard_top_restriction_and_microturn() {
    let mut s = ready(4);
    let card = s.hands[0][0];
    LostCities.apply_action(&mut s, &Discard(card)).unwrap();
    assert_eq!(s.current_player, PlayerId::FIRST);
    assert_eq!(s.phase, Phase::Draw);
    assert_eq!(s.discards[card.color() as usize].last(), Some(&card));
    assert_eq!(s.hands[0].len(), 7);
    assert!(!LostCities.is_turn_boundary(&s));
    assert!(
        !LostCities
            .legal_actions(&s)
            .any(|a| a == DrawDiscard(card.color()))
    );
    LostCities.apply_action(&mut s, &DrawDeck).unwrap();
    assert_eq!(LostCities.status(&s), PositionStatus::Chance);
    assert!(LostCities.legal_actions(&s).next().is_none());
    let draw = LostCities
        .sample_chance(&s, &mut SplitMix64::new(4))
        .unwrap();
    let before = s.deck.len();
    assert!(LostCities.apply_action(&mut s, &draw).is_err());
    LostCities.apply_chance_outcome(&mut s, &draw).unwrap();
    assert_eq!(s.deck.len(), before - 1);
    assert_eq!(s.current_player, PlayerId::SECOND);
    assert!(LostCities.is_turn_boundary(&s));
    let play = LostCities
        .legal_actions(&s)
        .find(|a| matches!(a, Play(_)))
        .unwrap();
    LostCities.apply_action(&mut s, &play).unwrap();
    assert!(
        LostCities
            .legal_actions(&s)
            .any(|a| a == DrawDiscard(card.color()))
    );
    LostCities
        .apply_action(&mut s, &DrawDiscard(card.color()))
        .unwrap();
    assert!(s.discards[card.color() as usize].is_empty());
    assert!(s.hands[1].contains(&card));
    LostCities.validate_state(&s).unwrap();
}
#[test]
fn scoring_cases_and_terminal_tie() {
    let cases = [
        (vec![], 0),
        (
            vec![N(Red, 2), N(Red, 3), N(Red, 7), N(Red, 8), N(Red, 10)],
            10,
        ),
        (vec![N(Red, 4), N(Red, 6), N(Red, 7)], -3),
        (vec![W(Red), N(Red, 4), N(Red, 6), N(Red, 7)], -6),
        (
            vec![
                W(Red),
                W(Red),
                N(Red, 4),
                N(Red, 5),
                N(Red, 6),
                N(Red, 7),
                N(Red, 8),
                N(Red, 10),
            ],
            80,
        ),
        (vec![W(Red), W(Red), W(Red)], -80),
        ((2..=9).map(|n| N(Red, n)).collect(), 44),
    ];
    for (cards, expected) in cases {
        assert_eq!(expedition_score(&cards), expected);
    }
    let mut s = ready(5);
    s.phase = Phase::DrawChance;
    s.hands[0].pop();
    s.deck = vec![W(White)];
    LostCities
        .apply_chance_outcome(&mut s, &DealCard(W(White)))
        .unwrap();
    assert_eq!(s.phase, Phase::Finished);
    assert_eq!(LostCities.terminal_utility(&s, PlayerId::FIRST), Some(0.));
    assert!(LostCities.legal_actions(&s).next().is_none());
    assert!(
        LostCities
            .apply_chance_outcome(&mut s, &DealCard(W(White)))
            .is_err()
    );
    s.expeditions[0][0] = vec![N(Red, 10)];
    s.expeditions[0][1] = vec![N(Green, 9)];
    assert_eq!(s.scores(), [-21, 0]);
    assert_eq!(LostCities.terminal_utility(&s, PlayerId::FIRST), Some(-1.));
    assert_eq!(LostCities.terminal_utility(&s, PlayerId::SECOND), Some(1.));
}
#[test]
fn chance_multiset_weights_and_independent_future_draws() {
    let mut s = ready(8);
    s.phase = Phase::DrawChance;
    s.deck = vec![W(Red), W(Red), N(Red, 2)];
    let outcomes = LostCities.chance_outcomes(&s).unwrap();
    assert_eq!(
        outcomes,
        vec![(DealCard(W(Red)), 2. / 3.), (DealCard(N(Red, 2)), 1. / 3.)]
    );
    let mut draws = HashSet::new();
    let mut wagers = 0;
    for seed in 0..3000 {
        let a = LostCities
            .sample_chance(&s, &mut SplitMix64::new(seed))
            .unwrap();
        draws.insert(a);
        if a == DealCard(W(Red)) {
            wagers += 1;
        }
    }
    assert_eq!(draws.len(), 2);
    assert!((1850..2150).contains(&wagers));
    LostCities
        .apply_chance_outcome(&mut s, &DealCard(W(Red)))
        .unwrap();
    assert_eq!(s.deck, vec![W(Red), N(Red, 2)]);
}
#[test]
fn hidden_worlds_have_equal_observation_and_hash() {
    let a = ready(19);
    let mut b = a.clone();
    let i = b.deck.iter().position(|c| *c != b.hands[1][0]).unwrap();
    std::mem::swap(&mut b.hands[1][0], &mut b.deck[i]);
    b.hands[1].sort();
    b.deck.sort();
    assert_ne!(a, b);
    LostCities.validate_state(&b).unwrap();
    let o = LostCities.observation(&a, PlayerId::FIRST);
    assert_eq!(o, LostCities.observation(&b, PlayerId::FIRST));
    assert_eq!(hash(&o), hash(&LostCities.observation(&b, PlayerId::FIRST)));
    assert_eq!(o.hand, a.hands[0]);
    assert_eq!(o.opponent_hand_size, 8);
    assert_eq!(o.deck_size, 44);
    assert_ne!(o, LostCities.observation(&a, PlayerId::SECOND));
    assert_ne!(
        LostCities.observation(&a, PlayerId::SECOND),
        LostCities.observation(&b, PlayerId::SECOND)
    );
}
fn check_determinizations(s: &LostCitiesState) {
    for observer in [PlayerId::FIRST, PlayerId::SECOND] {
        let o = LostCities.observation(s, observer);
        for seed in 0..20 {
            let d = LostCities
                .sample_determinization(&o, observer, &mut SplitMix64::new(seed))
                .unwrap();
            LostCities.validate_state(&d).unwrap();
            assert_eq!(LostCities.observation(&d, observer), o);
            assert_eq!(d.hands[observer.index()], s.hands[observer.index()]);
            assert_eq!(d.expeditions, s.expeditions);
            assert_eq!(d.discards, s.discards);
            assert_eq!(d.phase, s.phase);
            assert_eq!(d.current_player, s.current_player);
            assert_eq!(d.deck.len(), s.deck.len());
            assert_eq!(
                d.hands[1 - observer.index()].len(),
                s.hands[1 - observer.index()].len()
            );
            if LostCities.status(s) == PositionStatus::PlayerTurn(observer) {
                assert_eq!(
                    LostCities.legal_actions(s).collect::<Vec<_>>(),
                    LostCities.legal_actions(&d).collect::<Vec<_>>()
                );
            }
        }
    }
}
#[test]
fn determinizations_roundtrip_all_phases_and_preserve_conservation() {
    let mut s = LostCities.initial_state();
    let mut rng = SplitMix64::new(91);
    let mut visited = HashSet::new();
    for _ in 0..2000 {
        visited.insert(s.phase);
        check_determinizations(&s);
        match LostCities.status(&s) {
            PositionStatus::Chance => {
                let a = LostCities.sample_chance(&s, &mut rng).unwrap();
                LostCities.apply_chance_outcome(&mut s, &a).unwrap();
            }
            PositionStatus::PlayerTurn(_) => {
                let actions: Vec<_> = LostCities.legal_actions(&s).collect();
                let a = actions[rng.index(actions.len()).unwrap()];
                LostCities.apply_action(&mut s, &a).unwrap();
            }
            PositionStatus::Terminal => break,
            _ => unreachable!(),
        }
    }
    assert!(visited.contains(&Phase::Draw));
    assert!(visited.contains(&Phase::DrawChance));
    assert!(visited.contains(&Phase::Finished));
}
#[test]
fn seeded_hidden_assignments_vary_without_fixing_future_order() {
    let s = ready(42);
    let o = LostCities.observation(&s, PlayerId::FIRST);
    let mut worlds = HashSet::new();
    for seed in 0..20 {
        worlds.insert(
            LostCities
                .sample_determinization(&o, PlayerId::FIRST, &mut SplitMix64::new(seed))
                .unwrap(),
        );
    }
    assert!(worlds.len() > 1);
    let mut d = LostCities
        .sample_determinization(&o, PlayerId::FIRST, &mut SplitMix64::new(8))
        .unwrap();
    assert_eq!(
        d,
        LostCities
            .sample_determinization(&o, PlayerId::FIRST, &mut SplitMix64::new(8))
            .unwrap()
    );
    let a = LostCities.legal_actions(&d).next().unwrap();
    LostCities.apply_action(&mut d, &a).unwrap();
    LostCities.apply_action(&mut d, &DrawDeck).unwrap();
    let draws: HashSet<_> = (0..20)
        .map(|seed| {
            LostCities
                .sample_chance(&d, &mut SplitMix64::new(seed))
                .unwrap()
        })
        .collect();
    assert!(draws.len() > 1);
    assert!(
        LostCities
            .sample_determinization(&o, PlayerId::SECOND, &mut SplitMix64::new(8))
            .is_err()
    );
    let mut bad = o.clone();
    bad.deck_size += 1;
    assert!(
        LostCities
            .sample_determinization(&bad, PlayerId::FIRST, &mut SplitMix64::new(8))
            .is_err()
    );
}
#[test]
fn random_matches_and_environment_rng_are_reproducible() {
    for seed in 0..4 {
        let cfg = MatchConfig {
            seed,
            ..Default::default()
        };
        let a =
            play_match_with_trace(&LostCities, &mut RandomAgent, &mut RandomAgent, cfg).unwrap();
        let b =
            play_match_with_trace(&LostCities, &mut RandomAgent, &mut RandomAgent, cfg).unwrap();
        assert_eq!(a.chance_events, b.chance_events);
        assert_eq!(a.result.utilities, b.result.utilities);
        assert!(a.chance_events.len() >= 60); // 16 setup cards plus all 44 deck draws.
    }
}

#[test]
fn agent_rng_consumption_cannot_change_environment_events() {
    use meeple_bots_core::{Agent, AgentError, DecisionContext};
    struct Burn {
        own: SplitMix64,
        count: usize,
    }
    impl Agent<LostCities> for Burn {
        fn select_action<R: RandomSource + ?Sized>(
            &mut self,
            d: DecisionContext<'_, LostCities>,
            rng: &mut R,
        ) -> Result<LostCitiesAction, AgentError> {
            for _ in 0..self.count {
                rng.next_u64();
            }
            RandomAgent.select_action(d, &mut self.own)
        }
    }
    let run = |count| {
        play_match_with_trace(
            &LostCities,
            &mut Burn {
                own: SplitMix64::new(123),
                count,
            },
            &mut Burn {
                own: SplitMix64::new(456),
                count,
            },
            MatchConfig::default(),
        )
        .unwrap()
    };
    let a = run(0);
    let b = run(17);
    assert_eq!(a.chance_events, b.chance_events);
    assert_eq!(a.result, b.result);
}
