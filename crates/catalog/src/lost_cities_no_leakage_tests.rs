//! Information invariance through the real participant, including trusted callbacks.
use super::*;
use meeple_bots_core::{BanditPolicy, SearchBudget};
use meeple_bots_lost_cities::LostCitiesAction as A;
use meeple_bots_simulation::SplitMix64;
use meeple_bots_so_ismcts::{SearchResult, SoIsmctsConfig};
use std::num::NonZeroU32;

fn config(iterations: u32, tree_reuse: bool, selection_policy: BanditPolicy) -> AgentConfig {
    AgentConfig::SoIsmcts(SoIsmctsConfig {
        budget: SearchBudget::Iterations(NonZeroU32::new(iterations).unwrap()),
        exploration: 1.,
        tree_reuse,
        selection_policy,
    })
}

fn midgame(owner: PlayerId, turns: usize) -> LostCitiesState {
    let mut state = LostCities.initial_state();
    let mut rng = SplitMix64::new(19);
    while LostCities.status(&state) == PositionStatus::Chance {
        let event = LostCities.sample_chance(&state, &mut rng).unwrap();
        LostCities.apply_chance_outcome(&mut state, &event).unwrap();
    }
    for _ in 0..turns + owner.index() {
        let card = state.hands[state.current_player.index()][0];
        LostCities
            .apply_action(&mut state, &A::Discard(card))
            .unwrap();
        LostCities.apply_action(&mut state, &A::DrawDeck).unwrap();
        let event = LostCities.sample_chance(&state, &mut rng).unwrap();
        LostCities.apply_chance_outcome(&mut state, &event).unwrap();
    }
    LostCities.validate_state(&state).unwrap();
    assert_eq!(LostCities.status(&state), PositionStatus::PlayerTurn(owner));
    state
}

fn different_hidden_hand(state: &LostCitiesState, owner: PlayerId) -> LostCitiesState {
    let mut other = state.clone();
    let opponent = 1 - owner.index();
    let i = other
        .deck
        .iter()
        .position(|c| *c != other.hands[opponent][0])
        .unwrap();
    std::mem::swap(&mut other.hands[opponent][0], &mut other.deck[i]);
    other.hands[opponent].sort();
    other.deck.sort();
    LostCities.validate_state(&other).unwrap();
    assert_ne!(state.hands[opponent], other.hands[opponent]);
    assert_ne!(state.deck, other.deck);
    other
}

fn same_information(a: &LostCitiesState, b: &LostCitiesState, owner: PlayerId) {
    LostCities.validate_state(a).unwrap();
    LostCities.validate_state(b).unwrap();
    assert_eq!(
        LostCities.observation(a, owner),
        LostCities.observation(b, owner)
    );
    assert_eq!(
        LostCities.legal_actions(a).collect::<Vec<_>>(),
        LostCities.legal_actions(b).collect::<Vec<_>>()
    );
}

fn root_stats(result: &SearchResult<A, LostCitiesObservation>) -> Vec<(A, u64, u64, f64)> {
    let mut stats: Vec<_> = result.nodes[0]
        .edges
        .iter()
        .map(|e| (e.action, e.visits, e.availability, e.mean_utility()))
        .collect();
    stats.sort_by_key(|s| s.0);
    stats
}

fn assert_fresh_invariance(a: &LostCitiesState, b: &LostCitiesState, owner: PlayerId) {
    same_information(a, b, owner);
    let snapshots = (a.clone(), b.clone());
    for policy in [BanditPolicy::Uct, BanditPolicy::Ucb1Tuned] {
        for seed in [5, 99] {
            let AgentConfig::SoIsmcts(config) = config(32, false, policy) else {
                unreachable!()
            };
            let search = SoIsmctsAgent {
                config: config.clone(),
            };
            let run = |state: &LostCitiesState| {
                search
                    .search(
                        &LostCities,
                        &LostCities.observation(state, owner),
                        owner,
                        &LostCities.legal_actions(state).collect::<Vec<_>>(),
                        &mut SplitMix64::new(seed),
                    )
                    .unwrap()
            };
            let (left, right) = (run(a), run(b));
            assert_eq!(left.action, right.action);
            assert_eq!(left.diagnostics, right.diagnostics);
            assert_eq!(root_stats(&left), root_stats(&right));
            let mut first =
                LostCitiesParticipant::new(AgentConfig::SoIsmcts(config.clone())).unwrap();
            let mut second = LostCitiesParticipant::new(AgentConfig::SoIsmcts(config)).unwrap();
            first.on_match_start(&LostCities, a, owner);
            second.on_match_start(&LostCities, b, owner);
            let first_action = first
                .select_action(
                    DecisionContext::new(&LostCities, a, owner),
                    &mut SplitMix64::new(seed),
                )
                .unwrap();
            let second_action = second
                .select_action(
                    DecisionContext::new(&LostCities, b, owner),
                    &mut SplitMix64::new(seed),
                )
                .unwrap();
            assert_eq!(first_action, left.action);
            assert_eq!(second_action, first_action);
            assert!(
                LostCities
                    .legal_actions(a)
                    .any(|action| action == first_action)
            );
            assert_eq!(first.last_decision_stats(), second.last_decision_stats());
            assert_eq!((a, b), (&snapshots.0, &snapshots.1));
        }
    }
}

#[test]
fn fresh_so_search_is_invariant_to_real_opponent_hand() {
    for owner in [PlayerId::FIRST, PlayerId::SECOND] {
        let a = midgame(owner, 4);
        let b = different_hidden_hand(&a, owner);
        assert_fresh_invariance(&a, &b, owner);
    }
}

#[test]
fn fresh_so_search_cannot_know_the_next_real_draw() {
    // Reality has a canonical pool, NOT a deck order. Different environment RNG
    // states express "the next real card is X rather than Y" without invalidly
    // shuffling that pool. Neither environment stream is an input to search.
    for owner in [PlayerId::FIRST, PlayerId::SECOND] {
        let mut a = midgame(owner, 4);
        let action = LostCities.legal_actions(&a).next().unwrap();
        LostCities.apply_action(&mut a, &action).unwrap();
        let b = a.clone();
        let (mut env_a, mut env_b) = (SplitMix64::new(0), SplitMix64::new(1));
        assert_fresh_invariance(&a, &b, owner);
        LostCities.apply_action(&mut a, &A::DrawDeck).unwrap();
        let mut b = b;
        LostCities.apply_action(&mut b, &A::DrawDeck).unwrap();
        let event_a = LostCities.sample_chance(&a, &mut env_a).unwrap();
        let event_b = LostCities.sample_chance(&b, &mut env_b).unwrap();
        assert_ne!(event_a, event_b);
        LostCities.apply_chance_outcome(&mut a, &event_a).unwrap();
        LostCities.apply_chance_outcome(&mut b, &event_b).unwrap();
        LostCities.validate_state(&a).unwrap();
        LostCities.validate_state(&b).unwrap();
        // The owner's draw is now legitimately visible: invariance no longer applies.
        assert_ne!(
            LostCities.observation(&a, owner),
            LostCities.observation(&b, owner)
        );
    }
}

fn apply_action(participant: &mut LostCitiesParticipant, state: &mut LostCitiesState, action: A) {
    let actor = state.current_player;
    LostCities.apply_action(state, &action).unwrap();
    participant.on_action_applied(&LostCities, state, actor, &action);
}

fn apply_chance(participant: &mut LostCitiesParticipant, state: &mut LostCitiesState, event: A) {
    LostCities.apply_chance_outcome(state, &event).unwrap();
    participant.on_chance_applied(&LostCities, state, &event);
    LostCities.validate_state(state).unwrap();
}

#[test]
fn lifecycle_reuse_is_invariant_to_hidden_hands_and_private_draw_events() {
    for owner in [PlayerId::FIRST, PlayerId::SECOND] {
        let mut a = midgame(owner, 40);
        let play = LostCities.legal_actions(&a).next().unwrap();
        LostCities.apply_action(&mut a, &play).unwrap();
        let mut b = different_hidden_hand(&a, owner);
        same_information(&a, &b, owner);
        // A late-game fixture bounds rollouts; 512 iterations also expand the
        // third decision edge needed to exercise the private chance callback.
        let cfg = config(512, true, BanditPolicy::Uct);
        // Fixed legal history: a public own draw, then an opponent discard and
        // private draw. Require retained visits below, so a reset cannot hide a leak.
        use meeple_bots_lost_cities::{LostCitiesCard::Number, LostCitiesColor::*};
        let opponent_card = if owner == PlayerId::FIRST {
            Number(White, 7)
        } else {
            Number(Yellow, 9)
        };
        let path = [A::DrawDiscard(Red), A::Discard(opponent_card), A::DrawDeck];
        let mut left = LostCitiesParticipant::new(cfg.clone()).unwrap();
        let mut right = LostCitiesParticipant::new(cfg).unwrap();
        left.on_match_start(&LostCities, &a, owner);
        right.on_match_start(&LostCities, &b, owner);
        let (mut rng_a, mut rng_b) = (SplitMix64::new(7), SplitMix64::new(7));
        let action_a = left
            .select_action(DecisionContext::new(&LostCities, &a, owner), &mut rng_a)
            .unwrap();
        let action_b = right
            .select_action(DecisionContext::new(&LostCities, &b, owner), &mut rng_b)
            .unwrap();
        assert_eq!(action_a, action_b);
        assert_eq!(left.last_decision_stats(), right.last_decision_stats());
        for action in path {
            apply_action(&mut left, &mut a, action);
            apply_action(&mut right, &mut b, action);
            assert_eq!(
                LostCities.observation(&a, owner),
                LostCities.observation(&b, owner)
            );
        }
        let event_a = A::DealCard(a.deck[0]);
        let event_b = A::DealCard(*b.deck.iter().find(|c| A::DealCard(**c) != event_a).unwrap());
        assert_ne!(event_a, event_b);
        apply_chance(&mut left, &mut a, event_a);
        apply_chance(&mut right, &mut b, event_b);
        same_information(&a, &b, owner);
        let before = (a.clone(), b.clone());
        let action_a = left
            .select_action(DecisionContext::new(&LostCities, &a, owner), &mut rng_a)
            .unwrap();
        let action_b = right
            .select_action(DecisionContext::new(&LostCities, &b, owner), &mut rng_b)
            .unwrap();
        assert_eq!(action_a, action_b);
        assert_eq!(left.last_decision_stats(), right.last_decision_stats());
        assert_eq!((a.clone(), b.clone()), before);
        let reuse = left.last_decision_stats().tree_reuse.unwrap();
        assert_eq!(reuse.own_action_hits, 1);
        assert_eq!(reuse.opponent_action_hits, 2); // Includes the hidden draw callback.
        assert_eq!(reuse.transition_misses, 0);
        assert!(reuse.reused_root_visits > 0);
        left.on_match_end(&LostCities, &a);
        right.on_match_end(&LostCities, &b);
    }
}

// Exercise the real runner's RNG ownership and all participant callbacks, but
// discard search recommendations so only search work differs between executions.
struct FixedPlay(Option<LostCitiesParticipant>);
impl Agent<LostCities> for FixedPlay {
    fn select_action<R: RandomSource + ?Sized>(
        &mut self,
        d: DecisionContext<'_, LostCities>,
        rng: &mut R,
    ) -> Result<A, AgentError> {
        let legal: Vec<_> = d.legal_actions().collect();
        if let Some(participant) = &mut self.0 {
            let selected = participant.select_action(d, rng)?;
            assert!(legal.contains(&selected));
        }
        Ok(legal[0])
    }
    fn on_match_start(&mut self, g: &LostCities, s: &LostCitiesState, p: PlayerId) {
        if let Some(a) = &mut self.0 {
            a.on_match_start(g, s, p);
        }
    }
    fn on_action_applied(&mut self, g: &LostCities, s: &LostCitiesState, p: PlayerId, action: &A) {
        if let Some(a) = &mut self.0 {
            a.on_action_applied(g, s, p, action);
        }
    }
    fn on_chance_applied(&mut self, g: &LostCities, s: &LostCitiesState, event: &A) {
        if let Some(a) = &mut self.0 {
            a.on_chance_applied(g, s, event);
        }
    }
    fn on_match_end(&mut self, g: &LostCities, s: &LostCitiesState) {
        if let Some(a) = &mut self.0 {
            a.on_match_end(g, s);
        }
    }
}

#[test]
fn real_draws_are_identical_without_search_and_with_small_or_larger_search() {
    let run = |iterations, reuse| {
        let player = || {
            FixedPlay((iterations > 0).then(|| {
                LostCitiesParticipant::new(config(iterations, reuse, BanditPolicy::Uct)).unwrap()
            }))
        };
        report(
            play_match_with_trace(
                &LostCities,
                &mut player(),
                &mut player(),
                MatchConfig::default(),
            )
            .unwrap(),
        )
        .unwrap()
    };
    let control = run(0, false);
    assert_eq!(control.chance_events.len(), 60); // All setup events and 44 real draws.
    for reuse in [false, true] {
        for iterations in [1, 32] {
            let searched = run(iterations, reuse);
            assert_eq!(
                control
                    .moves
                    .iter()
                    .map(|m| (m.player, &m.action))
                    .collect::<Vec<_>>(),
                searched
                    .moves
                    .iter()
                    .map(|m| (m.player, &m.action))
                    .collect::<Vec<_>>()
            );
            assert_eq!(control.chance_events, searched.chance_events);
            assert_eq!(control.lost_cities_state, searched.lost_cities_state);
        }
    }
}
