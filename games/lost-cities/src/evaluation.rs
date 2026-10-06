//! Observation-only, known-card evaluations. No beliefs or future-card resources.
use crate::{
    LostCitiesCard, LostCitiesColor, LostCitiesObservation, Phase, accepts, expedition_score,
};
use meeple_bots_core::{AgentError, ObservationEvaluator, PlayerId};

/// Authoritative states and sampled worlds cannot enter the evaluator:
/// ```compile_fail
/// use meeple_bots_lost_cities::{LostCitiesEvaluator, LostCitiesSimulationWorld};
/// use meeple_bots_core::{ObservationEvaluator, PlayerId};
/// fn leak(world: &LostCitiesSimulationWorld) {
///     LostCitiesEvaluator::PublicScore { tau: 40.0 }.evaluate(world, PlayerId::FIRST);
/// }
/// ```
/// V0 uses public scores; V1 projects only the observer's known hand.
/// The opponent projection deliberately remains its current public score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LostCitiesEvaluator {
    Neutral,
    PublicScore { tau: f64 },
    KnownContinuation { tau: f64 },
}
impl LostCitiesEvaluator {
    pub fn validate(self) -> Result<(), AgentError> {
        match self {
            Self::Neutral => Ok(()),
            Self::PublicScore { tau } | Self::KnownContinuation { tau }
                if tau.is_finite() && tau > 0.0 =>
            {
                Ok(())
            }
            _ => Err(AgentError::message("tau must be finite and positive")),
        }
    }
}

/// Conservative reference: all remaining draws consume the deck. Discard draws
/// can extend real play. The final deck draw ends the round before its card can play.
pub fn remaining_play_opportunities(o: &LostCitiesObservation) -> usize {
    let actor = o.current_player == o.observer;
    match o.phase {
        Phase::Play => {
            if actor {
                o.deck_size.div_ceil(2)
            } else {
                o.deck_size / 2
            }
        }
        Phase::Draw | Phase::DrawChance => {
            let remaining = o.deck_size.saturating_sub(1);
            if actor {
                remaining / 2
            } else {
                remaining.div_ceil(2)
            }
        }
        _ => 0,
    }
}

fn public_score(o: &LostCitiesObservation, player: PlayerId) -> i16 {
    o.expeditions[player.index()]
        .iter()
        .map(|cards| expedition_score(cards))
        .sum()
}

/// Exact known-card continuations; at most 2^8 subsets across a single color.
fn color_scores(column: &[LostCitiesCard], hand: &[LostCitiesCard], budget: usize) -> Vec<i16> {
    let mut best = vec![expedition_score(column); budget + 1];
    let mut cards = column.to_vec();
    for mask in 1_usize..(1_usize << hand.len()) {
        let count = mask.count_ones() as usize;
        if count > budget {
            continue;
        }
        cards.truncate(column.len());
        let mut legal = true;
        for (index, card) in hand.iter().enumerate() {
            if mask & (1 << index) != 0 {
                if !accepts(&cards, *card) {
                    legal = false;
                    break;
                }
                cards.push(*card);
            }
        }
        if legal {
            best[count] = best[count].max(expedition_score(&cards));
        }
    }
    for k in 1..=budget {
        best[k] = best[k].max(best[k - 1]);
    }
    best
}

fn projected_score(o: &LostCitiesObservation) -> Result<i16, AgentError> {
    // Observation validity is normally established by the game sampler. Also bound
    // direct evaluation inputs so subset enumeration cannot become unbounded.
    if o.hand.len() > 8 {
        return Err(AgentError::message("Lost Cities hand exceeds eight cards"));
    }
    let budget = remaining_play_opportunities(o).min(o.hand.len());
    let mut portfolio = vec![0_i16; budget + 1];
    for color in LostCitiesColor::ALL {
        let mut hand: Vec<_> = o
            .hand
            .iter()
            .copied()
            .filter(|card| card.color() == color)
            .collect();
        hand.sort(); // Wagers first, then ascending numbers; physical wagers retained.
        let scores = color_scores(
            &o.expeditions[o.observer.index()][color as usize],
            &hand,
            budget,
        );
        let mut next = vec![i16::MIN; budget + 1];
        for total in 0..=budget {
            for used in 0..=total {
                next[total] = next[total].max(portfolio[total - used] + scores[used]);
            }
        }
        portfolio = next;
    }
    Ok(portfolio[budget])
}

impl ObservationEvaluator<LostCitiesObservation> for LostCitiesEvaluator {
    fn evaluate(&self, o: &LostCitiesObservation, root: PlayerId) -> Result<f64, AgentError> {
        self.validate()?;
        if root.index() >= 2 || root != o.observer {
            return Err(AgentError::message(
                "observation belongs to a different player",
            ));
        }
        let (own, tau) = match *self {
            Self::Neutral => return Ok(0.0),
            Self::PublicScore { tau } => (public_score(o, root), tau),
            Self::KnownContinuation { tau } => (projected_score(o)?, tau),
        };
        let opponent = public_score(o, PlayerId::new(1 - root.index() as u8));
        Ok((f64::from(own - opponent) / tau).tanh())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        LostCities,
        LostCitiesCard::{Number as N, Wager as W},
        LostCitiesColor::*,
    };
    use meeple_bots_core::{Game, ImperfectInformationGame, PositionStatus};
    use meeple_bots_simulation::SplitMix64;

    fn observation() -> LostCitiesObservation {
        LostCitiesObservation {
            observer: PlayerId::FIRST,
            current_player: PlayerId::FIRST,
            phase: Phase::Play,
            hand: vec![],
            opponent_hand_size: 8,
            deck_size: 44,
            expeditions: Default::default(),
            discards: Default::default(),
            blocked_discard: None,
        }
    }
    #[test]
    fn public_scores_orientation_range_and_tau() {
        let mut o = observation();
        let v0 = LostCitiesEvaluator::PublicScore { tau: 40.0 };
        assert_eq!(v0.evaluate(&o, o.observer).unwrap(), 0.0);
        o.expeditions[0][0] = vec![N(Red, 3)];
        let first = v0.evaluate(&o, o.observer).unwrap();
        assert_eq!(first, (-17.0_f64 / 40.0).tanh());
        o.observer = PlayerId::SECOND;
        assert_eq!(v0.evaluate(&o, o.observer).unwrap(), -first);
        assert!(v0.evaluate(&o, PlayerId::FIRST).is_err());
        for tau in [f64::MIN_POSITIVE, 1.0, 40.0, f64::MAX] {
            for evaluator in [
                LostCitiesEvaluator::Neutral,
                LostCitiesEvaluator::PublicScore { tau },
                LostCitiesEvaluator::KnownContinuation { tau },
            ] {
                let value = evaluator.evaluate(&o, o.observer).unwrap();
                assert!(value.is_finite() && (-1.0..=1.0).contains(&value));
            }
        }
        for tau in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                LostCitiesEvaluator::PublicScore { tau }
                    .evaluate(&o, o.observer)
                    .is_err()
            );
        }
    }
    #[test]
    fn ascending_continuations_and_blocked_cards() {
        assert_eq!(
            color_scores(&[N(Red, 3)], &[N(Red, 5), N(Red, 9)], 2),
            vec![-17, -8, -3]
        );
        assert_eq!(
            color_scores(&[N(Red, 8)], &[N(Red, 5), N(Red, 9)], 2),
            vec![-12, -3, -3]
        );
        assert_eq!(
            color_scores(&[N(Red, 3)], &[W(Red), N(Red, 9)], 2),
            vec![-17, -8, -8]
        );
    }
    #[test]
    fn unopened_can_stay_empty_and_opening_cost_is_once() {
        assert_eq!(color_scores(&[], &[N(Red, 9)], 1), vec![0, 0]);
        assert_eq!(
            color_scores(&[], &[N(Red, 6), N(Red, 9), N(Red, 10)], 3),
            vec![0, 0, 0, 5]
        );
    }
    #[test]
    fn wagers_use_real_multiplier_and_consume_a_play() {
        let hand = [W(Red), W(Red), N(Red, 6), N(Red, 9), N(Red, 10)];
        assert_eq!(color_scores(&[], &hand, 5), vec![0, 0, 0, 5, 10, 15]);
        assert_eq!(
            color_scores(&[W(Red)], &[N(Red, 6), N(Red, 9), N(Red, 10)], 3)[3],
            10
        );
        assert_eq!(color_scores(&[W(Red)], &[], 0), vec![-40]);
    }
    #[test]
    fn eight_card_bonus_is_after_multiplication_and_not_repeated() {
        let mut column = vec![W(Red)];
        column.extend((2..=7).map(|n| N(Red, n)));
        assert_eq!(color_scores(&column, &[N(Red, 8)], 1), vec![14, 50]);
        column.push(N(Red, 8));
        assert_eq!(color_scores(&column, &[N(Red, 9)], 1), vec![50, 68]);
    }
    #[test]
    fn global_budget_is_shared_by_colors_and_rival_stays_public() {
        let mut o = observation();
        o.expeditions[0][0] = vec![N(Red, 3)];
        o.expeditions[0][1] = vec![N(Green, 3)];
        o.hand = vec![N(Red, 9), N(Green, 10)];
        o.deck_size = 2;
        assert_eq!(projected_score(&o).unwrap(), -24);
        o.deck_size = 4;
        assert_eq!(projected_score(&o).unwrap(), -15);
        o.expeditions[1][2] = vec![W(Blue)];
        assert_eq!(
            LostCitiesEvaluator::KnownContinuation { tau: 40.0 }
                .evaluate(&o, o.observer)
                .unwrap(),
            (25.0_f64 / 40.0).tanh()
        );
    }
    #[test]
    fn phase_actor_and_last_draw_change_opportunities() {
        let mut o = observation();
        o.deck_size = 3;
        assert_eq!(remaining_play_opportunities(&o), 2);
        o.current_player = PlayerId::SECOND;
        assert_eq!(remaining_play_opportunities(&o), 1);
        o.phase = Phase::Draw;
        assert_eq!(remaining_play_opportunities(&o), 1);
        o.deck_size = 1;
        assert_eq!(remaining_play_opportunities(&o), 0);
        o.current_player = o.observer;
        assert_eq!(remaining_play_opportunities(&o), 0);
        o.phase = Phase::Play;
        assert_eq!(remaining_play_opportunities(&o), 1);
        o.phase = Phase::Finished;
        assert_eq!(remaining_play_opportunities(&o), 0);
    }
    #[test]
    fn hidden_hands_and_deck_orders_do_not_change_value() {
        let mut state = LostCities.initial_state();
        let mut rng = SplitMix64::new(19);
        while LostCities.status(&state) == PositionStatus::Chance {
            let event = LostCities.sample_chance(&state, &mut rng).unwrap();
            LostCities.apply_chance_outcome(&mut state, &event).unwrap();
        }
        let o = LostCities.observation(&state, PlayerId::FIRST);
        let mut other = state.clone();
        let different = other
            .deck
            .iter()
            .position(|card| *card != other.hands[1][0])
            .unwrap();
        std::mem::swap(&mut other.hands[1][0], &mut other.deck[different]);
        other.hands[1].sort();
        other.deck.sort();
        LostCities.validate_state(&other).unwrap();
        assert_ne!(state.hands[1], other.hands[1]);
        let o2 = LostCities.observation(&other, PlayerId::FIRST);
        assert_eq!(o, o2);
        let world_a = LostCities
            .sample_determinization(&o, o.observer, &mut SplitMix64::new(5))
            .unwrap();
        let world_b = LostCities
            .sample_determinization(&o, o.observer, &mut SplitMix64::new(99))
            .unwrap();
        assert_ne!(world_a.deck_order(), world_b.deck_order());
        assert_eq!(
            world_a.observation(o.observer),
            world_b.observation(o.observer)
        );
        for evaluator in [
            LostCitiesEvaluator::Neutral,
            LostCitiesEvaluator::PublicScore { tau: 40.0 },
            LostCitiesEvaluator::KnownContinuation { tau: 40.0 },
        ] {
            assert_eq!(
                evaluator
                    .evaluate(&world_a.observation(o.observer), o.observer)
                    .unwrap(),
                evaluator
                    .evaluate(&world_b.observation(o.observer), o.observer)
                    .unwrap()
            );
            assert_eq!(
                evaluator.evaluate(&o, o.observer).unwrap(),
                evaluator.evaluate(&o2, o2.observer).unwrap()
            );
        }
    }
}
