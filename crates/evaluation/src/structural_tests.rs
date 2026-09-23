use super::*;
use meeple_bots_core::{ImperfectInformationGame, PlayerId};

// Random setup with variable initial branching, a chance event inside a player's
// physical turn, and a second physical turn belonging to the same player.
struct ChanceGame;
impl Game for ChanceGame {
    type State = (u8, u8);
    type Action = u8;
    type Observation<'a> = &'a Self::State;
    type LegalActions<'a> = std::ops::Range<u8>;
    fn player_count(&self) -> u8 {
        2
    }
    fn initial_state(&self) -> Self::State {
        (0, 0)
    }
    fn status(&self, s: &Self::State) -> PositionStatus {
        match s.0 {
            0 | 2 => PositionStatus::Chance,
            1 | 3 | 4 => PositionStatus::PlayerTurn(PlayerId::FIRST),
            _ => PositionStatus::Terminal,
        }
    }
    fn is_turn_boundary(&self, s: &Self::State) -> bool {
        matches!(s.0, 1 | 4)
    }
    fn diagnostic_phase(&self, s: &Self::State) -> Option<&'static str> {
        // Deliberately labels chance/terminal too: the analyzer must ignore them.
        Some(if s.0 == 1 { "Choice" } else { "Forced" })
    }
    fn legal_actions<'a>(&'a self, s: &'a Self::State) -> Self::LegalActions<'a> {
        0..if s.0 == 1 { s.1 } else { 1 }
    }
    fn apply_action(&self, s: &mut Self::State, _: &u8) -> Result<(), IllegalAction> {
        s.0 += 1;
        Ok(())
    }
    fn chance_outcomes(&self, _: &Self::State) -> Result<Vec<(u8, f64)>, IllegalAction> {
        Ok(vec![(1, 0.5), (3, 0.5)])
    }
    fn apply_chance_outcome(&self, s: &mut Self::State, a: &u8) -> Result<(), IllegalAction> {
        s.1 = *a;
        s.0 += 1;
        Ok(())
    }
    fn observation<'a>(&'a self, s: &'a Self::State, _: PlayerId) -> Self::Observation<'a> {
        s
    }
    fn terminal_utility(&self, s: &Self::State, _: PlayerId) -> Option<f32> {
        (s.0 == 5).then_some(0.)
    }
}
#[test]
fn chance_setup_counts_events_and_preserves_decisions_and_physical_turns() {
    let cfg = EvaluationConfig {
        samples: NonZeroU32::new(64).unwrap(),
        ..Default::default()
    };
    let r = analyze_structure(&ChanceGame, cfg).unwrap();
    assert_eq!(r, analyze_structure(&ChanceGame, cfg).unwrap());
    assert_eq!(r.terminal_rate, 1.);
    assert_eq!((r.depth_p50, r.estimated_depth), (3, 3));
    assert_eq!((r.physical_turns_p50, r.player_turn_depth_p50), (2, 1));
    assert_eq!(
        (
            r.chance_events_mean,
            r.chance_events_p50,
            r.chance_events_p95
        ),
        (2., 2, 2)
    );
    assert_eq!(
        (r.initial_legal_actions_min, r.initial_legal_actions_max),
        (1, 3)
    );
    assert!((1. ..3.).contains(&r.initial_legal_actions_mean));
    // Only one of three decisions branches. Chance outcomes are excluded.
    assert!(r.effective_branching_factor <= 3_f64.powf(1. / 3.));
    assert_eq!(r.phases.len(), 2);
    let choice = r.phases.iter().find(|p| p.label == "Choice").unwrap();
    let forced = r.phases.iter().find(|p| p.label == "Forced").unwrap();
    assert_eq!(choice.samples, 64);
    assert_eq!(forced.samples, 128);
    assert_eq!((choice.legal_actions_min, choice.legal_actions_max), (1, 3));
    assert_eq!(choice.legal_actions_mean, r.initial_legal_actions_mean);
    assert_eq!((forced.legal_actions_p50, forced.legal_actions_p95), (1, 1));
    assert_eq!(forced.effective_branching_factor, 1.);
    let short = analyze_structure(
        &ChanceGame,
        EvaluationConfig {
            max_depth: NonZeroU32::new(1).unwrap(),
            ..cfg
        },
    )
    .unwrap();
    assert_eq!(short.depth_p50, 1);
    assert_eq!(short.chance_events_mean, 2.);
    assert!(short.depth_is_lower_bound);
}
#[test]
fn deterministic_structure_stays_exact_and_has_no_chance() {
    let r = analyze_structure(
        &meeple_bots_tic_tac_toe::TicTacToe,
        EvaluationConfig::default(),
    )
    .unwrap();
    assert_eq!(
        (r.initial_legal_actions_min, r.initial_legal_actions_max),
        (9, 9)
    );
    assert_eq!(r.chance_events_mean, 0.);
    assert!(r.phases.is_empty());
    assert_eq!(r.physical_turns_p50, r.depth_p50);
}

#[test]
fn calibration_adds_only_missing_phases_without_chance_or_terminal_probes() {
    let states = sample_calibration_states(&ChanceGame, 6, 42).unwrap();
    assert_eq!(states.len(), 2);
    assert!(
        states
            .iter()
            .all(|(_, s)| matches!(ChanceGame.status(s), PositionStatus::PlayerTurn(_)))
    );
    let game = meeple_bots_lost_cities::LostCities;
    let states = sample_calibration_states(&game, 120, 42).unwrap();
    assert!(states.len() <= 7);
    for phase in ["Play", "Draw"] {
        assert!(
            states
                .iter()
                .any(|(_, state)| game.diagnostic_phase(state) == Some(phase))
        );
    }
}
#[test]
fn lost_cities_structural_sampling_resolves_deals_and_draws() {
    let r = analyze_structure(
        &meeple_bots_lost_cities::LostCities,
        EvaluationConfig {
            samples: NonZeroU32::new(8).unwrap(),
            max_depth: NonZeroU32::new(2000).unwrap(),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r.terminal_rate, 1.);
    // Sixteen initial private deals plus 44 subsequent deck draws.
    assert_eq!(
        (
            r.chance_events_mean,
            r.chance_events_p50,
            r.chance_events_p95
        ),
        (60., 60, 60)
    );
    assert_eq!(r.depth_p50, 2 * r.physical_turns_p50);
}
#[test]
fn so_calibration_is_seeded_and_environment_is_unchanged() {
    use meeple_bots_lost_cities::LostCities;
    use meeple_bots_so_ismcts::SoIsmctsConfig;
    let cfg = SoIsmctsConfig {
        budget: SearchBudget::Iterations(NonZeroU32::new(8).unwrap()),
        ..Default::default()
    };
    let before = analyze_structure(&LostCities, EvaluationConfig::default()).unwrap();
    let a = so_ismcts::benchmark(&LostCities, cfg.clone(), 120, 42).unwrap();
    let b = so_ismcts::benchmark(&LostCities, cfg, 120, 42).unwrap();
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.iterations, 8);
        assert_eq!(x.determinizations, 8);
        assert_eq!(x.root_visits, y.root_visits);
        assert_eq!(x.nodes, y.nodes);
        assert_eq!(x.action_edges, y.action_edges);
        assert_eq!(x.root_visits.iter().sum::<u64>(), 8);
        assert!(x.determinization_milliseconds > 0.);
    }
    assert_eq!(
        before,
        analyze_structure(&LostCities, EvaluationConfig::default()).unwrap()
    );
}

/// Change only the opponent/deck assignment after the deal. The calibrator must
/// never feed that assignment to search: only FIRST's unchanged observation.
struct HiddenAssignment {
    swap: bool,
}
impl Game for HiddenAssignment {
    type State = meeple_bots_lost_cities::LostCitiesState;
    type Action = meeple_bots_lost_cities::LostCitiesAction;
    type Observation<'a> = meeple_bots_lost_cities::LostCitiesObservation;
    type LegalActions<'a> = std::vec::IntoIter<Self::Action>;
    fn player_count(&self) -> u8 {
        2
    }
    fn initial_state(&self) -> Self::State {
        meeple_bots_lost_cities::LostCities.initial_state()
    }
    fn status(&self, s: &Self::State) -> PositionStatus {
        meeple_bots_lost_cities::LostCities.status(s)
    }
    fn legal_actions<'a>(&'a self, s: &'a Self::State) -> Self::LegalActions<'a> {
        meeple_bots_lost_cities::LostCities
            .legal_actions(s)
            .collect::<Vec<_>>()
            .into_iter()
    }
    fn apply_action(&self, s: &mut Self::State, a: &Self::Action) -> Result<(), IllegalAction> {
        meeple_bots_lost_cities::LostCities.apply_action(s, a)
    }
    fn sample_chance<R: RandomSource + ?Sized>(
        &self,
        s: &Self::State,
        rng: &mut R,
    ) -> Result<Self::Action, IllegalAction> {
        meeple_bots_lost_cities::LostCities.sample_chance(s, rng)
    }
    fn apply_chance_outcome(
        &self,
        s: &mut Self::State,
        a: &Self::Action,
    ) -> Result<(), IllegalAction> {
        let dealing = matches!(s.phase, meeple_bots_lost_cities::Phase::Deal(_));
        meeple_bots_lost_cities::LostCities.apply_chance_outcome(s, a)?;
        if self.swap && dealing && s.phase == meeple_bots_lost_cities::Phase::Play {
            let different = s.deck.iter().position(|c| *c != s.hands[1][0]).unwrap();
            std::mem::swap(&mut s.hands[1][0], &mut s.deck[different]);
            s.hands[1].sort();
            s.deck.sort();
        }
        Ok(())
    }
    fn observation<'a>(&'a self, s: &'a Self::State, p: PlayerId) -> Self::Observation<'a> {
        meeple_bots_lost_cities::LostCities.observation(s, p)
    }
    fn terminal_utility(&self, s: &Self::State, p: PlayerId) -> Option<f32> {
        meeple_bots_lost_cities::LostCities.terminal_utility(s, p)
    }
}
impl meeple_bots_core::TwoPlayerZeroSumGame for HiddenAssignment {}
impl meeple_bots_core::ImperfectInformationGame for HiddenAssignment {
    type Determinization = meeple_bots_lost_cities::LostCitiesSimulationWorld;
    fn sample_determinization<R: RandomSource + ?Sized>(
        &self,
        o: &Self::Observation<'_>,
        p: PlayerId,
        rng: &mut R,
    ) -> Result<Self::Determinization, IllegalAction> {
        meeple_bots_lost_cities::LostCities.sample_determinization(o, p, rng)
    }
}
#[test]
fn calibration_is_indistinguishable_across_hidden_authoritative_assignments() {
    let a = HiddenAssignment { swap: false };
    let b = HiddenAssignment { swap: true };
    let root_a = sample_calibration_states(&a, 0, 42).unwrap().remove(0).1;
    let root_b = sample_calibration_states(&b, 0, 42).unwrap().remove(0).1;
    assert_ne!(root_a.hands[1], root_b.hands[1]);
    assert_ne!(root_a.deck, root_b.deck);
    assert_eq!(
        a.observation(&root_a, PlayerId::FIRST),
        b.observation(&root_b, PlayerId::FIRST)
    );
    let config = meeple_bots_so_ismcts::SoIsmctsConfig {
        budget: SearchBudget::Iterations(NonZeroU32::new(32).unwrap()),
        ..Default::default()
    };
    let first = so_ismcts::benchmark(&a, config.clone(), 0, 42).unwrap();
    let second = so_ismcts::benchmark(&b, config, 0, 42).unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(second.len(), 1);
    let x = &first[0];
    let y = &second[0];
    assert_eq!(x.root_visits, y.root_visits);
    assert_eq!(x.nodes, y.nodes);
    assert_eq!(x.action_edges, y.action_edges);
    assert_eq!(x.mean_availability, y.mean_availability);
    assert_eq!(x.determinizations, y.determinizations);
}
