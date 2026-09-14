//! Soft rollout horizons, including consecutive turns owned by the same player.
use super::*;
use meeple_bots_core::{Game, IllegalAction};
use meeple_bots_simulation::SplitMix64;
use std::cell::Cell;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct State {
    decisions: u32,
    pending: u8,
}
#[derive(Clone, PartialEq)]
struct Turns<const CHANCE: bool> {
    length: u32,
    terminal_at: u32,
}
impl<const C: bool> Game for Turns<C> {
    type State = State;
    type Action = bool;
    type Observation<'a> = &'a State;
    type LegalActions<'a> = std::option::IntoIter<bool>;
    fn player_count(&self) -> u8 {
        2
    }
    fn initial_state(&self) -> State {
        State {
            decisions: 0,
            pending: 0,
        }
    }
    fn status(&self, s: &State) -> PositionStatus {
        if s.decisions == self.terminal_at {
            PositionStatus::Terminal
        } else if s.pending > 0 {
            PositionStatus::Chance
        } else {
            PositionStatus::PlayerTurn(PlayerId::FIRST)
        }
    }
    fn is_turn_boundary(&self, s: &State) -> bool {
        s.decisions.is_multiple_of(self.length) && s.pending == 0
    }
    fn legal_actions<'a>(&'a self, s: &'a State) -> Self::LegalActions<'a> {
        matches!(self.status(s), PositionStatus::PlayerTurn(_))
            .then_some(false)
            .into_iter()
    }
    fn apply_action(&self, s: &mut State, a: &bool) -> Result<(), IllegalAction> {
        assert!(!a && matches!(self.status(s), PositionStatus::PlayerTurn(_)));
        s.decisions += 1;
        s.pending = if C { 2 } else { 0 };
        Ok(())
    }
    fn chance_outcomes(&self, s: &State) -> Result<Vec<(bool, f64)>, IllegalAction> {
        assert!(s.pending > 0);
        Ok(vec![(true, 1.0)])
    }
    fn apply_chance_outcome(&self, s: &mut State, a: &bool) -> Result<(), IllegalAction> {
        assert!(*a && self.status(s) == PositionStatus::Chance);
        s.pending -= 1;
        Ok(())
    }
    fn observation<'a>(&'a self, s: &'a State, _: PlayerId) -> &'a State {
        s
    }
    fn terminal_utility(&self, s: &State, _: PlayerId) -> Option<f32> {
        (self.status(s) == PositionStatus::Terminal).then_some(1.0)
    }
}
impl<const C: bool> PerfectInformationGame for Turns<C> {}
impl<const C: bool> TwoPlayerZeroSumGame for Turns<C> {}
impl DeterministicGame for Turns<false> {}

struct Record<'a>(&'a Cell<u32>);
impl<const C: bool> StateEvaluator<Turns<C>> for Record<'_> {
    fn evaluate(&self, g: &Turns<C>, s: &State, _: PlayerId) -> Result<f64, AgentError> {
        assert!(g.is_turn_boundary(s));
        assert!(matches!(g.status(s), PositionStatus::PlayerTurn(_)));
        self.0.set(s.decisions);
        Ok(0.0)
    }
}

#[test]
fn deterministic_rollout_finishes_only_the_current_turn() {
    // Player identity never changes: physical boundaries must come from the game.
    for (length, nominal, start, expected) in [
        (1, 5, 0, 5),
        (2, 3, 0, 4),
        (2, 4, 0, 4),
        (5, 1, 0, 5),
        (2, 0, 1, 2),
        (2, 0, 0, 0),
    ] {
        let game = Turns::<false> {
            length,
            terminal_at: 100,
        };
        let mut state = State {
            decisions: start,
            pending: 0,
        };
        let evaluated = Cell::new(u32::MAX);
        rollout(
            &game,
            &mut state,
            PlayerId::FIRST,
            nominal,
            &UniformRandom,
            &Record(&evaluated),
            &mut SplitMix64::new(42),
        )
        .unwrap();
        assert_eq!(state.decisions, expected);
        assert_eq!(evaluated.get(), expected);
    }
}

#[test]
fn terminal_mid_turn_overrides_completion_and_evaluator() {
    let game = Turns::<false> {
        length: 2,
        terminal_at: 1,
    };
    let mut state = game.initial_state();
    let evaluated = Cell::new(0);
    let utility = rollout(
        &game,
        &mut state,
        PlayerId::FIRST,
        1,
        &UniformRandom,
        &Record(&evaluated),
        &mut SplitMix64::new(42),
    )
    .unwrap();
    assert_eq!(utility, 1.0);
    assert_eq!(state.decisions, 1);
    assert_eq!(evaluated.get(), 0);
}

#[test]
fn stochastic_cutoff_resolves_chains_and_counts_only_decisions() {
    for selector in [SelectionPolicy::Uct, SelectionPolicy::Ucb1Tuned] {
        for (length, nominal, terminal_at, expected) in [
            (1, 3, 100, 3),
            (2, 3, 100, 4),
            (2, 4, 100, 4),
            (5, 1, 100, 5),
            (2, 1, 1, 0),
        ] {
            let game = Turns::<true> {
                length,
                terminal_at,
            };
            let evaluated = Cell::new(0);
            let mut agent = StochasticMctsAgent::new(
                MctsConfig {
                    budget: SearchBudget::Iterations(NonZeroU32::new(1).unwrap()),
                    exploration: 1.0,
                    selection_policy: selector,
                    rollout_depth: nominal,
                    rollout_policy: UniformRandom,
                },
                Record(&evaluated),
            );
            agent
                .select_action(
                    DecisionContext::new(&game, &game.initial_state(), PlayerId::FIRST),
                    &mut SplitMix64::new(42),
                )
                .unwrap();
            assert_eq!(evaluated.get(), expected);
        }
    }
}

#[test]
fn reuse_and_transposition_variants_share_turn_completion() {
    for reuse in [false, true] {
        for transpositions in [false, true] {
            let game = Turns::<false> {
                length: 2,
                terminal_at: 100,
            };
            let evaluated = Cell::new(0);
            let config = MctsConfig {
                budget: SearchBudget::Iterations(NonZeroU32::new(1).unwrap()),
                exploration: 1.0,
                selection_policy: SelectionPolicy::Uct,
                rollout_depth: 2,
                rollout_policy: UniformRandom,
            };
            let mut agent = TranspositionMctsAgent::new(
                MctsAgent::with_cutoff_evaluator(config, Record(&evaluated)),
                reuse,
                transpositions,
            );
            agent
                .select_action(
                    DecisionContext::new(&game, &game.initial_state(), PlayerId::FIRST),
                    &mut SplitMix64::new(42),
                )
                .unwrap();
            // One expansion + two nominal rollout decisions land inside turn two.
            assert_eq!(evaluated.get(), 4);

            let game = Turns::<true> {
                length: 2,
                terminal_at: 100,
            };
            evaluated.set(0);
            let mut agent = ReusableStochasticMctsAgent::new(
                StochasticMctsAgent::new(
                    MctsConfig {
                        rollout_depth: 3,
                        ..config
                    },
                    Record(&evaluated),
                ),
                reuse,
                transpositions,
            );
            agent
                .select_action(
                    DecisionContext::new(&game, &game.initial_state(), PlayerId::FIRST),
                    &mut SplitMix64::new(42),
                )
                .unwrap();
            assert_eq!(evaluated.get(), 4);
        }
    }
}
