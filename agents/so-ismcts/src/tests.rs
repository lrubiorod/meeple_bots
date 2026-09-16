use super::*;
use meeple_bots_core::{IllegalAction, TwoPlayerZeroSumGame};
use meeple_bots_simulation::SplitMix64;
use std::cell::Cell;
#[derive(Clone, Debug, Eq, PartialEq)]
enum A {
    Wait,
    Common,
    Rare,
}
#[derive(Clone)]
struct World {
    stage: u8,
    red: bool,
}
struct Toy {
    samples: Cell<u32>,
}
impl DeterminizedWorld for World {
    type Action = A;
    type Observation = u8;
    fn status(&self) -> PositionStatus {
        match self.stage {
            0 => PositionStatus::PlayerTurn(PlayerId::FIRST),
            1 => PositionStatus::PlayerTurn(PlayerId::SECOND),
            _ => PositionStatus::Terminal,
        }
    }
    fn observation(&self, _: PlayerId) -> u8 {
        self.stage
    }
    fn legal_actions(&self) -> Vec<A> {
        match self.stage {
            0 => vec![A::Wait],
            1 => {
                if self.red {
                    vec![A::Common, A::Rare]
                } else {
                    vec![A::Common]
                }
            }
            _ => vec![],
        }
    }
    fn apply_action(&mut self, a: &A) -> Result<(), IllegalAction> {
        assert!(
            self.legal_actions().contains(a),
            "illegal hidden-world action selected"
        );
        self.stage += 1;
        Ok(())
    }
    fn terminal_utility(&self, _: PlayerId) -> Option<f32> {
        (self.stage == 2).then_some(1.)
    }
}
impl Game for Toy {
    type State = World;
    type Action = A;
    type Observation<'a> = u8;
    type LegalActions<'a> = std::vec::IntoIter<A>;
    fn player_count(&self) -> u8 {
        2
    }
    fn initial_state(&self) -> World {
        World {
            stage: 0,
            red: false,
        }
    }
    fn status(&self, s: &World) -> PositionStatus {
        s.status()
    }
    fn legal_actions<'a>(&'a self, s: &'a World) -> Self::LegalActions<'a> {
        s.legal_actions().into_iter()
    }
    fn apply_action(&self, s: &mut World, a: &A) -> Result<(), IllegalAction> {
        s.apply_action(a)
    }
    fn observation<'a>(&'a self, s: &'a World, p: PlayerId) -> u8 {
        s.observation(p)
    }
    fn terminal_utility(&self, s: &World, p: PlayerId) -> Option<f32> {
        s.terminal_utility(p)
    }
}
impl TwoPlayerZeroSumGame for Toy {}
impl ImperfectInformationGame for Toy {
    type Determinization = World;
    fn sample_determinization<R: RandomSource + ?Sized>(
        &self,
        o: &u8,
        _: PlayerId,
        _: &mut R,
    ) -> Result<World, IllegalAction> {
        let n = self.samples.get();
        self.samples.set(n + 1);
        Ok(World {
            stage: *o,
            red: n % 2 == 0,
        })
    }
}
fn agent(n: u32) -> SoIsmctsAgent {
    SoIsmctsAgent {
        config: SoIsmctsConfig {
            iterations: NonZeroU32::new(n).unwrap(),
            exploration: 1.,
        },
    }
}
#[test]
fn fresh_world_each_iteration_shares_history_tree() {
    let game = Toy {
        samples: Cell::new(0),
    };
    let r = agent(30)
        .search(
            &game,
            &0,
            PlayerId::FIRST,
            &[A::Wait],
            &mut SplitMix64::new(2),
        )
        .unwrap();
    assert_eq!(game.samples.get(), 30);
    assert_eq!(r.diagnostics.completed_iterations, 30);
    assert_eq!(r.diagnostics.determinizations_sampled, 30);
    assert_eq!(r.nodes[0].visits, 30);
    assert_eq!(r.nodes[0].edges[0].outcomes.len(), 1);
    let child = r.nodes[0].edges[0].outcomes[0].1;
    assert_eq!(r.nodes[child].visits, 30);
    let e = &r.nodes[child].edges;
    let common = e.iter().find(|e| e.action == A::Common).unwrap();
    let rare = e.iter().find(|e| e.action == A::Rare).unwrap();
    assert_eq!(common.availability, 29);
    assert_eq!(rare.availability, 14);
    assert!(rare.visits <= 14);
    assert_eq!(r.action, A::Wait);
    assert_eq!(r.diagnostics.cutoff_simulations, 0);
    // O=u8 and A contain no World or hidden flag; tree diagnostic output cannot reveal red.
    assert!(!format!("{:?}", r.nodes).contains("red"));
}
#[test]
fn availability_counts_all_legal_actions_including_unexpanded() {
    let mut n: Node<A, u8> = Node {
        visits: 0,
        edges: vec![],
    };
    let mut rng = SplitMix64::new(7);
    for i in 0..10 {
        let legal = if i < 4 {
            vec![A::Common, A::Rare]
        } else {
            vec![A::Common]
        };
        let indices = available(&mut n, &legal);
        let chosen = choose(&n, &indices, 1., true, &mut rng);
        assert!(legal.contains(&n.edges[chosen].action));
        n.edges[chosen].visits += 1;
    }
    assert_eq!(n.edges[0].availability, 10);
    assert_eq!(n.edges[1].availability, 4);
    assert!(n.edges[1].visits <= 4);
}
#[test]
fn is_uct_uses_availability_and_actual_actor() {
    let n: Node<A, u8> = Node {
        visits: 1000,
        edges: vec![
            Edge {
                action: A::Common,
                visits: 10,
                availability: 100,
                total_utility: 0.,
                outcomes: vec![],
            },
            Edge {
                action: A::Rare,
                visits: 2,
                availability: 2,
                total_utility: 0.,
                outcomes: vec![],
            },
        ],
    };
    assert_eq!(choose(&n, &[0, 1], 1., true, &mut SplitMix64::new(1)), 0); // parent ln(1000) would choose Rare
    assert!(is_uct(8., 10, 10, 0., true) > is_uct(2., 10, 10, 0., true));
    assert!(is_uct(8., 10, 10, 0., false) < is_uct(2., 10, 10, 0., false));
}
#[test]
fn histories_are_not_globally_merged() {
    let mut nodes = vec![Node {
        visits: 0,
        edges: vec![],
    }];
    available(&mut nodes[0], &[A::Common, A::Rare]);
    let (a, _) = outcome(&mut nodes, 0, 0, 7u8);
    let (b, _) = outcome(&mut nodes, 0, 1, 7u8);
    assert_ne!(a, b);
    assert_eq!(outcome(&mut nodes, 0, 0, 7).0, a);
    assert_ne!(outcome(&mut nodes, 0, 0, 8).0, a);
}
use meeple_bots_lost_cities::{LostCities, LostCitiesAction as L, LostCitiesState};
fn ready() -> LostCitiesState {
    let g = LostCities;
    let mut s = g.initial_state();
    let mut r = SplitMix64::new(42);
    while g.status(&s) == PositionStatus::Chance {
        let a = g.sample_chance(&s, &mut r).unwrap();
        g.apply_chance_outcome(&mut s, &a).unwrap();
    }
    s
}
#[test]
fn lost_cities_visible_draws_split_hidden_draws_merge() {
    for actor in [PlayerId::FIRST, PlayerId::SECOND] {
        let mut s = ready();
        s.current_player = actor;
        let a = LostCities.legal_actions(&s).next().unwrap();
        LostCities.apply_action(&mut s, &a).unwrap();
        let o = LostCities.observation(&s, PlayerId::FIRST);
        let mut nodes = vec![Node {
            visits: 0,
            edges: vec![],
        }];
        available(&mut nodes[0], &[L::DrawDeck]);
        let mut tops = Vec::new();
        for seed in 0..30 {
            let mut w = LostCities
                .sample_determinization(&o, PlayerId::FIRST, &mut SplitMix64::new(seed))
                .unwrap();
            tops.push(w.deck_order()[0]);
            w.apply_action(&L::DrawDeck).unwrap();
            outcome(&mut nodes, 0, 0, w.observation(PlayerId::FIRST));
        }
        assert!(tops.iter().any(|c| *c != tops[0]));
        if actor == PlayerId::FIRST {
            assert!(nodes[0].edges[0].outcomes.len() > 1);
        } else {
            assert_eq!(nodes[0].edges[0].outcomes.len(), 1);
        }
    }
}
#[test]
fn lost_cities_seeded_search_reproducible_without_mutating_reality() {
    let s = ready();
    let before = s.clone();
    let o = LostCities.observation(&s, PlayerId::FIRST);
    let legal: Vec<_> = LostCities.legal_actions(&s).collect();
    let a = agent(24)
        .search(
            &LostCities,
            &o,
            PlayerId::FIRST,
            &legal,
            &mut SplitMix64::new(99),
        )
        .unwrap();
    let b = agent(24)
        .search(
            &LostCities,
            &o,
            PlayerId::FIRST,
            &legal,
            &mut SplitMix64::new(99),
        )
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(s, before);
    assert!(legal.contains(&a.action));
    assert_eq!(a.diagnostics.determinizations_sampled, 24);
    assert_eq!(a.diagnostics.terminal_simulations, 24);
    let max = a.nodes[0].edges.iter().map(|e| e.visits).max().unwrap();
    assert_eq!(
        a.nodes[0]
            .edges
            .iter()
            .find(|e| e.action == a.action)
            .unwrap()
            .visits,
        max
    );
}
#[test]
fn wrong_root_and_invalid_config_rejected() {
    let g = Toy {
        samples: Cell::new(0),
    };
    assert!(
        agent(1)
            .search(&g, &0, PlayerId::FIRST, &[A::Rare], &mut SplitMix64::new(1))
            .is_err()
    );
    for c in [f64::NAN, f64::INFINITY, -1.] {
        let mut a = agent(1);
        a.config.exploration = c;
        assert!(a.config.validate().is_err());
    }
    assert_eq!(
        LostCities::opponent(PlayerId::FIRST),
        Some(PlayerId::SECOND)
    );
}

#[test]
fn indistinguishable_authoritative_worlds_produce_identical_searches() {
    let state = ready();
    let observation = LostCities.observation(&state, PlayerId::FIRST);
    let mut rng = SplitMix64::new(752);
    let world = LostCities
        .sample_determinization(&observation, PlayerId::FIRST, &mut rng)
        .unwrap();
    assert_ne!(world.state().hands, state.hands);
    let other_observation = world.observation(PlayerId::FIRST);
    assert_eq!(observation, other_observation);
    let legal: Vec<_> = LostCities.legal_actions(&state).collect();
    let other_legal: Vec<_> = world.legal_actions().collect();
    let a = agent(16)
        .search(
            &LostCities,
            &observation,
            PlayerId::FIRST,
            &legal,
            &mut SplitMix64::new(5),
        )
        .unwrap();
    let b = agent(16)
        .search(
            &LostCities,
            &other_observation,
            PlayerId::FIRST,
            &other_legal,
            &mut SplitMix64::new(5),
        )
        .unwrap();
    assert_eq!(a, b);
}

// Consecutive decisions by the same player must not flip utility by depth parity.
struct MicroGame {
    next_player: PlayerId,
}
#[derive(Clone)]
struct MicroWorld {
    stage: u8,
    next_player: PlayerId,
    utility: f32,
}
impl DeterminizedWorld for MicroWorld {
    type Action = A;
    type Observation = u8;
    fn status(&self) -> PositionStatus {
        match self.stage {
            0 => PositionStatus::PlayerTurn(PlayerId::FIRST),
            1 => PositionStatus::PlayerTurn(self.next_player),
            _ => PositionStatus::Terminal,
        }
    }
    fn observation(&self, _: PlayerId) -> u8 {
        self.stage
    }
    fn legal_actions(&self) -> Vec<A> {
        match self.stage {
            0 => vec![A::Wait],
            1 => vec![A::Common, A::Rare],
            _ => vec![],
        }
    }
    fn apply_action(&mut self, action: &A) -> Result<(), IllegalAction> {
        assert!(self.legal_actions().contains(action));
        self.utility = if *action == A::Rare { 1. } else { -1. };
        self.stage += 1;
        Ok(())
    }
    fn terminal_utility(&self, observer: PlayerId) -> Option<f32> {
        (self.stage == 2).then_some(if observer == PlayerId::FIRST {
            self.utility
        } else {
            -self.utility
        })
    }
}
impl Game for MicroGame {
    type State = MicroWorld;
    type Action = A;
    type Observation<'a> = u8;
    type LegalActions<'a> = std::vec::IntoIter<A>;
    fn player_count(&self) -> u8 {
        2
    }
    fn initial_state(&self) -> MicroWorld {
        MicroWorld {
            stage: 0,
            next_player: self.next_player,
            utility: 0.,
        }
    }
    fn status(&self, s: &MicroWorld) -> PositionStatus {
        s.status()
    }
    fn legal_actions<'a>(&'a self, s: &'a MicroWorld) -> Self::LegalActions<'a> {
        s.legal_actions().into_iter()
    }
    fn apply_action(&self, s: &mut MicroWorld, a: &A) -> Result<(), IllegalAction> {
        s.apply_action(a)
    }
    fn observation<'a>(&'a self, s: &'a MicroWorld, p: PlayerId) -> u8 {
        s.observation(p)
    }
    fn terminal_utility(&self, s: &MicroWorld, p: PlayerId) -> Option<f32> {
        s.terminal_utility(p)
    }
}
impl TwoPlayerZeroSumGame for MicroGame {}
impl ImperfectInformationGame for MicroGame {
    type Determinization = MicroWorld;
    fn sample_determinization<R: RandomSource + ?Sized>(
        &self,
        o: &u8,
        _: PlayerId,
        _: &mut R,
    ) -> Result<MicroWorld, IllegalAction> {
        Ok(MicroWorld {
            stage: *o,
            ..self.initial_state()
        })
    }
}
#[test]
fn selection_and_backup_follow_actor_across_microturns() {
    for next_player in [PlayerId::FIRST, PlayerId::SECOND] {
        let game = MicroGame { next_player };
        let r = agent(40)
            .search(
                &game,
                &0,
                PlayerId::FIRST,
                &[A::Wait],
                &mut SplitMix64::new(1),
            )
            .unwrap();
        let child = r.nodes[0].edges[0].outcomes[0].1;
        let edges = &r.nodes[child].edges;
        let positive = edges.iter().find(|e| e.action == A::Rare).unwrap();
        let negative = edges.iter().find(|e| e.action == A::Common).unwrap();
        assert_eq!(positive.mean_utility(), 1.);
        assert_eq!(negative.mean_utility(), -1.);
        if next_player == PlayerId::FIRST {
            assert!(positive.visits > negative.visits);
        } else {
            assert!(negative.visits > positive.visits);
        }
    }
}
