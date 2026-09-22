//! Single-observer ISMCTS. Search accepts an observation, never an authoritative state.
//! No lifecycle implementation: the catalog adapter must not forward hidden callbacks.
pub use meeple_bots_core::SearchBudget;
use meeple_bots_core::{
    AgentError, DeterminizedWorld, Game, ImperfectInformationGame, PlayerId, PositionStatus,
    RandomSource, TwoPlayerZeroSumGame,
};
use std::time::Instant;

/// Safety cutoff for cyclic games, not a game horizon. Cutoffs back up neutral utility.
pub const MAX_SIMULATION_ACTIONS: usize = 10_000;
#[derive(Clone, Debug, PartialEq)]
pub struct SoIsmctsConfig {
    pub budget: SearchBudget,
    pub exploration: f64,
}
impl Default for SoIsmctsConfig {
    fn default() -> Self {
        Self {
            budget: SearchBudget::default(),
            exploration: std::f64::consts::SQRT_2,
        }
    }
}
impl SoIsmctsConfig {
    pub fn validate(&self) -> Result<(), AgentError> {
        if matches!(self.budget, SearchBudget::Time(t) if t.is_zero()) {
            return Err(AgentError::message(
                "SO-ISMCTS time budget must be positive",
            ));
        }
        if !self.exploration.is_finite() || self.exploration < 0. {
            return Err(AgentError::message(
                "SO-ISMCTS exploration must be finite and non-negative",
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Diagnostics {
    pub completed_iterations: u64,
    pub tree_nodes: usize,
    pub action_edges: usize,
    pub determinizations_sampled: u64,
    pub rollout_count: u64,
    pub terminal_simulations: u64,
    pub cutoff_simulations: u64,
}
/// Tree types deliberately have NO world/state type parameter or field.
/// Observations live only on outcomes of a specific history edge, never in a global map.
#[derive(Clone, Debug, PartialEq)]
pub struct Node<A, O> {
    pub visits: u64,
    pub edges: Vec<Edge<A, O>>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Edge<A, O> {
    pub action: A,
    pub visits: u64,
    pub availability: u64,
    pub total_utility: f64,
    pub outcomes: Vec<(O, usize)>,
}
impl<A, O> Edge<A, O> {
    pub fn mean_utility(&self) -> f64 {
        if self.visits == 0 {
            0.
        } else {
            self.total_utility / self.visits as f64
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult<A, O> {
    pub action: A,
    pub diagnostics: Diagnostics,
    pub nodes: Vec<Node<A, O>>,
}
#[derive(Clone, Debug, Default)]
pub struct SoIsmctsAgent {
    pub config: SoIsmctsConfig,
}
fn error(e: impl ToString) -> AgentError {
    AgentError::message(e.to_string())
}

/// Stored values are root-player utilities in [-1,1]. Only exploitation flips for opponent turns.
pub fn is_uct(
    total: f64,
    visits: u64,
    availability: u64,
    exploration: f64,
    root_turn: bool,
) -> f64 {
    assert!(visits > 0 && availability >= visits);
    (if root_turn { 1. } else { -1. }) * total / visits as f64
        + exploration * ((availability as f64).ln() / visits as f64).sqrt()
}
fn available<A: Clone + Eq, O>(node: &mut Node<A, O>, legal: &[A]) -> Vec<usize> {
    let mut indices = Vec::new();
    for action in legal {
        let index = if let Some(i) = node.edges.iter().position(|e| e.action == *action) {
            i
        } else {
            node.edges.push(Edge {
                action: action.clone(),
                visits: 0,
                availability: 0,
                total_utility: 0.,
                outcomes: vec![],
            });
            node.edges.len() - 1
        };
        // Duplicate physical representations must not double-count availability.
        if !indices.contains(&index) {
            node.edges[index].availability += 1;
            indices.push(index);
        }
    }
    indices
}
fn choose<A, O, R: RandomSource + ?Sized>(
    node: &Node<A, O>,
    legal: &[usize],
    c: f64,
    root_turn: bool,
    rng: &mut R,
) -> usize {
    let unvisited: Vec<_> = legal
        .iter()
        .copied()
        .filter(|&i| node.edges[i].visits == 0)
        .collect();
    if !unvisited.is_empty() {
        return unvisited[rng.index(unvisited.len()).unwrap()];
    }
    let mut best = f64::NEG_INFINITY;
    let mut ties = Vec::new();
    for &i in legal {
        let e = &node.edges[i];
        let score = is_uct(e.total_utility, e.visits, e.availability, c, root_turn);
        if score > best {
            best = score;
            ties.clear();
            ties.push(i);
        } else if score == best {
            ties.push(i);
        }
    }
    ties[rng.index(ties.len()).unwrap()]
}
fn outcome<A, O: Eq>(
    nodes: &mut Vec<Node<A, O>>,
    node: usize,
    edge: usize,
    observation: O,
) -> (usize, bool) {
    if let Some((_, child)) = nodes[node].edges[edge]
        .outcomes
        .iter()
        .find(|(o, _)| *o == observation)
    {
        return (*child, false);
    }
    let child = nodes.len();
    nodes.push(Node {
        visits: 0,
        edges: vec![],
    });
    nodes[node].edges[edge].outcomes.push((observation, child));
    (child, true)
}
impl SoIsmctsAgent {
    pub fn search<G, W, O, R>(
        &self,
        game: &G,
        observation: &O,
        observer: PlayerId,
        root_legal: &[G::Action],
        rng: &mut R,
    ) -> Result<SearchResult<G::Action, O>, AgentError>
    where
        G: TwoPlayerZeroSumGame
            + ImperfectInformationGame<Determinization = W>
            + for<'a> Game<Observation<'a> = O>
            + 'static,
        W: DeterminizedWorld<Action = G::Action, Observation = O>,
        O: Clone + Eq,
        G::Action: Clone + Eq,
        R: RandomSource + ?Sized,
    {
        self.config.validate()?;
        if game.player_count() != 2 || observer.index() >= 2 {
            return Err(error(
                "SO-ISMCTS requires a two-player game and a valid observer",
            ));
        }
        if root_legal.is_empty() {
            return Err(AgentError::NoLegalActions);
        }
        let mut nodes = vec![Node {
            visits: 0,
            edges: vec![],
        }];
        let mut diagnostics = Diagnostics::default();
        let started = Instant::now();
        loop {
            let exhausted = match self.config.budget {
                SearchBudget::Iterations(n) => {
                    diagnostics.completed_iterations >= u64::from(n.get())
                }
                SearchBudget::Time(t) => {
                    diagnostics.completed_iterations > 0 && started.elapsed() >= t
                }
            };
            if exhausted {
                break;
            }

            // This is the only world. It is dropped at the end of THIS iteration.
            let mut world = game
                .sample_determinization(observation, observer, rng)
                .map_err(error)?;
            diagnostics.determinizations_sampled += 1;
            let legal = world.legal_actions();
            if world.status() != PositionStatus::PlayerTurn(observer)
                || world.observation(observer) != *observation
                || legal.len() != root_legal.len()
                || !legal.iter().all(|a| root_legal.contains(a))
            {
                return Err(error(
                    "determinization does not preserve root observation/player/legal actions",
                ));
            }
            let mut node = 0;
            let mut path = Vec::new();
            let mut visited = vec![0];
            let mut steps = 0;
            loop {
                let player = match world.status() {
                    PositionStatus::Terminal => break,
                    PositionStatus::PlayerTurn(p) => p,
                    _ => return Err(error("SO-ISMCTS does not support chance nodes")),
                };
                if steps >= MAX_SIMULATION_ACTIONS {
                    break;
                }
                let legal = world.legal_actions();
                if legal.is_empty() {
                    return Err(AgentError::NoLegalActions);
                }
                let indices = available(&mut nodes[node], &legal);
                let edge = choose(
                    &nodes[node],
                    &indices,
                    self.config.exploration,
                    player == observer,
                    rng,
                );
                let expanding = nodes[node].edges[edge].visits == 0;
                world
                    .apply_action(&nodes[node].edges[edge].action)
                    .map_err(error)?;
                steps += 1;
                path.push((node, edge));
                let (child, new_outcome) =
                    outcome(&mut nodes, node, edge, world.observation(observer));
                node = child;
                visited.push(node);
                if expanding || new_outcome {
                    break;
                }
            }
            diagnostics.rollout_count += 1;
            while world.status() != PositionStatus::Terminal && steps < MAX_SIMULATION_ACTIONS {
                if !matches!(world.status(), PositionStatus::PlayerTurn(_)) {
                    return Err(error("SO-ISMCTS does not support chance nodes"));
                }
                let legal = world.legal_actions();
                let index = rng.index(legal.len()).ok_or(AgentError::NoLegalActions)?;
                world.apply_action(&legal[index]).map_err(error)?;
                steps += 1;
            }
            let utility = if world.status() == PositionStatus::Terminal {
                diagnostics.terminal_simulations += 1;
                world
                    .terminal_utility(observer)
                    .ok_or_else(|| error("missing terminal utility"))? as f64
            } else {
                diagnostics.cutoff_simulations += 1;
                0.
            };
            if !utility.is_finite() || !(-1. ..=1.).contains(&utility) {
                return Err(error("invalid terminal utility"));
            }
            for n in visited {
                nodes[n].visits += 1;
            }
            for (n, e) in path {
                nodes[n].edges[e].visits += 1;
                nodes[n].edges[e].total_utility += utility;
            }
            diagnostics.completed_iterations += 1;
        }
        diagnostics.tree_nodes = nodes.len();
        diagnostics.action_edges = nodes.iter().map(|n| n.edges.len()).sum();
        let most = nodes[0].edges.iter().map(|e| e.visits).max().unwrap();
        // Stable input action order resolves MostVisited ties; no hidden tie-break data.
        let action = root_legal
            .iter()
            .find(|a| {
                nodes[0]
                    .edges
                    .iter()
                    .any(|e| e.action == **a && e.visits == most)
            })
            .unwrap()
            .clone();
        Ok(SearchResult {
            action,
            diagnostics,
            nodes,
        })
    }
}
#[cfg(test)]
mod tests;
