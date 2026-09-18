# Single-observer ISMCTS baseline

`SoIsmctsAgent` is a separate, two-player zero-sum search agent. It accepts a player
observation, that player's legal root actions, game rules, and a search RNG. It has
no authoritative-state argument and does not implement the legacy `Agent` lifecycle.
The Lost Cities catalog adapter constructs the observation outside the search
boundary; its lifecycle callbacks are no-ops. Ordinary MCTS remains incompatible
with Lost Cities.

## One decision, many temporary worlds

Every iteration samples a **fresh complete determinization** from the same root
observation. Selection and expansion traverse a shared tree, then a uniform random
rollout finishes that same world. Terminal root utility (+1 win, 0 draw, -1 loss)
is backed up along the traversed path. The temporary world is dropped at iteration
end; only statistics and observable distinctions survive. Each real decision
starts a new tree, with the acting player as the fixed observer.

The optional core `DeterminizedWorld` capability supplies legal actions, observation,
action application, status and terminal utility without RNG-driven transitions.
Lost Cities implements it on its existing simulation world: unknown physical cards
are shuffled, allocated to the opponent hand, then kept as an ordered remaining
deck. `DrawDeck` consumes that order deterministically. There are **no Chance nodes**
in this search, nor another draw sample after determinization. Real Lost Cities
execution still resolves deck draws using the independent environment RNG.

## History tree and observable outcomes

Nodes store visits and action edges. Edges store action identity, visits, total
root utility, availability, and `(root observation, child node)` outcomes. No node
or edge contains a simulation world, hidden opponent hand or deck order.
Hypothetical opponent action identities are the union encountered across worlds;
they do not record which hidden world was sampled.

Observation snapshots do not encode full history, so observations are compared
**only beneath the same parent action edge**. There is no global observation map
or transposition merging. This preserves the history accumulated within this
search. The same action can have multiple children: an observer drawing red 8 or
blue 5 sees different hands. An opponent drawing those cards privately instead
produces the same observer observation and shares the child.

Sampling remains uniform using Lost Cities' existing observation snapshot; this
baseline does not add inference or a cross-decision private-history model.

## Availability and IS-UCT

At each selection/expansion visit, obtain the legal actions in the current world.
Record every legal action, including unexpanded ones, and increment its availability
exactly once. Unavailable edges receive no increment and cannot be chosen. Keeping
zero-visit edge records preserves availability before an action is first expanded.
A frontier node reached immediately before rollout has no selection opportunity;
its backed-up node visits may therefore exceed its outgoing availability counts.

Choose an unvisited legal action uniformly to expand. Otherwise select among only
currently legal edges using:

```text
Q = total_root_utility / visits
IS-UCT = actor_sign * Q + C * sqrt(ln(availability) / visits)
actor_sign = +1 for the root player, -1 for their opponent
```

Visits count actual selections. Availability counts opportunities to select that
action, not all parent visits: an opponent card action available in 4 of 10 worlds
gets availability 4. Actual player IDs determine perspective, including consecutive
Lost Cities Play/Draw micro-actions; depth parity is never used. Ties during search
are random. Final root selection is MostVisited among the supplied legal root
actions, with ties resolved in their stable input order. Every sampled world is
checked against the root observation, acting player and legal root action set.

## Configuration, diagnostics and limits

Configuration consists of positive `iterations` **or** positive `time_budget`
(seconds per decision), and finite nonnegative `exploration`. Both families reuse
the core `SearchBudget` type. Search uses the caller's seeded RNG. Timed search
checks the deadline between iterations and completes at least one; one simulation
may overrun the deadline. Fixed-iteration mode retains deterministic seeded behavior.
Uniform rollout and MostVisited are fixed. Unsupported MCTS options are rejected.
A simulation safety cap of 10,000 actions handles cycles with neutral cutoff utility;
it is **not** a game horizon or a scored game termination.

Results expose completed iterations, tree nodes, action records, determinizations
sampled, rollout evaluations (including already-terminal leaves), terminal/cutoff
counts and an inspectable tree. Edges expose visits, availability and root-perspective
Q. Python diagnostics expose child IDs, without any hidden-state handle. Match
traces carry the existing search iterations/nodes/root-action statistics; full
diagnostics are available through `search()`.

Single Observer uses one root perspective, including opponent turns. It does not
solve the opponent-model limitations addressed by MO-ISMCTS or RIS-MCTS. There is
no PIMC, redeterminization, heuristic rollout, RAVE, widening, bias, MAST, reuse,
transpositions, parallel search or stochastic-node search. `study` supports operating-budget calibration and exploration tuning; see
[Lost Cities study](../../python/studies.md#lost-cities-and-the-so-ismcts-study-profile). The Lost Cities debug GUI supports
SO-ISMCTS in either seat with separate iteration/exploration settings and an
observation-only search boundary, while displaying both hands for inspection.

## Python and CLI

```bash
.venv/bin/python -m meeple_bots match --game lost_cities \
  --first so_ismcts --second random --so-ismcts-iterations 1000 \
  --so-ismcts-exploration 1.0 --seed 42
```

Use `--second so_ismcts` for self-play. Both seats use the supplied search settings,
but each searches from its own observation with its own search RNG.

```python
from meeple_bots import LostCities, Match, RandomAgent, SoIsmctsAgent

game = LostCities()
agent = SoIsmctsAgent(iterations=1000, exploration=1.0)
result = Match(game=game, first=agent, second=RandomAgent(), seed=42).run()

state = game.initial_state(seed=42)  # Environment-owned administrative state.
observation = game.observation(state, observer=0)
legal = game.legal_actions(state)
search = agent.search(observation, legal, seed=123)  # No state passed to the agent.
print(search['action'], search['diagnostics'])
print(search['nodes'][0]['edges'])
```

Python `Batch` and `TournamentAgent` also accept `SoIsmctsAgent`; the CLI tournament
TOML parser and batch profile loader retain their existing MCTS/random configuration
formats. Administrative match traces still contain real private draws and must not
be supplied to search agents.
