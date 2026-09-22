# Single-observer ISMCTS baseline

`SoIsmctsAgent` is a separate, two-player zero-sum search agent. It accepts a player
observation, that player's legal root actions, game rules, and a search RNG. It has
no authoritative-state argument and does not implement the legacy `Agent` lifecycle.
The Lost Cities catalog adapter constructs the observation outside the search
boundary. With `tree_reuse=true`, its lifecycle callbacks advance an observation-only
`ReusableSoIsmcts` instance owned by that seat. Ordinary MCTS remains incompatible
with Lost Cities.

## One decision, many temporary worlds

Every iteration samples a **fresh complete determinization** from the same root
observation. Selection and expansion traverse a shared tree, then a uniform random
rollout finishes that same world. Terminal root utility (+1 win, 0 draw, -1 loss)
is backed up along the traversed path. The temporary world is dropped at iteration
end; only statistics and observable distinctions survive. By default each real
decision starts a new tree, with the acting player as the fixed observer.

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

## Availability and shared selection policies

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

The engine orients Q before calling the shared `BanditPolicy` in core. It supplies
`opportunities = edge.availability`; MCTS supplies parent visits instead. The
legacy Rust `is_uct` helper delegates to this same scoring policy.

`selection_policy="ucb1_tuned"` enables an experimental availability-aware variant.
For each selected edge, backup accumulates root utility and its square from the
same samples counted by visits. Population variance is `sum_squared / visits - Q²`;
opponent orientation flips only the mean. With `r = ln(availability) / visits`,
the shared score on the `[-1, 1]` scale is:

```text
oriented_Q + 2 * sqrt(r * min(1/4, variance/4 + sqrt(2*r)))
```

Numerical variance is clamped to its normalized `[0, 1/4]` range. UCB1-Tuned ignores
exploration C; finite nonnegative C remains accepted for configuration compatibility.
UCT is still the default and skips second-moment accumulation. Neither policy
changes MostVisited root selection. UCB1-Tuned's stationary-bandit assumptions do
not perfectly describe evolving tree-search statistics; no strength improvement
is implied.

Visits count actual selections. Availability counts opportunities to select that
action, not all parent visits: an opponent card action available in 4 of 10 worlds
gets availability 4. Actual player IDs determine perspective, including consecutive
Lost Cities Play/Draw micro-actions; depth parity is never used. Ties during search
are random. Final root selection is MostVisited among the supplied legal root
actions, with ties resolved in their stable input order. Every sampled world is
checked against the root observation, acting player and legal root action set.

## Configuration, diagnostics and limits

Configuration consists of positive `iterations` **or** positive `time_budget`
(seconds per decision), finite nonnegative `exploration`, and `selection_policy`
(`uct` by default or `ucb1_tuned`). Both families reuse
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
no PIMC, redeterminization, heuristic rollout, RAVE, widening, bias, MAST,
transpositions, parallel search or stochastic-node search. `study` supports
operating-budget calibration, UCT exploration, UCT/UCB1-Tuned selection and tree
reuse tuning. UCB1-Tuned baselines are accepted, but cannot tune inactive C. See
[Lost Cities study](../../python/studies.md#lost-cities-and-the-so-ismcts-study-profile). The Lost Cities debug GUI supports
SO-ISMCTS in either seat with separate iteration/exploration settings and an
observation-only search boundary, while displaying both hands for inspection.

## Optional realized-path tree reuse

Set `tree_reuse = true` in a SO-ISMCTS profile or pass it to `SoIsmctsAgent`.
Reuse is disabled by default. Each seat owns an independent tree and fixed observer;
cloning or starting/ending a match clears retained state. Standalone `search()` and
independent-position benchmarks always start fresh, including when reuse is enabled.

After each real action, reuse follows only that action's immediate edge and the
resulting owner observation. Real deck draws wait for the environment's chance
resolution; the hidden chance event itself is never passed to search. A missing
branch or incompatible owner/observation discards the retained tree. A hit compacts
the reached subtree, discarding unreachable nodes and remapping child indices.
Visits, availability, utility sums and UCB1-Tuned squared utility sums survive.
Every new iteration still samples a fresh determinization; no hidden world survives.
This is history-path reuse, not observation-based transposition merging.

Match diagnostics count newly completed iterations and newly allocated nodes;
`tree_reuse` separately records retained visits/nodes, hits, misses, pruning and
resets. Lifecycle maintenance is included in agent-total timing and deducted from
the next timed search, with a reserve for result finalization. At least one whole
iteration runs even when maintenance exhausts the allowance, so the deadline is
not a hard limit. Use equal wall-clock budgets for competitive reuse comparisons.

## Python and CLI

```bash
.venv/bin/python -m meeple_bots match --game lost_cities \
  --first so_ismcts --second random --so-ismcts-iterations 1000 \
  --so-ismcts-exploration 1.0 --seed 42
```

Add `--so-ismcts-selection-policy ucb1_tuned` for tuned selection (or `uct` for
IS-UCT). Python and scalar TOML profiles expose the same field:

```python
agent = SoIsmctsAgent(time_budget=0.1, selection_policy="ucb1_tuned")
```

```toml
agent = "so_ismcts"
time_budget = 0.1
selection_policy = "ucb1_tuned"
rollout = "uniform"
root_selection = "most_visited"
```

Configured `analyze --agent-config PATH` benchmarks support both selectors.
The debugging GUI currently retains its UCT iteration/exploration controls.

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

For cost calibration and competitive C tuning, use the same commands as other
search families:

```bash
meeple-bots analyze --game lost_cities --agent so_ismcts --samples 16 --target-time 500ms
meeple-bots study --game lost_cities --agent so_ismcts --budget 2h \
  --target-match-time 60s --output results/studies/lost-cities-so
```

`analyze` reports iterations/determinizations, latency, tree size and root coverage,
including isolated determinization cost. `study` races exploration C under equal
compute and uses fresh-seed confirmation. Neither treats hidden-world enumeration
as a search target. See [analysis](../../crates/evaluation/README.md) and
[study/local retuning](../../python/studies.md#lost-cities-and-the-so-ismcts-study-profile).

Python `Batch` and `TournamentAgent` also accept `SoIsmctsAgent`; the CLI tournament
TOML parser and batch profile loader retain their existing MCTS/random configuration
formats. Administrative match traces still contain real private draws and must not
be supplied to search agents.
