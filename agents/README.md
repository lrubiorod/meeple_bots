# Agents

[Back to the project overview](../README.md)

An agent receives one decision and returns one legal action. Rust agents are generic over the
concrete game, preserving strongly typed actions and static dispatch inside the simulation loop.

## Available agents

| Agent | Purpose | Implementation |
| --- | --- | --- |
| `HumanAgent` | Terminal or application-controlled input | Python boundary |
| `RandomAgent` | Uniform baseline and reproducible opponent | Rust |
| `MctsAgent` | Configurable Monte Carlo Tree Search | Rust |

## Random

`RandomAgent` selects uniformly from the current legal actions. It is the simplest baseline for
measuring whether a search agent learns anything useful from its budget.

```python
from meeple_bots import Match, RandomAgent

result = Match(first=RandomAgent(), second=RandomAgent(), seed=42).run()
```

Each player receives an independent random stream derived from the match seed, so the same match
configuration and seed reproduce the same decisions.

## Monte Carlo Tree Search

`MctsAgent` supports deterministic, perfect-information, two-player, zero-sum games. By default it
builds a new tree for every real decision and repeats four steps:

1. Select children with UCT until reaching a node with an unexpanded action.
2. Expand one randomly selected action.
3. Select simulated actions with the configured rollout policy until the game ends or the rollout
   limit is reached.
4. Backpropagate the utility from the root player's perspective.

The real action is the root child with the most visits; mean utility breaks visit ties.

Utilities stored in the tree always use the root player's perspective. During UCT selection, MCTS
reads the active player from `PositionStatus`: it maximizes the stored utility while the root
player keeps making decisions and minimizes it once control passes to the opponent. It does not
alternate maximize and minimize by tree depth. A game may therefore represent one physical turn
as several consecutive engine actions without changing the search semantics. Uniform rollout is
the default; informed policies use the same active-player perspective when ranking actions.

With `tree_reuse=True`, the configured wrapper follows every accepted own and opponent action,
re-roots at the matching expanded child, and discards nodes outside that subtree. This also works
when one player makes several consecutive engine actions. A missing child or any game/state
mismatch resets to a fresh tree rather than risking stale statistics. Cloning an agent for another
match and the explicit match-start lifecycle both discard retained data.

```python
from meeple_bots import Match, MctsAgent, RandomAgent

agent = MctsAgent(
    iterations=1_000,
    exploration=2.0**0.5,
    rollout_depth=256,
    tree_reuse=True,
)
result = Match(first=agent, second=RandomAgent(), seed=42).run()
```

### Configuration

| Parameter | Default | Meaning |
| --- | --- | --- |
| `iterations` | `1_000` | Exact iterations per decision; excludes `time_budget`. |
| `time_budget` | `None` | Approximate seconds per decision; excludes `iterations`. |
| `exploration` | `sqrt(2)` | UCT balance between utility and less-visited branches. |
| `rollout_depth` | `256` | Maximum simulated actions after expansion. |
| `cutoff_evaluator` | `NeutralEvaluator()` | Evaluator used only when a rollout reaches its depth cutoff. |
| `rollout_policy` | `UniformRandom()` | Policy used to select simulated actions outside the tree. |
| `progressive_bias` | `None` | Optional decaying heuristic prior added to UCT tree selection. |
| `root_diagnostics` | `false` | Record visits, utility, cached heuristic, and bias for expanded root actions. |
| `tree_reuse` | `false` | Retain the reachable subtree across decisions in the same match. |

The legacy `heuristic=INDEX` argument remains available as shorthand for
`cutoff_evaluator=GameHeuristic(INDEX)`.

More iterations usually improve coverage but also increase response time. Iteration count is not a
portable measure of work: action generation, branching, rollout length, and build profile all
affect the cost of one iteration.

Use an iteration budget for reproducible tests, debugging, and algorithmic comparisons. Use a time
budget to compare agents whose iterations have different costs:

```python
agent = MctsAgent(time_budget=2.0, rollout_depth=32)
```

The deadline is checked between iterations. MCTS always completes the current iteration and always
runs at least one, so elapsed time can exceed the requested duration by one expensive iteration.
Time-budget searches retain seeded randomness but are not exactly reproducible: CPU load and machine
speed change how many iterations finish. Match traces record the actual decision time, completed
iterations, and created nodes.

### Tree reuse

Tree reuse is deliberately optional. The unwrapped baseline `MctsAgent` keeps the original search
path and supports non-cloneable actions. The enabled reuse wrapper requires the concrete
perfect-information game, state, and action to be cloneable and comparable, but does not require
hashing or a transposition key. Utilities remain in the agent owner's perspective, while MAX/MIN
still follows the real active player stored in each state.

Per-move traces record transition attempts, hits and misses, own/opponent hits, retained root visits and
nodes, pruned nodes, and safe resets. `search_iterations` is always the new work performed for that
decision. On a reused root, `search_nodes` counts newly expanded nodes; retained nodes are reported
separately. Transition metrics are attached to the next decision because the observed action occurs
after the previous decision statistics were captured.

`analyze` benchmarks independent sampled positions. It accepts a profile containing `tree_reuse`,
but those timings do not estimate the accumulated benefit of reuse inside a continuous match. Use
paired tournaments with equal budgets to measure that benefit.

### Rollout policies

`UniformRandom` is the baseline and preserves the original MCTS behavior. It samples every legal
rollout action with equal probability and does not require a game heuristic.

`Greedy` always chooses a best-rated successor. `EpsilonGreedy` does so with probability
`1 - epsilon` and otherwise explores a uniformly random action. Both receive their own evaluator,
independent from the cutoff evaluator. They maximize for the root player and minimize for the
opponent by reading the active player from `PositionStatus`, so consecutive actions by one player
remain correct.

```python
from meeple_bots import EpsilonGreedy, GameHeuristic, MctsAgent, NeutralEvaluator

agent = MctsAgent(
    iterations=10_000,
    rollout_depth=32,
    cutoff_evaluator=NeutralEvaluator(),
    rollout_policy=EpsilonGreedy(
        epsilon=0.1,
        evaluator=GameHeuristic(1),
    ),
)
```

This example is an informed rollout without heuristic cutoff evaluation. The inverse combination
is also valid: `GameHeuristic(1)` as `cutoff_evaluator` with `UniformRandom()` rollouts. Informed
policies evaluate and copy every candidate successor, so their iterations are more expensive than
uniform-random iterations. Compare policies at equal wall-clock decision budgets as well as equal
iteration counts.

In Rust, `MctsAgent<C, P, B>` is generic over cutoff evaluator, rollout policy, and optional
selection bias. `Greedy<E>`
and `EpsilonGreedy<E>` are themselves generic over their rollout evaluator. The runtime catalog
uses configuration enums for the built-in variants, while a Rust integration can inject another
concrete evaluator or policy without duplicating the MCTS loop or paying for dynamic dispatch.

### Progressive Bias

Progressive Bias keeps the rollout policy unchanged and adds
`weight * heuristic(child) / (child_visits + 1)` to UCT selection. The child heuristic is computed
once when the node is expanded and cached. Its sign follows the real active player: nodes controlled
by the root player maximize it and opponent nodes minimize it, including games with consecutive
actions by the same player. Weight zero takes the exact baseline path and does not call the evaluator.

Conditions are checked on the parent state where the action is selected, while the evaluator scores
the resulting child. For SPOTF this allows H0 to guide only Collect without affecting gemstone
decisions:

```python
from meeple_bots import GameHeuristic, MctsAgent, ProgressiveBias, TurnPhaseIs, UniformRandom

agent = MctsAgent(
    time_budget=0.5,
    rollout_depth=130,
    cutoff_evaluator=GameHeuristic(0),
    rollout_policy=UniformRandom(),
    progressive_bias=ProgressiveBias(0.25, GameHeuristic(0), TurnPhaseIs("collect")),
    root_diagnostics=True,
)
```

### Cutoff evaluation

A rollout that reaches a terminal state uses the exact utility: `1.0` for a win, `0.0` for a draw,
and `-1.0` for a loss from the root player's perspective. A rollout that reaches its depth limit
uses the selected game heuristic or `0.0` when no heuristic is configured.

Heuristics belong to games, not to MCTS. This keeps the generic search independent from concrete
rules. Unsupported heuristic indices are rejected before a match starts.

Boop currently exposes two evaluators:

| Index | Summary |
| --- | --- |
| `0` | Difference between the players' cats currently on the board. |
| `1` | Cat progression, immediate threats, graduations, and safer board presence. |

See the [Boop guide](../games/boop/README.md#mcts-heuristics) for their exact interpretation.

Spirits of the Forest exposes only heuristic `0`, based on category progress that can still reach a
scoring threshold and stronger early- and mid-game gemstone conservation. See its
[game guide](../games/spirits-of-the-forest/README.md#mcts-heuristic).

### Reusable profiles

CLI batches and matches can load MCTS settings from TOML. Copy
[`configs/mcts/template.toml`](../configs/mcts/template.toml) and keep one profile per configuration
you want to compare:

```toml
name = "example-mcts"
iterations = 500
rollout_depth = 16
exploration = 1.4142135623730951
cutoff_evaluator = { kind = "neutral" }
rollout_policy = { kind = "epsilon_greedy", epsilon = 0.1, evaluator = { kind = "game_heuristic", index = 0 } }
tree_reuse = true
```

Exactly one of `iterations` or `time_budget` is required, together with `rollout_depth`.
`time_budget` is expressed in seconds. `exploration` defaults to `sqrt(2)`,
`cutoff_evaluator` defaults to neutral, `rollout_policy` defaults to uniform random, and
`tree_reuse` defaults to false. Evaluator
kinds currently supported by the catalog are `neutral` and `game_heuristic`. Rollout policy kinds
are `uniform_random`, `greedy`, and `epsilon_greedy`. The legacy `use_heuristic`,
`heuristic_index`, `rollout_heuristic_index`, and flat rollout fields remain accepted.

```bash
meeple-bots match --game boop --first mcts --second random \
  --first-mcts-config configs/mcts/template.toml --seed 42
```

### Choosing a budget

Use `evaluate_game` or `meeple-bots analyze` to estimate structural scale and local iteration cost:

```bash
meeple-bots analyze --game boop --samples 128 --max-depth 256 --seed 42
```

The recommendation is a starting point for experiments, not an Elo or strength guarantee. Compare
candidate agents at equal wall-clock time whenever possible. The
[evaluation guide](../crates/evaluation/README.md) explains the formula and limitations.

### Current limits and future work

The current implementation does not support chance transitions, hidden information,
transpositions, parallel search, or learned policies and values. Tree reuse is available for
compatible perfect-information games, without transposition sharing. The possible development stages
are recorded in the [MCTS roadmap](MCTS_ROADMAP.md) and its
[Spanish translation](MCTS_ROADMAP.es.md).

## Human

`HumanAgent` lives at the Python boundary. With no selector it prompts in the terminal; with a
selector callback it can obtain actions from another interface. See
[Human-controlled matches](../python/README.md#human-controlled-matches).
