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

`MctsAgent` supports deterministic, perfect-information, two-player, zero-sum games. For every real
decision it builds a new tree and repeats four steps:

1. Select children with UCT until reaching a node with an unexpanded action.
2. Expand one randomly selected action.
3. Play uniformly random actions until the game ends or the rollout limit is reached.
4. Backpropagate the utility from the root player's perspective.

The real action is the root child with the most visits; mean utility breaks visit ties.

Utilities stored in the tree always use the root player's perspective. During UCT selection, MCTS
reads the active player from `PositionStatus`: it maximizes the stored utility while the root
player keeps making decisions and minimizes it once control passes to the opponent. It does not
alternate maximize and minimize by tree depth. A game may therefore represent one physical turn
as several consecutive engine actions without changing the search semantics. Rollout actions are
still selected uniformly rather than by minimax.

The current agent builds a fresh tree for every real engine action. A later decision in the same
physical turn is considered by earlier simulations, then searched again from its authoritative
state when that decision is actually reached; tree reuse is not implemented yet.

```python
from meeple_bots import Match, MctsAgent, RandomAgent

agent = MctsAgent(
    iterations=1_000,
    exploration=2.0**0.5,
    rollout_depth=256,
)
result = Match(first=agent, second=RandomAgent(), seed=42).run()
```

### Configuration

| Parameter | Default | Meaning |
| --- | --- | --- |
| `iterations` | `1_000` | Tree-search iterations performed for each decision. |
| `exploration` | `sqrt(2)` | UCT balance between utility and less-visited branches. |
| `rollout_depth` | `256` | Maximum simulated actions after expansion. |
| `heuristic` | `None` | Optional zero-based game evaluator used at a rollout cutoff. |

More iterations usually improve coverage but also increase response time. Iteration count is not a
portable measure of work: action generation, branching, rollout length, and build profile all
affect the cost of one iteration.

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

Spirits of the Forest exposes heuristic `0`, which combines provisional majority scoring with
usable gemstones and active reservations. See its
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
use_heuristic = false
heuristic_index = 0
```

`iterations` and `rollout_depth` are required. `exploration` defaults to `sqrt(2)`,
`use_heuristic` defaults to `false`, and `heuristic_index` defaults to `0`. The index is ignored
unless heuristic use is enabled.

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

The current implementation does not support chance transitions, hidden information, tree reuse,
transpositions, parallel search, or learned policies and values. The possible development stages
are recorded in the [MCTS roadmap](MCTS_ROADMAP.md) and its
[Spanish translation](MCTS_ROADMAP.es.md).

## Human

`HumanAgent` lives at the Python boundary. With no selector it prompts in the terminal; with a
selector callback it can obtain actions from another interface. See
[Human-controlled matches](../python/README.md#human-controlled-matches).
