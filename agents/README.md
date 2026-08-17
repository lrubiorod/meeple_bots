# Agents

[Back to the project overview](../README.md)

Agents select one action from the legal actions exposed by a game. Rust agents are generic over the
concrete game type, so the simulation loop retains static dispatch and strongly typed actions.

## Random

`RandomAgent` selects uniformly from the current legal actions. It is useful as a baseline, a
rollout policy, and a reproducible test opponent.

```python
from meeple_bots import Match, RandomAgent

result = Match(first=RandomAgent(), second=RandomAgent(), seed=42).run()
```

Each player receives an independent random stream derived from the match seed.

## Monte Carlo Tree Search

`MctsAgent` uses UCT selection and uniformly random rollouts for deterministic, perfect-information,
two-player, zero-sum games.

```python
from meeple_bots import Match, MctsAgent, RandomAgent

agent = MctsAgent(
    iterations=1_000,
    exploration=2.0**0.5,
    rollout_depth=256,
)
result = Match(first=agent, second=RandomAgent(), seed=42).run()
```

### Parameters

- `iterations` controls how many MCTS simulations are performed for each decision. More iterations
  usually improve coverage but increase response time.
- `exploration` controls the UCT balance between known high-utility branches and less-visited
  branches. The default is `sqrt(2)`.
- `rollout_depth` limits actions simulated after tree expansion. A terminal result reached within
  the limit supplies its utility; a rollout that reaches the limit uses the selected game
  heuristic or `0.0` when none is selected.
- `heuristic` optionally selects a zero-based heuristic index provided by the game. It defaults to
  `None`.

Iteration count alone does not represent equal work across games. The cost of one iteration also
depends on legal-action generation, branching, and rollout length.

### Game heuristics

Heuristics belong to games rather than to MCTS. A game may expose no heuristics or several indexed
variants. Each variant evaluates a truncated state from the root player's perspective and returns a
normalized utility between `-1.0` and `1.0`.

The availability and meaning of each index are documented by the game that provides it. MCTS treats
the selected index as an opaque evaluator, which keeps the agent independent from concrete game
rules. For example, a game-specific heuristic can be selected for shorter rollouts:

```python
from meeple_bots import Boop, Match, MctsAgent, RandomAgent

result = Match(
    game=Boop(),
    first=MctsAgent(iterations=1_000, rollout_depth=16, heuristic=1),
    second=RandomAgent(),
).run()
```

Passing an index that the selected game does not provide is an error. See the
[games guide](../games/README.md) and each game's README for the available evaluators.

### Reusable profiles

CLI matches and batch simulations load MCTS parameters from TOML text files. Copy the provided
[`template.toml`](../configs/mcts/template.toml) and edit its values:

```toml
name = "boop-baseline"
iterations = 100
rollout_depth = 16
exploration = 1.4142135623730951
use_heuristic = false
heuristic_index = 0
```

`iterations` and `rollout_depth` are required. `exploration` defaults to `sqrt(2)`,
`use_heuristic` defaults to `false`, and `heuristic_index` defaults to `0`. The index is used only
when `use_heuristic` is true.

## Choosing a search budget

The game evaluator provides a simple starting point for choosing manual MCTS parameters:

```python
from meeple_bots import Boop, MctsAgent, evaluate_game

report = evaluate_game(Boop(), samples=128, max_depth=256, seed=42)
agent = MctsAgent(
    iterations=report.recommended_iterations,
    rollout_depth=report.recommended_rollout_depth,
)
```

It samples random games to estimate the initial action count, 95th-percentile depth, effective
branching factor, and approximate tree size. It then recommends a rounded iteration count using:

```text
initial actions × effective branching × estimated depth²
```

The count is capped at one million iterations. A short local calibration estimates the duration of
one iteration and of the recommendation on the current machine. These values estimate compute
requirements, not playing strength or Elo. See the
[game evaluation guide](../crates/evaluation/README.md) for details and limitations.

The equivalent CLI configuration is:

```bash
meeple-bots match --game boop --first mcts --second random \
  --mcts-iterations 50000 --mcts-rollout-depth 64 --seed 42
```

## Human

`HumanAgent` is implemented at the Python boundary. It either prompts in the terminal or delegates
to a Python selector function. See the [Python interface guide](../python/README.md#human-players).
