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
  the limit supplies its utility; a rollout that reaches the limit is evaluated as `0.0`.

Iteration count alone does not represent equal work across games. The cost of one iteration also
depends on legal-action generation, branching, and rollout length.

## Calibrated levels

The complexity evaluator can convert a target compute budget into fixed MCTS parameters for one
game and machine:

```python
from meeple_bots import Boop, MctsAgent, MctsLevel, evaluate_game_complexity

report = evaluate_game_complexity(Boop(), seed=42)
recommendation = report.recommend(MctsLevel.BALANCED)
agent = MctsAgent.from_recommendation(recommendation)
```

The built-in levels target these approximate decision times:

| Level | Target |
| --- | ---: |
| `FAST` | 100 ms |
| `BALANCED` | 500 ms |
| `THOROUGH` | 2000 ms |

They express compute budgets, not guaranteed strength or Elo. The resulting `iterations` and
`rollout_depth` are fixed before the match, preserving seeded reproducibility. See the
[complexity evaluation guide](../crates/evaluation/README.md) for details and limitations.

The equivalent CLI configuration is:

```bash
meeple-bots match --game boop --first mcts --second random \
  --mcts-level balanced --mcts-time-ms 750 --seed 42
```

Manual `--mcts-iterations` and `--mcts-rollout-depth` values cannot be combined with
`--mcts-level`.

## Search diagnostics

The strength evaluator can record the nodes and depths actually reached by MCTS, plus whether each
rollout found a terminal utility or stopped at `rollout_depth` and received `0.0`. It combines
these metrics with paired matches against random play and a four-times-iteration reference:

```bash
meeple-bots assess --game boop --mcts-level fast --matches 20 --seed 42
```

A high truncated-rollout rate is evidence specifically for a heuristic at the rollout cutoff. The
report presents iteration-budget sufficiency and benchmark confidence separately. A tiny fraction
of the estimated global tree is normal for MCTS and is not, by itself, evidence of weak play. See
the [strength assessment guide](../crates/evaluation/README.md#mcts-strength-assessment).

## Human

`HumanAgent` is implemented at the Python boundary. It either prompts in the terminal or delegates
to a Python selector function. See the [Python interface guide](../python/README.md#human-players).
