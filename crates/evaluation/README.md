# Game evaluation

[Rust architecture](../README.md) · [Project overview](../../README.md)

The evaluation crate answers two practical questions before running a large experiment:

1. What does the sampled game tree look like?
2. Approximately how expensive is a plausible MCTS decision on this machine?

It does not benchmark playing strength, select an optimal agent, or prove that a budget is
sufficient.

## Quick use

From the CLI:

```bash
meeple-bots analyze --game boop --samples 128 --max-depth 256 --seed 42
meeple-bots analyze --game boop --samples 128 --max-depth 256 --seed 42 --json
```

From Python:

```python
from meeple_bots import Boop, evaluate_game

report = evaluate_game(Boop(), samples=128, max_depth=256, seed=42)

print(report.initial_legal_actions)
print(report.estimated_depth)
print(report.recommended_iterations)
print(report.estimated_decision_time_ms)
```

Structural fields are reproducible for the same game, configuration, and seed. Timing fields also
depend on the machine, system load, and native build profile.

## How sampling works

For every sample, evaluation starts at the initial state and repeatedly chooses a legal action
uniformly at random. Sampling stops at a terminal position or at `max_depth`.

The report summarizes those sampled paths and then runs three neutral MCTS probes of 100 iterations
at the recommended rollout depth. Their median time per iteration calibrates the structural budget
on the current machine.

## Report fields

### Game shape

| Field | Meaning |
| --- | --- |
| `initial_legal_actions` | Exact number of choices in the initial position. |
| `effective_branching_factor` | Geometric mean of sampled legal-action counts. |
| `estimated_depth` | 95th percentile of sampled path lengths. |
| `depth_is_lower_bound` | Whether at least one sample hit `max_depth` before finishing. |
| `terminal_rate` | Fraction of samples that reached a terminal state. |
| `estimated_tree_log10` | Approximate base-10 logarithm of tree size. |

### Suggested search scale

| Field | Meaning |
| --- | --- |
| `recommended_rollout_depth` | Sampled P95 depth used for MCTS calibration. |
| `recommended_iterations` | Rounded structural estimate for an initial MCTS budget. |
| `iterations_capped` | Whether the estimate reached the 1,000,000 iteration ceiling. |
| `milliseconds_per_iteration` | Median local cost measured by the probe searches. |
| `estimated_decision_time_ms` | Suggested iterations multiplied by measured iteration cost. |

## Budget estimate

The uncapped iteration estimate is:

```text
initial actions × effective branching factor × estimated depth²
```

It is rounded upward to a readable `1`, `2`, or `5 × 10ⁿ` value and capped at 1,000,000. The
formula rewards root coverage, includes alternatives below the root, and penalizes long tactical
horizons. It is deliberately simple: it is not a theorem, Elo estimate, or guarantee of optimal
play.

The tree-size estimate is also an order-of-magnitude description. It does not model uneven
branches, cycles, transpositions, pruning, or search-policy preferences exactly.

## Reading the result

- If `depth_is_lower_bound` is true, increase `max_depth` before treating depth or tree size as
  representative.
- If the full-depth estimated time is acceptable, terminal rollouts may be sufficient.
- If it is too expensive, shorter rollouts reduce cost but make a cutoff heuristic more important.
- If `iterations_capped` is true, the structural formula exceeded its practical safety ceiling.
- Compare final agent configurations in real matches and at equal wall-clock time.

The report is most useful for choosing an initial range of experiments. It should not decide the
final MCTS parameters on its own.

## Rust API

`evaluate_game<G>` accepts an `EvaluationConfig` and returns `GameEvaluationReport`. It currently
supports deterministic, perfect-information, two-player, zero-sum games with cloneable states and
actions.
