# Game evaluation

[Back to the Rust architecture](../README.md) · [Back to the project overview](../../README.md)

The evaluation crate gives one compact view of a game's shape and the approximate local cost of a
reasonable MCTS configuration. It samples games and times a small MCTS probe; it does not play
benchmark matches or prove playing strength.

## Python API

```python
from meeple_bots import Boop, evaluate_game

report = evaluate_game(Boop(), samples=128, max_depth=256, seed=42)

print(report.initial_legal_actions)
print(report.estimated_depth)
print(report.recommended_iterations)
print(report.milliseconds_per_iteration)
print(report.estimated_decision_time_ms)
```

## Report

For every sample, the evaluator starts at the initial state and selects uniformly from the legal
actions until the game ends or `max_depth` is reached. A fixed seed makes all structural fields
reproducible.

| Field | Meaning |
| --- | --- |
| `initial_legal_actions` | Exact number of choices in the initial position. |
| `effective_branching_factor` | Geometric mean of sampled legal-action counts. |
| `estimated_depth` | 95th percentile of sampled game lengths. |
| `depth_is_lower_bound` | Whether at least one sample reached `max_depth` without finishing. |
| `terminal_rate` | Fraction of samples that reached a terminal state. |
| `estimated_tree_log10` | Approximate base-10 logarithm of the sampled game tree. |
| `recommended_rollout_depth` | The sampled P95 depth used for MCTS calibration. |
| `recommended_iterations` | Structural estimate for a reasonable MCTS budget. |
| `iterations_capped` | Whether that estimate was limited to 1,000,000 iterations. |
| `milliseconds_per_iteration` | Median local cost measured by three MCTS probes. |
| `estimated_decision_time_ms` | Recommended iterations multiplied by measured iteration cost. |

## Iteration estimate

The uncapped estimate is:

```text
initial legal actions × effective branching factor × estimated depth²
```

It is rounded upward to a readable `1`, `2`, or `5 × 10ⁿ` budget and capped at 1,000,000. The
formula rewards root coverage, accounts for alternatives below the root, and applies a quadratic
penalty to long tactical horizons. It is deliberately simple and is not a theorem, Elo estimate,
or guarantee of optimal play.

The tree-size estimate uses the initial action count, effective branching factor, and P95 depth. It
describes order of magnitude only; transpositions, uneven branches, cycles, and policy choices are
not modeled exactly.

## Hardware calibration and heuristics

The evaluator runs three neutral MCTS searches of 100 iterations at the recommended rollout
depth and uses their median time per iteration. Timing depends on CPU, system load, build profile,
and the initial state, so the total is approximate.

The report intentionally provides evidence instead of a universal heuristic verdict:

- If `depth_is_lower_bound` is true, increase `max_depth` or consider a heuristic for truncated
  rollouts.
- If full-depth `estimated_decision_time_ms` is acceptable, terminal rollouts may be sufficient.
- If the estimated time is too high, reducing rollout depth saves work but makes a state heuristic
  more useful at the cutoff.
- `iterations_capped` signals that even the simple structural estimate exceeded the practical
  ceiling.

## CLI

```bash
meeple-bots analyze --game boop --samples 128 --max-depth 256 --seed 42
meeple-bots analyze --game boop --samples 128 --max-depth 256 --seed 42 --json
```

The human-readable output separates game structure from the local MCTS estimate. Matches remain
manually configurable with `--mcts-iterations`, `--mcts-rollout-depth`, and the per-player heuristic
flags.

## Rust API

`evaluate_game<G>` accepts an `EvaluationConfig` and returns `GameEvaluationReport`. It supports
deterministic, perfect-information, two-player, zero-sum games with cloneable states and actions.
