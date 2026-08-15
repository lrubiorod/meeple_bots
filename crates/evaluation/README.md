# Complexity evaluation

[Back to the Rust architecture](../README.md) · [Back to the project overview](../../README.md)

The evaluation crate samples reachable positions and performs a short local MCTS calibration. Its
goal is to make branching, depth, and approximate computation cost visible before choosing fixed
agent parameters.

It does not calculate the exact game-tree size or guarantee equivalent playing strength between
different games.

## Python API

```python
from meeple_bots import Boop, MctsAgent, MctsLevel, evaluate_game_complexity

report = evaluate_game_complexity(
    Boop(),
    samples=128,
    max_depth=256,
    seed=42,
)

recommendation = report.recommend(MctsLevel.BALANCED)
agent = MctsAgent.from_recommendation(recommendation)
```

Use a custom target without repeating the calibration:

```python
recommendation = report.recommend(
    MctsLevel.BALANCED,
    time_budget_ms=750,
)
```

## Structural sampling

For each sample, the evaluator starts at the initial state and selects uniformly from legal actions
until reaching a terminal position or `max_depth`. Structural sampling is deterministic for a fixed
game, seed, sample count, and depth limit.

The report exposes:

| Field | Meaning |
| --- | --- |
| `initial_legal_actions` | Number of choices in the initial position. |
| `mean_branching_factor` | Arithmetic mean of sampled legal-action counts. |
| `effective_branching_factor` | Geometric mean, reducing domination by a few extreme positions. |
| `p95_branching_factor` | Branching factor at the sampled 95th percentile. |
| `maximum_branching_factor` | Largest sampled legal-action count. |
| `mean_plies` | Arithmetic mean sampled game length. |
| `median_plies` | 50th-percentile sampled game length. |
| `p75_plies` | 75th-percentile sampled game length. |
| `p95_plies` | 95th-percentile sampled game length. |
| `terminal_rate` | Fraction of samples reaching a terminal position. |
| `completed_samples` | Absolute number of terminal samples. |
| `estimated_tree_log10` | Approximate base-10 logarithm of the game-tree size. |
| `estimate_is_lower_bound` | Whether at least one sample was truncated by `max_depth`. |

The tree estimate is calculated from the effective branching factor and p95 game length. It is an
order-of-magnitude comparison, not an enumeration of distinct states. Transpositions, repeated
positions, uneven branches, and policy choice all limit its precision.

When samples hit `max_depth`, their true lengths are unknown. The report treats the estimate as a
lower bound and exposes the terminal rate rather than silently presenting truncated data as final.

## Depth recommendations

Recommended rollout depths use sampled game-length percentiles and explicit ceilings:

| Level | Sample statistic | Maximum depth | Default time target |
| --- | --- | ---: | ---: |
| `FAST` | median | 64 | 100 ms |
| `BALANCED` | p75 | 128 | 500 ms |
| `THOROUGH` | p95 | 256 | 2000 ms |

The caller's `max_depth` remains an additional upper bound for all three levels. These ceilings
prevent a game with cycles or very long sampled matches from producing unbounded rollouts.

MCTS returns utility `0.0` when a rollout reaches its limit without finding a terminal state and no
heuristic is selected. When configured, the selected game heuristic evaluates that truncated
position instead. A short recommendation still trades tactical visibility for response time.

## Hardware calibration

For each recommended depth, Rust times three small MCTS probes and uses the median milliseconds per
iteration. The target time is divided by that cost to estimate a fixed iteration count.

Recommendations obey these limits:

- At least one iteration per initial legal action, allowing every root action to be expanded once.
- At most 1,000,000 iterations.
- A positive rollout depth no greater than the selected level and analysis ceilings.

In a wide or expensive game, the minimum root coverage can exceed the requested target. Compare
`estimated_time_ms` with `target_time_ms` to identify that case.

Timing depends on processor, system load, compiler optimization, and the sampled initial search.
Structural metrics are reproducible, but timing fields and recommended iteration counts can vary.
Once a recommendation is applied, matches use fixed iteration and depth values and remain seeded
and reproducible.

## MCTS strength assessment

Complexity and iteration counts cannot predict absolute playing strength. The optional strength
assessment therefore combines search telemetry with paired matches:

```python
from meeple_bots import Boop, MctsAgent, evaluate_game_complexity, evaluate_mcts_strength

game = Boop()
complexity = evaluate_game_complexity(game, seed=42)
report = evaluate_mcts_strength(
    game,
    MctsAgent(iterations=1_000, rollout_depth=64),
    matches_per_opponent=20,
    seed=42,
    complexity_report=complexity,
)
```

The candidate plays the same even number of matches against `RandomAgent` and an MCTS reference
with four times as many iterations. Each seed is used twice with player positions swapped. The
report gives wins, draws, losses, score, mean utility, and a Wilson 95% confidence interval from the
candidate's perspective. Wilson intervals remain appropriately wide when a tiny sample happens to
contain identical results.

The top-level assessment separates four questions:

| Field | Meaning |
| --- | --- |
| `search_sufficiency` | Whether the iteration budget revisits branches enough to compare them. |
| `benchmark_confidence` | Confidence supported by the number of matches per opponent. |
| `strength_estimate` | Relative result, or `INCONCLUSIVE` when the sample is too small. |
| `cutoff_heuristic_evidence` | Evidence for evaluating rollouts truncated at the depth limit. |

Search is `INSUFFICIENT` below two mean iterations per root action or 10% existing-tree revisits,
`LIMITED` below ten iterations per action or 50% revisits, and `ADEQUATE` otherwise. These are
diagnostic thresholds, not guarantees of strong play. Benchmark confidence is `LOW` below 20
matches per opponent, `MODERATE` from 20 to 99, and `HIGH` from 100 onward.

The observed search metrics also include expanded nodes and actual tree and simulation depths. The
cutoff-specific evidence measures how often `rollout_depth` requires either a neutral fallback or a
heuristic estimate; it does not measure overall agent quality.

| Cutoff heuristic evidence | Truncated rollouts | Interpretation |
| --- | ---: | --- |
| `LOW` | below 10% | The depth limit rarely discards a rollout result. |
| `MODERATE` | 10% to below 50% | A heuristic may improve some simulations. |
| `HIGH` | 50% or more | Most simulations may lack a meaningful terminal utility. |

`tree_size_log10_gap` compares initial expanded nodes with the sampled tree estimate in orders of
magnitude. It describes scale only: MCTS is designed to search a selective fraction of the tree,
so this gap is not used to classify strength or heuristic evidence.

The result is relative, not an Elo rating or proof of optimal play. Beating random establishes only
a basic baseline, while the stronger reference is still the same algorithm and is not a game
solver. Wider confidence intervals mean that more matches are needed before drawing a conclusion.

## CLI

Print a human-readable report:

```bash
meeple-bots analyze --game boop --samples 128 --max-depth 256 --seed 42
```

Request machine-readable output:

```bash
meeple-bots analyze --game boop --samples 128 --max-depth 256 --seed 42 --json
```

Apply a calibrated configuration to each selected MCTS player:

```bash
meeple-bots match --game boop --first human --second mcts \
  --mcts-level balanced --mcts-time-ms 750 --seed 42
```

`--mcts-level` cannot be combined with `--mcts-iterations` or `--mcts-rollout-depth`.

Benchmark a calibrated or manually configured MCTS:

```bash
meeple-bots assess --game boop --mcts-level fast --matches 20 --seed 42
meeple-bots assess --game boop --mcts-iterations 1000 \
  --mcts-rollout-depth 64 --matches 20 --json
```

The assessment is opt-in because playing 40 total matches can take several minutes for expensive
games or large MCTS budgets.

The CLI prints setup status and a `Started`/`Completed` message for every match to standard error.
Each completion includes the candidate result, plies, and elapsed time. Progress remains visible
with `--json` without corrupting the JSON written to standard output.

## Rust API

`evaluate_game<G>` and `evaluate_mcts_strength<G>` are generic over deterministic,
perfect-information, two-player, zero-sum games whose state and action can be cloned.
`ComplexityConfig` controls structural sampling and calibration, while `StrengthConfig` controls
the candidate, paired matches, stronger reference, ply limit, and seed. The catalog performs
runtime game dispatch for Python.
