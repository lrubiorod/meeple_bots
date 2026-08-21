# Game evaluation

[Rust architecture](../README.md) · [Project overview](../../README.md)

The evaluation crate separates two different questions:

1. What does the sampled game tree look like?
2. Which MCTS configurations fit practical decision-time budgets on this machine?

It does not benchmark playing strength, select an optimal agent, or claim that any suggested
configuration is strong. The suggestions are starting points for head-to-head experiments.

## Quick use

From the CLI:

```bash
meeple-bots analyze --game boop --samples 128 --max-depth 256 --seed 42
meeple-bots analyze --game boop --target-time 5 --seed 42 --json
```

From Python:

```python
from meeple_bots import Boop, evaluate_game

report = evaluate_game(
    Boop(), samples=128, max_depth=256, seed=42, target_time=5
)

print(report.depth_p50, report.estimated_depth)
print(report.player_turn_depth_p50, report.player_turn_depth_p95)
for cost in report.rollout_costs:
    print(cost.rollout_depth, cost.iteration_budgets)
for experiment in report.suggested_experiments:
    print(experiment)
```

Structural fields are reproducible for the same game, configuration, and seed. Timing fields and
the iteration counts derived from them also depend on the machine, build profile, and system load.

## Structural sampling

Each sample starts at the initial state and chooses legal actions uniformly at random until it
reaches a terminal position or `max_depth`.

A ply is one call to `apply_action`. A player turn is a maximal consecutive block of plies whose
states report the same `PlayerId`. Evaluation always reads `Game::status`; it does not assume that
one action changes the player. A path containing action blocks of lengths `3, 2, 1` therefore has
six plies, three player turns, and two player changes.

MCTS follows the same rule. Selection maximizes at every node controlled by the root player and
minimizes at every node controlled by the opponent. Consecutive phases owned by one player do not
alternate maximize/minimize merely because tree depth increased.

### Structural fields

| Field | Meaning |
| --- | --- |
| `initial_legal_actions` | Exact number of choices in the initial position. |
| `effective_branching_factor` | Geometric mean of sampled legal-action counts per tree node. |
| `player_turn_choice_product_log10` | Mean sampled log10 product of legal-action counts within one same-player block. This is an observed interpretation aid, not an exact decision count. |
| `depth_p50` / `estimated_depth` | P50 and P95 sampled path lengths in plies. `estimated_depth` remains the compatibility name for P95. |
| `player_turn_depth_p50` / `player_turn_depth_p95` | P50 and P95 path lengths measured in same-player blocks. |
| `player_changes_p50` / `player_changes_p95` | P50 and P95 changes of active player along a path. |
| `actions_per_player_turn_*` | Mean, P95, and maximum sampled block length in plies. |
| `depth_is_lower_bound` | Whether at least one sample hit `max_depth` before finishing. |
| `terminal_rate` | Fraction of samples that reached a terminal state. |
| `estimated_tree_log10` | Approximate base-10 logarithm of structural tree size. |

The player-turn choice product can overstate strategically distinct choices when later phases are
forced or converge to equivalent states. It is deliberately displayed as an observed
approximation rather than a precise branching factor.

## Practical MCTS calibration

Evaluation derives up to four rollout depths from the sampled P95 ply depth. They span short,
medium, long, and sampled-full horizons. Each depth is also converted to approximate player turns
using the observed mean actions per player turn.

For every candidate depth, a neutral MCTS search is timed at the initial state and at up to two
seeded intermediate sampled positions. A small probe chooses an adaptive iteration count targeting
a short timing window; that count is clamped so analysis remains bounded. The median measured
cost across positions becomes `milliseconds_per_iteration` for that depth.

The standard 1, 2, 5, 10, and 20 second iteration budgets are computed from those measurements and
rounded down to two significant digits. They are approximations: tree reuse, allocator behavior,
CPU scaling, position shape, and system load can change real decision time.

`--target-time SECONDS` (or Python's `target_time`) controls the suggested benchmark points. It
does not make calibration run for that many seconds.

### Practical fields

| Field | Meaning |
| --- | --- |
| `calibration_positions` | Number of sampled positions timed per rollout depth. |
| `rollout_costs` | Candidate depths, approximate player-turn horizons, measured costs, and standard time budgets. |
| `target_time_seconds` | Requested per-decision target for suggested experiments. |
| `suggested_experiments` | Fast, Balanced, Wide, and when available Deep starting points. |

Fast, Balanced, and Wide keep the same medium rollout depth while increasing iterations. Deep uses
the next longer calibrated horizon at approximately the Balanced time target. This creates a small
experiment that distinguishes search-width gains from horizon gains:

- If Wide improves over Balanced, additional search width still helps.
- If Deep improves over Balanced at similar time, horizon or cutoff quality is likely the tighter
  constraint.
- If neither improves, test or improve a state heuristic before spending much more compute.

These are experiment interpretations, not automatic diagnoses. Playing strength must be measured
in matches, preferably with alternating sides, fixed seed sets, and equal wall-clock budgets.

## Compatibility fields

`recommended_rollout_depth`, `recommended_iterations`, `iterations_capped`,
`milliseconds_per_iteration`, and `estimated_decision_time_ms` remain available. They are aliases
for the Balanced experiment and its calibrated depth, not structural tree-size recommendations.
New integrations should prefer `rollout_costs` and `suggested_experiments`.

The tree-size estimate is still an order-of-magnitude structural description. It no longer drives
the practical iteration suggestions.

## Rust API

`evaluate_game<G>` accepts an `EvaluationConfig` and returns `GameEvaluationReport`. It supports
deterministic, perfect-information, two-player, zero-sum games with cloneable states and actions.
No game-specific phase API is required: player-turn blocks come from `Game::status`.
