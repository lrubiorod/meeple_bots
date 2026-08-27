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
meeple-bots analyze --game boop --target-time 1 \
  --agent 'iterations=5000,depth=120' \
  --agent 'name=equal-time,time_budget=1,depth=120' \
  --agent 'name=rollout-h1,iterations=20000,depth=32,ce=neutral,p=epsilon,rh=1,e=0.1' \
  --agent-config configs/mcts/heuristic.toml
```

From Python:

```python
from meeple_bots import Boop, MctsAgent, benchmark_mcts_agent, evaluate_game

report = evaluate_game(
    Boop(), samples=128, max_depth=256, seed=42, target_time=5
)

print(report.depth_p50, report.estimated_depth)
print(report.player_turn_depth_p50, report.player_turn_depth_p95)
for cost in report.rollout_costs:
    print(cost.rollout_depth, cost.iteration_budgets)
for experiment in report.suggested_experiments:
    print(experiment)

benchmark = benchmark_mcts_agent(
    Boop(),
    MctsAgent(iterations=10_000, rollout_depth=32, heuristic=0),
    median_depth=report.depth_p50,
    seed=42,
)
print(benchmark.decision_time_mean_ms, benchmark.decision_time_p95_ms)
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

This neutral calibration does not run a requested production profile. In particular, it does not
include a configured heuristic and its adaptive probe may use a different iteration count. Use
configured agent benchmarks when exact profile latency matters.

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

## Configured agent benchmarks

Repeat `--agent-config PATH` to measure any number of scalar MCTS profiles. Every profile is run
with its exact iteration or time budget, rollout depth, exploration constant, and evaluators on the
same seeded positions. Profile names must be unique. Measurements run sequentially so one compared
agent does not compete with another benchmark for CPU time.

Each path uses the [reusable scalar profile format](../../agents/README.md#reusable-profiles), not
a tournament TOML or an agent grid containing arrays.

For quick experiments, repeat `--agent [SPEC]` instead of creating files. `SPEC` contains
comma-separated `key=value` fields:

```bash
meeple-bots analyze --game spotf \
  --agent \
  --agent 'i=5000,d=120' \
  --agent 'name=timed,t=1,d=120' \
  --agent 'name=horizon,i=20000,d=32,h=0' \
  --agent 'name=informed,i=10000,d=32,ce=neutral,p=epsilon,rh=0,e=0.1' \
  --agent 'name=collect-h0,i=5000,d=32,h=0,p=epsilon,rh=0,e=0.25,phase=collect' \
  --agent 'name=collect-pb-h0,t=0.5,d=130,h=0,pb=0.25,pbh=0,pbphase=collect,rd=true' \
  --agent 'iterations=10000,exploration=0.8'
```

Supported fields are `name`, `iterations` or `time_budget`, `depth`, `exploration`, cutoff
`heuristic` or `cutoff_evaluator`, rollout `policy`, `rollout_heuristic`, `epsilon`, and optional
turn `phase`. Progressive Bias adds `progressive_bias`, `progressive_bias_heuristic`, and optional
`progressive_bias_phase`; `root_diagnostics` enables root-action statistics. Their short aliases
are `i` or `t`, `d`, `c`, `h` or `ce`, `p`, `rh`, `e`, `pb`, `pbh`, `pbphase`, and `rd`. `ce`
accepts `neutral` or `hINDEX`. Missing values default to 1000 iterations, depth 16,
square-root-of-two exploration, uniform-random rollouts, neutral cutoff evaluation, no Progressive
Bias, and disabled root diagnostics. A bare `--agent` uses every default. Automatic names encode
the effective configuration, for example `mcts-i5000-d120`, `mcts-h0-i20000-d32`, and
`mcts-h0-t0.5-d130-pb0.25-pbh0-pbphase-collect`; a non-default exploration constant is appended as
`-cVALUE`.

`p=random` and `p=uniform` abbreviate uniform-random rollout. `p=greedy` and `p=epsilon` require
`rh=INDEX`; epsilon-greedy uses `e=0.1` when omitted. `h=INDEX` configures only cutoff evaluation,
whereas `rh=INDEX` configures only informed rollout selection, so either mechanism can be enabled
alone or both can use different game heuristics.
For SPOTF, `phase=collect` applies the selected informed policy only during collection and uses
uniform random rollout actions during gemstone placement.

`pb=WEIGHT` enables Progressive Bias, `pbh=INDEX` selects its independent evaluator, and
`pbphase=collect` restricts it to SPOTF collection decisions. The evaluator is computed once when
a child is expanded and cached; its contribution then decays as that child accumulates visits.
`rd=true` records the root actions' visits, mean utility, cached heuristic value, final bias term,
and selected flag. It is disabled by default because detailed root records increase JSONL size.

Inline agents and `--agent-config` profiles can be mixed in one comparison. Explicit and generated
names must remain unique. Profiles are preferable when a configuration must be preserved as a
reproducible experimental artifact.

The shared positions are the initial state and, when reachable, states near one third and two
thirds of the sampled median game depth. This gives a small early/middle/late latency check without
making `analyze` run a long match suite. The output ranks every supplied profile from fastest to
slowest; comparison is not limited to two agents.

Each configured benchmark reports:

- exact agent configuration and sampled position count;
- mean, p50, p95, and maximum isolated decision latency;
- observed milliseconds per completed iteration;
- latency, actual iterations, and created nodes at each sampled ply;
- time relative to the fastest supplied profile;
- ratio to `--target-time` and the profile whose mean is closest to that target.

With only three positions, p95 is effectively the slowest sampled position; it is an orientation
metric, not a production latency guarantee. MCTS cost is not perfectly linear in iteration count
because larger searches build larger trees. Do not treat `milliseconds_per_iteration` as an exact
extrapolation far away from the measured configuration.

These are isolated single-decision measurements. A tournament with several workers can report
higher per-decision latency because multiple single-threaded MCTS searches share the machine. Run
benchmarks on an otherwise idle system for stable isolated comparisons, or under intentional load
when that load represents deployment.

A `time_budget` is checked only between complete MCTS iterations. The reported decision may exceed
the requested time by one iteration, and at least one iteration always runs. Seeded time-budget
searches are not exactly reproducible because system load changes the completed iteration count.

## Compatibility fields

`recommended_rollout_depth`, `recommended_iterations`, `iterations_capped`,
`milliseconds_per_iteration`, and `estimated_decision_time_ms` remain available. They are aliases
for the Balanced experiment and its calibrated depth, not structural tree-size recommendations.
New integrations should prefer `rollout_costs` and `suggested_experiments`.

The tree-size estimate is still an order-of-magnitude structural description. It no longer drives
the practical iteration suggestions.

## Rust API

`evaluate_game<G>` accepts an `EvaluationConfig` and returns `GameEvaluationReport`.
`benchmark_mcts_agent<G, A>` measures an already configured agent on reproducibly sampled states.
Both support deterministic, perfect-information, two-player, zero-sum games with cloneable states
and actions. No game-specific phase API is required: player-turn blocks come from `Game::status`.
