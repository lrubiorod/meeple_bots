# Game evaluation

[Rust architecture](../README.md) · [Project overview](../../README.md)

The evaluation crate separates two different questions:

1. What does the sampled game tree look like?
2. How much work fits a practical decision-time budget for a compatible search family?

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
  --agent-config configs/mcts/boop-baseline.toml
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

Each sample starts at the initial state and chooses player actions uniformly at random until it
reaches a terminal position or `max_depth`. Chance events, including initial deals, use a separate
environment RNG and are resolved without consuming player depth or contributing to branching.
`chance_events_mean/p50/p95` include setup events; a complete Lost Cities round has 60 physical
card draws (16 dealt cards plus 44 later deck draws).

A ply is one call to `apply_action`. A player turn is a maximal consecutive block of plies whose
states report the same `PlayerId`. Evaluation always reads `Game::status`; it does not assume that
one action changes the player. A path containing action blocks of lengths `3, 2, 1` therefore has
six plies, three player turns, and two player changes.

The legacy same-player-block metrics remain unchanged. Separate `physical_turns_p50/p95`
use `Game::is_turn_boundary`, including another physical turn by the same player. A truncated
sample counts the partial turn it entered. Neither turn metric uses depth parity.

MCTS selection maximizes at every node controlled by the root player and
minimizes at every node controlled by the opponent. Consecutive phases owned by one player do not
alternate maximize/minimize merely because tree depth increased.

### Structural fields

| Field | Meaning |
| --- | --- |
| `initial_legal_actions` | P50 initial player branching after setup; exact when the sampled min and max agree. |
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

Fast, Balanced, and Wide retain the legacy operating points at a medium rollout depth.
Deep uses the next longer calibrated horizon at approximately the Balanced time target.
Their iteration counts describe different resource allocations. Analyze does not select a
strongest operating point or infer a strategic improvement from additional compute.
Use `study`, batch or tournaments for competitive evidence under an explicit fairness budget.

## Configured agent benchmarks

Repeat `--agent-config PATH` to measure scalar profiles from the compatible MCTS or
SO-ISMCTS family. Each runs with its exact iteration/time budget and exploration
constant on the same seeded positions. MCTS also preserves rollout depth and evaluators;
SO-ISMCTS accepts `selection_policy = "uct"` or `"ucb1_tuned"`, retaining uniform
rollout and MostVisited root selection. UCB1-Tuned ignores exploration C. Profile names must be unique. Measurements run sequentially so one compared
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
making `analyze` run a long match suite. Legacy deterministic MCTS output ranks
supplied profiles from fastest to slowest; comparison is not limited to two agents.
Family-specific configured benchmarks also accept SO-ISMCTS TOMLs for compatible
games (see below), reporting search cost without competitive matches.

Each legacy MCTS configured benchmark reports:

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

These are isolated selection measurements, without a match lifecycle or maintenance callbacks.
They are not interchangeable with tournament `decision_seconds`, which includes lifecycle cost.
A tournament with several workers can report
higher per-decision latency because multiple single-threaded MCTS searches share the machine. Run
benchmarks on an otherwise idle system for stable isolated comparisons, or under intentional load
when that load represents deployment.

For MCTS, a `time_budget` targets approximate total agent cost, deducting measured pending maintenance and
reserving the previous decision's finalization cost. Preparation counts toward the allowance.
The search checks its remaining allowance between complete iterations; at least one always runs.
Indivisible operations, estimation error and final match cleanup can exceed the requested time. Seeded time-budget
searches are not exactly reproducible because system load changes the completed iteration count.

## Compatibility fields

`recommended_rollout_depth`, `recommended_iterations`, `iterations_capped`,
`milliseconds_per_iteration`, and `estimated_decision_time_ms` remain available. They are aliases
for the Balanced experiment and its calibrated depth, not structural tree-size recommendations.
For deterministic MCTS these are superseded by `rollout_costs` and
`suggested_experiments`. New multi-family integrations should use `analyze_game`
and its independent `structural` and `search_calibration` sections.

The tree-size estimate is still an order-of-magnitude structural description. It no longer drives
the practical iteration suggestions.

## Rust API

`evaluate_game<G>` accepts an `EvaluationConfig` and returns `GameEvaluationReport`.
`benchmark_mcts_agent<G, A>` measures an already configured agent on reproducibly sampled states.
`evaluate_game` preserves the deterministic perfect-information MCTS API and now includes its
independent `structural` report. `benchmark_mcts_agent` also accepts compatible stochastic
perfect-information agents. `analyze_structure<G: Game>` needs no search capability.
`so_ismcts::benchmark` requires the imperfect-information/determinized-world contracts and
passes only an owned observation, observer ID and legal actions to SO-ISMCTS.

### Rollout horizon interpretation

`rollout_depth` in cost estimates is the configured **soft decision limit**, not a
measurement of actual rollout actions. Calibration uses the real MCTS implementation,
including completion of the current physical turn and mandatory Chance resolution.
`approximate_player_turns` remains a nominal same-player-block estimate; it does not
measure overshoot. Terminal/cutoff counts reflect the actual stopping outcome.
Structural sampling's `max_depth` remains a separate hard sampling limit.


## One analyze command, multiple search families

The Python `analyze_game` coordinator returns `AnalysisReport` with independent `structural`,
`search_calibration`, `configured_agent_benchmarks` and `properties` sections. The CLI preserves
legacy flat MCTS JSON fields and rollout-depth experiments on deterministic games. SO-ISMCTS
has no artificial rollout-depth fields. Family compatibility is shared with `study`, resolved
from native capabilities, and can be explicit via `--search-family` or `--agent so_ismcts`.
Perfect-information MCTS is rejected for hidden-information games before sampling.

```bash
.venv/bin/python -m meeple_bots analyze --game connect6 --game-param board_size=13 \
  --samples 8 --max-depth 169 --target-time 500ms
.venv/bin/python -m meeple_bots analyze --game lost_cities --samples 16 \
  --max-depth 600 --target-time 500ms
.venv/bin/python -m meeple_bots analyze --game lost_cities --search-family so_ismcts \
  --samples 16 --max-depth 600 --target-match-time 60s --json
.venv/bin/python -m meeple_bots analyze --game lost_cities --target-time 500ms \
  --agent-config results/studies/lost-cities-so/candidates/best_agent.toml
```

`--agent-config` is repeatable. Each profile retains its exact resource budget and C. Comparisons
report latency, throughput, nodes and root diagnostics, never wins or a strongest configuration.
Legacy inline MCTS profiles and latency-ranked configured MCTS output remain supported.

```python
from meeple_bots import LostCities, SoIsmctsAgent, analyze_game, analyze_structure, benchmark_search_agent
structure = analyze_structure(LostCities(), samples=16, max_depth=600, seed=42)
report = analyze_game(LostCities(), samples=16, max_depth=600, target_time=.5, seed=42)
fixed = benchmark_search_agent(LostCities(), SoIsmctsAgent(iterations=100),
                               structure['depth_p50'], seed=42, target_time=.5)
```

Initial branching has sampled mean, P50, P95, min and max, reflecting setup/private hand
variation. CLI output stays compact when min equals max. `estimated_tree_log10` is a rough
physical **decision-tree** estimate: it excludes chance branching, is not a full enumeration,
and is not an information-set tree size. Capped samples are explicitly lower bounds. Lost Cities
has no finite rules-derived decision horizon; `max_depth` is only a sampling safety cap.

### SO-ISMCTS calibration

Representative early/mid/late decision states are generated with separate environment and policy
RNG streams. The calibration adapter obtains the acting player's observation outside the agent
boundary. Search receives that observation, never the authoritative hidden state. The same
positions and seed schedule are used for configured profiles. Fixed iteration searches reproduce
counts and root visits; wall-clock measurements and adaptive iteration counts are machine-dependent.

The default calibration uses an eight-iteration probe followed by an adaptive fixed iteration
measurement (aiming at at most 100 ms/position, capped at 4096 iterations). Even a tiny target
runs at least one iteration. It reports completed iterations, determinizations, latency,
information-tree nodes, action edges, root legal/visited actions, coverage, median/P10 visits,
and mean edge availability and availability/node-visit ratio. Root diagnostics refer to actual
measured work, **not** projected root visits at a different target time. Availability aggregates
include edges of varying depth and age; they are descriptive, not quality scores.

An isolated microbenchmark samples and drops 256 determinizations per representative observation,
using its own seeded RNG. It reports mean cost and throughput without guessing an exact percentage
of search time. No complete world or private card assignment is stored in reports. Current
SO-ISMCTS samples one complete world per iteration, so iterations equal determinizations; both
labels are retained because they describe different concepts.

### Operating points and interpretation

The budget table also shows `mean game` and `p95 game`: estimated accumulated search
time if **both players** use the row's decision budget at every decision. The JSON
fields are `estimated_mean_game_search_seconds` and `estimated_p95_game_search_seconds`:

```
estimated_mean_game_search_seconds = seconds * structural.decisions_mean
estimated_p95_game_search_seconds = seconds * structural.estimated_depth
```

`estimated_depth` is the sampled p95 of player decisions. These statistics already
include both seats; do not multiply by two or substitute physical turns. No additional
games are sampled. Missing or invalid statistics produce null fields / `N/A`.
Durations display hundredths of a second, adding minutes at 60 seconds (`1m 11.68s`).
These are estimates, not wall-clock limits: they exclude engine/Python overhead,
scheduling, serialization, rendering and iteration deadline overshoot. Capped
structural samples retain the report's existing lower-bound caveat.

Target estimates multiply measured throughput by the decision budget. A short table includes
100 ms, 250 ms, 500 ms, 1 s, 2 s and the requested target. These are linear cost estimates, not
optimal iterations or a ranking of search strength. `--target-match-time` is mutually exclusive
with `--target-time` and uses the same helper as study:

```
decision_seconds = target_match_seconds / (sampled_mean_player_decisions * 1.2)
```

Adequacy reuses the study diagnostic: median iterations / representative legal actions below
1/10/100 is VERY LOW/LOW/MEDIUM, otherwise HIGH. Measured root coverage and visit counts provide
additional context. The estimated target category uses projected iteration counts; shallow targets
print a warning without silently increasing the budget. These labels are neither confidence levels
nor strength guarantees. No percentage of possible hidden worlds covered is meaningful or reported.

**Analyze** describes game structure and search cost. **Study** compares/tunes C and other genuine
strategic parameters under equal compute. Batch/tournaments supply competitive evidence.
