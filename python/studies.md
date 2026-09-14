# Study configuration and analysis reference

[Python documentation](README.md) · [Project overview](../README.md)

## Run a study

Use a tournament when more than two configurations must be compared. Both tournaments and batches
created with `--output` preserve every action for later analysis.

```text
tournament TOML
       |
       v
tournament JSONL trace
       |
       v
extracted manifest and CSV tables
       |
       v
HTML report, figures, summary, and aggregate tables
```

### 1. Configure and run the tournament

Create `configs/tournaments/boop-study.toml` locally using the TOML example below,
then run it. Personal tournament definitions are ignored by Git. The shared
[`template-study.toml`](../configs/tournaments/template-study.toml) remains versioned
as an executable reference for agent configurations and parameter grids. The Boop
and SPOTF baseline profiles are also included under `configs/mcts/`. Personal study
scripts belong under the ignored `local/` directory.

```bash
meeple-bots tournament \
  --config configs/tournaments/boop-study.toml
```

A configuration defines shared execution settings and enough agent entries or matrix combinations
to produce at least two uniquely named agents:

```toml
game = "boop"
output = "../../results/tournaments/boop-study.jsonl"
matches_per_pair = 20
pairing_mode = "round_robin"
seat_mode = "paired"
seed = 42
max_plies = 10000
workers = "auto"

[[agents]]
name = "random"
kind = "random"

[[agents]]
name = "mcts-h0-10000"
kind = "mcts"
iterations = 10000
rollout_depth = 16
exploration = 1.4142135623730951
rollout_policy = "uniform_random"
# rollout_epsilon = 0.1
use_heuristic = true
heuristic_index = 0
self_play = true
```

The default `pairing_mode = "round_robin"` schedules every distinct pair of agents. The independent
default `seat_mode = "alternating"` alternates their seats while assigning a new seed to every
match. Use `seat_mode = "paired"` to run each seed twice with opposite seats. In paired mode,
`matches_per_pair` counts total games and must be even: `20` means ten distinct seeded positions and
twenty games. Self-play keeps distinct seeds because swapping one configuration with itself would
only duplicate the same match. `self_play = true` adds that same-configuration pairing without
including its games in competitive standings. Different names may intentionally use identical
parameters.

For ordered parameter sweeps, `pairing_mode = "adjacent"` reduces only the internal pairings of
each `[[agents]]` grid. Two variants from the same entry are paired when they differ in exactly one
array dimension and use consecutive positions in that array. Variants from different entries still
use round-robin pairings. For example, a grid with three `exploration` values and two
`rollout_depth` values creates the seven edges of the resulting 3-by-2 grid instead of all fifteen
internal pairs. Array order defines adjacency; values are not numerically sorted by the loader.

Tournament matches use the same thread pool as `Batch`. Set `workers = "auto"` or a positive TOML
integer. The CLI can override the file for one run:

```bash
meeple-bots tournament --config configs/tournaments/boop-study.toml --workers 6
```

Only the main thread updates standings, reports progress, and writes JSONL. Match numbers, seeds,
seats, and trace order therefore remain deterministic even when later matches finish computation
first. The JSONL header records the effective worker count and seat mode. Extraction preserves the
mode in `studies.csv`; paired games can also be identified by their shared pairing and seed.

An MCTS entry independently configures cutoff evaluation and rollout action selection:

```toml
cutoff_evaluator = { kind = "neutral" }
rollout_policy = { kind = "epsilon_greedy", epsilon = 0.1, evaluator = { kind = "game_heuristic", index = 0 } }
```

This example uses heuristic 0 during rollout selection but not at the depth cutoff. The inverse and
combined configurations are valid too. `uniform_random`, `greedy`, `epsilon_greedy`, and `mast` are the
built-in policies. Use `rollout_policy = { kind = "mast", epsilon = 0.1 }` in TOML or
`Mast(epsilon=0.1)` from `meeple_bots` in Python. MAST learns exact action averages per
player within each decision, using terminal results or the cutoff evaluator. It accepts
no rollout evaluator and resets its statistics even when the search tree is reused.
See the [MAST behavior and experiment notes](../agents/README.md#reusable-profiles).
Greedy and epsilon-greedy iterations evaluate legal successors; MAST instead maintains
an action table. These policies have different iteration costs, so
compare policies with the same `time_budget` as well as with equal iterations. A timed decision
finishes its current iteration and can exceed its search deadline. JSONL moves record actual
completed iterations and total agent cost; this total also includes lifecycle work outside the
search budget. Further configuration and grid examples appear below.

A conditional policy can restrict informed selection to a game phase. This SPOTF example evaluates
H0 only for `TakeTile` and `EndCollection`; gemstone actions use the uniform fallback:

```toml
rollout_policy = { kind = "conditional", condition = { kind = "turn_phase", phase = "collect" }, primary = { kind = "epsilon_greedy", epsilon = 0.25, evaluator = { kind = "game_heuristic", index = 0 } }, fallback = { kind = "uniform_random" } }
```

Conditional rollout configuration is preserved in JSONL and `agents.csv`. Turn-phase conditions
are currently supported by SPOTF; other games reject them explicitly.

Progressive Bias can guide UCT while preserving uniform rollouts. This example evaluates each new
child once with H0 during Collect, caches that value, and lets its influence decay with visits:

```toml
progressive_bias = { weight = 0.25, evaluator = { kind = "game_heuristic", index = 0 }, condition = { kind = "turn_phase", phase = "collect" } }
root_diagnostics = true
```

`root_diagnostics` is opt-in because it increases trace size. When enabled, every JSONL move stores
the expanded root actions' indices, visits, mean utility, cached heuristic value, final Progressive
Bias term, and selected flag. `extract` normalizes these records into `root_actions.csv`.

Tree reuse is independently opt-in:

```toml
tree_reuse = true
```

During a match it follows accepted actions from both seats, retains only the reached subtree, and
falls back to a fresh tree when the action was not expanded or the state does not match. JSONL moves
record reuse hits, retained visits/nodes, pruning, and resets; `extract` writes these columns to
Boop `turns.csv` and SPOTF `actions.csv`. The `analyze` command benchmarks isolated positions, so use
a paired tournament to evaluate reuse latency or strength across a continuous game.

Exact-state transpositions are also independently opt-in and can be combined with tree reuse:

```toml
transpositions = true
```

When disabled, the original tree backend is used unchanged. When enabled, equal states share one
node and its accumulated utility, while each action edge keeps an independent visit count for UCT
exploration. `search_nodes` then counts unique states created by the graph search.

An MCTS entry must define exactly one of `iterations` or `time_budget`, plus `rollout_depth`.
`rollout_depth` is a nominal player-decision limit: rollout finishes the current physical
turn before evaluating. Chance events do not count, and terminal stops immediately.
For example, Connect6 may execute 16 decisions for a depth of 15 when the last stone
would otherwise leave its turn incomplete. No boundary option is needed.
Either budget can be an array; `rollout_depth` and `exploration` can also use arrays.
For deterministic UCT-RAVE, set `selection_policy = "uct_rave"` and optionally
`rave_equivalence = 1000` (positive integer, also supports grids). See
[Classic UCT-RAVE](../agents/README.md#classic-uct-rave-deterministic-games). Structured
configuration also accepts arrays in `cutoff_evaluator.index`, `rollout_policy.evaluator.index`,
`rollout_policy.epsilon`, and the corresponding `rollout_policy.primary` fields of a conditional
policy. `progressive_bias.weight` also accepts an array. The loader creates their Cartesian product.
For example:

```toml
[[agents]]
name = "mcts-h0"
kind = "mcts"
iterations = [100, 1000, 10000]
rollout_depth = [8, 16, 32]
exploration = 1.4142135623730951
use_heuristic = true
heuristic_index = 0
```

An informed-rollout epsilon sweep remains fully structured:

```toml
cutoff_evaluator = { kind = "neutral" }
rollout_policy = { kind = "epsilon_greedy", epsilon = [0.0, 0.1, 0.25, 1.0], evaluator = { kind = "game_heuristic", index = 0 } }
```

The flat compatibility fields `heuristic_index`, `rollout_heuristic_index`, and
`rollout_epsilon` continue to accept arrays as well.

Generated names append only the fields written as arrays. The suffixes are `i` for iterations, `t`
for time budget, `d`
for rollout depth, `c` for exploration, `h` for cutoff heuristic index, `rh` for rollout heuristic
index, `e` for rollout epsilon, and `pb` for Progressive Bias weight. Inline `analyze` agents also
accept `tr=true` as shorthand for `tree_reuse=true` and `tp=true` as shorthand for
`transpositions=true`. The
example therefore creates names from `mcts-h0-i100-d8` through `mcts-h0-i10000-d32`. Scalar
entries retain their original names, so existing tournament files remain compatible.

Lists must be non-empty and cannot contain duplicate values. One matrix entry can generate at most
256 agents, preventing an accidental quadratic explosion in round-robin pairings. `use_heuristic`
and `self_play` remain scalar; when `self_play = true` is present on a matrix, every generated agent
gets its own self-play pairing. The JSONL tournament header records the fully expanded agents.

Relative output paths are resolved from the directory containing the TOML, not from the current
working directory. Missing parent directories are created. Existing trace files are protected;
pass `--overwrite` only when replacement is intentional or use `--output PATH` for another target.

The JSONL header stores the complete configuration and schema version. Each following line stores
roles, physical seats, seed, duration, result, and full action trace. Because each match is flushed
immediately, completed work remains available if a long tournament is interrupted.

### Decision timing

Decision timing in new traces uses `decision_timing_scope = "agent_total_v1"`:

- `decision_seconds` is total agent wall time attributed to that action.
- `selection_seconds` covers action selection and retrieval of search statistics.
- `maintenance_seconds` covers start, action-update and end callbacks, including tree/graph reuse.
- `decision_seconds = selection_seconds + maintenance_seconds`.

Maintenance belongs to the agent executing it, even after an opponent's action. Pending work is
charged to its next decision; any remainder at match end is added to its last decision. Therefore
final per-agent means and iterations/second include lifecycle overhead. A seat with no decisions
has its lifecycle cost in `result.unassigned_maintenance_seconds`, indexed by seat.
Game-rule transitions, observers, GUI animation delays, serialization and construction outside
the match runner are excluded. Human selection time includes waiting for human input.

Live move observers receive the cost known at publication time; terminal adjustments appear in
the final `MatchResult` and the GUI's completed history. `time_budget` targets approximate total
MCTS agent cost: measured pending maintenance and a finalization estimate reduce the search
allowance, including preparation. Iteration budgets are unchanged. A complete iteration is always
allowed; indivisible operations and final match cleanup can still exceed the target. For equal-cost
comparisons check measured totals. See [budget accounting](../agents/README.md#reusable-profiles).
Older runs used the entire time allowance for the search loop; do not interpret them as results
of this revised budgeting behavior.

`moves.csv` retains all three timing fields. Existing game-specific tables and reports use the
new total through `decision_seconds`. Extracted manifests and reports identify the timing scope.
Historical traces without the marker retain selection-only timing and unknown component fields.
They remain readable, but cannot be combined or resumed with new total-cost studies. Their
missing lifecycle costs cannot be recovered by re-extraction; repeat affected timing comparisons
when total cost is required.

Local Python studies can use the same executor without importing CLI internals:

```python
from pathlib import Path
from meeple_bots import ConnectFour, MctsAgent, RandomAgent
from meeple_bots.tournaments import TournamentAgent, TournamentConfig, run_tournament

config = TournamentConfig(
    game=ConnectFour(),
    output=Path("results/tournaments/my-study.jsonl"),
    pairing_mode="round_robin", seat_mode="paired",
    matches_per_pair=2, seed=42, max_plies=42, workers=1,
    agents=(
        TournamentAgent("control", MctsAgent(iterations=100, rollout_depth=42)),
        TournamentAgent("random", RandomAgent()),
    ),
)
summary = run_tournament(config)
```

`run_tournament` returns a summary and accepts optional `on_start` and `on_match`
callbacks. Console output remains the CLI's responsibility. Both entry points use
the same scheduler, worker pool, seat/seed handling, serialization, and trace writer.

For a study with its own ordering, `match_jobs`, `run_matches`, `tournament_header`,
and `TournamentTrace` expose those same pieces. `TournamentTrace(..., resume=True)`
validates an existing trace against its header and exposes `completed_match_numbers`;
the caller must skip those jobs without changing their original identities. Duplicate
numbers, conflicting plans, truncated records and incomplete results are rejected before
opening the file for append. Results must contain consistent plies/moves, winner/utilities,
finite timings and structured actions. The same validation runs before writing new results.
The writer checks pairing identity, seats and seed against the plan, including paired seeds,
self-play and seed wraparound. Game legality remains the responsibility of native replay during
extraction; resumption does not rerun games.

New headers include ordered `pairings`, making adjacent-grid schedules verifiable. Header equality
is still required. Older round-robin headers can be used unchanged to resume their original plan;
older adjacent headers without a pairing plan are rejected rather than guessed. This lower-level
resumption is for Python study scripts; the CLI still protects existing output and
requires `--overwrite` to replace it. Regenerate analysis using `extract` and `report`.

### 2. Extract analysis tables

```bash
meeple-bots extract \
  --input results/tournaments/boop-study.jsonl

meeple-bots extract \
  --input results/tournaments/spotf-study.jsonl
```

Compatible studies can be combined into one extraction. Inputs may follow one `--input` or repeat
the option:

```bash
meeple-bots extract \
  --input results/tournaments/spotf-study-1.jsonl \
          results/tournaments/spotf-study-2.jsonl \
  --input results/tournaments/spotf-study-3.jsonl \
  --output-dir results/tournaments/spotf-combined/data
```

Batch traces can be extracted alone or mixed with tournament traces:

```bash
meeple-bots extract \
  --input results/batches/spotf-comparison.jsonl \
          results/tournaments/spotf-study.jsonl \
  --output-dir results/spotf-combined/data
```

Multiple inputs must use the same game and tournament schema. An agent name may appear in several
studies only when its kind, iteration/time budget, rollout depth, exploration, and evaluators are
identical.
Conflicting definitions stop extraction before output is written; studies or matches are never
silently excluded. Give genuinely different configurations distinct names before combining them.

Combined matches receive a new global `match_number` while retaining `study_id` and
`source_match_number`. Pairings are likewise renumbered globally. `studies.csv` records every input,
its completion status, and processed match count. `manifest.json` stores the same information in
its `sources` list. The original single-input behavior remains available without `--output-dir`;
multiple inputs require an explicit destination.

Reports over combined data give every recorded match equal weight. When source studies use
different agents or numbers of matches, use the `study_id` column for source-level comparisons and
do not interpret the aggregate win rate as though every agent had faced an identical schedule.

The default output is `results/tournaments/boop-study/data/`. Use `--output-dir PATH` to override
it. Existing known outputs are protected unless `--overwrite` is supplied.

Extraction is available for every supported game. It starts with five generic study tables:

- `manifest.json`: sources, schemas, completeness, table names, and row counts;
- `studies.csv`: one row per source trace, including whether it came from a batch or tournament;
- `agents.csv`: one row per configured agent, with the full recorded configuration in
  `config_json` alongside the usual analysis columns;
- `matches.csv`: game-independent outcomes, seats, durations, and utilities.
- `moves.csv`: one row per ply with its player, agent, outcome, action JSON, decision time,
  search iterations/nodes, root diagnostics, and tree-reuse metrics.

`study_id` is unique within each extraction. It starts with the trace filename stem and,
when already used, receives the next available numeric suffix (including collisions with
original filenames that already contain suffixes). IDs are deterministic for the same ordered
inputs; reordering or combining different inputs may change them. `studies.csv` retains the
source path, and match-level tables reference the assigned ID and original match number.

`config_json` preserves nested evaluator parameters and diagnostics options with sorted object
keys. Only `name` and `self_play` are excluded: the name has its own column and self-play is
aggregated separately across studies. When combining traces, the same agent name must have the
same recorded configuration, including cutoff, rollout, fallback and progressive-bias evaluator
parameters. Object key order and equivalent JSON numbers such as `1` and `1.0` do not cause
conflicts. Missing fields are not filled with today's defaults; use distinct names when recorded
configurations differ. Existing traces retain their original fields and remain extractable.

Analysis schema 8 adds this column. Regenerate old extractions from the original JSONL to recover
parameters that earlier CSVs omitted; there is no need to rerun matches for this correction.

Boop additionally produces:

- `boop_matches.csv`: first graduation and winning mechanism per match;
- `turns.csv`: placements, phases, timings, search iterations/nodes, resolutions, and state metrics;
- `boops.csv`: one row per adjacent-piece interaction;
- `resolutions.csv`: one row per graduation or eight-piece recovery;
- `winning_lines.csv`: final cat-line positions and orientations.

The Boop extractor replays typed actions through the native rules and requires the replayed
winner to match the recorded winner. Its domain metrics come from that replay.

SPOTF additionally produces:

- `spotf_matches.csv`: final scores, tile counts, physical turns, and gemstone totals;
- `actions.csv`: every internal ply with phase, branching, timing/search work, surrounding state,
  reachable H0 progress, category viability, gemstone state, and both ply/tile progress;
- `player_turns.csv`: actions grouped into actual player turns, including aggregate decision cost,
  score/progress deltas, category viability, and gemstone attrition;
- `tile_takes.csv`: spirit, power source, reservation, sacrifice, search cost, and strategic deltas
  for every collected tile;
- `gemstone_actions.csv`: every place, move, or skip decision with game quarter and search cost;
- `categories.csv`: final counts, count gaps, scoring contribution, absence penalties, and lost
  majorities for all 12 categories and both players.

Splendor extraction emits the common tables plus `chance_events.csv`: `match_number`,
source provenance, `event_index`, `after_ply`, and `outcome_json`. Refills remain separate
from player moves and timing statistics. The extractor shares native replay validation
with tournament resume and rejects inconsistent outcomes, scores and final states.

The SPOTF extractor rebuilds the shuffled forest from each match seed. A trace is rejected if any
action is illegal or if its replayed winner or scores differ from the recorded result.

Extraction streams the trace and accepts interrupted studies. It marks the manifest as partial when
fewer matches than declared are available. A truncated final JSONL line is ignored and reported;
malformed records elsewhere are rejected.

Connect Four and tic-tac-toe use the generic tables, with native replay validation of every
completed match. The extractor checks action types and coordinates, active players, legal
transitions and a terminal final position, then compares the recorded winner and both utilities
with Rust's result. Actions after the game ends and unfinished matches are rejected. No game
rules are duplicated in Python. Validation failure does not replace existing extraction files,
even with `--overwrite`. Historical traces undergo the same checks when re-extracted.

A partial study may contain fewer completed matches than declared; an individual match must still
be complete and valid. `complete` describes study coverage, not just whether replay succeeded.
Games with richer analyzers add domain-specific tables without changing the common extraction
contract.

### 3. Generate a tournament report

Report readers preserve identifiers as literal text, including names such as `NA`, `NULL`,
`nan` or `001`. Boop/SPOTF treat only empty CSV cells as missing values, retaining real draws
and absent numeric measurements. To repair affected reports, regenerate them from the existing
extracted CSVs; no match rerun or re-extraction is needed for this reader correction.


Install the optional plotting and statistics dependencies once:

```bash
python -m pip install -e ".[report]"
```

Then generate the report:

```bash
meeple-bots report \
  --input results/tournaments/boop-study/data

meeple-bots report \
  --input results/tournaments/spotf-study/data
```

An input directory named `data` produces a sibling `report` directory. Other input names receive a
nested `report/` by default. Use `--output-dir PATH` or `--overwrite` to change that behavior.

Boop and SPOTF have registered report generators. Both accept a partial extraction but label the
result as preliminary. The SPOTF report compares competition, head-to-head results, first-player
advantage, plies versus physical turns, scores, spirit and power-source categories, collection
patterns, gemstone decisions, and sacrifices. It also compares measured MCTS latency, iterations
and nodes per second, budget utilization, search cost by phase, reachable scoring categories, and
gemstone conservation across game quarters. Strategic quarters use collected-tile progress rather
than internal plies. Search-specific figures are omitted when the input has no MCTS decision
metrics.

Connect Four and tic-tac-toe use a shared generic report through the same command:

```bash
meeple-bots report --input results/tournaments/connect-four-study/data
```

It produces `index.html`, `summary.json`, four CSV tables, and up to two figures.
Tables cover W/D/L and competitive score `(wins + 0.5 * draws) / games`, results
by agent and seat, results by opponent, and decision metrics. Decision metrics
include mean/median time, actual mean/median/min/max completed iterations, and
iterations per second. Configured iteration/time budgets are retained alongside
measurements. Missing measurements remain missing, including for random agents;
they are not interpreted as zero iterations. Throughput uses only decisions with
both iteration counts and positive elapsed time.

Competitive summaries exclude self-play; decision metrics include it. Partial
studies are marked preliminary. Scores are descriptive, without confidence
intervals, and aggregate scores depend on the opponents faced. The generic
report requires only the common extraction tables, so no game-specific analysis
or study script is needed. Existing output is protected unless `--overwrite` is
given. Regenerating a report does not rerun matches.

## Study artifacts

The default study layout keeps source traces, derived data, and presentation separate:

```text
results/tournaments/
├── boop-study.jsonl
└── boop-study/
    ├── data/
    │   ├── manifest.json
    │   └── *.csv
    └── report/
        ├── index.html
        ├── summary.json
        ├── figures/*.png
        └── tables/*.csv
```

Competitive report results exclude self-play. Strategic summaries retain it. Boop normalizes
board zones by cell count; SPOTF reports both internal plies and physical player turns.

The JSONL trace is the durable source record. Extracted tables and reports can be regenerated from
it without rerunning MCTS.


Analysis schema 9 adds `selection_policy` to `agents.csv`. Historical MCTS configurations
without that field mean `uct`; the original configuration remains preserved in `config_json`.
UCB1-Tuned studies should change only `selection_policy`, keeping budgets, rollouts, evaluators,
reuse and transpositions equal. The `exploration` setting only affects UCT.

## Automatic MCTS diagnosis

`study` optimizes **one MCTS family at a time**. Omitting `--heuristic` selects
uninformed full-depth optimization. Specifying `--heuristic h0` (or `h1`, or a
registered integer index) fixes that cutoff evaluator for the entire study.
There is no automatic full-depth-versus-cutoff competition. Compare the exported
winners yourself using the normal tournament workflow.

```bash
# Uninformed Connect6, with the rule-proven 121-decision maximum.
python -m meeple_bots study --game connect6 --game-param board_size=11 \
  --budget 4h --target-match-time 60s \
  --output results/studies/connect6-11-full

# Optimize only Splendor H1 with cutoff; H0 would be a separate study.
python -m meeple_bots study --game splendor --heuristic h1 \
  --budget 4h --target-match-time 60s \
  --output results/studies/splendor-h1

# Resume the same protocol/configuration/build with a larger TOTAL budget.
python -m meeple_bots study --game splendor --heuristic h1 \
  --budget 6h --target-match-time 60s \
  --output results/studies/splendor-h1 --resume
```

Connect6 currently has no game heuristic: its normal study works, while requesting
an unavailable heuristic produces an explicit error. Game parameters are retained
in checkpoints, all recreated match instances, worker transport and trace headers.

### Time budget and search work

`--budget` is the total study allowance, including calibration. Independently,
`--target-match-time` (default **60s**, accepting the same s/m/h syntax) controls
the approximate computational cost of a candidate match:

```
time per decision = target match time / (estimated game decisions * 1.2)
```

The initial paired pilot estimates game length in **player decisions**; Chance
events are recorded separately and never inflate this denominator. For example,
80 estimated decisions and a 60-second target give 0.625 seconds per decision.
This is an estimate, not a match timeout. Stronger agents may play longer or
shorter matches than the pilot. An individual simulation can exceed its deadline.

All competitive screens, selector comparisons (including RAVE), the mechanism
matrix, refinement and confirmation use this **same wall-clock budget per decision**.
Iteration count is measured work, not the criterion used to choose the winner.
The operating point is the measured median iterations available at that time;
there is no compulsory equal-iteration tournament or branching-times-constant
formula selecting the final budget. Equal-iteration experiments remain available
as separate tournaments to investigate sample efficiency.

Calibration runs isolated early/middle/late probes and records actual elapsed
time, completed iterations, throughput, median iterations, legal branching and,
when supplied by native root diagnostics, action coverage and visits (median/p10).
Candidate matches also report median iterations and observed throughput, latency
and terminal simulation fraction. Positions are sampled using reproducible random
play and are not guaranteed representative of all strong-agent positions.

**Search adequacy** is explicitly a heuristic diagnostic, not a confidence
probability. Its coarse categories use median iterations divided by representative
median legal branching:

| Approximate iterations / legal action | Category |
| --- | --- |
| < 1 | VERY LOW |
| 1–<10 | LOW |
| 10–<100 | MEDIUM |
| >= 100 | HIGH |

The thresholds are descriptive, not guarantees of search quality. Root coverage
and visit distributions provide context. For LOW/VERY LOW the study warns and
shows approximate 3x/10x larger match targets using linear throughput scaling.
It **continues with the requested target**, without silently increasing time.
Candidate adequacy uses the calibration branching as a reference, so it remains
an approximation across changing positions. Competitive confidence is reported
separately using actual paired games and intervals.

### Pipeline

1. **Structural calibration and horizon.** A short paired pilot and isolated
   native probes measure decision length, branching, search costs and terminal
   reach. The optional Rust `Game::maximum_decision_horizon()` supplies a
   rule-proven bound: Connect6 uses `N*N`, Connect Four 42 and Tic-Tac-Toe 9.
   Other games may return `None`. Without a hard bound, depths grow geometrically
   from 64, using two seeded sets of representative searches, until every sampled
   position has at least 99% terminal simulations. The probe is capped by the
   calibration allocation (8% of study budget) and `--max-plies`; failure to verify
   terminal reach is labelled **practical_unverified**, never proven full-depth.
   Hard bounds are used regardless of this empirical threshold.
2. **Depth screen (heuristic mode only).** Generate approximately 10%, 25%, 50%
   and 75% of the reference horizon, round/deduplicate and restrict to
   `1 <= depth < horizon`. A medium depth is the common control. Preserve two
   promising depths after initial coverage of at least four paired seeds per
   contrast. If coverage is insufficient, preserve all depths for tuning rather
   than treating them as disproven. A one-decision horizon has no distinct cutoff.
3. **UCT exploration per surviving depth.** Compare the starting UCT against
   C=0.25, 0.5, 1, 1.4 and 2 (deduplicating the starting C). Uniform rollout,
   no reuse/transpositions, no MAST or Progressive Bias. The time operating point
   stays fixed; depth-specific match measurements expose different iteration costs.
4. **Selectors.** Compare each depth's calibrated UCT with UCB1-Tuned.
5. **RAVE.** When the native catalog advertises UCT-RAVE, test k=100/1000/3000/10000/20000,
   starting from that depth's calibrated UCT exploration, against its best selector.
   Budget-limited plans prioritize 1000 and 20000 before filling the remaining
   grid, so the high-k region is not always omitted. Local refinement can extend
   beyond the initial range: a winner at 20000 tests 10000 and 40000.
   Stochastic Splendor does not advertise RAVE and skips this sweep. No new search
   algorithm is introduced.
6. **Tuned depth selection.** Only now compare the surviving depths with their
   tuned parameters/selectors. In normal mode there is only one full-depth family.
7. **Mechanisms.** Test all four reuse/transposition combinations with the selected
   configuration fixed. A six-contrast round robin covers interactions rather than
   independently making two greedy boolean decisions.
8. **Small refinement.** Check neighboring exploration, neighboring k when RAVE
   won, and neighboring cutoff depths in heuristic mode. Full depth never changes
   here. This is a local check, not a full Cartesian search.
9. **Confirmation.** Freeze the nominee and a runner-up from the *same family*,
   then use a fresh seed namespace with swapped seats. Confirmation reports whether
   the nominee's advantage is supported; it does not reselect a winner after
   seeing held-out outcomes. A nomination can remain provisional or fail confirmation.

`rollout_depth` always counts player decisions and is a soft limit: finish the
current physical turn before evaluating, resolve mandatory Chance, and stop
immediately at terminal. Calibration/search do not change the independent game,
chance and agent RNG semantics.

### Allocation, evidence and outputs

Phase resource priorities are recalculated from remaining time; unused allocation
flows forward. Confirmation initially receives a 30% priority, with its own
`--confirmation-pairs` cap (default 32). Screening uses `--max-pairs` (default 8).
Resource-limited comparisons can be omitted before play and are explicitly labelled
`phase_budget_not_evaluated`. Within per-depth tuning the planner interleaves depth
groups so one depth cannot consume all of another's allocation. After initial
coverage, optional rounds yield to later phases. A small total budget may leave
untested parameters or no independent confirmation. Completion is not evidence of
statistical significance; large targets require a larger total budget for useful
competitive evidence.

All competitive comparisons are timing-sensitive and run sequentially to avoid
CPU contention. `--workers` remains accepted for transport compatibility; it does
not parallelize these equal-time comparisons. Paired seeds and balanced seats
are retained; distinct phases have disjoint seed namespaces. Confirmation intervals
use the existing conservative 95% paired-seed Hoeffding bound. Adaptive screening
rankings are exploratory, and no multiple-comparison correction is claimed.

The output directory contains standard per-contrast traces, frozen `study.json`,
`summary.json`, `report.html`, stage candidates, and one of:

- `candidates/best_full_depth_agent.toml`
- `candidates/best_heuristic_cutoff_agent.toml`

These profiles use `time_budget = <derived seconds>`, the selected horizon,
selector/exploration, optional `rave_equivalence`, fixed evaluator, uniform rollout
and selected mechanisms. They load through the existing agent/TOML API. The report
separates calibration/search adequacy, actual costs, discarded/untested candidates,
observed head-to-head changes and independent confirmation. A +10 percentage-point
score advantage over parity is **not** a 20% increase in intrinsic playing strength.
Generated profiles never replace shared baselines automatically.

### Migration from mixed-family studies

Protocol 11 replaces the old full-versus-neutral/H0/H1 admission pipeline. Old
checkpoints are rejected without modification: use a new output directory rather
than mixing old matches with the new methodology. Resuming protocol 11 retains
frozen plans and flushed games; `--budget` is the new total allowance. Changes to
parameters, target time, evaluator or executable code require a new directory.
A Git commit alone still does not invalidate a compatible checkpoint.

`--baseline` remains accepted as a starting profile; its exploration is the initial
C, but its cutoff evaluator does **not** implicitly select the study mode. Study
initialization resets optional mechanisms, rollout and selector for calibration.
`--reference` is allowed only within the same evaluator/horizon family, as auxiliary
held-out evidence; external references do not select the nominee. Cross-family
comparisons belong in a separate tournament. `--decision-time` remains an explicit
seconds-per-decision override and is labelled as such in the report; normally use
`--target-match-time`. `--screening-time` now errors with migration guidance because
a cheaper screening budget would change the main comparison question.
