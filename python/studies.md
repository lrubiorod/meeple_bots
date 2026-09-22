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

## Incremental MCTS study

`study` improves one incumbent incrementally. With `--baseline`, the supplied
profile is the initial incumbent: its selector, evaluator, rollout, depth,
exploration, RAVE, PW, reuse and transpositions are preserved. Without a baseline,
calibration creates a neutral full-depth agent, or a cutoff agent using the
explicit `--heuristic hN`. A heuristic conflicting with a supplied baseline is an
error; the study never silently changes the evaluator.

### Optional stages

All search stages are opt-in. `--all-search` enables all supported stages:

| Flag | Stage |
| --- | --- |
| `--depth-search` | Compare heuristic cutoff depths against the current depth. |
| `--selection-search` | Compare UCT exploration candidates against the incumbent, then UCB1-Tuned against the winner. |
| `--rave-search` | Calibrate a RAVE copy before comparing it with the incumbent. |
| `--mechanism-search` | Test all four reuse/transposition combinations, each alternative against the incumbent. |
| `--pw-search` | Screen no-PW, random PW and RAVE-guided PW, then independently refine surviving policies. |
| `--widening-expansion-search` | Compare random and AMAF admission when the incumbent has PW enabled. |
| `--second-pass` | Append local exploration, RAVE k and PW k/alpha tuning on the retained champion. |

Order: depth, selection, RAVE, mechanisms, PW, optional widening admission, optional second local pass. `--all-search` includes admission but does not implicitly enable `--second-pass`.
Disabled stages carry the incumbent forward with no matches or allocation.
RAVE and PW flags enable tuning, not permission to retain those mechanisms in
an existing baseline. Unsupported stochastic backends skip these stages explicitly.

Each optional stage starts from the best candidate retained so far. A challenger
can replace the incumbent only after completing its planned comparison (minimum
four paired seeds), scoring at least 55%, with mean paired-seed score minus one
estimated standard error above 50%. Ties, unclear and incomplete comparisons
retain the incumbent as INCONCLUSIVE. This is a documented exploratory screening
rule, not a 95% significance claim. All tuners share it.

There is **no mandatory final refinement or independent confirmation**. A local
second pass is available explicitly through `--second-pass`. In a PW-only
study, both admission policies face no-PW before calibration; refined survivors
then challenge the retained screen winner. The joint k/alpha check remains part of PW calibration.
Results are exploratory; use a separate tournament for independent verification.
With no optional stages, the initial candidate is exported without competitive
matches (calibration still measures cost).

### Examples

```bash
# Initial full study; calibration generates the initial agent.
python -m meeple_bots study --game connect6 --game-param board_size=13 \
  --all-search --games-per-comparison 50 --target-match-time 60s \
  --output results/studies/connect6-13-initial

# Calibrate PW and compare it with the existing baseline, then stop.
python -m meeple_bots study --game connect6 --game-param board_size=13 \
  --baseline configs/mcts/connect6-baseline.toml --pw-search --games-per-comparison 50 \
  --output results/studies/connect6-13-incremental-pw

# Later, use the exported candidate as the next baseline.
python -m meeple_bots study --game connect6 --game-param board_size=13 \
  --baseline results/studies/connect6-13-incremental-pw/candidates/best_agent.toml \
  --selection-search --mechanism-search --games-per-comparison 50 \
  --output results/studies/connect6-13-incremental-mechanisms

# New heuristic family: calibrate a generic H1 candidate, then all supported stages.
python -m meeple_bots study --game splendor --heuristic h1 \
  --all-search --games-per-comparison 50 --output results/studies/splendor-h1-incremental
```

### Initialization, budgets and measurements

Without a supplied baseline, calibration uses a game-provided hard decision
horizon when available (Connect6 NxN: N*N). Otherwise it increases the rollout
safety depth until representative terminal reach is at least 99%, or reports
`practical_unverified` when limits prevent verification. A heuristic baseline
starts at a middle cutoff; depth search samples approximately 10/25/50/75% of
the reference horizon. Chance does not count as a player decision, and rollouts
still complete physical turns at cutoff.

With a supplied baseline, calibration measures cost and adequacy but **does not
replace its horizon**. `--tune cutoff-depth` estimates a reference hard/practical
horizon for candidate generation while preserving the original evaluator. Optional full-study depth search uses
shallower fractions of its current depth. A supplied neutral cutoff is not claimed to be proven
full-depth merely because the evaluator is neutral.

Search budgets follow these rules:

- Supplied `time_budget`: preserve seconds per decision.
- Supplied `iterations`: preserve the iteration budget for every comparison.
- `--decision-time SECONDS`: explicitly override either with equal-time search.
- No baseline: derive seconds per decision from `--target-match-time` (default
  60s) / (estimated player decisions * 1.2).

`--budget` is an optional global elapsed limit. Before a race, the coordinator
estimates its cost and reduces the challenger shortlist if necessary, preserving
the full evidence requested per comparison. `--target-match-time` does not override a
supplied baseline budget. Measured elapsed time, iterations/sec, iterations per
decision and root coverage remain diagnostics; search adequacy is distinct from
competitive confidence. LOW/VERY LOW adequacy warns without changing the request.
Larger target recommendations are suppressed when the baseline or an explicit
decision override controls compute. Study defaults to `--workers auto` (physical
cores minus one, minimum one). All match comparisons, including equal-time and
calibration pilot matches, use the configured worker limit. Actual concurrency
is bounded by available jobs in each paired round. Position probes remain
sequential. Parallel comparisons record `timing_mode="shared_cpu"`: throughput
can be affected by CPU contention. Use `--workers 1` for isolated comparisons.

### RAVE and PW calibration

RAVE starts from the current selector's exploration. It uses the current RAVE k
if present, otherwise 3000, with two coarse neighbors. Up to five adaptive rounds
check geometric gaps or expand improving boundaries, followed by local exploration
calibration. Only then does the calibrated copy face the incumbent. Stopping for
lack of improvement is not proof of a plateau.

Progressive Widening decides **how many** actions enter; expansion policy decides
**which** enter. Random and RAVE-guided admission are sibling alternatives: random
PW losing never prevents RAVE-guided PW from being evaluated.

The shared study race first compares no-PW against both policies at the incumbent's
alpha (default 0.5), using current k (default 1.5) and k multiplied by 1/3 and 8/3.
Budget screening reserves the current-k comparison for **both** policies before
adding k breadth, preserving the full planned sample (at least four paired seeds).
If even this minimum coverage is unaffordable or incomplete, the family is
inconclusive, not rejected. A policy is screened out only when its best completed
representative clearly loses under the same paired evidence rule used for promotion.
Rejecting the family requires representative evidence for every applicable policy.

Each surviving policy refines its own k and alpha, with separate tested-value
histories, then checks four joint neighbors (k times 0.5/2, alpha +/-0.125, bounded
to [0.05,1]). RAVE-guided tuning never inherits random PW's calibrated optimum.
A supported screen winner can be retained immediately; incomplete refinement
cannot displace it. Completed refinements challenge that retained winner.

AMAF applicability follows the deterministic backend capability (currently exposed
by availability of `uct_rave`). Rust's `collects_amaf()` activates statistics for
RAVE-guided admission itself, so **UCT-RAVE selection is not required**: UCT and
UCB1-Tuned can also use informed admission. If that capability is absent, guided
admission is explicitly non-applicable, not scored as a loss.

### Fixed games per comparison

Every enabled stage uses **50 games per comparison by default**, meaning 25
paired seeds with swapped seats. This is not 50 games shared among candidates:
comparing k=0.5 and k=4 against k=1.5 costs 100 games. An extra adaptive round
with two new challengers costs another 100 games. Later joint checks and the incumbent
comparison have their own full sample counts. No independent confirmation follows;
`--second-pass` optionally appends local tuning races.

- `--games-per-comparison 50`: global default for every comparison.
- `--stage-games pw=100 --stage-games selection=50`: exact per-comparison
  overrides for those stages; repeatable. Supported names: `depth`, `selection`,
  `rave`, `mechanisms`, `pw`.
- Counts must be even. The minimum effective evidence is 8 games (4 paired seeds);
  legacy requests for 4 or 6 games are raised to 8 with an explicit notice.
  Adaptive subrounds inherit their stage's count.
- Legacy `--max-pairs` remains an exact paired-seed count, **not a maximum**; stage overrides take precedence. Prefer
  the new game-count options.

Time per decision/iterations are unchanged. The study reports estimated duration
and actual elapsed/search work. With no `--budget`, it runs the fixed schedule
through the last enabled stage. With a budget, a conservative estimate (20%
headroom, configured workers bounded by available paired jobs) determines which
challengers fit. It retains full comparison evidence, never many one-game trials.
Omitted challengers and their estimated costs are recorded as `insufficient_budget`;
if none fit, that race retains its incumbent. Once a race starts, unexpected cost
can still exhaust the elapsed limit; it pauses between batches and resumes the
missing seats without promoting an incomplete result. Calibration and in-flight
batches may overshoot. Ctrl+C also preserves progress.

Budget-pruned races are frozen, not silently reintroduced on resume. A completed
study with `budget_limited=true` may have omitted challengers; start a new local
retune to explore them. Resume is for incomplete executed races.

After family screening, each surviving PW policy calibrates k, then alpha, with up to **five additional rounds per axis**.
The first additional round checks gaps around the coarse winner. Subsequent
rounds continue only for a winning challenger scoring at least 55%, with mean
paired-seed score minus one estimated standard error above 50%. This is an
exploratory trend heuristic, not confirmation or a significance guarantee.
Improving boundaries extend outward (alpha never exceeds 1); interior values
receive geometric k or arithmetic alpha neighbors. Stopping without a clear
trend is not proof of convergence. A four-neighbor joint check follows, then
each calibrated policy faces the retained screen winner with its complete fixed sample.

### Results and resume

`candidates/best_agent.toml` is the exported nominee. Generated families also keep
`best_full_depth_agent.toml` or `best_heuristic_cutoff_agent.toml` aliases. Reports
record requested stages, comparisons, discarded candidates, measured work,
fixed planned/completed games and exploratory comparison uncertainty. `baseline.toml`
records the supplied/generated starting profile; the initial depth-screen candidate
records any calibrated horizon or explicit budget override.

The exported profile is the last retained incumbent, including when interrupted:
unfinished internal RAVE/PW calibration does not replace it. The report explicitly
marks it `not_independently_confirmed`. `--reference`, `--confirmation-pairs` and
`--auxiliary-pairs` have been removed; compare profiles in a separate tournament.

Protocol **22** adds sibling PW screening and independent admission-policy calibration.
Older studies require a new output directory; their frozen phase/seed plans cannot resume under this protocol.
Older studies require a new output directory. Resume requires identical game
counts, flags and compatible code/native build; it completes exactly the missing
matches without replaying completed seats. Git commits alone do not invalidate it.
If the code or native binary changed, strict `--resume` still refuses to mix engines.
After reviewing the changes, `--resume --allow-engine-change` explicitly accepts a
mixed-engine continuation. This does **not** certify behavioral or performance
compatibility: even rebuilding identical sources can affect a time-budgeted search.
Game counts, flags, profiles, workers, seeds and protocol must still match.
The original request fingerprint stays intact. `engine_changes` records old/new
fingerprints, UTC acceptance time and exact completed match IDs per trace; the
checkpoint and reports mark `mixed_engines`. Existing matches are retained, so an
unfinished comparison or seed pair can span engines. Future resumes check the last
accepted engine and retain this warning. Restoring the original environment remains
the option for maintaining a single-engine experiment.

An optional `--budget` is cumulative: raise 2h to 2h30m to add 30 minutes, or omit
it on resume to remove the time limit. Changing game counts requires a new study.

```bash
python -m meeple_bots study --game connect6 --game-param board_size=13 \
  --baseline configs/mcts/connect6-baseline.toml --pw-search \
  --games-per-comparison 50 \
  --output results/studies/connect6-13-pw-fixed
# Add --resume to the identical command after an interruption.
```

### Local coordinate retuning

`--tune DIMENSION` requires `--agent-config PATH` (alias of `--baseline`). It is
exclusive with full-study search flags and `--second-pass`. No selection-family
switch, evaluator change, rollout adjustment or compute override is implicit.
In particular `--decision-time` is rejected: change the source profile first if
a different compute budget is intended. `--target-match-time` remains a calibration
reference and never replaces an explicit profile's iterations/time budget.

| Dimension | Only fields allowed to change |
| --- | --- |
| `exploration` | `exploration` (UCT/UCT-RAVE only) |
| `selection` | `selection_policy`; existing C and RAVE k remain fixed |
| `rave` | `rave_equivalence`; requires UCT-RAVE already selected |
| `progressive-widening` | k and alpha, PW must already be enabled |
| `progressive-widening-k` | k only |
| `progressive-widening-alpha` | alpha only |
| `widening-expansion` | random/rave admission, PW must already be enabled and AMAF available |
| `structure` | the four reuse/transposition combinations |
| `cutoff-depth` | rollout depth, evaluator unchanged |

`--tune widening-expansion` directly compares random and RAVE-guided admission
regardless of earlier study results. It freezes k, alpha, selection, exploration,
RAVE equivalence, reuse, transpositions, rollout and every other agent field.

Each generated candidate is checked against an allowlist of fields. Local output
is checked again against the original profile before saving. Inactive parameters
are serialized too, so selecting a mechanism later cannot silently reset a
previously configured k. Preservation means configuration **values**, not TOML
whitespace/comments. Calibration probes are temporary measurements; their smaller
budgets never become local race candidates.

Exploration starts with `C +/- max(0.125, C/3)` (bounded below at zero), RAVE with
approximately k/3 and 10k/3, PW k with k/3 and 8k/3, and alpha with +/-0.25
(bounded to [0.05,1]). Depth starts near half/1.5 times the incumbent and respects
the reference horizon. A first follow-up checks gaps; later rounds require a
supported improvement. Neighbor gaps are geometric for RAVE/PW k, arithmetic
for C/alpha/depth; boundary winners may extend outward. There are at most five
follow-up rounds per scalar tuner, preventing unbounded searches. Categorical
selection/admission/structure comparisons use one round. The coupled PW tuner
runs k, alpha, then one local k recheck, never enabling/disabling PW itself.

Full study and local mode call the same candidate generators in `_study_tuners.py`.
Full stages explicitly wrap them when enabling a new selector or screening the PW
family, calibrate private copies and then compare them with the retained incumbent. Local mode never
uses those enabling wrappers. Tournament execution, cost planning, paired seeds,
trace recovery and promotion are shared in `StudyRunner`.

`--second-pass` invokes the same local tuners on the current champion: C, RAVE k,
then PW k/alpha. Unsupported/inactive dimensions are explicitly skipped. It does
not rerun coarse selector searches or change evaluator/compute budget. We retain
one incumbent (no beam or new confirmation stage); an independent tournament
remains the intended verification step.

```bash
# Full supported search, optionally followed by a local pass.
python -m meeple_bots study --game connect6 --game-param board_size=13 \
  --all-search --second-pass --budget 4h --target-match-time 60s \
  --output results/studies/connect6-full-local-pass

# Retune ONLY C on an existing UCT-RAVE + PW champion.
python -m meeple_bots study --game connect6 --game-param board_size=13 \
  --agent-config configs/mcts/connect6-baseline.toml --tune exploration \
  --budget 30m --games-per-comparison 50 \
  --output results/studies/connect6-local-c

# The resulting champion can be used for the next coordinate.
python -m meeple_bots study --game connect6 --game-param board_size=13 \
  --agent-config results/studies/connect6-local-c/candidates/best_agent.toml \
  --tune rave --budget 30m --output results/studies/connect6-local-rave
```

Startup prints LOCAL RETUNE, the baseline snapshot, tunable and frozen values.
`baseline.toml`, `study.json`, standard paired traces, `summary.json`, `report.html`
and `candidates/best_agent.toml` preserve the experiment history. The local report
lists tested configurations, evidence/costs, exact changed and preserved fields,
and IMPROVED or INCONCLUSIVE. If no supported change remains, it explicitly says
“No sufficiently supported improvement found.” Search adequacy stays separate
from competitive uncertainty and from the absence of independent confirmation.

### Lost Cities and the SO-ISMCTS study profile

Study tools are registered by **agent family** in `_study_profiles.py`. The game
catalog determines compatible families; backend restrictions can further reduce
the family's tools (for example, stochastic MCTS does not support RAVE/PW).

| Family | Exploration | Selection | Tree reuse | RAVE | PW | Transpositions |
| --- | --- | --- | --- | --- | --- | --- |
| MCTS | UCT C | UCT/UCB1-Tuned (also UCT-RAVE where supported) | yes | deterministic backend | deterministic backend | yes |
| SO-ISMCTS | IS-UCT C | UCT/UCB1-Tuned | no | no | no | no |

SO-ISMCTS tree reuse is not integrated in the current public agent, so this profile
does not advertise it. MCTS retains its four-way reuse/transpositions mechanism
stage; `--tune tree-reuse` changes only reuse on an MCTS baseline.

`--selection-search` adds UCT exploration calibration and selector comparison for
SO-ISMCTS. `--all-search` resolves to these currently supported stages; it does not
invent RAVE, PW, transposition or reuse support. Without these flags, the original
bounded exploration plan remains unchanged. UCB1-Tuned baselines are now accepted;
`--tune exploration` rejects them because C is inactive. `--tune selection` works
from either selector, freezing C and the resource budget. Full selection search
from a tuned incumbent also evaluates UCT C challengers against that incumbent.

`study` selects a registered compatible search family using catalog metadata. Lost
Cities currently defaults to `so_ismcts`; `--agent so_ismcts` makes that choice
explicit. Ordinary MCTS is rejected for this imperfect-information game. Other
registered study games retain their existing MCTS plan. There is one coordinator:
profiles supply calibration/candidate policy, while the same planner, paired-match
executor, trace writer, evidence rules, checkpoints and HTML report execute studies.

```bash
.venv/bin/python -m meeple_bots study --game lost_cities \
  --budget 2h --target-match-time 60s \
  --output results/studies/lost-cities-so
```

Use a release native build for timing experiments (`maturin develop --release`).
`--workers 1` isolates compute; the default shares CPU across matches and reports
that fact. `--budget` limits cumulative elapsed study time, with checks between
batches. An in-flight match may overrun the limit.

The default SO-ISMCTS UCT profile has four bounded stages:

1. A paired Random-vs-Random structural pilot estimates total **player decisions**,
   counting Play and Draw separately. Three representative legal decision positions
   measure search throughput. This is a weak operating estimate, not a game horizon.
2. A coarse exploration race compares the original incumbent with C values from
   the shared exploration tuner (0.25, 0.5, 1.0, 1.4, 2.0, omitting duplicates).
3. At most two local refinement rounds test neighbors. Interior candidates bisect
   adjacent tested values; a boundary candidate may extend the range. Unsupported
   improvements stop further refinement; this is not proof of an optimum.
4. A frozen nominee is compared with the **original operating incumbent** on fresh
   paired seeds. This stage is skipped if tuning retained that same configuration.

By default, decision time is `target_match_time / (estimated_decisions * 1.2)`.
Every C candidate receives exactly this same configured time budget. SO-ISMCTS
checks its deadline between complete iterations, always completing at least one.
Actual elapsed time may exceed the target by one simulation. Iterations and fresh
determinizations completed per decision can vary; they are compute diagnostics,
**not** competitive hyperparameters. No hidden-world space is enumerated.

A supplied baseline preserves its explicit iteration/time operating budget, unless
full study explicitly supplies `--decision-time`. This also permits fixed-iteration
experiments. Local retuning always freezes the supplied budget.

The shared scheduler uses the same environment seed for both seats of a pair and
swaps candidates. Environment and each seat's agent have independent RNG streams;
search sampling cannot consume environment randomness. Pilot, coarse, refinement
and confirmation seeds occupy disjoint ranges. Fixed-iteration searches are seeded
and reproducible; timed searches vary with machine load.

Comparisons default to 50 games (25 paired seeds), with a minimum of 8 games. Budget
pruning reduces challenger count before evidence per retained comparison. If no
comparison fits, the report records insufficient budget. Coarse/refinement promotion
uses the existing exploratory rule (at least 55% score and one paired standard error
above parity). Confirmation requires a complete fixed sample and a conservative
95% paired interval above 50%. Inconclusive or losing confirmation retains the
original operating incumbent; an interrupted confirmation leaves an explicitly
provisional nominee. Random is only the structural pilot, never the C-selection target.

```bash
.venv/bin/python -m meeple_bots study --game lost_cities \
  --agent-config results/studies/lost-cities-so/candidates/best_agent.toml \
  --tune exploration --budget 30m --target-match-time 60s \
  --output results/studies/lost-cities-so-retune
```

Full study and local retune use the same exploration generator and comparisons.
Only C changes in local mode: iterations/time, uniform rollout, MostVisited, and
game configuration remain frozen. The target match time remains a reporting
reference for an explicit baseline budget, not a budget replacement. RAVE, widening,
reuse, other selectors, heuristics and other tuning dimensions are rejected.

`candidates/best_agent.toml` is directly usable, for example:

```toml
name = "best_agent"
agent = "so_ismcts"
exploration = 1.4142135623730951
rollout = "uniform"
root_selection = "most_visited"
time_budget = 0.2 # Example seconds per decision; actual calibration determines this.
```

Use `iterations = 1000` instead of `time_budget` for fixed-iteration operation.
To play with an exported profile:

```bash
.venv/bin/python -m meeple_bots match --game lost_cities \
  --first so_ismcts --first-agent-config results/studies/lost-cities-so/candidates/best_agent.toml \
  --second random --seed 42
```

The normal `study.json`, `summary.json`, `report.html`, candidate TOMLs and JSONL
traces identify the family, C comparisons, confirmation, and operating budget.
Calibration reports iterations/second, determinizations/second, medians per decision,
root action coverage, tree nodes and action records. One determinization is sampled
per iteration. Adequacy describes action sampling and throughput, not confidence
of winning or a percentage of possible hidden worlds covered.

Resume uses the identical command plus `--resume`. Family, baseline, game parameters,
phase plan, seeds and resource settings are frozen; changing MCTS into SO-ISMCTS is
never an engine-compatibility override. The normal explicit `--allow-engine-change`
option retains its mixed-engine provenance rules.


### Optional descriptive Random baseline

`--vs-random` defaults to false. After competitive tuning and confirmation, it
compares **only the final retained champion** against Random. Results never
participate in ranking, promotion, selector choice or confirmation. Console,
`summary.json`, and HTML label this explicitly as diagnostic only. The structural
Random-vs-Random pilot remains a separate calibration step.

The comparison uses `--games-per-comparison` (subject to the existing minimum
paired evidence), fresh environment seeds in its own namespace, both seats for
each seed, and the existing independent search RNG streams and worker pool.
Reported metrics include games, paired seeds, W/D/L, `(wins + draws/2)/games` and
the shared paired 95% interval. Search configuration and decision budget are not
changed to compensate for Random's lower cost.

This is the lowest-priority stage: it uses only remaining study budget, and is
explicitly skipped if full evidence cannot fit. An interrupted comparison resumes
only missing games from its trace. The checkpoint freezes family/profile version,
resolved tuners, stage plan and `vs_random`; changing these on resume is rejected.
A supplied agent config plus `--vs-random`, without tuning flags, calibrates and
exports that unchanged agent rather than adding competitive tuning.

```bash
# Initial SO-ISMCTS calibration, selectors, and a descriptive strength reference.
python -m meeple_bots study --game lost_cities --agent so_ismcts \
  --all-search --vs-random --budget 3h --target-match-time 40s \
  --games-per-comparison 100 --workers 7 \
  --output results/studies/lost-cities-so-first

# Local selector comparison; Random is optional and omitted here.
python -m meeple_bots study --game lost_cities --agent-config strong.toml \
  --tune selection --budget 1h --games-per-comparison 100 \
  --output results/studies/lost-cities-selection
```

With several workers, timed searches share CPU (`shared_cpu` provenance); the
configured compute conditions remain symmetric within each paired comparison.


SO-ISMCTS tournament TOML entries use `kind = "so_ismcts"`, with either
`time_budget` (seconds per decision) or `iterations`, plus `selection_policy`
(`uct` or `ucb1_tuned`) and optional `exploration`. C only affects UCT.
The loader validates game compatibility and rejects unsupported MCTS mechanisms.
Use `seat_mode = "paired"` and an even `matches_per_pair` for swapped-seat seeds.
