# Python interface

[Back to the project overview](../README.md)

The public `meeple_bots` package provides typed games, agents, matches, results, and evaluation
reports. Rust remains responsible for authoritative rules and automated simulation; Python provides
configuration, interactive input, browser interfaces, study orchestration, and reporting.

## Install for development

Python 3.11 or newer and a Rust toolchain compatible with the 2024 edition are required. Create the
virtual environment inside the repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "maturin>=1.9.4,<2.0"
python -m pip install --no-build-isolation -e .
```

The editable installation exposes Python source changes immediately. Rebuild the native extension
after changing Rust bindings:

```bash
maturin develop --release
```

Use the release profile for normal games and experiments. A plain `maturin develop` is useful when
debugging bindings, but its unoptimized MCTS timings are not representative.

Activate the environment with `source .venv/bin/activate` in each new terminal and leave it with
`deactivate`.

## Choose an interface

| Need | Recommended interface |
| --- | --- |
| Integrate a game into Python code | `Match`, `Batch`, and `evaluate_game` |
| Play or watch locally | `meeple-bots gui` |
| Run one visible terminal match | `meeple-bots match` |
| Compare two configurations repeatedly | `meeple-bots batch` |
| Run a reproducible round-robin study | `tournament`, `extract`, and `report` |

The installed command and module entry points are equivalent:

```bash
meeple-bots match --first mcts --second random --seed 42
python -m meeple_bots match --first mcts --second random --seed 42
```

## Python API

### Run one match

`Match` accepts one game, two agents, a seed, and a safety limit for the number of plies:

```python
from meeple_bots import Match, MctsAgent, RandomAgent, TicTacToe

result = Match(
    game=TicTacToe(),
    first=MctsAgent(iterations=1_000, rollout_depth=256),
    second=RandomAgent(),
    seed=42,
).run()

print(result.winner)
print(result.utilities)
print(result.final_board)
for move in result.moves:
    print(move.player, move.action)
```

`winner` is player `0`, player `1`, or `None` for a draw. Rows, columns, and player identifiers are
zero-based. The same game, agents, and seed reproduce the same automated decisions.

The result contains the complete action history and final board. Boop results also contain both
piece pools. Available games and action types are listed in the [games guide](../games/README.md).

MCTS configuration and heuristic indices are documented in the
[agents guide](../agents/README.md#monte-carlo-tree-search).

### Human-controlled matches

`HumanAgent()` uses the built-in terminal prompt:

```python
from meeple_bots import HumanAgent, Match, MctsAgent

result = Match(first=HumanAgent(), second=MctsAgent(), seed=42).run()
```

Pass a selector to connect another input source:

```python
from meeple_bots import HumanAgent, Match, RandomAgent


def choose_move(turn):
    print(turn.board)
    for index, action in enumerate(turn.legal_actions):
        print(index, action)
    return turn.legal_actions[0]


result = Match(first=HumanAgent(choose_move), second=RandomAgent()).run()
```

The selector receives a read-only `HumanTurn` containing the game, active player, board, legal
actions, and Boop pools when applicable. It must return one of the supplied legal actions; a wrong
type or illegal action stops the match with an error.

`HumanAgent` can also receive an `observe_action` callback for the state immediately after its own
accepted action. `Match` has an `observe_move` callback for every accepted move, including decision
time. These callbacks support custom interfaces without duplicating game rules in Python.

### Compare two agents

`Batch` runs automated participants repeatedly and alternates their seats by default:

```python
from meeple_bots import Batch, MctsAgent, RandomAgent, TicTacToe

result = Batch(
    game=TicTacToe(),
    agent_a=RandomAgent(),
    agent_b=MctsAgent(iterations=100, rollout_depth=9),
    matches=20,
    seed=42,
    workers="auto",
).run(lambda event: print(event.status, event.match_number))

print(result.agent_a_wins, result.agent_b_wins, result.draws)
```

Batch winners are participant-oriented: `0` means agent A, `1` means agent B, and `None` means a
draw. Match seeds increase sequentially from the batch seed. Durations are observational; results
remain reproducible for a fixed configuration and seed.

`Batch` runs independent matches with a `ThreadPoolExecutor`; each MCTS remains single-threaded.
On Linux, `workers="auto"` counts the physical CPU cores visible to the process and leaves one
free, with a minimum of one worker. For example, an 8-core/16-thread CPU uses 7 workers. Other
platforms fall back to the available CPU count when physical topology is unavailable. An explicit
positive integer selects a fixed limit, and the effective count never exceeds the number of
matches:

```python
Batch(..., workers=6).run()
```

Seeds, seats, returned games, and progress completions retain their logical match order. Several
`STARTED` events may arrive before the first `COMPLETED` event because they represent submission to
the pool. User callbacks and aggregation still run on the calling thread. Concurrent
`duration_seconds` values include CPU contention and should not be compared with an idle-machine
MCTS calibration.

### Estimate game and search scale

`evaluate_game` samples random paths and calibrates a small MCTS probe locally:

```python
from meeple_bots import Boop, evaluate_game

report = evaluate_game(
    Boop(), samples=128, max_depth=256, seed=42, target_time=5
)

print(report.initial_legal_actions)
print(report.depth_p50, report.estimated_depth)
print(report.player_turn_depth_p50, report.player_turn_depth_p95)
print(report.rollout_costs)
print(report.suggested_experiments)
```

The result separates structural complexity from locally measured MCTS budgets. Suggested
experiments are starting points, not playing-strength guarantees. See the
[game evaluation guide](../crates/evaluation/README.md) before interpreting them.

## Command-line workflows

| Command | Purpose | Main output |
| --- | --- | --- |
| `gui` | Play or watch a local browser match. | Interactive web page |
| `match` | Run and display one match. | Terminal text or JSON |
| `batch` | Compare two automated participants. | Summary and optional JSONL trace |
| `analyze` | Sample game structure and calibrate MCTS. | Evaluation report |
| `tournament` | Run configured round-robin pairings. | JSONL trace |
| `extract` | Convert tournament or batch traces into tables. | Manifest and CSV files |
| `report` | Build statistics and figures from extracted tables. | HTML, JSON, PNG, and CSV |

Use `meeple-bots COMMAND --help` for the complete options and defaults installed in the active
environment.

### Play or watch

Start the local browser interface:

```bash
meeple-bots gui
meeple-bots gui --game connect-four
meeple-bots gui --game boop
meeple-bots gui --game spotf
```

Each seat can be human, Random, or MCTS. Before a match, the page configures player types, MCTS
budget, seed, and minimum display interval. Boop also exposes both cutoff heuristics and asks humans
to choose a graduation or recovery when a placement has several legal resolutions.
Spirits of the Forest presents collection and gemstone decisions as separate phases, derives its
face-up forest from the match seed, and exposes heuristics `0` with constant gemstone value, `1`
with stronger early- and mid-game gemstone conservation, and `2` with reachable category progress
plus the same gemstone conservation as `1`.

The server binds to `127.0.0.1:8765` by default. Use `--host`, `--port`, or `--no-browser` to change
startup behavior, and `Ctrl+C` to stop it. Automated native matches release Python's GIL, keeping
the page responsive during long decisions.

For a terminal match:

```bash
meeple-bots match --game connect-four --first human --second mcts --seed 42
meeple-bots match --game boop --first random --second random --seed 9 --json
meeple-bots match --game spotf --first human --second mcts \
  --second-mcts-heuristic 0 --seed 42
```

`match` defaults to tic-tac-toe with MCTS as player 0 and Random as player 1. Common options select
the game, players, seed, ply limit, manual MCTS parameters, and JSON output.

Manual MCTS options apply to every MCTS player that does not load a profile. Heuristics are selected
per seat:

```bash
meeple-bots match --game boop --first human --second mcts \
  --mcts-iterations 1000 --mcts-rollout-depth 16 \
  --mcts-rollout-policy epsilon_greedy --mcts-rollout-heuristic 1 \
  --mcts-rollout-epsilon 0.1 --seed 42
```

Omitting the value after `--first-mcts-heuristic` or `--second-mcts-heuristic` selects index `0`.
Those seat-specific options configure cutoff evaluation only. `--mcts-rollout-heuristic` selects
the independent evaluator used by `greedy` or `epsilon_greedy` rollout policies. Unsupported
combinations and indices are rejected.

Load a complete profile for one seat when the configuration should be reusable:

```bash
meeple-bots match --game boop --first mcts --second human \
  --first-mcts-config configs/mcts/heuristic.toml
```

Do not combine a player's profile with that player's heuristic flag. The profile format is
documented in [Reusable profiles](../agents/README.md#reusable-profiles).

### Compare agents from the CLI

The `batch` command runs automated matches, reports progress on standard error, and alternates
seats by default. Every MCTS participant requires a profile:

```bash
meeple-bots batch --game boop --matches 20 \
  --agent-a random --agent-b mcts \
  --agent-b-config configs/mcts/template.toml --workers auto --seed 42
```

Compare two MCTS profiles:

```bash
meeple-bots batch --game boop --matches 20 \
  --agent-a mcts --agent-a-config configs/mcts/template.toml \
  --agent-b mcts --agent-b-config configs/mcts/heuristic.toml \
  --seed 42 --json
```

Use `--no-alternate-sides` only when keeping agent A in seat 0 is intentional. Batch summaries are
participant-oriented, so normal comparisons should leave alternation enabled.

Use `--output` to preserve every action in an extract-compatible JSONL trace:

```bash
meeple-bots batch --game boop --matches 100 \
  --agent-a mcts --agent-a-config configs/mcts/template.toml \
  --agent-b mcts --agent-b-config configs/mcts/heuristic.toml \
  --workers auto --seed 42 \
  --output results/batches/boop-comparison.jsonl
```

The trace uses the tournament schema with one pairing and marks its header with
`"study_type": "batch"`. It is written incrementally in deterministic match order and can be
passed directly to `extract`. Existing files are protected unless `--overwrite` is supplied.
`--json` continues to control the summary printed to standard output and can be used together with
`--output`.

### Analyze a game

```bash
meeple-bots analyze --game boop --samples 128 --max-depth 256 --seed 42
meeple-bots analyze --game boop --target-time 5 --seed 42 --json
```

Benchmark and rank any number of exact MCTS profiles on the same sampled positions:

```bash
meeple-bots analyze --game boop --target-time 1 --seed 42 \
  --agent \
  --agent 'i=5000,d=120' \
  --agent 'name=rollout-h1,i=20000,d=32,ce=neutral,p=epsilon,rh=1,e=0.1' \
  --agent-config configs/mcts/heuristic.toml
```

Each `--agent-config` uses the [reusable scalar profile](../agents/README.md#reusable-profiles)
format. For quick tests, repeat `--agent [SPEC]` with comma-separated fields. The cutoff evaluator
uses `h=INDEX` or `ce=neutral`; informed rollout uses `p=greedy|epsilon`, `rh=INDEX`, and optionally
`e=EPSILON`. A bare `--agent` defaults to 1000 iterations, depth 16, square-root-of-two
exploration, uniform-random rollout, and neutral cutoff evaluation. Inline agents and profiles can
be mixed; tournament grids are not accepted here.

This command is the CLI equivalent of `evaluate_game`. `--target-time` chooses the approximate
seconds per decision used for the suggested benchmark points; it does not lengthen calibration to
that duration. Repeated `--agent-config` values add sequential, exact-profile latency measurements
and a fastest-to-slowest comparison; they do not measure playing strength. Its measurements,
fields, and interpretation are kept in the [evaluation guide](../crates/evaluation/README.md).

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

```bash
meeple-bots tournament \
  --config configs/tournaments/boop-study.toml
```

A configuration defines shared execution settings and enough agent entries or matrix combinations
to produce at least two uniquely named agents:

The fully commented
[`configs/tournaments/template-study.toml`](../configs/tournaments/template-study.toml) shows
Random, uniform-rollout, cutoff-heuristic, epsilon-greedy, epsilon-sweep, and Cartesian-grid
examples in one executable study.

```toml
game = "boop"
output = "../../results/tournaments/boop-study.jsonl"
matches_per_pair = 20
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

The tournament schedules every distinct pair of agents and alternates their seats. `self_play =
true` adds one same-configuration pairing without including those games in competitive standings.
Different names may intentionally use identical parameters.

Tournament matches use the same thread pool as `Batch`. Set `workers = "auto"` or a positive TOML
integer. The CLI can override the file for one run:

```bash
meeple-bots tournament --config configs/tournaments/boop-study.toml --workers 6
```

Only the main thread updates standings, reports progress, and writes JSONL. Match numbers, seeds,
seats, and trace order therefore remain deterministic even when later matches finish computation
first. The JSONL header records the effective worker count.

An MCTS entry independently configures cutoff evaluation and rollout action selection:

```toml
cutoff_evaluator = { kind = "neutral" }
rollout_policy = { kind = "epsilon_greedy", epsilon = 0.1, evaluator = { kind = "game_heuristic", index = 1 } }
```

This example uses heuristic 1 during rollout selection but not at the depth cutoff. The inverse and
combined configurations are valid too. `uniform_random`, `greedy`, and `epsilon_greedy` are the
built-in policies. Informed iterations cost more because they evaluate every legal successor, so
compare policies at equal decision time as well as equal iterations. See the executable
[`template-study.toml`](../configs/tournaments/template-study.toml) for all combinations and grid
examples.

An MCTS entry can use arrays for `iterations`, `rollout_depth`, and `exploration`. Structured
configuration also accepts arrays in `cutoff_evaluator.index`, `rollout_policy.evaluator.index`,
and `rollout_policy.epsilon`. The loader creates their Cartesian product. For example:

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
rollout_policy = { kind = "epsilon_greedy", epsilon = [0.0, 0.1, 0.25, 1.0], evaluator = { kind = "game_heuristic", index = 1 } }
```

The flat compatibility fields `heuristic_index`, `rollout_heuristic_index`, and
`rollout_epsilon` continue to accept arrays as well.

Generated names append only the fields written as arrays. The suffixes are `i` for iterations, `d`
for rollout depth, `c` for exploration, `h` for cutoff heuristic index, `rh` for rollout heuristic
index, and `e` for rollout epsilon. The
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

The supplied `boop-study.toml` defines seven configurations, 21 cross-agent pairings, two selected
self-play pairings, and 20 matches per pairing: 460 matches in total.

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
studies only when its kind, iterations, rollout depth, exploration, and heuristic are identical.
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

Extraction is registered for Boop and SPOTF. It starts with four generic study tables:

- `manifest.json`: sources, schemas, completeness, table names, and row counts;
- `studies.csv`: one row per source trace, including whether it came from a batch or tournament;
- `agents.csv`: one row per configured agent;
- `matches.csv`: game-independent outcomes, seats, durations, and utilities.

Boop additionally produces:

- `boop_matches.csv`: first graduation and winning mechanism per match;
- `turns.csv`: placements, phases, zones, resolutions, boops, and state metrics;
- `boops.csv`: one row per adjacent-piece interaction;
- `resolutions.csv`: one row per graduation or eight-piece recovery;
- `winning_lines.csv`: final cat-line positions and orientations.

SPOTF additionally produces:

- `spotf_matches.csv`: final scores, tile counts, physical turns, and gemstone totals;
- `actions.csv`: every internal ply with phase, branching, and state before and after;
- `player_turns.csv`: actions and collected tiles grouped into actual player turns;
- `tile_takes.csv`: spirit, power source, reservation, and sacrifice for every tile;
- `gemstone_actions.csv`: every place, move, or skip decision;
- `categories.csv`: final counts and scoring contribution for all 12 categories and both players.

The SPOTF extractor rebuilds the shuffled forest from each match seed. A trace is rejected if any
action is illegal or if its replayed winner or scores differ from the recorded result.

Extraction streams the trace and accepts interrupted studies. It marks the manifest as partial when
fewer matches than declared are available. A truncated final JSONL line is ignored and reported;
malformed records elsewhere are rejected.

Tournaments can still record Connect Four and tic-tac-toe traces, but `extract` rejects them until
their analyzers are implemented.

### 3. Generate a tournament report

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
patterns, gemstone decisions, and sacrifices.

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

## Reproducibility and performance

- Match seeds create independent deterministic random streams for both seats.
- Batch and tournament match seeds advance sequentially from their configured base seed.
- Batch and tournament comparisons alternate seats unless explicitly configured otherwise.
- Timing fields vary with hardware, system load, and build profile.
- Use `maturin develop --release` before performance measurements or large studies.
- Preserve agent profiles and tournament TOML alongside results so experiments can be repeated.

## Package boundary

Game-specific Python presentation lives under `meeple_bots.games`. Generic GUI and report
dispatch live under `meeple_bots.gui` and `meeple_bots.reporting`. The private
`meeple_bots._native` module is an implementation detail; applications should import public values
from `meeple_bots`.

Rust remains the single source of truth for rules under the repository's top-level `games/`
workspace. Python game packages provide presentation, reporting, and integration rather than a
second rule implementation.

## Troubleshooting

### `ensurepip` is unavailable

On Debian or Ubuntu, installing the distribution's `python3-venv` package is the usual system-wide
solution. If system `pip` supports `--python`, bootstrap the local environment without installing
Python packages globally:

```bash
python3 -m venv --without-pip .venv
python3 -m pip --python .venv/bin/python install --upgrade pip "maturin>=1.9.4,<2.0"
source .venv/bin/activate
python -m pip install --no-build-isolation -e .
```

### MCTS is unexpectedly slow

Confirm that the virtual environment contains a release build:

```bash
maturin develop --release
```

Debug native builds are intentionally unoptimized and should not be used for search comparisons.
