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
).run(lambda event: print(event.status, event.match_number))

print(result.agent_a_wins, result.agent_b_wins, result.draws)
```

Batch winners are participant-oriented: `0` means agent A, `1` means agent B, and `None` means a
draw. Match seeds increase sequentially from the batch seed. Durations are observational; results
remain reproducible for a fixed configuration and seed.

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
| `batch` | Compare two automated participants. | Progress and aggregate result |
| `analyze` | Sample game structure and calibrate MCTS. | Evaluation report |
| `tournament` | Run configured round-robin pairings. | JSONL trace |
| `extract` | Convert a tournament trace into tables. | Manifest and CSV files |
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
face-up forest from the match seed, and exposes cutoff heuristic `0`.

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
  --second-mcts-heuristic 1 --seed 42
```

Omitting the value after `--first-mcts-heuristic` or `--second-mcts-heuristic` selects index `0`.
Unsupported indices and heuristics on unsupported games are rejected.

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
  --agent-b-config configs/mcts/template.toml --seed 42
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

### Analyze a game

```bash
meeple-bots analyze --game boop --samples 128 --max-depth 256 --seed 42
meeple-bots analyze --game boop --target-time 5 --seed 42 --json
```

This command is the CLI equivalent of `evaluate_game`. `--target-time` chooses the approximate
seconds per decision used for the suggested benchmark points; it does not lengthen calibration to
that duration. Its measurements, fields, and interpretation are kept in the
[evaluation guide](../crates/evaluation/README.md).

## Run a study

Use a tournament when more than two configurations must be compared or when every action trace
must be preserved for later analysis.

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

```toml
game = "boop"
output = "../../results/tournaments/boop-study.jsonl"
matches_per_pair = 20
seed = 42
max_plies = 10000

[[agents]]
name = "random"
kind = "random"

[[agents]]
name = "mcts-h0-10000"
kind = "mcts"
iterations = 10000
rollout_depth = 16
exploration = 1.4142135623730951
use_heuristic = true
heuristic_index = 0
self_play = true
```

The tournament schedules every distinct pair of agents and alternates their seats. `self_play =
true` adds one same-configuration pairing without including those games in competitive standings.
Different names may intentionally use identical parameters.

An MCTS entry can use arrays for `iterations`, `rollout_depth`, `exploration`, and
`heuristic_index`. The loader creates the Cartesian product, so this compact entry produces nine
agents:

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

Generated names append only the fields written as arrays. The suffixes are `i` for iterations, `d`
for rollout depth, `c` for exploration, and `h` for heuristic index. The example therefore creates
names from `mcts-h0-i100-d8` through `mcts-h0-i10000-d32`. Scalar entries retain their original
names, so existing tournament files remain compatible.

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
```

The default output is `results/tournaments/boop-study/data/`. Use `--output-dir PATH` to override
it. Existing known outputs are protected unless `--overwrite` is supplied.

Extraction is currently registered only for Boop. It starts with three generic tournament tables:

- `manifest.json`: schemas, completeness, table names, and row counts;
- `agents.csv`: one row per configured agent;
- `matches.csv`: game-independent outcomes, seats, durations, and utilities.

Boop additionally produces:

- `boop_matches.csv`: first graduation and winning mechanism per match;
- `turns.csv`: placements, phases, zones, resolutions, boops, and state metrics;
- `boops.csv`: one row per adjacent-piece interaction;
- `resolutions.csv`: one row per graduation or eight-piece recovery;
- `winning_lines.csv`: final cat-line positions and orientations.

Extraction streams the trace and accepts interrupted studies. It marks the manifest as partial when
fewer matches than declared are available. A truncated final JSONL line is ignored and reported;
malformed records elsewhere are rejected.

Tournaments can still record SPOTF, Connect Four, and tic-tac-toe traces, but `extract` rejects
them until their analyzers are implemented.

### 3. Generate a Boop report

Install the optional plotting and statistics dependencies once:

```bash
python -m pip install -e ".[report]"
```

Then generate the report:

```bash
meeple-bots report \
  --input results/tournaments/boop-study/data
```

An input directory named `data` produces a sibling `report` directory. Other input names receive a
nested `report/` by default. Use `--output-dir PATH` or `--overwrite` to change that behavior.

Boop is currently the only registered report generator. It accepts a partial extraction but labels
the result as preliminary.

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

Competitive report results exclude self-play. Strategic summaries retain it, normalize board
zones by cell count, and aggregate turn-level behavior by match so long games do not dominate.

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
