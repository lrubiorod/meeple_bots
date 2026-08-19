# Python interface

[Back to the project overview](../README.md)

The public `meeple_bots` package provides typed game, agent, match, result, and game-evaluation
objects. Its private PyO3 extension delegates game rules and simulation to Rust.

## Installation

Python 3.11 or newer and a Rust toolchain compatible with the 2024 edition are required. Create the
virtual environment inside the repository so the complete Python setup remains workspace-local:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "maturin>=1.9.4,<2.0"
python -m pip install --no-build-isolation -e .
```

The editable installation makes Python source changes immediately available. Rebuild the native
extension after Rust binding changes:

```bash
maturin develop
```

Activate the environment again with `source .venv/bin/activate` when opening a new terminal. Leave
it with `deactivate`.

### Systems without ensurepip

On Debian or Ubuntu, `python3 -m venv` can report that `ensurepip` is unavailable. Installing the
distribution's `python3-venv` package is the usual system-wide solution. If the system `pip`
supports `--python`, the environment can instead be bootstrapped without writing Python packages
globally:

```bash
python3 -m venv --without-pip .venv
python3 -m pip --python .venv/bin/python install --upgrade pip "maturin>=1.9.4,<2.0"
source .venv/bin/activate
python -m pip install --no-build-isolation -e .
```

## Running a match

`Match` accepts one game, two agents, a seed, and a safety limit for the number of plies:

```python
from meeple_bots import Match, MctsAgent, RandomAgent, TicTacToe

result = Match(
    game=TicTacToe(),
    first=MctsAgent(iterations=1_000, rollout_depth=256, heuristic=None),
    second=RandomAgent(),
    seed=42,
).run()

print(result.winner)
print(result.utilities)
for move in result.moves:
    print(move.player, move.action)
```

Player identifiers, rows, and columns are zero-based. `winner` is `None` for a draw. Reusing the
same seed and configuration reproduces the same match.

Available games and their action types are documented in the [games guide](../games/README.md).

An MCTS heuristic is selected by a zero-based index owned by the game. Availability and meaning
therefore vary between games, and selecting an unsupported index raises a validation error. See the
[games guide](../games/README.md) and each game's README for the available evaluators.

## Human players

`HumanAgent()` uses the built-in terminal prompt:

```python
from meeple_bots import HumanAgent, Match, MctsAgent

result = Match(first=HumanAgent(), second=MctsAgent(), seed=42).run()
```

Pass a selector function to connect another input source such as a graphical interface:

```python
from meeple_bots import HumanAgent, Match, RandomAgent


def choose_move(turn):
    print(turn.board)
    print(turn.legal_actions)
    return turn.legal_actions[0]


result = Match(first=HumanAgent(choose_move), second=RandomAgent()).run()
```

The selector receives a read-only `HumanTurn` containing the game, active player, board, legal
actions, and boop. pools when applicable. It must return one of `turn.legal_actions`; returning a
wrong type or illegal action stops the match with an error.

An optional `observe_action` callback receives a `HumanMoveObservation` with the board and pools
immediately after the selected action has been applied. The terminal CLI enables it automatically,
so interactive matches show the result of the human move before the automated opponent starts
thinking.

## Running simulation batches

`Batch` runs automated participants repeatedly, alternates their player positions by default, and
returns participant-oriented results. In `BatchMatchResult`, winner `0` means agent A, winner `1`
means agent B, and `None` means draw:

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

Seeds are assigned sequentially starting at the batch seed. Timing fields are observational; game
results remain reproducible for the same configuration and seed.

## Game evaluation

The Python API returns a compact estimate of a game's structure and MCTS compute requirements:

```python
from meeple_bots import Boop, MctsAgent, evaluate_game

report = evaluate_game(Boop(), samples=128, max_depth=256, seed=42)
print(report.initial_legal_actions)
print(report.estimated_depth)
print(report.recommended_iterations)
print(report.milliseconds_per_iteration)
print(report.estimated_decision_time_ms)

agent = MctsAgent(
    iterations=report.recommended_iterations,
    rollout_depth=report.recommended_rollout_depth,
)
```

The recommendation uses the initial action count, effective branching factor, and sampled
95th-percentile depth. It is capped at one million iterations. Timing comes from a short MCTS
calibration on the current machine, so it is approximate and can vary with load and build mode.
It indicates computational scale rather than playing strength. See the
[game evaluation guide](../crates/evaluation/README.md) for the formula and limitations.

## Command-line interface

The installed script and module entry points are equivalent while the virtual environment is
active:

```bash
meeple-bots match --first mcts --second random --seed 42
python -m meeple_bots match --first mcts --second random --seed 42
```

### match

The `match` command runs one game and prints its move history and final board.

| Option | Default | Description |
| --- | --- | --- |
| `--game` | `tic-tac-toe` | `tic-tac-toe`, `connect-four`, or `boop`. |
| `--first` | `mcts` | Player 0: `human`, `mcts`, or `random`. |
| `--second` | `random` | Player 1: `human`, `mcts`, or `random`. |
| `--seed` | `0` | Seed for reproducible random streams. |
| `--max-plies` | `10000` | Match safety limit. |
| `--mcts-iterations` | `1000` | Manual MCTS search iterations. |
| `--mcts-exploration` | `sqrt(2)` | MCTS exploration constant. |
| `--mcts-rollout-depth` | `256` | Manual rollout-depth limit. |
| `--first-mcts-config PATH` | disabled | Load player 0 MCTS parameters from a TOML profile. |
| `--second-mcts-config PATH` | disabled | Load player 1 MCTS parameters from a TOML profile. |
| `--first-mcts-heuristic [INDEX]` | disabled | Player 0 heuristic; omitting `INDEX` selects `0`. |
| `--second-mcts-heuristic [INDEX]` | disabled | Player 1 heuristic; omitting `INDEX` selects `0`. |
| `--json` | disabled | Emit machine-readable output. |

A profile supplies the complete MCTS configuration for its player. The shared manual MCTS options
apply only to MCTS players without a profile. Do not combine a player's profile with that player's
heuristic flag.

Examples:

```bash
meeple-bots match --game connect-four --first human --second mcts --seed 42
meeple-bots match --game boop --first random --second random --seed 9 --json
meeple-bots match --game boop --first human --second mcts \
  --mcts-iterations 50000 --mcts-rollout-depth 64
meeple-bots match --game boop --first mcts --second mcts \
  --first-mcts-heuristic --second-mcts-heuristic 0
meeple-bots match --game boop --first mcts --second human \
  --first-mcts-config configs/mcts/heuristic.toml
```

### analyze

The `analyze` command samples a game and prints its structural metrics, suggested MCTS iteration
count, and approximate timing on the current machine.

| Option | Default | Description |
| --- | --- | --- |
| `--game` | required | Game to analyze. |
| `--samples` | `128` | Number of sampled games. |
| `--max-depth` | `256` | Sampling depth ceiling and suggested rollout depth. |
| `--seed` | `0` | Seed for reproducible structural sampling. |
| `--json` | disabled | Emit machine-readable output. |

```bash
meeple-bots analyze --game boop --samples 128 --max-depth 256
meeple-bots analyze --game boop --samples 128 --max-depth 256 --json
```

### batch

The `batch` command runs automated matches and prints progress to standard error before and after
every game. Agent A and B swap player positions on alternating matches, so wins are summarized by
participant rather than by board position.

| Option | Default | Description |
| --- | --- | --- |
| `--game` | required | Game to simulate. |
| `--matches` | `20` | Number of matches. |
| `--agent-a` | `random` | Participant A: `random` or `mcts`. |
| `--agent-b` | `mcts` | Participant B: `random` or `mcts`. |
| `--agent-a-config` | disabled | TOML profile required when A is MCTS. |
| `--agent-b-config` | disabled | TOML profile required when B is MCTS. |
| `--seed` | `0` | Seed of the first match. |
| `--max-plies` | `10000` | Safety limit for each match. |
| `--no-alternate-sides` | disabled | Keep A as player 0 in every match. |
| `--json` | disabled | Emit the final report as JSON; progress remains on stderr. |

The `match` and `batch` commands share the same MCTS profiles. Copy
[`configs/mcts/template.toml`](../configs/mcts/template.toml) for each configuration. A profile is a
TOML text file:

```toml
name = "example-mcts"
iterations = 500
rollout_depth = 16
exploration = 1.4142135623730951
use_heuristic = false
heuristic_index = 0
```

Random against MCTS:

```bash
meeple-bots batch --game boop --matches 20 \
  --agent-a random --agent-b mcts \
  --agent-b-config configs/mcts/template.toml --seed 42
```

Two MCTS profiles against each other:

```bash
meeple-bots batch --game boop --matches 20 \
  --agent-a mcts --agent-a-config configs/mcts/template.toml \
  --agent-b mcts --agent-b-config configs/mcts/heuristic.toml \
  --seed 42 --json
```

Use `meeple-bots match --help`, `meeple-bots analyze --help`, or `meeple-bots batch --help` for the
options installed in the current environment.

### tournament

The `tournament` command loads a TOML study configuration, schedules every distinct pair of agents,
alternates their player positions, and optionally schedules selected self-play pairings. It writes
one compact JSON object per line so completed matches remain available if a long study is
interrupted:

```bash
meeple-bots tournament \
  --config configs/tournaments/boop-study.toml
```

The first JSONL record contains the complete tournament configuration and schema version. Every
remaining record contains agent roles, physical player positions, seed, duration, result, and the
full action trace. Existing output files are rejected by default; pass `--overwrite` intentionally
to replace one. `--output PATH` can override the destination configured in the TOML.

A tournament configuration defines shared execution parameters followed by at least two uniquely
named agents:

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

Relative output paths are resolved from the directory containing the tournament TOML, independently
of the process working directory. Missing parent directories are created automatically. Keeping
generated traces under `results/tournaments/` separates reproducible inputs in `configs/` from
potentially large study outputs.

`self_play = true` adds one same-configuration pairing for that agent without duplicating it in the
standings. Self-play games are recorded separately and do not count as wins or losses in the
cross-agent standings. Agents may still have identical parameters when different names are useful
for an experiment.

Run large studies with a release build of the native extension. The provided
[`boop-study.toml`](../configs/tournaments/boop-study.toml) contains Random plus 100, 1,000, and
10,000-iteration MCTS agents with and without heuristic 0. The two 10,000-iteration agents also run
self-play, producing 23 pairings and 460 matches with the default 20 matches per pairing.

### extract

The `extract` command reads the game from a tournament header, selects its Rust trace analyzer, and
writes analysis-ready CSV tables without running the agents or their MCTS searches again:

```bash
meeple-bots extract \
  --input results/tournaments/boop-study.jsonl
```

By default, this creates `results/tournaments/boop-study/data/`, grouping derived artifacts under a
directory named after the JSONL file. Use `--output-dir PATH` to choose another directory. Existing
extraction files are protected unless `--overwrite` is supplied.

Every supported game receives common tournament tables:

- `manifest.json`: source, schema, completeness, zone definitions, and row counts.
- `agents.csv`: one row per configured tournament agent.
- `matches.csv`: game-independent outcomes, sides, duration, and utilities.

The boop analyzer additionally produces:

- `boop_matches.csv`: first graduation and winning mechanism for each match.
- `turns.csv`: placements, phases, zones, resolutions, boop totals, and state metrics.
- `boops.csv`: one row per adjacent piece interaction.
- `resolutions.csv`: one row per graduation or eight-piece recovery.
- `winning_lines.csv`: positions and orientations of final cat lines.

Extraction is streaming and accepts an interrupted or still-growing study. It processes every
complete match available and marks `complete` as false in the manifest when the declared match
count has not been reached. A truncated final JSONL line is ignored and reported; malformed lines
elsewhere are rejected. Tournament schema version 1 is supported. Boop currently provides the only
game-specific analyzer; Connect Four and tic-tac-toe are recognized but report that tournament
analysis is not available until their analyzers are implemented.

### report

Install the optional statistical and plotting dependencies:

```bash
python -m pip install -e ".[report]"
```

Then generate a report from an extracted tournament directory:

```bash
meeple-bots report \
  --input results/tournaments/boop-study/data
```

An input directory named `data` produces a sibling `report` directory by default, resulting in the
grouped layout `boop-study/{data,report}/`. For any other input directory, `report/` is created
inside it. Use `--output-dir PATH` to choose another location and `--overwrite` to replace known
report artifacts. The command reads the game from `manifest.json`; Boop is currently the only
registered report generator.

The report directory contains:

- `index.html`: a navigable statistical report.
- `summary.json`: headline values for other tools.
- `figures/`: standalone PNG charts.
- `tables/`: the aggregate CSV data behind the charts.

Competitive results exclude self-play. Strategic plots retain it, normalize board zones by their
number of cells, and summarize turn-level behavior by match so long games do not dominate the
averages. A partial extraction is accepted but clearly labeled as preliminary.
