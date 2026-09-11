# Python API and command-line usage

[Python documentation](README.md) · [Project overview](../README.md)

## Choose an interface

| Need | Recommended interface |
| --- | --- |
| Integrate a game into Python code | `Match`, `Batch`, and `evaluate_game` |
| Play or watch locally | `meeple-bots gui` |
| Run one visible terminal match | `meeple-bots match` |
| Compare two configurations repeatedly | `meeple-bots batch` |
| Run a reproducible tournament study | `tournament`, `extract`, and `report` |

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
| `tournament` | Run configured round-robin or adjacent pairings. | JSONL trace |
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
meeple-bots gui --game cant-stop
meeple-bots gui --game splendor
```

Splendor supports human, Random and MCTS seats, with 256 iterations and rollout depth 64
as the initial GUI settings. The board shows public reservations and every refill in the
history. Click a card to buy or reserve, or click supply tokens to select gems.
Complete payment, required token returns and noble selection in the contextual panel.
See [the Splendor guide](../games/splendor/README.md) for the variant and session trace format.

Can't Stop uses a separate public-chance session and GUI. See [its guide](../games/cant-stop/README.md)
for dice events, supported MCTS options and the current tournament integration boundary.

Each seat can be human, Random, or MCTS. Before a match, the page configures player types, MCTS
budget, seed, and minimum display interval. The four deterministic-game GUIs start with these MCTS
reference settings for either seat:

| Game | Iterations per decision | Rollout depth | Reference profile |
| --- | ---: | ---: | --- |
| Tic-tac-toe | 10,000 | 9 | [tic-tac-toe-baseline.toml](../configs/mcts/tic-tac-toe-baseline.toml) |
| Connect Four | 50,000 | 42 | [connect-four-baseline.toml](../configs/mcts/connect-four-baseline.toml) |
| Boop | 15,000 | 16 | [boop-baseline.toml](../configs/mcts/boop-baseline.toml) |
| Spirits of the Forest | 100,000 | 130 | [spotf-baseline.toml](../configs/mcts/spotf-baseline.toml) |

Tic-tac-toe and Connect Four use UCT with exploration `sqrt(2)`, uniform rollouts long enough to
reach terminal states, and tree reuse, without heuristics or transpositions. Boop uses exploration
`0.25`, cutoff heuristic `0`, tree reuse and transpositions. SPOTF uses exploration `1.0`, cutoff
heuristic `0` and tree reuse, with transpositions disabled. Both use uniform rollouts and expose
exploration and transpositions as editable fields. These are intended as competent interactive
opponents, not guaranteed perfect or superhuman players. Decision time depends on the machine
and position. All displayed settings remain editable; switching player types preserves edits.
The GUI reads the reference TOML files at process startup. With an editable installation, edit
`configs/mcts/<game>-baseline.toml` and restart the GUI to load the new defaults. Installed wheels
include those same files and work without a repository checkout. Resolution does not depend on
the shell's working directory. Invalid profiles or settings unsupported by the GUI produce an error
instead of silently falling back to different defaults.
The profile only initializes the controls: changing a field affects the next match without writing
to the TOML. Restarting the GUI process restores the profile's values. Both iteration and time
budgets are supported. Load the file explicitly for CLI matches.

Boop also exposes both cutoff heuristics and asks humans
to choose a graduation or recovery when a placement has several legal resolutions.
Spirits of the Forest presents collection and gemstone decisions as separate phases, derives its
face-up forest from the match seed, and exposes only heuristic `0`, with reachable category progress
and stronger early- and mid-game gemstone conservation.

Invalid start requests preserve the current match. Restarting a GUI controller cancels its
previous execution: late callbacks and results cannot update the new match or consume its human
input. Cancellation stops publication and wakes human waiters; it does not immediately interrupt
an ongoing native simulation. A trace write already in progress may finish with the original
match's metadata, without updating the new match.

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
  --mcts-time-budget 2 --mcts-rollout-depth 16 \
  --mcts-rollout-policy epsilon_greedy --mcts-rollout-heuristic 1 \
  --mcts-rollout-epsilon 0.1 --seed 42
```

Omitting the value after `--first-mcts-heuristic` or `--second-mcts-heuristic` selects index `0`.
Those seat-specific options configure cutoff evaluation only. `--mcts-rollout-heuristic` selects
the independent evaluator used by `greedy` or `epsilon_greedy` rollout policies. Unsupported
combinations and indices are rejected.
`--mcts-iterations` and `--mcts-time-budget` are mutually exclusive. The latter is an approximate
number of seconds for each call to the agent, not for a complete multi-action player turn.

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
meeple-bots analyze --game spotf --target-time 1 --seed 42 \
  --agent \
  --agent 'i=5000,d=120' \
  --agent 'name=equal-time-random,t=1,d=120,p=random' \
  --agent 'name=rollout-h0,i=20000,d=32,ce=neutral,p=epsilon,rh=0,e=0.1' \
  --agent 'name=collect-h0,i=5000,d=32,h=0,p=epsilon,rh=0,e=0.25,phase=collect' \
  --agent-config configs/mcts/heuristic.toml
```

Each `--agent-config` uses the [reusable scalar profile](../agents/README.md#reusable-profiles)
format. For quick tests, repeat `--agent [SPEC]` with comma-separated fields. The cutoff evaluator
uses `h=INDEX` or `ce=neutral`; informed rollout uses `p=greedy|epsilon`, `rh=INDEX`, and optionally
`e=EPSILON`. For SPOTF, `phase=collect` wraps that policy in a conditional rollout whose fallback
is uniform random. Use `t=SECONDS` instead of `i=ITERATIONS` for a wall-clock profile. A bare `--agent`
defaults to 1000 iterations, depth 16, square-root-of-two
exploration, uniform-random rollout, and neutral cutoff evaluation. Inline agents and profiles can
be mixed; tournament grids are not accepted here.

This command is the CLI equivalent of `evaluate_game`. `--target-time` chooses the approximate
seconds per decision used for the suggested benchmark points; it does not lengthen calibration to
that duration. Repeated `--agent-config` values add sequential, exact-profile latency measurements
and a fastest-to-slowest comparison; they do not measure playing strength. Its measurements,
fields, and interpretation are kept in the [evaluation guide](../crates/evaluation/README.md).
