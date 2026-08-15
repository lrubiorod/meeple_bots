# Python interface

[Back to the project overview](../README.md)

The public `meeple_bots` package provides typed game, agent, match, result, and complexity-analysis
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

An MCTS heuristic is selected by a zero-based index owned by the game. Boop currently accepts
`heuristic=0`, which is a neutral implementation returning `0.0`. Tic-tac-toe and Connect Four
currently reject every heuristic index.

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

## Complexity analysis

The Python API can estimate a game's structure and recommend fixed MCTS parameters:

```python
from meeple_bots import Boop, MctsAgent, MctsLevel, evaluate_game_complexity

report = evaluate_game_complexity(Boop(), samples=128, max_depth=256, seed=42)
recommendation = report.recommend(MctsLevel.BALANCED, time_budget_ms=750)
agent = MctsAgent.from_recommendation(recommendation)
```

See the [complexity evaluation guide](../crates/evaluation/README.md) for metric definitions,
hardware calibration, depth limits, and interpretation.

Benchmark a concrete MCTS configuration and inspect its real search behavior:

```python
from meeple_bots import Boop, MctsAgent, evaluate_mcts_strength

report = evaluate_mcts_strength(
    Boop(),
    MctsAgent(iterations=1_000, rollout_depth=64),
    matches_per_opponent=20,
    seed=42,
)

print(report.strength_estimate)
print(report.search_sufficiency)
print(report.benchmark_confidence)
print(report.cutoff_heuristic_evidence)
print(report.search.truncated_rollout_rate)
```

The function plays paired matches against random and a four-times-iteration MCTS reference. Pass
an existing `complexity_report` to avoid repeating structural sampling and hardware calibration.
It is silent by default; pass a callback to observe synchronous match progress:

```python
def show_progress(event):
    print(event.match_number, event.total_matches, event.stage, event.elapsed_seconds)


report = evaluate_mcts_strength(
    Boop(),
    MctsAgent(iterations=1_000, rollout_depth=64),
    matches_per_opponent=20,
    progress=show_progress,
)
```

Every match emits `STARTED` before computation and `COMPLETED` with its result, plies, and elapsed
seconds. An exception raised by the callback cancels the benchmark.

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
| `--first-mcts-heuristic [INDEX]` | disabled | Player 0 heuristic; omitting `INDEX` selects `0`. |
| `--second-mcts-heuristic [INDEX]` | disabled | Player 1 heuristic; omitting `INDEX` selects `0`. |
| `--mcts-level` | disabled | Calibrated `fast`, `balanced`, or `thorough` level. |
| `--mcts-time-ms` | level default | Override a calibrated level's target time. |
| `--json` | disabled | Emit machine-readable output. |

Manual iteration or rollout-depth values cannot be combined with `--mcts-level`.

Examples:

```bash
meeple-bots match --game connect-four --first human --second mcts --seed 42
meeple-bots match --game boop --first random --second random --seed 9 --json
meeple-bots match --game boop --first human --second mcts --mcts-level balanced
meeple-bots match --game boop --first mcts --second mcts \
  --first-mcts-heuristic --second-mcts-heuristic 0
```

### analyze

The `analyze` command samples a game and prints structural metrics and recommendations for the
current machine.

| Option | Default | Description |
| --- | --- | --- |
| `--game` | required | Game to analyze. |
| `--samples` | `128` | Number of sampled games. |
| `--max-depth` | `256` | Sampling and recommendation depth ceiling. |
| `--seed` | `0` | Seed for reproducible structural sampling. |
| `--json` | disabled | Emit machine-readable output. |

```bash
meeple-bots analyze --game boop --samples 128 --max-depth 256
meeple-bots analyze --game boop --samples 128 --max-depth 256 --json
```

### assess

The opt-in `assess` command benchmarks one MCTS configuration and reports search telemetry,
relative results, confidence intervals, and evidence that a heuristic may be useful.

| Option | Default | Description |
| --- | --- | --- |
| `--game` | required | Game to assess. |
| `--matches` | `20` | Even number of matches against each opponent. |
| `--reference-multiplier` | `4` | Iteration multiplier for the stronger MCTS. |
| `--max-plies` | `10000` | Safety limit for every benchmark match. |
| `--samples` | `128` | Structural samples used for calibration and tree scale. |
| `--max-depth` | `256` | Structural sampling depth ceiling. |
| `--mcts-level` | `fast` | Calibrated candidate level when manual values are omitted. |
| `--mcts-time-ms` | level default | Override the selected level's target time. |
| `--mcts-iterations` | disabled | Manually configure candidate iterations. |
| `--mcts-rollout-depth` | disabled | Manually configure the rollout limit. |
| `--mcts-exploration` | `sqrt(2)` | Candidate and reference exploration constant. |
| `--mcts-heuristic [INDEX]` | disabled | Game heuristic for both MCTS variants; defaults to `0` when present without a value. |
| `--seed` | `0` | Reproducible sampling and paired-match seed. |
| `--json` | disabled | Emit machine-readable output. |

```bash
meeple-bots assess --game boop --mcts-level fast --matches 20 --seed 42
meeple-bots assess --game boop --mcts-level fast --mcts-heuristic --matches 20
meeple-bots assess --game connect-four --mcts-iterations 1000 \
  --mcts-rollout-depth 42 --matches 20 --json
```

The command runs twice `--matches` games in total and can take several minutes for expensive
configurations. Setup and per-match progress are printed immediately to standard error, including
when `--json` sends the final report to standard output.

Use `meeple-bots match --help`, `meeple-bots analyze --help`, or `meeple-bots assess --help` for
the options installed in the current environment.
