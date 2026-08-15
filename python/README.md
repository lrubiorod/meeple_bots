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

An MCTS heuristic is selected by a zero-based index owned by the game. Boop currently accepts
`heuristic=0`, which scores each cat as `0.1` for the root player and `-0.1` for the opponent.
Kittens and pieces in either pool do not affect the score. Tic-tac-toe and Connect Four currently
reject every heuristic index.

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
| `--first-mcts-heuristic [INDEX]` | disabled | Player 0 heuristic; omitting `INDEX` selects `0`. |
| `--second-mcts-heuristic [INDEX]` | disabled | Player 1 heuristic; omitting `INDEX` selects `0`. |
| `--json` | disabled | Emit machine-readable output. |

Examples:

```bash
meeple-bots match --game connect-four --first human --second mcts --seed 42
meeple-bots match --game boop --first random --second random --seed 9 --json
meeple-bots match --game boop --first human --second mcts \
  --mcts-iterations 50000 --mcts-rollout-depth 64
meeple-bots match --game boop --first mcts --second mcts \
  --first-mcts-heuristic --second-mcts-heuristic 0
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

Use `meeple-bots match --help` or `meeple-bots analyze --help` for the options installed in the
current environment.
