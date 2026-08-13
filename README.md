# Meeple Bots

A modular framework for simulating board games with intelligent agents. The performance-critical
engine is written in Rust and exposed through a Python interface.

## Rust Architecture

```text
python-bindings
       |
    catalog --------> simulation
     |  |                 |
  games agents --------> core
```

- `meeple_bots_core` defines games, agents, capabilities, players, and errors.
- `meeple_bots_simulation` runs reproducible matches and batches.
- `games/*` contains isolated game implementations.
- `agents/*` contains generic policies that only know about the `core` contracts.
- `meeple_bots_catalog` turns runtime configuration into monomorphized generic calls.
- `meeple_bots_python_bindings` is the PyO3 boundary and does not participate in critical loops.

Games retain their own `State` and `Action` types. Agents implement `Agent<G>`, so a concrete
simulation does not use trait objects or erase types. `DecisionContext` exposes observations to all
agents, but only exposes the authoritative state when the game implements
`PerfectInformationGame`.

Initial support covers sequential, deterministic, two-player, zero-sum, perfect-information games.
`PositionStatus::Chance` and per-player observations reserve the boundaries needed to add
randomness and hidden information through future capability traits.

## Current Implementations

- Game: tic-tac-toe with an allocation-free legal-action iterator.
- Agents: uniform random and Monte Carlo Tree Search.
- Simulation: reproducible seeds, independent RNG streams per player, turn limits, observers, and
  sequential batches.
- Catalog: runtime game and agent selection outside the match loop.

## Python Interface

Python 3.11 or newer and a Rust toolchain compatible with the 2024 edition are required. Create a
virtual environment in the repository, activate it, and install the build tools and project inside
it:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "maturin>=1.9.4,<2.0"
python -m pip install --no-build-isolation -e .
```

The virtual environment keeps the Python packages and executables isolated from the global Python
installation. Activate it again with `source .venv/bin/activate` whenever a new terminal is opened,
and leave it with `deactivate`.

On Debian or Ubuntu, `python3 -m venv` may report that `ensurepip` is unavailable. Installing the
distribution's `python3-venv` package is the usual system-wide solution. If a workspace-only setup
is preferred and the system `pip` supports `--python`, bootstrap the environment without
`ensurepip` instead:

```bash
python3 -m venv --without-pip .venv
python3 -m pip --python .venv/bin/python install --upgrade pip "maturin>=1.9.4,<2.0"
source .venv/bin/activate
python -m pip install --no-build-isolation -e .
```

The editable installation makes Python source changes immediately available. After changing the
Rust bindings, rebuild the native extension from the activated environment with `maturin develop`.

### Python API

Run an MCTS agent against a random agent from Python:

```python
from meeple_bots import Match, MctsAgent, RandomAgent, TicTacToe

result = Match(
    game=TicTacToe(),
    first=MctsAgent(),
    second=RandomAgent(),
    seed=42,
).run()

print(result.winner)
for move in result.moves:
    print(move.player, move.action.row, move.action.column)
```

Rows, columns, and player identifiers are zero-based. A `None` winner represents a draw. Reusing a
seed with the same configuration reproduces the same match.

Use `HumanAgent` to collect moves from a Python function. Calling it without arguments uses an
interactive terminal prompt:

```python
from meeple_bots import HumanAgent, Match, MctsAgent

result = Match(first=HumanAgent(), second=MctsAgent(), seed=42).run()
print(result.winner)
```

For a graphical interface or another input source, pass a function that receives a read-only
`HumanTurn` and returns a `TicTacToeAction`:

```python
from meeple_bots import HumanAgent, Match, RandomAgent, TicTacToeAction


def choose_move(turn):
    print(turn.board)
    print(turn.legal_actions)
    return TicTacToeAction(row=0, column=0)


result = Match(first=HumanAgent(choose_move), second=RandomAgent()).run()
```

The selector must return one of `turn.legal_actions`. Invalid interactive input is requested again;
an invalid value returned by a custom selector stops the match with an error.

### Command-line interface

The `match` command runs one match and prints its complete move history. Both entry points below are
equivalent when the virtual environment is active:

```bash
python -m meeple_bots match --first mcts --second random --seed 42
meeple-bots match --first mcts --second random --seed 42
```

The command accepts these options:

| Option | Default | Description |
| --- | --- | --- |
| `--game` | `tic-tac-toe` | Game to run; currently only tic-tac-toe is available. |
| `--first` | `mcts` | Agent for player 0: `human`, `mcts`, or `random`. |
| `--second` | `random` | Agent for player 1: `human`, `mcts`, or `random`. |
| `--seed` | `0` | Seed controlling the reproducible random streams. |
| `--max-plies` | `10000` | Safety limit for the number of actions in the match. |
| `--mcts-iterations` | `1000` | Search iterations used by every selected MCTS agent. |
| `--mcts-exploration` | `sqrt(2)` | Exploration constant used by MCTS. |
| `--mcts-rollout-depth` | `256` | Maximum number of actions in each MCTS rollout. |
| `--json` | disabled | Print machine-readable JSON instead of the human-readable report. |

For example, run two random agents and request JSON output:

```bash
meeple-bots match --first random --second random --seed 9 --json
```

To play against MCTS as player 0, enter a zero-based row and column when prompted:

```bash
meeple-bots match --first human --second mcts --seed 42
```

Use `meeple-bots match --help` to display the available options from the installed version.

## Development

Requires a Rust toolchain compatible with the 2024 edition.

```bash
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

From the activated virtual environment, run the Python tests with:

```bash
python -m unittest discover -s python/tests -v
```
