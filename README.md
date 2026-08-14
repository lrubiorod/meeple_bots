# Meeple Bots

Meeple Bots is a modular framework for running reproducible board-game matches between human,
random, and Monte Carlo Tree Search players. The generic simulation engine is written in Rust and
exposed through a typed Python API and command-line interface.

## Highlights

- Games: tic-tac-toe, standard 6x7 Connect Four, and standard two-player boop.
- Agents: interactive humans, uniform random play, and configurable MCTS.
- Reproducible matches with seeded, independent random streams.
- Complete move histories, utilities, winners, and authoritative final boards.
- Empirical complexity analysis, MCTS calibration, and relative-strength assessment.
- Statically dispatched Rust game and agent implementations behind a Python-friendly catalog.

## Quick start

Python 3.11 or newer and a Rust toolchain compatible with the 2024 edition are required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "maturin>=1.9.4,<2.0"
python -m pip install --no-build-isolation -e .
```

Run MCTS against a random player:

```bash
python -m meeple_bots match --first mcts --second random --seed 42
```

The equivalent Python API is:

```python
from meeple_bots import Match, MctsAgent, RandomAgent, TicTacToe

result = Match(
    game=TicTacToe(),
    first=MctsAgent(),
    second=RandomAgent(),
    seed=42,
).run()

print(result.winner)
print(result.final_board)
```

Select another game with `--game connect-four` or `--game boop`. Use `human` as either player to
enter moves interactively.

## Complexity-aware MCTS

Measure a game's branching, typical length, and approximate game-tree size while calibrating MCTS
for the current machine:

```bash
python -m meeple_bots analyze --game boop --samples 128 --max-depth 256
```

Apply one of the resulting compute levels to a match:

```bash
python -m meeple_bots match --game boop --first human --second mcts \
  --mcts-level balanced --seed 42
```

These levels represent compute budgets rather than guaranteed cross-game strength. See the
[complexity evaluation guide](crates/evaluation/README.md) for the metrics and calibration model.

An optional benchmark inspects the real search and compares one configuration against random play
and a four-times-iteration MCTS reference:

```bash
python -m meeple_bots assess --game boop --mcts-level fast --seed 42
```

## Documentation

- [Python interface](python/README.md): installation, public API, human players, and complete CLI
  reference.
- [Games](games/README.md): supported rulesets, identifiers, actions, and board conventions.
- [Agents](agents/README.md): Random, MCTS, manual parameters, and calibrated levels.
- [Rust architecture](crates/README.md): workspace structure, generic contracts, and dispatch model.
- [Complexity evaluation](crates/evaluation/README.md): metrics, recommendations, and limitations.

## Development

Run the Rust checks from the repository root:

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Run the Python tests from the activated virtual environment:

```bash
python -m unittest discover -s python/tests -v
```

After changing the Rust bindings, update the editable native extension with `maturin develop`.
