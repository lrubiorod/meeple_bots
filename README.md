# Meeple Bots

Meeple Bots is a modular framework for playing, simulating, and studying board games with human,
random, and Monte Carlo Tree Search players. Rust owns the game rules and performance-critical
simulation loop; Python provides the public API, command-line tools, browser interfaces, and study
workflow.

## What is included

| Area | Available features |
| --- | --- |
| Games | Tic-tac-toe, standard 6x7 Connect Four, and two-player boop. |
| Agents | Interactive human input, uniform random play, and configurable MCTS. |
| Interfaces | Typed Python API, command-line commands, and local browser playrooms. |
| Experiments | Reproducible batches, round-robin tournaments, trace extraction, and Boop reports. |
| Analysis | Structural game sampling and local MCTS cost estimation. |

Matches use seeded, independent random streams and return complete move histories, utilities, final
boards, and game-specific state such as Boop piece pools.

## Quick start

Python 3.11 or newer and a Rust toolchain compatible with the 2024 edition are required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "maturin>=1.9.4,<2.0"
python -m pip install --no-build-isolation -e .
```

Play in a local browser:

```bash
python -m meeple_bots gui
python -m meeple_bots gui --game connect-four
python -m meeple_bots gui --game boop
```

Run one terminal match:

```bash
python -m meeple_bots match --first mcts --second random --seed 42
```

Run the same kind of match through Python:

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

The editable installation exposes Python changes immediately. After changing Rust bindings,
rebuild the native extension with `maturin develop --release`.

## Choose a workflow

| Task | Entry point | Guide |
| --- | --- | --- |
| Play or watch a game | `meeple-bots gui` | [Python interface](python/README.md#play-or-watch) |
| Run a match | `meeple-bots match` or `Match` | [Python API](python/README.md#run-one-match) |
| Compare agents | `batch` or `Batch` | [Python API](python/README.md#compare-two-agents) |
| Configured study | `meeple-bots tournament` | [Study workflow](python/README.md#run-a-study) |
| Build study artifacts | `extract`, then `report` | [Artifacts](python/README.md#study-artifacts) |
| Estimate search cost | `analyze` or `evaluate_game` | [Evaluation](crates/evaluation/README.md) |

Use `meeple-bots COMMAND --help` for the options installed in the active environment.

## Documentation

Start with the guide that matches the question:

- [Games](games/README.md): rulesets, identifiers, actions, and input conventions.
- [Agents](agents/README.md): Random, MCTS, heuristics, profiles, and search budgets.
- [Python interface](python/README.md): installation, public API, CLI, GUI, and studies.
- [Rust architecture](crates/README.md): workspace layers, generic contracts, and dispatch.
- [Game evaluation](crates/evaluation/README.md): structural metrics, timing, and limitations.
- [MCTS roadmap](agents/MCTS_ROADMAP.md) ([Español](agents/MCTS_ROADMAP.es.md)): possible
  stages for evolving the search agents.

## Repository map

```text
agents/       Rust agent implementations and MCTS roadmap
configs/      Reusable MCTS profiles and tournament definitions
crates/       Shared Rust contracts, simulation, catalog, evaluation, and bindings
games/        Authoritative Rust rules for each game
python/       Public Python package, CLI, GUI, extraction, reporting, and tests
```

## Development

Run Rust validation from the repository root:

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Run Python tests from the activated virtual environment:

```bash
python -m unittest discover -s python/tests -v
```

Use a release build for MCTS experiments. An unoptimized native module can make the same search
dramatically slower and invalidate timing comparisons.
