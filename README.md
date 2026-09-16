# Meeple Bots

Meeple Bots is a modular framework for playing, simulating, and studying board games with human,
random, and Monte Carlo Tree Search players. Rust owns the game rules and performance-critical
simulation loop; Python provides the public API, command-line tools, browser interfaces, and study
workflow.

## What is included

| Area | Available features |
| --- | --- |
| Games | [Connect6](games/connect6/README.md), Tic-tac-toe, Connect Four, boop., two-player Spirits of the Forest, [public stochastic Splendor](games/splendor/README.md), and [imperfect-information Lost Cities](games/lost-cities/README.md). |
| Agents | Interactive human input, uniform random play, configurable MCTS, and [SO-ISMCTS for Lost Cities](agents/so-ismcts/README.md). |
| Interfaces | Typed Python API, command-line commands, and local browser playrooms. |
| Experiments | Reproducible batches, round-robin tournaments, trace extraction, and generic, Boop and SPOTF reports. |
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
python -m meeple_bots gui --game spotf
python -m meeple_bots gui --game cant-stop
python -m meeple_bots gui --game splendor
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

The editable installation exposes Python changes immediately. After changing any Rust crate used by Python (including games and agents),
rebuild the native extension with `maturin develop --release`.

## Choose a workflow

| Task | Entry point | Guide |
| --- | --- | --- |
| Play or watch a game | `meeple-bots gui` | [Python interface](python/usage.md#play-or-watch) |
| Run a match | `meeple-bots match` or `Match` | [Python API](python/usage.md#run-one-match) |
| Compare agents | `batch` or `Batch` | [Python API](python/usage.md#compare-two-agents) |
| Automatic MCTS diagnosis | `meeple-bots study` | [Budgeted diagnosis](python/studies.md#automatic-mcts-diagnosis) |
| Configured study | `meeple-bots tournament` | [Study workflow](python/studies.md#run-a-study) |
| Build study artifacts | `extract`, then `report` | [Artifacts](python/studies.md#study-artifacts) |
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
configs/      Shared MCTS baselines and templates; personal configurations stay local
crates/       Shared Rust contracts, simulation, catalog, evaluation, and bindings
games/        Authoritative Rust rules for each game
python/       Public Python package, CLI, GUI, extraction, reporting, and tests
```

Personal scripts and experiments belong under `local/`; tournament definitions under
`configs/tournaments/`; generated traces, tables, and reports under `results/`.
Personal configurations and generated data are ignored by Git. Shared references
remain versioned: `configs/mcts/template.toml`, `boop-baseline.toml`,
`spotf-baseline.toml`, `cant-stop-baseline.toml`, `tic-tac-toe-baseline.toml`, `connect-four-baseline.toml`, and
`configs/tournaments/template-study.toml`.
Game rules, agents, reusable tournament/extraction/report tools, and their regression
tests remain versioned. See [the Python guide](python/studies.md#1-configure-and-run-the-tournament)
to create your own local study; no personal experiment files are required.

## Development

Run Rust validation from the repository root:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
```

Rebuild the native extension after changing any Rust crate used by Python, then run the
Python suite from the activated virtual environment:

```bash
maturin develop --release --locked --extras report
python -m unittest discover -s python/tests -v
```

[CI](.github/workflows/ci.yml) runs these checks on pushes and pull requests, using Ubuntu 24.04,
Rust 1.97.1 and Python 3.12. Node 24 is installed explicitly for GUI JavaScript tests.
It installs report dependencies so analysis tests run too.
Superseded runs on the same branch are cancelled; no studies or generated results are uploaded.

Use a release build for MCTS experiments. An unoptimized native module can make the same search
dramatically slower and invalidate timing comparisons.

[Can't Stop](games/cant-stop/README.md) is the first public-chance game, available through its
GUI and Python session API. Its dice events are recorded separately from player decisions.
