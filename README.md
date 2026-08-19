# Meeple Bots

Meeple Bots is a modular framework for running reproducible board-game matches between human,
random, and Monte Carlo Tree Search players. The generic simulation engine is written in Rust and
exposed through a typed Python API and command-line interface.

## Highlights

- Games: tic-tac-toe, standard 6x7 Connect Four, and standard two-player boop.
- Agents: interactive humans, uniform random play, and configurable MCTS.
- Reproducible matches with seeded, independent random streams.
- Complete move histories, utilities, winners, and authoritative final boards.
- Reproducible simulation batches with side alternation and live progress.
- Simple game-tree sampling and local MCTS cost estimation.
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

Run 20 games between Random and a reusable MCTS profile:

```bash
python -m meeple_bots batch --game boop --matches 20 \
  --agent-a random --agent-b mcts \
  --agent-b-config configs/mcts/template.toml --seed 42
```

Progress is printed before and after every game. Copy
[`configs/mcts/template.toml`](configs/mcts/template.toml) to keep different MCTS configurations
and compare them in later batches.

Run a reproducible round-robin study across several agent strengths and preserve every action trace:

```bash
python -m meeple_bots tournament \
  --config configs/tournaments/boop-study.toml
```

The provided study compares Random with 100, 1,000, and 10,000-iteration MCTS agents, both with and
without boop heuristic 0. See the [Python interface](python/README.md#tournament) for the TOML
schema, self-play options, and JSONL output.

Replay the recorded matches with the analyzer registered for their game and extract CSV tables for
later statistics and plots:

```bash
python -m meeple_bots extract \
  --input results/tournaments/boop-study.jsonl
```

See the [extraction guide](python/README.md#extract) for the generated table schemas and partial
study behavior.

Install the optional reporting dependencies and turn those tables into a reproducible Boop report:

```bash
python -m pip install -e ".[report]"
python -m meeple_bots report \
  --input results/tournaments/boop-study/data
```

The trace remains at `boop-study.jsonl`; extracted tables and report artifacts are grouped under
`boop-study/data/` and `boop-study/report/`. The report contains an HTML overview, reusable PNG
figures, aggregate CSV tables, and a machine-readable summary.

## Game evaluation

Measure a game's initial actions, typical depth, effective branching, approximate game-tree size,
and local MCTS cost:

```bash
python -m meeple_bots analyze --game boop --samples 128 --max-depth 256 --seed 42
```

The report suggests a rounded iteration count for a reasonably explored MCTS decision and
estimates its duration on the current machine. Use those values as a starting point for a manual
match configuration:

```bash
python -m meeple_bots match --game boop --first human --second mcts \
  --mcts-iterations 50000 --mcts-rollout-depth 64 --seed 42
```

The estimate describes computational scale, not playing strength. See the
[game evaluation guide](crates/evaluation/README.md) for the formula and its limitations.

## Documentation

- [Python interface](python/README.md): installation, public API, human players, and complete CLI
  reference.
- [Games](games/README.md): supported rulesets, identifiers, actions, and board conventions.
- [Agents](agents/README.md): Random, MCTS, heuristics, and manual parameters.
- [Rust architecture](crates/README.md): workspace structure, generic contracts, and dispatch model.
- [Game evaluation](crates/evaluation/README.md): structural metrics, timing, and limitations.

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
