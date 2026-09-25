# Python development and troubleshooting

[Python documentation](README.md) · [Project overview](../README.md)

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
after changing any Rust crate used by Python, including games, agents, core, catalog and bindings:

```bash
maturin develop --release
```

Use the release profile for normal games and experiments. A plain `maturin develop` is useful when
debugging bindings, but its unoptimized MCTS timings are not representative.

Activate the environment with `source .venv/bin/activate` in each new terminal and leave it with
`deactivate`.

## Reproducibility and performance

- Match seeds create independent deterministic random streams for both seats.
- Batch and tournament match seeds advance sequentially from their configured base seed.
- Batch and tournament comparisons alternate seats unless explicitly configured otherwise.
- Timing fields vary with hardware, system load, and build profile.
- Use `maturin develop --release` before performance measurements or large studies.
- Preserve agent profiles and tournament TOML alongside results so experiments can be repeated.
- Traces record schema and configuration; they do not automatically identify the native build.
  Record the source revision, Rust version and build profile alongside local studies when needed.

## Package boundary

Game-specific Python presentation lives under `meeple_bots.games`. Generic GUI and report
dispatch live under `meeple_bots.gui` and `meeple_bots.reporting`. The private
`meeple_bots._native` module is an implementation detail; applications should import public values
from `meeple_bots`.

The Python facade is organized by responsibility:

- `_agent_config.py` defines agent/evaluator/rollout configuration values and their
  game-independent validation. `api.py` re-exports them and coordinates game-aware validation,
  match/batch execution and native conversion.
- `_mcts_profiles.py` parses TOML and inline MCTS profiles; `_search_profiles.py` selects
  the search family and decodes a TOML profile in one read. `tournament_config.py`
  owns tournament TOML validation and agent-grid expansion. The `cli/` package
  owns argparse, command orchestration, terminal views and the legacy analyze
  JSON compatibility projection. It delegates Study execution to `studies.py`.

Public imports remain under `meeple_bots` and `meeple_bots.api`; internal modules do not import
back from their entry points. CLI profile helper names remain available at their existing paths.
`meeple_bots.cli:main` and `build_parser` remain the command entrypoints.

Browser controllers share worker startup, cancellation, human waits, display pacing, trace
completion and stale-callback protection in `meeple_bots.gui.controller.GuiController`.
Game subclasses supply idle and seeded state, input validation, board/move presentation and
any game-specific final scores. Presentation hooks run under the shared condition lock;
`_prepare_start` runs before replacement so preparation failures preserve the current match.
Boop converts a browser action index to its legal action before passing it to the shared waiter.
HTTP adapters share atomic controller replacement in `meeple_bots.gui.application.GuiApplication`,
while each game retains its payload parsing. Cancellation remains asynchronous and does not
interrupt native search immediately.

Report adapters share extraction-table validation and competitive statistics through
`meeple_bots.reporting.common`. That module defines per-seat win/draw/loss records, score
aggregation, win-rate Wilson intervals and first-player advantage. Game modules select their
existing output columns and add domain-specific metrics and figures. Competitive results exclude
self-play; the historical overall first-player statistic includes it and considers decisive games
only. Wilson intervals describe win rate, not the score that awards half a point for draws.

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
