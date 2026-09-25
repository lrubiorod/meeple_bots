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
  game-independent validation. `api.py` remains the public facade and keeps the
  supported evaluation/benchmark entrypoints. `game_types.py` owns lightweight
  game/action values; `matches.models` owns match and batch result types;
  `matches.execution` owns `Match` and `Batch`; and `matches.human` owns the
  human callback and terminal adapter. The facade re-exports the same class objects.
- `native_config.py` converts game-independent agent values to Rust configurations.
  `native_bridge.py` owns game-aware validation and native trace/result adaptation.
  The Lost Cities SO-ISMCTS observation adapter remains game-owned. Match/Batch
  continue to invoke native execution; their seed and seat policies are distinct
  from Tournament's paired scheduling.
- `_mcts_profiles.py` parses TOML and inline MCTS profiles; `_search_profiles.py` selects
  the search family and decodes a TOML profile in one read. `tournament_config.py`
  owns tournament TOML validation and agent-grid expansion. The `cli/` package
  owns argparse, command orchestration and terminal views. It consumes the legacy
  Analyze JSON projection from `analysis.report` and delegates Study execution to `studies/`.

Study's stable `meeple_bots.studies` facade exposes `StudyRunner` and `run_study`.
Within `studies/`, `coordinator.py` owns mutable state and effect ordering;
`planning.py` builds frozen stages using `profiles.py` and `tuners.py`;
`race.py` evaluates completed paired evidence; `budget.py` allocates fixed
comparisons from explicit cost inputs; `calibration.py` keeps separate MCTS and
SO-ISMCTS measurements; `persistence.py` handles profile codecs, fingerprints,
checkpoint validation and trace recovery; and `report.py` renders progress,
diagnostics and artifacts. `search_metrics.py` owns only the shared `quantile`
and `search_adequacy` work diagnostics; `study_analysis.py` keeps compatibility
imports. Analyze's sampled-full horizon and Study's terminal-reach horizon remain
separate calculations.

Analyze's stable `meeple_bots.analysis` facade exposes the existing API.
`analysis/measurement.py` owns sampling, calibration, phase/root diagnostics and
operating points. `analysis/report.py` owns human output and both generic and
legacy JSON projections; `cli._evaluation_dict` remains a compatibility re-export
from `analysis.report`.

Probes retain `core.py`, `registry.py`, game fixtures, `runner.py`, and `cli.py`.
`probes/metrics.py` owns canonical action identity, aggregation, and important
action selection; `artifacts.py` owns version-1 capture/comparison I/O;
`compare.py` validates and joins saved captures; `report.py` renders summaries.
Offline comparison is descriptive and does not execute search. Package import
may still initialize native-dependent modules; native-free offline comparison
is not yet a supported installation mode.

Public imports remain under `meeple_bots` and `meeple_bots.api`. Implementation modules
should prefer authoritative owners directly; a few still import through the `api` or
`serialization` facades. This residual coupling forms no implementation import cycle.
CLI profile helper names remain available at their existing paths.
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

The GUI has three execution lifecycles. Tic-Tac-Toe, Connect Four, Boop, SPOTF and Connect6
continue through the Match-based `GuiController`. Can't Stop and Splendor use
`gui.step_session.NativeStepGui` for the identical condition-protected publication, human wait,
per-worker cancellation and post-step pacing; their native sessions still own chance and search
RNGs. The game adapters retain their distinct terminal predicates and trace formats. Can't Stop
identifies a decision by event count; Splendor also checks a controller UUID. Lost Cities keeps
its separate administrative loop, explicit chance/search RNG streams and observation-only SO
search; it does not save traces. The shared application constructs a replacement before cancelling
the previous controller. Native search may finish after cancellation, but its old worker cannot
publish into the replacement. Can't Stop's existing move payload has no session ID, so a click
from an earlier session with the same event count is not distinguishable at the server boundary;
the simple Match-based move payloads likewise have no session ID. Changing those contracts
requires a separate frontend/API compatibility decision.

The Python architecture refactor is complete through Phase 6. Public facades and
supported historical formats remain; private compatibility paths with current
test or documented consumers are retained until a separate removal decision.

Report adapters share extraction-table validation and competitive statistics through
`meeple_bots.reporting.common`. That module defines per-seat win/draw/loss records, score
aggregation, win-rate Wilson intervals and first-player advantage. Game modules select their
existing output columns and add domain-specific metrics and figures. Competitive results exclude
self-play; the historical overall first-player statistic includes it and considers decisive games
only. Wilson intervals describe win rate, not the score that awards half a point for draws.

Version-1 match/agent/action codecs and tournament trace validation live in
`matches.trace`. `serialization.py` and the trace names in `tournaments.py` re-export
the authoritative functions and class for existing imports; remove those compatibility
paths only after their documented callers migrate. `tournaments.py` still owns job
scheduling, standings and its paired seed policy. `extraction.pipeline` owns trace
streaming and atomic publication, `extraction.schema` owns CSV fields and common
decoders, and Boop/SPOTF extraction adapters own their game tables. The
`meeple_bots.extraction` facade preserves `extract_tournament`. Report dispatch is in
`reporting.dispatch`; dependency-light labels/intervals are in `reporting.base`,
while dataframe helpers remain in `reporting.common`. Renderers import helpers below
dispatch, and optional plotting libraries load only when a report is generated.

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
