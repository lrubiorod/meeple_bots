# Meeple Bots

A modular framework for simulating board games with intelligent agents. The performance-critical
engine is written in Rust, while the future user interface and analysis tools will live in Python.

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
- `meeple_bots_python_bindings` will be the PyO3 boundary and does not participate in critical
  loops.

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

## Development

Requires a Rust toolchain compatible with the 2024 edition.

```bash
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

The Python package is still a scaffold. Bindings will be implemented once the Rust contracts have
proven stable.
