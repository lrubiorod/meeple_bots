# Rust architecture

[Back to the project overview](../README.md)

Meeple Bots keeps concrete states and actions strongly typed while allowing games and agents to be
selected at runtime outside performance-critical loops.

## Workspace layout

```text
python-bindings
      |
   catalog
   ├── games ───────────────> core
   ├── agents ──────────────> core
   ├── simulation ──────────> core
   ├── evaluation ──> MCTS, simulation, core
   └── core
```

- `meeple_bots_core` defines games, agents, players, errors, randomness, and capability traits.
- `meeple_bots_simulation` runs reproducible matches and sequential batches.
- `meeple_bots_evaluation` samples game trees, calibrates MCTS, and benchmarks relative strength.
- `games/*` owns concrete state, action, and rule implementations.
- `agents/*` contains policies generic over the core contracts.
- `meeple_bots_catalog` maps runtime game and agent identifiers to monomorphized calls.
- `meeple_bots_python_bindings` translates public Python values at the PyO3 boundary.

## Game contract

Each game implements `Game` with its own associated types:

- `State`: the authoritative mutable position.
- `Action`: one complete legal decision.
- `Observation`: the information visible to a player.
- `LegalActions`: an iterator over currently legal actions.

The contract also provides initial state construction, position status, state transitions, and
terminal utilities. `PositionStatus` distinguishes player turns, terminal positions, and a reserved
chance boundary.

Capability traits describe assumptions required by generic algorithms:

- `DeterministicGame`
- `PerfectInformationGame`
- `HeuristicGame` for games that optionally expose indexed state evaluators.
- `TwoPlayerZeroSumGame`

MCTS declares these capabilities as trait bounds, so unsupported games fail at compile time instead
of relying on runtime checks inside the search.

## Agent contract

An `Agent<G>` receives a `DecisionContext` for a concrete game and selects one `G::Action`.
`DecisionContext` exposes legal actions and the player's observation; the authoritative state is
available only when `G` implements `PerfectInformationGame`.

Randomness is supplied through the small `RandomSource` abstraction. The simulation crate derives
an independent deterministic stream for each player from the match seed.

## Static and runtime dispatch

Matches use generic functions such as `play_match<G, A, B>`. Concrete game and agent combinations
are therefore monomorphized, without trait objects or erased actions in the match loop.

Runtime selection occurs once in the catalog. `GameId` and `AgentConfig` choose the corresponding
generic call before simulation begins. The same boundary converts typed traces and final states
into catalog reports for Python serialization.

## Simulation and observation

The simulation crate owns:

- Match and batch configuration.
- Independent seeded RNG streams.
- Turn and ply-limit enforcement.
- Agent and illegal-action error propagation.
- Compile-time observers and typed action traces.

The Python bindings do not participate in automated decision loops. Python is called during a
match only when a `HumanAgent` selector is active.

## Current boundaries

Current implementations are sequential, deterministic, two-player, zero-sum, and
perfect-information games. `PositionStatus::Chance` and per-player observations reserve extension
points, but stochastic transitions and hidden-information algorithms are not implemented yet.

For the empirical analysis layer, see the [evaluation crate guide](evaluation/README.md).
