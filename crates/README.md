# Rust architecture

[Back to the project overview](../README.md)

Meeple Bots keeps game states and actions strongly typed throughout simulation. The catalog selects a concrete game and constructs each participant independently. Search
implementations remain statically typed; participant variants dispatch at agent-method boundaries.

## Layers

```text
Python API and CLI
        |
PyO3 bindings
        |
runtime catalog
   |       |       |
 games   agents  evaluation
    \       |       /
     simulation and core contracts
```

| Workspace area | Responsibility |
| --- | --- |
| `crates/core` | Games, agents, players, errors, randomness, and capability traits. |
| `crates/simulation` | Reproducible matches, batches, observers, and typed traces. |
| `games/*` | Concrete state, action, legal-action, and transition rules. |
| `agents/*` | Search and selection policies generic over core contracts. |
| `crates/evaluation` | Structural sampling and local MCTS cost calibration. |
| `crates/catalog` | Runtime identifiers mapped to concrete generic calls. |
| `crates/python-bindings` | PyO3 conversion between Rust and the public Python model. |

Dependencies point inward toward `meeple_bots_core`. Concrete games do not depend on agents,
Python, or the catalog.

## Match execution

A normal automated match follows one short path:

1. Python converts a public game and agent configuration at the binding boundary.
2. The catalog selects a concrete game and constructs each seat independently.
3. `play_match<G, A, B>` creates the initial state and independent seeded RNG streams.
4. The active `Agent<G>` receives a read-only `DecisionContext` and returns one `G::Action`.
5. The game validates and applies the action to its authoritative state.
6. An optional observer records the action and agent time: selection (including statistics)
   plus lifecycle maintenance. Maintenance is attributed to the agent that performs it,
   including work after an opponent's action.
7. Terminal utilities and the typed trace are converted back to the public result model.

The Python interpreter does not participate in automated decision loops. Python is called during a
match only for a human selector or observer callback.

### Agent time accounting

`DecisionTiming` separates selection and maintenance. Each recorded decision includes that
player's pending maintenance since its previous decision plus the update after its new action.
Match-start work is charged to the first decision. At match end, remaining updates and cleanup
are added to each player's last decision through `on_remaining_maintenance`. If a seat never
acts, its lifecycle time is retained in `unassigned_maintenance_time` instead.

The sum of final decision times per seat therefore covers all measured agent calls during the
match. Game transitions, observer callbacks, serialization and agent construction outside the
runner are excluded. Live `on_action` timing includes work known then; retained traces receive
terminal adjustments before `on_finish`. A custom observer that retains times must handle
`on_remaining_maintenance` as well. Runs without timing do not sample the clock.

## Game contract

Each game implements `Game` with its own associated types:

| Associated type | Role |
| --- | --- |
| `State` | Complete authoritative position mutated by legal transitions. |
| `Action` | One complete player decision. |
| `Observation<'a>` | Information visible to one player. |
| `LegalActions<'a>` | Iterator over actions legal in the current state. |

The trait also constructs the initial state, reports whose turn it is, applies actions, and returns
normalized terminal utility. `PositionStatus` distinguishes player turns, terminal positions, and
a reserved chance boundary.

Capability traits state additional guarantees required by algorithms:

- `DeterministicGame`: the game never requests a chance transition.
- `PerfectInformationGame`: every player can observe the authoritative state.
- `HeuristicGame`: the game exposes one or more indexed state evaluators.
- `TwoPlayerZeroSumGame`: the current two-seat adversarial model.

MCTS expresses its supported domain through these trait bounds. An incompatible Rust game cannot
silently enter a search that assumes determinism or full state access.

## Agent contract

`Agent<G>` selects one `G::Action` from a `DecisionContext`. Every agent can access the game, legal
actions, active player, and player observation. Full state access exists only for agents whose game
implements `PerfectInformationGame`.

Randomness is explicit through `RandomSource`. The simulation crate derives a separate
deterministic stream for each seat from the match seed, avoiding accidental coupling between the
two agents' random choices.

## Static and runtime dispatch

The simulation functions are generic, so Rust monomorphizes every supported `G`, `A`, and `B`
combination. There are no erased action types or agent trait objects inside the match loop.

The catalog provides the dynamic edge needed by Python and the CLI. `GameId` selects the
concrete game. `ConfiguredAgent<M>` in `catalog/src/participant.rs` represents Random or the
configured, game-compatible MCTS type; each seat is built independently from `AgentConfig`.
It delegates selection, diagnostics and every lifecycle hook. Variant dispatch occurs at those
boundaries, not within MCTS iterations. Batch factories clone unplayed participant templates.
SPOTF continues to construct a seeded game for every match.

The bindings use one generic `PythonParticipant<M>` to add human selectors and their callbacks.
Observed and unobserved matches share participant construction, so adding a variant does not
require adding a branch for every opponent. Game-specific typed observers and action conversion
remain in the bindings; concrete traces and final states are converted into catalog reports.

## Search capability metadata

`game_search_capabilities(GameId)` exposes the heuristic indices and their named parameter
schemas (default, minimum and maximum), plus support for turn-phase conditions. Heuristic
metadata comes directly from `HeuristicGame`; condition support shares the catalog validator's
predicate. When adding a game, register this descriptor alongside its configured constructor.

The private binding `_native.game_search_capabilities` publishes these descriptors. Python's
`_capabilities` module caches them for API validation and GUI heuristic choices. Python retains
value/type checks and presentation; native validation remains authoritative for Rust callers and
for callers bypassing the Python facade. These descriptors cover the search metadata currently
needed by Python, not a dynamic game/plugin registry or a complete enumeration of core traits.
Rebuild the extension after changing a game's schema or catalog registration.

## Trace validation

`analyze_seeded_trace` dispatches to the game-specific Boop/SPOTF analyzers or to a generic
`Game` replay for Connect Four and tic-tac-toe. Generic replay checks the active player before
each accepted transition, rejects actions after termination and requires a terminal final state.
Its terminal utilities are returned to the extractor, which checks the declared outcome.
The binding only converts actions; legality remains in the Rust game implementation.

## Where changes belong

| Change | Primary location | Integration points |
| --- | --- | --- |
| New game rules | `games/<game>` | Catalog, bindings, Python model, and presentation. |
| New generic agent | `agents/<agent>` | Catalog, bindings, and Python configuration. |
| Match behavior | `crates/simulation` | Catalog and binding tests when public behavior changes. |
| Shared capability | `crates/core` | Every algorithm and game that relies on it. |
| Structural metric | `crates/evaluation` | Catalog, bindings, and Python report model. |
| Python-only UI or report | `python/src/meeple_bots` | Public API or CLI registration as needed. |

## Current boundaries

The implemented games are sequential, deterministic, perfect-information, two-player, and
zero-sum. `PositionStatus::Chance` and per-player observations reserve useful extension points, but
chance execution and hidden-information search are not implemented yet.

See the [agents guide](../agents/README.md) for current MCTS behavior, the
[MCTS roadmap](../agents/MCTS_ROADMAP.md) for possible extensions, and the
[evaluation guide](evaluation/README.md) for empirical analysis.
