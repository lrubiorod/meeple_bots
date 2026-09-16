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
an environment-owned chance boundary. Outcomes may be public or private.

Capability traits state additional guarantees required by algorithms:

- `DeterministicGame`: the game never requests a chance transition.
- `PerfectInformationGame`: every player can observe the authoritative state.
- `ImperfectInformationGame`: optionally sample a complete hidden assignment from a
  player observation and an explicit RNG, preserving that observation exactly.
- `HeuristicGame`: the game exposes one or more indexed state evaluators.
- `TwoPlayerZeroSumGame`: the current two-seat adversarial model.

MCTS expresses its supported domain through these trait bounds. An incompatible Rust game cannot
silently enter a search that assumes determinism or full state access.

## Agent contract

`Agent<G>` selects one `G::Action` from a `DecisionContext`. Every agent can access the game, legal
actions, active player, and player observation. `DecisionContext::state()` requires
`PerfectInformationGame`, but this restriction applies only to that accessor.

The lifecycle methods `on_match_start`, `on_action_applied` and `on_match_end` receive the full
`G::State` without that bound. An agent could retain private information received there; supplying
a filtered `observation()` alone would not prevent this. Simulation observers also receive full
state. The legacy lifecycle does not enforce hidden-information isolation. Lost Cities
registers only the audited RandomAgent (legal actions only, no-op callbacks); future
hidden-information searches require an observation-safe lifecycle. Lost Cities does
not implement `PerfectInformationGame`, so both MCTS backends are excluded by type
bounds as well as runtime catalog validation.

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

## Catalog modules

`catalog/src/configuration.rs` owns MCTS configuration types, policy/evaluator adapters,
game-specific constructors and validation. `participant.rs` delegates the agent contract to an
independently configured participant. `lib.rs` retains catalog identifiers, descriptors, errors,
match/evaluation dispatch and trace conversion. Public configuration types and constructors are
explicitly re-exported at the crate root, preserving existing Rust and binding imports.

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

## Adding a game

1. Implement authoritative rules in `games/<game>` using `Game`. Add only the capability
   traits whose contracts the game satisfies; test legal actions, turn order, terminal outcomes
   and utilities in that crate. Keep dependencies on core, not on concrete agents.
2. Register the crate in the workspace and catalog. Add `GameId`, configured agent
   constructors in `configuration.rs`, and `game_search_capabilities`. Expose heuristic
   parameter schemas from `HeuristicGame`, not from Python constants.
3. Connect match, observed-match, batch and trace dispatch. Preserve seeded setup and
   per-seat randomness. Use generic replay where possible; add a game-owned analyzer when
   domain-specific extraction requires one.
4. Add typed action/state conversion and human callbacks at the PyO3 boundary. In Python,
   register game/action types, native conversion, serialization and CLI identifiers. Export
   public values through `meeple_bots`; reuse shared GUI/report machinery when adding a UI.
5. Verify Random/MCTS and supported human pairings, observed versus retained traces,
   invalid replay rejection, batch seat swaps, diagnostics and capability metadata. Rebuild
   the extension before running Python integration tests.
6. Document identifiers, supported options, action encoding and extraction guarantees in
   the game guide. Keep reusable tests in Git and personal studies under `local/` or `results/`.

## Adding an agent

1. Implement `Agent<G>` in `agents/<agent>` against core contracts. Declare actual capability
   bounds; do not import individual game crates into reusable production logic.
2. Implement or deliberately retain defaults for all lifecycle hooks and decision statistics.
   Test legal selection, seeded behavior and state reset/reuse across matches.
3. Extend catalog configuration and participant construction/delegation. Handle selection,
   `last_decision_stats`, start, action updates and end consistently. Construct each seat
   independently; do not add branches for every possible opponent.
4. Extend binding configuration, Python values/validation, profile parsing and serialization.
   Preserve the complete agent configuration in traces so extraction can distinguish agents.
   Game rules should not need changes to accommodate the agent.
5. Test supported games and opponent types, traced/observed equivalence, batch independence,
   errors and timing attribution. If the adapter changes dispatch cost, measure a bounded
   release workload before claiming equivalent performance.
6. Update the agent guide and relevant help/profile examples. Run the affected Rust suites,
   rebuild the extension, then run Python integration tests; CI runs the full checks.

## Current boundaries

The implemented games are sequential, two-player and zero-sum. Lost Cities is the
first imperfect-information game; other games have perfect information. The generic
simulator resolves Chance with an independent environment RNG using `sample_chance`
and `apply_chance_outcome`; event actions never belong to player legal-action lists.
Authoritative traces record outcomes with `after_ply` for replay, including setup at
position zero. These traces disclose private draws and are administrative data, not
observations.

Lost Cities separates owned observations, stochastic environment states and complete
simulation worlds. Determinization samples an opponent hand plus an ordered deck;
simulation draws consume that order without resampling, while real draws use Chance. It implements no search agent, belief model or
information-set tree. Before registering SO-ISMCTS, provide an observation-safe
agent lifecycle and review action visibility, retained memory and observer access.
See [Lost Cities](../games/lost-cities/README.md) for the exact variant, observation
fields, conservation/roundtrip invariants and currently supported interfaces.

See the [agents guide](../agents/README.md) for current MCTS behavior, the
[MCTS roadmap](../agents/MCTS_ROADMAP.md) for possible extensions, and the
[evaluation guide](evaluation/README.md) for empirical analysis.
