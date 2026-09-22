# Can't Stop

[Back to games](../README.md)

`cant-stop` implements the basic two-player game: eleven columns (2–12), four fair six-sided
dice, three temporary runners and a race to secure three columns. No private hands, goals or
future random outcomes are part of the player-visible state. Player 0 starts; use swapped seats
when comparing opponents. This is a fixed-seat convention rather than a preliminary dice contest.

The basic rules follow [the rulebook](https://cdn.1j1ju.com/medias/d8/88/07-cant-stop-rulebook.pdf).
Choose one pairing of the four dice and apply both sums when possible within that pairing. If
only one runner slot remains, either eligible new column can use it. Equal sums can advance one
column twice; at the summit only the available step is applied. Runners cannot switch columns
mid-turn. A failed roll discards temporary progress. Stopping banks it and closes completed
columns to both players. A summit reached during the turn is not a claim until banked; a failed
roll can still lose it. There are no jumping, forced-reroll-on-occupied-space or multiplayer variants.

## Play

```bash
meeple-bots gui --game cant-stop
```

Both seats can be human, Random or MCTS. The GUI shows permanent markers, temporary runners,
closed columns, public dice, action choices and recent events. Click an advance, then choose
whether to roll again or stop. The environment performs dice rolls automatically.

[The reference profile](../../configs/mcts/cant-stop-baseline.toml) supplies initial MCTS values.
Edit controls for the next match without changing the file. The initial 2,000 iterations and
100-action horizon are an interactive starting point, not a strength-calibrated opponent.
H0 estimates relative column progress, giving completed columns greater weight and discounting
unbanked progress. Uniform rollouts may be weak at deciding when to stop; strength should be
measured before drawing conclusions about selection policies.

Saving a completed GUI match writes `cant_stop_session_v1` JSON under `results/gui/cant-stop/`:
seed, player settings, final public state and ordered player/dice events. It is distinct from the
historical deterministic tournament format. These files are local and ignored by Git.

## Python session API

```python
from meeple_bots import CantStopSession, MctsAgent, RandomAgent

session = CantStopSession(
    seed=42,
    first=MctsAgent(iterations=2_000, rollout_depth=100, heuristic=0),
    second=RandomAgent(),
)
while session.snapshot()["winner"] is None:
    state = session.step()  # one automated decision or one public dice event
print(state["winner"])
```

Use `None` for a human seat. When `waiting_human` is true, call `step(index)` with an index into
`legal_actions`; otherwise call `step()` without an action. Dice are environment events, never
legal player actions. Snapshots contain no RNG state. A session is sequential; use separate
sessions for concurrent matches.

Can't Stop exposes the GUI, Python session API and typed Rust simulator. It is not
registered in generic Python `Match`/`Batch`, tournament CLI, public `analyze`/`study`,
or tournament extraction. Other stochastic games already use those shared paths;
Can't Stop still needs its own catalog/transport integration. Its session traces
are a separate format and retain dice history.

## Engine integration

- `Game::sample_chance` samples an event from a supplied RNG; `apply_chance_outcome` is the environment transition entry point.
  `PositionStatus::Chance` is separate from `PlayerTurn`.
- The simulator has a third seeded RNG stream for actual events, independent of both agents.
  More search simulations cannot consume or reveal future real dice.
- `on_chance_applied` informs agents; `MatchObserver::on_chance` informs observers.
  `TracedChance::after_ply` interleaves outcomes with player actions, including consecutive rolls.
- `StochasticMctsAgent` is game-independent and samples successors during selection and rollout.
  Decision edges aggregate rewards across outcomes. Distinct sampled successor states have
  distinct continuation nodes. UCB never selects a chance result.
- Its Rust type accepts a generic `RolloutPolicy`, defaulting to `UniformRandom`. The search
  validates the policy, resolves chance before requesting a player action, and supplies the
  active and root players. Learning callbacks receive player actions only, with fresh memory
  for each decision. Shared policy, condition and selection-bias contracts support public
  chance games. Informed rollouts evaluate one independently sampled successor per candidate;
  the selected action samples chance afresh during simulation. This is a noisy heuristic
  estimate, not exact expectation.
- UCT and UCB1-Tuned, configurable rollouts, neutral/H0 cutoff and iteration/time budgets are supported.
  Time includes setup/search/finalization, with the previous finalization cost reserved and
  retained-tree lifecycle maintenance charged to the next decision. Depth counts player actions along a simulation;
  chance chains have a separate defensive bound.
- The deterministic implementations retain their existing search semantics.

Tests cover pairings, runner limits, doubles, summits, banking, busts, victory, rejection without
mutation, exact dice replay, RNG independence, expected-value search and GUI restart behavior.

Configured rollouts also accept `greedy`, `epsilon_greedy`, and `conditional`, with neutral/H0
evaluators. Conditions use `choose` (advance runners) or `continue` (roll again or stop),
checked on the player decision before acting. Other games reject these phase names.

`mast` learns averages keyed by player and action within each search. It credits tree and
rollout decisions, never dice events, and starts with fresh memory on the next decision.
It can also be used in either branch of a conditional rollout.

Progressive bias averages heuristic evaluations of sampled successors on each decision edge,
with the existing weight/(visits+1) decay and acting-player sign. Terminal successors use
terminal utility. Weight zero or a false phase condition skips evaluation. Root diagnostics
report the running heuristic mean and final bias term; no extra chance draws are needed.

With `tree_reuse=true`, the typed wrapper follows accepted player actions and each public
chance event, verifies the resulting states, and retains the observed continuation once
chance resolves. Unexplored outcomes reset the tree. Clone, match start and match end clear
retained search data. Maintenance is charged to the next time-budget decision.

`transpositions=true` merges exactly equal public states, independently of tree reuse.
Each incoming decision edge keeps its own visits, reward moments, and heuristic samples.
Revisiting a node ends tree traversal and uses the remaining rollout horizon, preventing
repeated tree credit within one simulation. No dice or state canonicalization is applied.


## Configure the mechanisms

The reference TOML stays uniform, with reuse and transpositions disabled. Copy it for an
experiment and replace only the setting being tested. These are independent examples:

```toml
rollout_policy = { kind = "epsilon_greedy", epsilon = 0.1, evaluator = { kind = "game_heuristic", index = 0 } }
```

```toml
rollout_policy = { kind = "conditional", condition = { kind = "turn_phase", phase = "continue" }, primary = { kind = "greedy", evaluator = { kind = "game_heuristic", index = 0 } }, fallback = { kind = "uniform_random" } }
```

```toml
rollout_policy = { kind = "mast", epsilon = 0.1 }
```

```toml
progressive_bias = { weight = 0.25, evaluator = { kind = "game_heuristic", index = 0 }, condition = { kind = "turn_phase", phase = "continue" } }
root_diagnostics = true
```

Set `tree_reuse = true` and/or `transpositions = true` to test retained search and state sharing.
All these mechanisms can be combined. H0 currently has no configurable parameters.
The example epsilon and bias weights are illustrative, not strength-calibrated values.

In the browser, **Políticas y memoria** exposes the rollout and fallback policy, phase,
progressive bias, tree reuse, transpositions and root diagnostics for each MCTS player.
Controls inherit the reference profile and apply to the next match. Session events and saved
GUI JSON include `search_nodes`, `root_actions` and `tree_reuse`; dice events have no search
statistics. Root visits include inherited visits when reuse succeeds, while `search_iterations`
counts only the work of the current decision. These added fields preserve the existing
`cant_stop_session_v1` event order and public dice history.

The Python API exposes the same options directly:

```python
from meeple_bots import CantStopSession, MctsAgent, Mast, ProgressiveBias, GameHeuristic, RandomAgent

session = CantStopSession(
    seed=42,
    first=MctsAgent(iterations=2000, rollout_depth=100, heuristic=0,
                    rollout_policy=Mast(0.1),
                    progressive_bias=ProgressiveBias(0.25, GameHeuristic(0)),
                    tree_reuse=True, transpositions=True, root_diagnostics=True),
    second=RandomAgent(),
)
```

These mechanisms are available through Can't Stop sessions. Generic match, tournament
and public analysis integration for this game remains outside that session API, as
explained above; the available mechanisms are not evidence of playing strength.
