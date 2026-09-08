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

This first stochastic integration exposes the GUI, Python session API and typed Rust simulator.
The generic Python `Match`/`Batch`, tournament CLI, `analyze`, and historical extractors still
support the four deterministic games. They must first gain explicit chance-event transport and
replay validation before accepting `cant-stop`; they do not silently omit its dice history.

## Engine integration

- `Game::sample_chance` samples an event from a supplied RNG; `apply_action` applies it.
  `PositionStatus::Chance` is separate from `PlayerTurn`.
- The simulator has a third seeded RNG stream for actual events, independent of both agents.
  More search simulations cannot consume or reveal future real dice.
- `on_chance_applied` informs agents; `MatchObserver::on_chance` informs observers.
  `TracedChance::after_ply` interleaves outcomes with player actions, including consecutive rolls.
- `StochasticMctsAgent` is game-independent and samples successors during selection and rollout.
  Decision edges aggregate rewards across outcomes. Distinct sampled successor states have
  distinct continuation nodes. UCB never selects a chance result.
- UCT and UCB1-Tuned, uniform rollout, neutral/H0 cutoff and iteration/time budgets are supported.
  Time includes setup/search/finalization, with the previous finalization cost reserved. There is
  no retained tree or lifecycle maintenance. Depth counts player actions along a simulation;
  chance chains have a separate defensive bound.
- Tree reuse, transpositions, MAST, informed rollouts and progressive bias are rejected by the
  configured integration. The deterministic implementations are unchanged.

Tests cover pairings, runner limits, doubles, summits, banking, busts, victory, rejection without
mutation, exact dice replay, RNG independence, expected-value search and GUI restart behavior.
