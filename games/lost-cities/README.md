# Lost Cities

**Lost Cities single-round experimental variant**: two players, five expeditions
(red, green, blue, yellow, white), 60 physical cards. Each color has numbers 2–10
and three interchangeable wagers. One deal is one Meeple Bots match; the official
three-round accumulated-score metagame is not implemented. No art or card text is included.

Each player receives eight cards through environment Chance transitions. A physical
turn has two player decisions: `Play(card)` or `Discard(card)`, then `DrawDeck` or
`DrawDiscard(color)`. Wagers precede all numbers; numbers strictly increase. A player
cannot take from the pile they just discarded onto. Only the top discard is drawable;
observations preserve the full public stacks. Duplicate wagers produce one strategic
action, but retain their physical multiplicity when sampled.

`DrawDeck` enters Chance. The engine samples one physical card uniformly without
replacement from a canonical multiset, using its existing environment RNG, independent
of agent RNGs. Drawing the last card immediately ends the round. Discard draws do not
consume the deck, so `maximum_decision_horizon()` is `None`. Generic match safety limits
are execution safeguards, not rules and not scored terminal positions.

An empty expedition scores zero. Otherwise its score is
`(sum(numbers) - 20) * (1 + wagers)`, plus **20 after multiplication** when at least
eight cards were played (including wagers). Hands never score. Totals sum all five
colors; terminal utilities are +1/-1 for winner/loser or 0/0 for a tie.

## Information boundary

- `LostCitiesState` is authoritative: both hands, public expeditions and discard
  stacks, deck pool, current player, phase and same-turn discard restriction.
- `LostCitiesObservation` is an owned `Clone + Eq + Hash` value: observer identity,
  exact own hand, opponent hand size, deck size, all public expeditions and discard
  stacks, current player, phase/setup progress and discard restriction. Scores are
  derivable from public expeditions. It contains **no state handle, opponent hand,
  deck composition, seed or future order**. Python observations are frozen/hashable
  and serialize only these fields.
- The optional core `ImperfectInformationGame` trait adds `sample_determinization`
  and an associated `Determinization` type to the existing observation contract.
  Lost Cities returns a separate `LostCitiesSimulationWorld`, not an environment state.
  Fisher-Yates shuffles all unknown physical cards with the supplied RNG, assigns
  the required opponent hand and keeps the remainder in exact future draw order.
  Identical wagers preserve their natural multiplicities. Validation checks all 60
  cards, public zones, phase hand sizes and agreement between order and remaining pool.

**Real game: chance chooses reality.** Environment states retain an unordered pool;
real matches, the debug GUI and replay still use normal stochastic Chance events.

**Simulation: determinization samples one complete possible reality.** The temporary
world owns the opponent hand and `deck_order` (next card first). Its `apply_action`
reuses real rules but resolves `DrawDeck` immediately from that order, without an RNG
or a new stochastic Chance branch. Observations taken during setup or a pending draw
are preserved exactly at sampling; `resolve_pending_draws()` explicitly advances
those pending transitions using the fixed order. No uncertainty is sampled twice.

**SO-ISMCTS tree: retain only observable distinctions and learned statistics.**
Never key a persistent tree by the complete world or put its hidden cards/deck order
in tree nodes. Two worlds with different deck orders have equal observations before
a draw; a hidden opponent draw remains indistinguishable to the observer. An own
draw legitimately changes the observer's hand. A fresh world must be sampled per
iteration and discarded afterward. See [SO-ISMCTS](../../agents/so-ismcts/README.md)
for the baseline implementation and its limitations.

The world provides `observation`, `legal_actions`, `apply_action` and validation;
Rust also exposes status/terminal utility and read-only administrative state/order
inspection. Python `world.state` is a summary with no executable environment handle.
Use `world.apply_action(action)` for simulation, not `game.apply_action(world, action)`.
Sampling is uniform without opponent inference or a private-history framework.
Observing every sampled world reproduces the original observation exactly.

## Agents and interfaces

`RandomAgent` and `SoIsmctsAgent` are registered. Random selects solely from legal
actions. SO-ISMCTS receives only a player observation and legal root actions through
a trusted catalog adapter with no-op lifecycle callbacks. Lost Cities does **not**
implement `PerfectInformationGame`; normal and stochastic MCTS fail their Rust type
bounds and are explicitly rejected by catalog/Python. The legacy `Agent` lifecycle
still accepts authoritative states: it is **not** a general hidden-agent security
boundary. Never forward those callbacks to an information-set search object.

Rust catalog, Python `LostCities`, CLI matches, batches and tournament JSONL are
available. Trace replay validates both player actions and setup/draw Chance events.
Authoritative positions, exact Chance distributions and saved match traces are
engine/administrative data, **not player observations**; traces disclose private draws.
Live match observers, study tuning, PIMC/MO-ISMCTS/RIS-MCTS and heuristics
are intentionally not implemented. A separate administrative
GUI supports human and random players (see below). Reports currently consist
of CLI scores, tournament summaries and validated JSON traces; no game-specific
HTML report or extraction analysis is provided.

```bash
.venv/bin/python -m meeple_bots match --game lost_cities \
  --first random --second random --seed 42
```

```python
from meeple_bots import LostCities

game = LostCities()
state = game.initial_state(seed=42)  # Resolves the sixteen setup Chance events.
observation = game.observation(state, observer=0)
world = game.sample_determinization(observation, observer=0, seed=123)
assert world.observation(0) == observation
assert world.legal_actions() == game.legal_actions(state)
```

## Open-hand debugging GUI

```bash
.venv/bin/python -m meeple_bots gui --game lost_cities
```

Choose Human or Random independently for both seats. The GUI intentionally shows
**both hands**, all expeditions and discard stacks, scores, remaining deck count,
an expandable unordered deck pool, and a transition log including private draws.
This is an administrative test interface, not a private player view or remote
multiplayer service. There is no handoff/reveal screen.

Click a card in the active human hand to see its legal Play expedition / Discard
options, then choose a draw source. Controls expose only Rust-generated legal actions. Random decisions choose uniformly from that same action list without
inspecting hands or deck composition. Environment chance still runs through Rust,
with a separate seeded RNG stream from each random policy. The GUI seed reproduces
GUI sessions; its Python orchestration does not promise identical random choices
to native tournament execution. The debug GUI uses only real-game execution, never determinized simulation worlds.

New Match cancels the previous worker, including human waits; stale/duplicate moves
are rejected using session and decision tokens. Step delay controls inspection pace.
A 10,000-decision execution safeguard reports an error without awarding a result.
GUI trace-file saving is not provided; the current transition log is available in
this debug view. Use native random tournaments for persisted validated match traces.
