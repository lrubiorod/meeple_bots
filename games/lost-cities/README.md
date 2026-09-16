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
  to the existing `Game::Observation` contract. Other games need no changes.
  Sampling subtracts known cards from the full multiset, uniformly deals the hidden
  opponent hand, and leaves the remaining pool for future Chance. It checks card
  conservation, phase hand sizes (including seven-card draw phases), public zones
  and observer identity. Invalid observations return an error.

**Determinization is not chance.** Determinization samples which unknown cards are
currently in the opponent's hand; Chance samples which remaining card is drawn next.
A determinization assigns no future order. Observing any compatible sampled state
again reproduces exactly the original observation; root legal actions are identical
across determinizations for the acting observer. This initial model uses only the
listed observation data, with no inference from play or private-history framework.

## Agents and interfaces

Only `RandomAgent` is registered, selecting solely from legal actions with no-op
lifecycle callbacks. Lost Cities does **not** implement `PerfectInformationGame`;
normal and stochastic MCTS therefore fail their Rust type bounds. Catalog and Python
also reject MCTS explicitly. The legacy `Agent` lifecycle still accepts authoritative
states: it is **not** a general hidden-agent security boundary. Future SO-ISMCTS must
use an observation-safe lifecycle; do not register arbitrary legacy agents here.

Rust catalog, Python `LostCities`, CLI matches, batches and tournament JSONL are
available. Trace replay validates both player actions and setup/draw Chance events.
Authoritative positions, exact Chance distributions and saved match traces are
engine/administrative data, **not player observations**; traces disclose private draws.
Live match observers, study, PIMC/ISMCTS/SO-ISMCTS/RIS-MCTS, heuristics and
information-set trees are intentionally not implemented. A separate administrative
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
assert game.observation(world, 0) == observation
assert game.legal_actions(world) == game.legal_actions(state)
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

Human controls expose only Rust-generated legal actions: first play/discard,
then draw. Random decisions choose uniformly from that same action list without
inspecting hands or deck composition. Environment chance still runs through Rust,
with a separate seeded RNG stream from each random policy. The GUI seed reproduces
GUI sessions; its Python orchestration does not promise identical random choices
to native tournament execution. Observations and determinization APIs are unchanged.

New Match cancels the previous worker, including human waits; stale/duplicate moves
are rejected using session and decision tokens. Step delay controls inspection pace.
A 10,000-decision execution safeguard reports an error without awarding a result.
GUI trace-file saving is not provided; the current transition log is available in
this debug view. Use native random tournaments for persisted validated match traces.
