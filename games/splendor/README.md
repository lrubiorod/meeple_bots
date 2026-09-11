# Splendor: public stochastic two-player rules

Rust owns the rules, setup, action validation and refill distribution. The initial market
contains four cards per tier and three nobles. Remaining cards are sorted ID sets; neither
state nor the game object contains a shuffled future deck. Catalog setup uses a separate
seed stream from simulation's environment RNG and each agent RNG.

The requested variant removes blind deck reservation and keeps visible reservations
public. A second explicit rule resolves blocked play: when no ordinary action is legal,
passing is mandatory. Two consecutive forced passes end the game as a draw; any ordinary
action resets the counter. Normal final-round scoring takes precedence if 15 points have
been reached. Passes include the mandatory end-of-turn noble choice if eligible and are
recorded as player actions. Player decisions include exact token returns, payments and any noble choice. Market tier/slot
indices in actions are zero-based; printed card tiers in the mechanical dataset are 1–3.
Gold payments are enumerated: spending gold instead of a colored token can change the
public bank and the opponent's legal take-two actions, so canonical colored-first payment
would remove strategically distinct choices. No heuristic, network or hidden-information
search is implemented.

## Mechanical data provenance

Color order is white, blue, green, red, black; gold is token index 5. Development IDs are
0–89 in the CSV row order (tiers 1–3); noble IDs are 0–9 in seal256's order. No artwork or
noble names are included.

Checked 2026-09-11:

- [Base rules, Space Cowboys 2014](https://bghub.org/r/splendor.pdf): setup, actions,
  payments, nobles and final-round/tiebreak rules.
- [bouk/splendimax card CSV](https://github.com/bouk/splendimax/blob/master/Splendor%20Cards.csv).
- [seal256/splendor mechanical constants](https://github.com/seal256/splendor/blob/master/pysplendor/splendor.py).
- [boardgamers data](https://github.com/boardgamers/splendor/blob/main/packages/engine/src/data.ts).
- [bouk noble requirements](https://github.com/bouk/splendimax/blob/master/src/noble.rs).

After parsing column names and color codes, all 90 bouk/seal card tuples matched exactly.
The boardgamers table does NOT match these sources despite its cross-verification comment;
it was not used for development cards. For example, its second black tier-1 card swaps
blue/red costs. Nine noble requirements match bouk (which lists only nine); the remaining
white-4/blue-4 noble matches seal and boardgamers. Boardgamers' white/green/red triple is
inconsistent with the white/black/red triple present in both bouk and seal and was rejected.

## Chance and search

`Game::chance_outcomes` enumerates event values and normalized probabilities;
`sample_chance` validates positive finite probabilities summing to one, then samples.
`apply_chance_outcome` is the environment transition entry point. Existing games that
implement direct sampling retain that implementation. Events share the existing action
transport enum, but `Refill` is never a player legal action and Splendor's `apply_action`
rejects it. The public Python API has distinct action/outcome classes.

The existing stochastic MCTS aggregates all sampled outcomes on the parent decision edge.
It resolves the chance transition, then looks up or creates the sampled decision-state
child. It does not precreate the whole outcome tree or apply UCB to outcomes. Decision
edge statistics estimate expected utility over refills. Full state equality/hash includes
remaining pools and pending refill. Reuse verifies actual chance transitions before
retaining sampled subtrees; unseen outcomes safely reset reuse. Deterministic search is
unchanged.

Simulation records every refill with its `after_ply`. Replay applies those exact events
in Rust, rather than rerunning the RNG. Fixed-seed reproducibility requires fixed-iteration
agents; time-budget agents also depend on runtime scheduling, as in other games.

## Python and CLI

`Splendor`, `SplendorAction`, `SplendorMoveKind`, `SplendorState`,
`SplendorPlayerState`, `SplendorChanceOutcome`, `SplendorSession` and `replay_splendor` are public.
`Splendor.initial_state(seed)` returns a position with native legal-action/transition methods.
Completed match snapshots are summaries; use replay to recover an executable position.

```
meeple-bots gui --game splendor
meeple-bots match --game splendor --first random --second random --seed 42
meeple-bots match --game splendor --first mcts --second random --seed 42
```

Match, batch and tournament traces include `chance_events` and `splendor_state`.
The browser GUI supports human, Random and MCTS seats, including UCT/UCB1-Tuned,
iterations, rollout depth, tree reuse and transpositions. It displays the bank, market,
public reserves, bonuses and nobles. Click a colored development card to buy or reserve it, or click supply tokens to
select gems (click a color twice to take two). Cards display victory points and printed
costs; nobles use distinct crown medallions. Complete any payment, token returns and
noble choice in the contextual panel before confirming; the browser does not calculate legality. Refills run through
the native session using the same RNG streams as catalog matches. The session forwards
agent lifecycle events, including chance, so tree reuse remains valid. Restart cancels
the previous worker and rejects stale browser submissions.

Optional GUI traces are saved under `results/gui/splendor/` in `splendor_session_v1`
format with seed, player configuration and ordered player/chance events. This is a
session trace, not a tournament trace; replay the recorded actions through native
positions without resampling. The GUI exposes the full event history.

No live match observer, game-specific report, strategic evaluator or automatic calibration
study is registered. The existing extractor and game reports are not registered for Splendor. Use the public
replay API to inspect/validate stochastic traces. The existing complexity evaluator requires `DeterministicGame`;
Splendor deliberately does not claim that capability.
