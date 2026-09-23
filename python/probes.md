# Behavioral search probes

[Python guide](README.md)

| Tool | Purpose |
| --- | --- |
| Tests | Correctness and information invariants |
| `analyze` | Structural/search-cost characterization |
| `probe` | Behavior from fixed positions across search seeds and iteration budgets |
| `study` | Competitive strength in full matches |

Probes are measurements, not strategic pass/fail gates. A dominant action is the
most frequently selected action in this sample, not a claim of optimality.
These distributions can reveal whether more search changes or reinforces a
preference. They do not establish full-game strength.

## Running

Use the existing environment, from the repository root:

```bash
.venv/bin/python -m meeple_bots probe --game lost_cities --list
.venv/bin/python -m meeple_bots probe --game lost_cities --suite ordering --list

.venv/bin/python -m meeple_bots probe \
  --game lost_cities --agent so_ismcts \
  --probe preserve-low-sequence-early \
  --iterations 1000,5000,20000 --seeds 16 \
  --output results/probes/lost-cities-ordering-v1
```

For an initial complete baseline:

```bash
.venv/bin/python -m meeple_bots probe \
  --game lost_cities --agent so_ismcts \
  --iterations 1000,5000,20000 --seeds 16 \
  --output results/probes/lost-cities-so-baseline-v1
```

This schedules 480 independent searches and 4.16 million determinizations across
ten positions. Uniform rollouts in early positions can be long; try one probe
first. No wall-clock estimate is implied. The larger `1000,5000,20000,100000`
with 32 seeds schedules 1,280 searches and 40.32 million determinizations.
The implementation is sequential and reports progress periodically within each
budget and when each budget finishes; it does not print every seed.

`--seeds N` runs search seeds `0..N-1`; `--seed K` starts at K instead.
The fixture is built once per case, independently of search seeds. Each budget
and seed invokes a fresh search, with no tree reuse across runs, even when the
configuration enables match-lifecycle tree reuse. In current SO-ISMCTS one
iteration samples one fresh determinization. Existing TOML profiles are accepted
with `--agent-config`; `--iterations` replaces their time/iteration budget for
these measurements. Selector and other agent settings remain unchanged.

## Initial Lost Cities suite

IDs have prefix `lost_cities.`; short CLI names accept underscores or hyphens.

| Short ID | Suite/tag | Diagnostic situation |
| --- | --- | --- |
| `preserve_low_sequence_early` | ordering | Red 4, 6, 8, empty red expedition, 44 deck cards |
| `preserve_low_sequence_late` | ordering | Same focal cards, four deck cards |
| `avoid_large_irreversible_jump` | ordering | Red expedition at 3, own Red 5 and 9 |
| `marginal_expedition_early` | commitment | Red 4 without other red support, 44 deck cards |
| `marginal_expedition_late` | commitment | Same red commitment question, four deck cards |
| `wager_with_support` | wagers | Red wager with Red 4, 6 and 9 |
| `wager_without_support` | wagers | Red wager without numbered red cards |
| `dangerous_discard` | discard | Own Red 9, opponent's red expedition at 3 |
| `safer_discard` | discard | Own Red 9, opponent's red expedition at 10 |
| `discard_pile_vs_deck` | draw | Own Red 4, visible Red 6 discard, deck available |

All fixtures start from fixed seeded deals and use legal actions and explicit
legal chance outcomes. They are reachable, card-conserving positions, not direct
mutations of state fields. The scripted late fixtures deliberately retain focal
cards; public discard history, the rest of the hand and hidden cards also change.
They are interpretable contrasts, not controlled causal estimates of deck size
alone, nor representative samples of normal play. Fixture metadata records every
transition for replay. Draw choices in fixture construction are independent of
search RNG and are not a rollout policy.

The game-specific builder converts the authoritative state to the legitimate
root observation. Only that observation and **all** legal root actions enter the
normal `agent.search` API. Neither authoritative fixture metadata nor a sampled
world is passed to the agent. Rust samples determinizations normally.

## Reading results

Each budget's table shows selected count/share, median visits, median
availability and median Q across seeds. A final matrix compares selection shares
across budgets. Candidate-action lists focus the tables only: search, raw results
and aggregate JSON retain every legal action. The dominant selection is computed
over all actions, so it can fall outside the displayed focus. Focus percentages
need not sum to 100%.

Q is mean terminal utility from the **root player's perspective**, not the actor
at a deeper node. For current Lost Cities this is win/draw/loss utility
`+1 / 0 / -1`, not expedition points or expected score difference. Unvisited edges
have no measured Q: the capture stores `null`, and median Q excludes those seeds.
Availability counts simulations where an edge was legal. At these fixed
SO-ISMCTS roots all legal actions are available at every iteration. An adapter
without availability (for example MCTS) records `null`, rendered as `—`.

Elapsed search seconds include the standalone API call and its diagnostics
conversion. They are incidental measurements, not deterministic results or a
search-speed benchmark. Action choices/statistics are reproducible with the same
fixture, config, seeds and native build; elapsed time and capture timestamps are
not.

## Artifacts

`--output` must name a new directory; existing directories are rejected without
modification. Each capture writes:

- `metadata.json`: schema version, native binary SHA-256, creation time, config,
  budgets/seeds, case descriptions/tags/notes, root observations and administrative
  fixture states plus replay transitions. These full-state records are not agent inputs.
- `runs.jsonl`: one record per case/budget/seed, with all root actions nested inside.
  Includes actual agent config, selected action, root visits, completed iterations,
  determinization count, elapsed time, visits, availability and Q.
- `summary.json`: aggregates for every legal action, with config and root player.
- `report.txt`: compact human tables and the cross-budget selection matrix.

Raw records are flushed after each completed search. Interrupted runs leave
partial raw data and may have no summaries. This first version does not resume or
overwrite captures; use a new directory. Preserve baseline directories and repeat
the same command with a new output path after future agent changes. There is no
automatic cross-capture comparison or tuning.

## Extending

`meeple_bots/probes/core.py` defines `ProbeCase` and `ProbePosition`.
`registry.py` registers concrete cases and filters IDs/games/tags.
`runner.py` handles budgets, seeds and persistence; `report.py` aggregates generic
action metrics. `games/lost_cities.py` owns all card/expedition semantics and labels.

A new game adds builders and labels and registers its cases; runner/reporting
need no strategic game knowledge. The current built-in suite and executable
search integration are Lost Cities/SO-ISMCTS. A different search API can provide
the observation-only adapter documented in `runner.observation_search`, returning
a selected action, root visits, root-action metrics and optional diagnostics.
It must use the game's legitimate agent input, retain all legal actions, start
fresh and normalize Q to root-player utility. Missing availability is supported;
no MCTS agent or perfect-information suite is added here.

Correctness tests cover fixture reachability/card conservation, legal candidates,
observation invariance under hidden changes, determinization compatibility,
observation-only search input, reproducibility, aggregation, CLI and persistence.
Tiny native smoke runs assert legal results and diagnostic shape only. No test
requires Red 4 (or any strategically preferred action) to win a race.
