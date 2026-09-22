# Agents

[Back to the project overview](../README.md)

An agent receives one decision and returns one legal action. Rust agents are generic over the
concrete game, preserving strongly typed actions and static dispatch inside the simulation loop.

## Available agents

| Agent | Purpose | Implementation |
| --- | --- | --- |
| `HumanAgent` | Terminal or application-controlled input | Python boundary |
| `RandomAgent` | Uniform baseline and reproducible opponent | Rust |
| `MctsAgent` | Configurable Monte Carlo Tree Search | Rust |
| [`SoIsmctsAgent`](so-ismcts/README.md) | Observation-only Single-Observer ISMCTS; currently Lost Cities | Rust |

## Random

`RandomAgent` selects uniformly from the current legal actions. It is the simplest baseline for
measuring whether a search agent learns anything useful from its budget.

```python
from meeple_bots import Match, RandomAgent

result = Match(first=RandomAgent(), second=RandomAgent(), seed=42).run()
```

Each player receives an independent random stream derived from the match seed, so the same match
configuration and seed reproduce the same decisions.

## Monte Carlo Tree Search

The Rust `MctsAgent` supports deterministic, perfect-information, two-player, zero-sum games.
The public Python `MctsAgent` configuration also dispatches to public-chance MCTS
for compatible stochastic games (see below); hidden-information games use a separate
[SO-ISMCTS agent](so-ismcts/README.md). By default deterministic MCTS
builds a new tree for every real decision and repeats four steps:

1. Select children with UCT until reaching a node with an unexpanded action.
2. Expand one randomly selected action.
3. Select simulated actions with the configured rollout policy until the game ends or the rollout
   limit is reached.
4. Backpropagate the utility from the root player's perspective.

The real action is the root child with the most visits; mean utility breaks visit ties.

Utilities stored in the tree always use the root player's perspective. During UCT selection, MCTS
reads the active player from `PositionStatus`: it maximizes the stored utility while the root
player keeps making decisions and minimizes it once control passes to the opponent. It does not
alternate maximize and minimize by tree depth. A game may therefore represent one physical turn
as several consecutive engine actions without changing the search semantics. Uniform rollout is
the default; informed policies use the same active-player perspective when ranking actions.

With `tree_reuse=True`, the configured wrapper follows every accepted own and opponent action,
re-roots at the matching expanded child, and discards nodes outside that subtree. This also works
when one player makes several consecutive engine actions. A missing child or any game/state
mismatch resets to a fresh tree rather than risking stale statistics. Cloning an agent for another
match and the explicit match-start lifecycle both discard retained data.

With `transpositions=True`, MCTS uses an exact-state graph instead of a tree. Different action
sequences that produce equal states share visits and utility, while each incoming action keeps its
own exploration count. This backend is independent from `tree_reuse`: it can build a fresh graph
for every decision or retain and prune the reachable graph across match actions.

```python
from meeple_bots import Match, MctsAgent, RandomAgent

agent = MctsAgent(
    iterations=1_000,
    exploration=2.0**0.5,
    rollout_depth=256,
    tree_reuse=True,
    transpositions=True,
)
result = Match(first=agent, second=RandomAgent(), seed=42).run()
```

### Configuration

| Parameter | Default | Meaning |
| --- | --- | --- |
| `selection_policy` | `"uct"` | Tree selection: `uct`, `ucb1_tuned`, or deterministic `uct_rave`. |
| `rave_equivalence` | `1000` | Positive integer k for Classic UCT-RAVE; ignored by other selectors. |
| `iterations` | `1_000` | Exact iterations per decision; excludes `time_budget`. |
| `time_budget` | `None` | Approximate total agent seconds per decision; excludes `iterations`. |
| `exploration` | `sqrt(2)` | UCT balance between utility and less-visited branches. |
| `rollout_depth` | `256` | Soft limit in player decisions; complete the current physical turn before cutoff. |
| `cutoff_evaluator` | `NeutralEvaluator()` | Evaluator used only when a rollout reaches its depth cutoff. |
| `rollout_policy` | `UniformRandom()` | Policy used to select simulated actions outside the tree. |
| `progressive_widening` | `false` | Limit expanded actions in deterministic MCTS. |
| `progressive_widening_k` | `1.5` | Positive widening coefficient. |
| `progressive_widening_alpha` | `0.5` | Widening exponent in `(0, 1]`. |
| `progressive_widening_expansion` | `"random"` | Admission policy: `random` or `rave`. |
| `progressive_bias` | `None` | Optional decaying heuristic prior added to UCT tree selection. |
| `root_diagnostics` | `false` | Record visits, utility, cached heuristic, and bias for expanded root actions. |
| `tree_reuse` | `false` | Retain the reachable subtree across decisions in the same match. |
| `transpositions` | `false` | Merge exactly equal states reached through different action sequences. |

`rollout_depth` counts player decisions/actions, not Chance events. It is always a
**soft limit**: after reaching it, simulate only the remaining decisions of the current
physical turn, resolve mandatory Chance, and evaluate at the first turn boundary.
Terminal positions stop immediately and use terminal utility, even halfway through a turn.
For Connect6, a nominal depth of 15 that ends after the first stone continues to 16;
a depth that already ends after the second stone adds nothing. Single-action turns
retain their previous behavior. No additional configuration flag is required.

Games expose `Game::is_turn_boundary(state)` explicitly; changing `PlayerId` is not the
boundary test. Games with several microactions must override the one-action default.
Mandatory chance preparation before the next player's first decision can preserve a
boundary, but no evaluator receives a pending Chance state. A turn must eventually end
under the rollout policy; extensions are not capped separately and can be longer in
push-your-luck games such as Can't Stop.

The deterministic engine counts rollout decisions after tree traversal. The stochastic
engine retains its existing shared decision horizon for traversal plus rollout. This
change does not alter either counting origin or Selection/Expansion/Backup; it only
completes a pending physical turn before cutoff. Tree reuse and transpositions use
the same cutoff rule as their corresponding non-reusing engine.

Reported `rollout_depth` remains the configured nominal value, not the actual number
of simulated actions. Terminal/cutoff simulation counts classify where the simulation
really stopped. No per-rollout action count is currently exported; do not infer it from
the configured horizon. Actual decisions can be fewer (terminal) or more (turn completion).

The legacy `heuristic=INDEX` argument remains available as shorthand for
`cutoff_evaluator=GameHeuristic(INDEX)`.

More iterations usually improve coverage but also increase response time. Iteration count is not a
portable measure of work: action generation, branching, rollout length, and build profile all
affect the cost of one iteration.

Use an iteration budget for reproducible tests, debugging, and algorithmic comparisons. Use a time
budget to compare agents whose iterations have different costs:

```python
agent = MctsAgent(time_budget=2.0, rollout_depth=32)
```

The deadline is checked between iterations. MCTS always completes the current iteration and always
runs at least one, so elapsed time can exceed the requested duration by one expensive iteration.
Time-budget searches retain seeded randomness but are not exactly reproducible: CPU load and machine
speed change how many iterations finish. Match traces record the actual decision time, completed
iterations, and created nodes.

### UCB1-Tuned

`selection_policy="ucb1_tuned"` enables the variance-aware rule from
[Auer, Cesa-Bianchi and Fischer (2002)](https://doi.org/10.1023/A:1013689704352).
It uses the canonical formula, without an additional exploration multiplier; `exploration` is
used by UCT and Classic UCT-RAVE. For parent visits `N`, action visits `n`, empirical reward mean `m` and
population variance `v` on `[0, 1]`, its score is:

```text
m + sqrt((ln(N) / n) * min(1/4, v + sqrt(2 * ln(N) / n)))
```

Utilities are stored on `[-1, 1]`. Their variance is divided by four, and the bonus is multiplied
by two to express the score on the existing utility scale. The exploitation term is reversed
at opponent nodes; the variance is unchanged. Unvisited actions have infinite priority.
Tuned records squared utilities on tree nodes and both first and second moments on graph edges.
With transpositions, its mean and variance use the same edge samples, rather than combining an
edge count with a shared child's statistics. Root diagnostics and final visit-count tie breaks
also use the edge mean. UCT retains its existing scoring and shared-child semantics. Reuse
preserves the moments; resetting the search clears them. Progressive bias remains additive on
the existing utility scale. UCT does not perform the extra moment accumulation, although the
node/edge storage includes the new fields.

```toml
selection_policy = "ucb1_tuned"
```

The setting works in profiles, inline analysis agents, tournament grids, Python `MctsAgent`, and
GUI selection controls. CLI matches also accept `--mcts-selection-policy ucb1_tuned`.
The original bandit model assumes stationary rewards; MCTS samples evolve with tree search, so
this is an experimental selection alternative, not a guarantee of better play.

### Tree reuse

Tree reuse is deliberately optional. The unwrapped baseline `MctsAgent` keeps the original search
path and supports non-cloneable actions. The enabled reuse wrapper requires the concrete
perfect-information game, state, and action to be cloneable and comparable, but does not require
hashing or a transposition key. Utilities remain in the agent owner's perspective, while MAX/MIN
still follows the real active player stored in each state.

Per-move traces record transition attempts, hits and misses, own/opponent hits, retained root visits and
nodes, pruned nodes, and safe resets. `search_iterations` is always the new work performed for that
decision. On a reused root, `search_nodes` counts newly expanded nodes; retained nodes are reported
separately. Transition metrics are attached to the next decision because the observed action occurs
after the previous decision statistics were captured.

`analyze` benchmarks independent sampled positions. It accepts a profile containing `tree_reuse`,
but those timings do not estimate the accumulated benefit of reuse inside a continuous match. Use
paired tournaments with equal budgets to measure that benefit.

### Rollout policies

`UniformRandom` is the baseline and preserves the original MCTS behavior. It samples every legal
rollout action with equal probability and does not require a game heuristic.

`Greedy` always chooses a best-rated successor. `EpsilonGreedy` does so with probability
`1 - epsilon` and otherwise explores a uniformly random action. Both receive their own evaluator,
independent from the cutoff evaluator. They maximize for the root player and minimize for the
opponent by reading the active player from `PositionStatus`, so consecutive actions by one player
remain correct.

```python
from meeple_bots import EpsilonGreedy, GameHeuristic, MctsAgent, NeutralEvaluator

agent = MctsAgent(
    iterations=10_000,
    rollout_depth=32,
    cutoff_evaluator=NeutralEvaluator(),
    rollout_policy=EpsilonGreedy(
        epsilon=0.1,
        evaluator=GameHeuristic(1),
    ),
)
```

This example is an informed rollout without heuristic cutoff evaluation. The inverse combination
is also valid: `GameHeuristic(1)` as `cutoff_evaluator` with `UniformRandom()` rollouts. Informed
policies evaluate and copy every candidate successor, so their iterations are more expensive than
uniform-random iterations. Compare policies at equal wall-clock decision budgets as well as equal
iteration counts.

In Rust, `MctsAgent<C, P, B>` is generic over cutoff evaluator, rollout policy, and optional
selection bias. `Greedy<E>`
and `EpsilonGreedy<E>` are themselves generic over their rollout evaluator. The runtime catalog
uses configuration enums for the built-in variants, while a Rust integration can inject another
concrete evaluator or policy without duplicating the MCTS loop or paying for dynamic dispatch.

### Progressive Bias

Progressive Bias keeps the rollout policy unchanged and adds
`weight * heuristic(child) / (child_visits + 1)` to UCT selection. The child heuristic is computed
once when the node is expanded and cached. Its sign follows the real active player: nodes controlled
by the root player maximize it and opponent nodes minimize it, including games with consecutive
actions by the same player. Weight zero takes the exact baseline path and does not call the evaluator.

Conditions are checked on the parent state where the action is selected, while the evaluator scores
the resulting child. For SPOTF this allows H0 to guide only Collect without affecting gemstone
decisions:

```python
from meeple_bots import GameHeuristic, MctsAgent, ProgressiveBias, TurnPhaseIs, UniformRandom

agent = MctsAgent(
    time_budget=0.5,
    rollout_depth=130,
    cutoff_evaluator=GameHeuristic(0),
    rollout_policy=UniformRandom(),
    progressive_bias=ProgressiveBias(0.25, GameHeuristic(0), TurnPhaseIs("collect")),
    root_diagnostics=True,
)
```

### Cutoff evaluation

A rollout that reaches a terminal state uses the exact utility: `1.0` for a win, `0.0` for a draw,
and `-1.0` for a loss from the root player's perspective. A rollout that reaches its depth limit
uses the selected game heuristic or `0.0` when no heuristic is configured.

Heuristics belong to games, not to MCTS. This keeps the generic search independent from concrete
rules. Unsupported heuristic indices are rejected before a match starts. A heuristic may expose
named numeric parameters with defaults and bounds. TOML places them in a generic `params` table:

```toml
cutoff_evaluator = { kind = "game_heuristic", index = 0, params = { gemstone_early_bonus = 4.0 } }
```

Omitted parameters use the default declared by the game. Unknown names, invalid values, and
parameters supplied to a heuristic with no configurable parameters are rejected before play.
Tournament grids may put arrays inside `params` just like other grid dimensions.

Boop currently exposes two evaluators:

| Index | Summary |
| --- | --- |
| `0` | Difference between the players' cats currently on the board. |
| `1` | Cat progression, immediate threats, graduations, and safer board presence. |

See the [Boop guide](../games/boop/README.md#mcts-heuristics) for their exact interpretation.

Spirits of the Forest exposes only heuristic `0`, based on category progress that can still reach a
scoring threshold and parameterized early- and mid-game gemstone conservation. See its
[game guide](../games/spirits-of-the-forest/README.md#mcts-heuristic).

### Reusable profiles

CLI batches and matches can load MCTS settings from TOML. Copy
[`configs/mcts/template.toml`](../configs/mcts/template.toml) and keep one profile per configuration
you want to compare:

```toml
name = "example-mcts"
iterations = 500
rollout_depth = 16
exploration = 1.4142135623730951
cutoff_evaluator = { kind = "neutral" }
rollout_policy = { kind = "epsilon_greedy", epsilon = 0.1, evaluator = { kind = "game_heuristic", index = 0 } }
tree_reuse = true
transpositions = true
```

Exactly one of `iterations` or `time_budget` is required, together with `rollout_depth`.
`time_budget` is expressed in seconds and targets total agent cost per decision. MCTS subtracts
its measured lifecycle maintenance since the previous decision and includes preparation in the
selection timer. It reserves the previous decision's finalization cost (diagnostics, action
extraction and temporary tree disposal) before searching. This accounting applies to plain MCTS,
tree reuse and transpositions; iteration budgets remain exact and unchanged.
At least one complete iteration runs even when maintenance consumes the allowance. Uninterruptible
operations, changing finalization costs, external statistics copying and end-of-match cleanup can
still exceed the target. There is no next decision to compensate final cleanup. First decisions
have no finalization estimate. Only this agent's work counts, not opponent thinking or GUI delays.
Recorded `decision_seconds` remains the measured total; equal configured budgets are approximate,
so comparisons must still check observed totals.
See [agent timing in studies](../python/studies.md#1-configure-and-run-the-tournament).
`exploration` defaults to `sqrt(2)`,
`cutoff_evaluator` defaults to neutral, `rollout_policy` defaults to uniform random, and
`tree_reuse` and `transpositions` default to false. Evaluator
kinds currently supported by the catalog are `neutral` and `game_heuristic`. Rollout policy kinds
are `uniform_random`, `greedy`, `epsilon_greedy`, `conditional`, and `mast`. The legacy `use_heuristic`,
`heuristic_index`, `rollout_heuristic_index`, and flat rollout fields remain accepted.

MAST (Move-Average Sampling Technique) learns a mean utility for each exact action and
acting player across simulations within one decision:

```toml
rollout_policy = { kind = "mast", epsilon = 0.1 }
```

This implementation uses epsilon-greedy sampling: choose uniformly with probability
`epsilon`, otherwise choose a legal action with the highest learned mean, breaking ties
uniformly. Unseen actions have value zero. `epsilon` must be finite and in `[0, 1]`;
Python and TOML default it to `0.1`. MAST has no rollout evaluator of its own.
Every action occurrence in tree selection, expansion, and rollout receives the final
simulation utility, with its sign adjusted for the acting player. Truncated simulations
use the configured cutoff evaluator, so the learned values can inherit its biases.
Statistics reset at every decision, including with tree reuse or transpositions enabled,
and are never shared between agents or matches. A standalone Rust `select_action` call
has no search memory and therefore samples uniformly.

The configurable Rust policy requires actions implementing `Clone + Eq + Hash` to copy
and identify table keys. Concrete `UniformRandom`, `Greedy`, and `EpsilonGreedy` policies
retain their previous action requirements. In SPOTF, conditional policies can include
MAST in either branch; both branches share the same action table and the full trajectory
trains it. In Boop, action identity includes piece, position, and resolution; SPOTF uses
the complete action, including positions and any sacrifice. No symmetry or feature
grouping is applied.

For an initial study, compare uniform rollouts and MAST with identical search settings
and paired seats. Measure elapsed time as well as wins, then confirm results with equal
time budgets. SPOTF's shrinking tile supply makes complete rollouts useful for a separate
experiment without cutoff feedback; Boop with a short horizon tests the combination of
MAST and its cutoff heuristic. Neither setup guarantees stronger play.

```bash
meeple-bots match --game boop --first mcts --second random \
  --first-mcts-config configs/mcts/template.toml --seed 42
```

### Choosing a budget

Use `evaluate_game` or `meeple-bots analyze` to estimate structural scale and local iteration cost:

```bash
meeple-bots analyze --game boop --samples 128 --max-depth 256 --seed 42
```

The recommendation is a starting point for experiments, not an Elo or strength guarantee. Compare
candidate agents at equal wall-clock time whenever possible. The
[evaluation guide](../crates/evaluation/README.md) explains the formula and limitations.

### Current limits and future work

The deterministic MCTS backend does not handle chance or hidden information.
Public-chance MCTS and observation-only SO-ISMCTS are separate implementations.
Parallel search and learned policies/values are not provided. Tree reuse and exact-state transpositions are optional for
compatible perfect-information games. The possible development stages are recorded in the
[MCTS roadmap](MCTS_ROADMAP.md) and its
[Spanish translation](MCTS_ROADMAP.es.md).

## Human

`HumanAgent` lives at the Python boundary. With no selector it prompts in the terminal; with a
selector callback it can obtain actions from another interface. See
[Human-controlled matches](../python/usage.md#human-controlled-matches).

## Public-chance MCTS

`StochasticMctsAgent` accepts two-player, perfect-information games with public chance events.
It samples those events with the search RNG, independently of actual match outcomes. Its decision
edges accumulate results across chance outcomes, with distinct successor states as continuations.
UCT and UCB1-Tuned therefore optimize player actions, never chance outcomes. The search supports
uniform, informed, conditional and MAST rollouts, configurable cutoff evaluation and progressive
bias. Informed rollouts evaluate independently sampled candidate successors; actual rollout
transitions sample again. Progressive bias averages sampled successor evaluations per action.

`ReusableStochasticMctsAgent<G, C, P, B>` optionally retains the observed subtree after player and
chance events, and independently supports exact-state transpositions. Statistics remain on
incoming decision edges, including UCB1-Tuned moments. Cycles end tree traversal and continue
with the remaining rollout horizon; depth counts player actions rather than chance events.
MAST memory is fresh each decision even with a retained tree. Deterministic search retains its
existing behavior.
See [Can't Stop integration](../games/cant-stop/README.md#engine-integration) for supported options.

## Classic UCT-RAVE (deterministic games)

Normal Q asks: **How good was action A when actually chosen from state S?**
AMAF asks: **When A appeared at or after S in a simulation for the player active
at S, how did that simulation end?** RAVE blends these estimates early and
increasingly trusts direct evidence as the state accumulates real visits:

```text
beta = sqrt(k / (3 * N(s) + k))
Q_RAVE = (1 - beta) * Q + beta * Q_AMAF
score = oriented(Q_RAVE) + exploration * sqrt(ln(N(s)) / N(s,a))
```

Both means use root-player utility. `oriented` reverses the blended mean when
selecting for the opponent, exactly as ordinary UCT does. Without AMAF samples,
the effective beta is zero. Real edge visits, never AMAF visits, control the
exploration denominator and whether an action has been explored.

AMAF tables belong to each state's outgoing legal actions, including unexpanded
actions. Every visited state receives at most one AMAF credit per matching action
per simulation. Only actions in its trajectory suffix by its actual active
`PlayerId` count; the direct action counts too. Opponent actions and actions not
legal at that state do not count. Selection, expansion and rollout all contribute
to the trajectory, including the extra microactions needed to complete a turn.
Thus Connect6 credits both consecutive placements by one player without assuming
ply parity or limiting AMAF to that physical turn.

RAVE changes selection and adds a backup. It does not change expansion, rollout
policy, MAST memory or final root choice (real visits, then real mean utility).
Progressive Bias remains an additional term with its existing decay. AMAF is
state-local, whereas MAST uses cross-state move statistics to guide rollouts.
The transposition backend stores direct RAVE Q on outgoing edges (as it already
does for UCB1-Tuned), alongside normal node statistics. Ordinary UCT retains its
existing transposition scoring unchanged. Exact shared states share a single
AMAF table; a repeated node receives one AMAF update per simulation. Rerooting
and graph compaction move the whole table with the retained node.

```toml
iterations = 1000
rollout_depth = 16
selection_policy = "uct_rave"
rave_equivalence = 1000
exploration = 1.4142135623730951
rollout_policy = { kind = "uniform_random" }
cutoff_evaluator = { kind = "neutral" }
tree_reuse = false
transpositions = false
```

`rave_equivalence` is a positive u32 integer. Python/config defaults to 1000, a
starting value, not a universally optimal setting. It is ignored by other selectors.
Tournaments accept a grid such as `rave_equivalence = [100, 1000, 10000]`;
serialized RAVE profiles and extracted agent tables retain the chosen value.

```python
MctsAgent(iterations=1000, rollout_depth=16,
          selection_policy="uct_rave", rave_equivalence=1000)
```

```bash
meeple-bots match --game connect6 --game-param board_size=11 \
  --first mcts --second random --mcts-iterations 1000 \
  --mcts-selection-policy uct_rave --mcts-rave-equivalence 1000
```

Rust uses `SelectionPolicy::UctRave { rave_equivalence: DEFAULT_RAVE_EQUIVALENCE }`.
Use the existing typed `TreeReuseMctsAgent::new(inner, false)` or
`TranspositionMctsAgent::new(inner, false, false)` adapter even if reuse is off;
these provide action cloning/equality for trajectory identity. The catalog does
this automatically. Basic `MctsAgent` still supports non-cloneable actions for
UCT/UCB1-Tuned and reports an explicit error if asked for RAVE without that adapter.

AMAF is most plausible when action ordering is partly interchangeable, making
Connect6 an interesting experiment. RAVE adds trajectory recording and local
matching work; it need not improve strength or equal-time throughput in every game.
This implementation records each action once, reuses trajectory buffers and avoids
per-node suffix allocations; lookup uses equality scans rather than assuming a
hashable Action. No stochastic RAVE is implemented: stochastic agents/catalog
reject UCT-RAVE explicitly. No GRAVE, PoolRAVE or game-specific extensions are used.

### Progressive Widening (deterministic MCTS)

Progressive Widening (PW) limits admission into the persistent tree, independently
of UCT, UCB1-Tuned or UCT-RAVE selection. It is disabled by default.

```toml
progressive_widening = true
progressive_widening_k = 1.5
progressive_widening_alpha = 0.5
```

The exact limit is `min(legal_count, max(1, floor(k * N^alpha)))`. `N` is the
state node's completed normal visits **before** the current simulation's backup.
At `N=0` the first child can open; a terminal node with no legal actions has limit
zero. `k` must be finite and positive; `alpha` must be finite and in `(0, 1]`.
The defaults are experimental starting points, not universally optimal values.
This follows the power-law admission convention used in
[progressive widening literature](https://ntrs.nasa.gov/api/citations/20160005033/downloads/20160005033.pdf);
this implementation explicitly chooses floor rounding and a one-child minimum.

If pending actions remain and the active count is below the limit, expand exactly
one pending action using the expansion policy (uniform random by default).
Otherwise select among active children.
For example, with `k=1, alpha=0.5`, the visit with `N=4` can open the second action;
with `N=3` the search must revisit the first. The visit with `N=25` can open the
fifth. No new turn/depth semantics are introduced.

- **Legal:** an action descriptor generated by the game.
- **Active:** removed from pending actions and admitted as a child/edge.
- **Visited:** traversed and backed up with a normal sample. Admission and its
  first traversal happen in the same expansion step; backup supplies its visit.
- **AMAF-sampled:** appeared for the same player in a simulation suffix. This can
  happen while an action is still inactive; it neither activates the action nor
  supplies a normal visit.

All legal descriptors remain lightweight pending/AMAF records. Child nodes and,
with transpositions, child states are materialized only upon expansion. Reuse
preserves pending actions, active children, visits and AMAF; transpositions share
these properties at the exact state node. Root choice still uses normal visits/Q.
Rollouts remain unrestricted and retain end-of-turn cutoff completion.

Rust search/decision stats expose `root_expansion = (legal, active, limit)` after
search. Python `benchmark_mcts_agent(...).position_timings` exposes the same tuple,
plus actual iterations and elapsed time. Coverage is `active / legal`.
With `root_diagnostics=True`, existing `root_visits` provides visits for active
children (including enough information for their median), and match root-action
records contain only admitted actions. Even without PW, a small iteration budget
may leave legal actions unexpanded: active does not necessarily equal legal.

The same fields work in Python `MctsAgent`, profiles, inline profiles and tournament
grids; they are preserved in study candidate serialization and extraction's
`agents.csv`. Incremental studies tune PW only when `--pw-search` (or
`--all-search`) is requested; `--baseline` preserves the starting agent. See
[study stages](../python/studies.md#incremental-mcts-study). Compare identical agents
with `progressive_widening = [false, true]` and the same `time_budget` in a tournament.
A CLI example is:

```sh
meeple-bots match --game connect6 --game-param board_size=13 \
  --first mcts --second random --mcts-iterations 1000 \
  --mcts-progressive-widening --mcts-progressive-widening-k 1.5 \
  --mcts-progressive-widening-alpha 0.5
```

Stochastic games reject PW explicitly. Stochastic/chance widening and
heuristic-guided admission are deliberately left for later.

#### RAVE-guided admission

```toml
progressive_widening = true
progressive_widening_k = 1.5
progressive_widening_alpha = 0.5
progressive_widening_expansion = "rave" # default: "random"
```

PW still decides **when** another action can enter, with the same limit and
rounding. `rave` decides **which** pending action enters: maximize
`Q_AMAF = amaf_total_utility / amaf_visits` from the active player's perspective
(the root-oriented stored utility is negated at opponent nodes). Only node-local,
still-unexpanded legal actions with AMAF samples are considered. Unsampled
moves are ignored even when all sampled means are negative; they have no fake
zero prior. Exact best-mean ties use the agent RNG uniformly. If no pending
action has samples, fall back to the existing uniform random admission.

This differs from UCT-RAVE **selection**, which blends normal Q and AMAF Q among
already admitted actions. Admission uses neither that blend nor an exploration
bonus. AMAF collection is also enabled for guided admission with plain UCT or
UCB1-Tuned; their selection scores remain unchanged. No new rollout, backup,
root-choice or turn-boundary semantics are introduced. Existing reuse and graph
compaction preserve the state-local AMAF/pending records used by this policy.
In Rust the typed tree/graph adapter supplies action identity, as with UCT-RAVE.

`rave` expansion requires PW enabled; Python/native config validation rejects
`progressive_widening=false` combined with `expansion="rave"`. With PW absent,
Rust never runs the admission policy. CLI: `--mcts-progressive-widening-expansion
rave` together with `--mcts-progressive-widening`.

Rust `MctsSearchStats.widening` counts admissions during the current search:
`expansions_total`, `expansions_random`, `expansions_rave_guided`, and
`rave_fallbacks_no_amaf` (fallbacks are included in random). Generic decision
stats and Python benchmark position timings expose the same counters as
`widening_expansions = (total, random, rave_guided, no_amaf_fallbacks)`.
These distinguish configured guidance from actual use of sampled AMAF evidence.
Profiles, study candidates and extracted agent configuration retain the policy.
The study PW family screen compares no PW, random PW and RAVE-guided PW before
rejecting the family. Guided PW does not depend on random PW winning: surviving
policies refine k/alpha independently with separate histories. AMAF collection can
be enabled by guided admission even with UCT/UCB1-Tuned selection. Limited budgets
reduce parameter breadth before dropping an applicable expansion policy.

Local `--tune widening-expansion` requires PW enabled and directly compares random
and RAVE-guided admission with k, alpha and all other agent fields frozen. See the
[study guide](../python/studies.md#incremental-mcts-study).
