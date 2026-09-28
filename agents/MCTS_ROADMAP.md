# Agent and search development roadmap

[**English**](MCTS_ROADMAP.md) | [Español](MCTS_ROADMAP.es.md)

The historical filename is retained so existing links keep working. Meeple Bots aims to build
strong, generic agents for modern board games, including public-chance and imperfect-information
games, **and to extract useful strategic information from their decisions**. This is a research
roadmap, not a commitment to implement every algorithm. Agent strength, generic ownership,
reproducibility, and interpretable strategic evidence matter more than collecting techniques.

## Current foundation

The following capabilities are implemented, although their strategic benefit depends on the game
and measured budget:

| Area | Current capability |
| --- | --- |
| Measurement | Search diagnostics and time/iteration budgets; Analyze, Probe, Study, seeded matches and paired experiments. |
| Generic search | UCT and UCB1-Tuned shared selection; configurable rollout and cutoff evaluation, heuristic/epsilon rollouts, MAST, Progressive Bias, RAVE, Progressive Widening and RAVE-guided admission. |
| Search memory | Optional tree reuse and exact-state transpositions for compatible perfect-information search; observation-path reuse for Lost Cities SO-ISMCTS. |
| Game families | Deterministic perfect-information MCTS, public-chance MCTS for supported games, and observation-based SO-ISMCTS for Lost Cities. |
| Evaluation ownership | `StateEvaluator` is defined in Rust core and consumed by MCTS; game heuristics remain game-owned. |

The core mechanisms from the old modular-search, informed-search, reuse, chance and observation
stages are therefore **implemented foundations**, not proposed first steps. Some old completion signals,
including universal equal-time strength gains, exhaustive reference benchmarks, configurable
memory limits and a known-equilibrium hidden-information test bed, have **not** been established.
The [agent guide](README.md) describes supported mechanisms and configurations; the
[Probe](../python/probes.md), [Analyze](../crates/evaluation/README.md) and
[Study](../python/studies.md) guides describe the measurement tools. There is currently no
learned value or policy model, PUCT, Alpha-Beta, AlphaZero training loop or CFR agent.

## Architectural and experimental rules

- Games own rules, legality, chance, terminal utility and game-specific features. Generic Rust
  searches consume game contracts and optional evaluation; they must not branch on game names.
  Catalog/adapters compose supported combinations. Rust owns rules, search and simulation; Python
  owns experiment orchestration, training, dataset publication, analysis and reporting. Inference
  backend choice remains open; a Python callback on every Rust leaf is not a default design.
- Keep deterministic perfect-information, public-chance and imperfect-information search as
  distinct semantic families. An algorithm declares the capabilities it needs; there is no
  universal search or requirement that every game support every agent. Chance outcomes are not
  player actions. Random, MCTS, stochastic MCTS, SO-ISMCTS and future algorithms remain independent
  consumers of small shared contracts; neural infrastructure is optional.
- Orient values by the explicitly requested player and select using the **actual actor**, not
  search-depth parity. Consecutive actions by the same player occur in current games. A model may
  advise search, but never replaces authoritative legality, transitions or terminal outcomes.
- The general `Agent` lifecycle is trusted and may receive authoritative state. Lost Cities uses
  a trusted adapter to filter events before they reach observation-only SO search. Future hidden-
  information evaluators and agents need the same explicit information boundary; the generic
  lifecycle alone is not a sandbox.
- Evaluate with fresh match seeds, balanced seats, saved configurations and model/data provenance.
  Fixed iterations help diagnose reproducibility; **equal measured wall-clock** is the primary
  fairness comparison for algorithms with different per-iteration cost. Use held-out data,
  ablations and behavioral probes. Do not turn an unforced strategic preference into a CI pass/fail
  assertion; reserve correctness tests for invariants and mathematically forced outcomes.

## Next: diagnose strategic bottlenecks

**Goal and reason.** Establish why current searches make questionable decisions before building
a large learning system. Lost Cities SO-ISMCTS is the first case because reported apparently
obvious mistakes give a concrete hypothesis to test, not a proven diagnosis.

**Possible work.** Use existing fixed-position Probes across multiple iteration budgets and
search seeds. Track the selected action, root visits, root-player Q, availability, stability and
convergence. Separate insufficient budget/sample efficiency, weak rollout or cutoff signal, and
structural limitations of determinization/SO-ISMCTS. Repeat decisive observations with fresh
positions and measured time. Probe captures describe behavior; tournaments and studies measure
strength.

**Risk and completion signal.** More iterations can reinforce a biased estimate; a stable choice
is not automatically sound. Finish with reproducible captures and a clear classification of
whether errors disappear, persist or remain uncertain as budget grows. No strategic CI oracle is
required.

## Learning I: scalar value before policy

**Goal and reason.** Test whether stronger strategic evaluation changes decisions in agents that
already search. A value function is the quantity estimated; a neural network is one possible
approximator. Compare a neutral baseline, a handcrafted signal where available, and a learned
value. A linear/statistical model may be a useful control; a small MLP is a concrete first neural
experiment. GPU, transformers, batching, policy training and an AlphaZero loop are not prerequisites.

**Possible work.** Keep the scalar evaluation contract generic (`StateEvaluator` already has a
neutral Rust owner). Put game-specific feature encoding and model adaptation at a game-aware
boundary, not in generic search. There is no universal board tensor. A first dataset can pair a
legitimate decision observation/state, acting-player perspective and final outcome. For Lost
Cities, use only the acting player's observation: own hand, public expeditions/discards/scores,
remaining deck count and legitimately observed phase/history may be encoded; the opponent's
hidden hand and true future deck order may not. The target estimates expected outcome conditional
on available information and the data policy, not hidden truth. Hidden outcomes make targets noisy.
Keep terminal utility authoritative. SO-ISMCTS has no learned-value integration today; its future
observation-safe evaluation path must be established explicitly, not inferred from MCTS's trait.
Version the encoder/model and record provenance; avoid silently reusing search statistics under a
changed model.

**Risks.** Weak self-play can copy its own blind spots; correlated train/test games can exaggerate
progress; hidden-state leakage can make an invalid model look excellent; prediction accuracy need
not improve play; inference overhead can erase search gains. Separate train/validation/test games,
use fresh evaluation seeds, and report calibration and equal-time play.

**Completion signal.** The encoder passes information-boundary checks; held-out prediction and
calibration are reported; an appropriate search path consumes the value without a game-specific
branch; baseline/handcrafted/learned comparisons use equal measured wall-clock and seat balance;
and the Lost Cities probes are repeated. A well-measured negative result is valid research.

**Decision point.** If stronger value substantially corrects the observed SO choices, continue
toward policy-guided observation-safe search. If mistakes persist at large budgets despite better
value, prioritize information-set/game-theoretic or belief-aware methods. A learned value cannot
by itself remove strategy fusion, determinization bias, opponent-modelling limits or belief mismatch.

## Learning II: a legal-action policy

**Goal and reason.** Learn `P(action | information/state)` to improve action ordering, expansion
and sample efficiency, and to expose strategic preferences. Wide phases such as SPOTF Gem
placement motivate this without teaching generic search what a Gem is.

**Possible work.** Associate scores or probabilities with the game's supplied **legal actions**;
a conceptual output is `[(action, score), ...]`. A fixed global action vector is optional and
may be inappropriate for dynamic or phase-dependent actions. Game-owned encoders/adapters can
supply stable indexing when useful. For hidden information, policy input is a legitimate
observation. Keep policy advice separate from game legality and from scalar value.

**Risk and completion signal.** Action identity, normalization, masking and model version must
remain stable across training and inference. Show legal-action association, held-out policy
measurements and equal-time agent comparisons without a concrete-game branch in generic search.

## Guided search: PUCT after the policy boundary

**Goal and reason.** Test whether legal-action priors plus value improve search, then export
root visits as an analyzable policy signal.

**Possible work.** Add decision-edge priors, generic normalization/validation, PUCT selection and
root visit distributions linked to legal actions and model context. Start on Tic-Tac-Toe or
Connect Four with simple priors/value. Policy-guided Progressive Widening can later choose which
pending action enters the tree; RAVE-guided widening already exists but its AMAF scores are **not**
learned policy priors. RAVE/MAST integration, stochastic or SO-PUCT, and batched inference are not
requirements for a first experiment.

**Risk and completion signal.** Preserve actual-actor orientation and distinguish decision edges
from chance outcomes; do not mistake an admitted-child diagnostic for a full legal-action policy.
Show legal actions, validated priors, reproducible root visit output and equal-time comparisons.

## Self-play branch: AlphaZero-style experiments

**Goal and reason.** Study policy/value improvement with exact Rust simulators on games where
classic deterministic, sequential, perfect-information assumptions fit. Tic-Tac-Toe and Connect
Four are the first pipeline checks; Connect6, Boop and SPOTF need explicit encodings and action
associations, including same-player/phase behavior. This branch is not every game's destination.

**Possible work.** Combine policy/value inference, PUCT, root exploration noise, temperature-based
self-play action sampling, legal root visit targets, player-oriented final outcomes, frozen model
versions within games and an evaluation gate. Rust exposes search decisions and exact simulation;
Python publishes datasets, trains/selects models and orchestrates experiments. Chance games such
as Can't Stop and Splendor need an explicit chance-aware extension; ordinary PUCT does not apply
unchanged. Lost Cities needs observation/belief-aware treatment, not a classic omniscient state
network.

**Risk and completion signal.** Self-play can reinforce blind spots and model changes can
invalidate reused search data. Demonstrate a reproducible end-to-end loop, legal and versioned
training records, held-out/equal-time evaluation and interpretable decision outputs. Winning a
single match is not sufficient evidence.

## Imperfect-information research branch

**Goal and reason.** Investigate intentional mixed strategies and information-set reasoning when
SO-ISMCTS errors appear structural. Win rate against one opponent alone does not establish a
robust hidden-information strategy; exploitability or distance to a known equilibrium can be more
informative on suitable small games. CFR-family work is relevant to both agent strength and
strategy extraction, without presuming it beats SO-ISMCTS.

**Possible work.** Reposition Kuhn Poker as a future compact known-equilibrium test bed for CFR
and mixed strategies, then study CFR variants/Deep CFR or neural regret and public-belief methods.
Compare with SO-ISMCTS on richer hidden-information problems only after validating information
boundaries and measurement. ReBeL-like approaches are a later belief-aware direction, not an alias
for SO-ISMCTS or AlphaZero.

**Risk and completion signal.** Distinguish strategy fusion, determinization bias, opponent
modelling and belief mismatch; a learned value or policy alone fixes none automatically. A small
reference experiment should report reproducible strategy/exploitability-oriented evidence before
scaling to a modern game.

## Optional reference branches and research horizon

- **Alpha-Beta:** Optional educational/reference search for deterministic, perfect-information,
  two-player zero-sum games. Compare systematic minimax with MCTS, solve small positions and test
  the shared scalar evaluator. Actor identity, not ply parity, must control maximizing/minimizing.
  It is not on the critical path to learned agents or a shortcut into hidden-information games.
- **Expectiminimax:** Optional public-chance reference for small games; stochastic MCTS already
  supplies the main current chance-search path.
- **Advanced work:** Deep CFR, public-belief/ReBeL-like methods, Student of Games ideas, Gumbel
  search, sampled-action search, batched inference and parallel search remain research options.
  MuZero-style learned dynamics are low priority while exact Rust simulation is available; revisit
  only when rules are unavailable or simulation becomes prohibitively expensive.

## Strategy extraction and interpretability

This is a cross-cutting deliverable, not a side effect of higher win rate. Preserve root visits,
Q/value estimates, legal-action distributions, SO availability where relevant, phase/context,
legitimate observation features, model/encoder version and stability across seeds. Ask where
value swings occur, which actions remain robust across determinizations, which motifs correlate
with value, where a policy concentrates or mixes, which phases cost search, and where agents
disagree. Analyze uncertainty separately from strong preference; a network does not explain
itself automatically.

Value trajectories and action-conditioned comparisons are useful, but `V(after) - V(before)` is
not automatically an action's strategic value: perspective, chance, hidden information and the
opponent's response can change the meaning. Prefer controlled root/counterfactual comparisons and
record the assumptions behind an interpretation. Probes describe decisions, held-out datasets
assess models, and paired studies/tournaments assess playing strength.

## Suggested implementation order

1. Use the existing Probe/Analyze/Study infrastructure to diagnose Lost Cities SO decisions across
   fixed positions, budgets and seeds; record measured time.
2. Use the already core-owned scalar evaluator contract; establish an observation-safe or
   state-specific game encoder and a small learned value model.
3. Integrate value into a suitable search path without hidden-state leakage; compare neutral,
   handcrafted (where available) and learned signals at equal wall-clock with fresh seeds.
4. Re-run behavioral probes and let evidence choose between policy-guided SO experiments and an
   information-set/belief-aware research branch.
5. Add a learned legal-action policy, then generic decision priors, PUCT and root visit-policy
   export on a simple perfect-information game.
6. Validate an AlphaZero-style self-play loop on that suitable game, then extend to more complex
   compatible games with game-owned adapters.
7. Build a compact CFR/equilibrium baseline and compare hidden-information methods when the
   diagnostic evidence warrants it; pursue advanced techniques only for concrete limitations.

These steps are conditional, not a single compulsory algorithm hierarchy. Alpha-Beta and
expectiminimax remain optional reference branches.

## Priority overview

| Direction | Status | Near-term priority | Key dependency |
| --- | --- | --- | --- |
| Behavioral diagnosis and strategy extraction | Next / cross-cutting | Very high | Existing probes and reproducible positions |
| Observation-safe encoding and learned value | Next | Very high | Legitimate input, held-out data, neutral evaluator |
| Learned legal-action policy | After value | High | Action association and model provenance |
| PUCT and policy-guided widening | After policy | High | Priors and legal root-policy output |
| AlphaZero-style self-play | Suitable-game branch | Medium/high after foundations | PUCT, versioned data and evaluation gate |
| CFR and belief-aware information-set methods | Research branch | High when SO errors persist | Compact reference and game-theoretic metrics |
| Alpha-Beta / expectiminimax | Optional reference | Optional / educational | Eligible game semantics |
| Batched/parallel search and advanced methods | Research horizon | Evidence-driven | Measured bottleneck |
| MuZero-style dynamics | Research horizon | Low with exact simulators | Unavailable or costly exact rules |

## Evaluation checklist

For each experiment, ask whether legality, actor orientation and information boundaries hold;
whether configuration, source/model/encoder and dataset provenance are recorded; whether
train/validation/test games and match seeds are separated; whether seats and opponents are
balanced; whether held-out calibration, equal-wall-clock strength and inference overhead are
reported; whether a baseline and ablation isolate the new signal; and whether outputs can support
careful strategy analysis. Iterations remain a useful diagnostic, not the main fairness measure
across methods with different costs.

## Current recommendation

First determine whether current strategic mistakes, especially in Lost Cities SO-ISMCTS, arise
from inadequate budget/sample efficiency, weak strategic evaluation, or deeper information-set
limitations. Use fixed-position probes across budgets and seeds. Then train a small value model
on legitimate observation/state encodings, integrate it through a generic scalar contract and
compare it against unchanged baselines at equal wall-clock. If better value resolves the observed
errors, proceed toward learned policy and guided search; if they persist, prioritize CFR or
belief-aware methods rather than merely scaling the network or search budget.
