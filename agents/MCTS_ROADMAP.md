# MCTS development roadmap

[**English**](MCTS_ROADMAP.md) | [Español](MCTS_ROADMAP.es.md)

This document records possible directions for evolving the Meeple Bots search agents. It is a
horizon rather than a committed schedule: each stage should be justified with measurements before
becoming part of the project.

The roadmap deliberately progresses from observable, classical MCTS improvements to stochastic
and imperfect-information search, and only then to learned policies and value functions. This keeps
each new concept small enough to understand and test independently.

## Guiding principles

- Keep the game rules independent from search algorithms.
- Never expose authoritative hidden state to an imperfect-information agent.
- Compare agents using both playing strength and wall-clock cost. An iteration does not have the
  same cost across games or search variants.
- Add one search idea at a time and preserve a simple baseline for comparison.
- Prefer explicit capabilities over runtime assumptions. Unsupported combinations should fail
  clearly, ideally at compile time.
- Make deterministic experiments reproducible from their seed and configuration.
- Treat advanced algorithms as optional agents or policies rather than continually complicating
  the basic MCTS implementation.

## Stage 0: establish reliable baselines

### Goal

Make the current MCTS behavior measurable before changing its algorithm.

### Possible work

- Record the number of created nodes, maximum tree depth, rollout actions, terminal rollouts, and
  heuristic cutoffs for each decision.
- Record root action visits, mean utilities, and the selected action.
- Support search budgets expressed as iterations and as wall-clock time.
- Add benchmark configurations for Tic-Tac-Toe, Connect Four, and Boop.
- Compare agents by win rate, seat, decision time, nodes per second, and memory use.
- Validate Tic-Tac-Toe decisions against its solved game tree.

### Completion signal

Experiments can explain not only which agent wins, but how it spends its search budget.

## Stage 1: make classical MCTS policies modular

### Goal

Create explicit extension points without changing the default UCT behavior.

### Possible work

- Separate the tree-selection policy from node storage.
- Separate the rollout policy from the cutoff evaluator.
- Separate the final root action-selection rule from UCT selection.
- Keep uniform random rollout, neutral cutoff evaluation, and most-visited final selection as the
  baseline implementations.
- Allow policies to inspect only the game capabilities they genuinely require.

### Concepts to explore

- `TreePolicy`: selects a child during tree traversal.
- `ExpansionPolicy`: selects which unexpanded action to add.
- `RolloutPolicy`: selects simulated actions outside the tree.
- `CutoffEvaluator`: estimates non-terminal states at the rollout limit.
- `RootSelectionPolicy`: chooses the real action after search.

### Completion signal

Alternative policies can be tested without duplicating the complete MCTS loop.

## Stage 2: improve simulation quality

### Goal

Obtain more useful information from each iteration without machine learning.

### Possible work

- Add heuristic or rule-based rollout policies.
- Support epsilon mixtures: usually choose a heuristic action, but sometimes choose randomly.
- Add progressive bias so game knowledge influences early selection and fades as visits grow.
- Compare full rollouts with shorter heuristic-evaluated rollouts.
- Investigate implicit minimax backups for tactical games.
- Tune exploration independently for each game and search policy.

### Risks

- A biased rollout policy may repeatedly miss unusual winning lines.
- Strong heuristics can hide bugs by making small test agents appear competent.
- A more expensive iteration may perform worse under the same time budget despite needing fewer
  iterations.

### Completion signal

At least one informed policy outperforms uniform random rollout at equal wall-clock cost, and the
baseline remains available.

## Stage 3: reuse and share search information

### Goal

Avoid discarding valid work within a match and across equivalent search paths.

### Possible work

- Preserve the selected subtree after a real action.
- Re-root after observing the opponent's actual action.
- Verify that a reused root matches the current search context; reset safely when it does not.
- Define an explicit search-session lifecycle so agents cannot accidentally reuse a tree in a new
  match.
- Add transposition tables for paths that reach the same complete state.
- Add configurable memory limits and pruning policies.

### Design constraints

- Tree reuse must remain optional because not every search variant attaches statistics to exact
  states.
- Reuse for perfect-information games must not force `State: Eq + Hash` on unrelated agents unless
  the benefit justifies the restriction.
- Imperfect-information agents must key reuse by legal observations, information sets, or public
  histories, never by hidden authoritative state.
- Changing the utility perspective, rules, rollout policy, or cutoff evaluator should invalidate
  incompatible statistics.

### Completion signal

Reused search produces the same legal decisions as a fresh search, resets safely on mismatch, and
provides a measured strength or latency improvement.

## Stage 4: support stochastic games

### Goal

Represent random environment transitions explicitly and search through them correctly.

### Possible work

- Distinguish player-decision, chance, and terminal nodes.
- Extend the game contract to enumerate chance outcomes with probabilities when practical.
- Also support sampling a chance outcome from a generative model.
- Validate that outcome probabilities are finite, non-negative, and normalized.
- Back up expected values through chance nodes rather than maximizing or minimizing them.
- Add progressive widening or outcome sampling when the chance space is very large.
- Keep environment randomness separate from agent randomness so matches remain reproducible.
- Add a small dice game as a focused stochastic-search test bed.

### Questions to answer

- Should chance be resolved by the simulation runner or through a game-owned transition API?
- When should search enumerate every result, and when should it sample results?
- How are chance outcomes represented in action traces and replay analysis?

### Completion signal

A stochastic reference game has reproducible matches, statistically correct outcome frequencies,
and MCTS decisions that agree with exact expectations on small positions.

## Stage 5: introduce imperfect-information search

### Goal

Let agents reason from observations without gaining access to hidden state.

### Possible work

- Define public observations, private observations, and observable action histories.
- Define how an agent samples a complete state compatible with its information.
- Add belief or determinization validation to prevent impossible sampled states.
- Implement plain determinization as a deliberately simple baseline.
- Implement Information Set MCTS as a separate agent or search backend.
- Store statistics by information set or observable history rather than complete hidden state.
- Add Kuhn Poker as a compact test game containing chance, hidden cards, bluffing, and a known
  equilibrium.
- Measure exploitability or distance from a known equilibrium in addition to win rate.

### Concepts to study

- Information set: states a player cannot distinguish from one another.
- Belief state: a probability distribution over possible underlying states.
- Strategy fusion: incorrectly choosing different actions in hidden states the player cannot
  distinguish.
- Non-locality: the value of a decision may depend on strategy choices elsewhere in the game.
- Mixed strategy: intentionally randomizing actions to avoid exploitation.

### Risks

- A determinized perfect-information search can accidentally act as if it knows hidden facts.
- Win rate against one opponent does not demonstrate a sound hidden-information strategy.
- Reusing statistics collected under an old belief distribution can bias a later search.

### Completion signal

Tests demonstrate that the agent cannot inspect hidden state, impossible determinizations are
rejected, and the agent approaches a sensible strategy in a game with a known solution.

## Stage 6: add policy-guided tree search

### Goal

Guide expansion with prior action preferences while retaining exact game simulation.

### Possible work

- Introduce an action-prior interface that returns a normalized distribution over legal actions.
- Implement PUCT selection alongside UCT.
- Start with uniform priors to establish equivalence with the baseline.
- Add hand-written priors for one game.
- Train tabular or simple linear priors from recorded games.
- Use root exploration noise and action-selection temperature only in self-play configurations.
- Store visit distributions as improved policy targets.

### Learning progression

1. Uniform priors.
2. Hand-written priors.
3. Frequency tables learned from MCTS games.
4. A small statistical model.
5. A neural policy if simpler approaches have reached their limits.

### Completion signal

Guided search improves playing strength or reduces the simulations required to reach the baseline's
strength, without making illegal actions possible.

## Stage 7: learn position values through self-play

### Goal

Replace long or weak rollouts with a value estimate trained from experience.

### Possible work

- Define a stable, game-specific state encoding.
- Record training examples containing state, root visit distribution, current player, and final
  utility.
- Train a value model to predict the eventual result.
- Combine policy and value prediction in one model when useful.
- Add an evaluation gate: a new model replaces the current model only after a reproducible match
  suite.
- Track calibration as well as prediction error; a value of `0.8` should have a meaningful
  interpretation.
- Prevent training and evaluation matches from sharing random streams or data unintentionally.

### AlphaZero-style loop

```text
current policy and value
          |
          v
    guided MCTS
          |
          v
      self-play
          |
          v
states + visit distributions + final utilities
          |
          v
 train a better policy and value
```

### Risks

- Self-play can amplify its own blind spots.
- Training infrastructure can dominate the complexity of the game and search code.
- Neural inference can cost more than the simulations it replaces for small games.
- Strength gains must be compared at equal total compute, not just equal MCTS iterations.

### Completion signal

An AlphaZero-style experimental agent learns from self-play and beats its unguided MCTS baseline at
a documented compute budget.

## Stage 8: advanced game-theoretic and learned-model research

### Goal

Keep longer-term research directions visible without treating them as immediate requirements.

### Possible directions

- Counterfactual Regret Minimization as a baseline for two-player zero-sum imperfect-information
  games.
- Public belief states and depth-limited solving inspired by ReBeL.
- Search that combines policy improvement with game-theoretic reasoning, inspired by Student of
  Games.
- Gumbel action selection for reliable policy improvement with small simulation budgets.
- Sampled action search and progressive widening for extremely large or continuous action spaces.
- Batched leaf evaluation and parallel tree search.
- Learned dynamics inspired by MuZero when exact rules are unavailable or prohibitively expensive
  to simulate.

### Scope warning

MuZero is not automatically an upgrade over AlphaZero-style search for Meeple Bots. The project
already owns exact Rust game simulators, so replacing them with learned dynamics would introduce
model error and substantial training complexity. It becomes relevant only for environments whose
rules are unknown, inaccessible, or too expensive to execute during search.

Game-theoretic search is similarly a distinct branch rather than a small modification to UCT. It
should follow a well-tested imperfect-information model and include exploitability-oriented
evaluation.

## Suggested implementation order

The stages do not need to be completed as one linear project. A practical order is:

1. Search diagnostics and time-based comparisons.
2. Modular rollout and tree policies.
3. Informed rollouts and progressive bias.
4. Optional tree reuse and transpositions for perfect-information games.
5. Explicit chance nodes and a small dice game.
6. Observation histories, determinization, and Kuhn Poker.
7. Information Set MCTS.
8. PUCT with uniform and hand-written priors.
9. Tabular or lightweight learned policy and value models.
10. An AlphaZero-style self-play experiment.
11. Game-theoretic search or learned dynamics only when a concrete game requires them.

## Priority overview

| Direction | Learning value | Expected project value | Complexity | Near-term priority |
| --- | --- | --- | --- | --- |
| Search diagnostics | High | High | Low | Very high |
| Modular search policies | High | High | Medium | Very high |
| Better rollout policies | High | High | Low to medium | High |
| Tree reuse | Medium | Medium to high | Medium | Medium |
| Transposition tables | Medium | Game-dependent | Medium | Medium |
| Chance nodes | High | High | Medium | High |
| Determinization baseline | High | Medium | Medium | High after chance |
| Information Set MCTS | Very high | High | High | High after determinization |
| PUCT and action priors | Very high | High | Medium | Medium |
| Learned value function | Very high | Potentially high | High | Later |
| AlphaZero-style self-play | Very high | Game-dependent | Very high | Later |
| ReBeL or Student of Games ideas | Very high | Specialized | Very high | Research horizon |
| MuZero-style learned dynamics | High | Low with exact simulators | Very high | Research horizon |

## Evaluation checklist for every stage

Before accepting a new search technique, answer:

- Does it preserve legal play and the agent's information boundary?
- Is the experiment reproducible from configuration and seed?
- Does it improve strength at equal wall-clock time?
- What additional memory does it require?
- Does it work consistently across seats and opponents?
- Can its contribution be isolated with an ablation or baseline comparison?
- Does it introduce game-specific knowledge into a generic crate?
- Is there a safe fallback when the required game capability is unavailable?
- Are configuration, traces, Python bindings, and user documentation affected?

## Current recommendation

The most useful immediate path is to instrument the existing MCTS, modularize its policies, and
experiment with informed rollouts. In parallel, the game contracts can be designed for explicit
chance transitions without weakening the existing perfect-information boundary. A small stochastic
game followed by Kuhn Poker would provide a controlled progression from chance to hidden
information.

Policy-guided MCTS should begin with hand-written or tabular priors before introducing neural
networks. This exposes the essential AlphaZero idea while keeping experiments understandable.
Learned dynamics and advanced game-theoretic solvers should remain visible as longer-term options,
activated by the requirements of a concrete game rather than by the roadmap alone.
