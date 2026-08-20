# boop.

[Games](../README.md) · [Project overview](../../README.md)

Meeple Bots implements the standard two-player rules of boop. on a 6x6 bed. This guide explains
the turn model used by the Rust rules, Python actions, terminal input, and MCTS heuristics.

## Quick start

```bash
meeple-bots match --game boop --first human --second random --seed 42
meeple-bots gui --game boop
```

Rows and columns are zero-based in the CLI and Python API.

## Objective and pieces

Each player starts with eight kittens in their pool. Kittens graduate into cats, which then return
to the player's pool and can be placed on later turns.

A player wins after booping by either:

- forming a horizontal, vertical, or diagonal line of three cats, or
- having all eight cats on the bed.

If one placement creates winning cat lines for both players, the active player wins.

## Turn sequence

One legal action represents the complete turn:

1. Choose an available kitten or cat and an empty cell.
2. Place the piece and resolve every adjacent boop simultaneously from the pre-boop position.
3. Check for a win. A winning turn ends immediately.
4. If there is no winner, select one required graduation or recovery resolution when available.
5. Pass the turn to the opponent.

Encoding the complete decision in one action keeps Random, MCTS, terminal humans, and browser
humans on the same legal-action boundary.

## Booping

The placed piece attempts to move every adjacent piece one cell directly away from itself,
including diagonal neighbors.

- A kitten can boop kittens but not cats.
- A cat can boop kittens and cats.
- A piece with an occupied destination does not move.
- A piece pushed beyond the board returns to its owner's pool.
- Booped pieces do not trigger chain reactions.
- A player's own pieces are affected.

All movements caused by one placement are computed from the same board position. One moved piece
therefore cannot open or block the destination of another boop during that placement.

## Graduation and recovery

After a non-winning boop, every line of three owned pieces is a possible graduation. The player
must select one line:

- kittens in the selected line become cats in the player's pool;
- cats in the line return to the pool unchanged.

Longer or intersecting lines may produce several legal three-piece choices. The action records the
exact selected positions.

If all eight of the player's pieces are on the bed, recovering one owned piece is another possible
resolution. A recovered kitten graduates; a recovered cat returns unchanged. When graduations and
recoveries are both available, every valid choice is represented as a separate legal action.

Winning is checked after booping and before any graduation or recovery.

## Python action model

`BoopAction` contains the placement and optional end-of-turn resolution:

```python
from meeple_bots import BoopAction, BoopPieceKind

placement = BoopAction(
    piece=BoopPieceKind.KITTEN,
    row=2,
    column=3,
)
```

A resolved action may contain:

- `BoopGraduateLine`, identifying the exact three positions removed;
- `BoopRecoverPiece`, identifying the exact piece recovered.

Board cells contain `BoopPiece(player, kind)` values. Completed match results include `pools`, with
the kittens and cats remaining in each player's pool.

When constructing actions manually, choose one of the complete legal actions supplied by the
current turn rather than reconstructing resolution rules in application code.

## Terminal input and display

Enter `k` for a kitten or `c` for a cat, followed by row and column:

```text
k 2 3
c 4 1
```

The terminal displays player 0 kittens and cats as `x` and `X`, and player 1 pieces as `o` and `O`.
If a placement has multiple legal resolutions, a second prompt lists the complete choices.

## MCTS heuristics

Boop has longer games and many more legal actions than tic-tac-toe or Connect Four. Graduation and
recovery choices increase the branching factor further, so truncated rollouts are often useful.

| Index | Evaluation |
| --- | --- |
| `0` | Difference in cats currently present on the board, scaled by `0.1`. |
| `1` | Cat progression, immediate winning pairs, graduation potential, and safer board presence. |

Heuristic `1` assigns `0.9` to an immediate cat-line completion for the active player. Other
non-terminal values are limited to `[-0.95, 0.95]`, preserving `1.0` and `-1.0` for terminal wins
and losses.

Example with the strategic evaluator:

```bash
meeple-bots match --game boop --first human --second mcts \
  --mcts-iterations 1000 --mcts-rollout-depth 16 \
  --second-mcts-heuristic 1 --seed 42
```

Use the [game evaluation guide](../../crates/evaluation/README.md) to estimate search scale on the
current machine and the [agents guide](../../agents/README.md) for MCTS configuration details.
