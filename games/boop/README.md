# boop.

[Back to the games overview](../README.md) · [Back to the project overview](../../README.md)

Meeple Bots implements the standard two-player rules of boop. on a 6x6 bed.

## Pieces and objective

Each player starts with eight kittens in their pool. Cats enter the pool when kittens graduate.
The active player chooses an available kitten or cat and places it in an empty cell.

A player wins by forming a horizontal, vertical, or diagonal line of three cats, or by having all
eight cats on the bed at the end of a turn. If a single boop creates winning cat lines for both
players, the active player wins.

## Booping

The placed piece attempts to move every adjacent piece one cell directly away, including diagonal
neighbors.

- A kitten can boop kittens but not cats.
- A cat can boop kittens and cats.
- A piece cannot move when its destination is occupied.
- A piece pushed beyond the board returns to its owner's pool.
- Booped pieces do not cause chain reactions.
- A player's own pieces are affected by boops.

All movements caused by one placement are resolved from the same pre-boop board position.

## Graduation and recovery

After booping, a line of three pieces belonging to the active player must be removed. Kittens in
the selected line graduate into cats in the player's pool; cats return to the pool unchanged.

When several lines are available, the action identifies the exact selected line. A line containing
more than three pieces therefore produces multiple legal three-piece resolutions where applicable.

If all eight pieces are on the bed, the player may instead recover one of them. A recovered kitten
graduates into a cat; a recovered cat returns to the pool. When both graduation and recovery are
available, every valid choice is represented as a separate legal action.

Winning is checked after booping and before graduation or recovery.

## Python model

`BoopAction` contains the complete deterministic turn:

```python
from meeple_bots import BoopAction, BoopPieceKind

placement = BoopAction(
    piece=BoopPieceKind.KITTEN,
    row=2,
    column=3,
)
```

Actions returned by a `HumanTurn` can also contain:

- `BoopGraduateLine`: the exact three positions removed.
- `BoopRecoverPiece`: the exact piece recovered.

Board cells contain `BoopPiece(player, kind)` values. `MatchResult.pools` reports the kittens and
cats remaining in each player's pool.

## Interactive input

Use `k` for a kitten or `c` for a cat followed by a zero-based row and column:

```text
k 2 3
c 4 1
```

The terminal displays player 0 kittens/cats as `x`/`X` and player 1 kittens/cats as `o`/`O`. When
a placement has several end-of-turn resolutions, a second prompt lists the legal choices.

```bash
meeple-bots match --game boop --first human --second random --seed 42
```

## MCTS considerations

boop. has substantially more legal actions and longer games than tic-tac-toe or Connect Four.
Graduation and recovery choices increase the branching factor further. Small manual configurations
are useful for quick development runs:

```bash
meeple-bots match --game boop --first human --second mcts \
  --mcts-iterations 20 --mcts-rollout-depth 128 --seed 42
```

Use the [complexity evaluation guide](../../crates/evaluation/README.md) to measure the current
ruleset and obtain a hardware-aware configuration.
