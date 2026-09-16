# Games

[Back to the project overview](../README.md)

Every game owns its authoritative Rust state, action type, legal-action generator, transition
rules, and terminal utilities. Python exposes matching game and action values without duplicating
the rules.

## Supported games

| Game | CLI identifier | Python type | Human input |
| --- | --- | --- | --- |
| Tic-tac-toe | `tic-tac-toe` | `TicTacToe` | `row column` |
| Connect Four | `connect-four` | `ConnectFour` | `column` |
| boop. | `boop` | `Boop` | `k/c row column` |
| Spirits of the Forest | `spotf` | `SpiritsOfTheForest` | legal action index |
| [Lost Cities](lost-cities/README.md) | `lost_cities` | `LostCities` | Random agents only |
| [Can't Stop](cant-stop/README.md) | `cant-stop` (GUI) | `CantStopSession` | GUI action index |

Rows, board indices and player identifiers are zero-based; Can't Stop column labels are sums 2–12.
All current games are sequential, two-player and zero-sum. Lost Cities is the first
imperfect-information game; the others have perfect information. Lost Cities uses
private card draws, Splendor public refills and Can't Stop public dice. See each
game guide for its interfaces and integration limits.

## Tic-tac-toe

The standard 3x3 game. Players place pieces in empty cells and win with a horizontal, vertical, or
diagonal line of three. A full board without a winner is a draw.

- Python action: `TicTacToeAction(row, column)`
- Board: 3 rows by 3 columns
- MCTS cutoff heuristics: none

```bash
meeple-bots match --game tic-tac-toe --first human --second mcts --seed 42
```

## Connect Four

The standard 6x7 game. An action selects a column; the rules apply gravity and place the piece in
its lowest free row. Four connected pieces horizontally, vertically, or diagonally win. A full
board without a winner is a draw.

- Python action: `ConnectFourAction(column)`
- Board: 6 rows by 7 columns
- MCTS cutoff heuristics: none

```bash
meeple-bots match --game connect-four --first human --second mcts --seed 42
```

## boop.

boop. uses a 6x6 board, kittens and cats, displacement rules, and mandatory end-of-turn choices.
Each legal action includes both the placement and any required graduation or recovery, so human,
Random, and MCTS players share one complete action model.

- Python action: `BoopAction(piece, row, column, resolution)`
- Board: 6 rows by 6 columns
- MCTS cutoff heuristics: indices `0` and `1`

```bash
meeple-bots match --game boop --first human --second mcts --seed 42
```

Read the [Boop rules and interface guide](boop/README.md) before constructing its actions directly.

## Spirits of the Forest

The two-player base game uses a seeded 4x12 forest, set collection, public gemstone reservations,
and majority scoring. Favor tokens are omitted, so the complete position is visible to both
players. A physical turn is split into collection and gemstone decisions to keep search branching
manageable.

- Python actions: `TakeSpiritTile`, `EndSpiritCollection`, `PlaceSpiritGemstone`,
  `MoveSpiritGemstone`, and `SkipSpiritGemstone`
- Forest: 4 rows by 12 tiles
- MCTS cutoff heuristic: index `0` for reachable category progress and phase-dependent gemstone
  conservation

```bash
meeple-bots match --game spotf --first human --second mcts \
  --second-mcts-heuristic 0 --seed 42
```

See the [rules and interface guide](spirits-of-the-forest/README.md) for the exact variant and
public types.

## Adding a game

A new game begins as an independent Rust crate under `games/`:

1. Define its complete `State`, one-turn `Action`, and legal-action iterator.
2. Implement `Game`, including observation, transitions, and terminal utility.
3. Implement only the capability traits that the rules genuinely satisfy.
4. Add catalog dispatch and Python binding conversions.
5. Add the public Python game, action, board, and result representations.
6. Register CLI and presentation support where appropriate.
7. Test rules in Rust and public behavior in Python.

See the [Rust architecture guide](../crates/README.md#game-contract) for the contracts and
[where changes belong](../crates/README.md#where-changes-belong) for integration boundaries.
