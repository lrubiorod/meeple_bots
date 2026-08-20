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

Rows, columns, and player identifiers are zero-based. All current games are sequential,
deterministic, perfect-information, two-player, and zero-sum.

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
