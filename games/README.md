# Games

[Back to the project overview](../README.md)

Each game owns its strongly typed Rust `State` and `Action` representations while implementing the
shared `Game` contract. Python exposes matching game and action types.

## Supported games

| Game | CLI identifier | Python game | Action | Human input |
| --- | --- | --- | --- | --- |
| Tic-tac-toe | `tic-tac-toe` | `TicTacToe` | `TicTacToeAction(row, column)` | `row column` |
| Connect Four | `connect-four` | `ConnectFour` | `ConnectFourAction(column)` | `column` |
| boop. | `boop` | `Boop` | `BoopAction(piece, row, column, resolution)` | `k/c row column` |

Rows, columns, and player identifiers are zero-based. All current games are deterministic,
sequential, two-player, zero-sum, and perfect-information rulesets.

## Tic-tac-toe

The standard 3x3 game. Players alternate placing pieces in empty cells and win with three pieces
in a horizontal, vertical, or diagonal line. A full board without a winning line is a draw.

```bash
meeple-bots match --game tic-tac-toe --first human --second mcts --seed 42
```

## Connect Four

The standard 6x7 game. An action selects a column; the Rust rules apply gravity and place the piece
in its lowest available row. Four connected pieces horizontally, vertically, or diagonally win.
A full board without a winner is a draw.

```bash
meeple-bots match --game connect-four --first human --second mcts --seed 42
```

## boop.

boop. uses a 6x6 board, two piece ranks, displacement rules, and explicit graduation or recovery
choices. These choices are encoded in the action so Random, MCTS, and human players all operate on
the same complete legal-action set.

See the [boop. rules and interface guide](boop/README.md) for the full model.

```bash
meeple-bots match --game boop --first human --second mcts --seed 42
```

## Adding another game

A new Rust game belongs under `games/` and implements the contracts defined by `meeple_bots_core`.
Runtime selection is added at the catalog boundary, while Python-facing types and serialization are
added in the bindings and public package. See the [Rust architecture guide](../crates/README.md) for
the dependency and dispatch model.
