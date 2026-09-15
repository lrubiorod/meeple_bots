# Connect6

Standard Connect6 uses a 19×19 board. Black places one stone in the opening;
all later physical turns contain two sequential `Place(position)` actions by the
same player. Six **or more** contiguous stones win horizontally, vertically or
on either diagonal. Victory is checked after each stone, so the second placement
is omitted if the first wins. A full board without a winner is a draw.

Positions are zero-based row-major indices. Every empty cell is one legal action.
`Connect6State` stores a byte per cell, the current player, placements remaining,
and cached winner. Rust equality/hash include this information, without move
history: non-winning A→B and B→A placements yield equal states.

## Initialization

```bash
meeple-bots match --game connect6 --first random --second random --seed 42
meeple-bots match --game connect6 --game-param board_size=11 --first mcts --second random --seed 42
meeple-bots analyze --game connect6 --game-param board_size=11
meeple-bots study --game connect6 --game-param board_size=11 --output results/studies/connect6-11 --budget 30m
```

`--game-param NAME=INTEGER` is shared by match, batch, analyze and study.
Tournament TOML uses a top-level inline table:

```toml
game = "connect6"
game_params = { board_size = 11 }
# Other tournament settings and [[agents]] follow the existing tournament format.
```

```python
from meeple_bots import Connect6, Connect6Action, create_game

standard = Connect6()  # 19×19
small = Connect6(board_size=11)
assert small == create_game("connect6", {"board_size": 11})
state = small.initial_state().apply_action(Connect6Action(position=0))
assert state.current_player == 1 and state.placements_remaining == 2
```

`DEFAULT_BOARD_SIZE` in `src/lib.rs` is the authoritative default. Rust catalog
validation rejects unknown parameters and sizes below six. The only upper bound
is technical: the cell count must fit u32 match counters and allocation indices.
Connection length remains six for every size.

The catalog carries validated parameters in the game identifier. Python workers
transport immutable game configurations; experiment headers, match traces,
analysis results and study resume requests preserve normalized parameters.
Traces replay ordered actions against the native rules. Python state snapshots
are not pickleable; game configurations are. Existing parameterless games keep
their defaults and reject unsupported parameters.

The existing deterministic MCTS reads the active player from each state, including
consecutive decisions by one player. UCT, UCB1-Tuned, tree reuse and transpositions
need no Connect6-specific search code. Utility is +1 / 0 / −1. No heuristic, RAVE or symmetry reduction is provided.

## Browser play

```bash
meeple-bots gui --game connect6
```

Choose board size in the settings (default 19), then click an empty intersection
to place each stone. The status displays the active color and remaining placements.
Each seat supports human, random or MCTS; MCTS controls appear only for MCTS seats.
The initial MCTS settings come from [`connect6-baseline.toml`](../../configs/mcts/connect6-baseline.toml), a
provisional one-second profile calibrated on 13x13. This profile does not set the board size;
select 13 explicitly to use its intended board size. Configure iterations/time, selector,
exploration, depth, tree reuse and transpositions independently for each player.
The interface includes move history, last-move marking, pacing, seeded restarts
and optional extractable traces under `results/gui/connect6` retaining board size.

MCTS controls include Progressive Widening (PW) for either player, with editable
`k > 0` and `0 < alpha <= 1`. The k/alpha inputs appear only when PW is enabled;
all MCTS controls are hidden for human/random players. PW defaults load from
`configs/mcts/connect6-baseline.toml`, and saved traces retain the chosen values.
