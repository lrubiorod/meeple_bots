# Spirits of the Forest

[Back to the games guide](../README.md)

Meeple Bots implements the base two-player game with three public gemstones per player and no
favor tokens. Removing favor tokens removes the hidden information and the only randomness is the
face-up forest created before play.

The implementation follows the
[official rulebook](https://thundergryph.com/wp-content/uploads/2022/08/soft_rulebook_en_web.pdf).
It uses the 48-tile composition supplied for this project and does not include expansions,
scenarios, or the three- and four-player rules.

## Setup and objective

The match seed shuffles 48 face-up tiles into four rows of twelve. The shuffle uses a separate
random stream from both agents, so an identical match configuration and seed reproduces the forest
and automated decisions.

Players collect spirit symbols and Fire, Moon, and Sun icons. For each of the nine spirits and
three power sources, the player with the majority scores their symbol count. Tied majorities score
for both players; having none of a category costs three points. The highest total wins, with fewer
collected tiles breaking a tie and an exact second tie producing a draw.

## Turn phases

A physical turn is represented by several engine actions:

1. Take an exposed tile from either end of a row.
2. After taking one single-spirit tile, optionally take a second exposed single tile of the same
   spirit or end collection.
3. Place, move, or skip a gemstone.

A double-spirit tile ends collection immediately. The first player may take only one tile on the
first turn. Taking an opponent-reserved tile permanently removes one chosen gemstone belonging to
the active player; the opponent recovers their reservation.

Each phase decision counts as one `ply`, while the Rust state tracks completed physical turns
separately. Splitting the turn keeps MCTS branching manageable and lets the GUI present each choice
directly.

## Python and CLI

```python
from meeple_bots import Match, MctsAgent, RandomAgent, SpiritsOfTheForest

result = Match(
    game=SpiritsOfTheForest(),
    first=MctsAgent(iterations=500, rollout_depth=64, heuristic=0),
    second=RandomAgent(),
    seed=42,
    max_plies=256,
).run()

print(result.scores)
print(result.spirit_collections)
print(result.gemstone_pools)
```

```bash
meeple-bots gui --game spotf
meeple-bots match --game spotf --first human --second mcts \
  --second-mcts-heuristic 0 --seed 42
```

The public action union contains `TakeSpiritTile`, `EndSpiritCollection`,
`PlaceSpiritGemstone`, `MoveSpiritGemstone`, and `SkipSpiritGemstone`. Human selectors receive
only currently legal instances of these types.

## MCTS heuristics

Three state evaluators are available for cutoff evaluation or informed rollouts:

- Index `0` combines provisional score difference, remaining usable gemstones, and a small
  reservation bonus. Every usable gemstone has a constant weight of `0.5`.
- Index `1` keeps the same score and reservation terms but values each usable gemstone according to
  the fraction of forest tiles remaining: `0.5 + 4 * remaining_fraction^2`. A sacrificed gemstone
  therefore costs `4.5` raw heuristic points at the start, `1.5` halfway through the forest, and
  approaches the index `0` cost near the end.
- Index `2` keeps the gemstone terms from index `1`, but replaces provisional majority scoring
  with reachable progress. For each spirit and power source, every collected symbol is worth one
  raw point only while `collected + remaining >= ceil(total / 2)`. A mathematically lost category
  is worth zero; once no symbol remains, a player with none receives the real `-3` penalty. All
  forest symbols are treated as potentially reachable, including reservations.

All results are normalized to `[-1, 1]`; terminal states always use exact match utility. Neutral
terminal rollouts remain available by leaving `heuristic=None`.

## Tournament analysis

Completed SPOTF tournament traces can be replayed and converted into analysis tables and an HTML
report:

```bash
meeple-bots extract --input results/tournaments/spotf-study.jsonl
meeple-bots report --input results/tournaments/spotf-study/data
```

The native replay analyzer reconstructs the shuffled forest from the match seed and validates every
recorded action. It separates engine plies from physical turns, records tile collection and gemstone
behavior, and breaks the final score down across the nine spirits and three power sources. See the
[Python guide](../../python/README.md#2-extract-analysis-tables) for the generated table list.
