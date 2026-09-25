"""Lightweight Python game values and observer payloads used by the public API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from .connect6 import Connect6, Connect6Action
from .lost_cities import LostCities, LostCitiesAction
from .splendor import Splendor, SplendorAction

@dataclass(frozen=True, slots=True)
class TicTacToe:
    """The standard 3x3 tic-tac-toe rules."""

@dataclass(frozen=True, slots=True)
class ConnectFour:
    """The standard 6x7 Connect Four rules with gravity."""

@dataclass(frozen=True, slots=True)
class Boop:
    """The standard two-player rules for boop. on a 6x6 bed."""

@dataclass(frozen=True, slots=True)
class SpiritsOfTheForest:
    """Two-player Spirits of the Forest without favor tokens."""

@dataclass(frozen=True, slots=True)
class TicTacToeAction:
    """A zero-based row and column on the tic-tac-toe board."""

    row: int
    column: int

    def __post_init__(self) -> None:
        for name, value in (("row", self.row), ("column", self.column)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= value < 3:
                raise ValueError(f"{name} must be between 0 and 2")

@dataclass(frozen=True, slots=True)
class ConnectFourAction:
    """A zero-based column in which to drop a Connect Four piece."""

    column: int

    def __post_init__(self) -> None:
        if isinstance(self.column, bool) or not isinstance(self.column, int):
            raise TypeError("column must be an integer")
        if not 0 <= self.column < 7:
            raise ValueError("column must be between 0 and 6")

class BoopPieceKind(str, Enum):
    """The two ranks of pieces in boop."""

    KITTEN = "kitten"
    CAT = "cat"

@dataclass(frozen=True, slots=True)
class BoopPosition:
    """A zero-based position on the 6x6 boop. board."""

    row: int
    column: int

    def __post_init__(self) -> None:
        for name, value in (("row", self.row), ("column", self.column)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= value < 6:
                raise ValueError(f"{name} must be between 0 and 5")

@dataclass(frozen=True, slots=True)
class BoopGraduateLine:
    """The exact line of three pieces selected for graduation."""

    positions: tuple[BoopPosition, BoopPosition, BoopPosition]

    def __post_init__(self) -> None:
        if not isinstance(self.positions, tuple) or len(self.positions) != 3:
            raise TypeError("positions must be a tuple containing exactly three positions")
        if not all(isinstance(position, BoopPosition) for position in self.positions):
            raise TypeError("every graduation position must be a BoopPosition")

@dataclass(frozen=True, slots=True)
class BoopRecoverPiece:
    """The piece selected for recovery when all eight pieces are on the board."""

    position: BoopPosition

    def __post_init__(self) -> None:
        if not isinstance(self.position, BoopPosition):
            raise TypeError("position must be a BoopPosition")

BoopResolution: TypeAlias = BoopGraduateLine | BoopRecoverPiece | None

@dataclass(frozen=True, slots=True)
class BoopAction:
    """A placement and any mandatory end-of-turn resolution in boop."""

    piece: BoopPieceKind
    row: int
    column: int
    resolution: BoopResolution = None

    def __post_init__(self) -> None:
        if not isinstance(self.piece, BoopPieceKind):
            raise TypeError("piece must be a BoopPieceKind")
        BoopPosition(self.row, self.column)

@dataclass(frozen=True, slots=True)
class BoopPiece:
    """A kitten or cat on the board and its owner."""

    player: int
    kind: BoopPieceKind

    def __post_init__(self) -> None:
        if self.player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        if not isinstance(self.kind, BoopPieceKind):
            raise TypeError("kind must be a BoopPieceKind")

@dataclass(frozen=True, slots=True)
class BoopPool:
    """The kittens and cats currently available to one player."""

    kittens: int
    cats: int

    def __post_init__(self) -> None:
        for name, value in (("kittens", self.kittens), ("cats", self.cats)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= value <= 8:
                raise ValueError(f"{name} must be between 0 and 8")

class Spirit(str, Enum):
    MOSS = "moss"
    FLOWERS = "flowers"
    FRUITS = "fruits"
    MUSHROOMS = "mushrooms"
    WATER = "water"
    VINES = "vines"
    BRANCHES = "branches"
    LEAVES = "leaves"
    WEBS = "webs"

class PowerSource(str, Enum):
    FIRE = "fire"
    MOON = "moon"
    SUN = "sun"

class SpiritsTurnPhase(str, Enum):
    COLLECT = "collect"
    PLACE_GEMSTONE = "place_gemstone"

@dataclass(frozen=True, slots=True)
class ForestPosition:
    row: int
    column: int

    def __post_init__(self) -> None:
        for name, value, upper in (("row", self.row, 4), ("column", self.column, 12)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= value < upper:
                raise ValueError(f"{name} must be between 0 and {upper - 1}")

@dataclass(frozen=True, slots=True)
class SpiritGemstoneSacrifice:
    """A sacrificed gem; a missing source means a gem from the available supply."""

    source: ForestPosition | None = None

    def __post_init__(self) -> None:
        if self.source is not None and not isinstance(self.source, ForestPosition):
            raise TypeError("source must be a ForestPosition or None")

@dataclass(frozen=True, slots=True)
class TakeSpiritTile:
    position: ForestPosition
    sacrifice: SpiritGemstoneSacrifice | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.position, ForestPosition):
            raise TypeError("position must be a ForestPosition")
        if self.sacrifice is not None and not isinstance(
            self.sacrifice, SpiritGemstoneSacrifice
        ):
            raise TypeError("sacrifice must be a SpiritGemstoneSacrifice or None")

@dataclass(frozen=True, slots=True)
class EndSpiritCollection:
    pass

@dataclass(frozen=True, slots=True)
class PlaceSpiritGemstone:
    target: ForestPosition

    def __post_init__(self) -> None:
        if not isinstance(self.target, ForestPosition):
            raise TypeError("target must be a ForestPosition")

@dataclass(frozen=True, slots=True)
class MoveSpiritGemstone:
    source: ForestPosition
    target: ForestPosition

    def __post_init__(self) -> None:
        if not isinstance(self.source, ForestPosition) or not isinstance(
            self.target, ForestPosition
        ):
            raise TypeError("source and target must be ForestPosition values")

@dataclass(frozen=True, slots=True)
class SkipSpiritGemstone:
    pass

SpiritsOfTheForestAction: TypeAlias = (
    TakeSpiritTile
    | EndSpiritCollection
    | PlaceSpiritGemstone
    | MoveSpiritGemstone
    | SkipSpiritGemstone
)

@dataclass(frozen=True, slots=True)
class SpiritTile:
    spirit: Spirit
    spirit_symbols: int
    power_source: PowerSource | None
    gemstone: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.spirit, Spirit):
            raise TypeError("spirit must be a Spirit")
        if self.spirit_symbols not in (1, 2):
            raise ValueError("spirit_symbols must be 1 or 2")
        if self.power_source is not None and not isinstance(
            self.power_source, PowerSource
        ):
            raise TypeError("power_source must be a PowerSource or None")
        if self.gemstone not in (None, 0, 1):
            raise ValueError("gemstone must be player 0, player 1, or None")

@dataclass(frozen=True, slots=True)
class SpiritCollection:
    spirit_symbols: tuple[int, ...]
    power_sources: tuple[int, ...]
    tiles: int

    def __post_init__(self) -> None:
        if len(self.spirit_symbols) != 9 or len(self.power_sources) != 3:
            raise ValueError("a collection needs 9 spirit and 3 power-source counts")
        values = (*self.spirit_symbols, *self.power_sources, self.tiles)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("collection counts must be non-negative integers")

@dataclass(frozen=True, slots=True)
class SpiritGemstonePool:
    available: int
    placed: int
    removed: int

    def __post_init__(self) -> None:
        values = (self.available, self.placed, self.removed)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("gemstone counts must be non-negative integers")
        if sum(values) != 3:
            raise ValueError("each player must account for exactly three gemstones")

Game: TypeAlias = TicTacToe | ConnectFour | Boop | SpiritsOfTheForest | Splendor | LostCities | Connect6

GameAction: TypeAlias = (
    Connect6Action | SplendorAction | LostCitiesAction |
    TicTacToeAction | ConnectFourAction | BoopAction | SpiritsOfTheForestAction
)

BoardCell: TypeAlias = int | BoopPiece | SpiritTile | None

GameBoard: TypeAlias = tuple[tuple[BoardCell, ...], ...]

@dataclass(frozen=True, slots=True)
class HumanTurn:
    """Read-only position presented to a human move selector."""

    game: Game
    player: int
    board: GameBoard
    legal_actions: tuple[GameAction, ...]
    pools: tuple[BoopPool, BoopPool] | None = None
    spirit_collections: tuple[SpiritCollection, SpiritCollection] | None = None
    gemstone_pools: tuple[SpiritGemstonePool, SpiritGemstonePool] | None = None
    scores: tuple[int, int] | None = None
    phase: SpiritsTurnPhase | None = None

MoveSelector: TypeAlias = Callable[[HumanTurn], GameAction]

@dataclass(frozen=True, slots=True)
class HumanMoveObservation:
    """State immediately after an accepted human action."""

    game: Game
    player: int
    action: GameAction
    board: GameBoard
    pools: tuple[BoopPool, BoopPool] | None = None
    spirit_collections: tuple[SpiritCollection, SpiritCollection] | None = None
    gemstone_pools: tuple[SpiritGemstonePool, SpiritGemstonePool] | None = None
    scores: tuple[int, int] | None = None
    phase: SpiritsTurnPhase | None = None
    active_player: int | None = None

HumanMoveObserver: TypeAlias = Callable[[HumanMoveObservation], None]

@dataclass(frozen=True, slots=True)
class MatchMoveObservation:
    """Live action and agent time so far, excluding game and observer work.

    Final MatchResult moves also include lifecycle work remaining at match end.
    """

    game: Game
    player: int
    action: GameAction
    board: GameBoard
    decision_seconds: float
    search_iterations: int | None = None
    search_nodes: int | None = None
    pools: tuple[BoopPool, BoopPool] | None = None
    spirit_collections: tuple[SpiritCollection, SpiritCollection] | None = None
    gemstone_pools: tuple[SpiritGemstonePool, SpiritGemstonePool] | None = None
    scores: tuple[int, int] | None = None
    phase: SpiritsTurnPhase | None = None
    active_player: int | None = None

MatchMoveObserver: TypeAlias = Callable[[MatchMoveObservation], None]

