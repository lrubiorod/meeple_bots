"""Cached search metadata from the installed native game catalog."""

from functools import cache

from . import _native


@cache
def game_search_capabilities(game: str) -> dict:
    return _native.game_search_capabilities(game.replace("-", "_"))


def heuristic_indices(game: str) -> tuple[int, ...]:
    return tuple(game_search_capabilities(game)["heuristics"])
