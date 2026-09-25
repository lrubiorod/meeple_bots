"""Shared integer game parameters. Rust owns parameter names, defaults and bounds."""

from ._game_parameters import normalize_game_parameters

PLAYABLE_GAMES = ["lost_cities", "connect6", "splendor", "boop", "connect-four", "spotf", "tic-tac-toe"]




def game_parameters(game) -> dict[str, int]:
    from .connect6 import Connect6

    return {'board_size': game.board_size} if isinstance(game, Connect6) else {}


def create_game(name: str, parameters: dict[str, int] | None = None):
    from .game_types import Boop, ConnectFour, TicTacToe, SpiritsOfTheForest
    from .splendor import Splendor
    from .connect6 import Connect6
    from .lost_cities import LostCities

    factories = {
        'lost-cities': LostCities, 'boop': Boop, 'connect-four': ConnectFour, 'tic-tac-toe': TicTacToe,
        'spotf': SpiritsOfTheForest, 'splendor': Splendor, 'connect6': Connect6,
    }
    name = name.replace('_', '-')
    if name not in factories:
        raise ValueError(f'unknown game: {name}')
    return factories[name](**normalize_game_parameters(name, parameters))
