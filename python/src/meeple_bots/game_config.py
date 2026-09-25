"""Shared integer game parameters. Rust owns parameter names, defaults and bounds."""

from . import _native

PLAYABLE_GAMES = ["lost_cities", "connect6", "splendor", "boop", "connect-four", "spotf", "tic-tac-toe"]


def normalize_game_parameters(name: str, parameters: dict[str, int] | None = None) -> dict[str, int]:
    parameters = {} if parameters is None else parameters
    if not isinstance(parameters, dict):
        raise TypeError('game_params must be a mapping')
    for key, value in parameters.items():
        if not isinstance(key, str) or type(value) is not int:
            raise ValueError('game parameters require string names and integer values')
    try:
        return dict(_native.normalize_game_parameters(name.replace('-', '_'), parameters))
    except OverflowError as error:
        raise ValueError('game parameter values must fit signed 64-bit integers') from error


def game_parameters(game) -> dict[str, int]:
    from .connect6 import Connect6

    return {'board_size': game.board_size} if isinstance(game, Connect6) else {}


def create_game(name: str, parameters: dict[str, int] | None = None):
    from .api import Boop, ConnectFour, TicTacToe, SpiritsOfTheForest, Splendor
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
