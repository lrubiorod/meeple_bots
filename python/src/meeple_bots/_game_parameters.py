"""Validate game parameters against the native catalog, without constructing games."""

from . import _native

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

