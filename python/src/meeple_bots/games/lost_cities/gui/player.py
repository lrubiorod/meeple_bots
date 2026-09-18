"""Lost Cities GUI configuration, separate from perfect-information MCTS controls."""
from dataclasses import dataclass, field

from ...._agent_config import SoIsmctsAgent
from ....gui.player import GuiPlayer


@dataclass(frozen=True, slots=True)
class SoIsmctsGuiPlayer:
    iterations: int = 1000
    exploration: float = 2 ** 0.5
    kind: str = field(default='so_ismcts', init=False)

    def __post_init__(self):
        SoIsmctsAgent(self.iterations, self.exploration)

    def as_dict(self):
        return {'kind': self.kind, 'iterations': self.iterations, 'exploration': self.exploration}


def parse_player(raw):
    if not isinstance(raw, dict) or raw.get('kind') not in ('human', 'random', 'so_ismcts'):
        raise ValueError('Lost Cities GUI supports human, random, and so_ismcts players')
    allowed = {'kind', 'iterations', 'exploration'} if raw['kind'] == 'so_ismcts' else {'kind'}
    unexpected = raw.keys() - allowed
    if unexpected:
        raise ValueError('Unsupported player settings: ' + ', '.join(sorted(unexpected)))
    if raw['kind'] == 'so_ismcts':
        return SoIsmctsGuiPlayer(**{k: v for k, v in raw.items() if k != 'kind'})
    return GuiPlayer(raw['kind'])
