"""Lost Cities GUI configuration, separate from perfect-information MCTS controls."""
from dataclasses import dataclass, field

from ...._agent_config import SoIsmctsAgent
from ....gui.player import GuiPlayer


@dataclass(frozen=True, slots=True)
class SoIsmctsGuiPlayer:
    iterations: int | None = None
    exploration: float = 2 ** 0.5
    time_budget: float | None = None
    kind: str = field(default='so_ismcts', init=False)

    def __post_init__(self):
        agent = SoIsmctsAgent(self.iterations, self.exploration, time_budget=self.time_budget)
        object.__setattr__(self, 'iterations', agent.iterations)

    def as_dict(self):
        result = {'kind': self.kind, 'iterations': self.iterations, 'exploration': self.exploration}
        if self.time_budget is not None:
            result['time_budget'] = self.time_budget
        return result


def parse_player(raw):
    if not isinstance(raw, dict) or raw.get('kind') not in ('human', 'random', 'so_ismcts'):
        raise ValueError('Lost Cities GUI supports human, random, and so_ismcts players')
    allowed = {'kind', 'iterations', 'time_budget', 'exploration'} if raw['kind'] == 'so_ismcts' else {'kind'}
    unexpected = raw.keys() - allowed
    if unexpected:
        raise ValueError('Unsupported player settings: ' + ', '.join(sorted(unexpected)))
    if raw['kind'] == 'so_ismcts':
        return SoIsmctsGuiPlayer(**{k: v for k, v in raw.items() if k != 'kind'})
    return GuiPlayer(raw['kind'])
