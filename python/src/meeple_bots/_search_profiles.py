"""Compatible search-family resolution and profile loading shared by analyze/study."""
import tomllib
from ._agent_config import MctsAgent, SoIsmctsAgent
from ._capabilities import game_search_capabilities

# Capabilities decide eligibility; a future family can join this registry without
# tying game IDs permanently to one search algorithm.
FAMILY_REQUIREMENTS = {
    'mcts': lambda caps: 'mcts' in caps.get('search_agents', []) and not caps.get('imperfect_information'),
    'so_ismcts': lambda caps: 'so_ismcts' in caps.get('search_agents', []) and caps.get('imperfect_information', False),
}


def resolve_family(game, requested=None, baseline=None):
    caps = game_search_capabilities(game)
    compatible = [name for name, supports in FAMILY_REQUIREMENTS.items() if supports(caps)]
    if baseline is not None and not isinstance(baseline, (MctsAgent, SoIsmctsAgent)):
        raise ValueError('baseline must belong to a compatible search-agent family')
    inferred = 'so_ismcts' if isinstance(baseline, SoIsmctsAgent) else 'mcts' if isinstance(baseline, MctsAgent) else None
    family = requested or inferred or (compatible[0] if len(compatible) == 1 else None)
    if family not in compatible:
        raise ValueError(f'{family or "unspecified family"} is not a compatible search agent for {game}; compatible: {compatible}')
    if inferred and family != inferred:
        raise ValueError('agent family conflicts with baseline configuration')
    return family


def load_search_profile(path):
    return load_named_search_profile(path).agent


def load_named_search_profile(path):
    """Read a profile once, retaining the caller-visible name projection."""
    from types import SimpleNamespace
    from ._mcts_profiles import _mcts_profile_from_values
    values = tomllib.loads(path.read_text())
    if values.get('agent', 'mcts') == 'mcts':
        agent = _mcts_profile_from_values(values, path).agent
    elif values.get('agent') == 'so_ismcts':
        agent = so_from_values(values)
    else:
        raise ValueError('unknown search agent family')
    return SimpleNamespace(name=values.get('name', path.stem), agent=agent)


def so_from_values(values):
    unknown = values.keys() - {'agent', 'name', 'iterations', 'time_budget', 'exploration', 'rollout', 'root_selection', 'selection_policy', 'tree_reuse'}
    if unknown:
        raise ValueError(f'unsupported SO-ISMCTS fields: {sorted(unknown)}')
    if values.get('rollout', 'uniform') != 'uniform' or values.get('root_selection', 'most_visited') != 'most_visited':
        raise ValueError('SO-ISMCTS requires uniform rollout and most_visited root selection')
    return SoIsmctsAgent(iterations=values.get('iterations'), time_budget=values.get('time_budget'),
                         exploration=values.get('exploration', 2**.5),
                         selection_policy=values.get('selection_policy', 'uct'), tree_reuse=values.get('tree_reuse', False))

