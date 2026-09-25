"""Search-family policy for the shared study coordinator, not another executor."""

class MctsStudyProfile:
    name = 'mcts'
    version = 1
    supported_tuners = ('exploration', 'selection', 'structure', 'tree-reuse', 'rave',
        'progressive-widening', 'progressive-widening-k', 'progressive-widening-alpha',
        'widening-expansion', 'cutoff-depth')
    mechanisms = ('tree_reuse', 'transpositions', 'rave', 'progressive_widening')


    def resolved_tuners(self, caps):
        return tuple(t for t in self.supported_tuners if 'uct_rave' in caps['selection_policies']
                     or t not in ('rave', 'progressive-widening', 'progressive-widening-k',
                                  'progressive-widening-alpha', 'widening-expansion'))


class SoIsmctsStudyProfile:
    name = 'so_ismcts'
    version = 3
    supported_tuners = ('exploration', 'selection', 'tree-reuse')
    mechanisms = ('tree_reuse',)

    def resolved_tuners(self, caps):
        return self.supported_tuners


PROFILES = {'mcts': MctsStudyProfile(), 'so_ismcts': SoIsmctsStudyProfile()}

def profile_for_agent(agent):
    from .._agent_config import SoIsmctsAgent
    return PROFILES['so_ismcts' if isinstance(agent, SoIsmctsAgent) else 'mcts']
