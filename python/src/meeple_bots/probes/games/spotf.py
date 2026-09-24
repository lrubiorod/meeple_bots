"""Public SPOTF replay fixtures and hypothesis annotations, never expected moves."""
from functools import partial
import json
from pathlib import Path

from ...spotf import SpotfPosition
from ...api import _spirits_action_from_mapping, _spirits_action_description
from ..core import ProbeCase, ProbePosition

FIXTURES = json.loads(Path(__file__).with_name('spotf_positions.json').read_text())


def position(name):
    fixture = FIXTURES[name]
    public = SpotfPosition(fixture['seed'], tuple(fixture['action_indices']))
    return ProbePosition(public, public.legal_actions(), public.root_player, fixture)


def search(agent, observation, legal_actions, *, seed):
    if not isinstance(observation, SpotfPosition):
        raise TypeError('SPOTF probe search requires a public SpotfPosition')
    if legal_actions != observation.legal_actions():
        raise ValueError('probe must search all legal root actions')
    return observation.search(agent, seed=seed)


CASES = tuple(ProbeCase(
    id='spotf.' + name, game='spotf', description=f['description'],
    tags=(f['phase'], 'contrast' if name.startswith('reservation_context') else 'replay'),
    builder=partial(position, name), action_label=_spirits_action_description,
    candidate_actions=tuple(_spirits_action_from_mapping(a) for a in f['candidates']),
    notes=f['notes'], search=search,
) for name, f in FIXTURES.items())
