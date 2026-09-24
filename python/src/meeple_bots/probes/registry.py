"""Concrete case registration; runner/reporting do not import game rules."""
from .games.lost_cities import CASES as LOST_CITIES_CASES
from .games.spotf import CASES as SPOTF_CASES

CASES = (*LOST_CITIES_CASES, *SPOTF_CASES)


def select_cases(*, game=None, suite=None, probe=None, cases=CASES):
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate probe IDs')
    selected = tuple(c for c in cases if (game is None or c.game == game)
                     and (suite is None or suite in c.tags)
                     and (probe is None or probe.replace('-', '_') in (c.id, c.id.split('.', 1)[-1])))
    if not selected:
        raise ValueError('no probes match the requested game/suite/probe')
    return selected
