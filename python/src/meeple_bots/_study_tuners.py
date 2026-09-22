"""Candidate generation only; studies owns execution, evidence and promotion."""
from dataclasses import fields, is_dataclass, replace
from collections.abc import Mapping
from itertools import product
import math

TUNING_FIELDS = {
    'exploration': ('exploration',),
    'selection': ('selection_policy',),
    'rave': ('rave_equivalence',),
    'progressive-widening': ('progressive_widening_k', 'progressive_widening_alpha'),
    'progressive-widening-k': ('progressive_widening_k',),
    'progressive-widening-alpha': ('progressive_widening_alpha',),
    'widening-expansion': ('progressive_widening_expansion',),
    'structure': ('tree_reuse', 'transpositions'),
    'tree-reuse': ('tree_reuse',),
    'cutoff-depth': ('rollout_depth',),
}


def config_fields(value):
    if is_dataclass(value):
        return {f.name: config_fields(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: config_fields(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [config_fields(v) for v in value]
    return value


def changes(base, candidate):
    before, after = config_fields(base), config_fields(candidate)
    return {key: {'before': before[key], 'after': after[key]} for key in before if before[key] != after[key]}


def assert_frozen(base, candidate, dimension):
    unexpected = changes(base, candidate).keys() - set(TUNING_FIELDS[dimension])
    if unexpected:
        raise ValueError(f'{dimension} tuner changed frozen fields: {sorted(unexpected)}')


def validate_tuner(dimension, base, selectors):
    from ._study_profiles import profile_for_agent
    profile = profile_for_agent(base)
    if dimension not in profile.supported_tuners:
        raise ValueError(f"tuner '{dimension}' is not supported by agent family '{profile.name}'")
    if dimension.startswith('progressive-widening') or dimension == 'widening-expansion':
        if not base.progressive_widening:
            raise ValueError(f'{dimension} requires PW already enabled; local tuning never enables it implicitly')
    if dimension == 'rave' and base.selection_policy != 'uct_rave':
        raise ValueError('rave_equivalence affects UCT-RAVE selection; --tune rave requires selection_policy=uct_rave')
    if dimension == 'widening-expansion' and 'uct_rave' not in selectors:
        raise ValueError('widening-expansion is not applicable: AMAF unavailable for this backend')
    if dimension == 'exploration' and base.selection_policy == 'ucb1_tuned':
        raise ValueError('UCB1-Tuned does not use exploration; choose another tuning dimension')
    if dimension == 'selection' and base.selection_policy not in selectors:
        raise ValueError('selection policy unavailable')


def neighbors(value, tested, *, geometric, lower, upper=None):
    """Check adjacent gaps; extend unbounded sides. At most two candidates."""
    below = max((v for v in tested if v < value), default=None)
    above = min((v for v in tested if v > value), default=None)
    midpoint = (lambda a, b: math.sqrt(a) * math.sqrt(b)) if geometric else (lambda a, b: (a+b)/2)
    step = max(.125, abs(value)/2)
    candidates = [midpoint(below, value) if below is not None else value/2 if geometric else value-step,
                  midpoint(value, above) if above is not None else value*2 if geometric else value+step]
    return sorted({max(lower, min(upper, v) if upper is not None else v) for v in candidates
                   if math.isfinite(v)} - set(tested) - {value})


def proposals(dimension, base, *, selectors=(), horizon=None, tested=(), coarse=False):
    """Return challengers only. The coordinator always retains the incumbent."""
    if dimension == 'selection':
        candidates = [replace(base, selection_policy=s) for s in selectors]
    elif dimension == 'tree-reuse':
        candidates = [replace(base, tree_reuse=r) for r in (False, True)]
    elif dimension == 'structure':
        candidates = [replace(base, tree_reuse=r, transpositions=t) for r, t in product((False, True), repeat=2)]
    elif dimension == 'widening-expansion':
        candidates = [replace(base, progressive_widening_expansion=p) for p in ('random', 'rave')]
    elif dimension == 'progressive-widening':
        # Small joint recheck; the coordinator precedes this with k and alpha rounds.
        candidates = [replace(base, progressive_widening_k=base.progressive_widening_k*f,
                              progressive_widening_alpha=min(1., max(.05, base.progressive_widening_alpha+d)))
                      for f, d in product((.5, 2.), (-.125, .125))]
    else:
        field = TUNING_FIELDS[dimension][0]
        value = getattr(base, field)
        if tested:
            values = neighbors(value, tested, geometric=dimension in ('rave', 'progressive-widening-k'),
                               lower=1 if dimension in ('rave', 'cutoff-depth') else .05 if dimension == 'progressive-widening-alpha' else 0.,
                               upper=2**32-1 if dimension == 'rave' else 1. if dimension == 'progressive-widening-alpha' else max(1, horizon-1) if dimension == 'cutoff-depth' and horizon else None)
        elif dimension == 'exploration':
            values = sorted({.25, .5, 1., 1.4, 2., value}) if coarse else [max(0., value-max(.125, value/3)), value+max(.125, value/3)]
        elif dimension == 'rave':
            values = [max(1, value//3), min(2**32-1, round(value*10/3))]
        elif dimension == 'progressive-widening-k':
            values = [value/3, value*8/3]
        elif dimension == 'progressive-widening-alpha':
            values = [max(.05, value-.25), min(1., value+.25)]
        elif dimension == 'cutoff-depth':
            h = horizon or max(2, value*2)
            values = [max(1, min(h-1, round(h*f))) for f in (.1, .25, .5, .75)] if coarse else [max(1, round(value*.5)), max(1, min(h-1, round(value*1.5)))]
        else:
            raise ValueError(dimension)
        if dimension in ('rave', 'cutoff-depth'):
            values = [round(v) for v in values]
        candidates = [replace(base, **{field: v}) for v in sorted(set(values))]
    unique = []
    for candidate in candidates:
        assert_frozen(base, candidate, dimension)
        if candidate != base and candidate not in unique:
            unique.append(candidate)
    return unique
