"""Independent fixed-observation searches, with streaming raw persistence."""
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from math import isfinite

from ..serialization import agent_dict
from .core import action_dict
from .artifacts import (capture_metadata, write_capture_json, open_runs,
                        append_run, write_capture_report, write_variant_artifacts)
from .metrics import action_key, aggregate
from .report import render


def observation_search(agent, observation, legal_actions, *, seed):
    """Reuse the normal fresh observation-only API; no state or fixture argument.

    Other search APIs can supply an adapter with this same signature and return
    action, root_visits, root_actions and diagnostics. Q must face the root player.
    Availability is optional. No adapter may filter legal_actions.
    """
    result = agent.search(observation, legal_actions, seed=seed)
    root = result['nodes'][0]
    return {'action': result['action'], 'root_visits': root['visits'],
            'root_actions': root['edges'], 'diagnostics': result['diagnostics']}


def run_probes(cases, agent, *, iterations=(1000,), seeds=32, seed=0, output,
               progress=print, search=observation_search, decision_seconds=None, variant_name="probe"):
    cases, iterations = tuple(cases), tuple(iterations)
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError('provide nonempty cases with unique IDs')
    if not iterations or len(set(iterations)) != len(iterations) or any(type(n) is not int or not 1 <= n <= 2**32-1 for n in iterations):
        raise ValueError('iterations must be distinct positive u32 budgets')
    if type(seeds) is not int or seeds < 1 or type(seed) is not int or not 0 <= seed <= 2**64-seeds:
        raise ValueError('search seeds must form a nonempty u64 range')
    if decision_seconds is not None and (not isfinite(decision_seconds) or decision_seconds <= 0):
        raise ValueError('decision time must be finite and positive')
    if search is observation_search and not callable(getattr(agent, 'search', None)) and any(c.search is None for c in cases):
        raise ValueError('this agent needs an observation-only probe search adapter')
    configs = ([replace(agent, iterations=None, time_budget=decision_seconds)] if decision_seconds is not None else
               [replace(agent, iterations=n, time_budget=None) for n in iterations])
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'probe output already exists: {output}; use a new directory')
    positions = [c.builder() for c in cases]
    for case, position in zip(cases, positions):
        if not position.legal_actions or any(a not in position.legal_actions for a in case.candidate_actions):
            raise ValueError(f'{case.id}: invalid legal/candidate actions')
    metadata = capture_metadata(cases, positions, agent, iterations, decision_seconds,
                                variant_name, seed, seeds)
    output.mkdir(parents=True, exist_ok=False)
    write_capture_json(output, 'metadata.json', metadata)
    runs = []
    with open_runs(output) as raw:
        for index, (case, position) in enumerate(zip(cases, positions), 1):
            progress(f'Probe {index}/{len(cases)}: {case.id}')
            legal = {action_key(action_dict(a)): a for a in position.legal_actions}
            focus = {action_key(action_dict(a)) for a in case.candidate_actions}
            for config in configs:
                last_update = perf_counter()
                for count, search_seed in enumerate(range(seed, seed+seeds), 1):
                    started = perf_counter()
                    result = (case.search or search)(config, position.observation, position.legal_actions, seed=search_seed)
                    elapsed = perf_counter() - started
                    selected = action_dict(result['action'])
                    if action_key(selected) not in legal:
                        raise ValueError('search returned an illegal action')
                    edges = {action_key(action_dict(e['action'])): e for e in result['root_actions']}
                    if set(edges) != set(legal):
                        raise ValueError('root diagnostics must include every legal action')
                    root = []
                    for key, action in legal.items():
                        edge = edges[key]
                        root.append({'action': action_dict(action), 'label': case.action_label(action),
                                     'focused': key in focus, 'visits': edge['visits'],
                                     'availability': edge.get('availability'),
                                     'q': edge['q'] if edge['visits'] else None,
                                     **{k: edge[k] for k in ('heuristic_value', 'progressive_bias') if k in edge}})
                    row = {'probe_id': case.id, 'game': case.game, 'root_player': position.root_player,
                           'agent': agent_dict(variant_name, config), 'iterations': config.iterations,
                           'decision_seconds': config.time_budget, 'variant': variant_name,
                           'search_seed': search_seed, 'selected_action': selected,
                           'root_visits': result['root_visits'], 'elapsed_seconds': elapsed,
                           'diagnostics': result.get('diagnostics', {}), 'root_actions': root}
                    append_run(raw, row)
                    runs.append(row)
                    now = perf_counter()
                    if count == seeds or now-last_update >= 5:
                        budget_label = f'{config.iterations:,} iterations' if config.iterations is not None else f'{config.time_budget:g}s/decision'
                        progress(f'  {budget_label}: {count}/{seeds} complete')
                        last_update = now
    summaries = aggregate(runs)
    write_capture_json(output, 'summary.json', summaries)
    write_capture_report(output, render(summaries))
    return summaries


def run_variants(cases, variants, *, output, **kwargs):
    """Run named configs over the same cases/seeds; each capture remains standalone."""
    from .report import render_variants
    cases = tuple(cases)
    output = Path(output)
    if not variants or any(not name or not all(c.isalnum() or c in '_-' for c in name) for name in variants):
        raise ValueError('variant names must contain only letters, numbers, _ or -')
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for name, agent in variants.items():
        summaries.extend(run_probes(cases, agent, output=output/name, variant_name=name, **kwargs))
    write_variant_artifacts(output, summaries, render_variants(summaries))
    return summaries
