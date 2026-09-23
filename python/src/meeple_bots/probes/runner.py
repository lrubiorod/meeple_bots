"""Independent fixed-observation searches, with streaming raw persistence."""
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from ..serialization import agent_dict
from .core import action_dict
from .report import action_key, aggregate, render


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
               progress=print, search=observation_search):
    cases, iterations = tuple(cases), tuple(iterations)
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError('provide nonempty cases with unique IDs')
    if not iterations or len(set(iterations)) != len(iterations) or any(type(n) is not int or not 1 <= n <= 2**32-1 for n in iterations):
        raise ValueError('iterations must be distinct positive u32 budgets')
    if type(seeds) is not int or seeds < 1 or type(seed) is not int or not 0 <= seed <= 2**64-seeds:
        raise ValueError('search seeds must form a nonempty u64 range')
    if search is observation_search and not callable(getattr(agent, 'search', None)):
        raise ValueError('this agent needs an observation-only probe search adapter')
    configs = [replace(agent, iterations=n, time_budget=None) for n in iterations]
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'probe output already exists: {output}; use a new directory')
    positions = [c.builder() for c in cases]
    for case, position in zip(cases, positions):
        if not position.legal_actions or any(a not in position.legal_actions for a in case.candidate_actions):
            raise ValueError(f'{case.id}: invalid legal/candidate actions')
    # Exact executable fingerprint helps distinguish future baseline captures.
    from .. import _native
    native_path = Path(_native.__file__)
    metadata = {'version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'native_sha256': sha256(native_path.read_bytes()).hexdigest(),
                'iterations': list(iterations), 'search_seeds': list(range(seed, seed+seeds)),
                'agent': agent_dict('probe', agent), 'q_orientation': 'root_player',
                'fresh_search_per_run': True,
                'probes': [{'id': c.id, 'game': c.game, 'description': c.description, 'tags': c.tags,
                            'notes': c.notes, 'root_player': p.root_player,
                            'candidate_actions': [action_dict(a) for a in c.candidate_actions],
                            'observation': asdict(p.observation) if is_dataclass(p.observation) else p.observation,
                            'fixture': p.fixture} for c, p in zip(cases, positions)]}
    output.mkdir(parents=True, exist_ok=False)
    def write(name, data):
        (output/name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    write('metadata.json', metadata)
    runs = []
    with (output/'runs.jsonl').open('x') as raw:
        for index, (case, position) in enumerate(zip(cases, positions), 1):
            progress(f'Probe {index}/{len(cases)}: {case.id}')
            legal = {action_key(action_dict(a)): a for a in position.legal_actions}
            focus = {action_key(action_dict(a)) for a in case.candidate_actions}
            for config in configs:
                last_update = perf_counter()
                for count, search_seed in enumerate(range(seed, seed+seeds), 1):
                    started = perf_counter()
                    result = search(config, position.observation, position.legal_actions, seed=search_seed)
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
                                     'q': edge['q'] if edge['visits'] else None})
                    row = {'probe_id': case.id, 'game': case.game, 'root_player': position.root_player,
                           'agent': agent_dict('probe', config), 'iterations': config.iterations,
                           'search_seed': search_seed, 'selected_action': selected,
                           'root_visits': result['root_visits'], 'elapsed_seconds': elapsed,
                           'diagnostics': result.get('diagnostics', {}), 'root_actions': root}
                    raw.write(json.dumps(row, allow_nan=False) + '\n')
                    raw.flush()
                    runs.append(row)
                    now = perf_counter()
                    if count == seeds or now-last_update >= 5:
                        progress(f'  {config.iterations:,} iterations: {count}/{seeds} complete')
                        last_update = now
    summaries = aggregate(runs)
    write('summary.json', summaries)
    (output/'report.txt').write_text(render(summaries))
    return summaries
