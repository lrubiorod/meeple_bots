"""Pure in-memory probe action identity, aggregation, and display selection."""
import json
from collections import defaultdict
from statistics import median


def action_key(action):
    return json.dumps(action, sort_keys=True, separators=(',', ':'))


def aggregate(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[(run['probe_id'], run.get('variant', 'probe'), run['iterations'], run.get('decision_seconds'))].append(run)
    summaries = []
    for (probe_id, variant, iterations, decision_seconds), rows in groups.items():
        actions = []
        indexes = [{action_key(e['action']): e for e in r['root_actions']} for r in rows]
        keys = dict.fromkeys(key for index in indexes for key in index)
        for key in keys:
            samples = [index[key] for index in indexes if key in index]
            edge = samples[0]
            selected = sum(action_key(r['selected_action']) == key for r in rows)
            def mid(field):
                values = [e.get(field) for e in samples if e.get(field) is not None]
                return median(values) if values else None
            actions.append({'action': edge['action'], 'label': edge.get('label', key),
                            'focused': any(e.get('focused', False) for e in samples),
                            'selected_count': selected, 'selected_percentage': 100 * selected / len(rows),
                            'median_visits': mid('visits'), 'median_availability': mid('availability'),
                            'median_q': mid('q')})
        seconds = [r.get('diagnostics', {}).get('search_seconds', r['elapsed_seconds']) for r in rows]
        work = [r.get('diagnostics', {}).get('completed_iterations', r['iterations']) for r in rows]
        summaries.append({'variant': variant, 'decision_seconds': decision_seconds,
                          'median_iterations': median(work), 'median_decision_seconds': median(seconds),
                          'iterations_per_second': sum(work) / sum(seconds) if sum(seconds) else 0, 'probe_id': probe_id, 'iterations': iterations, 'runs': len(rows),
                          'agent': rows[0]['agent'], 'root_player': rows[0]['root_player'], 'actions': actions})
    return summaries



def action_roles(actions, top_k=3):
    """Focus plus selected top-K and every tied leader; ties use canonical ordering."""
    if top_k < 1:
        raise ValueError('top_k must be positive')
    ranked = sorted(actions, key=lambda a: (-a['selected_count'], action_key(a['action'])))
    maximum = ranked[0]['selected_count'] if ranked else 0
    top = {action_key(a['action']) for a in ranked[:top_k] if a['selected_count']}
    roles = {}
    for a in actions:
        labels = []
        if a['focused']:
            labels.append('focus')
        if a['selected_count'] == maximum and maximum:
            labels.append('dominant')
        elif action_key(a['action']) in top:
            labels.append('top')
        if labels:
            roles[action_key(a['action'])] = '+'.join(labels)
    return roles


def important_actions(rows, top_k=3):
    keys = set().union(*(set(action_roles(r['actions'], top_k)) for r in rows))
    actions = {action_key(a['action']): a for r in rows for a in r['actions']}
    return [actions[k] for k in sorted(keys)]

