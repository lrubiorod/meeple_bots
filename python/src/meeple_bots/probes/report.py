"""Game-independent aggregation and human rendering; never strategic assertions."""
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

def render(summaries):
    lines = ['Behavioral probes — measurements, no strategic verdicts.',
             'Q: root-player utility; unvisited Q and unavailable metrics are shown as —.',
             'Tables show focus + top-3 selections + tied dominants; search and raw results include every legal action.']
    groups = defaultdict(list)
    for summary in summaries:
        groups[(summary['probe_id'], summary.get('variant', 'probe'))].append(summary)
    def fmt(value):
        return '—' if value is None else f'{value:.3f}'
    for (probe_id, variant), budgets in groups.items():
        lines.append(f'\nProbe: {probe_id}; variant: {variant}')
        agent = budgets[0]['agent']
        lines.append(f'Agent: {agent["type"]}; selector: {agent["selection_policy"]}; root player: {budgets[0]["root_player"]}')
        for row in budgets:
            budget = f'Iterations: {row["iterations"]:,}' if row["iterations"] is not None else f'Decision time: {row["decision_seconds"]:g}s'
            lines.append(f'  Search cost: {row["iterations_per_second"]:,.0f} iter/s; median {row["median_iterations"]:g} iterations; median latency {1000*row["median_decision_seconds"]:.2f} ms')
            lines += [f'  {budget}; search seeds: {row["runs"]}',
                      '  Action                         Role             Selected       Median visits  Median avail  Median Q']
            roles = action_roles(row['actions'])
            for a in important_actions([row]):
                share = f'{a["selected_count"]}/{row["runs"]} ({a["selected_percentage"]:.1f}%)'
                lines.append(f'  {a["label"]:<30} {roles[action_key(a["action"])]:<16} {share:<16} {fmt(a["median_visits"]):>12} {fmt(a["median_availability"]):>13} {fmt(a["median_q"]):>9}')
            best = max(a['selected_count'] for a in row['actions'])
            leaders = [a for a in row['actions'] if a['selected_count'] == best]
            lines.append('  Dominant selection' + (' (tie)' if len(leaders) > 1 else '') + ': ' +
                         ', '.join(f'{a["label"]} ({a["selected_percentage"]:.1f}%)' for a in leaders))
        if len(budgets) > 1:
            lines.append('  Selection share across budgets (all seeds):')
            lines.append('  ' + f'{"Action":<30}' + ''.join(f'{r["iterations"]:>10,}' for r in budgets))
            for a in important_actions(budgets):
                shares = [next((e['selected_percentage'] for e in r['actions'] if action_key(e['action']) == action_key(a['action'])), 0) for r in budgets]
                lines.append('  ' + f'{a["label"]:<30}' + ''.join(f'{s:>9.1f}%' for s in shares))
    return '\n'.join(lines) + '\n'


def render_variants(summaries):
    """Compact cross-config selection shares at each fixed budget."""
    groups = defaultdict(list)
    for row in summaries:
        groups[(row['probe_id'], row['iterations'], row.get('decision_seconds'))].append(row)
    lines = ['Named variant comparison; selection shares are measurements, not strength.']
    for (probe, iterations, seconds), rows in groups.items():
        lines.append(f'\n{probe}: {iterations} iterations' if iterations is not None else f'\n{probe}: {seconds}s/decision')
        lines.append(f'{"Action":<38}' + ''.join(f'{r["variant"]:>20}' for r in rows))
        focus = important_actions(rows)
        for action in focus:
            shares = [next((a['selected_percentage'] for a in r['actions'] if action_key(a['action']) == action_key(action['action'])), 0) for r in rows]
            lines.append(f'{action["label"]:<38}' + ''.join(f'{s:>19.1f}%' for s in shares))
        if len(focus) < len(rows[0]['actions']):
            keys = {action_key(a['action']) for a in focus}
            shares = [sum(a['selected_percentage'] for a in r['actions'] if action_key(a['action']) not in keys) for r in rows]
            lines.append(f'{"Other":<38}' + ''.join(f'{s:>19.1f}%' for s in shares))
    return '\n'.join(lines) + '\n\n' + render(summaries)
