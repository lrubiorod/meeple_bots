"""Game-independent aggregation and human rendering; never strategic assertions."""
import json
from collections import defaultdict
from statistics import median


def action_key(action):
    return json.dumps(action, sort_keys=True, separators=(',', ':'))


def aggregate(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[(run['probe_id'], run['iterations'])].append(run)
    summaries = []
    for (probe_id, iterations), rows in groups.items():
        actions = []
        for edge in rows[0]['root_actions']:
            key = action_key(edge['action'])
            samples = [next(e for e in r['root_actions'] if action_key(e['action']) == key) for r in rows]
            selected = sum(action_key(r['selected_action']) == key for r in rows)
            def mid(field):
                values = [e[field] for e in samples if e[field] is not None]
                return median(values) if values else None
            actions.append({'action': edge['action'], 'label': edge['label'], 'focused': edge['focused'],
                            'selected_count': selected, 'selected_percentage': 100 * selected / len(rows),
                            'median_visits': mid('visits'), 'median_availability': mid('availability'),
                            'median_q': mid('q')})
        summaries.append({'probe_id': probe_id, 'iterations': iterations, 'runs': len(rows),
                          'agent': rows[0]['agent'], 'root_player': rows[0]['root_player'], 'actions': actions})
    return summaries


def render(summaries):
    lines = ['Behavioral probes — measurements, no strategic verdicts.',
             'Q: root-player utility; unvisited Q and unavailable metrics are shown as —.',
             'Focus filters tables only; search and raw results include every legal action.']
    groups = defaultdict(list)
    for summary in summaries:
        groups[summary['probe_id']].append(summary)
    def fmt(value):
        return '—' if value is None else f'{value:.3f}'
    for probe_id, budgets in groups.items():
        lines.append(f'\nProbe: {probe_id}')
        agent = budgets[0]['agent']
        lines.append(f'Agent: {agent["type"]}; selector: {agent["selection_policy"]}; root player: {budgets[0]["root_player"]}')
        for row in budgets:
            lines += [f'  Iterations: {row["iterations"]:,}; search seeds: {row["runs"]}',
                      '  Action                         Selected       Median visits  Median avail  Median Q']
            focused = [a for a in row['actions'] if a['focused']]
            for a in focused or row['actions']:
                share = f'{a["selected_count"]}/{row["runs"]} ({a["selected_percentage"]:.1f}%)'
                lines.append(f'  {a["label"]:<30} {share:<16} {fmt(a["median_visits"]):>12} {fmt(a["median_availability"]):>13} {fmt(a["median_q"]):>9}')
            best = max(a['selected_count'] for a in row['actions'])
            leaders = [a for a in row['actions'] if a['selected_count'] == best]
            lines.append('  Dominant selection' + (' (tie)' if len(leaders) > 1 else '') + ': ' +
                         ', '.join(f'{a["label"]} ({a["selected_percentage"]:.1f}%)' for a in leaders))
        if len(budgets) > 1:
            lines.append('  Selection share across budgets (all seeds):')
            lines.append('  ' + f'{"Action":<30}' + ''.join(f'{r["iterations"]:>10,}' for r in budgets))
            for a in budgets[0]['actions']:
                if any(e['focused'] for e in budgets[0]['actions']) and not a['focused']:
                    continue
                shares = [next(e['selected_percentage'] for e in r['actions'] if e['action'] == a['action']) for r in budgets]
                lines.append('  ' + f'{a["label"]:<30}' + ''.join(f'{s:>9.1f}%' for s in shares))
    return '\n'.join(lines) + '\n'
