"""Human rendering of precomputed behavioral probe summaries."""
import json
from collections import defaultdict
from .metrics import action_key, aggregate, action_roles, important_actions


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


def render_comparison(data):
    lines = ['Probe comparison', f'Game: {data["game"]}; baseline: {data["baseline"]}',
             'Q: root-player utility; estimates, not objective move quality.',
             'Fixed-iteration captures diagnose sample complexity, not equal-time competitive strength.',
             'Same numeric search seeds do not imply statistically paired samples.',
             'Missing metrics/actions: —. Share deltas are percentage points (pp).']
    baseline_config = data['sources'][data['baseline']]['metadata'].get('agent', {})
    for name in [data['baseline'], *sorted(set(data['sources']) - {data['baseline']})]:
        source = data['sources'][name]
        lines.append(f'{name}: {source["directory"]}')
        config = source['metadata'].get('agent', {})
        if name != data['baseline']:
            diff = {k: {'baseline': baseline_config.get(k), 'variant': config.get(k)}
                    for k in sorted(set(baseline_config) | set(config)) if baseline_config.get(k) != config.get(k)}
            lines.append(f'  Saved config differences vs {data["baseline"]}: {json.dumps(diff, sort_keys=True)}')
    lines.extend('WARNING: ' + warning for warning in data['warnings'])
    def fmt(value, precision=3):
        return '—' if value is None else f'{value:.{precision}f}'
    for group in data['groups']:
        budget = (f'{group["iterations"]:,} iterations' if group['iterations'] is not None
                  else f'{group["decision_seconds"]:g}s/decision')
        lines += [f'\nProbe: {group["probe_id"]} — {budget}', 'Dominant selection:']
        for row in group['variants']:
            leaders = [a for a in row['actions'] if a['dominant']]
            lines.append(f'  {row["variant"]}: ' + ' / '.join(a['label'] for a in leaders)
                         + f' ({leaders[0]["selected_percentage"]:.1f}% each)')
        lines.append('Dominant agreement: ' + group['dominant_agreement'])
        lines.append('Search cost:')
        for row in group['variants']:
            lines.append(f'  {row["variant"]}: {row["iterations_per_second"]:,.0f} iter/s; '
                         f'median latency {1000*row["median_decision_seconds"]:.2f} ms; {row["runs"]} seeds')
        lines.append('Action | Variant | Role | Selected (share) | Median visits | Median avail | Median Q | Share Δ vs baseline')
        indexes = {r['variant']: {action_key(a['action']): a for a in r['actions']} for r in group['variants']}
        for action in important_actions(group['variants'], data['top_k']):
            for row in group['variants']:
                a = indexes[row['variant']].get(action_key(action['action']))
                prefix = f'{action["label"]} | {row["variant"]} | '
                if a is None:
                    lines.append(prefix + 'missing | — | — | — | — | —')
                    continue
                delta = a['share_delta_pp']
                lines.append(prefix + f'{a["role"] or "comparison"} | {a["selected_count"]}/{row["runs"]} '
                             f'({a["selected_percentage"]:.1f}%) | {fmt(a["median_visits"], 1)} | '
                             f'{fmt(a["median_availability"], 1)} | {fmt(a["median_q"])} | '
                             + ('—' if delta is None else f'{delta:+.1f} pp'))
    return '\n'.join(lines) + '\n'
