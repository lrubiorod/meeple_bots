"""Offline comparison of archived probe captures. No search or game dependencies."""
import json
from pathlib import Path

from .report import action_key, action_roles, aggregate, important_actions


def _budget(row):
    return (row['iterations'], row.get('decision_seconds'))


def compare_captures(inputs, *, baseline=None, top_k=3):
    if len(inputs) < 2 or any(not name.strip() for name in inputs):
        raise ValueError('comparison requires at least two named inputs')
    if top_k < 1:
        raise ValueError('top_k must be positive')
    baseline = baseline or next(iter(inputs))
    if baseline not in inputs:
        raise ValueError('baseline must name an input variant')
    names = [baseline, *sorted(set(inputs) - {baseline})]
    sources, captures, warnings = {}, {}, []
    for name in names:
        path = Path(inputs[name])
        meta = json.loads((path / 'metadata.json').read_text())
        if meta.get('version', 1) != 1:
            raise ValueError(f'{name}: unsupported capture schema version')
        if meta.get('q_orientation', 'root_player') != 'root_player':
            raise ValueError(f'{name}: incompatible Q orientation')
        for field in ('q_orientation', 'version'):
            if field not in meta:
                warnings.append(f'{name}: missing {field}; assuming legacy version-1 conventions')
        rows = [json.loads(line) for line in (path / 'runs.jsonl').read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f'{name}: no saved runs')
        seen = set()
        for row in rows:
            identity = (row['probe_id'], _budget(row), row['search_seed'])
            if identity in seen:
                raise ValueError(f'{name}: duplicate probe/budget/seed record: {identity}')
            seen.add(identity)
            keys = [action_key(e['action']) for e in row['root_actions']]
            if len(keys) != len(set(keys)) or action_key(row['selected_action']) not in keys:
                raise ValueError(f'{name}: invalid root action identities or selected action')
            row['variant'] = name
        sources[name] = {'directory': str(path), 'metadata': meta}
        captures[name] = rows

    reference = captures[baseline]
    game_set = {r['game'] for r in reference}
    probe_ids = {r['probe_id'] for r in reference}
    if len(game_set) != 1:
        raise ValueError('comparison requires one game')
    game = next(iter(game_set))
    reference_probes = {p['id']: p for p in sources[baseline]['metadata'].get('probes', [])}
    for name in names:
        rows = captures[name]
        ref_meta = sources[baseline]['metadata']
        own_meta = sources[name]['metadata']
        for field in ('action_schema_version', 'fixture_version'):
            if field in ref_meta and field in own_meta and ref_meta[field] != own_meta[field]:
                raise ValueError(f'{name}: incompatible {field}')
        if {r['game'] for r in rows} != {game}:
            raise ValueError(f'{name}: incompatible games')
        if {r['probe_id'] for r in rows} != probe_ids:
            raise ValueError(f'{name}: incompatible probe IDs')
        probes = {p['id']: p for p in sources[name]['metadata'].get('probes', [])}
        for probe in sorted(probe_ids):
            roots = {r['root_player'] for r in reference + rows if r['probe_id'] == probe}
            if len(roots) != 1:
                raise ValueError(f'{name}/{probe}: incompatible root players')
            a, b = reference_probes.get(probe, {}), probes.get(probe, {})
            for field in ('observation', 'fixture_version', 'action_schema_version'):
                if field in a and field in b and action_key(a[field]) != action_key(b[field]):
                    raise ValueError(f'{name}/{probe}: incompatible {field}')
            if 'observation' not in a or 'observation' not in b:
                if 'fixture' in a and 'fixture' in b and a['fixture'] != b['fixture']:
                    raise ValueError(f'{name}/{probe}: incompatible fixture')
                warnings.append(f'{name}/{probe}: observation unavailable; state compatibility cannot be fully verified')
            for row in (r for r in rows if r['probe_id'] == probe):
                if b.get('root_player', row['root_player']) != row['root_player']:
                    raise ValueError(f'{name}/{probe}: metadata root player differs from raw runs')
                # Historical annotations may live only in metadata.
                focus = {action_key(x) for x in b.get('candidate_actions', [])}
                for edge in row['root_actions']:
                    edge['focused'] = edge.get('focused', False) or action_key(edge['action']) in focus

    summaries = aggregate([r for name in names for r in captures[name]])
    lookup = {(s['probe_id'], _budget(s), s['variant']): s for s in summaries}
    groups = []
    for probe in sorted(probe_ids):
        budgets = {name: {_budget(r) for r in captures[name] if r['probe_id'] == probe} for name in names}
        union = set.union(*budgets.values())
        for budget in sorted(union, key=lambda b: (b[0] is None, b[0] or 0, b[1] or 0)):
            available = [name for name in names if budget in budgets[name]]
            missing = sorted(set(names) - set(available))
            if missing:
                warnings.append(f'{probe}/{budget}: missing budgets for {", ".join(missing)}; no comparison for absent samples')
            rows = [lookup[probe, budget, name] for name in available]
            seeds = {name: sorted(r['search_seed'] for r in captures[name]
                                 if r['probe_id'] == probe and _budget(r) == budget) for name in available}
            if any(s != seeds[available[0]] for s in seeds.values()):
                warnings.append(f'{probe}/{budget}: different seed sets: {seeds}')
            root_sets = {}
            for name in available:
                selected_rows = [r for r in captures[name] if r['probe_id'] == probe and _budget(r) == budget]
                sets = [{action_key(e['action']) for e in r['root_actions']} for r in selected_rows]
                root_sets[name] = set.union(*sets)
                if any(s != sets[0] for s in sets):
                    warnings.append(f'{name}/{probe}/{budget}: root action sets differ between seeds')
            reference_set = root_sets[available[0]]
            for name in available[1:]:
                if root_sets[name] != reference_set:
                    warnings.append(f'{probe}/{budget}: root action set mismatch {name} vs {available[0]}; '
                                    f'missing={sorted(reference_set-root_sets[name])}; additional={sorted(root_sets[name]-reference_set)}')
            leaders = {}
            for row in rows:
                best = max(a['selected_count'] for a in row['actions'])
                leaders[row['variant']] = [a['action'] for a in row['actions'] if a['selected_count'] == best]
                roles = action_roles(row['actions'], top_k)
                for action in row['actions']:
                    action['dominant'] = action['selected_count'] == best
                    action['role'] = roles.get(action_key(action['action']), '')
            base_actions = {action_key(a['action']): a for a in lookup.get((probe, budget, baseline), {}).get('actions', [])}
            for row in rows:
                for action in row['actions']:
                    base = base_actions.get(action_key(action['action']))
                    action['share_delta_pp'] = action['selected_percentage'] - base['selected_percentage'] if base else None
                    action['median_q_delta'] = (action['median_q'] - base['median_q']
                                                if base and action['median_q'] is not None and base['median_q'] is not None else None)
            signatures = {name: tuple(sorted(action_key(a) for a in values)) for name, values in leaders.items()}
            pairs = [f'{a}/{b} same' for i, a in enumerate(available) for b in available[i+1:] if signatures[a] == signatures[b]]
            agreement = ('unavailable' if len(available) < 2 else 'all same' if len(set(signatures.values())) == 1
                         else ', '.join(pairs) if pairs else 'all different')
            groups.append({'probe_id': probe, 'iterations': budget[0], 'decision_seconds': budget[1],
                           'seed_sets': seeds, 'dominant_actions': leaders, 'dominant_agreement': agreement,
                           'variants': rows})
    return {'version': 1, 'game': game, 'baseline': baseline, 'top_k': top_k,
            'q_orientation': 'root_player', 'sources': sources, 'warnings': warnings, 'groups': groups,
            'seed_selections': [{'variant': name, 'probe_id': r['probe_id'], 'iterations': r['iterations'],
                                 'decision_seconds': r.get('decision_seconds'), 'search_seed': r['search_seed'],
                                 'selected_action': r['selected_action']} for name in names
                                for r in sorted(captures[name], key=lambda r: (r['probe_id'], str(_budget(r)), r['search_seed']))]}


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


def write_comparison(inputs, *, output, baseline=None, top_k=3):
    output = Path(output)
    if output.exists():
        raise ValueError(f'output already exists: {output}')
    data = compare_captures(inputs, baseline=baseline, top_k=top_k)
    report = render_comparison(data)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'comparison.json').write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    (output / 'comparison.txt').write_text(report)
    return data
