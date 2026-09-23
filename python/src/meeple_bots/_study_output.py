"""Human study presentation; no candidate, evidence or allocation decisions."""
from statistics import mean


def compute_budget(state):
    request, cal = state['request'], state['calibration']
    source = ('explicit --decision-time' if request['decision_seconds'] is not None else
              'baseline/config' if request['baseline_supplied'] else '--target-match-time')
    fixed = cal.get('fixed_iterations')
    timings = cal.get('position_timings', [])
    seconds = (mean(t['milliseconds'] for t in timings) / 1000 if timings else None) if fixed else cal['decision_seconds']
    decisions = cal.get('estimated_game_decisions', cal['mean_plies'])
    return dict(source=source, iterations=fixed, decision_seconds=None if fixed else seconds,
                estimated_decision_seconds=seconds, estimated_player_decisions=decisions,
                estimated_search_game_seconds=seconds * decisions if seconds is not None else None,
                target_match_time_active=source == '--target-match-time')


def announce_plan(runner):
    from .studies import MAX_EXTENSION_ROUNDS, _phase_enabled
    r, c = runner.state['request'], runner.state['calibration']
    out = runner.progress
    def field(label, value):
        out(f'  {label + ":":<29}{value}')
    b = compute_budget(runner.state)
    out('Study\nTarget')
    field('Game', r['game'])
    field('Agent family', 'SO-ISMCTS' if runner.family == 'so_ismcts' else 'MCTS')
    mode = ('local retune: ' + r['tune'] if r.get('tune') else
            'incremental baseline optimization' if r['baseline_supplied'] else r['mode'].replace('_', ' '))
    field('Mode', mode)
    field('Total study budget', f'{runner.budget:g}s' if runner.budget is not None else 'unlimited')
    field('Selection policies', ', '.join(r['selection_policies']))
    out('Supported tuning')
    out('  ' + ', '.join(r['supported_tuners']))
    unsupported = sorted({'tree_reuse', 'transpositions', 'rave', 'progressive_widening'} - set(runner.family_profile.mechanisms))
    if unsupported:
        field('Unsupported by family', ', '.join(unsupported))
    out('Search horizon')
    horizon = c['horizon']
    depth = c.get('operating_point', {}).get('depth', horizon['depth'])
    field('Rollout depth', f'{depth} plies' if depth is not None else 'no configurable rollout horizon')
    cutoff = r['mode'] == 'heuristic_cutoff' and horizon['kind'] != 'baseline'
    field('Mode', 'heuristic cutoff' if cutoff else horizon['kind'].replace('_', ' '))
    if depth != horizon['depth']:
        field('Reference full horizon', f"{horizon['depth']} plies")
        field('Reference horizon mode', horizon['kind'].replace('_', ' '))
    if 'minimum_position_terminal_fraction' in horizon:
        field('Min probe terminal reach', f"{horizon['minimum_position_terminal_fraction']:.1%} (reference horizon)")
    if 'target_terminal_fraction' in horizon:
        field('Probe terminal criterion', f">= {horizon['target_terminal_fraction']:.0%} at every sampled position")
    out('Compute budget')
    if b['iterations']:
        field('Iterations/decision', f"{b['iterations']:,} (fixed)")
        if b['estimated_decision_seconds'] is not None:
            field('Measured decision time', f"~{b['estimated_decision_seconds']*1000:g} ms (not a time limit)")
    else:
        field('Derived decision time' if b['target_match_time_active'] else 'Decision time',
              f"{b['decision_seconds']*1000:g} ms")
    field('Source', b['source'])
    field('Estimated decisions', f"{b['estimated_player_decisions']:g}")
    if b['target_match_time_active']:
        field('Target match time', f"{r['target_match_time']:g}s")
        field('Safety margin', f"{r['safety_margin']:.2f}")
        field('Derivation', f"{r['target_match_time']:g} / ({b['estimated_player_decisions']:g} × {r['safety_margin']:g}) s/decision")
    else:
        field('Target match time', f"{r['target_match_time']:g}s (overridden; not used)")
    if b['estimated_search_game_seconds'] is not None:
        field('Estimated search/game', f"~{b['estimated_search_game_seconds']:.2f}s")
    out('  Search compute estimate, not a hard wall-clock match limit.')
    out('Search calibration')
    a = c['search_adequacy']
    field('Adequacy', a['category'] + ' (search population, not strength)')
    field('Median iterations', f"{a['median_iterations_per_decision']:,.0f} / decision")
    field('Representative branching', f"{a['representative_branching']:g}")
    for warning in c.get('warnings', []):
        out('  Warning: ' + warning)
    out('Planned stages')
    out('  Calibration')
    specs = r['tuner_specs']
    if r.get('tune'):
        field('Local retune', r['tune'])
        out(f'    Adaptive dimensions: initial round + max {MAX_EXTENSION_ROUNDS} extensions; categorical dimensions: one round.')
    elif runner.profile:
        dimensions = {s['dimension'] for n, s in specs.items() if not n.startswith('second-')}
        for dimension in ('exploration', 'selection', 'tree-reuse'):
            field(dimension.replace('-', ' ').capitalize(), 'enabled' if dimension in dimensions else 'disabled')
    else:
        for label, phase, flag in [('Depth screen', 'depth_screen', 'depth-search'),
                                   ('Exploration', 'exploration', 'selection-search'),
                                   ('Selection', 'selectors', 'selection-search'),
                                   ('RAVE', 'rave', 'rave-search'),
                                   ('Structure', 'mechanisms', 'mechanism-search'),
                                   ('Progressive widening', 'pw_screen', 'pw-search')]:
            active = phase in runner.phase_names and _phase_enabled(phase, r)
            unavailable = phase in ('rave', 'pw_screen') and 'uct_rave' not in r['selection_policies']
            status = 'enabled' if active else 'unsupported by game/backend' if unavailable else f'disabled (enable with --{flag})'
            if phase == 'depth_screen' and r['mode'] != 'heuristic_cutoff':
                status = 'not applicable to this horizon mode'
            field(label, status)
            if active and phase == 'rave':
                out(f'    equivalence: initial range + max {MAX_EXTENSION_ROUNDS} adaptive extensions; exploration; compare')
            if active and phase == 'pw_screen':
                out(f'    screen; tune k (max {MAX_EXTENSION_ROUNDS} extensions); tune alpha (max {MAX_EXTENSION_ROUNDS} extensions); refine; compare')
        if any(n.startswith('admission-') for n in specs):
            field('Widening expansion', 'enabled')
    if r['second_pass']:
        dimensions = dict.fromkeys(s['dimension'] for n, s in specs.items() if n.startswith('second-'))
        field('Second pass', ', '.join(dimensions))
        out(f'    Adaptive chains: initial round + max {MAX_EXTENSION_ROUNDS} extensions; final joint recheck unchanged.')
    if 'confirmation' in runner.phase_names:
        out('  Confirmation')
    if r['vs_random']:
        out('  Random baseline: final retained champion; diagnostic only; after competitive stages.')
    out('Comparison policy')
    out('  Equal fixed iterations' if b['iterations'] else '  Equal wall-clock decision budgets')
    field('Games/comparison', max(8, r['games_per_comparison']))
    if r['stage_games']:
        field('Stage game overrides', ', '.join(f'{k}={max(8,v)}' for k,v in r['stage_games'].items()))
    field('Workers', f"{r['workers']} ({'shared_cpu' if r['workers'] > 1 else 'isolated'})")
    out('  Paired seeds and swapped seats. Budget pressure removes candidates, not evidence.')


def announce_extension(runner, phase, completed=False):
    from .studies import MAX_EXTENSION_ROUNDS, _group_leaders
    name = phase['name']
    fields = {'rave': 'rave_equivalence', 'pw_k': 'progressive_widening_k', 'pw_alpha': 'progressive_widening_alpha'}
    if '_extend_' in name:
        stem, suffix = name.rsplit('_extend_', 1)
        field, round_index = fields.get(stem), int(suffix)
        earlier = [p for n, p in runner.state['phases'].items() if n == stem or n.startswith(stem + '_extend_')]
    elif phase.get('round', 0) and phase.get('chain'):
        from ._study_tuners import TUNING_FIELDS
        field, round_index = TUNING_FIELDS[phase['dimension']][0], phase['round']
        earlier = [p for p in runner.state['phases'].values() if p.get('chain') == phase['chain']]
    else:
        return
    if not field or not phase['contrasts']:
        return
    prefix = name.rsplit('_extend_', 1)[0] if '_extend_' in name else phase['dimension']
    if not completed:
        tested = [v[field] for p in earlier if p is not phase for v in p['agents'].values() if field in v]
        expanding = tested and any(phase['agents'][c['b']][field] < min(tested) or phase['agents'][c['b']][field] > max(tested) for c in phase['contrasts'])
        reason = 'Expanding boundary range beyond previously tested values' if expanding else 'Refining untested neighboring values'
        runner.progress(f'{prefix}: {reason} ({round_index}/{MAX_EXTENSION_ROUNDS}).')
    elif round_index == MAX_EXTENSION_ROUNDS:
        phase['extension_limit_reached'] = True
        boundary = False
        for group, leader in _group_leaders(phase).items():
            members = phase['groups'][group]
            values = [phase['agents'][m][field] for m in members]
            challengers = {c['b'] for c in phase['contrasts']}
            boundary |= leader in challengers and phase['agents'][leader][field] in (min(values), max(values))
        reason = 'Boundary candidate won at extension limit' if boundary else 'Adaptive extension limit reached'
        runner.progress(f'{prefix}: {reason} ({round_index}/{MAX_EXTENSION_ROUNDS}). Keeping best supported candidate; no further expansion in this chain.')
