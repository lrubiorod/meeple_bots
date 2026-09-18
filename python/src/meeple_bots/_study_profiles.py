"""Search-family policy for the shared study coordinator, not another executor."""
import json
from statistics import mean, median
from time import perf_counter
import tomllib

from ._agent_config import MctsAgent, SoIsmctsAgent
from ._capabilities import game_search_capabilities


# Capabilities decide eligibility; a future family can join this registry without
# tying game IDs permanently to one search algorithm.
FAMILY_REQUIREMENTS = {
    'mcts': lambda caps: 'mcts' in caps.get('search_agents', []) and not caps.get('imperfect_information'),
    'so_ismcts': lambda caps: 'so_ismcts' in caps.get('search_agents', []) and caps.get('imperfect_information', False),
}


def resolve_family(game, requested=None, baseline=None):
    caps = game_search_capabilities(game)
    compatible = [name for name, supports in FAMILY_REQUIREMENTS.items() if supports(caps)]
    if baseline is not None and not isinstance(baseline, (MctsAgent, SoIsmctsAgent)):
        raise ValueError('baseline must belong to a compatible search-agent family')
    inferred = 'so_ismcts' if isinstance(baseline, SoIsmctsAgent) else 'mcts' if isinstance(baseline, MctsAgent) else None
    family = requested or inferred or (compatible[0] if len(compatible) == 1 else None)
    if family not in compatible:
        raise ValueError(f'{family or "unspecified family"} is not a compatible study agent for {game}; compatible: {compatible}')
    if inferred and family != inferred:
        raise ValueError('agent family conflicts with baseline configuration')
    return family


def load_search_profile(path):
    from ._mcts_profiles import _load_mcts_profile
    values = tomllib.loads(path.read_text())
    if values.get('agent', 'mcts') == 'mcts':
        return _load_mcts_profile(path).agent
    if values.get('agent') != 'so_ismcts':
        raise ValueError('unknown search agent family')
    return so_from_values(values)


def so_from_values(values):
    unknown = values.keys() - {'agent', 'name', 'iterations', 'time_budget', 'exploration', 'rollout', 'root_selection'}
    if unknown:
        raise ValueError(f'unsupported SO-ISMCTS fields: {sorted(unknown)}')
    if values.get('rollout', 'uniform') != 'uniform' or values.get('root_selection', 'most_visited') != 'most_visited':
        raise ValueError('SO-ISMCTS requires uniform rollout and most_visited root selection')
    return SoIsmctsAgent(iterations=values.get('iterations'), time_budget=values.get('time_budget'),
                         exploration=values.get('exploration', 2**.5))


class SoIsmctsStudyProfile:
    name = 'so_ismcts'

    def specs(self):
        return {name: {'dimension': 'exploration', 'round': i, 'chain': 'exploration', 'coarse': i == 0}
                for i, name in enumerate(('exploration_coarse', 'exploration_refine_1', 'exploration_refine_2'))}

    def calibrate(self, runner):
        from .studies import _study_budget, profile_values
        from .study_analysis import search_adequacy
        if runner.state['calibration']:
            return
        request = runner.state['request']
        runner.progress('Study mode: ' + ('LOCAL RETUNE' if request['tune'] else 'FULL STUDY'))
        runner.progress('Agent family: SO-ISMCTS; game: ' + request['game'])
        runner.progress(f'Total study budget: {runner.budget}; target match time: {request["target_match_time"]}s')
        # The normal trace writer/executor persists the paired structural pilot too.
        pilot = runner.state.setdefault('calibration_progress', {}).setdefault('pilot', {
            'name': 'calibration', 'agents': {'random-a': None, 'random-b': None},
            'contrasts': [{'a': 'random-a', 'b': 'random-b'}], 'planned_pairs': 1})
        rows = runner._batch(pilot, [0], 0, pilot=True)[0]
        expected = mean(row['result']['plies'] for row in rows)
        fixed = runner.base.iterations if request['baseline_supplied'] and request['decision_seconds'] is None else None
        seconds = request['decision_seconds'] or (runner.base.time_budget if request['baseline_supplied'] else None)
        seconds = seconds or request['target_match_time'] / (expected * request['safety_margin'])
        cal = {'mean_plies': expected, 'estimated_game_decisions': expected,
               'target_match_time': request['target_match_time'], 'safety_margin': request['safety_margin'],
               'decision_seconds': seconds, 'fixed_iterations': fixed,
               'decision_time_source': 'baseline' if request['baseline_supplied'] else 'target_match_time',
               'horizon': {'kind': 'unbounded', 'depth': None}, 'pilot': pilot, 'position_timings': []}
        agent = _study_budget(runner.base, cal)
        # Sample only legitimate observations. Environment sampling and policy/search seeds
        # have separate streams; positions span both microphases and early/middle play.
        from random import Random
        environment, policy = Random(request['seed'] ^ 0x8EBC6AF0), Random(request['seed'] ^ 0xA0761D64)
        game = runner.game
        position = game.initial_state(request['seed'])
        targets = {0, max(1, round(expected/4)), max(2, round(expected/2))}
        decision = 0
        while position.status != 'terminal' and decision <= max(targets):
            if position.status == 'chance':
                position = game.apply_chance_outcome(position, game.sample_chance(position, environment.getrandbits(64)))
                continue
            legal = game.legal_actions(position)
            if decision in targets:
                if runner.budget is not None and runner.spent >= runner.budget:
                    return  # Resume can repeat unfinished throughput measurements; pilot is saved.
                observation = game.observation(position, position.current_player)
                start = perf_counter()
                result = agent.search(observation, legal, seed=request['seed'] + decision)
                elapsed = perf_counter() - start
                d = result['diagnostics']; root = result['nodes'][0]['edges']
                cal['position_timings'].append({'sampled_ply': decision, 'milliseconds': elapsed*1000,
                    'iterations': d['completed_iterations'], 'determinizations': d['determinizations_sampled'],
                    'legal_actions': len(legal), 'root_action_coverage': sum(e['visits'] > 0 for e in root)/len(legal),
                    'nodes': d['tree_nodes'], 'action_edges': d['action_edges'],
                    'terminal_simulations': d['terminal_simulations'], 'cutoff_simulations': d['cutoff_simulations']})
            position = game.apply_action(position, policy.choice(legal))
            decision += 1
        timings = cal['position_timings']
        elapsed = sum(t['milliseconds'] for t in timings)/1000
        iterations = sum(t['iterations'] for t in timings)
        cal.update(seconds_per_iteration=elapsed/iterations, iterations_per_second=iterations/elapsed,
                   determinizations_per_second=sum(t['determinizations'] for t in timings)/elapsed,
                   median_iterations_per_decision=median(t['iterations'] for t in timings),
                   median_determinizations_per_decision=median(t['determinizations'] for t in timings),
                   root_action_coverage=median(t['root_action_coverage'] for t in timings),
                   search_adequacy=search_adequacy(timings, request['target_match_time']))
        cal['warnings'] = ['Random-vs-random pilot length and sampled throughput are preliminary operating estimates, not a hard game horizon.']
        runner.state['calibration'] = cal
        runner.state['operating_baseline'] = profile_values(agent)
        runner.state['selected_candidate'] = {'phase': None, 'name': 'incumbent', 'profile': profile_values(agent)}
        runner.save()

    def announce(self, runner):
        cal = runner.state['calibration']
        runner.progress('Agent family: SO-ISMCTS; baseline: ' + json.dumps(runner.state['operating_baseline']))
        runner.progress(f'Estimated decisions/game: {cal["estimated_game_decisions"]:.1f}; decision budget: ' +
                        (f'{cal["fixed_iterations"]} iterations' if cal['fixed_iterations'] else f'{cal["decision_seconds"]:.6f}s'))
        runner.progress(f'Median iterations/decision: {cal["median_iterations_per_decision"]}; determinizations/decision: {cal["median_determinizations_per_decision"]}; search adequacy: {cal["search_adequacy"]["category"]}')
        runner.progress('Planned stages: calibration, exploration coarse race, bounded local refinement, fresh-seed confirmation')
        if runner.state['request']['tune']:
            runner.progress('Tuning: exploration. Frozen: search budget, uniform rollout, most_visited, game configuration.')

    def build_phase(self, name, state, base):
        from .studies import _build_tuning_phase
        if name != 'confirmation':
            return _build_tuning_phase(name, state, base)
        original = state['operating_baseline']
        finalist = state.get('selected_candidate', {}).get('profile', original)
        return {'name': name, 'agents': {'incumbent': original, 'finalist': finalist},
                'groups': {'main': ['incumbent', 'finalist']}, 'accept': True, 'incremental': True,
                'evidence_policy': 'confirmation', 'tie_priority': {'incumbent': [0], 'finalist': [1]},
                'contrasts': [] if original == finalist else [{'a': 'incumbent', 'b': 'finalist', 'primary': True, 'factor': 'exploration'}],
                'status': 'pending', 'skip_reason': 'no_supported_improvement'}


PROFILES = {'so_ismcts': SoIsmctsStudyProfile()}
