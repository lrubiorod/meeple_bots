"""Analyze structural sampling, search measurement, and operating points."""
from dataclasses import asdict, dataclass, replace
from math import isfinite
from statistics import mean, median

from .. import _native
from .._agent_config import MctsAgent, SoIsmctsAgent
from .._capabilities import game_search_capabilities
from .._search_budget import decision_budget
from .._search_profiles import resolve_family
from ..api import Game, _positive_u32, _non_negative_u32, _MAX_U64, _native_game, benchmark_mcts_agent, evaluate_game, game_parameters
from ..serialization import agent_dict
from ..search_metrics import quantile, search_adequacy


@dataclass(frozen=True)
class AnalysisReport:
    structural: dict
    search_calibration: dict
    configured_agent_benchmarks: tuple[dict, ...]
    properties: dict
    legacy_report: object = None


def _validate_game(game):
    if not isinstance(game, Game):
        raise TypeError("game must be a registered game instance")


def _validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= _MAX_U64:
        raise ValueError("seed must be an unsigned 64-bit integer")


def _validate_target(target):
    if isinstance(target, bool) or not isinstance(target, (int, float)):
        raise TypeError("target time must be a number")
    if not isfinite(target) or not 0 < target <= 3600:
        raise ValueError("target time must be finite, positive and at most 3600 seconds")


def analyze_structure(game, samples=128, max_depth=256, seed=0):
    """Sample player decisions and environment events separately, without any agent."""
    _validate_game(game)
    _positive_u32("samples", samples)
    _positive_u32("max_depth", max_depth)
    _validate_seed(seed)
    return _native.analyze_structure(_native_game(game), samples, max_depth, seed,
                                     game_params=game_parameters(game))


def _mcts_timings(game, agent, depth, seed):
    return [asdict(t) for t in benchmark_mcts_agent(game, agent, depth, seed).position_timings]


def _so_timings(game, agent, depth, seed):
    return _native.benchmark_so_ismcts(_native_game(game), depth, seed,
                                      iterations=agent.iterations, exploration=agent.exploration,
                                      time_budget=agent.time_budget, selection_policy=agent.selection_policy, tree_reuse=agent.tree_reuse)


# Family-specific measurement only. Report construction and operating points are shared.
_ANALYZERS = {'mcts': _mcts_timings, 'so_ismcts': _so_timings}


def sampled_full_horizon(p95):
    """Ceiling of 1.5 times sampled decision p95, within MCTS's u32 config range."""
    return max(1, min(2**32 - 1, (3 * p95 + 1) // 2))


def operating_points(iterations_per_second, target_time, family):
    budgets = sorted({.1, .25, .5, 1., 2., target_time})
    return [dict(seconds=t, iterations=max(1, int(iterations_per_second*t)),
                 **({'determinizations': max(1, int(iterations_per_second*t))} if family == 'so_ismcts' else {}),
                 target=t == target_time) for t in budgets]


def _add_game_search_estimates(budget_table, structural):
    """Use existing player-decision samples, summed across both seats, not turns."""
    for row in budget_table:
        for statistic, field in (
            ('decisions_mean', 'estimated_mean_game_search_seconds'),
            ('estimated_depth', 'estimated_p95_game_search_seconds'),
        ):
            decisions = structural.get(statistic)
            valid = (isinstance(decisions, (int, float)) and not isinstance(decisions, bool)
                     and isfinite(decisions) and decisions >= 0)
            row[field] = row['seconds'] * decisions if valid else None


def summarize_search(timings, target_time, family):
    elapsed = sum(t['milliseconds'] for t in timings) / 1000
    iterations = sum(t['iterations'] for t in timings)
    throughput = iterations / elapsed
    for t in timings:
        if not t.get('root_visits') and family == 'mcts':
            t.update(root_actions_visited=None, root_action_coverage=None,
                     median_root_visits_per_action=None, p10_root_visits_per_action=None)
            continue
        visits = list(t.get('root_visits', ()))
        visits += [0] * max(0, t['legal_actions'] - len(visits))
        t['root_actions_visited'] = sum(v > 0 for v in visits)
        t['root_action_coverage'] = t['root_actions_visited'] / max(1, t['legal_actions'])
        t['median_root_visits_per_action'] = median(visits) if visits else 0
        t['p10_root_visits_per_action'] = quantile(visits, .1)
    adequacy = search_adequacy(timings, target_time)
    # The shared helper calls these match targets in study; here the unit is per decision.
    for suggestion in adequacy['suggested_targets']:
        suggestion['target_time_seconds'] = suggestion.pop('target_match_time')
    # Root visits/coverage remain actual measurements, never extrapolated as guarantees.
    projected = [{**t, 'iterations': t['iterations']*target_time/(t['milliseconds']/1000)} for t in timings]
    target_adequacy = search_adequacy(projected, target_time)
    result = dict(family=family, sampled_positions=len(timings), position_timings=timings,
        milliseconds_per_iteration=elapsed*1000/iterations,
        iterations_per_second=throughput,
        median_iterations_per_decision=median(t['iterations'] for t in timings),
        decision_time_mean_ms=mean(t['milliseconds'] for t in timings),
        decision_time_p50_ms=quantile([t['milliseconds'] for t in timings], .5),
        decision_time_p95_ms=quantile([t['milliseconds'] for t in timings], .95),
        mean_tree_nodes=mean(t['nodes'] for t in timings),
        target_time_seconds=target_time,
        estimated_iterations_at_target=max(1, int(throughput*target_time)),
        search_adequacy=adequacy, estimated_target_adequacy=target_adequacy['category'],
        budget_table=operating_points(throughput, target_time, family),
        warnings=[], interpretation='Search-cost diagnostic, not strength or confidence. Target estimates scale measured throughput linearly; use study to tune C under equal compute.')
    if target_adequacy['category'] in ('VERY LOW', 'LOW'):
        result['warnings'].append('Requested target appears undersearched; it has not been increased.')
    if family == 'so_ismcts':
        determinization_ms = mean(t['determinization_milliseconds'] for t in timings)
        result.update(determinizations_per_second=sum(t['determinizations'] for t in timings)/elapsed,
                      median_determinizations_per_decision=median(t['determinizations'] for t in timings),
                      estimated_determinizations_at_target=max(1, int(throughput*target_time)),
                      isolated_determinization_mean_ms=determinization_ms,
                      isolated_determinizations_per_second=1000/determinization_ms,
                      mean_action_edges=mean(t['action_edges'] for t in timings),
                      mean_action_availability=mean(t['mean_availability'] for t in timings),
                      mean_availability_ratio=mean(t['mean_availability_ratio'] for t in timings))
    completed = [t for t in timings if t.get('terminal_simulations') is not None and t.get('cutoff_simulations') is not None]
    terminal = sum(t['terminal_simulations'] for t in completed)
    cutoff = sum(t['cutoff_simulations'] for t in completed)
    result['rollout_terminal_rate'] = terminal / (terminal + cutoff) if terminal + cutoff else None
    result['rollout_cutoff_rate'] = cutoff / (terminal + cutoff) if terminal + cutoff else None
    result['phase_diagnostics'] = []
    for phase in sorted({t['phase'] for t in timings if t.get('phase')}):
        probes = [t for t in timings if t.get('phase') == phase]
        measured = search_adequacy(probes, target_time)
        projected = [{**t, 'iterations': t['iterations']*target_time/(t['milliseconds']/1000)} for t in probes]
        result['phase_diagnostics'].append(dict(label=phase, sampled_positions=len(probes),
            root_legal_actions=[t['legal_actions'] for t in probes],
            root_actions_visited=[t['root_actions_visited'] for t in probes],
            search_adequacy=measured,
            estimated_target_adequacy=search_adequacy(projected, target_time)['category']))
    return result


def benchmark_search_agent(game, agent, median_depth, seed=0, target_time=.5):
    """Measure an exact profile on reproducible positions; compare costs, never wins."""
    _validate_game(game)
    _validate_seed(seed)
    _non_negative_u32("median_depth", median_depth)
    _validate_target(target_time)
    family = resolve_family(_native_game(game), baseline=agent)
    timings = _ANALYZERS[family](game, agent, median_depth, seed)
    return {**summarize_search(timings, target_time, family), 'agent': agent_dict('benchmark', agent)}


def analyze_game(game, samples=128, max_depth=256, seed=0, target_time=None,
                 search_family=None, profiles=(), target_match_time=None):
    """One analysis coordinator with additive family-specific measurement backends.

    Profiles are (name, agent) pairs. Short default probes adapt their iteration count;
    exact configured profiles retain their original time/iteration budget.
    """
    _validate_game(game)
    _positive_u32("samples", samples)
    _positive_u32("max_depth", max_depth)
    _validate_seed(seed)
    family = resolve_family(_native_game(game), search_family, profiles[0][1] if profiles else None)
    for _, agent in profiles:
        resolve_family(_native_game(game), family, agent)
    if target_time is not None and target_match_time is not None:
        raise ValueError('choose target_time or target_match_time, not both')
    for t in (target_time, target_match_time):
        if t is not None:
            _validate_target(t)
    target = target_time if target_time is not None else 5.
    caps = game_search_capabilities(_native_game(game))
    # Preserve the established deterministic MCTS rollout-depth calibration/output.
    legacy = None
    if family == 'mcts' and not caps.get('stochastic') and target_match_time is None:
        legacy = evaluate_game(game, samples, max_depth, seed, target)
        structural = legacy.structural
    else:
        structural = analyze_structure(game, samples, max_depth, seed)
    if target_match_time is not None:
        target = decision_budget(target_match_time, max(1, structural['decisions_mean']))
        if family == 'mcts' and not caps.get('stochastic'):
            legacy = evaluate_game(game, samples, max_depth, seed, target)
    depth = structural['depth_p50']
    default = (SoIsmctsAgent(iterations=8) if family == 'so_ismcts' else
               MctsAgent(iterations=8, rollout_depth=sampled_full_horizon(structural['estimated_depth']), root_diagnostics=True))
    measure = _ANALYZERS[family]
    probe = measure(game, default, depth, seed)
    cost = sum(t['milliseconds'] for t in probe)/sum(t['iterations'] for t in probe)
    # Bounded, adaptive measurement window, not an optimization over iteration counts.
    count = max(1, min(4096, round(min(.1, target)*1000/cost)))
    operating = replace(default, iterations=count)
    calibration = benchmark_search_agent(game, operating, depth, seed, target)
    if family == 'mcts':
        calibration['rollout_horizon'] = dict(kind='sampled_full_safety',
            sampled_p95=structural['estimated_depth'], depth=default.rollout_depth,
            multiplier=1.5, capped=3 * structural['estimated_depth'] > 2 * (2**32 - 1))
        if structural['depth_is_lower_bound']:
            calibration['warnings'].append('Sampled-full safety horizon is based on truncated structural samples; increase --max-depth before relying on it.')
    calibration['measurement'] = 'Short adaptive fixed-iteration probe; root diagnostics refer to measured work, target counts are estimates.'
    calibration['target_match_time'] = target_match_time
    calibration['safety_margin'] = 1.2 if target_match_time is not None else None
    calibration['expected_decisions'] = structural['decisions_mean']
    benchmarks = tuple({'name': name, **benchmark_search_agent(game, agent, depth, seed, target)}
                       for name, agent in profiles)
    for result in (calibration, *benchmarks):
        _add_game_search_estimates(result['budget_table'], structural)
    return AnalysisReport(structural, calibration, benchmarks,
        dict(game=_native_game(game), players=caps['players'], stochastic=caps.get('stochastic', False),
             imperfect_information=caps.get('imperfect_information', False), search_family=family,
             game_params=game_parameters(game)), legacy)

