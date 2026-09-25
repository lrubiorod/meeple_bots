"""Analyze human output and generic/legacy JSON projections."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from statistics import median

from ..api import GameEvaluationReport, MctsAgentBenchmark, SampledDecisionTiming
from ..game_config import game_parameters
from ..serialization import (
    game_name as _game_name, _evaluator_dict, _rollout_policy_name,
    _rollout_policy_evaluator, _rollout_policy_epsilon,
    _conditional_rollout_fields, _progressive_bias_fields,
)
from .measurement import sampled_full_horizon


def _format_game_search_seconds(seconds):
    if seconds is None:
        return 'N/A'
    # Round before splitting so minute boundaries never render as "60.00s".
    minutes, remainder = divmod(round(seconds * 100), 6000)
    return (f'{minutes}m ' if minutes else '') + f'{remainder / 100:.2f}s'


def report_dict(report):
    return dict(properties=report.properties, structural=report.structural,
                search_calibration=report.search_calibration,
                configured_agent_benchmarks=list(report.configured_agent_benchmarks))


def _print_root_diagnostics(adequacy, visited, estimated):
    coverage = adequacy['median_root_coverage']
    coverage_text = f"{coverage:.1%}" if coverage is not None else "unavailable"
    known = [v for v in visited if v is not None]
    print(f"  Root legal actions (median): {adequacy['representative_branching']}; median visited: {median(known) if known else 'unavailable'}; coverage: {coverage_text}")
    print(f"  Root visits/action median/p10: {adequacy['median_visits_per_root_action']} / {adequacy['p10_visits_per_root_action']}")
    print(f"  Measured adequacy: {adequacy['category']}; estimated at target: {estimated}")


def print_analysis(report, include_structure=True):
    s, c, p = report.structural, report.search_calibration, report.properties
    rows = [('Default calibration', c)] + [(x['name'], x) for x in report.configured_agent_benchmarks]
    print(f"Game: {p['game']}")
    if include_structure:
        print('\nStructure')
        print(f"  Players: {p['players']}; Stochastic: {p['stochastic']}; Hidden information: {p['imperfect_information']}")
        print(f"  Structural sampling: {s['samples']} samples; terminal rate {s['terminal_rate']:.1%}")
        if s['initial_legal_actions_min'] == s['initial_legal_actions_max']:
            print(f"  Initial legal actions: {s['initial_legal_actions_min']}")
        else:
            print(f"  Initial legal actions mean/p50/p95/min/max: {s['initial_legal_actions_mean']:.2f} / {s['initial_legal_actions_p50']} / {s['initial_legal_actions_p95']} / {s['initial_legal_actions_min']} / {s['initial_legal_actions_max']}")
        print(f"  Effective decision branching: {s['effective_branching_factor']:.2f}")
        choices = s['player_turn_choice_product_log10']
        choice_text = f"~{10**choices:,.1f} (10^{choices:.2f})" if choices <= 6 else f"~10^{choices:.2f}"
        print(f"  Observed choices across one player turn: {choice_text} (geometric mean of sampled phase products)")
        print(f"  Physical decision-tree estimate: 10^{s['estimated_tree_log10']:.1f}; excludes chance branching, not an information-set tree size")
        print('\nDepth structure')
        print(f"  Player decisions / tree depth mean/p50/p95: {s['decisions_mean']:.1f} / {s['depth_p50']} / {s['estimated_depth']}")
        print(f"  Physical turns p50/p95: {s['physical_turns_p50']} / {s['physical_turns_p95']} (game turn boundaries)")
        print(f"  Player turns p50/p95: {s['player_turn_depth_p50']} / {s['player_turn_depth_p95']} (consecutive same-player decision blocks)")
        print(f"  Player changes p50/p95: {s['player_changes_p50']} / {s['player_changes_p95']}")
        print(f"  Actions per player turn mean/p95/max: {s['actions_per_player_turn_mean']:.2f} / {s['actions_per_player_turn_p95']} / {s['actions_per_player_turn_max']}")
        print(f"  Chance events mean/p50/p95: {s['chance_events_mean']:.2f} / {s['chance_events_p50']} / {s['chance_events_p95']}")
        if s['depth_is_lower_bound']:
            print('  WARNING: some samples hit the safety cap; depth estimates are lower bounds.')
    if report.legacy_report is not None:
        print('\nRollout horizon diagnostics')
        print(f"  full = ceil(1.5 × sampled p95 depth {s['estimated_depth']}) = {sampled_full_horizon(s['estimated_depth'])}; soft safety horizon, not a proven game bound.")
        for cost in report.legacy_report.rollout_costs:
            label = 'full' if cost.rollout_depth == sampled_full_horizon(s['estimated_depth']) else 'depth'
            print(f"  {label} {cost.rollout_depth:<4} | ~{cost.milliseconds_per_iteration:.4f} ms/iteration | ~{cost.approximate_player_turns:.1f} player turns")
    print('\nSearch calibration')
    print(f"  Search family: {c['family']}")
    print('  Independent positions; these probes do not measure match tree-reuse benefits.')
    for label, row in rows:
        print(f"\n{label}: {row['sampled_positions']} positions (early/mid/late plus missing phases)")
        if row.get('rollout_horizon'):
            horizon = row['rollout_horizon']
            print(f"  Rollout horizon: full = {horizon['depth']} plies; ceil(1.5 × sampled p95 {horizon['sampled_p95']}); soft safety horizon.")
            if horizon['capped']:
                print('  WARNING: horizon capped at the u32 configuration limit.')
        if row.get('rollout_terminal_rate') is not None:
            print(f"  Rollouts reaching terminal: {row['rollout_terminal_rate']:.1%}; cutoff rate: {row['rollout_cutoff_rate']:.1%}")
        print(f"  Agent configuration: {row['agent']}")
        print(f"  Mean iteration: {row['milliseconds_per_iteration']:.4f} ms; iterations/s: {row['iterations_per_second']:,.0f}")
        print(f"  Measured median iterations/decision: {row['median_iterations_per_decision']:,.0f}; mean latency: {row['decision_time_mean_ms']:.2f} ms")
        print(f"  Latency p50/p95/max: {row['decision_time_p50_ms']:.2f} / {row['decision_time_p95_ms']:.2f} / {max(t['milliseconds'] for t in row['position_timings']):.2f} ms")
        print(f"  Mean tree nodes: {row['mean_tree_nodes']:.0f}")
        if row['family'] == 'so_ismcts':
            print(f"  Determinizations/s: {row['determinizations_per_second']:,.0f}; median/decision: {row['median_determinizations_per_decision']:,.0f}")
            print(f"  Isolated determinization: {row['isolated_determinization_mean_ms']*1000:.2f} us (256 samples/observation)")
            print(f"  Mean action edges: {row['mean_action_edges']:.0f}; mean availability: {row['mean_action_availability']:.2f}; mean availability/node visits: {row['mean_availability_ratio']:.3f}")
        print('  Root diagnostics')
        _print_root_diagnostics(row['search_adequacy'], [t['root_actions_visited'] for t in row['position_timings']], row['estimated_target_adequacy'])
        if row.get('target_match_time'):
            print(f"  Target derived from {row['target_match_time']}s / ({row['expected_decisions']:.1f} decisions × {row['safety_margin']})")
        for warning in row['warnings']:
            print('  WARNING: ' + warning)
    phases = {phase['label']: phase for phase in s.get('phases', [])}
    phase_labels = sorted(set(phases) | {phase['label'] for _, row in rows for phase in row.get('phase_diagnostics', [])})
    if phase_labels:
        print('\nPhase diagnostics')
        for label in phase_labels:
            print(label)
            if label in phases:
                phase = phases[label]
                print(f"  Structural decision samples: {phase['samples']}")
                print(f"  Legal actions mean/p50/p95: {phase['legal_actions_mean']:.2f} / {phase['legal_actions_p50']} / {phase['legal_actions_p95']}; min/max: {phase['legal_actions_min']} / {phase['legal_actions_max']}")
                print(f"  Effective decision branching: {phase['effective_branching_factor']:.2f}")
            for name, row in rows:
                found = next((x for x in row.get('phase_diagnostics', []) if x['label'] == label), None)
                if found is None:
                    print(f"  {name}: no search probe for this phase")
                    continue
                print(f"  {name}: {found['sampled_positions']} search probe(s)")
                _print_root_diagnostics(found['search_adequacy'], found['root_actions_visited'], found['estimated_target_adequacy'])
    print('\nOperating points (linear estimates)')
    for label, row in rows:
        print(f"  {label}")
        print('  decision | iterations' + (' | determinizations' if row['family'] == 'so_ismcts' else '') + ' | mean game | p95 game')
        for b in row['budget_table']:
            game_times = ' | '.join(_format_game_search_seconds(b.get(field)) for field in
                                  ('estimated_mean_game_search_seconds', 'estimated_p95_game_search_seconds'))
            print(f"    {b['seconds']:g}s | {b['iterations']:,}" + (f" | {b['determinizations']:,}" if 'determinizations' in b else '') + f' | {game_times}' + (' <- target' if b['target'] else ''))
    print('  Game columns estimate accumulated search time with both players using this decision budget; not wall-clock limits.')
    print('\nInterpretation')
    print('Root diagnostics describe measured probes; target operating points are estimates based on measured throughput.')
    print('Adequacy describes search population, not playing strength, confidence or optimality.')
    print('Analyze measures search cost / adequacy; study measures competitive strength through equal-compute comparisons.')


@dataclass(frozen=True, slots=True)
class _ConfiguredMctsBenchmark:
    name: str
    benchmark: MctsAgentBenchmark


def _evaluation_dict(
    report: GameEvaluationReport,
    configured_benchmarks: Sequence[_ConfiguredMctsBenchmark] = (),
) -> dict[str, object]:
    return {
        "game": _game_name(report.game),
        **({"game_params": game_parameters(report.game)} if game_parameters(report.game) else {}),
        "samples": report.samples,
        "max_depth": report.max_depth,
        "terminal_rate": report.terminal_rate,
        "initial_legal_actions": report.initial_legal_actions,
        "effective_branching_factor": report.effective_branching_factor,
        "player_turn_choice_product_log10": report.player_turn_choice_product_log10,
        "depth_p50": report.depth_p50,
        "estimated_depth": report.estimated_depth,
        "player_turn_depth_p50": report.player_turn_depth_p50,
        "player_turn_depth_p95": report.player_turn_depth_p95,
        "player_changes_p50": report.player_changes_p50,
        "player_changes_p95": report.player_changes_p95,
        "actions_per_player_turn_mean": report.actions_per_player_turn_mean,
        "actions_per_player_turn_p95": report.actions_per_player_turn_p95,
        "actions_per_player_turn_max": report.actions_per_player_turn_max,
        "depth_is_lower_bound": report.depth_is_lower_bound,
        "estimated_tree_log10": report.estimated_tree_log10,
        "calibration_positions": report.calibration_positions,
        "target_time_seconds": report.target_time_seconds,
        "rollout_costs": [
            {
                "rollout_depth": cost.rollout_depth,
                "approximate_player_turns": cost.approximate_player_turns,
                "milliseconds_per_iteration": cost.milliseconds_per_iteration,
                "iteration_budgets": [
                    {
                        "seconds": budget.seconds,
                        "iterations": budget.iterations,
                    }
                    for budget in cost.iteration_budgets
                ],
            }
            for cost in report.rollout_costs
        ],
        "suggested_experiments": [
            {
                "label": experiment.label,
                "iterations": experiment.iterations,
                "iterations_capped": experiment.iterations_capped,
                "rollout_depth": experiment.rollout_depth,
                "approximate_player_turns": experiment.approximate_player_turns,
                "estimated_decision_time_ms": experiment.estimated_decision_time_ms,
            }
            for experiment in report.suggested_experiments
        ],
        "configured_agent_benchmarks": _configured_benchmark_dicts(
            report,
            configured_benchmarks,
        ),
        "recommended_rollout_depth": report.recommended_rollout_depth,
        "recommended_iterations": report.recommended_iterations,
        "iterations_capped": report.iterations_capped,
        "milliseconds_per_iteration": report.milliseconds_per_iteration,
        "estimated_decision_time_ms": report.estimated_decision_time_ms,
    }


def _configured_benchmark_dicts(
    report: GameEvaluationReport,
    configured_benchmarks: Sequence[_ConfiguredMctsBenchmark],
) -> list[dict[str, object]]:
    ranked = sorted(
        configured_benchmarks,
        key=lambda configured: configured.benchmark.decision_time_mean_ms,
    )
    if not ranked:
        return []
    fastest_ms = ranked[0].benchmark.decision_time_mean_ms
    target_ms = report.target_time_seconds * 1_000.0
    rows = []
    for rank, configured in enumerate(ranked, start=1):
        benchmark = configured.benchmark
        agent = benchmark.agent
        rows.append(
            {
                "rank": rank,
                "name": configured.name,
                "iterations": agent.iterations,
                "time_budget": agent.time_budget,
                "rollout_depth": agent.rollout_depth,
                "exploration": agent.exploration,
                "selection_policy": agent.selection_policy,
                **({"rave_equivalence": agent.rave_equivalence} if agent.selection_policy == "uct_rave" else {}),
                **({"progressive_widening": agent.progressive_widening, "progressive_widening_k": agent.progressive_widening_k, "progressive_widening_alpha": agent.progressive_widening_alpha, "progressive_widening_expansion": agent.progressive_widening_expansion} if getattr(agent, "progressive_widening", False) else {}),
                "heuristic": agent.heuristic,
                "cutoff_evaluator": _evaluator_dict(agent.cutoff_evaluator),
                "rollout_policy": _rollout_policy_name(agent),
                "rollout_evaluator": _evaluator_dict(
                    _rollout_policy_evaluator(agent.rollout_policy)
                ),
                "rollout_epsilon": _rollout_policy_epsilon(agent.rollout_policy),
                **_conditional_rollout_fields(agent.rollout_policy),
                **_progressive_bias_fields(agent.progressive_bias),
                "root_diagnostics": agent.root_diagnostics,
                "tree_reuse": agent.tree_reuse,
                "transpositions": agent.transpositions,
                "sampled_positions": benchmark.sampled_positions,
                "decision_time_mean_ms": benchmark.decision_time_mean_ms,
                "decision_time_p50_ms": benchmark.decision_time_p50_ms,
                "decision_time_p95_ms": benchmark.decision_time_p95_ms,
                "decision_time_max_ms": benchmark.decision_time_max_ms,
                "milliseconds_per_iteration": benchmark.milliseconds_per_iteration,
                "relative_to_fastest": benchmark.decision_time_mean_ms / fastest_ms,
                "target_time_ratio": benchmark.decision_time_mean_ms / target_ms,
                "position_timings": [
                    {
                        "sampled_ply": timing.sampled_ply,
                        "milliseconds": timing.milliseconds,
                        "iterations": timing.iterations,
                        "nodes": timing.nodes,
                    }
                    for timing in benchmark.position_timings
                ],
            }
        )
    return rows


def analysis_json(analysis, profiles, game):
    report = analysis.legacy_report
    benchmarks = []
    if report is not None:
        allowed = {f.name for f in fields(SampledDecisionTiming)}
        for profile, measured in zip(profiles, analysis.configured_agent_benchmarks):
            benchmarks.append(_ConfiguredMctsBenchmark(profile.name, MctsAgentBenchmark(
                game=game, agent=profile.agent,
                sampled_positions=measured['sampled_positions'],
                decision_time_mean_ms=measured['decision_time_mean_ms'],
                decision_time_p50_ms=measured['decision_time_p50_ms'],
                decision_time_p95_ms=measured['decision_time_p95_ms'],
                decision_time_max_ms=max(t['milliseconds'] for t in measured['position_timings']),
                milliseconds_per_iteration=measured['milliseconds_per_iteration'],
                position_timings=tuple(SampledDecisionTiming(**{k:v for k,v in t.items() if k in allowed}) for t in measured['position_timings']))))
    data = report_dict(analysis)
    if report is not None:
        legacy = _evaluation_dict(report, benchmarks)
        # Preserve configured MCTS benchmark schema; add generic diagnostics separately.
        legacy.update({k:v for k,v in data.items() if k != 'configured_agent_benchmarks'})
        details = {row['name']: row for row in data['configured_agent_benchmarks']}
        for row in legacy['configured_agent_benchmarks']:
            row['search_diagnostics'] = details[row['name']]
        data = legacy
    return data
