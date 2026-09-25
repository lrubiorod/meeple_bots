"""Legacy deterministic-MCTS analyze JSON compatibility projection."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

from ..api import GameEvaluationReport, MctsAgentBenchmark, SampledDecisionTiming
from ..analysis import report_dict
from ..game_config import game_parameters
from ..serialization import (
    game_name as _game_name, _evaluator_dict, _rollout_policy_name,
    _rollout_policy_evaluator, _rollout_policy_epsilon,
    _conditional_rollout_fields, _progressive_bias_fields,
)

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
