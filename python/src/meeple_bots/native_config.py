"""Convert game-independent agent values into Rust agent configurations."""

from __future__ import annotations

from . import _native
from ._agent_config import (
    BaseRolloutPolicy, ConditionalRollout, Greedy, Mast, MctsAgent,
    ProgressiveBias, RandomAgent, RolloutPolicy, SoIsmctsAgent,
    StateEvaluator, GameHeuristic, UniformRandom,
)

def _native_evaluator(
    evaluator: StateEvaluator | None,
) -> tuple[str, int | None, dict[str, float] | None]:
    if isinstance(evaluator, GameHeuristic):
        return "game_heuristic", evaluator.index, dict(evaluator.params)
    return "neutral", None, None


def _native_progressive_bias(
    bias: ProgressiveBias | None,
) -> tuple[
    float | None,
    str | None,
    int | None,
    dict[str, float] | None,
    str | None,
]:
    if bias is None:
        return None, None, None, None, None
    evaluator, heuristic, params = _native_evaluator(bias.evaluator)
    phase = None if bias.condition is None else bias.condition.phase
    return float(bias.weight), evaluator, heuristic, params, phase


def _native_rollout_policy(
    policy: RolloutPolicy,
) -> tuple[
    str,
    str | None,
    int | None,
    dict[str, float] | None,
    float | None,
    str | None,
    str | None,
    str | None,
    int | None,
    dict[str, float] | None,
    float | None,
]:
    if isinstance(policy, ConditionalRollout):
        primary = _native_base_rollout_policy(policy.primary)
        fallback = _native_base_rollout_policy(policy.fallback)
        return (*primary, policy.condition.phase, *fallback)
    return (*_native_base_rollout_policy(policy), None, None, None, None, None, None)


def _native_base_rollout_policy(
    policy: BaseRolloutPolicy,
) -> tuple[str, str | None, int | None, dict[str, float] | None, float | None]:
    if isinstance(policy, UniformRandom):
        return "uniform_random", None, None, None, None
    if isinstance(policy, Mast):
        return "mast", None, None, None, float(policy.epsilon)
    if isinstance(policy, Greedy):
        evaluator, heuristic, params = _native_evaluator(policy.evaluator)
        return "greedy", evaluator, heuristic, params, None
    evaluator, heuristic, params = _native_evaluator(policy.evaluator)
    return (
        "epsilon_greedy",
        evaluator,
        heuristic,
        params,
        float(policy.epsilon),
    )


def native_agent_config(agent: RandomAgent | SoIsmctsAgent | MctsAgent):
    if isinstance(agent, RandomAgent):
        return _native.AgentConfig.random()
    if isinstance(agent, SoIsmctsAgent):
        return _native.AgentConfig.so_ismcts(agent.iterations, agent.exploration, agent.time_budget, agent.selection_policy, agent.tree_reuse)
    if isinstance(agent, MctsAgent):
        (
            policy,
            rollout_evaluator,
            rollout_heuristic,
            rollout_params,
            epsilon,
            rollout_condition_phase,
            fallback_policy,
            fallback_evaluator,
            fallback_heuristic,
            fallback_params,
            fallback_epsilon,
        ) = _native_rollout_policy(agent.rollout_policy)
        cutoff_evaluator, cutoff_heuristic, cutoff_params = _native_evaluator(
            agent.cutoff_evaluator
        )
        return _native.AgentConfig.mcts(
            agent.iterations,
            float(agent.exploration),
            agent.rollout_depth,
            cutoff_evaluator,
            cutoff_heuristic,
            cutoff_params,
            policy,
            rollout_evaluator,
            rollout_heuristic,
            rollout_params,
            epsilon,
            agent.time_budget,
            rollout_condition_phase,
            fallback_policy,
            fallback_evaluator,
            fallback_heuristic,
            fallback_params,
            fallback_epsilon,
            *_native_progressive_bias(agent.progressive_bias),
            agent.root_diagnostics,
            agent.tree_reuse,
            agent.transpositions,
            agent.selection_policy,
            rave_equivalence=agent.rave_equivalence,
            progressive_widening=agent.progressive_widening,
            progressive_widening_k=agent.progressive_widening_k,
            progressive_widening_alpha=agent.progressive_widening_alpha,
            progressive_widening_expansion=agent.progressive_widening_expansion,
        )

