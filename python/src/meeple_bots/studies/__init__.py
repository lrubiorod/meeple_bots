"""Study public and private compatibility facade.

Lazy exports preserve Study entrypoints without loading the coordinator until needed.
Analyze obtains shared ``quantile`` and ``search_adequacy`` from ``search_metrics``;
Study retains its own practical horizon, race/evidence, promotion and paired statistics.
"""
from importlib import import_module

_OWNERS = {
    **dict.fromkeys(('StudyRunner', 'run_study', 'GAMES', 'duration_seconds', 'generic_baseline',
                     'run_matches', 'benchmark_mcts_agent'), 'coordinator'),
    **dict.fromkeys(('MAX_EXTENSION_ROUNDS', 'PHASES', 'STAGES', 'SEED_STRIDE',
                     'stage_for_phase', '_phase_enabled', 'cutoff_depths', '_pw_survivors',
                     '_build_phase', '_build_pw_phase', 'tuning_specs', '_build_tuning_phase',
                     'mcts_plan', 'so_specs', 'build_so_phase'), 'planning'),
    **dict.fromkeys(('_rank', '_promising', '_group_leaders', '_evidence_complete',
                     '_reverse_result', 'summarize_contrast'), 'race'),
    **dict.fromkeys(('_study_budget', 'estimated_round_seconds', 'cost_pair', 'plan_phase'), 'budget'),
    **dict.fromkeys(('STUDY_VERSION', 'FINGERPRINT_ALGORITHM', 'policy_values',
                     'profile_values', 'agent_from_values', '_toml', 'export_profile',
                     '_package_source_root', '_python_source_entries', '_python_tree_digest',
                     '_legacy_python_digest', '_fingerprint', '_engine_change_reason', '_save'),
                    'persistence'),
}


def __getattr__(name):
    owner = _OWNERS.get(name)
    if owner is None:
        raise AttributeError(name)
    return getattr(import_module(f'{__name__}.{owner}'), name)


def __dir__():
    return sorted(set(globals()) | set(_OWNERS))
