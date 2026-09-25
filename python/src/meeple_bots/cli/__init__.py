"""Stable CLI entrypoints and transitional documented helper exports."""

from .commands import main, _match_agent
from .parser import build_parser
from .display import _serialized_rollout_policy_description
from .analyze_compat import _evaluation_dict
from ..tournament_config import _load_tournament_config, _load_tournament_agents
from ..tournaments import tournament_pairings as _tournament_pairings
from ..serialization import agent_dict as _batch_agent_dict
from .._mcts_profiles import (
    _MctsProfile, _configured_cutoff_evaluator, _configured_evaluator,
    _configured_legacy_rollout_evaluator, _configured_progressive_bias,
    _configured_rollout_policy, _configured_root_diagnostics,
    _configured_transpositions, _configured_tree_reuse,
    _inline_agent_integer, _inline_agent_name, _inline_evaluator,
    _load_mcts_profile, _mcts_budget_kwargs, _parse_inline_mcts_profile,
    sqrt_two,
)
