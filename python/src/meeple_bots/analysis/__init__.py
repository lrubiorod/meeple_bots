"""Stable Analyze imports; measurement and presentation have separate owners."""
from .measurement import (
    AnalysisReport, analyze_game, analyze_structure, benchmark_search_agent,
    sampled_full_horizon, operating_points, summarize_search,
    _add_game_search_estimates, _ANALYZERS, evaluate_game,
)
from .report import report_dict, print_analysis, _format_game_search_seconds, _print_root_diagnostics
