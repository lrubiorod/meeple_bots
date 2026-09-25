"""Compatibility surface for Study evidence/reporting and shared work metrics.

Analyze still imports quantile/search_adequacy here until Phase 3.
"""
from importlib import import_module
from statistics import median


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(len(ordered)-1, low+1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position-low)


def search_adequacy(timings, target_match_time):
    """Heuristic work diagnostic, NOT a probability or competitive confidence.

    Ratios below 1/10/100 simulations per representative legal action are
    VERY LOW/LOW/MEDIUM; >=100 is HIGH. Thresholds are descriptive, not guarantees.
    """
    def category(ratio):
        return "VERY LOW" if ratio < 1 else "LOW" if ratio < 10 else "MEDIUM" if ratio < 100 else "HIGH"
    iterations = median([t["iterations"] for t in timings]) if timings else 0
    branching = max(1, median([t["legal_actions"] for t in timings])) if timings else 1
    elapsed = sum(t["milliseconds"] for t in timings) / 1000
    ratio = iterations / branching
    visits, coverages = [], []
    for t in timings:
        if t.get("root_visits"):
            values = list(t["root_visits"]) + [0] * max(0, t["legal_actions"] - len(t["root_visits"]))
            visits.extend(values)
            coverages.append(sum(v > 0 for v in values) / max(1, t["legal_actions"]))
    recommendations = []
    if category(ratio) in ("VERY LOW", "LOW") and ratio > 0:
        for multiplier in (3, 10):
            recommendations.append({"target_match_time": target_match_time * multiplier,
                                    "estimated_category": category(ratio * multiplier),
                                    "estimated_iterations_per_decision": iterations * multiplier})
    return {"category": category(ratio), "median_iterations_per_decision": iterations,
            "actual_elapsed_seconds": elapsed,
            "iterations_completed": sum(t["iterations"] for t in timings),
            "iterations_per_second": sum(t["iterations"] for t in timings) / elapsed if elapsed else 0,
            "representative_branching": branching, "iterations_per_legal_action": ratio,
            "median_root_coverage": median(coverages) if coverages else None,
            "median_visits_per_root_action": median(visits) if visits else None,
            "p10_visits_per_root_action": quantile(visits, .1),
            "suggested_targets": recommendations,
            "interpretation": "Heuristic search-work diagnostic; neither win probability nor statistical confidence. Linear time scaling is approximate."}


_REPORT_NAMES = ('mechanism_effects', '_fmt', '_table', '_curve', 'cutoff_screening', 'study_diagnostics', 'write_study_report', 'family_study_diagnostics', 'write_family_study_report')
_EVIDENCE_NAMES = ('_timings', 'paired_interval', 'summarize_contrast')


def __getattr__(name):
    if name in _EVIDENCE_NAMES:
        return getattr(import_module('.studies.race', __package__), name)
    if name in _REPORT_NAMES:
        return getattr(import_module('.studies.report', __package__), name)
    raise AttributeError(name)


def __dir__():
    return sorted(set(globals()) | set(_EVIDENCE_NAMES) | set(_REPORT_NAMES))
