"""Operating resource conversion shared by analyze and study."""
from math import isfinite


def decision_budget(target_match_time, expected_decisions, safety_margin=1.2):
    if any(not isfinite(v) or v <= 0 for v in (target_match_time, expected_decisions, safety_margin)):
        raise ValueError("match time, expected decisions and safety margin must be finite and positive")
    return target_match_time / (expected_decisions * safety_margin)
