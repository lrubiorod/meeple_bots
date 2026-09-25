"""Study-only budget estimates and fixed-evidence allocation; no clocks or I/O."""
from copy import deepcopy
from dataclasses import replace
from statistics import median
from .._agent_config import MctsAgent, SoIsmctsAgent

def _study_budget(agent: MctsAgent | SoIsmctsAgent, calibration: dict) -> MctsAgent:
    if calibration.get("fixed_iterations") is not None:
        return replace(agent, iterations=calibration["fixed_iterations"], time_budget=None)
    return replace(agent, iterations=None, time_budget=calibration["decision_seconds"])


def estimated_round_seconds(phase: dict, calibration: dict) -> float:
    return calibration["mean_plies"] * sum(
        (phase["agents"][c[role]].get("time_budget") or phase["agents"][c[role]]["iterations"] * calibration["seconds_per_iteration"])
        for c in phase["contrasts"] for role in ("a", "b") if phase["agents"][c[role]] is not None)


def cost_pair(phase, contrast, calibration, prior_phases):
    result = contrast.get("result", {})
    elapsed = sum(result.get(t, {}).get("total_seconds", 0) for t in ("timing_a", "timing_b"))
    if result.get("seed_pairs", 0) and elapsed:
        return elapsed / result["seed_pairs"]
    if calibration.get("fixed_iterations"):
        return estimated_round_seconds({**phase, "contrasts": [contrast]}, calibration)
    # Recent completed candidate games replace the weak pilot's length estimate.
    # Normalize by decision budgets; retain 20% headroom for different opponents.
    for previous in reversed(prior_phases):
        if previous.get("status") != "complete":
            continue
        lengths = []
        for c in previous["contrasts"]:
            r = c.get("result", {})
            seconds = sum(r.get(t, {}).get("total_seconds", 0) for t in ("timing_a", "timing_b"))
            if r.get("seed_pairs", 0) >= 2 and seconds:
                budgets = sum((previous["agents"][c[role]] or {}).get("time_budget", 0) for role in ("a", "b"))
                if budgets:
                    lengths.append(seconds / r["seed_pairs"] / budgets)
        if lengths:
            return 1.2 * median(lengths) * sum((phase["agents"][contrast[role]] or {}).get("time_budget", 0) for role in ("a", "b"))
    return estimated_round_seconds({**phase, "contrasts": [contrast]}, calibration)


def plan_phase(phase, request, remaining, pair_costs, workers, stage):
    phase = deepcopy(phase)
    default_pairs = request["max_pairs"]
    for ci, contrast in enumerate(phase["contrasts"]):
        pairs = max(4, request.get("stage_games", {}).get(stage, 2 * default_pairs) // 2)
        contrast.update(target_pairs=pairs, workers=workers[ci],
                        timing_mode="isolated" if workers[ci] == 1 else "shared_cpu")
    # Keep complete evidence for a smaller shortlist. Include 20% headroom,
    # limiting parallelism to the paired jobs available in this race.
    retained, dropped = [], []
    for contrast, pair_cost in zip(phase["contrasts"], pair_costs, strict=True):
        cost = 1.2 * pair_cost * contrast["target_pairs"]
        contrast["estimated_seconds"] = cost
    minimum = phase.get("minimum_policy_comparisons", 0)
    representatives = phase["contrasts"][:minimum]
    if representatives:
        cost = sum(c["estimated_seconds"] for c in representatives) / min(request["workers"], 2*minimum)
        if cost > remaining:
            dropped = [{**c, "reason": "insufficient_budget_for_policy_coverage"} for c in phase["contrasts"]]
        else:
            retained.extend(representatives)
    for contrast in ([] if dropped else phase["contrasts"][minimum:]):
        proposed_cost = sum(c["estimated_seconds"] for c in [*retained, contrast]) / min(request["workers"], 2*(len(retained)+1))
        if proposed_cost <= remaining:
            retained.append(contrast)
        else:
            dropped.append({**contrast, "reason": "insufficient_budget"})
    phase["contrasts"] = retained
    phase["discarded_comparisons"] = dropped
    phase.update(planned_pairs=max((c["target_pairs"] for c in phase["contrasts"]), default=0),
                 planned_games=sum(2*c["target_pairs"] for c in phase["contrasts"]),
                 estimated_seconds=sum(c["estimated_seconds"] for c in phase["contrasts"]) / max(1, min(request["workers"], 2*len(phase["contrasts"]))),
                 allocation_mode="fixed_games")
    return phase
