"""Pure paired evidence, ranking, and promotion policy for Study."""
from collections import defaultdict
import math
from statistics import mean, median
from ..search_metrics import quantile

def _timings(rows, role):
    moves = []
    for row in rows:
        player = row["agent_a_player"] if role == "a" else 1 - row["agent_a_player"]
        selected = [m for m in row["result"]["moves"] if m["player"] == player]
        for move in selected:
            moves.append((move, (move["ply"] - 1) / row["result"]["plies"]))
    seconds = [m["decision_seconds"] for m, _ in moves]
    iterations = [m.get("search_iterations") or 0 for m, _ in moves]
    measured = [m for m, _ in moves if m.get("terminal_simulations") is not None
                and m.get("cutoff_simulations") is not None]
    terminal = sum(m["terminal_simulations"] for m in measured)
    cutoffs = sum(m["cutoff_simulations"] for m in measured)
    total = sum(seconds)
    attempts = sum((m.get("tree_reuse") or {}).get("transition_attempts", 0) for m, _ in moves)
    hits = sum((m.get("tree_reuse") or {}).get("transition_hits", 0) for m, _ in moves)
    return {"terminal_simulations": terminal if measured else None,
            "cutoff_simulations": cutoffs if measured else None,
            "terminal_fraction": terminal / (terminal + cutoffs) if terminal + cutoffs else None,
            "completion_measured_decisions": len(measured),
            "decisions": len(moves), "mean_seconds": mean(seconds) if seconds else None,
            "p50_seconds": median(seconds) if seconds else None,
            "p95_seconds": quantile(seconds, .95), "total_seconds": total,
            "mean_iterations": mean(iterations) if iterations else None,
            "median_iterations": median(iterations) if iterations else None,
            "iterations_per_second": sum(iterations) / total if total else None,
            "mean_nodes": mean([m.get("search_nodes") or 0 for m, _ in moves]) if moves else None,
            "maintenance_seconds": sum(m.get("maintenance_seconds", 0) for m, _ in moves),
            "reuse_hit_rate": hits / attempts if attempts else None,
            "quarters": [{"quarter": q+1, "decisions": sum(int(f*4) == q for _, f in moves),
                          "seconds": sum(m["decision_seconds"] for m, f in moves if int(f*4) == q),
                          "iterations": sum(m.get("search_iterations") or 0 for m, f in moves if int(f*4) == q)}
                         for q in range(4)]}


def paired_interval(values):
    """Distribution-free 95% Hoeffding bound for bounded independent seed blocks.

    Intentionally conservative for small samples and perfect sweeps. No false claim
    of equivalence from a degenerate bootstrap or an insignificant difference.
    """
    if not values:
        return [0.0, 1.0]
    radius = math.sqrt(math.log(40) / (2 * len(values)))
    return [max(0.0, mean(values)-radius), min(1.0, mean(values)+radius)]


def summarize_contrast(rows):
    seeds = defaultdict(list)
    for row in rows:
        seeds[row["result"]["seed"]].append(row)
    complete = [group for group in seeds.values()
                if len(group) == 2 and {r["agent_a_player"] for r in group} == {0, 1}]
    paired_rows = [row for group in complete for row in group]
    def score(row):
        return .5 if row["winner"] is None else float(row["winner"] == "agent_b")
    blocks = [mean(score(row) for row in group) for group in complete]
    interval = paired_interval(blocks)
    seats = {}
    for seat in (0, 1):
        group = [r for r in paired_rows if 1-r["agent_a_player"] == seat]
        seats[str(seat)] = {"wins": sum(r["winner"] == "agent_b" for r in group),
                            "draws": sum(r["winner"] is None for r in group),
                            "losses": sum(r["winner"] == "agent_a" for r in group)}
    return {"games": len(paired_rows), "unpaired_games": len(rows)-len(paired_rows),
            "seed_pairs": len(blocks), "score_b": mean(blocks) if blocks else None,
            "ci95_b": interval, "wins_b": sum(r["winner"] == "agent_b" for r in paired_rows),
            "draws": sum(r["winner"] is None for r in paired_rows),
            "losses_b": sum(r["winner"] == "agent_a" for r in paired_rows),
            "verdict": "b_ahead" if interval[0] > .5 else "a_ahead" if interval[1] < .5 else "inconclusive",
            "by_seat_b": seats, "seed_scores_b": {str(g[0]["result"]["seed"]): mean(score(r) for r in g) for g in complete},
            "timing_a": _timings(paired_rows, "a"), "timing_b": _timings(paired_rows, "b")}


def _rank(phase: dict) -> list[str]:
    """Exploratory ordering only; not a claim of a universal strongest agent."""
    scores = {name: [] for name in phase["agents"]}
    for contrast in phase["contrasts"]:
        if contrast.get("purpose") == "attribution":
            continue
        result = contrast.get("result", {})
        complete = result.get("seed_pairs", 0) >= max(2, contrast.get("target_pairs", 2))
        if phase.get("evidence_policy"):
            complete = result.get("seed_pairs", 0) >= max(4, contrast.get("target_pairs", 4))
            if not complete or not _promising(result) or (phase.get("evidence_policy") == "confirmation" and result.get("ci95_b", [0])[0] <= .5):
                scores[contrast["a"]].append(.5)
                continue
        if result.get("score_b") is not None and (not phase.get("incremental") or complete):
            scores[contrast["a"]].append(1 - result["score_b"])
            scores[contrast["b"]].append(result["score_b"])
    # In a common-control screen every challenger is scored against the same parent.
    # The parent's baseline score is 0.5, independent of how weak the challengers are.
    controls = {c["a"] for c in phase["contrasts"]}
    challengers = {c["b"] for c in phase["contrasts"]}
    for name in controls - challengers:
        if scores[name]:
            scores[name] = [.5]
    return sorted((name for name, values in scores.items() if values),
                  key=lambda name: (-sum(scores[name]) / len(scores[name]),
                                    tuple(phase.get("tie_priority", {}).get(name, [1]))))


def _promising(result):
    """Exploratory trend, not statistical confirmation: >=55% and one SE above parity."""
    if result.get("score_b", 0) < .55:
        return False
    values = list(result.get("seed_scores_b", {}).values())
    if len(values) < 2:
        return result.get("score_b", 0) >= .6
    average = mean(values)
    se = math.sqrt(sum((v-average)**2 for v in values) / (len(values)-1) / len(values))
    return average - se > .5


def _evidence_complete(contrast):
    return contrast.get("result", {}).get("seed_pairs", 0) >= max(4, contrast.get("target_pairs", 4))


def _reverse_result(result):
    return {**result, "score_b": 1-result["score_b"],
            "seed_scores_b": {k: 1-v for k, v in result.get("seed_scores_b", {}).items()}}


def _group_leaders(phase: dict) -> dict[str, str]:
    leaders = {}
    for group, members in phase["groups"].items():
        subset = {**phase, "agents": {n: phase["agents"][n] for n in members},
                  "contrasts": [c for c in phase["contrasts"] if c["a"] in members and c["b"] in members]}
        leaders[group] = next(iter(_rank(subset)), phase.get("fallback", members[0]))
        if phase.get("pw_policies") and phase.get("fallback") and leaders[group] == members[0]:
            # Disabling an already widened baseline also needs supported evidence.
            evidence = next((c for c in phase["contrasts"] if c["b"] == phase["fallback"]), None)
            if evidence is None or not _evidence_complete(evidence) or not _promising(_reverse_result(evidence["result"])):
                leaders[group] = phase["fallback"]
    return leaders
