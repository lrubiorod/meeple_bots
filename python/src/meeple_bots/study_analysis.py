"""Paired-seed summaries and a dependency-free HTML report for automatic studies."""
from collections import defaultdict
from html import escape
import json
import math
from pathlib import Path
from statistics import mean, median


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(len(ordered)-1, low+1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position-low)


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


def mechanism_effects(phase):
    effects = []
    for factor in ("selector", "tree_reuse", "transpositions"):
        contrasts = [c for c in phase["contrasts"] if c["factor"] == factor]
        if not contrasts:
            continue
        # Each seed is one statistical block across all backgrounds as well as seats.
        shared = None
        for contrast in contrasts:
            keys = set(contrast.get("result", {}).get("seed_scores_b", {}))
            shared = keys if shared is None else shared & keys
        blocks = [mean(c["result"]["seed_scores_b"][seed] for c in contrasts) for seed in sorted(shared or [])]
        interval = paired_interval(blocks)
        effects.append({"factor": factor, "backgrounds": len(contrasts), "seed_blocks": len(blocks),
                        "score_enabled_or_tuned": mean(blocks) if blocks else None, "ci95": interval,
                        "verdict": "enabled_or_tuned_ahead" if interval[0] > .5 else "disabled_or_uct_ahead" if interval[1] < .5 else "inconclusive",
                        "interpretation": "conditional on each tuned family: fixed selector, heuristic, rollout, horizon, exploration and search budget"})
    return effects


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>" for row in rows) + "</tbody></table>"


def _curve(phase):
    points = [(c, c.get("result", {})) for c in phase["contrasts"] if c["factor"] == "anchor" and c.get("result", {}).get("score_b") is not None]
    if not points:
        return ""
    def cost(c, r):
        return phase.get("isolated_costs", {}).get(c["b"], r["timing_b"]["mean_seconds"])
    max_time = max(cost(c, r) for c, r in points) or 1
    svg = ['<svg viewBox="0 0 760 280" role="img" aria-label="Observed score against the fixed anchor versus estimated isolated decision seconds">',
           '<path d="M55 15V230H735" fill="none" stroke="black"/>',
           '<text x="55" y="265">Estimated isolated seconds →</text><text x="5" y="15">1.0</text><text x="5" y="230">0.0</text>']
    for c, r in points:
        x = 55 + 660 * cost(c, r) / max_time
        y = 230 - 210 * r["score_b"]
        lo, hi = r["ci95_b"]
        color = "#276baf" if c["b"].startswith("full-") else "#a84d16"
        svg.append(f'<line x1="{x}" x2="{x}" y1="{230-210*hi}" y2="{230-210*lo}" stroke="{color}" opacity=".4"/>')
        svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"><title>{escape(c["b"])}: score {_fmt(r["score_b"])}; {_fmt(cost(c, r))}s</title></circle>')
    svg.append(f'<text x="650" y="250">{max_time:.4f}s</text></svg>')
    return "".join(svg)


def cutoff_screening(phase):
    """Keep an uncertain cutoff; rejection requires all candidates below the threshold."""
    observed = [c for c in phase.get("contrasts", []) if c.get("result", {}).get("score_b") is not None]
    ordered = sorted(observed, key=lambda c: (-c["result"]["score_b"],
                     tuple(phase.get("tie_priority", {}).get(c["b"], [1]))))
    best = ordered[0] if ordered else None
    complete = phase.get("status") == "complete"
    enough = bool(observed) and len(observed) == len(phase.get("contrasts", [])) and all(
        c["result"].get("seed_pairs", 0) >= 8 for c in observed)
    rejected = enough and all(c["result"].get("ci95_b", [0, 1])[1] < .45 for c in observed)
    admitted = not rejected if complete and best else None
    reason = ("screening_incomplete" if admitted is None else "all_cutoffs_below_threshold" if rejected else
              "insufficient_seed_pairs" if not enough else
              "score_at_least_threshold" if best["result"]["score_b"] >= .45 else "inconclusive_keep_cutoff")
    return {"status": "pending" if admitted is None else "rejected" if rejected else
                      "provisional" if reason in ("insufficient_seed_pairs", "inconclusive_keep_cutoff") else "admitted",
            "admitted": admitted, "threshold_score": .45, "minimum_seed_pairs": 8,
            "candidate": best["b"] if admitted else None,
            "best_observed_candidate": best["b"] if best else None,
            "best_observed_score": best["result"]["score_b"] if best else None,
            "evidence": "below_threshold" if rejected else "insufficient" if not enough else "screening_only",
            "reason": reason}


def study_diagnostics(state):
    """Describe admission and measured effects; never infer universal strength gains."""
    if state.get("request", {}).get("version", 0) >= 11:
        return family_study_diagnostics(state)
    phases = state.get("phases", {})
    horizon = phases.get("horizon_check", phases.get("horizons", {}))
    selection = cutoff_screening(horizon)
    effects = []
    for phase_name, phase in phases.items():
        for contrast in phase.get("contrasts", []):
            if contrast.get("purpose") not in ("attribution", "ablation") and contrast["factor"] not in (
                    "tree_reuse", "transpositions", "combined_mechanisms", "iterations", "cutoff_equal_time", "parameters", "refinement"):
                continue
            a = phase.get("agents", {}).get(contrast["a"])
            b = phase.get("agents", {}).get(contrast["b"])
            if not a or not b:
                continue
            changes = {key: {"before": a.get(key), "after": b.get(key)}
                       for key in sorted(a.keys() | b.keys()) if a.get(key) != b.get(key)}
            if not changes:
                continue
            result = contrast.get("result", {})
            score = result.get("score_b")
            mode = ("equal_time" if a.get("time_budget") is not None and a.get("time_budget") == b.get("time_budget")
                    else "equal_iterations" if a.get("iterations") is not None and a.get("iterations") == b.get("iterations")
                    else "unequal_budget")
            family = "full" if b["rollout_depth"] == 1024 else "cutoff"
            effects.append({"phase": phase_name, "family": family,
                            "evidence_source": "held_out_ablation" if contrast.get("purpose") == "ablation" else "screening", "before": contrast["a"], "after": contrast["b"],
                            "factor": next(iter(changes)) if len(changes) == 1 else "combined_changes",
                            "changes": changes, "isolated_factor": len(changes) == 1,
                            "comparison": mode, "timing_mode": contrast.get("timing_mode", "isolated"),
                            "phase_complete": phase.get("status") == "complete",
                            "score_percent": 100 * score if score is not None else None,
                            "advantage_pp": 100 * (score - .5) if score is not None else None,
                            "advantage_ci95_pp": [100 * (v - .5) for v in result.get("ci95_b", [0, 1])],
                            "seed_pairs": result.get("seed_pairs", 0),
                            "verdict": result.get("verdict", "not_run"), "trace": contrast.get("trace")})
    rankings = []
    for family, mode, source in sorted({(e["family"], e["comparison"], e["evidence_source"]) for e in effects}):
        measured = [e for e in effects if e["family"] == family and e["comparison"] == mode and e["evidence_source"] == source
                    and e["phase_complete"] and e["advantage_pp"] is not None and e["isolated_factor"]]
        measured.sort(key=lambda e: (-e["advantage_pp"], e["phase"], e["before"], e["after"]))
        if measured:
            rankings.append({"family": family, "comparison": mode, "evidence_source": source, "effects": measured,
                             "interpretation": "Descriptive ranking of observed head-to-head advantages in different contexts; not a causal ranking across contexts or a percentage increase in playing strength. Screening is not held-out validation."})
    confirmation = phases.get("confirmation", {})
    final_match = next((c for c in confirmation.get("contrasts", []) if c["factor"] == "cutoff_confirmation"), None)
    verdict = final_match.get("result", {}).get("verdict") if final_match else None
    primary = next((c for c in confirmation.get("contrasts", []) if c.get("primary")), None)
    finished = (primary.get("result", {}).get("seed_pairs", 0) >= primary["target_pairs"]
                if primary and primary.get("target_pairs") else confirmation.get("status") == "complete")
    winner = None
    if not finished:
        final_status = "pending"
    elif final_match is None:
        final_status, winner = "only_full_survived", "finalist-full"
    elif verdict in ("a_ahead", "b_ahead"):
        final_status = "resolved"
        winner = final_match["b"] if verdict == "b_ahead" else final_match["a"]
    else:
        final_status = "inconclusive"
    return {"search_complete": finished, "cutoff_selection": selection, "improvement_comparisons": effects, "improvement_rankings": rankings,
            "final_selection": {"status": final_status, "candidate": winner,
                                "provisional_candidate": winner or ("finalist-full" if "finalist-full" in confirmation.get("agents", {}) else None),
                                "interpretation": "Relative to tested finalists at equal time; phase completion does not imply a decisive result."},
            "phase_evidence": {name: {"execution": p.get("status"),
                 "minimum_observed_seed_pairs": min((c.get("result", {}).get("seed_pairs", 0) for c in p.get("contrasts", [])), default=0),
                 "decisive_contrasts": sum(c.get("result", {}).get("verdict") in ("a_ahead", "b_ahead") for c in p.get("contrasts", [])),
                 "contrasts": len(p.get("contrasts", []))} for name, p in phases.items()}}


def write_study_report(output: Path, state: dict):
    if state.get("request", {}).get("version", 0) >= 11:
        return write_family_study_report(output, state)
    summary = {"status": state["status"], "game": state["request"]["game"],
               "budget_seconds": state["budget_seconds"], "spent_seconds": state["spent_seconds"],
               "calibration": state["calibration"], "phases": state["phases"],
               "candidate_profiles": state.get("candidate_profiles", {}), "last_error": state.get("last_error"),
               "budget_plan": state.get("budget_plan")}
    summary.update(study_diagnostics(state))
    mechanisms = state["phases"].get("mechanisms")
    summary["mechanism_effects"] = mechanism_effects(mechanisms) if mechanisms else []
    caveats = [
        "Terminal references use neutral evaluation and a 1024-step safety cap, not a known maximum game length. Any cutoff means the reference is truncated; completion counters are reported below. A single simulation can exceed the decision deadline.",
        "Depths 16/32/64 with neutral/H0/H1 are screened at equal short time; two cutoffs are checked at target time. Low-sample admission is provisional. Rejection of shortlisted cutoffs is not a claim about untested alternatives. Full and optional cutoff families are tuned separately, then compared on held-out seeds.",
        "All results are preliminary and relative to the tested opponents and budgets; candidates are not automatically promoted to baselines.",
        "Intervals use paired seed blocks and a conservative 95% Hoeffding bound. Adaptive screening intervals and rankings are descriptive, with no claim of inferiority for candidates not prioritized. Confirmation sample sizes are fixed before play; no multiple-comparison correction.",
        "All four mechanism settings are screened after parameter and iteration tuning; one winner per family enters local refinement and two refined alternatives per family receive target-time checks. Screening rankings remain provisional. No global optimum is guaranteed.",
        "The primary confirmation has its own cap (default 32 pairs); auxiliaries and ablations have a smaller independent cap (default 4). All phases have separate seed namespaces. Inconclusive is not evidence of equal strength or iteration saturation.",
        "Calibration, cost benchmarks and timed matches run in isolation. All fixed-iteration matches may share CPU; their latency is not isolated performance. Timing requires a release native build.",
        "The external reference is excluded from screening and selection and participates only in auxiliary equal-time confirmation.",
        "Phases run in order. After initial coverage, exploratory work is reduced to protect later allocations. Auxiliary comparisons can be omitted. Insufficient total budget leaves provisional candidates; primary search completion is reported separately from optional diagnostics. A running batch of swapped-seat pairs may exceed the deadline.",
        "Seeds vary the search RNG and, where supported, the initial setup. Distinct seeds need not mean distinct starting boards.",
    ]
    summary["cutoff_decisions"] = []
    confirmation = state["phases"].get("confirmation", {})
    for contrast in confirmation.get("contrasts", []):
        if contrast["factor"] != "cutoff_confirmation":
            continue
        result = contrast.get("result", {})
        timing = result.get("timing_a", {})
        verified = (timing.get("cutoff_simulations") == 0
                    and (timing.get("terminal_simulations") or 0) > 0
                    and timing.get("completion_measured_decisions") == timing.get("decisions"))
        decision = ("pending_confirmation" if confirmation.get("status") != "complete" else
                    "reference_truncated_or_unverified" if not verified else
                    "cutoff_supported" if result.get("verdict") == "b_ahead" else "retain_terminal")
        summary["cutoff_decisions"].append({"terminal": contrast["a"], "cutoff": contrast["b"],
                                            "decision": decision})
    summary["limitations"] = caveats
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    parts = ['<!doctype html><meta charset="utf-8"><title>MCTS diagnostic study</title>',
             '<style>body{font:16px system-ui;max-width:1250px;margin:2rem auto;padding:0 1rem;color:#183040}table{border-collapse:collapse;width:100%;margin:1rem 0}td,th{border:1px solid #ccd5dd;padding:.45rem;text-align:left}th{background:#eef3f6}svg{width:100%;max-width:760px}li{margin:.5rem 0}code{background:#eef3f6}</style>',
             f'<h1>{escape(summary["game"])} — MCTS diagnostic study</h1>',
             f'<p>Status: <b>{escape(state["status"])}</b>. Budget used: {state["spent_seconds"]:.1f}/{state["budget_seconds"]:.1f}s.</p>',
             '<p><a href="study.json">Frozen plans and provenance</a> · <a href="summary.json">Machine-readable results</a> · <a href="baseline.toml">Starting profile</a></p>',
             '<h2>Interpretation</h2><ul>' + ''.join(f'<li>{escape(c)}</li>' for c in caveats) + '</ul>']
    plan = summary.get("budget_plan")
    if plan:
        parts.append('<h2>Initial budget estimate</h2>')
        parts.append(_table(['Phase', 'Estimated seconds', 'Allocated seconds'],
                           [(p['phase'], _fmt(p['estimated_seconds'], 1), _fmt(p['allocated_seconds'], 1)) for p in plan['phases']]))
    final = summary["final_selection"]
    parts.append(f'<p>Primary search complete: {summary["search_complete"]}. Final comparison: <b>{escape(final["status"])}</b>; selected candidate: {escape(str(final["candidate"]))}. Execution completion and statistical evidence are separate.</p>')
    selection = summary["cutoff_selection"]
    parts.append('<h2>Cutoff admission</h2>')
    parts.append(_table(['Status', 'Selected cutoff', 'Best observed cutoff', 'Observed score %', 'Threshold %', 'Reason'],
                        [(selection['status'], selection['candidate'], selection['best_observed_candidate'],
                          _fmt(100 * selection['best_observed_score'] if selection['best_observed_score'] is not None else None, 1),
                          45, selection['reason'])]))
    parts.append('<h2>Observed improvements</h2><p>Score = wins + half draws. A 60% score against the previous candidate is +10 percentage points above parity, not a 20% increase in playing strength. Rankings compare observed advantages in different contexts and are exploratory. Equal-time and equal-iteration results are kept separate; combined changes are not attributed to one factor.</p>')
    for group in summary['improvement_rankings']:
        parts.append(f'<h3>{escape(group["family"])} — {escape(group["comparison"])} — {escape(group["evidence_source"])}</h3>')
        parts.append(_table(['Factor', 'Before', 'After', 'Phase', 'Score %', 'Advantage pp', '95% interval pp', 'Seed pairs', 'Conclusion'],
                            [(e['factor'], e['before'], e['after'], e['phase'], _fmt(e['score_percent'], 1),
                              _fmt(e['advantage_pp'], 1), ' – '.join(_fmt(v, 1) for v in e['advantage_ci95_pp']),
                              e['seed_pairs'], e['verdict']) for e in group['effects']]))
    parts.append('<details><summary>All change measurements, including combined and pending comparisons</summary>')
    parts.append(_table(['Phase', 'Before', 'After', 'Changed parameters', 'Budget comparison', 'Advantage pp', 'Phase complete'],
                        [(e['phase'], e['before'], e['after'], json.dumps(e['changes'], sort_keys=True), e['comparison'],
                          _fmt(e['advantage_pp'], 1), e['phase_complete']) for e in summary['improvement_comparisons']]))
    parts.append('</details>')
    if summary["cutoff_decisions"]:
        parts.append('<h2>Held-out cutoff decisions</h2>')
        parts.append(_table(['Terminal control', 'Cutoff candidate', 'Conclusion'],
                            [(d['terminal'], d['cutoff'], d['decision']) for d in summary['cutoff_decisions']]))
    if summary["mechanism_effects"]:
        parts.append('<h2>Mechanism effects across backgrounds</h2>')
        parts.append(_table(['Factor', 'Seed blocks', 'Score enabled/Tuned', '95% interval', 'Conclusion'],
                            [(e['factor'], e['seed_blocks'], _fmt(e['score_enabled_or_tuned']), ' – '.join(_fmt(x) for x in e['ci95']), e['verdict']) for e in summary['mechanism_effects']]))
    if state.get("last_error"):
        parts.append(f'<p>Stopped: {escape(state["last_error"])}</p>')
    for name, phase in state["phases"].items():
        parts.append(f'<h2>{escape(name)} — {escape(phase["status"])}</h2>')
        parts.append(_table(['A', 'B', 'Question', 'Max workers', 'Timing', 'Paired seeds', 'Target pairs', 'Allocation decision', 'B W/D/L', 'B score', '95% interval', 'Conclusion', 'A mean s', 'B mean s', 'B p95 s'], [
            (c['a'], c['b'], c['factor'], c.get('workers', 1), c.get('timing_mode', 'isolated'), (r := c.get('result', {})).get('seed_pairs', 0), c.get('target_pairs', phase.get('planned_pairs')), c.get('stop_reason', 'allocated'),
             f"{r.get('wins_b', 0)}/{r.get('draws', 0)}/{r.get('losses_b', 0)}", _fmt(r.get('score_b')),
             ' – '.join(_fmt(x) for x in r.get('ci95_b', [0, 1])), r.get('verdict', 'not_run'),
             _fmt(r.get('timing_a', {}).get('mean_seconds')), _fmt(r.get('timing_b', {}).get('mean_seconds')),
             _fmt(r.get('timing_b', {}).get('p95_seconds'))) for c in phase['contrasts']]))
        parts.append('<details><summary>Search cost and retained-state diagnostics</summary>')
        cost_rows = []
        for contrast in phase['contrasts']:
            for role in ('a', 'b'):
                timing = contrast.get('result', {}).get('timing_' + role)
                if timing is None:
                    continue
                cost_rows.append((contrast[role], contrast['b' if role == 'a' else 'a'], timing['decisions'],
                                  _fmt(timing['iterations_per_second'], 1), _fmt(timing['mean_nodes'], 1),
                                  _fmt(timing['maintenance_seconds']), _fmt(timing['reuse_hit_rate']),
                                  timing.get('terminal_simulations'), timing.get('cutoff_simulations'),
                                  _fmt(timing.get('terminal_fraction')),
                                  ' / '.join(str(q['iterations']) for q in timing['quarters'])))
        parts.append(_table(['Agent', 'Opponent', 'Decisions', 'Iterations/s', 'Mean nodes', 'Maintenance s', 'Reuse hit rate', 'Terminal simulations', 'Cutoff simulations', 'Terminal fraction', 'Iterations Q1/Q2/Q3/Q4'], cost_rows))
        parts.append('</details>')
        if name == 'iterations':
            parts.append(_curve(phase))
            parts.append('<p>Fast/balanced/strong labels describe observed trade-offs. A 5-point screening tolerance selects the balanced candidate; it does not prove equivalence.</p>')
        parts.append('<ul>' + ''.join(f'<li><a href="{escape(c["trace"], quote=True)}">{escape(c["a"])} vs {escape(c["b"])}</a></li>' for c in phase['contrasts'] if 'trace' in c) + '</ul>')
    parts.append('<h2>Candidate profiles</h2><ul>' + ''.join(f'<li><a href="{escape(path, quote=True)}">{escape(name)}</a></li>' for name, path in state.get('candidate_profiles', {}).items()) + '</ul>')
    (output / 'report.html').write_text('\n'.join(parts), encoding='utf-8')


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


def family_study_diagnostics(state):
    phases = state.get("phases", {})
    confirmation = phases.get("confirmation", {})
    primary = next((c for c in confirmation.get("contrasts", []) if c.get("primary")), {})
    result = primary.get("result", {})
    complete = confirmation.get("status") == "complete"
    tested = (primary.get("target_pairs", 0) >= 2
              and result.get("seed_pairs", 0) >= primary["target_pairs"])
    candidate = "finalist" if "finalist" in confirmation.get("agents", {}) else None
    verdict = result.get("verdict", "not_measured")
    effects = []
    costs = []
    for name, phase in phases.items():
        for c in phase.get("contrasts", []):
            a, b = (phase["agents"][c[role]] for role in ("a", "b"))
            r = c.get("result", {})
            score = r.get("score_b")
            changes = {k: {"before": a.get(k), "after": b.get(k)} for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
            effects.append({"phase": name, "before": c["a"], "after": c["b"], "changes": changes,
                            "score_b": score, "advantage_pp": 100*(score-.5) if score is not None else None,
                            "ci95_b": r.get("ci95_b"), "seed_pairs": r.get("seed_pairs", 0),
                            "reason": c.get("stop_reason", "evaluated" if score is not None else "pending"),
                            "evidence": "held_out" if name == "confirmation" else "exploratory"})
            for role in ("a", "b"):
                timing = r.get("timing_" + role)
                if timing and timing.get("decisions"):
                    branching = (state.get("calibration") or {}).get("search_adequacy", {}).get("representative_branching", 1)
                    adequacy = search_adequacy([{"iterations": timing["median_iterations"], "legal_actions": branching,
                                                "milliseconds": timing["p50_seconds"] * 1000}], state["request"]["target_match_time"])
                    costs.append({"phase": name, "candidate": c[role], "search_adequacy": adequacy["category"],
                                  "median_iterations_per_decision": timing["median_iterations"],
                                  "iterations_per_second": timing["iterations_per_second"],
                                  "actual_elapsed_seconds": timing["total_seconds"], "mean_decision_seconds": timing["mean_seconds"],
                                  "terminal_fraction": timing["terminal_fraction"]})
    return {"mode": state["request"]["mode"], "search_complete": complete,
            "final_selection": {"candidate": candidate, "status": "confirmed" if complete and tested and verdict == "b_ahead" else "provisional",
                                "competitive_confidence": verdict if complete and tested else "not_measured",
                                "score_b": result.get("score_b"), "ci95_b": result.get("ci95_b"),
                                "seed_pairs": result.get("seed_pairs", 0),
                                "interpretation": "Nominee fixed before confirmation. Inconclusive evidence does not establish equivalence; a loss is reported without post-hoc reselection."},
            "improvement_comparisons": effects, "candidate_search_costs": costs}


def write_family_study_report(output, state):
    summary = {**state, **family_study_diagnostics(state)}
    summary["limitations"] = [
        "Only the requested evaluator family is optimized; compare separate studies in a tournament.",
        "All competitive contrasts use the same decision time, in isolation. Iterations are measured work, not the selection budget.",
        "Practical full-depth means >=99% terminal simulations at every sampled position; this finite empirical probe is not a rules guarantee. Unverified safety horizons are explicitly labelled.",
        "Rollout depth counts player decisions, excludes Chance, and completes the physical turn before cutoff. Terminal stops immediately.",
        "Search adequacy thresholds are heuristic diagnostics, separate from paired-seed competitive confidence.",
        "95% intervals use conservative paired-seed Hoeffding bounds. Screening is adaptive/exploratory, not independent confirmation; no multiple-comparison correction.",
        "The final nominee is frozen before fresh confirmation seeds. Budget-limited/untested alternatives are not proven inferior.",
        "Pilot length and random representative positions are preliminary estimates. Actual game costs can differ; target match time is not a match deadline.",
        "Phase allocations roll forward; match pairs and individual searches can overshoot time budgets. Old study protocols require a new output directory.",
    ]
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    cal = state.get("calibration") or {}
    parts = ['<!doctype html><meta charset="utf-8"><title>MCTS family study</title>',
             '<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:.5rem;text-align:left}</style>',
             f'<h1>{escape(state["request"]["game"])} — {escape(state["request"]["mode"])}</h1>',
             f'<p>Status: {escape(state["status"])}. Used {state["spent_seconds"]:.1f}/{state["budget_seconds"]:.1f}s.</p>',
             '<p><a href="summary.json">Summary and frozen comparisons</a> · <a href="study.json">Checkpoint</a></p>',
             f'<p>RAVE search requested: {state["request"].get("rave_search", False)}; supported: {"uct_rave" in state["request"].get("selection_policies", [])}.</p>',
             '<h2>Calibration and search adequacy</h2>',
             _table(['Measurement', 'Value'], [(k, json.dumps(cal.get(k))) for k in
                    ('target_match_time', 'estimated_game_decisions', 'safety_margin', 'decision_seconds', 'decision_time_source', 'fixed_iterations', 'cutoff_depths')])]
    parts.append(_table(['Sampled player decisions', 'Search ms', 'Iterations', 'Legal actions', 'Terminal / cutoff simulations'],
                        [(t['sampled_ply'], t['milliseconds'], t['iterations'], t['legal_actions'],
                          f"{t.get('terminal_simulations')} / {t.get('cutoff_simulations')}") for t in cal.get('position_timings', [])]))
    parts.append('<h2>Incremental stages</h2>' + _table(['Stage', 'Requested'], [(key, state['request'].get(key, False)) for key in ('selection_search', 'rave_search', 'mechanism_search', 'pw_search', 'depth_search')]))
    parts.append(f'<p>PW search requested: {state["request"].get("pw_search", False)}; supported: {state["request"].get("pw_supported", False)}.</p>')
    if state.get('pw_budget'):
        parts.append('<h2>Shared PW budget</h2><pre>' + escape(json.dumps(state['pw_budget'], indent=2)) + '</pre>')
    if state.get('rave_budget'):
        parts.append('<h2>Shared RAVE budget</h2><pre>' + escape(json.dumps(state['rave_budget'], indent=2)) + '</pre>')
    horizon = cal.get("horizon", {})
    parts.append(_table(['Horizon kind', 'Decision depth', 'Measured terminal fraction'], [(horizon.get('kind'), horizon.get('depth'), horizon.get('terminal_fraction'))]))
    parts.append('<pre>' + escape(json.dumps(cal.get('search_adequacy', {}), indent=2)) + '</pre>')
    parts.extend('<p>Warning: ' + escape(w) + '</p>' for w in cal.get('warnings', []))
    parts.append('<h2>Independent competitive evidence</h2><pre>' + escape(json.dumps(summary['final_selection'], indent=2)) + '</pre>')
    for name, phase in state['phases'].items():
        parts.append(f'<h2>{escape(name)} — {escape(phase["status"])}</h2>')
        if phase.get('pw_decisions'):
            parts.append(_table(['Family', 'PW calibration decision'], phase['pw_decisions'].items()))
        if phase.get('rave_decisions'):
            parts.append(_table(['Depth family', 'RAVE calibration decision'], phase['rave_decisions'].items()))
        parts.append(_table(['A', 'B', 'Changed factor', 'Pairs', 'B score', '95% CI', 'Disposition'], [
            (c['a'], c['b'], c['factor'], (r := c.get('result', {})).get('seed_pairs', 0), r.get('score_b'),
             r.get('ci95_b'), c.get('stop_reason', r.get('verdict', 'pending'))) for c in phase['contrasts']]))
    parts.append('<h2>Measured candidate search work</h2>')
    fields = ('phase', 'candidate', 'search_adequacy', 'median_iterations_per_decision', 'iterations_per_second', 'mean_decision_seconds', 'terminal_fraction')
    parts.append(_table(fields, [[row[k] for k in fields] for row in summary['candidate_search_costs']]))
    parts.append('<h2>Candidate profiles</h2><ul>' + ''.join(f'<li><a href="{escape(path)}">{escape(name)}</a></li>' for name, path in state.get('candidate_profiles', {}).items()) + '</ul>')
    parts.append('<h2>Limitations</h2><ul>' + ''.join('<li>' + escape(s) + '</li>' for s in summary['limitations']) + '</ul>')
    (output / 'report.html').write_text('\n'.join(parts))
