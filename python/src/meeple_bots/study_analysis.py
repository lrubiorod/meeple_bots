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
    total = sum(seconds)
    attempts = sum((m.get("tree_reuse") or {}).get("transition_attempts", 0) for m, _ in moves)
    hits = sum((m.get("tree_reuse") or {}).get("transition_hits", 0) for m, _ in moves)
    return {"decisions": len(moves), "mean_seconds": mean(seconds) if seconds else None,
            "p50_seconds": median(seconds) if seconds else None,
            "p95_seconds": quantile(seconds, .95), "total_seconds": total,
            "mean_iterations": mean(iterations) if iterations else None,
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
                        "interpretation": "conditional on fixed heuristic, rollout, horizon, UCT exploration and decision time"})
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
    max_time = max(r["timing_b"]["mean_seconds"] for _, r in points) or 1
    svg = ['<svg viewBox="0 0 760 280" role="img" aria-label="Observed score against the fixed anchor versus mean decision seconds">',
           '<path d="M55 15V230H735" fill="none" stroke="black"/>',
           '<text x="55" y="265">Mean decision seconds →</text><text x="5" y="15">1.0</text><text x="5" y="230">0.0</text>']
    for c, r in points:
        x = 55 + 660 * r["timing_b"]["mean_seconds"] / max_time
        y = 230 - 210 * r["score_b"]
        lo, hi = r["ci95_b"]
        color = "#276baf" if c["b"].startswith("family0") else "#a84d16"
        svg.append(f'<line x1="{x}" x2="{x}" y1="{230-210*hi}" y2="{230-210*lo}" stroke="{color}" opacity=".4"/>')
        svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"><title>{escape(c["b"])}: score {_fmt(r["score_b"])}; {_fmt(r["timing_b"]["mean_seconds"])}s</title></circle>')
    svg.append(f'<text x="650" y="250">{max_time:.4f}s</text></svg>')
    return "".join(svg)


def write_study_report(output: Path, state: dict):
    summary = {"status": state["status"], "game": state["request"]["game"],
               "budget_seconds": state["budget_seconds"], "spent_seconds": state["spent_seconds"],
               "calibration": state["calibration"], "phases": state["phases"],
               "candidate_profiles": state.get("candidate_profiles", {}), "last_error": state.get("last_error")}
    mechanisms = state["phases"].get("mechanisms")
    summary["mechanism_effects"] = mechanism_effects(mechanisms) if mechanisms else []
    caveats = [
        "All results are preliminary and relative to the tested opponents and budgets; candidates are not automatically promoted to baselines.",
        "Intervals use paired seed blocks and a conservative 95% Hoeffding bound. No multiple-comparison correction; screening rankings are exploratory.",
        "Only confirmation uses held-out seeds. Inconclusive is not evidence of equal strength or iteration saturation.",
        "Calibration, timed matches and iteration-anchor cost measurements run in isolation. Other fixed-iteration matches may share CPU; their latency is not isolated performance. Timing requires a release native build.",
        "The calibrated external reference is excluded from screening and selection. Equal-time and original-budget reference contrasts answer different questions.",
        "Phases run in order. An insufficient budget leaves partial results without promoting that phase. A running batch of swapped-seat pairs may exceed the deadline.",
        "Seeds vary the search RNG and, where supported, the initial setup. Distinct seeds need not mean distinct starting boards.",
    ]
    summary["limitations"] = caveats
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    parts = ['<!doctype html><meta charset="utf-8"><title>MCTS diagnostic study</title>',
             '<style>body{font:16px system-ui;max-width:1250px;margin:2rem auto;padding:0 1rem;color:#183040}table{border-collapse:collapse;width:100%;margin:1rem 0}td,th{border:1px solid #ccd5dd;padding:.45rem;text-align:left}th{background:#eef3f6}svg{width:100%;max-width:760px}li{margin:.5rem 0}code{background:#eef3f6}</style>',
             f'<h1>{escape(summary["game"])} — MCTS diagnostic study</h1>',
             f'<p>Status: <b>{escape(state["status"])}</b>. Budget used: {state["spent_seconds"]:.1f}/{state["budget_seconds"]:.1f}s.</p>',
             '<p><a href="study.json">Frozen plans and provenance</a> · <a href="summary.json">Machine-readable results</a> · <a href="baseline.toml">Starting profile</a></p>',
             '<h2>Interpretation</h2><ul>' + ''.join(f'<li>{escape(c)}</li>' for c in caveats) + '</ul>']
    if summary["mechanism_effects"]:
        parts.append('<h2>Mechanism effects across backgrounds</h2>')
        parts.append(_table(['Factor', 'Seed blocks', 'Score enabled/Tuned', '95% interval', 'Conclusion'],
                            [(e['factor'], e['seed_blocks'], _fmt(e['score_enabled_or_tuned']), ' – '.join(_fmt(x) for x in e['ci95']), e['verdict']) for e in summary['mechanism_effects']]))
    if state.get("last_error"):
        parts.append(f'<p>Stopped: {escape(state["last_error"])}</p>')
    for name, phase in state["phases"].items():
        parts.append(f'<h2>{escape(name)} — {escape(phase["status"])}</h2>')
        parts.append(_table(['A', 'B', 'Question', 'Max workers', 'Timing', 'Paired seeds', 'B W/D/L', 'B score', '95% interval', 'Conclusion', 'A mean s', 'B mean s', 'B p95 s'], [
            (c['a'], c['b'], c['factor'], c.get('workers', 1), c.get('timing_mode', 'isolated'), (r := c.get('result', {})).get('seed_pairs', 0),
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
                                  ' / '.join(str(q['iterations']) for q in timing['quarters'])))
        parts.append(_table(['Agent', 'Opponent', 'Decisions', 'Iterations/s', 'Mean nodes', 'Maintenance s', 'Reuse hit rate', 'Iterations Q1/Q2/Q3/Q4'], cost_rows))
        parts.append('</details>')
        if name == 'iterations':
            parts.append(_curve(phase))
            parts.append('<p>Fast/balanced/strong labels describe observed trade-offs. A 5-point screening tolerance selects the balanced candidate; it does not prove equivalence.</p>')
        parts.append('<ul>' + ''.join(f'<li><a href="{escape(c["trace"], quote=True)}">{escape(c["a"])} vs {escape(c["b"])}</a></li>' for c in phase['contrasts'] if 'trace' in c) + '</ul>')
    parts.append('<h2>Candidate profiles</h2><ul>' + ''.join(f'<li><a href="{escape(path, quote=True)}">{escape(name)}</a></li>' for name, path in state.get('candidate_profiles', {}).items()) + '</ul>')
    (output / 'report.html').write_text('\n'.join(parts), encoding='utf-8')
