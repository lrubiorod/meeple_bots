"""Study presentation and diagnostic reports; no search execution."""
from html import escape
import json
from pathlib import Path
from statistics import mean
from .race import paired_interval
from ..search_metrics import search_adequacy
from .planning import MAX_EXTENSION_ROUNDS, _phase_enabled
from .race import _group_leaders
from .tuners import TUNING_FIELDS


def compute_budget(state):
    request, cal = state['request'], state['calibration']
    source = ('explicit --decision-time' if request['decision_seconds'] is not None else
              'baseline/config' if request['baseline_supplied'] else '--target-match-time')
    fixed = cal.get('fixed_iterations')
    timings = cal.get('position_timings', [])
    seconds = (mean(t['milliseconds'] for t in timings) / 1000 if timings else None) if fixed else cal['decision_seconds']
    decisions = cal.get('estimated_game_decisions', cal['mean_plies'])
    return dict(source=source, iterations=fixed, decision_seconds=None if fixed else seconds,
                estimated_decision_seconds=seconds, estimated_player_decisions=decisions,
                estimated_search_game_seconds=seconds * decisions if seconds is not None else None,
                target_match_time_active=source == '--target-match-time')


def announce_plan(state, family, family_profile, phase_names, base, budget, progress):
    r, c = state['request'], state['calibration']
    out = progress
    def field(label, value):
        out(f'  {label + ":":<{max(29, len(label) + 2)}}{value}')
    b = compute_budget(state)
    out('Study\nTarget')
    field('Game', r['game'])
    field('Agent family', 'SO-ISMCTS' if family == 'so_ismcts' else 'MCTS')
    mode = ('local retune: ' + r['tune'] if r.get('tune') else
            'incremental baseline optimization' if r['baseline_supplied'] else r['mode'].replace('_', ' '))
    field('Mode', mode)
    field('Total study budget', f'{budget:g}s' if budget is not None else 'unlimited')
    out('Capabilities')
    field('Available selection policies', ', '.join(r['selection_policies']))
    out('Supported tuning')
    out('  ' + ', '.join(r['supported_tuners']))
    unsupported = sorted({'tree_reuse', 'transpositions', 'rave', 'progressive_widening'} - set(family_profile.mechanisms))
    if unsupported:
        field('Unsupported by family', ', '.join(unsupported))
    out('Search horizon')
    horizon = c['horizon']
    depth = c.get('operating_point', {}).get('depth', horizon['depth'])
    field('Rollout depth', f'{depth} plies' if depth is not None else 'no configurable rollout horizon')
    cutoff = r['mode'] == 'heuristic_cutoff' and horizon['kind'] != 'baseline'
    field('Mode', 'heuristic cutoff' if cutoff else horizon['kind'].replace('_', ' '))
    if depth != horizon['depth']:
        field('Reference full horizon', f"{horizon['depth']} plies")
        field('Reference horizon mode', horizon['kind'].replace('_', ' '))
    if 'minimum_position_terminal_fraction' in horizon:
        field('Min probe terminal reach', f"{horizon['minimum_position_terminal_fraction']:.1%} (reference horizon)")
    if 'target_terminal_fraction' in horizon:
        field('Probe terminal criterion', f">= {horizon['target_terminal_fraction']:.0%} at every sampled position")
    out('Compute budget')
    if b['iterations']:
        field('Iterations/decision', f"{b['iterations']:,} (fixed)")
        if b['estimated_decision_seconds'] is not None:
            field('Measured decision time', f"~{b['estimated_decision_seconds']*1000:g} ms (not a time limit)")
    else:
        field('Derived decision time' if b['target_match_time_active'] else 'Decision time',
              f"{b['decision_seconds']*1000:g} ms")
    field('Source', b['source'])
    field('Estimated decisions', f"{b['estimated_player_decisions']:g}")
    if b['target_match_time_active']:
        field('Target match time', f"{r['target_match_time']:g}s")
        field('Safety margin', f"{r['safety_margin']:.2f}")
        field('Derivation', f"{r['target_match_time']:g} / ({b['estimated_player_decisions']:g} × {r['safety_margin']:g}) s/decision")
    else:
        field('Target match time', f"{r['target_match_time']:g}s (overridden; not used)")
    if b['estimated_search_game_seconds'] is not None:
        field('Estimated search/game', f"~{b['estimated_search_game_seconds']:.2f}s")
    out('  Search compute estimate, not a hard wall-clock match limit.')
    out('Search calibration')
    a = c['search_adequacy']
    field('Adequacy', a['category'] + ' (search population, not strength)')
    field('Median iterations', f"{a['median_iterations_per_decision']:,.0f} / decision")
    field('Representative branching', f"{a['representative_branching']:g}")
    for warning in c.get('warnings', []):
        out('  Warning: ' + warning)
    out('Planned stages')
    out('  Calibration')
    specs = r['tuner_specs']
    policy_names = {'uct': 'UCT', 'ucb1_tuned': 'UCB1-Tuned', 'uct_rave': 'UCT-RAVE'}
    def selection_candidates(policies):
        out('    Candidate policies: ' + ', '.join(policy_names.get(p, p) for p in policies))
    if r.get('tune'):
        field('Local retune', r['tune'])
        if r['tune'] == 'selection':
            selection_candidates(r['selection_policies'])
        if r['tune'] == 'widening-expansion':
            field('PW admission strategies', 'random, rave-guided')
        out(f'    Adaptive dimensions: initial round + max {MAX_EXTENSION_ROUNDS} extensions; categorical dimensions: one round.')
    elif (family != "mcts"):
        dimensions = {s['dimension'] for n, s in specs.items() if not n.startswith('second-')}
        for dimension in ('exploration', 'selection', 'tree-reuse'):
            field(dimension.replace('-', ' ').capitalize(), 'enabled' if dimension in dimensions else 'disabled')
            if dimension == 'selection' and dimension in dimensions:
                selection_candidates(r['selection_policies'])
    else:
        for label, phase, flag in [('Depth screen', 'depth_screen', 'depth-search'),
                                   ('Exploration', 'exploration', 'selection-search'),
                                   ('Selection', 'selectors', 'selection-search'),
                                   ('RAVE selector search', 'rave', 'rave-search'),
                                   ('Structure', 'mechanisms', 'mechanism-search'),
                                   ('Progressive widening', 'pw_screen', 'pw-search')]:
            active = phase in phase_names and _phase_enabled(phase, r)
            unavailable = phase in ('rave', 'pw_screen') and 'uct_rave' not in r['selection_policies']
            status = 'enabled' if active else 'unsupported by game/backend' if unavailable else f'disabled (enable with --{flag})'
            if phase == 'depth_screen' and r['mode'] != 'heuristic_cutoff':
                status = 'not applicable to this horizon mode'
            field(label, status)
            if active and phase == 'selectors':
                selection_candidates(p for p in r['selection_policies'] if p != 'uct_rave')
                out('    Compared against the retained incumbent (which may use another policy).')
            if active and phase == 'rave':
                selection_candidates(['uct_rave'])
                out(f'    equivalence: initial range + max {MAX_EXTENSION_ROUNDS} adaptive extensions; exploration; compare')
            if active and phase == 'pw_screen':
                out(f'    screen; tune k (max {MAX_EXTENSION_ROUNDS} extensions); tune alpha (max {MAX_EXTENSION_ROUNDS} extensions); refine; compare')
                field('PW admission strategies', 'random, rave-guided' if 'uct_rave' in r['selection_policies'] else 'random')
        if any(n.startswith('admission-') for n in specs):
            field('Widening expansion', 'conditional on retained candidate having PW enabled')
            field('PW admission strategies', 'random, rave-guided' if 'uct_rave' in r['selection_policies'] else 'unavailable (AMAF unsupported)')
    if r['second_pass']:
        dimensions = dict.fromkeys(s['dimension'] for n, s in specs.items() if n.startswith('second-'))
        field('Second pass', 'conditional tuning of the retained candidate')
        for dimension in dimensions:
            if dimension == 'rave':
                if not (_phase_enabled('rave', r) or base.selection_policy == 'uct_rave'):
                    continue
                condition = 'only if retained selector is UCT-RAVE'
            elif dimension.startswith('progressive-widening'):
                if not (_phase_enabled('pw_screen', r) or getattr(base, 'progressive_widening', False)):
                    continue
                condition = 'only if retained candidate has PW enabled'
            else:
                condition = 'only if retained selector uses exploration'
            out(f'    {dimension} ({condition})')
        out(f'    Adaptive chains: initial round + max {MAX_EXTENSION_ROUNDS} extensions; final joint recheck unchanged.')
    if 'confirmation' in phase_names:
        out('  Confirmation')
    if r['vs_random']:
        out('  Random baseline: final retained champion; diagnostic only; after competitive stages.')
    out('Comparison policy')
    out('  Equal fixed iterations' if b['iterations'] else '  Equal wall-clock decision budgets')
    field('Games/comparison', max(8, r['games_per_comparison']))
    if r['stage_games']:
        field('Stage game overrides', ', '.join(f'{k}={max(8,v)}' for k,v in r['stage_games'].items()))
    field('Workers', f"{r['workers']} ({'shared_cpu' if r['workers'] > 1 else 'isolated'})")
    out('  Paired seeds and swapped seats. Budget pressure removes candidates, not evidence.')


def extension_notice(state, phase, completed=False):
    name = phase['name']
    fields = {'rave': 'rave_equivalence', 'pw_k': 'progressive_widening_k', 'pw_alpha': 'progressive_widening_alpha'}
    if '_extend_' in name:
        stem, suffix = name.rsplit('_extend_', 1)
        field, round_index = fields.get(stem), int(suffix)
        earlier = [p for n, p in state['phases'].items() if n == stem or n.startswith(stem + '_extend_')]
    elif phase.get('round', 0) and phase.get('chain'):
        field, round_index = TUNING_FIELDS[phase['dimension']][0], phase['round']
        earlier = [p for p in state['phases'].values() if p.get('chain') == phase['chain']]
    else:
        return None, False
    if not field or not phase['contrasts']:
        return None, False
    prefix = name.rsplit('_extend_', 1)[0] if '_extend_' in name else phase['dimension']
    if not completed:
        tested = [v[field] for p in earlier if p is not phase for v in p['agents'].values() if field in v]
        expanding = tested and any(phase['agents'][c['b']][field] < min(tested) or phase['agents'][c['b']][field] > max(tested) for c in phase['contrasts'])
        reason = 'Expanding boundary range beyond previously tested values' if expanding else 'Refining untested neighboring values'
        return f'{prefix}: {reason} ({round_index}/{MAX_EXTENSION_ROUNDS}).', False
    elif round_index == MAX_EXTENSION_ROUNDS:
        boundary = False
        for group, leader in _group_leaders(phase).items():
            members = phase['groups'][group]
            values = [phase['agents'][m][field] for m in members]
            challengers = {c['b'] for c in phase['contrasts']}
            boundary |= leader in challengers and phase['agents'][leader][field] in (min(values), max(values))
        reason = 'Boundary candidate won at extension limit' if boundary else 'Adaptive extension limit reached'
        return f'{prefix}: {reason} ({round_index}/{MAX_EXTENSION_ROUNDS}). Keeping best supported candidate; no further expansion in this chain.', True
    return None, False


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
             f'<p>Status: <b>{escape(state["status"])}</b>. Budget used: {state["spent_seconds"]:.1f}/{state["budget_seconds"] if state["budget_seconds"] is not None else "unlimited"}s.</p>',
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
        if phase.get('descriptive'):
            continue
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
    if state["request"].get("agent_family") == "so_ismcts":
        selected = state.get("selected_candidate", {})
        confirmed = complete and tested and result.get("ci95_b", [0])[0] > .5
        for cost in costs:
            cost["median_determinizations_per_decision"] = cost["median_iterations_per_decision"]
            cost["determinizations_per_second"] = cost["iterations_per_second"]
        return {"agent_family": "so_ismcts", "mode": state["request"]["mode"],
                "search_complete": all(phases.get(n, {}).get("status") == "complete"
                    for n in state["request"]["phase_names"] if n != "random_baseline"),
                "final_selection": {"candidate": selected.get("name"), "phase": selected.get("phase"),
                    "status": "confirmed" if confirmed else "provisional",
                    "competitive_confidence": verdict if complete and tested else "not_measured",
                    "confirmation_result": "IMPROVED" if confirmed else "INCONCLUSIVE" if complete else "PENDING",
                    "score_b": result.get("score_b"), "ci95_b": result.get("ci95_b"),
                    "seed_pairs": result.get("seed_pairs", 0),
                    "interpretation": "Fresh-seed confirmation against the original operating incumbent; no sufficiently supported improvement found." if not confirmed else "Improvement confirmed on fresh paired seeds."},
                "improvement_comparisons": effects, "candidate_search_costs": costs}
    if state["request"].get("version", 0) >= 20:
        selected = state.get("selected_candidate", {})
        return {"mode": state["request"]["mode"], "search_complete": all(phases.get(n, {}).get("status") == "complete"
                    for n in state["request"]["phase_names"] if n != "random_baseline"),
                "final_selection": {"candidate": selected.get("name"), "phase": selected.get("phase"),
                                    "status": "provisional", "competitive_confidence": "not_independently_confirmed",
                                    "interpretation": "Retained incumbent after completed stages and any requested local pass; no independent confirmation."},
                "improvement_comparisons": effects, "candidate_search_costs": costs}
    return {"mode": state["request"]["mode"], "search_complete": complete,
            "final_selection": {"candidate": candidate, "status": "confirmed" if complete and tested and verdict == "b_ahead" else "provisional",
                                "competitive_confidence": verdict if complete and tested else "not_measured",
                                "score_b": result.get("score_b"), "ci95_b": result.get("ci95_b"),
                                "seed_pairs": result.get("seed_pairs", 0),
                                "interpretation": "Nominee fixed before confirmation. Inconclusive evidence does not establish equivalence; a loss is reported without post-hoc reselection."},
            "improvement_comparisons": effects, "candidate_search_costs": costs}


def write_family_study_report(output, state):
    summary = {**state, **family_study_diagnostics(state), "agent_family": state["request"].get("agent_family", "mcts")}
    summary["limitations"] = [
        "Only the requested evaluator family is optimized; compare separate studies in a tournament.",
        "Candidate compute budgets are equal within each race: baseline iterations/time or derived decision time. Parallel comparisons share CPU; iterations and elapsed search work are measured.",
        "Practical full-depth means >=99% terminal simulations at every sampled position; this finite empirical probe is not a rules guarantee. Unverified safety horizons are explicitly labelled.",
        "Rollout depth counts player decisions, excludes Chance, and completes the physical turn before cutoff. Terminal stops immediately.",
        "Search adequacy thresholds are heuristic diagnostics, separate from paired-seed competitive confidence.",
        "95% intervals use conservative paired-seed Hoeffding bounds. Screening is adaptive/exploratory, not independent confirmation; no multiple-comparison correction.",
        "The exported winner has exploratory evidence only; an optional second local pass is not independent confirmation.",
        "Pilot length and random representative positions are preliminary estimates. Actual game costs can differ; target match time is not a match deadline.",
        "Fixed evidence per comparison. Budget screening can omit challengers explicitly; elapsed limits may overshoot by an in-flight match batch. Old study protocols require a new output directory.",
    ]
    if summary["agent_family"] == "so_ismcts":
        summary["limitations"] = [
            "Equal decision-time budgets are operating resources; iteration/determinization counts are diagnostics, not optimized hyperparameters. Fixed-iteration profiles retain their explicit budget.",
            "Pilot decision counts and throughput are estimates; Lost Cities has no hard horizon. A search finishes at least one iteration and may overrun by one simulation.",
            "One fresh complete determinization per iteration. Search adequacy measures action sampling, not percentage of possible hidden worlds.",
            "Coarse/refinement races are exploratory. Final confirmation uses disjoint paired seeds and a conservative 95% bound; inconclusive confirmation retains the original incumbent.",
            "Single-observer baseline with uniform rollout and MostVisited; no advanced search techniques or opponent inference. Parallel matches share CPU.",
        ]
    if state.get("mixed_engines"):
        summary["limitations"].insert(0, "Mixed-engine study: an explicit resume accepted changed code/build. Games from different engines may occur within the same comparison or seed pair; throughput and strength may differ. Engine fingerprints and exact pre-change match IDs are recorded in engine_changes.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    cal = state.get("calibration") or {}
    parts = ['<!doctype html><meta charset="utf-8"><title>Search-agent study</title>',
             '<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:.5rem;text-align:left}</style>',
             f'<h1>{escape(state["request"]["game"])} — {escape(summary["agent_family"])} — {escape(state["request"]["mode"])}</h1>',
             f'<p>Status: {escape(state["status"])}. Used {state["spent_seconds"]:.1f}/{state["budget_seconds"] if state["budget_seconds"] is not None else "unlimited"}s.</p>',
             '<p><a href="summary.json">Summary and frozen comparisons</a> · <a href="study.json">Checkpoint</a></p>',
             '<h2>Calibration and search adequacy</h2>',
             _table(['Measurement', 'Value'], [(k, json.dumps(cal.get(k))) for k in
                    ('target_match_time', 'estimated_game_decisions', 'safety_margin', 'decision_seconds', 'decision_time_source', 'fixed_iterations', 'cutoff_depths')])]
    if summary["agent_family"] == "so_ismcts":
        parts.append('<h2>Operating search budget and determinization throughput</h2><pre>' + escape(json.dumps({
            k: cal.get(k) for k in ('estimated_game_decisions', 'decision_seconds', 'fixed_iterations',
            'seconds_per_iteration', 'iterations_per_second', 'determinizations_per_second',
            'median_iterations_per_decision', 'median_determinizations_per_decision', 'root_action_coverage')}, indent=2)) + '</pre>')
    parts.append(_table(['Sampled player decisions', 'Search ms', 'Iterations', 'Legal actions', 'Terminal / cutoff simulations'],
                        [(t['sampled_ply'], t['milliseconds'], t['iterations'], t['legal_actions'],
                          f"{t.get('terminal_simulations')} / {t.get('cutoff_simulations')}") for t in cal.get('position_timings', [])]))
    if state.get("mixed_engines"):
        parts.append('<p><strong>' + escape(summary["limitations"][0]) + '</strong></p>')
    if state.get('budget_limited'):
        parts.append('<p><strong>Budget limited: some challengers were not tested. See discarded comparisons below.</strong></p>')
    parts.append('<h2>Fixed game budgets</h2><pre>' + escape(json.dumps({k: state['request'].get(k) for k in ('games_per_comparison', 'stage_games')}, indent=2)) + '</pre>')
    if summary["agent_family"] == "so_ismcts":
        parts.append('<h2>Study stages</h2>' + _table(['Stage', 'Status'], [(name, state['phases'].get(name, {}).get('status', 'pending')) for name in state['request']['phase_names']]))
    else:
        parts.append('<h2>Incremental stages</h2>' + _table(['Stage', 'Requested'], [(key, state['request'].get(key, False)) for key in ('selection_search', 'rave_search', 'mechanism_search', 'pw_search', 'depth_search')]))
        parts.append(f'<p>RAVE search requested: {state["request"].get("rave_search", False)}; supported: {"uct_rave" in state["request"].get("selection_policies", [])}.</p>')
        parts.append(f'<p>PW search requested: {state["request"].get("pw_search", False)}; supported: {state["request"].get("pw_supported", False)}.</p>')
    if state.get('pw_budget'):
        parts.append('<h2>Shared PW budget</h2><pre>' + escape(json.dumps(state['pw_budget'], indent=2)) + '</pre>')
    if state.get('rave_budget'):
        parts.append('<h2>Shared RAVE budget</h2><pre>' + escape(json.dumps(state['rave_budget'], indent=2)) + '</pre>')
    horizon = cal.get("horizon", {})
    parts.append(_table(['Horizon kind', 'Decision depth', 'Measured terminal fraction'], [(horizon.get('kind'), horizon.get('depth'), horizon.get('terminal_fraction'))]))
    parts.append('<pre>' + escape(json.dumps(cal.get('search_adequacy', {}), indent=2)) + '</pre>')
    parts.extend('<p>Warning: ' + escape(w) + '</p>' for w in cal.get('warnings', []))
    if state.get('local_retune'):
        parts.append('<h2>LOCAL RETUNE: frozen fields and changes</h2><pre>' + escape(json.dumps(state['local_retune'], indent=2)) + '</pre>')
    if state['request'].get('vs_random'):
        parts.append('<h2>RANDOM BASELINE</h2><p>Diagnostic only. Not used for candidate selection.</p><pre>' + escape(json.dumps(state.get('random_baseline', {}), indent=2)) + '</pre>')
    parts.append('<h2>Selection evidence</h2><pre>' + escape(json.dumps(summary['final_selection'], indent=2)) + '</pre>')
    for name, phase in state['phases'].items():
        if phase.get('discarded_comparisons') or phase.get('skip_reason'):
            parts.append('<pre>' + escape(json.dumps({'phase': name, 'skipped': phase.get('skip_reason'), 'discarded': phase.get('discarded_comparisons', [])}, indent=2)) + '</pre>')
        parts.append(f'<h2>{escape(name)} — {escape(phase["status"])}</h2>')
        if 'planned_games' in phase:
            parts.append(f'<p>Fixed planned games: {phase["planned_games"]}; estimated duration: {phase["estimated_seconds"]:.1f}s (not a limit).</p>')
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
