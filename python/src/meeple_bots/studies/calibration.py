"""MCTS and SO-ISMCTS Study calibration using explicit execution callbacks."""
from dataclasses import asdict, replace
from statistics import mean, median
from time import perf_counter
from .._agent_config import MctsAgent, NeutralEvaluator
from .._search_budget import decision_budget
from .budget import _study_budget
from .persistence import profile_values
from .planning import cutoff_depths
from ..search_metrics import search_adequacy

def _timed(agent: MctsAgent, seconds: float) -> MctsAgent:
    return replace(agent, iterations=None, time_budget=seconds, root_diagnostics=False)


def _iterations(agent: MctsAgent, count: int) -> MctsAgent:
    return replace(agent, iterations=max(1, min(2**32 - 1, count)), time_budget=None, root_diagnostics=False)


def calibrate_mcts(state, game, base, reference, spent, pair, save, progress, benchmark_mcts_agent):
    if state["calibration"]:
        return
    request = state["request"]
    progress("Calibrating search cost and horizon...")
    work = state.setdefault("calibration_progress", {})
    pilot_agent = replace(_iterations(base, 32), rollout_depth=64)
    pilot = {"name": "calibration", "agents": {"pilot": profile_values(pilot_agent), "random": None},
             "contrasts": [{"a": "pilot", "b": "random", "factor": "sanity"}]}
    if "mean_plies" not in work:
        rows = pair(pilot, 0, 0, pilot=True)
        # Simulation's plies/moves count only player decisions; chance_events is separate.
        work["mean_plies"] = mean(r["result"]["plies"] for r in rows)
        work["pilot"] = pilot
        save()
    length = max(1, work["mean_plies"])
    target = request["decision_seconds"] or (base.time_budget if request["baseline_supplied"] else None) or decision_budget(request["target_match_time"], length, request["safety_margin"])
    fixed_iterations = base.iterations if request["baseline_supplied"] and request["decision_seconds"] is None else None
    if request["baseline_supplied"] and request.get("tune") != "cutoff-depth":
        work["horizon"] = {"depth": base.rollout_depth, "kind": "baseline", "samples": []}
    samples = work.setdefault("horizon_samples", [])
    depth = samples[-1]["depth"] if samples else min(64, request["max_plies"])
    neutral = replace(base, heuristic=None, cutoff_evaluator=NeutralEvaluator())
    while "horizon" not in work:
        if not samples or samples[-1]["depth"] != depth:
            # Two independent search/state seeds, same depths for stable comparisons.
            benches = [benchmark_mcts_agent(game, _iterations(replace(neutral, rollout_depth=depth), 64),
                                            round(length), request["seed"] + offset) for offset in (17, 31)]
            timings = [asdict(t) for b in benches for t in b.position_timings]
            hard = benches[0].maximum_decision_horizon
            terminal = sum(t["terminal_simulations"] or 0 for t in timings)
            cutoff = sum(t["cutoff_simulations"] or 0 for t in timings)
            fractions = [t["terminal_simulations"] / max(1, (t["terminal_simulations"] or 0) + (t["cutoff_simulations"] or 0))
                         for t in timings if t["terminal_simulations"] is not None]
            samples.append({"depth": depth, "hard_maximum": hard, "terminal_fraction": terminal / max(1, terminal+cutoff),
                            "minimum_position_terminal_fraction": min(fractions, default=0), "position_timings": timings})
            save()
        sample = samples[-1]
        hard = sample["hard_maximum"]
        if hard is not None and depth != hard:
            depth = hard
            continue
        reached = sample["minimum_position_terminal_fraction"] >= .99
        if hard is not None or reached or len(samples) >= 6 or depth >= request["max_plies"]:
            kind = "hard_maximum" if hard is not None else "practical_full_depth" if reached else "practical_unverified"
            work["horizon"] = {"depth": depth, "kind": kind, "terminal_fraction": sample["terminal_fraction"],
                               "minimum_position_terminal_fraction": sample["minimum_position_terminal_fraction"],
                               "target_terminal_fraction": .99, "samples": samples}
            break
        depth = min(depth * 2, request["max_plies"])
    horizon = work["horizon"]
    depths = cutoff_depths(horizon["depth"]) if request["mode"] == "heuristic_cutoff" else []
    if request["mode"] == "heuristic_cutoff" and not depths and not request["baseline_supplied"]:
        raise ValueError("the reference horizon leaves no distinct heuristic cutoff depth")
    if reference and not request["baseline_supplied"]:
        if (request["mode"] == "full_depth" and reference.rollout_depth != horizon["depth"]) or (
                request["mode"] == "heuristic_cutoff" and reference.rollout_depth >= horizon["depth"]):
            raise ValueError("reference horizon belongs to another family; compare it in a separate tournament")
    operating_depth = base.rollout_depth if request["baseline_supplied"] else depths[len(depths)//2] if depths else horizon["depth"]
    if "position_timings" not in work:
        probe = replace(base if fixed_iterations else _timed(base, target), rollout_depth=operating_depth, root_diagnostics=True)
        bench = benchmark_mcts_agent(game, probe, round(length), request["seed"] + 47)
        work["position_timings"] = [asdict(t) for t in bench.position_timings]
        save()
    adequacy = search_adequacy(work["position_timings"], request["target_match_time"])
    if request["decision_seconds"] is not None or request["baseline_supplied"]:
        # Changing the match target cannot affect an explicit per-decision override.
        adequacy["suggested_targets"] = []
    if fixed_iterations:
        target = mean(t["milliseconds"] for t in work["position_timings"]) / 1000
    center = max(1, round(adequacy["median_iterations_per_decision"]))
    cal = {"mean_plies": length, "estimated_game_decisions": length, "target_match_time": request["target_match_time"],
           "safety_margin": request["safety_margin"], "safety_adjusted_decisions": length * request["safety_margin"],
           "decision_seconds": target, "fixed_iterations": fixed_iterations, "decision_time_source": "explicit_override" if request["decision_seconds"] else "baseline" if request["baseline_supplied"] else "target_match_time",
           "estimated_match_seconds": length * target, "horizon": horizon, "cutoff_depths": depths,
           "center_iterations": center, "operating_point": {"iterations": center, "depth": operating_depth,
               "interpretation": "Measured work at the requested time; diagnostic only, never a winner-selection budget."},
           "seconds_per_iteration": 1 / max(1e-9, adequacy["iterations_per_second"]),
           "position_timings": work["position_timings"], "search_adequacy": adequacy,
           "pilot": work["pilot"], "seconds": spent()}
    warnings = []
    if horizon["kind"] == "practical_unverified":
        warnings.append("Full-depth is impractical or unverified within calibration limits; this is an explicit neutral safety cutoff, not proven terminal search.")
    if adequacy["category"] in ("LOW", "VERY LOW"):
        warnings.append("Requested target provides little search per decision; results describe under-searched agents. The requested target is unchanged.")
    cal["warnings"] = warnings
    state["calibration"] = cal
    save()


def calibrate_so(state, game, base, budget, spent, batch, save, progress):
    if state['calibration']:
        return
    request = state['request']
    progress('Calibrating search cost with structural Random-vs-Random pilot...')
    # The normal trace writer/executor persists the paired structural pilot too.
    pilot = state.setdefault('calibration_progress', {}).setdefault('pilot', {
        'name': 'calibration', 'agents': {'random-a': None, 'random-b': None},
        'contrasts': [{'a': 'random-a', 'b': 'random-b'}], 'planned_pairs': 1})
    rows = batch(pilot, [0], 0, pilot=True)[0]
    expected = mean(row['result']['plies'] for row in rows)
    fixed = base.iterations if request['baseline_supplied'] and request['decision_seconds'] is None else None
    seconds = request['decision_seconds'] or (base.time_budget if request['baseline_supplied'] else None)
    seconds = seconds or decision_budget(request['target_match_time'], expected, request['safety_margin'])
    cal = {'mean_plies': expected, 'estimated_game_decisions': expected,
           'target_match_time': request['target_match_time'], 'safety_margin': request['safety_margin'],
           'decision_seconds': seconds, 'fixed_iterations': fixed,
           'decision_time_source': 'explicit_override' if request['decision_seconds'] is not None else 'baseline' if request['baseline_supplied'] else 'target_match_time',
           'horizon': {'kind': 'unbounded', 'depth': None}, 'pilot': pilot, 'position_timings': []}
    agent = _study_budget(base, cal)
    # Sample only legitimate observations. Environment sampling and policy/search seeds
    # have separate streams; positions span both microphases and early/middle play.
    from random import Random
    environment, policy = Random(request['seed'] ^ 0x8EBC6AF0), Random(request['seed'] ^ 0xA0761D64)
    game = game
    position = game.initial_state(request['seed'])
    targets = {0, max(1, round(expected/4)), max(2, round(expected/2))}
    decision = 0
    while position.status != 'terminal' and decision <= max(targets):
        if position.status == 'chance':
            position = game.apply_chance_outcome(position, game.sample_chance(position, environment.getrandbits(64)))
            continue
        legal = game.legal_actions(position)
        if decision in targets:
            if budget is not None and spent() >= budget:
                return  # Resume can repeat unfinished throughput measurements; pilot is saved.
            observation = game.observation(position, position.current_player)
            start = perf_counter()
            result = agent.search(observation, legal, seed=request['seed'] + decision)
            elapsed = perf_counter() - start
            d = result['diagnostics']; root = result['nodes'][0]['edges']
            cal['position_timings'].append({'sampled_ply': decision, 'milliseconds': elapsed*1000,
                'iterations': d['completed_iterations'], 'determinizations': d['determinizations_sampled'],
                'legal_actions': len(legal), 'root_action_coverage': sum(e['visits'] > 0 for e in root)/len(legal),
                'nodes': d['tree_nodes'], 'action_edges': d['action_edges'],
                'terminal_simulations': d['terminal_simulations'], 'cutoff_simulations': d['cutoff_simulations']})
        position = game.apply_action(position, policy.choice(legal))
        decision += 1
    timings = cal['position_timings']
    elapsed = sum(t['milliseconds'] for t in timings)/1000
    iterations = sum(t['iterations'] for t in timings)
    cal.update(seconds_per_iteration=elapsed/iterations, iterations_per_second=iterations/elapsed,
               determinizations_per_second=sum(t['determinizations'] for t in timings)/elapsed,
               median_iterations_per_decision=median(t['iterations'] for t in timings),
               median_determinizations_per_decision=median(t['determinizations'] for t in timings),
               root_action_coverage=median(t['root_action_coverage'] for t in timings),
               search_adequacy=search_adequacy(timings, request['target_match_time']))
    cal['warnings'] = ['Random-vs-random pilot length and sampled throughput are preliminary operating estimates, not a hard game horizon.']
    state['calibration'] = cal
    state['operating_baseline'] = profile_values(agent)
    state['selected_candidate'] = {'phase': None, 'name': 'incumbent', 'profile': profile_values(agent)}
    save()
