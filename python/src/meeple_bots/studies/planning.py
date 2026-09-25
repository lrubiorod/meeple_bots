"""Deterministic Study stage definitions and candidate composition."""
from dataclasses import replace
from .._agent_config import MctsAgent
from .persistence import profile_values, agent_from_values
from .race import _group_leaders, _promising, _evidence_complete, _reverse_result
from .tuners import TUNING_FIELDS, proposals, assert_frozen, validate_tuner
from .budget import _study_budget

MAX_EXTENSION_ROUNDS = 3


PHASES = ("depth_screen", "exploration", "selectors", "rave",
          *(f"rave_extend_{i}" for i in range(1, MAX_EXTENSION_ROUNDS + 1)),
          "rave_exploration", "rave_compare", "mechanisms", "pw_screen", "pw_k",
          *(f"pw_k_extend_{i}" for i in range(1, MAX_EXTENSION_ROUNDS + 1)), "pw_alpha",
          *(f"pw_alpha_extend_{i}" for i in range(1, MAX_EXTENSION_ROUNDS + 1)), "pw_refine", "pw_compare")


STAGES = ("depth", "selection", "rave", "mechanisms", "pw")


SEED_STRIDE = 100_000


def stage_for_phase(name):
    if name == "depth_screen": return "depth"
    if name in ("exploration", "selectors"): return "selection"
    if name.startswith("rave"): return "rave"
    if name.startswith("pw_"): return "pw"
    return name


def budget_stage(phase):
    return {"exploration": "selection", "selection": "selection", "rave": "rave", "structure": "mechanisms", "tree-reuse": "mechanisms", "cutoff-depth": "depth"}.get(phase.get("tuner"), "pw" if phase.get("tuner") else stage_for_phase(phase["name"]))


def _phase_enabled(name: str, request: dict) -> bool:
    if name == "depth_screen":
        return request.get("depth_search", False) and request["mode"] == "heuristic_cutoff"
    if name in ("exploration", "selectors"):
        return request.get("selection_search", False)
    if name.startswith("rave"):
        return request.get("rave_search", False) and "uct_rave" in request["selection_policies"]
    if name == "mechanisms":
        return request.get("mechanism_search", False)
    if name.startswith("pw_"):
        return request.get("pw_search", False) and request.get("pw_supported", False)
    return False


def cutoff_depths(horizon: int) -> list[int]:
    """Small shallow/medium/deep grid, strictly below the reference horizon."""
    return sorted({max(1, min(horizon - 1, round(horizon * f)))
                   for f in (.1, .25, .5, .75)}) if horizon > 1 else []


def _pw_survivors(screen):
    """A loss by one admission policy says nothing about its sibling."""
    survivors, decisions = {}, {}
    for policy in screen["pw_policies"]:
        evidence = [c for c in screen["contrasts"] if c["policy"] == policy and _evidence_complete(c)]
        if not evidence:
            decisions[policy] = "insufficient_budget_or_evidence"
            continue
        best = max(evidence, key=lambda c: c["result"]["score_b"])
        result = best["result"]
        if _promising(_reverse_result(result)):
            decisions[policy] = "screened_out"
        else:
            survivors[policy] = agent_from_values(screen["agents"][best["b"]])
            decisions[policy] = "survived"
    decisions["family"] = ("rejected" if all(v == "screened_out" for v in decisions.values())
                           else "inconclusive" if "insufficient_budget_or_evidence" in decisions.values()
                           else "survivors")
    return survivors, decisions


def _build_phase(name: str, state: dict, base: MctsAgent, reference: MctsAgent | None) -> dict:
    if name in state["request"].get("tuner_specs", {}):
        return _build_tuning_phase(name, state, base)
    """Frozen incremental comparisons within one evaluator family and search budget."""
    cal, phases, request = state["calibration"], state["phases"], state["request"]
    horizon = cal["horizon"]["depth"]
    start = _study_budget(base, cal)
    if not request.get("baseline_supplied", False):
        depth = cal["cutoff_depths"][len(cal["cutoff_depths"])//2] if request["mode"] == "heuristic_cutoff" else horizon
        start = replace(start, rollout_depth=depth)
    agents, contrasts, groups, priority = {}, [], {}, {}
    decisions = {}

    def add(label, agent, group="main"):
        agents[label] = profile_values(_study_budget(agent, cal))
        groups.setdefault(group, []).append(label)
        priority[label] = [len(agents)]
        return label

    def compare(a, b, factor, **extra):
        if agents[a] != agents[b]:
            contrasts.append({"a": a, "b": b, "factor": factor, **extra})

    def carry(previous):
        result = {}
        for group, leader in _group_leaders(phases[previous]).items():
            result[group] = agent_from_values(phases[previous]["agents"][leader])
        return result

    def finish():
        return {"name": name, "agents": agents, "groups": groups, "contrasts": contrasts,
                "incremental": True, "evidence_policy": "paired_55_one_se", "tie_priority": priority,
                "rave_decisions": decisions if name.startswith("rave") else {},
                "pw_decisions": decisions if name.startswith("pw_") else {}, "status": "pending"}

    if name != "depth_screen" and not _phase_enabled(name, request):
        previous = PHASES[PHASES.index(name)-1]
        for group, parent in carry(previous).items():
            add("incumbent", parent, group)
        decisions["main"] = "disabled" if not request.get("rave_search" if name.startswith("rave") else "pw_search", False) else "unavailable"
        return finish()

    if name == "depth_screen":
        control = add("initial", start)
        if _phase_enabled(name, request):
            for candidate in proposals("cutoff-depth", start, horizon=horizon, coarse=True):
                compare(control, add(f"d{candidate.rollout_depth}", candidate), "rollout_depth")
    elif name == "exploration":
        parent = next(iter(carry("depth_screen").values()))
        control = add("incumbent", parent)
        uct = replace(parent, selection_policy="uct")
        for candidate in [uct, *proposals("exploration", uct, coarse=True)]:
            compare(control, add(f"uct-c{candidate.exploration:g}", candidate), "exploration_and_selector")
    elif name == "selectors":
        parent = next(iter(carry("exploration").values()))
        control = add("incumbent", parent)
        for candidate in proposals("selection", parent, selectors=[s for s in request["selection_policies"] if s != "uct_rave"]):
            compare(control, add(candidate.selection_policy, candidate), "selection_policy")
    elif name == "rave":
        for group, parent in carry("selectors").items():
            if request.get("rave_search", False) and "uct_rave" in request["selection_policies"]:
                initial_k = parent.rave_equivalence if parent.selection_policy == "uct_rave" else 3000
                control = add(f"{group}-rave-k{initial_k}", replace(parent, selection_policy="uct_rave", rave_equivalence=initial_k), group)
                for candidate in proposals("rave", agent_from_values(agents[control])):
                    compare(control, add(f"{group}-rave-k{candidate.rave_equivalence}", candidate, group), "rave_equivalence")
            else:
                add(f"{group}-selector", carry("selectors")[group], group)
                decisions[group] = "unavailable" if request.get("rave_search") else "disabled"
    elif name.startswith("rave_extend_"):
        previous = PHASES[PHASES.index(name)-1]
        prior = phases[previous]
        for group, parent in carry(previous).items():
            control = add(f"{group}-incumbent", parent, group)
            evidence = [c for c in prior["contrasts"] if c["b"] in prior["groups"][group]]
            if not evidence:
                decisions[group] = prior.get("rave_decisions", {}).get(group, "unavailable")
                continue
            complete = all(c.get("result", {}).get("seed_pairs", 0) >= max(4, c.get("target_pairs", 4)) for c in evidence)
            if not complete:
                decisions[group] = "insufficient_budget_or_evidence"
                continue
            # The first refinement always checks gaps. Later rounds require an
            # improving challenger, independently of the previous non-RAVE selector.
            improved = any(prior["agents"][c["b"]]["rave_equivalence"] == parent.rave_equivalence
                           and _promising(c["result"]) for c in evidence)
            if previous != "rave" and not improved:
                decisions[group] = "no_clear_improvement_not_proven_plateau"
                continue
            tested = {v["rave_equivalence"] for stage in PHASES[PHASES.index("rave"):PHASES.index(name)]
                      for member in phases[stage]["groups"][group]
                      if (v := phases[stage]["agents"][member])["selection_policy"] == "uct_rave"}
            k = parent.rave_equivalence
            for candidate in proposals("rave", parent, tested=tested):
                if candidate.rave_equivalence not in tested:
                    compare(control, add(f"{group}-rave-k{candidate.rave_equivalence}", candidate, group), "rave_equivalence")
            decisions[group] = "refining_geometric_neighbors" if any(c["a"] == control for c in contrasts) else "no_untested_neighbors"
    elif name == "rave_exploration":
        prior = phases[f"rave_extend_{MAX_EXTENSION_ROUNDS}"]
        for group, parent in carry(f"rave_extend_{MAX_EXTENSION_ROUNDS}").items():
            control = add(f"{group}-rave", parent, group)
            evidence = [c for c in prior["contrasts"] if c["b"] in prior["groups"][group]]
            ready = all(c.get("result", {}).get("seed_pairs", 0) >= max(4, c.get("target_pairs", 4)) for c in evidence)
            reason = prior.get("rave_decisions", {}).get(group, "unavailable")
            if not ready or reason in ("insufficient_budget_or_evidence", "unavailable", "disabled"):
                decisions[group] = "insufficient_budget_or_evidence" if not ready else reason
                continue
            decisions[group] = "extension_limit" if evidence else reason
            for candidate in proposals("exploration", parent):
                compare(control, add(f"{group}-rave-c{candidate.exploration:g}", candidate, group), "exploration")
    elif name == "rave_compare":
        prior = phases["rave_exploration"]
        for group, parent in carry("selectors").items():
            control = add(f"{group}-selector", parent, group)
            evidence = [c for c in prior["contrasts"] if c["b"] in prior["groups"][group]]
            ready = bool(evidence) and all(c.get("result", {}).get("seed_pairs", 0) >= max(4, c.get("target_pairs", 4)) for c in evidence)
            if ready:
                candidate = add(f"{group}-calibrated-rave", carry("rave_exploration")[group], group)
                compare(control, candidate, "calibrated_rave_vs_selector")
                decisions[group] = "bounded_calibration_complete"
            else:
                decisions[group] = "rave_unavailable_or_calibration_incomplete"
    elif name == "mechanisms":
        parent = next(iter(carry("rave_compare").values()))
        control = add("incumbent", parent)
        for candidate in proposals("structure", parent):
            compare(control, add(f"r{int(candidate.tree_reuse)}-t{int(candidate.transpositions)}", candidate), "mechanisms")
    elif name.startswith("pw_"):
        return _build_pw_phase(name, state)
    else:
        raise ValueError(f"unknown study phase: {name}")
    return finish()


def _build_pw_phase(name, state):
    phases, request = state["phases"], state["request"]
    original_phase = phases["mechanisms"]
    original = agent_from_values(original_phase["agents"][next(iter(_group_leaders(original_phase).values()))])
    phase = {"name": name, "agents": {}, "groups": {}, "contrasts": [], "tie_priority": {},
             "incremental": True, "evidence_policy": "paired_55_one_se", "pw_decisions": {}, "status": "pending"}

    def add(label, agent, group="main"):
        phase["agents"][label] = profile_values(agent)
        phase["groups"].setdefault(group, []).append(label)
        phase["tie_priority"][label] = [len(phase["agents"])]
        return label

    def compare(control, candidate, policy):
        if phase["agents"][control] != phase["agents"][candidate]:
            phase["contrasts"].append({"a": control, "b": candidate, "factor": name, "policy": policy})

    if name == "pw_screen":
        # PW decides HOW MANY actions enter; admission decides WHICH enter.
        # The deterministic backend collects AMAF for rave admission even under UCT.
        policies = ["random", "rave"] if "uct_rave" in request["selection_policies"] else ["random"]
        phase["pw_policies"] = policies
        phase["accept"] = True
        if "rave" not in policies:
            phase["pw_decisions"]["rave"] = "not_applicable_amaf_unavailable"
        control = add("incumbent", replace(original, progressive_widening=False, progressive_widening_expansion="random"))
        ks = [original.progressive_widening_k, original.progressive_widening_k/3, original.progressive_widening_k*8/3]
        for k in ks:
            for policy in policies:
                candidate = replace(original, progressive_widening=True, progressive_widening_expansion=policy, progressive_widening_k=k)
                label = add(f"{policy}-k{k:g}", candidate)
                compare(control, label, policy)
                if candidate == original:
                    phase["fallback"] = label
        # Retain at least one representative of EVERY applicable policy together.
        phase["minimum_policy_comparisons"] = len(policies)
        return phase

    screen = phases["pw_screen"]
    survivors, decisions = _pw_survivors(screen)
    phase["pw_decisions"] = {**screen["pw_decisions"], **decisions}
    previous = phases[PHASES[PHASES.index(name)-1]]
    if name == "pw_compare":
        # Preserve a supported screen winner even when refinement runs out of budget.
        leader = next(iter(_group_leaders(screen).values()))
        control = add("incumbent", agent_from_values(screen["agents"][leader]))
        for policy, parent in survivors.items():
            evidence = [c for c in previous["contrasts"] if c.get("policy") == policy]
            if (policy in previous["groups"] and evidence and all(_evidence_complete(c) for c in evidence)
                    and previous["pw_decisions"].get(policy) != "calibration_incomplete"):
                parent = agent_from_values(previous["agents"][_group_leaders(previous)[policy]])
            compare(control, add(f"calibrated-pw-{policy}", parent), policy)
        return phase

    axis = "alpha" if name.startswith("pw_alpha") else "k"
    dimension = "progressive-widening" if name == "pw_refine" else f"progressive-widening-{axis}"
    key = f"progressive_widening_{axis}"
    for policy, parent in survivors.items():
        if name != "pw_k" and policy in previous["groups"]:
            parent = agent_from_values(previous["agents"][_group_leaders(previous)[policy]])
        control = add(f"{policy}-incumbent", parent, policy)
        evidence = [c for c in previous["contrasts"] if c.get("policy") == policy]
        reason = previous.get("pw_decisions", {}).get(policy)
        if name != "pw_k" and (any(not _evidence_complete(c) for c in evidence)
                               or reason == "calibration_incomplete"
                               or (not evidence and previous.get("discarded_comparisons") and any(c.get("policy") == policy for c in previous["discarded_comparisons"]))):
            phase["pw_decisions"][policy] = "calibration_incomplete"
            continue
        if "_extend_" in name and not name.endswith("_extend_1"):
            improving = any(previous["agents"][c["b"]][key] == getattr(parent, key) and _promising(c["result"]) for c in evidence)
            if not improving:
                phase["pw_decisions"][policy] = "no_clear_improvement_not_proven_plateau"
                continue
        stages = PHASES[PHASES.index("pw_screen" if axis == "k" else "pw_alpha"):PHASES.index(name)]
        tested = {v[key] for stage in stages for c in phases[stage]["contrasts"]
                  if c.get("policy") == policy and _evidence_complete(c)
                  for role in ("a", "b") if (v := phases[stage]["agents"][c[role]]).get("progressive_widening")
                  and v["progressive_widening_expansion"] == policy}
        for i, candidate in enumerate(proposals(dimension, parent, tested=tested)):
            compare(control, add(f"{policy}-{i}", candidate, policy), policy)
        phase["pw_decisions"][policy] = "refining_neighbors"
    if not phase["groups"]:
        add("incumbent", original)
    return phase


def tuning_specs(dimension, prefix="local"):
    axes = ["progressive-widening-k", "progressive-widening-alpha", "progressive-widening-k"] if dimension == "progressive-widening" else [dimension]
    specs = {}
    for step, axis in enumerate(axes):
        rounds = 1 if axis in ("selection", "structure", "tree-reuse", "widening-expansion") or step == 2 else 1 + MAX_EXTENSION_ROUNDS
        chain = f"{prefix}-{step}-{axis}"
        for round_index in range(rounds):
            specs[f"{chain}-{round_index}"] = {"dimension": axis, "round": round_index, "chain": chain}
    return specs


def _build_tuning_phase(name, state, base):
    request = state["request"]
    spec = request["tuner_specs"][name]
    dimension = spec["dimension"]
    selected = state.get("selected_candidate")
    parent = agent_from_values(selected["profile"]) if selected else base
    phases = [p for p in state["phases"].values() if p.get("chain") == spec["chain"]]
    phase = {"name": name, **spec, "tuner": dimension, "agents": {"incumbent": profile_values(parent)},
             "groups": {"main": ["incumbent"]}, "contrasts": [], "incremental": True,
             "evidence_policy": "paired_55_one_se", "accept": True,
             "tie_priority": {"incumbent": [0]}, "status": "pending"}
    try:
        validate_tuner(dimension, parent, request["selection_policies"])
    except ValueError as error:
        phase["skip_reason"] = str(error)
        return phase
    if phases and (not phases[-1]["contrasts"] or (spec["round"] > 1 and next(iter(_group_leaders(phases[-1]).values())) == "incumbent")):
        phase["skip_reason"] = "no_supported_improvement; not proof of a plateau"
        return phase
    field = TUNING_FIELDS[dimension][0]
    tested = {getattr(agent_from_values(v), field) for p in phases for v in p["agents"].values()}
    candidates = proposals(dimension, parent, selectors=request["selection_policies"],
                           horizon=state["calibration"]["horizon"]["depth"], tested=tested, coarse=spec.get("coarse", False))
    for i, candidate in enumerate(candidates):
        assert_frozen(parent, candidate, dimension)
        if request.get("tune"):
            assert_frozen(base, candidate, request["tune"])
        label = f"{dimension}-{i}"
        phase["agents"][label] = profile_values(candidate)
        phase["groups"]["main"].append(label)
        phase["tie_priority"][label] = [i+1]
        phase["contrasts"].append({"a": "incumbent", "b": label, "factor": dimension})
    return phase



def mcts_plan(profile, *, tune=None, admission=False, second_pass=False):
    specs = tuning_specs(tune) if tune else {}
    if not tune and admission:
        specs.update(tuning_specs('widening-expansion', prefix='admission'))
    if second_pass:
        for dimension in ('exploration', 'rave', 'progressive-widening'):
            specs.update(tuning_specs(dimension, prefix='second'))
    return specs, tuple(specs) if tune else (*PHASES, *specs)


def so_specs(profile, base=None, *, tune=None, selection=False, mechanisms=False,
          second_pass=False, fixed=False):
    if tune:
        if tune == 'exploration':
            return so_specs(profile)
        return tuning_specs(tune)
    specs = {}
    if not fixed and (base is None or base.selection_policy == 'uct' or selection):
        specs.update({name: {'dimension': 'exploration', 'round': i, 'chain': 'exploration', 'coarse': i == 0}
            for i, name in enumerate(('exploration_coarse', 'exploration_refine_1', 'exploration_refine_2'))})
    if selection:
        specs.update(tuning_specs('selection', prefix='selector'))
    for dimension in profile.mechanisms if mechanisms else ():
        specs.update(tuning_specs(dimension.replace('_', '-'), prefix='mechanism'))
    if second_pass:
        specs.update(tuning_specs('exploration', prefix='second'))
    return specs


def build_so_phase(name, state, base):
    if name != 'confirmation':
        # Full selection searches include UCT C challengers even from a tuned incumbent.
        spec = state['request']['tuner_specs'][name]
        selected = state.get('selected_candidate', {}).get('profile', {})
        if (spec['dimension'] == 'exploration' and selected.get('selection_policy', 'uct') == 'ucb1_tuned'
                and state['request'].get('selection_search') and not state['request'].get('tune')):
            temporary = {**state, 'selected_candidate': {'profile': profile_values(replace(
                agent_from_values(selected), selection_policy='uct'))}}
            phase = _build_tuning_phase(name, temporary, base)
            # Every challenge remains head-to-head against the frozen tuned incumbent.
            phase['agents']['incumbent'] = selected
            return phase
        return _build_tuning_phase(name, state, base)
    original = state['operating_baseline']
    finalist = state.get('selected_candidate', {}).get('profile', original)
    return {'name': name, 'agents': {'incumbent': original, 'finalist': finalist},
            'groups': {'main': ['incumbent', 'finalist']}, 'accept': True, 'incremental': True,
            'evidence_policy': 'confirmation', 'tie_priority': {'incumbent': [0], 'finalist': [1]},
            'contrasts': [] if original == finalist else [{'a': 'incumbent', 'b': 'finalist', 'primary': True, 'factor': state['request'].get('tune') or 'configuration'}],
            'status': 'pending', 'skip_reason': 'no_supported_improvement'}
