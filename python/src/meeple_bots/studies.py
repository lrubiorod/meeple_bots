"""Budgeted MCTS diagnosis built on the standard tournament executor and traces.

The registry still owns game capabilities. This coordinator never implements game rules.
Each contrast has its own standard trace so contrasts can share seeds without changing the
historical tournament seed schedule. Plans are frozen before their first match.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import subprocess
from statistics import median, mean
from time import perf_counter
from typing import Callable

from . import _native
from ._search_budget import decision_budget
from ._concurrency import WorkerSetting, resolve_workers
from ._agent_config import (
    ConditionalRollout, EpsilonGreedy, GameHeuristic, Greedy, Mast, MctsAgent,
    NeutralEvaluator, RandomAgent, SoIsmctsAgent, UniformRandom,
)
from ._capabilities import heuristic_indices, game_search_capabilities
from ._mcts_profiles import (
    _configured_cutoff_evaluator, _configured_progressive_bias, _configured_rollout_policy,
    _load_mcts_profile,
)
from .game_config import create_game, game_parameters
from .connect6 import Connect6
from .api import Boop, ConnectFour, SpiritsOfTheForest, TicTacToe, benchmark_mcts_agent
from .splendor import Splendor
from .lost_cities import LostCities
from ._study_profiles import PROFILES, resolve_family, load_search_profile, so_from_values
from .serialization import _evaluator_dict
from .tournaments import (
    TournamentAgent, TournamentConfig, TournamentTrace, match_jobs, run_matches,
    tournament_header,
)
from ._study_tuners import TUNING_FIELDS, proposals, config_fields, changes, assert_frozen, validate_tuner
from .study_analysis import summarize_contrast, write_study_report, study_diagnostics, search_adequacy

GAMES = {"lost_cities": LostCities,"connect6": Connect6, "boop": Boop, "spotf": SpiritsOfTheForest, "connect-four": ConnectFour,
         "tic-tac-toe": TicTacToe, "splendor": Splendor}
MAX_EXTENSION_ROUNDS = 3
STUDY_VERSION = 23
PHASES = ("depth_screen", "exploration", "selectors", "rave",
          *(f"rave_extend_{i}" for i in range(1, MAX_EXTENSION_ROUNDS + 1)),
          "rave_exploration", "rave_compare", "mechanisms", "pw_screen", "pw_k",
          *(f"pw_k_extend_{i}" for i in range(1, MAX_EXTENSION_ROUNDS + 1)), "pw_alpha",
          *(f"pw_alpha_extend_{i}" for i in range(1, MAX_EXTENSION_ROUNDS + 1)), "pw_refine", "pw_compare")
STAGES = ("depth", "selection", "rave", "mechanisms", "pw")


def stage_for_phase(name):
    if name == "depth_screen": return "depth"
    if name in ("exploration", "selectors"): return "selection"
    if name.startswith("rave"): return "rave"
    if name.startswith("pw_"): return "pw"
    return name


# Disjoint, fixed seed namespaces, including calibration. Never adapt seeds to results.
SEED_STRIDE = 100_000


def duration_seconds(value: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(s|m|h)?", value.strip())
    if not match:
        raise ValueError("budget must be a positive duration, e.g. 90s, 20m or 2h")
    seconds = float(match[1]) * {None: 1, "s": 1, "m": 60, "h": 3600}[match[2]]
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("budget must be finite and positive")
    return seconds


def policy_values(policy) -> dict:
    if isinstance(policy, ConditionalRollout):
        return {"kind": "conditional", "condition": {"kind": "turn_phase", "phase": policy.condition.phase},
                "primary": policy_values(policy.primary), "fallback": policy_values(policy.fallback)}
    if isinstance(policy, UniformRandom):
        return {"kind": "uniform_random"}
    if isinstance(policy, Mast):
        return {"kind": "mast", "epsilon": policy.epsilon}
    values = {"kind": "greedy" if isinstance(policy, Greedy) else "epsilon_greedy",
              "evaluator": _evaluator_dict(policy.evaluator)}
    if isinstance(policy, EpsilonGreedy):
        values["epsilon"] = policy.epsilon
    return values


def profile_values(agent: MctsAgent | SoIsmctsAgent) -> dict:
    if isinstance(agent, SoIsmctsAgent):
        return {"agent": "so_ismcts", "exploration": agent.exploration, "selection_policy": agent.selection_policy, "tree_reuse": agent.tree_reuse,
                "rollout": "uniform", "root_selection": "most_visited",
                **({"iterations": agent.iterations} if agent.iterations is not None else {"time_budget": agent.time_budget})}
    values = {"iterations": agent.iterations} if agent.iterations is not None else {"time_budget": agent.time_budget}
    values.update(exploration=agent.exploration, rollout_depth=agent.rollout_depth,
                  selection_policy=agent.selection_policy, cutoff_evaluator=_evaluator_dict(agent.cutoff_evaluator),
                  rollout_policy=policy_values(agent.rollout_policy), tree_reuse=agent.tree_reuse,
                  transpositions=agent.transpositions, root_diagnostics=agent.root_diagnostics)
    # Preserve inactive parameters too: coordinate tuning must not reset them.
    values.update(progressive_widening=agent.progressive_widening, progressive_widening_k=agent.progressive_widening_k, progressive_widening_alpha=agent.progressive_widening_alpha, progressive_widening_expansion=agent.progressive_widening_expansion)
    values["rave_equivalence"] = agent.rave_equivalence
    if agent.progressive_bias is not None:
        bias = agent.progressive_bias
        values["progressive_bias"] = {"weight": bias.weight, "evaluator": _evaluator_dict(bias.evaluator)}
        if bias.condition is not None:
            values["progressive_bias"]["condition"] = {"kind": "turn_phase", "phase": bias.condition.phase}
    return values


def agent_from_values(values: dict) -> MctsAgent | SoIsmctsAgent:
    if values.get("agent") == "so_ismcts":
        return so_from_values(values)
    return MctsAgent(iterations=values.get("iterations"), time_budget=values.get("time_budget"),
                     exploration=values["exploration"], rollout_depth=values["rollout_depth"],
                     selection_policy=values["selection_policy"],
                     rave_equivalence=values.get("rave_equivalence", 1000),
                     progressive_widening=values.get("progressive_widening", False),
                     progressive_widening_k=values.get("progressive_widening_k", 1.5),
                     progressive_widening_alpha=values.get("progressive_widening_alpha", 0.5),
                     progressive_widening_expansion=values.get("progressive_widening_expansion", "random"),
                     cutoff_evaluator=_configured_cutoff_evaluator(values, "study"),
                     rollout_policy=_configured_rollout_policy(values, "study"),
                     progressive_bias=_configured_progressive_bias(values, "study"),
                     tree_reuse=values["tree_reuse"], transpositions=values["transpositions"],
                     root_diagnostics=values["root_diagnostics"])


def _toml(value) -> str:
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{json.dumps(k)} = {_toml(v)}" for k, v in value.items()) + " }"
    return json.dumps(value, allow_nan=False)


def export_profile(path: Path, name: str, agent: MctsAgent | SoIsmctsAgent) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# Generated study candidate; strength claims require held-out confirmation.\n"
    text += f"name = {json.dumps(name)}\n"
    text += "".join(f"{key} = {_toml(value)}\n" for key, value in profile_values(agent).items())
    path.write_text(text, encoding="utf-8")


def generic_baseline(game: str) -> MctsAgent:
    return MctsAgent(iterations=1000, rollout_depth=64, exploration=math.sqrt(2),
                     cutoff_evaluator=NeutralEvaluator(), rollout_policy=UniformRandom(),
                     tree_reuse=False, transpositions=False)


def _study_budget(agent: MctsAgent | SoIsmctsAgent, calibration: dict) -> MctsAgent:
    if calibration.get("fixed_iterations") is not None:
        return replace(agent, iterations=calibration["fixed_iterations"], time_budget=None)
    return replace(agent, iterations=None, time_budget=calibration["decision_seconds"])


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


def _timed(agent: MctsAgent, seconds: float) -> MctsAgent:
    return replace(agent, iterations=None, time_budget=seconds, root_diagnostics=False)


def _iterations(agent: MctsAgent, count: int) -> MctsAgent:
    return replace(agent, iterations=max(1, min(2**32 - 1, count)), time_budget=None, root_diagnostics=False)


def _fingerprint() -> dict:
    def digest(path):
        return sha256(Path(path).read_bytes()).hexdigest()
    package = Path(__file__).parent
    # Refuse mixing different executors/search builds on resume, including uncommitted code.
    sources = sorted(package.glob("*.py"))
    python_hash = sha256(b"".join(p.name.encode() + p.read_bytes() for p in sources)).hexdigest()
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package, capture_output=True,
                                  text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"git_revision": revision, "native_sha256": digest(_native.__file__), "python_sha256": python_hash}


def _save(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


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


def cutoff_depths(horizon: int) -> list[int]:
    """Small shallow/medium/deep grid, strictly below the reference horizon."""
    return sorted({max(1, min(horizon - 1, round(horizon * f)))
                   for f in (.1, .25, .5, .75)}) if horizon > 1 else []


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


def _evidence_complete(contrast):
    return contrast.get("result", {}).get("seed_pairs", 0) >= max(4, contrast.get("target_pairs", 4))


def _reverse_result(result):
    return {**result, "score_b": 1-result["score_b"],
            "seed_scores_b": {k: 1-v for k, v in result.get("seed_scores_b", {}).items()}}


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


class StudyRunner:
    def __init__(self, game: str, baseline: MctsAgent | SoIsmctsAgent | None = None, *, output: Path, budget: float | None = None,
                 reference: MctsAgent | None = None, seed: int = 42, max_pairs: int | None = None,
                 games_per_comparison: int = 50, stage_games: dict[str, int] | None = None,
                 decision_seconds: float | None = None, screening_seconds: float | None = None, max_plies: int = 10000,
                 heuristic: int | None = None, target_match_time: float = 60.0, rave_search: bool = False, pw_search: bool = False,
                 selection_search: bool = False, mechanism_search: bool = False, depth_search: bool = False, all_search: bool = False,
                 tune: str | None = None, second_pass: bool = False, widening_expansion_search: bool = False,
                 agent_family: str | None = None, vs_random: bool = False, workers: WorkerSetting = "auto", resume: bool = False, allow_engine_change: bool = False, game_params: dict | None = None, progress: Callable[[str], None] = print):
        if allow_engine_change and not resume:
            raise ValueError("--allow-engine-change requires --resume and an existing study")
        self.family = resolve_family(game, agent_family, baseline)
        self.family_profile = PROFILES[self.family]
        self.profile = self.family_profile if self.family != "mcts" else None
        if game not in GAMES:
            raise ValueError("automatic studies require generic tournament transport; supported: " + ", ".join(GAMES))
        if budget is not None and (not math.isfinite(budget) or budget <= 0):
            raise ValueError("study budget must be finite and positive")
        stage_games = dict(stage_games or {})
        if set(stage_games) - set(STAGES):
            raise ValueError("unknown stage-games stage; expected " + ", ".join(STAGES))
        for label, count in {"games_per_comparison": games_per_comparison, **stage_games}.items():
            if type(count) is not int or count < 4 or count % 2 or count >= 2 * SEED_STRIDE:
                raise ValueError(f"{label} must be an even number of games between 4 and 199998")
        max_pairs = games_per_comparison // 2 if max_pairs is None else max_pairs
        for label, count in dict(max_pairs=max_pairs).items():
            if type(count) is not int or not 2 <= count < SEED_STRIDE:
                raise ValueError(f"{label} must be between 2 and 99999")
        if screening_seconds is not None and (not math.isfinite(screening_seconds) or screening_seconds <= 0):
            raise ValueError("screening time must be finite and positive")
        if type(seed) is not int or not 0 <= seed < 2**64 - (len(PHASES) + 1) * SEED_STRIDE:
            raise ValueError("seed leaves insufficient room for disjoint study phases")
        if type(max_plies) is not int or not 1 <= max_plies < 2**32:
            raise ValueError("max_plies must be a positive u32")
        if decision_seconds is not None and (not math.isfinite(decision_seconds) or decision_seconds <= 0):
            raise ValueError("decision time must be finite and positive")
        if not math.isfinite(target_match_time) or target_match_time <= 0:
            raise ValueError("target match time must be finite and positive")
        if screening_seconds is not None:
            raise ValueError("--screening-time was removed: all candidates use the same derived decision budget")
        if heuristic is not None and (type(heuristic) is not int or heuristic not in heuristic_indices(game)):
            raise ValueError(f"unknown cutoff heuristic H{heuristic} for {game}")
        supplied = baseline is not None
        for name, value in dict(vs_random=vs_random, all_search=all_search, selection_search=selection_search,
                                mechanism_search=mechanism_search, second_pass=second_pass).items():
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.profile:
            for flag, dimension in ((rave_search, 'rave'), (pw_search, 'progressive-widening'),
                                    (depth_search, 'cutoff-depth'), (widening_expansion_search, 'widening-expansion')):
                if flag:
                    raise ValueError(f"tuner '{dimension}' is not supported by agent family '{self.family}'")
            if heuristic is not None or reference is not None:
                raise ValueError('SO-ISMCTS does not support heuristic/reference options')
            if stage_games.keys() - {'selection', 'mechanisms'}:
                raise ValueError('SO-ISMCTS supports stage-games selection and mechanisms')
            baseline = baseline or SoIsmctsAgent()
            selectors = game_search_capabilities(game)['selection_policies']
            if tune is not None:
                if not supplied or decision_seconds is not None:
                    raise ValueError('--tune requires --agent-config and freezes its search budget')
                if any((selection_search, mechanism_search, all_search, second_pass)):
                    raise ValueError('--tune cannot be combined with full-study stage flags')
                validate_tuner(tune, baseline, selectors)
            selection_search = selection_search or all_search
            if mechanism_search and not self.profile.mechanisms:
                raise ValueError(f"mechanism search is not supported by agent family '{self.family}'")
            mechanism_search = mechanism_search or (all_search and bool(self.profile.mechanisms))
            fixed = supplied and vs_random and not any((tune, selection_search, mechanism_search, second_pass))
            specs = self.profile.specs(baseline, tune=tune, selection=selection_search,
                                       mechanisms=mechanism_search, second_pass=second_pass, fixed=fixed)
            self.phase_names = (*specs, 'confirmation')
            pw_supported = False
        else:
            if tune is not None:
                if baseline is None:
                    raise ValueError("--tune requires --baseline/--agent-config")
                if any((all_search, selection_search, rave_search, pw_search, mechanism_search, depth_search, second_pass, widening_expansion_search)):
                    raise ValueError("--tune cannot be combined with full-study stages or --second-pass")
                if decision_seconds is not None:
                    raise ValueError("--tune freezes the agent search budget; remove --decision-time")
                validate_tuner(tune, baseline, game_search_capabilities(game)["selection_policies"])
            if baseline is None:
                baseline = replace(generic_baseline(game), cutoff_evaluator=NeutralEvaluator() if heuristic is None else GameHeuristic(heuristic))
            elif heuristic is not None and baseline.cutoff_evaluator != GameHeuristic(heuristic):
                raise ValueError("--heuristic conflicts with the supplied baseline evaluator")
            evaluator = baseline.cutoff_evaluator
            heuristic = evaluator.index if isinstance(evaluator, GameHeuristic) else None
            if reference and reference.cutoff_evaluator != evaluator:
                raise ValueError("reference must belong to the same evaluator family; compare families in a separate tournament")
            selectors = game_search_capabilities(game)["selection_policies"]
            for label, agent in (("baseline", baseline), ("reference", reference)):
                if agent and agent.selection_policy not in selectors:
                    raise ValueError(f"{label} selection policy is unavailable for this game")
            for name, value in dict(rave_search=rave_search, pw_search=pw_search, selection_search=selection_search,
                                    mechanism_search=mechanism_search, depth_search=depth_search, all_search=all_search, second_pass=second_pass, widening_expansion_search=widening_expansion_search).items():
                if type(value) is not bool:
                    raise ValueError(f"{name} must be a boolean")
            rave_search, pw_search, selection_search, mechanism_search, depth_search = [
                all_search or value for value in (rave_search, pw_search, selection_search, mechanism_search, depth_search)]
            specs, self.phase_names = self.family_profile.plan(
                tune=tune, admission=widening_expansion_search or all_search, second_pass=second_pass)
            if seed >= 2**64 - (len(self.phase_names)+1)*SEED_STRIDE:
                raise ValueError("seed leaves insufficient room for tuning phases")
            # Current catalog exposes UCT-RAVE exactly for deterministic search backends.
            pw_supported = "uct_rave" in selectors
            if not pw_supported and any(a and a.progressive_widening for a in (baseline, reference)):
                raise ValueError("PW requires a deterministic backend")
            if reference is not None:
                raise ValueError("--reference was removed with confirmation; compare it in a separate tournament")
        if supplied and vs_random and not any((tune, all_search, selection_search, mechanism_search,
                rave_search, pw_search, depth_search, second_pass, widening_expansion_search)):
            self.phase_names = ('confirmation',) if self.profile else ()
        if vs_random:
            self.phase_names = (*self.phase_names, 'random_baseline')
        if seed >= 2**64 - (len(self.phase_names)+1)*SEED_STRIDE:
            raise ValueError('seed leaves insufficient room for tuning and random baseline')
        worker_count = resolve_workers(workers)
        self.game = create_game(game, game_params)
        self.base, self.reference = baseline, reference
        self.output, self.budget = output.resolve(), budget
        self.progress, self.resume = progress, resume
        request = {"version": STUDY_VERSION, "max_extension_rounds": MAX_EXTENSION_ROUNDS, "execution_mode": "local_retune" if tune else "full_study", "minimum_evidence_pairs": 4, "tune": tune, "second_pass": second_pass, "widening_expansion_search": widening_expansion_search, "tuner_specs": specs, "phase_names": list(self.phase_names), "games_per_comparison": 2 * max_pairs, "stage_games": stage_games, "baseline_supplied": supplied, "selection_search": selection_search, "mechanism_search": mechanism_search, "depth_search": depth_search, "pw_search": pw_search, "pw_supported": pw_supported, "rave_search": rave_search, "mode": "full_depth" if heuristic is None else "heuristic_cutoff",
                   "heuristic": heuristic, "target_match_time": target_match_time, "safety_margin": 1.2,
                   "selection_policies": selectors, "game": game, **({"game_params": game_parameters(self.game)} if game_parameters(self.game) else {}), "baseline": profile_values(baseline),
                   "reference": profile_values(reference) if reference else None,
                   "seed": seed, "max_pairs": max_pairs,
                   "decision_seconds": decision_seconds, "screening_seconds": screening_seconds,
                   "max_plies": max_plies, "workers": worker_count, "engine": _fingerprint()}
        request.update(agent_family=self.family, family_profile_version=self.family_profile.version,
                       supported_tuners=list(self.family_profile.resolved_tuners(game_search_capabilities(game))),
                       vs_random=vs_random, all_search=all_search)
        if self.profile:
            request.update(mode="information_set")
        self.path = self.output / "study.json"
        if self.path.exists():
            if not resume:
                raise FileExistsError(f"study exists: {self.path}; use --resume")
            self.state = json.loads(self.path.read_text())
            if self.state["request"].get("version") != STUDY_VERSION:
                raise ValueError("old study protocol / extension plan cannot resume with the 3-extension limit; existing checkpoint is unchanged; use a new output directory")
            saved_request = self.state["request"]
            # Permission to change binaries is never permission to change the frozen plan.
            if {k: v for k, v in saved_request.items() if k != "engine"} != {k: v for k, v in request.items() if k != "engine"}:
                raise ValueError("study configuration or engine changed: configuration differs; use a new output directory")
            previous_engine = self.state.get("active_engine", saved_request["engine"])
            current_engine = request["engine"]
            # Git revision is provenance, not an execution compatibility key.
            if {k: v for k, v in previous_engine.items() if k != "git_revision"} != {k: v for k, v in current_engine.items() if k != "git_revision"}:
                if not allow_engine_change:
                    raise ValueError("study configuration or engine changed: engine hashes differ; restore the original environment or explicitly use --resume --allow-engine-change to record a mixed-engine continuation")
                completed = {}
                for trace in sorted((self.output / "traces").glob("*.jsonl")):
                    with trace.open() as handle:
                        next(handle)  # Standard tournament header.
                        completed[str(trace.relative_to(self.output))] = [json.loads(line)["match_number"] for line in handle]
                self.state.setdefault("engine_changes", []).append({
                    "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
                    "previous_engine": previous_engine, "new_engine": current_engine,
                    "completed_matches_before_change": completed,
                })
                self.state["active_engine"] = current_engine
                self.state["mixed_engines"] = True
            if self.state.get("mixed_engines"):
                self.progress("Warning: mixed-engine study. Existing games are retained; code/build changes can affect search throughput and strength, especially with time budgets. See engine_changes in study.json.")
        else:
            if allow_engine_change:
                raise ValueError("--allow-engine-change requires --resume and an existing study")
            if self.output.exists() and any(self.output.iterdir()):
                raise FileExistsError("study output directory must be empty")
            self.output.mkdir(parents=True, exist_ok=True)
            self.state = {"request": request, "budget_seconds": budget, "spent_seconds": 0,
                          "calibration": None, "phases": {}, "status": "pending"}
            export_profile(self.output / "baseline.toml", "starting-baseline", baseline)
            if reference:
                export_profile(self.output / "reference.toml", "held-out-reference", reference)
        if budget is not None and self.state.get("budget_seconds") is not None and budget < self.state["budget_seconds"]:
            raise ValueError("resumed budget cannot be smaller than the original budget")
        self.state["budget_seconds"] = budget
        self.started = perf_counter()
        self.previous_spent = self.state["spent_seconds"]
        self.save()

    @property
    def spent(self):
        return self.previous_spent + perf_counter() - self.started

    def save(self):
        if self.profile:
            self.state.setdefault("selected_candidate", {"phase": None, "name": "incumbent", "profile": profile_values(self.base)})
        screen = self.state["phases"].get("pw_screen")
        if screen and "pw_policies" in screen:
            screen["pw_decisions"].update(_pw_survivors(screen)[1])
        accepted = {"depth_screen", "exploration", "selectors", "rave_compare", "mechanisms", "pw_compare"}
        selected = next((p for name, p in reversed(list(self.state["phases"].items()))
                         if (name in accepted or p.get("accept")) and p["status"] == "complete"), None)
        if selected:
            winner = next(iter(_group_leaders(selected).values()))
            self.state["selected_candidate"] = {"phase": selected["name"], "name": winner, "profile": selected["agents"][winner]}
        if self.state["request"].get("tune"):
            final = agent_from_values(self.state.get("selected_candidate", {}).get("profile", profile_values(self.base)))
            dimension = self.state["request"]["tune"]
            assert_frozen(self.base, final, dimension)
            modified = changes(self.base, final)
            self.state["local_retune"] = {"dimension": dimension, "changed_fields": modified,
                "preserved_fields": {k: v for k, v in config_fields(self.base).items() if k not in modified},
                "result": "IMPROVED" if modified else "INCONCLUSIVE",
                "message": "Supported exploratory improvement found." if modified else "No sufficiently supported improvement found."}
            self.state.setdefault("selected_candidate", {"phase": None, "name": "incumbent", "profile": profile_values(self.base)})
        self.state.update(game=self.state["request"]["game"], agent_family=self.family)
        from ._study_output import compute_budget
        if self.state.get("calibration"):
            self.state["calibration"]["compute_budget"] = compute_budget(self.state)
        self.state["spent_seconds"] = self.spent
        self.state.update(study_diagnostics(self.state))
        self.state["random_baseline"] = self._random_summary()
        _save(self.path, self.state)

    def _random_summary(self):
        phase = self.state['phases'].get('random_baseline', {})
        contrast = next(iter(phase.get('contrasts', [])), {})
        r = contrast.get('result', {})
        return {'enabled': self.state['request'].get('vs_random', False),
                'status': ('skipped' if phase.get('completion_reason') == 'insufficient_budget' else
                    phase.get('status', 'pending' if self.state['request'].get('vs_random') else 'disabled')),
                'reason': phase.get('completion_reason') or ('insufficient_budget' if self.state.get('status') == 'budget_exhausted' else None), 'champion': phase.get('frozen_champion'),
                'games': r.get('games', 0), 'unpaired_games': r.get('unpaired_games', 0),
                'paired_seeds': r.get('seed_pairs', 0), 'verdict': r.get('verdict', 'not_measured'),
                'wins': r.get('wins_b', 0), 'draws': r.get('draws', 0), 'losses': r.get('losses_b', 0),
                'score': r.get('score_b'), 'ci95': r.get('ci95_b'),
                'interpretation': 'Diagnostic only. Not used for candidate selection.'}

    def _random_phase(self):
        champion = self.state.get('selected_candidate', {}).get('profile')
        if champion is None:
            champion = profile_values(_study_budget(self.base, self.state['calibration']))
            self.state['selected_candidate'] = {'phase': None, 'name': 'incumbent', 'profile': champion}
        return {'name': 'random_baseline', 'agents': {'random': None, 'champion': champion},
                'groups': {}, 'accept': False, 'descriptive': True, 'evidence_policy': 'descriptive',
                'frozen_champion': champion, 'status': 'pending',
                'contrasts': [{'a': 'random', 'b': 'champion', 'factor': 'random_reference'}]}

    def _trace_config(self, phase: dict, index: int, *, pilot=False):
        contrast = phase["contrasts"][index]
        agents = tuple(TournamentAgent(n, RandomAgent() if phase["agents"][n] is None else agent_from_values(phase["agents"][n]))
                       for n in (contrast["a"], contrast["b"]))
        path = self.output / "traces" / f"{phase['name']}-{index:02d}.jsonl"
        phase_index = 0 if pilot else self.phase_names.index(phase["name"]) + 1
        return TournamentConfig(game=self.game, output=path, pairing_mode="round_robin", seat_mode="paired",
                                matches_per_pair=2 if pilot else 2 * contrast.get("target_pairs", phase["planned_pairs"]),
                                seed=self.state["request"]["seed"] + phase_index * SEED_STRIDE,
                                max_plies=self.state["request"]["max_plies"],
                                workers=self._contrast_workers(phase, index), agents=agents)

    def _contrast_workers(self, phase: dict, index: int) -> int:
        return self.state["request"]["workers"]

    def _batches(self, phase: dict, round_index: int):
        parallel = []
        for index, contrast in enumerate(phase["contrasts"]):
            if contrast.get("screening_status") == "not_prioritized":
                continue
            if round_index >= contrast.get("target_pairs", phase.get("planned_pairs", SEED_STRIDE)):
                continue
            if contrast.get("result", {}).get("seed_pairs", 0) > round_index:
                continue
            if self._contrast_workers(phase, index) == 1:
                yield [index]
            else:
                parallel.append(index)
        workers = self.state["request"]["workers"]
        for offset in range(0, len(parallel), workers):
            yield parallel[offset:offset + workers]

    def _batch(self, phase: dict, indices: list[int], round_index: int, *, pilot=False):
        # Only the coordinator writes traces/state, even when matches run in threads.
        configs, owners, jobs = {}, {}, []
        with ExitStack() as stack:
            for index in indices:
                config = self._trace_config(phase, index, pilot=pilot)
                configs[index] = config
                header = tournament_header(config, config.output, config.workers)
                trace = stack.enter_context(TournamentTrace(config.output, header, resume=config.output.exists()))
                pair = list(match_jobs([(config.agents[0], config.agents[1])], config))[2*round_index:2*round_index+2]
                for job in pair:
                    if job.match_number not in trace.completed_match_numbers:
                        jobs.append(job)
                        owners[id(job)] = trace
            workers = min(self._contrast_workers(phase, i) for i in indices)
            for job, outcome in run_matches(self.game, jobs, max_plies=self.state["request"]["max_plies"],
                                           workers=min(workers, max(1, len(jobs)))):
                owners[id(job)].write(job, outcome)
                self.save()
        rows_by_index = {}
        for index, config in configs.items():
            rows = [json.loads(line) for line in config.output.read_text().splitlines()[1:]]
            contrast = phase["contrasts"][index]
            contrast["result"] = summarize_contrast(rows)
            if phase.get("evidence_policy") not in ("confirmation", "descriptive"):
                contrast["result"]["verdict"] = "exploratory"
            contrast["trace"] = str(config.output.relative_to(self.output))
            rows_by_index[index] = rows
        self.save()
        return rows_by_index

    def _pair(self, phase: dict, index: int, round_index: int, *, pilot=False):
        return self._batch(phase, [index], round_index, pilot=pilot)[index]

    def calibrate(self):
        if self.profile:
            return self.profile.calibrate(self)
        if self.state["calibration"]:
            return
        request = self.state["request"]
        self.progress("Calibrating search cost and horizon...")
        work = self.state.setdefault("calibration_progress", {})
        pilot_agent = replace(_iterations(self.base, 32), rollout_depth=64)
        pilot = {"name": "calibration", "agents": {"pilot": profile_values(pilot_agent), "random": None},
                 "contrasts": [{"a": "pilot", "b": "random", "factor": "sanity"}]}
        if "mean_plies" not in work:
            rows = self._pair(pilot, 0, 0, pilot=True)
            # Simulation's plies/moves count only player decisions; chance_events is separate.
            work["mean_plies"] = mean(r["result"]["plies"] for r in rows)
            work["pilot"] = pilot
            self.save()
        length = max(1, work["mean_plies"])
        target = request["decision_seconds"] or (self.base.time_budget if request["baseline_supplied"] else None) or decision_budget(request["target_match_time"], length, request["safety_margin"])
        fixed_iterations = self.base.iterations if request["baseline_supplied"] and request["decision_seconds"] is None else None
        if request["baseline_supplied"] and request.get("tune") != "cutoff-depth":
            work["horizon"] = {"depth": self.base.rollout_depth, "kind": "baseline", "samples": []}
        samples = work.setdefault("horizon_samples", [])
        depth = samples[-1]["depth"] if samples else min(64, request["max_plies"])
        neutral = replace(self.base, heuristic=None, cutoff_evaluator=NeutralEvaluator())
        while "horizon" not in work:
            if not samples or samples[-1]["depth"] != depth:
                # Two independent search/state seeds, same depths for stable comparisons.
                benches = [benchmark_mcts_agent(self.game, _iterations(replace(neutral, rollout_depth=depth), 64),
                                                round(length), request["seed"] + offset) for offset in (17, 31)]
                timings = [asdict(t) for b in benches for t in b.position_timings]
                hard = benches[0].maximum_decision_horizon
                terminal = sum(t["terminal_simulations"] or 0 for t in timings)
                cutoff = sum(t["cutoff_simulations"] or 0 for t in timings)
                fractions = [t["terminal_simulations"] / max(1, (t["terminal_simulations"] or 0) + (t["cutoff_simulations"] or 0))
                             for t in timings if t["terminal_simulations"] is not None]
                samples.append({"depth": depth, "hard_maximum": hard, "terminal_fraction": terminal / max(1, terminal+cutoff),
                                "minimum_position_terminal_fraction": min(fractions, default=0), "position_timings": timings})
                self.save()
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
        if self.reference and not request["baseline_supplied"]:
            if (request["mode"] == "full_depth" and self.reference.rollout_depth != horizon["depth"]) or (
                    request["mode"] == "heuristic_cutoff" and self.reference.rollout_depth >= horizon["depth"]):
                raise ValueError("reference horizon belongs to another family; compare it in a separate tournament")
        operating_depth = self.base.rollout_depth if request["baseline_supplied"] else depths[len(depths)//2] if depths else horizon["depth"]
        if "position_timings" not in work:
            probe = replace(self.base if fixed_iterations else _timed(self.base, target), rollout_depth=operating_depth, root_diagnostics=True)
            bench = benchmark_mcts_agent(self.game, probe, round(length), request["seed"] + 47)
            work["position_timings"] = [asdict(t) for t in bench.position_timings]
            self.save()
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
               "pilot": work["pilot"], "seconds": self.spent}
        warnings = []
        if horizon["kind"] == "practical_unverified":
            warnings.append("Full-depth is impractical or unverified within calibration limits; this is an explicit neutral safety cutoff, not proven terminal search.")
        if adequacy["category"] in ("LOW", "VERY LOW"):
            warnings.append("Requested target provides little search per decision; results describe under-searched agents. The requested target is unchanged.")
        cal["warnings"] = warnings
        self.state["calibration"] = cal
        self.save()

    def _estimated_round_seconds(self, phase: dict) -> float:
        cal = self.state["calibration"]
        return cal["mean_plies"] * sum(
            (phase["agents"][c[role]].get("time_budget") or phase["agents"][c[role]]["iterations"] * cal["seconds_per_iteration"])
            for c in phase["contrasts"] for role in ("a", "b") if phase["agents"][c[role]] is not None)

    def _recover(self):
        duration = 0.0
        phases = list(self.state["phases"].values())
        cal = self.state.get("calibration") or self.state.get("calibration_progress")
        if cal and "pilot" in cal:
            phases = [cal["pilot"], *phases]
        for phase in phases:
            for index, contrast in enumerate(phase["contrasts"]):
                if contrast.get("target_pairs") == 0:
                    continue
                config = self._trace_config(phase, index, pilot=phase["name"] == "calibration")
                if not config.output.exists():
                    if contrast.get("result", {}).get("games", 0):
                        raise ValueError(f"missing study trace: {config.output}")
                    continue
                with TournamentTrace(config.output, tournament_header(config, config.output, config.workers), resume=True):
                    pass
                rows = [json.loads(line) for line in config.output.read_text().splitlines()[1:]]
                contrast["result"] = summarize_contrast(rows)
                if phase.get("evidence_policy") not in ("confirmation", "descriptive"):
                    contrast["result"]["verdict"] = "exploratory"
                duration += sum(row["duration_seconds"] for row in rows) / config.workers
        # A process can terminate after flushing a match but before saving the state.
        # Never forget the elapsed time already represented by durable results.
        self.previous_spent = max(self.previous_spent, duration)

    def _cost_pair(self, phase, contrast):
        result = contrast.get("result", {})
        elapsed = sum(result.get(t, {}).get("total_seconds", 0) for t in ("timing_a", "timing_b"))
        if result.get("seed_pairs", 0) and elapsed:
            return elapsed / result["seed_pairs"]
        if self.state["calibration"].get("fixed_iterations"):
            return self._estimated_round_seconds({**phase, "contrasts": [contrast]})
        # Recent completed candidate games replace the weak pilot's length estimate.
        # Normalize by decision budgets; retain 20% headroom for different opponents.
        for previous in reversed(list(self.state["phases"].values())):
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
        return self._estimated_round_seconds({**phase, "contrasts": [contrast]})

    def _plan_phase(self, phase, index):
        request = self.state["request"]
        stage = {"exploration": "selection", "selection": "selection", "rave": "rave", "structure": "mechanisms", "tree-reuse": "mechanisms", "cutoff-depth": "depth"}.get(phase.get("tuner"), "pw" if phase.get("tuner") else stage_for_phase(phase["name"]))
        default_pairs = request["max_pairs"]
        for ci, contrast in enumerate(phase["contrasts"]):
            pairs = max(4, request.get("stage_games", {}).get(stage, 2 * default_pairs) // 2)
            contrast.update(target_pairs=pairs, workers=self._contrast_workers(phase, ci),
                            timing_mode="isolated" if self._contrast_workers(phase, ci) == 1 else "shared_cpu")
        # Keep complete evidence for a smaller shortlist. Include 20% headroom,
        # limiting parallelism to the paired jobs available in this race.
        remaining = max(0., self.budget-self.spent) if self.budget is not None else math.inf
        retained, dropped = [], []
        for contrast in phase["contrasts"]:
            cost = 1.2 * self._cost_pair(phase, contrast) * contrast["target_pairs"]
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
        if dropped:
            self.progress(f"{phase['name']}: insufficient budget for {len(dropped)} challengers; retaining {len(retained)} with full evidence.")
        phase.update(planned_pairs=max((c["target_pairs"] for c in phase["contrasts"]), default=0),
                     planned_games=sum(2*c["target_pairs"] for c in phase["contrasts"]),
                     estimated_seconds=sum(c["estimated_seconds"] for c in phase["contrasts"]) / max(1, min(request["workers"], 2*len(phase["contrasts"]))),
                     allocation_mode="fixed_games")

    def _announce_plan(self):
        from ._study_output import announce_plan
        announce_plan(self)

    def run(self):
        try:
            if self.resume:
                self._recover()
            if all(self.state["phases"].get(name, {}).get("status") == "complete" for name in self.phase_names):
                self.state["status"] = "complete"
                self.state.pop("last_error", None)
                return self.state
            if self.budget is not None and self.spent >= self.budget:
                self.state["status"] = "budget_exhausted"
                return self.state
            self.calibrate()
            if self.state["calibration"] is None:
                self.state["status"] = "budget_exhausted"
                return self.state
            self._announce_plan()
            self.state.pop("last_error", None)
            self.state["status"] = "running"
            for index, name in enumerate(self.phase_names):
                phase = self.state["phases"].get(name)
                if phase and phase["status"] == "complete":
                    continue
                if self.budget is not None and self.spent >= self.budget:
                    break
                if phase is None:
                    phase = (self._random_phase() if name == 'random_baseline' else
                             self.profile.build_phase(name, self.state, self.base) if self.profile else
                             _build_phase(name, self.state, self.base, self.reference))
                    self._plan_phase(phase, index)
                    self.state["phases"][name] = phase
                    self.save()
                if not phase["contrasts"] or not phase["planned_pairs"]:
                    phase["status"] = "complete"
                    phase["completion_reason"] = "insufficient_budget" if phase.get("discarded_comparisons") else phase.get("skip_reason", "no_changed_parameters")
                    if name == 'random_baseline':
                        self.progress('Random baseline skipped: ' + phase['completion_reason'])
                    self.save()
                    continue
                phase["status"] = "running"
                from ._study_output import announce_extension
                announce_extension(self, phase)
                self.progress(f"{name}: {len(phase['contrasts'])} contrasts, {phase['planned_games']} fixed games; estimate {phase['estimated_seconds']:.0f}s (not a limit).")
                for round_index in range(phase["planned_pairs"]):
                    for indices in self._batches(phase, round_index):
                        if self.budget is not None and self.spent >= self.budget:
                            phase["status"] = "budget_exhausted"
                            self.state["status"] = "budget_exhausted"
                            return self.state
                        self._batch(phase, indices, round_index)
                    self.progress(f"{name}: paired round {round_index+1}/{phase['planned_pairs']} complete ({self.spent:.0f}s used).")
                if any(c.get("result", {}).get("seed_pairs", 0) < c["target_pairs"] for c in phase["contrasts"]):
                    raise RuntimeError(f"{name}: fixed comparison games remain incomplete")
                phase["status"] = "complete"
                phase["completion_reason"] = "fixed_games_finished"
                if not phase.get('descriptive'):
                    phase["winner"] = next(iter(_group_leaders(phase).values()))
                    control = phase["contrasts"][0]["a"]
                    phase["outcome"] = "IMPROVED" if phase["winner"] != control else "INCONCLUSIVE"
                announce_extension(self, phase, completed=True)
                self.save()
            complete = all(self.state["phases"].get(n, {}).get("status") == "complete" for n in self.phase_names)
            self.state["status"] = "complete" if complete else "budget_exhausted"
            if complete and any(p.get("discarded_comparisons") for p in self.state["phases"].values()):
                self.state["budget_limited"] = True
        except BaseException as error:
            self.state["status"] = "interrupted"
            self.state["last_error"] = str(error) or type(error).__name__
            raise
        finally:
            self.save()
            self.export_candidates()
            write_study_report(self.output, self.state)
            if self.state['request'].get('vs_random'):
                self.progress('RANDOM BASELINE: ' + json.dumps(self._random_summary(), sort_keys=True))
            if self.state.get("local_retune"):
                self.progress(self.state["local_retune"]["message"])
                self.progress("Changed fields: " + json.dumps(self.state["local_retune"]["changed_fields"], sort_keys=True))
        return self.state

    def export_candidates(self):
        profiles = {}
        for phase in self.state["phases"].values():
            if phase["status"] == "complete" and not phase.get('descriptive'):
                for leader in _group_leaders(phase).values():
                    profiles[f"{phase['name']}-{leader}"] = phase["agents"][leader]
        selected = self.state.get("selected_candidate")
        if selected:
            profiles["best_agent"] = selected["profile"]
            if not self.profile and not self.state["request"]["baseline_supplied"]:
                label = "best_full_depth_agent" if self.state["request"]["mode"] == "full_depth" else "best_heuristic_cutoff_agent"
                profiles[label] = selected["profile"]
        directory = self.output / "candidates"
        for name, values in profiles.items():
            export_profile(directory / f"{name}.toml", name, agent_from_values(values))
        self.state["candidate_profiles"] = {name: str((directory / f"{name}.toml").relative_to(self.output)) for name in profiles}
        self.save()


def run_study(game: str, *, output: Path, budget: float | None = None, baseline: Path | None = None,
              reference: Path | None = None, **kwargs) -> dict:
    if game not in GAMES:
        raise ValueError("automatic studies require generic tournament transport; supported: " + ", ".join(GAMES))
    base = load_search_profile(baseline) if baseline else None
    ref = load_search_profile(reference) if reference else None
    return StudyRunner(game, base, output=output, budget=budget, reference=ref, **kwargs).run()
