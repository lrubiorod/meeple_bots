"""Budgeted MCTS diagnosis built on the standard tournament executor and traces.

The registry still owns game capabilities. This coordinator never implements game rules.
Each contrast has its own standard trace so contrasts can share seeds without changing the
historical tournament seed schedule. Plans are frozen before their first match.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, replace
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import re
import subprocess
from statistics import median, mean
from time import perf_counter
from typing import Callable

from . import _native
from ._concurrency import WorkerSetting, resolve_workers
from ._agent_config import (
    ConditionalRollout, EpsilonGreedy, GameHeuristic, Greedy, Mast, MctsAgent,
    NeutralEvaluator, RandomAgent, UniformRandom,
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
from .serialization import _evaluator_dict
from .tournaments import (
    TournamentAgent, TournamentConfig, TournamentTrace, match_jobs, run_matches,
    tournament_header,
)
from .study_analysis import summarize_contrast, write_study_report, study_diagnostics, search_adequacy

GAMES = {"connect6": Connect6, "boop": Boop, "spotf": SpiritsOfTheForest, "connect-four": ConnectFour,
         "tic-tac-toe": TicTacToe, "splendor": Splendor}
PHASES = ("depth_screen", "exploration", "selectors", "rave", "rave_extend_1", "rave_extend_2", "rave_extend_3", "rave_extend_4", "rave_extend_5",
          "rave_exploration", "rave_compare", "family_selection",
          "mechanisms", "refinement", "confirmation")
# Relative resource priorities; unspent allocations flow forward. Confirmation owns 30%.
# RAVE owns a single 25% pool, booked at entry rather than split among stages.
PHASE_WEIGHTS = (.08, .12, .07, .25, 0, 0, 0, 0, 0, 0, 0, .03, .08, .07, .30)
RACING_PHASES = set(PHASES) - {"confirmation"}
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


def profile_values(agent: MctsAgent) -> dict:
    values = {"iterations": agent.iterations} if agent.iterations is not None else {"time_budget": agent.time_budget}
    values.update(exploration=agent.exploration, rollout_depth=agent.rollout_depth,
                  selection_policy=agent.selection_policy, cutoff_evaluator=_evaluator_dict(agent.cutoff_evaluator),
                  rollout_policy=policy_values(agent.rollout_policy), tree_reuse=agent.tree_reuse,
                  transpositions=agent.transpositions, root_diagnostics=agent.root_diagnostics)
    if agent.progressive_widening:
        values.update(progressive_widening=agent.progressive_widening, progressive_widening_k=agent.progressive_widening_k, progressive_widening_alpha=agent.progressive_widening_alpha)
    if agent.selection_policy == "uct_rave":
        values["rave_equivalence"] = agent.rave_equivalence
    if agent.progressive_bias is not None:
        bias = agent.progressive_bias
        values["progressive_bias"] = {"weight": bias.weight, "evaluator": _evaluator_dict(bias.evaluator)}
        if bias.condition is not None:
            values["progressive_bias"]["condition"] = {"kind": "turn_phase", "phase": bias.condition.phase}
    return values


def agent_from_values(values: dict) -> MctsAgent:
    return MctsAgent(iterations=values.get("iterations"), time_budget=values.get("time_budget"),
                     exploration=values["exploration"], rollout_depth=values["rollout_depth"],
                     selection_policy=values["selection_policy"],
                     rave_equivalence=values.get("rave_equivalence", 1000),
                     progressive_widening=values.get("progressive_widening", False),
                     progressive_widening_k=values.get("progressive_widening_k", 1.5),
                     progressive_widening_alpha=values.get("progressive_widening_alpha", 0.5),
                     cutoff_evaluator=_configured_cutoff_evaluator(values, "study"),
                     rollout_policy=_configured_rollout_policy(values, "study"),
                     progressive_bias=_configured_progressive_bias(values, "study"),
                     tree_reuse=values["tree_reuse"], transpositions=values["transpositions"],
                     root_diagnostics=values["root_diagnostics"])


def _toml(value) -> str:
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{json.dumps(k)} = {_toml(v)}" for k, v in value.items()) + " }"
    return json.dumps(value, allow_nan=False)


def export_profile(path: Path, name: str, agent: MctsAgent) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# Generated study candidate; strength claims require held-out confirmation.\n"
    text += f"name = {json.dumps(name)}\n"
    text += "".join(f"{key} = {_toml(value)}\n" for key, value in profile_values(agent).items())
    path.write_text(text, encoding="utf-8")


def generic_baseline(game: str) -> MctsAgent:
    return MctsAgent(iterations=1000, rollout_depth=64, exploration=math.sqrt(2),
                     cutoff_evaluator=NeutralEvaluator(), rollout_policy=UniformRandom(),
                     tree_reuse=False, transpositions=False)


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
        if result.get("score_b") is not None:
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


def cutoff_depths(horizon: int) -> list[int]:
    """Small shallow/medium/deep grid, strictly below the reference horizon."""
    return sorted({max(1, min(horizon - 1, round(horizon * f)))
                   for f in (.1, .25, .5, .75)}) if horizon > 1 else []


def _group_leaders(phase: dict) -> dict[str, str]:
    leaders = {}
    for group, members in phase["groups"].items():
        subset = {**phase, "agents": {n: phase["agents"][n] for n in members},
                  "contrasts": [c for c in phase["contrasts"] if c["a"] in members and c["b"] in members]}
        leaders[group] = next(iter(_rank(subset)), members[0])
    return leaders


def _build_phase(name: str, state: dict, base: MctsAgent, reference: MctsAgent | None) -> dict:
    """Frozen sequential screens within ONE evaluator family; all matches use equal time."""
    cal, phases, request = state["calibration"], state["phases"], state["request"]
    seconds, horizon = cal["decision_seconds"], cal["horizon"]["depth"]
    start = replace(_timed(base, seconds), rollout_depth=horizon, tree_reuse=False, transpositions=False,
                    selection_policy="uct", rollout_policy=UniformRandom(), progressive_bias=None)
    agents, contrasts, groups, priority = {}, [], {}, {}
    decisions = {}

    def add(label, agent, group="main"):
        agents[label] = profile_values(_timed(agent, seconds))
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

    if name == "depth_screen":
        depths = cal["cutoff_depths"] if request["mode"] == "heuristic_cutoff" else [horizon]
        for depth in depths:
            add(f"depth-{depth}", replace(start, rollout_depth=depth))
        names = list(agents)
        if len(names) > 1:
            control = names[len(names)//2]
            for candidate in names:
                if candidate != control:
                    compare(control, candidate, "rollout_depth")
    elif name == "exploration":
        screen = phases["depth_screen"]
        coverage = all(c.get("result", {}).get("seed_pairs", 0) >= 4 for c in screen["contrasts"])
        ordered = _rank(screen) or list(screen["agents"])
        shortlist = ordered[:2] if coverage else list(screen["agents"])
        # No early depth elimination on one minimal batch; scarce budgets keep
        # the depths rather than pretending that unmeasured depths lost.
        for winner in shortlist:
            parent = agent_from_values(screen["agents"][winner])
            group = f"d{parent.rollout_depth}"
            control = add(f"{group}-uct-start", parent, group)
            for c in sorted({.25, .5, 1., 1.4, 2., parent.exploration}):
                if c != parent.exploration:
                    candidate = add(f"{group}-uct-c{c:g}", replace(parent, exploration=c), group)
                    compare(control, candidate, "exploration")
    elif name == "selectors":
        for group, parent in carry("exploration").items():
            control = add(f"{group}-uct", parent, group)
            if "ucb1_tuned" in request["selection_policies"]:
                candidate = add(f"{group}-tuned", replace(parent, selection_policy="ucb1_tuned"), group)
                compare(control, candidate, "selection_policy")
    elif name == "rave":
        for group, parent in carry("exploration").items():
            if request.get("rave_search", False) and "uct_rave" in request["selection_policies"]:
                control = add(f"{group}-rave-k3000", replace(parent, selection_policy="uct_rave", rave_equivalence=3000), group)
                for k in (1000, 10000):
                    candidate = add(f"{group}-rave-k{k}", replace(parent, selection_policy="uct_rave", rave_equivalence=k), group)
                    compare(control, candidate, "rave_equivalence")
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
            complete = all(c.get("result", {}).get("seed_pairs", 0) >= max(2, c.get("target_pairs", 2)) for c in evidence)
            if not complete:
                decisions[group] = "insufficient_budget_or_evidence"
                continue
            # The first refinement always checks gaps. Later rounds require an
            # improving challenger, independently of the previous non-RAVE selector.
            improved = any(prior["agents"][c["b"]]["rave_equivalence"] == parent.rave_equivalence
                           and c["result"].get("score_b", 0) >= .55 for c in evidence)
            if previous != "rave" and not improved:
                decisions[group] = "no_clear_improvement_not_proven_plateau"
                continue
            tested = {v["rave_equivalence"] for stage in PHASES[PHASES.index("rave"):PHASES.index(name)]
                      for member in phases[stage]["groups"][group]
                      if (v := phases[stage]["agents"][member])["selection_policy"] == "uct_rave"}
            k = parent.rave_equivalence
            below = max((v for v in tested if v < k), default=None)
            above = min((v for v in tested if v > k), default=None)
            proposed = []
            if above is None and improved:
                proposed.append(min(2**32 - 1, k * 2))
            if below is None:
                proposed.append(max(1, k // 2))
            if below is not None:
                proposed.append(round(math.sqrt(below * k)))
            if above is not None:
                proposed.append(round(math.sqrt(k * above)))
            for candidate_k in dict.fromkeys(proposed):
                if candidate_k not in tested:
                    compare(control, add(f"{group}-rave-k{candidate_k}", replace(parent, rave_equivalence=candidate_k), group), "rave_equivalence")
            decisions[group] = "refining_geometric_neighbors" if any(c["a"] == control for c in contrasts) else "no_untested_neighbors"
    elif name == "rave_exploration":
        prior = phases["rave_extend_5"]
        for group, parent in carry("rave_extend_5").items():
            control = add(f"{group}-rave", parent, group)
            evidence = [c for c in prior["contrasts"] if c["b"] in prior["groups"][group]]
            ready = all(c.get("result", {}).get("seed_pairs", 0) >= max(2, c.get("target_pairs", 2)) for c in evidence)
            reason = prior.get("rave_decisions", {}).get(group, "unavailable")
            if not ready or reason in ("insufficient_budget_or_evidence", "unavailable", "disabled"):
                decisions[group] = "insufficient_budget_or_evidence" if not ready else reason
                continue
            decisions[group] = "extension_limit" if evidence else reason
            for factor in (.5, 2.):
                value = max(.01, parent.exploration * factor)
                compare(control, add(f"{group}-rave-c{value:g}", replace(parent, exploration=value), group), "exploration")
    elif name == "rave_compare":
        prior = phases["rave_exploration"]
        for group, parent in carry("selectors").items():
            control = add(f"{group}-selector", parent, group)
            evidence = [c for c in prior["contrasts"] if c["b"] in prior["groups"][group]]
            ready = bool(evidence) and all(c.get("result", {}).get("seed_pairs", 0) >= max(2, c.get("target_pairs", 2)) for c in evidence)
            if ready:
                candidate = add(f"{group}-calibrated-rave", carry("rave_exploration")[group], group)
                compare(control, candidate, "calibrated_rave_vs_selector")
                decisions[group] = "bounded_calibration_complete"
            else:
                decisions[group] = "rave_unavailable_or_calibration_incomplete"
    elif name == "family_selection":
        for group, parent in carry("rave_compare").items():
            add(f"tuned-{group}", parent)
        names = list(agents)
        for i, a in enumerate(names):
            for b in names[i+1:]:
                compare(a, b, "tuned_depth")
    elif name == "mechanisms":
        parent = next(iter(carry("family_selection").values()))
        for reuse, trans in product((False, True), repeat=2):
            add(f"r{int(reuse)}-t{int(trans)}", replace(parent, tree_reuse=reuse, transpositions=trans))
        names = list(agents)
        # Full 2x2 round robin: the joint cell also competes against each single mechanism.
        for i, a in enumerate(names):
            for b in names[i+1:]:
                compare(a, b, "mechanisms")
    elif name == "refinement":
        parent = next(iter(carry("mechanisms").values()))
        control = add("incumbent", parent)
        if parent.selection_policy in ("uct", "uct_rave"):
            for multiplier in (.7, 1.4):
                c = max(.01, parent.exploration * multiplier)
                compare(control, add(f"refine-c{c:g}", replace(parent, exploration=c)), "exploration")
        if parent.selection_policy == "uct_rave":
            for k in (max(1, parent.rave_equivalence // 2), parent.rave_equivalence * 2):
                compare(control, add(f"refine-k{k}", replace(parent, rave_equivalence=k)), "rave_equivalence")
        if request["mode"] == "heuristic_cutoff":
            for depth in sorted({max(1, min(horizon-1, round(parent.rollout_depth * f))) for f in (.75, 1.25)}):
                if depth != parent.rollout_depth:
                    compare(control, add(f"refine-d{depth}", replace(parent, rollout_depth=depth)), "rollout_depth")
    elif name == "confirmation":
        prior = phases["refinement"]
        ranked = _rank(prior) or list(prior["agents"])
        winner = agent_from_values(prior["agents"][ranked[0]])
        # Lock the nominee before fresh seeds; confirmation cannot select a new winner by noise.
        add("finalist", winner)
        alternatives = [agent_from_values(prior["agents"][n]) for n in ranked[1:]]
        alternatives.append(agent_from_values(next(iter(phases["depth_screen"]["agents"].values()))))
        challenger = next((a for a in alternatives if profile_values(a) != profile_values(winner)), None)
        if challenger:
            compare(add("runner-up", challenger), "finalist", "confirmation", primary=True)
        if reference:
            compare(add("reference", reference), "finalist", "reference", primary=not contrasts)
    else:
        raise ValueError(f"unknown study phase: {name}")
    return {"name": name, "agents": agents, "groups": groups, "contrasts": contrasts,
            "tie_priority": priority, "rave_decisions": decisions, "status": "pending"}


class StudyRunner:
    def __init__(self, game: str, baseline: MctsAgent, *, output: Path, budget: float,
                 reference: MctsAgent | None = None, seed: int = 42, max_pairs: int = 8,
                 confirmation_pairs: int = 32, auxiliary_pairs: int = 4,
                 decision_seconds: float | None = None, screening_seconds: float | None = None, max_plies: int = 10000,
                 heuristic: int | None = None, target_match_time: float = 60.0, rave_search: bool = False,
                 workers: WorkerSetting = 1, resume: bool = False, game_params: dict | None = None, progress: Callable[[str], None] = print):
        if game not in GAMES:
            raise ValueError("automatic studies require generic tournament transport; supported: " + ", ".join(GAMES))
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("study budget must be finite and positive")
        if type(max_pairs) is not int or not 2 <= max_pairs < SEED_STRIDE:
            raise ValueError("max_pairs must be between 2 and 99999")
        if type(confirmation_pairs) is not int or not 2 <= confirmation_pairs < SEED_STRIDE:
            raise ValueError("confirmation_pairs must be between 2 and 99999")
        if type(auxiliary_pairs) is not int or not 2 <= auxiliary_pairs < SEED_STRIDE:
            raise ValueError("auxiliary_pairs must be between 2 and 99999")
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
        evaluator = NeutralEvaluator() if heuristic is None else GameHeuristic(heuristic)
        baseline = replace(baseline, heuristic=None, cutoff_evaluator=evaluator,
                           selection_policy="uct", rollout_policy=UniformRandom(), progressive_bias=None,
                           tree_reuse=False, transpositions=False)
        if reference and (reference.cutoff_evaluator != evaluator or reference.rollout_policy != UniformRandom()
                          or reference.progressive_bias is not None):
            raise ValueError("reference must belong to the same evaluator family with uniform rollout and no bias; compare families in a separate tournament")
        selectors = game_search_capabilities(game)["selection_policies"]
        if reference and reference.selection_policy not in selectors:
            raise ValueError("reference selection policy is unavailable for this game")
        if type(rave_search) is not bool:
            raise ValueError("rave_search must be a boolean")
        if reference and reference.selection_policy == "uct_rave" and not rave_search:
            raise ValueError("a RAVE reference requires --rave-search")
        worker_count = resolve_workers(workers)
        self.game = create_game(game, game_params)
        self.base, self.reference = baseline, reference
        self.output, self.budget = output.resolve(), budget
        self.progress, self.resume = progress, resume
        request = {"version": 16, "rave_search": rave_search, "mode": "full_depth" if heuristic is None else "heuristic_cutoff",
                   "heuristic": heuristic, "target_match_time": target_match_time, "safety_margin": 1.2,
                   "selection_policies": selectors, "game": game, **({"game_params": game_parameters(self.game)} if game_parameters(self.game) else {}), "baseline": profile_values(baseline),
                   "reference": profile_values(reference) if reference else None,
                   "seed": seed, "max_pairs": max_pairs, "confirmation_pairs": confirmation_pairs, "auxiliary_pairs": auxiliary_pairs,
                   "decision_seconds": decision_seconds, "screening_seconds": screening_seconds,
                   "max_plies": max_plies, "workers": worker_count, "engine": _fingerprint()}
        self.path = self.output / "study.json"
        if self.path.exists():
            if not resume:
                raise FileExistsError(f"study exists: {self.path}; use --resume")
            self.state = json.loads(self.path.read_text())
            if self.state["request"].get("version") != 16:
                raise ValueError("old study protocol cannot resume with single-family tuning; use a new output directory")
            # Git revision is provenance, not an execution compatibility key.
            # Keep the original revision in the saved request.
            saved_request = self.state["request"]
            compatible_request = {**request, "engine": {
                **request["engine"], "git_revision": saved_request["engine"].get("git_revision")}}
            if saved_request != compatible_request:
                raise ValueError("study configuration or engine changed; use a new output directory")
        else:
            if self.output.exists() and any(self.output.iterdir()):
                raise FileExistsError("study output directory must be empty")
            self.output.mkdir(parents=True, exist_ok=True)
            self.state = {"request": request, "budget_seconds": budget, "spent_seconds": 0,
                          "calibration": None, "phases": {}, "status": "pending"}
            export_profile(self.output / "baseline.toml", "starting-baseline", baseline)
            if reference:
                export_profile(self.output / "reference.toml", "held-out-reference", reference)
        if budget < self.state.get("budget_seconds", 0):
            raise ValueError("resumed budget cannot be smaller than the original budget")
        self.state["budget_seconds"] = budget
        self.started = perf_counter()
        self.previous_spent = self.state["spent_seconds"]
        self.save()

    @property
    def spent(self):
        return self.previous_spent + perf_counter() - self.started

    def save(self):
        self.state["spent_seconds"] = self.spent
        self.state.update(study_diagnostics(self.state))
        _save(self.path, self.state)

    def _trace_config(self, phase: dict, index: int, *, pilot=False):
        contrast = phase["contrasts"][index]
        agents = tuple(TournamentAgent(n, RandomAgent() if phase["agents"][n] is None else agent_from_values(phase["agents"][n]))
                       for n in (contrast["a"], contrast["b"]))
        path = self.output / "traces" / f"{phase['name']}-{index:02d}.jsonl"
        phase_index = 0 if pilot else PHASES.index(phase["name"]) + 1
        return TournamentConfig(game=self.game, output=path, pairing_mode="round_robin", seat_mode="paired",
                                matches_per_pair=2 if pilot else 2 * contrast.get("target_pairs", phase["planned_pairs"]),
                                seed=self.state["request"]["seed"] + phase_index * SEED_STRIDE,
                                max_plies=self.state["request"]["max_plies"],
                                workers=self._contrast_workers(phase, index), agents=agents)

    def _contrast_workers(self, phase: dict, index: int) -> int:
        contrast = phase["contrasts"][index]
        # Anchor timings choose fast/balanced candidates, so must remain isolated.
        if phase["name"] == "calibration":
            return 1
        if any(phase["agents"][contrast[role]].get("time_budget") is not None for role in ("a", "b")):
            return 1
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
            if phase["name"] != "confirmation":
                contrast["result"]["verdict"] = "exploratory"
            contrast["trace"] = str(config.output.relative_to(self.output))
            rows_by_index[index] = rows
        self.save()
        return rows_by_index

    def _pair(self, phase: dict, index: int, round_index: int, *, pilot=False):
        return self._batch(phase, [index], round_index, pilot=pilot)[index]

    def calibrate(self):
        if self.state["calibration"]:
            return
        request = self.state["request"]
        label = "Full-depth optimization" if request["mode"] == "full_depth" else f"Heuristic cutoff optimization (H{request['heuristic']})"
        self.progress(f"Study mode: {label}. Target match time: {request['target_match_time']:g}s.")
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
        target = request["decision_seconds"] or request["target_match_time"] / (length * request["safety_margin"])
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
            if hard is not None or reached or self.spent >= self.budget * .08 or depth >= request["max_plies"]:
                kind = "hard_maximum" if hard is not None else "practical_full_depth" if reached else "practical_unverified"
                work["horizon"] = {"depth": depth, "kind": kind, "terminal_fraction": sample["terminal_fraction"],
                                   "minimum_position_terminal_fraction": sample["minimum_position_terminal_fraction"],
                                   "target_terminal_fraction": .99, "samples": samples}
                break
            depth = min(depth * 2, request["max_plies"])
        horizon = work["horizon"]
        depths = cutoff_depths(horizon["depth"]) if request["mode"] == "heuristic_cutoff" else []
        if request["mode"] == "heuristic_cutoff" and not depths:
            raise ValueError("the reference horizon leaves no distinct heuristic cutoff depth")
        if self.reference:
            if (request["mode"] == "full_depth" and self.reference.rollout_depth != horizon["depth"]) or (
                    request["mode"] == "heuristic_cutoff" and self.reference.rollout_depth >= horizon["depth"]):
                raise ValueError("reference horizon belongs to another family; compare it in a separate tournament")
        operating_depth = depths[len(depths)//2] if depths else horizon["depth"]
        if "position_timings" not in work:
            probe = replace(_timed(self.base, target), rollout_depth=operating_depth, root_diagnostics=True)
            bench = benchmark_mcts_agent(self.game, probe, round(length), request["seed"] + 47)
            work["position_timings"] = [asdict(t) for t in bench.position_timings]
            self.save()
        adequacy = search_adequacy(work["position_timings"], request["target_match_time"])
        if request["decision_seconds"] is not None:
            # Changing the match target cannot affect an explicit per-decision override.
            adequacy["suggested_targets"] = []
        center = max(1, round(adequacy["median_iterations_per_decision"]))
        cal = {"mean_plies": length, "estimated_game_decisions": length, "target_match_time": request["target_match_time"],
               "safety_margin": request["safety_margin"], "safety_adjusted_decisions": length * request["safety_margin"],
               "decision_seconds": target, "decision_time_source": "explicit_override" if request["decision_seconds"] else "target_match_time",
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
        self.progress(f"Horizon: {horizon['depth']} ({horizon['kind']}); estimated decisions: {length:.1f}; time/decision: {target:.4f}s.")
        self.progress(f"Search adequacy: {adequacy['category']}; median iterations: {center}; representative branching: {adequacy['representative_branching']:g}.")
        for warning in warnings:
            self.progress("Warning: " + warning)
        for recommendation in adequacy["suggested_targets"]:
            self.progress(f"Approximate target {recommendation['target_match_time']:.1f}s/match -> {recommendation['estimated_category']} (linear throughput estimate, not a guarantee).")
        self.save()

    def _estimated_round_seconds(self, phase: dict) -> float:
        cal = self.state["calibration"]
        return cal["mean_plies"] * sum(
            phase["agents"][c[role]]["time_budget"]
            for c in phase["contrasts"] for role in ("a", "b"))

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
                if phase["name"] != "confirmation":
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
                    budgets = sum(previous["agents"][c[role]].get("time_budget", 0) for role in ("a", "b"))
                    if budgets:
                        lengths.append(seconds / r["seed_pairs"] / budgets)
            if lengths:
                return 1.2 * median(lengths) * sum(phase["agents"][contrast[role]]["time_budget"] for role in ("a", "b"))
        return self._estimated_round_seconds({**phase, "contrasts": [contrast]})

    def _plan_rave(self, phase, weights):
        pool = self.state.get("rave_budget")
        if pool is None:
            entry = PHASES.index("rave")
            allocation = max(0, self.budget - self.spent) * weights[entry] / sum(weights[entry:])
            pool = self.state["rave_budget"] = {
                "allocated_seconds": allocation, "started_spent": self.spent,
                "interpretation": "Shared by all RAVE stages; future mandatory comparisons are reserved before optional rounds."}
            self.progress(f"RAVE shared budget: {allocation/60:.1f}min; minimum coverage first, then adaptive rounds.")
        spent = max(0, self.spent - pool["started_spent"])
        remaining = min(max(0, pool["allocated_seconds"] - spent), max(0, self.budget - self.spent))
        # Keep two seed pairs for each exploration contrast and final comparison;
        # the initial screen also protects the obligatory first geometric refinement.
        future_pairs = 10 if phase["name"] == "rave" else 2 if phase["name"] == "rave_exploration" else 0 if phase["name"] == "rave_compare" else 6
        reserve = 0.0
        for members in phase["groups"].values():
            control = members[0]
            reserve += future_pairs * self._cost_pair(phase, {"a": control, "b": control})
        available = max(0, remaining - reserve)
        phase.update(allocated_seconds=available, started_spent=self.spent,
                     shared_budget_remaining_seconds=remaining, reserved_followup_seconds=reserve)
        contrasts = phase["contrasts"]
        costs = [max(.001, self._cost_pair(phase, c)) for c in contrasts]
        minimum = 2 * sum(costs)
        # All initial/gap/exploration comparisons need coverage before any receives
        # extra samples. Never strand calibration by funding only one contrast.
        counts = [2 if minimum <= available + 1e-9 else 0 for _ in contrasts]
        left = available - sum(n*c for n,c in zip(counts, costs))
        cap = self.state["request"]["max_pairs"]
        if phase["name"] != "rave_compare":
            cap = min(cap, 4)  # Cheap sequential calibration; preserve optional exploration.
        if counts and all(counts):
            for _ in range(2, cap):
                for i, cost in enumerate(costs):
                    if left + 1e-9 >= cost:
                        counts[i] += 1
                        left -= cost
        for contrast, count in zip(contrasts, counts):
            contrast.update(target_pairs=count, workers=1, timing_mode="isolated")
            if not count:
                contrast.update(screening_status="not_prioritized", stop_reason="shared_rave_budget_insufficient_coverage")
        phase["planned_pairs"] = max(counts, default=0)
        phase["estimated_seconds"] = sum(n*c for n,c in zip(counts, costs))
        pool["spent_seconds_at_last_plan"] = spent

    def _plan_phase(self, phase, index):
        enabled = self.state["request"]["rave_search"] and "uct_rave" in self.state["request"]["selection_policies"]
        weights = [weight if enabled or not name.startswith("rave") else 0
                   for name, weight in zip(PHASES, PHASE_WEIGHTS)]
        if enabled and phase["name"].startswith("rave"):
            self._plan_rave(phase, weights)
            return
        available = max(0, self.budget - self.spent) * weights[index] / sum(weights[index:])
        request = self.state["request"]
        phase.update(allocated_seconds=available, started_spent=self.spent)
        estimate = sum(self._cost_pair(phase, c) for c in phase["contrasts"])
        cap = request["max_pairs"]
        pairs = min(cap, max(2, int(available / max(.001, estimate))))
        remaining = available
        # Round-robin groups when resources are scarce: one depth cannot consume
        # another depth's entire tuning allocation merely by insertion order.
        ordered = list(enumerate(phase["contrasts"]))
        if phase["name"] in ("exploration", "selectors") or phase["name"].startswith("rave"):
            group_of = {n: g for g, members in phase["groups"].items() for n in members}
            buckets = {}
            for item in ordered:
                buckets.setdefault(group_of[item[1]["b"]], []).append(item)
            ordered = [bucket[i] for i in range(max(map(len, buckets.values()), default=0))
                       for bucket in buckets.values() if i < len(bucket)]
        for ci, contrast in ordered:
            cost = max(.001, self._cost_pair(phase, contrast))
            limit = (request["confirmation_pairs"] if contrast.get("primary") else request["auxiliary_pairs"]) if phase["name"] == "confirmation" else pairs
            count = min(limit, int(remaining / cost))
            count = count if count >= 2 else 0
            contrast["target_pairs"] = count
            contrast["workers"] = self._contrast_workers(phase, ci)
            contrast["timing_mode"] = "isolated" if contrast["workers"] == 1 else "shared_cpu"
            if count == 0:
                contrast["screening_status"] = "not_prioritized"
                contrast["stop_reason"] = "phase_budget_not_evaluated"
            remaining -= count * cost
        phase["planned_pairs"] = max((c["target_pairs"] for c in phase["contrasts"]), default=0)
        phase["estimated_seconds"] = sum(c["target_pairs"] * self._cost_pair(phase, c) for c in phase["contrasts"])

    def _prioritize(self, phase, completed_round):
        # Complete initial coverage, then respect the phase allocation. No early
        # per-depth elimination: every shortlisted depth reaches selector/RAVE tuning.
        return

    def _announce_plan(self):
        requested = self.state["request"]["rave_search"]
        supported = "uct_rave" in self.state["request"]["selection_policies"]
        self.progress("RAVE search: " + ("enabled" if requested and supported else "unavailable for this game" if requested else "disabled (enable with --rave-search)"))
        self.progress("Equal-time comparisons; unused phase budget rolls forward; confirmation has a reserved allocation.")

    def run(self):
        try:
            if self.resume:
                self._recover()
            if all(self.state["phases"].get(name, {}).get("status") == "complete" for name in PHASES):
                self.state["status"] = "complete"
                self.state.pop("last_error", None)
                return self.state
            if self.spent >= self.budget:
                self.state["status"] = "budget_exhausted"
                return self.state
            self.calibrate()
            self._announce_plan()
            self.state.pop("last_error", None)
            self.state["status"] = "running"
            for index, name in enumerate(PHASES):
                phase = self.state["phases"].get(name)
                if phase and phase["status"] == "complete":
                    continue
                if self.spent >= self.budget:
                    break
                if phase is None:
                    phase = _build_phase(name, self.state, self.base, self.reference)
                    self._plan_phase(phase, index)
                    self.state["phases"][name] = phase
                    self.save()
                if not phase["contrasts"] or not phase["planned_pairs"]:
                    phase["status"] = "complete"
                    phase["completion_reason"] = "no_changed_parameters" if not phase["contrasts"] else "phase_budget_not_evaluated"
                    self.save()
                    continue
                phase["status"] = "running"
                self.progress(f"{name}: {len(phase['contrasts'])} contrasts, up to {phase['planned_pairs']} pairs; estimate {phase['estimated_seconds']:.0f}s; allocation {phase['allocated_seconds']:.0f}s.")
                for round_index in range(phase["planned_pairs"]):
                    for indices in self._batches(phase, round_index):
                        estimated_batch = sum(self._cost_pair(phase, phase["contrasts"][ci]) for ci in indices)
                        if self.spent + estimated_batch > self.budget:
                            phase["status"] = "budget_exhausted"
                            self.state["status"] = "budget_exhausted"
                            return self.state
                        # After the first two complete rounds, protect later phases.
                        if name in RACING_PHASES and round_index >= 2 and self.spent - phase["started_spent"] + estimated_batch > phase["allocated_seconds"]:
                            for c in phase["contrasts"]:
                                if c.get("result", {}).get("seed_pairs", 0) < c["target_pairs"]:
                                    c["screening_status"] = "not_prioritized"
                                    c["stop_reason"] = "phase_budget"
                            break
                        self._batch(phase, indices, round_index)
                    self._prioritize(phase, round_index + 1)
                    self.progress(f"{name}: paired round {round_index+1}/{phase['planned_pairs']} complete ({self.spent:.0f}s used).")
                phase["status"] = "complete"
                phase["completion_reason"] = "allocated_search_finished"
                self.save()
            complete = all(self.state["phases"].get(n, {}).get("status") == "complete" for n in PHASES)
            self.state["status"] = "complete" if complete else "budget_exhausted"
        except BaseException as error:
            self.state["status"] = "interrupted"
            self.state["last_error"] = str(error) or type(error).__name__
            raise
        finally:
            self.save()
            self.export_candidates()
            write_study_report(self.output, self.state)
        return self.state

    def export_candidates(self):
        profiles = {}
        for phase in self.state["phases"].values():
            if phase["status"] == "complete" and phase["name"] != "confirmation":
                for leader in _group_leaders(phase).values():
                    profiles[f"{phase['name']}-{leader}"] = phase["agents"][leader]
        confirmation = self.state["phases"].get("confirmation", {})
        if "finalist" in confirmation.get("agents", {}):
            label = "best_full_depth_agent" if self.state["request"]["mode"] == "full_depth" else "best_heuristic_cutoff_agent"
            profiles[label] = confirmation["agents"]["finalist"]
        directory = self.output / "candidates"
        for name, values in profiles.items():
            export_profile(directory / f"{name}.toml", name, agent_from_values(values))
        self.state["candidate_profiles"] = {name: str((directory / f"{name}.toml").relative_to(self.output)) for name in profiles}
        self.save()


def run_study(game: str, *, output: Path, budget: float, baseline: Path | None = None,
              reference: Path | None = None, **kwargs) -> dict:
    if game not in GAMES:
        raise ValueError("automatic studies require generic tournament transport; supported: " + ", ".join(GAMES))
    base = _load_mcts_profile(baseline).agent if baseline else generic_baseline(game)
    ref = _load_mcts_profile(reference).agent if reference else None
    return StudyRunner(game, base, output=output, budget=budget, reference=ref, **kwargs).run()
