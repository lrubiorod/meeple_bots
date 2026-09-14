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
from time import perf_counter
from typing import Callable

from . import _native
from ._concurrency import WorkerSetting, resolve_workers
from ._agent_config import (
    ConditionalRollout, EpsilonGreedy, GameHeuristic, Greedy, Mast, MctsAgent,
    NeutralEvaluator, RandomAgent, UniformRandom,
)
from ._capabilities import heuristic_indices
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
from .study_analysis import summarize_contrast, write_study_report, study_diagnostics, cutoff_screening

GAMES = {"connect6": Connect6, "boop": Boop, "spotf": SpiritsOfTheForest, "connect-four": ConnectFour,
         "tic-tac-toe": TicTacToe, "splendor": Splendor}
TERMINAL_SAFETY_DEPTH = 1024

PHASES = ("horizons", "horizon_check", "parameters", "iterations", "mechanisms",
          "refinement", "target_check", "confirmation", "ablations")
PHASE_WEIGHTS = (.12, .08, .14, .10, .10, .10, .10, .21, .05)
RACING_PHASES = {"horizons", "parameters", "mechanisms", "refinement"}
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
    indices = heuristic_indices(game)
    return MctsAgent(iterations=1000, rollout_depth=32, exploration=1.0,
                     cutoff_evaluator=GameHeuristic(indices[0]) if indices else NeutralEvaluator(),
                     rollout_policy=UniformRandom(), tree_reuse=False, transpositions=False)


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


def mechanism_plan(base: MctsAgent, seconds: float) -> tuple[dict, list[dict]]:
    agents, cells = {}, {}
    for selector, reuse, transpositions in product(("uct", "ucb1_tuned"), (False, True), (False, True)):
        name = f"{selector}-r{int(reuse)}-t{int(transpositions)}"
        cells[selector, reuse, transpositions] = name
        agents[name] = profile_values(replace(_timed(base, seconds), selection_policy=selector,
                                            tree_reuse=reuse, transpositions=transpositions))
    contrasts = []
    for cell, name in cells.items():
        for dimension, factor in enumerate(("selector", "tree_reuse", "transpositions")):
            if cell[dimension] not in ("uct", False):
                continue
            other = list(cell)
            other[dimension] = "ucb1_tuned" if dimension == 0 else True
            contrasts.append({"a": name, "b": cells[tuple(other)], "factor": factor})
    return agents, contrasts


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


def _width_candidates(phase: dict) -> dict[str, dict]:
    contrasts = [c for c in phase["contrasts"] if c["factor"] == "anchor"]
    if not contrasts:
        return {}
    def cost(contrast):
        return phase.get("isolated_costs", {}).get(contrast["b"], contrast["result"]["timing_b"]["mean_seconds"])
    ordered = sorted(contrasts, key=lambda c: (-c["result"]["score_b"], cost(c)))
    strong = ordered[0]
    close = [c for c in ordered if c["result"]["score_b"] >= strong["result"]["score_b"] - .05]
    balanced = min(close, key=cost)
    quick = min(ordered, key=cost)
    return {label: phase["agents"][contrast["b"]] for label, contrast in
            (("fast", quick), ("balanced", balanced), ("strong", strong))}


def _terminal_reference(agent: MctsAgent, seconds: float) -> MctsAgent:
    """Roll out toward terminal, with an explicit safety cap for unbounded games."""
    return replace(_timed(agent, seconds), rollout_depth=TERMINAL_SAFETY_DEPTH,
                   heuristic=None, cutoff_evaluator=NeutralEvaluator())


def _family(values: dict) -> str:
    return "full" if values["rollout_depth"] == TERMINAL_SAFETY_DEPTH else "cutoff"


def _leaders(phase: dict, count: int = 1) -> list[str]:
    ranking = [name for name in _rank(phase) if name != "anchor"]
    return [name for family in ("full", "cutoff")
            for name in [n for n in ranking if _family(phase["agents"][n]) == family][:count]]


def _horizon_survivors(phase: dict) -> list[str]:
    selection = cutoff_screening(phase)
    return ["full"] + ([selection["candidate"]] if selection["candidate"] else [])


def _build_phase(name: str, state: dict, base: MctsAgent, reference: MctsAgent | None) -> dict:
    target = state["calibration"]["decision_seconds"]
    seconds = target if name in ("horizon_check", "target_check", "confirmation", "ablations") else state["calibration"].get("screening_seconds", target)
    n = state["calibration"]["center_iterations"]
    fixed_profiles = {}
    tie_priority = {}
    contrasts = []
    if name == "horizons":
        parent = replace(base, tree_reuse=False, transpositions=False)
        agents = {"full": profile_values(_terminal_reference(parent, seconds))}
        evaluators = [NeutralEvaluator()]
        game = state.get("request", {}).get("game")
        if game is not None:
            registered = heuristic_indices(game)
            evaluators.extend(GameHeuristic(index) for index in (0, 1) if index in registered)
        if base.cutoff_evaluator not in evaluators:
            evaluators.append(base.cutoff_evaluator)
        for depth, evaluator in product((16, 32, 64), evaluators):
            label = "neutral" if isinstance(evaluator, NeutralEvaluator) else f"h{evaluator.index}"
            if isinstance(evaluator, GameHeuristic) and evaluator.params:
                label += "-custom"
            candidate = f"cutoff-d{depth}-{label}"
            agents[candidate] = profile_values(replace(_timed(parent, seconds), rollout_depth=depth,
                                                       heuristic=None, cutoff_evaluator=evaluator))
            tie_priority[candidate] = [int(evaluator != base.cutoff_evaluator), abs(depth - base.rollout_depth)]
            contrasts.append({"a": "full", "b": candidate, "factor": "cutoff_equal_time"})
    elif name == "horizon_check":
        horizon = state["phases"]["horizons"]
        agents = {"full": profile_values(_terminal_reference(base, target))}
        ordered = sorted(horizon["contrasts"], key=lambda c: (-c.get("result", {}).get("score_b", 0),
                         tuple(horizon.get("tie_priority", {}).get(c["b"], [1]))))
        for contrast in ordered[:2]:
            candidate = contrast["b"]
            tie_priority[candidate] = horizon.get("tie_priority", {}).get(candidate, [1])
            agents[candidate] = profile_values(_timed(agent_from_values(horizon["agents"][candidate]), target))
            contrasts.append({"a": "full", "b": candidate, "factor": "cutoff_equal_time"})
    elif name == "parameters":
        horizons = state["phases"].get("horizon_check", state["phases"]["horizons"])
        agents = {}
        for winner in _horizon_survivors(horizons):
            parent = _timed(agent_from_values(horizons["agents"][winner]), seconds)
            family = _family(horizons["agents"][winner])
            control = f"{family}-parent"
            agents[control] = profile_values(parent)
            tie_priority[control] = [0]
            for selector in ("uct", "ucb1_tuned"):
                exploration = sorted({base.exploration*f for f in (.25, .5, 1, 2)}) if selector == "uct" else [base.exploration]
                for c in exploration:
                    values = profile_values(replace(parent, selection_policy=selector, exploration=c))
                    if values == agents[control]:
                        continue
                    candidate = f"{family}-{selector}-c{c:g}"
                    agents[candidate] = values
                    tie_priority[candidate] = [int(selector != parent.selection_policy) + int(c != parent.exploration)]
                    contrasts.append({"a": control, "b": candidate, "factor": "parameters"})
    elif name == "iterations":
        parameters = state["phases"]["parameters"]
        agents = {"anchor": profile_values(_iterations(base, n))}
        for winner in _leaders(parameters):
            parent = agent_from_values(parameters["agents"][winner])
            family = _family(parameters["agents"][winner])
            center = state["calibration"].get("families", {}).get(family, {}).get("center_iterations", n)
            for count in sorted({max(1, min(2**32-1, round(center*f))) for f in (.25, .5, 1, 2, 4)}):
                candidate = f"{family}-i{count}"
                agents[candidate] = profile_values(_iterations(parent, count))
                contrasts.append({"a": "anchor", "b": candidate, "factor": "anchor"})
    elif name == "mechanisms":
        width = state["phases"]["iterations"]
        agents = {}
        for family in ("full", "cutoff"):
            subset = {**width, "contrasts": [c for c in width["contrasts"]
                      if c["factor"] == "anchor" and _family(width["agents"][c["b"]]) == family]}
            if not subset["contrasts"]:
                continue
            parent = agent_from_values(_width_candidates(subset)["balanced"])
            cube, _ = mechanism_plan(parent, seconds)
            for candidate, values in cube.items():
                if values["selection_policy"] == parent.selection_policy:
                    agents[f"{family}-{candidate}"] = profile_values(_iterations(agent_from_values(values), parent.iterations))
                    tie_priority[f"{family}-{candidate}"] = [int(values["tree_reuse"] != parent.tree_reuse) + int(values["transpositions"] != parent.transpositions)]
            prefix = f"{family}-{parent.selection_policy}"
            for reuse, transpositions in ((True, False), (False, True), (True, True)):
                factor = "combined_mechanisms" if reuse and transpositions else "tree_reuse" if reuse else "transpositions"
                contrasts.append({"a": prefix+"-r0-t0", "b": prefix+f"-r{int(reuse)}-t{int(transpositions)}", "factor": factor})
    elif name == "refinement":
        mechanisms = state["phases"]["mechanisms"]
        agents = {"anchor": profile_values(_timed(base, seconds))}
        for winner in _leaders(mechanisms):
            parent = agent_from_values(mechanisms["agents"][winner])
            control = f"{winner}-parent"
            agents[control] = profile_values(_timed(parent, seconds))
            fixed_profiles[control] = profile_values(parent)
            tie_priority[control] = [0]
            variants = set()
            if _family(mechanisms["agents"][winner]) == "cutoff":
                variants.update((max(1, min(TERMINAL_SAFETY_DEPTH-1, round(parent.rollout_depth*f))), parent.exploration) for f in (.75, 1.25))
            if parent.selection_policy == "uct":
                variants.update((parent.rollout_depth, parent.exploration*f) for f in (.5, 2))
            for depth, c in sorted(variants):
                candidate = f"{winner}-d{depth}-c{c:g}"
                tuned = replace(parent, rollout_depth=depth, exploration=c)
                agents[candidate] = profile_values(_timed(tuned, seconds))
                fixed_profiles[candidate] = profile_values(tuned)
                tie_priority[candidate] = [int(depth != parent.rollout_depth) + int(c != parent.exploration)]
                contrasts.append({"a": control, "b": candidate, "factor": "refinement"})
        # A family without local variants still needs a measured representative.
        for winner in _leaders(mechanisms):
            control = f"{winner}-parent"
            if not any(c["a"] == control for c in contrasts):
                contrasts.append({"a": "anchor", "b": control, "factor": "refinement"})
    elif name == "target_check":
        refined = state["phases"]["refinement"]
        agents = {"anchor": profile_values(_timed(base, target))}
        # Validate two alternatives per family at deployment time before final selection.
        for candidate in _leaders(refined, 2):
            agents[candidate] = profile_values(_timed(agent_from_values(refined["agents"][candidate]), target))
            fixed_profiles[candidate] = refined["fixed_profiles"][candidate]
            tie_priority[candidate] = refined.get("tie_priority", {}).get(candidate, [1])
            contrasts.append({"a": "anchor", "b": candidate, "factor": "target_time_check"})
    elif name == "confirmation":
        refined = state["phases"].get("target_check", state["phases"]["refinement"])
        agents = {"anchor": profile_values(_timed(base, target))}
        finalists = []
        for winner in _leaders(refined):
            family = _family(refined["agents"][winner])
            candidate = f"finalist-{family}"
            agents[candidate] = profile_values(_timed(agent_from_values(refined["agents"][winner]), target))
            finalists.append(candidate)
            fixed_profiles[f"tuned-{family}-balanced"] = refined["fixed_profiles"][winner]
        if len(finalists) == 2:
            contrasts.append({"a": "finalist-full", "b": "finalist-cutoff", "factor": "cutoff_confirmation", "primary": True})
        for candidate in finalists:
            contrasts.append({"a": "anchor", "b": candidate, "factor": "confirmation", "primary": len(finalists) == 1})
        if reference:
            agents["reference-time"] = profile_values(_timed(reference, target))
            for candidate in finalists:
                contrasts.append({"a": "reference-time", "b": candidate, "factor": "reference_equal_time"})
    elif name == "ablations":
        confirmation = state["phases"]["confirmation"]
        candidates = [key for key in confirmation["agents"] if key.startswith("finalist-")]
        comparison = next((c for c in confirmation["contrasts"] if c["factor"] == "cutoff_confirmation"), None)
        if comparison:
            verdict = comparison.get("result", {}).get("verdict")
            if verdict == "b_ahead":
                candidates = [comparison["b"]]
            elif verdict == "a_ahead":
                candidates = [comparison["a"]]
        if len(candidates) > 1:
            candidates = ["finalist-full"]  # Conservative provisional choice; not evidence of superiority.
        agents = {}
        for candidate in candidates:
            final = agent_from_values(confirmation["agents"][candidate])
            agents[candidate] = profile_values(_timed(final, seconds))
            family = _family(agents[candidate])
            horizons = state["phases"].get("horizon_check", state["phases"]["horizons"])
            origin = next(key for key in _horizon_survivors(horizons) if _family(horizons["agents"][key]) == family)
            parent = agent_from_values(horizons["agents"][origin])
            reversions = {"selection_policy": parent.selection_policy, "exploration": parent.exploration,
                          "tree_reuse": parent.tree_reuse, "transpositions": parent.transpositions,
                          "rollout_depth": parent.rollout_depth, "cutoff_evaluator": NeutralEvaluator()}
            for field, value in reversions.items():
                if getattr(final, field) == value or (field == "exploration" and final.selection_policy != "uct"):
                    continue
                reverted = f"{candidate}-revert-{field}"
                agents[reverted] = profile_values(replace(_timed(final, seconds), heuristic=None, **{field: value}))
                contrasts.append({"a": reverted, "b": candidate, "factor": field, "purpose": "ablation"})
            if final.tree_reuse and final.transpositions:
                reverted = f"{candidate}-revert-both-mechanisms"
                agents[reverted] = profile_values(replace(_timed(final, seconds), tree_reuse=False, transpositions=False))
                contrasts.append({"a": reverted, "b": candidate, "factor": "combined_mechanisms", "purpose": "ablation"})
    else:
        raise ValueError(f"unknown study phase: {name}")
    return {"name": name, "agents": agents, "contrasts": contrasts, "fixed_profiles": fixed_profiles, "tie_priority": tie_priority, "status": "pending"}

class StudyRunner:
    def __init__(self, game: str, baseline: MctsAgent, *, output: Path, budget: float,
                 reference: MctsAgent | None = None, seed: int = 42, max_pairs: int = 8,
                 confirmation_pairs: int = 32, auxiliary_pairs: int = 4,
                 decision_seconds: float | None = None, screening_seconds: float | None = None, max_plies: int = 10000,
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
        worker_count = resolve_workers(workers)
        self.game = create_game(game, game_params)
        self.base, self.reference = baseline, reference
        self.output, self.budget = output.resolve(), budget
        self.progress, self.resume = progress, resume
        request = {"version": 10, "game": game, **({"game_params": game_parameters(self.game)} if game_parameters(self.game) else {}), "baseline": profile_values(baseline),
                   "reference": profile_values(reference) if reference else None,
                   "seed": seed, "max_pairs": max_pairs, "confirmation_pairs": confirmation_pairs, "auxiliary_pairs": auxiliary_pairs,
                   "decision_seconds": decision_seconds, "screening_seconds": screening_seconds,
                   "max_plies": max_plies, "workers": worker_count, "engine": _fingerprint()}
        self.path = self.output / "study.json"
        if self.path.exists():
            if not resume:
                raise FileExistsError(f"study exists: {self.path}; use --resume")
            self.state = json.loads(self.path.read_text())
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
        self.progress("Calibration: short paired games, then isolated early/middle/late timings.")
        pilot = {"name": "calibration", "agents": {"pilot": profile_values(_iterations(self.base, min(self.base.iterations or 128, 128))),
                                                       "random": None},
                 "contrasts": [{"a": "pilot", "b": "random", "factor": "sanity"}]}
        # Native construction validates the input policies for this registered game
        # before an expensive screening run, including the held-out reference.
        for agent in (self.base, self.reference):
            if agent is not None:
                benchmark_mcts_agent(self.game, _iterations(agent, 1), 0, self.state["request"]["seed"])
        rows = self._pair(pilot, 0, 0, pilot=True)
        mean_plies = sum(r["result"]["plies"] for r in rows) / len(rows)
        bench_agent = _iterations(self.base, min(self.base.iterations or 256, 256))
        bench = benchmark_mcts_agent(self.game, bench_agent, max(1, round(mean_plies)), self.state["request"]["seed"])
        seconds_per_iteration = max(1e-9, bench.milliseconds_per_iteration / 1000)
        # Reserve enough runtime for several rounds across ~50 contrasts. Small
        # budgets imply cheap screening, not a promise of strong play.
        target = self.state["request"]["decision_seconds"] or min(1.0, max(.001, self.budget / (800 * max(1, mean_plies))))
        screen = self.state["request"]["screening_seconds"]
        if screen is None:
            screen = min(target / 4, max(.000001, self.budget * .4 / (300 * max(1, mean_plies))))
        self.state["calibration"] = {"mean_plies": mean_plies, "decision_seconds": target, "screening_seconds": screen,
                                     "center_iterations": max(1, min(2**32-1, round(screen / seconds_per_iteration))),
                                     "seconds_per_iteration": seconds_per_iteration,
                                     "position_timings": [asdict(t) for t in bench.position_timings],
                                     "pilot": pilot, "seconds": self.spent}
        self.progress(f"Estimated match length: {mean_plies:.0f} actions; screening: {screen:.4f}s/decision; final target: {target:.4f}s/decision; iteration center: {self.state['calibration']['center_iterations']}.")
        self.save()

    def calibrate_families(self):
        calibration = self.state["calibration"]
        families = calibration.setdefault("families", {})
        phase = self.state["phases"]["parameters"]
        for winner in _leaders(phase):
            values = phase["agents"][winner]
            family = _family(values)
            if family in families:
                continue
            if self.spent >= self.budget:
                return
            self.progress(f"Calibrating {family} iteration costs on early/middle/late positions.")
            bench = benchmark_mcts_agent(self.game, _iterations(agent_from_values(values), 8),
                                         max(1, round(calibration["mean_plies"])), self.state["request"]["seed"])
            cost = max(1e-9, bench.milliseconds_per_iteration / 1000)
            families[family] = {"candidate": winner, "profile": values,
                                "seconds_per_iteration": cost,
                                "center_iterations": max(1, min(2**32-1, round(calibration.get("screening_seconds", calibration["decision_seconds"]) / cost))),
                                "position_timings": [asdict(t) for t in bench.position_timings]}
            self.save()

    def _estimated_round_seconds(self, phase: dict) -> float:
        cal = self.state["calibration"]
        def cost(values):
            iteration_cost = cal["seconds_per_iteration"]
            for sample in cal.get("families", {}).values():
                if all(values.get(key) == sample["profile"].get(key)
                       for key in ("rollout_depth", "selection_policy", "exploration", "cutoff_evaluator")):
                    iteration_cost = sample["seconds_per_iteration"]
                    break
            return values.get("time_budget", values.get("iterations", 0) * iteration_cost)
        # Two swapped games: each agent takes approximately one full game's plies.
        return cal["mean_plies"] * sum(cost(phase["agents"][c["a"]]) + cost(phase["agents"][c["b"]]) for c in phase["contrasts"])

    def _recover(self):
        duration = 0.0
        phases = list(self.state["phases"].values())
        cal = self.state.get("calibration")
        if cal:
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
        if result.get("seed_pairs", 0):
            return sum(result.get(t, {}).get("total_seconds", 0) for t in ("timing_a", "timing_b")) / result["seed_pairs"]
        return self._estimated_round_seconds({**phase, "contrasts": [contrast]})

    def _plan_phase(self, phase, index):
        available = max(0, self.budget - self.spent) * PHASE_WEIGHTS[index] / sum(PHASE_WEIGHTS[index:])
        request = self.state["request"]
        phase.update(allocated_seconds=available, started_spent=self.spent)
        estimate = self._estimated_round_seconds(phase)
        cap = request["max_pairs"]
        if phase["name"] in ("horizon_check", "target_check", "ablations"):
            cap = min(cap, request["auxiliary_pairs"])
        pairs = min(cap, max(2, int(available / max(.001, estimate))))
        remaining = available
        for ci, contrast in enumerate(phase["contrasts"]):
            cost = max(.001, self._cost_pair(phase, contrast))
            if phase["name"] == "confirmation":
                limit = request["confirmation_pairs"] if contrast.get("primary") else min(request["max_pairs"], request["auxiliary_pairs"])
                # The primary contrast is first and receives the reserved budget first.
                count = min(limit, int(remaining / cost))
                count = max(2, count) if contrast.get("primary") else count if count >= 2 else 0
            elif phase["name"] == "ablations":
                count = min(pairs, int(remaining / cost))
                count = count if count >= 2 else 0
            else:
                count = pairs
            contrast["target_pairs"] = count
            contrast["workers"] = self._contrast_workers(phase, ci)
            contrast["timing_mode"] = "isolated" if contrast["workers"] == 1 else "shared_cpu"
            if count == 0:
                contrast["screening_status"] = "not_prioritized"
                contrast["stop_reason"] = "auxiliary_budget"
            remaining -= count * cost
        phase["planned_pairs"] = max((c["target_pairs"] for c in phase["contrasts"]), default=0)
        phase["estimated_seconds"] = sum(c["target_pairs"] * self._cost_pair(phase, c) for c in phase["contrasts"])

    def _prioritize(self, phase, completed_round):
        if phase["name"] not in RACING_PHASES or completed_round not in (2, 4):
            return
        if completed_round in phase.get("prioritized_rounds", []):
            return
        # This is resource allocation, not a hypothesis test or a proof of inferiority.
        for family in ("full", "cutoff"):
            candidates = [c for c in phase["contrasts"] if _family(phase["agents"][c["b"]]) == family
                          and c.get("screening_status") != "not_prioritized"]
            candidates.sort(key=lambda c: (-c.get("result", {}).get("score_b", 0),
                                           tuple(phase.get("tie_priority", {}).get(c["b"], [1]))))
            for contrast in candidates[2:]:
                contrast["screening_status"] = "not_prioritized"
                contrast["stop_reason"] = "exploratory_ranking"
        phase.setdefault("prioritized_rounds", []).append(completed_round)
        self.save()

    def _benchmark_widths(self, phase):
        costs = phase.setdefault("isolated_costs", {})
        samples = phase.setdefault("isolated_benchmarks", {})
        for name in dict.fromkeys(c["b"] for c in phase["contrasts"]):
            if name in costs:
                continue
            if self.spent >= self.budget:
                return False
            agent = agent_from_values(phase["agents"][name])
            # A small isolated sample estimates cost; contested match times never rank speed.
            bench = benchmark_mcts_agent(self.game, _iterations(agent, min(128, agent.iterations)),
                                         max(1, round(self.state["calibration"]["mean_plies"])),
                                         self.state["request"]["seed"])
            costs[name] = bench.milliseconds_per_iteration / 1000 * agent.iterations
            samples[name] = [asdict(t) for t in bench.position_timings]
            self.save()
        return True

    def _announce_plan(self):
        if "budget_plan" in self.state:
            return
        cal = self.state["calibration"]
        # Conservative upper count for two families. Later plans use actual survivors/costs.
        counts = (9, 2, 8, 10, 6, 6, 4, 3 + (2 if self.reference else 0), 4)
        estimates = []
        for i, (name, count) in enumerate(zip(PHASES, counts)):
            seconds = cal["decision_seconds"] if name in ("horizon_check", "target_check", "confirmation", "ablations") else cal.get("screening_seconds", cal["decision_seconds"])
            pairs = 2 * count
            if name == "confirmation":
                pairs = self.state["request"]["confirmation_pairs"] + (count-1)*self.state["request"]["auxiliary_pairs"]
            if name == "ablations":
                pairs = count*self.state["request"]["auxiliary_pairs"]
            estimated = 2 * cal["mean_plies"] * seconds * pairs
            estimates.append({"phase": name, "estimated_seconds": estimated,
                              "allocated_seconds": self.budget * PHASE_WEIGHTS[i]})
        total = sum(p["estimated_seconds"] for p in estimates)
        self.state["budget_plan"] = {"phases": estimates, "estimated_seconds": total,
                                      "fits_budget": total <= self.budget-self.spent,
                                      "interpretation": "Preliminary estimate before survivor selection; excludes optional racing rounds and does not assume linear worker scaling."}
        self.progress(f"Preliminary plan: {total/3600:.2f}h; budget {self.budget/3600:.2f}h. Final comparison gets a reserved allocation; auxiliary work is capped.")
        for estimate in estimates:
            self.progress(f"  {estimate['phase']}: estimate {estimate['estimated_seconds']/60:.1f}min; allocation {estimate['allocated_seconds']/60:.1f}min.")
        if total > self.budget-self.spent:
            self.progress("Requested evidence exceeds the estimated budget: sample sizes/auxiliary work will be reduced; the result may remain inconclusive.")
        self.save()

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
                    if name == "iterations":
                        self.calibrate_families()
                        if self.spent >= self.budget:
                            break
                    phase = _build_phase(name, self.state, self.base, self.reference)
                    self._plan_phase(phase, index)
                    self.state["phases"][name] = phase
                    self.save()
                if name == "iterations" and not self._benchmark_widths(phase):
                    break
                if not phase["contrasts"] or not phase["planned_pairs"]:
                    phase["status"] = "complete"
                    phase["completion_reason"] = "no_changed_parameters" if not phase["contrasts"] else "auxiliary_budget"
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
        directory = self.output / "candidates"
        profiles = {}
        width = self.state["phases"].get("iterations")
        if width and width["status"] == "complete":
            profiles.update(_width_candidates(width))
        horizons = self.state["phases"].get("horizons")
        if horizons and horizons["status"] == "complete":
            profiles.update({name: horizons["agents"][name] for name in _horizon_survivors(horizons)})
        params = self.state["phases"].get("target_check", self.state["phases"].get("refinement"))
        if params and params["status"] == "complete":
            for name in _leaders(params):
                profiles["parameter-" + _family(params["agents"][name])] = params["agents"][name]
        confirmation = self.state["phases"].get("confirmation")
        if confirmation:
            profiles.update(confirmation.get("fixed_profiles", {}))
            profiles.update({name: values for name, values in confirmation["agents"].items()
                             if name.startswith(("finalist", "tuned", "terminal-"))})
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
