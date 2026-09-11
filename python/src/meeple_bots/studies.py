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
from .api import Boop, ConnectFour, SpiritsOfTheForest, TicTacToe, benchmark_mcts_agent
from .serialization import _evaluator_dict
from .tournaments import (
    TournamentAgent, TournamentConfig, TournamentTrace, match_jobs, run_matches,
    tournament_header,
)
from .study_analysis import summarize_contrast, write_study_report

GAMES = {"boop": Boop, "spotf": SpiritsOfTheForest, "connect-four": ConnectFour,
         "tic-tac-toe": TicTacToe}
PHASES = ("mechanisms", "iterations", "parameters", "confirmation")
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
        result = contrast.get("result", {})
        if result.get("score_b") is not None:
            scores[contrast["a"]].append(1 - result["score_b"])
            scores[contrast["b"]].append(result["score_b"])
    return sorted((name for name, values in scores.items() if values),
                  key=lambda name: (-sum(scores[name]) / len(scores[name]), name))


def _rounds(phase: dict) -> int:
    return min((c.get("result", {}).get("seed_pairs", 0) for c in phase["contrasts"]), default=0)


def _width_candidates(phase: dict) -> dict[str, dict]:
    contrasts = [c for c in phase["contrasts"] if c["factor"] == "anchor"]
    ordered = sorted(contrasts, key=lambda c: (-c["result"]["score_b"], c["result"]["timing_b"]["mean_seconds"]))
    strong = ordered[0]
    close = [c for c in ordered if c["result"]["score_b"] >= strong["result"]["score_b"] - .05]
    balanced = min(close, key=lambda c: c["result"]["timing_b"]["mean_seconds"])
    quick = min(ordered, key=lambda c: c["result"]["timing_b"]["mean_seconds"])
    return {label: phase["agents"][contrast["b"]] for label, contrast in
            (("fast", quick), ("balanced", balanced), ("strong", strong))}


def _build_phase(name: str, state: dict, base: MctsAgent, reference: MctsAgent | None) -> dict:
    calibration = state["calibration"]
    seconds = calibration["decision_seconds"]
    n = calibration["center_iterations"]
    if name == "mechanisms":
        agents, contrasts = mechanism_plan(base, seconds)
    elif name == "iterations":
        mechanisms = state["phases"]["mechanisms"]
        finalists = _rank(mechanisms)[:2]
        agents = {"anchor": profile_values(_iterations(base, n))}
        contrasts = []
        for index, winner in enumerate(finalists):
            parent = agent_from_values(mechanisms["agents"][winner])
            previous = None
            for count in sorted({max(1, min(2**32-1, round(n * factor))) for factor in (.25, .5, 1, 2, 4)}):
                candidate = f"family{index}-i{count}"
                agents[candidate] = profile_values(_iterations(parent, count))
                contrasts.append({"a": "anchor", "b": candidate, "factor": "anchor"})
                if previous:
                    contrasts.append({"a": previous, "b": candidate, "factor": "iterations"})
                previous = candidate
        # All families use the same fixed anchor. The calibrated external reference is held out.
    elif name == "parameters":
        mechanisms = state["phases"]["mechanisms"]
        # Carry both selectors, avoiding exclusion of UCT before its exploration is tuned.
        ranking = _rank(mechanisms)
        winners = [next(x for x in ranking if mechanisms["agents"][x]["selection_policy"] == selector)
                   for selector in ("uct", "ucb1_tuned")]
        agents = {"anchor": profile_values(_timed(base, seconds))}
        contrasts = []
        for winner in winners:
            parent = agent_from_values(mechanisms["agents"][winner])
            depths = sorted({max(1, min(2**32-1, round(base.rollout_depth * f))) for f in (.5, 1, 2)})
            exploration = sorted({base.exploration, base.exploration / 4, base.exploration / 2,
                                  base.exploration * 2}) if parent.selection_policy == "uct" else [parent.exploration]
            # A small cross-product measures exploration/horizon interactions at equal time.
            for depth, c in product(depths, exploration):
                candidate = f"{parent.selection_policy}-d{depth}-c{c:g}"
                agents[candidate] = profile_values(replace(_timed(parent, seconds), rollout_depth=depth, exploration=c))
                contrasts.append({"a": "anchor", "b": candidate, "factor": "parameters"})
    else:
        parameters = state["phases"]["parameters"]
        ranked = [name for name in _rank(parameters) if name != "anchor"][:2]
        agents = {"anchor": profile_values(_timed(base, seconds))}
        contrasts = []
        for index, winner in enumerate(ranked):
            candidate = f"finalist{index}"
            agents[candidate] = parameters["agents"][winner]
            contrasts.append({"a": "anchor", "b": candidate, "factor": "confirmation"})
        if len(ranked) == 2:
            contrasts.append({"a": "finalist0", "b": "finalist1", "factor": "confirmation"})
        agents["anchor-iterations"] = profile_values(_iterations(base, n))
        # Confirm the exported iteration trade-offs as well as tuned time-limited finalists.
        unique_width = {}
        for label, values in _width_candidates(state["phases"]["iterations"]).items():
            key = json.dumps(values, sort_keys=True)
            if key in unique_width:
                continue
            candidate = "width-" + label
            unique_width[key] = candidate
            agents[candidate] = values
            contrasts.append({"a": "anchor-iterations", "b": candidate, "factor": "quality_confirmation"})
        # Iteration recommendations were measured before parameter tuning. Explicitly
        # test their combination with each tuned finalist instead of silently exporting
        # an unmeasured depth/exploration/iteration combination as a new baseline.
        balanced_count = _width_candidates(state["phases"]["iterations"])["balanced"]["iterations"]
        for index in range(len(ranked)):
            candidate = f"tuned{index}-balanced"
            agents[candidate] = profile_values(_iterations(agent_from_values(agents[f"finalist{index}"]), balanced_count))
            contrasts.append({"a": "anchor-iterations", "b": candidate, "factor": "combined_confirmation"})
        if reference:
            # Same-time matches test parameter choices; the original reference is also
            # measured at its actual iteration budget, explicitly labeled as unequal cost.
            agents["reference-time"] = profile_values(_timed(reference, seconds))
            agents["reference-original"] = profile_values(reference)
            for index in range(len(ranked)):
                contrasts.append({"a": "reference-time", "b": f"finalist{index}", "factor": "reference_equal_time"})
                contrasts.append({"a": "reference-original", "b": f"finalist{index}", "factor": "reference_original_budget"})
            for candidate in [*unique_width.values(), *(f"tuned{i}-balanced" for i in range(len(ranked)))]:
                contrasts.append({"a": "reference-original", "b": candidate, "factor": "reference_original_budget"})
    return {"name": name, "agents": agents, "contrasts": contrasts, "status": "pending"}


class StudyRunner:
    def __init__(self, game: str, baseline: MctsAgent, *, output: Path, budget: float,
                 reference: MctsAgent | None = None, seed: int = 42, max_pairs: int = 16,
                 decision_seconds: float | None = None, max_plies: int = 10000,
                 workers: WorkerSetting = 1, resume: bool = False, progress: Callable[[str], None] = print):
        if game not in GAMES:
            raise ValueError("automatic studies require generic tournament transport; supported: " + ", ".join(GAMES))
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("study budget must be finite and positive")
        if type(max_pairs) is not int or not 2 <= max_pairs < SEED_STRIDE:
            raise ValueError("max_pairs must be between 2 and 99999")
        if type(seed) is not int or not 0 <= seed < 2**64 - 5 * SEED_STRIDE:
            raise ValueError("seed leaves insufficient room for disjoint study phases")
        if type(max_plies) is not int or not 1 <= max_plies < 2**32:
            raise ValueError("max_plies must be a positive u32")
        if decision_seconds is not None and (not math.isfinite(decision_seconds) or decision_seconds <= 0):
            raise ValueError("decision time must be finite and positive")
        worker_count = resolve_workers(workers)
        self.game = GAMES[game]()
        self.base, self.reference = baseline, reference
        self.output, self.budget = output.resolve(), budget
        self.progress, self.resume = progress, resume
        request = {"version": 1, "game": game, "baseline": profile_values(baseline),
                   "reference": profile_values(reference) if reference else None,
                   "seed": seed, "max_pairs": max_pairs, "decision_seconds": decision_seconds,
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
        _save(self.path, self.state)

    def _trace_config(self, phase: dict, index: int, *, pilot=False):
        contrast = phase["contrasts"][index]
        agents = tuple(TournamentAgent(n, RandomAgent() if phase["agents"][n] is None else agent_from_values(phase["agents"][n]))
                       for n in (contrast["a"], contrast["b"]))
        path = self.output / "traces" / f"{phase['name']}-{index:02d}.jsonl"
        phase_index = 0 if pilot else PHASES.index(phase["name"]) + 1
        return TournamentConfig(game=self.game, output=path, pairing_mode="round_robin", seat_mode="paired",
                                matches_per_pair=2 if pilot else 2 * phase["planned_pairs"],
                                seed=self.state["request"]["seed"] + phase_index * SEED_STRIDE,
                                max_plies=self.state["request"]["max_plies"],
                                workers=self._contrast_workers(phase, index), agents=agents)

    def _contrast_workers(self, phase: dict, index: int) -> int:
        contrast = phase["contrasts"][index]
        # Anchor timings choose fast/balanced candidates, so must remain isolated.
        if phase["name"] == "calibration" or contrast["factor"] == "anchor":
            return 1
        if any(phase["agents"][contrast[role]].get("time_budget") is not None for role in ("a", "b")):
            return 1
        return self.state["request"]["workers"]

    def _batches(self, phase: dict, round_index: int):
        parallel = []
        for index, contrast in enumerate(phase["contrasts"]):
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
        target = self.state["request"]["decision_seconds"] or max(.001, self.budget / (800 * max(1, mean_plies)))
        self.state["calibration"] = {"mean_plies": mean_plies, "decision_seconds": target,
                                     "center_iterations": max(1, min(2**32-1, round(target / seconds_per_iteration))),
                                     "seconds_per_iteration": seconds_per_iteration,
                                     "position_timings": [asdict(t) for t in bench.position_timings],
                                     "pilot": pilot, "seconds": self.spent}
        self.progress(f"Estimated horizon: {mean_plies:.0f} actions; screening budget: {target:.4f}s/decision; iteration center: {self.state['calibration']['center_iterations']}.")
        self.save()

    def _estimated_round_seconds(self, phase: dict) -> float:
        cal = self.state["calibration"]
        def cost(values):
            return values.get("time_budget", values.get("iterations", 0) * cal["seconds_per_iteration"])
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
                config = self._trace_config(phase, index, pilot=phase["name"] == "calibration")
                if not config.output.exists():
                    if contrast.get("result", {}).get("games", 0):
                        raise ValueError(f"missing study trace: {config.output}")
                    continue
                with TournamentTrace(config.output, tournament_header(config, config.output, config.workers), resume=True):
                    pass
                rows = [json.loads(line) for line in config.output.read_text().splitlines()[1:]]
                contrast["result"] = summarize_contrast(rows)
                duration += sum(row["duration_seconds"] for row in rows) / config.workers
        # A process can terminate after flushing a match but before saving the state.
        # Never forget the elapsed time already represented by durable results.
        self.previous_spent = max(self.previous_spent, duration)

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
            self.state.pop("last_error", None)
            fractions = (.30, .30, .20, .20)
            for index, name in enumerate(PHASES):
                phase = self.state["phases"].get(name)
                if phase and phase["status"] == "complete":
                    continue
                if self.spent >= self.budget:
                    break
                # Stop at the first incomplete phase. Resumption never promotes a
                # partially sampled candidate pool or changes a frozen pairing plan.
                if phase is None:
                    phase = _build_phase(name, self.state, self.base, self.reference)
                    available = max(0, self.budget - self.spent) * fractions[index] / sum(fractions[index:])
                    estimate = self._estimated_round_seconds(phase)
                    phase["planned_pairs"] = max(2, min(self.state["request"]["max_pairs"], int(available / max(.001, estimate))))
                    phase["estimated_seconds"] = estimate * phase["planned_pairs"]
                    phase["allocated_seconds"] = available
                    for ci, contrast in enumerate(phase["contrasts"]):
                        contrast["workers"] = self._contrast_workers(phase, ci)
                        contrast["timing_mode"] = "isolated" if contrast["workers"] == 1 else "shared_cpu"
                    self.state["phases"][name] = phase
                    self.save()
                self.progress(f"{name}: up to {max(c['workers'] for c in phase['contrasts'])} workers; {len(phase['contrasts'])} contrasts × {phase['planned_pairs']} paired seeds; estimate {phase['estimated_seconds']:.0f}s.")
                for round_index in range(phase["planned_pairs"]):
                    for indices in self._batches(phase, round_index):
                        # Conservative admission: use summed pair costs, never assume
                        # linear CPU scaling. Already running batches finish together.
                        estimated_batch = 0.0
                        for ci in indices:
                            contrast = phase["contrasts"][ci]
                            estimate = self._estimated_round_seconds({**phase, "contrasts": [contrast]})
                            result = contrast.get("result", {})
                            if result.get("seed_pairs", 0):
                                estimate = (result["timing_a"]["total_seconds"] + result["timing_b"]["total_seconds"]) / result["seed_pairs"]
                            estimated_batch += estimate
                        if self.spent + estimated_batch > self.budget:
                            self.progress(f"{name}: insufficient remaining time for the next batch; resume with a larger total budget.")
                            phase["status"] = "budget_exhausted"
                            self.state["status"] = "budget_exhausted"
                            return self.state
                        self._batch(phase, indices, round_index)
                    if self.spent >= self.budget:
                        break
                    self.progress(f"{name}: paired round {round_index + 1}/{phase['planned_pairs']} complete ({self.spent:.0f}s used).")
                if _rounds(phase) < phase["planned_pairs"]:
                    phase["status"] = "budget_exhausted"
                    break
                phase["status"] = "complete"
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
        params = self.state["phases"].get("parameters")
        if params and params["status"] == "complete":
            winner = next(n for n in _rank(params) if n != "anchor")
            profiles["parameter-finalist"] = params["agents"][winner]
        confirmation = self.state["phases"].get("confirmation")
        if confirmation:
            profiles.update({name: values for name, values in confirmation["agents"].items()
                             if name.startswith(("finalist", "tuned"))})
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
