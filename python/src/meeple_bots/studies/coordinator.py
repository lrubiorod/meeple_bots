"""Budgeted MCTS diagnosis built on the standard tournament executor and traces.

The registry still owns game capabilities. This coordinator never implements game rules.
Each contrast has its own standard trace so contrasts can share seeds without changing the
historical tournament seed schedule. Plans are frozen before their first match.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import json
import math
from pathlib import Path
import re
from time import perf_counter
from typing import Callable

from .._concurrency import WorkerSetting, resolve_workers
from .._agent_config import GameHeuristic, MctsAgent, NeutralEvaluator, RandomAgent, SoIsmctsAgent, UniformRandom
from .._capabilities import heuristic_indices, game_search_capabilities
from ..game_config import create_game, game_parameters
from ..connect6 import Connect6
from ..api import Boop, ConnectFour, SpiritsOfTheForest, TicTacToe, benchmark_mcts_agent
from ..splendor import Splendor
from ..lost_cities import LostCities
from .profiles import PROFILES
from .._search_profiles import resolve_family, load_search_profile
from ..tournaments import (
    TournamentAgent, TournamentConfig, TournamentTrace, match_jobs, run_matches,
    tournament_header,
)
from .tuners import config_fields, changes, assert_frozen, validate_tuner
from .report import write_study_report, study_diagnostics
from .race import _group_leaders, summarize_contrast
from .planning import (MAX_EXTENSION_ROUNDS, PHASES, STAGES, SEED_STRIDE,
    _pw_survivors, _build_phase,
    mcts_plan, so_specs, build_so_phase, budget_stage)
from .budget import _study_budget, estimated_round_seconds, cost_pair, plan_phase
from .calibration import calibrate_mcts, calibrate_so
from .report import compute_budget, announce_plan, extension_notice
from .persistence import (STUDY_VERSION, profile_values, agent_from_values, export_profile,
    export_candidate_profiles,
    _fingerprint, _save,
    load_checkpoint, recover_traces)

GAMES = {"lost_cities": LostCities,"connect6": Connect6, "boop": Boop, "spotf": SpiritsOfTheForest, "connect-four": ConnectFour,
         "tic-tac-toe": TicTacToe, "splendor": Splendor}


# Disjoint, fixed seed namespaces, including calibration. Never adapt seeds to results.


def duration_seconds(value: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(s|m|h)?", value.strip())
    if not match:
        raise ValueError("budget must be a positive duration, e.g. 90s, 20m or 2h")
    seconds = float(match[1]) * {None: 1, "s": 1, "m": 60, "h": 3600}[match[2]]
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("budget must be finite and positive")
    return seconds


def generic_baseline(game: str) -> MctsAgent:
    return MctsAgent(iterations=1000, rollout_depth=64, exploration=math.sqrt(2),
                     cutoff_evaluator=NeutralEvaluator(), rollout_policy=UniformRandom(),
                     tree_reuse=False, transpositions=False)


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
            specs = so_specs(self.profile, baseline, tune=tune, selection=selection_search,
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
            specs, self.phase_names = mcts_plan(self.family_profile,
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
        self.base = baseline
        self.output, self.budget = output.resolve(), budget
        self.progress, self.resume = progress, resume
        request = {"version": STUDY_VERSION, "max_extension_rounds": MAX_EXTENSION_ROUNDS, "execution_mode": "local_retune" if tune else "full_study", "minimum_evidence_pairs": 4, "tune": tune, "second_pass": second_pass, "widening_expansion_search": widening_expansion_search, "tuner_specs": specs, "phase_names": list(self.phase_names), "games_per_comparison": 2 * max_pairs, "stage_games": stage_games, "baseline_supplied": supplied, "selection_search": selection_search, "mechanism_search": mechanism_search, "depth_search": depth_search, "pw_search": pw_search, "pw_supported": pw_supported, "rave_search": rave_search, "mode": "full_depth" if heuristic is None else "heuristic_cutoff",
                   "heuristic": heuristic, "target_match_time": target_match_time, "safety_margin": 1.2,
                   "selection_policies": selectors, "game": game, **({"game_params": game_parameters(self.game)} if game_parameters(self.game) else {}), "baseline": profile_values(baseline),
                   "reference": None,
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
            self.state = load_checkpoint(self.path, request, self.output, allow_engine_change)
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
        self._prepare_snapshot()
        _save(self.path, self.state)

    def _prepare_snapshot(self):
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
        if self.state.get("calibration"):
            self.state["calibration"]["compute_budget"] = compute_budget(self.state)
        self.state["spent_seconds"] = self.spent
        self.state.update(study_diagnostics(self.state))
        self.state["random_baseline"] = self._random_summary()

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
            return calibrate_so(self.state, self.game, self.base, self.budget,
                                lambda: self.spent, self._batch, self.save, self.progress)
        return calibrate_mcts(self.state, self.game, self.base,
                              lambda: self.spent, self._pair, self.save, self.progress,
                              benchmark_mcts_agent)

    def _estimated_round_seconds(self, phase: dict) -> float:
        return estimated_round_seconds(phase, self.state["calibration"])

    def _cost_pair(self, phase, contrast):
        return cost_pair(phase, contrast, self.state["calibration"],
                         tuple(self.state["phases"].values()))

    def _plan_phase(self, phase, index):
        request = self.state["request"]
        remaining = max(0., self.budget-self.spent) if self.budget is not None else math.inf
        costs = [self._cost_pair(phase, contrast) for contrast in phase["contrasts"]]
        workers = [self._contrast_workers(phase, ci) for ci in range(len(phase["contrasts"]))]
        planned = plan_phase(phase, request, remaining, costs, workers, budget_stage(phase))
        phase.clear()
        phase.update(planned)
        dropped = phase["discarded_comparisons"]
        if dropped:
            self.progress(f"{phase['name']}: insufficient budget for {len(dropped)} challengers; retaining {len(phase['contrasts'])} with full evidence.")

    def _recover(self):
        # A process can terminate after flushing a match but before saving state.
        self.previous_spent = max(self.previous_spent, recover_traces(self.state, self._trace_config))

    def _announce_plan(self):
        announce_plan(self.state, self.family, self.family_profile,
                      self.phase_names, self.base, self.budget, self.progress)

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
                             build_so_phase(name, self.state, self.base) if self.profile else
                             _build_phase(name, self.state, self.base, None))
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
                message, limit_reached = extension_notice(self.state, phase)
                if limit_reached:
                    phase['extension_limit_reached'] = True
                if message:
                    self.progress(message)
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
                message, limit_reached = extension_notice(self.state, phase, completed=True)
                if limit_reached:
                    phase['extension_limit_reached'] = True
                if message:
                    self.progress(message)
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
        self.state["candidate_profiles"] = export_candidate_profiles(self.output, profiles)
        self.save()


def run_study(game: str, *, output: Path, budget: float | None = None, baseline: Path | None = None,
              reference: Path | None = None, **kwargs) -> dict:
    if game not in GAMES:
        raise ValueError("automatic studies require generic tournament transport; supported: " + ", ".join(GAMES))
    base = load_search_profile(baseline) if baseline else None
    ref = load_search_profile(reference) if reference else None
    return StudyRunner(game, base, output=output, budget=budget, reference=ref, **kwargs).run()
