"""Study profile codecs, source identity, and atomic checkpoint I/O."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from .. import _native
from .._agent_config import (ConditionalRollout, EpsilonGreedy, Greedy,
    Mast, MctsAgent, SoIsmctsAgent, UniformRandom)
from .._mcts_profiles import (_configured_cutoff_evaluator, _configured_progressive_bias,
    _configured_rollout_policy)
from .._search_profiles import so_from_values
from ..serialization import _evaluator_dict

STUDY_VERSION = 23


FINGERPRINT_ALGORITHM = "python-tree-v2"


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


def export_candidate_profiles(output: Path, profiles: dict[str, dict]) -> dict[str, str]:
    directory = output / "candidates"
    for name, values in profiles.items():
        export_profile(directory / f"{name}.toml", name, agent_from_values(values))
    return {name: str((directory / f"{name}.toml").relative_to(output)) for name in profiles}


def _package_source_root() -> Path:
    return Path(__file__).parent.parent


def _python_source_entries(package: Path) -> list[tuple[str, bytes]]:
    """Production modules only, in checkout-independent relative-path order."""
    ignored = {"__pycache__", "build", "dist", "target", ".venv", "tests", "docs", "local", "results"}
    paths = (path for path in package.rglob("*.py")
             if path.is_file() and not ignored.intersection(path.relative_to(package).parts))
    return [(path.relative_to(package).as_posix(), path.read_bytes())
            for path in sorted(paths, key=lambda path: path.relative_to(package).as_posix())]


def _python_tree_digest(package: Path) -> str:
    digest = sha256()
    digest.update(FINGERPRINT_ALGORITHM.encode("ascii") + b"\0")
    for relative_path, contents in _python_source_entries(package):
        name = relative_path.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _legacy_python_digest(package: Path) -> str:
    """Exact historical root-only algorithm for unversioned checkpoints."""
    sources = sorted(package.glob("*.py"))
    return sha256(b"".join(path.name.encode() + path.read_bytes() for path in sources)).hexdigest()


def _fingerprint() -> dict:
    package = _package_source_root()
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package, capture_output=True,
                                  text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"fingerprint_algorithm": FINGERPRINT_ALGORITHM, "git_revision": revision,
            "native_sha256": sha256(Path(_native.__file__).read_bytes()).hexdigest(),
            "python_sha256": _python_tree_digest(package)}


def _engine_change_reason(saved: dict, current: dict) -> str | None:
    algorithm = saved.get("fingerprint_algorithm", "legacy-root-v1")
    if algorithm == "legacy-root-v1":
        expected_python = _legacy_python_digest(_package_source_root())
    elif algorithm == FINGERPRINT_ALGORITHM:
        expected_python = current["python_sha256"]
    else:
        raise ValueError(f"unsupported study engine fingerprint algorithm: {algorithm}")
    if saved.get("python_sha256") != expected_python:
        return "Python source fingerprint differs"
    if saved.get("native_sha256") != current["native_sha256"]:
        return "native engine fingerprint differs"
    return None


def _save(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def load_checkpoint(path: Path, request: dict, output: Path, allow_engine_change: bool) -> dict:
    state = json.loads(path.read_text())
    if state["request"].get("version") != STUDY_VERSION:
        raise ValueError("old study protocol / extension plan cannot resume with the 3-extension limit; existing checkpoint is unchanged; use a new output directory")
    saved_request = state["request"]
    # Permission to change binaries is never permission to change the frozen plan.
    if {k: v for k, v in saved_request.items() if k != "engine"} != {k: v for k, v in request.items() if k != "engine"}:
        raise ValueError("study configuration or engine changed: configuration differs; use a new output directory")
    previous_engine = state.get("active_engine", saved_request["engine"])
    current_engine = request["engine"]
    # Legacy checkpoints retain their historical root-only comparison.
    engine_change = _engine_change_reason(previous_engine, current_engine)
    if engine_change:
        if not allow_engine_change:
            raise ValueError(f"study engine/source fingerprint changed: {engine_change}; restore the original environment or explicitly use --resume --allow-engine-change to record a mixed-engine continuation")
        completed = {}
        for trace in sorted((output / "traces").glob("*.jsonl")):
            with trace.open() as handle:
                next(handle)  # Standard tournament header.
                completed[str(trace.relative_to(output))] = [json.loads(line)["match_number"] for line in handle]
        state.setdefault("engine_changes", []).append({
            "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
            "previous_engine": previous_engine, "new_engine": current_engine,
            "completed_matches_before_change": completed,
        })
        state["active_engine"] = current_engine
        state["mixed_engines"] = True
    return state


def recover_traces(state: dict, trace_config) -> float:
    """Recover durable paired results before the coordinator resumes scheduling."""
    from .race import summarize_contrast
    from ..tournaments import TournamentTrace, tournament_header

    duration = 0.0
    phases = list(state["phases"].values())
    cal = state.get("calibration") or state.get("calibration_progress")
    if cal and "pilot" in cal:
        phases = [cal["pilot"], *phases]
    for phase in phases:
        for index, contrast in enumerate(phase["contrasts"]):
            if contrast.get("target_pairs") == 0:
                continue
            config = trace_config(phase, index, pilot=phase["name"] == "calibration")
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
    return duration
