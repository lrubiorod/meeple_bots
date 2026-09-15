"""Read TOML and inline MCTS profiles independently of CLI execution."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from ._agent_config import (
    ConditionalRollout, EpsilonGreedy, GameHeuristic, Greedy, Mast, MctsAgent,
    NeutralEvaluator, ProgressiveBias, TurnPhaseIs, UniformRandom,
)
from .serialization import _evaluator_heuristic_index, _rollout_policy_evaluator


def sqrt_two() -> float:
    return 2.0**0.5


@dataclass(frozen=True, slots=True)
class _MctsProfile:
    name: str
    agent: MctsAgent


def _mcts_budget_kwargs(
    values: dict[str, object],
    context: str,
) -> dict[str, object]:
    has_iterations = "iterations" in values
    has_time = "time_budget" in values
    if has_iterations == has_time:
        requirement = "exactly one of iterations or time_budget"
        raise ValueError(f"{context} must define {requirement}")
    if has_iterations:
        return {"iterations": values["iterations"]}
    return {"time_budget": values["time_budget"]}


def _load_mcts_profile(path: Path) -> _MctsProfile:
    with path.open("rb") as profile_file:
        values = tomllib.load(profile_file)
    allowed = {
        "name",
        "iterations",
        "time_budget",
        "exploration",
        "selection_policy",
        "rave_equivalence",
        "progressive_widening",
        "progressive_widening_k",
        "progressive_widening_alpha",
        "rollout_depth",
        "use_heuristic",
        "heuristic_index",
        "cutoff_evaluator",
        "rollout_policy",
        "rollout_evaluator",
        "rollout_use_heuristic",
        "rollout_heuristic_index",
        "rollout_epsilon",
        "progressive_bias",
        "root_diagnostics",
        "tree_reuse",
        "transpositions",
    }
    unknown = sorted(values.keys() - allowed)
    if unknown:
        raise ValueError(f"unknown MCTS profile fields: {', '.join(unknown)}")
    missing = sorted({"rollout_depth"} - values.keys())
    if missing:
        raise ValueError(f"missing MCTS profile fields: {', '.join(missing)}")
    budget = _mcts_budget_kwargs(values, "MCTS profile")

    name = values.get("name", path.stem)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("MCTS profile name must be a non-empty string")
    return _MctsProfile(
        name=name.strip(),
        agent=MctsAgent(
            **budget,
            selection_policy=values.get("selection_policy", "uct"),
            rave_equivalence=values.get("rave_equivalence", 1000),
            progressive_widening=values.get("progressive_widening", False),
            progressive_widening_k=values.get("progressive_widening_k", 1.5),
            progressive_widening_alpha=values.get("progressive_widening_alpha", 0.5),
            exploration=values.get("exploration", sqrt_two()),
            rollout_depth=values["rollout_depth"],
            cutoff_evaluator=_configured_cutoff_evaluator(values, "MCTS profile"),
            rollout_policy=_configured_rollout_policy(values, "MCTS profile"),
            progressive_bias=_configured_progressive_bias(values, "MCTS profile"),
            root_diagnostics=_configured_root_diagnostics(values, "MCTS profile"),
            tree_reuse=_configured_tree_reuse(values, "MCTS profile"),
            transpositions=_configured_transpositions(values, "MCTS profile"),
        ),
    )


def _parse_inline_mcts_profile(spec: str) -> _MctsProfile:
    values: dict[str, str] = {}
    aliases = {
        "name": "name",
        "i": "iterations",
        "iterations": "iterations",
        "t": "time_budget",
        "time": "time_budget",
        "time_budget": "time_budget",
        "d": "rollout_depth",
        "depth": "rollout_depth",
        "rollout_depth": "rollout_depth",
        "c": "exploration",
        "exploration": "exploration",
        "selection_policy": "selection_policy",
        "rave_equivalence": "rave_equivalence",
        "progressive_widening": "progressive_widening",
        "progressive_widening_k": "progressive_widening_k",
        "progressive_widening_alpha": "progressive_widening_alpha",
        "h": "heuristic",
        "heuristic": "heuristic",
        "ce": "cutoff_evaluator",
        "cutoff": "cutoff_evaluator",
        "cutoff_evaluator": "cutoff_evaluator",
        "p": "rollout_policy",
        "policy": "rollout_policy",
        "rollout_policy": "rollout_policy",
        "e": "rollout_epsilon",
        "epsilon": "rollout_epsilon",
        "rollout_epsilon": "rollout_epsilon",
        "rh": "rollout_heuristic",
        "rollout_h": "rollout_heuristic",
        "rollout_heuristic": "rollout_heuristic",
        "phase": "rollout_phase",
        "scope": "rollout_phase",
        "rollout_phase": "rollout_phase",
        "pb": "progressive_bias_weight",
        "progressive_bias": "progressive_bias_weight",
        "progressive_bias_weight": "progressive_bias_weight",
        "pbh": "progressive_bias_heuristic",
        "progressive_bias_heuristic": "progressive_bias_heuristic",
        "pbphase": "progressive_bias_phase",
        "progressive_bias_phase": "progressive_bias_phase",
        "rd": "root_diagnostics",
        "root_diagnostics": "root_diagnostics",
        "tr": "tree_reuse",
        "tree_reuse": "tree_reuse",
        "tp": "transpositions",
        "transpositions": "transpositions",
    }
    if spec.strip():
        for raw_field in spec.split(","):
            field = raw_field.strip()
            key, separator, value = field.partition("=")
            key = key.strip().lower()
            value = value.strip()
            if not separator or not key or not value:
                raise ValueError(
                    "inline analyze agents must use comma-separated key=value fields"
                )
            canonical = aliases.get(key)
            if canonical is None:
                raise ValueError(f"unknown inline analyze agent field: {key}")
            if canonical in values:
                raise ValueError(f"duplicate inline analyze agent field: {key}")
            values[canonical] = value

    if "iterations" in values and "time_budget" in values:
        raise ValueError("inline analyze agent cannot combine iterations with time_budget")
    iterations = (
        _inline_agent_integer(values["iterations"], "iterations")
        if "iterations" in values
        else None if "time_budget" in values else 1_000
    )
    if "time_budget" in values:
        try:
            time_budget = float(values["time_budget"])
        except ValueError as error:
            raise ValueError("inline analyze agent time budget must be a number") from error
    else:
        time_budget = None
    rollout_depth = _inline_agent_integer(
        values.get("rollout_depth", "16"),
        "depth",
    )
    exploration_text = values.get("exploration", str(sqrt_two()))
    try:
        exploration = float(exploration_text)
    except ValueError as error:
        raise ValueError("inline analyze agent exploration must be a number") from error
    heuristic_text = values.get("heuristic", "none")
    heuristic = (
        None
        if heuristic_text.lower() == "none"
        else _inline_agent_integer(heuristic_text, "heuristic")
    )
    if "cutoff_evaluator" in values and "heuristic" in values:
        raise ValueError("inline analyze agent cannot combine h with ce/cutoff")
    cutoff_evaluator = (
        _inline_evaluator(values["cutoff_evaluator"], "cutoff")
        if "cutoff_evaluator" in values
        else (NeutralEvaluator() if heuristic is None else GameHeuristic(heuristic))
    )
    policy_name = values.get("rollout_policy", "uniform_random").lower()
    policy_values: dict[str, object] = {"rollout_policy": policy_name}
    if "rollout_epsilon" in values:
        try:
            policy_values["rollout_epsilon"] = float(values["rollout_epsilon"])
        except ValueError as error:
            raise ValueError("inline analyze agent rollout epsilon must be a number") from error
    if "rollout_heuristic" in values:
        policy_values["rollout_heuristic_index"] = _inline_agent_integer(
            values["rollout_heuristic"],
            "rollout heuristic",
        )
    rollout_policy = _configured_rollout_policy(
        policy_values,
        "inline analyze agent",
    )
    if "rollout_phase" in values:
        rollout_policy = ConditionalRollout(
            condition=TurnPhaseIs(values["rollout_phase"]),
            primary=rollout_policy,
            fallback=UniformRandom(),
        )
    progressive_fields = {
        "progressive_bias_weight",
        "progressive_bias_heuristic",
        "progressive_bias_phase",
    }
    progressive_bias = None
    if progressive_fields & values.keys():
        if "progressive_bias_weight" not in values:
            raise ValueError("inline progressive bias requires pb=<weight>")
        if "progressive_bias_heuristic" not in values:
            raise ValueError("inline progressive bias requires pbh=<heuristic index>")
        try:
            bias_weight = float(values["progressive_bias_weight"])
        except ValueError as error:
            raise ValueError("inline progressive bias weight must be a number") from error
        progressive_bias = ProgressiveBias(
            bias_weight,
            GameHeuristic(
                _inline_agent_integer(
                    values["progressive_bias_heuristic"],
                    "progressive bias heuristic",
                )
            ),
            (
                TurnPhaseIs(values["progressive_bias_phase"])
                if "progressive_bias_phase" in values
                else None
            ),
        )
    root_diagnostics_text = values.get("root_diagnostics", "false").lower()
    if root_diagnostics_text not in {"true", "false"}:
        raise ValueError("inline root_diagnostics must be true or false")
    tree_reuse_text = values.get("tree_reuse", "false").lower()
    if tree_reuse_text not in {"true", "false"}:
        raise ValueError("inline tree_reuse must be true or false")
    transpositions_text = values.get("transpositions", "false").lower()
    if transpositions_text not in {"true", "false"}:
        raise ValueError("inline transpositions must be true or false")
    pw_text = values.get("progressive_widening", "false").lower()
    if pw_text not in {"true", "false"}:
        raise ValueError("inline progressive_widening must be true or false")
    agent = MctsAgent(
        iterations=iterations,
        time_budget=time_budget,
        selection_policy=values.get("selection_policy", "uct"),
        progressive_widening=pw_text == "true",
        progressive_widening_k=float(values.get("progressive_widening_k", 1.5)),
        progressive_widening_alpha=float(values.get("progressive_widening_alpha", 0.5)),
        rave_equivalence=_inline_agent_integer(str(values.get("rave_equivalence", 1000)), "rave_equivalence"),
        exploration=exploration,
        rollout_depth=rollout_depth,
        cutoff_evaluator=cutoff_evaluator,
        rollout_policy=rollout_policy,
        progressive_bias=progressive_bias,
        root_diagnostics=root_diagnostics_text == "true",
        tree_reuse=tree_reuse_text == "true",
        transpositions=transpositions_text == "true",
    )
    name = values.get("name")
    if name is not None and not name.strip():
        raise ValueError("inline analyze agent name must be non-empty")
    return _MctsProfile(
        name=name.strip() if name is not None else _inline_agent_name(agent),
        agent=agent,
    )


def _inline_agent_integer(value: str, field: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"inline analyze agent {field} must be an integer") from error


def _inline_evaluator(value: str, field: str) -> NeutralEvaluator | GameHeuristic:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"neutral", "none"}:
        return NeutralEvaluator()
    if normalized.startswith("h"):
        return GameHeuristic(_inline_agent_integer(normalized[1:], field))
    raise ValueError(f"inline analyze agent {field} evaluator must be neutral or hINDEX")


def _inline_agent_name(agent: MctsAgent) -> str:
    parts = ["mcts"]
    if agent.heuristic is not None:
        parts.append(f"h{agent.heuristic}")
    parts.append(
        f"i{agent.iterations}"
        if agent.iterations is not None
        else f"t{agent.time_budget}"
    )
    parts.append(f"d{agent.rollout_depth}")
    if agent.selection_policy != "uct":
        parts.append(agent.selection_policy)
    if agent.selection_policy == "uct_rave":
        parts.append(f"k{agent.rave_equivalence}")
    if agent.progressive_widening:
        parts.append(f"pw-k{agent.progressive_widening_k}-a{agent.progressive_widening_alpha}")
    if agent.exploration != sqrt_two():
        parts.append(f"c{agent.exploration}")
    rollout_heuristic = _evaluator_heuristic_index(
        _rollout_policy_evaluator(agent.rollout_policy)
    )
    primary_policy = (
        agent.rollout_policy.primary
        if isinstance(agent.rollout_policy, ConditionalRollout)
        else agent.rollout_policy
    )
    if isinstance(primary_policy, EpsilonGreedy):
        parts.extend(("p-epsilon-greedy", f"e{primary_policy.epsilon}"))
    elif isinstance(primary_policy, Greedy):
        parts.append("p-greedy")
    if isinstance(agent.rollout_policy, ConditionalRollout):
        parts.append(f"phase-{agent.rollout_policy.condition.phase}")
    if rollout_heuristic is not None:
        parts.append(f"rh{rollout_heuristic}")
    if agent.progressive_bias is not None:
        parts.append(f"pb{agent.progressive_bias.weight}")
        bias_heuristic = _evaluator_heuristic_index(
            agent.progressive_bias.evaluator
        )
        if bias_heuristic is not None:
            parts.append(f"pbh{bias_heuristic}")
        if agent.progressive_bias.condition is not None:
            parts.append(f"pbphase-{agent.progressive_bias.condition.phase}")
    if agent.tree_reuse:
        parts.append("reuse")
    if agent.transpositions:
        parts.append("transpositions")
    return "-".join(parts)


def _configured_cutoff_evaluator(
    values: dict[str, object],
    context: str,
) -> NeutralEvaluator | GameHeuristic:
    if "cutoff_evaluator" in values:
        legacy = {"use_heuristic", "heuristic_index"} & values.keys()
        if legacy:
            raise ValueError(
                f"{context} cutoff_evaluator cannot be combined with: "
                + ", ".join(sorted(legacy))
            )
        return _configured_evaluator(values["cutoff_evaluator"], f"{context} cutoff")

    use_heuristic = values.get("use_heuristic", False)
    if not isinstance(use_heuristic, bool):
        raise TypeError(f"{context} use_heuristic must be a boolean")
    heuristic_index = values.get("heuristic_index", 0)
    if isinstance(heuristic_index, bool) or not isinstance(heuristic_index, int):
        raise TypeError(f"{context} heuristic_index must be an integer")
    return GameHeuristic(heuristic_index) if use_heuristic else NeutralEvaluator()


def _configured_evaluator(
    value: object,
    context: str,
) -> NeutralEvaluator | GameHeuristic:
    if isinstance(value, str):
        kind = value
        fields: dict[str, object] = {}
    elif isinstance(value, dict):
        unknown = sorted(value.keys() - {"kind", "index", "params"})
        if unknown:
            raise ValueError(f"unknown {context} evaluator fields: {', '.join(unknown)}")
        kind = value.get("kind")
        fields = value
    else:
        raise TypeError(f"{context} evaluator must be a string or TOML table")

    if not isinstance(kind, str):
        raise TypeError(f"{context} evaluator kind must be a string")
    normalized = kind.strip().lower().replace("-", "_")
    if normalized == "neutral":
        unsupported = sorted({"index", "params"} & fields.keys())
        if unsupported:
            raise ValueError(
                f"{context} neutral evaluator cannot use: {', '.join(unsupported)}"
            )
        return NeutralEvaluator()
    if normalized not in {"game_heuristic", "heuristic"}:
        raise ValueError(f"{context} evaluator kind must be neutral or game_heuristic")
    index = fields.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError(f"{context} game_heuristic evaluator requires an integer index")
    params = fields.get("params", {})
    if not isinstance(params, dict):
        raise TypeError(f"{context} game_heuristic params must be a TOML table")
    return GameHeuristic(index, params)


def _configured_progressive_bias(
    values: dict[str, object],
    context: str,
) -> ProgressiveBias | None:
    raw = values.get("progressive_bias")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError(f"{context} progressive_bias must be a TOML table")
    unknown = sorted(raw.keys() - {"weight", "evaluator", "condition"})
    if unknown:
        raise ValueError(
            f"unknown {context} progressive_bias fields: {', '.join(unknown)}"
        )
    weight = raw.get("weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise TypeError(f"{context} progressive_bias requires a numeric weight")
    if "evaluator" not in raw:
        raise ValueError(f"{context} progressive_bias requires an evaluator")
    evaluator = _configured_evaluator(
        raw["evaluator"], f"{context} progressive bias"
    )
    condition_raw = raw.get("condition")
    condition = None
    if condition_raw is not None:
        if not isinstance(condition_raw, dict):
            raise TypeError(
                f"{context} progressive_bias condition must be a TOML table"
            )
        unknown_condition = sorted(condition_raw.keys() - {"kind", "phase"})
        if unknown_condition:
            raise ValueError(
                f"unknown {context} progressive_bias condition fields: "
                + ", ".join(unknown_condition)
            )
        if condition_raw.get("kind") != "turn_phase":
            raise ValueError(
                f"{context} progressive_bias condition kind must be turn_phase"
            )
        phase = condition_raw.get("phase")
        if not isinstance(phase, str):
            raise TypeError(
                f"{context} progressive_bias turn_phase condition requires a phase"
            )
        condition = TurnPhaseIs(phase)
    return ProgressiveBias(float(weight), evaluator, condition)


def _configured_root_diagnostics(values: dict[str, object], context: str) -> bool:
    enabled = values.get("root_diagnostics", False)
    if not isinstance(enabled, bool):
        raise TypeError(f"{context} root_diagnostics must be a boolean")
    return enabled


def _configured_tree_reuse(values: dict[str, object], context: str) -> bool:
    enabled = values.get("tree_reuse", False)
    if not isinstance(enabled, bool):
        raise TypeError(f"{context} tree_reuse must be a boolean")
    return enabled


def _configured_transpositions(values: dict[str, object], context: str) -> bool:
    enabled = values.get("transpositions", False)
    if not isinstance(enabled, bool):
        raise TypeError(f"{context} transpositions must be a boolean")
    return enabled


def _configured_rollout_policy(
    values: dict[str, object],
    context: str,
) -> UniformRandom | Greedy | EpsilonGreedy | Mast | ConditionalRollout:
    policy = values.get("rollout_policy", "uniform_random")
    if isinstance(policy, dict):
        kind = policy.get("kind")
        if not isinstance(kind, str):
            raise TypeError(f"{context} rollout policy kind must be a string")
        flat_fields = {
            "rollout_evaluator",
            "rollout_use_heuristic",
            "rollout_heuristic_index",
            "rollout_epsilon",
        }
        if flat_fields & values.keys():
            raise ValueError(
                f"{context} structured rollout_policy cannot use flat rollout evaluator fields"
            )
        normalized_policy = kind.strip().lower().replace("-", "_")
        if normalized_policy == "conditional":
            unknown = sorted(
                policy.keys() - {"kind", "condition", "primary", "fallback"}
            )
            if unknown:
                raise ValueError(
                    f"unknown {context} conditional rollout fields: {', '.join(unknown)}"
                )
            condition = policy.get("condition")
            if not isinstance(condition, dict):
                raise TypeError(f"{context} conditional rollout condition must be a table")
            condition_unknown = sorted(condition.keys() - {"kind", "phase"})
            if condition_unknown:
                raise ValueError(
                    f"unknown {context} rollout condition fields: "
                    + ", ".join(condition_unknown)
                )
            condition_kind = condition.get("kind")
            if not isinstance(condition_kind, str):
                raise TypeError(f"{context} rollout condition kind must be a string")
            if condition_kind.strip().lower().replace("-", "_") != "turn_phase":
                raise ValueError(
                    f"{context} rollout condition kind must be turn_phase"
                )
            phase = condition.get("phase")
            if not isinstance(phase, str):
                raise TypeError(f"{context} rollout condition phase must be a string")
            primary = policy.get("primary")
            fallback = policy.get("fallback")
            if not isinstance(primary, dict) or not isinstance(fallback, dict):
                raise TypeError(
                    f"{context} conditional rollout primary and fallback must be tables"
                )
            primary_policy = _configured_rollout_policy(
                {"rollout_policy": primary}, f"{context} primary"
            )
            fallback_policy = _configured_rollout_policy(
                {"rollout_policy": fallback}, f"{context} fallback"
            )
            if isinstance(primary_policy, ConditionalRollout) or isinstance(
                fallback_policy, ConditionalRollout
            ):
                raise ValueError(f"{context} conditional rollout branches cannot be conditional")
            return ConditionalRollout(
                condition=TurnPhaseIs(phase),
                primary=primary_policy,
                fallback=fallback_policy,
            )
        unknown = sorted(policy.keys() - {"kind", "epsilon", "evaluator"})
        if unknown:
            raise ValueError(f"unknown {context} rollout policy fields: {', '.join(unknown)}")
        evaluator = (
            _configured_evaluator(policy["evaluator"], f"{context} rollout")
            if "evaluator" in policy
            else None
        )
        epsilon = policy.get("epsilon")
    elif isinstance(policy, str):
        normalized_policy = policy.strip().lower().replace("-", "_")
        evaluator = _configured_legacy_rollout_evaluator(
            values,
            context,
            share_legacy_cutoff=normalized_policy
            not in {"uniform_random", "uniform", "random", "mast"},
        )
        epsilon = values.get("rollout_epsilon")
    else:
        raise TypeError(f"{context} rollout_policy must be a string or TOML table")

    if normalized_policy in {"uniform_random", "uniform", "random"}:
        if evaluator is not None or epsilon is not None:
            raise ValueError(f"{context} uniform_random rollout accepts no evaluator or epsilon")
        return UniformRandom()
    if normalized_policy == "mast":
        if evaluator is not None:
            raise ValueError(f"{context} mast rollout does not accept an evaluator")
        return Mast(0.1 if epsilon is None else epsilon)
    if evaluator is None:
        raise ValueError(f"{context} {normalized_policy} rollout requires an evaluator")
    if normalized_policy == "greedy":
        if epsilon is not None:
            raise ValueError(f"{context} greedy rollout does not accept epsilon")
        return Greedy(evaluator)
    if normalized_policy not in {
        "epsilon_greedy_heuristic",
        "epsilon_greedy",
        "epsilon",
    }:
        raise ValueError(
            f"{context} rollout_policy must be uniform_random, greedy, epsilon_greedy, or mast"
        )
    epsilon = 0.1 if epsilon is None else epsilon
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError(f"{context} rollout_epsilon must be a number")
    return EpsilonGreedy(float(epsilon), evaluator)


def _configured_legacy_rollout_evaluator(
    values: dict[str, object],
    context: str,
    *,
    share_legacy_cutoff: bool,
) -> NeutralEvaluator | GameHeuristic | None:
    if "rollout_evaluator" in values:
        legacy = {"rollout_use_heuristic", "rollout_heuristic_index"} & values.keys()
        if legacy:
            raise ValueError(
                f"{context} rollout_evaluator cannot be combined with: "
                + ", ".join(sorted(legacy))
            )
        return _configured_evaluator(values["rollout_evaluator"], f"{context} rollout")
    if "rollout_use_heuristic" in values:
        enabled = values["rollout_use_heuristic"]
        if not isinstance(enabled, bool):
            raise TypeError(f"{context} rollout_use_heuristic must be a boolean")
        if not enabled:
            return NeutralEvaluator()
    if "rollout_heuristic_index" in values or values.get("rollout_use_heuristic") is True:
        index = values.get("rollout_heuristic_index", 0)
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"{context} rollout_heuristic_index must be an integer")
        return GameHeuristic(index)

    # Compatibility with the first informed-rollout format, where one heuristic was shared.
    if share_legacy_cutoff and values.get("use_heuristic") is True:
        index = values.get("heuristic_index", 0)
        if isinstance(index, int) and not isinstance(index, bool):
            return GameHeuristic(index)
    return None
