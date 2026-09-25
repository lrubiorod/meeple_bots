"""Tournament TOML decoding and deterministic agent grid expansion."""
from __future__ import annotations

import tomllib
from itertools import product
from pathlib import Path

from ._concurrency import resolve_workers
from ._mcts_profiles import (
    _configured_cutoff_evaluator, _configured_progressive_bias,
    _configured_rollout_policy, _configured_root_diagnostics,
    _configured_transpositions, _configured_tree_reuse,
    _mcts_budget_kwargs, sqrt_two,
)
from ._agent_config import MctsAgent, RandomAgent
from .game_types import TicTacToe, ConnectFour, Boop, SpiritsOfTheForest
from .matches.execution import Match
from .game_config import PLAYABLE_GAMES, create_game
from .serialization import game_name as _game_name
from .tournaments import TournamentAgent as _TournamentAgent, TournamentConfig as _TournamentConfig

_PLAYABLE_GAMES = PLAYABLE_GAMES

_TOURNAMENT_GRID_FIELDS = (
    (("iterations",), "i"),
    (("time_budget",), "t"),
    (("rollout_depth",), "d"),
    (("exploration",), "c"),
    (("selection_policy",), "selection"),
    (("rave_equivalence",), "rave_equivalence"),
    (("progressive_widening",), "progressive_widening"),
    (("progressive_widening_k",), "progressive_widening_k"),
    (("progressive_widening_alpha",), "progressive_widening_alpha"),
    (("progressive_widening_expansion",), "progressive_widening_expansion"),
    (("heuristic_index",), "h"),
    (("cutoff_evaluator", "index"), "h"),
    (("rollout_heuristic_index",), "rh"),
    (("rollout_policy", "evaluator", "index"), "rh"),
    (("rollout_policy", "primary", "evaluator", "index"), "rh"),
    (("rollout_epsilon",), "e"),
    (("rollout_policy", "epsilon"), "e"),
    (("rollout_policy", "primary", "epsilon"), "e"),
    (("progressive_bias", "weight"), "pb"),
)
_EVALUATOR_PARAM_GRID_PATHS = (
    (("cutoff_evaluator", "params"), "hp-"),
    (("rollout_policy", "evaluator", "params"), "rhp-"),
    (("rollout_policy", "primary", "evaluator", "params"), "rhp-"),
    (("rollout_policy", "fallback", "evaluator", "params"), "rfhp-"),
    (("progressive_bias", "evaluator", "params"), "pbhp-"),
)
_MAX_AGENTS_PER_TOURNAMENT_GRID = 256
_MISSING_GRID_VALUE = object()


def _load_tournament_config(path: Path) -> _TournamentConfig:
    with path.open("rb") as config_file:
        values = tomllib.load(config_file)
    return tournament_config_from_values(values, path)


def tournament_config_from_values(values: dict[str, object], path: Path) -> _TournamentConfig:
    """Build one tournament plan; path anchors relative artifact references."""
    allowed = {
        "game",
        "game_params",
        "output",
        "pairing_mode",
        "seat_mode",
        "matches_per_pair",
        "seed",
        "max_plies",
        "workers",
        "agents",
    }
    unknown = sorted(values.keys() - allowed)
    if unknown:
        raise ValueError(f"unknown tournament fields: {', '.join(unknown)}")

    game_name = values.get("game")
    if game_name not in _PLAYABLE_GAMES:
        raise ValueError(
            "tournament game must be boop, connect-four, spotf, tic-tac-toe, or splendor"
        )
    game = create_game(game_name, values.get("game_params"))
    raw_output = values.get("output")
    if raw_output is not None and (
        not isinstance(raw_output, str) or not raw_output.strip()
    ):
        raise ValueError("tournament output must be a non-empty path string")
    output = None if raw_output is None else Path(raw_output)
    if output is not None and not output.is_absolute():
        output = (path.parent / output).resolve()
    pairing_mode = values.get("pairing_mode", "round_robin")
    if not isinstance(pairing_mode, str):
        raise TypeError("tournament pairing_mode must be a string")
    if pairing_mode not in {"round_robin", "adjacent"}:
        raise ValueError("tournament pairing_mode must be round_robin or adjacent")
    seat_mode = values.get("seat_mode", "alternating")
    if not isinstance(seat_mode, str):
        raise TypeError("tournament seat_mode must be a string")
    if seat_mode not in {"alternating", "paired"}:
        raise ValueError("tournament seat_mode must be alternating or paired")
    matches_per_pair = _positive_tournament_integer(
        "matches_per_pair", values.get("matches_per_pair")
    )
    if seat_mode == "paired" and matches_per_pair % 2 != 0:
        raise ValueError("tournament matches_per_pair must be even for paired seats")
    max_plies = _positive_tournament_integer(
        "max_plies", values.get("max_plies", 10_000)
    )
    workers = resolve_workers(values.get("workers", "auto"))
    seed = values.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("tournament seed must be an integer")
    if not 0 <= seed <= 2**64 - 1:
        raise ValueError("tournament seed must be between 0 and 18446744073709551615")

    raw_agents = values.get("agents")
    if not isinstance(raw_agents, list) or not raw_agents:
        raise ValueError("tournament agents must contain at least one entry")
    agents = tuple(
        agent
        for index, raw in enumerate(raw_agents, start=1)
        for agent in _load_tournament_agents(raw, index, game)
    )
    if len(agents) < 2:
        raise ValueError(
            "tournament agents must expand to at least two configurations"
        )
    names = [agent.name for agent in agents]
    if len(names) != len(set(names)):
        raise ValueError("tournament agent names must be unique")
    return _TournamentConfig(
        game=game,
        output=output,
        pairing_mode=pairing_mode,
        seat_mode=seat_mode,
        matches_per_pair=matches_per_pair,
        seed=seed,
        max_plies=max_plies,
        workers=workers,
        agents=agents,
    )


def _load_tournament_agents(
    values: object,
    index: int,
    game: TicTacToe | ConnectFour | Boop | SpiritsOfTheForest,
) -> tuple[_TournamentAgent, ...]:
    if not isinstance(values, dict):
        raise TypeError(f"tournament agent {index} must be a TOML table")
    allowed = {
        "name",
        "kind",
        "iterations",
        "time_budget",
        "exploration",
        "selection_policy",
        "rave_equivalence",
        "progressive_widening",
        "progressive_widening_k",
        "progressive_widening_alpha",
        "progressive_widening_expansion",
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
        "self_play",
    }
    unknown = sorted(values.keys() - allowed)
    if unknown:
        raise ValueError(
            f"unknown fields for tournament agent {index}: {', '.join(unknown)}"
        )
    name = values.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"tournament agent {index} name must be a non-empty string")
    kind = values.get("kind")
    if kind not in {"mcts", "random", "so_ismcts"}:
        raise ValueError(f"tournament agent {name} kind must be mcts, so_ismcts or random")
    self_play = values.get("self_play", False)
    if not isinstance(self_play, bool):
        raise TypeError(f"tournament agent {name} self_play must be a boolean")

    mcts_fields = {
        "iterations",
        "time_budget",
        "exploration",
        "selection_policy",
        "rave_equivalence",
        "progressive_widening",
        "progressive_widening_k",
        "progressive_widening_alpha",
        "progressive_widening_expansion",
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
    if kind == "random":
        unexpected = sorted(values.keys() & mcts_fields)
        if unexpected:
            raise ValueError(
                f"random tournament agent {name} cannot use: {', '.join(unexpected)}"
            )
        return (
            _TournamentAgent(
                name=name.strip(),
                agent=RandomAgent(),
                self_play=self_play,
                template_index=index,
                grid_position=(),
            ),
        )

    if kind == "so_ismcts":
        from ._search_profiles import resolve_family, so_from_values
        resolve_family(_game_name(game), "so_ismcts")
        agent = so_from_values({k: v for k, v in values.items()
                                if k not in ("kind", "self_play")})
        return (_TournamentAgent(name=name.strip(), agent=agent, self_play=self_play,
                                 template_index=index, grid_position=()),)

    missing = sorted({"rollout_depth"} - values.keys())
    if missing:
        raise ValueError(
            f"missing fields for tournament agent {name}: {', '.join(missing)}"
        )
    _mcts_budget_kwargs(values, f"tournament agent {name}")
    use_heuristic = values.get("use_heuristic", False)
    if not isinstance(use_heuristic, bool):
        raise TypeError(f"tournament agent {name} use_heuristic must be a boolean")
    if isinstance(values.get("heuristic_index"), list) and not use_heuristic:
        raise ValueError(
            f"tournament agent {name} cannot vary heuristic_index when "
            "use_heuristic is false"
        )

    grid_fields: list[tuple[tuple[str, ...], str]] = []
    grid_options: list[list[object]] = []
    combination_count = 1
    used_suffixes: dict[str, tuple[str, ...]] = {}
    all_grid_fields = (*_TOURNAMENT_GRID_FIELDS, *_heuristic_param_grid_fields(values))
    for path, suffix in all_grid_fields:
        raw_options = _nested_tournament_value(values, path)
        if not isinstance(raw_options, list):
            continue
        field = ".".join(path)
        previous_path = used_suffixes.get(suffix)
        if previous_path is not None:
            raise ValueError(
                f"tournament agent {name} cannot vary both "
                f"{'.'.join(previous_path)} and {field}"
            )
        used_suffixes[suffix] = path
        options = raw_options
        if not options:
            raise ValueError(f"tournament agent {name} {field} list cannot be empty")
        if any(value in options[:option_index] for option_index, value in enumerate(options)):
            raise ValueError(
                f"tournament agent {name} {field} list contains duplicate values"
            )
        grid_fields.append((path, suffix))
        grid_options.append(options)
        combination_count *= len(options)
    if combination_count > _MAX_AGENTS_PER_TOURNAMENT_GRID:
        raise ValueError(
            f"tournament agent {name} expands to {combination_count} configurations; "
            f"the maximum is {_MAX_AGENTS_PER_TOURNAMENT_GRID}"
        )

    grid_positions = (
        product(*(range(len(options)) for options in grid_options))
        if grid_options
        else [()]
    )
    expanded = []
    for grid_position in grid_positions:
        combination = tuple(
            options[position]
            for options, position in zip(
                grid_options,
                grid_position,
                strict=True,
            )
        )
        concrete = dict(values)
        for (path, _suffix), value in zip(grid_fields, combination, strict=True):
            _set_nested_tournament_value(concrete, path, value)
        agent = _build_tournament_mcts_agent(concrete, name, game)
        suffix = "".join(
            f"-{field_suffix}{value}"
            for (_path, field_suffix), value in zip(
                grid_fields,
                combination,
                strict=True,
            )
        )
        expanded.append(
            _TournamentAgent(
                name=f"{name.strip()}{suffix}",
                agent=agent,
                self_play=self_play,
                template_index=index,
                grid_position=tuple(grid_position),
            )
        )
    return tuple(expanded)


def _nested_tournament_value(
    values: dict[str, object],
    path: tuple[str, ...],
) -> object:
    current: object = values
    for field in path:
        if not isinstance(current, dict) or field not in current:
            return _MISSING_GRID_VALUE
        current = current[field]
    return current


def _heuristic_param_grid_fields(
    values: dict[str, object],
) -> tuple[tuple[tuple[str, ...], str], ...]:
    fields = []
    for params_path, suffix_prefix in _EVALUATOR_PARAM_GRID_PATHS:
        params = _nested_tournament_value(values, params_path)
        if not isinstance(params, dict):
            continue
        fields.extend(
            (params_path + (name,), suffix_prefix + name + "-")
            for name in sorted(params)
            if isinstance(params[name], list)
        )
    return tuple(fields)


def _set_nested_tournament_value(
    values: dict[str, object],
    path: tuple[str, ...],
    value: object,
) -> None:
    current = values
    for field in path[:-1]:
        child = current.get(field)
        if not isinstance(child, dict):
            raise ValueError(f"tournament grid path {'.'.join(path)} is not a table")
        copied_child = dict(child)
        current[field] = copied_child
        current = copied_child
    current[path[-1]] = value


def _build_tournament_mcts_agent(
    values: dict[str, object],
    name: str,
    game: TicTacToe | ConnectFour | Boop | SpiritsOfTheForest,
) -> MctsAgent:
    agent = MctsAgent(
        **_mcts_budget_kwargs(values, f"tournament agent {name}"),
        selection_policy=values.get("selection_policy", "uct"),
        rave_equivalence=values.get("rave_equivalence", 1000),
        progressive_widening=values.get("progressive_widening", False),
        progressive_widening_k=values.get("progressive_widening_k", 1.5),
        progressive_widening_alpha=values.get("progressive_widening_alpha", 0.5),
        progressive_widening_expansion=values.get("progressive_widening_expansion", "random"),
        exploration=values.get("exploration", sqrt_two()),
        rollout_depth=values["rollout_depth"],
        cutoff_evaluator=_configured_cutoff_evaluator(
            values,
            f"tournament agent {name}",
        ),
        rollout_policy=_configured_rollout_policy(values, f"tournament agent {name}"),
        progressive_bias=_configured_progressive_bias(
            values, f"tournament agent {name}"
        ),
        root_diagnostics=_configured_root_diagnostics(
            values, f"tournament agent {name}"
        ),
        tree_reuse=_configured_tree_reuse(values, f"tournament agent {name}"),
        transpositions=_configured_transpositions(
            values, f"tournament agent {name}"
        ),
    )
    Match(game=game, first=agent, second=RandomAgent())
    return agent


def _positive_tournament_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"tournament {name} must be an integer")
    if not 1 <= value <= 2**32 - 1:
        raise ValueError(f"tournament {name} must be between 1 and 4294967295")
    return value
