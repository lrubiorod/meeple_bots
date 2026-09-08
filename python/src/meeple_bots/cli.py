"""Command-line interface for running observable matches."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from ._mcts_profiles import (
    _MctsProfile,
    _configured_cutoff_evaluator,
    _configured_evaluator,
    _configured_legacy_rollout_evaluator,
    _configured_progressive_bias,
    _configured_rollout_policy,
    _configured_root_diagnostics,
    _configured_transpositions,
    _configured_tree_reuse,
    _inline_agent_integer,
    _inline_agent_name,
    _inline_evaluator,
    _load_mcts_profile,
    _mcts_budget_kwargs,
    _parse_inline_mcts_profile,
    sqrt_two,
)

from .api import (
    Batch,
    BatchProgress,
    BatchProgressStatus,
    BatchResult,
    Boop,
    BoopAction,
    BoopGraduateLine,
    BoopPiece,
    BoopPieceKind,
    BoopRecoverPiece,
    ConnectFour,
    ConnectFourAction,
    ConditionalRollout,
    EndSpiritCollection,
    EpsilonGreedy,
    GameEvaluationReport,
    GameHeuristic,
    Greedy,
    HumanAgent,
    HumanMoveObservation,
    Match,
    MatchResult,
    Mast,
    MctsAgent,
    MctsAgentBenchmark,
    NeutralEvaluator,
    MoveSpiritGemstone,
    PlaceSpiritGemstone,
    ProgressiveBias,
    RandomAgent,
    SkipSpiritGemstone,
    SpiritTile,
    SpiritsOfTheForest,
    TakeSpiritTile,
    TicTacToe,
    TicTacToeAction,
    TurnPhaseIs,
    UniformRandom,
    benchmark_mcts_agent,
    evaluate_game,
)
from ._concurrency import WorkerSetting, resolve_workers
from .extraction import extract_tournament
from .gui import run_gui
from .reporting import generate_study_report
from .serialization import (
    match_result_dict as _result_dict,
    agent_dict as _batch_agent_dict,
    _progressive_bias_fields,
    _rollout_policy_name,
    _base_rollout_policy_name,
    _rollout_policy_evaluator,
    _rollout_policy_epsilon,
    _conditional_rollout_fields,
    _evaluator_heuristic_index,
    _evaluator_dict,
    game_name as _game_name,
    trace_match_dict as _trace_match_dict,
    write_jsonl as _write_jsonl,
)
from .tournaments import (
    TournamentAgent as _TournamentAgent,
    TournamentConfig as _TournamentConfig,
    tournament_pairings as _tournament_pairings,
    run_tournament,
)


_PLAYABLE_GAMES = ["boop", "connect-four", "spotf", "tic-tac-toe"]
_TOURNAMENT_GRID_FIELDS = (
    (("iterations",), "i"),
    (("time_budget",), "t"),
    (("rollout_depth",), "d"),
    (("exploration",), "c"),
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


def _game_tag(value: str) -> str:
    return "spotf" if value == "spirits-of-the-forest" else value


def _worker_setting(value: str) -> WorkerSetting:
    if value == "auto":
        return "auto"
    try:
        workers = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "workers must be 'auto' or a positive integer"
        ) from error
    if workers < 1:
        raise argparse.ArgumentTypeError(
            "workers must be 'auto' or a positive integer"
        )
    return workers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meeple-bots")
    commands = parser.add_subparsers(dest="command", required=True)
    gui = commands.add_parser("gui", help="play or watch a game in a local browser")
    gui.add_argument(
        "--game",
        type=_game_tag,
        choices=_PLAYABLE_GAMES,
        default="tic-tac-toe",
    )
    gui.add_argument("--host", default="127.0.0.1")
    gui.add_argument("--port", type=int, default=8765)
    gui.add_argument(
        "--no-browser",
        action="store_true",
        help="serve the interface without opening a browser automatically",
    )

    match = commands.add_parser("match", help="run and display one match")
    match.add_argument(
        "--game", type=_game_tag, choices=_PLAYABLE_GAMES, default="tic-tac-toe"
    )
    match.add_argument("--first", choices=["human", "mcts", "random"], default="mcts")
    match.add_argument("--second", choices=["human", "mcts", "random"], default="random")
    match.add_argument("--seed", type=int, default=0)
    match.add_argument("--max-plies", type=int, default=10_000)
    match_budget = match.add_mutually_exclusive_group()
    match_budget.add_argument("--mcts-iterations", type=int)
    match_budget.add_argument(
        "--mcts-time-budget",
        type=float,
        help="approximate wall-clock seconds per MCTS decision",
    )
    match.add_argument("--mcts-exploration", type=float, default=sqrt_two())
    match.add_argument("--mcts-rollout-depth", type=int)
    match.add_argument(
        "--mcts-rollout-policy",
        choices=[
            "uniform_random",
            "uniform",
            "random",
            "greedy",
            "mast",
            "epsilon_greedy_heuristic",
            "epsilon_greedy",
            "epsilon",
        ],
        default="uniform_random",
        help="rollout action policy shared by manually configured MCTS players",
    )
    match.add_argument(
        "--mcts-rollout-epsilon",
        type=float,
        help="random-action probability for epsilon-greedy heuristic rollouts",
    )
    match.add_argument(
        "--mcts-rollout-heuristic",
        type=int,
        help="game heuristic used only to rank informed rollout actions",
    )
    match.add_argument(
        "--mcts-tree-reuse",
        action="store_true",
        help="reuse the reachable MCTS subtree across decisions in this match",
    )
    match.add_argument(
        "--mcts-transpositions",
        action="store_true",
        help="merge exactly equal MCTS states reached through different action sequences",
    )
    match.add_argument("--first-mcts-config", type=Path)
    match.add_argument("--second-mcts-config", type=Path)
    match.add_argument(
        "--first-mcts-heuristic",
        nargs="?",
        const=0,
        type=int,
        help="use the selected game's heuristic (index 0 when no value is given)",
    )
    match.add_argument(
        "--second-mcts-heuristic",
        nargs="?",
        const=0,
        type=int,
        help="use the selected game's heuristic (index 0 when no value is given)",
    )
    match.add_argument("--json", action="store_true", help="print machine-readable JSON")

    batch = commands.add_parser("batch", help="run and summarize automated matches")
    batch.add_argument(
        "--game", type=_game_tag, choices=_PLAYABLE_GAMES, required=True
    )
    batch.add_argument("--matches", type=int, default=20)
    batch.add_argument("--agent-a", choices=["mcts", "random"], default="random")
    batch.add_argument("--agent-b", choices=["mcts", "random"], default="mcts")
    batch.add_argument("--agent-a-config", type=Path)
    batch.add_argument("--agent-b-config", type=Path)
    batch.add_argument("--seed", type=int, default=0)
    batch.add_argument("--max-plies", type=int, default=10_000)
    batch.add_argument(
        "--workers",
        type=_worker_setting,
        default="auto",
        help="parallel matches: 'auto' uses physical cores minus one (default: auto)",
    )
    batch.add_argument(
        "--no-alternate-sides",
        action="store_false",
        dest="alternate_sides",
        help="keep agent A as player 0 in every match",
    )
    batch.add_argument(
        "--output",
        type=Path,
        help="save full match traces as extract-compatible JSONL",
    )
    batch.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing trace file instead of refusing to run",
    )
    batch.add_argument("--json", action="store_true", help="print machine-readable JSON")

    tournament = commands.add_parser(
        "tournament",
        help="run a configured tournament and save full match traces",
    )
    tournament.add_argument("--config", type=Path, required=True)
    tournament.add_argument(
        "--workers",
        type=_worker_setting,
        help="override tournament workers with 'auto' or a positive integer",
    )
    tournament.add_argument(
        "--output",
        type=Path,
        help="override the output path configured in the tournament TOML",
    )
    tournament.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file instead of refusing to run",
    )
    tournament.add_argument("--json", action="store_true", help="print summary as JSON")

    extract = commands.add_parser(
        "extract",
        help="extract game-specific analysis tables from tournament or batch traces",
    )
    extract.add_argument(
        "--input",
        type=Path,
        nargs="+",
        action="extend",
        required=True,
        help="one or more compatible tournament or batch JSONL traces",
    )
    extract.add_argument(
        "--output-dir",
        type=Path,
        help="output location; required for multiple study traces",
    )
    extract.add_argument(
        "--overwrite",
        action="store_true",
        help="replace known extraction files that already exist",
    )
    extract.add_argument("--json", action="store_true", help="print summary as JSON")

    report = commands.add_parser(
        "report",
        help="generate statistics and figures from extracted study tables",
    )
    report.add_argument("--input", type=Path, required=True)
    report.add_argument(
        "--output-dir",
        type=Path,
        help="write the report here instead of beside the extracted tables",
    )
    report.add_argument(
        "--overwrite",
        action="store_true",
        help="replace known report files that already exist",
    )
    report.add_argument("--json", action="store_true", help="print summary as JSON")

    analyze = commands.add_parser("analyze", help="measure game complexity and calibrate MCTS")
    analyze.add_argument(
        "--game", type=_game_tag, choices=_PLAYABLE_GAMES, required=True
    )
    analyze.add_argument("--samples", type=int, default=128)
    analyze.add_argument("--max-depth", type=int, default=256)
    analyze.add_argument("--seed", type=int, default=0)
    analyze.add_argument(
        "--target-time",
        type=float,
        default=5.0,
        help="target seconds per decision for suggested experiments (default: 5)",
    )
    analyze.add_argument(
        "--agent-config",
        type=Path,
        action="append",
        default=[],
        help="benchmark an exact MCTS profile; repeat to compare any number of agents",
    )
    analyze.add_argument(
        "--agent",
        nargs="?",
        const="",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "benchmark an inline MCTS agent; repeat as needed, for example "
            "--agent 'i=5000,d=120,h=0'"
        ),
    )
    analyze.add_argument("--json", action="store_true", help="print machine-readable JSON")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "gui":
            run_gui(
                game=args.game,
                host=args.host,
                port=args.port,
                open_browser=not args.no_browser,
            )
            return 0
        if args.command == "tournament":
            return _run_tournament(args)
        if args.command == "extract":
            summary = extract_tournament(
                args.input,
                args.output_dir,
                overwrite=args.overwrite,
            )
            if args.json:
                print(json.dumps(summary, indent=2))
            else:
                _print_extraction_summary(summary)
            return 0
        if args.command == "report":
            summary = generate_study_report(
                args.input,
                args.output_dir,
                overwrite=args.overwrite,
            )
            if args.json:
                print(json.dumps(summary, indent=2))
            else:
                _print_report_summary(summary)
            return 0
        game = _game(args.game)
        if args.command == "analyze":
            profiles = _analyze_profiles(args.agent_config, args.agent, game)
            report = evaluate_game(
                game,
                samples=args.samples,
                max_depth=args.max_depth,
                seed=args.seed,
                target_time=args.target_time,
            )
            benchmarks = []
            for profile in profiles:
                print(
                    f"Benchmarking {profile.name}: "
                    f"{_mcts_budget_description(profile.agent)}, "
                    f"depth {profile.agent.rollout_depth}, "
                    f"cutoff={_evaluator_name(profile.agent.cutoff_evaluator)}, "
                    f"rollout={_rollout_policy_description(profile.agent)}, "
                    "progressive_bias="
                    f"{_progressive_bias_description(profile.agent.progressive_bias)}",
                    file=sys.stderr,
                    flush=True,
                )
                benchmarks.append(
                    _ConfiguredMctsBenchmark(
                        name=profile.name,
                        benchmark=benchmark_mcts_agent(
                            game,
                            profile.agent,
                            report.depth_p50,
                            seed=args.seed,
                        ),
                    )
                )
            if args.json:
                print(json.dumps(_evaluation_dict(report, benchmarks), indent=2))
            else:
                _print_evaluation(report, benchmarks)
            return 0
        if args.command == "batch":
            return _run_batch(args, game)

        mcts = _mcts_configuration(args)
        first = _match_agent(
            args.first,
            args.first_mcts_config,
            mcts,
            args.first_mcts_heuristic,
            "--first-mcts-config",
            "--first-mcts-heuristic",
        )
        second = _match_agent(
            args.second,
            args.second_mcts_config,
            mcts,
            args.second_mcts_heuristic,
            "--second-mcts-config",
            "--second-mcts-heuristic",
        )
        result = Match(
            game=game,
            first=first,
            second=second,
            seed=args.seed,
            max_plies=args.max_plies,
        ).run()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.json:
        payload = _result_dict(result)
        payload["players"] = [
            _agent_dict(args.first, first),
            _agent_dict(args.second, second),
        ]
        print(json.dumps(payload, indent=2))
    else:
        _print_result(result, args.first, args.second, first, second)
    return 0


def _print_extraction_summary(summary: dict[str, object]) -> None:
    inputs = summary.get("inputs", [summary["input"]])
    print(f"Inputs ({len(inputs)}):")
    for input_path in inputs:
        print(f"  {input_path}")
    print(f"Output directory: {summary['output_dir']}")
    print(
        f"Matches: {summary['processed_matches']}/{summary['declared_matches']} "
        f"({'complete' if summary['complete'] else 'partial'})"
    )
    print(f"Rows: {summary['row_counts']}")


def _print_report_summary(summary: dict[str, object]) -> None:
    print(f"Input directory: {summary['input_dir']}")
    print(f"Output directory: {summary['output_dir']}")
    print(f"Game: {summary['game']}")
    print(f"Matches: {summary['matches']}")
    print(f"Study: {'complete' if summary['complete'] else 'partial'}")
    print(f"Artifacts: {summary['figures']} figures, {summary['tables']} tables")


@dataclass(frozen=True, slots=True)
class _ConfiguredMctsBenchmark:
    name: str
    benchmark: MctsAgentBenchmark


def _run_tournament(args: argparse.Namespace) -> int:
    config = _load_tournament_config(args.config)
    total_matches = 0

    def started(header: dict) -> None:
        nonlocal total_matches
        total_matches = header["total_matches"]
        print(
            f"Starting tournament: {header['game']}, {len(header['agents'])} agents, "
            f"{header['total_pairings']} pairings, {total_matches} matches, {header['workers']} workers",
            file=sys.stderr, flush=True,
        )

    def completed(row: dict) -> None:
        print(
            f"[{row['match_number']}/{total_matches}] {row['agent_a']} vs {row['agent_b']}: "
            f"winner={row['winner'] or 'draw'}, plies={row['result']['plies']}, "
            f"time={row['duration_seconds']:.3f}s",
            file=sys.stderr, flush=True,
        )

    summary = run_tournament(
        config, output=args.output, workers=args.workers, overwrite=args.overwrite,
        on_start=started, on_match=completed,
    )
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_tournament_summary(summary)
    return 0


def _load_tournament_config(path: Path) -> _TournamentConfig:
    with path.open("rb") as config_file:
        values = tomllib.load(config_file)
    allowed = {
        "game",
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
            "tournament game must be boop, connect-four, spotf, or tic-tac-toe"
        )
    game = _game(game_name)
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
    if kind not in {"mcts", "random"}:
        raise ValueError(f"tournament agent {name} kind must be mcts or random")
    self_play = values.get("self_play", False)
    if not isinstance(self_play, bool):
        raise TypeError(f"tournament agent {name} self_play must be a boolean")

    mcts_fields = {
        "iterations",
        "time_budget",
        "exploration",
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


def _print_tournament_summary(summary: dict[str, object]) -> None:
    print(f"Game: {summary['game']}")
    print(f"Agents: {summary['agents']}")
    print(f"Pairings: {summary['pairings']}")
    print(f"Seat mode: {summary['seat_mode']}")
    print(f"Matches: {summary['matches']}")
    print(f"Workers: {summary['workers']}")
    print(f"Total time: {summary['elapsed_seconds']:.3f}s")
    print(f"Trace output: {summary['output']}")
    print()
    print("Standings (self-play excluded):")
    standings = summary["standings"]
    if not isinstance(standings, dict):
        raise RuntimeError("tournament standings have an invalid shape")
    for name, result in standings.items():
        print(
            f"  {name}: {result['wins']}W {result['losses']}L {result['draws']}D "
            f"({result['games']} games, {result['self_play_games']} self-play)"
        )


def _run_batch(
    args: argparse.Namespace,
    game: TicTacToe | ConnectFour | Boop | SpiritsOfTheForest,
) -> int:
    agent_a, name_a = _batch_agent(args.agent_a, args.agent_a_config, "--agent-a-config")
    agent_b, name_b = _batch_agent(args.agent_b, args.agent_b_config, "--agent-b-config")
    if args.overwrite and args.output is None:
        raise ValueError("--overwrite requires --output")
    batch = Batch(
        game=game,
        agent_a=agent_a,
        agent_b=agent_b,
        matches=args.matches,
        seed=args.seed,
        max_plies=args.max_plies,
        alternate_sides=args.alternate_sides,
        workers=args.workers,
    )
    _print_batch_setup(batch, game, name_a, agent_a, name_b, agent_b)
    if args.output is None:
        result = batch.run(
            progress=lambda event: _print_batch_progress(event, name_a, name_b)
        )
    else:
        trace_agents, self_play = _batch_trace_agents(name_a, agent_a, name_b, agent_b)
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(
                f"batch trace already exists; use --overwrite to replace: {args.output}"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_mode = "w" if args.overwrite else "x"
        with args.output.open(output_mode, encoding="utf-8") as trace_output:
            _write_jsonl(
                trace_output,
                {
                    "record_type": "tournament",
                    "schema_version": 1,
                    "decision_timing_scope": "agent_total_v1",
                    "study_type": "batch",
                    "game": _game_name(game),
                    "output": str(args.output),
                    "matches_per_pair": batch.matches,
                    "seed": batch.seed,
                    "max_plies": batch.max_plies,
                    "workers": min(resolve_workers(batch.workers), batch.matches),
                    "total_pairings": 1,
                    "total_matches": batch.matches,
                    "agents": trace_agents,
                },
            )

            def report_progress(event: BatchProgress) -> None:
                _print_batch_progress(event, name_a, name_b)
                if event.status is not BatchProgressStatus.COMPLETED:
                    return
                if event.result is None or event.match_result is None:
                    raise RuntimeError("completed batch progress is missing its match trace")
                winner = (
                    None
                    if event.result.winner is None
                    else "agent_a"
                    if event.result.winner == 0
                    else "agent_b"
                )
                _write_jsonl(
                    trace_output,
                    _trace_match_dict(
                        result=event.match_result,
                        match_number=event.match_number,
                        pairing_number=1,
                        pairing_match_number=event.match_number,
                        agent_a=name_a,
                        agent_b=name_b,
                        self_play=self_play,
                        agent_a_player=event.agent_a_player,
                        winner=winner,
                        duration_seconds=event.result.duration_seconds,
                    ),
                )

            result = batch.run(progress=report_progress)
    if args.json:
        print(
            json.dumps(
                _batch_dict(
                    result,
                    game,
                    name_a,
                    agent_a,
                    name_b,
                    agent_b,
                    output=args.output,
                ),
                indent=2,
            )
        )
    else:
        _print_batch_result(
            result,
            game,
            name_a,
            agent_a,
            name_b,
            agent_b,
            output=args.output,
        )
    return 0


def _batch_trace_agents(
    name_a: str,
    agent_a: RandomAgent | MctsAgent,
    name_b: str,
    agent_b: RandomAgent | MctsAgent,
) -> tuple[list[dict[str, object]], bool]:
    if name_a == name_b:
        if agent_a != agent_b:
            raise ValueError(
                f'batch agents share the name "{name_a}" but have different configurations; '
                "use distinct profile names"
            )
        description = _batch_agent_dict(name_a, agent_a)
        description["self_play"] = True
        return [description], True

    descriptions = []
    for name, agent in ((name_a, agent_a), (name_b, agent_b)):
        description = _batch_agent_dict(name, agent)
        description["self_play"] = False
        descriptions.append(description)
    return descriptions, False


def _batch_agent(
    kind: str,
    config_path: Path | None,
    option: str,
) -> tuple[RandomAgent | MctsAgent, str]:
    if kind == "random":
        if config_path is not None:
            raise ValueError(f"{option} can only be used with an MCTS agent")
        return RandomAgent(), "random"
    if config_path is None:
        raise ValueError(f"{option} is required for an MCTS agent")
    profile = _load_mcts_profile(config_path)
    return profile.agent, profile.name


def _analyze_profiles(
    paths: Sequence[Path],
    inline_specs: Sequence[str],
    game: TicTacToe | ConnectFour | Boop | SpiritsOfTheForest,
) -> tuple[_MctsProfile, ...]:
    profiles = tuple(_load_mcts_profile(path) for path in paths) + tuple(
        _parse_inline_mcts_profile(spec) for spec in inline_specs
    )
    names = [profile.name for profile in profiles]
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        raise ValueError(
            "analyze agent profile names must be unique: " + ", ".join(duplicates)
        )
    for profile in profiles:
        Match(game=game, first=profile.agent, second=RandomAgent())
    return profiles


def _print_batch_setup(
    batch: Batch,
    game: TicTacToe | ConnectFour | Boop | SpiritsOfTheForest,
    name_a: str,
    agent_a: RandomAgent | MctsAgent,
    name_b: str,
    agent_b: RandomAgent | MctsAgent,
) -> None:
    print(
        f"Starting batch: {_game_name(game)}, {batch.matches} matches, "
        f"alternate sides: {'yes' if batch.alternate_sides else 'no'}, "
        f"workers: {min(resolve_workers(batch.workers), batch.matches)}",
        file=sys.stderr,
    )
    print(f"Agent A: {_batch_agent_description(name_a, agent_a)}", file=sys.stderr)
    print(f"Agent B: {_batch_agent_description(name_b, agent_b)}", file=sys.stderr)
    print(file=sys.stderr, flush=True)


def _print_batch_progress(event: BatchProgress, name_a: str, name_b: str) -> None:
    prefix = f"[{event.match_number}/{event.total_matches}]"
    player_a = event.agent_a_player
    player_b = 1 - player_a
    if event.status is BatchProgressStatus.STARTED:
        print(
            f"{prefix} starting: A ({name_a})=P{player_a}, "
            f"B ({name_b})=P{player_b}, seed={event.seed}",
            file=sys.stderr,
            flush=True,
        )
        return

    result = event.result
    if result is None:
        raise RuntimeError("completed batch progress is missing its match result")
    winner = (
        "draw"
        if result.winner is None
        else f"A ({name_a})"
        if result.winner == 0
        else f"B ({name_b})"
    )
    print(
        f"{prefix} completed: winner={winner}, plies={result.plies}, "
        f"match={result.duration_seconds:.3f}s, elapsed={event.elapsed_seconds:.3f}s",
        file=sys.stderr,
        flush=True,
    )


def _batch_agent_description(name: str, agent: RandomAgent | MctsAgent) -> str:
    if isinstance(agent, RandomAgent):
        return name
    return (
        f"{name} ({_mcts_budget_description(agent)}, "
        f"rollout_depth={agent.rollout_depth}, "
        f"exploration={agent.exploration:.6f}, "
        f"cutoff={_evaluator_name(agent.cutoff_evaluator)}, "
        f"rollout={_rollout_policy_description(agent)}, "
        f"progressive_bias={_progressive_bias_description(agent.progressive_bias)}, "
        f"tree_reuse={'yes' if agent.tree_reuse else 'no'}, "
        f"transpositions={'yes' if agent.transpositions else 'no'})"
    )


def _progressive_bias_description(bias: ProgressiveBias | None) -> str:
    if bias is None:
        return "none"
    condition = "always" if bias.condition is None else bias.condition.phase
    return (
        f"weight={bias.weight:g}/{_evaluator_name(bias.evaluator)}/"
        f"condition={condition}"
    )


def _mcts_budget_description(agent: MctsAgent) -> str:
    if agent.iterations is not None:
        return f"iterations={agent.iterations}"
    return f"time_budget={agent.time_budget:g}s"


def _batch_dict(
    result: BatchResult,
    game: TicTacToe | ConnectFour | Boop | SpiritsOfTheForest,
    name_a: str,
    agent_a: RandomAgent | MctsAgent,
    name_b: str,
    agent_b: RandomAgent | MctsAgent,
    *,
    output: Path | None = None,
) -> dict[str, object]:
    payload = {
        "game": _game_name(game),
        "seed": result.seed,
        "matches": result.matches,
        "workers": result.workers,
        "alternate_sides": result.alternate_sides,
        "agents": {
            "a": _batch_agent_dict(name_a, agent_a),
            "b": _batch_agent_dict(name_b, agent_b),
        },
        "summary": {
            "agent_a_wins": result.agent_a_wins,
            "agent_b_wins": result.agent_b_wins,
            "draws": result.draws,
            "total_plies": result.total_plies,
            "average_plies": result.average_plies,
            "elapsed_seconds": result.elapsed_seconds,
        },
        "games": [
            {
                "match_number": game_result.match_number,
                "seed": game_result.seed,
                "agent_a_player": game_result.agent_a_player,
                "winner": (
                    None
                    if game_result.winner is None
                    else "agent_a"
                    if game_result.winner == 0
                    else "agent_b"
                ),
                "plies": game_result.plies,
                "utilities": list(game_result.utilities),
                "duration_seconds": game_result.duration_seconds,
            }
            for game_result in result.games
        ],
    }
    if output is not None:
        payload["output"] = str(output)
    return payload


def _print_batch_result(
    result: BatchResult,
    game: TicTacToe | ConnectFour | Boop | SpiritsOfTheForest,
    name_a: str,
    agent_a: RandomAgent | MctsAgent,
    name_b: str,
    agent_b: RandomAgent | MctsAgent,
    *,
    output: Path | None = None,
) -> None:
    print(f"Game: {_game_name(game)}")
    print(f"Matches: {result.matches}")
    print(f"Workers: {result.workers}")
    print(f"Alternate sides: {'yes' if result.alternate_sides else 'no'}")
    print(f"Agent A: {_batch_agent_description(name_a, agent_a)}")
    print(f"Agent B: {_batch_agent_description(name_b, agent_b)}")
    if output is not None:
        print(f"Trace: {output}")
    print()
    print("Results:")
    print(
        f"  Agent A wins: {result.agent_a_wins} "
        f"({result.agent_a_wins / result.matches:.1%})"
    )
    print(
        f"  Agent B wins: {result.agent_b_wins} "
        f"({result.agent_b_wins / result.matches:.1%})"
    )
    print(f"  Draws: {result.draws} ({result.draws / result.matches:.1%})")
    print(f"  Average plies: {result.average_plies:.1f}")
    print(f"  Total time: {result.elapsed_seconds:.3f}s")


def _game(name: str) -> TicTacToe | ConnectFour | Boop | SpiritsOfTheForest:
    if name == "boop":
        return Boop()
    if name == "connect-four":
        return ConnectFour()
    if name == "spotf":
        return SpiritsOfTheForest()
    return TicTacToe()


def _mcts_configuration(args: argparse.Namespace) -> MctsAgent:
    return MctsAgent(
        iterations=(
            args.mcts_iterations
            if args.mcts_iterations is not None
            else None if args.mcts_time_budget is not None else 1_000
        ),
        time_budget=args.mcts_time_budget,
        exploration=args.mcts_exploration,
        rollout_depth=(
            256 if args.mcts_rollout_depth is None else args.mcts_rollout_depth
        ),
        rollout_policy=_configured_rollout_policy(
            {
                "rollout_policy": args.mcts_rollout_policy,
                **(
                    {"rollout_epsilon": args.mcts_rollout_epsilon}
                    if args.mcts_rollout_epsilon is not None
                    else {}
                ),
                **(
                    {"rollout_heuristic_index": args.mcts_rollout_heuristic}
                    if args.mcts_rollout_heuristic is not None
                    else {}
                ),
            },
            "manual MCTS configuration",
        ),
        tree_reuse=args.mcts_tree_reuse,
        transpositions=args.mcts_transpositions,
    )


def _match_agent(
    name: str,
    config_path: Path | None,
    manual_mcts: MctsAgent,
    heuristic: int | None,
    config_option: str,
    heuristic_option: str,
) -> HumanAgent | MctsAgent | RandomAgent:
    if config_path is None:
        return _agent(name, manual_mcts, heuristic, heuristic_option)
    if name != "mcts":
        raise ValueError(f"{config_option} requires the corresponding player to be MCTS")
    if heuristic is not None:
        raise ValueError(f"{config_option} cannot be combined with {heuristic_option}")
    return _load_mcts_profile(config_path).agent


def _agent(
    name: str,
    mcts: MctsAgent,
    heuristic: int | None,
    option: str,
) -> HumanAgent | MctsAgent | RandomAgent:
    if heuristic is not None and name != "mcts":
        raise ValueError(f"{option} requires the corresponding player to be MCTS")
    if name == "human":
        return HumanAgent(observe_action=_print_human_move)
    if name == "mcts":
        return MctsAgent(
            iterations=mcts.iterations,
            time_budget=mcts.time_budget,
            exploration=mcts.exploration,
            rollout_depth=mcts.rollout_depth,
            cutoff_evaluator=(
                NeutralEvaluator() if heuristic is None else GameHeuristic(heuristic)
            ),
            rollout_policy=mcts.rollout_policy,
            progressive_bias=mcts.progressive_bias,
            root_diagnostics=mcts.root_diagnostics,
            tree_reuse=mcts.tree_reuse,
            transpositions=mcts.transpositions,
        )
    return RandomAgent()


def _agent_dict(name: str, agent) -> dict[str, object]:
    cutoff_evaluator = agent.cutoff_evaluator if isinstance(agent, MctsAgent) else None
    rollout_evaluator = (
        _rollout_policy_evaluator(agent.rollout_policy)
        if isinstance(agent, MctsAgent)
        else None
    )
    return {
        "type": name,
        "iterations": agent.iterations if isinstance(agent, MctsAgent) else None,
        "time_budget": agent.time_budget if isinstance(agent, MctsAgent) else None,
        "rollout_depth": agent.rollout_depth if isinstance(agent, MctsAgent) else None,
        "heuristic": agent.heuristic if isinstance(agent, MctsAgent) else None,
        "cutoff_evaluator": _evaluator_dict(cutoff_evaluator),
        "rollout_policy": (
            _rollout_policy_name(agent) if isinstance(agent, MctsAgent) else None
        ),
        "rollout_evaluator": _evaluator_dict(rollout_evaluator),
        "rollout_epsilon": (
            _rollout_policy_epsilon(agent.rollout_policy)
            if isinstance(agent, MctsAgent)
            else None
        ),
        **(
            _conditional_rollout_fields(agent.rollout_policy)
            if isinstance(agent, MctsAgent)
            else {}
        ),
        **(
            _progressive_bias_fields(agent.progressive_bias)
            if isinstance(agent, MctsAgent)
            else _progressive_bias_fields(None)
        ),
        "root_diagnostics": (
            agent.root_diagnostics if isinstance(agent, MctsAgent) else False
        ),
        "tree_reuse": agent.tree_reuse if isinstance(agent, MctsAgent) else False,
        "transpositions": (
            agent.transpositions if isinstance(agent, MctsAgent) else False
        ),
    }


def _heuristic_name(heuristic: int | None) -> str:
    return "none" if heuristic is None else str(heuristic)


def _evaluator_name(evaluator: NeutralEvaluator | GameHeuristic | None) -> str:
    if isinstance(evaluator, GameHeuristic):
        params = ""
        if evaluator.params:
            assignments = ",".join(
                f"{name}={value:g}" for name, value in evaluator.params.items()
            )
            params = f";{assignments}"
        return f"game_heuristic({evaluator.index}{params})"
    return "neutral" if evaluator is not None else "none"


def _serialized_evaluator_name(evaluator: object) -> str:
    if evaluator is None:
        return "none"
    if not isinstance(evaluator, dict):
        raise TypeError("serialized evaluator must be an object")
    kind = evaluator.get("kind")
    if kind == "game_heuristic":
        params = evaluator.get("params")
        suffix = ""
        if isinstance(params, dict) and params:
            assignments = ",".join(
                f"{name}={value}" for name, value in params.items()
            )
            suffix = f";{assignments}"
        return f"game_heuristic({evaluator.get('index')}{suffix})"
    return str(kind)


def _serialized_rollout_policy_description(fields: dict[str, object]) -> str:
    if fields["rollout_policy"] != "conditional":
        policy = str(fields["rollout_policy"])
        if fields["rollout_epsilon"] is not None:
            policy += f"(epsilon={fields['rollout_epsilon']})"
        evaluator = _serialized_evaluator_name(fields["rollout_evaluator"])
        return policy if evaluator == "none" else f"{policy}, evaluator={evaluator}"

    condition = fields["rollout_condition"]
    if not isinstance(condition, dict):
        raise TypeError("serialized conditional rollout requires a condition")
    primary = str(fields["rollout_primary_policy"])
    if fields["rollout_epsilon"] is not None:
        primary += f"(epsilon={fields['rollout_epsilon']})"
    primary_evaluator = _serialized_evaluator_name(fields["rollout_evaluator"])
    if primary_evaluator != "none":
        primary += f", evaluator={primary_evaluator}"
    fallback = str(fields["rollout_fallback_policy"])
    fallback_epsilon = fields["rollout_fallback_epsilon"]
    if fallback_epsilon is not None:
        fallback += f"(epsilon={fallback_epsilon})"
    fallback_evaluator = _serialized_evaluator_name(
        fields["rollout_fallback_evaluator"]
    )
    if fallback_evaluator != "none":
        fallback += f", evaluator={fallback_evaluator}"
    return f"conditional({condition.get('phase')} ? {primary} : {fallback})"


def _rollout_policy_description(agent: MctsAgent) -> str:
    if isinstance(agent.rollout_policy, ConditionalRollout):
        primary = _base_rollout_policy_name(agent.rollout_policy.primary)
        evaluator = _rollout_policy_evaluator(agent.rollout_policy.primary)
        if evaluator is not None:
            primary += f"/{_evaluator_name(evaluator)}"
        epsilon = _rollout_policy_epsilon(agent.rollout_policy.primary)
        if epsilon is not None:
            primary += f"/epsilon={epsilon}"
        fallback = _base_rollout_policy_name(agent.rollout_policy.fallback)
        return (
            f"conditional/{agent.rollout_policy.condition.phase}"
            f"?{primary}:{fallback}"
        )
    policy = _rollout_policy_name(agent)
    evaluator = _rollout_policy_evaluator(agent.rollout_policy)
    if evaluator is not None:
        policy += f"/{_evaluator_name(evaluator)}"
    epsilon = _rollout_policy_epsilon(agent.rollout_policy)
    if epsilon is not None:
        policy += f"/epsilon={epsilon}"
    return policy


def _print_result(
    result: MatchResult,
    first: str,
    second: str,
    first_agent,
    second_agent,
) -> None:
    print(f"Player 0: {_agent_name(first, first_agent)}")
    print(f"Player 1: {_agent_name(second, second_agent)}")
    print()
    for ply, move in enumerate(result.moves, start=1):
        if isinstance(move.action, TicTacToeAction):
            selected = f"row {move.action.row}, column {move.action.column}"
        elif isinstance(move.action, ConnectFourAction):
            selected = f"column {move.action.column}"
        elif isinstance(
            move.action,
            (
                TakeSpiritTile,
                EndSpiritCollection,
                PlaceSpiritGemstone,
                MoveSpiritGemstone,
                SkipSpiritGemstone,
            ),
        ):
            from .api import _spirits_action_description

            selected = _spirits_action_description(move.action)
        else:
            selected = (
                f"{move.action.piece.value} at row {move.action.row}, "
                f"column {move.action.column}"
            )
            resolution = _resolution_text(move.action)
            if resolution:
                selected += f"; {resolution}"
        print(f"{ply}. Player {move.player} -> {selected}")
    print()
    print("Final board:")
    _print_board(result.final_board)
    if result.pools is not None:
        for player, pool in enumerate(result.pools):
            print(f"Player {player} pool: {pool.kittens} kittens, {pool.cats} cats")
    if result.scores is not None:
        print(f"Scores: player 0 = {result.scores[0]}, player 1 = {result.scores[1]}")
    print()
    print("Result: draw" if result.winner is None else f"Winner: player {result.winner}")
    print(f"Utilities: {list(result.utilities)}")
    print(f"Plies: {result.plies}")
    print(f"Seed: {result.seed}")


def _agent_name(name: str, agent) -> str:
    if isinstance(agent, MctsAgent):
        return (
            f"{name} (cutoff {_evaluator_name(agent.cutoff_evaluator)}, "
            f"rollout {_rollout_policy_description(agent)}, "
            f"progressive bias {_progressive_bias_description(agent.progressive_bias)})"
        )
    return name


def _print_board(board) -> None:
    print("    " + " ".join(str(column) for column in range(len(board[0]))))
    for row, cells in enumerate(board):
        print(f"{row} | " + " ".join(_piece_symbol(cell) for cell in cells))


def _print_human_move(observation: HumanMoveObservation) -> None:
    print(file=sys.stderr)
    print(f"Board after player {observation.player}'s move:", file=sys.stderr)
    print(
        "    " + " ".join(str(column) for column in range(len(observation.board[0]))),
        file=sys.stderr,
    )
    for row, cells in enumerate(observation.board):
        rendered = " ".join(_piece_symbol(cell) for cell in cells)
        print(f"{row} | {rendered}", file=sys.stderr)
    if observation.pools is not None:
        for player, pool in enumerate(observation.pools):
            print(
                f"Player {player} pool: {pool.kittens} kittens, {pool.cats} cats",
                file=sys.stderr,
            )


def _piece_symbol(piece) -> str:
    if piece is None:
        return "."
    if isinstance(piece, int):
        return "X" if piece == 0 else "O"
    if isinstance(piece, SpiritTile):
        gemstone = "" if piece.gemstone is None else str(piece.gemstone)
        return f"{piece.spirit.value[:2].upper()}{piece.spirit_symbols}{gemstone}"
    if not isinstance(piece, BoopPiece):
        raise TypeError("unknown board piece")
    if piece.player == 0:
        return "x" if piece.kind is BoopPieceKind.KITTEN else "X"
    return "o" if piece.kind is BoopPieceKind.KITTEN else "O"


def _resolution_text(action: BoopAction) -> str:
    if isinstance(action.resolution, BoopGraduateLine):
        positions = ", ".join(
            f"({position.row}, {position.column})"
            for position in action.resolution.positions
        )
        return f"graduate {positions}"
    if isinstance(action.resolution, BoopRecoverPiece):
        position = action.resolution.position
        return f"recover ({position.row}, {position.column})"
    return ""


def _evaluation_dict(
    report: GameEvaluationReport,
    configured_benchmarks: Sequence[_ConfiguredMctsBenchmark] = (),
) -> dict[str, object]:
    return {
        "game": _game_name(report.game),
        "samples": report.samples,
        "max_depth": report.max_depth,
        "terminal_rate": report.terminal_rate,
        "initial_legal_actions": report.initial_legal_actions,
        "effective_branching_factor": report.effective_branching_factor,
        "player_turn_choice_product_log10": report.player_turn_choice_product_log10,
        "depth_p50": report.depth_p50,
        "estimated_depth": report.estimated_depth,
        "player_turn_depth_p50": report.player_turn_depth_p50,
        "player_turn_depth_p95": report.player_turn_depth_p95,
        "player_changes_p50": report.player_changes_p50,
        "player_changes_p95": report.player_changes_p95,
        "actions_per_player_turn_mean": report.actions_per_player_turn_mean,
        "actions_per_player_turn_p95": report.actions_per_player_turn_p95,
        "actions_per_player_turn_max": report.actions_per_player_turn_max,
        "depth_is_lower_bound": report.depth_is_lower_bound,
        "estimated_tree_log10": report.estimated_tree_log10,
        "calibration_positions": report.calibration_positions,
        "target_time_seconds": report.target_time_seconds,
        "rollout_costs": [
            {
                "rollout_depth": cost.rollout_depth,
                "approximate_player_turns": cost.approximate_player_turns,
                "milliseconds_per_iteration": cost.milliseconds_per_iteration,
                "iteration_budgets": [
                    {
                        "seconds": budget.seconds,
                        "iterations": budget.iterations,
                    }
                    for budget in cost.iteration_budgets
                ],
            }
            for cost in report.rollout_costs
        ],
        "suggested_experiments": [
            {
                "label": experiment.label,
                "iterations": experiment.iterations,
                "iterations_capped": experiment.iterations_capped,
                "rollout_depth": experiment.rollout_depth,
                "approximate_player_turns": experiment.approximate_player_turns,
                "estimated_decision_time_ms": experiment.estimated_decision_time_ms,
            }
            for experiment in report.suggested_experiments
        ],
        "configured_agent_benchmarks": _configured_benchmark_dicts(
            report,
            configured_benchmarks,
        ),
        "recommended_rollout_depth": report.recommended_rollout_depth,
        "recommended_iterations": report.recommended_iterations,
        "iterations_capped": report.iterations_capped,
        "milliseconds_per_iteration": report.milliseconds_per_iteration,
        "estimated_decision_time_ms": report.estimated_decision_time_ms,
    }


def _configured_benchmark_dicts(
    report: GameEvaluationReport,
    configured_benchmarks: Sequence[_ConfiguredMctsBenchmark],
) -> list[dict[str, object]]:
    ranked = sorted(
        configured_benchmarks,
        key=lambda configured: configured.benchmark.decision_time_mean_ms,
    )
    if not ranked:
        return []
    fastest_ms = ranked[0].benchmark.decision_time_mean_ms
    target_ms = report.target_time_seconds * 1_000.0
    rows = []
    for rank, configured in enumerate(ranked, start=1):
        benchmark = configured.benchmark
        agent = benchmark.agent
        rows.append(
            {
                "rank": rank,
                "name": configured.name,
                "iterations": agent.iterations,
                "time_budget": agent.time_budget,
                "rollout_depth": agent.rollout_depth,
                "exploration": agent.exploration,
                "heuristic": agent.heuristic,
                "cutoff_evaluator": _evaluator_dict(agent.cutoff_evaluator),
                "rollout_policy": _rollout_policy_name(agent),
                "rollout_evaluator": _evaluator_dict(
                    _rollout_policy_evaluator(agent.rollout_policy)
                ),
                "rollout_epsilon": _rollout_policy_epsilon(agent.rollout_policy),
                **_conditional_rollout_fields(agent.rollout_policy),
                **_progressive_bias_fields(agent.progressive_bias),
                "root_diagnostics": agent.root_diagnostics,
                "tree_reuse": agent.tree_reuse,
                "transpositions": agent.transpositions,
                "sampled_positions": benchmark.sampled_positions,
                "decision_time_mean_ms": benchmark.decision_time_mean_ms,
                "decision_time_p50_ms": benchmark.decision_time_p50_ms,
                "decision_time_p95_ms": benchmark.decision_time_p95_ms,
                "decision_time_max_ms": benchmark.decision_time_max_ms,
                "milliseconds_per_iteration": benchmark.milliseconds_per_iteration,
                "relative_to_fastest": benchmark.decision_time_mean_ms / fastest_ms,
                "target_time_ratio": benchmark.decision_time_mean_ms / target_ms,
                "position_timings": [
                    {
                        "sampled_ply": timing.sampled_ply,
                        "milliseconds": timing.milliseconds,
                        "iterations": timing.iterations,
                        "nodes": timing.nodes,
                    }
                    for timing in benchmark.position_timings
                ],
            }
        )
    return rows


def _print_evaluation(
    report: GameEvaluationReport,
    configured_benchmarks: Sequence[_ConfiguredMctsBenchmark] = (),
) -> None:
    depth_note = " (lower bound)" if report.depth_is_lower_bound else ""
    choices_log10 = report.player_turn_choice_product_log10
    choices_text = (
        f"~{10**choices_log10:,.1f} (10^{choices_log10:.2f})"
        if choices_log10 <= 6
        else f"~10^{choices_log10:.2f}"
    )

    print(f"Game: {_game_name(report.game)}")
    print()
    print("Structural complexity:")
    print(f"  Initial legal actions: {report.initial_legal_actions}")
    print(f"  Branching per tree node: {report.effective_branching_factor:.2f}")
    print(
        f"  Observed choices across one player turn: {choices_text} "
        "(geometric mean of sampled phase products)"
    )
    print(f"  Terminal samples: {report.terminal_rate:.1%}")
    print(f"  Estimated tree size: 10^{report.estimated_tree_log10:.1f}")
    print()
    print("Depth structure:")
    print(f"  Tree depth p50: {report.depth_p50} plies{depth_note}")
    print(f"  Tree depth p95: {report.estimated_depth} plies{depth_note}")
    print(f"  Player-turn depth p50: {report.player_turn_depth_p50}")
    print(f"  Player-turn depth p95: {report.player_turn_depth_p95}")
    print(
        f"  Player changes p50 / p95: {report.player_changes_p50} / "
        f"{report.player_changes_p95}"
    )
    print("  Actions per player turn:")
    print(f"    mean: {report.actions_per_player_turn_mean:.2f}")
    print(f"    p95: {report.actions_per_player_turn_p95}")
    print(f"    max sampled: {report.actions_per_player_turn_max}")
    print()
    print("Practical MCTS configuration:")
    print(
        f"  Measured at {report.calibration_positions} sampled position(s); "
        "timings are machine-dependent approximations."
    )
    print("  Candidate rollout depths and approximate iteration budgets:")
    for cost in report.rollout_costs:
        budgets = ", ".join(
            f"{budget.seconds}s={budget.iterations:,}" for budget in cost.iteration_budgets
        )
        print(
            f"    depth {cost.rollout_depth:<4} ≈ {cost.approximate_player_turns:>5.1f} "
            f"player turns | ~{cost.milliseconds_per_iteration:.4f} ms/iteration "
            f"| {budgets}"
        )
    print()
    print("Suggested starting experiments (benchmark points, not strength guarantees):")
    for experiment in report.suggested_experiments:
        cap_note = " (capped)" if experiment.iterations_capped else ""
        print(
            f"  {experiment.label}: {experiment.iterations:,} iterations{cap_note}, "
            f"depth {experiment.rollout_depth} ≈ "
            f"{experiment.approximate_player_turns:.1f} player turns, "
            f"~{experiment.estimated_decision_time_ms / 1_000:.2f} s"
        )
    if configured_benchmarks:
        print()
        print("Configured MCTS benchmarks (fastest to slowest):")
        print(
            "  Exact profiles measured sequentially on the same sampled positions; "
            "timings are isolated latencies."
        )
        rows = _configured_benchmark_dicts(report, configured_benchmarks)
        for row in rows:
            rollout_policy = _serialized_rollout_policy_description(row)
            timings = ", ".join(
                f"ply {timing['sampled_ply']}={timing['milliseconds']:.2f} ms/"
                f"{timing['iterations']:,}i/{timing['nodes']:,} nodes"
                for timing in row["position_timings"]
            )
            budget = (
                f"{row['iterations']:,} iterations"
                if row["iterations"] is not None
                else f"{row['time_budget']:g}s time budget"
            )
            print(
                f"  {row['rank']}. {row['name']}: {budget}, "
                f"depth {row['rollout_depth']}, "
                f"cutoff={_serialized_evaluator_name(row['cutoff_evaluator'])}, "
                f"rollout={rollout_policy}"
            )
            print(
                f"     mean={row['decision_time_mean_ms']:.2f} ms, "
                f"p50={row['decision_time_p50_ms']:.2f} ms, "
                f"p95={row['decision_time_p95_ms']:.2f} ms, "
                f"max={row['decision_time_max_ms']:.2f} ms, "
                f"relative={row['relative_to_fastest']:.2f}x"
            )
            print(
                f"     ~{row['milliseconds_per_iteration']:.6f} ms/iteration | "
                f"{timings}"
            )
        target_ms = report.target_time_seconds * 1_000.0
        closest = min(
            rows,
            key=lambda row: abs(row["decision_time_mean_ms"] - target_ms),
        )
        print(
            f"  Closest to the {report.target_time_seconds:g}s target: "
            f"{closest['name']} ({closest['decision_time_mean_ms'] / 1_000:.3f}s mean)."
        )
    print()
    balanced = next(
        experiment
        for experiment in report.suggested_experiments
        if experiment.label == "Balanced"
    )
    comparison_iterations = " / ".join(
        f"{experiment.iterations:,}"
        for experiment in report.suggested_experiments
        if experiment.label in {"Fast", "Balanced", "Wide"}
    )
    print("Next experiment:")
    print(
        f"  Compare {comparison_iterations} iterations at depth "
        f"{balanced.rollout_depth}."
    )
    print(
        "  If extra iterations stop improving results, compare the Deep point; "
        "if that also plateaus, introduce or improve the state heuristic."
    )
    print(
        "  If Deep beats Balanced at similar time, the game is more horizon/heuristic "
        "constrained; if only Wide improves, it is more search-width constrained."
    )
    if report.depth_is_lower_bound:
        print(
            "  Some samples did not finish: increase --max-depth before treating "
            "the full-depth row as representative."
        )
