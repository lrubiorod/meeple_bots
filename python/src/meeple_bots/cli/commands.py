"""Command dispatch and orchestration; configuration and views have separate owners."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .._mcts_profiles import (
    _MctsProfile, _configured_rollout_policy, _load_mcts_profile,
    _parse_inline_mcts_profile,
)
from ..api import (
    Batch, BatchProgress, BatchProgressStatus, BatchResult, Boop, ConnectFour,
    GameHeuristic, HumanAgent, Match, MctsAgent, NeutralEvaluator, RandomAgent,
    SoIsmctsAgent, SpiritsOfTheForest, TicTacToe,
)
from .._concurrency import resolve_workers
from ..extraction import extract_tournament
from ..game_config import create_game, game_parameters
from ..gui import run_gui
from ..reporting import generate_study_report
from ..serialization import (
    match_result_dict as _result_dict, agent_dict as _batch_agent_dict,
    _progressive_bias_fields, _rollout_policy_name, _rollout_policy_evaluator,
    _rollout_policy_epsilon, _conditional_rollout_fields, _evaluator_dict,
    game_name as _game_name, trace_match_dict as _trace_match_dict,
    write_jsonl as _write_jsonl,
)
from ..tournament_config import _load_tournament_config
from ..tournaments import run_tournament
from .analyze_compat import analysis_json
from .display import (
    _print_extraction_summary, _print_report_summary, _print_tournament_summary,
    _print_tournament_start, _print_tournament_progress,
    _print_batch_setup, _print_batch_progress, _print_batch_result,
    _print_result, _print_human_move,
)
from .parser import build_parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "probe":
            from ..probes.cli import run as run_probe
            return run_probe(args)
        if args.command == "gui":
            run_gui(
                game=args.game,
                host=args.host,
                port=args.port,
                open_browser=not args.no_browser,
            )
            return 0
        if args.command == "study":
            return _run_study(args)
        if args.command == "tournament":
            return _run_tournament(args)
        if args.command == "extract":
            return _run_extract(args)
        if args.command == "report":
            return _run_report(args)
        game = create_game(args.game, args.game_params)
        if args.command == "analyze":
            return _run_analyze(args, game)
        if args.command == "batch":
            return _run_batch(args, game)

        result, first, second = _run_match(args, game, argv)
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


def _run_study(args: argparse.Namespace):
    from ..studies import duration_seconds, run_study
    output = args.output or Path("results/studies") / f"{args.game}-study"
    stage_games = {}
    for entry in args.stage_games:
        stage, separator, count = entry.partition("=")
        if not separator or stage in stage_games:
            raise ValueError("--stage-games requires unique STAGE=GAMES entries")
        stage_games[stage] = int(count)
    if args.max_pairs is not None and args.games_per_comparison != 50:
        raise ValueError("use either --games-per-comparison or --max-pairs")
    result = run_study(args.game, output=output, budget=duration_seconds(args.budget) if args.budget else None,
                       baseline=args.baseline, agent_family=args.agent, seed=args.seed, tune=args.tune, second_pass=args.second_pass, widening_expansion_search=args.widening_expansion_search,
                       heuristic=args.heuristic, rave_search=args.rave_search, pw_search=args.pw_search,
                       selection_search=args.selection_search, mechanism_search=args.mechanism_search, depth_search=args.depth_search, all_search=args.all_search, vs_random=args.vs_random, target_match_time=duration_seconds(args.target_match_time),
                       games_per_comparison=args.games_per_comparison, stage_games=stage_games,
                       max_pairs=args.max_pairs, decision_seconds=args.decision_time,
                       screening_seconds=args.screening_time,
                       max_plies=args.max_plies, workers=args.workers, resume=args.resume, allow_engine_change=args.allow_engine_change, game_params=args.game_params,
                       progress=lambda message: print(message, file=sys.stderr, flush=True))
    summary = {"status": result["status"], "spent_seconds": result["spent_seconds"],
               "report": str(output.resolve() / "report.html"),
               "candidate_profiles": result.get("candidate_profiles", {}),
               "budget_limited": result.get("budget_limited", False),
               "local_retune": result.get("local_retune")}
    print(json.dumps(summary, indent=2) if args.json else f"Study {summary['status']}: {summary['report']}")
    return 0


def _run_extract(args: argparse.Namespace):
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


def _run_report(args: argparse.Namespace):
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


def _run_analyze(args: argparse.Namespace, game):
    from ..analysis import analyze_game, print_analysis
    from .._search_profiles import resolve_family
    from ..api import _native_game
    inline = [x for x in args.agent if x not in ("mcts", "so_ismcts")]
    families = [x for x in args.agent if x in ("mcts", "so_ismcts")]
    if len(set(families + ([args.search_family] if args.search_family else []))) > 1:
        raise ValueError("conflicting search families")
    family = args.search_family or (families[0] if families else None)
    profiles = _analyze_profiles(args.agent_config, inline, game)
    family = resolve_family(_native_game(game), family, profiles[0].agent if profiles else None)
    for profile in profiles:
        print(f"Benchmarking {profile.name}", file=sys.stderr, flush=True)
    analysis = analyze_game(game, args.samples, args.max_depth, args.seed,
                            args.target_time, family,
                            [(p.name, p.agent) for p in profiles], args.target_match_time)
    data = analysis_json(analysis, profiles, game)
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print_analysis(analysis)
    return 0


def _run_match(args: argparse.Namespace, game, argv: Sequence[str] | None):
    using_so = "so_ismcts" in (args.first, args.second)
    supplied = list(sys.argv[1:] if argv is None else argv)
    if using_so and any(a.startswith(("--mcts-", "--first-mcts-", "--second-mcts-")) for a in supplied):
        raise ValueError("MCTS options are not supported by SO-ISMCTS; use --so-ismcts-iterations and --so-ismcts-exploration")
    if not using_so and any(a.startswith("--so-ismcts-") for a in supplied):
        raise ValueError("SO-ISMCTS options require a so_ismcts participant")
    mcts = _mcts_configuration(args)
    first = SoIsmctsAgent(args.so_ismcts_iterations, args.so_ismcts_exploration, selection_policy=args.so_ismcts_selection_policy, tree_reuse=args.so_ismcts_tree_reuse) if args.first == "so_ismcts" and args.first_mcts_config is None else _match_agent(
        args.first,
        args.first_mcts_config,
        mcts,
        args.first_mcts_heuristic,
        "--first-mcts-config",
        "--first-mcts-heuristic",
    )
    second = SoIsmctsAgent(args.so_ismcts_iterations, args.so_ismcts_exploration, selection_policy=args.so_ismcts_selection_policy, tree_reuse=args.so_ismcts_tree_reuse) if args.second == "so_ismcts" and args.second_mcts_config is None else _match_agent(
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
    return result, first, second


def _run_tournament(args: argparse.Namespace) -> int:
    config = _load_tournament_config(args.config)
    total_matches = 0

    def started(header: dict) -> None:
        nonlocal total_matches
        total_matches = header["total_matches"]
        _print_tournament_start(header)

    def completed(row: dict) -> None:
        _print_tournament_progress(row, total_matches)

    summary = run_tournament(
        config, output=args.output, workers=args.workers, overwrite=args.overwrite,
        on_start=started, on_match=completed,
    )
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_tournament_summary(summary)
    return 0



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
                    **({"game_params": game_parameters(game)} if game_parameters(game) else {}),
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
    from .._search_profiles import load_named_search_profile
    profiles = tuple(load_named_search_profile(path) for path in paths) + tuple(
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
                    **({"game_params": game_parameters(game)} if game_parameters(game) else {}),
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


def _game(name: str):
    return create_game(name)


def _mcts_configuration(args: argparse.Namespace) -> MctsAgent:
    return MctsAgent(
        iterations=(
            args.mcts_iterations
            if args.mcts_iterations is not None
            else None if args.mcts_time_budget is not None else 1_000
        ),
        time_budget=args.mcts_time_budget,
        exploration=args.mcts_exploration,
        selection_policy=args.mcts_selection_policy,
        rave_equivalence=args.mcts_rave_equivalence,
        progressive_widening=args.mcts_progressive_widening,
        progressive_widening_k=args.mcts_progressive_widening_k,
        progressive_widening_alpha=args.mcts_progressive_widening_alpha,
        progressive_widening_expansion=args.mcts_progressive_widening_expansion,
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
    if name == "so_ismcts":
        from .._study_profiles import load_search_profile
        agent = load_search_profile(config_path)
        if not isinstance(agent, SoIsmctsAgent):
            raise ValueError("agent config family must match so_ismcts")
        if heuristic is not None:
            raise ValueError("SO-ISMCTS does not support heuristics")
        return agent
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
            selection_policy=mcts.selection_policy,
            rave_equivalence=mcts.rave_equivalence,
            progressive_widening=mcts.progressive_widening,
            progressive_widening_k=mcts.progressive_widening_k,
            progressive_widening_alpha=mcts.progressive_widening_alpha,
            progressive_widening_expansion=mcts.progressive_widening_expansion,
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
    if isinstance(agent, SoIsmctsAgent):
        return {"type": "so_ismcts", "iterations": agent.iterations, "exploration": agent.exploration,
                "selection_policy": agent.selection_policy, "tree_reuse": agent.tree_reuse,
                "rollout_policy": "uniform", "root_selection": "most_visited"}
    cutoff_evaluator = agent.cutoff_evaluator if isinstance(agent, MctsAgent) else None
    rollout_evaluator = (
        _rollout_policy_evaluator(agent.rollout_policy)
        if isinstance(agent, MctsAgent)
        else None
    )
    return {
        "type": name,
        "selection_policy": agent.selection_policy if isinstance(agent, MctsAgent) else None,
        **({"rave_equivalence": agent.rave_equivalence} if isinstance(agent, MctsAgent) and agent.selection_policy == "uct_rave" else {}),
        **({"progressive_widening": agent.progressive_widening, "progressive_widening_k": agent.progressive_widening_k, "progressive_widening_alpha": agent.progressive_widening_alpha, "progressive_widening_expansion": agent.progressive_widening_expansion} if getattr(agent, "progressive_widening", False) and isinstance(agent, MctsAgent) else {}),
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
