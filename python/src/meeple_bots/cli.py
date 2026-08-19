"""Command-line interface for running observable matches."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

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
    GameEvaluationReport,
    HumanAgent,
    HumanMoveObservation,
    Match,
    MatchResult,
    MctsAgent,
    RandomAgent,
    TicTacToe,
    TicTacToeAction,
    evaluate_game,
)
from .extraction import extract_tournament
from .reporting import generate_study_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meeple-bots")
    commands = parser.add_subparsers(dest="command", required=True)
    match = commands.add_parser("match", help="run and display one match")
    match.add_argument(
        "--game", choices=["boop", "connect-four", "tic-tac-toe"], default="tic-tac-toe"
    )
    match.add_argument("--first", choices=["human", "mcts", "random"], default="mcts")
    match.add_argument("--second", choices=["human", "mcts", "random"], default="random")
    match.add_argument("--seed", type=int, default=0)
    match.add_argument("--max-plies", type=int, default=10_000)
    match.add_argument("--mcts-iterations", type=int)
    match.add_argument("--mcts-exploration", type=float, default=sqrt_two())
    match.add_argument("--mcts-rollout-depth", type=int)
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
        "--game", choices=["boop", "connect-four", "tic-tac-toe"], required=True
    )
    batch.add_argument("--matches", type=int, default=20)
    batch.add_argument("--agent-a", choices=["mcts", "random"], default="random")
    batch.add_argument("--agent-b", choices=["mcts", "random"], default="mcts")
    batch.add_argument("--agent-a-config", type=Path)
    batch.add_argument("--agent-b-config", type=Path)
    batch.add_argument("--seed", type=int, default=0)
    batch.add_argument("--max-plies", type=int, default=10_000)
    batch.add_argument(
        "--no-alternate-sides",
        action="store_false",
        dest="alternate_sides",
        help="keep agent A as player 0 in every match",
    )
    batch.add_argument("--json", action="store_true", help="print machine-readable JSON")

    tournament = commands.add_parser(
        "tournament",
        help="run a configured round-robin tournament and save full match traces",
    )
    tournament.add_argument("--config", type=Path, required=True)
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
        help="extract game-specific analysis tables from a tournament trace",
    )
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument(
        "--output-dir",
        type=Path,
        help="write tables here instead of beside the tournament trace",
    )
    extract.add_argument(
        "--overwrite",
        action="store_true",
        help="replace known extraction files that already exist",
    )
    extract.add_argument("--json", action="store_true", help="print summary as JSON")

    report = commands.add_parser(
        "report",
        help="generate statistics and figures from extracted tournament tables",
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
        "--game", choices=["boop", "connect-four", "tic-tac-toe"], required=True
    )
    analyze.add_argument("--samples", type=int, default=128)
    analyze.add_argument("--max-depth", type=int, default=256)
    analyze.add_argument("--seed", type=int, default=0)
    analyze.add_argument("--json", action="store_true", help="print machine-readable JSON")

    return parser


def sqrt_two() -> float:
    return 2.0**0.5


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
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
            report = evaluate_game(
                game,
                samples=args.samples,
                max_depth=args.max_depth,
                seed=args.seed,
            )
            if args.json:
                print(json.dumps(_evaluation_dict(report), indent=2))
            else:
                _print_evaluation(report)
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
    print(f"Input: {summary['input']}")
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
class _MctsProfile:
    name: str
    agent: MctsAgent


@dataclass(frozen=True, slots=True)
class _TournamentAgent:
    name: str
    agent: RandomAgent | MctsAgent
    self_play: bool


@dataclass(frozen=True, slots=True)
class _TournamentConfig:
    game: TicTacToe | ConnectFour | Boop
    output: Path | None
    matches_per_pair: int
    seed: int
    max_plies: int
    agents: tuple[_TournamentAgent, ...]


def _run_tournament(args: argparse.Namespace) -> int:
    config = _load_tournament_config(args.config)
    output_path = args.output if args.output is not None else config.output
    if output_path is None:
        raise ValueError("tournament output is required in the config or with --output")
    pairings = _tournament_pairings(config.agents)
    total_matches = len(pairings) * config.matches_per_pair
    output_mode = "w" if args.overwrite else "x"
    standings = {
        agent.name: {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "self_play_games": 0,
        }
        for agent in config.agents
    }
    pairing_results = []
    tournament_started = perf_counter()
    match_number = 0

    print(
        f"Starting tournament: {_game_name(config.game)}, {len(config.agents)} agents, "
        f"{len(pairings)} pairings, {total_matches} matches",
        file=sys.stderr,
        flush=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(output_mode, encoding="utf-8") as output:
        _write_jsonl(
            output,
            {
                "record_type": "tournament",
                "schema_version": 1,
                "game": _game_name(config.game),
                "output": str(output_path),
                "matches_per_pair": config.matches_per_pair,
                "seed": config.seed,
                "max_plies": config.max_plies,
                "total_pairings": len(pairings),
                "total_matches": total_matches,
                "agents": [
                    _tournament_agent_dict(agent) for agent in config.agents
                ],
            },
        )

        for pairing_number, (agent_a, agent_b) in enumerate(pairings, start=1):
            self_play = agent_a is agent_b
            agent_a_wins = 0
            agent_b_wins = 0
            draws = 0
            total_plies = 0
            pairing_started = perf_counter()

            for pairing_match_number in range(1, config.matches_per_pair + 1):
                match_number += 1
                agent_a_player = (pairing_match_number - 1) % 2
                first, second = (
                    (agent_a.agent, agent_b.agent)
                    if agent_a_player == 0
                    else (agent_b.agent, agent_a.agent)
                )
                match_seed = (config.seed + match_number - 1) & (2**64 - 1)
                match_started = perf_counter()
                result = Match(
                    game=config.game,
                    first=first,
                    second=second,
                    seed=match_seed,
                    max_plies=config.max_plies,
                ).run()
                duration_seconds = perf_counter() - match_started
                total_plies += result.plies

                if result.winner is None:
                    winner = None
                    draws += 1
                elif result.winner == agent_a_player:
                    winner = "agent_a"
                    agent_a_wins += 1
                else:
                    winner = "agent_b"
                    agent_b_wins += 1

                if self_play:
                    standings[agent_a.name]["self_play_games"] += 1
                else:
                    _update_tournament_standings(
                        standings,
                        agent_a.name,
                        agent_b.name,
                        winner,
                    )

                players = (
                    [agent_a.name, agent_b.name]
                    if agent_a_player == 0
                    else [agent_b.name, agent_a.name]
                )
                _write_jsonl(
                    output,
                    {
                        "record_type": "match",
                        "match_number": match_number,
                        "pairing_number": pairing_number,
                        "pairing_match_number": pairing_match_number,
                        "agent_a": agent_a.name,
                        "agent_b": agent_b.name,
                        "self_play": self_play,
                        "agent_a_player": agent_a_player,
                        "players": players,
                        "winner": winner,
                        "duration_seconds": duration_seconds,
                        "result": _result_dict(result),
                    },
                )
                print(
                    f"[{match_number}/{total_matches}] {agent_a.name} vs {agent_b.name}: "
                    f"winner={winner or 'draw'}, plies={result.plies}, "
                    f"time={duration_seconds:.3f}s",
                    file=sys.stderr,
                    flush=True,
                )

            pairing_results.append(
                {
                    "agent_a": agent_a.name,
                    "agent_b": agent_b.name,
                    "self_play": self_play,
                    "matches": config.matches_per_pair,
                    "agent_a_wins": agent_a_wins,
                    "agent_b_wins": agent_b_wins,
                    "draws": draws,
                    "average_plies": total_plies / config.matches_per_pair,
                    "elapsed_seconds": perf_counter() - pairing_started,
                }
            )

    summary = {
        "game": _game_name(config.game),
        "agents": len(config.agents),
        "pairings": len(pairings),
        "matches": total_matches,
        "matches_per_pair": config.matches_per_pair,
        "seed": config.seed,
        "output": str(output_path),
        "elapsed_seconds": perf_counter() - tournament_started,
        "standings": standings,
        "pairing_results": pairing_results,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_tournament_summary(summary)
    return 0


def _load_tournament_config(path: Path) -> _TournamentConfig:
    with path.open("rb") as config_file:
        values = tomllib.load(config_file)
    allowed = {"game", "output", "matches_per_pair", "seed", "max_plies", "agents"}
    unknown = sorted(values.keys() - allowed)
    if unknown:
        raise ValueError(f"unknown tournament fields: {', '.join(unknown)}")

    game_name = values.get("game")
    if game_name not in {"boop", "connect-four", "tic-tac-toe"}:
        raise ValueError("tournament game must be boop, connect-four, or tic-tac-toe")
    game = _game(game_name)
    raw_output = values.get("output")
    if raw_output is not None and (
        not isinstance(raw_output, str) or not raw_output.strip()
    ):
        raise ValueError("tournament output must be a non-empty path string")
    output = None if raw_output is None else Path(raw_output)
    if output is not None and not output.is_absolute():
        output = (path.parent / output).resolve()
    matches_per_pair = _positive_tournament_integer(
        "matches_per_pair", values.get("matches_per_pair")
    )
    max_plies = _positive_tournament_integer(
        "max_plies", values.get("max_plies", 10_000)
    )
    seed = values.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("tournament seed must be an integer")
    if not 0 <= seed <= 2**64 - 1:
        raise ValueError("tournament seed must be between 0 and 18446744073709551615")

    raw_agents = values.get("agents")
    if not isinstance(raw_agents, list) or len(raw_agents) < 2:
        raise ValueError("tournament agents must contain at least two entries")
    agents = tuple(
        _load_tournament_agent(raw, index, game)
        for index, raw in enumerate(raw_agents, start=1)
    )
    names = [agent.name for agent in agents]
    if len(names) != len(set(names)):
        raise ValueError("tournament agent names must be unique")
    return _TournamentConfig(
        game=game,
        output=output,
        matches_per_pair=matches_per_pair,
        seed=seed,
        max_plies=max_plies,
        agents=agents,
    )


def _load_tournament_agent(
    values: object,
    index: int,
    game: TicTacToe | ConnectFour | Boop,
) -> _TournamentAgent:
    if not isinstance(values, dict):
        raise TypeError(f"tournament agent {index} must be a TOML table")
    allowed = {
        "name",
        "kind",
        "iterations",
        "exploration",
        "rollout_depth",
        "use_heuristic",
        "heuristic_index",
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
        "exploration",
        "rollout_depth",
        "use_heuristic",
        "heuristic_index",
    }
    if kind == "random":
        unexpected = sorted(values.keys() & mcts_fields)
        if unexpected:
            raise ValueError(
                f"random tournament agent {name} cannot use: {', '.join(unexpected)}"
            )
        return _TournamentAgent(name=name.strip(), agent=RandomAgent(), self_play=self_play)

    missing = sorted({"iterations", "rollout_depth"} - values.keys())
    if missing:
        raise ValueError(
            f"missing fields for tournament agent {name}: {', '.join(missing)}"
        )
    use_heuristic = values.get("use_heuristic", False)
    if not isinstance(use_heuristic, bool):
        raise TypeError(f"tournament agent {name} use_heuristic must be a boolean")
    heuristic_index = values.get("heuristic_index", 0)
    if isinstance(heuristic_index, bool) or not isinstance(heuristic_index, int):
        raise TypeError(f"tournament agent {name} heuristic_index must be an integer")
    agent = MctsAgent(
        iterations=values["iterations"],
        exploration=values.get("exploration", sqrt_two()),
        rollout_depth=values["rollout_depth"],
        heuristic=heuristic_index if use_heuristic else None,
    )
    Match(game=game, first=agent, second=RandomAgent())
    return _TournamentAgent(name=name.strip(), agent=agent, self_play=self_play)


def _positive_tournament_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"tournament {name} must be an integer")
    if not 1 <= value <= 2**32 - 1:
        raise ValueError(f"tournament {name} must be between 1 and 4294967295")
    return value


def _tournament_pairings(
    agents: tuple[_TournamentAgent, ...],
) -> list[tuple[_TournamentAgent, _TournamentAgent]]:
    pairings = [
        (agent_a, agent_b)
        for index, agent_a in enumerate(agents)
        for agent_b in agents[index + 1 :]
    ]
    pairings.extend((agent, agent) for agent in agents if agent.self_play)
    return pairings


def _update_tournament_standings(
    standings: dict[str, dict[str, int]],
    agent_a: str,
    agent_b: str,
    winner: str | None,
) -> None:
    standings[agent_a]["games"] += 1
    standings[agent_b]["games"] += 1
    if winner is None:
        standings[agent_a]["draws"] += 1
        standings[agent_b]["draws"] += 1
    elif winner == "agent_a":
        standings[agent_a]["wins"] += 1
        standings[agent_b]["losses"] += 1
    else:
        standings[agent_b]["wins"] += 1
        standings[agent_a]["losses"] += 1


def _tournament_agent_dict(agent: _TournamentAgent) -> dict[str, object]:
    description = _batch_agent_dict(agent.name, agent.agent)
    description["self_play"] = agent.self_play
    return description


def _write_jsonl(output, value: dict[str, object]) -> None:
    output.write(json.dumps(value, separators=(",", ":")) + "\n")
    output.flush()


def _print_tournament_summary(summary: dict[str, object]) -> None:
    print(f"Game: {summary['game']}")
    print(f"Agents: {summary['agents']}")
    print(f"Pairings: {summary['pairings']}")
    print(f"Matches: {summary['matches']}")
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
    game: TicTacToe | ConnectFour | Boop,
) -> int:
    agent_a, name_a = _batch_agent(args.agent_a, args.agent_a_config, "--agent-a-config")
    agent_b, name_b = _batch_agent(args.agent_b, args.agent_b_config, "--agent-b-config")
    batch = Batch(
        game=game,
        agent_a=agent_a,
        agent_b=agent_b,
        matches=args.matches,
        seed=args.seed,
        max_plies=args.max_plies,
        alternate_sides=args.alternate_sides,
    )
    _print_batch_setup(batch, game, name_a, agent_a, name_b, agent_b)
    result = batch.run(
        progress=lambda event: _print_batch_progress(event, name_a, name_b)
    )
    if args.json:
        print(
            json.dumps(
                _batch_dict(result, game, name_a, agent_a, name_b, agent_b),
                indent=2,
            )
        )
    else:
        _print_batch_result(result, game, name_a, agent_a, name_b, agent_b)
    return 0


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


def _load_mcts_profile(path: Path) -> _MctsProfile:
    with path.open("rb") as profile_file:
        values = tomllib.load(profile_file)
    allowed = {
        "name",
        "iterations",
        "exploration",
        "rollout_depth",
        "use_heuristic",
        "heuristic_index",
    }
    unknown = sorted(values.keys() - allowed)
    if unknown:
        raise ValueError(f"unknown MCTS profile fields: {', '.join(unknown)}")
    missing = sorted({"iterations", "rollout_depth"} - values.keys())
    if missing:
        raise ValueError(f"missing MCTS profile fields: {', '.join(missing)}")

    name = values.get("name", path.stem)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("MCTS profile name must be a non-empty string")
    use_heuristic = values.get("use_heuristic", False)
    if not isinstance(use_heuristic, bool):
        raise TypeError("MCTS profile use_heuristic must be a boolean")
    heuristic_index = values.get("heuristic_index", 0)
    if isinstance(heuristic_index, bool) or not isinstance(heuristic_index, int):
        raise TypeError("MCTS profile heuristic_index must be an integer")

    return _MctsProfile(
        name=name.strip(),
        agent=MctsAgent(
            iterations=values["iterations"],
            exploration=values.get("exploration", sqrt_two()),
            rollout_depth=values["rollout_depth"],
            heuristic=heuristic_index if use_heuristic else None,
        ),
    )


def _print_batch_setup(
    batch: Batch,
    game: TicTacToe | ConnectFour | Boop,
    name_a: str,
    agent_a: RandomAgent | MctsAgent,
    name_b: str,
    agent_b: RandomAgent | MctsAgent,
) -> None:
    print(
        f"Starting batch: {_game_name(game)}, {batch.matches} matches, "
        f"alternate sides: {'yes' if batch.alternate_sides else 'no'}",
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
        f"{name} (iterations={agent.iterations}, rollout_depth={agent.rollout_depth}, "
        f"exploration={agent.exploration:.6f}, "
        f"heuristic={_heuristic_name(agent.heuristic)})"
    )


def _batch_agent_dict(name: str, agent: RandomAgent | MctsAgent) -> dict[str, object]:
    if isinstance(agent, RandomAgent):
        return {"name": name, "type": "random"}
    return {
        "name": name,
        "type": "mcts",
        "iterations": agent.iterations,
        "rollout_depth": agent.rollout_depth,
        "exploration": agent.exploration,
        "heuristic": agent.heuristic,
    }


def _batch_dict(
    result: BatchResult,
    game: TicTacToe | ConnectFour | Boop,
    name_a: str,
    agent_a: RandomAgent | MctsAgent,
    name_b: str,
    agent_b: RandomAgent | MctsAgent,
) -> dict[str, object]:
    return {
        "game": _game_name(game),
        "seed": result.seed,
        "matches": result.matches,
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


def _print_batch_result(
    result: BatchResult,
    game: TicTacToe | ConnectFour | Boop,
    name_a: str,
    agent_a: RandomAgent | MctsAgent,
    name_b: str,
    agent_b: RandomAgent | MctsAgent,
) -> None:
    print(f"Game: {_game_name(game)}")
    print(f"Matches: {result.matches}")
    print(f"Alternate sides: {'yes' if result.alternate_sides else 'no'}")
    print(f"Agent A: {_batch_agent_description(name_a, agent_a)}")
    print(f"Agent B: {_batch_agent_description(name_b, agent_b)}")
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


def _game(name: str) -> TicTacToe | ConnectFour | Boop:
    if name == "boop":
        return Boop()
    if name == "connect-four":
        return ConnectFour()
    return TicTacToe()


def _mcts_configuration(args: argparse.Namespace) -> MctsAgent:
    return MctsAgent(
        iterations=1_000 if args.mcts_iterations is None else args.mcts_iterations,
        exploration=args.mcts_exploration,
        rollout_depth=(
            256 if args.mcts_rollout_depth is None else args.mcts_rollout_depth
        ),
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
            exploration=mcts.exploration,
            rollout_depth=mcts.rollout_depth,
            heuristic=heuristic,
        )
    return RandomAgent()


def _agent_dict(name: str, agent) -> dict[str, object]:
    return {
        "type": name,
        "heuristic": agent.heuristic if isinstance(agent, MctsAgent) else None,
    }


def _heuristic_name(heuristic: int | None) -> str:
    return "none" if heuristic is None else str(heuristic)


def _result_dict(result: MatchResult) -> dict[str, object]:
    return {
        "seed": result.seed,
        "plies": result.plies,
        "utilities": list(result.utilities),
        "winner": result.winner,
        "moves": [
            {
                "ply": ply,
                "player": move.player,
                "action": _action_dict(move.action),
            }
            for ply, move in enumerate(result.moves, start=1)
        ],
    }


def _action_dict(
    action: TicTacToeAction | ConnectFourAction | BoopAction,
) -> dict[str, object]:
    if isinstance(action, TicTacToeAction):
        return {
            "type": "tic_tac_toe",
            "row": action.row,
            "column": action.column,
        }
    if isinstance(action, ConnectFourAction):
        return {"type": "connect_four", "column": action.column}
    return {
        "type": "boop",
        "piece": action.piece.value,
        "row": action.row,
        "column": action.column,
        "resolution": _resolution_dict(action),
    }


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
    print()
    print("Result: draw" if result.winner is None else f"Winner: player {result.winner}")
    print(f"Utilities: {list(result.utilities)}")
    print(f"Plies: {result.plies}")
    print(f"Seed: {result.seed}")


def _agent_name(name: str, agent) -> str:
    if isinstance(agent, MctsAgent):
        return f"{name} (heuristic {_heuristic_name(agent.heuristic)})"
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
    if not isinstance(piece, BoopPiece):
        raise TypeError("unknown board piece")
    if piece.player == 0:
        return "x" if piece.kind is BoopPieceKind.KITTEN else "X"
    return "o" if piece.kind is BoopPieceKind.KITTEN else "O"


def _resolution_dict(action: BoopAction) -> dict[str, object]:
    if isinstance(action.resolution, BoopGraduateLine):
        return {
            "type": "graduate",
            "positions": [
                {"row": position.row, "column": position.column}
                for position in action.resolution.positions
            ],
        }
    if isinstance(action.resolution, BoopRecoverPiece):
        return {
            "type": "recover",
            "row": action.resolution.position.row,
            "column": action.resolution.position.column,
        }
    return {"type": "none"}


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


def _evaluation_dict(report: GameEvaluationReport) -> dict[str, object]:
    return {
        "game": _game_name(report.game),
        "samples": report.samples,
        "max_depth": report.max_depth,
        "terminal_rate": report.terminal_rate,
        "initial_legal_actions": report.initial_legal_actions,
        "effective_branching_factor": report.effective_branching_factor,
        "estimated_depth": report.estimated_depth,
        "depth_is_lower_bound": report.depth_is_lower_bound,
        "estimated_tree_log10": report.estimated_tree_log10,
        "recommended_rollout_depth": report.recommended_rollout_depth,
        "recommended_iterations": report.recommended_iterations,
        "iterations_capped": report.iterations_capped,
        "milliseconds_per_iteration": report.milliseconds_per_iteration,
        "estimated_decision_time_ms": report.estimated_decision_time_ms,
    }


def _print_evaluation(report: GameEvaluationReport) -> None:
    depth_note = " (lower bound)" if report.depth_is_lower_bound else ""
    iteration_note = (
        " (capped; structural estimate is higher)" if report.iterations_capped else ""
    )

    print(f"Game: {_game_name(report.game)}")
    print()
    print("Game structure:")
    print(f"  Initial legal actions: {report.initial_legal_actions}")
    print(f"  Effective branching factor: {report.effective_branching_factor:.2f}")
    print(f"  Estimated depth (p95): {report.estimated_depth}{depth_note}")
    print(f"  Terminal samples: {report.terminal_rate:.1%}")
    print(f"  Estimated tree size: 10^{report.estimated_tree_log10:.1f}")
    print()
    print("MCTS estimate for this machine:")
    print(f"  Rollout depth: {report.recommended_rollout_depth}")
    print(f"  Recommended iterations: {report.recommended_iterations:,}{iteration_note}")
    print(f"  Cost per iteration: ~{report.milliseconds_per_iteration:.6f} ms")
    print(f"  Estimated decision time: ~{report.estimated_decision_time_ms:.1f} ms")
    print()
    print("Interpretation:")
    if report.depth_is_lower_bound:
        print(
            "  - Some samples did not finish. Increase --max-depth or consider "
            "a heuristic for truncated rollouts."
        )
    else:
        print("  - Sampled full-depth rollouts usually reach a terminal result.")
    print(
        "  - If the estimated decision time is too high, reducing rollout depth "
        "usually makes a state heuristic more useful."
    )


def _game_name(game: TicTacToe | ConnectFour | Boop) -> str:
    if isinstance(game, TicTacToe):
        return "tic-tac-toe"
    if isinstance(game, ConnectFour):
        return "connect-four"
    return "boop"
