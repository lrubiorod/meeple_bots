"""Argparse schema and CLI-only argument adapters."""
from __future__ import annotations

import argparse
from pathlib import Path

from .._concurrency import WorkerSetting
from .._mcts_profiles import sqrt_two
from ..game_config import PLAYABLE_GAMES as _PLAYABLE_GAMES


class GameParameterAction(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        key, separator, raw = value.partition('=')
        if not separator or not key:
            raise argparse.ArgumentError(self, 'expected NAME=INTEGER')
        try:
            number = int(raw)
        except ValueError as error:
            raise argparse.ArgumentError(self, 'expected NAME=INTEGER') from error
        parameters = dict(getattr(namespace, self.dest, None) or {})
        if key in parameters:
            raise argparse.ArgumentError(self, f'duplicate game parameter: {key}')
        parameters[key] = number
        setattr(namespace, self.dest, parameters)


def _analysis_duration(text):
    """Analyze accepts fractional seconds and explicit ms/s/m/h suffixes."""
    value = str(text).strip()
    try:
        for suffix, scale in (("ms", .001), ("s", 1.), ("m", 60.), ("h", 3600.)):
            if value.endswith(suffix):
                return float(value[:-len(suffix)]) * scale
        return float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected seconds or a duration such as 500ms") from error


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
        choices=[*_PLAYABLE_GAMES, "cant-stop"],
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
    match.add_argument("--first", choices=["human", "mcts", "so_ismcts", "random"], default="mcts")
    match.add_argument("--second", choices=["human", "mcts", "so_ismcts", "random"], default="random")
    match.add_argument("--so-ismcts-tree-reuse", action="store_true")
    match.add_argument("--so-ismcts-selection-policy", choices=("uct", "ucb1_tuned"), default="uct")
    match.add_argument("--so-ismcts-iterations", type=int, default=1000)
    match.add_argument("--so-ismcts-exploration", type=float, default=2 ** 0.5)
    match.add_argument("--seed", type=int, default=0)
    match.add_argument("--max-plies", type=int, default=10_000)
    match_budget = match.add_mutually_exclusive_group()
    match_budget.add_argument("--mcts-iterations", type=int)
    match_budget.add_argument(
        "--mcts-time-budget",
        type=float,
        help="approximate wall-clock seconds per MCTS decision",
    )
    match.add_argument("--mcts-progressive-widening", action="store_true")
    match.add_argument("--mcts-progressive-widening-k", type=float, default=1.5)
    match.add_argument("--mcts-progressive-widening-alpha", type=float, default=0.5)
    match.add_argument("--mcts-progressive-widening-expansion", choices=("random", "rave"), default="random")
    match.add_argument("--mcts-rave-equivalence", type=int, default=1000)
    match.add_argument("--mcts-selection-policy", choices=("uct", "ucb1_tuned", "uct_rave"), default="uct")
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
    match.add_argument("--first-mcts-config", "--first-agent-config", dest="first_mcts_config", type=Path)
    match.add_argument("--second-mcts-config", "--second-agent-config", dest="second_mcts_config", type=Path)
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

    analyze = commands.add_parser("analyze", help="measure game structure and search-family cost")
    analyze.add_argument(
        "--game", type=_game_tag, choices=_PLAYABLE_GAMES, required=True
    )
    analyze.add_argument("--samples", type=int, default=128)
    analyze.add_argument("--max-depth", type=int, default=256)
    analyze.add_argument("--seed", type=int, default=0)
    analyze.add_argument("--target-time", type=_analysis_duration, default=None,
                         help="target decision budget, e.g. 500ms or 1s (default: 5s)")
    analyze.add_argument("--target-match-time", type=_analysis_duration,
                         help="derive decision budget from sampled mean decisions and margin 1.2")
    analyze.add_argument("--search-family", choices=("mcts", "so_ismcts"),
                         help="compatible search family; otherwise inferred")
    analyze.add_argument(
        "--agent-config",
        type=Path,
        action="append",
        default=[],
        help="benchmark an exact compatible search profile; repeat to compare costs",
    )
    analyze.add_argument(
        "--agent",
        nargs="?",
        const="",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "select mcts/so_ismcts, or benchmark an inline MCTS profile; for example "
            "--agent 'i=5000,d=120,h=0'"
        ),
    )
    analyze.add_argument("--json", action="store_true", help="print machine-readable JSON")

    study = commands.add_parser("study", help="automatically diagnose MCTS mechanisms, budgets and parameters")
    study.add_argument("--game", choices=tuple(_PLAYABLE_GAMES), required=True)
    study.add_argument("--baseline", "--agent-config", dest="baseline", type=Path, help="initial incumbent; preserves its evaluator, parameters and search budget")
    from ..studies.tuners import TUNING_FIELDS
    study.add_argument("--vs-random", action="store_true", help="final champion vs Random; descriptive only, never used for selection")
    study.add_argument("--tune", choices=tuple(TUNING_FIELDS), help="retune only this dimension; requires --agent-config/--baseline")
    study.add_argument("--second-pass", action="store_true", help="append local C/RAVE/PW retuning on the final incumbent")
    study.add_argument("--heuristic", type=lambda v: int(v.lower().removeprefix("h")), help="cutoff heuristic for a generated baseline; must match an explicit baseline")
    study.add_argument("--target-match-time", default="60s", help="approximate compute per match, e.g. 60s or 2m; independent of --budget")
    study.add_argument("--selection-search", action="store_true", help="tune UCT exploration and compare selectors against the incumbent")
    study.add_argument("--mechanism-search", action="store_true", help="compare reuse/transposition combinations against the incumbent")
    study.add_argument("--depth-search", action="store_true", help="compare cutoff depths with the same heuristic")
    study.add_argument("--all-search", action="store_true", help="enable all supported optional search stages")
    study.add_argument("--widening-expansion-search", action="store_true", help="compare random vs AMAF admission when the incumbent has PW enabled")
    study.add_argument("--pw-search", action="store_true", help="calibrate Progressive Widening before comparing it with the best deterministic candidate")
    study.add_argument("--rave-search", action="store_true", help="enable progressive RAVE calibration and comparison when supported; disabled by default")
    study.add_argument("--budget", help="optional total elapsed limit; limits candidate count while preserving comparison evidence")
    study.add_argument("--games-per-comparison", type=int, default=50, help="fixed games per contrast; even, minimum effective 8; default 50")
    study.add_argument("--stage-games", action="append", default=[], metavar="STAGE=GAMES", help="override fixed games for depth, selection, rave, mechanisms or pw")
    study.add_argument("--output", type=Path, help="study directory; default: results/studies/GAME-study")
    study.add_argument("--seed", type=int, default=42)
    study.add_argument("--max-pairs", type=int, help="legacy alias: exact seed pairs per comparison, not a cap")
    study.add_argument("--decision-time", type=float, help="explicit seconds per decision override; otherwise preserve baseline budget or derive from --target-match-time")
    study.add_argument("--screening-time", type=float, help="removed: use --target-match-time for all comparisons")
    study.add_argument("--workers", type=_worker_setting, default="auto", help="parallel match workers, including equal-time comparisons (default: physical cores minus one)")
    study.add_argument("--max-plies", type=int, default=10000)
    study.add_argument("--agent", choices=("mcts", "so_ismcts"), help="search family; inferred from game capabilities or agent config")
    study.add_argument("--resume", action="store_true", help="resume a frozen study; --budget may be increased")
    study.add_argument("--allow-engine-change", action="store_true", help="with --resume, explicitly accept changed code/build and record mixed-engine provenance; configuration must still match")
    study.add_argument("--json", action="store_true")

    from ..probes.cli import add_parser as add_probe_parser
    add_probe_parser(commands)
    for command in (match, batch, analyze, study):
        command.add_argument("--game-param", dest="game_params", action=GameParameterAction, metavar="NAME=INTEGER", help="game initialization parameter; repeat for multiple parameters")
    return parser
