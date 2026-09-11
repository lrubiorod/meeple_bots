"""Shared tournament execution for the CLI and local study scripts.

Configuration parsing and console presentation belong to the caller. This module
owns match scheduling, execution, summaries, and durable version-1 traces.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .api import (
    Boop, ConnectFour, SpiritsOfTheForest, TicTacToe,
    RandomAgent, MctsAgent, Match, MatchResult,
)
from ._concurrency import WorkerSetting, ordered_parallel_map, resolve_workers
from .serialization import agent_dict, game_name, trace_match_dict, write_jsonl


@dataclass(frozen=True, slots=True)
class TournamentAgent:
    name: str
    agent: RandomAgent | MctsAgent
    self_play: bool = False
    template_index: int = 0
    grid_position: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class TournamentConfig:
    game: TicTacToe | ConnectFour | Boop | SpiritsOfTheForest
    output: Path | None
    pairing_mode: str
    seat_mode: str
    matches_per_pair: int
    seed: int
    max_plies: int
    workers: WorkerSetting
    agents: tuple[TournamentAgent, ...]


@dataclass(frozen=True, slots=True)
class MatchJob:
    match_number: int
    pairing_number: int
    pairing_match_number: int
    agent_a: TournamentAgent
    agent_b: TournamentAgent
    self_play: bool
    agent_a_player: int
    seed: int


@dataclass(frozen=True, slots=True)
class MatchOutcome:
    result: MatchResult
    started_at: float
    finished_at: float

    @property
    def duration_seconds(self) -> float:
        return self.finished_at - self.started_at


@dataclass(slots=True)
class _TournamentPairingStats:
    agent_a: TournamentAgent
    agent_b: TournamentAgent
    self_play: bool
    agent_a_wins: int = 0
    agent_b_wins: int = 0
    draws: int = 0
    total_plies: int = 0
    started_at: float | None = None
    finished_at: float | None = None


def run_match(
    job: MatchJob,
    game: TicTacToe | ConnectFour | Boop | SpiritsOfTheForest,
    max_plies: int,
) -> MatchOutcome:
    first, second = (
        (job.agent_a.agent, job.agent_b.agent)
        if job.agent_a_player == 0
        else (job.agent_b.agent, job.agent_a.agent)
    )
    started_at = perf_counter()
    result = Match(
        game=game,
        first=first,
        second=second,
        seed=job.seed,
        max_plies=max_plies,
    ).run()
    return MatchOutcome(
        result=result,
        started_at=started_at,
        finished_at=perf_counter(),
    )


def match_jobs(
    pairings: list[tuple[TournamentAgent, TournamentAgent]],
    config: TournamentConfig,
) -> Iterator[MatchJob]:
    match_number = 0
    seed_offset = 0
    for pairing_number, (agent_a, agent_b) in enumerate(pairings, start=1):
        paired_seats = config.seat_mode == "paired" and agent_a is not agent_b
        for pairing_match_number in range(1, config.matches_per_pair + 1):
            match_number += 1
            pairing_seed_offset = (
                (pairing_match_number - 1) // 2
                if paired_seats
                else pairing_match_number - 1
            )
            yield MatchJob(
                match_number=match_number,
                pairing_number=pairing_number,
                pairing_match_number=pairing_match_number,
                agent_a=agent_a,
                agent_b=agent_b,
                self_play=agent_a is agent_b,
                agent_a_player=(pairing_match_number - 1) % 2,
                seed=(config.seed + seed_offset + pairing_seed_offset) & (2**64 - 1),
            )
        seed_offset += (
            config.matches_per_pair // 2 if paired_seats else config.matches_per_pair
        )


def _tournament_pairing_result(
    stats: _TournamentPairingStats,
    config: TournamentConfig,
) -> dict[str, object]:
    if stats.started_at is None or stats.finished_at is None:
        raise RuntimeError("tournament pairing did not run any matches")
    return {
        "agent_a": stats.agent_a.name,
        "agent_b": stats.agent_b.name,
        "self_play": stats.self_play,
        "matches": config.matches_per_pair,
        "agent_a_wins": stats.agent_a_wins,
        "agent_b_wins": stats.agent_b_wins,
        "draws": stats.draws,
        "average_plies": stats.total_plies / config.matches_per_pair,
        "elapsed_seconds": stats.finished_at - stats.started_at,
    }


def tournament_pairings(
    agents: tuple[TournamentAgent, ...],
    pairing_mode: str = "round_robin",
) -> list[tuple[TournamentAgent, TournamentAgent]]:
    pairings = [
        (agent_a, agent_b)
        for index, agent_a in enumerate(agents)
        for agent_b in agents[index + 1 :]
        if pairing_mode == "round_robin"
        or agent_a.template_index != agent_b.template_index
        or _adjacent_grid_variants(agent_a, agent_b)
    ]
    pairings.extend((agent, agent) for agent in agents if agent.self_play)
    return pairings


def _adjacent_grid_variants(
    agent_a: TournamentAgent,
    agent_b: TournamentAgent,
) -> bool:
    if len(agent_a.grid_position) != len(agent_b.grid_position):
        return False
    differences = [
        abs(position_a - position_b)
        for position_a, position_b in zip(
            agent_a.grid_position,
            agent_b.grid_position,
            strict=True,
        )
        if position_a != position_b
    ]
    return differences == [1]


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


def _tournament_agent_dict(agent: TournamentAgent) -> dict[str, object]:
    description = agent_dict(agent.name, agent.agent)
    description["self_play"] = agent.self_play
    return description


def tournament_header(config: TournamentConfig, output: Path, workers: int) -> dict[str, object]:
    """Describe the exact execution using the shared extraction trace schema."""
    pairings = tournament_pairings(config.agents, config.pairing_mode)
    return {
        "record_type": "tournament", "schema_version": 1,
        "decision_timing_scope": "agent_total_v1",
        "study_type": "tournament", "game": game_name(config.game),
        "output": str(output), "pairing_mode": config.pairing_mode,
        "seat_mode": config.seat_mode, "matches_per_pair": config.matches_per_pair,
        "seed": config.seed, "max_plies": config.max_plies, "workers": workers,
        "pairings": [[a.name, b.name] for a, b in pairings],
        "total_pairings": len(pairings),
        "total_matches": len(pairings) * config.matches_per_pair,
        "agents": [_tournament_agent_dict(agent) for agent in config.agents],
    }


def match_record(job: MatchJob, outcome: MatchOutcome) -> dict[str, object]:
    """Serialize an outcome, translating physical seats into tournament roles."""
    winner = outcome.result.winner
    return trace_match_dict(
        result=outcome.result, match_number=job.match_number,
        pairing_number=job.pairing_number, pairing_match_number=job.pairing_match_number,
        agent_a=job.agent_a.name, agent_b=job.agent_b.name,
        self_play=job.self_play, agent_a_player=job.agent_a_player,
        winner=None if winner is None else "agent_a" if winner == job.agent_a_player else "agent_b",
        duration_seconds=outcome.duration_seconds,
    )


def run_matches(
    game: TicTacToe | ConnectFour | Boop | SpiritsOfTheForest,
    jobs: Iterable[MatchJob],
    *,
    max_plies: int,
    workers: WorkerSetting = 1,
) -> Iterator[tuple[MatchJob, MatchOutcome]]:
    """Execute caller-ordered jobs; study scripts can interleave their own plans."""
    def execute(job: MatchJob) -> MatchOutcome:
        return run_match(job, game, max_plies)

    yield from ordered_parallel_map(execute, jobs, resolve_workers(workers))


def _validate_completed_record(record: dict, header: dict) -> None:
    """Validate persisted job identity and required result data without running games."""
    def integer(value, label, minimum=0, maximum=2**64 - 1):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"Invalid {label}")
        return value

    def number(value, label, minimum=0):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < minimum):
            raise ValueError(f"Invalid {label}")
        return value

    pairs = header.get("pairings")
    if pairs is None:
        if header.get("pairing_mode", "round_robin") != "round_robin":
            raise ValueError("Cannot verify legacy adjacent trace without an explicit pairing plan")
        agents = header["agents"]
        pairs = [[a["name"], b["name"]] for i, a in enumerate(agents) for b in agents[i+1:]]
        pairs.extend([a["name"], a["name"]] for a in agents if a.get("self_play", False))
    per_pair = integer(header.get("matches_per_pair"), "matches_per_pair", 1)
    pairing = integer(record.get("pairing_number"), "pairing_number", 1, len(pairs))
    within = integer(record.get("pairing_match_number"), "pairing_match_number", 1, per_pair)
    if record.get("match_number") != (pairing - 1) * per_pair + within:
        raise ValueError("Match number does not match its pairing")
    a, b = pairs[pairing - 1]
    seat = (within - 1) % 2
    players = [a, b] if seat == 0 else [b, a]
    if (record.get("agent_a") != a or record.get("agent_b") != b
            or type(record.get("self_play")) is not bool or record["self_play"] != (a == b)
            or type(record.get("agent_a_player")) is not int or record["agent_a_player"] != seat
            or record.get("players") != players):
        raise ValueError("Recorded agents or seats differ from the pairing plan")
    paired = header.get("seat_mode") == "paired"
    offset = sum(per_pair // 2 if paired and x != y else per_pair for x, y in pairs[:pairing-1])
    offset += (within - 1) // 2 if paired and a != b else within - 1
    seed = (header["seed"] + offset) & (2**64 - 1)
    result = record.get("result")
    if not isinstance(result, dict):
        raise ValueError("Missing match result")
    if integer(result.get("seed"), "result seed") != seed:
        raise ValueError("Recorded seed differs from the pairing plan")
    plies = integer(result.get("plies"), "plies", 1, header["max_plies"])
    moves = result.get("moves")
    if not isinstance(moves, list) or len(moves) != plies:
        raise ValueError("Move count differs from result plies")
    if "winner" not in result:
        raise ValueError("Missing result winner")
    winner = result["winner"]
    if winner is not None:
        integer(winner, "winner", 0, 1)
    expected_role = None if winner is None else "agent_a" if winner == seat else "agent_b"
    if "winner" not in record or record["winner"] != expected_role:
        raise ValueError("Winner role differs from result winner")
    utilities = result.get("utilities")
    if not isinstance(utilities, list) or len(utilities) != 2:
        raise ValueError("Utilities must contain two values")
    for utility in utilities:
        number(utility, "utility", -1)
        if utility > 1:
            raise ValueError("Utility is outside [-1, 1]")
    utility_winner = None if utilities[0] == 0 else 0 if utilities[0] > 0 else 1
    if utilities[0] != -utilities[1] or utility_winner != winner:
        raise ValueError("Utilities differ from winner")
    number(record.get("duration_seconds"), "duration_seconds")
    if header["game"] == "splendor":
        from types import SimpleNamespace
        from .splendor import SplendorAction, SplendorState, SplendorChanceOutcome, ChanceEvent, replay_splendor
        if not isinstance(result.get("chance_events"), list) or not isinstance(result.get("splendor_state"), dict):
            raise ValueError("Splendor trace requires chance events and final state")
        try:
            parsed_moves = tuple(SimpleNamespace(player=m['player'], action=SplendorAction.from_dict(m['action'])) for m in moves)
            events = tuple(ChanceEvent(integer(e['after_ply'], 'chance after_ply', 1), SplendorChanceOutcome(integer(e['outcome']['card'], 'refill card', 0, 89))) for e in result['chance_events'])
            if any(e['outcome'].get('kind') != 'refill' or e['outcome'].get('type') != 'splendor' for e in result['chance_events']):
                raise ValueError("invalid Splendor chance outcome")
            state = replay_splendor(seed, parsed_moves, events)
            if state != SplendorState.from_dict(result['splendor_state']):
                raise ValueError("Splendor replay final state differs")
            if result.get('scores') != [player.prestige for player in state.players]:
                raise ValueError("Splendor replay scores differ")
            if tuple(state._native_position().utilities()) != tuple(utilities):
                raise ValueError("Splendor replay utilities differ")
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("invalid Splendor trace") from error
    action_type = {
        "tic-tac-toe": "tic_tac_toe", "connect-four": "connect_four",
        "boop": "boop", "spotf": "spotf", "splendor": "splendor",
    }[header["game"]]
    if action_type == "spotf":
        for field in ("scores", "collections", "gemstone_pools"):
            if not isinstance(result.get(field), list) or len(result[field]) != 2:
                raise ValueError(f"Missing or invalid spotf {field}")
    for ply, move in enumerate(moves, 1):
        if not isinstance(move, dict) or integer(move.get("ply"), "move ply", 1) != ply:
            raise ValueError("Invalid move sequence")
        integer(move.get("player"), "move player", 0, 1)
        action = move.get("action")
        if not isinstance(action, dict) or action.get("type") != action_type:
            raise ValueError("Invalid move action")
        if action_type in {"tic_tac_toe", "connect_four", "boop"}:
            integer(action.get("column"), "action column", 0, {"tic_tac_toe": 2, "connect_four": 6, "boop": 5}[action_type])
            if action_type != "connect_four":
                integer(action.get("row"), "action row", 0, 2 if action_type == "tic_tac_toe" else 5)
        if action_type == "boop":
            resolution = action.get("resolution")
            if action.get("piece") not in {"kitten", "cat"} or not isinstance(resolution, dict):
                raise ValueError("Invalid boop action")
            kind = resolution.get("type")
            if kind not in {"none", "recover", "graduate"}:
                raise ValueError("Invalid boop resolution")
            positions = resolution.get("positions") if kind == "graduate" else [resolution] if kind == "recover" else []
            if not isinstance(positions, list) or (kind == "graduate" and len(positions) != 3):
                raise ValueError("Invalid graduation positions")
            for position in positions:
                if not isinstance(position, dict):
                    raise ValueError("Invalid resolution position")
                integer(position.get("row"), "resolution row")
                integer(position.get("column"), "resolution column")
        if action_type == "spotf":
            fields = {
                "take_tile": ("row", "column"), "end_collection": (),
                "place_gemstone": ("row", "column"),
                "move_gemstone": ("source_row", "source_column", "target_row", "target_column"),
                "skip_gemstone": (),
            }.get(action.get("kind"))
            if fields is None:
                raise ValueError("Invalid spotf action")
            for field in fields:
                integer(action.get(field), field)
            sacrifice = action.get("sacrifice")
            if sacrifice is not None:
                if not isinstance(sacrifice, dict) or sacrifice.get("kind") not in {"available", "forest"}:
                    raise ValueError("Invalid sacrifice")
                if sacrifice["kind"] == "forest":
                    integer(sacrifice.get("row"), "sacrifice row")
                    integer(sacrifice.get("column"), "sacrifice column")
        number(move.get("decision_seconds"), "decision_seconds")
        for field in ("search_iterations", "search_nodes"):
            if move.get(field) is not None:
                integer(move[field], field)
        if header.get("decision_timing_scope") == "agent_total_v1":
            selection = number(move.get("selection_seconds"), "selection_seconds")
            maintenance = number(move.get("maintenance_seconds"), "maintenance_seconds")
            if not math.isclose(move["decision_seconds"], selection + maintenance, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("Decision time differs from component times")


class TournamentTrace:
    """Write and flush standard traces, optionally continuing a matching local plan.

    Resumption rejects conflicting headers, invalid match numbers, duplicate
    records, incomplete results, incorrect job identities and truncated lines.
    The same structural validator runs before writes; game legality is checked
    separately by extraction. It never silently discards recorded work.
    The caller skips completed jobs while retaining the original job identities.
    """

    def __init__(self, path: Path, header: dict[str, object], *, overwrite=False, resume=False):
        if overwrite and resume:
            raise ValueError("overwrite and resume are mutually exclusive")
        self.path = path
        self.header = header
        self.overwrite = overwrite
        self.resume = resume
        self.completed_match_numbers: set[int] = set()
        self._output = None

    def _check_number(self, record: dict) -> int:
        number = record.get("match_number")
        if type(number) is not int or not 1 <= number <= self.header["total_matches"]:
            raise ValueError(f"Invalid match number in {self.path}")
        if number in self.completed_match_numbers:
            raise ValueError(f"Duplicate match number in {self.path}: {number}")
        return number

    def __enter__(self):
        continuing = self.resume and self.path.exists()
        if continuing:
            with self.path.open(encoding="utf-8") as source:
                first_line = source.readline()
                if not first_line.endswith("\n"):
                    raise ValueError(f"Truncated trace header: {self.path}")
                if json.loads(first_line) != self.header:
                    raise ValueError(f"Trace header differs: {self.path}")
                for line in source:
                    if not line.endswith("\n"):
                        raise ValueError(f"Truncated trace line: {self.path}")
                    record = json.loads(line)
                    if not isinstance(record, dict) or record.get("record_type") != "match":
                        raise ValueError(f"Invalid match record in {self.path}")
                    number = self._check_number(record)
                    try:
                        _validate_completed_record(record, self.header)
                    except (ValueError, KeyError, TypeError) as error:
                        raise ValueError(f"Invalid match {number} in {self.path}: {error}") from error
                    self.completed_match_numbers.add(number)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._output = self.path.open(
            "a" if continuing else "w" if self.overwrite else "x", encoding="utf-8"
        )
        if not continuing:
            write_jsonl(self._output, self.header)
        return self

    def write(self, job: MatchJob, outcome: MatchOutcome) -> dict[str, object]:
        if self._output is None:
            raise RuntimeError("use TournamentTrace inside a with block")
        record = match_record(job, outcome)
        number = self._check_number(record)
        _validate_completed_record(record, self.header)
        write_jsonl(self._output, record)
        self.completed_match_numbers.add(number)
        return record

    def __exit__(self, *args):
        if self._output is not None:
            self._output.close()
            self._output = None


def run_tournament(
    config: TournamentConfig,
    *,
    output: Path | None = None,
    workers: WorkerSetting | None = None,
    overwrite: bool = False,
    on_start: Callable[[dict], None] | None = None,
    on_match: Callable[[dict], None] | None = None,
) -> dict[str, object]:
    """Execute a tournament without requiring argparse or writing console output."""
    output_path = output if output is not None else config.output
    if output_path is None:
        raise ValueError("tournament output is required in the config or with --output")
    if config.matches_per_pair < 1 or config.max_plies < 1:
        raise ValueError("matches_per_pair and max_plies must be positive")
    if config.seat_mode not in {"paired", "alternating"}:
        raise ValueError("tournament seat_mode must be alternating or paired")
    if config.seat_mode == "paired" and config.matches_per_pair % 2:
        raise ValueError("tournament matches_per_pair must be even for paired seats")
    if config.pairing_mode not in {"round_robin", "adjacent"}:
        raise ValueError("tournament pairing_mode must be round_robin or adjacent")
    if len({agent.name for agent in config.agents}) != len(config.agents):
        raise ValueError("tournament agent names must be unique")
    pairings = tournament_pairings(config.agents, config.pairing_mode)
    total_matches = len(pairings) * config.matches_per_pair
    if not total_matches:
        raise ValueError("tournament must contain at least one pairing")
    worker_count = min(resolve_workers(config.workers if workers is None else workers), total_matches)
    header = tournament_header(config, output_path, worker_count)
    standings = {
        agent.name: dict(games=0, wins=0, losses=0, draws=0, self_play_games=0)
        for agent in config.agents
    }
    stats = [_TournamentPairingStats(a, b, a is b) for a, b in pairings]
    started = perf_counter()
    if on_start is not None:
        on_start(header)
    with TournamentTrace(output_path, header, overwrite=overwrite) as trace:
        for job, outcome in run_matches(
            config.game, match_jobs(pairings, config), max_plies=config.max_plies, workers=worker_count
        ):
            row = trace.write(job, outcome)
            pairing = stats[job.pairing_number - 1]
            pairing.total_plies += outcome.result.plies
            pairing.started_at = outcome.started_at if pairing.started_at is None else min(pairing.started_at, outcome.started_at)
            pairing.finished_at = outcome.finished_at if pairing.finished_at is None else max(pairing.finished_at, outcome.finished_at)
            winner = row["winner"]
            if winner is None:
                pairing.draws += 1
            elif winner == "agent_a":
                pairing.agent_a_wins += 1
            else:
                pairing.agent_b_wins += 1
            if job.self_play:
                standings[job.agent_a.name]["self_play_games"] += 1
            else:
                _update_tournament_standings(standings, job.agent_a.name, job.agent_b.name, winner)
            if on_match is not None:
                on_match(row)
    return {
        "game": game_name(config.game), "agents": len(config.agents),
        "pairing_mode": config.pairing_mode, "seat_mode": config.seat_mode,
        "pairings": len(pairings), "matches": total_matches, "workers": worker_count,
        "matches_per_pair": config.matches_per_pair, "seed": config.seed,
        "output": str(output_path), "elapsed_seconds": perf_counter() - started,
        "standings": standings,
        "pairing_results": [_tournament_pairing_result(pairing, config) for pairing in stats],
    }
