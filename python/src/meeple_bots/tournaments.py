"""Shared tournament execution for the CLI and local study scripts.

Configuration parsing and console presentation belong to the caller. This module
owns match scheduling, execution, summaries, and durable version-1 traces.
"""

from __future__ import annotations

import json
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
        "study_type": "tournament", "game": game_name(config.game),
        "output": str(output), "pairing_mode": config.pairing_mode,
        "seat_mode": config.seat_mode, "matches_per_pair": config.matches_per_pair,
        "seed": config.seed, "max_plies": config.max_plies, "workers": workers,
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


class TournamentTrace:
    """Write and flush standard traces, optionally continuing a matching local plan.

    Resumption rejects conflicting headers, invalid match numbers, duplicate
    records, and truncated lines. It never silently discards recorded work.
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
                    if not isinstance(record.get("result"), dict):
                        raise ValueError(f"Missing match result in {self.path}")
                    self.completed_match_numbers.add(self._check_number(record))
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
