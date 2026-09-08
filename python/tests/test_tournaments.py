"""Shared execution and trace persistence, independent of personal studies."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path

from meeple_bots import Boop, ConnectFour, SpiritsOfTheForest, MctsAgent, RandomAgent, TicTacToe
from meeple_bots.cli import main
from meeple_bots.extraction import extract_tournament
from meeple_bots.tournaments import (
    TournamentAgent, TournamentConfig, TournamentTrace,
    match_jobs, run_matches, run_tournament, tournament_header, tournament_pairings,
)


class TournamentTests(unittest.TestCase):
    def config(self, path):
        return TournamentConfig(
            game=TicTacToe(), output=path, pairing_mode="round_robin",
            seat_mode="paired", matches_per_pair=4, seed=17,
            max_plies=9, workers=1,
            agents=(
                TournamentAgent("search", MctsAgent(iterations=8, rollout_depth=9)),
                TournamentAgent("random", RandomAgent()),
            ),
        )

    def records(self, path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_cli_and_python_execute_identical_seeded_games(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config(root / "python.jsonl")
            starts, completions = [], []
            console = io.StringIO()
            with redirect_stdout(console), redirect_stderr(console):
                summary = run_tournament(config, on_start=starts.append, on_match=completions.append)
            self.assertEqual(console.getvalue(), "")
            self.assertEqual(summary["matches"], 4)
            self.assertEqual(len(completions), 4)
            self.assertEqual(starts[0]["seat_mode"], "paired")
            toml = root / "study.toml"
            toml.write_text('''game = "tic-tac-toe"
output = "cli.jsonl"
matches_per_pair = 4
seat_mode = "paired"
seed = 17
max_plies = 9
workers = 1
[[agents]]
name = "search"
kind = "mcts"
iterations = 8
rollout_depth = 9
[[agents]]
name = "random"
kind = "random"
''')
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(["tournament", "--config", str(toml)]), 0)
            python_rows = self.records(config.output)
            cli_rows = self.records(root / "cli.jsonl")
            self.assertEqual(python_rows[0]["agents"], cli_rows[0]["agents"])
            for a, b in zip(python_rows[1:], cli_rows[1:], strict=True):
                self.assertEqual(a["players"], b["players"])
                self.assertEqual(a["winner"], b["winner"])
                self.assertEqual(a["result"]["seed"], b["result"]["seed"])
                self.assertEqual(
                    [move["action"] for move in a["result"]["moves"]],
                    [move["action"] for move in b["result"]["moves"]],
                )

    def test_resume_keeps_job_identity_and_produces_extractable_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config(root / "resumed.jsonl")
            header = tournament_header(config, config.output, 1)
            jobs = list(match_jobs(tournament_pairings(config.agents), config))
            self.assertEqual([job.seed for job in jobs], [17, 17, 18, 18])
            with TournamentTrace(config.output, header) as trace:
                # An interrupted study may have executed jobs in shuffled order.
                for job, outcome in run_matches(config.game, [jobs[2]], max_plies=9):
                    trace.write(job, outcome)
            original = config.output.read_bytes()
            with TournamentTrace(config.output, header, resume=True) as trace:
                self.assertEqual(trace.completed_match_numbers, {3})
                pending = [job for job in jobs if job.match_number not in trace.completed_match_numbers]
                for job, outcome in run_matches(config.game, pending, max_plies=9, workers=2):
                    trace.write(job, outcome)
            self.assertTrue(config.output.read_bytes().startswith(original))
            records = self.records(config.output)
            self.assertEqual([r["match_number"] for r in records[1:]], [3, 1, 2, 4])
            summary = extract_tournament(config.output, root / "data")
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["processed_matches"], 4)

    def test_resume_rejects_conflicts_duplicates_and_truncation_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary) / "matches.jsonl")
            run_tournament(config)
            original = config.output.read_bytes()
            header = self.records(config.output)[0]
            with self.assertRaisesRegex(ValueError, "header differs"):
                with TournamentTrace(config.output, dict(header, seed=999), resume=True):
                    pass
            self.assertEqual(config.output.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                run_tournament(config)
            self.assertEqual(config.output.read_bytes(), original)
            for broken in (
                original + original.splitlines(keepends=True)[1],
                original[:-1],
                json.dumps(header).encode(),
            ):
                config.output.write_bytes(broken)
                with self.assertRaises(ValueError):
                    with TournamentTrace(config.output, header, resume=True):
                        pass
                self.assertEqual(config.output.read_bytes(), broken)

    def test_resume_rejects_invalid_results_and_job_identity_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary) / "matches.jsonl")
            run_tournament(config)
            records = self.records(config.output)
            corruptions = [
                (("result",), {}),
                (("result", "seed"), 999),
                (("result", "plies"), 999),
                (("result", "moves"), []),
                (("result", "utilities"), [0]),
                (("result", "utilities"), [float("nan"), 0]),
                (("result", "winner"), True),
                (("result", "moves", 0, "ply"), 2),
                (("result", "moves", 0, "player"), 2),
                (("result", "moves", 0, "action"), {}),
                (("result", "moves", 0, "decision_seconds"), -1),
                (("result", "moves", 0, "selection_seconds"), 42),
                (("result", "moves", 0, "search_iterations"), True),
                (("players",), ["random", "search"]),
                (("agent_a_player",), True),
                (("agent_a",), "random"),
                (("self_play",), True),
                (("pairing_match_number",), 2),
                (("pairing_number",), 2),
                (("duration_seconds",), float("inf")),
            ]
            for path, value in corruptions:
                with self.subTest(path=path):
                    broken = copy.deepcopy(records)
                    target = broken[1]
                    for field in path[:-1]:
                        target = target[field]
                    target[path[-1]] = value
                    config.output.write_text("".join(json.dumps(row) + "\n" for row in broken))
                    original = config.output.read_bytes()
                    trace = TournamentTrace(config.output, records[0], resume=True)
                    with self.assertRaisesRegex(ValueError, "match 1"):
                        with trace:
                            pass
                    self.assertNotIn(1, trace.completed_match_numbers)
                    self.assertEqual(config.output.read_bytes(), original)
            for field in ("seed", "plies", "winner", "utilities", "moves"):
                broken = copy.deepcopy(records)
                del broken[1]["result"][field]
                config.output.write_text("".join(json.dumps(row) + "\n" for row in broken))
                original = config.output.read_bytes()
                with self.assertRaises(ValueError):
                    with TournamentTrace(config.output, records[0], resume=True):
                        pass
                self.assertEqual(config.output.read_bytes(), original)

    def test_resume_validates_adjacent_self_play_and_wrapping_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            agents = tuple(
                TournamentAgent(str(index), RandomAgent(), self_play=index == 0,
                                grid_position=(index,))
                for index in range(3)
            )
            for mode in ("paired", "alternating"):
                config = replace(self.config(root / f"{mode}.jsonl"),
                                 agents=agents, pairing_mode="adjacent", seat_mode=mode,
                                 seed=2**64-2)
                run_tournament(config)
                records = self.records(config.output)
                self.assertEqual(records[0]["pairings"], [["0", "1"], ["1", "2"], ["0", "0"]])
                with TournamentTrace(config.output, records[0], resume=True) as trace:
                    self.assertEqual(trace.completed_match_numbers, set(range(1, 13)))

    def test_completed_results_for_all_games_can_be_resumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            for index, game in enumerate((TicTacToe(), ConnectFour(), Boop(), SpiritsOfTheForest())):
                with self.subTest(game=type(game).__name__):
                    config = replace(self.config(Path(temporary) / f"{index}.jsonl"),
                                     game=game, max_plies=10000, seed=42,
                                     agents=(TournamentAgent("a", RandomAgent()), TournamentAgent("b", RandomAgent())))
                    header = tournament_header(config, config.output, 1)
                    job = next(match_jobs(tournament_pairings(config.agents), config))
                    _, outcome = next(run_matches(game, [job], max_plies=config.max_plies))
                    with TournamentTrace(config.output, header) as trace:
                        trace.write(job, outcome)
                    with TournamentTrace(config.output, header, resume=True) as trace:
                        self.assertEqual(trace.completed_match_numbers, {1})

    def test_legacy_round_robin_is_verifiable_but_adjacent_needs_a_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary) / "legacy.jsonl")
            run_tournament(config)
            records = self.records(config.output)
            del records[0]["pairings"]
            config.output.write_text("".join(json.dumps(row) + "\n" for row in records))
            with TournamentTrace(config.output, records[0], resume=True) as trace:
                self.assertEqual(trace.completed_match_numbers, {1, 2, 3, 4})
            records[0]["pairing_mode"] = "adjacent"
            config.output.write_text("".join(json.dumps(row) + "\n" for row in records))
            original = config.output.read_bytes()
            with self.assertRaisesRegex(ValueError, "legacy adjacent"):
                with TournamentTrace(config.output, records[0], resume=True):
                    pass
            self.assertEqual(config.output.read_bytes(), original)

    def test_write_rejects_an_outcome_from_another_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary) / "matches.jsonl")
            jobs = list(match_jobs(tournament_pairings(config.agents), config))
            _, outcome = next(run_matches(config.game, [jobs[0]], max_plies=9))
            with TournamentTrace(config.output, tournament_header(config, config.output, 1)) as trace:
                original = config.output.read_bytes()
                with self.assertRaisesRegex(ValueError, "seed"):
                    trace.write(jobs[2], outcome)
                self.assertEqual(config.output.read_bytes(), original)
                self.assertFalse(trace.completed_match_numbers)

    def test_invalid_python_plan_fails_before_creating_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary) / "matches.jsonl")
            for invalid in (
                replace(config, matches_per_pair=3),
                replace(config, agents=()),
                replace(config, agents=(config.agents[0], config.agents[0])),
            ):
                with self.assertRaises(ValueError):
                    run_tournament(invalid)
                self.assertFalse(config.output.exists())


if __name__ == "__main__":
    unittest.main()
