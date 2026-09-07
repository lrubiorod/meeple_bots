"""Shared execution and trace persistence, independent of personal studies."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path

from meeple_bots import MctsAgent, RandomAgent, TicTacToe
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
