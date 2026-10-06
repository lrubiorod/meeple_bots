"""Shared execution and trace persistence, independent of personal studies."""

import copy
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path

from meeple_bots import (Boop, ConnectFour, Connect6, LostCities, Splendor,
                        SpiritsOfTheForest, Match, MatchTermination, MctsAgent, RandomAgent, TicTacToe)
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

    def test_ply_limit_draw_preserves_valid_traces_across_games(self):
        with tempfile.TemporaryDirectory() as temporary:
            for index, game in enumerate((TicTacToe(), ConnectFour(), Boop(),
                                         SpiritsOfTheForest(), Connect6(6), Splendor(), LostCities())):
                with self.subTest(game=type(game).__name__):
                    config = replace(self.config(Path(temporary) / f'{index}.jsonl'),
                        game=game, max_plies=1,
                        matches_per_pair=2, workers=2, draw_on_ply_limit=True,
                        agents=(TournamentAgent('a', RandomAgent()), TournamentAgent('b', RandomAgent())))
                    summary = run_tournament(config)
                    self.assertEqual(summary['ply_limit_draws'], 2)
                    self.assertEqual(summary['pairing_results'][0]['draws'], 2)
                    self.assertEqual(summary['pairing_results'][0]['ply_limit_draws'], 2)
                    rows = self.records(config.output)
                    self.assertTrue(rows[0]['draw_on_ply_limit'])
                    for row in rows[1:]:
                        self.assertTrue(row['result']['ply_limit_reached'])
                        self.assertEqual(row['result']['termination'], 'ply_limit')
                        self.assertEqual(row['result']['plies'], config.max_plies)
                        self.assertEqual(row['result']['utilities'], [0, 0])
                        self.assertIsNone(row['winner'])
                    with TournamentTrace(config.output, rows[0], resume=True) as trace:
                        self.assertEqual(trace.completed_match_numbers, {1, 2})
                    data = Path(temporary) / f'data-{index}'
                    self.assertTrue(extract_tournament(config.output, data)['complete'])
                    with (data / 'matches.csv').open() as source:
                        self.assertTrue(all(r['termination'] == 'ply_limit' for r in csv.DictReader(source)))
                    with (data / 'moves.csv').open() as source:
                        self.assertTrue(all(r['terminal_after'] == 'False' and r['outcome'] == 'draw'
                                            for r in csv.DictReader(source)))
                    # Resume must retain completed draws and execute only the missing job.
                    config.output.write_text(''.join(json.dumps(row) + '\n' for row in rows[:-1]))
                    with TournamentTrace(config.output, rows[0], resume=True) as trace:
                        jobs = match_jobs(tournament_pairings(config.agents), config)
                        pending = [job for job in jobs if job.match_number not in trace.completed_match_numbers]
                        self.assertEqual(len(pending), 1)
                        for job, outcome in run_matches(config.game, pending, max_plies=1,
                                                       draw_on_ply_limit=True):
                            trace.write(job, outcome)
                        self.assertEqual(trace.completed_match_numbers, {1, 2})
                    self.assertEqual(len(self.records(config.output)), 3)
                    # Legacy traces without the explicit reason remain readable.
                    for row in rows[1:]:
                        del row['result']['termination']
                    config.output.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                    with TournamentTrace(config.output, rows[0], resume=True) as trace:
                        self.assertEqual(trace.completed_match_numbers, {1, 2})
                    self.assertTrue(extract_tournament(config.output, data, overwrite=True)['complete'])
                    # Neither a false terminal reason nor an illegal actor becomes valid
                    # merely because this match was administratively adjudicated.
                    for corruption in ('reason', 'actor'):
                        broken = copy.deepcopy(rows)
                        if corruption == 'reason':
                            broken[1]['result']['termination'] = 'game_terminal'
                            broken[1]['result'].pop('ply_limit_reached')
                        else:
                            broken[1]['result']['moves'][0]['player'] = 1
                        config.output.write_text(''.join(json.dumps(row) + '\n' for row in broken))
                        with self.assertRaises(ValueError):
                            with TournamentTrace(config.output, rows[0], resume=True):
                                pass
                        with self.assertRaises(ValueError):
                            extract_tournament(config.output, data, overwrite=True)

    def test_real_terminal_has_priority_and_false_adjudication_is_rejected_for_all_games(self):
        with tempfile.TemporaryDirectory() as temporary:
            for index, game in enumerate((TicTacToe(), ConnectFour(), Boop(),
                                         SpiritsOfTheForest(), Connect6(6), Splendor(), LostCities())):
                with self.subTest(game=type(game).__name__):
                    options = dict(game=game, first=RandomAgent(), second=RandomAgent(), seed=17)
                    full = Match(**options).run()
                    exact = Match(**options, max_plies=full.plies, draw_on_ply_limit=True).run()
                    self.assertIs(exact.termination, MatchTermination.GAME_TERMINAL)
                    self.assertEqual(exact, full)
                    config = replace(self.config(Path(temporary) / f'terminal-{index}.jsonl'),
                                     game=game, max_plies=full.plies, matches_per_pair=2,
                                     draw_on_ply_limit=True,
                                     agents=(TournamentAgent('a', RandomAgent()), TournamentAgent('b', RandomAgent())))
                    header = tournament_header(config, config.output, 1)
                    jobs = list(match_jobs(tournament_pairings(config.agents), config))
                    with TournamentTrace(config.output, header) as trace:
                        for job, outcome in run_matches(game, jobs, max_plies=full.plies,
                                                       draw_on_ply_limit=True):
                            trace.write(job, outcome)
                    rows = self.records(config.output)
                    # First job uses seed 17 and must have ended by the game rules.
                    self.assertEqual(rows[1]['result']['termination'], 'game_terminal')
                    self.assertTrue(extract_tournament(config.output, Path(temporary) / f'full-{index}')['complete'])
                    rows[1]['result'].update(termination='ply_limit', ply_limit_reached=True,
                                              utilities=[0, 0], winner=None)
                    rows[1]['winner'] = None
                    config.output.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                    with self.assertRaises(ValueError):
                        with TournamentTrace(config.output, header, resume=True):
                            pass
                    with self.assertRaises(ValueError):
                        extract_tournament(config.output, Path(temporary) / f'bad-{index}')

    def test_ply_limit_resolves_chance_and_missing_final_outcome_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(self.config(Path(temporary) / 'chance.jsonl'), game=LostCities(),
                             max_plies=2, matches_per_pair=2, draw_on_ply_limit=True,
                             agents=(TournamentAgent('a', RandomAgent()), TournamentAgent('b', RandomAgent())))
            run_tournament(config)
            rows = self.records(config.output)
            self.assertEqual(rows[1]['result']['chance_events'][-1]['after_ply'], 2)
            self.assertEqual(rows[1]['result']['lost_cities_state']['status'], 'player')
            rows[1]['result']['chance_events'].pop()
            config.output.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaises(ValueError):
                with TournamentTrace(config.output, rows[0], resume=True):
                    pass
            with self.assertRaises(ValueError):
                extract_tournament(config.output, Path(temporary) / 'data')

    def test_lost_cities_adjudication_is_opt_in_and_terminal_result_has_priority(self):
        game = LostCities()
        options = dict(game=game, first=RandomAgent(), second=RandomAgent(), seed=42)
        with self.assertRaisesRegex(RuntimeError, 'ply limit'):
            Match(**options, max_plies=32).run()
        limited = Match(**options, max_plies=32, draw_on_ply_limit=True).run()
        self.assertTrue(limited.ply_limit_reached)
        self.assertEqual(limited.lost_cities_state.status, 'player')
        self.assertEqual(len(limited.moves), 32)
        self.assertEqual(limited.utilities, (0, 0))
        terminal = Match(**options).run()
        exact = Match(**options, max_plies=terminal.plies, draw_on_ply_limit=True).run()
        self.assertFalse(exact.ply_limit_reached)
        self.assertIs(exact.termination, MatchTermination.GAME_TERMINAL)
        self.assertEqual(exact.utilities, terminal.utilities)
        self.assertEqual(exact.winner, terminal.winner)
        self.assertEqual(exact.lost_cities_state, terminal.lost_cities_state)

    def test_adjudicated_lost_cities_trace_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(self.config(Path(temporary) / 'limited.jsonl'),
                game=LostCities(), max_plies=32, matches_per_pair=2, draw_on_ply_limit=True,
                agents=(TournamentAgent('a', RandomAgent()), TournamentAgent('b', RandomAgent())))
            run_tournament(config)
            original = self.records(config.output)
            for field, value in (('ply_limit_reached', False), ('ply_limit_reached', 'true'),
                                 ('utilities', [1, -1]), ('plies', 31)):
                with self.subTest(field=field, value=value):
                    rows = copy.deepcopy(original)
                    rows[1]['result'][field] = value
                    config.output.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                    with self.assertRaises(ValueError):
                        with TournamentTrace(config.output, rows[0], resume=True):
                            pass
            rows = copy.deepcopy(original)
            del rows[0]['draw_on_ply_limit']
            config.output.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaises(ValueError):
                with TournamentTrace(config.output, rows[0], resume=True):
                    pass

    def test_cli_loads_ply_limit_draw_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'limited.toml'
            path.write_text('''game = "lost_cities"
output = "limited.jsonl"
matches_per_pair = 2
seat_mode = "paired"
max_plies = 32
draw_on_ply_limit = true
workers = 1
[[agents]]
name = "a"
kind = "random"
[[agents]]
name = "b"
kind = "random"
''')
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(['tournament', '--config', str(path)]), 0)
            rows = self.records(Path(temporary) / 'limited.jsonl')
            self.assertTrue(rows[0]['draw_on_ply_limit'])
            self.assertTrue(all(row['result']['ply_limit_reached'] for row in rows[1:]))
            from meeple_bots.tournament_config import _load_tournament_config
            path.write_text(path.read_text().replace('draw_on_ply_limit = true', 'draw_on_ply_limit = "true"'))
            with self.assertRaisesRegex(TypeError, 'must be a boolean'):
                _load_tournament_config(path)

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
