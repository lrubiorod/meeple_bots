import csv
import io
import importlib.util
import json
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from meeple_bots import (
    Batch,
    BatchProgressStatus,
    Boop,
    BoopAction,
    BoopPiece,
    BoopPieceKind,
    ConnectFour,
    ConnectFourAction,
    HumanAgent,
    HumanMoveObservation,
    Match,
    MatchMoveObservation,
    MctsAgent,
    RandomAgent,
    TicTacToe,
    TicTacToeAction,
    evaluate_game,
)
from meeple_bots.cli import main
from meeple_bots.games.connect_four.gui import ConnectFourGui
from meeple_bots.games.tic_tac_toe.gui import GuiPlayer, TicTacToeGui
from meeple_bots.gui.server import run_gui
from meeple_bots.reporting import wilson_interval


REPORT_DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(module) is not None
    for module in ("matplotlib", "pandas", "seaborn")
)


class MatchApiTests(unittest.TestCase):
    @staticmethod
    def wait_for_gui(gui, predicate, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = gui.snapshot()
            if predicate(state):
                return state
            time.sleep(0.005)
        raise AssertionError(f"GUI state did not arrive: {gui.snapshot()}")

    def test_wilson_interval_handles_known_and_invalid_counts(self) -> None:
        low, high = wilson_interval(5, 10)

        self.assertAlmostEqual(low, 0.236593, places=6)
        self.assertAlmostEqual(high, 0.763407, places=6)
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))
        with self.assertRaises(ValueError):
            wilson_interval(2, 1)

    def test_batch_alternates_sides_and_aggregates_results(self) -> None:
        events = []
        result = Batch(
            game=TicTacToe(),
            agent_a=RandomAgent(),
            agent_b=RandomAgent(),
            matches=4,
            seed=42,
        ).run(events.append)

        self.assertEqual(result.matches, 4)
        self.assertEqual(
            result.agent_a_wins + result.agent_b_wins + result.draws,
            result.matches,
        )
        self.assertEqual([game.seed for game in result.games], [42, 43, 44, 45])
        self.assertEqual(
            [game.agent_a_player for game in result.games],
            [0, 1, 0, 1],
        )
        self.assertEqual(len(events), 8)
        self.assertEqual(events[0].status, BatchProgressStatus.STARTED)
        self.assertEqual(events[1].status, BatchProgressStatus.COMPLETED)
        self.assertIsNone(events[0].result)
        self.assertEqual(events[1].result, result.games[0])

    def test_batch_results_are_reproducible_except_for_timing(self) -> None:
        batch = Batch(
            game=TicTacToe(),
            agent_a=RandomAgent(),
            agent_b=MctsAgent(iterations=4, rollout_depth=9),
            matches=4,
            seed=7,
        )

        first = batch.run()
        repeated = batch.run()
        first_games = [
            (game.seed, game.agent_a_player, game.winner, game.plies, game.utilities)
            for game in first.games
        ]
        repeated_games = [
            (game.seed, game.agent_a_player, game.winner, game.plies, game.utilities)
            for game in repeated.games
        ]

        self.assertEqual(first_games, repeated_games)

    def test_same_seed_produces_same_report(self) -> None:
        match = Match(
            game=TicTacToe(),
            first=MctsAgent(iterations=100),
            second=RandomAgent(),
            seed=42,
        )

        self.assertEqual(match.run(), match.run())

    def test_report_contains_every_move(self) -> None:
        result = Match(
            game=TicTacToe(),
            first=RandomAgent(),
            second=RandomAgent(),
            seed=7,
        ).run()

        self.assertEqual(len(result.moves), result.plies)
        for ply, move in enumerate(result.moves):
            self.assertEqual(move.player, ply % 2)
            self.assertIn(move.action.row, range(3))
            self.assertIn(move.action.column, range(3))
        if result.winner is not None:
            other = 1 - result.winner
            self.assertGreater(result.utilities[result.winner], result.utilities[other])

    def test_match_observer_receives_every_tic_tac_toe_move(self) -> None:
        observations = []
        result = Match(
            game=TicTacToe(),
            first=RandomAgent(),
            second=RandomAgent(),
            seed=7,
            observe_move=observations.append,
        ).run()

        self.assertEqual(len(observations), result.plies)
        self.assertTrue(all(isinstance(item, MatchMoveObservation) for item in observations))
        self.assertTrue(all(item.decision_seconds >= 0 for item in observations))
        self.assertEqual(observations[-1].board, result.final_board)
        self.assertEqual(
            [item.action for item in observations],
            [move.action for move in result.moves],
        )

    def test_match_observer_receives_every_connect_four_move(self) -> None:
        observations = []
        result = Match(
            game=ConnectFour(),
            first=RandomAgent(),
            second=RandomAgent(),
            seed=11,
            observe_move=observations.append,
        ).run()

        self.assertEqual(len(observations), result.plies)
        self.assertTrue(
            all(isinstance(item.action, ConnectFourAction) for item in observations)
        )
        self.assertTrue(all(item.decision_seconds >= 0 for item in observations))
        self.assertEqual(observations[-1].board, result.final_board)
        self.assertEqual(
            [item.action for item in observations],
            [move.action for move in result.moves],
        )

    def test_tic_tac_toe_gui_runs_agents_and_accepts_human_moves(self) -> None:
        watched = TicTacToeGui()
        watched.start(
            GuiPlayer("random"),
            GuiPlayer("random"),
            seed=4,
            minimum_move_seconds=0,
        )
        finished = self.wait_for_gui(watched, lambda state: state["status"] == "finished")
        self.assertGreaterEqual(len(finished["moves"]), 5)
        self.assertEqual(
            sum(cell is not None for cell in finished["board"]),
            len(finished["moves"]),
        )

        played = TicTacToeGui()
        played.start(
            GuiPlayer("human"),
            GuiPlayer("human"),
            minimum_move_seconds=0,
        )
        waiting = self.wait_for_gui(
            played,
            lambda state: state["status"] == "waiting_human",
        )
        self.assertEqual(waiting["active_player"], 0)
        played.submit_move(1, 1)
        waiting = self.wait_for_gui(
            played,
            lambda state: state["status"] == "waiting_human"
            and state["active_player"] == 1,
        )
        self.assertEqual(waiting["board"][4], 0)
        played.cancel()

    def test_connect_four_gui_runs_agents_and_accepts_columns(self) -> None:
        watched = ConnectFourGui()
        watched.start(
            GuiPlayer("random"),
            GuiPlayer("random"),
            seed=6,
            minimum_move_seconds=0,
        )
        finished = self.wait_for_gui(watched, lambda state: state["status"] == "finished")
        self.assertGreaterEqual(len(finished["moves"]), 7)
        self.assertEqual(sum(cell is not None for cell in finished["board"]), len(finished["moves"]))

        played = ConnectFourGui()
        played.start(
            GuiPlayer("human"),
            GuiPlayer("human"),
            minimum_move_seconds=0,
        )
        waiting = self.wait_for_gui(
            played,
            lambda state: state["status"] == "waiting_human",
        )
        self.assertEqual(waiting["legal_actions"], list(range(7)))
        played.submit_move(3)
        waiting = self.wait_for_gui(
            played,
            lambda state: state["status"] == "waiting_human"
            and state["active_player"] == 1,
        )
        self.assertEqual(waiting["board"][5 * 7 + 3], 0)
        self.assertEqual(waiting["last_move"], [5, 3])
        played.cancel()

    def test_gui_dispatches_connect_four_without_changing_the_server(self) -> None:
        with patch("meeple_bots.gui.server.serve_gui") as serve:
            run_gui(game="connect-four", open_browser=False)

        application, page = serve.call_args.args
        self.assertEqual(application.snapshot()["game"], "connect-four")
        self.assertIn("Connect Four", page)

    def test_invalid_mcts_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MctsAgent(iterations=0)
        with self.assertRaises(TypeError):
            MctsAgent(heuristic=True)
        with self.assertRaises(ValueError):
            MctsAgent(heuristic=-1)
        with self.assertRaises(TypeError):
            HumanAgent(observe_action="not callable")

    def test_boop_match_accepts_both_heuristics(self) -> None:
        for heuristic in (0, 1):
            result = Match(
                game=Boop(),
                first=MctsAgent(iterations=4, rollout_depth=1, heuristic=heuristic),
                second=RandomAgent(),
                seed=19,
            ).run()

            self.assertGreater(result.plies, 0)
            self.assertEqual(len(result.moves), result.plies)

        with self.assertRaisesRegex(ValueError, "available indices: 0..1"):
            Match(game=Boop(), first=MctsAgent(heuristic=2))

    def test_games_without_heuristics_reject_an_index(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "tic-tac-toe does not provide MCTS heuristics",
        ):
            Match(game=TicTacToe(), first=MctsAgent(heuristic=0))

        with self.assertRaisesRegex(
            ValueError,
            "connect-four does not provide MCTS heuristics",
        ):
            Match(
                game=ConnectFour(),
                first=MctsAgent(iterations=1, heuristic=0),
            )

    def test_game_evaluation_has_reproducible_structural_metrics(self) -> None:
        first = evaluate_game(TicTacToe(), samples=16, seed=42)
        repeated = evaluate_game(TicTacToe(), samples=16, seed=42)

        self.assertEqual(first.initial_legal_actions, 9)
        self.assertEqual(first.terminal_rate, 1.0)
        self.assertLessEqual(first.estimated_depth, 9)
        self.assertFalse(first.depth_is_lower_bound)
        self.assertEqual(
            first.effective_branching_factor,
            repeated.effective_branching_factor,
        )
        self.assertEqual(first.estimated_depth, repeated.estimated_depth)
        self.assertEqual(first.recommended_iterations, repeated.recommended_iterations)
        self.assertGreater(first.milliseconds_per_iteration, 0.0)
        self.assertAlmostEqual(
            first.estimated_decision_time_ms,
            first.recommended_iterations * first.milliseconds_per_iteration,
        )

    def test_game_evaluation_marks_a_truncated_depth(self) -> None:
        report = evaluate_game(TicTacToe(), samples=8, max_depth=1)

        self.assertEqual(report.terminal_rate, 0.0)
        self.assertEqual(report.estimated_depth, 1)
        self.assertTrue(report.depth_is_lower_bound)

    def test_scripted_humans_receive_positions_and_finish_a_match(self) -> None:
        first_moves = iter([(0, 0), (0, 1), (0, 2)])
        second_moves = iter([(1, 0), (1, 1)])
        observed_turns = []

        def select_first(turn):
            observed_turns.append(turn)
            return TicTacToeAction(*next(first_moves))

        def select_second(turn):
            observed_turns.append(turn)
            return TicTacToeAction(*next(second_moves))

        result = Match(
            first=HumanAgent(select_first),
            second=HumanAgent(select_second),
        ).run()

        self.assertEqual(result.winner, 0)
        self.assertEqual(result.plies, 5)
        self.assertEqual(observed_turns[0].player, 0)
        self.assertIsInstance(observed_turns[0].game, TicTacToe)
        self.assertEqual(observed_turns[0].board, ((None,) * 3,) * 3)
        self.assertEqual(len(observed_turns[0].legal_actions), 9)

    def test_scripted_humans_can_finish_connect_four(self) -> None:
        first_moves = iter([0, 1, 2, 3])
        second_moves = iter([0, 1, 2])
        observed_turns = []

        def select_first(turn):
            observed_turns.append(turn)
            return ConnectFourAction(next(first_moves))

        def select_second(turn):
            observed_turns.append(turn)
            return ConnectFourAction(next(second_moves))

        result = Match(
            game=ConnectFour(),
            first=HumanAgent(select_first),
            second=HumanAgent(select_second),
        ).run()

        self.assertEqual(result.winner, 0)
        self.assertEqual(result.plies, 7)
        self.assertIsInstance(observed_turns[0].game, ConnectFour)
        self.assertEqual(observed_turns[0].board, ((None,) * 7,) * 6)
        self.assertEqual(len(observed_turns[0].legal_actions), 7)
        self.assertTrue(
            all(isinstance(move.action, ConnectFourAction) for move in result.moves)
        )

    def test_random_agents_can_finish_boop(self) -> None:
        result = Match(
            game=Boop(),
            first=RandomAgent(),
            second=RandomAgent(),
            seed=9,
            max_plies=1_000,
        ).run()

        self.assertIsNotNone(result.winner)
        self.assertTrue(all(isinstance(move.action, BoopAction) for move in result.moves))
        self.assertEqual(len(result.final_board), 6)
        self.assertTrue(
            all(
                cell is None or isinstance(cell, BoopPiece)
                for row in result.final_board
                for cell in row
            )
        )
        self.assertIsNotNone(result.pools)

    def test_human_boop_selector_receives_pools_and_typed_actions(self) -> None:
        observed_turns = []

        def inspect_turn(turn):
            observed_turns.append(turn)
            raise RuntimeError("inspection complete")

        with self.assertRaisesRegex(RuntimeError, "inspection complete"):
            Match(
                game=Boop(),
                first=HumanAgent(inspect_turn),
                second=RandomAgent(),
            ).run()

        turn = observed_turns[0]
        self.assertIsInstance(turn.game, Boop)
        self.assertEqual(len(turn.legal_actions), 36)
        self.assertTrue(
            all(
                action.piece is BoopPieceKind.KITTEN
                for action in turn.legal_actions
            )
        )
        self.assertEqual(turn.pools[0].kittens, 8)
        self.assertEqual(turn.pools[0].cats, 0)

    def test_human_boop_observer_receives_the_state_after_boops(self) -> None:
        first_targets = iter([(2, 2), (2, 3)])
        observations = []

        def select_target(turn):
            row, column = next(first_targets)
            return next(
                action
                for action in turn.legal_actions
                if action.piece is BoopPieceKind.KITTEN
                and action.row == row
                and action.column == column
            )

        def select_safe_move(turn):
            return next(
                action
                for action in turn.legal_actions
                if action.piece is BoopPieceKind.KITTEN
                and action.row == 0
                and action.column == 0
            )

        def observe(move):
            observations.append(move)
            if len(observations) == 2:
                raise RuntimeError("observation complete")

        with self.assertRaisesRegex(RuntimeError, "observation complete"):
            Match(
                game=Boop(),
                first=HumanAgent(select_target, observe_action=observe),
                second=HumanAgent(select_safe_move),
            ).run()

        result = observations[1]
        self.assertIsInstance(result, HumanMoveObservation)
        self.assertEqual(result.player, 0)
        self.assertIsNone(result.board[2][2])
        self.assertEqual(
            result.board[2][1],
            BoopPiece(player=0, kind=BoopPieceKind.KITTEN),
        )
        self.assertEqual(
            result.board[2][3],
            BoopPiece(player=0, kind=BoopPieceKind.KITTEN),
        )

    def test_cli_prompts_for_human_moves(self) -> None:
        output = io.StringIO()
        prompts = io.StringIO()
        moves = ["0 0", "1 0", "0 1", "1 1", "0 2"]

        with (
            patch("builtins.input", side_effect=moves),
            redirect_stdout(output),
            redirect_stderr(prompts),
        ):
            exit_code = main(["match", "--first", "human", "--second", "human"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Winner: player 0", output.getvalue())
        self.assertIn("Final board:", output.getvalue())
        self.assertIn("0 | X X X", output.getvalue())
        self.assertIn("enter row and column", prompts.getvalue())
        self.assertIn("Board after player 0's move:", prompts.getvalue())

    def test_cli_shows_final_board_for_automated_match(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "match",
                    "--first",
                    "random",
                    "--second",
                    "random",
                    "--seed",
                    "9",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Final board:", output.getvalue())
        self.assertIn("    0 1 2", output.getvalue())

    def test_cli_shows_final_connect_four_board_with_gravity(self) -> None:
        output = io.StringIO()
        prompts = io.StringIO()
        moves = ["0", "0", "1", "1", "2", "2", "3"]

        with (
            patch("builtins.input", side_effect=moves),
            redirect_stdout(output),
            redirect_stderr(prompts),
        ):
            exit_code = main(
                [
                    "match",
                    "--game",
                    "connect-four",
                    "--first",
                    "human",
                    "--second",
                    "human",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("4 | O O O . . . .", output.getvalue())
        self.assertIn("5 | X X X X . . .", output.getvalue())

    def test_cli_can_return_json(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "match",
                    "--first",
                    "random",
                    "--second",
                    "random",
                    "--seed",
                    "9",
                    "--json",
                ]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["seed"], 9)
        self.assertEqual(len(payload["moves"]), payload["plies"])

    def test_cli_can_run_connect_four(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "match",
                    "--game",
                    "connect-four",
                    "--first",
                    "random",
                    "--second",
                    "random",
                    "--seed",
                    "9",
                    "--json",
                ]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertGreaterEqual(payload["plies"], 7)
        self.assertTrue(
            all(move["action"]["type"] == "connect_four" for move in payload["moves"])
        )

    def test_cli_can_run_boop(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "match",
                    "--game",
                    "boop",
                    "--first",
                    "random",
                    "--second",
                    "random",
                    "--seed",
                    "9",
                    "--max-plies",
                    "1000",
                    "--json",
                ]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(
            all(move["action"]["type"] == "boop" for move in payload["moves"])
        )

    def test_cli_heuristic_flag_defaults_to_zero(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "match",
                    "--game",
                    "boop",
                    "--first",
                    "mcts",
                    "--second",
                    "random",
                    "--mcts-iterations",
                    "1",
                    "--mcts-rollout-depth",
                    "1",
                    "--first-mcts-heuristic",
                    "--seed",
                    "9",
                    "--json",
                ]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["players"][0]["heuristic"], 0)
        self.assertIsNone(payload["players"][1]["heuristic"])

    def test_cli_rejects_heuristics_for_non_mcts_players(self) -> None:
        errors = io.StringIO()
        with redirect_stderr(errors):
            exit_code = main(
                [
                    "match",
                    "--game",
                    "boop",
                    "--first",
                    "random",
                    "--first-mcts-heuristic",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("requires the corresponding player to be MCTS", errors.getvalue())

    def test_cli_match_loads_an_mcts_profile(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory, "match-profile.toml")
            profile.write_text(
                '\n'.join(
                    [
                        'name = "match-mcts"',
                        "iterations = 1",
                        "rollout_depth = 1",
                        "use_heuristic = true",
                        "heuristic_index = 1",
                    ]
                ),
                encoding="utf-8",
            )
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "match",
                        "--game",
                        "boop",
                        "--first",
                        "mcts",
                        "--second",
                        "random",
                        "--first-mcts-config",
                        str(profile),
                        "--seed",
                        "9",
                        "--json",
                    ]
                )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["players"][0]["heuristic"], 1)

    def test_cli_match_rejects_a_profile_for_a_non_mcts_player(self) -> None:
        errors = io.StringIO()
        with redirect_stderr(errors):
            exit_code = main(
                [
                    "match",
                    "--first",
                    "random",
                    "--first-mcts-config",
                    "profile.toml",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn(
            "--first-mcts-config requires the corresponding player to be MCTS",
            errors.getvalue(),
        )

    def test_cli_match_rejects_a_profile_and_heuristic_for_the_same_player(
        self,
    ) -> None:
        errors = io.StringIO()
        with redirect_stderr(errors):
            exit_code = main(
                [
                    "match",
                    "--first-mcts-config",
                    "profile.toml",
                    "--first-mcts-heuristic",
                    "0",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn(
            "--first-mcts-config cannot be combined with --first-mcts-heuristic",
            errors.getvalue(),
        )

    def test_cli_analyze_returns_simple_evaluation_json(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "analyze",
                    "--game",
                    "tic-tac-toe",
                    "--samples",
                    "8",
                    "--json",
                ]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["game"], "tic-tac-toe")
        self.assertEqual(payload["initial_legal_actions"], 9)
        self.assertLessEqual(payload["estimated_depth"], 9)
        self.assertIn("recommended_iterations", payload)
        self.assertIn("milliseconds_per_iteration", payload)
        self.assertIn("estimated_decision_time_ms", payload)

    def test_cli_batch_loads_an_mcts_profile_and_reports_progress(self) -> None:
        output = io.StringIO()
        progress = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory, "test-profile.toml")
            profile.write_text(
                '\n'.join(
                    [
                        'name = "test-mcts"',
                        "iterations = 4",
                        "rollout_depth = 9",
                        "exploration = 1.4142135623730951",
                        "use_heuristic = false",
                        "heuristic_index = 0",
                    ]
                ),
                encoding="utf-8",
            )
            with redirect_stdout(output), redirect_stderr(progress):
                exit_code = main(
                    [
                        "batch",
                        "--game",
                        "tic-tac-toe",
                        "--matches",
                        "4",
                        "--agent-b-config",
                        str(profile),
                        "--seed",
                        "42",
                        "--json",
                    ]
                )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["matches"], 4)
        self.assertEqual(payload["agents"]["a"]["type"], "random")
        self.assertEqual(payload["agents"]["b"]["name"], "test-mcts")
        self.assertEqual(payload["agents"]["b"]["iterations"], 4)
        self.assertEqual(
            payload["summary"]["agent_a_wins"]
            + payload["summary"]["agent_b_wins"]
            + payload["summary"]["draws"],
            4,
        )
        self.assertEqual(
            [game["agent_a_player"] for game in payload["games"]],
            [0, 1, 0, 1],
        )
        self.assertIn("[1/4] starting", progress.getvalue())
        self.assertIn("[4/4] completed", progress.getvalue())

    def test_cli_batch_requires_a_profile_for_each_mcts_agent(self) -> None:
        errors = io.StringIO()
        with redirect_stderr(errors):
            exit_code = main(
                [
                    "batch",
                    "--game",
                    "tic-tac-toe",
                    "--matches",
                    "2",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("--agent-b-config is required", errors.getvalue())

    def test_cli_batch_profile_can_enable_the_boop_heuristic(self) -> None:
        output = io.StringIO()
        progress = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory, "heuristic.toml")
            profile.write_text(
                '\n'.join(
                    [
                        'name = "strategic"',
                        "iterations = 1",
                        "rollout_depth = 1",
                        "use_heuristic = true",
                        "heuristic_index = 1",
                    ]
                ),
                encoding="utf-8",
            )
            with redirect_stdout(output), redirect_stderr(progress):
                exit_code = main(
                    [
                        "batch",
                        "--game",
                        "boop",
                        "--matches",
                        "1",
                        "--agent-b-config",
                        str(profile),
                        "--json",
                    ]
                )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["agents"]["b"]["name"], "strategic")
        self.assertEqual(payload["agents"]["b"]["heuristic"], 1)

    def test_cli_tournament_runs_round_robin_and_selected_self_play(self) -> None:
        output = io.StringIO()
        progress = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "tournament.toml")
            trace = Path(directory, "generated", "matches.jsonl")
            config.write_text(
                '\n'.join(
                    [
                        'game = "tic-tac-toe"',
                        'output = "generated/matches.jsonl"',
                        "matches_per_pair = 2",
                        "seed = 17",
                        "max_plies = 9",
                        "",
                        "[[agents]]",
                        'name = "alpha"',
                        'kind = "random"',
                        "self_play = true",
                        "",
                        "[[agents]]",
                        'name = "beta"',
                        'kind = "random"',
                    ]
                ),
                encoding="utf-8",
            )

            with redirect_stdout(output), redirect_stderr(progress):
                exit_code = main(
                    [
                        "tournament",
                        "--config",
                        str(config),
                        "--json",
                    ]
                )

            records = [json.loads(line) for line in trace.read_text().splitlines()]

        summary = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["pairings"], 2)
        self.assertEqual(summary["matches"], 4)
        self.assertEqual(summary["output"], str(trace))
        self.assertEqual(summary["standings"]["alpha"]["games"], 2)
        self.assertEqual(summary["standings"]["alpha"]["self_play_games"], 2)
        self.assertEqual(records[0]["record_type"], "tournament")
        self.assertEqual(records[0]["schema_version"], 1)
        self.assertEqual(len(records), 5)
        self.assertEqual(
            [record["result"]["seed"] for record in records[1:]],
            [17, 18, 19, 20],
        )
        self.assertEqual(
            [record["agent_a_player"] for record in records[1:3]],
            [0, 1],
        )
        self.assertTrue(all(record["result"]["moves"] for record in records[1:]))
        self.assertIn("[4/4]", progress.getvalue())

    def test_cli_tournament_rejects_duplicate_agent_names(self) -> None:
        errors = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "tournament.toml")
            config.write_text(
                '\n'.join(
                    [
                        'game = "tic-tac-toe"',
                        "matches_per_pair = 1",
                        "[[agents]]",
                        'name = "same"',
                        'kind = "random"',
                        "[[agents]]",
                        'name = "same"',
                        'kind = "random"',
                    ]
                ),
                encoding="utf-8",
            )
            with redirect_stderr(errors):
                exit_code = main(
                    [
                        "tournament",
                        "--config",
                        str(config),
                        "--output",
                        str(Path(directory, "matches.jsonl")),
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("agent names must be unique", errors.getvalue())

    def test_cli_extracts_analysis_tables_from_a_boop_tournament(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = self._create_small_boop_tournament(root)
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["extract", "--input", str(trace), "--json"])

            summary = json.loads(output.getvalue())
            output_dir = Path(summary["output_dir"])
            manifest = json.loads((output_dir / "manifest.json").read_text())
            matches = self._read_csv(output_dir / "matches.csv")
            boop_matches = self._read_csv(output_dir / "boop_matches.csv")
            turns = self._read_csv(output_dir / "turns.csv")
            agents = self._read_csv(output_dir / "agents.csv")
            boops = self._read_csv(output_dir / "boops.csv")

            self.assertEqual(exit_code, 0)
            self.assertEqual(output_dir, root / "boop-study" / "data")
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["processed_matches"], 1)
            self.assertEqual(len(matches), 1)
            self.assertEqual(len(boop_matches), 1)
            self.assertEqual(boop_matches[0]["match_number"], matches[0]["match_number"])
            self.assertNotIn("win_by_cat_line", matches[0])
            self.assertIn("win_by_cat_line", boop_matches[0])
            self.assertEqual(len(agents), 2)
            self.assertEqual(len(turns), int(matches[0]["plies"]))
            self.assertEqual(manifest["row_counts"]["turns"], len(turns))
            self.assertEqual(manifest["row_counts"]["boops"], len(boops))
            self.assertTrue({turn["zone"] for turn in turns} <= {"center", "middle", "outer"})
            self.assertEqual(turns[0]["strategic_phase"], "all_kittens")
            self.assertEqual(turns[-1]["terminal_after"], "True")
            for filename in (
                "resolutions.csv",
                "winning_lines.csv",
            ):
                self.assertTrue((output_dir / filename).is_file())
            for filename in (
                "agents.csv",
                "matches.csv",
                "boop_matches.csv",
                "turns.csv",
                "boops.csv",
                "resolutions.csv",
                "winning_lines.csv",
            ):
                with Path(output_dir, filename).open(encoding="utf-8", newline="") as source:
                    header = next(csv.reader(source))
                self.assertEqual(len(header), len(set(header)), filename)

            errors = io.StringIO()
            with redirect_stderr(errors):
                repeated_exit = main(["extract", "--input", str(trace)])
            self.assertEqual(repeated_exit, 1)
            self.assertIn("--overwrite", errors.getvalue())
            with redirect_stdout(io.StringIO()):
                overwritten_exit = main(
                    ["extract", "--input", str(trace), "--overwrite"]
                )
            self.assertEqual(overwritten_exit, 0)

    def test_cli_extract_accepts_a_partial_trace_and_ignores_a_truncated_tail(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete_trace = self._create_small_boop_tournament(root)
            records = complete_trace.read_text(encoding="utf-8").splitlines()
            header = json.loads(records[0])
            header["total_matches"] = 2
            partial_trace = root / "partial.jsonl"
            partial_trace.write_text(
                json.dumps(header) + "\n" + records[1] + "\n" + '{"record_type"',
                encoding="utf-8",
            )

            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    ["extract", "--input", str(partial_trace), "--json"]
                )
            summary = json.loads(output.getvalue())
            manifest = json.loads(
                Path(summary["output_dir"], "manifest.json").read_text()
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["processed_matches"], 1)
            self.assertFalse(summary["complete"])
            self.assertTrue(summary["truncated_last_line"])
            self.assertFalse(manifest["complete"])

    def test_cli_extract_reports_games_without_an_analyzer_before_creating_output(
        self,
    ) -> None:
        for game in ("connect-four", "tic-tac-toe"):
            with self.subTest(game=game), tempfile.TemporaryDirectory() as directory:
                trace = Path(directory, f"{game}.jsonl")
                trace.write_text(
                    json.dumps(
                        {
                            "record_type": "tournament",
                            "schema_version": 1,
                            "game": game,
                            "total_matches": 0,
                            "agents": [],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                errors = io.StringIO()
                with redirect_stderr(errors):
                    exit_code = main(["extract", "--input", str(trace)])

                self.assertEqual(exit_code, 1)
                self.assertIn(
                    f"tournament analysis is not available for {game}",
                    errors.getvalue(),
                )
                self.assertFalse(Path(directory, game).exists())

    def test_cli_report_rejects_an_unimplemented_game_before_creating_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = root / "connect-four"
            data = study / "data"
            data.mkdir(parents=True)
            (data / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "game": "connect-four",
                    }
                ),
                encoding="utf-8",
            )
            errors = io.StringIO()
            with redirect_stderr(errors):
                exit_code = main(["report", "--input", str(data)])

            self.assertEqual(exit_code, 1)
            self.assertIn(
                "tournament report is not available for connect-four",
                errors.getvalue(),
            )
            self.assertFalse((study / "report").exists())

    @unittest.skipUnless(
        REPORT_DEPENDENCIES_AVAILABLE,
        "optional report dependencies are not installed",
    )
    def test_cli_generates_a_partial_boop_report_and_protects_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = self._create_small_boop_tournament(root)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["extract", "--input", str(trace)]), 0)
            study = root / "boop-study"
            data = study / "data"
            manifest_path = data / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["complete"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["report", "--input", str(data), "--json"])
            summary = json.loads(output.getvalue())
            report = study / "report"

            self.assertEqual(exit_code, 0)
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["figures"], 11)
            self.assertEqual(summary["tables"], 16)
            self.assertTrue((report / "index.html").is_file())
            self.assertTrue((report / "summary.json").is_file())
            self.assertEqual(len(list((report / "figures").glob("*.png"))), 11)
            self.assertEqual(len(list((report / "tables").glob("*.csv"))), 16)
            self.assertIn("Preliminary", (report / "index.html").read_text())

            errors = io.StringIO()
            with redirect_stderr(errors):
                repeated_exit = main(["report", "--input", str(data)])
            self.assertEqual(repeated_exit, 1)
            self.assertIn("--overwrite", errors.getvalue())

    @unittest.skipUnless(
        REPORT_DEPENDENCIES_AVAILABLE,
        "optional report dependencies are not installed",
    )
    def test_zone_density_accounts_for_different_zone_sizes(self) -> None:
        import pandas as pd

        from meeple_bots.games.boop.reporting import _placement_cells, _zone_rates

        turns = pd.DataFrame(
            {
                "match_number": [1, 1],
                "agent": ["alpha", "alpha"],
                "zone": ["center", "outer"],
            }
        )
        rates = _zone_rates(turns, ["match_number", "agent"]).set_index("zone")

        self.assertEqual(rates.loc["center", "move_share"], 0.5)
        self.assertEqual(rates.loc["outer", "move_share"], 0.5)
        self.assertEqual(rates.loc["center", "density_per_cell"], 0.125)
        self.assertEqual(rates.loc["outer", "density_per_cell"], 0.025)

        placements = _placement_cells(
            pd.DataFrame(
                {
                    "match_number": [1, 2, 2, 2],
                    "row": [2, 0, 0, 0],
                    "column": [2, 0, 0, 0],
                }
            )
        ).set_index(["row", "column"])
        self.assertEqual(placements.loc[(2, 2), "placement_share"], 0.5)
        self.assertEqual(placements.loc[(0, 0), "placement_share"], 0.5)

    @staticmethod
    def _create_small_boop_tournament(root: Path) -> Path:
        config = root / "boop-tournament.toml"
        trace = root / "boop-study.jsonl"
        config.write_text(
            "\n".join(
                [
                    'game = "boop"',
                    'output = "boop-study.jsonl"',
                    "matches_per_pair = 1",
                    "seed = 91",
                    "max_plies = 1000",
                    "",
                    "[[agents]]",
                    'name = "alpha"',
                    'kind = "random"',
                    "",
                    "[[agents]]",
                    'name = "beta"',
                    'kind = "random"',
                ]
            ),
            encoding="utf-8",
        )
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            exit_code = main(["tournament", "--config", str(config)])
        if exit_code != 0:
            raise AssertionError("failed to create the boop extraction fixture")
        return trace

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as source:
            return list(csv.DictReader(source))


if __name__ == "__main__":
    unittest.main()
