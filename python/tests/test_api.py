import io
import json
import tempfile
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
    MctsAgent,
    RandomAgent,
    TicTacToe,
    TicTacToeAction,
    evaluate_game,
)
from meeple_bots.cli import main


class MatchApiTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
