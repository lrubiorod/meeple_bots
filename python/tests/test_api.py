import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from meeple_bots import (
    Boop,
    BoopAction,
    BoopPiece,
    BoopPieceKind,
    ConnectFour,
    ConnectFourAction,
    HumanAgent,
    Match,
    MctsAgent,
    RandomAgent,
    TicTacToe,
    TicTacToeAction,
)
from meeple_bots.cli import main


class MatchApiTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
