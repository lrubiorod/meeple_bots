import io
import json
import unittest
from contextlib import redirect_stdout

from meeple_bots import Match, MctsAgent, RandomAgent, TicTacToe
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


if __name__ == "__main__":
    unittest.main()
