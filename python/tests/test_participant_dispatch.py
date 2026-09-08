"""All participant pairings use the same traced and observed match behavior."""

import random
import unittest

from meeple_bots import (
    Boop, ConnectFour, HumanAgent, Match, MctsAgent, RandomAgent,
    SpiritsOfTheForest, TicTacToe,
)


class ParticipantDispatchTests(unittest.TestCase):
    def test_all_pairings_preserve_traces_and_human_observers(self):
        for game in (Boop(), ConnectFour(), SpiritsOfTheForest(), TicTacToe()):
            for first in ("random", "mcts", "human"):
                for second in ("random", "mcts", "human"):
                    with self.subTest(game=game, first=first, second=second):
                        def run(observed):
                            human_moves = [[], []]
                            def participant(kind, seat):
                                if kind == "random":
                                    return RandomAgent()
                                if kind == "mcts":
                                    return MctsAgent(iterations=8, rollout_depth=2,
                                                     root_diagnostics=True, tree_reuse=True)
                                rng = random.Random(seat + 1)
                                return HumanAgent(
                                    select_action=lambda turn: rng.choice(turn.legal_actions),
                                    observe_action=human_moves[seat].append,
                                )
                            moves = []
                            result = Match(
                                game=game, first=participant(first, 0), second=participant(second, 1),
                                seed=19, observe_move=moves.append if observed else None,
                            ).run()
                            for seat, kind in enumerate((first, second)):
                                if kind == "human":
                                    self.assertEqual(
                                        [move.action for move in human_moves[seat]],
                                        [move.action for move in result.moves if move.player == seat],
                                    )
                            if observed:
                                self.assertEqual([move.action for move in moves],
                                                 [move.action for move in result.moves])
                            return result
                        direct, observed = run(False), run(True)
                        self.assertEqual(direct.winner, observed.winner)
                        self.assertEqual(direct.final_board, observed.final_board)
                        def decisions(result):
                            return [(move.player, move.action, move.search_iterations,
                                     move.root_actions, move.tree_reuse) for move in result.moves]
                        self.assertEqual(decisions(direct), decisions(observed))
