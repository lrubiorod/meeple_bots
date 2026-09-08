"""UCB1-Tuned reaches native search and survives reusable experiment formats."""

import csv
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from meeple_bots import Boop, ConnectFour, Match, MctsAgent, RandomAgent, SpiritsOfTheForest, TicTacToe
from meeple_bots._mcts_profiles import _load_mcts_profile, _parse_inline_mcts_profile
from meeple_bots.cli import _load_tournament_config
from meeple_bots.extraction import extract_tournament
from meeple_bots.serialization import agent_dict
from meeple_bots.tournaments import run_tournament


class UcbTunedTests(unittest.TestCase):
    def test_validation_and_profile_loading(self):
        with self.assertRaisesRegex(ValueError, "selection_policy"):
            MctsAgent(selection_policy="unknown")
        inline = _parse_inline_mcts_profile("iterations=8,depth=9,selection_policy=ucb1_tuned")
        self.assertEqual(inline.agent.selection_policy, "ucb1_tuned")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "agent.toml"
            path.write_text('iterations=8\nrollout_depth=9\nselection_policy="ucb1_tuned"\n')
            agent = _load_mcts_profile(path).agent
            self.assertEqual(agent, inline.agent)
            self.assertEqual(agent_dict("tuned", agent)["selection_policy"], "ucb1_tuned")

    def test_all_games_and_backends_accept_tuned(self):
        for game in (TicTacToe(), ConnectFour(), Boop(), SpiritsOfTheForest()):
            for transpositions in (False, True):
                with self.subTest(game=type(game).__name__, transpositions=transpositions):
                    result = Match(
                        game=game,
                        first=MctsAgent(iterations=8, rollout_depth=2, selection_policy="ucb1_tuned",
                                        tree_reuse=True, transpositions=transpositions),
                        second=RandomAgent(), seed=42,
                    ).run()
                    self.assertTrue(result.moves)
                    self.assertTrue(all(m.search_iterations == 8 for m in result.moves if m.player == 0))

    def test_tournament_and_extraction_preserve_selection_policy(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "study.toml"
            path.write_text('''game="tic-tac-toe"
output="matches.jsonl"
matches_per_pair=2
seed=42
workers=1
seat_mode="paired"
[[agents]]
name="control"
kind="mcts"
iterations=8
rollout_depth=9
[[agents]]
name="tuned"
kind="mcts"
iterations=8
rollout_depth=9
selection_policy="ucb1_tuned"
''')
            config = _load_tournament_config(path)
            self.assertEqual(config.agents[0].agent.selection_policy, "uct")
            self.assertEqual(config.agents[1].agent.selection_policy, "ucb1_tuned")
            self.assertEqual(replace(config.agents[1].agent, selection_policy="uct"), config.agents[0].agent)
            run_tournament(config)
            extract_tournament(config.output, root / "data")
            with (root / "data/agents.csv").open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([r["selection_policy"] for r in rows], ["uct", "ucb1_tuned"])
