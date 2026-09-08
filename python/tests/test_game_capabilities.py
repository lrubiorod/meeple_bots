"""Native capability metadata drives Python validation and GUI choices."""

import unittest
from unittest.mock import patch

from meeple_bots import Boop, ConnectFour, SpiritsOfTheForest, TicTacToe, GameHeuristic, MctsAgent, Match
from meeple_bots import _native
from meeple_bots.api import _native_agent
from meeple_bots._capabilities import game_search_capabilities, heuristic_indices


class GameCapabilityTests(unittest.TestCase):
    def test_published_schemas_match_native_validation(self):
        for name, game, indices, phases in (
            ("boop", Boop(), (0, 1), False),
            ("spotf", SpiritsOfTheForest(), (0,), True),
            ("connect-four", ConnectFour(), (), False),
            ("tic-tac-toe", TicTacToe(), (), False),
        ):
            with self.subTest(game=name):
                metadata = game_search_capabilities(name)
                self.assertEqual(heuristic_indices(name), indices)
                self.assertEqual(metadata["turn_phase_conditions"], phases)
                for index, parameters in metadata["heuristics"].items():
                    defaults = {key: spec["default"] for key, spec in parameters.items()}
                    Match(game=game, first=MctsAgent(iterations=4, rollout_depth=1,
                          cutoff_evaluator=GameHeuristic(index, defaults))).run()
                    invalid = [{"unknown_parameter": 1.0}]
                    for key, spec in parameters.items():
                        if spec["minimum"] is not None:
                            invalid.append({key: spec["minimum"] - 1.0})
                        if spec["maximum"] is not None:
                            invalid.append({key: spec["maximum"] + 1.0})
                    for params in invalid:
                        agent = MctsAgent(iterations=4, rollout_depth=1,
                                          cutoff_evaluator=GameHeuristic(index, params))
                        with self.assertRaises(ValueError):
                            Match(game=game, first=agent)
                        # Bypass Python validation to check the independent native boundary.
                        with self.assertRaises(RuntimeError):
                            _native.run_match(name, _native_agent(agent, game),
                                              _native.AgentConfig.random(), 0, 10000)
                with self.assertRaises(ValueError):
                    Match(game=game, first=MctsAgent(heuristic=len(indices)))

    def test_python_uses_published_parameters_instead_of_game_constants(self):
        metadata = {"heuristics": {3: {"new_parameter": {
            "default": 2.0, "minimum": 1.0, "maximum": 4.0,
        }}}, "turn_phase_conditions": False}
        with patch("meeple_bots.api.game_search_capabilities", return_value=metadata):
            Match(game=ConnectFour(), first=MctsAgent(
                cutoff_evaluator=GameHeuristic(3, {"new_parameter": 2})))
            for value in (0, 5):
                with self.assertRaises(ValueError):
                    Match(game=ConnectFour(), first=MctsAgent(
                        cutoff_evaluator=GameHeuristic(3, {"new_parameter": value})))
