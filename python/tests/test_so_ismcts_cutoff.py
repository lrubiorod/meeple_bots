"""Cutoff correctness and configuration provenance; no strategic action labels."""
import tempfile
import unittest
from pathlib import Path

from meeple_bots import LostCities, SoIsmctsAgent, GameHeuristic, NeutralEvaluator, Match, RandomAgent, _native
from meeple_bots._search_profiles import load_search_profile
from meeple_bots.cli import _load_tournament_agents
from meeple_bots.serialization import agent_dict
from meeple_bots.studies.persistence import export_profile


class SoIsmctsCutoffTests(unittest.TestCase):
    def test_observation_search_and_hidden_world_invariance(self):
        game = LostCities()
        state = game.initial_state(42)
        observation = game.observation(state, 0)
        world = game.sample_determinization(observation, 0, seed=71)
        self.assertNotEqual(world.state.hands[1], state.hands[1])
        for evaluator in (NeutralEvaluator(), GameHeuristic(0), GameHeuristic(1, {'tau': 30})):
            agent = SoIsmctsAgent(iterations=12, rollout_depth=0, cutoff_evaluator=evaluator)
            result = agent.search(observation, game.legal_actions(state), seed=99)
            self.assertEqual(result, agent.search(world.observation(0), world.legal_actions(), seed=99))
            d = result['diagnostics']
            self.assertEqual(d['terminal_simulations'], 0)
            self.assertEqual(d['cutoff_simulations'], 12)
            self.assertEqual(d['heuristic_evaluations'], 12)
            self.assertEqual(d['safety_cutoff_simulations'], 0)
            for edge in result['nodes'][0]['edges']:
                self.assertTrue(-1 <= edge['q'] <= 1)
                if isinstance(evaluator, NeutralEvaluator):
                    self.assertEqual(edge['q'], 0)
            with self.assertRaisesRegex(TypeError, 'Observation'):
                agent.search(world, world.legal_actions())

    def test_native_match_adapter_with_cutoff_and_reuse(self):
        for reuse in (False, True):
            result = Match(game=LostCities(), first=SoIsmctsAgent(iterations=2, tree_reuse=reuse,
                rollout_depth=0, cutoff_evaluator=GameHeuristic(1)), second=RandomAgent(), seed=42).run()
            self.assertEqual(result.lost_cities_state.status, 'terminal')
            self.assertTrue(any(m.cutoff_simulations for m in result.moves if m.player == 0))

    def test_configuration_validation_python_and_native(self):
        for kwargs in (dict(rollout_depth=-1), dict(rollout_depth=True),
                       dict(rollout_depth=1.5), dict(cutoff_evaluator=GameHeuristic(0))):
            with self.assertRaises((TypeError, ValueError)):
                SoIsmctsAgent(**kwargs)
        # Native generic configuration also accepts descriptors for future games.
        _native.AgentConfig.so_ismcts(iterations=1, rollout_depth=0,
            cutoff_evaluator='game_heuristic', cutoff_heuristic=7, cutoff_params={'other': .5})
        for evaluator in (GameHeuristic(2), GameHeuristic(0, {'tau': 0}), GameHeuristic(1, {'eta': .5})):
            agent = SoIsmctsAgent(iterations=1, rollout_depth=0, cutoff_evaluator=evaluator)
            with self.assertRaises(ValueError):
                agent.search(LostCities().observation(LostCities().initial_state(42), 0), ())
        self.assertIsNone(SoIsmctsAgent().rollout_depth)
