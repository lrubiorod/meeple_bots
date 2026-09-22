"""Observation-only search, diagnostics, compatibility and small seeded matches."""
import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr

from meeple_bots import (
    LostCities, LostCitiesAction, Match, Batch, RandomAgent, SoIsmctsAgent, MctsAgent,
    TicTacToe, _native,
)
from meeple_bots.cli import main
from meeple_bots.serialization import agent_dict


class SoIsmctsTests(unittest.TestCase):
    def test_observation_only_search_reproducibility_and_statistics(self):
        game = LostCities()
        state = game.initial_state(seed=42)
        observation = game.observation(state, 0)
        legal = game.legal_actions(state)
        agent = SoIsmctsAgent(iterations=24, exploration=1)
        result = agent.search(observation, legal, seed=71)
        self.assertEqual(result, agent.search(observation, legal, seed=71))
        self.assertEqual(state, game.initial_state(seed=42))
        self.assertIn(result['action'], legal)
        diagnostics = result['diagnostics']
        self.assertEqual(diagnostics['completed_iterations'], 24)
        self.assertEqual(diagnostics['determinizations_sampled'], 24)
        self.assertEqual(diagnostics['rollout_count'], 24)
        self.assertEqual(diagnostics['tree_nodes'], len(result['nodes']))
        self.assertEqual(diagnostics['terminal_simulations'], 24)
        self.assertEqual(diagnostics['cutoff_simulations'], 0)
        edges = result['nodes'][0]['edges']
        self.assertEqual(sum(e['visits'] for e in edges), 24)
        self.assertTrue(all(e['availability'] == 24 for e in edges))
        best = max(e['visits'] for e in edges)
        self.assertEqual(next(e for e in edges if LostCitiesAction.from_dict(e['action']) == result['action'])['visits'], best)
        for node in result['nodes']:
            self.assertEqual(set(node), {'visits', 'edges'})
            for edge in node['edges']:
                self.assertEqual(set(edge), {'action', 'visits', 'availability', 'q', 'children'})
                self.assertLessEqual(edge['visits'], edge['availability'])
        with self.assertRaisesRegex(TypeError, 'Observation'):
            agent.search(state, legal)
        with self.assertRaisesRegex(ValueError, 'root observation/player/legal'):
            agent.search(observation, legal[:1])
        # Neither hidden hand nor deck order can influence identical observation input.
        world = game.sample_determinization(observation, 0, seed=532)
        self.assertNotEqual(world.state.hands[1], state.hands[1])
        self.assertEqual(result, agent.search(world.observation(0), world.legal_actions(), seed=71))

    def test_validation_and_compatibility_both_python_and_native(self):
        for kwargs in ({'iterations': 0}, {'exploration': -1}, {'exploration': float('nan')}, {'exploration': float('inf')}):
            with self.assertRaises(ValueError):
                SoIsmctsAgent(**kwargs)
        for kwargs in ({'rave_equivalence': 100}, {'tree_reuse': True}, {'rollout_policy': 'uniform'}):
            with self.assertRaises(TypeError):
                SoIsmctsAgent(**kwargs)
        with self.assertRaisesRegex(ValueError, 'only supported by Lost Cities'):
            Match(game=TicTacToe(), first=SoIsmctsAgent(2))
        with self.assertRaisesRegex(ValueError, 'imperfect information'):
            Match(game=LostCities(), first=MctsAgent())
        with self.assertRaisesRegex(RuntimeError, 'only supported by Lost Cities'):
            _native.run_match('tic_tac_toe', _native.AgentConfig.so_ismcts(2, 1), _native.AgentConfig.random(), 1, 100)
        for iterations, exploration in [(0, 1), (2, float('nan'))]:
            with self.assertRaises(ValueError):
                _native.AgentConfig.so_ismcts(iterations, exploration)
        self.assertEqual(agent_dict('search', SoIsmctsAgent(3))['type'], 'so_ismcts')

    def test_time_budget_and_actual_completed_iterations(self):
        game = LostCities()
        state = game.initial_state(42)
        for c in (.25, 2.0):
            agent = SoIsmctsAgent(time_budget=.003, exploration=c)
            result = agent.search(game.observation(state, 0), game.legal_actions(state), seed=7)
            d = result['diagnostics']
            self.assertGreater(d['completed_iterations'], 0)
            self.assertEqual(d['completed_iterations'], d['determinizations_sampled'])
            self.assertIsNone(agent.iterations)
        for kw in ({'time_budget': 0}, {'time_budget': float('nan')}, {'iterations': 10, 'time_budget': .1}):
            with self.assertRaises(ValueError):
                SoIsmctsAgent(**kw)

    def test_seeded_small_matches_and_batch(self):
        for opponent in [RandomAgent(), SoIsmctsAgent(2)]:
            result = Match(game=LostCities(), first=SoIsmctsAgent(2), second=opponent, seed=42).run()
            self.assertEqual(result.lost_cities_state.status, 'terminal')
            self.assertEqual(result.lost_cities_state.deck, ())
            self.assertTrue(all(m.search_iterations == 2 for m in result.moves if m.player == 0))
        # Exercise the second player's own observation independently.
        result = Match(game=LostCities(), first=RandomAgent(), second=SoIsmctsAgent(2), seed=12).run()
        self.assertTrue(all(m.search_iterations == 2 for m in result.moves if m.player == 1))
        Batch(game=LostCities(), agent_a=SoIsmctsAgent(1), agent_b=RandomAgent(), matches=1).run()

    def test_cli_configuration_and_rejected_advanced_options(self):
        base = ['match', '--game', 'lost_cities', '--first', 'so_ismcts', '--second', 'random',
                '--so-ismcts-iterations', '2', '--so-ismcts-exploration', '1', '--seed', '42', '--json']
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(base), 0)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload['players'][0], {
            'type': 'so_ismcts', 'iterations': 2, 'exploration': 1.0, 'selection_policy': 'uct',
            'rollout_policy': 'uniform', 'root_selection': 'most_visited',
        })
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(main(base + ['--mcts-selection-policy', 'uct_rave']), 1)
        self.assertIn('not supported by SO-ISMCTS', err.getvalue())


if __name__ == '__main__':
    unittest.main()
