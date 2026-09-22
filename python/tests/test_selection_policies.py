"""Fixed-seed snapshots captured BEFORE extracting the shared bandit policies.

Digests cover complete move diagnostics/tree output, excluding wall-clock measurements.
Changing these expectations requires explaining an intentional search behavior change.
"""
import hashlib
import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from meeple_bots import (Match, MctsAgent, RandomAgent, TicTacToe, LostCities,
                         SoIsmctsAgent, benchmark_search_agent, _native)
from meeple_bots._search_profiles import load_search_profile
from meeple_bots.studies import export_profile, StudyRunner


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=list).encode()).hexdigest()


class SelectionRegressionTests(unittest.TestCase):
    def test_mcts_pre_refactor_moves_and_diagnostics(self):
        expected = {
            ('uct', False, False): 'c77a076adfaa788bb373845ce62f5526490b517f88f5c2cb4251148d09ea5541',
            ('uct', True, False): 'c563cb6b6a55e1d4bee05fe690d223e441a80311e659eb7368ca2dbdd7e66d13',
            ('uct', False, True): '4e567966021dcac2fa06cf438957a1ccb275e63a13d9c706e0c7e311ed09c6b4',
            ('uct', True, True): 'd30077888ab8afe9dcb6f98d99a9c67ace0a35e2b2f6abda4048a224622657b4',
            ('ucb1_tuned', False, False): '6dc9e09cebeee0f68374277ed42f74e55a0f963fc896f5e705f48f8af535d5e7',
            ('ucb1_tuned', True, False): '5db947553e9c4c5b530655c4a876479e94be6b117b18682e81cb84b743521b67',
            ('ucb1_tuned', False, True): 'e9d1cef0c4059881183a7e07d35d34bb4f20aa25f0d0b0de8bcb087cd28a0936',
            ('ucb1_tuned', True, True): '87d8e714165d4cd90f1998ece5b9b91997a45fa38b04980336959104cad01420',
        }
        for (policy, reuse, transpositions), value in expected.items():
            with self.subTest(policy=policy, reuse=reuse, transpositions=transpositions):
                agent = MctsAgent(iterations=128, selection_policy=policy, tree_reuse=reuse,
                                  transpositions=transpositions, root_diagnostics=True)
                result = Match(game=TicTacToe(), first=agent, second=RandomAgent(), seed=42).run()
                moves = [{k: v for k, v in asdict(m).items() if 'seconds' not in k} for m in result.moves]
                self.assertEqual(digest(moves), value)

    def test_so_uct_pre_refactor_full_tree_and_diagnostics(self):
        game = LostCities()
        state = game.initial_state(seed=42)
        for seed, expected in [(17, '354588b86c78597b1930dbb25bbe099b2081fe49ce19be8bd0def09fc3a4c036'),
                               (42, '967ee2db03afa0442c7d0c4245b39bc2c778dcdcbfe59eb3e7be206b58f9d655')]:
            result = SoIsmctsAgent(iterations=256).search(game.observation(state, 0), game.legal_actions(state), seed=seed)
            result['action'] = result['action'].to_dict()
            self.assertEqual(digest(result), expected)

    def test_so_policies_profiles_and_benchmarks(self):
        game = LostCities()
        state = game.initial_state(seed=42)
        for policy in ('uct', 'ucb1_tuned'):
            with self.subTest(policy=policy):
                agent = SoIsmctsAgent(iterations=32, selection_policy=policy)
                args = (game.observation(state, 0), game.legal_actions(state))
                result = agent.search(*args, seed=12)
                self.assertIn(result['action'], args[1])
                self.assertEqual(result, agent.search(*args, seed=12))
                if policy == 'ucb1_tuned':
                    self.assertEqual(result, replace(agent, exploration=99).search(*args, seed=12))
                for node in result['nodes']:
                    self.assertTrue(all(e['visits'] <= e['availability'] for e in node['edges']))
                with TemporaryDirectory() as tmp:
                    path = Path(tmp)/'agent.toml'
                    export_profile(path, 'test', agent)
                    self.assertEqual(load_search_profile(path), agent)
                benchmark = benchmark_search_agent(game, agent, 20, seed=42)
                self.assertEqual(benchmark['agent']['selection_policy'], policy)
                self.assertTrue(all(t['iterations'] == 32 for t in benchmark['position_timings']))

    def test_invalid_policy_rejected_and_tuned_is_not_c_tuned(self):
        for value in ('uct_rave', 'is_uct', '', None):
            with self.assertRaises(ValueError):
                SoIsmctsAgent(selection_policy=value)
        with self.assertRaisesRegex(ValueError, 'selection_policy'):
            _native.AgentConfig.so_ismcts(selection_policy='uct_rave')
        with TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'does not use exploration'):
            StudyRunner('lost_cities', SoIsmctsAgent(selection_policy='ucb1_tuned'), output=Path(tmp), tune='exploration')
