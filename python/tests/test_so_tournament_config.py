import unittest
from meeple_bots import LostCities, TicTacToe
from meeple_bots.cli import _load_tournament_agents


class SoTournamentConfigTests(unittest.TestCase):
    def test_selectors_and_equal_time_budget(self):
        for policy in ('uct', 'ucb1_tuned'):
            (entry,) = _load_tournament_agents(dict(name=policy, kind='so_ismcts',
                selection_policy=policy, time_budget=1.0), 1, LostCities())
            self.assertEqual(entry.agent.selection_policy, policy)
            self.assertEqual(entry.agent.time_budget, 1.0)
            self.assertIsNone(entry.agent.iterations)

    def test_compatibility_and_unsupported_fields_rejected(self):
        base = dict(name='so', kind='so_ismcts', iterations=1)
        with self.assertRaisesRegex(ValueError, 'compatible'):
            _load_tournament_agents(base, 1, TicTacToe())
        with self.assertRaisesRegex(ValueError, 'unsupported SO-ISMCTS'):
            _load_tournament_agents(dict(base, transpositions=True), 1, LostCities())
