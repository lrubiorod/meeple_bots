"""Random progressive widening configuration and deterministic integration."""
import csv
import io
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from meeple_bots import Connect6, Match, MctsAgent, RandomAgent, Splendor, benchmark_mcts_agent
from meeple_bots.cli import main, _load_tournament_agents
from meeple_bots._mcts_profiles import _load_mcts_profile, _parse_inline_mcts_profile
from meeple_bots.serialization import agent_dict
from meeple_bots.studies import profile_values, agent_from_values
from meeple_bots.extraction import extract_tournament
from meeple_bots.tournaments import TournamentConfig, TournamentAgent, run_tournament


class ProgressiveWideningTests(unittest.TestCase):
    def agent(self, **kwargs):
        return MctsAgent(iterations=16, rollout_depth=3, progressive_widening=True,
                         progressive_widening_k=1.0, progressive_widening_alpha=0.5, **kwargs)

    def test_config_roundtrip_and_validation(self):
        self.assertFalse(MctsAgent().progressive_widening)
        for field, values in {'progressive_widening': [1, 'true'],
                              'progressive_widening_k': [0, -1, float('nan'), float('inf'), True],
                              'progressive_widening_alpha': [0, -1, 1.1, float('nan'), float('inf'), True]}.items():
            for value in values:
                with self.assertRaises((ValueError, TypeError)):
                    MctsAgent(**{field: value})
        agent = self.agent()
        self.assertEqual(agent_from_values(profile_values(agent)), agent)
        self.assertTrue(agent_dict('pw', agent)['progressive_widening'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'pw.toml'
            path.write_text('iterations=16\nrollout_depth=3\nprogressive_widening=true\nprogressive_widening_k=1.0\nprogressive_widening_alpha=0.5\n')
            self.assertEqual(_load_mcts_profile(path).agent, agent)
        self.assertEqual(_parse_inline_mcts_profile('iterations=16,depth=3,progressive_widening=true,progressive_widening_k=1,progressive_widening_alpha=0.5').agent, agent)

    def test_guided_config_roundtrip_and_native_diagnostics(self):
        self.assertEqual(MctsAgent().progressive_widening_expansion, "random")
        with self.assertRaisesRegex(ValueError, 'requires'):
            MctsAgent(progressive_widening_expansion="rave")
        with self.assertRaisesRegex(ValueError, 'random or rave'):
            self.agent(progressive_widening_expansion="unknown")
        agent = self.agent(progressive_widening_expansion="rave")
        self.assertEqual(agent_from_values(profile_values(agent)), agent)
        self.assertEqual(agent_dict('guided', agent)['progressive_widening_expansion'], 'rave')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'guided.toml'
            path.write_text('iterations=16\nrollout_depth=3\nprogressive_widening=true\nprogressive_widening_k=1.0\nprogressive_widening_alpha=0.5\nprogressive_widening_expansion="rave"\n')
            self.assertEqual(_load_mcts_profile(path).agent, agent)
        self.assertEqual(_parse_inline_mcts_profile('iterations=16,depth=3,progressive_widening=true,progressive_widening_k=1,progressive_widening_alpha=0.5,progressive_widening_expansion=rave').agent, agent)
        for policy in ('uct', 'ucb1_tuned', 'uct_rave'):
            for reuse in (False, True):
                for transpositions in (False, True):
                    configured = replace(agent, selection_policy=policy, tree_reuse=reuse, transpositions=transpositions)
                    report = benchmark_mcts_agent(Connect6(6), configured, median_depth=4, seed=42)
                    totals = [t.widening_expansions for t in report.position_timings]
                    self.assertTrue(any(t[2] > 0 for t in totals))
                    for total, random, guided, fallback in totals:
                        self.assertEqual(total, random + guided)
                        self.assertEqual(random, fallback)
                    first = Match(Connect6(6), configured, RandomAgent(), seed=42).run()
                    second = Match(Connect6(6), configured, RandomAgent(), seed=42).run()
                    self.assertEqual([m.action for m in first.moves], [m.action for m in second.moves])
        agents = _load_tournament_agents(dict(name='pw', kind='mcts', iterations=4, rollout_depth=3,
            progressive_widening=True, progressive_widening_expansion=['random', 'rave']), 0, Connect6(6))
        self.assertEqual([a.agent.progressive_widening_expansion for a in agents], ['random', 'rave'])

    def test_selectors_backends_reproducibility_and_diagnostics(self):
        game = Connect6(6)
        for policy in ('uct', 'ucb1_tuned', 'uct_rave'):
            for reuse in (False, True):
                for transpositions in (False, True):
                    agent = self.agent(selection_policy=policy, tree_reuse=reuse, transpositions=transpositions, root_diagnostics=True)
                    first = Match(game, agent, RandomAgent(), seed=42).run()
                    second = Match(game, agent, RandomAgent(), seed=42).run()
                    self.assertEqual([m.action for m in first.moves], [m.action for m in second.moves])
                    self.assertGreater(first.plies, 0)
                    report = benchmark_mcts_agent(game, agent, median_depth=4, seed=42)
                    for timing in report.position_timings:
                        legal, active, limit = timing.root_expansion
                        self.assertEqual(legal, timing.legal_actions)
                        self.assertLessEqual(active, limit)
                        self.assertLess(active, legal)
                        self.assertEqual(active, len(timing.root_visits))
                        self.assertEqual(timing.iterations, 16)
        with self.assertRaisesRegex((ValueError, RuntimeError), 'deterministic'):
            Match(Splendor(), self.agent(), RandomAgent()).run()

    def test_cli_grid_and_extraction(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(['match', '--game', 'connect6', '--game-param', 'board_size=6',
                '--first', 'mcts', '--second', 'random', '--mcts-iterations', '4', '--mcts-rollout-depth', '3',
                '--mcts-progressive-widening', '--mcts-progressive-widening-expansion', 'rave', '--mcts-progressive-widening-k', '1', '--json']), 0)
        self.assertIn('"progressive_widening": true', output.getvalue())
        agents = _load_tournament_agents(dict(name='pw', kind='mcts', iterations=4, rollout_depth=3,
            progressive_widening=[False, True], progressive_widening_k=1), 0, Connect6(6))
        self.assertEqual([a.agent.progressive_widening for a in agents], [False, True])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = TournamentConfig(Connect6(6), root/'trace.jsonl', 'round_robin', 'paired', 2, 42, 36, 1,
                tuple(TournamentAgent(a.name, a.agent) for a in agents))
            run_tournament(config)
            self.assertTrue(extract_tournament(config.output, root/'data')['complete'])
            with (root/'data/agents.csv').open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual({r['progressive_widening'] for r in rows}, {'False', 'True'})
            self.assertEqual(next(r for r in rows if r['progressive_widening']=='True')['progressive_widening_k'], '1')
