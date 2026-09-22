"""Noncompetitive analysis: structural chance sampling and observation-only calibration."""
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from meeple_bots import (LostCities, Connect6, TicTacToe, Splendor, MctsAgent, SoIsmctsAgent,
                         analyze_game, analyze_structure, benchmark_search_agent)
from meeple_bots.analysis import operating_points, summarize_search, print_analysis
from meeple_bots.cli import main
from meeple_bots._search_budget import decision_budget
from meeple_bots import _native


class AnalysisTests(unittest.TestCase):
    def test_structural_chance_and_random_initial_distribution(self):
        a = analyze_structure(LostCities(), samples=32, max_depth=2000, seed=42)
        self.assertEqual(a, analyze_structure(LostCities(), samples=32, max_depth=2000, seed=42))
        self.assertEqual(a['terminal_rate'], 1)
        self.assertEqual(a['chance_events_mean'], 60)
        self.assertEqual(a['chance_events_p95'], 60)
        self.assertLess(a['initial_legal_actions_min'], a['initial_legal_actions_max'])
        self.assertEqual(a['depth_p50'], 2*a['physical_turns_p50'])
        b = analyze_structure(TicTacToe(), samples=8)
        self.assertEqual(b['initial_legal_actions_min'], 9)
        self.assertEqual(b['initial_legal_actions_max'], 9)
        self.assertEqual(b['chance_events_mean'], 0)

    def test_cant_stop_generic_structural_sampler(self):
        s = _native.analyze_structure('cant_stop', 2, 100, 42)
        self.assertGreater(s['chance_events_mean'], 0)
        self.assertLessEqual(s['estimated_depth'], 100)

    def test_so_analysis_auto_family_and_no_mcts_fields(self):
        report = analyze_game(LostCities(), samples=4, max_depth=600, target_time=.002)
        self.assertEqual(report.properties['search_family'], 'so_ismcts')
        self.assertIsNone(report.legacy_report)
        c = report.search_calibration
        self.assertNotIn('rollout_depth', c)
        self.assertNotIn('recommended_iterations', c)
        self.assertGreater(c['isolated_determinization_mean_ms'], 0)
        self.assertEqual(c['iterations_per_second'], c['determinizations_per_second'])
        self.assertEqual(c['estimated_iterations_at_target'], c['estimated_determinizations_at_target'])
        for t in c['position_timings']:
            self.assertEqual(t['iterations'], t['determinizations'])
            self.assertEqual(sum(t['root_visits']), t['iterations'])
            self.assertGreater(t['legal_actions'], 0)
        output = io.StringIO()
        with redirect_stdout(output):
            print_analysis(report)
        self.assertIn('not an information-set tree size', output.getvalue())
        self.assertIn('determinizations', output.getvalue())

    def test_wrong_family_rejected_before_sampling(self):
        with patch('meeple_bots.analysis.analyze_structure') as sample:
            with self.assertRaisesRegex(ValueError, 'compatible'):
                analyze_game(LostCities(), search_family='mcts')
            sample.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'compatible'):
            benchmark_search_agent(LostCities(), MctsAgent(iterations=1), 10)
        with self.assertRaisesRegex(ValueError, 'compatible'):
            analyze_game(TicTacToe(), search_family='so_ismcts')

    def test_seeded_fixed_search_reproducibility_and_environment(self):
        game, agent = LostCities(), SoIsmctsAgent(iterations=12)
        before = analyze_structure(game, samples=2, max_depth=500, seed=7)
        a = benchmark_search_agent(game, agent, 100, seed=7)
        b = benchmark_search_agent(game, agent, 100, seed=7)
        for x, y in zip(a['position_timings'], b['position_timings']):
            for field in ('iterations', 'determinizations', 'root_visits', 'nodes', 'action_edges', 'legal_actions', 'sampled_ply'):
                self.assertEqual(x[field], y[field])
            self.assertEqual(x['iterations'], 12)
        self.assertEqual(before, analyze_structure(game, samples=2, max_depth=500, seed=7))
        # Reports contain diagnostics only, no private cards/ordered decks/observations.
        encoded = json.dumps(a)
        for key in ('opponent_hand', 'deck_order', 'observation', 'hands'):
            self.assertNotIn('"'+key+'"', encoded)

    def test_operating_points_coverage_and_undersearch_warning(self):
        row = next(x for x in operating_points(2000, .5, 'so_ismcts') if x['target'])
        self.assertEqual(row['iterations'], 1000)
        self.assertEqual(row['determinizations'], 1000)
        timings = [dict(milliseconds=100, iterations=10, legal_actions=4, root_visits=[7,3], nodes=11)]
        s = summarize_search(timings, .001, 'mcts')
        t = s['position_timings'][0]
        self.assertEqual(t['root_action_coverage'], .5)
        self.assertEqual(t['root_actions_visited'], 2)
        self.assertEqual(t['median_root_visits_per_action'], 1.5)
        self.assertEqual(t['p10_root_visits_per_action'], 0)
        self.assertEqual(s['estimated_target_adequacy'], 'VERY LOW')
        self.assertTrue(s['warnings'])
        self.assertAlmostEqual(decision_budget(60, 80), .625)

    def test_match_target_and_configured_budgets(self):
        agent = SoIsmctsAgent(iterations=8, exploration=.75)
        report = analyze_game(LostCities(), samples=2, max_depth=500,
                              target_match_time=1, profiles=[('fixed', agent)])
        c = report.search_calibration
        self.assertAlmostEqual(c['target_time_seconds'], 1/(report.structural['decisions_mean']*1.2))
        self.assertEqual(report.configured_agent_benchmarks[0]['agent']['iterations'], 8)
        with self.assertRaises(ValueError):
            analyze_game(LostCities(), target_time=.5, target_match_time=60)

    def test_configured_cli_profiles_and_explicit_family(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=[]
            for i in (1,2):
                p=Path(directory)/f'agent{i}.toml'
                p.write_text(f'agent="so_ismcts"\nname="so{i}"\niterations={i}\nexploration=1.0\n')
                paths.extend(['--agent-config', str(p)])
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code=main(['analyze','--game','lost_cities','--agent','so_ismcts',
                           '--samples','2','--max-depth','500','--target-time','5ms','--json',*paths])
            self.assertEqual(code,0,err.getvalue())
            data=json.loads(out.getvalue())
            self.assertEqual(data['properties']['search_family'],'so_ismcts')
            self.assertEqual(len(data['configured_agent_benchmarks']),2)
            self.assertNotIn('win_rate',out.getvalue())
            self.assertNotIn('rollout_depth',out.getvalue())

    def test_stochastic_perfect_info_uses_mcts(self):
        r=analyze_game(Splendor(),samples=2,max_depth=20,target_time=.001)
        self.assertEqual(r.properties['search_family'],'mcts')
        self.assertGreater(r.structural['chance_events_mean'],0)
        self.assertGreater(r.search_calibration['iterations_per_second'],0)

    def test_invalid_analysis_inputs_are_rejected_before_sampling(self):
        with self.assertRaises(TypeError):
            analyze_game(object())
        with self.assertRaises(TypeError):
            analyze_structure(LostCities(), samples=True)
        with self.assertRaises(TypeError):
            benchmark_search_agent(LostCities(), SoIsmctsAgent(), 10, seed=True)
        for time in (0, float('nan'), -1):
            with self.subTest(time=time), self.assertRaises(ValueError):
                benchmark_search_agent(LostCities(), SoIsmctsAgent(), 10, target_time=time)

    def test_existing_mcts_report_and_configured_agent_retained(self):
        r=analyze_game(TicTacToe(),samples=2,target_time=.001,
                       profiles=[('fixed',MctsAgent(iterations=2,rollout_depth=3))])
        self.assertIsNotNone(r.legacy_report)
        self.assertTrue(r.legacy_report.rollout_costs)
        self.assertEqual(r.structural['initial_legal_actions_min'],9)
        self.assertEqual(r.configured_agent_benchmarks[0]['agent']['iterations'],2)


if __name__ == '__main__':
    unittest.main()
