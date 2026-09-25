"""Noncompetitive analysis: structural chance sampling and observation-only calibration."""
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch, Mock

from meeple_bots import (LostCities, Connect6, TicTacToe, Splendor, SpiritsOfTheForest, MctsAgent, SoIsmctsAgent,
                         analyze_game, analyze_structure, benchmark_search_agent)
from meeple_bots.analysis import (operating_points, summarize_search, print_analysis,
                                 _add_game_search_estimates, _format_game_search_seconds, sampled_full_horizon, _ANALYZERS)
from meeple_bots.cli import main
from meeple_bots._search_budget import decision_budget
from meeple_bots.search_metrics import quantile, search_adequacy
from meeple_bots import study_analysis
from meeple_bots.analysis import AnalysisReport
from meeple_bots.analysis.measurement import AnalysisReport as MeasuredReport
from meeple_bots import _native


class AnalysisTests(unittest.TestCase):
    def test_analysis_facade_keeps_one_report_type(self):
        self.assertIs(AnalysisReport, MeasuredReport)

    def test_shared_work_metrics_keep_boundary_values_and_study_imports(self):
        self.assertIs(study_analysis.quantile, quantile)
        self.assertIs(study_analysis.search_adequacy, search_adequacy)
        self.assertIsNone(quantile([], .95))
        self.assertEqual(quantile([1, 2, 3, 4], .95), 3.8499999999999996)
        for iterations, category in ((9, 'VERY LOW'), (10, 'LOW'), (100, 'MEDIUM'), (1000, 'HIGH')):
            result = search_adequacy([{'iterations': iterations, 'legal_actions': 10,
                                       'milliseconds': 100}], 60)
            self.assertEqual(result['category'], category)
            self.assertTrue(all('target_match_time' in row for row in result['suggested_targets']))

    def test_sampled_full_horizon_rounding(self):
        for p95, expected in ((100, 150), (101, 152), (124, 186), (1, 2), (0, 1), (2**32-1, 2**32-1)):
            self.assertEqual(sampled_full_horizon(p95), expected)

    def test_default_full_calibration_ignores_legacy_and_shares_probes(self):
        from meeple_bots.api import evaluate_game
        from meeple_bots.study_analysis import search_adequacy
        game = SpiritsOfTheForest()
        legacy = evaluate_game(game, samples=2, max_depth=200, seed=42, target_time=.001)
        # Deliberately contradictory compatibility field must not influence any probe.
        legacy = replace(legacy, recommended_rollout_depth=1)
        measured = Mock(wraps=_ANALYZERS['mcts'])
        with patch('meeple_bots.analysis.measurement.evaluate_game', return_value=legacy), patch.dict(_ANALYZERS, mcts=measured):
            report = analyze_game(game, samples=2, max_depth=200, seed=42, target_time=.001)
        full = sampled_full_horizon(report.structural['estimated_depth'])
        self.assertTrue(all(call.args[1].rollout_depth == full for call in measured.call_args_list))
        calibration = report.search_calibration
        self.assertEqual(calibration['agent']['rollout_depth'], full)
        self.assertEqual(calibration['agent']['cutoff_evaluator'], {'kind': 'neutral'})
        timings = calibration['position_timings']
        throughput = sum(t['iterations'] for t in timings) / (sum(t['milliseconds'] for t in timings)/1000)
        self.assertEqual(calibration['iterations_per_second'], throughput)
        for point in calibration['budget_table']:
            self.assertEqual(point['iterations'], max(1, int(throughput * point['seconds'])))
        self.assertEqual(calibration['search_adequacy']['category'], search_adequacy(timings, .001)['category'])
        self.assertEqual({p['label'] for p in calibration['phase_diagnostics']}, {'Collect', 'Gem placement'})
        for phase in calibration['phase_diagnostics']:
            probes = [t for t in timings if t['phase'] == phase['label']]
            self.assertEqual(phase['search_adequacy']['iterations_completed'], sum(t['iterations'] for t in probes))
        self.assertEqual(legacy.rollout_costs[-1].rollout_depth, full)
        self.assertTrue(any(c.rollout_depth < full for c in legacy.rollout_costs))
        output = io.StringIO()
        with redirect_stdout(output): print_analysis(report)
        self.assertIn(f'full {full}', output.getvalue())
        self.assertIn(f'full = {full} plies', output.getvalue())
        self.assertIn('Rollouts reaching terminal:', output.getvalue())
        self.assertFalse(any('truncated structural' in w for w in calibration['warnings']))

    def test_truncated_full_horizon_warning_and_configured_budget_preserved(self):
        baseline = MctsAgent(iterations=2, rollout_depth=3)
        report = analyze_game(TicTacToe(), samples=2, max_depth=1, target_time=.001, profiles=[('fixed', baseline)])
        self.assertEqual(report.search_calibration['agent']['rollout_depth'], 2)
        self.assertTrue(any('truncated structural' in w for w in report.search_calibration['warnings']))
        self.assertEqual(report.configured_agent_benchmarks[0]['agent']['rollout_depth'], 3)

    def test_terminal_and_neutral_cutoff_diagnostics(self):
        full = benchmark_search_agent(TicTacToe(), MctsAgent(iterations=1, rollout_depth=100, root_diagnostics=True), 0)
        self.assertEqual(full['rollout_terminal_rate'], 1)
        self.assertEqual(full['rollout_cutoff_rate'], 0)
        short = benchmark_search_agent(TicTacToe(), MctsAgent(iterations=1, rollout_depth=1, root_diagnostics=True), 0)
        self.assertEqual(short['rollout_terminal_rate'], 0)
        self.assertEqual(short['rollout_cutoff_rate'], 1)

    def test_cli_has_one_operating_section_and_no_legacy_presets(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(['analyze', '--game', 'tic-tac-toe', '--samples', '2', '--target-time', '1ms'])
        self.assertEqual(code, 0)
        text = output.getvalue()
        for old in ('Fast', 'Balanced', 'Wide', 'Deep', 'Suggested starting experiments',
                    'Practical MCTS configuration', 'Candidate rollout depths', 'Budget table', '1s='):
            self.assertNotIn(old, text)
        for section in ('Structure', 'Depth structure', 'Rollout horizon diagnostics',
                        'Search calibration', 'Operating points', 'Interpretation'):
            self.assertIn(section, text)
        self.assertEqual(text.count('Operating points'), 1)
        self.assertEqual(text.count('decision | iterations | mean game | p95 game'), 1)
        self.assertEqual(text.count('<- target'), 1)
        self.assertIn('ms/iteration', text)
        self.assertIn('Actions per player turn', text)
        self.assertIn('Observed choices across one player turn', text)
        self.assertNotIn('Phase diagnostics', text)

    def test_phase_adequacy_can_be_lower_than_global(self):
        timings = [dict(milliseconds=10, iterations=1000, legal_actions=2,
                        root_visits=[500, 500], nodes=1001, phase='Collect') for _ in range(2)]
        timings.append(dict(milliseconds=10, iterations=1000, legal_actions=200,
                            root_visits=[5]*200, nodes=1001, phase='Gem placement'))
        result = summarize_search(timings, .01, 'mcts')
        self.assertEqual(result['search_adequacy']['category'], 'HIGH')
        phases = {p['label']: p for p in result['phase_diagnostics']}
        self.assertEqual(phases['Gem placement']['search_adequacy']['category'], 'LOW')
        self.assertEqual(phases['Gem placement']['search_adequacy']['median_visits_per_root_action'], 5)
        self.assertEqual(phases['Gem placement']['root_actions_visited'], [200])

    def test_phases_cover_spotf_and_lost_cities_without_chance(self):
        for game, labels in ((SpiritsOfTheForest(), {'Collect', 'Gem placement'}),
                             (LostCities(), {'Play', 'Draw'})):
            with self.subTest(game=game):
                structure = analyze_structure(game, samples=4, max_depth=600, seed=42)
                self.assertEqual({p['label'] for p in structure['phases']}, labels)
                self.assertEqual(sum(p['samples'] for p in structure['phases']),
                                 structure['samples'] * structure['decisions_mean'])
                agent = (SoIsmctsAgent(iterations=8) if isinstance(game, LostCities)
                         else MctsAgent(iterations=8, root_diagnostics=True))
                report = benchmark_search_agent(game, agent, structure['depth_p50'], seed=42)
                self.assertLessEqual(report['sampled_positions'], 7)
                self.assertEqual({p['label'] for p in report['phase_diagnostics']}, labels)
                self.assertTrue(all(t['phase'] in labels and t['legal_actions'] > 0
                                    for t in report['position_timings']))
        report = analyze_game(SpiritsOfTheForest(), samples=2, max_depth=150, seed=42, target_time=.001)
        output = io.StringIO()
        with redirect_stdout(output):
            print_analysis(report)
        for label in ('Phase diagnostics', 'Collect', 'Gem placement', 'Structural decision samples'):
            self.assertIn(label, output.getvalue())

    def test_game_search_estimates_use_decisions_and_preserve_operating_points(self):
        for family in ('mcts', 'so_ismcts', 'future_family'):
            with self.subTest(family=family):
                rows = operating_points(2000, .1, family)
                original = [dict(row) for row in rows]
                _add_game_search_estimates(rows, dict(decisions_mean=100, estimated_depth=150,
                                                      physical_turns_p50=50, physical_turns_p95=75))
                target = next(row for row in rows if row['target'])
                self.assertEqual(target['estimated_mean_game_search_seconds'], 10)
                self.assertEqual(target['estimated_p95_game_search_seconds'], 15)
                for before, after in zip(original, rows):
                    self.assertEqual(before, {key: after[key] for key in before})

    def test_unavailable_game_search_statistics(self):
        for invalid in (None, -1, float('nan'), float('inf'), '100', True):
            rows = operating_points(2000, .1, 'mcts')
            _add_game_search_estimates(rows, {'decisions_mean': invalid})
            self.assertIsNone(rows[0]['estimated_mean_game_search_seconds'])
            self.assertIsNone(rows[0]['estimated_p95_game_search_seconds'])
        rows = operating_points(2000, .1, 'mcts')
        _add_game_search_estimates(rows, {'decisions_mean': 100})
        self.assertEqual(rows[0]['estimated_mean_game_search_seconds'], 10)
        self.assertIsNone(rows[0]['estimated_p95_game_search_seconds'])

    def test_game_search_duration_format(self):
        for seconds, expected in ((None, 'N/A'), (0, '0.00s'), (.125, '0.12s'),
                                  (2.87, '2.87s'), (28.67, '28.67s'),
                                  (71.68, '1m 11.68s'), (3601.25, '60m 1.25s'),
                                  (59.999, '1m 0.00s')):
            with self.subTest(seconds=seconds):
                self.assertEqual(_format_game_search_seconds(seconds), expected)

    def test_game_columns_reuse_structural_samples_for_all_profiles(self):
        with patch('meeple_bots.analysis.measurement.analyze_structure', wraps=analyze_structure) as sample:
            report = analyze_game(LostCities(), samples=2, max_depth=500, target_time=.1,
                                  profiles=[('fixed', SoIsmctsAgent(iterations=2))])
            sample.assert_called_once()
        for calibration in (report.search_calibration, *report.configured_agent_benchmarks):
            target = next(row for row in calibration['budget_table'] if row['target'])
            self.assertAlmostEqual(target['estimated_mean_game_search_seconds'],
                                   .1 * report.structural['decisions_mean'])
            self.assertAlmostEqual(target['estimated_p95_game_search_seconds'],
                                   .1 * report.structural['estimated_depth'])
        output = io.StringIO()
        with redirect_stdout(output):
            print_analysis(report)
        self.assertIn('determinizations | mean game | p95 game', output.getvalue())
        targets = [line.strip() for line in output.getvalue().splitlines() if '<- target' in line]
        self.assertEqual(len(targets), 2)
        self.assertTrue(all(line.startswith('0.1s |') for line in targets))
        self.assertIn('not wall-clock limits', output.getvalue())

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
        with patch('meeple_bots.analysis.measurement.analyze_structure') as sample:
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
        for calibration in (r.search_calibration, *r.configured_agent_benchmarks):
            target = next(row for row in calibration['budget_table'] if row['target'])
            self.assertAlmostEqual(target['estimated_mean_game_search_seconds'],
                                   .001 * r.structural['decisions_mean'])
            self.assertAlmostEqual(target['estimated_p95_game_search_seconds'],
                                   .001 * r.structural['estimated_depth'])
        output = io.StringIO()
        with redirect_stdout(output):
            print_analysis(r)
        self.assertIn('decision | iterations | mean game | p95 game', output.getvalue())


if __name__ == '__main__':
    unittest.main()
