"""Single-family study plans, measured work, paired statistics and resumable transport."""
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, PropertyMock
from types import SimpleNamespace

from meeple_bots import (MctsAgent, NeutralEvaluator, GameHeuristic, ConditionalRollout,
                        TurnPhaseIs, EpsilonGreedy, Mast, ProgressiveBias,
                        Connect6, Splendor, TicTacToe, benchmark_mcts_agent)
from meeple_bots._mcts_profiles import _load_mcts_profile
from meeple_bots.cli import build_parser, main
from meeple_bots.extraction import extract_tournament
from meeple_bots.studies import (StudyRunner, _build_phase, agent_from_values, duration_seconds,
                                export_profile, generic_baseline, profile_values, PHASES, cutoff_depths)
from meeple_bots.study_analysis import summarize_contrast, study_diagnostics, search_adequacy
from meeple_bots.api import SampledDecisionTiming


class StudyTests(unittest.TestCase):
    def test_completion_statistics_keep_legacy_measurements_unknown(self):
        rows = [self.row(7, seat, 'agent_b') for seat in (0, 1)]
        self.assertIsNone(summarize_contrast(rows)['timing_a']['terminal_fraction'])
        for row in rows:
            row['result']['moves'][0].update(terminal_simulations=7, cutoff_simulations=3)
        timing = summarize_contrast(rows)['timing_a']
        self.assertEqual(timing['terminal_simulations'], 14)
        self.assertEqual(timing['cutoff_simulations'], 6)
        self.assertEqual(timing['terminal_fraction'], .7)

    def test_profiles_preserve_independent_policies_and_parameters(self):
        agent = MctsAgent(iterations=1000, heuristic=0,
                          rollout_policy=ConditionalRollout(TurnPhaseIs('collect'), EpsilonGreedy(.2, GameHeuristic(0)), Mast(.1)),
                          progressive_bias=ProgressiveBias(.3, GameHeuristic(0), TurnPhaseIs('collect')),
                          tree_reuse=True, transpositions=True)
        self.assertEqual(agent_from_values(profile_values(agent)), agent)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'candidate.toml'
            export_profile(path, 'candidate', agent)
            self.assertEqual(_load_mcts_profile(path).agent, agent)

    def test_durations_and_cli(self):
        self.assertEqual(duration_seconds('2h'), 7200)
        self.assertEqual(duration_seconds('1.5m'), 90)
        for value in ('0', '-1s', 'nan', 'infinity', '2d'):
            with self.assertRaises(ValueError):
                duration_seconds(value)
        args = build_parser().parse_args(['study', '--game', 'boop', '--budget', '2h'])
        self.assertEqual(args.max_pairs, 8)
        self.assertEqual(args.confirmation_pairs, 32)
        self.assertIsNone(args.baseline)
        self.assertEqual(args.workers, 1)
        self.assertEqual(build_parser().parse_args(["study", "--game", "boop", "--budget", "2h", "--workers", "auto"]).workers, "auto")

    @staticmethod
    def row(seed, seat, winner):
        return {'winner': winner, 'agent_a_player': seat,
                'result': {'seed': seed, 'plies': 1, 'moves': [
                    {'ply': 1, 'player': seat, 'decision_seconds': .01, 'search_iterations': 10, 'search_nodes': 10}]}}

    def test_statistics_exclude_unpaired_games_and_do_not_claim_equivalence(self):
        rows = [self.row(7, 0, 'agent_b'), self.row(7, 1, 'agent_a'), self.row(8, 0, 'agent_b')]
        result = summarize_contrast(rows)
        self.assertEqual(result['score_b'], .5)
        self.assertEqual(result['seed_pairs'], 1)
        self.assertEqual(result['unpaired_games'], 1)
        self.assertEqual(result['verdict'], 'inconclusive')
        self.assertEqual(result['ci95_b'], [0, 1])
        ties = [self.row(seed, seat, None) for seed in range(20) for seat in (0, 1)]
        result = summarize_contrast(ties)
        self.assertLess(result['ci95_b'][0], .5)
        self.assertGreater(result['ci95_b'][1], .5)

    def test_resume_ignores_git_revision_but_rejects_changed_engine_hashes(self):
        with TemporaryDirectory() as tmp:
            base = MctsAgent(iterations=4, rollout_depth=9)
            options = dict(output=Path(tmp), budget=60, progress=lambda _: None)
            original_engine = {'git_revision': 'original', 'python_sha256': 'python', 'native_sha256': 'native'}
            with patch('meeple_bots.studies._fingerprint', return_value=original_engine):
                StudyRunner('tic-tac-toe', base, **options)
            new_commit = {**original_engine, 'git_revision': 'new-commit'}
            with patch('meeple_bots.studies._fingerprint', return_value=new_commit):
                resumed = StudyRunner('tic-tac-toe', base, resume=True, **options)
                self.assertEqual(resumed.state['request']['engine'], original_engine)
            for key in ('python_sha256', 'native_sha256'):
                with self.subTest(key=key), patch('meeple_bots.studies._fingerprint',
                                                return_value={**new_commit, key: 'changed'}):
                    with self.assertRaisesRegex(ValueError, 'configuration or engine changed'):
                        StudyRunner('tic-tac-toe', base, resume=True, **options)

    def test_interruption_reuses_flushed_single_seat_and_budget_stops_before_promotion(self):
        from meeple_bots.studies import run_matches as real_run
        with TemporaryDirectory() as tmp:
            output = Path(tmp)/'study'
            base = MctsAgent(iterations=4, rollout_depth=9)
            kwargs = dict(output=output, budget=60, decision_seconds=.00001,
                          max_pairs=2, confirmation_pairs=2, max_plies=9, progress=lambda _: None)
            def interrupt(*args, **options):
                for item in real_run(*args, **options):
                    yield item
                    raise RuntimeError('simulated interruption')
            with patch('meeple_bots.studies.run_matches', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    StudyRunner('tic-tac-toe', base, **kwargs).run()
            pilot = output/'traces/calibration-00.jsonl'
            prefix = pilot.read_bytes()
            self.assertEqual(len(prefix.splitlines()), 2)
            state = StudyRunner('tic-tac-toe', base, resume=True, **kwargs).run()
            self.assertEqual(state['status'], 'complete')
            self.assertTrue(pilot.read_bytes().startswith(prefix))
            self.assertEqual(len(pilot.read_bytes().splitlines()), 3)
        with TemporaryDirectory() as tmp:
            runner = StudyRunner('tic-tac-toe', base, output=Path(tmp)/'short', budget=.000001,
                                 decision_seconds=.01, max_pairs=2, confirmation_pairs=2, max_plies=9, progress=lambda _: None)
            state = runner.run()
            self.assertEqual(state['status'], 'budget_exhausted')
            self.assertEqual(state['phases'], {})
            self.assertEqual(state['candidate_profiles'], {})

    def state_for_plan(self, heuristic=None, rave=True):
        return {'request': {'version': 11, 'game': 'boop', 'mode': 'full_depth' if heuristic is None else 'heuristic_cutoff',
                            'heuristic': heuristic, 'target_match_time': 60, 'rave_search': rave,
                            'selection_search': True, 'mechanism_search': True, 'depth_search': True,
                            'selection_policies': ['uct', 'ucb1_tuned'] + (['uct_rave'] if rave else [])},
                'calibration': {'decision_seconds': .01, 'horizon': {'depth': 100},
                                'cutoff_depths': cutoff_depths(100)}, 'phases': {}}

    def populate(self, state, heuristic=None):
        base = replace(generic_baseline('boop'), cutoff_evaluator=NeutralEvaluator() if heuristic is None else GameHeuristic(heuristic))
        for name in PHASES:
            phase = _build_phase(name, state, base, None)
            for c in phase['contrasts']:
                c['result'] = {'score_b': .7 if name == 'rave' and phase['agents'][c['b']].get('rave_equivalence') == 10000 else .6, 'seed_pairs': 4}
            phase['status'] = 'complete'
            state['phases'][name] = phase
        return state

    def test_modes_and_all_candidate_evaluators_are_separate(self):
        for heuristic in (None, 0, 1):
            state = self.populate(self.state_for_plan(heuristic), heuristic)
            for phase in state['phases'].values():
                for values in phase['agents'].values():
                    expected = {'kind': 'neutral'} if heuristic is None else {'kind': 'game_heuristic', 'index': heuristic}
                    self.assertEqual(values['cutoff_evaluator'], expected)
                    self.assertEqual(values['time_budget'], .01)
                    self.assertNotIn('iterations', values)
                    if heuristic is None:
                        self.assertEqual(values['rollout_depth'], 100)
                    else:
                        self.assertLess(values['rollout_depth'], 100)
            self.assertNotIn('horizon_check', state['phases'])
            self.assertNotIn('horizons', state['phases'])

    def test_cutoff_grid_small_horizons_and_rounding(self):
        self.assertEqual(cutoff_depths(100), [10, 25, 50, 75])
        self.assertEqual(cutoff_depths(1), [])
        for horizon in range(2, 200):
            values = cutoff_depths(horizon)
            self.assertEqual(values, sorted(set(values)))
            self.assertTrue(all(1 <= x < horizon for x in values))
            self.assertLessEqual(len(values), 4)

    def test_depth_stage_is_incremental_and_partial_evidence_keeps_incumbent(self):
        state = self.populate(self.state_for_plan(0), 0)
        screen = state['phases']['depth_screen']
        for c in screen['contrasts']:
            c['target_pairs'] = 4
            c['result']['seed_pairs'] = 2
        phase = _build_phase('exploration', state, replace(generic_baseline('boop'), cutoff_evaluator=GameHeuristic(0)), None)
        self.assertEqual(len(phase['groups']), 1)
        self.assertEqual(phase['agents']['incumbent']['rollout_depth'], screen['agents']['initial']['rollout_depth'])

    def test_rave_capability_and_matrix(self):
        for enabled in (False, True):
            state = self.populate(self.state_for_plan(rave=enabled))
            rave = state['phases']['rave']
            samples = [v for v in rave['agents'].values() if v['selection_policy'] == 'uct_rave']
            self.assertEqual({v['rave_equivalence'] for v in samples}, {1000, 3000, 10000} if enabled else set())
            mechanism = state['phases']['mechanisms']
            self.assertEqual({(v['tree_reuse'], v['transpositions']) for v in mechanism['agents'].values()},
                             {(False, False), (True, False), (False, True), (True, True)})
            self.assertEqual(len(mechanism['contrasts']), 3)
            self.assertEqual(len({v['selection_policy'] for v in mechanism['agents'].values()}), 1)

    def test_default_target_and_custom_cli(self):
        parser = build_parser()
        defaults = parser.parse_args(['study', '--game', 'connect6', '--budget', '4h'])
        self.assertEqual(duration_seconds(defaults.target_match_time), 60)
        self.assertIsNone(defaults.heuristic)
        args = parser.parse_args(['study', '--game', 'splendor', '--budget', '4h', '--heuristic', 'h1', '--target-match-time', '2m'])
        self.assertEqual(args.heuristic, 1)
        self.assertEqual(duration_seconds(args.target_match_time), 120)
        with TemporaryDirectory() as tmp:
            for heuristic in (None, 1):
                runner = StudyRunner('splendor', output=Path(tmp)/str(heuristic), budget=60,
                                     heuristic=heuristic, progress=lambda _: None)
                self.assertEqual(runner.state['request']['mode'], 'full_depth' if heuristic is None else 'heuristic_cutoff')
                self.assertNotIn('uct_rave', runner.state['request']['selection_policies'])
            with self.assertRaisesRegex(ValueError, 'unknown cutoff heuristic'):
                StudyRunner('connect6', generic_baseline('connect6'), output=Path(tmp)/'bad', budget=60, heuristic=1)

    @staticmethod
    def fake_benchmark(game, agent, median_depth, seed):
        count = agent.iterations or max(1, int(agent.time_budget * 400))
        if agent.selection_policy == 'uct_rave':
            count = max(1, count // 3)
        terminal = count if agent.rollout_depth >= 128 else 0
        timings = tuple(SampledDecisionTiming(ply, count * 2.5, count, count+1, 300, terminal, count-terminal, ())
                        for ply in (0, 20, 40))
        return SimpleNamespace(maximum_decision_horizon=None, position_timings=timings)

    def calibrated_fake(self, tmp, **kwargs):
        runner = StudyRunner('boop', output=Path(tmp), budget=1000, progress=lambda _: None, **kwargs)
        rows = [{'result': {'plies': 80, 'chance_events': [None]*500}} for _ in range(2)]
        with patch.object(runner, '_pair', return_value=rows), patch('meeple_bots.studies.benchmark_mcts_agent', side_effect=self.fake_benchmark):
            runner.calibrate()
        return runner

    def test_practical_horizon_and_target_formula_excludes_chance(self):
        with TemporaryDirectory() as tmp:
            runner = self.calibrated_fake(tmp)
            cal = runner.state['calibration']
            self.assertEqual(cal['horizon']['kind'], 'practical_full_depth')
            self.assertEqual([s['depth'] for s in cal['horizon']['samples']], [64, 128])
            self.assertEqual(cal['decision_seconds'], .625)
            self.assertEqual(cal['estimated_game_decisions'], 80)
            self.assertEqual(cal['safety_adjusted_decisions'], 96)
            self.assertEqual(cal['center_iterations'], 250)
            self.assertEqual(cal['search_adequacy']['category'], 'VERY LOW')
            self.assertTrue(cal['warnings'])
            self.assertEqual(runner.state['request']['target_match_time'], 60)
            self.assertEqual([r['target_match_time'] for r in cal['search_adequacy']['suggested_targets']], [180, 600])
            self.assertNotIn('competitive_confidence', cal['search_adequacy'])
            self.assertIn('competitive_confidence', study_diagnostics(runner.state)['final_selection'])

    def test_smaller_target_measures_less_work_without_modifying_target(self):
        with TemporaryDirectory() as tmp:
            small = self.calibrated_fake(Path(tmp)/'small', target_match_time=6)
            large = self.calibrated_fake(Path(tmp)/'large', target_match_time=60)
            self.assertLess(small.state['calibration']['center_iterations'], large.state['calibration']['center_iterations'])
            self.assertEqual(small.state['request']['target_match_time'], 6)

    def test_hard_horizon_and_native_work_probes_preserve_board_size(self):
        for size in (9, 11):
            bench = benchmark_mcts_agent(Connect6(size), MctsAgent(iterations=4, rollout_depth=size*size, root_diagnostics=True), 12, 42)
            self.assertEqual(bench.maximum_decision_horizon, size*size)
            self.assertEqual(bench.position_timings[0].legal_actions, size*size)
            self.assertTrue(all(t.cutoff_simulations == 0 for t in bench.position_timings))
            self.assertTrue(bench.position_timings[0].root_visits)
        with TemporaryDirectory() as tmp:
            runner = StudyRunner('connect6', output=Path(tmp), budget=60,
                                 game_params={'board_size': 11}, target_match_time=.001, progress=lambda _: None)
            with patch.object(runner, '_pair', return_value=[{'result': {'plies': 40}}]*2):
                runner.calibrate()
            self.assertEqual(runner.state['calibration']['horizon']['depth'], 121)
            for name in PHASES:
                phase = _build_phase(name, runner.state, runner.base, None)
                runner._plan_phase(phase, PHASES.index(name))
                runner.state['phases'][name] = phase
                for i in range(len(phase['contrasts'])):
                    self.assertEqual(runner._trace_config(phase, i).game.board_size, 11)

    def test_native_time_budgets_and_different_policy_throughput(self):
        # Generous time ratio avoids requiring exact real-time scheduling in CI.
        reports = [benchmark_mcts_agent(Connect6(6), MctsAgent(iterations=None, time_budget=seconds, rollout_depth=36), 0, 10)
                   for seconds in (.002, .05)]
        self.assertGreater(reports[1].position_timings[0].iterations, reports[0].position_timings[0].iterations)
        uct = MctsAgent(iterations=None, time_budget=.1)
        rave = replace(uct, selection_policy='uct_rave')
        a = self.fake_benchmark(None, uct, 0, 0)
        b = self.fake_benchmark(None, rave, 0, 0)
        self.assertNotEqual(a.position_timings[0].iterations, b.position_timings[0].iterations)

    def test_native_stochastic_sampling_counts_decisions_and_resolves_chance(self):
        bench = benchmark_mcts_agent(Splendor(), MctsAgent(iterations=2, rollout_depth=2), 6, 42)
        self.assertIsNone(bench.maximum_decision_horizon)
        self.assertEqual([t.sampled_ply for t in bench.position_timings], [0, 2, 4])
        self.assertTrue(all(t.terminal_simulations + t.cutoff_simulations == 2 for t in bench.position_timings))
        self.assertTrue(all(t.legal_actions > 0 for t in bench.position_timings))

    def test_complete_small_study_fresh_confirmation_extract_and_resume(self):
        with TemporaryDirectory() as tmp:
            options = dict(output=Path(tmp), budget=60, decision_seconds=.00001, max_pairs=2, confirmation_pairs=2,
                           max_plies=9, workers=2, rave_search=True, progress=lambda _: None)
            runner = StudyRunner('tic-tac-toe', generic_baseline('tic-tac-toe'), **options)
            state = runner.run()
            self.assertEqual(state['status'], 'complete')
            self.assertIn('best_agent', state['candidate_profiles'])
            report = json.loads((Path(tmp)/'summary.json').read_text())
            self.assertEqual(report['calibration']['target_match_time'], 60)
            self.assertEqual(report['calibration']['decision_seconds'], .00001)
            seeds = {}
            files = {}
            for path in (Path(tmp)/'traces').glob('*.jsonl'):
                files[path] = path.read_bytes()
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                seeds[path.name] = {r['result']['seed'] for r in rows[1:]}
                self.assertEqual(rows[0]['workers'], 1)
            confirmation = set.union(*(v for k,v in seeds.items() if k.startswith('confirmation')))
            screening = set.union(*(v for k,v in seeds.items() if not k.startswith('confirmation')))
            self.assertFalse(confirmation & screening)
            first = next((Path(tmp)/'traces').glob('confirmation*.jsonl'))
            extracted = extract_tournament(first, Path(tmp)/'extracted')
            self.assertTrue(extracted['complete'])
            with patch('meeple_bots.studies.run_matches', side_effect=AssertionError('replay')):
                resumed = StudyRunner('tic-tac-toe', generic_baseline('tic-tac-toe'), resume=True, **options).run()
            self.assertEqual(resumed['status'], 'complete')
            self.assertTrue(all(p.read_bytes() == data for p,data in files.items()))

    def test_phase_budget_protects_confirmation_and_never_changes_time(self):
        with TemporaryDirectory() as tmp:
            runner = self.calibrated_fake(tmp)
            phase = _build_phase('depth_screen', runner.state, runner.base, None)
            runner._plan_phase(phase, 0)
            phase = _build_phase('exploration', {**runner.state, 'phases': {'depth_screen': phase}}, runner.base, None)
            runner._plan_phase(phase, 1)
            self.assertLessEqual(phase['estimated_seconds'], phase['allocated_seconds'])
            self.assertTrue(all(v['time_budget'] == .625 for v in phase['agents'].values()))

    def test_old_checkpoint_rejected_without_modification(self):
        with TemporaryDirectory() as tmp:
            options = dict(output=Path(tmp), budget=60, progress=lambda _: None)
            runner = StudyRunner('tic-tac-toe', generic_baseline('tic-tac-toe'), **options)
            old = runner.state
            old['request']['version'] = 10
            runner.path.write_text(json.dumps(old))
            before = runner.path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'old study protocol'):
                StudyRunner('tic-tac-toe', generic_baseline('tic-tac-toe'), resume=True, **options)
            self.assertEqual(runner.path.read_bytes(), before)

    def test_root_coverage_is_measured_and_unvisited_actions_are_zero(self):
        diagnostic = search_adequacy([{'iterations': 3, 'legal_actions': 10, 'milliseconds': 20,
                                      'root_visits': [2, 1]}], 60)
        self.assertEqual(diagnostic['median_root_coverage'], .2)
        self.assertEqual(diagnostic['median_visits_per_root_action'], 0)
        self.assertEqual(diagnostic['p10_visits_per_root_action'], 0)
        self.assertEqual(diagnostic['iterations_completed'], 3)
        self.assertEqual(diagnostic['iterations_per_second'], 150)

    def test_practical_horizon_reports_failure_instead_of_claiming_full_depth(self):
        with TemporaryDirectory() as tmp:
            runner = self.calibrated_fake(tmp, max_plies=64)
            horizon = runner.state['calibration']['horizon']
            self.assertEqual(horizon['kind'], 'practical_unverified')
            self.assertEqual(horizon['terminal_fraction'], 0)
            self.assertTrue(any('unverified' in w for w in runner.state['calibration']['warnings']))

    def test_heuristic_study_transport_keeps_chance_traces_and_no_cross_family(self):
        with TemporaryDirectory() as tmp:
            runner = StudyRunner('splendor', heuristic=1, depth_search=True, output=Path(tmp), budget=60,
                                 target_match_time=.001, max_pairs=2, max_plies=3000, progress=lambda _: None)
            runner.state['calibration'] = {'decision_seconds': .00001, 'horizon': {'depth': 32},
                                           'cutoff_depths': [4, 8], 'mean_plies': 80}
            phase = _build_phase('depth_screen', runner.state, runner.base, None)
            phase['planned_pairs'] = 1
            runner.state['phases']['depth_screen'] = phase
            rows = runner._pair(phase, 0, 0)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r['result']['chance_events'] for r in rows))
            self.assertTrue(all(len(r['result']['moves']) == r['result']['plies'] for r in rows))
            path = runner._trace_config(phase, 0).output
            self.assertTrue(extract_tournament(path, Path(tmp)/'data')['complete'])
            for values in phase['agents'].values():
                self.assertEqual(values['cutoff_evaluator'], {'kind': 'game_heuristic', 'index': 1})
                self.assertLess(values['rollout_depth'], 32)

    def test_nominee_stays_provisional_when_confirmation_inconclusive(self):
        state = self.populate(self.state_for_plan())
        primary = state['phases']['confirmation']['contrasts'][0]
        primary['target_pairs'] = 4
        primary['result'] = {'score_b': .5, 'seed_pairs': 4, 'ci95_b': [.1, .9], 'verdict': 'inconclusive'}
        diagnostic = study_diagnostics(state)
        self.assertEqual(diagnostic['final_selection']['candidate'], 'finalist')
        self.assertEqual(diagnostic['final_selection']['status'], 'provisional')
        self.assertEqual(diagnostic['final_selection']['competitive_confidence'], 'inconclusive')

    def test_explicit_decision_override_does_not_recommend_ineffective_match_targets(self):
        with TemporaryDirectory() as tmp:
            runner = self.calibrated_fake(tmp, decision_seconds=.001)
            self.assertEqual(runner.state['calibration']['decision_time_source'], 'explicit_override')
            self.assertEqual(runner.state['calibration']['search_adequacy']['suggested_targets'], [])

    def test_unsupported_reference_rejected_before_creating_output(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp)/'invalid'
            with self.assertRaisesRegex(ValueError, 'selection policy is unavailable'):
                StudyRunner('splendor', generic_baseline('splendor'), output=output, budget=60,
                            reference=MctsAgent(selection_policy='uct_rave'))
            self.assertFalse(output.exists())

    def test_rave_progression_and_selector_comparison_order(self):
        state = self.populate(self.state_for_plan())
        for name, k in [('rave', 10000), ('rave_extend_1', 20000), ('rave_extend_2', 40000), ('rave_extend_3', 80000), ('rave_extend_4', 160000), ('rave_extend_5', 320000)]:
            phase = state['phases'][name]
            self.assertEqual(len(phase['contrasts']), 2)
            self.assertIn(k, [phase['agents'][c['b']]['rave_equivalence'] for c in phase['contrasts']])
            self.assertTrue(all(v['selection_policy'] == 'uct_rave' for v in phase['agents'].values()))
        tuned = state['phases']['rave_exploration']
        self.assertEqual(len(tuned['contrasts']), 2)
        self.assertTrue(all(v['selection_policy'] == 'uct_rave' for v in tuned['agents'].values()))
        comparison = state['phases']['rave_compare']
        self.assertEqual(len(comparison['contrasts']), 1)
        c = comparison['contrasts'][0]
        self.assertEqual(comparison['agents'][c['a']]['selection_policy'], 'ucb1_tuned')
        self.assertEqual(comparison['agents'][c['b']]['selection_policy'], 'uct_rave')
        self.assertEqual(comparison['agents'][c['b']]['rave_equivalence'], 320000)
        self.assertGreater(PHASES.index('rave_compare'), PHASES.index('rave_exploration'))

    def test_rave_stops_expansion_without_claiming_plateau(self):
        for score in (.4, .5, .54):
            state = self.populate(self.state_for_plan())
            for c in state['phases']['rave_extend_1']['contrasts']:
                c['result']['score_b'] = score
            for name in ('rave_extend_2', 'rave_extend_3', 'rave_extend_4', 'rave_extend_5', 'rave_exploration'):
                phase = _build_phase(name, state, generic_baseline('boop'), None)
                state['phases'][name] = phase
                if name != 'rave_exploration':
                    self.assertFalse(phase['contrasts'])
                    self.assertEqual(set(phase['rave_decisions'].values()), {'no_clear_improvement_not_proven_plateau'})
            self.assertEqual(len(phase['contrasts']), 2)

    def test_rave_checks_geometric_gaps_and_tracks_interior_winner(self):
        state = self.populate(self.state_for_plan())
        for c in state['phases']['rave']['contrasts']:
            c['result']['score_b'] = .4
        phase = _build_phase('rave_extend_1', state, generic_baseline('boop'), None)
        self.assertEqual({v['rave_equivalence'] for v in phase['agents'].values()}, {1732, 3000, 5477})
        for c in phase['contrasts']:
            c['result'] = {'score_b': .7 if phase['agents'][c['b']]['rave_equivalence'] == 5477 else .4, 'seed_pairs': 4}
        state['phases']['rave_extend_1'] = phase
        second = _build_phase('rave_extend_2', state, generic_baseline('boop'), None)
        ks = {v['rave_equivalence'] for v in second['agents'].values()}
        self.assertEqual(ks, {4054, 5477, 7401})
        self.assertTrue(all(k < 10000 for k in ks))

    def test_high_rave_boundary_also_checks_lower_gap(self):
        state = self.populate(self.state_for_plan())
        phase = state['phases']['rave_extend_1']
        self.assertEqual({v['rave_equivalence'] for v in phase['agents'].values()}, {5477, 10000, 20000})

    def test_incomplete_rave_calibration_never_competes_with_selector(self):
        state = self.populate(self.state_for_plan())
        state['phases']['rave']['contrasts'][0]['result'] = {}
        for name in ('rave_extend_1', 'rave_extend_2', 'rave_extend_3', 'rave_extend_4', 'rave_extend_5', 'rave_exploration', 'rave_compare'):
            phase = _build_phase(name, state, generic_baseline('boop'), None)
            state['phases'][name] = phase
            self.assertFalse(phase['contrasts'])
        self.assertTrue(all(v['selection_policy'] != 'uct_rave' for v in phase['agents'].values()))

    def test_progressive_rave_keeps_allocation_and_frozen_resume(self):
        with TemporaryDirectory() as tmp:
            runner = self.calibrated_fake(tmp, rave_search=True)
            planned = self.populate(self.state_for_plan())
            runner.state['phases'] = planned['phases']
            phase = _build_phase('rave_extend_1', runner.state, runner.base, None)
            runner._plan_phase(phase, PHASES.index('rave_extend_1'))
            runner.state['phases']['rave_extend_1'] = phase
            runner.save()
            self.assertLessEqual(phase['estimated_seconds'], phase['allocated_seconds'])
            for planned_phase in runner.state['phases'].values():
                planned_phase.setdefault('planned_pairs', 2)
            runner.save()
            seeds = [runner._trace_config(runner.state['phases'][name], 0).seed
                     for name in ('rave_extend_1', 'rave_extend_2', 'rave_extend_3', 'rave_extend_4', 'rave_extend_5', 'rave_exploration', 'rave_compare')]
            self.assertEqual(len(set(seeds)), 7)
            resumed = StudyRunner('boop', output=Path(tmp), budget=1000, resume=True, rave_search=True, progress=lambda _: None)
            self.assertEqual(resumed.state['phases']['rave_extend_1'], phase)
            self.assertEqual(resumed.state['rave_budget'], runner.state['rave_budget'])

    def test_incremental_baseline_is_preserved_and_disabled_stages_are_free(self):
        from meeple_bots.studies import _phase_enabled
        base = MctsAgent(time_budget=.01, rollout_depth=169, exploration=.125,
                         selection_policy='uct_rave', rave_equivalence=20000,
                         tree_reuse=True, transpositions=True)
        with TemporaryDirectory() as tmp:
            runner = StudyRunner('connect6', base, output=Path(tmp), budget=60,
                                 game_params={'board_size': 13}, progress=lambda _: None)
            self.assertEqual(runner.base, base)
            rows = [{'result': {'plies': 80}}]*2
            with patch.object(runner, '_pair', return_value=rows), patch('meeple_bots.studies.benchmark_mcts_agent', side_effect=self.fake_benchmark):
                runner.calibrate()
            self.assertEqual(runner.state['calibration']['horizon']['kind'], 'baseline')
            for name in PHASES:
                phase = _build_phase(name, runner.state, base, None)
                runner._plan_phase(phase, PHASES.index(name))
                runner.state['phases'][name] = phase
                if name not in ('refinement', 'confirmation'):
                    self.assertFalse(phase['contrasts'])
                    self.assertEqual(phase['allocated_seconds'], 0)
                    self.assertEqual(phase['planned_pairs'], 0)
                    self.assertTrue(all(agent_from_values(v) == base for v in phase['agents'].values()))
                else:
                    self.assertTrue(_phase_enabled(name, runner.state['request']))
            self.assertTrue(runner.state['phases']['refinement']['contrasts'])

    def test_incremental_native_pipeline_all_stages_export_and_resume(self):
        with TemporaryDirectory() as tmp:
            base = MctsAgent(iterations=4, rollout_depth=9)
            opts = dict(output=Path(tmp), budget=60, all_search=True, max_pairs=2,
                        confirmation_pairs=2, max_plies=9, progress=lambda _: None)
            runner = StudyRunner('tic-tac-toe', base, **opts)
            state = runner.run()
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(state['phases']['confirmation']['status'], 'complete')
            self.assertTrue((Path(tmp)/'candidates/best_agent.toml').exists())
            for phase in state['phases'].values():
                for agent in phase['agents'].values():
                    self.assertEqual(agent['iterations'], 4)
                    self.assertNotIn('time_budget', agent)
            with patch('meeple_bots.studies.run_matches', side_effect=AssertionError('unexpected rerun')):
                resumed = StudyRunner('tic-tac-toe', base, resume=True, **opts).run()
            self.assertEqual(resumed['status'], 'complete')
            report = (Path(tmp)/'report.html').read_text()
            self.assertIn('Incremental stages', report)
            self.assertIn('fixed_iterations', report)

    def test_incremental_fixed_iterations_and_explicit_time_override(self):
        base = MctsAgent(iterations=77, rollout_depth=16, heuristic=1)
        for override in (None, .02):
            with TemporaryDirectory() as tmp:
                runner = StudyRunner('splendor', base, output=Path(tmp), budget=60,
                                     decision_seconds=override, progress=lambda _: None)
                with patch.object(runner, '_pair', return_value=[{'result': {'plies': 80}}]*2), patch('meeple_bots.studies.benchmark_mcts_agent', side_effect=self.fake_benchmark):
                    runner.calibrate()
                phase = _build_phase('depth_screen', runner.state, base, None)
                got = agent_from_values(phase['agents']['initial'])
                expected = base if override is None else replace(base, iterations=None, time_budget=override)
                self.assertEqual(got, expected)
                self.assertEqual(runner.state['request']['heuristic'], 1)
                with self.assertRaisesRegex(ValueError, 'conflicts'):
                    StudyRunner('splendor', base, output=Path(tmp)/'conflict', budget=60, heuristic=0)

    def test_incremental_only_pw_copies_current_and_confirmation_uses_initial(self):
        base = MctsAgent(time_budget=.01, rollout_depth=169, exploration=.125,
                         selection_policy='uct_rave', rave_equivalence=20000, tree_reuse=True, transpositions=True)
        state = self.state_for_plan()
        state['request'].update(baseline_supplied=True, selection_search=False, mechanism_search=False,
                                depth_search=False, rave_search=False, pw_search=True, pw_supported=True)
        state['calibration']['horizon']['depth'] = 169
        for name in PHASES:
            phase = _build_phase(name, state, base, None)
            for c in phase['contrasts']:
                c['result'] = {'seed_pairs': 4, 'score_b': .6}
            state['phases'][name] = phase
            if name not in ('refinement', 'confirmation') and not name.startswith('pw_'):
                self.assertFalse(phase['contrasts'])
                self.assertEqual(agent_from_values(next(iter(phase['agents'].values()))), base)
        final = state['phases']['confirmation']
        self.assertEqual(agent_from_values(final['agents']['runner-up']), base)
        self.assertTrue(final['agents']['finalist']['progressive_widening'])
        compare = state['phases']['pw_compare']
        compare['contrasts'][0]['result']['score_b'] = .4
        refined = _build_phase('refinement', state, base, None)
        self.assertEqual(agent_from_values(refined['agents']['incumbent']), base)

    def test_all_search_enables_supported_stages_and_generated_baseline(self):
        with TemporaryDirectory() as tmp:
            runner = StudyRunner('connect6', output=Path(tmp), budget=60, all_search=True,
                                 game_params={'board_size': 11}, progress=lambda _: None)
            self.assertFalse(runner.state['request']['baseline_supplied'])
            for key in ('selection_search', 'rave_search', 'mechanism_search', 'pw_search', 'depth_search'):
                self.assertTrue(runner.state['request'][key])
            args = build_parser().parse_args(['study', '--game', 'connect6', '--budget', '4h', '--all-search'])
            self.assertTrue(args.all_search)

    def test_pw_opt_in_and_calibrate_before_cross_comparison(self):
        args = ['study', '--game', 'connect6', '--budget', '4h']
        self.assertFalse(build_parser().parse_args(args).pw_search)
        self.assertTrue(build_parser().parse_args(args + ['--pw-search']).pw_search)
        for rave in (False, True):
            state = self.state_for_plan(rave=rave)
            state['request'].update(pw_search=True, pw_supported=True)
            self.populate(state)
            original = next(iter(state['phases']['pw_compare']['agents'].values()))
            for stage in ('pw_k', 'pw_alpha', 'pw_refine'):
                phase = state['phases'][stage]
                self.assertTrue(phase['contrasts'])
                for values in phase['agents'].values():
                    self.assertTrue(values['progressive_widening'])
                    for key in original:
                        if not key.startswith('progressive_widening'):
                            self.assertEqual(values[key], original[key])
            comparison = state['phases']['pw_compare']
            self.assertEqual(len(comparison['contrasts']), 1)
            self.assertFalse(comparison['agents']['incumbent'].get('progressive_widening', False))
            self.assertTrue(comparison['agents']['calibrated-pw']['progressive_widening'])
            self.assertGreater(PHASES.index('pw_compare'), PHASES.index('pw_refine'))

    def test_pw_incomplete_and_unsupported_keep_incumbent(self):
        for enabled, supported in [(False, True), (True, False), (True, True)]:
            state = self.state_for_plan()
            state['request'].update(pw_search=enabled, pw_supported=supported)
            self.populate(state)
            if enabled and supported:
                state['phases']['pw_k']['contrasts'][0]['result']['seed_pairs'] = 1
                for name in ('pw_alpha', 'pw_refine', 'pw_compare'):
                    state['phases'][name] = _build_phase(name, state, generic_baseline('boop'), None)
            comparison = state['phases']['pw_compare']
            self.assertFalse(comparison['contrasts'])
            self.assertFalse(any(v.get('progressive_widening', False) for v in comparison['agents'].values()))

    def test_pw_pool_reserves_followups_and_resume_preserves_plan(self):
        with TemporaryDirectory() as tmp:
            runner = self.calibrated_fake(tmp, pw_search=True)
            state = self.state_for_plan()
            state['request'].update(pw_search=True, pw_supported=True)
            runner.state['phases'] = self.populate(state)['phases']
            phase = _build_phase('pw_k', runner.state, runner.base, None)
            runner._plan_phase(phase, PHASES.index('pw_k'))
            self.assertGreater(phase['reserved_followup_seconds'], 0)
            self.assertLessEqual(phase['estimated_seconds'], phase['allocated_seconds'])
            runner.state['phases']['pw_k'] = phase
            runner.save()
            resumed = StudyRunner('boop', output=Path(tmp), budget=1000,
                                  resume=True, pw_search=True, progress=lambda _: None)
            self.assertEqual(resumed.state['pw_budget'], runner.state['pw_budget'])
            self.assertEqual(resumed.state['phases']['pw_k'], phase)
            with self.assertRaisesRegex(ValueError, 'configuration or engine changed'):
                StudyRunner('boop', output=Path(tmp), budget=1000,
                            resume=True, pw_search=False, progress=lambda _: None)
            phase['planned_pairs'] = 2
            seeds = []
            for name in ('pw_k', 'pw_alpha', 'pw_refine', 'pw_compare', 'confirmation'):
                item = runner.state['phases'][name]
                item.setdefault('planned_pairs', 2)
                seeds.append(runner._trace_config(item, 0).seed)
            self.assertEqual(len(seeds), len(set(seeds)))

    def test_high_rave_winner_refines_above_original_grid(self):
        state = self.populate(self.state_for_plan())
        for values in state['phases']['pw_compare']['agents'].values():
            values.update(selection_policy='uct_rave', rave_equivalence=20000)
        phase = _build_phase('refinement', state, generic_baseline('boop'), None)
        ks = {phase['agents'][c['b']]['rave_equivalence'] for c in phase['contrasts'] if c['factor'] == 'rave_equivalence'}
        self.assertEqual(ks, {10000, 40000})

    def test_rave_search_is_explicit_cli_opt_in(self):
        args = ['study', '--game', 'connect6', '--budget', '4h']
        self.assertFalse(build_parser().parse_args(args).rave_search)
        self.assertTrue(build_parser().parse_args(args + ['--rave-search']).rave_search)
        state = self.state_for_plan()
        state['request']['rave_search'] = False
        self.populate(state)
        for name, phase in state['phases'].items():
            self.assertTrue(all(v['selection_policy'] != 'uct_rave' for v in phase['agents'].values()))
            if name.startswith('rave'):
                self.assertFalse(phase['contrasts'])

    def test_disabled_rave_has_no_allocation_and_resume_rejects_option_change(self):
        with TemporaryDirectory() as tmp:
            runner = self.calibrated_fake(tmp)
            state = self.state_for_plan()
            state['request']['rave_search'] = False
            runner.state['phases'] = self.populate(state)['phases']
            for name, phase in runner.state['phases'].items():
                runner._plan_phase(phase, PHASES.index(name))
                if name.startswith('rave'):
                    self.assertEqual(phase['allocated_seconds'], 0)
                    self.assertEqual(phase['planned_pairs'], 0)
            runner.save()
            with self.assertRaisesRegex(ValueError, 'configuration or engine changed'):
                StudyRunner('boop', output=Path(tmp), budget=1000,
                            resume=True, rave_search=True, progress=lambda _: None)

    def test_shared_rave_pool_covers_initial_screen_and_keeps_unspent_balance(self):
        with TemporaryDirectory() as tmp, patch.object(StudyRunner, 'spent', new_callable=PropertyMock, return_value=0) as clock:
            runner = self.calibrated_fake(tmp, rave_search=True)
            runner.budget = 14400
            runner.state['calibration'].update(mean_plies=91.5, decision_seconds=60/109.8)
            runner.state['phases'] = self.populate(self.state_for_plan())['phases']
            # No competitive timings yet: reproduce the original conservative 100s/pair estimate.
            phase = _build_phase('rave', runner.state, runner.base, None)
            runner._plan_phase(phase, PHASES.index('rave'))
            self.assertTrue(all(c['target_pairs'] >= 2 for c in phase['contrasts']))
            pool = runner.state['rave_budget']['allocated_seconds']
            self.assertGreater(pool, 14400*.25)
            self.assertGreaterEqual(phase['reserved_followup_seconds'], 999)
            self.assertLessEqual(phase['estimated_seconds'] + phase['reserved_followup_seconds'], pool)
            phase['status'] = 'complete'
            for c in phase['contrasts']:
                c['result'] = {'seed_pairs': c['target_pairs'], 'score_b': .7,
                               'timing_a': {'total_seconds': 15*c['target_pairs']},
                               'timing_b': {'total_seconds': 15*c['target_pairs']}}
            runner.state['phases'] = {'rave': phase}
            clock.return_value = 240
            extension = _build_phase('rave_extend_1', runner.state, runner.base, None)
            runner._plan_phase(extension, PHASES.index('rave_extend_1'))
            self.assertEqual(runner.state['rave_budget']['allocated_seconds'], pool)
            self.assertAlmostEqual(extension['shared_budget_remaining_seconds'], pool-240)
            self.assertAlmostEqual(runner._cost_pair(extension, extension['contrasts'][0]), 36)
            self.assertAlmostEqual(extension['reserved_followup_seconds'], 216)
            self.assertTrue(all(c['target_pairs'] >= 2 for c in extension['contrasts']))
            # Final comparison receives the remaining pool, with no upstream subquota.
            final = {**extension, 'name': 'rave_compare'}
            runner._plan_phase(final, PHASES.index('rave_compare'))
            self.assertEqual(final['reserved_followup_seconds'], 0)
            self.assertAlmostEqual(final['allocated_seconds'], pool-240)

    def test_shared_rave_budget_requires_complete_coverage_before_extra_pairs(self):
        with TemporaryDirectory() as tmp, patch.object(StudyRunner, 'spent', new_callable=PropertyMock, return_value=0):
            runner = self.calibrated_fake(tmp, rave_search=True)
            runner.state['phases'] = self.populate(self.state_for_plan())['phases']
            runner.budget = 2000
            # 100s per pair. Reserve 1000s follow-up; initial coverage requires 400s.
            runner.state['rave_budget'] = {'allocated_seconds': 1399, 'started_spent': 0}
            phase = _build_phase('rave', runner.state, runner.base, None)
            runner._plan_phase(phase, PHASES.index('rave'))
            self.assertEqual([c['target_pairs'] for c in phase['contrasts']], [0, 0])
            runner.state['rave_budget']['allocated_seconds'] = 1400
            phase = _build_phase('rave', runner.state, runner.base, None)
            runner._plan_phase(phase, PHASES.index('rave'))
            self.assertEqual([c['target_pairs'] for c in phase['contrasts']], [2, 2])



if __name__ == '__main__':
    unittest.main()
