"""Automatic study plans, paired statistics, persistence and real native integration."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from meeple_bots import MctsAgent, NeutralEvaluator, GameHeuristic, ConditionalRollout, TurnPhaseIs, EpsilonGreedy, Mast, ProgressiveBias
from meeple_bots._mcts_profiles import _load_mcts_profile
from meeple_bots.cli import build_parser
from meeple_bots.extraction import extract_tournament
from meeple_bots.studies import (
    StudyRunner, _build_phase, agent_from_values, duration_seconds, export_profile,
    generic_baseline, mechanism_plan, profile_values, PHASES, _family, _horizon_survivors,
)
from meeple_bots.study_analysis import mechanism_effects, summarize_contrast, write_study_report, study_diagnostics


class StudyTests(unittest.TestCase):
    def test_splendor_calibration_chance_traces_candidates_and_resume(self):
        from meeple_bots import NeutralEvaluator, Splendor, benchmark_mcts_agent
        base = MctsAgent(iterations=2, rollout_depth=4, heuristic=0)
        self.assertEqual(generic_baseline('splendor').cutoff_evaluator, GameHeuristic(0))
        args = build_parser().parse_args(['study', '--game', 'splendor', '--budget', '2h'])
        self.assertEqual(args.game, 'splendor')
        benchmark = benchmark_mcts_agent(Splendor(), base, 60, 42)
        self.assertGreaterEqual(benchmark.sampled_positions, 2)
        self.assertGreater(benchmark.milliseconds_per_iteration, 0)
        # Exercise every real phase using only two contrasts each, with tiny search budgets.
        def small_plan(*args):
            phase = _build_phase(*args)
            if phase['name'] == 'horizons':
                phase['contrasts'] = phase['contrasts'][:1]
            else:
                selected = []
                for family in ('full', 'cutoff'):
                    candidates = [c for c in phase['contrasts']
                                  if _family(phase['agents'][c['b']]) == family
                                  and (phase['name'] != 'iterations' or c['factor'] == 'anchor')]
                    if candidates:
                        selected.append(candidates[0])
                phase['contrasts'] = selected
            return phase
        with TemporaryDirectory() as tmp:
            options = dict(output=Path(tmp), budget=60, workers=2,
                           decision_seconds=.00001, max_pairs=2, confirmation_pairs=2, max_plies=3000,
                           progress=lambda _: None)
            with patch('meeple_bots.studies._build_phase', side_effect=small_plan):
                result = StudyRunner('splendor', base, **options).run()
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(result['candidate_profiles'])
            for path in result['candidate_profiles'].values():
                self.assertIsInstance(_load_mcts_profile(Path(tmp)/path).agent.cutoff_evaluator, NeutralEvaluator)
            traces = {p: p.read_bytes() for p in (Path(tmp)/'traces').glob('*.jsonl')}
            self.assertTrue(traces)
            for data in traces.values():
                rows = [json.loads(line) for line in data.splitlines()][1:]
                self.assertTrue(all(row['result']['chance_events'] for row in rows))
                self.assertTrue(all(row['result']['splendor_state']['finished'] for row in rows))
            self.assertTrue((Path(tmp)/'report.html').exists())
            with patch('meeple_bots.studies.run_matches', side_effect=AssertionError('replayed matches')):
                resumed = StudyRunner('splendor', base, resume=True, **options).run()
            self.assertEqual(resumed['status'], 'complete')
            for path, data in traces.items():
                self.assertEqual(path.read_bytes(), data)

    def test_registered_h0_crosses_cutoff_with_selector_depth_and_exploration(self):
        for game in ('splendor', 'boop', 'spotf', 'tic-tac-toe', 'connect-four'):
            with self.subTest(game=game):
                state = {'request': {'game': game}, 'calibration': {'decision_seconds': .01, 'center_iterations': 4}}
                phase = _build_phase('horizons', state, generic_baseline(game), None)
                families = {}
                for name, values in phase['agents'].items():
                    if name == 'full':
                        continue
                    key = (values['selection_policy'], values['rollout_depth'], values['exploration'])
                    families.setdefault(key, set()).add(values['cutoff_evaluator']['kind'])
                    self.assertEqual(agent_from_values(values).cutoff_evaluator,
                                     GameHeuristic(0) if name.endswith('-h0') else NeutralEvaluator())
                expected = {'neutral', 'game_heuristic'} if game in ('splendor', 'boop', 'spotf') else {'neutral'}
                self.assertEqual(len(families), 3)
                self.assertTrue(all(kinds == expected for kinds in families.values()))

    def test_cutoff_screening_preserves_custom_baseline_without_duplicate_candidates(self):
        state = {'request': {'game': 'boop'}, 'calibration': {'decision_seconds': .01, 'center_iterations': 4}}
        for evaluator, expected in ((NeutralEvaluator(), 2), (GameHeuristic(0), 2),
                                    (GameHeuristic(1), 3)):
            phase = _build_phase('horizons', state, MctsAgent(cutoff_evaluator=evaluator), None)
            self.assertEqual(len(phase['agents']), 1 + 3 * expected)
            self.assertTrue(any(agent_from_values(v).cutoff_evaluator == evaluator
                                for n, v in phase['agents'].items() if n != 'anchor'))
        # Parameterized H0 must coexist with the default H0 without overwriting its names.
        custom = GameHeuristic(0, {'gemstone_early_bonus': 2.0})
        state['request']['game'] = 'spotf'
        phase = _build_phase('horizons', state, MctsAgent(cutoff_evaluator=custom), None)
        self.assertEqual(len(phase['agents']), 10)
        self.assertTrue(any(n.endswith('-h0-custom') for n in phase['agents']))
        self.assertTrue(any(agent_from_values(v).cutoff_evaluator == custom for v in phase['agents'].values()))

    def test_terminal_controls_and_cutoff_admission_boundary(self):
        state = {'request': {'game': 'splendor'},
                 'calibration': {'decision_seconds': .01, 'center_iterations': 4}, 'phases': {}}
        phase = _build_phase('horizons', state, generic_baseline('splendor'), None)
        self.assertEqual(len(phase['contrasts']), 6)
        for contrast in phase['contrasts']:
            control, cutoff = (phase['agents'][contrast[role]] for role in ('a', 'b'))
            self.assertEqual(control['rollout_depth'], 1024)
            self.assertEqual(control['cutoff_evaluator'], {'kind': 'neutral'})
            self.assertIn(cutoff['rollout_depth'], (16, 32, 64))
            for field in ('time_budget', 'selection_policy', 'exploration', 'tree_reuse', 'transpositions'):
                self.assertEqual(control[field], cutoff[field])
            contrast['result'] = {'score_b': .1, 'seed_pairs': 16, 'ci95_b': [0, .44]}
        phase['status'] = 'complete'
        self.assertEqual(_horizon_survivors(phase), ['full'])
        phase['contrasts'][0]['result'].update(score_b=.45, ci95_b=[.1, .8])
        self.assertEqual(_horizon_survivors(phase), ['full', phase['contrasts'][0]['b']])
        state['phases']['horizons'] = phase
        params = _build_phase('parameters', state, generic_baseline('splendor'), None)
        for family in ('full', 'cutoff'):
            candidates = [a for name, a in params['agents'].items() if name != 'anchor' and _family(a) == family]
            self.assertEqual({a['selection_policy'] for a in candidates}, {'uct', 'ucb1_tuned'})
        phase['status'] = 'complete'
        with TemporaryDirectory() as tmp:
            runner = StudyRunner('splendor', generic_baseline('splendor'), output=Path(tmp), budget=1)
            runner.state['phases']['horizons'] = phase
            runner.export_candidates()
            self.assertIn('full', runner.state['candidate_profiles'])

    def test_cutoff_selection_distinguishes_pending_rejected_and_admitted(self):
        phase = {'status': 'pending', 'contrasts': [
            {'a': 'full', 'b': 'cutoff', 'factor': 'cutoff_equal_time', 'result': {'score_b': .7}}]}
        state = {'phases': {'horizons': phase}}
        self.assertIsNone(study_diagnostics(state)['cutoff_selection']['admitted'])
        phase['status'] = 'complete'
        selection = study_diagnostics(state)['cutoff_selection']
        self.assertTrue(selection['admitted'])
        self.assertEqual(selection['candidate'], 'cutoff')
        phase['contrasts'][0]['result'].update(score_b=.1, seed_pairs=16, ci95_b=[0, .44])
        selection = study_diagnostics(state)['cutoff_selection']
        self.assertFalse(selection['admitted'])
        self.assertIsNone(selection['candidate'])
        self.assertEqual(selection['best_observed_score'], .1)

    def test_improvements_are_direct_conditional_advantages_not_strength_percentages(self):
        original = profile_values(MctsAgent(time_budget=.01, rollout_depth=1024))
        agents = {'parent': original, 'explore': {**original, 'exploration': 2.0},
                  'reuse': {**original, 'tree_reuse': True},
                  'combined': {**original, 'exploration': 2.0, 'tree_reuse': True}}
        phase = {'status': 'complete', 'agents': agents, 'contrasts': []}
        for candidate, score in (('explore', .6), ('reuse', .7), ('combined', .9)):
            phase['contrasts'].append({'a': 'parent', 'b': candidate, 'purpose': 'attribution',
                                      'factor': 'example', 'result': {'score_b': score, 'seed_pairs': 10,
                                      'ci95_b': [.4, .9], 'verdict': 'inconclusive'}})
        state = {'phases': {'parameters': phase}}
        diagnostics = study_diagnostics(state)
        effects = diagnostics['improvement_comparisons']
        self.assertAlmostEqual(effects[0]['score_percent'], 60)
        self.assertAlmostEqual(effects[0]['advantage_pp'], 10)
        self.assertAlmostEqual(effects[0]['advantage_ci95_pp'][0], -10)
        self.assertEqual(effects[2]['factor'], 'combined_changes')
        ranking = diagnostics['improvement_rankings'][0]
        self.assertEqual(ranking['comparison'], 'equal_time')
        self.assertEqual([e['after'] for e in ranking['effects']], ['reuse', 'explore'])
        self.assertTrue(all(e['verdict'] == 'inconclusive' for e in ranking['effects']))
        phase['status'] = 'pending'
        self.assertEqual(study_diagnostics(state)['improvement_rankings'], [])

    def test_attribution_controls_do_not_change_selection_rankings(self):
        from meeple_bots.studies import _rank
        for selector in ('uct', 'ucb1_tuned'):
            base = MctsAgent(selection_policy=selector)
            state = {'calibration': {'decision_seconds': .01, 'center_iterations': 4}, 'phases': {}}
            horizon = _build_phase('horizons', state, base, None)
            for c in horizon['contrasts']:
                c['result'] = {'score_b': .5}
            horizon['status'] = 'complete'
            state['phases']['horizons'] = horizon
            phase = _build_phase('parameters', state, base, None)
            for c in phase['contrasts']:
                c['result'] = {'score_b': 0 if c.get('purpose') == 'attribution' else .6}
            ranking = _rank(phase)
            without = {**phase, 'contrasts': [c for c in phase['contrasts'] if c.get('purpose') != 'attribution']}
            self.assertEqual(ranking, _rank(without))
            self.assertFalse(any(n.startswith(('parent-', 'uct-control-')) for n in ranking))
            exploration = [c for c in phase['contrasts'] if c['factor'] == 'exploration']
            self.assertTrue(exploration)
            for c in exploration:
                a, b = (phase['agents'][c[role]] for role in ('a', 'b'))
                self.assertEqual([key for key in a if a[key] != b[key]], ['exploration'])
                self.assertEqual(a['selection_policy'], 'uct')

    def test_completion_statistics_keep_legacy_measurements_unknown(self):
        rows = [self.row(7, seat, 'agent_b') for seat in (0, 1)]
        self.assertIsNone(summarize_contrast(rows)['timing_a']['terminal_fraction'])
        for row in rows:
            row['result']['moves'][0].update(terminal_simulations=7, cutoff_simulations=3)
        timing = summarize_contrast(rows)['timing_a']
        self.assertEqual(timing['terminal_simulations'], 14)
        self.assertEqual(timing['cutoff_simulations'], 6)
        self.assertEqual(timing['terminal_fraction'], .7)

    def test_cutoff_recommendation_requires_complete_untruncated_confirmation(self):
        timing = {'decisions': 2, 'terminal_simulations': 20, 'cutoff_simulations': 0,
                  'completion_measured_decisions': 2}
        contrast = {'a': 'terminal-finalist0', 'b': 'finalist0', 'factor': 'cutoff_confirmation',
                    'result': {'verdict': 'b_ahead', 'timing_a': timing}}
        state = {'status': 'complete', 'request': {'game': 'splendor'}, 'budget_seconds': 1,
                 'spent_seconds': 1, 'calibration': {}, 'phases': {
                     'confirmation': {'status': 'complete', 'contrasts': [contrast]}}}
        # Report table diagnostics need the ordinary timing keys too.
        timing.update(iterations_per_second=20, mean_nodes=1, maintenance_seconds=0,
                      reuse_hit_rate=None, quarters=[])
        with TemporaryDirectory() as tmp:
            def decision():
                write_study_report(Path(tmp), state)
                return json.loads((Path(tmp) / 'summary.json').read_text())['cutoff_decisions'][0]['decision']
            self.assertEqual(decision(), 'cutoff_supported')
            contrast['result']['verdict'] = 'inconclusive'
            self.assertEqual(decision(), 'retain_terminal')
            timing['cutoff_simulations'] = 1
            self.assertEqual(decision(), 'reference_truncated_or_unverified')
            state['phases']['confirmation']['status'] = 'partial'
            self.assertEqual(decision(), 'pending_confirmation')

    def test_cube_has_twelve_isolated_contrasts(self):
        base = MctsAgent(iterations=20, rollout_depth=16, heuristic=0)
        agents, contrasts = mechanism_plan(base, .01)
        self.assertEqual(len(agents), 8)
        self.assertEqual(len(contrasts), 12)
        for contrast in contrasts:
            a, b = agents[contrast['a']], agents[contrast['b']]
            self.assertEqual(sum(a[k] != b[k] for k in a), 1)
            self.assertEqual(a['time_budget'], .01)
            self.assertEqual(a['cutoff_evaluator'], {'kind': 'game_heuristic', 'index': 0})
            self.assertNotIn('iterations', a)
        self.assertEqual(base.iterations, 20)

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

    def test_generic_baseline_does_not_read_calibrated_boop_parameters(self):
        base = generic_baseline('boop')
        self.assertEqual((base.iterations, base.rollout_depth, base.exploration), (1000, 32, 1.0))
        self.assertFalse(base.tree_reuse)
        self.assertFalse(base.transpositions)
        self.assertEqual(base.heuristic, 0)
        self.assertIsNone(generic_baseline('tic-tac-toe').heuristic)

    def test_durations_and_cli(self):
        self.assertEqual(duration_seconds('2h'), 7200)
        self.assertEqual(duration_seconds('1.5m'), 90)
        for value in ('0', '-1s', 'nan', 'infinity', '2d'):
            with self.assertRaises(ValueError):
                duration_seconds(value)
        args = build_parser().parse_args(['study', '--game', 'boop', '--budget', '2h'])
        self.assertEqual(args.max_pairs, 16)
        self.assertEqual(args.confirmation_pairs, 64)
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

    def test_mechanism_uncertainty_clusters_backgrounds_by_seed(self):
        agents, contrasts = mechanism_plan(MctsAgent(iterations=10), .01)
        for c in contrasts:
            c['result'] = summarize_contrast([self.row(7, 0, 'agent_b'), self.row(7, 1, 'agent_b')])
        effects = mechanism_effects({'agents': agents, 'contrasts': contrasts})
        self.assertTrue(all(effect['seed_blocks'] == 1 for effect in effects))
        self.assertTrue(all(effect['backgrounds'] == 4 for effect in effects))

    def test_end_to_end_resumes_without_replaying_and_traces_are_extractable(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp)/'study'
            base = MctsAgent(iterations=4, rollout_depth=9)
            reference = MctsAgent(iterations=8, rollout_depth=9)
            runner = StudyRunner('tic-tac-toe', base, reference=reference, output=output, budget=60, workers=3,
                                 decision_seconds=.00001, max_pairs=2, confirmation_pairs=2, max_plies=9, progress=lambda _: None)
            state = runner.run()
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(set(state['phases']), set(PHASES))
            seeds = []
            for name, phase in state['phases'].items():
                self.assertEqual(phase['status'], 'complete')
                phase_seeds = set()
                for c in phase['contrasts']:
                    self.assertEqual(c['result']['seed_pairs'], 2)
                    rows = [json.loads(line) for line in (output/c['trace']).read_text().splitlines()]
                    phase_seeds.update(row['result']['seed'] for row in rows[1:])
                self.assertTrue(all(not phase_seeds & previous for previous in seeds))
                seeds.append(phase_seeds)
            confirmation = state['phases']['confirmation']
            self.assertEqual(confirmation['agents']['reference-original'], profile_values(reference))
            self.assertNotIn('iterations', confirmation['agents']['reference-time'])
            self.assertTrue(any(c['factor'] == 'quality_confirmation' for c in confirmation['contrasts']))
            self.assertTrue((output/'report.html').exists())
            saved = json.loads((output/'study.json').read_text())
            summary = json.loads((output/'summary.json').read_text())
            for key in ('cutoff_selection', 'improvement_comparisons', 'improvement_rankings', 'phase_evidence', 'final_selection'):
                self.assertEqual(saved[key], summary[key])
            self.assertIsInstance(summary['cutoff_selection']['admitted'], bool)
            self.assertTrue(summary['improvement_rankings'])
            for path in state['candidate_profiles'].values():
                _load_mcts_profile(output/path)
            trace_path = output/state['phases']['mechanisms']['contrasts'][0]['trace']
            original = trace_path.read_bytes()
            extract_tournament(trace_path, output/'extracted')
            # A final batch may legitimately finish after the total deadline.
            state['spent_seconds'] = 61
            (output/'study.json').write_text(json.dumps(state))
            with patch('meeple_bots.studies.run_matches', side_effect=AssertionError('resume replayed games')):
                resumed = StudyRunner('tic-tac-toe', base, reference=reference, output=output, budget=60, workers=3,
                                      decision_seconds=.00001, max_pairs=2, confirmation_pairs=2, max_plies=9,
                                      resume=True, progress=lambda _: None).run()
            self.assertEqual(resumed['status'], 'complete')
            self.assertEqual(trace_path.read_bytes(), original)
            with trace_path.open('a') as file:
                file.write('{')
            with self.assertRaisesRegex(ValueError, 'Truncated trace line'):
                StudyRunner('tic-tac-toe', base, reference=reference, output=output, budget=60, workers=3,
                            decision_seconds=.00001, max_pairs=2, confirmation_pairs=2, max_plies=9,
                            resume=True, progress=lambda _: None).run()
            with self.assertRaisesRegex(ValueError, 'configuration or engine changed'):
                StudyRunner('tic-tac-toe', MctsAgent(iterations=5, rollout_depth=9), output=output,
                            budget=60, resume=True, max_pairs=2, confirmation_pairs=2)

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

    def test_worker_policy_and_parallel_interruption_resume(self):
        from meeple_bots.studies import run_matches as real_run
        with TemporaryDirectory() as tmp:
            base = MctsAgent(iterations=4, rollout_depth=9)
            kwargs = dict(output=Path(tmp)/'study', budget=60, workers=3,
                          decision_seconds=.00001, max_pairs=2, confirmation_pairs=2, max_plies=9, progress=lambda _: None)
            runner = StudyRunner('tic-tac-toe', base, **kwargs)
            fixed = profile_values(base)
            timed = {**fixed, 'time_budget': .01}
            timed.pop('iterations')
            phase = {'name': 'confirmation', 'agents': {'a': fixed, 'b': fixed, 'ref': timed},
                     'contrasts': [{'a': 'a', 'b': 'b', 'factor': 'quality_confirmation'},
                                   {'a': 'a', 'b': 'ref', 'factor': 'reference_original_budget'},
                                   {'a': 'a', 'b': 'b', 'factor': 'anchor'}]}
            self.assertEqual([runner._contrast_workers(phase, i) for i in range(3)], [3, 1, 1])
            self.assertEqual(list(runner._batches(phase, 0)), [[1], [2], [0]])
            def interrupt_parallel(game, jobs, **options):
                jobs = list(jobs)
                for job in jobs:
                    if getattr(job.agent_a.agent, 'time_budget', None) is not None or getattr(job.agent_b.agent, 'time_budget', None) is not None:
                        self.assertEqual(options['workers'], 1)
                for item in real_run(game, jobs, **options):
                    yield item
                    if options['workers'] > 1:
                        raise RuntimeError('parallel interruption')
            with patch('meeple_bots.studies.run_matches', side_effect=interrupt_parallel):
                with self.assertRaisesRegex(RuntimeError, 'parallel interruption'):
                    runner.run()
            phase = runner.state['phases']['iterations']
            self.assertTrue(all(c['workers'] == 1 for c in phase['contrasts'] if c['factor'] == 'anchor'))
            paths = list((kwargs['output']/'traces').glob('*.jsonl'))
            prefixes = {p: p.read_bytes() for p in paths}
            state = StudyRunner('tic-tac-toe', base, resume=True, **kwargs).run()
            self.assertEqual(state['status'], 'complete')
            for path, prefix in prefixes.items():
                self.assertTrue(path.read_bytes().startswith(prefix))
            self.assertTrue(any(c['workers'] == 3 for c in state['phases']['confirmation']['contrasts']))
            with self.assertRaisesRegex(ValueError, 'configuration or engine changed'):
                StudyRunner('tic-tac-toe', base, resume=True, **{**kwargs, 'workers': 2})

    def _synthetic_pipeline(self):
        base = MctsAgent(iterations=4, rollout_depth=32, exploration=1)
        state = {'calibration': {'decision_seconds': .01, 'center_iterations': 4}, 'phases': {}}
        for name in PHASES[:5]:
            phase = _build_phase(name, state, base, None)
            for contrast in phase['contrasts']:
                contrast['result'] = {'score_b': .5, 'seed_pairs': 16, 'ci95_b': [.16, .84],
                                      'timing_b': {'mean_seconds': .01}}
            phase['status'] = 'complete'
            state['phases'][name] = phase
        return base, state

    def test_ties_preserve_parent_and_ignore_candidate_names(self):
        from meeple_bots.studies import _leaders, _rank
        base, state = self._synthetic_pipeline()
        parameters = state['phases']['parameters']
        for leader in _leaders(parameters):
            values = parameters['agents'][leader]
            self.assertEqual(values['selection_policy'], base.selection_policy)
            self.assertEqual(values['exploration'], base.exploration)
        refined = state['phases']['refinement']
        for leader in _leaders(refined):
            values = refined['agents'][leader]
            self.assertEqual(values['rollout_depth'], 1024 if leader.startswith('full') else 32)
            self.assertEqual(values['exploration'], base.exploration)
            self.assertFalse(values['tree_reuse'])
            self.assertFalse(values['transpositions'])
        renames = {name: f'renamed-{len(parameters["agents"])-i}' for i, name in enumerate(parameters['agents'])}
        renamed = {**parameters, 'agents': {renames[n]: v for n, v in parameters['agents'].items()},
                   'tie_priority': {renames[n]: v for n, v in parameters['tie_priority'].items()},
                   'contrasts': [{**c, 'a': renames[c['a']], 'b': renames[c['b']]} for c in parameters['contrasts']]}
        self.assertEqual(_rank(renamed), [renames[n] for n in _rank(parameters)])

    def test_mechanisms_are_a_complete_balanced_round_robin(self):
        from itertools import combinations
        _, state = self._synthetic_pipeline()
        phase = state['phases']['mechanisms']
        for family in ('full', 'cutoff'):
            names = [n for n in phase['agents'] if n.startswith(family)]
            expected = {frozenset(pair) for pair in combinations(names, 2)}
            actual = [frozenset((c['a'], c['b'])) for c in phase['contrasts'] if c['a'].startswith(family)]
            self.assertEqual(len(actual), 6)
            self.assertEqual(set(actual), expected)

    def test_family_calibration_is_independent_frozen_and_used_by_iteration_ladder(self):
        from types import SimpleNamespace
        base, state = self._synthetic_pipeline()
        with TemporaryDirectory() as tmp:
            runner = StudyRunner('tic-tac-toe', base, output=Path(tmp), budget=60, progress=lambda _: None)
            runner.state['phases'] = {'parameters': state['phases']['parameters']}
            runner.state['calibration'] = {**state['calibration'], 'mean_plies': 9,
                                           'seconds_per_iteration': .001}
            def benchmark(game, agent, plies, seed):
                self.assertEqual(agent.iterations, 8)
                return SimpleNamespace(milliseconds_per_iteration=2 if agent.rollout_depth == 1024 else .1,
                                       position_timings=[])
            with patch('meeple_bots.studies.benchmark_mcts_agent', side_effect=benchmark) as mocked:
                runner.calibrate_families()
                runner.calibrate_families()
                self.assertEqual(mocked.call_count, 2)
            families = runner.state['calibration']['families']
            self.assertEqual(families['full']['center_iterations'], 5)
            self.assertEqual(families['cutoff']['center_iterations'], 100)
            saved = json.loads((Path(tmp)/'study.json').read_text())
            self.assertEqual(saved['calibration']['families'], families)
            phase = _build_phase('iterations', runner.state, base, None)
            for family, center in (('full', 5), ('cutoff', 100)):
                counts = {v['iterations'] for k,v in phase['agents'].items() if k.startswith(family)}
                self.assertEqual(counts, {max(1, round(center*f)) for f in (.25, .5, 1, 2, 4)})

    def test_low_sample_cutoff_remains_provisional_and_complete_is_not_decisive(self):
        from meeple_bots.study_analysis import paired_interval
        phase = {'status': 'complete', 'contrasts': [
            {'a': 'full', 'b': 'cutoff', 'factor': 'cutoff_equal_time',
             'result': {'score_b': 0, 'seed_pairs': 2, 'ci95_b': paired_interval([0, 0])}}]}
        selection = study_diagnostics({'phases': {'horizons': phase}})['cutoff_selection']
        self.assertEqual(selection['status'], 'provisional')
        self.assertTrue(selection['admitted'])
        phase['contrasts'][0]['result'].update(seed_pairs=16, ci95_b=paired_interval([0]*16))
        self.assertFalse(study_diagnostics({'phases': {'horizons': phase}})['cutoff_selection']['admitted'])
        self.assertLess(paired_interval([.75]*16)[0], .5)
        self.assertGreater(paired_interval([.75]*64)[0], .5)
        final = {'status': 'complete', 'contrasts': [
            {'a': 'finalist-full', 'b': 'finalist-cutoff', 'factor': 'cutoff_confirmation',
             'result': {'score_b': .5, 'verdict': 'inconclusive'}}]}
        summary = study_diagnostics({'phases': {'confirmation': final}})
        self.assertEqual(summary['final_selection']['status'], 'inconclusive')
        self.assertIsNone(summary['final_selection']['candidate'])

    def test_final_ablations_reverse_one_change_at_equal_time_and_keep_both_if_uncertain(self):
        base, state = self._synthetic_pipeline()
        confirmation = _build_phase('confirmation', state, base, None)
        for name in ('finalist-full', 'finalist-cutoff'):
            confirmation['agents'][name].update(exploration=2, tree_reuse=True, transpositions=True)
        match = next(c for c in confirmation['contrasts'] if c['factor'] == 'cutoff_confirmation')
        match['result'] = {'verdict': 'b_ahead'}
        confirmation['status'] = 'complete'
        state['phases']['confirmation'] = confirmation
        phase = _build_phase('ablations', state, base, None)
        self.assertTrue(phase['contrasts'])
        self.assertEqual({c['b'] for c in phase['contrasts']}, {'finalist-cutoff'})
        self.assertTrue({'tree_reuse', 'transpositions', 'exploration'} <= {c['factor'] for c in phase['contrasts']})
        for c in phase['contrasts']:
            a,b = (phase['agents'][c[role]] for role in ('a', 'b'))
            changed = {key for key in a if a[key] != b[key]}
            self.assertEqual(changed, {'tree_reuse', 'transpositions'} if c['factor'] == 'combined_mechanisms' else {c['factor']})
            self.assertEqual(a['time_budget'], b['time_budget'])
            c['result'] = {'score_b': .7, 'verdict': 'b_ahead'}
        phase['status'] = 'complete'
        summaries = study_diagnostics({'phases': {'ablations': phase}})
        self.assertTrue(all(e['evidence_source'] == 'held_out_ablation' for e in summaries['improvement_comparisons']))
        match['result']['verdict'] = 'inconclusive'
        repeated = _build_phase('ablations', state, base, None)
        self.assertEqual({c['b'] for c in repeated['contrasts']}, {'finalist-full', 'finalist-cutoff'})

    def test_heuristic_winner_can_be_ablated_to_neutral_without_alias_conflict(self):
        base, state = self._synthetic_pipeline()
        confirmation = _build_phase('confirmation', state, base, None)
        finalist = confirmation['agents']['finalist-cutoff']
        finalist['cutoff_evaluator'] = {'kind': 'game_heuristic', 'index': 0}
        self.assertEqual(agent_from_values(finalist).heuristic, 0)
        match = next(c for c in confirmation['contrasts'] if c['factor'] == 'cutoff_confirmation')
        match['result'] = {'verdict': 'b_ahead'}
        confirmation['status'] = 'complete'
        state['phases']['confirmation'] = confirmation
        phase = _build_phase('ablations', state, base, None)
        contrast = next(c for c in phase['contrasts'] if c['factor'] == 'cutoff_evaluator')
        neutral = phase['agents'][contrast['a']]
        winner = phase['agents'][contrast['b']]
        self.assertEqual(neutral['cutoff_evaluator'], {'kind': 'neutral'})
        self.assertIsNone(agent_from_values(neutral).heuristic)
        self.assertEqual(winner, finalist)
        self.assertEqual([key for key in winner if winner[key] != neutral[key]], ['cutoff_evaluator'])
        self.assertEqual(confirmation['agents']['finalist-cutoff'], finalist)

    def test_ablation_interrupt_resumes_frozen_pairs_without_recalibrating(self):
        from meeple_bots.studies import run_matches as real_run
        base, state = self._synthetic_pipeline()
        confirmation = _build_phase('confirmation', state, base, None)
        confirmation['agents']['finalist-full'].update(tree_reuse=True, transpositions=True)
        for c in confirmation['contrasts']:
            c['result'] = {'verdict': 'a_ahead'}
        confirmation['status'] = 'complete'
        state['phases']['confirmation'] = confirmation
        state['calibration'].update(mean_plies=9, seconds_per_iteration=.001,
                                    pilot={'name': 'calibration', 'agents': {}, 'contrasts': []})
        for phase in state['phases'].values():
            phase['planned_pairs'] = 2
        with TemporaryDirectory() as tmp:
            options = dict(output=Path(tmp), budget=60, max_pairs=2, confirmation_pairs=2,
                           decision_seconds=.00001, max_plies=9, progress=lambda _: None)
            runner = StudyRunner('tic-tac-toe', base, **options)
            state['calibration']['decision_seconds'] = .00001
            for values in confirmation['agents'].values():
                if 'time_budget' in values:
                    values['time_budget'] = .00001
            runner.state.update(state)
            def interrupt(*args, **kwargs):
                for item in real_run(*args, **kwargs):
                    yield item
                    raise RuntimeError('ablation interruption')
            with patch('meeple_bots.studies.write_study_report'), patch('meeple_bots.studies.run_matches', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'ablation interruption'):
                    runner.run()
            phase = runner.state['phases']['ablations']
            self.assertEqual(phase['planned_pairs'], 2)
            self.assertEqual(phase['status'], 'pending')
            first = next((Path(tmp)/'traces').glob('ablations-*.jsonl'))
            prefix = first.read_bytes()
            with patch('meeple_bots.studies.write_study_report'), patch('meeple_bots.studies.benchmark_mcts_agent', side_effect=AssertionError('recalibrated')):
                resumed = StudyRunner('tic-tac-toe', base, resume=True, **options).run()
            self.assertEqual(resumed['status'], 'complete')
            self.assertTrue(first.read_bytes().startswith(prefix))
            for contrast in resumed['phases']['ablations']['contrasts']:
                self.assertEqual(contrast['result']['seed_pairs'], 2)
                rows = [json.loads(line) for line in (Path(tmp)/contrast['trace']).read_text().splitlines()[1:]]
                self.assertEqual(len(rows), 4)
                self.assertTrue(all(row['result']['seed'] >= 42 + len(PHASES)*100_000 for row in rows))

    def test_empty_ablation_phase_finishes_and_confirmation_budget_is_independent(self):
        base, state = self._synthetic_pipeline()
        confirmation = _build_phase('confirmation', state, base, None)
        for c in confirmation['contrasts']:
            c['result'] = {'verdict': 'a_ahead'}
        confirmation['status'] = 'complete'
        state['phases']['confirmation'] = confirmation
        phase = _build_phase('ablations', state, base, None)
        self.assertEqual(phase['contrasts'], [])
        with TemporaryDirectory() as tmp:
            runner = StudyRunner('tic-tac-toe', base, output=Path(tmp), budget=60, progress=lambda _: None)
            self.assertEqual(runner.state['request']['confirmation_pairs'], 64)
            runner.state.update(state)
            runner.state['calibration'].update(mean_plies=9, seconds_per_iteration=.001)
            with patch('meeple_bots.studies.run_matches', side_effect=AssertionError('empty ablations launched matches')), patch('meeple_bots.studies.write_study_report'):
                result = runner.run()
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['phases']['ablations']['completion_reason'], 'no_changed_parameters')
            self.assertEqual(result['phases']['ablations']['planned_pairs'], 64)

    def test_families_survive_all_phases_and_full_depth_is_never_refined(self):
        for cutoff_score in (.4, .5):
            base = MctsAgent(iterations=4, rollout_depth=16)
            reference = MctsAgent(iterations=100, rollout_depth=64)
            state = {'calibration': {'decision_seconds': .01, 'center_iterations': 4}, 'phases': {}}
            self.assertEqual(PHASES, ('horizons', 'parameters', 'iterations', 'mechanisms', 'refinement', 'confirmation', 'ablations'))
            for name in PHASES[:5]:
                phase = _build_phase(name, state, base, reference)
                self.assertEqual(phase, _build_phase(name, state, base, None))
                for c in phase['contrasts']:
                    values = phase['agents'][c['b']]
                    score = cutoff_score if name == 'horizons' else .8 if values['tree_reuse'] and values['transpositions'] else .5
                    c['result'] = {'score_b': score, 'seed_pairs': 1000, 'ci95_b': [score-.01, score+.01], 'timing_b': {'mean_seconds': .01}}
                phase['status'] = 'complete'
                state['phases'][name] = phase
            families = {'full', 'cutoff'} if cutoff_score >= .45 else {'full'}
            for name in PHASES[1:5]:
                phase = state['phases'][name]
                self.assertEqual({_family(a) for n, a in phase['agents'].items() if n != 'anchor'}, families)
            mechanisms = state['phases']['mechanisms']
            for family in families:
                cells = [a for a in mechanisms['agents'].values() if _family(a) == family]
                self.assertEqual({(a['tree_reuse'], a['transpositions']) for a in cells},
                                 {(False, False), (False, True), (True, False), (True, True)})
                self.assertEqual(len({a['iterations'] for a in cells}), 1)
            refined = state['phases']['refinement']
            for name, values in refined['agents'].items():
                self.assertEqual(values['time_budget'], .01)
                if name.startswith('full-'):
                    self.assertEqual(values['rollout_depth'], 1024)
                    self.assertEqual(values['cutoff_evaluator'], {'kind': 'neutral'})
            confirmation = _build_phase('confirmation', state, base, reference)
            self.assertEqual(confirmation['agents']['reference-original'], profile_values(reference))
            self.assertIn('finalist-full', confirmation['agents'])
            self.assertEqual('finalist-cutoff' in confirmation['agents'], cutoff_score >= .45)
            if cutoff_score >= .45:
                contrast = next(c for c in confirmation['contrasts'] if c['factor'] == 'cutoff_confirmation')
                self.assertEqual(contrast['a'], 'finalist-full')
                self.assertEqual(contrast['b'], 'finalist-cutoff')
                self.assertEqual(confirmation['agents'][contrast['a']]['time_budget'], confirmation['agents'][contrast['b']]['time_budget'])
            for name, values in confirmation['agents'].items():
                if name.startswith('tuned'):
                    self.assertIn(values, refined['fixed_profiles'].values())
                    self.assertTrue(values['tree_reuse'] and values['transpositions'])


if __name__ == '__main__':
    unittest.main()
