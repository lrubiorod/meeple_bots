"""Automatic study plans, paired statistics, persistence and real native integration."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from meeple_bots import MctsAgent, GameHeuristic, ConditionalRollout, TurnPhaseIs, EpsilonGreedy, Mast, ProgressiveBias
from meeple_bots._mcts_profiles import _load_mcts_profile
from meeple_bots.cli import build_parser
from meeple_bots.extraction import extract_tournament
from meeple_bots.studies import (
    StudyRunner, _build_phase, agent_from_values, duration_seconds, export_profile,
    generic_baseline, mechanism_plan, profile_values, PHASES,
)
from meeple_bots.study_analysis import mechanism_effects, summarize_contrast


class StudyTests(unittest.TestCase):
    def test_splendor_calibration_chance_traces_candidates_and_resume(self):
        from meeple_bots import NeutralEvaluator, Splendor, benchmark_mcts_agent
        base = MctsAgent(iterations=2, rollout_depth=4)
        self.assertIsInstance(generic_baseline('splendor').cutoff_evaluator, NeutralEvaluator)
        args = build_parser().parse_args(['study', '--game', 'splendor', '--budget', '2h'])
        self.assertEqual(args.game, 'splendor')
        benchmark = benchmark_mcts_agent(Splendor(), base, 60, 42)
        self.assertGreaterEqual(benchmark.sampled_positions, 2)
        self.assertGreater(benchmark.milliseconds_per_iteration, 0)
        # Exercise every real phase using only two contrasts each, with tiny search budgets.
        def small_plan(*args):
            phase = _build_phase(*args)
            if phase['name'] in ('parameters', 'refinement'):
                phase['contrasts'] = [next(c for c in phase['contrasts'] if phase['agents'][c['b']]['selection_policy'] == selector)
                                      for selector in ('uct', 'ucb1_tuned')]
            elif phase['name'] == 'iterations':
                phase['contrasts'] = [next(c for c in phase['contrasts'] if c['factor'] == 'anchor' and phase['agents'][c['b']]['selection_policy'] == selector) for selector in ('uct', 'ucb1_tuned')]
            else:
                phase['contrasts'] = phase['contrasts'][:2] if phase['name'] != 'mechanisms' else [phase['contrasts'][0], phase['contrasts'][5]]
            return phase
        with TemporaryDirectory() as tmp:
            options = dict(output=Path(tmp), budget=60, workers=2,
                           decision_seconds=.00001, max_pairs=2, max_plies=3000,
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
                                 decision_seconds=.00001, max_pairs=2, max_plies=9, progress=lambda _: None)
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
                                      decision_seconds=.00001, max_pairs=2, max_plies=9,
                                      resume=True, progress=lambda _: None).run()
            self.assertEqual(resumed['status'], 'complete')
            self.assertEqual(trace_path.read_bytes(), original)
            with trace_path.open('a') as file:
                file.write('{')
            with self.assertRaisesRegex(ValueError, 'Truncated trace line'):
                StudyRunner('tic-tac-toe', base, reference=reference, output=output, budget=60, workers=3,
                            decision_seconds=.00001, max_pairs=2, max_plies=9,
                            resume=True, progress=lambda _: None).run()
            with self.assertRaisesRegex(ValueError, 'configuration or engine changed'):
                StudyRunner('tic-tac-toe', MctsAgent(iterations=5, rollout_depth=9), output=output,
                            budget=60, resume=True, max_pairs=2)

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
                          max_pairs=2, max_plies=9, progress=lambda _: None)
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
                                 decision_seconds=.01, max_pairs=2, max_plies=9, progress=lambda _: None)
            state = runner.run()
            self.assertEqual(state['status'], 'budget_exhausted')
            self.assertEqual(state['phases'], {})
            self.assertEqual(state['candidate_profiles'], {})

    def test_worker_policy_and_parallel_interruption_resume(self):
        from meeple_bots.studies import run_matches as real_run
        with TemporaryDirectory() as tmp:
            base = MctsAgent(iterations=4, rollout_depth=9)
            kwargs = dict(output=Path(tmp)/'study', budget=60, workers=3,
                          decision_seconds=.00001, max_pairs=2, max_plies=9, progress=lambda _: None)
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

    def test_late_mechanisms_preserve_tuned_parameters_and_feed_iteration_search(self):
        base = MctsAgent(iterations=4, rollout_depth=16, exploration=1.0,
                         tree_reuse=True, transpositions=True)
        reference = MctsAgent(iterations=15000, rollout_depth=64)
        state = {'calibration': {'decision_seconds': .01, 'center_iterations': 4}, 'phases': {}}
        self.assertEqual(PHASES, ('parameters', 'iterations', 'mechanisms', 'refinement', 'confirmation'))
        for name in PHASES[:-1]:
            phase = _build_phase(name, state, base, reference)
            self.assertEqual(phase, _build_phase(name, state, base, None))
            for c in phase['contrasts']:
                # Synthetic strong evidence for both mechanisms, to verify propagation.
                values = phase['agents'][c['b']]
                winner = 'agent_b' if values['tree_reuse'] and values['transpositions'] else 'agent_a'
                c['result'] = summarize_contrast([self.row(seed, seat, winner)
                                                 for seed in range(2) for seat in (0, 1)])
                c['result']['timing_b']['mean_seconds'] = .01
            state['phases'][name] = phase
        params = state['phases']['parameters']
        self.assertTrue(all(not a['tree_reuse'] and not a['transpositions'] for a in params['agents'].values()))
        mechanisms = state['phases']['mechanisms']
        for selector in ('uct', 'ucb1_tuned'):
            cells = [a for a in mechanisms['agents'].values() if a['selection_policy'] == selector]
            self.assertEqual({(a['tree_reuse'], a['transpositions']) for a in cells},
                             {(False, False), (False, True), (True, False), (True, True)})
            self.assertEqual(len({(a['rollout_depth'], a['exploration']) for a in cells}), 1)
        iterations = state['phases']['iterations']
        self.assertTrue(all(not a['tree_reuse'] and not a['transpositions'] for n,a in iterations['agents'].items() if n != 'anchor'))
        for selector in ('uct', 'ucb1_tuned'):
            counts = {a['iterations'] for a in mechanisms['agents'].values() if a['selection_policy'] == selector}
            self.assertEqual(len(counts), 1)
            self.assertTrue(counts <= {a['iterations'] for n,a in iterations['agents'].items() if n != 'anchor' and a['selection_policy'] == selector})
        refinement = state['phases']['refinement']
        self.assertTrue(any(a['tree_reuse'] and a['transpositions'] for a in refinement['agents'].values()))
        self.assertGreater(len({a['iterations'] for n,a in refinement['agents'].items() if n != 'anchor'}), 1)
        confirmation = _build_phase('confirmation', state, base, reference)
        self.assertEqual(confirmation['agents']['reference-original'], profile_values(reference))
        for name, values in confirmation['agents'].items():
            if name.startswith('tuned'):
                self.assertIn(values, refinement['agents'].values())
                self.assertTrue(values['tree_reuse'] and values['transpositions'])


if __name__ == '__main__':
    unittest.main()
