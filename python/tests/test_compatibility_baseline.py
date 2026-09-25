"""Small persistence and source-identity contracts for the Python refactor."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from meeple_bots import MctsAgent, TicTacToe, _native
from meeple_bots import studies
from meeple_bots._study_profiles import load_search_profile
from meeple_bots.analysis import analyze_game, report_dict
from meeple_bots.api import evaluate_game
from meeple_bots.cli import _evaluation_dict, main
from meeple_bots.probes.compare import compare_captures
from meeple_bots.tournaments import (TournamentAgent, TournamentConfig, TournamentTrace,
                                     match_jobs, tournament_header, tournament_pairings)
from meeple_bots import RandomAgent


FIXTURES = Path(__file__).parent / 'fixtures' / 'compat'


class SourceFingerprintTests(unittest.TestCase):
    @staticmethod
    def tree(root):
        root.mkdir(exist_ok=True)
        (root / 'studies').mkdir()
        (root / 'root.py').write_text('root = 1\n')
        (root / 'studies' / 'race.py').write_text('race = 1\n')

    def test_tree_digest_is_recursive_path_sensitive_and_checkout_independent(self):
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            a, b = Path(first), Path(second)
            self.tree(a)
            # Create in the opposite order; filesystem enumeration must not matter.
            (b / 'studies').mkdir()
            (b / 'studies' / 'race.py').write_text('race = 1\n')
            (b / 'root.py').write_text('root = 1\n')
            original = studies._python_tree_digest(a)
            self.assertEqual(original, studies._python_tree_digest(b))
            self.assertEqual([n for n, _ in studies._python_source_entries(a)], ['root.py', 'studies/race.py'])
            for path in ('docs/note.py', 'tests/test_fake.py', 'results/run.py', 'local/setup.py',
                         '__pycache__/cached.py', 'build/generated.py', 'dist/wheel.py'):
                target = a / path
                target.parent.mkdir(exist_ok=True)
                target.write_text('irrelevant = True\n')
            self.assertEqual(original, studies._python_tree_digest(a))
            (a / 'root.py').write_text('root = 2\n')
            self.assertNotEqual(original, studies._python_tree_digest(a))
            (a / 'root.py').write_text('root = 1\n')
            nested = a / 'studies' / 'race.py'
            nested.write_text('race = 2\n')
            self.assertNotEqual(original, studies._python_tree_digest(a))
            nested.write_text('race = 1\n')
            nested.rename(a / 'studies' / 'coordinator.py')
            self.assertNotEqual(original, studies._python_tree_digest(a))
            (a / 'studies' / 'coordinator.py').rename(nested)
            (a / 'studies' / 'new.py').write_text('new = 1\n')
            self.assertNotEqual(original, studies._python_tree_digest(a))
            (a / 'studies' / 'new.py').unlink()
            nested.unlink()
            self.assertNotEqual(original, studies._python_tree_digest(a))

    def test_legacy_hash_ignores_nested_files_exactly_as_before(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.tree(root)
            old = studies._legacy_python_digest(root)
            (root / 'studies' / 'race.py').write_text('changed\n')
            self.assertEqual(old, studies._legacy_python_digest(root))

    def test_runtime_fingerprint_covers_package_and_native_engine(self):
        actual = studies._fingerprint()
        self.assertEqual(actual['fingerprint_algorithm'], studies.FINGERPRINT_ALGORITHM)
        self.assertEqual(actual['python_sha256'], studies._python_tree_digest(studies._package_source_root()))
        self.assertEqual(len(actual['native_sha256']), 64)


class StudyCompatibilityTests(unittest.TestCase):
    def test_version_23_request_plan_and_seed_namespaces(self):
        expected = json.loads((FIXTURES / 'study-plan-v23.json').read_text())
        with TemporaryDirectory() as tmp:
            runner = studies.StudyRunner('boop', output=Path(tmp), budget=60, seed=71,
                                         all_search=True, second_pass=True,
                                         stage_games={'pw': 10}, progress=lambda _: None)
            request = runner.state['request']
            self.assertIn('reference', request)
            self.assertIsNone(request['reference'])
            actual = {'study_protocol': request['version'],
                      'max_extension_rounds': request['max_extension_rounds'],
                      'phase_names': request['phase_names'], 'seed': request['seed'],
                      'seed_stride': studies.SEED_STRIDE,
                      'games_per_comparison': request['games_per_comparison'],
                      'stage_games': request['stage_games'], 'tuner_specs': request['tuner_specs'],
                      'engine_algorithm': request['engine']['fingerprint_algorithm']}
            self.assertEqual(actual, expected)
            self.assertIn('rave_extend_3', request['phase_names'])
            self.assertIn('pw_k_extend_3', request['phase_names'])
            self.assertIn('pw_alpha_extend_3', request['phase_names'])
            self.assertIn('random_baseline', studies.StudyRunner(
                'boop', output=Path(tmp) / 'random', budget=60, seed=71,
                vs_random=True, progress=lambda _: None).phase_names)

    def test_interrupted_study_matches_uninterrupted_semantic_results(self):
        real_run = studies.run_matches
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = MctsAgent(iterations=4, rollout_depth=9)
            options = dict(budget=60, max_pairs=2, max_plies=9, workers=1,
                           tune='exploration', seed=71, progress=lambda _: None)
            control = studies.StudyRunner('tic-tac-toe', base,
                                           output=root / 'control', **options).run()
            checkpoint = json.loads((root / 'control' / 'study.json').read_text())
            phase = next(p for p in checkpoint['phases'].values() if p['contrasts'])
            projection = {'protocol': checkpoint['request']['version'],
                          'fingerprint_algorithm': checkpoint['request']['engine']['fingerprint_algorithm'],
                          'seed': checkpoint['request']['seed'],
                          'phase_names': checkpoint['request']['phase_names'],
                          'first_phase': phase['name'],
                          'candidate_exploration': {name: agent['exploration']
                                                    for name, agent in phase['agents'].items()},
                          'completed_comparisons': [[c['a'], c['b'], c['target_pairs'],
                                                     c['result']['seed_pairs'], c['result']['score_b'],
                                                     c['result']['verdict']] for c in phase['contrasts']]}
            self.assertEqual(projection, json.loads((FIXTURES / 'study-checkpoint-v23.json').read_text()))

            interrupted = False
            def stop_once(*args, **kwargs):
                nonlocal interrupted
                for item in real_run(*args, **kwargs):
                    yield item
                    if not interrupted:
                        interrupted = True
                        raise RuntimeError('synthetic interruption')

            output = root / 'resume'
            with patch('meeple_bots.studies.coordinator.run_matches', side_effect=stop_once):
                with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                    studies.StudyRunner('tic-tac-toe', base, output=output, **options).run()
            resumed = studies.StudyRunner('tic-tac-toe', base, output=output,
                                          resume=True, **options).run()
            self.assertEqual(control['status'], 'complete')
            self.assertEqual(resumed['status'], 'complete')
            self.assertEqual(control['selected_candidate'], resumed['selected_candidate'])
            self.assertEqual(control['request']['phase_names'], resumed['request']['phase_names'])
            def competitive(contrast):
                result = contrast.get('result', {})
                return (contrast['a'], contrast['b'], contrast.get('target_pairs'),
                        {k: v for k, v in result.items() if k not in {'timing_a', 'timing_b'}})
            for phase_name in control['phases']:
                a, b = control['phases'][phase_name], resumed['phases'][phase_name]
                self.assertEqual(a['status'], b['status'])
                self.assertEqual(a['agents'], b['agents'])
                self.assertEqual([competitive(c) for c in a['contrasts']],
                                 [competitive(c) for c in b['contrasts']])
            def identities(directory):
                return {path.name: [(row['match_number'], row['result']['seed'],
                                     row['agent_a_player'], row['winner'])
                                    for row in [json.loads(line) for line in path.read_text().splitlines()[1:]]]
                        for path in (directory / 'traces').glob('*.jsonl')}
            self.assertEqual(identities(root / 'control'), identities(output))

    def test_resume_matrix_for_new_and_legacy_fingerprints(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            SourceFingerprintTests.tree(source)
            native = root / 'engine.so'
            native.write_bytes(b'native version one')
            opts = dict(output=root / 'study', budget=60, max_pairs=2,
                        max_plies=9, progress=lambda _: None)
            base = MctsAgent(iterations=4, rollout_depth=9)
            with patch('meeple_bots.studies.persistence._package_source_root', return_value=source), \
                    patch.object(_native, '__file__', str(native)):
                runner = studies.StudyRunner('tic-tac-toe', base, **opts)
                stored = runner.state['request']['engine']
                self.assertEqual(stored['fingerprint_algorithm'], 'python-tree-v2')
                studies.StudyRunner('tic-tac-toe', base, resume=True, **opts)
                before = (opts['output'] / 'study.json').read_bytes()
                for changed in (source / 'root.py', source / 'studies' / 'race.py', native):
                    original = changed.read_bytes()
                    changed.write_bytes(original + b'change')
                    with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
                        studies.StudyRunner('tic-tac-toe', base, resume=True, **opts)
                    self.assertEqual((opts['output'] / 'study.json').read_bytes(), before)
                    changed.write_bytes(original)
                with self.assertRaisesRegex(ValueError, 'configuration differs'):
                    studies.StudyRunner('tic-tac-toe', base, resume=True,
                                        allow_engine_change=True, seed=99, **opts)
                state = json.loads((opts['output'] / 'study.json').read_text())
                state['request']['phase_names'].append('invented_phase')
                (opts['output'] / 'study.json').write_text(json.dumps(state))
                with self.assertRaisesRegex(ValueError, 'configuration differs'):
                    studies.StudyRunner('tic-tac-toe', base, resume=True, **opts)
                state['request']['phase_names'].pop()
                legacy = {k: v for k, v in stored.items() if k != 'fingerprint_algorithm'}
                legacy['python_sha256'] = studies._legacy_python_digest(source)
                state['request']['engine'] = legacy
                (opts['output'] / 'study.json').write_text(json.dumps(state))
                resumed = studies.StudyRunner('tic-tac-toe', base, resume=True, **opts)
                self.assertEqual(resumed.state['request']['engine'], legacy)
                self.assertNotIn('active_engine', resumed.state)
                source.joinpath('root.py').write_text('changed\n')
                with self.assertRaisesRegex(ValueError, 'Python source fingerprint differs'):
                    studies.StudyRunner('tic-tac-toe', base, resume=True, **opts)
                mixed = studies.StudyRunner('tic-tac-toe', base, resume=True,
                                            allow_engine_change=True, **opts)
                self.assertEqual(mixed.state['request']['engine'], legacy)
                self.assertEqual(mixed.state['active_engine']['fingerprint_algorithm'], 'python-tree-v2')
                self.assertEqual(len(mixed.state['engine_changes']), 1)

    def test_unknown_algorithm_rejected_without_reinterpretation(self):
        with TemporaryDirectory() as tmp:
            opts = dict(output=Path(tmp), budget=60, progress=lambda _: None)
            base = MctsAgent(iterations=4, rollout_depth=9)
            studies.StudyRunner('tic-tac-toe', base, **opts)
            path = Path(tmp) / 'study.json'
            state = json.loads(path.read_text())
            state['request']['engine']['fingerprint_algorithm'] = 'future-v3'
            path.write_text(json.dumps(state))
            for override in (False, True):
                with self.assertRaisesRegex(ValueError, 'unsupported study engine fingerprint algorithm'):
                    studies.StudyRunner('tic-tac-toe', base, resume=True,
                                        allow_engine_change=override, **opts)


class FormatCompatibilityTests(unittest.TestCase):
    def test_documented_import_paths(self):
        for name in ('meeple_bots', 'meeple_bots.api', 'meeple_bots.cli',
                     'meeple_bots._mcts_profiles', 'meeple_bots.gui.controller',
                     'meeple_bots.gui.application', 'meeple_bots.reporting.common'):
            self.assertIsNotNone(importlib.import_module(name))
        self.assertTrue(callable(main))
        help_result = subprocess.run([sys.executable, '-m', 'meeple_bots', '--help'],
                                     capture_output=True, text=True, check=True)
        self.assertIn('study', help_result.stdout)

    def test_profile_fixtures_decode_and_round_trip(self):
        for name, family in (('mcts-modern.toml', 'MctsAgent'),
                             ('mcts-legacy.toml', 'MctsAgent'),
                             ('so-ismcts.toml', 'SoIsmctsAgent')):
            agent = load_search_profile(FIXTURES / name)
            self.assertEqual(type(agent).__name__, family)
            self.assertEqual(studies.agent_from_values(studies.profile_values(agent)), agent)
        self.assertEqual(load_search_profile(FIXTURES / 'mcts-legacy.toml').cutoff_evaluator.index, 0)

    def test_analyze_current_and_legacy_json_projections(self):
        expected = json.loads((FIXTURES / 'analyze-keys-v1.json').read_text())
        report = analyze_game(TicTacToe(), samples=2, max_depth=9, target_time=.001)
        self.assertEqual(sorted(report_dict(report)), expected['generic'])
        legacy = evaluate_game(TicTacToe(), samples=2, max_depth=9, target_time=.001)
        self.assertEqual(sorted(_evaluation_dict(legacy)), expected['legacy'])

    def test_version_one_probe_capture_retains_all_edges_and_nullability(self):
        with TemporaryDirectory() as tmp:
            from shutil import copytree
            a, b = Path(tmp) / 'A', Path(tmp) / 'B'
            copytree(FIXTURES / 'probe-v1', a)
            copytree(FIXTURES / 'probe-v1', b)
            compared = compare_captures({'A': a, 'B': b}, baseline='A')
            actions = compared['groups'][0]['variants'][0]['actions']
            self.assertEqual(len(actions), 2)
            self.assertEqual([x['selected_count'] for x in actions], [1, 0])
            self.assertEqual([x['median_availability'] for x in actions], [4, None])
            self.assertEqual([x['median_q'] for x in actions], [.25, None])
            self.assertEqual(compared['seed_selections'][0]['search_seed'], 7)

    def test_paired_trace_fixture_resumes_with_exact_header_and_job_identity(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trace.jsonl'
            config = TournamentConfig(game=TicTacToe(), output=path, pairing_mode='round_robin',
                                      seat_mode='paired', matches_per_pair=2, seed=17,
                                      max_plies=9, workers=1,
                                      agents=(TournamentAgent('a', RandomAgent()),
                                              TournamentAgent('b', RandomAgent())))
            header = tournament_header(config, path, 1)
            jobs = list(match_jobs(tournament_pairings(config.agents), config))
            self.assertEqual([(j.match_number, j.seed, j.agent_a_player) for j in jobs],
                             [(1, 17, 0), (2, 17, 1)])
            fixture = (FIXTURES / 'tournament-trace-v1.jsonl').read_text()
            path.write_text(fixture.replace('__TRACE_PATH__', str(path)))
            self.assertEqual(json.loads(path.read_text().splitlines()[0]), header)
            with TournamentTrace(path, header, resume=True) as trace:
                self.assertEqual(trace.completed_match_numbers, {1})


if __name__ == '__main__':
    unittest.main()
