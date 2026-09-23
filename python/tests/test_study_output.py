"""Readable plans preserve frozen budgets, phases and competitive evidence."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from meeple_bots import MctsAgent, SoIsmctsAgent
from meeple_bots.studies import StudyRunner, PHASES, MAX_EXTENSION_ROUNDS, tuning_specs
from meeple_bots._study_output import announce_extension
import test_studies


class StudyOutputTests(unittest.TestCase):
    def calibrated(self, path, **kwargs):
        return test_studies.StudyTests().calibrated_fake(path, **kwargs)

    def output(self, runner):
        lines = []
        runner.progress = lines.append
        runner._announce_plan()
        return '\n'.join(lines)

    def test_explicit_budget_and_grouped_plan(self):
        with TemporaryDirectory() as tmp:
            runner = self.calibrated(tmp, decision_seconds=.25, pw_search=True, second_pass=True)
            text = self.output(runner)
            for expected in ('250 ms', 'explicit --decision-time', '60s (overridden; not used)',
                             '~20.00s', 'not a hard wall-clock match limit', 'practical full depth',
                             '128 plies', 'Progressive widening:', 'Second pass:', 'max 3 extensions'):
                self.assertIn(expected, text)
            self.assertEqual(text.count('Agent family:'), 1)
            self.assertIn('RAVE selector search:        disabled', text)
            self.assertNotIn('Resolved stages:', text)
            self.assertNotIn('pw_k_extend_1', text)
            self.assertNotIn('second-0-exploration-0', text)
            request = json.loads(runner.path.read_text())['request']
            self.assertIn('pw_k_extend_3', request['phase_names'])
            self.assertEqual(request['max_extension_rounds'], 3)
            b = runner.state['calibration']['compute_budget']
            self.assertEqual(b['decision_seconds'], .25)
            self.assertEqual(b['estimated_search_game_seconds'], 20)
            self.assertFalse(b['target_match_time_active'])

    def test_available_policies_and_conditional_second_pass_preserve_state(self):
        with TemporaryDirectory() as tmp:
            for rave in (False, True):
                runner = self.calibrated(Path(tmp)/str(rave), selection_search=True,
                                         pw_search=True, second_pass=True, rave_search=rave)
                before = json.dumps(runner.state, sort_keys=True)
                checkpoint = runner.path.read_bytes()
                text = self.output(runner)
                self.assertIn('Available selection policies: uct, ucb1_tuned, uct_rave', text)
                self.assertIn('Candidate policies: UCT, UCB1-Tuned\n', text)
                self.assertIn('RAVE selector search:        ' + ('enabled' if rave else 'disabled'), text)
                self.assertIn('PW admission strategies:     random, rave-guided', text)
                second = text.split('Second pass:', 1)[1].split('Comparison policy', 1)[0]
                self.assertEqual('    rave (' in second, rave)
                self.assertEqual('Candidate policies: UCT-RAVE\n' in text, rave)
                self.assertIn('progressive-widening-k (only if retained candidate has PW enabled)', second)
                self.assertIn('progressive-widening-alpha (only if retained candidate has PW enabled)', second)
                self.assertIn('second-0-rave-0', runner.phase_names)
                self.assertEqual(json.dumps(runner.state, sort_keys=True), before)
                self.assertEqual(runner.path.read_bytes(), checkpoint)

    def test_second_pass_respects_existing_baseline_mechanisms(self):
        with TemporaryDirectory() as tmp:
            for enabled in (False, True):
                base = MctsAgent(iterations=20, selection_policy='uct_rave' if enabled else 'uct',
                                 progressive_widening=enabled)
                runner = self.calibrated(Path(tmp)/str(enabled), baseline=base, second_pass=True)
                text = self.output(runner)
                second = text.split('Second pass:', 1)[1].split('Comparison policy', 1)[0]
                self.assertEqual('rave (only if retained selector is UCT-RAVE)' in second, enabled)
                self.assertEqual('progressive-widening-k (' in second, enabled)
                self.assertIn('RAVE selector search:        disabled', text)

    def test_active_target_and_baseline_precedence(self):
        with TemporaryDirectory() as tmp:
            runner = self.calibrated(Path(tmp)/'target')
            text = self.output(runner)
            for expected in ('625 ms', '--target-match-time', '60 / (80 × 1.2)', '1.20', '~50.00s'):
                self.assertIn(expected, text)
            self.assertNotIn('overridden', text)
            self.assertIn('Derived decision time:', text)
            runner = self.calibrated(Path(tmp)/'base', baseline=MctsAgent(time_budget=.2))
            text = self.output(runner)
            self.assertIn('baseline/config', text)
            self.assertIn('200 ms', text)
            runner = self.calibrated(Path(tmp)/'fixed', baseline=MctsAgent(iterations=20))
            text = self.output(runner)
            self.assertIn('20 (fixed)', text)
            self.assertIn('not a time limit', text)
            self.assertIsNone(runner.state['calibration']['compute_budget']['decision_seconds'])
            runner = self.calibrated(Path(tmp)/'override', baseline=MctsAgent(iterations=20),
                                     decision_seconds=.25)
            text = self.output(runner)
            self.assertIn('explicit --decision-time', text)
            self.assertNotIn('Iterations/decision:', text)

    def test_cutoff_and_reference_horizons_are_distinct(self):
        with TemporaryDirectory() as tmp:
            runner = self.calibrated(tmp, heuristic=0)
            text = self.output(runner)
            self.assertIn('heuristic cutoff', text)
            self.assertIn('Reference full horizon:', text)
            self.assertIn('Reference horizon mode:', text)
            self.assertIn('practical full depth', text)
            self.assertIn('>= 99% at every sampled position', text)

    def test_planned_extension_cost_keeps_fixed_race_evidence(self):
        helper = test_studies.StudyTests()
        state = helper.state_for_plan()
        state['request'].update(pw_search=True, pw_supported=True)
        helper.populate(state)
        with TemporaryDirectory() as tmp:
            runner = self.calibrated(tmp, rave_search=True, pw_search=True,
                                     games_per_comparison=8, workers=1)
            runner.budget = None
            with patch.object(runner, '_cost_pair', return_value=1):
                for stem in ('rave', 'pw_k', 'pw_alpha'):
                    phases = [state['phases'][n] for n in runner.phase_names
                              if n.startswith(stem + '_extend_')]
                    self.assertEqual(len(phases), 3)
                    total = 0
                    for phase in phases:
                        candidates = json.dumps(phase['agents'], sort_keys=True)
                        count = len(phase['contrasts'])
                        runner._plan_phase(phase, runner.phase_names.index(phase['name']))
                        self.assertEqual(json.dumps(phase['agents'], sort_keys=True), candidates)
                        self.assertEqual(len(phase['contrasts']), count)
                        self.assertEqual(phase['planned_games'], 8 * count)
                        self.assertTrue(all(c['target_pairs'] == 4 for c in phase['contrasts']))
                        total += phase['estimated_seconds']
                    self.assertAlmostEqual(total, 4.8 * sum(len(p['contrasts']) for p in phases))

    def test_so_family_uses_same_output_and_correct_override_source(self):
        with TemporaryDirectory() as tmp:
            runner = StudyRunner('lost_cities', SoIsmctsAgent(iterations=1), output=Path(tmp),
                                 decision_seconds=.0001, workers=1, progress=lambda _: None)
            runner.calibrate()
            text = self.output(runner)
            self.assertEqual(text.count('Agent family:'), 1)
            self.assertIn('SO-ISMCTS', text)
            self.assertIn('explicit --decision-time', text)
            self.assertIn('Confirmation', text)
            self.assertEqual(runner.state['calibration']['decision_time_source'], 'explicit_override')

    def test_extension_limits_and_local_second_pass_round_semantics(self):
        self.assertEqual(MAX_EXTENSION_ROUNDS, 3)
        for stem in ('rave', 'pw_k', 'pw_alpha'):
            self.assertEqual([p for p in PHASES if p.startswith(stem+'_extend_')],
                             [f'{stem}_extend_{i}' for i in (1, 2, 3)])
        for dimension in ('exploration', 'rave', 'cutoff-depth', 'progressive-widening'):
            specs = tuning_specs(dimension, prefix='second')
            self.assertLessEqual(max(s['round'] for s in specs.values()), 3)
            for chain in {s['chain'] for s in specs.values()}:
                rounds = [s['round'] for s in specs.values() if s['chain'] == chain]
                self.assertIn(rounds, ([0], [0, 1, 2, 3]))
        for dimension in ('selection', 'structure', 'tree-reuse', 'widening-expansion'):
            self.assertEqual(len(tuning_specs(dimension)), 1)

    def test_third_extension_keeps_supported_boundary_winner(self):
        helper = test_studies.StudyTests()
        state = helper.populate(helper.state_for_plan())
        phase = state['phases']['rave_extend_3']
        lines = []
        runner = SimpleNamespace(state=state, progress=lines.append)
        before = json.dumps(phase['agents'], sort_keys=True)
        announce_extension(runner, phase)
        announce_extension(runner, phase, completed=True)
        self.assertTrue(phase['extension_limit_reached'])
        self.assertIn('Boundary candidate won at extension limit (3/3)', '\n'.join(lines))
        self.assertEqual(json.dumps(phase['agents'], sort_keys=True), before)
        self.assertNotIn('rave_extend_4', state['phases'])
        self.assertEqual(len(phase['contrasts']), 2)
        self.assertTrue(all(c['result']['seed_pairs'] >= 4 for c in phase['contrasts']))

    def test_old_extension_plan_rejected_without_modification(self):
        with TemporaryDirectory() as tmp:
            runner = StudyRunner('tic-tac-toe', output=Path(tmp), progress=lambda _: None)
            old = runner.state
            old['request']['version'] = 22
            old['request']['phase_names'].append('rave_extend_5')
            runner.path.write_text(json.dumps(old))
            before = runner.path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'old study protocol / extension plan'):
                StudyRunner('tic-tac-toe', output=Path(tmp), resume=True, progress=lambda _: None)
            self.assertEqual(runner.path.read_bytes(), before)
