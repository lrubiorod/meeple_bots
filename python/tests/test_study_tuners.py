"""Coordinate tuning freezes configuration and reuses standard study races."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

from meeple_bots import MctsAgent, GameHeuristic
from meeple_bots.cli import build_parser
from meeple_bots._mcts_profiles import _load_mcts_profile
from meeple_bots._study_tuners import TUNING_FIELDS, proposals, changes, assert_frozen, config_fields
from meeple_bots.studies import StudyRunner, _build_phase, _group_leaders, profile_values, agent_from_values, PHASES, tuning_specs


class StudyTunerTests(unittest.TestCase):
    def champion(self, **kw):
        return replace(MctsAgent(iterations=4, rollout_depth=9, selection_policy='uct_rave',
            exploration=.75, rave_equivalence=1234, progressive_widening=True,
            progressive_widening_k=1.7, progressive_widening_alpha=.6,
            progressive_widening_expansion='rave', tree_reuse=True, transpositions=True), **kw)

    def test_all_dimensions_freeze_unrelated_fields(self):
        base = self.champion()
        for dimension in TUNING_FIELDS:
            with self.subTest(dimension=dimension):
                candidates = proposals(dimension, base, selectors=('uct', 'ucb1_tuned', 'uct_rave'), horizon=169)
                self.assertTrue(candidates)
                for candidate in candidates:
                    self.assertLessEqual(set(changes(base, candidate)), set(TUNING_FIELDS[dimension]))
                    assert_frozen(base, candidate, dimension)
                    self.assertEqual(agent_from_values(profile_values(candidate)), candidate)
        with self.assertRaisesRegex(ValueError, 'frozen fields'):
            assert_frozen(base, replace(base, exploration=1., tree_reuse=False), 'exploration')

    def test_dormant_fields_and_heuristic_parameters_survive_roundtrip(self):
        base = self.champion(selection_policy='uct', progressive_widening=False, progressive_widening_expansion='random')
        self.assertEqual(agent_from_values(profile_values(base)), base)
        heuristic = self.champion(cutoff_evaluator=GameHeuristic(0))
        self.assertEqual(changes(heuristic, replace(heuristic, exploration=1.)), {'exploration': {'before': .75, 'after': 1.}})
        self.assertEqual(config_fields(heuristic)['cutoff_evaluator']['index'], 0)

    def test_cli_local_requires_baseline_and_rejects_hidden_overrides(self):
        args = build_parser().parse_args(['study', '--game', 'connect6', '--agent-config', 'champion.toml', '--tune', 'exploration'])
        self.assertEqual(args.baseline, Path('champion.toml'))
        self.assertEqual(args.tune, 'exploration')
        self.assertIsNone(build_parser().parse_args(['study', '--game', 'connect6']).tune)
        with TemporaryDirectory() as tmp:
            for opts in ({'tune': 'exploration'}, {'baseline': self.champion(), 'tune': 'exploration', 'decision_seconds': .1},
                         {'baseline': self.champion(), 'tune': 'rave', 'all_search': True}):
                with self.assertRaises(ValueError):
                    StudyRunner('connect6', output=Path(tmp)/'bad', **opts)
            for dimension, base in [('rave', self.champion(selection_policy='uct')),
                                    ('progressive-widening-k', MctsAgent()),
                                    ('exploration', self.champion(selection_policy='ucb1_tuned'))]:
                with self.assertRaises(ValueError):
                    StudyRunner('connect6', base, output=Path(tmp)/'bad', tune=dimension)

    def setup_race(self, tmp, **kwargs):
        runner = StudyRunner('connect6', self.champion(), output=Path(tmp), tune='exploration',
            game_params={'board_size': 13}, games_per_comparison=8, workers=1, progress=lambda _: None, **kwargs)
        runner.state['calibration'] = {'horizon': {'depth': 169}, 'fixed_iterations': 4,
            'mean_plies': 50, 'seconds_per_iteration': .001, 'decision_seconds': .01}
        phase = _build_phase(runner.phase_names[0], runner.state, runner.base, None)
        return runner, phase

    def test_budget_reduces_candidates_not_evidence(self):
        with TemporaryDirectory() as tmp:
            runner, phase = self.setup_race(tmp, budget=30)
            with patch.object(runner, '_cost_pair', return_value=5):
                runner._plan_phase(phase, 0)
            self.assertEqual(len(phase['contrasts']), 1)
            self.assertEqual(len(phase['discarded_comparisons']), 1)
            self.assertEqual(phase['contrasts'][0]['target_pairs'], 4)
            self.assertEqual(phase['planned_games'], 8)
            self.assertIn('incumbent', phase['agents'])

    def test_inconclusive_and_incomplete_keep_incumbent(self):
        with TemporaryDirectory() as tmp:
            runner, phase = self.setup_race(tmp)
            runner._plan_phase(phase, 0)
            for score, pairs in [(0.51, 4), (.52, 4), (.49, 4), (.9, 3)]:
                for c in phase['contrasts']:
                    c['result'] = {'score_b': score, 'seed_pairs': pairs, 'seed_scores_b': {str(i): score for i in range(pairs)}}
                self.assertEqual(_group_leaders(phase)['main'], 'incumbent')
            c = phase['contrasts'][1]
            c['result'] = {'score_b': .75, 'seed_pairs': 4, 'seed_scores_b': dict(zip('abcd', [1, .5, 1, .5]))}
            self.assertEqual(_group_leaders(phase)['main'], c['b'])

    def test_boundary_winner_extends_interval_and_keeps_frozen_config(self):
        with TemporaryDirectory() as tmp:
            runner, phase = self.setup_race(tmp)
            runner._plan_phase(phase, 0)
            for c in phase['contrasts']:
                c['result'] = {'score_b': .75 if phase['agents'][c['b']]['exploration'] > .75 else .25,
                    'seed_pairs': 4, 'seed_scores_b': {str(i): .75 if phase['agents'][c['b']]['exploration'] > .75 else .25 for i in range(4)}}
            phase['status'] = 'complete'
            runner.state['phases'][phase['name']] = phase
            runner.save()
            next_phase = _build_phase(runner.phase_names[1], runner.state, runner.base, None)
            self.assertGreater(max(a['exploration'] for a in next_phase['agents'].values()), 1.)
            for a in next_phase['agents'].values():
                assert_frozen(runner.base, agent_from_values(a), 'exploration')

    def test_full_and_local_call_same_generator_and_second_pass_uses_champion(self):
        with TemporaryDirectory() as tmp:
            runner = StudyRunner('connect6', self.champion(), output=Path(tmp), all_search=True, second_pass=True, progress=lambda _: None)
            runner.state['calibration'] = {'horizon': {'depth': 169}, 'fixed_iterations': 4, 'cutoff_depths': []}
            first = _build_phase('depth_screen', runner.state, runner.base, None)
            first['status'] = 'complete'
            runner.state['phases']['depth_screen'] = first
            with patch('meeple_bots.studies.proposals', wraps=proposals) as generator:
                _build_phase('exploration', runner.state, runner.base, None)
                self.assertEqual(generator.call_args.args[0], 'exploration')
            updated = replace(runner.base, exploration=1.5, rave_equivalence=9000)
            runner.state['selected_candidate'] = {'profile': profile_values(updated)}
            name = next(n for n in runner.phase_names if n.startswith('second-0-exploration'))
            with patch('meeple_bots.studies.proposals', wraps=proposals) as generator:
                local = _build_phase(name, runner.state, runner.base, None)
                self.assertEqual(generator.call_args.args[:2], ('exploration', updated))
            self.assertEqual(local['agents']['incumbent'], profile_values(updated))
            self.assertLess(runner.phase_names.index('pw_compare'), runner.phase_names.index(name))

    def test_local_native_export_resume_preserves_board_and_budget(self):
        with TemporaryDirectory() as tmp:
            base = self.champion(iterations=None, time_budget=.00001)
            opts = dict(output=Path(tmp), tune='exploration', game_params={'board_size': 6},
                games_per_comparison=8, workers=2, max_plies=36, progress=lambda _: None)
            state = StudyRunner('connect6', base, **opts).run()
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(state['request']['execution_mode'], 'local_retune')
            self.assertEqual(state['request']['game_params'], {'board_size': 6})
            self.assertIn('local_retune', state)
            for phase in state['phases'].values():
                for config in phase['agents'].values():
                    candidate = agent_from_values(config)
                    assert_frozen(base, candidate, 'exploration')
                    self.assertEqual(candidate.time_budget, base.time_budget)
            result = _load_mcts_profile(Path(tmp)/'candidates/best_agent.toml').agent
            assert_frozen(base, result, 'exploration')
            self.assertEqual(state['local_retune']['changed_fields'], changes(base, result))
            report = (Path(tmp)/'report.html').read_text()
            self.assertIn('LOCAL RETUNE', report)
            seeds = [c['trace'] for p in state['phases'].values() for c in p['contrasts']]
            self.assertEqual(len(seeds), len(set(seeds)))
            with patch('meeple_bots.studies.run_matches', side_effect=AssertionError('replayed')):
                self.assertEqual(StudyRunner('connect6', base, resume=True, **opts).run()['status'], 'complete')

    def test_coupled_pw_plan_keeps_enable_and_expansion_frozen(self):
        specs = tuning_specs('progressive-widening')
        dimensions = [s['dimension'] for s in specs.values()]
        self.assertEqual(dimensions[:6], ['progressive-widening-k']*6)
        self.assertEqual(dimensions[6:12], ['progressive-widening-alpha']*6)
        self.assertEqual(dimensions[-1], 'progressive-widening-k')

    def test_native_second_pass_and_no_flags_still_use_full_mode(self):
        with TemporaryDirectory() as tmp:
            base = self.champion()
            state = StudyRunner('tic-tac-toe', base, output=Path(tmp), second_pass=True,
                games_per_comparison=8, max_plies=9, workers=2, progress=lambda _: None).run()
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(state['request']['execution_mode'], 'full_study')
            self.assertTrue(any(name.startswith('second-') for name in state['phases']))
            winner = _load_mcts_profile(Path(tmp)/'candidates/best_agent.toml').agent
            self.assertLessEqual(set(changes(base, winner)), {'exploration', 'rave_equivalence', 'progressive_widening_k', 'progressive_widening_alpha'})

    def test_budget_omission_is_explicit_in_local_output(self):
        with TemporaryDirectory() as tmp:
            runner, _ = self.setup_race(tmp, budget=30)
            with patch.object(runner, '_cost_pair', return_value=1e9):
                state = runner.run()
            self.assertEqual(state['local_retune']['result'], 'INCONCLUSIVE')
            self.assertTrue(state['budget_limited'])
            self.assertEqual(state['local_retune']['changed_fields'], {})
            self.assertEqual(_load_mcts_profile(Path(tmp)/'candidates/best_agent.toml').agent, runner.base)
            self.assertIn('No sufficiently supported improvement found.', (Path(tmp)/'report.html').read_text())
