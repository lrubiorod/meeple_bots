"""Offline report invariants, without strategic or search assertions."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from meeple_bots.cli import main
from meeple_bots.probes.compare import compare_captures, render_comparison, write_comparison
from meeple_bots.probes.report import aggregate, action_roles, important_actions, render


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def capture(self, name, choices=(8, 8, 9, 10), budgets=(1000,), seeds=None, game='test', missing=()):
        path = self.root / name
        path.mkdir()
        meta = {'version': 1, 'q_orientation': 'root_player', 'agent': {'type': 'mcts', 'selection_policy': 'uct'},
                'probes': [{'id': 'test.position', 'game': game, 'root_player': 0,
                            'observation': {'public': [1, 2]}, 'candidate_actions': [{'move': 0, 'kind': 'take'}]}]}
        (path / 'metadata.json').write_text(json.dumps(meta))
        rows = []
        for budget in budgets:
            for seed, selected in zip(seeds or range(len(choices)), choices):
                rows.append({'probe_id': 'test.position', 'game': game, 'root_player': 0,
                             'iterations': budget, 'search_seed': seed,
                             'selected_action': {'kind': 'take', 'move': selected},
                             'agent': meta['agent'], 'elapsed_seconds': .1,
                             'root_actions': [{'action': {'move': i, 'kind': 'take'}, 'label': f'action-{i}',
                                               'focused': i == 0, 'visits': i + 1, 'q': i / 100,
                                               'availability': budget if name == 'control' else None}
                                              for i in reversed(range(100)) if i not in missing]})
        (path / 'runs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return path

    def test_two_and_three_variants_union_metrics_baseline_and_determinism(self):
        inputs = {'control': self.capture('control'), 'guided': self.capture('guided', (11, 11, 12, 13))}
        two = compare_captures(inputs)
        inputs['bias'] = self.capture('bias', (14, 14, 15, 16))
        data = compare_captures(inputs, baseline='control')
        self.assertEqual(data, compare_captures(inputs, baseline='control'))
        self.assertEqual(len(two['groups'][0]['variants']), 2)
        rows = data['groups'][0]['variants']
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(len(r['actions']) == 100 for r in rows))
        shown = important_actions(rows)
        self.assertEqual({a['action']['move'] for a in shown}, {0, 8, 9, 10, 11, 12, 13, 14, 15, 16})
        text = render_comparison(data)
        self.assertIn('root-player utility', text)
        self.assertIn('+50.0 pp', text)
        self.assertIn('median latency', text)
        self.assertNotIn('action-99', text)
        self.assertIn('action-8 | control | dominant | 2/4 (50.0%) | 9.0 | 1000.0 | 0.080', text)
        self.assertEqual(data['groups'][0]['dominant_agreement'], 'all different')
        self.assertEqual(len(data['seed_selections']), 12)

    def test_ties_focus_dedup_and_single_report(self):
        p = self.capture('control', (0, 8, 9, 10))
        rows = [json.loads(line) for line in (p/'runs.jsonl').read_text().splitlines()]
        summary = aggregate(rows)
        roles = action_roles(summary[0]['actions'])
        self.assertEqual(roles['{"kind":"take","move":0}'], 'focus+dominant')
        self.assertEqual(len(important_actions(summary)), 4)  # all four tied leaders
        text = render(summary)
        self.assertIn('Dominant selection (tie)', text)
        self.assertEqual(text.count('action-8'), 2)  # table with metrics and dominant line
        self.assertNotIn('action-99', text)

    def test_seeds_budgets_action_union_and_missing_values(self):
        a = self.capture('control', budgets=(1000, 5000))
        b = self.capture('guided', seeds=(10, 11, 12, 13), missing=(0,))
        data = compare_captures({'control': a, 'guided': b})
        warnings = '\n'.join(data['warnings'])
        for phrase in ('different seed sets', 'missing budgets', 'root action set mismatch', 'missing=', 'additional='):
            self.assertIn(phrase, warnings)
        self.assertIn('action-0 | guided | missing', render_comparison(data))
        self.assertEqual(len(data['groups']), 2)

    def test_incompatible_game_state_root_and_schema(self):
        a = self.capture('control')
        b = self.capture('guided', game='other')
        with self.assertRaisesRegex(ValueError, 'games'):
            compare_captures({'a': a, 'b': b})
        runs = [json.loads(x) for x in (b/'runs.jsonl').read_text().splitlines()]
        for r in runs:
            r['game'] = 'test'
            r['root_player'] = 1
        (b/'runs.jsonl').write_text('\n'.join(map(json.dumps, runs)))
        with self.assertRaisesRegex(ValueError, 'root players'):
            compare_captures({'a': a, 'b': b})
        for r in runs:
            r['root_player'] = 0
        (b/'runs.jsonl').write_text('\n'.join(map(json.dumps, runs)))
        meta = json.loads((b/'metadata.json').read_text())
        meta['probes'][0]['observation'] = {'public': [3]}
        (b/'metadata.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'observation'):
            compare_captures({'a': a, 'b': b})
        meta['version'] = 99
        (b/'metadata.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'schema'):
            compare_captures({'a': a, 'b': b})

    def test_legacy_metadata_warning(self):
        a, b = self.capture('control'), self.capture('guided')
        meta = json.loads((b/'metadata.json').read_text())
        del meta['probes'][0]['observation']
        (b/'metadata.json').write_text(json.dumps(meta))
        self.assertIn('observation unavailable', '\n'.join(compare_captures({'a': a, 'b': b})['warnings']))

    def test_cli_persistence_no_search_no_overwrite(self):
        a, b = self.capture('control'), self.capture('guided')
        original = (a/'runs.jsonl').read_bytes()
        output = self.root / 'comparison'
        with patch('meeple_bots.probes.runner.run_probes', side_effect=AssertionError('must not search')), redirect_stdout(io.StringIO()):
            self.assertEqual(main(['probe', 'compare', '--input', f'arbitrary={a}', '--input', f'other={b}',
                                   '--baseline', 'other', '--output', str(output)]), 0)
        data = json.loads((output/'comparison.json').read_text())
        self.assertEqual(data['baseline'], 'other')
        self.assertEqual((output/'comparison.txt').read_text(), render_comparison(data))
        self.assertEqual((a/'runs.jsonl').read_bytes(), original)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            write_comparison({'a': a, 'b': b}, output=output)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            compare_captures({'a': a})
        with self.assertRaisesRegex(ValueError, 'baseline'):
            compare_captures({'a': a, 'b': b}, baseline='missing')
