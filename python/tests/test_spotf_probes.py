"""Correctness and experimental isolation; no strategic action assertions."""
from dataclasses import fields, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from meeple_bots import (MctsAgent, RandomAgent, SpiritsOfTheForest, ConditionalRollout,
                         EpsilonGreedy, GameHeuristic, ProgressiveBias, TurnPhaseIs, UniformRandom)
from meeple_bots.api import _native_agent, TakeSpiritTile
from meeple_bots.spotf import SpotfPosition
from meeple_bots.probes import run_probes, select_cases
from meeple_bots.probes.runner import run_variants
from meeple_bots.probes.games.spotf import CASES, FIXTURES
from meeple_bots.probes.report import render_variants

def profiles():
    """Self-contained configurations exercising independent conditional mechanisms."""
    control = MctsAgent(time_budget=.5, rollout_depth=128)
    condition = TurnPhaseIs('collect')
    return {
        'A': control,
        'B': replace(control, rollout_policy=ConditionalRollout(
            condition, EpsilonGreedy(.1, GameHeuristic(0)), UniformRandom())),
        'C': replace(control, progressive_bias=ProgressiveBias(.25, GameHeuristic(0), condition)),
    }


class SpotfProbeTests(unittest.TestCase):
    def test_fixtures_replay_legally_and_are_public(self):
        self.assertEqual(len(select_cases(game='spotf')), 11)
        for case in CASES:
            p = case.builder()
            observation = p.observation
            replay = SpotfPosition(observation.seed)
            for index in observation.action_indices:
                replay = replay.apply(replay.legal_actions()[index])
            self.assertEqual(replay.state, observation.state)
            self.assertEqual(p.legal_actions, replay.legal_actions())
            self.assertTrue(all(a in p.legal_actions for a in case.candidate_actions))
            self.assertEqual(observation.state[3].value, p.fixture['phase'])
            board, collections, gems, _, player, _ = replay.state
            self.assertEqual(sum(t is not None for row in board for t in row) + sum(c.tiles for c in collections), 48)
            for owner, pool in enumerate(gems):
                self.assertEqual(pool.available+pool.placed+pool.removed, 3)
                self.assertEqual(pool.placed, sum(t is not None and t.gemstone == owner for row in board for t in row))
            self.assertEqual(p.root_player, player)
        self.assertGreaterEqual(FIXTURES['high_branching_gem_position']['public_context']['legal_action_count'], 80)

    def test_contrast_changes_only_public_collection_allocation(self):
        a = next(c for c in CASES if c.id.endswith('context_a')).builder().observation.state
        b = next(c for c in CASES if c.id.endswith('context_b')).builder().observation.state
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[2:], b[2:])
        self.assertNotEqual(a[1], b[1])
        self.assertEqual(a[1], tuple(reversed(b[1])))

    def test_unrecoverable_annotation_has_public_evidence(self):
        context = FIXTURES['unrecoverable_category']['public_context']
        self.assertTrue(any(a+r < b for a,b,r in zip(context['own_symbols'],context['opponent_symbols'],context['remaining_symbols'])))
        case = next(c for c in CASES if c.id.endswith('avoid_unnecessary_gem_sacrifice'))
        self.assertEqual(case.builder().fixture['phase'], 'collect')
        self.assertTrue(any(isinstance(a,TakeSpiritTile) and a.sacrifice for a in case.candidate_actions))
        self.assertTrue(any(isinstance(a,TakeSpiritTile) and a.sacrifice is None for a in case.candidate_actions))

    def test_frozen_configs_change_one_field_only(self):
        p = profiles()
        def changes(x,y):
            return {f.name for f in fields(x) if getattr(x,f.name)!=getattr(y,f.name)}
        self.assertEqual(changes(p['A'],p['B']), {'rollout_policy'})
        self.assertEqual(changes(p['A'],p['C']), {'progressive_bias'})
        self.assertEqual(p['B'].rollout_policy.primary.epsilon, .1)
        self.assertEqual(p['B'].rollout_policy.condition.phase, 'collect')
        self.assertEqual(p['C'].progressive_bias.condition.phase, 'collect')
        self.assertEqual(p['C'].progressive_bias.weight, .25)
        self.assertFalse(p['A'].progressive_widening)
        self.assertEqual(p['A'].time_budget, .5)

    def test_native_reproducibility_legal_actions_and_bias_phase(self):
        p = profiles()
        for case in (CASES[0], next(c for c in CASES if 'place_gemstone' in c.tags)):
            pos = case.builder().observation
            for name, agent in p.items():
                agent = replace(agent, iterations=32, time_budget=None)
                a = pos.search(agent,seed=17); b = pos.search(agent,seed=17)
                a['diagnostics'].pop('search_seconds'); b['diagnostics'].pop('search_seconds')
                self.assertEqual(a,b)
                self.assertIn(a['action'],pos.legal_actions())
                self.assertEqual(len(a['root_actions']),len(pos.legal_actions()))
                self.assertEqual(a['root_visits'],32)
                self.assertEqual(sum(e['visits'] for e in a['root_actions']),32)
                measured=[e['heuristic_value'] for e in a['root_actions'] if e['visits']]
                if name=='C' and pos.state[3].value=='collect':
                    self.assertTrue(any(v is not None for v in measured))
                elif name == 'C':
                    self.assertTrue(all(v == 0 for v in measured))
                    self.assertTrue(all(e['progressive_bias'] in (None, 0) for e in a['root_actions']))
                else:
                    self.assertTrue(all(v is None for v in measured))
            with self.assertRaises(ValueError):
                pos._position.search(_native_agent(RandomAgent(),SpiritsOfTheForest()),0)
            with self.assertRaises(ValueError):
                pos._position.apply(99999)

    def test_named_variants_time_metrics_and_focus_preserve_actions(self):
        with TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            rows=run_variants(CASES[:1], profiles(), iterations=(4,8), seeds=2, output=out,progress=lambda _:None)
            self.assertEqual(len(rows),6)
            self.assertIn('Other',render_variants(rows))
            self.assertEqual({r['variant'] for r in rows},{'A','B','C'})
            self.assertTrue(all(r['iterations_per_second']>0 for r in rows))
            self.assertTrue(all(r['median_iterations']==r['iterations'] for r in rows))
            raw=[json.loads(l) for l in (out/'C/runs.jsonl').read_text().splitlines()]
            self.assertTrue(all(len(r['root_actions'])==len(CASES[0].builder().legal_actions) for r in raw))
            self.assertTrue(all(e['availability'] is None for r in raw for e in r['root_actions']))
            timed=run_probes(CASES[:1],profiles()['A'],decision_seconds=.003,seeds=2,
                             output=Path(tmp)/'timed',progress=lambda _:None)
            self.assertIsNone(timed[0]['iterations'])
            self.assertEqual(timed[0]['decision_seconds'],.003)
            self.assertGreater(timed[0]['median_iterations'],0)
            def capture(case,name):
                path=Path(tmp)/name
                run_probes((case,),profiles()['A'],iterations=(8,),seeds=1,output=path,progress=lambda _:None)
                row=json.loads((path/'runs.jsonl').read_text())
                return row['selected_action'],[(e['action'],e['visits'],e['q']) for e in row['root_actions']]
            self.assertEqual(capture(CASES[0],'focus'),capture(replace(CASES[0],candidate_actions=()),'all'))

    def test_every_fixture_native_smoke(self):
        for case in CASES:
            p=case.builder()
            r=case.search(MctsAgent(iterations=2),p.observation,p.legal_actions,seed=0)
            self.assertIn(r['action'],p.legal_actions)
