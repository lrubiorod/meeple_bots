"""Family capabilities and a strictly descriptive, resumable Random baseline."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from meeple_bots import MctsAgent, SoIsmctsAgent
from meeple_bots._study_profiles import PROFILES
from meeple_bots._study_tuners import changes, proposals
from meeple_bots.studies import StudyRunner, profile_values, agent_from_values, SEED_STRIDE
from meeple_bots.tournaments import match_jobs


class FamilyStudyTests(unittest.TestCase):
    def runner(self, path, base=None, **kwargs):
        return StudyRunner('lost_cities', base or SoIsmctsAgent(iterations=1),
            output=Path(path), games_per_comparison=8, workers=1, progress=lambda _: None, **kwargs)

    def test_capabilities_are_real_and_defaults_unchanged(self):
        self.assertEqual(PROFILES['so_ismcts'].supported_tuners, ('exploration', 'selection', 'tree-reuse'))
        self.assertEqual(PROFILES['so_ismcts'].mechanisms, ('tree_reuse',))
        for dimension in ('rave', 'structure', 'progressive-widening', 'unknown'):
            with TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "agent family 'so_ismcts'"):
                self.runner(tmp, tune=dimension)
        with TemporaryDirectory() as tmp:
            r=self.runner(tmp, all_search=True)
            self.assertFalse(r.state['request']['vs_random'])
            self.assertEqual({s['dimension'] for s in r.state['request']['tuner_specs'].values()}, {'exploration','selection','tree-reuse'})
        with TemporaryDirectory() as tmp:
            r=StudyRunner('connect6', MctsAgent(iterations=1), output=Path(tmp), progress=lambda _: None)
            self.assertEqual(r.family_profile.name, 'mcts')
            self.assertEqual(len(proposals('structure', MctsAgent())), 3)

    def test_selection_from_both_policies_freezes_compute(self):
        for selector in ('uct', 'ucb1_tuned'):
            base=SoIsmctsAgent(time_budget=.0001, selection_policy=selector)
            with TemporaryDirectory() as tmp:
                r=self.runner(tmp,base,tune='selection'); r.calibrate()
                name=next(iter(r.state['request']['tuner_specs']))
                p=r.profile.build_phase(name,r.state,base)
                challenger=agent_from_values(p['agents'][p['contrasts'][0]['b']])
                self.assertEqual(set(changes(base,challenger)), {'selection_policy'})
                self.assertEqual(challenger.time_budget, base.time_budget)
        with TemporaryDirectory() as tmp:
            r=self.runner(tmp,SoIsmctsAgent(iterations=1,selection_policy='ucb1_tuned'),selection_search=True)
            r.calibrate()
            p=r.profile.build_phase('exploration_coarse',r.state,r.base)
            self.assertTrue(p['contrasts'])
            self.assertTrue(all(p['agents'][c['b']]['selection_policy']=='uct' for c in p['contrasts']))
            self.assertEqual(p['agents']['incumbent']['selection_policy'],'ucb1_tuned')
        with TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'does not use exploration'):
            self.runner(tmp,SoIsmctsAgent(selection_policy='ucb1_tuned'),tune='exploration')

    def test_tree_reuse_local_and_mechanism_candidates_freeze_other_fields(self):
        for selector in ('uct', 'ucb1_tuned'):
            for enabled in (False, True):
                base = SoIsmctsAgent(time_budget=.0001, selection_policy=selector, tree_reuse=enabled)
                for flags in ({'tune': 'tree-reuse'}, {'mechanism_search': True}):
                    with TemporaryDirectory() as tmp:
                        r = self.runner(tmp, base, **flags)
                        r.calibrate()
                        name = next(n for n, s in r.state['request']['tuner_specs'].items()
                                    if s['dimension'] == 'tree-reuse')
                        phase = r.profile.build_phase(name, r.state, base)
                        self.assertEqual(len(phase['contrasts']), 1)
                        challenger = agent_from_values(phase['agents'][phase['contrasts'][0]['b']])
                        self.assertEqual(set(changes(base, challenger)), {'tree_reuse'})
                        self.assertEqual(challenger.tree_reuse, not enabled)
                        self.assertNotIn('transpositions', phase['agents'][phase['contrasts'][0]['b']])

    def test_random_native_fixed_profile_reports_and_does_not_select(self):
        with TemporaryDirectory() as tmp:
            r=self.runner(tmp,vs_random=True)
            self.assertEqual(r.phase_names, ('confirmation','random_baseline'))
            state=r.run(); baseline=state['random_baseline']
            self.assertEqual(baseline['games'],8)
            self.assertEqual(baseline['paired_seeds'],4)
            self.assertEqual(baseline['wins']+baseline['draws']+baseline['losses'],8)
            self.assertEqual(baseline['score'], (baseline['wins']+.5*baseline['draws'])/8)
            original=profile_values(r.base)
            self.assertEqual(state['selected_candidate']['profile'],original)
            self.assertEqual(baseline['champion'],original)
            self.assertFalse(state['improvement_comparisons'])
            self.assertIn('Diagnostic only', (Path(tmp)/'report.html').read_text())
            self.assertEqual(json.loads((Path(tmp)/'summary.json').read_text())['random_baseline'],baseline)
            phase=state['phases']['random_baseline']
            phase['contrasts'][0]['result'].update(score_b=0,ci95_b=[0,0])
            r.save()
            self.assertEqual(r.state['selected_candidate']['profile'],original)
            phase['contrasts'][0]['result'].update(score_b=1,ci95_b=[1,1])
            r.save()
            self.assertEqual(r.state['selected_candidate']['profile'],original)
            config=r._trace_config(phase,0)
            jobs=list(match_jobs([(config.agents[0],config.agents[1])],config))
            self.assertEqual(jobs[0].seed,jobs[1].seed)
            self.assertNotEqual(jobs[0].agent_a_player,jobs[1].agent_a_player)
            self.assertEqual(jobs[0].seed,42+2*SEED_STRIDE)
            with self.assertRaisesRegex(ValueError,'configuration differs'):
                self.runner(tmp,resume=True)

    def test_random_budget_pruning_and_final_champion(self):
        with TemporaryDirectory() as tmp:
            r=self.runner(tmp,vs_random=True);r.calibrate()
            chosen=profile_values(SoIsmctsAgent(iterations=1,selection_policy='ucb1_tuned'))
            r.state['selected_candidate']['profile']=chosen
            phase=r._random_phase()
            self.assertEqual(phase['agents']['champion'],chosen)
            self.assertFalse(phase['accept'])
            r.budget=r.spent+.1
            with patch.object(r,'_cost_pair',return_value=10):
                r._plan_phase(phase,1)
            self.assertFalse(phase['contrasts'])
            self.assertEqual(phase['discarded_comparisons'][0]['target_pairs'],4)

    def test_random_resume_only_missing_games_and_local_selection(self):
        from meeple_bots.studies import run_matches as real_run
        with TemporaryDirectory() as tmp:
            def interrupt(game,jobs,**kwargs):
                for row in real_run(game,jobs,**kwargs):
                    yield row
                    if any(isinstance(a.agent,SoIsmctsAgent) for a in (jobs[0].agent_a,jobs[0].agent_b)):
                        raise RuntimeError('interrupt Random comparison')
            with patch('meeple_bots.studies.run_matches',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'interrupt Random'):
                    self.runner(tmp,vs_random=True).run()
            trace=Path(tmp)/'traces/random_baseline-00.jsonl'; prefix=trace.read_bytes()
            self.assertEqual(len(prefix.splitlines()),2)
            state=self.runner(tmp,vs_random=True,resume=True).run()
            self.assertEqual(state['random_baseline']['games'],8)
            self.assertTrue(trace.read_bytes().startswith(prefix))
            self.assertEqual(len(trace.read_text().splitlines()),9)
        with TemporaryDirectory() as tmp:
            state=self.runner(tmp,tune='selection',vs_random=True).run()
            self.assertEqual(state['random_baseline']['games'],8)
            self.assertEqual(state['random_baseline']['champion'],state['selected_candidate']['profile'])

    def test_mcts_random_uses_same_coordinator(self):
        with TemporaryDirectory() as tmp:
            base=MctsAgent(iterations=1)
            r=StudyRunner('tic-tac-toe',base,output=Path(tmp),vs_random=True,
                games_per_comparison=8,workers=1,progress=lambda _:None)
            self.assertEqual(r.phase_names,('random_baseline',))
            state=r.run()
            self.assertEqual(state['random_baseline']['games'],8)
            self.assertEqual(state['selected_candidate']['profile'],profile_values(base))
