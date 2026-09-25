"""Family dispatch and SO-ISMCTS policy on the shared study execution pipeline."""
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from meeple_bots import SoIsmctsAgent, MctsAgent
from meeple_bots.cli import main
from meeple_bots.studies import StudyRunner, _group_leaders, profile_values, agent_from_values, export_profile, SEED_STRIDE
from meeple_bots._study_profiles import load_search_profile
from meeple_bots.studies.planning import build_so_phase
from meeple_bots._study_tuners import proposals, changes
from meeple_bots.tournaments import match_jobs


class SoIsmctsStudyTests(unittest.TestCase):
    def make_runner(self, directory, baseline=None, **kw):
        return StudyRunner('lost_cities', baseline, output=Path(directory), workers=1,
                           games_per_comparison=8, progress=lambda _: None, **kw)

    def calibrated(self, directory, baseline=None, **kw):
        runner=self.make_runner(directory, baseline or SoIsmctsAgent(iterations=1), **kw)
        runner.calibrate()
        return runner

    def test_family_resolution_cli_and_hidden_state_compatibility(self):
        with TemporaryDirectory() as tmp:
            r=self.make_runner(tmp)
            self.assertEqual(r.family, 'so_ismcts')
            self.assertEqual(r.state['request']['agent_family'], 'so_ismcts')
            self.assertEqual(tuple(r.phase_names), ('exploration_coarse', 'exploration_refine_1', 'exploration_refine_2', 'confirmation'))
        for args in ({'agent_family':'mcts'}, {'baseline':MctsAgent()}, {'pw_search':True}, {'rave_search':True}):
            with TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                self.make_runner(tmp, **args)
        with TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'compatible'):
            StudyRunner('tic-tac-toe', SoIsmctsAgent(), output=Path(tmp))
        for explicit in ([], ['--agent','so_ismcts']):
            with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
                code=main(['study','--game','lost_cities','--budget','0.000001s','--output',tmp,'--json',*explicit])
                self.assertEqual(code,0)
                self.assertEqual(json.loads(out.getvalue())['status'],'budget_exhausted')
                state=json.loads((Path(tmp)/'study.json').read_text())
                self.assertEqual(state['request']['agent_family'],'so_ismcts')

    def test_target_time_calibration_and_equal_compute_candidates(self):
        with TemporaryDirectory() as tmp:
            r=self.make_runner(tmp,target_match_time=.1)
            r.calibrate()
            c=r.state['calibration']
            self.assertAlmostEqual(c['decision_seconds'],.1/(c['estimated_game_decisions']*1.2))
            self.assertIsNone(c['fixed_iterations'])
            self.assertEqual(c['median_iterations_per_decision'],c['median_determinizations_per_decision'])
            self.assertEqual(c['iterations_per_second'],c['determinizations_per_second'])
            self.assertGreater(c['seconds_per_iteration'],0)
            self.assertGreater(c['root_action_coverage'],0)
            self.assertEqual(c['horizon'],{'kind':'unbounded','depth':None})
            phase=build_so_phase('exploration_coarse',r.state,r.base)
            base=agent_from_values(phase['agents']['incumbent'])
            self.assertGreaterEqual(len(phase['contrasts']),3)
            for name,v in phase['agents'].items():
                a=agent_from_values(v)
                self.assertEqual(a.time_budget,base.time_budget)
                self.assertIsNone(a.iterations)
                self.assertLessEqual(set(changes(base,a)),{'exploration'})

    def test_local_retune_frozen_budget_profile_roundtrip_and_boundary_expansion(self):
        base=SoIsmctsAgent(iterations=3,exploration=.5)
        with TemporaryDirectory() as tmp:
            r=self.calibrated(tmp,base,tune='exploration')
            phase=build_so_phase('exploration_coarse',r.state,base)
            for v in phase['agents'].values():
                a=agent_from_values(v)
                self.assertEqual(a.iterations,3)
                self.assertLessEqual(set(changes(base,a)),{'exploration'})
            file=Path(tmp)/'profile.toml';export_profile(file,'test',base)
            self.assertEqual(load_search_profile(file),base)
            timed=SoIsmctsAgent(time_budget=.001,exploration=.75)
            export_profile(file,'timed',timed)
            self.assertEqual(load_search_profile(file),timed)
        generated=proposals('exploration',replace(base,exploration=2),tested={.25,.5,1,1.4,2})
        self.assertTrue(any(a.exploration>2 for a in generated))
        self.assertLessEqual(len(generated),2)
        for bad in ({'tune':'rave'}, {'tune':'exploration','decision_seconds':.01}):
            with TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                self.make_runner(tmp,base,**bad)

    def test_budget_prunes_breadth_not_evidence_and_inconclusive_keeps_control(self):
        with TemporaryDirectory() as tmp:
            r=self.calibrated(tmp)
            phase=build_so_phase('exploration_coarse',r.state,r.base)
            r.budget=r.spent+10
            with patch.object(r,'_cost_pair',return_value=1):
                r._plan_phase(phase,0)
            self.assertEqual(len(phase['contrasts']),2)
            self.assertTrue(phase['discarded_comparisons'])
            self.assertTrue(all(c['target_pairs']==4 for c in phase['contrasts']))
            for c in phase['contrasts']:
                c['result']={'seed_pairs':4,'score_b':.5,'seed_scores_b':{str(i):.5 for i in range(4)}}
            self.assertEqual(_group_leaders(phase)['main'],'incumbent')
            r.budget=r.spent+.01
            phase=build_so_phase('exploration_coarse',r.state,r.base)
            with patch.object(r,'_cost_pair',return_value=1):
                r._plan_phase(phase,0)
            self.assertFalse(phase['contrasts'])
            self.assertTrue(all(c['reason']=='insufficient_budget' for c in phase['discarded_comparisons']))

    def test_confirmation_fresh_paired_seeds_and_conservative_promotion(self):
        with TemporaryDirectory() as tmp:
            r=self.calibrated(tmp)
            r.state['selected_candidate']['profile']=profile_values(SoIsmctsAgent(iterations=1,exploration=.75))
            phase=build_so_phase('confirmation',r.state,r.base)
            r._plan_phase(phase,3)
            config=r._trace_config(phase,0)
            jobs=list(match_jobs([(config.agents[0],config.agents[1])],config))
            self.assertEqual(len(jobs),8)
            # Existing scheduler supplies the same environmental seed with swapped seats.
            self.assertEqual(jobs[0].seed,jobs[1].seed)
            self.assertNotEqual(jobs[0].agent_a_player,jobs[1].agent_a_player)
            self.assertEqual(jobs[0].seed,r.state['request']['seed']+4*SEED_STRIDE)
            c=phase['contrasts'][0]
            c['result']={'seed_pairs':4,'score_b':.75,'ci95_b':[.1,1], 'seed_scores_b':{str(i):.75 for i in range(4)}}
            self.assertEqual(_group_leaders(phase)['main'],'incumbent')
            c['result']['ci95_b']=[.51,.99]
            self.assertEqual(_group_leaders(phase)['main'],'finalist')

    def test_interrupted_so_race_resumes_only_missing_seats(self):
        from meeple_bots.studies import run_matches as real_run
        with TemporaryDirectory() as tmp:
            base=SoIsmctsAgent(iterations=1)
            def interrupt(game, jobs, **kw):
                for item in real_run(game, jobs, **kw):
                    yield item
                    # Pilot random agents have no iteration configuration.
                    if isinstance(jobs[0].agent_a.agent, SoIsmctsAgent):
                        raise RuntimeError('pause during C race')
            with patch('meeple_bots.studies.coordinator.run_matches', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'pause during C race'):
                    self.make_runner(tmp,base).run()
            trace=Path(tmp)/'traces/exploration_coarse-00.jsonl'
            prefix=trace.read_bytes()
            self.assertEqual(len(prefix.splitlines()),2)
            state=self.make_runner(tmp,base,resume=True).run()
            self.assertEqual(state['status'],'complete')
            self.assertEqual(state['request']['agent_family'],'so_ismcts')
            self.assertTrue(trace.read_bytes().startswith(prefix))
            rows=[json.loads(line) for line in trace.read_text().splitlines()[1:]]
            self.assertEqual(len(rows),8)
            self.assertEqual(len({row['match_number'] for row in rows}),8)

    def test_local_time_budget_and_cli_profile_are_preserved(self):
        base=SoIsmctsAgent(time_budget=.0001, exploration=.75)
        with TemporaryDirectory() as tmp:
            root=Path(tmp);file=root/'base.toml';export_profile(file,'base',base)
            r=self.calibrated(root/'study',base,tune='exploration',target_match_time=100)
            self.assertEqual(r.state['calibration']['decision_seconds'],.0001)
            for v in build_so_phase('exploration_coarse',r.state,base)['agents'].values():
                candidate=agent_from_values(v)
                self.assertEqual(candidate.time_budget,.0001)
                self.assertLessEqual(set(changes(base,candidate)),{'exploration'})
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code=main(['study','--game','lost_cities','--agent-config',str(file),'--tune','exploration',
                           '--budget','0.000001s','--output',str(root/'cli')])
            self.assertEqual(code,0)
            champion=load_search_profile(root/'cli/candidates/best_agent.toml')
            self.assertEqual(champion,base)

    def test_small_native_study_resume_reporting_and_usable_champion(self):
        with TemporaryDirectory() as tmp:
            base=SoIsmctsAgent(iterations=1)
            r=self.make_runner(tmp,base)
            state=r.run()
            self.assertEqual(state['status'],'complete')
            champion=load_search_profile(Path(tmp)/'candidates/best_agent.toml')
            self.assertIsInstance(champion,SoIsmctsAgent)
            self.assertEqual(champion.iterations,1)
            summary=json.loads((Path(tmp)/'summary.json').read_text())
            self.assertEqual(summary['agent_family'],'so_ismcts')
            self.assertIn('so_ismcts',(Path(tmp)/'report.html').read_text())
            self.assertEqual(summary['final_selection']['confirmation_result'],'INCONCLUSIVE')
            self.assertEqual(champion,base)
            traces={p.name:p.read_bytes() for p in (Path(tmp)/'traces').glob('*.jsonl')}
            resumed=self.make_runner(tmp,base,resume=True).run()
            self.assertEqual(resumed['request']['agent_family'],'so_ismcts')
            self.assertEqual(traces,{p.name:p.read_bytes() for p in (Path(tmp)/'traces').glob('*.jsonl')})
            with self.assertRaisesRegex(ValueError,'configuration differs'):
                self.make_runner(tmp,base,resume=True,target_match_time=61)
            # The generic match profile alias can consume the exported family.
            from meeple_bots.cli import _match_agent
            loaded=_match_agent('so_ismcts',Path(tmp)/'candidates/best_agent.toml',MctsAgent(),None,'--first-agent-config','--first-mcts-heuristic')
            self.assertEqual(loaded,champion)


if __name__=='__main__':
    unittest.main()
