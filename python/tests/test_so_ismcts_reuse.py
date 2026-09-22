"""Real-path reuse through native matches; standalone search remains fresh."""
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from meeple_bots import LostCities, SoIsmctsAgent, RandomAgent, Match, benchmark_search_agent

class ReuseTests(unittest.TestCase):
    def test_profile_export_preserves_reuse_and_selector(self):
        from meeple_bots.studies import export_profile
        from meeple_bots._search_profiles import load_search_profile
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'agent.toml'
            for policy in ('uct', 'ucb1_tuned'):
                agent = SoIsmctsAgent(time_budget=.001, selection_policy=policy, tree_reuse=True)
                export_profile(path, 'reuse', agent)
                self.assertEqual(load_search_profile(path), agent)

    def test_standalone_is_fresh_even_when_config_requests_reuse(self):
        g=LostCities(); s=g.initial_state(42); o=g.observation(s,0); legal=g.legal_actions(s)
        for policy in ('uct','ucb1_tuned'):
            a=SoIsmctsAgent(iterations=64, selection_policy=policy)
            self.assertEqual(a.search(o,legal,seed=7), replace(a,tree_reuse=True).search(o,legal,seed=7))
        with self.assertRaises(TypeError): SoIsmctsAgent(tree_reuse=1)

    def test_matches_are_seeded_per_seat_and_report_only_new_work(self):
        g=LostCities()
        for policy in ('uct','ucb1_tuned'):
            agent=SoIsmctsAgent(iterations=64,selection_policy=policy,tree_reuse=True)
            first=Match(game=g,first=agent,second=agent,seed=42).run()
            second=Match(game=g,first=agent,second=agent,seed=42).run()
            clean=lambda r: [{k:v for k,v in asdict(m).items() if 'seconds' not in k} for m in r.moves]
            self.assertEqual(clean(first),clean(second))
            self.assertEqual(first.chance_events,second.chance_events)
            for seat in (0,1):
                moves=[m for m in first.moves if m.player==seat]
                self.assertTrue(all(m.search_iterations==64 for m in moves))
                self.assertTrue(any(m.tree_reuse.transition_hits for m in moves))
                self.assertTrue(any(m.tree_reuse.reused_root_visits for m in moves))
                self.assertTrue(any(m.tree_reuse.pruned_nodes for m in moves))
                self.assertEqual(moves[0].tree_reuse.reused_root_visits,0)
            self.assertEqual(first.lost_cities_state.status,'terminal')

    def test_time_budget_and_independent_position_benchmark(self):
        a=SoIsmctsAgent(time_budget=.0001,tree_reuse=True)
        result=Match(game=LostCities(),first=a,second=RandomAgent(),seed=7).run()
        self.assertTrue(all(m.search_iterations>=1 for m in result.moves if m.player==0))
        self.assertTrue(all(m.maintenance_seconds>=0 for m in result.moves))
        b=benchmark_search_agent(LostCities(),SoIsmctsAgent(iterations=4,tree_reuse=True),20)
        self.assertTrue(b['agent']['tree_reuse'])
        self.assertTrue(all(t['iterations']==4 for t in b['position_timings']))
