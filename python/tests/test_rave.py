"""Classic deterministic UCT-RAVE configuration and native integration."""
import csv
import io
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from meeple_bots import Boop, ConnectFour, Mast, ProgressiveBias, GameHeuristic, Connect6, Match, MctsAgent, RandomAgent, Splendor, benchmark_mcts_agent
from meeple_bots.cli import main, _load_tournament_agents
from meeple_bots._mcts_profiles import _load_mcts_profile, _parse_inline_mcts_profile
from meeple_bots.serialization import agent_dict
from meeple_bots.studies import profile_values, agent_from_values
from meeple_bots.extraction import extract_tournament
from meeple_bots.tournaments import TournamentConfig, TournamentAgent, run_tournament


class RaveTests(unittest.TestCase):
    def agent(self, **kwargs):
        return MctsAgent(iterations=16, rollout_depth=3,selection_policy='uct_rave',rave_equivalence=37,**kwargs)

    def test_parameters_round_trip_and_invalid_values(self):
        self.assertEqual(MctsAgent(selection_policy='uct_rave').rave_equivalence,1000)
        for invalid in (0,-1,True,1.5,2**32):
            with self.assertRaises((ValueError,TypeError)): MctsAgent(rave_equivalence=invalid)
        agent=self.agent()
        self.assertEqual(agent_from_values(profile_values(agent)),agent)
        self.assertEqual(agent_dict('rave',agent)['rave_equivalence'],37)
        self.assertNotIn('rave_equivalence',agent_dict('uct',MctsAgent()))
        with tempfile.TemporaryDirectory() as tmp:
            profile=Path(tmp)/'rave.toml'
            profile.write_text('iterations=16\nrollout_depth=3\nselection_policy="uct_rave"\nrave_equivalence=37\n')
            self.assertEqual(_load_mcts_profile(profile).agent,agent)
        inline=_parse_inline_mcts_profile('iterations=16,depth=3,selection_policy=uct_rave,rave_equivalence=37')
        self.assertEqual(inline.agent,agent)

    def test_complete_matches_all_backends_and_stochastic_rejection(self):
        game=Connect6(6)
        for reuse in (False,True):
            for transpositions in (False,True):
                agent=self.agent(tree_reuse=reuse,transpositions=transpositions)
                first=Match(game,agent,RandomAgent(),seed=42).run()
                second=Match(game,agent,RandomAgent(),seed=42).run()
                self.assertEqual([m.action for m in first.moves],[m.action for m in second.moves])
                state=game.initial_state()
                for move in first.moves:
                    self.assertEqual(state.current_player,move.player)
                    state=state.apply_action(move.action)
                self.assertTrue(state.terminal)
        with self.assertRaisesRegex((ValueError,RuntimeError),'deterministic'):
            Match(Splendor(),self.agent(),RandomAgent()).run()
        benchmark_mcts_agent(game,self.agent(),median_depth=4,seed=42)

    def test_mast_and_progressive_bias_remain_independent(self):
        for transpositions in (False, True):
            mast = self.agent(rollout_policy=Mast(0.5), tree_reuse=True, transpositions=transpositions)
            self.assertGreater(Match(ConnectFour(),mast,RandomAgent(),seed=42).run().plies,0)
            bias = self.agent(progressive_bias=ProgressiveBias(0.25,GameHeuristic(0)), transpositions=transpositions)
            benchmark_mcts_agent(Boop(),bias,median_depth=4,seed=42)

    def test_cli_and_tournament_grid_extract_preserve_k(self):
        output=io.StringIO()
        with redirect_stdout(output),redirect_stderr(io.StringIO()):
            self.assertEqual(main(['match','--game','connect6','--game-param','board_size=6',
                '--first','mcts','--second','random','--mcts-iterations','16','--mcts-rollout-depth','3',
                '--mcts-selection-policy','uct_rave','--mcts-rave-equivalence','37','--json']),0)
        self.assertIn('uct_rave',output.getvalue())
        values=dict(name='rave',kind='mcts',iterations=16,rollout_depth=3,
                    selection_policy='uct_rave',rave_equivalence=[37,1000])
        agents=_load_tournament_agents(values,0,Connect6(6))
        self.assertEqual([a.agent.rave_equivalence for a in agents],[37,1000])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config=TournamentConfig(Connect6(6),root/'trace.jsonl','round_robin','paired',2,42,36,1,
                    tuple(TournamentAgent(a.name,a.agent) for a in agents))
            run_tournament(config)
            self.assertTrue(extract_tournament(config.output,root/'data')['complete'])
            with (root/'data/agents.csv').open() as source:
                rows=list(csv.DictReader(source))
            self.assertEqual({r['rave_equivalence'] for r in rows},{'37','1000'})

if __name__=='__main__': unittest.main()
