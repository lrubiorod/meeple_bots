"""Can't Stop sessions, public chance events and GUI lifecycle integration."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from meeple_bots import CantStopSession, MctsAgent, RandomAgent
from meeple_bots.cli import build_parser
from meeple_bots.games.cant_stop.gui import CantStopApplication, PAGE
from meeple_bots.gui.baselines import CANT_STOP_BASELINE
from meeple_bots.gui.server import run_gui


class CantStopTests(unittest.TestCase):
    def wait_for(self, app, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = app.snapshot()
            if predicate(state):
                return state
            if state['status'] == 'error':
                self.fail(state['message'])
            time.sleep(0.005)
        self.fail('GUI did not reach expected state')

    def test_session_exposes_json_safe_public_state_and_seeded_events(self):
        outcomes = []
        for _ in range(2):
            session = CantStopSession(seed=42, first=RandomAgent(), second=RandomAgent())
            while session.snapshot()['winner'] is None:
                session.step()
            state = session.snapshot()
            json.dumps(state)
            self.assertIsInstance(state['progress'][0], list)
            outcomes.append([(e['player'], e['action']) for e in state['events']])
            self.assertTrue(any(e['action']['kind'] == 'dice' for e in state['events']))
        self.assertEqual(*outcomes)

    def test_human_indices_and_chance_cannot_be_spoofed(self):
        session = CantStopSession()
        before = session.snapshot()
        for action in (0, True, -1):
            with self.assertRaises(ValueError):
                session.step(action)
            self.assertEqual(session.snapshot(), before)
        state = session.step()
        self.assertTrue(state['waiting_human'])
        self.assertTrue(all(a['kind'] == 'advance' for a in state['legal_actions']))
        self.assertEqual(session.step(0)['phase'], 'continue')
        with self.assertRaises(ValueError):
            session.step(1000)

    def test_mcts_policies_finish_and_unsupported_options_fail_explicitly(self):
        for policy in ('uct', 'ucb1_tuned'):
            session = CantStopSession(first=MctsAgent(iterations=16, rollout_depth=40, heuristic=0, selection_policy=policy), second=RandomAgent())
            while session.snapshot()['winner'] is None:
                session.step()
            self.assertTrue(all(e['search_iterations'] == 16 for e in session.snapshot()['events'] if e['player'] == 0))
        for kwargs in ({'heuristic': 1},):
            with self.assertRaises(ValueError):
                CantStopSession(first=MctsAgent(**kwargs))

    def test_informed_and_conditional_rollouts(self):
        from meeple_bots import Greedy, EpsilonGreedy, ConditionalRollout, TurnPhaseIs, GameHeuristic, UniformRandom
        for policy in (Greedy(GameHeuristic(0)), EpsilonGreedy(0.25, GameHeuristic(0)),
                       ConditionalRollout(TurnPhaseIs("continue"), Greedy(GameHeuristic(0)), UniformRandom())):
            session = CantStopSession(first=MctsAgent(iterations=8, rollout_depth=12, rollout_policy=policy), second=RandomAgent())
            for _ in range(20):
                if session.snapshot()['winner'] is not None:
                    break
                session.step()
            self.assertTrue(session.snapshot()['events'])
        for policy in (Greedy(GameHeuristic(99)),
                       ConditionalRollout(TurnPhaseIs("collect"), UniformRandom(), UniformRandom())):
            with self.assertRaises(ValueError):
                CantStopSession(first=MctsAgent(rollout_policy=policy))

    def test_mast_and_conditional_mast_preserve_uniform_at_epsilon_one(self):
        from meeple_bots import Mast, ConditionalRollout, TurnPhaseIs, UniformRandom
        histories = []
        for policy in (UniformRandom(), Mast(1.0), ConditionalRollout(TurnPhaseIs("choose"), Mast(1.0), UniformRandom())):
            session = CantStopSession(seed=42, first=MctsAgent(iterations=16, rollout_depth=20, rollout_policy=policy), second=RandomAgent())
            for _ in range(30):
                if session.snapshot()['winner'] is not None:
                    break
                session.step()
            histories.append([(e['player'], e['action']) for e in session.snapshot()['events']])
        self.assertEqual(histories[0], histories[1])
        self.assertEqual(histories[0], histories[2])
        session = CantStopSession(first=MctsAgent(iterations=16, rollout_depth=20, rollout_policy=Mast(0.1)), second=RandomAgent())
        for _ in range(10):
            session.step()

    def test_progressive_bias_and_its_validation(self):
        from meeple_bots import ProgressiveBias, TurnPhaseIs, GameHeuristic
        for bias in (ProgressiveBias(0.25, GameHeuristic(0)), ProgressiveBias(0.25, GameHeuristic(0), TurnPhaseIs("continue"))):
            session = CantStopSession(first=MctsAgent(iterations=16, rollout_depth=20, progressive_bias=bias, root_diagnostics=True), second=RandomAgent())
            for _ in range(10):
                session.step()
        for bias in (ProgressiveBias(0.25, GameHeuristic(99)), ProgressiveBias(0.25, GameHeuristic(0), TurnPhaseIs("collect"))):
            with self.assertRaises(ValueError):
                CantStopSession(first=MctsAgent(progressive_bias=bias))

    def test_tree_reuse_sessions_finish(self):
        from meeple_bots import Mast, ProgressiveBias, GameHeuristic
        for extra in ({}, {'rollout_policy': Mast(0.1), 'progressive_bias': ProgressiveBias(0.25, GameHeuristic(0))}):
            session = CantStopSession(first=MctsAgent(iterations=16, rollout_depth=20, tree_reuse=True, **extra), second=RandomAgent())
            for _ in range(3000):
                if session.snapshot()['winner'] is not None:
                    break
                session.step()
            self.assertIsNotNone(session.snapshot()['winner'])

    def test_transpositions_with_and_without_reuse(self):
        from meeple_bots import Mast, ProgressiveBias, GameHeuristic
        for reuse in (False, True):
            for selection in ('uct', 'ucb1_tuned'):
                session = CantStopSession(first=MctsAgent(iterations=16, rollout_depth=20, tree_reuse=reuse, transpositions=True, selection_policy=selection, rollout_policy=Mast(0.1), progressive_bias=ProgressiveBias(0.25, GameHeuristic(0))), second=RandomAgent())
                for _ in range(3000):
                    if session.snapshot()['winner'] is not None:
                        break
                    session.step()
                self.assertIsNotNone(session.snapshot()['winner'])

    def test_combined_options_export_consistent_diagnostics(self):
        from meeple_bots import Greedy, EpsilonGreedy, Mast, ConditionalRollout, TurnPhaseIs, ProgressiveBias, GameHeuristic, UniformRandom
        for policy in (UniformRandom(), Greedy(GameHeuristic(0)), EpsilonGreedy(0.1, GameHeuristic(0)), Mast(0.1), ConditionalRollout(TurnPhaseIs("continue"), Mast(0.1), Greedy(GameHeuristic(0)))):
            for selection in ('uct', 'ucb1_tuned'):
                for reuse, transpositions in ((False, False), (False, True), (True, False), (True, True)):
                    with self.subTest(policy=policy, selection=selection, reuse=reuse, transpositions=transpositions):
                        session = CantStopSession(first=MctsAgent(iterations=8, rollout_depth=12, rollout_policy=policy, selection_policy=selection, tree_reuse=reuse, transpositions=transpositions, progressive_bias=ProgressiveBias(0.25, GameHeuristic(0)), root_diagnostics=True), second=RandomAgent())
                        for _ in range(8):
                            session.step()
                        for event in session.snapshot()['events']:
                            if event['search_iterations'] is None:
                                self.assertEqual(event['root_actions'], [])
                                self.assertIsNone(event['tree_reuse'])
                                continue
                            inherited = event['tree_reuse']['reused_root_visits'] if reuse else 0
                            self.assertEqual(sum(a['visits'] for a in event['root_actions']), 8 + inherited)
                            self.assertEqual(sum(a['selected'] for a in event['root_actions']), 1)
                            self.assertGreater(event['search_nodes'], 0)
                        json.dumps(session.snapshot(), allow_nan=False)

    def test_policy_profile_roundtrips_through_gui(self):
        from meeple_bots.gui import baselines
        from meeple_bots.gui.player import parse_gui_player
        from meeple_bots._mcts_profiles import _load_mcts_profile
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'cant-stop-baseline.toml'
            path.write_text('iterations = 16\nrollout_depth = 20\nrollout_policy = { kind = "conditional", condition = { kind = "turn_phase", phase = "continue" }, primary = { kind = "epsilon_greedy", epsilon = 0.2, evaluator = { kind = "game_heuristic", index = 0 } }, fallback = { kind = "mast", epsilon = 0.1 } }\nprogressive_bias = { weight = 0.25, evaluator = { kind = "game_heuristic", index = 0 } }\ntree_reuse = true\ntranspositions = true\nroot_diagnostics = true\n')
            with patch.object(baselines, '_baseline_path', return_value=path):
                defaults = baselines._load_gui_baseline(path.name)
            fields = json.loads(json.dumps(defaults.as_dict()))
            parsed = parse_gui_player(fields, 'first', default_rollout_depth=100, available_heuristics=(0,), with_policies=True)
            self.assertEqual(parsed.to_agent(), _load_mcts_profile(path).agent)
            app = CantStopApplication()
            try:
                state = app.start({'first': fields, 'second': {'kind': 'human'}, 'minimum_move_seconds': 0})
                self.assertEqual(state['players'][0], fields)
            finally:
                app.cancel()

    def test_gui_human_moves_restart_and_invalid_start_are_atomic(self):
        app = CantStopApplication()
        try:
            app.start({'first': {'kind': 'human'}, 'second': {'kind': 'human'}, 'minimum_move_seconds': 0})
            state = self.wait_for(app, lambda s: s['status'] == 'waiting_human')
            previous = deepcopy(state)
            with self.assertRaises(ValueError):
                app.start({'first': {'kind': 'mcts', 'heuristic': 99}, 'second': {'kind': 'random'}})
            self.assertEqual(app.snapshot(), previous)
            with self.assertRaises(ValueError):
                app.move({'action': 0, 'turn': len(state['events']) - 1})
            app.move({'action': 0, 'turn': len(state['events'])})
            state = self.wait_for(app, lambda s: s['status'] == 'waiting_human' and s['phase'] == 'continue')
            self.assertGreater(len(state['events']), len(previous['events']))
            app.start({'first': {'kind': 'human'}, 'second': {'kind': 'human'}, 'seed': 12, 'minimum_move_seconds': 0})
            state = self.wait_for(app, lambda s: s['status'] == 'waiting_human')
            self.assertEqual(len(state['events']), 1)
        finally:
            app.cancel()

    def test_gui_saves_dice_history_and_cli_dispatches(self):
        with TemporaryDirectory() as directory:
            app = CantStopApplication(Path(directory))
            try:
                app.start({'first': {'kind': 'random'}, 'second': {'kind': 'random'}, 'minimum_move_seconds': 0, 'save_trace': True})
                state = self.wait_for(app, lambda s: s['trace_path'] is not None)
                data = json.loads(Path(state['trace_path']).read_text())
                self.assertEqual(data['format'], 'cant_stop_session_v1')
                self.assertEqual(data['winner'], state['winner'])
                self.assertTrue(any(e['action']['kind'] == 'dice' for e in data['events']))
            finally:
                app.cancel()
        args = build_parser().parse_args(['gui', '--game', 'cant-stop', '--no-browser'])
        self.assertEqual(args.game, 'cant-stop')
        with patch('meeple_bots.gui.server.serve_gui') as serve:
            run_gui(game='cant-stop', open_browser=False)
            self.assertIsInstance(serve.call_args.args[0], CantStopApplication)

    def test_gui_defaults_and_browser_script_parse(self):
        self.assertIn('const baseline = ' + json.dumps(CANT_STOP_BASELINE.as_dict()), PAGE)
        if shutil.which('node'):
            script = PAGE.split('<script>', 1)[1].split('</script>', 1)[0]
            subprocess.run(['node', '--check'], input=script, text=True, check=True, capture_output=True)

    @unittest.skipUnless(shutil.which('node'), 'Node is required for GUI interaction tests')
    def test_polling_preserves_action_buttons_until_the_decision_changes(self):
        script = PAGE.split('<script>', 1)[1].split('</script>', 1)[0]
        harness = r"""
const assert=require('node:assert/strict');
class Element {
 constructor(){this.children=[];this.value='';this.classList={add(){},toggle(){}};}
 append(child){this.children.push(child);}
 replaceChildren(){this.children=[];}
}
const elements=new Map();
global.document={createElement:()=>new Element(),querySelector(selector){
 if(!elements.has(selector))elements.set(selector,new Element());
 return elements.get(selector);
}};
global.setTimeout=()=>{};
let reply={status:'waiting_human',phase:'choose',active_player:0,winner:null,
 heights:Array(11).fill(3),progress:[Array(11).fill(0),Array(11).fill(0)],
 runners:Array(11).fill(0),claimed:Array(11).fill(null),dice:[1,2,3,4],
 events:[{player:null,action:{kind:'dice',dice:[1,2,3,4]},search_iterations:null}],
 legal_actions:[{kind:'advance',columns:[3,7]},{kind:'advance',columns:[5,5]}]};
const sent=[];
let completeMove;
global.fetch=async(path,options)=>{
 if(path==='/api/move'){
  sent.push(JSON.parse(options.body));
  return new Promise(resolve=>{completeMove=()=>resolve({ok:true,json:async()=>structuredClone(reply)});});
 }
 return {ok:true,json:async()=>structuredClone(reply)};
};
"""
        assertions = r"""
(async()=>{
 await new Promise(setImmediate); // Initial page poll finishes.
 assert.deepEqual(config(1),{...baseline,kind:'mcts'});
 document.querySelector('#primary-policy-1').value='mast';
 document.querySelector('#primary-epsilon-1').value='0.2';
 document.querySelector('#rollout-phase-1').value='continue';
 document.querySelector('#fallback-policy-1').value='greedy';
 document.querySelector('#fallback-evaluator-1').value='none';
 document.querySelector('#reuse-1').checked=true;
 document.querySelector('#transpositions-1').checked=true;
 document.querySelector('#bias-1').checked=true;
 document.querySelector('#bias-weight-1').value='0.5';
 document.querySelector('#bias-phase-1').value='choose';
 const settings=config(1);
 assert.deepEqual(settings.rollout_policy,{kind:'conditional',condition:{kind:'turn_phase',phase:'continue'},primary:{kind:'mast',epsilon:0.2},fallback:{kind:'greedy',evaluator:{kind:'neutral'}}});
 assert.equal(settings.tree_reuse,true);assert.equal(settings.transpositions,true);
 assert.equal(settings.progressive_bias.weight,0.5);assert.equal(settings.progressive_bias.condition.phase,'choose');
 const actions=document.querySelector('#actions');
 const pressed=actions.children[0];
 // Repeated refreshes between pointer-down and pointer-up must retain the target.
 for(let i=0;i<5;i++){
  await poll();
  assert.equal(actions.children[0],pressed);
  assert.equal(pressed.disabled,false);
 }
 const sending=pressed.onclick();
 assert.equal(actions.children[0],pressed);
 assert.equal(pressed.disabled,true);
 await pressed.onclick(); // A second activation cannot submit the same move twice.
 assert.deepEqual(sent,[{action:0,turn:1}]);
 await poll();
 assert.equal(actions.children[0],pressed);
 assert.equal(pressed.disabled,true);
 reply={...reply,status:'playing'};
 completeMove();await sending;
 assert.equal(actions.children.length,0);
 // Even identical legal actions in a later turn must acquire the new turn token.
 reply={...reply,status:'waiting_human',events:[...reply.events,reply.events[0]]};
 await poll();
 assert.notEqual(actions.children[0],pressed);
 const next=actions.children[0].onclick();
 assert.deepEqual(sent[1],{action:0,turn:2});
 completeMove();await next;
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
        subprocess.run(['node'], input=harness + script + assertions, text=True,
                       check=True, capture_output=True, timeout=10)
