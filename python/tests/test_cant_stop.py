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
        for kwargs in ({'tree_reuse': True}, {'transpositions': True}, {'heuristic': 1}):
            with self.assertRaises(ValueError):
                CantStopSession(first=MctsAgent(**kwargs))

    def test_gui_human_moves_restart_and_invalid_start_are_atomic(self):
        app = CantStopApplication()
        try:
            app.start({'first': {'kind': 'human'}, 'second': {'kind': 'human'}, 'minimum_move_seconds': 0})
            state = self.wait_for(app, lambda s: s['status'] == 'waiting_human')
            previous = deepcopy(state)
            with self.assertRaises(ValueError):
                app.start({'first': {'kind': 'mcts', 'tree_reuse': True}, 'second': {'kind': 'random'}})
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
