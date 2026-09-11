"""Interactive Splendor parity, human validation and browser lifecycle regressions."""
import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from meeple_bots import Match, MctsAgent, RandomAgent, Splendor, SplendorSession
from meeple_bots.games.splendor.gui import SplendorApplication
from meeple_bots.cli import build_parser
from meeple_bots.serialization import match_result_dict


def wait_for(app, predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = app.snapshot()
        if state['status'] == 'error':
            raise AssertionError(state['message'])
        if predicate(state):
            return state
        time.sleep(.005)
    raise AssertionError('GUI did not reach expected state')


class SplendorGuiTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node is required for browser interaction tests")
    def test_board_selection_and_required_returns(self):
        from meeple_bots.games.splendor.gui.page import PAGE
        script = PAGE.split("<script>", 1)[1].split("</script>", 1)[0]
        script = script.replace("}poll();", "}")
        dom = r"""
const assert = require('node:assert/strict');
class Element {
 constructor(){this.children=[];this.value='';this.style={};this.dataset={};}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this.children=nodes;}
 setAttribute(){}
 scrollIntoView(){}
}
const elements=new Map();
const document={getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},createElement(){return new Element();}};
const matchMedia=()=>({matches:false});
"""
        assertions = r"""
state={status:'waiting_human',active_player:0,legal_actions:[],cards:[{bonus:0,points:1,cost:[1,0,0,0,0]}],market:[[0]],holdings:[{reserved:[0]}],noble_data:[{requirements:[4,4,0,0,0]}]};
const play=(kind,extra={})=>({kind,payment:zeros(),returned:zeros(),noble:null,...extra});
state.legal_actions=[play('take_same',{color:0}),play('take_different',{colors:7})];
selection={type:'gems',tokens:[1,0,0,0,0,0]};
assert.deepEqual(nextTokens(0),[2,0,0,0,0,0]);
selection.tokens=nextTokens(0);
assert.equal(nextTokens(1),null);
assert.deepEqual(nextTokens(0),zeros());
selection={type:'gems',tokens:[1,1,0,0,0,0]};resetDetails();drawDecision();
assert.equal(choice,null);assert.equal($('confirm').disabled,true);
selection.tokens=nextTokens(2);drawDecision();assert.equal(choice,1);
// The selected gem action cannot be confirmed before the exact native return is chosen.
state.legal_actions=[play('take_different',{colors:7,returned:[1,0,0,0,0,0]})];
resetDetails();drawDecision();assert.equal(choice,null);
selection.returned=[1,0,0,0,0,0];drawDecision();assert.equal(choice,0);
// Card decisions preserve payment and mandatory noble choices.
state.legal_actions=[play('buy_visible',{tier:0,slot:0,payment:[1,0,0,0,0,0],noble:0}),play('reserve_visible',{tier:0,slot:0})];
selection={type:'card',tier:0,slot:0,mode:'buy_visible'};resetDetails();drawDecision();
assert.equal(choice,0);assert.equal(selection.noble,0);
selection.mode='reserve_visible';resetDetails();drawDecision();assert.equal(choice,1);
assert.equal(targetActions({reserved:true,player:1,index:0}).length,0);
state.status='playing';drawDecision();assert.equal(choice,null);assert.equal($('confirm').disabled,true);
"""
        result = subprocess.run(["node"], input=dom + script + assertions,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_session_matches_catalog_including_chance_and_mcts_reuse(self):
        for agent in (RandomAgent(), MctsAgent(iterations=8, rollout_depth=20,
                                               tree_reuse=True, transpositions=True)):
            with self.subTest(agent=type(agent).__name__):
                session = SplendorSession(seed=42, first=agent, second=RandomAgent())
                for _ in range(3000):
                    state = session.snapshot()
                    if state['finished']:
                        break
                    session.step()
                self.assertTrue(state['finished'])
                report = json.loads(json.dumps(match_result_dict(Match(Splendor(), agent, RandomAgent(), seed=42).run())))
                moves = [e for e in state['events'] if e['player'] is not None]
                self.assertEqual([(e['player'], e['action']) for e in moves],
                                 [(m['player'], m['action']) for m in report['moves']])
                self.assertEqual([e['action'] for e in state['events'] if e['player'] is None],
                                 [e['outcome'] for e in report['chance_events']])
                self.assertEqual(state['utilities'], report['utilities'])
                self.assertEqual(state['holdings'], report['splendor_state']['players'])

    def test_human_decisions_cannot_choose_chance_or_illegal_actions(self):
        session = SplendorSession(seed=42)
        initial = session.snapshot()
        for bad in (-1, True, 100000):
            with self.assertRaises(ValueError):
                session.step(bad)
            self.assertEqual(session.snapshot(), initial)
        reserve = next(i for i, a in enumerate(initial['legal_actions'])
                       if a['kind'] == 'reserve_visible')
        state = session.step(reserve)
        self.assertEqual(state['phase'], 'chance')
        self.assertEqual(state['legal_actions'], [])
        with self.assertRaises(ValueError):
            session.step(0)
        self.assertEqual(session.snapshot(), state)
        state = session.step()
        self.assertTrue(state['waiting_human'])
        self.assertEqual(state['events'][-1]['action']['kind'], 'refill')
        self.assertEqual(len(state['cards']), 90)
        self.assertEqual(len(state['noble_data']), 10)

    def test_restart_rejects_stale_click_and_invalid_start_preserves_match(self):
        app = SplendorApplication()
        payload = dict(first={'kind': 'human'}, second={'kind': 'human'},
                       seed=42, minimum_move_seconds=0)
        try:
            app.start(payload)
            old = wait_for(app, lambda s: s['status'] == 'waiting_human')
            with self.assertRaises(ValueError):
                app.start({**payload, 'seed': -1})
            self.assertEqual(app.snapshot(), old)
            app.start(payload)
            new = wait_for(app, lambda s: s['status'] == 'waiting_human')
            with self.assertRaises(ValueError):
                app.move({'action': 0, 'turn': 0, 'session_id': old['session_id']})
            self.assertEqual(app.snapshot(), new)
            app.move({'action': 0, 'turn': 0, 'session_id': new['session_id']})
            current = wait_for(app, lambda s: len(s['events']) == 1 and s['status'] == 'waiting_human')
            self.assertEqual(current['active_player'], 1)
            with self.assertRaises(ValueError):
                app.move({'action': 0, 'turn': 0, 'session_id': new['session_id']})
        finally:
            app.cancel()
            app._game._thread.join(2)
            self.assertFalse(app._game._thread.is_alive())

    def test_automated_gui_finishes_and_saves_explicit_events(self):
        with TemporaryDirectory() as tmp:
            app = SplendorApplication(Path(tmp))
            try:
                app.start(dict(first={'kind': 'random'}, second={'kind': 'random'},
                               seed=42, minimum_move_seconds=0, save_trace=True))
                state = wait_for(app, lambda s: s['trace_path'] is not None)
                self.assertEqual(state['status'], 'finished')
                saved = json.loads(Path(state['trace_path']).read_text())
                self.assertEqual(saved['format'], 'splendor_session_v1')
                self.assertEqual(saved['events'], state['events'])
                replay = Splendor().initial_state(saved['seed'])._position
                for event in saved['events']:
                    replay = replay.apply(event['action'])
                self.assertTrue(replay.snapshot()['finished'])
                self.assertEqual(replay.snapshot()['players'], saved['holdings'])
            finally:
                app.cancel()
        self.assertEqual(build_parser().parse_args(['gui', '--game', 'splendor']).game, 'splendor')
