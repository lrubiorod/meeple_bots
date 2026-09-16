"""Open-hand debug GUI: legal Rust transitions and isolated controller lifetimes."""
import json
import shutil
import subprocess
import time
import unittest
from unittest.mock import patch

from meeple_bots.cli import build_parser
from meeple_bots.gui.server import run_gui
from meeple_bots.games.lost_cities.gui import LostCitiesApplication, PAGE


class LostCitiesGuiTests(unittest.TestCase):
    def setUp(self):
        self.app = LostCitiesApplication()
        self.addCleanup(self.app.cancel)

    def wait(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = self.app.snapshot()
            if state['status'] == 'error':
                self.fail(state['message'])
            if predicate(state):
                return state
            time.sleep(.005)
        self.fail('GUI did not reach expected state')

    def start(self, first='human', second='random', **kw):
        return self.app.start({'first': {'kind': first}, 'second': {'kind': second},
                               'seed': 42, 'minimum_move_seconds': 0, **kw})

    def move(self, state, kind):
        index = next(i for i, a in enumerate(state['legal_actions']) if a['kind'] == kind)
        payload = {'action': index, 'turn': state['turn'], 'session_id': state['session_id']}
        self.app.move(payload)
        return payload

    def test_both_hands_and_admin_pool_visible_without_reveal(self):
        self.start('human', 'human')
        state = self.wait(lambda s: s['status'] == 'waiting_human')
        self.assertEqual([len(h) for h in state['hands']], [8, 8])
        self.assertEqual(len(state['deck']), 44)
        self.assertEqual(state['phase'], 'play')
        json.dumps(state)  # Never exports an executable native position.
        self.assertNotIn('_position', state)
        card = state['legal_actions'][0].get('card')
        self.assertIsNotNone(card)
        state['hands'] = ()
        self.assertEqual(len(self.app.snapshot()['hands'][0]), 8)

    def test_microturn_chance_and_stale_actions(self):
        self.start('human', 'human')
        state = self.wait(lambda s: s['status'] == 'waiting_human')
        old = self.move(state, 'discard')
        draw = self.wait(lambda s: s['status'] == 'waiting_human' and s['turn'] > state['turn'])
        self.assertEqual(draw['phase'], 'draw')
        self.assertEqual(draw['current_player'], 0)
        self.assertEqual(len(draw['hands'][0]), 7)
        self.assertFalse(any(a['kind'] == 'draw_discard' and a['color'] == draw['blocked_discard'] for a in draw['legal_actions']))
        with self.assertRaises(ValueError):
            self.app.move(old)
        self.move(draw, 'draw_deck')
        next_turn = self.wait(lambda s: s['status'] == 'waiting_human' and s['current_player'] == 1)
        self.assertEqual(next_turn['phase'], 'play')
        self.assertEqual(len(next_turn['deck']), 43)
        self.assertEqual([len(h) for h in next_turn['hands']], [8, 8])
        self.assertTrue(next_turn['events'][-1]['chance'])
        self.assertEqual(next_turn['events'][-1]['action']['kind'], 'deal_card')

    def test_invalid_inputs_preserve_current_match_and_restart_cancels_waiter(self):
        self.start('human', 'human')
        before = self.wait(lambda s: s['status'] == 'waiting_human')
        for payload in ({'first': {'kind': 'mcts'}}, {'seed': -1}, {'seed': True},
                        {'minimum_move_seconds': float('nan')}, {'minimum_move_seconds': -1}):
            with self.assertRaises(ValueError):
                self.app.start(payload)
            self.assertEqual(self.app.snapshot()['session_id'], before['session_id'])
        for action in (-1, 9999, True, None):
            with self.assertRaises(ValueError):
                self.app.move({'action': action, 'turn': before['turn'], 'session_id': before['session_id']})
        old_worker = self.app._game._thread
        self.start('human', 'human')
        after = self.wait(lambda s: s['status'] == 'waiting_human')
        self.assertNotEqual(before['session_id'], after['session_id'])
        old_worker.join(1)
        self.assertFalse(old_worker.is_alive())
        with self.assertRaises(ValueError):
            self.app.move({'action': 0, 'turn': before['turn'], 'session_id': before['session_id']})

    def test_random_players_finish_legally_and_reproducibly(self):
        self.start('random', 'random')
        first = self.wait(lambda s: s['status'] == 'finished')
        self.assertEqual(len(first['deck']), 0)
        self.assertEqual([len(h) for h in first['hands']], [8, 8])
        self.assertEqual(sum(e['chance'] for e in first['events']), 44)
        self.assertEqual(first['legal_actions'], [])
        self.assertEqual(sum(len(h) for h in first['hands']) + sum(len(c) for p in first['expeditions'] for c in p) + sum(len(c) for c in first['discards']), 60)
        self.start('random', 'random')
        second = self.wait(lambda s: s['status'] == 'finished')
        self.assertEqual(first['events'], second['events'])
        self.assertEqual(first['scores'], second['scores'])

    def test_human_random_both_seat_orders(self):
        for first, second, human in [('human', 'random', 0), ('random', 'human', 1)]:
            self.start(first, second)
            s = self.wait(lambda s: s['status'] == 'waiting_human')
            self.assertEqual(s['current_player'], human)
            self.move(s, 'discard')
            s = self.wait(lambda s: s['status'] == 'waiting_human' and s['phase'] == 'draw')
            self.move(s, 'draw_deck')
            s = self.wait(lambda s: s['status'] == 'waiting_human' and s['phase'] == 'play')
            self.assertEqual(s['current_player'], human)
            self.assertTrue(any(e['player'] != human for e in s['events']))

    def test_cli_dispatch_and_browser_syntax(self):
        self.assertEqual(build_parser().parse_args(['gui', '--game', 'lost_cities']).game, 'lost_cities')
        with patch('meeple_bots.gui.server.serve_gui') as serve:
            run_gui(game='lost_cities', open_browser=False)
            self.assertIsInstance(serve.call_args.args[0], LostCitiesApplication)
            serve.call_args.args[0].cancel()
        self.assertIn('Open-hand debug view', PAGE)
        self.assertNotIn('value="mcts"', PAGE)
        if shutil.which('node'):
            subprocess.run(['node', '--check'], input=PAGE.split('<script>')[1].split('</script>')[0],
                           text=True, check=True, capture_output=True)

    @unittest.skipUnless(shutil.which('node'), 'Node is required for browser interaction tests')
    def test_card_selection_filters_legal_actions_and_preserves_server_indices(self):
        script = PAGE.split('<script>')[1].split('</script>')[0]
        harness = r'''
const assert=require('node:assert/strict');
class Element {
 constructor(){this.children=[];this.style={};this.dataset={};this.innerHTML='';this.textContent='';}
 replaceChildren(){this.children=[];}
 append(el){this.children.push(el);}
 querySelectorAll(){return [];}
}
const elements=new Map();
global.document={getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},createElement(){return new Element();}};
global.setInterval=()=>{};
let requests=[];
global.fetch=async(path,options)=>{if(options)requests.push(JSON.parse(options.body));return {ok:true,json:async()=>({status:'idle',message:''})};};
'''
        assertions = r'''
(async()=>{
 await new Promise(setImmediate);
 const s={session_id:'test',turn:3,status:'waiting_human',phase:'play',current_player:0,
 hands:[[[0,2],[1,0],[1,0]],[[2,3]]],players:[{kind:'human'},{kind:'random'}],scores:[0,0],
 expeditions:[[[],[],[],[],[]],[[],[],[],[],[]]],discards:[[],[],[],[],[]],deck:[],events:[],
 legal_actions:[{kind:'play',card:[0,2]},{kind:'discard',card:[0,2]},{kind:'discard',card:[1,0]}]};
 render(s);assert.equal($('actions').children.length,0);
 assert.match($('p0').innerHTML,/data-hand-index/);assert.doesNotMatch($('p1').innerHTML,/data-hand-index/);
 selectCard(s,[0,2]);assert.deepEqual($('actions').children.map(b=>b.textContent),['Play expedition','Discard']);
 const button=$('actions').children[0];render(structuredClone(s));assert.equal($('actions').children[0],button);
 // A wager that cannot be played still offers discard, at its original server index.
 selectCard(s,[1,0]);assert.deepEqual($('actions').children.map(b=>b.textContent),['Discard']);
 await $('actions').children[0].onclick();assert.deepEqual(requests,[{action:2,turn:3,session_id:'test'}]);
 render({...s,turn:4,phase:'draw',legal_actions:[{kind:'draw_deck'},{kind:'draw_discard',color:2}]});
 assert.equal(selectedCard,null);assert.equal($('actions').children.length,2);
 assert.doesNotMatch($('p0').innerHTML,/data-hand-index/);
 render({...s,turn:5,status:'playing',legal_actions:[]});assert.equal($('actions').children.length,0);
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
        subprocess.run(['node'], input=harness + script + assertions, text=True,
                       check=True, capture_output=True, timeout=10)
