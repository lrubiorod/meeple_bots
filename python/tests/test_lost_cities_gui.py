"""Open-hand debug GUI: legal Rust transitions and isolated controller lifetimes."""
import json
from random import Random
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from meeple_bots.cli import build_parser
from meeple_bots.gui.server import run_gui
from meeple_bots.games.lost_cities.gui import LostCitiesApplication, PAGE
from meeple_bots.games.lost_cities.gui.player import parse_player
from meeple_bots import LostCities, LostCitiesAction


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
                        {'save_trace': 1}, {'save_trace': 'yes'},
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

    def test_chance_and_agent_rng_streams_keep_distinct_seed_namespaces(self):
        seeds = []

        def make_rng(seed):
            seeds.append(seed)
            return Random(seed)

        with patch('meeple_bots.games.lost_cities.gui.controller.Random', side_effect=make_rng):
            self.start('human', 'human')
            self.wait(lambda s: s['status'] == 'waiting_human')
        self.assertEqual(seeds, [
            42 ^ 0x8EBC6AF09C88C6E3,
            42 ^ 0xA0761D6478BD642F,
            42 ^ 0xE7037ED1A0B428DB,
        ])
        self.assertEqual(len(set(seeds)), 3)

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

    def test_so_players_finish_reproducibly_with_independent_settings(self):
        for first, second in [('so_ismcts', 'random'), ('random', 'so_ismcts'), ('so_ismcts', 'so_ismcts')]:
            payload = {'first': {'kind': first}, 'second': {'kind': second}, 'seed': 42, 'minimum_move_seconds': 0}
            for seat, iterations in [('first', 2), ('second', 3)]:
                if payload[seat]['kind'] == 'so_ismcts':
                    payload[seat].update(iterations=iterations, exploration=.75)
            self.app.start(payload)
            state = self.wait(lambda s: s['status'] == 'finished')
            self.assertEqual(len(state['deck']), 0)
            self.assertEqual(sum(e['chance'] for e in state['events']), 44)
            for e in state['events']:
                seat = 'first' if e['player'] == 0 else 'second'
                if not e['chance'] and payload[seat]['kind'] == 'so_ismcts':
                    self.assertEqual(e['search']['completed_iterations'], payload[seat]['iterations'])
                else:
                    self.assertNotIn('search', e)
            self.app.start(payload)
            repeated = self.wait(lambda s: s['status'] == 'finished')
            self.assertEqual(state['events'], repeated['events'])

    def test_so_time_budget_reaches_native_observation_search(self):
        from meeple_bots import LostCitiesObservation, SoIsmctsAgent

        original_search = SoIsmctsAgent.search
        calls = []

        def record_search(agent, observation, legal_actions, *, seed):
            calls.append((agent.iterations, agent.time_budget, observation, legal_actions))
            return original_search(agent, observation, legal_actions, seed=seed)

        with patch.object(SoIsmctsAgent, 'search', record_search):
            self.app.start({'first': {'kind': 'so_ismcts', 'time_budget': .001},
                            'second': {'kind': 'human'}, 'seed': 42,
                            'minimum_move_seconds': 0})
            state = self.wait(lambda s: s['status'] == 'waiting_human')

        self.assertEqual(state['players'][0]['iterations'], None)
        self.assertEqual(state['players'][0]['time_budget'], .001)
        self.assertGreaterEqual(len(calls), 2)  # Play and draw are separate decisions.
        for iterations, seconds, observation, legal in calls:
            self.assertIsNone(iterations)
            self.assertEqual(seconds, .001)
            self.assertIsInstance(observation, LostCitiesObservation)
            self.assertFalse(hasattr(observation, 'deck'))
            self.assertFalse(hasattr(observation, 'hands'))
            self.assertTrue(legal)
        self.assertTrue(all(e['search']['completed_iterations'] >= 1
                            for e in state['events'] if 'search' in e))

    def test_save_completed_match_with_private_chance_events(self):
        with TemporaryDirectory() as tmp:
            app = LostCitiesApplication(Path(tmp))
            self.addCleanup(app.cancel)
            app.start({'first': {'kind': 'so_ismcts', 'time_budget': .001},
                       'second': {'kind': 'random'}, 'seed': 42,
                       'minimum_move_seconds': 0, 'save_trace': True})
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                state = app.snapshot()
                if state['status'] == 'error':
                    self.fail(state['message'])
                if state['trace_path']:
                    break
                time.sleep(.005)
            else:
                self.fail('completed match was not saved')
            self.assertEqual(state['status'], 'finished')
            self.assertIsNone(state['trace_error'])
            saved = json.loads(Path(state['trace_path']).read_text())
            self.assertEqual(saved['format'], 'lost_cities_session_v1')
            self.assertEqual(saved['seed'], 42)
            self.assertEqual(saved['players'][0]['time_budget'], .001)
            self.assertEqual(saved['events'], state['events'])
            self.assertEqual(sum(e['chance'] for e in saved['events']), 44)
            self.assertTrue(any('search' in e for e in saved['events']))
            game = LostCities()
            position = game.initial_state(saved['seed'])
            for event in saved['events']:
                action = LostCitiesAction.from_dict(event['action'])
                position = (game.apply_chance_outcome(position, action) if event['chance']
                            else game.apply_action(position, action))
            self.assertEqual(position.status, 'terminal')
            actual = json.loads(json.dumps(position.to_dict()))
            for key in ('hands', 'deck', 'expeditions', 'discards', 'scores', 'phase'):
                self.assertEqual(saved[key], actual[key])

    def test_cancelled_and_unsaved_matches_do_not_write_files(self):
        with TemporaryDirectory() as tmp:
            app = LostCitiesApplication(Path(tmp))
            self.addCleanup(app.cancel)
            app.start({'first': {'kind': 'human'}, 'second': {'kind': 'human'},
                       'seed': 42, 'minimum_move_seconds': 0, 'save_trace': True})
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and app.snapshot()['status'] != 'waiting_human':
                time.sleep(.005)
            self.assertEqual(app.snapshot()['status'], 'waiting_human')
            app.start({'first': {'kind': 'random'}, 'second': {'kind': 'random'},
                       'seed': 42, 'minimum_move_seconds': 0})
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and app.snapshot()['status'] != 'finished':
                time.sleep(.005)
            self.assertEqual(app.snapshot()['status'], 'finished')
            self.assertIsNone(app.snapshot()['trace_path'])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_so_default_and_iteration_payloads_keep_previous_budget(self):
        self.assertEqual(parse_player({'kind': 'so_ismcts'}).as_dict(),
                         {'kind': 'so_ismcts', 'iterations': 1000,
                          'exploration': 2 ** .5})
        self.assertEqual(parse_player({'kind': 'so_ismcts', 'iterations': 17}).as_dict()['iterations'], 17)
        self.assertEqual(parse_player({'kind': 'so_ismcts', 'time_budget': .25}).as_dict(),
                         {'kind': 'so_ismcts', 'iterations': None,
                          'exploration': 2 ** .5, 'time_budget': .25})

    def test_search_boundary_and_restart_while_thinking(self):
        from threading import Event
        from meeple_bots import SoIsmctsAgent, LostCitiesObservation
        entered, release = Event(), Event()
        calls = []
        def blocked_search(agent, observation, legal_actions, *, seed):
            calls.append((observation, legal_actions, seed))
            entered.set()
            release.wait(3)
            return {'action': legal_actions[0], 'diagnostics': {'completed_iterations': agent.iterations}}
        with patch.object(SoIsmctsAgent, 'search', blocked_search):
            try:
                self.app.start({'first': {'kind': 'so_ismcts', 'iterations': 2}, 'second': {'kind': 'human'}, 'seed': 42, 'minimum_move_seconds': 0})
                self.assertTrue(entered.wait(2))
                state = self.app.snapshot()  # Must not wait for the search lock.
                observation, legal, seed = calls[0]
                self.assertIsInstance(observation, LostCitiesObservation)
                self.assertEqual(observation.observer, 0)
                self.assertEqual(observation.hand, state['hands'][0])
                self.assertFalse(hasattr(observation, 'deck'))
                self.assertFalse(hasattr(observation, 'hands'))
                self.assertFalse(hasattr(observation, '_position'))
                old_worker = self.app._game._thread
                self.start('human', 'human')
                current = self.wait(lambda s: s['status'] == 'waiting_human')
                self.assertNotEqual(state['session_id'], current['session_id'])
            finally:
                release.set()
            old_worker.join(2)
            self.assertFalse(old_worker.is_alive())
            self.assertEqual(self.app.snapshot()['events'], [])

    def test_human_so_both_seats_and_rejected_search_settings(self):
        for first, second, human in [('human', 'so_ismcts', 0), ('so_ismcts', 'human', 1)]:
            payload = {'first': {'kind': first}, 'second': {'kind': second}, 'seed': 42, 'minimum_move_seconds': 0}
            for p in (payload['first'], payload['second']):
                if p['kind'] == 'so_ismcts': p['iterations'] = 2
            self.app.start(payload)
            state = self.wait(lambda s: s['status'] == 'waiting_human')
            self.assertEqual(state['current_player'], human)
            self.move(state, 'discard')
            draw = self.wait(lambda s: s['status'] == 'waiting_human' and s['phase'] == 'draw')
            self.move(draw, 'draw_deck')
            self.wait(lambda s: s['status'] == 'waiting_human' and s['phase'] == 'play')
        before = self.app.snapshot()['session_id']
        for fields in ({'iterations': 0}, {'iterations': True}, {'exploration': float('nan')},
                       {'exploration': -1}, {'tree_reuse': True}, {'time_budget': 0},
                       {'time_budget': -1}, {'time_budget': True}, {'time_budget': 'bad'},
                       {'time_budget': float('nan')}, {'iterations': 2, 'time_budget': .01},
                       {'progressive_widening': True}):
            with self.assertRaises((ValueError, TypeError)):
                self.app.start({'first': {'kind': 'so_ismcts', **fields}})
            self.assertEqual(self.app.snapshot()['session_id'], before)

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
 $('first').value='so_ismcts';$('first').onchange();
 $('first-iterations').value='17';$('first-exploration').value='0.8';
 $('second').value='so_ismcts';$('second').onchange();
 $('second-iterations').value='29';$('second-exploration').value='1.2';
 assert.deepEqual(playerConfig('first'),{kind:'so_ismcts',iterations:17,exploration:0.8});
 assert.deepEqual(playerConfig('second'),{kind:'so_ismcts',iterations:29,exploration:1.2});
 assert.equal($('search-settings').children[0].hidden,false);
 $('first').value='human';$('first').onchange();
 assert.deepEqual(playerConfig('first'),{kind:'human'});
 assert.equal($('search-settings').children[0].hidden,true);

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
 $('first').value='so_ismcts';$('first').onchange();
 $('first-budget-mode').value='time';$('first-budget-mode').onchange();
 $('first-time-budget').value='0.125';$('seed').value='42';$('delay').value='0';$('save-trace').checked=true;
 assert.equal($('first-iterations-label').hidden,true);
 assert.equal($('first-time-label').hidden,false);
 assert.deepEqual(playerConfig('first'),{kind:'so_ismcts',iterations:null,time_budget:0.125,exploration:0.8});
 assert.throws(()=>{$('first-time-budget').value='';playerConfig('first');},/positive number of seconds/);
 $('first-time-budget').value='0.125';
 await $('start').onclick();
 assert.deepEqual(requests[1],{first:{kind:'so_ismcts',iterations:null,time_budget:0.125,exploration:0.8},second:{kind:'so_ismcts',iterations:29,exploration:1.2},seed:42,minimum_move_seconds:0,save_trace:true});
 render({...s,trace_path:'results/gui/lost_cities/match.json'});
 assert.match($('trace').textContent,/Saved match: results\/gui\/lost_cities\/match.json/);
 $('first-budget-mode').value='iterations';$('first-budget-mode').onchange();
 assert.equal($('first-iterations-label').hidden,false);
 assert.deepEqual(playerConfig('first'),{kind:'so_ismcts',iterations:17,exploration:0.8});
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
        subprocess.run(['node'], input=harness + script + assertions, text=True,
                       check=True, capture_output=True, timeout=10)
