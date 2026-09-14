"""Interactive Connect6 uses native turns and the shared GUI lifecycle."""
import json
from pathlib import Path
import tempfile
import time
import unittest
import shutil
import subprocess

from meeple_bots.games.connect6.gui import Connect6Application
from meeple_bots.extraction import extract_tournament
from meeple_bots.gui.baselines import CONNECT6_BASELINE


class Connect6GuiTests(unittest.TestCase):
    def wait(self, app, predicate):
        deadline = time.monotonic()+5
        while time.monotonic()<deadline:
            state=app.snapshot()
            if state['status']=='error': self.fail(state['message'])
            if predicate(state): return state
            time.sleep(.005)
        self.fail(str(app.snapshot()))

    def start(self, app, **kwargs):
        app.start(dict(first={'kind':'human'}, second={'kind':'human'}, board_size=6,
                       minimum_move_seconds=0, **kwargs))
        return self.wait(app,lambda s:s['status']=='waiting_human')

    def move(self, app, state, position):
        return app.move(dict(position=position,turn=len(state['moves']),session_id=state['session_id']))

    def test_opening_two_stones_stale_clicks_and_restart(self):
        app=Connect6Application()
        self.addCleanup(app.cancel)
        self.assertEqual(app.snapshot()['board_size'],19)
        self.assertEqual(app.snapshot()['players'][1],CONNECT6_BASELINE.as_dict())
        state=self.start(app)
        for position, player, remaining in ((0,1,2),(1,1,1),(2,0,2)):
            previous=state
            self.move(app,state,position)
            state=self.wait(app,lambda s:s['status']=='waiting_human' and len(s['moves'])==position+1)
            self.assertEqual((state['active_player'],state['placements_remaining']),(player,remaining))
            with self.assertRaises(ValueError): self.move(app,previous,position+10)
        with self.assertRaises(ValueError): self.move(app,state,0)
        with self.assertRaises(ValueError): app.start({'board_size':5})
        self.assertEqual(app.snapshot()['session_id'],state['session_id'])
        self.start(app)
        with self.assertRaises(ValueError): self.move(app,state,3)

    def test_win_on_first_stone_finishes_without_second_prompt(self):
        app=Connect6Application()
        self.addCleanup(app.cancel)
        state=self.start(app)
        sequence=(0,30,31,1,2,32,33,3,4,34,24,5)
        for ply, position in enumerate(sequence,1):
            self.move(app,state,position)
            expected='finished' if ply==len(sequence) else 'waiting_human'
            state=self.wait(app,lambda s:s['status']==expected and len(s['moves'])==ply)
        self.assertEqual(state['winner'],0)
        self.assertEqual(state['placements_remaining'],1)
        self.assertEqual(state['legal_actions'],[])

    @unittest.skipUnless(shutil.which('node'), 'Node is needed to exercise browser JavaScript')
    def test_browser_keeps_cells_and_sends_current_decision(self):
        from meeple_bots.games.connect6.gui import PAGE
        script = PAGE.split('<script>')[1].split('</script>')[0]
        harness = r"""
const elements=new Map();
class Element {
 constructor(){this.children=[];this.value='human';this.checked=false;this.classList={toggle(){},contains(){return false;}};}
 addEventListener(){} setAttribute(){} appendChild(e){this.children.push(e);} replaceChildren(){this.children=[];}
 style={}; dataset={};
}
global.document={querySelector(k){if(!elements.has(k))elements.set(k,new Element());return elements.get(k);},createElement(){return new Element();}};
global.window={setTimeout(){}};
global.fetch=async()=>({ok:true,json:async()=>({})});
"""
        # Disable the initial asynchronous poll; exercise rendering and submission directly.
        script = script.rsplit('    poll();', 1)[0]
        checks = r"""
state={board_size:6,board:Array(36).fill(null),status:'waiting_human',active_player:1,
 placements_remaining:2,legal_actions:[0,1],moves:[],session_id:'session'};
render(); const first=board.children[0]; render();
if(board.children.length!==36 || first!==board.children[0])throw Error('poll replaced buttons');
let sent; global.fetch=async(path,options)=>{sent=JSON.parse(options.body);return {ok:true,json:async()=>({...state,status:'playing'})};};
(async()=>{await play(1);if(sent.position!==1 || sent.turn!==0 || sent.session_id!=='session')throw Error('stale decision payload');
document.querySelector('#player-0').value='mcts';document.querySelector('#selection-policy-0').value='ucb1_tuned';
document.querySelector('#exploration-0').value='0.5';document.querySelector('#transpositions-0').checked=true;
const config=playerConfig(0);if(config.selection_policy!=='ucb1_tuned'||config.exploration!==0.5||!config.transpositions)throw Error('missing MCTS settings');})();
"""
        result=subprocess.run([shutil.which('node'), '-e', harness+script+checks], capture_output=True, text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_automated_modes_trace_and_extraction(self):
        for selector in ('uct','ucb1_tuned'):
            with self.subTest(selector=selector), tempfile.TemporaryDirectory() as tmp:
                app=Connect6Application(Path(tmp))
                try:
                    app.start(dict(board_size=6, seed=42,minimum_move_seconds=0,save_trace=True,
                                   first=dict(kind='mcts',iterations=8,rollout_depth=36,
                                              selection_policy=selector,transpositions=True,tree_reuse=True),
                                   second=dict(kind='random')))
                    state=self.wait(app,lambda s:s['status']=='finished')
                    self.assertIsNone(state['trace_error'])
                    path=Path(state['trace_path'])
                    header=json.loads(path.read_text().splitlines()[0])
                    self.assertEqual(header['game_params'],{'board_size':6})
                    self.assertTrue(extract_tournament(path,Path(tmp)/'data')['complete'])
                finally: app.cancel()


if __name__=='__main__': unittest.main()
