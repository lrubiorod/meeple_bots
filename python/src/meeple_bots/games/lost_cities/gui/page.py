"""Dependency-free inspection board with neutral, code-rendered cards."""
PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lost Cities · Debug table</title>
<style>
:root{color-scheme:dark;font:15px system-ui;background:#101c24;color:#e5edf1}*{box-sizing:border-box}body{max-width:1300px;margin:auto;padding:24px}h1{margin:0;font-size:28px}h2{font-size:18px}small,.muted{color:#a8bac5}header{display:flex;justify-content:space-between;gap:20px;align-items:center;margin-bottom:22px}.badge{border:1px solid #588293;border-radius:20px;padding:7px 12px}section{background:#192b36;border:1px solid #304752;border-radius:12px;padding:18px;margin:16px 0}.controls{display:flex;flex-wrap:wrap;gap:16px;align-items:end}label{display:grid;gap:6px}select,input,button{font:inherit;padding:9px;border-radius:6px;border:1px solid #526a77;background:#233d4c;color:inherit}input{width:130px}button{cursor:pointer}button:hover{background:#365b6f}button:disabled{opacity:.5;cursor:default}#start{background:#c1dfbb;color:#142218;font-weight:700}.columns{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}.column{border-top:4px solid var(--c);background:#10212c;padding:12px;border-radius:6px;min-height:92px}.cards{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}.card{display:inline-flex;align-items:center;justify-content:center;min-width:40px;height:52px;padding:5px;background:#edf1e9;color:#17252c;border:7px solid var(--c);border-radius:6px;font-weight:bold}.hand .card{height:76px;min-width:58px;border-width:10px}.hand button.card:hover{background:#fff;transform:translateY(-3px)}.hand button.card[aria-pressed="true"]{outline:3px solid #fff;outline-offset:3px;transform:translateY(-3px)}.card:focus-visible{outline:3px solid #fff;outline-offset:3px}.r{--c:#ed625d}.g{--c:#45b875}.b{--c:#4e98ee}.y{--c:#f0c844}.w{--c:#dce3e7}.empty{color:#8299a8;font-size:13px}.player-head{display:flex;justify-content:space-between;align-items:center}.active{outline:2px solid #b8d8b4}#status{font-weight:600}#error{color:#ffb4ac;white-space:pre-wrap}#actions{display:flex;flex-wrap:wrap;gap:8px}#log{max-height:260px;overflow:auto;font:13px monospace;line-height:1.7}summary{cursor:pointer}details .cards{margin:8px 0} @media(max-width:650px){body{padding:12px}.columns{gap:4px}.column{padding:5px}.card{min-width:28px;height:38px}header{align-items:start}.badge{font-size:12px}}
</style>
<header><div><h1>Lost Cities</h1><small>Single-round experimental variant · 5 expeditions · 60 cards</small></div><span class="badge">Open-hand debug view</span></header>
<div class="controls"><label>Player 1<select id="first"><option value="human">Human</option><option value="random">Random</option></select></label><label>Player 2<select id="second"><option value="random">Random</option><option value="human">Human</option></select></label><label>Seed<input id="seed" type="number" min="0" value="42"></label><label>Step delay (s)<input id="delay" type="number" min="0" max="10" step="0.1" value="0.4"></label><button id="start">New match</button></div>
<p class="muted">Both hands and private draws are visible for inspection. Random players choose only from legal actions.</p>
<p id="status" role="status">Start a match.</p><p id="error" role="alert"></p>
<section id="p1"></section><section><div class="player-head"><h2>Shared discard piles</h2><strong id="deck"></strong></div><div id="discards" class="columns"></div><details><summary>Inspect remaining deck pool (unordered)</summary><div id="pool" class="cards"></div></details></section><section id="p0"></section>
<section><h2 id="decision">Legal actions</h2><div id="actions"></div></section><section><h2>Transition log</h2><div id="log"></div></section>
<script>
const $=id=>document.getElementById(id), colors=['Red','Green','Blue','Yellow','White'], classes=['r','g','b','y','w'];
let actionKey='', busy=false, polling=false, revision=0, selectedCard=null;
function card(c){return `<span class="card ${classes[c[0]]}" title="${colors[c[0]]} ${c[1]||'Wager'}">${c[1]||'W'}</span>`;}
function cards(cs){return cs.length?cs.map(card).join(''):'<span class="empty">Empty</span>';}
function hand(cs,enabled){return enabled?cs.map((c,i)=>`<button class="card ${classes[c[0]]}" data-hand-index="${i}" aria-label="${colors[c[0]]} ${c[1]||'Wager'}" aria-pressed="false">${c[1]||'W'}</button>`).join(''):cards(cs);}
function sameCard(a,b){return a&&b&&a[0]===b[0]&&a[1]===b[1];}
function showActions(s){
 $('actions').replaceChildren();
 if(s.status!=='waiting_human')return;
 if(s.phase==='play'&&!selectedCard){$('decision').textContent=`Player ${s.current_player+1}: select a card from your hand`;return;}
 $('decision').textContent=s.phase==='play'?`${colors[selectedCard[0]]} ${selectedCard[1]||'wager'} — choose an action`:`Player ${s.current_player+1}: draw one card`;
 s.legal_actions.forEach((a,i)=>{
  if(s.phase==='play'&&!sameCard(a.card,selectedCard))return;
  const b=document.createElement('button');b.textContent=s.phase==='play'?(a.kind==='play'?'Play expedition':'Discard'):describe(a);
  b.onclick=()=>submitAction(s,i);$('actions').append(b);
 });
}
function selectCard(s,c){
 if(busy||s.status!=='waiting_human'||s.phase!=='play')return;
 selectedCard=c;
 for(const b of $('p'+s.current_player).querySelectorAll('[data-hand-index]'))b.setAttribute('aria-pressed',String(sameCard(s.hands[s.current_player][Number(b.dataset.handIndex)],c)));
 showActions(s);
}
async function submitAction(s,i){
 if(busy)return;busy=true;revision++;
 for(const el of $('actions').children)el.disabled=true;
 try{render(await post('/api/move',{action:i,turn:s.turn,session_id:s.session_id}));$('error').textContent='';}
 catch(e){$('error').textContent=e.message;actionKey='';}
 finally{busy=false;await poll();}
}
function describe(a){const name=a.card?`${colors[a.card[0]]} ${a.card[1]||'wager'}`:'';return a.kind==='play'?`Play ${name}`:a.kind==='discard'?`Discard ${name}`:a.kind==='draw_deck'?'Draw from deck':a.kind==='draw_discard'?`Take ${colors[a.color]} discard`:`Deck draw: ${name}`;}
async function post(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.error||'Request failed');return data;}
function render(s){
 $('status').textContent=s.status==='idle'?s.message:s.status==='finished'||s.status==='error'?s.message:`Player ${s.current_player+1} · ${s.phase==='draw_chance'?'Chance draw':s.phase.toUpperCase()} · ${s.status==='waiting_human'?'Choose an action':'Running'}`;
 if(!s.hands)return;
 const key=JSON.stringify([s.session_id,s.turn,s.status]);
 if(key===actionKey)return;
 actionKey=key;selectedCard=null;
 for(let p=0;p<2;p++){$('p'+p).className=s.current_player===p&&s.status!=='finished'?'active':'';$('p'+p).innerHTML=`<div class="player-head"><h2>Player ${p+1} · ${s.players[p].kind}</h2><strong>Score ${s.scores[p]}</strong></div><div class="columns">${s.expeditions[p].map((cs,c)=>`<div class="column ${classes[c]}"><strong>${colors[c]}</strong><div class="cards">${cards(cs)}</div></div>`).join('')}</div><p class="muted">Hand · ${s.hands[p].length} cards</p><div class="cards hand">${hand(s.hands[p],s.status==='waiting_human'&&s.phase==='play'&&s.current_player===p)}</div>`;}
 $('deck').textContent=`Deck: ${s.deck.length} cards`;$('pool').innerHTML=cards(s.deck);
 $('discards').innerHTML=s.discards.map((cs,c)=>`<div class="column ${classes[c]}"><strong>${colors[c]}</strong><div class="cards">${cards(cs)}</div><small>${cs.length?'Top: '+(cs[cs.length-1][1]||'Wager'):''}${s.blocked_discard===c?' · Cannot draw this turn':''}</small></div>`).join('');
 $('decision').textContent='Legal actions';
 for(let p=0;p<2;p++)for(const b of $('p'+p).querySelectorAll('[data-hand-index]'))b.onclick=()=>selectCard(s,s.hands[p][Number(b.dataset.handIndex)]);
 showActions(s);
 $('log').textContent=s.events.map((e,i)=>`${i+1}. P${e.player+1}${e.chance?' [chance]':''}: ${describe(e.action)}`).join('\n');$('log').style.whiteSpace='pre-wrap';
}
async function poll(){if(polling||busy)return;polling=true;const version=revision;try{const r=await fetch('/api/state');const state=await r.json();if(version===revision&&!busy)render(state);}catch(e){$('error').textContent=e.message;}finally{polling=false;}}
$('start').onclick=async()=>{if(busy)return;busy=true;revision++;$('start').disabled=true;try{render(await post('/api/start',{first:{kind:$('first').value},second:{kind:$('second').value},seed:Number($('seed').value),minimum_move_seconds:Number($('delay').value)}));$('error').textContent='';}catch(e){$('error').textContent=e.message;}finally{busy=false;$('start').disabled=false;}};
setInterval(poll,250);poll();
</script></html>'''
