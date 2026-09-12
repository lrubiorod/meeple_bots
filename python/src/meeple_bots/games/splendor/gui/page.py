"""Direct board interaction backed exclusively by native legal-action combinations."""

import json

from ....gui.baselines import SPLENDOR_BASELINE

PAGE = r'''<!doctype html>
<html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Splendor · Meeple Bots</title>
<style>
:root{font-family:system-ui,sans-serif;color:#efece0;background:#112b2b;color-scheme:dark}*{box-sizing:border-box}body{margin:auto;max-width:1440px;padding:20px}button,input,select{font:inherit}button{cursor:pointer}button:disabled{cursor:default}button:focus-visible,summary:focus-visible{outline:3px solid #ffdd88;outline-offset:4px}[hidden]{display:none!important}header{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:16px}h1{font:38px Georgia,serif;margin:0}h2{font-size:15px;margin:0 0 12px}h3{font-size:14px;margin:12px 0 8px}.eyebrow{font-size:10px;letter-spacing:2px;color:#bbc3ac;text-transform:uppercase}.muted,small{color:#a9bbb0}small{font-size:11px}.panel,details{background:#183535;border:1px solid #3e5650;border-radius:12px;padding:16px}.row{display:flex;flex-wrap:wrap;align-items:center;gap:8px}summary{cursor:pointer}#settings{margin-bottom:14px}#configs>section{padding:10px;flex:1;min-width:250px}label{display:inline-flex;flex-direction:column;gap:4px;margin:5px;font-size:12px}input,select,.plain{border:1px solid #65796a;background:#244240;color:inherit;border-radius:6px;padding:8px}input[type=number]{width:100px}.plain:hover{background:#36564c}.primary{background:#ecc67b;border:0;color:#24312a;font-weight:700;padding:11px 18px;border-radius:8px}.primary:disabled{opacity:.4}.error{color:#ffb1a5}#notice{padding:10px 0;font-size:16px}#hint{color:#bacaba;margin:0 0 16px;font-size:13px}.layout{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:22px;align-items:start}.table{min-width:0}.sidebar{position:sticky;top:16px;display:grid;gap:14px}.market-row{display:grid;grid-template-columns:56px repeat(4,minmax(0,1fr));gap:12px;margin:13px 0}.deck{border:1px solid #6c7761;border-radius:9px;background:repeating-linear-gradient(45deg,#28423b,#28423b 5px,#30493f 5px,#30493f 6px);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;font-size:12px;color:#e8d5a5}.deck strong{font:26px Georgia,serif}.development{position:relative;width:100%;min-width:0;min-height:142px;border:1px solid #ffffff55;border-radius:10px;color:var(--ink);background:linear-gradient(145deg,#ffffff24,transparent 50%),var(--color);padding:12px;text-align:left;box-shadow:0 4px 8px #0003;display:flex;flex-direction:column;justify-content:space-between;gap:12px;transition:transform .12s,box-shadow .12s}.development:disabled{opacity:1}.development.playable:hover{transform:translateY(-4px);box-shadow:0 7px 14px #0005,0 0 0 2px #efce84}.development.selected{box-shadow:0 0 0 3px #ffd778,0 7px 14px #0005;transform:translateY(-3px)}.card-head{display:flex;justify-content:space-between;align-items:start}.points{font:700 34px Georgia,serif}.points small{font:10px system-ui;color:inherit;display:block;letter-spacing:1px}.bonus{display:flex;flex-direction:column;align-items:center;font-size:10px;gap:4px}.jewel{font-size:32px;line-height:1;text-shadow:0 2px 1px #0003}.costs{display:flex;flex-wrap:wrap;gap:5px}.counter{display:inline-flex;align-items:center;justify-content:center;width:27px;height:27px;background:var(--color);color:var(--ink);border-radius:50%;border:1px solid #ffffff90;box-shadow:0 1px 3px #0005;font-weight:750;font-size:14px}.card-id{font-size:10px;opacity:.65}.card-foot{display:flex;gap:5px;align-items:end;justify-content:space-between}.empty{min-height:142px;border:1px dashed #4c6556;border-radius:10px;display:grid;place-items:center;color:#71897c;font-size:12px}.nobles{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px}.noble{position:relative;border:2px solid #cda558;border-radius:40px 40px 9px 9px;background:linear-gradient(#f8e8bd,#cfb780);color:#473821;padding:10px 18px;min-width:130px;display:flex;flex-direction:column;align-items:center;gap:7px;box-shadow:0 3px 7px #0003}.noble:disabled{opacity:1}.noble .crown{font-size:27px;line-height:1}.noble .noble-score{font:700 19px Georgia,serif}.noble small{color:#6b5834}.noble.eligible{outline:3px solid #8cceae;cursor:pointer}.noble.selected{outline:4px solid #ffdf78}.supply{display:grid;grid-template-columns:repeat(3,1fr);gap:16px 8px;text-align:center}.supply-item{display:flex;align-items:center;flex-direction:column;gap:7px}.chip{position:relative;border:5px dashed #ffffff65;border-radius:50%;background:var(--color);color:var(--ink);width:64px;height:64px;box-shadow:0 3px 0 #0006,0 5px 9px #0003;display:grid;place-items:center;font-size:26px;font-weight:750}.chip:disabled{opacity:.55}.chip.selected{outline:3px solid #ffd778;outline-offset:3px}.chip:not(:disabled):hover{transform:translateY(-2px)}.picked{position:absolute;right:-9px;top:-8px;border-radius:12px;background:#f4d38b;color:#332d22;font:700 12px system-ui;padding:3px 6px}.supply-item small{font-size:11px}#decision{border-color:#c5a76a}#decision-title{font:22px Georgia,serif;margin:0 0 9px}#decision-help{font-size:12px;color:#b7c7b8}.choice-row{display:flex;gap:7px;flex-wrap:wrap;margin:10px 0}.choice-option{padding:8px;background:#284b43;border:1px solid #607a62;border-radius:8px;color:inherit}.choice-option.selected{border-color:#f2d18c;background:#496047}.choice-option:disabled{opacity:.4}.choice-option .counter{margin:2px}.actions-footer{display:flex;gap:8px;margin-top:16px}.seats{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:20px}.seat.active{border-color:#d8b774}.seat-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}.seat-head h2{margin:0}.score{font:26px Georgia,serif;color:#f2d18c}.holdings{display:flex;gap:6px;flex-wrap:wrap;margin:7px 0}.holding{background:#23413b;border-radius:7px;padding:5px;display:flex;flex-direction:column;align-items:center;gap:4px}.holding small{font-size:9px}.reserves{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.reserves .development{min-height:110px;padding:8px}.reserves .points{font-size:24px}.reserves .counter{width:22px;height:22px;font-size:12px}.reserves .jewel{font-size:23px}.reserves .bonus{font-size:8px}.reserves .card-id{display:none}#history{max-height:240px;overflow:auto;font-size:12px;line-height:1.8}.history-panel{margin-top:20px}.legend{font-size:11px;color:#a8bcae;margin-top:10px}#selection-preview .development{max-width:155px;margin:12px auto}.return-chips{display:flex;flex-wrap:wrap;gap:14px;margin:15px 0}.return-chips .chip{width:43px;height:43px;font-size:18px;border-width:3px}.noble-choice{padding:6px 10px;border-radius:20px 20px 6px 6px;background:#dfc78f;color:#423522;border:2px solid #ac8f51}.noble-choice.selected{outline:2px solid #ffe2a1}
@media(max-width:1000px){.layout{grid-template-columns:minmax(0,1fr) 280px;gap:14px}.market-row{gap:8px;grid-template-columns:38px repeat(4,minmax(0,1fr))}.development{padding:8px;min-height:130px}.counter{width:23px;height:23px;font-size:12px}.seats{grid-template-columns:1fr}.noble{min-width:110px;padding:8px 12px}}
@media(max-width:720px){body{padding:12px}.layout{display:flex;flex-direction:column}.table,.sidebar{width:100%}.sidebar{position:static;display:flex;flex-direction:column}.market-row{grid-template-columns:28px repeat(4,minmax(0,1fr));gap:5px}.development{padding:6px;min-height:117px;gap:8px}.points{font-size:25px}.points small{font-size:0}.points small::before{content:"★";font-size:9px}.deck>span{font-size:8px}.jewel{font-size:23px}.bonus{font-size:8px}.costs{gap:3px}.counter{width:21px;height:21px;font-size:11px}.card-id{display:none}.nobles{gap:8px}.noble{flex:1;min-width:0;padding:8px}.noble .counter{width:22px;height:22px}.noble small{font-size:9px}.supply{grid-template-columns:repeat(6,minmax(0,1fr));gap:6px}.chip{width:43px;height:43px;font-size:20px;border-width:3px}.supply-item small{font-size:9px}.panel{padding:12px}.seats{grid-template-columns:1fr}.reserves .development{min-height:125px}header small{display:none}.empty{min-height:117px}#decision{scroll-margin-top:12px}}
</style>
<header><div><span class="eyebrow">Meeple Bots · Mesa de juego</span><h1>Splendor</h1></div><small>2 jugadores · Reservas públicas</small></header>
<details id="settings"><summary>Nueva partida / configurar jugadores</summary><div class="row" id="configs"></div><div class="row"><label>Seed<input id="seed" type="number" min="0" max="9007199254740991" value="42"></label><label>Pausa entre eventos (s)<input id="delay" type="number" min="0" step="0.1" value="0.3"></label><label>Guardar traza<input id="save" type="checkbox"></label><button id="start" class="primary">Empezar / reiniciar</button></div></details>
<div id="error" class="error" role="alert"></div><div id="notice" role="status" aria-live="polite"></div><p id="hint">Pulsa una carta para comprar o reservar. Pulsa las fichas del suministro para tomar gemas.</p>
<div class="layout"><main class="table"><h2>Nobles <small>· requisitos de descuentos, no de fichas</small></h2><div id="nobles" class="nobles"></div><div id="market"></div><p class="legend">El color de la carta es su descuento permanente · ★ puntos de victoria · círculos inferiores: coste impreso</p><div id="players" class="seats"></div></main>
<aside class="sidebar"><section class="panel"><h2>Suministro de gemas</h2><div id="bank" class="supply"></div><p class="legend">Pulsa colores distintos o dos veces el mismo color. Vuelve a pulsar para quitarlo. El oro se obtiene al reservar.</p></section><section id="decision" class="panel"><h2 id="decision-title">Tu jugada</h2><p id="decision-help"></p><div id="selection-preview"></div><div id="modes" class="choice-row"></div><div id="variants"></div><div class="actions-footer"><button id="confirm" class="primary" disabled>Confirmar</button><button id="cancel" class="plain" hidden>Cancelar</button></div></section></aside></div>
<details class="history-panel"><summary>Historial de jugadas y azar</summary><div id="history"></div></details>
<script>
const names=['Blanco','Azul','Verde','Rojo','Negro','Oro'];
const colors=['#eee7d6','#377bad','#479064','#b95149','#333b48','#dbb65e'];
const inks=['#3b403e','#fff5e5','#fff5e5','#fff5e5','#fff5e5','#3b3022'];
const $=id=>document.getElementById(id);
let state=null,version='',selection=null,choice=null,busy=false;
const paint=i=>`--color:${colors[i]};--ink:${inks[i]}`;
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const total=a=>a.reduce((x,y)=>x+y,0);
const subset=(a,b)=>a.every((n,i)=>n<=b[i]);
const zeros=()=>[0,0,0,0,0,0];
const human=()=>state?.status==='waiting_human'&&!busy;
const legal=()=>state.legal_actions.map((a,index)=>({a,index}));
const counters=values=>values.map((n,i)=>n?`<span class="counter" style="${paint(i)}" title="${n} ${names[i]}" aria-label="${n} ${names[i]}">${n}</span>`:'').join('');
function textGems(values){return values.map((n,i)=>n?`${n} ${names[i]}`:'').filter(Boolean).join(', ')||'Nada'}
const baseline = __MCTS_BASELINE__;
function updatePlayerControls(i){
 $('mcts'+i).hidden=$('kind'+i).value!=='mcts';
 const timed=$('budget'+i).value==='time';
 $('iteration-field'+i).hidden=timed;$('time-field'+i).hidden=!timed;
}
function playerConfig(i){
 const kind=$('kind'+i).value;if(kind!=='mcts')return {kind};
 const timed=$('budget'+i).value==='time';
 return {kind,iterations:timed?null:Number($('iterations'+i).value),
  time_budget:timed?Number($('time'+i).value):null,
  rollout_depth:Number($('depth'+i).value),exploration:Number($('exploration'+i).value),
  heuristic:$('heuristic'+i).value==='none'?null:Number($('heuristic'+i).value),
  selection_policy:$('policy'+i).value,tree_reuse:$('reuse'+i).checked,
  transpositions:$('trans'+i).checked,root_diagnostics:$('diagnostics'+i).checked};
}
for(let i=0;i<2;i++){
 const box=document.createElement('section');
 box.innerHTML=`<h2>Jugador ${i+1}</h2>
 <label>Control<select id="kind${i}"><option value="human">Humano</option><option value="random">Random</option><option value="mcts">MCTS</option></select></label>
 <div id="mcts${i}" hidden>
 <label>Presupuesto<select id="budget${i}"><option value="iterations">Iteraciones</option><option value="time">Tiempo</option></select></label>
 <label id="iteration-field${i}">Iteraciones<input id="iterations${i}" type="number" min="1" step="1"></label>
 <label id="time-field${i}" hidden>Segundos / decisión<input id="time${i}" type="number" min="0.001" step="0.05"></label>
 <label>Profundidad<input id="depth${i}" type="number" min="1" step="1"></label>
 <label>Exploración<input id="exploration${i}" type="number" min="0" step="0.05"></label>
 <label>Selección<select id="policy${i}"><option value="uct">UCT</option><option value="ucb1_tuned">UCB1-Tuned</option></select></label>
 <label>Evaluación al corte<select id="heuristic${i}"><option value="none">Neutral</option><option value="0">H0 · Prestigio</option></select></label>
 <label>Reutilizar árbol<input id="reuse${i}" type="checkbox"></label>
 <label>Transposiciones<input id="trans${i}" type="checkbox"></label>
 <label>Diagnóstico de raíz<input id="diagnostics${i}" type="checkbox"></label>
 </div>`;
 $('configs').append(box);
 $('kind'+i).value=i===0?'human':'mcts';
 $('budget'+i).value=baseline.time_budget===null?'iterations':'time';
 $('iterations'+i).value=baseline.iterations??1000;$('time'+i).value=baseline.time_budget??1;
 $('depth'+i).value=baseline.rollout_depth;$('exploration'+i).value=baseline.exploration;
 $('policy'+i).value=baseline.selection_policy;$('heuristic'+i).value=baseline.heuristic===null?'none':String(baseline.heuristic);
 $('reuse'+i).checked=baseline.tree_reuse;$('trans'+i).checked=baseline.transpositions;
 $('diagnostics'+i).checked=baseline.root_diagnostics??false;
 $('kind'+i).onchange=()=>updatePlayerControls(i);$('budget'+i).onchange=()=>updatePlayerControls(i);
 updatePlayerControls(i);
}
function actionName(a){switch(a.kind){case 'take_different':return 'Tomar '+names.filter((_,i)=>a.colors&(1<<i)).join(' + ');case 'take_same':return 'Tomar 2 '+names[a.color];case 'reserve_visible':return `Reservar nivel ${a.tier+1}, posición ${a.slot+1}`;case 'buy_visible':return `Comprar nivel ${a.tier+1}, posición ${a.slot+1}`;case 'buy_reserved':return `Comprar reserva ${a.index+1}`;case 'pass':return 'Pasar (sin acciones disponibles)';case 'refill':return `Refill: carta #${a.card}`;}}

function taken(a){const values=zeros();if(a.kind==='take_same')values[a.color]=2;if(a.kind==='take_different')for(let i=0;i<5;i++)values[i]=a.colors&(1<<i)?1:0;return values;}
function gemActions(){return legal().filter(x=>['take_same','take_different'].includes(x.a.kind));}
function targetActions(target){return legal().filter(({a})=>target.reserved
 ? a.kind==='buy_reserved'&&a.index===target.index&&target.player===state.active_player
 : ['buy_visible','reserve_visible'].includes(a.kind)&&a.tier===target.tier&&a.slot===target.slot);}
function selectedCard(){return selection?.type==='card'?(selection.reserved?state.holdings[selection.player].reserved[selection.index]:state.market[selection.tier][selection.slot]):null;}
function baseActions(){if(!selection)return [];if(selection.type==='gems')return gemActions().filter(x=>same(taken(x.a),selection.tokens));if(selection.type==='pass')return legal().filter(x=>x.a.kind==='pass');return targetActions(selection).filter(x=>x.a.kind===selection.mode);}
function resetDetails(){if(selection){selection.payment=null;selection.returned=zeros();selection.noble=null;}}
function selectCard(target){if(!human())return;selection={type:'card',...target,mode:null};resetDetails();draw();if(matchMedia('(max-width:720px)').matches)$('decision').scrollIntoView({behavior:'smooth',block:'start'});}
function nextTokens(color){const current=selection?.type==='gems'?[...selection.tokens]:zeros();if(current[color]===2){current[color]=0;return current;}if(current[color]===1){current[color]=2;if(gemActions().some(x=>subset(current,taken(x.a))))return current;current[color]=0;return current;}current[color]=1;return gemActions().some(x=>subset(current,taken(x.a)))?current:null;}
function selectGem(color){if(!human())return;const tokens=nextTokens(color);if(!tokens)return;selection=total(tokens)?{type:'gems',tokens}:null;resetDetails();draw();}
function chooseMode(mode){selection.mode=mode;resetDetails();draw();}
function cardElement(id,target=null){if(id===null){const e=document.createElement('div');e.className='empty';e.textContent='Vacío';return e;}const c=state.cards[id],button=document.createElement('button');const playable=!!target&&human()&&targetActions(target).length>0;
 button.className='development'+(playable?' playable':'')+(selection?.type==='card'&&selectedCard()===id?' selected':'');button.style.cssText=paint(c.bonus);button.disabled=!playable;button.dataset.card=id;button.setAttribute('aria-label',`Carta ${id}, ${names[c.bonus]}, ${c.points} puntos. Coste: ${textGems(c.cost)}${playable?'. Pulsa para comprar o reservar':''}`);
 button.innerHTML=`<div class="card-head"><div class="points">${c.points}<small>★ PUNTOS</small></div><div class="bonus"><span class="jewel">◆</span>+1 ${names[c.bonus]}</div></div><div class="card-foot"><div class="costs">${counters(c.cost)}</div><span class="card-id">#${id}</span></div>`;
 if(target)button.onclick=()=>selectCard(target);return button;
}
function nobleOptions(){let xs=baseActions();if(!xs.length)return [];if(selection.payment)xs=xs.filter(x=>same(x.a.payment,selection.payment));xs=xs.filter(x=>same(x.a.returned,selection.returned));return [...new Set(xs.map(x=>x.a.noble))].filter(n=>n!==null);}
function chooseNoble(id){selection.noble=id;draw();}
function drawBoard(){
 $('market').replaceChildren();for(const tier of [2,1,0]){const row=document.createElement('div');row.className='market-row';const deck=document.createElement('div');deck.className='deck';deck.innerHTML=`<span>${'◆'.repeat(tier+1)}</span><strong>${state.remaining[tier].length}</strong><small>N${tier+1}</small>`;deck.title=`Nivel ${tier+1}: ${state.remaining[tier].length} cartas restantes`;row.append(deck);state.market[tier].forEach((id,slot)=>row.append(cardElement(id,{tier,slot})));$('market').append(row);}
 $('bank').replaceChildren();state.bank.forEach((count,i)=>{const item=document.createElement('div');item.className='supply-item';const chip=document.createElement('button');const picked=selection?.type==='gems'?selection.tokens[i]:0;chip.className='chip'+(picked?' selected':'');chip.style.cssText=paint(i);chip.dataset.gem=i;chip.disabled=!human()||i===5||nextTokens(i)===null;chip.setAttribute('aria-label',`${names[i]}: ${count} disponibles, ${picked} seleccionadas`);chip.innerHTML=`${count}${picked?`<span class="picked">+${picked}</span>`:''}`;chip.onclick=()=>selectGem(i);const label=document.createElement('small');label.textContent=names[i];item.append(chip,label);$('bank').append(item);});
 $('nobles').replaceChildren();const eligible=nobleOptions();for(const id of state.nobles){const n=state.noble_data[id],button=document.createElement('button');button.className='noble'+(eligible.includes(id)?' eligible':'')+(selection?.noble===id?' selected':'');button.dataset.noble=id;button.disabled=!human()||!eligible.includes(id);button.setAttribute('aria-label',`Noble ${id}, 3 puntos, requiere ${textGems(n.requirements)}${eligible.includes(id)?'. Elegir este noble':''}`);button.innerHTML=`<span class="crown">♛</span><span class="noble-score">3 ★</span><div class="costs">${counters(n.requirements)}</div><small>NOBLE ${id}</small>`;button.onclick=()=>chooseNoble(id);$('nobles').append(button);}
 $('players').replaceChildren();state.holdings.forEach((p,i)=>{const section=document.createElement('section');section.className='panel seat'+(state.active_player===i&&!state.finished?' active':'');section.innerHTML=`<div class="seat-head"><h2>Jugador ${i+1} <small>· ${state.players[i].kind}</small></h2><span class="score">${p.prestige} ★</span></div><small>FICHAS · ${total(p.tokens)}/10</small><div class="holdings">${p.tokens.map((n,c)=>`<div class="holding"><span class="counter" style="${paint(c)}" title="${names[c]}">${n}</span><small>${names[c]}</small></div>`).join('')}</div><small>DESCUENTOS PERMANENTES</small><div class="holdings">${counters(p.bonuses)||'—'}</div><p class="legend">${p.purchased.length} cartas · Nobles: ${p.nobles.map(n=>'♛ '+n).join(', ')||'—'}</p><h3>Reservas públicas · ${p.reserved.length}/3</h3>`;const reserves=document.createElement('div');reserves.className='reserves';p.reserved.forEach((id,index)=>reserves.append(cardElement(id,{reserved:true,player:i,index})));section.append(reserves);$('players').append(section);});
}
function button(label,callback,className='choice-option'){const b=document.createElement('button');b.className=className;b.textContent=label;b.onclick=callback;return b;}
function drawDecision(){choice=null;$('selection-preview').replaceChildren();$('modes').replaceChildren();$('variants').replaceChildren();$('cancel').hidden=!selection;$('confirm').disabled=true;$('confirm').hidden=!selection;$('confirm').textContent='Confirmar';
 $('decision-title').textContent=selection?.type==='gems'?'Tomar gemas':selection?.type==='card'?'Carta seleccionada':'Tu jugada';
 $('decision-help').textContent=human()?'Pulsa una carta o las fichas del suministro.':'Las acciones estarán disponibles en tu turno.';
 if(!human())return;
 if(!selection){if(legal().some(x=>x.a.kind==='pass'))$('modes').append(button('Pasar · sin acciones legales',()=>{selection={type:'pass'};resetDetails();draw();}));return;}
 if(selection.type==='card'){
  $('selection-preview').append(cardElement(selectedCard()));
  const options=targetActions(selection);for(const [kind,label] of [[selection.reserved?'buy_reserved':'buy_visible','Comprar'],['reserve_visible','Reservar']]){if(selection.reserved&&kind==='reserve_visible')continue;const b=button(label,()=>chooseMode(kind),'choice-option'+(selection.mode===kind?' selected':''));b.disabled=!options.some(x=>x.a.kind===kind);$('modes').append(b);}
  $('decision-help').textContent=selection.mode==='reserve_visible'?'La reserva queda pública. Recibes oro si queda en el suministro.':'Compra con tus descuentos y fichas, o reserva la carta para más adelante.';
 }else if(selection.type==='gems'){$('selection-preview').innerHTML=`<div class="choice-row">${counters(selection.tokens)}</div>`;$('decision-help').textContent='Selecciona en el suministro la combinación que quieres tomar.';}
 let xs=baseActions();if(!xs.length){if(selection.type==='gems')$('decision-help').textContent='Completa la selección con las fichas resaltables del suministro.';return;}
 const payments=[...new Map(xs.map(x=>[JSON.stringify(x.a.payment),x.a.payment])).values()];
 if(!selection.payment||!payments.some(p=>same(p,selection.payment)))selection.payment=payments[0];
 if(payments.some(p=>total(p)>0)){
  const title=document.createElement('h3');title.textContent=payments.length===1?'Pagas':'Elige con qué pagar';$('variants').append(title);const row=document.createElement('div');row.className='choice-row';for(const pay of payments){const b=button('',()=>{selection.payment=pay;selection.returned=zeros();selection.noble=null;draw();},'choice-option'+(same(pay,selection.payment)?' selected':''));b.innerHTML=counters(pay)||'Gratis';b.setAttribute('aria-label','Pagar '+textGems(pay));row.append(b);}$('variants').append(row);
 }
 xs=xs.filter(x=>same(x.a.payment,selection.payment));const returns=[...new Map(xs.map(x=>[JSON.stringify(x.a.returned),x.a.returned])).values()];
 const required=total(returns[0]);if(required){const title=document.createElement('h3');title.textContent=`Devuelve ${required} fichas · ${total(selection.returned)}/${required}`;$('variants').append(title);const row=document.createElement('div');row.className='return-chips';for(let c=0;c<6;c++){if(!returns.some(r=>r[c]>0))continue;const next=[...selection.returned];next[c]++;if(!returns.some(r=>subset(next,r)))next[c]=0;const b=button('',()=>{selection.returned=next;selection.noble=null;draw();},'chip'+(selection.returned[c]?' selected':''));b.style.cssText=paint(c);b.dataset.return=c;b.innerHTML=`${selection.returned[c]}`;b.setAttribute('aria-label',`Devolver ${names[c]}: ${selection.returned[c]} seleccionadas. Pulsar para cambiar.`);b.disabled=!returns.some(r=>subset(next,r));row.append(b);}$('variants').append(row);}
 xs=xs.filter(x=>same(x.a.returned,selection.returned));const nobles=[...new Set(xs.map(x=>x.a.noble))];if(nobles.length===1)selection.noble=nobles[0];if(nobles.some(n=>n!==null)){const h=document.createElement('h3');h.textContent=nobles.length>1?'Elige un noble':'Recibes este noble';$('variants').append(h);const row=document.createElement('div');row.className='choice-row';for(const n of nobles){if(n===null)continue;const b=button(`♛ Noble ${n} · 3 ★`,()=>chooseNoble(n),'noble-choice'+(selection.noble===n?' selected':''));b.title=textGems(state.noble_data[n].requirements);row.append(b);}$('variants').append(row);}
 const exact=xs.find(x=>x.a.noble===selection.noble);choice=exact?.index??null;$('confirm').disabled=choice===null;
 $('confirm').textContent=selection.type==='gems'?'Tomar gemas':selection.type==='pass'?'Pasar':selection.mode==='reserve_visible'?'Reservar':'Comprar';
}
function draw(){drawDecision();drawBoard();}
$('cancel').onclick=()=>{selection=null;draw();};
function render(s){const v=`${s.session_id}:${s.events.length}:${s.status}:${s.trace_path}`;state=s;if(v===version)return;version=v;selection=null;
 $('notice').textContent=s.status==='finished'?(s.utilities[0]===0?'Empate':`Gana el jugador ${s.utilities[0]>0?1:2}`):s.status==='idle'?'Configura los jugadores y empieza una partida.':s.status==='error'?s.message:s.phase==='chance'?'Azar: reponiendo el mercado…':`Turno del jugador ${s.active_player+1} · ${s.status==='waiting_human'?'elige sobre el tablero':'pensando…'}${s.final_round?' · Última ronda':''}`;
 if(s.status==='idle')$('settings').open=true;draw();$('history').replaceChildren();for(const [i,e] of s.events.entries()){const line=document.createElement('div');line.textContent=`${i+1}. ${e.player===null?'Azar':'J'+(e.player+1)}: ${actionName(e.action)}${e.player===null?'':` · Pago: ${textGems(e.action.payment)} · Devuelve: ${textGems(e.action.returned)} · Noble: ${e.action.noble??'—'}`}`;$('history').append(line);}if(s.trace_path){const p=document.createElement('p');p.textContent='Traza guardada: '+s.trace_path;$('history').append(p);}
}
async function request(path,payload){const r=await fetch(path,payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const s=await r.json();if(!r.ok)throw Error(s.error);return s;}
$('start').onclick=async()=>{try{$('error').textContent='';const seed=Number($('seed').value);if(!Number.isSafeInteger(seed)||seed<0)throw Error('El seed debe ser un entero entre 0 y 9007199254740991.');render(await request('/api/start',{first:playerConfig(0),second:playerConfig(1),seed,minimum_move_seconds:Number($('delay').value),save_trace:$('save').checked}));$('settings').open=false;}catch(e){$('error').textContent=e.message;}};
$('confirm').onclick=async()=>{try{if(choice===null||busy)return;busy=true;$('confirm').disabled=true;const next=await request('/api/move',{action:choice,turn:state.events.length,session_id:state.session_id});busy=false;version='';render(next);}catch(e){$('error').textContent=e.message;version='';}finally{busy=false;}};
async function poll(){try{render(await request('/api/state'));}catch(e){$('error').textContent=e.message;}setTimeout(poll,350);}poll();
</script></html>'''


PAGE = PAGE.replace("__MCTS_BASELINE__", json.dumps(SPLENDOR_BASELINE.as_dict()))
