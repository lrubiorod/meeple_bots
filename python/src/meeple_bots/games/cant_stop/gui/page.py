"""Self-contained Can't Stop browser board; presentation only, no game rules."""
import json
from ....gui.baselines import CANT_STOP_BASELINE

PAGE = r'''<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Can't Stop · Meeple Bots</title><style>
:root{font-family:system-ui,sans-serif;color:#e7ebee;background:#14202a}body{margin:0;padding:24px;max-width:1300px;margin:auto}h1{margin:0;font-size:32px}p{color:#afbec9}.layout{display:grid;grid-template-columns:minmax(450px,1fr) 310px;gap:24px}.panel{background:#20303d;border-radius:14px;padding:18px;margin-top:18px}.board{display:flex;align-items:flex-end;justify-content:space-around;gap:6px;min-height:420px}.column{display:flex;flex-direction:column;align-items:center;gap:5px;flex:1}.cell{height:24px;width:100%;max-width:43px;border:1px solid #526372;border-radius:6px;display:flex;justify-content:center;align-items:center;gap:2px;background:#172531}.cell.top{border-color:#edc47d}.token{display:inline-block;width:10px;height:10px;border-radius:50%}.p0{background:#45bde5}.p1{background:#ef8d72}.temporary{background:#fff;border:1px solid #172531;width:9px;height:9px;border-radius:2px}.closed{opacity:.5}.colnum{font-weight:700;margin-top:5px}.dice{display:flex;gap:10px;margin:18px 0}.die{background:#fff;color:#14202a;border-radius:9px;width:46px;height:46px;display:grid;place-items:center;font-size:30px;font-weight:bold}button{border:0;border-radius:8px;padding:11px;background:#e9bd74;color:#15232f;font-weight:700;cursor:pointer;margin:4px}button:disabled{opacity:.5;cursor:default}input,select{box-sizing:border-box;width:100%;padding:7px;border-radius:6px;border:1px solid #526372;background:#14202a;color:#e7ebee}label{display:block;margin:9px 0;font-size:13px}fieldset{border:1px solid #526372;border-radius:9px;margin:12px 0;padding:10px}.settings{display:grid;grid-template-columns:1fr 1fr;gap:9px}.settings label{margin:2px 0}.hidden{display:none}.log{max-height:220px;overflow:auto;font-size:13px}.log div{border-bottom:1px solid #344655;padding:5px}#message{min-height:24px;font-size:18px}#error{color:#ffb1a0}input[type=checkbox]{width:auto}small{color:#afbec9}@media(max-width:850px){.layout{grid-template-columns:1fr}.board{min-height:400px}body{padding:12px}}
</style></head><body>
<h1>Can't Stop</h1><p>Avanza, asegura tus columnas o arriesga otra tirada. Gana quien consolide tres columnas.</p>
<div class="layout"><main><div class="panel"><div id="message">Configura los jugadores y empieza.</div><div class="dice" id="dice"></div><div id="board" class="board"></div><p><span class="token p0"></span> Jugador 1 &nbsp; <span class="token p1"></span> Jugador 2 &nbsp; <span class="token temporary"></span> Avance provisional</p><div id="actions"></div><div id="error" role="alert"></div></div>
<div class="panel"><b>Últimos eventos</b><div id="history" class="log"></div><small id="timing"></small></div></main>
<aside class="panel"><b>Nueva partida</b><div id="players"></div><label>Semilla<input id="seed" type="number" min="0" value="0"></label><label>Pausa entre eventos (s)<input id="delay" type="number" min="0" step="0.1" value="0.4"></label><label><input id="save" type="checkbox"> Guardar partida y tiradas</label><button id="start">Empezar / Reiniciar</button><p id="trace"></p><small>Los dados son públicos. Plantarse conserva el avance; fallar elimina solo el progreso provisional del turno. Las columnas se conquistan al plantarse.</small></aside></div>
<script>
const baseline = __BASELINE__;
let state=null,pending=false,renderedDecision=null;
const $=s=>document.querySelector(s);
function policyControls(i,prefix){return `<label>Política<select id="${prefix}-policy-${i}"><option value="uniform_random">Uniforme</option><option value="greedy">Greedy</option><option value="epsilon_greedy">Epsilon-greedy</option><option value="mast">MAST</option></select></label><label>Evaluación<select id="${prefix}-evaluator-${i}"><option value="0">H0 · Progreso</option><option value="none">Neutral</option></select></label><label>Epsilon<input id="${prefix}-epsilon-${i}" type="number" min="0" max="1" step="0.05"></label>`;}
function setPolicy(i,prefix,policy){
 $(`#${prefix}-policy-${i}`).value=policy.kind;
 $(`#${prefix}-evaluator-${i}`).value=policy.evaluator?.kind==='neutral'?'none':'0';
 $(`#${prefix}-epsilon-${i}`).value=policy.epsilon??0.1;
}
function evaluator(value){return value==='none'?{kind:'neutral'}:{kind:'game_heuristic',index:Number(value)};}
function readPolicy(i,prefix){
 const kind=$(`#${prefix}-policy-${i}`).value,policy={kind};
 if(kind==='greedy'||kind==='epsilon_greedy')policy.evaluator=evaluator($(`#${prefix}-evaluator-${i}`).value);
 if(kind==='mast'||kind==='epsilon_greedy')policy.epsilon=Number($(`#${prefix}-epsilon-${i}`).value);
 return policy;
}
for(let i=0;i<2;i++){
 const box=document.createElement('fieldset');
 box.innerHTML=`<legend>Jugador ${i+1}</legend><label>Agente<select id="kind-${i}"><option value="human">Humano</option><option value="random">Aleatorio</option><option value="mcts">MCTS</option></select></label><div id="mcts-${i}" class="settings"><label>Presupuesto<select id="mode-${i}"><option value="iterations">Iteraciones</option><option value="time">Tiempo</option></select></label><label id="budget-label-${i}">Cantidad<input id="budget-${i}" type="number" min="0.001" step="any"></label><label>Profundidad<input id="depth-${i}" type="number" min="1"></label><label>Exploración<input id="exploration-${i}" type="number" min="0" step="any"></label><label>Selección<select id="selection-${i}"><option value="uct">UCT</option><option value="ucb1_tuned">UCB1-Tuned</option></select></label><label>Cutoff<select id="heuristic-${i}"><option value="0">H0 · Progreso</option><option value="none">Neutral</option></select></label><details style="grid-column:1/-1"><summary>Políticas y memoria</summary><label>Aplicar rollout<select id="rollout-phase-${i}"><option value="always">Siempre</option><option value="choose">Al elegir avances</option><option value="continue">Al decidir seguir o parar</option></select></label>${policyControls(i,'primary')}<div id="fallback-${i}"><b>En las demás fases</b>${policyControls(i,'fallback')}</div><label><input id="bias-${i}" type="checkbox"> Sesgo progresivo</label><div id="bias-settings-${i}"><label>Peso<input id="bias-weight-${i}" type="number" min="0" step="0.05"></label><label>Evaluación del sesgo<select id="bias-evaluator-${i}"><option value="0">H0 · Progreso</option><option value="none">Neutral</option></select></label><label>Aplicar sesgo<select id="bias-phase-${i}"><option value="always">Siempre</option><option value="choose">Al elegir avances</option><option value="continue">Al decidir seguir o parar</option></select></label></div><label><input id="reuse-${i}" type="checkbox"> Reutilizar árbol</label><label><input id="transpositions-${i}" type="checkbox"> Compartir estados iguales</label><label><input id="diagnostics-${i}" type="checkbox"> Guardar diagnóstico de raíz</label></details></div>`;
 $('#players').append(box);
 $(`#kind-${i}`).value=i===0?'human':'mcts';
 $(`#mode-${i}`).value=baseline.time_budget===null?'iterations':'time';
 $(`#budget-${i}`).value=baseline.time_budget??baseline.iterations;
 $(`#depth-${i}`).value=baseline.rollout_depth;$(`#exploration-${i}`).value=baseline.exploration;
 $(`#selection-${i}`).value=baseline.selection_policy;$(`#heuristic-${i}`).value=baseline.heuristic===null?'none':String(baseline.heuristic);
 const policy=baseline.rollout_policy??{kind:'uniform_random'},conditional=policy.kind==='conditional';
 $(`#rollout-phase-${i}`).value=conditional?policy.condition.phase:'always';
 setPolicy(i,'primary',conditional?policy.primary:policy);setPolicy(i,'fallback',conditional?policy.fallback:{kind:'uniform_random'});
 const bias=baseline.progressive_bias;
 $(`#bias-${i}`).checked=bias!==null&&bias!==undefined;$(`#bias-weight-${i}`).value=bias?.weight??0.25;
 $(`#bias-evaluator-${i}`).value=bias?.evaluator.kind==='neutral'?'none':'0';$(`#bias-phase-${i}`).value=bias?.condition?.phase??'always';
 $(`#reuse-${i}`).checked=baseline.tree_reuse;$(`#transpositions-${i}`).checked=baseline.transpositions;$(`#diagnostics-${i}`).checked=baseline.root_diagnostics??false;
 const update=()=>{
  $(`#mcts-${i}`).classList.toggle('hidden',$(`#kind-${i}`).value!=='mcts');$(`#exploration-${i}`).disabled=$(`#selection-${i}`).value!=='uct';
  $(`#fallback-${i}`).classList.toggle('hidden',$(`#rollout-phase-${i}`).value==='always');
  $(`#bias-settings-${i}`).classList.toggle('hidden',!$(`#bias-${i}`).checked);
  for(const prefix of ['primary','fallback']){const kind=$(`#${prefix}-policy-${i}`).value;$(`#${prefix}-evaluator-${i}`).disabled=!['greedy','epsilon_greedy'].includes(kind);$(`#${prefix}-epsilon-${i}`).disabled=!['mast','epsilon_greedy'].includes(kind);}
 };
 for(const id of ['kind','selection','rollout-phase','bias','primary-policy','fallback-policy'])$(`#${id}-${i}`).onchange=update;
 $(`#mode-${i}`).onchange=()=>{$(`#budget-${i}`).value=$(`#mode-${i}`).value==='time'?1:baseline.iterations??2000;};update();
}
function config(i){
 const kind=$(`#kind-${i}`).value;if(kind!=='mcts')return{kind};
 const timed=$(`#mode-${i}`).value==='time',h=$(`#heuristic-${i}`).value;
 const phase=$(`#rollout-phase-${i}`).value,primary=readPolicy(i,'primary');
 const rollout_policy=phase==='always'?primary:{kind:'conditional',condition:{kind:'turn_phase',phase},primary,fallback:readPolicy(i,'fallback')};
 const biasPhase=$(`#bias-phase-${i}`).value;
 const progressive_bias=$(`#bias-${i}`).checked?{weight:Number($(`#bias-weight-${i}`).value),evaluator:evaluator($(`#bias-evaluator-${i}`).value),condition:biasPhase==='always'?null:{kind:'turn_phase',phase:biasPhase}}:null;
 return{kind,iterations:timed?null:Number($(`#budget-${i}`).value),time_budget:timed?Number($(`#budget-${i}`).value):null,rollout_depth:Number($(`#depth-${i}`).value),exploration:Number($(`#exploration-${i}`).value),selection_policy:$(`#selection-${i}`).value,heuristic:h==='none'?null:Number(h),tree_reuse:$(`#reuse-${i}`).checked,transpositions:$(`#transpositions-${i}`).checked,rollout_policy,progressive_bias,root_diagnostics:$(`#diagnostics-${i}`).checked};
}
async function api(path,payload){const r=await fetch(path,payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const value=await r.json();if(!r.ok)throw Error(value.error??r.statusText);return value;}
function actionLabel(a){return a.kind==='advance'?'Avanzar '+a.columns.join(' + '):a.kind==='roll'?'Volver a tirar':a.kind==='stop'?'Plantarse':'Dados: '+a.dice.join(' · ');}
function render(){
 if(!state)return;
 $('#board').replaceChildren();
 for(let c=0;c<11;c++){
  const col=document.createElement('div');col.className='column';
  if(state.claimed[c]!==null)col.classList.add('closed');
  for(let h=state.heights[c];h>0;h--){const cell=document.createElement('div');cell.className='cell'+(h===state.heights[c]?' top':'');
   for(let p=0;p<2;p++)if(state.progress[p][c]===h){const token=document.createElement('span');token.className='token p'+p;cell.append(token);}
   if(state.runners[c]===h){const token=document.createElement('span');token.className='token temporary';cell.append(token);}col.append(cell);
  }
  const label=document.createElement('div');label.className='colnum';label.textContent=String(c+2)+(state.claimed[c]!==null?' ✓':'');col.append(label);$('#board').append(col);
 }
 $('#dice').replaceChildren();for(const d of state.dice){const el=document.createElement('div');el.className='die';el.textContent=d||'–';$('#dice').append(el);}
 $('#message').textContent=state.status==='idle'?'Configura y empieza.':state.status==='error'?state.message:state.winner!==null?`¡Gana el jugador ${state.winner+1}!`:state.last_bust?`Tirada fallida. Turno del jugador ${state.active_player+1}.`:state.phase==='roll'?`Jugador ${state.active_player+1}: lanzando dados…`:state.status==='waiting_human'?`Jugador ${state.active_player+1}: ${state.phase==='choose'?'elige los avances':'¿seguir o plantarse?'}`:`Jugador ${state.active_player+1}: pensando…`;
 // Preserve button identity across polls so pointer and keyboard clicks are not interrupted.
 const decision=state.status==='waiting_human'?JSON.stringify([state.events.length,state.active_player,state.legal_actions]):null;
 if(decision!==renderedDecision){
  const actions=$('#actions');actions.replaceChildren();renderedDecision=decision;
  if(decision!==null)state.legal_actions.forEach((a,i)=>{const b=document.createElement('button');b.textContent=actionLabel(a);const turn=state.events.length;b.onclick=()=>move(i,turn);actions.append(b);});
 }
 for(const button of $('#actions').children)button.disabled=pending;
 $('#history').replaceChildren();for(const e of state.events.slice(-30).reverse()){const row=document.createElement('div');row.textContent=(e.player===null?'Azar':`J${e.player+1}`)+' · '+actionLabel(e.action)+(e.action.kind==='dice'&&e.bust?' · Fallo':'');$('#history').append(row);}
 const last=state.events.slice().reverse().find(e=>e.search_iterations!==null);
 $('#timing').textContent=last?`${last.search_iterations} iteraciones · ${last.search_nodes??'–'} nodos · ${(last.decision_seconds*1000).toFixed(1)} ms${last.tree_reuse?' · '+last.tree_reuse.reused_nodes+' nodos conservados':''}`:'';
 $('#trace').textContent=state.trace_path?`Guardado: ${state.trace_path}`:'';
}
async function move(action,turn){if(pending)return;pending=true;render();try{state=await api('/api/move',{action,turn});$('#error').textContent='';}catch(e){$('#error').textContent=e.message;}finally{pending=false;render();}}
$('#start').onclick=async()=>{try{state=await api('/api/start',{first:config(0),second:config(1),seed:Number($('#seed').value),minimum_move_seconds:Number($('#delay').value),save_trace:$('#save').checked});$('#error').textContent='';render();}catch(e){$('#error').textContent=e.message;}};
async function poll(){try{state=await api('/api/state');render();}catch(e){$('#error').textContent=e.message;}setTimeout(poll,150);}poll();
</script></body></html>'''
PAGE = PAGE.replace("__BASELINE__", json.dumps(CANT_STOP_BASELINE.as_dict()))
