"""Single-page browser client for Spirits of the Forest."""

PAGE = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Meeple Bots · Spirits of the Forest</title>
<style>
:root{--ink:#243128;--paper:#f4f0e5;--panel:#fffdf6;--first:#2374ab;--second:#c74758}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#dbe9d1,#f2e3c2);color:var(--ink);font:15px system-ui,sans-serif}
.shell{max-width:1500px;margin:auto;padding:20px}h1{margin:0;font-family:Georgia,serif;font-size:clamp(1.8rem,4vw,3.4rem)}
.subtitle{margin:.2rem 0 1rem;color:#536257}.panel{background:var(--panel);border:1px solid #cabf9f;border-radius:16px;box-shadow:0 8px 24px #32402a1c;padding:16px;margin-bottom:14px}
.config{display:flex;gap:12px;align-items:end;flex-wrap:wrap}.config label,.mcts-config label{display:grid;gap:4px;font-size:.82rem}.config input,.config select,.config button,.mcts-config input,.mcts-config select{font:inherit;padding:8px;border:1px solid #a99e80;border-radius:8px;background:white}.config button,.action{cursor:pointer;background:#315f44;color:white;border:0}[hidden]{display:none!important}.mcts-configs{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin-bottom:16px}.mcts-config{display:flex;gap:10px;align-items:end;flex-wrap:wrap;border:1px solid #c8bea4;border-radius:12px;padding:12px;background:#f7f1e4}.mcts-config strong{flex-basis:100%}
.players{display:grid;grid-template-columns:1fr;gap:12px}.player{border-left:5px solid var(--color);padding:10px;background:#f6f2e8;border-radius:8px}.player.active{outline:2px solid var(--color)}.score{font-size:1.5rem;font-weight:800}
.counts{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:5px;font-size:.75rem;margin-top:6px}.count{background:#fff;padding:5px;border-radius:6px;border-left:5px solid var(--counter);display:flex;justify-content:space-between;gap:5px}.count i{width:9px;height:9px;border-radius:50%;background:var(--counter);display:inline-block;margin-right:3px}
.forest-wrap{overflow-x:auto;padding:8px 0}.forest{display:grid;grid-template-columns:repeat(12,minmax(72px,1fr));gap:7px;min-width:920px}.tile{position:relative;min-height:82px;border:2px solid #ffffff99;border-radius:12px;background:var(--tile);color:#1f211e;box-shadow:0 3px 7px #0003;padding:6px;cursor:default}.tile.legal{cursor:pointer;outline:4px solid #2a8b52;transform:translateY(-2px)}.tile.source-option{outline-color:#2374ab}.tile.payment-option{outline-color:#d07a16}.tile.selected{outline:5px solid #2d2930;transform:translateY(-3px)}.tile.empty{visibility:hidden}.tile strong{display:block;font-size:.8rem;text-transform:uppercase}.symbols{font:700 1.35rem Georgia,serif;margin-top:7px}.source{position:absolute;right:6px;bottom:5px;font-size:1.3rem}.gem{position:absolute;right:4px;top:4px;width:18px;height:18px;border-radius:50%;background:var(--gem);border:2px solid white;box-shadow:0 1px 3px #000}
.actions{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin-top:10px;min-height:34px}.action{padding:8px 11px;border-radius:8px}.action.secondary{background:#74684c}.instruction{font-weight:650;margin-right:4px}.status{display:flex;justify-content:space-between;gap:10px;font-size:1.05rem}.history{max-height:250px;overflow:auto}.move{display:grid;grid-template-columns:42px 1fr auto;gap:8px;border-bottom:1px solid #ddd2b8;padding:7px;cursor:pointer}.move:hover{background:#f3eddf}.error{color:#a21c2b;min-height:1.2em}.legend{display:flex;flex-wrap:wrap;gap:8px;font-size:.8rem;margin-top:9px}.legend span{padding:4px 7px;border-radius:5px;background:#eee5d4}
@media(max-width:1100px){.counts{grid-template-columns:repeat(6,1fr)}}
@media(max-width:700px){.shell{padding:10px}.counts{grid-template-columns:repeat(3,1fr)}}
</style>
</head>
<body><div class="shell">
<h1>Spirits of the Forest</h1><p class="subtitle">Meeple Bots · 2 jugadores · sin fichas de favor</p>
<section class="panel config">
<label>Jugador 1<select id="player-0"><option value="human">Humano</option><option value="random">Random</option><option value="mcts">MCTS</option></select></label>
<label>Jugador 2<select id="player-1"><option value="mcts">MCTS</option><option value="human">Humano</option><option value="random">Random</option></select></label>
<label><input id="save-trace" type="checkbox" checked> Guardar JSONL</label>
<label>Semilla<input id="seed" type="number" min="0" value="0"></label>
<label>Ritmo (s)<input id="pace" type="number" min="0" max="10" step="0.1" value="0.4"></label>
<button id="start">Nueva partida</button><div class="error" id="error"></div>
</section>
<section class="mcts-configs">
<div class="mcts-config" id="mcts-config-0" hidden>
<strong>MCTS · Jugador 1</strong>
<label>Presupuesto<select id="budget-mode-0"><option value="time">Tiempo</option><option value="iterations">Iteraciones</option></select></label>
<label id="time-budget-label-0">Tiempo por decisión (s)<input id="time-budget-0" type="number" min="0.001" step="0.1" value="1"></label>
<label id="iterations-label-0" hidden>Iteraciones por decisión<input id="iterations-0" type="number" min="1" value="500"></label>
<label>Profundidad<input id="depth-0" type="number" min="1" value="130"></label>
<label>Exploración<input id="exploration-0" type="number" min="0" step="0.1" value="1"></label>
<label>Heurística<select id="heuristic-0"><option value="0">H0 · Progreso alcanzable</option><option value="none" selected>Ninguna</option></select></label>
<label><input id="tree-reuse-0" type="checkbox" checked> Reutilizar árbol</label>
</div>
<div class="mcts-config" id="mcts-config-1">
<strong>MCTS · Jugador 2</strong>
<label>Presupuesto<select id="budget-mode-1"><option value="time">Tiempo</option><option value="iterations">Iteraciones</option></select></label>
<label id="time-budget-label-1">Tiempo por decisión (s)<input id="time-budget-1" type="number" min="0.001" step="0.1" value="1"></label>
<label id="iterations-label-1" hidden>Iteraciones por decisión<input id="iterations-1" type="number" min="1" value="500"></label>
<label>Profundidad<input id="depth-1" type="number" min="1" value="130"></label>
<label>Exploración<input id="exploration-1" type="number" min="0" step="0.1" value="1"></label>
<label>Heurística<select id="heuristic-1"><option value="0">H0 · Progreso alcanzable</option><option value="none" selected>Ninguna</option></select></label>
<label><input id="tree-reuse-1" type="checkbox" checked> Reutilizar árbol</label>
</div>
</section>
<section class="players" id="players"></section>
<section class="panel">
<div class="status"><strong id="status">Configura la partida</strong><span id="phase"></span></div>
<div class="forest-wrap"><div class="forest" id="forest"></div></div>
<div class="actions" id="actions"></div>
<div class="legend"><span>☀ Sol</span><span>☾ Luna</span><span>🔥 Fuego</span><span>Contadores: obtenido/total</span><span>Gema azul: J1</span><span>Gema roja: J2</span></div>
</section>
<section class="panel"><strong>Historial</strong><div class="history" id="history"></div></section>
</div>
<script>
const spiritDefinitions=[
  {key:'moss',label:'Musgo',total:5,color:'#65a95b'},
  {key:'flowers',label:'Flores',total:6,color:'#9ca3a8'},
  {key:'fruits',label:'Frutas',total:6,color:'#d85c57'},
  {key:'mushrooms',label:'Setas',total:7,color:'#9a7156'},
  {key:'water',label:'Agua',total:7,color:'#67a7d8'},
  {key:'vines',label:'Vides',total:8,color:'#e38b45'},
  {key:'branches',label:'Ramas',total:8,color:'#d2b62e'},
  {key:'leaves',label:'Hojas',total:8,color:'#e797ae'},
  {key:'webs',label:'Telarañas',total:10,color:'#a27ac3'},
];
const sourceDefinitions=[
  {key:'fire',label:'🔥',total:9,color:'#c74731'},
  {key:'moon',label:'☾',total:9,color:'#7767a8'},
  {key:'sun',label:'☀',total:9,color:'#d2a91f'},
];
const definitions=[...spiritDefinitions,...sourceDefinitions];
const colors=Object.fromEntries(spiritDefinitions.map(item=>[item.key,item.color]));
const labels=Object.fromEntries(spiritDefinitions.map(item=>[item.key,item.label]));
const sources={sun:'☀',moon:'☾',fire:'🔥'};

function cellIndex(row,column){return row*12+column}
function takeActionsAt(actions,index){return actions.filter(action=>action.kind==='take_tile'&&cellIndex(action.row,action.column)===index)}
function placeActionAt(actions,index){return actions.find(action=>action.kind==='place_gemstone'&&cellIndex(action.row,action.column)===index)}
function moveSources(actions){return new Set(actions.filter(action=>action.kind==='move_gemstone').map(action=>cellIndex(action.source_row,action.source_column)))}
function reduceBoardClick(actions,selection,index){
  if(selection?.mode==='sacrifice'){
    const action=takeActionsAt(actions,selection.target).find(candidate=>candidate.sacrifice?.kind==='forest'&&cellIndex(candidate.sacrifice.row,candidate.sacrifice.column)===index);
    return action?{actionIndex:action.index,selection:null}:{actionIndex:null,selection};
  }
  if(selection?.mode==='move'){
    const action=actions.find(candidate=>candidate.kind==='move_gemstone'&&cellIndex(candidate.source_row,candidate.source_column)===selection.source&&cellIndex(candidate.target_row,candidate.target_column)===index);
    if(action)return{actionIndex:action.index,selection:null};
    if(moveSources(actions).has(index))return{actionIndex:null,selection:{mode:'move',source:index}};
    return{actionIndex:null,selection};
  }
  const takes=takeActionsAt(actions,index);
  if(takes.some(action=>action.sacrifice))return{actionIndex:null,selection:{mode:'sacrifice',target:index}};
  if(takes.length===1)return{actionIndex:takes[0].index,selection:null};
  const place=placeActionAt(actions,index);
  if(place)return{actionIndex:place.index,selection:null};
  if(moveSources(actions).has(index))return{actionIndex:null,selection:{mode:'move',source:index}};
  return{actionIndex:null,selection:null};
}
function reconcileSelection(actions,selection){
  if(selection?.mode==='sacrifice'&&takeActionsAt(actions,selection.target).some(action=>action.sacrifice))return selection;
  if(selection?.mode==='move'&&moveSources(actions).has(selection.source))return selection;
  return null;
}
function interactionCells(actions,selection){
  const clickable=new Set(),sourceOptions=new Set(),paymentOptions=new Set(),destinations=new Set(),selected=new Set();
  if(selection?.mode==='sacrifice'){
    selected.add(selection.target);
    for(const action of takeActionsAt(actions,selection.target))if(action.sacrifice?.kind==='forest'){
      const index=cellIndex(action.sacrifice.row,action.sacrifice.column);clickable.add(index);paymentOptions.add(index);
    }
  }else if(selection?.mode==='move'){
    for(const source of moveSources(actions)){clickable.add(source);sourceOptions.add(source)}
    selected.add(selection.source);
    for(const action of actions)if(action.kind==='move_gemstone'&&cellIndex(action.source_row,action.source_column)===selection.source){
      const target=cellIndex(action.target_row,action.target_column);clickable.add(target);destinations.add(target);
    }
  }else{
    for(const action of actions){
      if(action.kind==='take_tile'||action.kind==='place_gemstone')clickable.add(cellIndex(action.row,action.column));
    }
    for(const source of moveSources(actions)){clickable.add(source);sourceOptions.add(source)}
  }
  return{clickable,sourceOptions,paymentOptions,destinations,selected};
}
function availableSacrifice(actions,selection){
  if(selection?.mode!=='sacrifice')return null;
  return takeActionsAt(actions,selection.target).find(action=>action.sacrifice?.kind==='available')||null;
}
function counterEntries(collection){
  const values=[...(collection?.spirit_symbols||Array(9).fill(0)),...(collection?.power_sources||Array(3).fill(0))];
  return definitions.map((definition,index)=>({...definition,value:values[index]}));
}

const spiritsGuiLogic={cellIndex,reduceBoardClick,reconcileSelection,interactionCells,availableSacrifice,counterEntries};
if(typeof module!=='undefined'&&module.exports)module.exports=spiritsGuiLogic;

function initializeGui(){
  let state=null,review=null,pending=false,selection=null;
  const forest=document.querySelector('#forest');
  for(let i=0;i<48;i++){
    const button=document.createElement('button');button.className='tile empty';button.addEventListener('click',()=>chooseTile(i));forest.appendChild(button);
  }
  async function api(path,options={}){const response=await fetch(path,options),payload=await response.json();if(!response.ok)throw Error(payload.error||'Error');return payload}
  function playerConfig(index){
    const kind=document.querySelector(`#player-${index}`).value;
    if(kind!=='mcts')return{kind};
    const heuristic=document.querySelector(`#heuristic-${index}`).value,mode=document.querySelector(`#budget-mode-${index}`).value;
    return{kind,iterations:mode==='iterations'?Number(document.querySelector(`#iterations-${index}`).value):null,time_budget:mode==='time'?Number(document.querySelector(`#time-budget-${index}`).value):null,exploration:Number(document.querySelector(`#exploration-${index}`).value),rollout_depth:Number(document.querySelector(`#depth-${index}`).value),heuristic:heuristic==='none'?null:Number(heuristic),tree_reuse:document.querySelector(`#tree-reuse-${index}`).checked};
  }
  function updateBudget(index){const timed=document.querySelector(`#budget-mode-${index}`).value==='time';document.querySelector(`#time-budget-label-${index}`).hidden=!timed;document.querySelector(`#iterations-label-${index}`).hidden=timed}
  function updateMctsConfig(index){document.querySelector(`#mcts-config-${index}`).hidden=document.querySelector(`#player-${index}`).value!=='mcts'}
  for(const index of [0,1]){document.querySelector(`#player-${index}`).onchange=()=>updateMctsConfig(index);document.querySelector(`#budget-mode-${index}`).onchange=()=>updateBudget(index);updateMctsConfig(index);updateBudget(index)}
  document.querySelector('#start').onclick=async()=>{try{review=null;selection=null;document.querySelector('#error').textContent='';state=await api('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({first:playerConfig(0),second:playerConfig(1),seed:Number(document.querySelector('#seed').value),minimum_move_seconds:Number(document.querySelector('#pace').value),save_trace:document.querySelector('#save-trace').checked})});render()}catch(error){document.querySelector('#error').textContent=error.message}};
  setInterval(async()=>{if(!state||!['playing','waiting_human'].includes(state.status))return;try{state=await api('/api/state');render()}catch(error){}},250);
  function frame(){if(review===null)return state;if(review===0)return state.initial||state;return state.moves[review-1]}
  function liveActions(){return review===null&&state?.status==='waiting_human'?(state.legal_actions||[]):[]}
  function chooseTile(index){
    if(pending)return;
    const result=reduceBoardClick(liveActions(),selection,index);selection=result.selection;
    if(result.actionIndex!==null)play(result.actionIndex);else render();
  }
  async function play(index){
    if(pending)return;pending=true;selection=null;document.querySelector('#error').textContent='';
    try{state=await api('/api/move',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:index})});render()}catch(error){document.querySelector('#error').textContent=error.message}
    pending=false;
  }
  function render(){
    if(!state)return;
    const current=frame(),actions=liveActions();selection=reconcileSelection(actions,selection);
    document.querySelector('#status').textContent=review===null?(state.trace_error?`${state.message} · Error al guardar: ${state.trace_error}`:state.trace_path?`${state.message} · Guardada en ${state.trace_path}`:state.message):`Revisando acción ${review}`;
    document.querySelector('#phase').textContent=current.phase==='collect'?'Recogida':'Gemas';
    renderPlayers(current);renderForest(current,actions);renderActions(actions);renderHistory();
  }
  function renderPlayers(current){
    document.querySelector('#players').innerHTML=[0,1].map(player=>{
      const collection=current.collections?.[player]||{spirit_symbols:Array(9).fill(0),power_sources:Array(3).fill(0),tiles:0};
      const gems=current.gemstone_pools?.[player]||{available:3,placed:0,removed:0};
      const counts=counterEntries(collection).map(item=>`<span class="count" style="--counter:${item.color}"><span><i></i>${item.label}</span><b>${item.value}/${item.total}</b></span>`).join('');
      return`<div class="panel player ${review===null&&current.active_player===player?'active':''}" style="--color:${player?'var(--second)':'var(--first)'}"><b>Jugador ${player+1}</b><span class="score"> ${current.scores?.[player]??0} pts</span><div>${collection.tiles} losetas · gemas ${gems.available} disponibles / ${gems.placed} bosque / ${gems.removed} retiradas</div><div class="counts">${counts}</div></div>`;
    }).join('');
  }
  function renderForest(current,actions){
    const view=interactionCells(actions,selection);
    for(let index=0;index<48;index++){
      const element=forest.children[index],tile=current.forest[index];element.className='tile';element.title='';
      if(!tile){element.classList.add('empty');element.innerHTML='';continue}
      element.style.setProperty('--tile',colors[tile.spirit]);
      element.classList.toggle('legal',view.clickable.has(index));
      element.classList.toggle('source-option',view.sourceOptions.has(index));
      element.classList.toggle('payment-option',view.paymentOptions.has(index));
      element.classList.toggle('selected',view.selected.has(index));
      if(view.paymentOptions.has(index))element.title='Sacrificar esta gema propia';
      else if(view.sourceOptions.has(index))element.title='Seleccionar esta gema para moverla';
      else if(view.destinations.has(index))element.title='Mover la gema seleccionada aquí';
      const gem=tile.gemstone==null?'':`<i class="gem" style="--gem:${tile.gemstone?'var(--second)':'var(--first)'}"></i>`;
      element.innerHTML=`${gem}<strong>${labels[tile.spirit]}</strong><div class="symbols">${'●'.repeat(tile.spirit_symbols)}</div><span class="source">${sources[tile.power_source]||''}</span>`;
    }
  }
  function actionLabel(action){
    if(action.kind==='take_tile'){let label=`Tomar ${action.row+1}:${action.column+1}`;if(action.sacrifice)label+=action.sacrifice.kind==='available'?' · retirar gema disponible':` · retirar gema propia ${action.sacrifice.row+1}:${action.sacrifice.column+1}`;return label}
    if(action.kind==='end_collection')return'Terminar recogida';
    if(action.kind==='place_gemstone')return`Poner gema ${action.row+1}:${action.column+1}`;
    if(action.kind==='move_gemstone')return`Mover ${action.source_row+1}:${action.source_column+1} → ${action.target_row+1}:${action.target_column+1}`;
    return'No usar gema';
  }
  function addActionButton(box,label,onClick,secondary=false){const button=document.createElement('button');button.className=`action${secondary?' secondary':''}`;button.textContent=label;button.onclick=onClick;box.appendChild(button)}
  function addInstruction(box,message){const text=document.createElement('span');text.className='instruction';text.textContent=message;box.appendChild(text)}
  function renderActions(actions){
    const box=document.querySelector('#actions');box.innerHTML='';
    if(review!==null){addActionButton(box,'Volver al directo',()=>{review=null;selection=null;render()},true);return}
    if(selection?.mode==='sacrifice'){
      addInstruction(box,'Reserva rival: elige una gema propia naranja para retirarla de la partida.');
      const available=availableSacrifice(actions,selection);if(available)addActionButton(box,'Retirar una gema disponible',()=>play(available.index));
      addActionButton(box,'Cancelar selección',()=>{selection=null;render()},true);return;
    }
    if(selection?.mode==='move'){
      addInstruction(box,'Elige en verde el destino de la gema seleccionada; otra gema azul cambia el origen.');
      addActionButton(box,'Cancelar selección',()=>{selection=null;render()},true);return;
    }
    if(actions.some(action=>action.kind==='move_gemstone'))addInstruction(box,'Selecciona una gema propia azul y después su destino.');
    for(const action of actions.filter(action=>action.kind==='end_collection'||action.kind==='skip_gemstone'))addActionButton(box,actionLabel(action),()=>play(action.index),true);
  }
  window.reviewSpiritsMove=move=>{review=move;selection=null;render()};
  function renderHistory(){
    const start='<div class="move" onclick="reviewSpiritsMove(0)"><span>0</span><span>Posición inicial</span><small></small></div>';
    document.querySelector('#history').innerHTML=start+(state.moves.length?state.moves.map(move=>`<div class="move" onclick="reviewSpiritsMove(${move.ply})"><span>${move.ply}</span><span>J${move.player+1} · ${actionLabel(move)}</span><small>${(move.decision_seconds||0).toFixed(3)} s</small></div>`).reverse().join(''):'');
  }
}
if(typeof document!=='undefined')initializeGui();
</script>
</body></html>"""
