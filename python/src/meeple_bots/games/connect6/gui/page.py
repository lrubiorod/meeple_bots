"""Self-contained browser page for the Connect6 GUI."""

import json

from ....gui.baselines import CONNECT6_BASELINE

PAGE = r"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Meeple Bots · Connect6</title>
  <style>
    :root {
      color-scheme: dark;
      --ink: #f6f3ea;
      --muted: #a8adb7;
      --panel: rgba(23, 28, 39, .86);
      --line: rgba(255, 255, 255, .1);
      --first: #ffb45c;
      --second: #69d5c7;
      --accent: #ffd166;
      --board: #263c78;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(circle at 15% 10%, rgba(105, 213, 199, .15), transparent 34rem),
        radial-gradient(circle at 85% 85%, rgba(255, 180, 92, .13), transparent 30rem),
        #0d1119;
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: .18;
      background-image: linear-gradient(var(--line) 1px, transparent 1px),
                        linear-gradient(90deg, var(--line) 1px, transparent 1px);
      background-size: 42px 42px;
      mask-image: linear-gradient(to bottom, black, transparent 80%);
    }
    .shell { width: min(1280px, calc(100% - 32px)); margin: 0 auto; padding: 34px 0; }
    header { display: flex; align-items: end; justify-content: space-between; gap: 24px; margin-bottom: 26px; }
    .eyebrow { color: var(--accent); letter-spacing: .18em; text-transform: uppercase; font-size: .72rem; font-weight: 800; }
    h1 { margin: 6px 0 0; font-family: Georgia, serif; font-size: clamp(2rem, 5vw, 4.25rem); line-height: .95; font-weight: 500; }
    .connection { color: var(--muted); font-size: .86rem; display: flex; align-items: center; gap: 8px; }
    .dot { width: 9px; height: 9px; border-radius: 99px; background: #7ce38b; box-shadow: 0 0 14px #7ce38b; }
    main { display: grid; grid-template-columns: minmax(280px, .82fr) minmax(430px, 1.55fr) minmax(250px, .72fr); gap: 18px; align-items: start; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 22px; padding: 22px; box-shadow: 0 18px 60px rgba(0,0,0,.2); backdrop-filter: blur(16px); }
    .panel-title { margin: 0 0 18px; font-size: .78rem; text-transform: uppercase; letter-spacing: .14em; color: var(--muted); }
    .players { display: grid; gap: 13px; }
    .player { border: 1px solid var(--line); border-radius: 16px; padding: 15px; transition: border-color .2s, background .2s; }
    .player.active { border-color: color-mix(in srgb, var(--player-color) 68%, transparent); background: color-mix(in srgb, var(--player-color) 8%, transparent); }
    .player-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .player-name { font-weight: 750; }
    .token { width: 25px; height: 25px; border-radius: 50%; background: var(--player-color); box-shadow: 0 0 18px color-mix(in srgb, var(--player-color) 35%, transparent); }
    label { display: block; margin-top: 11px; color: var(--muted); font-size: .77rem; }
    select, input[type="number"] { width: 100%; margin-top: 5px; border: 1px solid var(--line); border-radius: 10px; padding: 9px 10px; background: #101621; color: var(--ink); font: inherit; }
    .mcts-options { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
    .mcts-options.hidden { display: none; }
    .budget-field.hidden { display: none; }
    .pace { margin: 18px 0; }
    .pace-line { display: flex; justify-content: space-between; color: var(--muted); font-size: .8rem; }
    input[type="range"] { width: 100%; accent-color: var(--accent); }
    .seed-row { display: grid; grid-template-columns: 1fr auto; gap: 9px; align-items: end; }
    button { font: inherit; }
    .start { width: 100%; border: 0; border-radius: 12px; padding: 12px 16px; background: var(--accent); color: #201804; font-weight: 850; cursor: pointer; }
    .start:hover { filter: brightness(1.06); }
    .arena { text-align: center; padding-bottom: 28px; }
    .status { min-height: 48px; }
    .status strong { display: block; font-size: 1.18rem; }
    .status span { color: var(--muted); font-size: .78rem; }
    .column-controls { width: min(100%, 610px); margin: 16px auto 5px; display: grid; grid-template-columns: repeat(7, 1fr); gap: 6px; padding: 0 11px; }
    .column-button { height: 34px; border: 0; border-radius: 9px; color: var(--muted); background: transparent; cursor: default; transition: color .14s, background .14s, transform .14s; }
    .column-button.legal { color: var(--accent); cursor: pointer; }
    .column-button.legal:hover { transform: translateY(3px); background: rgba(255,209,102,.08); }
    .board { width: min(100%, 610px); aspect-ratio: 7 / 6; margin: 0 auto; padding: 11px; display: grid; grid-template-columns: repeat(7, 1fr); grid-template-rows: repeat(6, 1fr); gap: 7px; border: 1px solid rgba(130,165,255,.28); border-radius: 22px; background: linear-gradient(145deg, #2d478f, #172852); box-shadow: inset 0 0 35px rgba(0,0,0,.25), 0 22px 50px rgba(0,0,0,.25); }
    .cell { position: relative; min-width: 0; border: 0; border-radius: 50%; background: rgba(8,12,19,.82); box-shadow: inset 0 5px 12px rgba(0,0,0,.55), 0 1px 0 rgba(255,255,255,.09); cursor: default; transition: transform .14s, box-shadow .14s; }
    .cell.first { background: radial-gradient(circle at 36% 30%, #ffd39d, var(--first) 45%, #b96717); }
    .cell.second { background: radial-gradient(circle at 36% 30%, #b8fff5, var(--second) 45%, #187b73); }
    .cell.legal { cursor: pointer; }
    .cell.legal:hover, .cell.column-hover { box-shadow: inset 0 0 0 3px rgba(255,209,102,.6), inset 0 5px 12px rgba(0,0,0,.45); transform: scale(.94); }
    .cell.last::after { content: ""; position: absolute; inset: 17%; border: 3px solid rgba(255,255,255,.65); border-radius: 50%; }
    .legend { display: flex; justify-content: center; gap: 20px; margin-top: 18px; color: var(--muted); font-size: .8rem; }
    .legend span { display: flex; align-items: center; gap: 7px; }
    .mini-token { width: 10px; height: 10px; border-radius: 50%; background: var(--color); }
    .thinking::after { content: ""; display: inline-block; width: 7px; height: 7px; margin-left: 8px; border-radius: 50%; background: var(--accent); animation: pulse 1s infinite alternate; }
    @keyframes pulse { to { opacity: .25; transform: scale(.7); } }
    .history { max-height: 510px; overflow: auto; }
    .move { display: grid; grid-template-columns: 30px 1fr auto; align-items: center; gap: 10px; padding: 11px 0; border-bottom: 1px solid var(--line); }
    .move:last-child { border-bottom: 0; }
    .move-number { color: var(--muted); font-size: .72rem; }
    .move-action { display: flex; align-items: center; gap: 8px; }
    .move-action b { width: 15px; height: 15px; border-radius: 50%; background: var(--move-color); }
    .move-action small, .move-time { color: var(--muted); font-size: .72rem; }
    .empty { color: var(--muted); font-size: .86rem; padding: 22px 0; text-align: center; }
    .error { color: #ff8b8b; margin-top: 10px; min-height: 1.2em; font-size: .8rem; }
    @media (max-width: 1050px) { main { grid-template-columns: minmax(260px,.8fr) minmax(430px,1.3fr); } .history-panel { grid-column: 1 / -1; } }
    @media (max-width: 720px) { .shell { width: min(100% - 20px, 620px); padding-top: 22px; } header { align-items: start; flex-direction: column; } main { grid-template-columns: 1fr; } .history-panel { grid-column: auto; } .config { order: 2; } .arena { order: 1; padding-inline: 10px; } .history-panel { order: 3; } .board { gap: 4px; padding: 7px; border-radius: 15px; } }
    .board { width:min(100%,760px); aspect-ratio:1; padding:10px; gap:1px; background:#c9a36a; border-radius:8px; }
    .cell { border-radius:0; background:linear-gradient(#765832,#765832) center/1px 100% no-repeat,linear-gradient(#765832,#765832) center/100% 1px no-repeat; box-shadow:none; aspect-ratio:1; }
    .cell.first { border-radius:50%; background:radial-gradient(circle at 35% 30%,#555,#111 65%); }
    .cell.second { border-radius:50%; background:radial-gradient(circle at 35% 30%,#fff,#ddd 65%); }
    .cell.last::after { inset:36%; background:#d74131; border:0; }
    .cell:disabled { opacity:1; }
    .cell.legal:hover { background-color:#ffffff60; transform:none; }
    :root { --first:#333; --second:#eee; }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div><div class="eyebrow">Meeple Bots · Playroom</div><h1>Connect6</h1></div>
      <div class="connection"><span class="dot"></span> Motor local conectado</div>
    </header>
    <main>
      <section class="panel config">
        <h2 class="panel-title">Configurar partida</h2>
        <div class="players">
          <div class="player" id="player-card-0" style="--player-color:var(--first)">
            <div class="player-head"><span class="player-name">Negras</span><span class="token"></span></div>
            <label>Control<select id="player-0"><option value="human">Humano</option><option value="mcts">MCTS</option><option value="random">Random</option></select></label>
            <div class="mcts-options hidden" id="mcts-0"><label>Selección<select id="selection-policy-0"><option value="uct">UCT</option><option value="uct_rave">UCT-RAVE</option><option value="ucb1_tuned">UCB1-Tuned (ignora exploración)</option></select></label><label class="hidden" id="rave-label-0" title="Valores mayores mantienen el peso de AMAF durante más visitas">Equivalencia RAVE (k)<input id="rave-equivalence-0" type="number" min="1" step="1" value="1000"></label><label>Presupuesto<select id="budget-mode-0"><option value="iterations">Iteraciones</option><option value="time">Tiempo</option></select></label><label class="budget-field" id="iterations-label-0">Iteraciones<input id="iterations-0" type="number" min="1" value="1000"></label><label class="budget-field hidden" id="time-label-0">Tiempo por decisión (s)<input id="time-budget-0" type="number" min="0.001" step="0.1" value="1"></label><label>Profundidad<input id="depth-0" type="number" min="1" value="64"></label><label>Exploración<input id="exploration-0" type="number" min="0" step="0.1"></label><label><input id="transpositions-0" type="checkbox"> Transposiciones</label><label><input id="tree-reuse-0" type="checkbox"> Reutilizar árbol</label><label title="Limita las acciones abiertas y amplía el árbol a medida que recibe visitas"><input id="progressive-widening-0" type="checkbox"> Progressive Widening (PW)</label><label class="hidden" id="pw-k-label-0" title="k positivo: controla cuántas acciones se abren">Amplitud PW (k)<input id="pw-k-0" type="number" min="0.001" step="any" value="1.5"></label><label class="hidden" id="pw-alpha-label-0" title="0 &lt; alpha ≤ 1: controla el crecimiento con las visitas">Crecimiento PW (alpha)<input id="pw-alpha-0" type="number" min="0.001" max="1" step="any" value="0.5"></label></div>
          </div>
          <div class="player" id="player-card-1" style="--player-color:var(--second)">
            <div class="player-head"><span class="player-name">Blancas</span><span class="token"></span></div>
            <label>Control<select id="player-1"><option value="mcts">MCTS</option><option value="human">Humano</option><option value="random">Random</option></select></label>
            <div class="mcts-options" id="mcts-1"><label>Selección<select id="selection-policy-1"><option value="uct">UCT</option><option value="uct_rave">UCT-RAVE</option><option value="ucb1_tuned">UCB1-Tuned (ignora exploración)</option></select></label><label class="hidden" id="rave-label-1" title="Valores mayores mantienen el peso de AMAF durante más visitas">Equivalencia RAVE (k)<input id="rave-equivalence-1" type="number" min="1" step="1" value="1000"></label><label>Presupuesto<select id="budget-mode-1"><option value="iterations">Iteraciones</option><option value="time">Tiempo</option></select></label><label class="budget-field" id="iterations-label-1">Iteraciones<input id="iterations-1" type="number" min="1" value="1000"></label><label class="budget-field hidden" id="time-label-1">Tiempo por decisión (s)<input id="time-budget-1" type="number" min="0.001" step="0.1" value="1"></label><label>Profundidad<input id="depth-1" type="number" min="1" value="64"></label><label>Exploración<input id="exploration-1" type="number" min="0" step="0.1"></label><label><input id="transpositions-1" type="checkbox"> Transposiciones</label><label><input id="tree-reuse-1" type="checkbox"> Reutilizar árbol</label><label title="Limita las acciones abiertas y amplía el árbol a medida que recibe visitas"><input id="progressive-widening-1" type="checkbox"> Progressive Widening (PW)</label><label class="hidden" id="pw-k-label-1" title="k positivo: controla cuántas acciones se abren">Amplitud PW (k)<input id="pw-k-1" type="number" min="0.001" step="any" value="1.5"></label><label class="hidden" id="pw-alpha-label-1" title="0 &lt; alpha ≤ 1: controla el crecimiento con las visitas">Crecimiento PW (alpha)<input id="pw-alpha-1" type="number" min="0.001" max="1" step="any" value="0.5"></label></div>
          </div>
        </div>
        <div class="pace"><div class="pace-line"><span>Intervalo mínimo entre jugadas</span><b id="pace-value">0.6 s</b></div><input id="pace" type="range" min="0" max="3" step="0.1" value="0.6"></div>
        <label><input id="save-trace" type="checkbox" checked> Guardar JSONL para análisis posterior</label>
        <label>Tamaño del tablero<input id="board-size" type="number" min="6"></label><div class="seed-row"><label>Semilla<input id="seed" type="number" min="0" value="0"></label><button class="start" id="start">Nueva partida</button></div>
        <div class="error" id="error"></div>
      </section>
      <section class="panel arena">
        <div class="status"><strong id="status">Configura y comienza una partida</strong><span id="timing"></span></div>
        <p>Conecta 6 · Negras abre con 1 piedra; después, 2 por turno.</p>
        <div class="board" id="board"></div>
        <div class="legend"><span><i class="mini-token" style="--color:var(--first)"></i>Negras</span><span><i class="mini-token" style="--color:var(--second)"></i>Blancas</span></div>
      </section>
      <section class="panel history-panel">
        <h2 class="panel-title">Historial en vivo</h2>
        <div class="history" id="history"><div class="empty">Las jugadas aparecerán aquí.</div></div>
      </section>
    </main>
  </div>
  <script>
    const board = document.querySelector('#board');

    const errorBox = document.querySelector('#error');
    let state = null;
    let requestPending = false;
    let revision = 0;
    let initialized = false;

    function ensureBoard() {
      const n=state.board_size;
      if(board.children.length===n*n) return;
      board.replaceChildren();
      board.style.gridTemplateColumns=`repeat(${n}, 1fr)`;
      board.style.gridTemplateRows=`repeat(${n}, 1fr)`;
      for(let p=0;p<n*n;p++) {
        const cell=document.createElement('button');
        cell.className='cell'; cell.title=`Fila ${Math.floor(p/n)+1}, columna ${p%n+1}`;
        cell.setAttribute('aria-label',cell.title);
        cell.addEventListener('click',()=>play(p)); board.appendChild(cell);
      }
    }
    for (const player of [0, 1]) {
      document.querySelector(`#player-${player}`).addEventListener('change', updateAgentFields);
      document.querySelector(`#selection-policy-${player}`).addEventListener('change', updateAgentFields);
      document.querySelector(`#progressive-widening-${player}`).addEventListener('change', updateAgentFields);
    }
    for (const player of [0, 1]) document.querySelector(`#budget-mode-${player}`).addEventListener('change', () => updateBudgetFields(player));
    const pace = document.querySelector('#pace');
    pace.addEventListener('input', () => document.querySelector('#pace-value').textContent = `${Number(pace.value).toFixed(1)} s`);
    document.querySelector('#start').addEventListener('click', start);

    function playerConfig(index) {
      const kind = document.querySelector(`#player-${index}`).value;
      if (kind !== 'mcts') return {kind};
      const timed = document.querySelector(`#budget-mode-${index}`).value === 'time';
      const selection = document.querySelector(`#selection-policy-${index}`).value;
      return {kind, iterations:timed?null:Number(document.querySelector(`#iterations-${index}`).value), time_budget:timed?Number(document.querySelector(`#time-budget-${index}`).value):null, rollout_depth:Number(document.querySelector(`#depth-${index}`).value), selection_policy:selection, ...(selection === 'uct_rave' ? {rave_equivalence:Number(document.querySelector(`#rave-equivalence-${index}`).value)} : {}), exploration:Number(document.querySelector(`#exploration-${index}`).value), transpositions:document.querySelector(`#transpositions-${index}`).checked, tree_reuse:document.querySelector(`#tree-reuse-${index}`).checked, progressive_widening:document.querySelector(`#progressive-widening-${index}`).checked, progressive_widening_k:Number(document.querySelector(`#pw-k-${index}`).value), progressive_widening_alpha:Number(document.querySelector(`#pw-alpha-${index}`).value)};
    }
    function updateBudgetFields(index) { const timed=document.querySelector(`#budget-mode-${index}`).value==='time'; document.querySelector(`#iterations-label-${index}`).classList.toggle('hidden',timed); document.querySelector(`#time-label-${index}`).classList.toggle('hidden',!timed); }
    function updateAgentFields() {
      for (const player of [0, 1]) {
        document.querySelector(`#mcts-${player}`).classList.toggle('hidden', document.querySelector(`#player-${player}`).value !== 'mcts');
        const selection = document.querySelector(`#selection-policy-${player}`).value;
        document.querySelector(`#exploration-${player}`).disabled = selection === 'ucb1_tuned';
        document.querySelector(`#rave-label-${player}`).classList.toggle('hidden', selection !== 'uct_rave');
        const pw = document.querySelector(`#progressive-widening-${player}`).checked;
        document.querySelector(`#pw-k-label-${player}`).classList.toggle('hidden', !pw);
        document.querySelector(`#pw-alpha-label-${player}`).classList.toggle('hidden', !pw);
      }
    }
    async function api(path, options = {}) {
      const response = await fetch(path, options);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'Error desconocido');
      return payload;
    }
    async function start() {
      if(requestPending) return;
      requestPending=true; ++revision;
      errorBox.textContent = '';
      try {
        state = await api('/api/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({board_size:Number(document.querySelector('#board-size').value), first:playerConfig(0), second:playerConfig(1), seed:Number(document.querySelector('#seed').value), minimum_move_seconds:Number(pace.value), save_trace:document.querySelector('#save-trace').checked})});
        render();
      } catch (error) { errorBox.textContent = error.message; }
      finally { requestPending=false; }
    }
    async function play(column) {
      if (requestPending || !isLegal(column)) return;
      requestPending = true; ++revision;
      errorBox.textContent = '';
      try {
        state = await api('/api/move', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({position:column, turn:state.moves.length, session_id:state.session_id})});
        render();
      } catch (error) { errorBox.textContent = error.message; }
      requestPending = false;
    }
    function isLegal(column) { return state?.status === 'waiting_human' && state.legal_actions.includes(column); }
    function render() {
      if (!state) return;
      ensureBoard();
      if(!initialized) { document.querySelector('#board-size').value=state.board_size; initialized=true; }
      document.querySelector('#status').textContent = translateMessage(state);
      document.querySelector('#status').classList.toggle('thinking', state.status === 'playing');
      const seconds = state.last_decision_seconds;
      document.querySelector('#timing').textContent = seconds == null ? '' : `Última decisión: ${formatTime(seconds)}`;
      for(let index=0;index<state.board.length;index++) {
        const cell=board.children[index], value=state.board[index], n=state.board_size;
        cell.className=`cell ${value===0?'first':value===1?'second':''}`;
        cell.classList.toggle('legal',isLegal(index));
        cell.disabled=!isLegal(index);
        cell.classList.toggle('last', state.last_move?.[0]===Math.floor(index/n) && state.last_move?.[1]===index%n);
      }
      for (const player of [0,1]) document.querySelector(`#player-card-${player}`).classList.toggle('active', state.active_player === player);
      const history = document.querySelector('#history');
      history.innerHTML = state.moves.length ? state.moves.map(move => `<div class="move" style="--move-color:${move.player === 0 ? 'var(--first)' : 'var(--second)'}"><span class="move-number">${String(move.ply).padStart(2,'0')}</span><span class="move-action"><b></b><small>(${move.row + 1}, ${move.column + 1})</small></span><span class="move-time">${formatTime(move.decision_seconds)}</span></div>`).reverse().join('') : '<div class="empty">Las jugadas aparecerán aquí.</div>';
    }
    function translateMessage(value) {
      if (value.status === 'idle') return 'Configura y comienza una partida';
      if (value.status === 'finished') { const result=value.winner==null?'Tablas':`Gana el jugador ${value.winner+1}`; return value.trace_error?`${result} · Error al guardar: ${value.trace_error}`:value.trace_path?`${result} · Guardada en ${value.trace_path}`:result; }
      if (value.status === 'error') return `Error: ${value.message}`;
      if (value.status === 'waiting_human') return `${value.active_player===0?"Negras":"Blancas"}: coloca una piedra · quedan ${value.placements_remaining}`;
      return `Pensando: ${value.active_player===0?"negras":"blancas"} · quedan ${value.placements_remaining}`;
    }
    function formatTime(seconds) { return seconds < .001 ? `${(seconds * 1e6).toFixed(0)} µs` : seconds < 1 ? `${(seconds * 1e3).toFixed(1)} ms` : `${seconds.toFixed(2)} s`; }
    async function poll() {
      const version=revision;
      try { if(!requestPending) { const next=await api('/api/state'); if(version===revision && !requestPending) {state=next; render();} } } catch (_) {}
      window.setTimeout(poll, 120);
    }
    const baseline = __MCTS_BASELINE__;
    for (const player of [0, 1]) {
      document.querySelector(`#budget-mode-${player}`).value = baseline.time_budget === null ? 'iterations' : 'time';
    document.querySelector(`#iterations-${player}`).value = baseline.iterations ?? 1000;
    document.querySelector(`#time-budget-${player}`).value = baseline.time_budget ?? 1;
      document.querySelector(`#depth-${player}`).value = baseline.rollout_depth;
      document.querySelector(`#exploration-${player}`).value = baseline.exploration;
      document.querySelector(`#rave-equivalence-${player}`).value = baseline.rave_equivalence ?? 1000;
      document.querySelector(`#progressive-widening-${player}`).checked = baseline.progressive_widening ?? false;
      document.querySelector(`#pw-k-${player}`).value = baseline.progressive_widening_k ?? 1.5;
      document.querySelector(`#pw-alpha-${player}`).value = baseline.progressive_widening_alpha ?? 0.5;
      document.querySelector(`#transpositions-${player}`).checked = baseline.transpositions;
    document.querySelector(`#selection-policy-${player}`).value = baseline.selection_policy;
      document.querySelector(`#tree-reuse-${player}`).checked = baseline.tree_reuse;
    }
    updateAgentFields();
    for (const player of [0, 1]) updateBudgetFields(player);
    poll();
  </script>
</body>
</html>
"""

PAGE = PAGE.replace("__MCTS_BASELINE__", json.dumps(CONNECT6_BASELINE.as_dict()))
