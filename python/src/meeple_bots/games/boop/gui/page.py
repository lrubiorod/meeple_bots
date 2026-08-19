"""Self-contained browser page for the Boop GUI."""

PAGE = r"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Meeple Bots · Boop</title>
  <style>
    :root {
      color-scheme: dark;
      --ink: #f6f3ea;
      --muted: #a8adb7;
      --panel: rgba(23, 28, 39, .88);
      --line: rgba(255, 255, 255, .1);
      --first: #ffb45c;
      --first-dark: #a85816;
      --second: #69d5c7;
      --second-dark: #187b73;
      --accent: #ffd166;
      --bed-a: #2d4168;
      --bed-b: #263958;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(circle at 13% 8%, rgba(105, 213, 199, .15), transparent 34rem),
        radial-gradient(circle at 86% 88%, rgba(255, 180, 92, .13), transparent 31rem),
        #0d1119;
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: .18;
      background-image: linear-gradient(var(--line) 1px, transparent 1px), linear-gradient(90deg, var(--line) 1px, transparent 1px);
      background-size: 42px 42px;
      mask-image: linear-gradient(to bottom, black, transparent 80%);
    }
    .shell { width: min(1420px, calc(100% - 32px)); margin: 0 auto; padding: 34px 0; }
    header { display: flex; align-items: end; justify-content: space-between; gap: 24px; margin-bottom: 26px; }
    .eyebrow { color: var(--accent); letter-spacing: .18em; text-transform: uppercase; font-size: .72rem; font-weight: 800; }
    h1 { margin: 6px 0 0; font-family: Georgia, serif; font-size: clamp(2rem, 5vw, 4.25rem); line-height: .95; font-weight: 500; }
    .connection { color: var(--muted); font-size: .86rem; display: flex; align-items: center; gap: 8px; }
    .dot { width: 9px; height: 9px; border-radius: 99px; background: #7ce38b; box-shadow: 0 0 14px #7ce38b; }
    main { display: grid; grid-template-columns: minmax(275px, .78fr) minmax(520px, 1.5fr) minmax(275px, .76fr); gap: 18px; align-items: start; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 22px; padding: 22px; box-shadow: 0 18px 60px rgba(0,0,0,.2); backdrop-filter: blur(16px); }
    .panel-title { margin: 0 0 18px; font-size: .78rem; text-transform: uppercase; letter-spacing: .14em; color: var(--muted); }
    .players { display: grid; gap: 13px; }
    .player { border: 1px solid var(--line); border-radius: 16px; padding: 15px; transition: border-color .2s, background .2s; }
    .player.active { border-color: color-mix(in srgb, var(--player-color) 68%, transparent); background: color-mix(in srgb, var(--player-color) 8%, transparent); }
    .player-head, .pool-line { display: flex; justify-content: space-between; align-items: center; }
    .player-head { margin-bottom: 9px; }
    .player-name { font-weight: 750; }
    .token { width: 25px; height: 25px; border-radius: 50%; background: var(--player-color); box-shadow: 0 0 18px color-mix(in srgb, var(--player-color) 35%, transparent); }
    .pool-line { padding: 7px 9px; margin-bottom: 8px; border-radius: 10px; background: rgba(255,255,255,.035); color: var(--muted); font-size: .74rem; }
    .pool-line b { color: var(--ink); font-size: .78rem; }
    label { display: block; margin-top: 11px; color: var(--muted); font-size: .77rem; }
    select, input[type="number"] { width: 100%; margin-top: 5px; border: 1px solid var(--line); border-radius: 10px; padding: 9px 10px; background: #101621; color: var(--ink); font: inherit; }
    .mcts-options { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
    .mcts-options .wide { grid-column: 1 / -1; }
    .mcts-options.hidden { display: none; }
    .pace { margin: 18px 0; }
    .pace-line { display: flex; justify-content: space-between; gap: 10px; color: var(--muted); font-size: .8rem; }
    input[type="range"] { width: 100%; accent-color: var(--accent); }
    .seed-row { display: grid; grid-template-columns: 1fr auto; gap: 9px; align-items: end; }
    button { font: inherit; }
    .start { border: 0; border-radius: 12px; padding: 12px 16px; background: var(--accent); color: #201804; font-weight: 850; cursor: pointer; }
    .start:hover { filter: brightness(1.06); }
    .arena { text-align: center; padding-bottom: 25px; }
    .status { min-height: 48px; }
    .status strong { display: block; font-size: 1.18rem; }
    .status span { color: var(--muted); font-size: .78rem; }
    .replay-controls { display: flex; justify-content: center; align-items: center; gap: 7px; margin: 8px 0 5px; }
    .replay-button { min-width: 38px; border: 1px solid var(--line); border-radius: 9px; padding: 6px 9px; color: var(--muted); background: rgba(255,255,255,.035); cursor: pointer; }
    .replay-button:hover:not(:disabled) { color: var(--ink); border-color: rgba(255,209,102,.58); }
    .replay-button:disabled { opacity: .3; cursor: default; }
    .replay-position { min-width: 142px; color: var(--muted); font-size: .72rem; }
    .live-button { color: var(--accent); }
    .piece-picker { display: flex; justify-content: center; gap: 9px; min-height: 47px; margin: 9px 0 14px; }
    .piece-choice { min-width: 120px; border: 1px solid var(--line); border-radius: 13px; padding: 8px 13px; color: var(--muted); background: rgba(255,255,255,.025); cursor: default; transition: .16s ease; }
    .piece-choice.available { cursor: pointer; }
    .piece-choice.selected { color: var(--ink); border-color: rgba(255,209,102,.65); background: rgba(255,209,102,.08); transform: translateY(-1px); }
    .piece-choice small { display: block; opacity: .7; font-size: .67rem; }
    .board-wrap { width: min(100%, 610px); margin: 0 auto; display: grid; grid-template-columns: 20px 1fr; grid-template-rows: 20px 1fr; gap: 6px; }
    .column-labels { grid-column: 2; display: grid; grid-template-columns: repeat(6, 1fr); color: var(--muted); font-size: .66rem; }
    .row-labels { grid-row: 2; display: grid; grid-template-rows: repeat(6, 1fr); align-items: center; color: var(--muted); font-size: .66rem; }
    .board { grid-column: 2; grid-row: 2; aspect-ratio: 1; padding: 12px; display: grid; grid-template-columns: repeat(6, 1fr); grid-template-rows: repeat(6, 1fr); gap: 7px; border: 1px solid rgba(156,181,230,.3); border-radius: 30px; background: linear-gradient(145deg, #354b76, #1d2d4b); box-shadow: inset 0 0 38px rgba(0,0,0,.28), 0 24px 55px rgba(0,0,0,.27); }
    .cell { position: relative; min-width: 0; border: 1px solid rgba(255,255,255,.055); border-radius: 15px; background: var(--bed-a); box-shadow: inset 0 1px rgba(255,255,255,.055); cursor: default; transition: transform .14s, border-color .14s, filter .14s; }
    .cell:nth-child(odd) { background: var(--bed-b); }
    .cell.legal { cursor: pointer; border-color: rgba(255,209,102,.42); }
    .cell.legal::after { content: ""; position: absolute; inset: 42%; border-radius: 50%; background: var(--accent); opacity: .62; }
    .cell.legal:hover { transform: scale(.95); border-color: var(--accent); filter: brightness(1.13); }
    .cell.last { border-color: var(--accent); box-shadow: inset 0 0 0 2px rgba(255,209,102,.5), 0 0 18px rgba(255,209,102,.34); }
    .cell.last::before { content: ""; position: absolute; inset: -5px; z-index: 3; pointer-events: none; border: 3px solid var(--accent); border-radius: inherit; animation: last-move 1.1s ease-in-out infinite alternate; }
    .cell.candidate { outline: 3px solid rgba(255,209,102,.72); outline-offset: -5px; }
    .piece { position: absolute; left: 50%; top: 52%; translate: -50% -50%; display: grid; place-items: center; border-radius: 50%; background: radial-gradient(circle at 35% 28%, var(--piece-light), var(--piece-color) 48%, var(--piece-dark)); color: rgba(17,22,31,.76); box-shadow: 0 6px 14px rgba(0,0,0,.3), inset 0 1px rgba(255,255,255,.4); font-size: .65rem; font-weight: 900; }
    .piece.first { --piece-color: var(--first); --piece-light: #ffe0b8; --piece-dark: var(--first-dark); }
    .piece.second { --piece-color: var(--second); --piece-light: #c9fff8; --piece-dark: var(--second-dark); }
    .piece.kitten { width: 52%; height: 52%; }
    .piece.cat { width: 70%; height: 70%; border: 3px solid rgba(255,255,255,.36); border-radius: 18%; font-size: .74rem; }
    .legend { display: flex; justify-content: center; flex-wrap: wrap; gap: 16px; margin-top: 16px; color: var(--muted); font-size: .76rem; }
    .legend span { display: flex; align-items: center; gap: 7px; }
    .mini-token { width: 10px; height: 10px; border-radius: 50%; background: var(--color); }
    .thinking::after { content: ""; display: inline-block; width: 7px; height: 7px; margin-left: 8px; border-radius: 50%; background: var(--accent); animation: pulse 1s infinite alternate; }
    @keyframes pulse { to { opacity: .25; transform: scale(.7); } }
    @keyframes last-move { to { opacity: .48; box-shadow: 0 0 12px rgba(255,209,102,.42); } }
    .resolution { width: min(100%, 610px); margin: 15px auto 0; padding: 14px; border: 1px solid rgba(255,209,102,.28); border-radius: 15px; background: rgba(255,209,102,.055); text-align: left; }
    .resolution.hidden { display: none; }
    .resolution-head { display: flex; justify-content: space-between; gap: 10px; align-items: center; margin-bottom: 9px; }
    .resolution-head strong { font-size: .84rem; }
    .resolution-close { border: 0; background: transparent; color: var(--muted); cursor: pointer; }
    .resolution-options { display: grid; gap: 7px; }
    .resolution-option { border: 1px solid var(--line); border-radius: 10px; padding: 9px 11px; color: var(--ink); background: rgba(255,255,255,.035); text-align: left; cursor: pointer; }
    .resolution-option:hover { border-color: var(--accent); }
    .history { max-height: 590px; overflow: auto; }
    .move { display: grid; grid-template-columns: 28px 1fr auto; align-items: start; gap: 9px; padding: 11px 7px; border-bottom: 1px solid var(--line); border-radius: 9px; cursor: pointer; }
    .move:hover { background: rgba(255,255,255,.035); }
    .move.reviewed { background: rgba(255,209,102,.08); box-shadow: inset 3px 0 var(--accent); }
    .move:last-child { border-bottom: 0; }
    .move-number { color: var(--muted); font-size: .7rem; padding-top: 2px; }
    .move-action { display: flex; gap: 8px; align-items: flex-start; }
    .move-action b { flex: 0 0 auto; width: 13px; height: 13px; margin-top: 2px; border-radius: 50%; background: var(--move-color); }
    .move-action small { color: var(--muted); font-size: .72rem; line-height: 1.42; }
    .move-time { color: var(--muted); font-size: .68rem; white-space: nowrap; }
    .empty { color: var(--muted); font-size: .86rem; padding: 22px 0; text-align: center; }
    .error { color: #ff8b8b; margin-top: 10px; min-height: 1.2em; font-size: .8rem; }
    @media (max-width: 1120px) { main { grid-template-columns: minmax(270px,.78fr) minmax(500px,1.35fr); } .history-panel { grid-column: 1 / -1; } }
    @media (max-width: 800px) { .shell { width: min(100% - 20px, 680px); padding-top: 22px; } header { align-items: start; flex-direction: column; } main { grid-template-columns: 1fr; } .history-panel { grid-column: auto; } .config { order: 2; } .arena { order: 1; padding-inline: 10px; } .history-panel { order: 3; } .board { gap: 4px; padding: 8px; border-radius: 20px; } .cell { border-radius: 10px; } }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div><div class="eyebrow">Meeple Bots · Playroom</div><h1>Boop</h1></div>
      <div class="connection"><span class="dot"></span> Motor local conectado</div>
    </header>
    <main>
      <section class="panel config">
        <h2 class="panel-title">Configurar partida</h2>
        <div class="players">
          <div class="player" id="player-card-0" style="--player-color:var(--first)">
            <div class="player-head"><span class="player-name">Jugador 1</span><span class="token"></span></div>
            <div class="pool-line"><span>Reserva</span><b id="pool-0">8 kittens · 0 cats</b></div>
            <label>Control<select id="player-0"><option value="human">Humano</option><option value="mcts">MCTS</option><option value="random">Random</option></select></label>
            <div class="mcts-options hidden" id="mcts-0"><label>Iteraciones<input id="iterations-0" type="number" min="1" value="1000"></label><label>Profundidad<input id="depth-0" type="number" min="1" value="15"></label><label class="wide">Heurística<select id="heuristic-0"><option value="none">Sin heurística</option><option value="0">0 · Balance de gatos</option><option value="1">1 · Estratégica</option></select></label></div>
          </div>
          <div class="player" id="player-card-1" style="--player-color:var(--second)">
            <div class="player-head"><span class="player-name">Jugador 2</span><span class="token"></span></div>
            <div class="pool-line"><span>Reserva</span><b id="pool-1">8 kittens · 0 cats</b></div>
            <label>Control<select id="player-1"><option value="mcts">MCTS</option><option value="human">Humano</option><option value="random">Random</option></select></label>
            <div class="mcts-options" id="mcts-1"><label>Iteraciones<input id="iterations-1" type="number" min="1" value="1000"></label><label>Profundidad<input id="depth-1" type="number" min="1" value="15"></label><label class="wide">Heurística<select id="heuristic-1"><option value="0">0 · Balance de gatos</option><option value="1">1 · Estratégica</option><option value="none">Sin heurística</option></select></label></div>
          </div>
        </div>
        <div class="pace"><div class="pace-line"><span>Intervalo mínimo entre jugadas</span><b id="pace-value">0.6 s</b></div><input id="pace" type="range" min="0" max="3" step="0.1" value="0.6"></div>
        <div class="seed-row"><label>Semilla<input id="seed" type="number" min="0" value="0"></label><button class="start" id="start">Nueva partida</button></div>
        <div class="error" id="error"></div>
      </section>
      <section class="panel arena">
        <div class="status"><strong id="status">Configura y comienza una partida</strong><span id="timing"></span></div>
        <div class="replay-controls">
          <button class="replay-button" id="previous-move" title="Movimiento anterior">←</button>
          <span class="replay-position" id="replay-position">En directo</span>
          <button class="replay-button" id="next-move" title="Movimiento siguiente">→</button>
          <button class="replay-button live-button" id="live-move">● Directo</button>
        </div>
        <div class="piece-picker">
          <button class="piece-choice selected" id="choose-kitten" data-piece="kitten">Kitten<small id="kitten-count">elige una casilla</small></button>
          <button class="piece-choice" id="choose-cat" data-piece="cat">Cat<small id="cat-count">elige una casilla</small></button>
        </div>
        <div class="board-wrap">
          <div class="column-labels"><span>A</span><span>B</span><span>C</span><span>D</span><span>E</span><span>F</span></div>
          <div class="row-labels"><span>1</span><span>2</span><span>3</span><span>4</span><span>5</span><span>6</span></div>
          <div class="board" id="board"></div>
        </div>
        <div class="resolution hidden" id="resolution">
          <div class="resolution-head"><strong>Elige cómo resolver la jugada</strong><button class="resolution-close" id="resolution-close">Cancelar</button></div>
          <div class="resolution-options" id="resolution-options"></div>
        </div>
        <div class="legend"><span><i class="mini-token" style="--color:var(--first)"></i>Jugador 1</span><span><i class="mini-token" style="--color:var(--second)"></i>Jugador 2</span><span>K = kitten · C = cat</span></div>
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
    const resolutionPanel = document.querySelector('#resolution');
    const resolutionOptions = document.querySelector('#resolution-options');
    let state = null;
    let selectedPiece = 'kitten';
    let requestPending = false;
    let highlightedPositions = [];
    let reviewPly = null;

    for (let index = 0; index < 36; index++) {
      const cell = document.createElement('button');
      cell.className = 'cell';
      cell.dataset.row = Math.floor(index / 6);
      cell.dataset.column = index % 6;
      cell.title = coordinate(Number(cell.dataset.row), Number(cell.dataset.column));
      cell.addEventListener('click', () => chooseCell(Number(cell.dataset.row), Number(cell.dataset.column)));
      board.appendChild(cell);
    }
    for (const player of [0, 1]) document.querySelector(`#player-${player}`).addEventListener('change', updateAgentFields);
    for (const button of document.querySelectorAll('.piece-choice')) button.addEventListener('click', () => selectPiece(button.dataset.piece));
    document.querySelector('#resolution-close').addEventListener('click', closeResolution);
    document.querySelector('#previous-move').addEventListener('click', previousMove);
    document.querySelector('#next-move').addEventListener('click', nextMove);
    document.querySelector('#live-move').addEventListener('click', returnToLive);
    document.querySelector('#history').addEventListener('click', event => {
      const movement = event.target.closest('[data-ply]');
      if (movement) reviewMove(Number(movement.dataset.ply));
    });
    const pace = document.querySelector('#pace');
    pace.addEventListener('input', () => document.querySelector('#pace-value').textContent = `${Number(pace.value).toFixed(1)} s`);
    document.querySelector('#start').addEventListener('click', start);

    function playerConfig(index) {
      const heuristic = document.querySelector(`#heuristic-${index}`).value;
      return {
        kind: document.querySelector(`#player-${index}`).value,
        iterations: Number(document.querySelector(`#iterations-${index}`).value),
        rollout_depth: Number(document.querySelector(`#depth-${index}`).value),
        heuristic: heuristic === 'none' ? null : Number(heuristic),
      };
    }
    function updateAgentFields() {
      for (const player of [0, 1]) document.querySelector(`#mcts-${player}`).classList.toggle('hidden', document.querySelector(`#player-${player}`).value !== 'mcts');
    }
    async function api(path, options = {}) {
      const response = await fetch(path, options);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'Error desconocido');
      return payload;
    }
    async function start() {
      errorBox.textContent = '';
      reviewPly = null;
      closeResolution();
      try {
        state = await api('/api/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({first:playerConfig(0), second:playerConfig(1), seed:Number(document.querySelector('#seed').value), minimum_move_seconds:Number(pace.value)})});
        render();
      } catch (error) { errorBox.textContent = error.message; }
    }
    function selectPiece(piece) {
      if (!availablePieces().has(piece)) return;
      selectedPiece = piece;
      closeResolution();
      render();
    }
    function actionsFor(row, column) {
      if (reviewPly !== null || state?.status !== 'waiting_human') return [];
      return state.legal_actions.filter(action => action.piece === selectedPiece && action.row === row && action.column === column);
    }
    function chooseCell(row, column) {
      if (requestPending) return;
      const candidates = actionsFor(row, column);
      if (candidates.length === 1) play(candidates[0].index);
      else if (candidates.length > 1) showResolutions(candidates);
    }
    function showResolutions(candidates) {
      resolutionOptions.innerHTML = '';
      resolutionPanel.classList.remove('hidden');
      for (const action of candidates) {
        const button = document.createElement('button');
        button.className = 'resolution-option';
        button.textContent = resolutionLabel(action.resolution);
        button.addEventListener('mouseenter', () => { highlightedPositions = action.resolution.positions; renderBoard(); });
        button.addEventListener('mouseleave', () => { highlightedPositions = []; renderBoard(); });
        button.addEventListener('click', () => play(action.index));
        resolutionOptions.appendChild(button);
      }
    }
    function closeResolution() {
      highlightedPositions = [];
      resolutionPanel.classList.add('hidden');
      if (state) renderBoard();
    }
    async function play(action) {
      if (requestPending) return;
      requestPending = true;
      errorBox.textContent = '';
      closeResolution();
      try {
        state = await api('/api/move', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({action})});
        render();
      } catch (error) { errorBox.textContent = error.message; }
      requestPending = false;
    }
    function availablePieces() {
      if (reviewPly !== null) return new Set();
      return new Set((state?.legal_actions || []).map(action => action.piece));
    }
    function previousMove() {
      if (!state?.moves.length) return;
      const current = reviewPly === null ? state.moves.length : reviewPly;
      reviewMove(Math.max(0, current - 1));
    }
    function nextMove() {
      if (reviewPly === null || reviewPly >= state.moves.length) return;
      reviewMove(reviewPly + 1);
    }
    function reviewMove(ply) {
      if (!state || ply < 0 || ply > state.moves.length) return;
      reviewPly = ply;
      closeResolution();
      render();
    }
    function returnToLive() {
      reviewPly = null;
      closeResolution();
      render();
    }
    function displayedFrame() {
      if (reviewPly === null) return {board:state.board, pools:state.pools, lastAction:state.last_action};
      if (reviewPly === 0) return {board:Array(36).fill(null), pools:[{kittens:8,cats:0},{kittens:8,cats:0}], lastAction:null};
      const movement = state.moves[reviewPly - 1];
      return {board:movement.board, pools:movement.pools, lastAction:movement};
    }
    function render() {
      if (!state) return;
      const reviewing = reviewPly !== null;
      const frame = displayedFrame();
      document.querySelector('#status').textContent = reviewing ? (reviewPly === 0 ? 'Revisando la posición inicial' : `Revisando el movimiento ${reviewPly}`) : translateMessage(state);
      document.querySelector('#status').classList.toggle('thinking', !reviewing && state.status === 'playing');
      const last = frame.lastAction;
      const seconds = last?.decision_seconds ?? state.last_decision_seconds;
      document.querySelector('#timing').textContent = last == null ? '' : `Última colocación: ${last.piece === 'kitten' ? 'Kitten' : 'Cat'} en ${coordinate(last.row, last.column)} · Decisión: ${formatTime(seconds)}`;
      document.querySelector('#replay-position').textContent = reviewing ? `${reviewPly === 0 ? 'Inicio' : `Movimiento ${reviewPly}`} de ${state.moves.length}` : `En directo · ${state.moves.length} movimientos`;
      document.querySelector('#previous-move').disabled = !state.moves.length || reviewPly === 0;
      document.querySelector('#next-move').disabled = reviewPly === null || reviewPly >= state.moves.length;
      document.querySelector('#live-move').disabled = reviewPly === null;
      for (const player of [0,1]) {
        document.querySelector(`#player-card-${player}`).classList.toggle('active', !reviewing && state.active_player === player);
        const pool = frame.pools[player];
        document.querySelector(`#pool-${player}`).textContent = `${pool.kittens} kittens · ${pool.cats} cats`;
      }
      const available = availablePieces();
      if (!available.has(selectedPiece) && available.size) selectedPiece = available.values().next().value;
      const activePool = reviewing || state.active_player == null ? null : state.pools[state.active_player];
      document.querySelector('#kitten-count').textContent = activePool ? `${activePool.kittens} en reserva` : 'elige una casilla';
      document.querySelector('#cat-count').textContent = activePool ? `${activePool.cats} en reserva` : 'elige una casilla';
      for (const button of document.querySelectorAll('.piece-choice')) {
        button.classList.toggle('available', available.has(button.dataset.piece));
        button.classList.toggle('selected', button.dataset.piece === selectedPiece && available.has(button.dataset.piece));
      }
      renderBoard();
      renderHistory();
    }
    function renderBoard() {
      if (!state) return;
      const frame = displayedFrame();
      for (let index = 0; index < 36; index++) {
        const cell = board.children[index], value = frame.board[index], row = Math.floor(index / 6), column = index % 6;
        cell.className = 'cell';
        cell.classList.toggle('legal', actionsFor(row, column).length > 0);
        cell.classList.toggle('last', frame.lastAction?.row === row && frame.lastAction?.column === column);
        cell.classList.toggle('candidate', highlightedPositions.some(position => position[0] === row && position[1] === column));
        cell.innerHTML = value == null ? '' : `<span class="piece ${value.player === 0 ? 'first' : 'second'} ${value.kind}">${value.kind === 'kitten' ? 'K' : 'C'}</span>`;
      }
    }
    function renderHistory() {
      const history = document.querySelector('#history');
      history.innerHTML = state.moves.length ? state.moves.map(move => {
        const resolution = move.resolution.type === 'none' ? '' : `<br>${resolutionLabel(move.resolution)}`;
        return `<div class="move ${reviewPly === move.ply ? 'reviewed' : ''}" data-ply="${move.ply}" style="--move-color:${move.player === 0 ? 'var(--first)' : 'var(--second)'}"><span class="move-number">${String(move.ply).padStart(2,'0')}</span><span class="move-action"><b></b><small>${move.piece === 'kitten' ? 'Kitten' : 'Cat'} en ${coordinate(move.row, move.column)}${resolution}</small></span><span class="move-time">${formatTime(move.decision_seconds)}</span></div>`;
      }).reverse().join('') : '<div class="empty">Las jugadas aparecerán aquí.</div>';
    }
    function resolutionLabel(resolution) {
      const cells = resolution.positions.map(position => coordinate(position[0], position[1])).join(' · ');
      if (resolution.type === 'graduate') return `Graduar ${cells}`;
      if (resolution.type === 'recover') return `Recuperar ${cells}`;
      return 'Continuar sin resolución';
    }
    function coordinate(row, column) { return `${String.fromCharCode(65 + column)}${row + 1}`; }
    function translateMessage(value) {
      if (value.status === 'idle') return 'Configura y comienza una partida';
      if (value.status === 'finished') return value.winner == null ? 'Tablas' : `Gana el jugador ${value.winner + 1}`;
      if (value.status === 'error') return `Error: ${value.message}`;
      if (value.status === 'waiting_human') return `Turno del jugador ${value.active_player + 1}: elige ficha y casilla`;
      return `Pensando: jugador ${value.active_player + 1}`;
    }
    function formatTime(seconds) { return seconds < .001 ? `${(seconds * 1e6).toFixed(0)} µs` : seconds < 1 ? `${(seconds * 1e3).toFixed(1)} ms` : `${seconds.toFixed(2)} s`; }
    async function poll() {
      try { state = await api('/api/state'); render(); } catch (_) {}
      window.setTimeout(poll, 120);
    }
    updateAgentFields();
    poll();
  </script>
</body>
</html>
"""
