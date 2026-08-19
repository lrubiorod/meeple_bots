"""Self-contained browser page for the tic-tac-toe GUI."""

PAGE = r"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Meeple Bots · Tic-tac-toe</title>
  <style>
    :root {
      color-scheme: dark;
      --ink: #f6f3ea;
      --muted: #a8adb7;
      --panel: rgba(23, 28, 39, .86);
      --line: rgba(255, 255, 255, .1);
      --x: #ffb45c;
      --o: #69d5c7;
      --accent: #ffd166;
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
    .shell { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 34px 0; }
    header { display: flex; align-items: end; justify-content: space-between; gap: 24px; margin-bottom: 26px; }
    .eyebrow { color: var(--accent); letter-spacing: .18em; text-transform: uppercase; font-size: .72rem; font-weight: 800; }
    h1 { margin: 6px 0 0; font-family: Georgia, serif; font-size: clamp(2rem, 5vw, 4.25rem); line-height: .95; font-weight: 500; }
    .connection { color: var(--muted); font-size: .86rem; display: flex; align-items: center; gap: 8px; }
    .dot { width: 9px; height: 9px; border-radius: 99px; background: #7ce38b; box-shadow: 0 0 14px #7ce38b; }
    main { display: grid; grid-template-columns: minmax(280px, 1fr) minmax(320px, 1.12fr) minmax(250px, .82fr); gap: 18px; align-items: start; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 22px; padding: 22px; box-shadow: 0 18px 60px rgba(0,0,0,.2); backdrop-filter: blur(16px); }
    .panel-title { margin: 0 0 18px; font-size: .78rem; text-transform: uppercase; letter-spacing: .14em; color: var(--muted); }
    .players { display: grid; gap: 13px; }
    .player { border: 1px solid var(--line); border-radius: 16px; padding: 15px; transition: border-color .2s, background .2s; }
    .player.active { border-color: color-mix(in srgb, var(--player-color) 68%, transparent); background: color-mix(in srgb, var(--player-color) 8%, transparent); }
    .player-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .player-name { font-weight: 750; }
    .mark { font-family: Georgia, serif; font-size: 1.6rem; color: var(--player-color); }
    label { display: block; margin-top: 11px; color: var(--muted); font-size: .77rem; }
    select, input[type="number"] { width: 100%; margin-top: 5px; border: 1px solid var(--line); border-radius: 10px; padding: 9px 10px; background: #101621; color: var(--ink); font: inherit; }
    .mcts-options { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
    .mcts-options.hidden { display: none; }
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
    .board { width: min(100%, 430px); aspect-ratio: 1; margin: 22px auto 0; display: grid; grid-template-columns: repeat(3, 1fr); gap: 9px; }
    .cell { position: relative; border: 1px solid var(--line); border-radius: 18px; color: var(--ink); background: rgba(255,255,255,.035); cursor: default; font-family: Georgia, serif; font-size: clamp(3rem, 8vw, 6rem); line-height: 1; transition: transform .14s, border-color .14s, background .14s; }
    .cell.legal { cursor: pointer; border-color: rgba(255, 209, 102, .34); }
    .cell.legal:hover { transform: translateY(-3px); border-color: var(--accent); background: rgba(255, 209, 102, .08); }
    .cell.x { color: var(--x); text-shadow: 0 0 28px rgba(255,180,92,.2); }
    .cell.o { color: var(--o); text-shadow: 0 0 28px rgba(105,213,199,.2); }
    .cell.last::after { content: ""; position: absolute; inset: 7px; border: 2px solid rgba(255,255,255,.22); border-radius: 13px; }
    .legend { display: flex; justify-content: center; gap: 20px; margin-top: 18px; color: var(--muted); font-size: .8rem; }
    .thinking::after { content: ""; display: inline-block; width: 7px; height: 7px; margin-left: 8px; border-radius: 50%; background: var(--accent); animation: pulse 1s infinite alternate; }
    @keyframes pulse { to { opacity: .25; transform: scale(.7); } }
    .history { max-height: 460px; overflow: auto; }
    .move { display: grid; grid-template-columns: 30px 1fr auto; align-items: center; gap: 10px; padding: 11px 0; border-bottom: 1px solid var(--line); }
    .move:last-child { border-bottom: 0; }
    .move-number { color: var(--muted); font-size: .72rem; }
    .move-action b { color: var(--move-color); font-family: Georgia, serif; font-size: 1.15rem; }
    .move-action small, .move-time { color: var(--muted); font-size: .72rem; }
    .empty { color: var(--muted); font-size: .86rem; padding: 22px 0; text-align: center; }
    .error { color: #ff8b8b; margin-top: 10px; min-height: 1.2em; font-size: .8rem; }
    @media (max-width: 920px) { main { grid-template-columns: 1fr 1fr; } .history-panel { grid-column: 1 / -1; } }
    @media (max-width: 640px) { .shell { width: min(100% - 20px, 520px); padding-top: 22px; } header { align-items: start; flex-direction: column; } main { grid-template-columns: 1fr; } .history-panel { grid-column: auto; } .config { order: 2; } .arena { order: 1; } .history-panel { order: 3; } }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div><div class="eyebrow">Meeple Bots · Playroom</div><h1>Tic-tac-toe</h1></div>
      <div class="connection"><span class="dot"></span> Motor local conectado</div>
    </header>
    <main>
      <section class="panel config">
        <h2 class="panel-title">Configurar partida</h2>
        <div class="players">
          <div class="player" id="player-card-0" style="--player-color: var(--x)">
            <div class="player-head"><span class="player-name">Jugador 1</span><span class="mark">X</span></div>
            <label>Control<select id="player-0"><option value="human">Humano</option><option value="mcts">MCTS</option><option value="random">Random</option></select></label>
            <div class="mcts-options hidden" id="mcts-0"><label>Iteraciones<input id="iterations-0" type="number" min="1" value="1000"></label><label>Profundidad<input id="depth-0" type="number" min="1" value="9"></label></div>
          </div>
          <div class="player" id="player-card-1" style="--player-color: var(--o)">
            <div class="player-head"><span class="player-name">Jugador 2</span><span class="mark">O</span></div>
            <label>Control<select id="player-1"><option value="mcts">MCTS</option><option value="human">Humano</option><option value="random">Random</option></select></label>
            <div class="mcts-options" id="mcts-1"><label>Iteraciones<input id="iterations-1" type="number" min="1" value="1000"></label><label>Profundidad<input id="depth-1" type="number" min="1" value="9"></label></div>
          </div>
        </div>
        <div class="pace"><div class="pace-line"><span>Intervalo mínimo entre jugadas</span><b id="pace-value">0.6 s</b></div><input id="pace" type="range" min="0" max="3" step="0.1" value="0.6"></div>
        <div class="seed-row"><label>Semilla<input id="seed" type="number" min="0" value="0"></label><button class="start" id="start">Nueva partida</button></div>
        <div class="error" id="error"></div>
      </section>
      <section class="panel arena">
        <div class="status"><strong id="status">Configura y comienza una partida</strong><span id="timing"></span></div>
        <div class="board" id="board"></div>
        <div class="legend"><span style="color:var(--x)">X · Jugador 1</span><span style="color:var(--o)">O · Jugador 2</span></div>
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

    for (let index = 0; index < 9; index++) {
      const cell = document.createElement('button');
      cell.className = 'cell';
      cell.dataset.row = Math.floor(index / 3);
      cell.dataset.column = index % 3;
      cell.addEventListener('click', () => play(Number(cell.dataset.row), Number(cell.dataset.column)));
      board.appendChild(cell);
    }

    for (const player of [0, 1]) {
      document.querySelector(`#player-${player}`).addEventListener('change', updateAgentFields);
    }
    const pace = document.querySelector('#pace');
    pace.addEventListener('input', () => document.querySelector('#pace-value').textContent = `${Number(pace.value).toFixed(1)} s`);
    document.querySelector('#start').addEventListener('click', start);

    function playerConfig(index) {
      return {
        kind: document.querySelector(`#player-${index}`).value,
        iterations: Number(document.querySelector(`#iterations-${index}`).value),
        rollout_depth: Number(document.querySelector(`#depth-${index}`).value),
      };
    }

    function updateAgentFields() {
      for (const player of [0, 1]) {
        const mcts = document.querySelector(`#player-${player}`).value === 'mcts';
        document.querySelector(`#mcts-${player}`).classList.toggle('hidden', !mcts);
      }
    }

    async function api(path, options = {}) {
      const response = await fetch(path, options);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'Error desconocido');
      return payload;
    }

    async function start() {
      errorBox.textContent = '';
      try {
        state = await api('/api/start', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({first: playerConfig(0), second: playerConfig(1), seed: Number(document.querySelector('#seed').value), minimum_move_seconds: Number(pace.value)}),
        });
        render();
      } catch (error) { errorBox.textContent = error.message; }
    }

    async function play(row, column) {
      if (requestPending || !isLegal(row, column)) return;
      requestPending = true;
      errorBox.textContent = '';
      try {
        state = await api('/api/move', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({row, column})});
        render();
      } catch (error) { errorBox.textContent = error.message; }
      requestPending = false;
    }

    function isLegal(row, column) {
      return state?.status === 'waiting_human' && state.legal_actions.some(action => action[0] === row && action[1] === column);
    }

    function render() {
      if (!state) return;
      document.querySelector('#status').textContent = translateMessage(state);
      document.querySelector('#status').classList.toggle('thinking', state.status === 'playing');
      const seconds = state.last_decision_seconds;
      document.querySelector('#timing').textContent = seconds == null ? '' : `Última decisión: ${formatTime(seconds)}`;
      for (let index = 0; index < 9; index++) {
        const cell = board.children[index];
        const value = state.board[index];
        const row = Math.floor(index / 3), column = index % 3;
        cell.textContent = value === 0 ? 'X' : value === 1 ? 'O' : '';
        cell.className = `cell ${value === 0 ? 'x' : value === 1 ? 'o' : ''}`;
        cell.classList.toggle('legal', value == null && isLegal(row, column));
        cell.classList.toggle('last', state.last_move?.[0] === row && state.last_move?.[1] === column);
      }
      for (const player of [0, 1]) document.querySelector(`#player-card-${player}`).classList.toggle('active', state.active_player === player);
      const history = document.querySelector('#history');
      history.innerHTML = state.moves.length ? state.moves.map(move => `<div class="move" style="--move-color:${move.player === 0 ? 'var(--x)' : 'var(--o)'}"><span class="move-number">${String(move.ply).padStart(2, '0')}</span><span class="move-action"><b>${move.player === 0 ? 'X' : 'O'}</b> <small>fila ${move.row + 1}, col. ${move.column + 1}</small></span><span class="move-time">${formatTime(move.decision_seconds)}</span></div>`).reverse().join('') : '<div class="empty">Las jugadas aparecerán aquí.</div>';
    }

    function translateMessage(value) {
      if (value.status === 'idle') return 'Configura y comienza una partida';
      if (value.status === 'finished') return value.winner == null ? 'Tablas' : `Gana el jugador ${value.winner + 1}`;
      if (value.status === 'error') return `Error: ${value.message}`;
      if (value.status === 'waiting_human') return `Turno del jugador ${value.active_player + 1}`;
      return `Pensando: jugador ${value.active_player + 1}`;
    }

    function formatTime(seconds) {
      return seconds < .001 ? `${(seconds * 1e6).toFixed(0)} µs` : seconds < 1 ? `${(seconds * 1e3).toFixed(1)} ms` : `${seconds.toFixed(2)} s`;
    }

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
