"""Administrative, open-hand GUI. Rust validates every action and chance event."""
from copy import deepcopy
import json
from pathlib import Path
from random import Random
import threading
from uuid import uuid4

from ....lost_cities import LostCities
from ...._agent_config import SoIsmctsAgent


class LostCitiesGui:
    def __init__(self, trace_dir=Path('results/gui/lost_cities')):
        self._trace_dir = trace_dir
        self._condition = threading.Condition()
        self._cancelled = threading.Event()
        self._session_id = uuid4().hex
        self._pending = None
        self._state = {'status': 'idle', 'session_id': self._session_id, 'turn': 0,
                       'message': 'Start a single-round test match.', 'legal_actions': [],
                       'events': [], 'players': [{'kind': 'human'}, {'kind': 'random'}],
                       'trace_path': None, 'trace_error': None}

    def snapshot(self):
        with self._condition:
            return deepcopy(self._state)

    def cancel(self):
        with self._condition:
            self._cancelled.set()
            self._condition.notify_all()

    def start(self, first, second, *, seed=0, minimum_move_seconds=.4, save_trace=False):
        if any(p.kind not in ('human', 'random', 'so_ismcts') for p in (first, second)):
            raise ValueError('Lost Cities GUI supports human, random, and so_ismcts players')
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError('seed must be an integer between 0 and 2^64 - 1')
        if type(minimum_move_seconds) not in (int, float) or not 0 <= minimum_move_seconds <= 10:
            raise ValueError('minimum_move_seconds must be between 0 and 10')
        if not isinstance(save_trace, bool):
            raise ValueError('save_trace must be a boolean')
        searches = tuple(SoIsmctsAgent(p.iterations, p.exploration, time_budget=p.time_budget) if p.kind == 'so_ismcts' else None for p in (first, second))
        game = LostCities()
        position = game.initial_state(seed)
        with self._condition:
            self.cancel()
            self._cancelled = cancelled = threading.Event()
            self._session_id = uuid4().hex
            self._pending = None
            self._state = {**position.to_dict(), 'status': 'playing', 'session_id': self._session_id,
                           'turn': 0, 'events': [], 'legal_actions': [], 'message': '',
                           'players': [p.as_dict() if p.kind == 'so_ismcts' else {'kind': p.kind} for p in (first, second)],
                           'winner': None, 'trace_path': None, 'trace_error': None}
            self._thread = threading.Thread(target=self._run,
                args=(game, position, (first, second), seed, minimum_move_seconds, cancelled, searches, save_trace), daemon=True)
            self._thread.start()

    def submit_move(self, action, turn, session_id):
        if type(action) is not int or type(turn) is not int:
            raise ValueError('action and turn must be integers')
        with self._condition:
            if (self._cancelled.is_set() or session_id != self._session_id
                    or turn != self._state['turn'] or self._state['status'] != 'waiting_human'):
                raise ValueError('this decision is no longer active')
            if not 0 <= action < len(self._state['legal_actions']):
                raise ValueError('illegal action index')
            self._pending = action
            self._state['status'] = 'playing'
            self._state['legal_actions'] = []
            self._condition.notify_all()

    def _run(self, game, position, players, seed, delay, cancelled, searches, save_trace):
        # Independent streams: GUI presentation never participates in policy input.
        chance_rng = Random(seed ^ 0x8EBC6AF09C88C6E3)
        agent_rngs = [Random(seed ^ 0xA0761D6478BD642F), Random(seed ^ 0xE7037ED1A0B428DB)]
        decisions = 0
        try:
            while not cancelled.is_set():
                terminal = position.status == 'terminal'
                chance = position.status == 'chance'
                active = position.current_player
                actions = () if terminal or chance else game.legal_actions(position)
                human = not terminal and not chance and players[active].kind == 'human'
                with self._condition:
                    if cancelled.is_set():
                        return
                    self._state.update(position.to_dict())
                    self._state.update(status='finished' if terminal else 'waiting_human' if human else 'playing',
                                       legal_actions=[a.to_dict() for a in actions] if human else [])
                    if terminal:
                        utility = game.terminal_utility(position, 0)
                        winner = None if utility == 0 else 0 if utility > 0 else 1
                        self._state.update(winner=winner, message='Draw' if winner is None else f'Player {winner + 1} wins')
                        completed = deepcopy(self._state) if save_trace else None
                    else:
                        # Execution safeguard, never a game rule or a scored result.
                        if not chance and decisions >= 10000:
                            raise RuntimeError('Match safety limit reached (10000 decisions); no result awarded')
                        if human:
                            self._condition.wait_for(lambda: cancelled.is_set() or self._pending is not None)
                            if cancelled.is_set():
                                return
                            action = actions[self._pending]
                            self._pending = None
                        elif not chance and searches[active] is None:
                            # Uniform policy receives only the available action list, not either hand or deck.
                            action = agent_rngs[active].choice(actions)
                if terminal:
                    if completed is not None and not cancelled.is_set():
                        trace_path = trace_error = None
                        try:
                            trace_path = self._write_trace(completed, seed, cancelled)
                        except (OSError, TypeError, ValueError) as error:
                            trace_error = str(error)
                        with self._condition:
                            if not cancelled.is_set():
                                self._state['trace_path'] = None if trace_path is None else str(trace_path)
                                self._state['trace_error'] = trace_error
                    return
                diagnostics = None
                if not chance and searches[active] is not None:
                    # Search runs outside the controller lock and receives no administrative state.
                    observation = game.observation(position, active)
                    result = searches[active].search(observation, actions, seed=agent_rngs[active].getrandbits(64))
                    action, diagnostics = result['action'], result['diagnostics']
                if cancelled.is_set():
                    return
                if chance:
                    action = game.sample_chance(position, chance_rng.getrandbits(64))
                    position = game.apply_chance_outcome(position, action)
                else:
                    position = game.apply_action(position, action)
                    decisions += 1
                with self._condition:
                    if cancelled.is_set():
                        return
                    self._state['events'].append({'player': active, 'chance': chance, 'action': action.to_dict(), **({'search': diagnostics} if diagnostics is not None else {})})
                    self._state['turn'] += 1
                    self._state['status'] = 'playing'
                    self._state['legal_actions'] = []
                if cancelled.wait(delay):
                    return
        except (ValueError, TypeError, RuntimeError) as error:
            with self._condition:
                if not cancelled.is_set():
                    self._state.update(status='error', message=str(error), legal_actions=[])

    def _write_trace(self, completed, seed, cancelled):
        if cancelled.is_set():
            return None
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        path = self._trace_dir / f"lost-cities-{seed}-{completed['session_id']}.json"
        temporary = path.with_suffix('.json.tmp')
        try:
            with temporary.open('x', encoding='utf-8') as output:
                json.dump({'format': 'lost_cities_session_v1', 'seed': seed, **completed}, output)
            if cancelled.is_set():
                return None
            temporary.replace(path)
            return path
        finally:
            temporary.unlink(missing_ok=True)
