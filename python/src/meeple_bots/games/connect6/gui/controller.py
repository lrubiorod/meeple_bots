"""Connect6 presentation over the shared cancellable match worker."""
from pathlib import Path
from uuid import uuid4

from ....connect6 import Connect6, Connect6Action
from ....gui.controller import GuiController
from ....gui.player import GuiPlayer
from ....gui.baselines import CONNECT6_BASELINE
from ....api import HumanAgent, RandomAgent, MctsAgent
from copy import deepcopy


class Connect6Gui(GuiController):
    def __init__(self, trace_dir=Path('results/gui/connect6'), board_size=None):
        game = Connect6(board_size)
        self._position = game.initial_state()
        self._session_id = uuid4().hex
        GuiController.__init__(self, trace_dir, game=game, game_name='connect6',
                               max_plies=game.board_size**2,
                               players=(GuiPlayer('human'), CONNECT6_BASELINE), delay=0.4)

    def snapshot(self):
        with self._condition:
            return deepcopy(self._state)

    def _agent(self, configured, cancelled):
        if configured.kind == 'human':
            return HumanAgent(lambda turn: self._select_human_action(turn, cancelled))
        if configured.kind == 'random':
            return RandomAgent()
        return MctsAgent(**{k:v for k,v in configured.as_dict().items() if k != 'kind'})

    def _prepare_start(self, seed):
        self._position = self._game.initial_state()
        self._session_id = uuid4().hex
        return {}

    def submit_move(self, position, turn, session_id):
        action = Connect6Action(position)
        with self._condition:
            if type(turn) is not int or session_id != self._session_id or turn != len(self._state['moves']):
                raise ValueError('this decision is no longer active')
            if self._state['status'] != 'waiting_human' or action not in self._legal_actions:
                raise ValueError('that cell is not a legal move')
            self._pending_action = action
            self._state.update(status='playing', legal_actions=[], message='Applying move...')
            self._condition.notify_all()

    def _present_human_turn(self, turn):
        self._legal_actions = turn.legal_actions
        self._state.update(status='waiting_human', active_player=turn.player,
                           legal_actions=[a.position for a in turn.legal_actions])

    def _present_move(self, observation):
        # Native replay supplies turn progression, including immediate terminal wins.
        self._position = self._position.apply_action(observation.action)
        row, column = divmod(observation.action.position, self._game.board_size)
        self._state['moves'].append(dict(ply=len(self._state['moves'])+1, player=observation.player,
                                        row=row, column=column, decision_seconds=observation.decision_seconds))
        self._state.update(board=[v for r in self._position.board for v in r],
                           active_player=self._position.current_player,
                           placements_remaining=self._position.placements_remaining,
                           last_move=[row,column], last_decision_seconds=observation.decision_seconds,
                           legal_actions=[], status='playing')

    def _initial_state(self):
        return dict(game='connect6', session_id=self._session_id, board_size=self._game.board_size,
                    status='idle', message='Configure and start a match',
                    board=[v for r in self._position.board for v in r],
                    players=[p.as_dict() for p in self._players], active_player=None,
                    placements_remaining=self._position.placements_remaining,
                    winner=None, moves=[], legal_actions=[], last_move=None,
                    last_decision_seconds=None, seed=0, minimum_move_seconds=self._minimum_move_seconds,
                    save_trace=self._save_trace, trace_path=None, trace_error=None)
