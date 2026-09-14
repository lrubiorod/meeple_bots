"""Thread-safe tic-tac-toe controller used by the local web interface."""

from __future__ import annotations

import threading
from pathlib import Path

from ....api import (
    HumanAgent,
    HumanTurn,
    MatchMoveObservation,
    MctsAgent,
    RandomAgent,
    TicTacToe,
    TicTacToeAction,
)
from ....gui.player import GuiPlayer
from ....gui.baselines import TIC_TAC_TOE_BASELINE
from ....gui.controller import GuiController


class TicTacToeGui(GuiController):
    """Coordinate a live match between browser input and native agents."""

    def __init__(self, trace_dir: Path = Path("results/gui/tic-tac-toe")) -> None:
        super().__init__(
            trace_dir,
            game=TicTacToe(),
            game_name="tic-tac-toe",
            max_plies=9,
            players=(
                GuiPlayer("human", rollout_depth=9),
                TIC_TAC_TOE_BASELINE,
            ),
            delay=0.6,
        )

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                **self._state,
                "board": list(self._state["board"]),
                "players": list(self._state["players"]),
                "moves": [dict(move) for move in self._state["moves"]],
                "legal_actions": [list(action) for action in self._state["legal_actions"]],
            }

    def submit_move(self, row: int, column: int) -> None:
        action = TicTacToeAction(row=row, column=column)
        with self._condition:
            if self._state["status"] != "waiting_human":
                raise ValueError("the match is not waiting for a human move")
            if action not in self._legal_actions:
                raise ValueError("that cell is not a legal move")
            self._pending_action = action
            self._state["status"] = "playing"
            self._state["message"] = "Applying move..."
            self._condition.notify_all()

    def _agent(self, configured: GuiPlayer, cancelled: threading.Event):
        if configured.kind == "human":
            return HumanAgent(lambda turn: self._select_human_action(turn, cancelled))
        if configured.kind == "random":
            return RandomAgent()
        return MctsAgent(
            iterations=configured.iterations,
            time_budget=configured.time_budget,
            exploration=configured.exploration,
            selection_policy=configured.selection_policy,
            rave_equivalence=configured.rave_equivalence,
            rollout_depth=configured.rollout_depth,
            tree_reuse=configured.tree_reuse,
            transpositions=configured.transpositions,
        )

    def _present_human_turn(self, turn: HumanTurn) -> None:
        self._legal_actions = tuple(
            action
            for action in turn.legal_actions
            if isinstance(action, TicTacToeAction)
        )
        self._state["status"] = "waiting_human"
        self._state["active_player"] = turn.player
        self._state["message"] = f"Player {turn.player + 1}, choose a cell"
        self._state["legal_actions"] = [
            [action.row, action.column] for action in self._legal_actions
        ]

    def _present_move(self, observation: MatchMoveObservation) -> None:
        action = observation.action
        if not isinstance(action, TicTacToeAction):
            raise TypeError("tic-tac-toe observer received another game's action")
        flat_board = [cell for row in observation.board for cell in row]
        moves = list(self._state["moves"])
        moves.append(
            {
                "ply": len(moves) + 1,
                "player": observation.player,
                "row": action.row,
                "column": action.column,
                "decision_seconds": observation.decision_seconds,
            }
        )
        self._state["board"] = flat_board
        self._state["moves"] = moves
        self._state["active_player"] = 1 - observation.player
        self._state["last_move"] = [action.row, action.column]
        self._state["last_decision_seconds"] = observation.decision_seconds
        self._state["legal_actions"] = []
        self._state["message"] = f"Player {2 - observation.player} is thinking"

    def _initial_state(self) -> dict[str, object]:
        return {
            "game": "tic-tac-toe",
            "status": "idle",
            "message": "Configure and start a match",
            "board": [None] * 9,
            "players": [player.as_dict() for player in self._players],
            "active_player": None,
            "winner": None,
            "moves": [],
            "legal_actions": [],
            "last_move": None,
            "last_decision_seconds": None,
            "seed": 0,
            "minimum_move_seconds": self._minimum_move_seconds,
            "save_trace": self._save_trace,
            "trace_path": None,
            "trace_error": None,
        }
