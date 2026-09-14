"""Thread-safe Connect Four controller used by the local web interface."""

from __future__ import annotations

import threading
from pathlib import Path

from ....api import (
    ConnectFour,
    ConnectFourAction,
    HumanAgent,
    HumanTurn,
    MatchMoveObservation,
    MctsAgent,
    RandomAgent,
)
from ....gui.player import GuiPlayer
from ....gui.baselines import CONNECT_FOUR_BASELINE
from ....gui.controller import GuiController


class ConnectFourGui(GuiController):
    """Coordinate a live Connect Four match between browser input and agents."""

    def __init__(self, trace_dir: Path = Path("results/gui/connect-four")) -> None:
        super().__init__(
            trace_dir,
            game=ConnectFour(),
            game_name="connect-four",
            max_plies=42,
            players=(
                GuiPlayer("human", rollout_depth=64),
                CONNECT_FOUR_BASELINE,
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
                "legal_actions": list(self._state["legal_actions"]),
            }

    def submit_move(self, column: int) -> None:
        action = ConnectFourAction(column=column)
        with self._condition:
            if self._state["status"] != "waiting_human":
                raise ValueError("the match is not waiting for a human move")
            if action not in self._legal_actions:
                raise ValueError("that column is not a legal move")
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
            if isinstance(action, ConnectFourAction)
        )
        self._state["status"] = "waiting_human"
        self._state["active_player"] = turn.player
        self._state["message"] = f"Player {turn.player + 1}, choose a column"
        self._state["legal_actions"] = [
            action.column for action in self._legal_actions
        ]

    def _present_move(self, observation: MatchMoveObservation) -> None:
        action = observation.action
        if not isinstance(action, ConnectFourAction):
            raise TypeError("Connect Four observer received another game's action")
        previous_board = list(self._state["board"])
        flat_board = [cell for row in observation.board for cell in row]
        placed_index = next(
            index
            for index, (before, after) in enumerate(zip(previous_board, flat_board))
            if before is None and after == observation.player
        )
        row, column = divmod(placed_index, 7)
        moves = list(self._state["moves"])
        moves.append(
            {
                "ply": len(moves) + 1,
                "player": observation.player,
                "row": row,
                "column": column,
                "decision_seconds": observation.decision_seconds,
            }
        )
        self._state["board"] = flat_board
        self._state["moves"] = moves
        self._state["active_player"] = 1 - observation.player
        self._state["last_move"] = [row, column]
        self._state["last_decision_seconds"] = observation.decision_seconds
        self._state["legal_actions"] = []
        self._state["message"] = f"Player {2 - observation.player} is thinking"

    def _initial_state(self) -> dict[str, object]:
        return {
            "game": "connect-four",
            "status": "idle",
            "message": "Configure and start a match",
            "board": [None] * 42,
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
