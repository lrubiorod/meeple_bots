"""Thread-safe Connect Four controller used by the local web interface."""

from __future__ import annotations

import threading
from time import monotonic

from ....api import (
    ConnectFour,
    ConnectFourAction,
    HumanAgent,
    HumanTurn,
    Match,
    MatchMoveObservation,
    MctsAgent,
    RandomAgent,
)
from ....gui.player import GuiPlayer


class ConnectFourGui:
    """Coordinate a live Connect Four match between browser input and agents."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._cancelled = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending_action: ConnectFourAction | None = None
        self._legal_actions: tuple[ConnectFourAction, ...] = ()
        self._players = (
            GuiPlayer("human", rollout_depth=64),
            GuiPlayer("mcts", rollout_depth=64),
        )
        self._minimum_move_seconds = 0.6
        self._last_published = monotonic()
        self._state: dict[str, object] = self._initial_state()

    def start(
        self,
        first: GuiPlayer,
        second: GuiPlayer,
        *,
        seed: int = 0,
        minimum_move_seconds: float = 0.6,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError("seed must be an integer between 0 and 2^64 - 1")
        if (
            isinstance(minimum_move_seconds, bool)
            or not isinstance(minimum_move_seconds, (int, float))
            or not 0 <= minimum_move_seconds <= 10
        ):
            raise ValueError("minimum_move_seconds must be between 0 and 10")

        self.cancel()
        with self._condition:
            self._cancelled = threading.Event()
            self._pending_action = None
            self._legal_actions = ()
            self._players = (first, second)
            self._minimum_move_seconds = float(minimum_move_seconds)
            self._last_published = monotonic()
            self._state = self._initial_state()
            self._state.update(
                {
                    "status": "playing",
                    "active_player": 0,
                    "seed": seed,
                    "minimum_move_seconds": self._minimum_move_seconds,
                    "players": [first.as_dict(), second.as_dict()],
                }
            )
            self._thread = threading.Thread(
                target=self._run_match,
                args=(seed,),
                name="meeple-bots-connect-four",
                daemon=True,
            )
            self._thread.start()

    def cancel(self) -> None:
        self._cancelled.set()
        with self._condition:
            self._condition.notify_all()

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

    def _run_match(self, seed: int) -> None:
        try:
            result = Match(
                game=ConnectFour(),
                first=self._agent(0),
                second=self._agent(1),
                seed=seed,
                max_plies=42,
                observe_move=self._observe_move,
            ).run()
        except (RuntimeError, TypeError, ValueError) as error:
            if self._cancelled.is_set():
                return
            with self._condition:
                self._state["status"] = "error"
                self._state["message"] = str(error)
                self._condition.notify_all()
            return

        if self._cancelled.is_set():
            return
        with self._condition:
            self._state["status"] = "finished"
            self._state["active_player"] = None
            self._state["winner"] = result.winner
            self._state["message"] = (
                "Draw" if result.winner is None else f"Player {result.winner + 1} wins"
            )
            self._state["legal_actions"] = []
            self._condition.notify_all()

    def _agent(self, player: int):
        configured = self._players[player]
        if configured.kind == "human":
            return HumanAgent(self._select_human_action)
        if configured.kind == "random":
            return RandomAgent()
        return MctsAgent(
            iterations=configured.iterations,
            exploration=configured.exploration,
            rollout_depth=configured.rollout_depth,
        )

    def _select_human_action(self, turn: HumanTurn) -> ConnectFourAction:
        with self._condition:
            if self._cancelled.is_set():
                raise RuntimeError("match cancelled")
            self._pending_action = None
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
            self._condition.notify_all()

            while self._pending_action is None and not self._cancelled.is_set():
                self._condition.wait()
            if self._cancelled.is_set():
                raise RuntimeError("match cancelled")
            action = self._pending_action
            if action is None:
                raise RuntimeError("human move was not supplied")
            return action

    def _observe_move(self, observation: MatchMoveObservation) -> None:
        elapsed = monotonic() - self._last_published
        remaining = max(0.0, self._minimum_move_seconds - elapsed)
        if self._cancelled.wait(remaining):
            return

        with self._condition:
            if self._cancelled.is_set():
                return
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
            self._last_published = monotonic()
            self._condition.notify_all()

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
        }
