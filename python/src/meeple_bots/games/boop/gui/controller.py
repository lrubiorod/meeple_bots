"""Thread-safe Boop controller used by the local web interface."""

from __future__ import annotations

import threading
from copy import deepcopy
from time import monotonic

from ....api import (
    Boop,
    BoopAction,
    BoopGraduateLine,
    BoopPiece,
    BoopRecoverPiece,
    HumanAgent,
    HumanTurn,
    Match,
    MatchMoveObservation,
    MctsAgent,
    RandomAgent,
)
from ....gui.player import GuiPlayer


class BoopGui:
    """Coordinate a live Boop match between browser input and native agents."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._cancelled = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending_action: int | None = None
        self._legal_actions: tuple[BoopAction, ...] = ()
        self._players = (
            GuiPlayer("human", rollout_depth=15),
            GuiPlayer("mcts", iterations=1_000, rollout_depth=15, heuristic=0),
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
                name="meeple-bots-boop",
                daemon=True,
            )
            self._thread.start()

    def cancel(self) -> None:
        self._cancelled.set()
        with self._condition:
            self._condition.notify_all()

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return deepcopy(self._state)

    def submit_move(self, action_index: int) -> None:
        with self._condition:
            if self._state["status"] != "waiting_human":
                raise ValueError("the match is not waiting for a human move")
            if not 0 <= action_index < len(self._legal_actions):
                raise ValueError("that action is not a legal move")
            self._pending_action = action_index
            self._state["status"] = "playing"
            self._state["message"] = "Applying move..."
            self._state["legal_actions"] = []
            self._condition.notify_all()

    def _run_match(self, seed: int) -> None:
        try:
            result = Match(
                game=Boop(),
                first=self._agent(0),
                second=self._agent(1),
                seed=seed,
                max_plies=10_000,
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
            heuristic=configured.heuristic,
        )

    def _select_human_action(self, turn: HumanTurn) -> BoopAction:
        with self._condition:
            if self._cancelled.is_set():
                raise RuntimeError("match cancelled")
            self._pending_action = None
            self._legal_actions = tuple(
                action for action in turn.legal_actions if isinstance(action, BoopAction)
            )
            if turn.pools is None:
                raise RuntimeError("Boop turn did not include piece pools")
            self._state["board"] = [
                _serialize_piece(cell) for row in turn.board for cell in row
            ]
            self._state["pools"] = [
                {"kittens": pool.kittens, "cats": pool.cats} for pool in turn.pools
            ]
            self._state["status"] = "waiting_human"
            self._state["active_player"] = turn.player
            self._state["message"] = f"Player {turn.player + 1}, choose a piece and cell"
            self._state["legal_actions"] = [
                _serialize_action(action, index)
                for index, action in enumerate(self._legal_actions)
            ]
            self._condition.notify_all()

            while self._pending_action is None and not self._cancelled.is_set():
                self._condition.wait()
            if self._cancelled.is_set():
                raise RuntimeError("match cancelled")
            action_index = self._pending_action
            if action_index is None:
                raise RuntimeError("human move was not supplied")
            return self._legal_actions[action_index]

    def _observe_move(self, observation: MatchMoveObservation) -> None:
        elapsed = monotonic() - self._last_published
        remaining = max(0.0, self._minimum_move_seconds - elapsed)
        if self._cancelled.wait(remaining):
            return

        with self._condition:
            if self._cancelled.is_set():
                return
            action = observation.action
            if not isinstance(action, BoopAction):
                raise TypeError("Boop observer received another game's action")
            if observation.pools is None:
                raise TypeError("Boop observer did not receive piece pools")
            serialized_action = _serialize_action(action)
            serialized_board = [
                _serialize_piece(cell) for row in observation.board for cell in row
            ]
            serialized_pools = [
                {"kittens": pool.kittens, "cats": pool.cats}
                for pool in observation.pools
            ]
            moves = list(self._state["moves"])
            moves.append(
                {
                    **serialized_action,
                    "ply": len(moves) + 1,
                    "player": observation.player,
                    "decision_seconds": observation.decision_seconds,
                    "board": serialized_board,
                    "pools": serialized_pools,
                }
            )
            self._state["board"] = serialized_board
            self._state["pools"] = serialized_pools
            self._state["moves"] = moves
            self._state["active_player"] = 1 - observation.player
            self._state["last_action"] = serialized_action
            self._state["last_decision_seconds"] = observation.decision_seconds
            self._state["legal_actions"] = []
            self._state["message"] = f"Player {2 - observation.player} is thinking"
            self._last_published = monotonic()
            self._condition.notify_all()

    def _initial_state(self) -> dict[str, object]:
        return {
            "game": "boop",
            "status": "idle",
            "message": "Configure and start a match",
            "board": [None] * 36,
            "pools": [
                {"kittens": 8, "cats": 0},
                {"kittens": 8, "cats": 0},
            ],
            "players": [player.as_dict() for player in self._players],
            "active_player": None,
            "winner": None,
            "moves": [],
            "legal_actions": [],
            "last_action": None,
            "last_decision_seconds": None,
            "seed": 0,
            "minimum_move_seconds": self._minimum_move_seconds,
        }


def _serialize_piece(piece: object) -> dict[str, object] | None:
    if piece is None:
        return None
    if not isinstance(piece, BoopPiece):
        raise TypeError("Boop board contains another game's piece")
    return {"player": piece.player, "kind": piece.kind.value}


def _serialize_action(action: BoopAction, index: int | None = None) -> dict[str, object]:
    resolution = action.resolution
    if isinstance(resolution, BoopGraduateLine):
        serialized_resolution: dict[str, object] = {
            "type": "graduate",
            "positions": [[position.row, position.column] for position in resolution.positions],
        }
    elif isinstance(resolution, BoopRecoverPiece):
        serialized_resolution = {
            "type": "recover",
            "positions": [[resolution.position.row, resolution.position.column]],
        }
    else:
        serialized_resolution = {"type": "none", "positions": []}
    serialized: dict[str, object] = {
        "piece": action.piece.value,
        "row": action.row,
        "column": action.column,
        "resolution": serialized_resolution,
    }
    if index is not None:
        serialized["index"] = index
    return serialized
