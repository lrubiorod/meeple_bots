"""Thread-safe Boop controller used by the local web interface."""

from __future__ import annotations

import threading
from copy import deepcopy
from pathlib import Path

from ....api import (
    Boop,
    BoopAction,
    BoopGraduateLine,
    BoopPiece,
    BoopRecoverPiece,
    HumanAgent,
    HumanTurn,
    MatchMoveObservation,
    MctsAgent,
    RandomAgent,
)
from ....gui.baselines import BOOP_BASELINE
from ....gui.player import GuiPlayer
from ....gui.controller import GuiController


class BoopGui(GuiController):
    """Coordinate a live Boop match between browser input and native agents."""

    def __init__(self, trace_dir: Path = Path("results/gui/boop")) -> None:
        super().__init__(
            trace_dir,
            game=Boop(),
            game_name="boop",
            max_plies=10_000,
            players=(
                GuiPlayer("human", rollout_depth=15),
                BOOP_BASELINE,
            ),
            delay=0.6,
        )

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return deepcopy(self._state)

    def submit_move(self, action_index: int) -> None:
        with self._condition:
            if self._state["status"] != "waiting_human":
                raise ValueError("the match is not waiting for a human move")
            if not 0 <= action_index < len(self._legal_actions):
                raise ValueError("that action is not a legal move")
            self._pending_action = self._legal_actions[action_index]
            self._state["status"] = "playing"
            self._state["message"] = "Applying move..."
            self._state["legal_actions"] = []
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
            rollout_depth=configured.rollout_depth,
            heuristic=configured.heuristic,
            tree_reuse=configured.tree_reuse,
            transpositions=configured.transpositions,
        )

    def _present_human_turn(self, turn: HumanTurn) -> None:
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

    def _present_move(self, observation: MatchMoveObservation) -> None:
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
            "save_trace": self._save_trace,
            "trace_path": None,
            "trace_error": None,
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
