"""Thread-safe controller for the Spirits of the Forest browser interface."""

from __future__ import annotations

import threading
from pathlib import Path
from time import monotonic

from ....api import (
    EndSpiritCollection,
    HumanAgent,
    HumanTurn,
    Match,
    MatchMoveObservation,
    MctsAgent,
    MoveSpiritGemstone,
    PlaceSpiritGemstone,
    RandomAgent,
    SkipSpiritGemstone,
    SpiritCollection,
    SpiritGemstonePool,
    SpiritTile,
    SpiritsOfTheForest,
    SpiritsOfTheForestAction,
    TakeSpiritTile,
    _initial_spirits_state,
)
from ....gui.player import GuiPlayer
from ....gui.trace import write_gui_trace


class SpiritsOfTheForestGui:
    def __init__(self, trace_dir: Path = Path("results/gui/spotf")) -> None:
        self._condition = threading.Condition()
        self._cancelled = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending_action: SpiritsOfTheForestAction | None = None
        self._legal_actions: tuple[SpiritsOfTheForestAction, ...] = ()
        self._players = (
            GuiPlayer("human", rollout_depth=64),
            GuiPlayer("mcts", rollout_depth=64),
        )
        self._minimum_move_seconds = 0.4
        self._trace_dir = trace_dir
        self._save_trace = False
        self._last_published = monotonic()
        self._state = self._empty_state()

    def start(
        self,
        first: GuiPlayer,
        second: GuiPlayer,
        *,
        seed: int = 0,
        minimum_move_seconds: float = 0.4,
        save_trace: bool = False,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError("seed must be an integer between 0 and 2^64 - 1")
        if (
            isinstance(minimum_move_seconds, bool)
            or not isinstance(minimum_move_seconds, (int, float))
            or not 0 <= minimum_move_seconds <= 10
        ):
            raise ValueError("minimum_move_seconds must be between 0 and 10")
        if not isinstance(save_trace, bool):
            raise ValueError("save_trace must be a boolean")
        self.cancel()
        board, collections, gems, phase, active, scores = _initial_spirits_state(seed)
        with self._condition:
            self._cancelled = threading.Event()
            self._pending_action = None
            self._legal_actions = ()
            self._players = (first, second)
            self._minimum_move_seconds = float(minimum_move_seconds)
            self._save_trace = save_trace
            self._last_published = monotonic()
            self._state = self._empty_state()
            self._state.update(
                {
                    "status": "playing",
                    "message": "Player 1 is thinking",
                    "forest": _serialize_board(board),
                    "collections": _serialize_collections(collections),
                    "gemstone_pools": _serialize_gems(gems),
                    "scores": list(scores),
                    "phase": phase.value,
                    "active_player": active,
                    "seed": seed,
                    "minimum_move_seconds": self._minimum_move_seconds,
                    "players": [first.as_dict(), second.as_dict()],
                    "save_trace": save_trace,
                    "initial": {
                        "forest": _serialize_board(board),
                        "collections": _serialize_collections(collections),
                        "gemstone_pools": _serialize_gems(gems),
                        "scores": list(scores),
                        "phase": phase.value,
                        "active_player": active,
                    },
                }
            )
            self._thread = threading.Thread(
                target=self._run_match,
                args=(seed,),
                name="meeple-bots-spotf",
                daemon=True,
            )
            self._thread.start()

    def cancel(self) -> None:
        self._cancelled.set()
        with self._condition:
            self._condition.notify_all()

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            initial = self._state["initial"]
            return {
                **self._state,
                "forest": [None if tile is None else dict(tile) for tile in self._state["forest"]],
                "collections": [dict(collection) for collection in self._state["collections"]],
                "gemstone_pools": [dict(pool) for pool in self._state["gemstone_pools"]],
                "scores": list(self._state["scores"]),
                "players": [dict(player) for player in self._state["players"]],
                "legal_actions": [dict(action) for action in self._state["legal_actions"]],
                "initial": (
                    None
                    if initial is None
                    else {
                        **initial,
                        "forest": [
                            None if tile is None else dict(tile)
                            for tile in initial["forest"]
                        ],
                        "collections": [dict(item) for item in initial["collections"]],
                        "gemstone_pools": [
                            dict(item) for item in initial["gemstone_pools"]
                        ],
                        "scores": list(initial["scores"]),
                    }
                ),
                "moves": [
                    {
                        **move,
                        "forest": [None if tile is None else dict(tile) for tile in move["forest"]],
                        "collections": [dict(item) for item in move["collections"]],
                        "gemstone_pools": [dict(item) for item in move["gemstone_pools"]],
                        "scores": list(move["scores"]),
                    }
                    for move in self._state["moves"]
                ],
            }

    def submit_move(self, index: int) -> None:
        with self._condition:
            if self._state["status"] != "waiting_human":
                raise ValueError("the match is not waiting for a human action")
            action = self._legal_actions[index] if 0 <= index < len(self._legal_actions) else None
            if action is None:
                raise ValueError("that action index is not legal")
            self._pending_action = action
            self._state["status"] = "playing"
            self._state["message"] = "Applying action..."
            self._condition.notify_all()

    def _run_match(self, seed: int) -> None:
        started = monotonic()
        try:
            result = Match(
                game=SpiritsOfTheForest(),
                first=self._agent(0),
                second=self._agent(1),
                seed=seed,
                max_plies=256,
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
        trace_path = None
        trace_error = None
        if self._save_trace:
            try:
                trace_path = write_gui_trace(
                    self._trace_dir,
                    game="spotf",
                    max_plies=256,
                    result=result,
                    players=self._players,
                    duration_seconds=monotonic() - started,
                )
            except Exception as error:
                trace_error = str(error)
        with self._condition:
            self._state.update(
                status="finished",
                active_player=None,
                winner=result.winner,
                scores=list(result.scores or (0, 0)),
                message=(
                    "Empate"
                    if result.winner is None
                    else f"Gana el jugador {result.winner + 1}"
                ),
                legal_actions=[],
                trace_path=None if trace_path is None else str(trace_path),
                trace_error=trace_error,
            )
            self._condition.notify_all()

    def _agent(self, player: int):
        configured = self._players[player]
        if configured.kind == "human":
            return HumanAgent(self._select_human_action)
        if configured.kind == "random":
            return RandomAgent()
        return MctsAgent(
            iterations=configured.iterations,
            time_budget=configured.time_budget,
            exploration=configured.exploration,
            rollout_depth=configured.rollout_depth,
            heuristic=configured.heuristic,
            tree_reuse=configured.tree_reuse,
        )

    def _select_human_action(self, turn: HumanTurn) -> SpiritsOfTheForestAction:
        with self._condition:
            if self._cancelled.is_set():
                raise RuntimeError("match cancelled")
            self._pending_action = None
            self._legal_actions = tuple(turn.legal_actions)
            self._state.update(
                status="waiting_human",
                message=f"Jugador {turn.player + 1}: elige una acción",
                active_player=turn.player,
                phase=turn.phase.value if turn.phase else None,
                legal_actions=[
                    _serialize_action(action, index)
                    for index, action in enumerate(self._legal_actions)
                ],
            )
            self._condition.notify_all()
            while self._pending_action is None and not self._cancelled.is_set():
                self._condition.wait()
            if self._cancelled.is_set():
                raise RuntimeError("match cancelled")
            if self._pending_action is None:
                raise RuntimeError("human action was not supplied")
            return self._pending_action

    def _observe_move(self, observation: MatchMoveObservation) -> None:
        if self._cancelled.wait(
            max(0.0, self._minimum_move_seconds - (monotonic() - self._last_published))
        ):
            return
        with self._condition:
            if self._cancelled.is_set():
                return
            action = observation.action
            if not isinstance(
                action,
                (TakeSpiritTile, EndSpiritCollection, PlaceSpiritGemstone, MoveSpiritGemstone, SkipSpiritGemstone),
            ):
                raise TypeError("Spirits observer received another game's action")
            forest = _serialize_board(observation.board)
            collections = _serialize_collections(observation.spirit_collections or ())
            gems = _serialize_gems(observation.gemstone_pools or ())
            moves = list(self._state["moves"])
            serialized_action = _serialize_action(action)
            moves.append(
                {
                    **serialized_action,
                    "ply": len(moves) + 1,
                    "player": observation.player,
                    "decision_seconds": observation.decision_seconds,
                    "forest": forest,
                    "collections": collections,
                    "gemstone_pools": gems,
                    "scores": list(observation.scores or (0, 0)),
                    "phase": observation.phase.value if observation.phase else None,
                    "active_player": observation.active_player,
                }
            )
            self._state.update(
                forest=forest,
                collections=collections,
                gemstone_pools=gems,
                scores=list(observation.scores or (0, 0)),
                phase=observation.phase.value if observation.phase else None,
                active_player=observation.active_player,
                moves=moves,
                legal_actions=[],
                last_decision_seconds=observation.decision_seconds,
                message=f"Jugador {(observation.active_player or 0) + 1} está pensando",
            )
            self._last_published = monotonic()
            self._condition.notify_all()

    def _empty_state(self) -> dict[str, object]:
        return {
            "game": "spotf",
            "status": "idle",
            "message": "Configura y comienza una partida",
            "forest": [None] * 48,
            "collections": [],
            "gemstone_pools": [],
            "scores": [0, 0],
            "players": [player.as_dict() for player in self._players],
            "active_player": None,
            "winner": None,
            "phase": None,
            "moves": [],
            "legal_actions": [],
            "last_decision_seconds": None,
            "seed": 0,
            "minimum_move_seconds": self._minimum_move_seconds,
            "save_trace": self._save_trace,
            "trace_path": None,
            "trace_error": None,
            "initial": None,
        }


def _serialize_board(board) -> list[dict[str, object] | None]:
    return [
        None
        if tile is None
        else {
            "spirit": tile.spirit.value,
            "spirit_symbols": tile.spirit_symbols,
            "power_source": None if tile.power_source is None else tile.power_source.value,
            "gemstone": tile.gemstone,
        }
        for row in board
        for tile in row
    ]


def _serialize_collections(collections) -> list[dict[str, object]]:
    return [
        {
            "spirit_symbols": list(collection.spirit_symbols),
            "power_sources": list(collection.power_sources),
            "tiles": collection.tiles,
        }
        for collection in collections
        if isinstance(collection, SpiritCollection)
    ]


def _serialize_gems(pools) -> list[dict[str, int]]:
    return [
        {"available": pool.available, "placed": pool.placed, "removed": pool.removed}
        for pool in pools
        if isinstance(pool, SpiritGemstonePool)
    ]


def _serialize_action(action: SpiritsOfTheForestAction, index: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {"index": index}
    if isinstance(action, TakeSpiritTile):
        result.update(kind="take_tile", row=action.position.row, column=action.position.column)
        if action.sacrifice is None:
            result["sacrifice"] = None
        elif action.sacrifice.source is None:
            result["sacrifice"] = {"kind": "available"}
        else:
            result["sacrifice"] = {
                "kind": "forest",
                "row": action.sacrifice.source.row,
                "column": action.sacrifice.source.column,
            }
    elif isinstance(action, EndSpiritCollection):
        result["kind"] = "end_collection"
    elif isinstance(action, PlaceSpiritGemstone):
        result.update(kind="place_gemstone", row=action.target.row, column=action.target.column)
    elif isinstance(action, MoveSpiritGemstone):
        result.update(
            kind="move_gemstone",
            source_row=action.source.row,
            source_column=action.source.column,
            target_row=action.target.row,
            target_column=action.target.column,
        )
    else:
        result["kind"] = "skip_gemstone"
    return result
