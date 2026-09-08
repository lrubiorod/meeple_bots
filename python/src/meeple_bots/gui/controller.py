"""Shared worker lifecycle; game controllers own only their presentation and input."""

from __future__ import annotations

from abc import ABC, abstractmethod
import threading
from pathlib import Path
from time import monotonic

from ..api import Game, HumanTurn, Match, MatchMoveObservation, MatchResult
from .player import GuiPlayer
from .trace import write_gui_trace


class GuiController(ABC):
    """Serialize state updates and isolate callbacks from cancelled executions.

    Presentation hooks run under the condition lock. Start preparation runs before
    cancelling the current match, so a preparation failure preserves it.
    """

    def __init__(
        self, trace_dir: Path, *, game: Game, game_name: str,
        max_plies: int, players: tuple[GuiPlayer, GuiPlayer], delay: float,
    ) -> None:
        self._condition = threading.Condition()
        self._cancelled = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending_action = None
        self._legal_actions = ()
        self._players = players
        self._minimum_move_seconds = delay
        self._trace_dir = trace_dir
        self._game = game
        self._game_name = game_name
        self._max_plies = max_plies
        self._save_trace = False
        self._last_published = monotonic()
        self._state = self._initial_state()

    def _prepare_start(self, seed: int) -> dict[str, object]:
        return {}

    def start(
        self,
        first: GuiPlayer,
        second: GuiPlayer,
        *,
        seed: int = 0,
        minimum_move_seconds: float = 0.6,
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

        initial = self._prepare_start(seed)
        with self._condition:
            self.cancel()
            self._cancelled = threading.Event()
            self._pending_action = None
            self._legal_actions = ()
            self._players = (first, second)
            self._minimum_move_seconds = float(minimum_move_seconds)
            self._save_trace = save_trace
            self._last_published = monotonic()
            self._state = self._initial_state()
            self._state.update(
                {
                    "status": "playing",
                    "active_player": 0,
                    "seed": seed,
                    "minimum_move_seconds": self._minimum_move_seconds,
                    "players": [first.as_dict(), second.as_dict()],
                    "save_trace": save_trace,
                }
            )
            self._state.update(initial)
            self._thread = threading.Thread(
                target=self._run_match,
                args=(seed, self._cancelled, self._players, save_trace),
                name=f"meeple-bots-{self._game_name}",
                daemon=True,
            )
            self._thread.start()

    def cancel(self) -> None:
        with self._condition:
            self._cancelled.set()
            self._condition.notify_all()

    def _run_match(
        self,
        seed: int,
        cancelled: threading.Event,
        players: tuple[GuiPlayer, GuiPlayer],
        save_trace: bool,
    ) -> None:
        # Each worker keeps its own cancellation token and configuration.
        if cancelled.is_set():
            return
        started = monotonic()
        try:
            result = Match(
                game=self._game,
                first=self._agent(players[0], cancelled),
                second=self._agent(players[1], cancelled),
                seed=seed,
                max_plies=self._max_plies,
                observe_move=lambda move: self._observe_move(move, cancelled),
            ).run()
        except (RuntimeError, TypeError, ValueError) as error:
            with self._condition:
                if cancelled.is_set():
                    return
                self._state["status"] = "error"
                self._state["message"] = str(error)
                self._condition.notify_all()
            return

        if cancelled.is_set():
            return
        trace_path = None
        trace_error = None
        if save_trace:
            try:
                trace_path = write_gui_trace(
                    self._trace_dir,
                    game=self._game_name,
                    max_plies=self._max_plies,
                    result=result,
                    players=players,
                    duration_seconds=monotonic() - started,
                )
            except Exception as error:
                trace_error = str(error)
        with self._condition:
            if cancelled.is_set():
                return
            for displayed, recorded in zip(self._state["moves"], result.moves):
                displayed["decision_seconds"] = recorded.decision_seconds
            if result.moves:
                self._state["last_decision_seconds"] = result.moves[-1].decision_seconds
            self._finish_state(result)
            self._state["trace_path"] = None if trace_path is None else str(trace_path)
            self._state["trace_error"] = trace_error
            self._condition.notify_all()

    def _finish_state(self, result: MatchResult) -> None:
        self._state.update(
            status="finished", active_player=None, winner=result.winner,
            message="Draw" if result.winner is None else f"Player {result.winner + 1} wins",
            legal_actions=[],
        )

    def _select_human_action(self, turn: HumanTurn, cancelled: threading.Event):
        with self._condition:
            if cancelled.is_set():
                raise RuntimeError("match cancelled")
            self._pending_action = None
            self._present_human_turn(turn)
            self._condition.notify_all()
            while self._pending_action is None and not cancelled.is_set():
                self._condition.wait()
            if cancelled.is_set():
                raise RuntimeError("match cancelled")
            return self._pending_action

    def _observe_move(
        self, observation: MatchMoveObservation, cancelled: threading.Event
    ) -> None:
        with self._condition:
            if cancelled.is_set():
                return
            remaining = max(
                0.0, self._minimum_move_seconds - (monotonic() - self._last_published)
            )
        if cancelled.wait(remaining):
            return

        with self._condition:
            if cancelled.is_set():
                return
            self._present_move(observation)
            self._last_published = monotonic()
            self._condition.notify_all()

    @abstractmethod
    def _initial_state(self) -> dict[str, object]:
        """Return the game's idle browser state."""

    @abstractmethod
    def _agent(self, configured: GuiPlayer, cancelled: threading.Event):
        """Construct a participant using this game's supported configuration."""

    @abstractmethod
    def _present_human_turn(self, turn: HumanTurn) -> None:
        """Publish legal input and the current position under the lock."""

    @abstractmethod
    def _present_move(self, observation: MatchMoveObservation) -> None:
        """Apply an observed move to the browser state under the lock."""
