"""Shared lifecycle for native sessions advanced one action or chance event at a time."""

from copy import deepcopy
import json
from math import isfinite
from pathlib import Path
import threading
from uuid import uuid4

from ..api import MctsAgent, RandomAgent
from .player import ConfiguredGuiPlayer, GuiPlayer


class NativeStepGui:
    """Own publication, cancellation and human waits for a native step session.

    The condition protects published state and pending human input. Native
    ``step`` and pacing run outside it. Each worker retains its cancellation
    event, so a completed old step cannot publish into a replacement session.
    Chance order and RNG streams remain owned by the native session.
    """

    def __init__(
        self, trace_dir: Path, *, session_type, baseline: GuiPlayer,
        finished, trace_slug: str, trace_format: str,
        session_id: str | None = None,
    ) -> None:
        self._condition = threading.Condition()
        self._cancelled = threading.Event()
        self._trace_dir = trace_dir
        self._session_type = session_type
        self._finished = finished
        self._trace_slug = trace_slug
        self._trace_format = trace_format
        self._session_id = session_id
        self._players = (GuiPlayer("human"), baseline)
        self._state = self._decorate(session_type().snapshot(), "idle")
        self._pending = None

    def _decorate(self, state, status):
        decorated = {**state}
        if self._session_id is not None:
            decorated["session_id"] = self._session_id
        decorated.update(status=status, players=[p.as_dict() for p in self._players],
                         message="", trace_path=None)
        return decorated

    def snapshot(self):
        with self._condition:
            return deepcopy(self._state)

    def cancel(self):
        with self._condition:
            self._cancelled.set()
            self._condition.notify_all()

    @staticmethod
    def _agent(player):
        if player.kind == "human":
            return None
        if player.kind == "random":
            return RandomAgent()
        if isinstance(player, ConfiguredGuiPlayer):
            return player.to_agent()
        return MctsAgent(**{k: v for k, v in player.as_dict().items() if k != "kind"})

    def start(self, first, second, *, seed=0, minimum_move_seconds=0.4, save_trace=False):
        if isinstance(minimum_move_seconds, bool) or not isinstance(minimum_move_seconds, (int, float)) or not isfinite(minimum_move_seconds) or minimum_move_seconds < 0:
            raise ValueError("minimum_move_seconds must be finite and non-negative")
        if not isinstance(save_trace, bool):
            raise ValueError("save_trace must be a boolean")
        session = self._session_type(seed=seed, first=self._agent(first), second=self._agent(second))
        with self._condition:
            self.cancel()
            cancelled = self._cancelled = threading.Event()
            self._players = (first, second)
            self._pending = None
            self._state = self._decorate(session.snapshot(), "playing")
            self._thread = threading.Thread(target=self._run, args=(session, cancelled, minimum_move_seconds, save_trace, seed, (first, second)), daemon=True)
            self._thread.start()

    def _submit_move(self, action, turn, session_id=None):
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError("action must be an integer")
        if isinstance(turn, bool) or not isinstance(turn, int):
            raise ValueError("turn must be an integer")
        with self._condition:
            if ((self._session_id is not None and session_id != self._session_id)
                    or self._state["status"] != "waiting_human" or turn != len(self._state["events"])):
                raise ValueError("this decision is no longer active")
            if not 0 <= action < len(self._state["legal_actions"]):
                raise ValueError("illegal action index")
            self._pending = action
            self._state["status"] = "playing"
            self._condition.notify_all()

    def _run(self, session, cancelled, delay, save_trace, seed, players):
        try:
            while not cancelled.is_set():
                state = session.snapshot()
                status = "finished" if self._finished(state) else "waiting_human" if state["waiting_human"] else "playing"
                with self._condition:
                    if cancelled.is_set():
                        return
                    self._state = self._decorate(state, status)
                    if status == "finished":
                        break
                    if status == "waiting_human":
                        self._condition.wait_for(lambda: cancelled.is_set() or self._pending is not None)
                        if cancelled.is_set():
                            return
                        action, self._pending = self._pending, None
                    else:
                        action = None
                if cancelled.is_set():
                    return
                session.step(action)
                if cancelled.wait(delay):
                    return
            if save_trace and not cancelled.is_set():
                self._trace_dir.mkdir(parents=True, exist_ok=True)
                path = self._trace_dir / f"{self._trace_slug}-{seed}-{uuid4().hex}.json"
                payload = {"format": self._trace_format, "seed": seed, **session.snapshot(),
                           "players": [p.as_dict() for p in players]}
                path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                with self._condition:
                    if not cancelled.is_set():
                        self._state["trace_path"] = str(path)
        except (TypeError, ValueError, RuntimeError, OSError) as error:
            with self._condition:
                if not cancelled.is_set():
                    self._state["status"] = "error"
                    self._state["message"] = str(error)
