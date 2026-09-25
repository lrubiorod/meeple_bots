"""Deterministic lifecycle contracts for the two native step-session GUIs."""

from importlib import import_module
import threading
import unittest
from unittest.mock import patch

from meeple_bots.gui.player import GuiPlayer


GAMES = (
    ("cant_stop", "CantStopGui", "CantStopSession", "winner"),
    ("splendor", "SplendorGui", "SplendorSession", "finished"),
)


class StepGuiLifecycleTests(unittest.TestCase):
    def test_old_native_step_cannot_publish_or_complete_trace_after_replacement(self):
        for game, controller_name, session_name, terminal_key in GAMES:
            with self.subTest(game=game):
                module = import_module(f"meeple_bots.games.{game}.gui.controller")
                entered = threading.Event()
                release = threading.Event()

                class Session:
                    def __init__(self, *, seed=0, first=None, second=None):
                        self.seed = seed
                        self.events = []
                        self.terminal = False

                    def snapshot(self):
                        return {
                            "seed": self.seed, "events": list(self.events),
                            "legal_actions": [], "waiting_human": False,
                            "winner": 0 if self.terminal else None,
                            "finished": self.terminal,
                        }

                    def step(self, action=None):
                        if self.seed == 1:
                            entered.set()
                            if not release.wait(2):
                                raise RuntimeError("test release timed out")
                        self.events.append({"action": action})
                        self.terminal = True

                with patch.object(module, session_name, Session):
                    gui = getattr(module, controller_name)()
                    old = None
                    try:
                        gui.start(GuiPlayer("human"), GuiPlayer("human"),
                                  seed=1, minimum_move_seconds=0, save_trace=True)
                        old = gui._thread
                        self.assertTrue(entered.wait(2))
                        gui.start(GuiPlayer("human"), GuiPlayer("human"),
                                  seed=2, minimum_move_seconds=0)
                        gui._thread.join(2)
                        self.assertFalse(gui._thread.is_alive())
                        expected = gui.snapshot()
                        self.assertEqual(expected["seed"], 2)
                        self.assertEqual(expected["status"], "finished")
                        self.assertIsNone(expected["trace_path"])
                        release.set()
                        old.join(2)
                        self.assertFalse(old.is_alive())
                        self.assertEqual(gui.snapshot(), expected)
                        self.assertEqual(expected[terminal_key], 0 if terminal_key == "winner" else True)
                    finally:
                        release.set()
                        gui.cancel()
                        if old is not None:
                            old.join(2)

    def test_cancel_wakes_human_wait_without_consuming_replacement_input(self):
        for game, controller_name, session_name, _ in GAMES:
            with self.subTest(game=game):
                module = import_module(f"meeple_bots.games.{game}.gui.controller")

                class Session:
                    def __init__(self, *, seed=0, first=None, second=None):
                        self.seed = seed

                    def snapshot(self):
                        return {
                            "seed": self.seed, "events": [],
                            "legal_actions": ["move"], "waiting_human": True,
                            "winner": None, "finished": False,
                        }

                    def step(self, action=None):
                        raise AssertionError("a cancelled human wait must not step")

                with patch.object(module, session_name, Session):
                    gui = getattr(module, controller_name)()
                    published = {1: threading.Event(), 2: threading.Event()}
                    decorate = gui._decorate

                    def signal_publish(state, status):
                        result = decorate(state, status)
                        if status == "waiting_human":
                            published[state["seed"]].set()
                        return result

                    old = None
                    try:
                        gui._decorate = signal_publish
                        gui.start(GuiPlayer("human"), GuiPlayer("human"), seed=1,
                                  minimum_move_seconds=0)
                        old = gui._thread
                        self.assertTrue(published[1].wait(2))
                        previous = gui.snapshot()
                        gui.start(GuiPlayer("human"), GuiPlayer("human"), seed=2,
                                  minimum_move_seconds=0)
                        self.assertTrue(published[2].wait(2))
                        old.join(2)
                        self.assertFalse(old.is_alive())
                        current = gui.snapshot()
                        self.assertEqual(current["seed"], 2)
                        if game == "splendor":
                            # The application replaces the controller to rotate this ID.
                            self.assertEqual(previous["session_id"], current["session_id"])
                            with self.assertRaisesRegex(ValueError, "no longer active"):
                                gui.submit_move(0, 0, "stale-session")
                        with self.assertRaisesRegex(ValueError, "no longer active"):
                            if game == "splendor":
                                gui.submit_move(0, 1, current["session_id"])
                            else:
                                gui.submit_move(0, 1)
                        self.assertEqual(gui.snapshot(), current)
                    finally:
                        gui.cancel()
                        if old is not None:
                            old.join(2)
                        if hasattr(gui, "_thread"):
                            gui._thread.join(2)
