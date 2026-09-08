"""Regression tests for execution isolation across all browser controllers."""

from importlib import import_module
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from meeple_bots.gui.player import GuiPlayer
from meeple_bots.gui import controller as lifecycle


GAMES = (
    ("connect_four", "ConnectFour"),
    ("tic_tac_toe", "TicTacToe"),
    ("boop", "Boop"),
    ("spirits_of_the_forest", "SpiritsOfTheForest"),
)
RESULT = SimpleNamespace(moves=(), winner=None, scores=(0, 0))


class GuiLifecycleTests(unittest.TestCase):
    def test_game_defaults_and_failed_preparation_preserve_idle_state(self):
        for game, name in GAMES:
            module = import_module(f"meeple_bots.games.{game}.gui.controller")
            with self.subTest(game=game):
                gui = getattr(module, name + "Gui")()
                before = gui.snapshot()
                depth = 15 if game == "boop" else 9 if game == "tic_tac_toe" else 64
                self.assertEqual(before["players"][0]["rollout_depth"], depth)
                self.assertEqual(before["minimum_move_seconds"],
                                 0.4 if game == "spirits_of_the_forest" else 0.6)
                if game == "boop":
                    self.assertEqual(before["players"][1]["heuristic"], 0)
                    self.assertEqual(before["players"][1]["iterations"], 15000)
                with patch.object(gui, "_prepare_start", side_effect=ValueError("bad initial state")):
                    with self.assertRaisesRegex(ValueError, "bad initial state"):
                        gui.start(GuiPlayer("random"), GuiPlayer("random"))
                self.assertEqual(gui.snapshot(), before)
                self.assertFalse(gui._cancelled.is_set())
                self.assertIsNone(gui._thread)

    def test_invalid_browser_start_preserves_current_match(self):
        for game, name in GAMES:
            module = import_module(f"meeple_bots.games.{game}.gui.application")
            with self.subTest(game=game), patch.object(lifecycle, "Match") as match:
                match.return_value.run.return_value = RESULT
                app = getattr(module, name + "Application")()
                app.start({"first": {"kind": "random"}, "second": {"kind": "random"}})
                previous = app._game
                previous._thread.join(2)
                before = previous.snapshot()
                try:
                    for invalid in ({"seed": -1}, {"minimum_move_seconds": -1},
                                    {"save_trace": "yes"}, {"first": {"kind": "unknown"}}):
                        with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                            app.start(invalid)
                        self.assertIs(app._game, previous)
                        self.assertFalse(previous._cancelled.is_set())
                        self.assertEqual(app.snapshot(), before)
                finally:
                    app.cancel()

    def test_old_callbacks_results_and_errors_cannot_modify_restart(self):
        for game, name in GAMES:
            module = import_module(f"meeple_bots.games.{game}.gui.controller")
            for failure in (False, True):
                with self.subTest(game=game, failure=failure):
                    entered = threading.Event()
                    release = threading.Event()
                    gui = getattr(module, name + "Gui")()
                    callbacks = {}

                    def make_match(**kwargs):
                        def run():
                            if kwargs["seed"] == 1:
                                callbacks.update(kwargs)
                                entered.set()
                                if not release.wait(2):
                                    raise RuntimeError("test release timed out")
                                if failure:
                                    raise ValueError("old match failed")
                            return RESULT
                        return SimpleNamespace(run=run)

                    with patch.object(lifecycle, "Match", side_effect=make_match), \
                            patch.object(lifecycle, "write_gui_trace") as write_trace:
                        try:
                            gui.start(GuiPlayer("human"), GuiPlayer("random"), seed=1,
                                      minimum_move_seconds=0, save_trace=True)
                            old = gui._thread
                            self.assertTrue(entered.wait(2))
                            gui.start(GuiPlayer("random"), GuiPlayer("random"), seed=2,
                                      minimum_move_seconds=0)
                            gui._thread.join(2)
                            self.assertFalse(gui._thread.is_alive())
                            expected = gui.snapshot()
                            # A stale callback must return before even reading a new position.
                            callbacks["observe_move"](None)
                            with self.assertRaisesRegex(RuntimeError, "match cancelled"):
                                callbacks["first"].select_action(None)
                            release.set()
                            old.join(2)
                            self.assertFalse(old.is_alive())
                            self.assertEqual(gui.snapshot(), expected)
                            self.assertEqual(expected["seed"], 2)
                            self.assertEqual(expected["status"], "finished")
                            write_trace.assert_not_called()
                        finally:
                            release.set()
                            gui.cancel()
                            old.join(2)
                            gui._thread.join(2)

    def test_trace_finishing_after_restart_keeps_original_players(self):
        for game, name in GAMES:
            module = import_module(f"meeple_bots.games.{game}.gui.controller")
            with self.subTest(game=game):
                entered = threading.Event()
                release = threading.Event()
                gui = getattr(module, name + "Gui")()
                players = (GuiPlayer("human"), GuiPlayer("random"))

                def write_trace(*args, **kwargs):
                    entered.set()
                    if not release.wait(2):
                        raise RuntimeError("test release timed out")
                    return "old-trace.json"

                with patch.object(lifecycle, "Match") as match, \
                        patch.object(lifecycle, "write_gui_trace", side_effect=write_trace) as trace:
                    match.return_value.run.return_value = RESULT
                    try:
                        gui.start(*players, seed=1, save_trace=True)
                        old = gui._thread
                        self.assertTrue(entered.wait(2))
                        gui.start(GuiPlayer("random"), GuiPlayer("random"), seed=2)
                        gui._thread.join(2)
                        expected = gui.snapshot()
                        release.set()
                        old.join(2)
                        self.assertFalse(old.is_alive())
                        self.assertEqual(trace.call_args.kwargs["players"], players)
                        self.assertEqual(gui.snapshot(), expected)
                        self.assertIsNone(expected["trace_path"])
                    finally:
                        release.set()
                        gui.cancel()
                        old.join(2)
                        gui._thread.join(2)

    def test_restart_wakes_old_human_without_consuming_new_input(self):
        for game, name in GAMES:
            module = import_module(f"meeple_bots.games.{game}.gui.controller")
            with self.subTest(game=game):
                gui = getattr(module, name + "Gui")()
                cancelled_turn = threading.Event()
                # No legal actions are needed: this turn waits until it is cancelled.
                turn = SimpleNamespace(player=0, legal_actions=(), board=(), pools=(), phase=None)

                def make_match(**kwargs):
                    def run():
                        if kwargs["seed"] == 1:
                            try:
                                kwargs["first"].select_action(turn)
                            except RuntimeError as error:
                                if str(error) == "match cancelled":
                                    cancelled_turn.set()
                                raise
                        return RESULT
                    return SimpleNamespace(run=run)

                with patch.object(lifecycle, "Match", side_effect=make_match):
                    gui.start(GuiPlayer("human"), GuiPlayer("random"), seed=1)
                    old = gui._thread
                    try:
                        with gui._condition:
                            self.assertTrue(gui._condition.wait_for(
                                lambda: gui._state["status"] == "waiting_human", timeout=2
                            ))
                            gui.start(GuiPlayer("random"), GuiPlayer("random"), seed=2)
                            # Make input available before the cancelled waiter reacquires the lock.
                            pending = object()
                            gui._pending_action = pending
                        old.join(2)
                        gui._thread.join(2)
                        self.assertFalse(old.is_alive())
                        self.assertTrue(cancelled_turn.is_set())
                        self.assertIs(gui._pending_action, pending)
                        self.assertEqual(gui.snapshot()["seed"], 2)
                        self.assertEqual(gui.snapshot()["status"], "finished")
                    finally:
                        gui.cancel()
                        old.join(2)
                        gui._thread.join(2)
