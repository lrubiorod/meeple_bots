"""Reference profiles and installed GUI defaults stay in sync."""

import json
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from meeple_bots._mcts_profiles import _load_mcts_profile
from meeple_bots.gui.baselines import BOOP_BASELINE, SPOTF_BASELINE, CONNECT_FOUR_BASELINE, TIC_TAC_TOE_BASELINE
from meeple_bots.gui import controller as lifecycle
from meeple_bots.gui import baselines
from meeple_bots.gui.player import parse_gui_player


class GuiBaselineTests(unittest.TestCase):
    def test_toml_edits_change_defaults_and_allow_manual_overrides(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.toml"
            with patch.object(baselines, "_baseline_path", return_value=path):
                path.write_text("iterations = 20000\nrollout_depth = 42\n")
                self.assertEqual(baselines._load_gui_baseline(path.name).iterations, 20000)
                path.write_text("time_budget = 0.5\nrollout_depth = 30\ntranspositions = true\n")
                defaults = baselines._load_gui_baseline(path.name)
                self.assertIsNone(defaults.iterations)
                self.assertEqual(defaults.time_budget, 0.5)
                self.assertEqual(defaults.rollout_depth, 30)
                self.assertTrue(defaults.transpositions)
                player = parse_gui_player(
                    {"kind": "mcts", "iterations": 30000}, "first",
                    default_rollout_depth=42, default_mcts=defaults,
                )
                self.assertEqual(player.iterations, 30000)
                self.assertIsNone(player.time_budget)
                self.assertEqual(baselines._load_gui_baseline(path.name), defaults)

    def test_unsupported_profile_settings_are_not_silently_discarded(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.toml"
            path.write_text('iterations = 100\nrollout_depth = 42\nrollout_policy = { kind = "mast" }\n')
            with patch.object(baselines, "_baseline_path", return_value=path):
                with self.assertRaisesRegex(ValueError, "not supported by the GUI"):
                    baselines._load_gui_baseline(path.name)

    def test_boop_search_settings_can_override_baseline(self):
        player = parse_gui_player(
            {"kind": "mcts", "heuristic": None, "exploration": 0.8,
             "transpositions": False, "tree_reuse": False},
            "first", default_rollout_depth=16, available_heuristics=(0, 1),
            default_mcts=BOOP_BASELINE,
        )
        self.assertIsNone(player.heuristic)
        self.assertEqual(player.exploration, 0.8)
        self.assertFalse(player.transpositions)
        self.assertFalse(player.tree_reuse)
        self.assertEqual(player.iterations, BOOP_BASELINE.iterations)

    def test_reference_profiles_match_gui_and_browser_defaults(self):
        root = Path(__file__).resolve().parents[2]
        for slug, module_name, name, baseline in (
            ("tic-tac-toe", "tic_tac_toe", "TicTacToe", TIC_TAC_TOE_BASELINE),
            ("boop", "boop", "Boop", BOOP_BASELINE),
            ("spotf", "spirits_of_the_forest", "SpiritsOfTheForest", SPOTF_BASELINE),
            ("connect-four", "connect_four", "ConnectFour", CONNECT_FOUR_BASELINE),
        ):
            with self.subTest(game=slug):
                profile = _load_mcts_profile(root / "configs/mcts" / f"{slug}-baseline.toml")
                module = import_module(f"meeple_bots.games.{module_name}.gui.controller")
                gui = getattr(module, name + "Gui")()
                agent = gui._agent(baseline, gui._cancelled)
                self.assertEqual(profile.agent, agent)
                self.assertEqual(gui.snapshot()["players"][1], baseline.as_dict())
                page = import_module(f"meeple_bots.games.{module_name}.gui.page").PAGE
                self.assertIn("const baseline = " + json.dumps(baseline.as_dict()), page)
                app_module = import_module(f"meeple_bots.games.{module_name}.gui.application")
                app = getattr(app_module, name + "Application")()
                with patch.object(lifecycle.threading.Thread, "start"):
                    state = app.start({"first": {"kind": "mcts"}, "second": {"kind": "mcts"}})
                self.assertEqual(state["players"], [baseline.as_dict()] * 2)
                app.cancel()

    def test_explicit_budgets_and_reuse_override_defaults(self):
        for baseline in (TIC_TAC_TOE_BASELINE, CONNECT_FOUR_BASELINE, BOOP_BASELINE, SPOTF_BASELINE):
            for overrides in ({"iterations": 16, "tree_reuse": False}, {"time_budget": 0.01}):
                player = parse_gui_player({"kind": "mcts", **overrides}, "first",
                                          default_rollout_depth=baseline.rollout_depth,
                                          default_mcts=baseline, available_heuristics=(0,))
                if "iterations" in overrides:
                    self.assertEqual(player.iterations, 16)
                    self.assertFalse(player.tree_reuse)
                    self.assertIsNone(player.time_budget)
                else:
                    self.assertIsNone(player.iterations)
                    self.assertEqual(player.time_budget, 0.01)
