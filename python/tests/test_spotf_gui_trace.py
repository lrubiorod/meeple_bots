from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from meeple_bots import MctsAgent, RandomAgent, SpiritsOfTheForest
from meeple_bots.api import Match
from meeple_bots.extraction import extract_tournament
from meeple_bots.games.spirits_of_the_forest.gui.page import PAGE
from meeple_bots.games.spirits_of_the_forest.gui.trace import write_gui_trace
from meeple_bots.gui.player import GuiPlayer, parse_gui_player


class SpotfGuiTraceTests(unittest.TestCase):
    def test_page_has_independent_mcts_configs(self) -> None:
        for index in (0, 1):
            self.assertIn(f'id="mcts-config-{index}"', PAGE)
            self.assertIn(f'id="budget-mode-{index}"', PAGE)
            self.assertIn(f'id="tree-reuse-{index}"', PAGE)
        self.assertNotIn('id="budget-mode"', PAGE)
        self.assertIn("[hidden]{display:none!important}", PAGE)

    def test_gui_player_accepts_time_budget_and_tree_reuse(self) -> None:
        player = parse_gui_player(
            {
                "kind": "mcts",
                "iterations": None,
                "time_budget": 0.01,
                "exploration": 1.0,
                "rollout_depth": 13,
                "heuristic": 2,
                "tree_reuse": True,
            },
            "second",
            default_rollout_depth=64,
            available_heuristics=(0, 1, 2),
        )

        self.assertIsNone(player.iterations)
        self.assertEqual(player.time_budget, 0.01)
        self.assertTrue(player.tree_reuse)
        self.assertEqual(player.as_dict()["time_budget"], 0.01)

    def test_gui_player_rejects_two_mcts_budgets(self) -> None:
        with self.assertRaises(ValueError):
            parse_gui_player(
                {"kind": "mcts", "iterations": 10, "time_budget": 0.01},
                "second",
                default_rollout_depth=64,
            )

    def test_completed_gui_trace_is_extractable(self) -> None:
        players = (
            GuiPlayer("random"),
            GuiPlayer(
                "mcts",
                iterations=8,
                rollout_depth=4,
                exploration=1.0,
                heuristic=2,
                tree_reuse=True,
            ),
        )
        result = Match(
            game=SpiritsOfTheForest(),
            first=RandomAgent(),
            second=MctsAgent(
                iterations=8,
                rollout_depth=4,
                exploration=1.0,
                heuristic=2,
                tree_reuse=True,
            ),
            seed=42,
            max_plies=256,
        ).run()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = write_gui_trace(
                root / "traces",
                result=result,
                players=players,
                duration_seconds=0.5,
            )
            records = [
                json.loads(line)
                for line in trace.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["study_type"], "batch")
            self.assertTrue(records[0]["agents"][1]["tree_reuse"])
            self.assertEqual(records[0]["agents"][1]["iterations"], 8)
            self.assertFalse(list(trace.parent.glob("*.tmp")))

            summary = extract_tournament(trace, root / "extracted")
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["processed_matches"], 1)


if __name__ == "__main__":
    unittest.main()
