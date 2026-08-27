from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from meeple_bots import (
    ForestPosition,
    MctsAgent,
    RandomAgent,
    SpiritGemstoneSacrifice,
    SpiritsOfTheForest,
    TakeSpiritTile,
)
from meeple_bots.api import Match
from meeple_bots.extraction import extract_tournament
from meeple_bots.games.spirits_of_the_forest.gui.controller import _serialize_action
from meeple_bots.games.spirits_of_the_forest.gui.page import PAGE
from meeple_bots.gui.player import GuiPlayer, parse_gui_player
from meeple_bots.gui.trace import write_gui_trace


class SpotfGuiInteractionTests(unittest.TestCase):
    def test_forest_sacrifice_serializes_target_and_payment_separately(self) -> None:
        action = TakeSpiritTile(
            ForestPosition(0, 11),
            SpiritGemstoneSacrifice(ForestPosition(2, 3)),
        )

        self.assertEqual(
            _serialize_action(action, 7),
            {
                "index": 7,
                "kind": "take_tile",
                "row": 0,
                "column": 11,
                "sacrifice": {
                    "kind": "forest",
                    "row": 2,
                    "column": 3,
                },
            },
        )

    def test_page_explains_counter_totals_and_guided_sacrifices(self) -> None:
        self.assertIn("Contadores: obtenido/total", PAGE)
        self.assertIn("Reserva rival: elige una gema propia", PAGE)
        self.assertIn(".players{display:grid;grid-template-columns:1fr", PAGE)
        self.assertIn("grid-template-columns:repeat(12,minmax(0,1fr))", PAGE)
        self.assertEqual(PAGE.count('value="0">H0 · Progreso alcanzable'), 2)
        self.assertNotIn("H1 ·", PAGE)
        self.assertNotIn("H2 ·", PAGE)
        self.assertEqual(PAGE.count('value="none" selected>Ninguna'), 2)
        self.assertIn("heuristic==='none'?null:Number(heuristic)", PAGE)

    @unittest.skipUnless(shutil.which("node"), "Node is required for embedded GUI tests")
    def test_embedded_javascript_guides_every_board_selection(self) -> None:
        script = PAGE.split("<script>", 1)[1].rsplit("</script>", 1)[0]
        assertions = r"""
{
const assert=require('node:assert/strict');
const {
  reduceBoardClick,
  reconcileSelection,
  interactionCells,
  availableSacrifice,
  counterEntries,
}=module.exports;

const direct=[
  {index:3,kind:'take_tile',row:0,column:0,sacrifice:null},
  {index:4,kind:'place_gemstone',row:1,column:2},
];
assert.deepEqual(reduceBoardClick(direct,null,0),{actionIndex:3,selection:null});
assert.deepEqual(reduceBoardClick(direct,null,14),{actionIndex:4,selection:null});

const singleSacrifice=[
  {index:10,kind:'take_tile',row:0,column:1,sacrifice:{kind:'available'}},
];
assert.deepEqual(
  reduceBoardClick(singleSacrifice,null,1),
  {actionIndex:null,selection:{mode:'sacrifice',target:1}},
);
assert.equal(
  availableSacrifice(singleSacrifice,{mode:'sacrifice',target:1}).index,
  10,
);

const sacrifices=[
  {index:5,kind:'take_tile',row:0,column:11,sacrifice:{kind:'available'}},
  {index:6,kind:'take_tile',row:0,column:11,sacrifice:{kind:'forest',row:2,column:3}},
];
let result=reduceBoardClick(sacrifices,null,11);
assert.deepEqual(result,{actionIndex:null,selection:{mode:'sacrifice',target:11}});
assert.deepEqual(reconcileSelection(sacrifices,result.selection),result.selection);
assert.equal(availableSacrifice(sacrifices,result.selection).index,5);
let view=interactionCells(sacrifices,result.selection);
assert.deepEqual([...view.paymentOptions],[27]);
assert.deepEqual([...view.selected],[11]);
assert.deepEqual(
  reduceBoardClick(sacrifices,result.selection,27),
  {actionIndex:6,selection:null},
);
assert.equal(reconcileSelection([],result.selection),null);

const moves=[
  {index:8,kind:'move_gemstone',source_row:0,source_column:0,target_row:1,target_column:0},
  {index:9,kind:'move_gemstone',source_row:0,source_column:1,target_row:1,target_column:0},
];
assert.deepEqual(reduceBoardClick(moves,null,12),{actionIndex:null,selection:null});
result=reduceBoardClick(moves,null,0);
assert.deepEqual(result,{actionIndex:null,selection:{mode:'move',source:0}});
view=interactionCells(moves,result.selection);
assert.deepEqual([...view.sourceOptions],[0,1]);
assert.deepEqual([...view.destinations],[12]);
assert.deepEqual(
  reduceBoardClick(moves,result.selection,12),
  {actionIndex:8,selection:null},
);

const counters=counterEntries({
  spirit_symbols:[0,1,2,3,4,5,6,7,8],
  power_sources:[2,3,4],
});
assert.deepEqual(counters.map(item=>item.total),[5,6,6,7,7,8,8,8,10,9,9,9]);
assert.deepEqual(counters.map(item=>item.value),[0,1,2,3,4,5,6,7,8,2,3,4]);
assert.ok(counters.every(item=>item.color.startsWith('#')));
}
"""

        completed = subprocess.run(
            ["node"],
            input=f"{script}\n{assertions}",
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


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
                "heuristic": 0,
                "tree_reuse": True,
            },
            "second",
            default_rollout_depth=64,
            available_heuristics=(0,),
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
                heuristic=0,
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
                heuristic=0,
                tree_reuse=True,
            ),
            seed=42,
            max_plies=256,
        ).run()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = write_gui_trace(
                root / "traces",
                game="spotf",
                max_plies=256,
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
            self.assertEqual(summary["row_counts"]["moves"], result.plies)
            self.assertTrue(Path(summary["output_dir"], "moves.csv").is_file())


if __name__ == "__main__":
    unittest.main()
