from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from meeple_bots import Boop, ConnectFour, RandomAgent, TicTacToe
from meeple_bots.api import Match
from meeple_bots.extraction import extract_tournament
from meeple_bots.games.boop.gui.page import PAGE as BOOP_PAGE
from meeple_bots.games.connect_four.gui.page import PAGE as CONNECT_FOUR_PAGE
from meeple_bots.games.tic_tac_toe.gui.page import PAGE as TIC_TAC_TOE_PAGE
from meeple_bots.gui.player import GuiPlayer
from meeple_bots.gui.trace import write_gui_trace


class OtherGuiTraceTests(unittest.TestCase):
    def test_pages_have_independent_mcts_budgets_and_trace_toggle(self) -> None:
        for page in (TIC_TAC_TOE_PAGE, CONNECT_FOUR_PAGE, BOOP_PAGE):
            for index in (0, 1):
                self.assertIn(f'id="mcts-{index}"', page)
                self.assertIn(f'id="budget-mode-{index}"', page)
                self.assertIn(f'id="iterations-{index}"', page)
                self.assertIn(f'id="time-budget-{index}"', page)
                self.assertIn(f'id="tree-reuse-{index}"', page)
            self.assertIn('id="save-trace"', page)
            self.assertIn("if (kind !== 'mcts') return {kind}", page)

    @unittest.skipUnless(shutil.which("node"), "Node is required for GUI syntax tests")
    def test_pages_have_valid_embedded_javascript(self) -> None:
        pages = {
            "tic-tac-toe": TIC_TAC_TOE_PAGE,
            "connect-four": CONNECT_FOUR_PAGE,
            "boop": BOOP_PAGE,
        }
        for game, page in pages.items():
            with self.subTest(game=game):
                script = page.split("<script>", 1)[1].rsplit("</script>", 1)[0]
                completed = subprocess.run(
                    ["node", "--check"],
                    input=script,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_completed_gui_traces_are_extractable(self) -> None:
        cases = (
            ("tic-tac-toe", TicTacToe(), 9),
            ("connect-four", ConnectFour(), 42),
            ("boop", Boop(), 10_000),
        )
        players = (
            GuiPlayer("random"),
            GuiPlayer("random"),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for game_name, game, max_plies in cases:
                with self.subTest(game=game_name):
                    result = Match(
                        game=game,
                        first=RandomAgent(),
                        second=RandomAgent(),
                        seed=42,
                        max_plies=max_plies,
                    ).run()
                    trace = write_gui_trace(
                        root / game_name / "traces",
                        game=game_name,
                        max_plies=max_plies,
                        result=result,
                        players=players,
                        duration_seconds=0.5,
                    )

                    records = [
                        json.loads(line)
                        for line in trace.read_text(encoding="utf-8").splitlines()
                    ]
                    self.assertEqual(records[0]["game"], game_name)
                    self.assertEqual(records[1]["result"]["plies"], result.plies)
                    summary = extract_tournament(
                        trace,
                        root / game_name / "extracted",
                    )
                    with Path(summary["output_dir"], "moves.csv").open(
                        encoding="utf-8",
                        newline="",
                    ) as source:
                        moves = list(csv.DictReader(source))
                    self.assertTrue(summary["complete"])
                    self.assertEqual(summary["processed_matches"], 1)
                    self.assertEqual(len(moves), result.plies)
                    self.assertEqual(moves[-1]["terminal_after"], "True")
                    self.assertFalse(list(trace.parent.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
