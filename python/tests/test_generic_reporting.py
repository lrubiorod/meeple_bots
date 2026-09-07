"""Validate common report metrics and the tournament/extract/report workflow."""

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from meeple_bots.cli import main
from meeple_bots.reporting import generate_study_report


REPORT_AVAILABLE = all(
    importlib.util.find_spec(module) is not None for module in ("pandas", "matplotlib")
)


@unittest.skipUnless(REPORT_AVAILABLE, "optional report dependencies are not installed")
class GenericReportingTests(unittest.TestCase):
    def test_competitive_scores_seats_and_self_play(self):
        import pandas as pd
        from meeple_bots.reporting.generic import _competitive_results, _performance

        matches = pd.DataFrame([
            ("NA", "beta", "0", "false"),
            ("beta", "NA", "0", "false"),
            ("NA", "beta", "", "false"),
            ("beta", "NA", "", "false"),
            ("NA", "NA", "0", "true"),
        ], columns=["player_0_agent", "player_1_agent", "winner_player", "self_play"])
        results = _competitive_results(matches)
        performance = _performance(results, ["agent"]).set_index("agent")
        self.assertEqual(performance.loc["NA"].to_dict(), {
            "games": 4, "wins": 1, "draws": 2, "losses": 1, "score": 0.5,
        })
        seats = _performance(results, ["agent", "seat"]).set_index(["agent", "seat"])
        self.assertEqual(seats.loc[("NA", 0), "score"], 0.75)
        self.assertEqual(seats.loc[("NA", 1), "score"], 0.25)
        pairwise = _performance(results, ["agent", "opponent"])
        self.assertEqual(len(pairwise), 2)
        self.assertTrue((pairwise["games"] == 4).all())
        self.assertTrue(_competitive_results(matches.iloc[-1:]).empty)

    def test_actual_iterations_and_throughput_use_matching_measurements(self):
        import pandas as pd
        from meeple_bots.reporting.generic import _decision_performance

        agents = pd.DataFrame([
            ("mast", "", 0.2), ("random", "", ""), ("unplayed", 50, ""),
        ], columns=["agent_name", "iterations", "time_budget"])
        moves = pd.DataFrame([
            ("mast", 0.1, 100), ("mast", 0.3, 30),
            ("mast", "", 70), ("mast", 0.2, ""),
            ("random", 0.01, ""),
        ], columns=["agent", "decision_seconds", "search_iterations"])
        metrics = _decision_performance(moves, agents).set_index("agent")
        mast = metrics.loc["mast"]
        self.assertEqual(mast["decisions"], 4)
        self.assertEqual(mast["timed_decisions"], 3)
        self.assertEqual(mast["search_decisions"], 3)
        self.assertEqual(mast["throughput_decisions"], 2)
        self.assertAlmostEqual(mast["mean_decision_seconds"], 0.2)
        self.assertAlmostEqual(mast["mean_iterations"], 200 / 3)
        self.assertEqual(mast["p50_iterations"], 70)
        self.assertEqual(mast["min_iterations"], 30)
        self.assertEqual(mast["max_iterations"], 100)
        self.assertAlmostEqual(mast["iterations_per_second"], 325)
        self.assertEqual(mast["configured_time_budget"], 0.2)
        self.assertTrue(pd.isna(metrics.loc["random", "mean_iterations"]))
        self.assertTrue(pd.isna(metrics.loc["unplayed", "mean_decision_seconds"]))
        legacy = _decision_performance(moves.drop(columns="search_iterations"), agents)
        self.assertTrue((legacy["search_decisions"] == 0).all())

    def test_generic_reports_from_real_tournaments_and_protect_existing_output(self):
        import pandas as pd

        for game in ("connect-four", "tic-tac-toe"):
            with self.subTest(game=game), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = root / "study.toml"
                config.write_text(f'''game = "{game}"
output = "matches.jsonl"
matches_per_pair = 2
seat_mode = "paired"
workers = 1
max_plies = 42
[[agents]]
name = "NA"
kind = "mcts"
iterations = 8
rollout_depth = 42
[[agents]]
name = "random"
kind = "random"
''', encoding="utf-8")
                data = root / "matches" / "data"
                report = data.parent / "report"
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(main(["tournament", "--config", str(config)]), 0)
                    self.assertEqual(main(["extract", "--input", str(root / "matches.jsonl")]), 0)
                    self.assertEqual(main(["report", "--input", str(data)]), 0)
                summary = json.loads((report / "summary.json").read_text())
                self.assertEqual(summary["matches"], 2)
                self.assertEqual(summary["competitive_matches"], 2)
                self.assertEqual(summary["tables"], 4)
                self.assertTrue(summary["search_data_available"])
                self.assertTrue(summary["complete"])
                self.assertTrue((report / "figures" / "score.png").is_file())
                metrics = pd.read_csv(report / "tables" / "decision_performance.csv", keep_default_na=False)
                self.assertEqual(metrics.loc[metrics.agent == "NA", "mean_iterations"].iloc[0], "8.0")
                self.assertEqual(metrics.loc[metrics.agent == "random", "mean_iterations"].iloc[0], "")
                original = (report / "summary.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    generate_study_report(data)
                self.assertEqual((report / "summary.json").read_bytes(), original)

                manifest_path = data / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["complete"] = False
                manifest_path.write_text(json.dumps(manifest))
                generate_study_report(data, overwrite=True)
                self.assertIn("Preliminary", (report / "index.html").read_text())

                # Bad extraction must not replace a previously generated report.
                original = (report / "summary.json").read_bytes()
                manifest["row_counts"]["matches"] += 1
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "expected"):
                    generate_study_report(data, overwrite=True)
                self.assertEqual((report / "summary.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
