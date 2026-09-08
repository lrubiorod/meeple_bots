"""Validate common report metrics and the tournament/extract/report workflow."""

import csv
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
    def test_specialized_reports_preserve_names_and_real_missing_values(self):
        import pandas as pd
        from meeple_bots.games.boop import reporting as boop
        from meeple_bots.games.spirits_of_the_forest import reporting as spotf

        for module in (boop, spotf):
            for name in ("NA", "N/A", "NULL", "nan", "001"):
                with self.subTest(report=module.__name__, name=name), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    manifest = {"tables": {}, "row_counts": {}}
                    fixtures = {
                        "agents": (["agent_name", "iterations"], [[name, ""], ["beta", 8]]),
                        "matches": (
                            ["agent_a", "agent_b", "player_0_agent", "player_1_agent", "winner_agent", "winner_player", "self_play"],
                            [[name, "beta", name, "beta", name, 0, False],
                             [name, "beta", "beta", name, "beta", 0, False],
                             [name, "beta", name, "beta", "", "", False]],
                        ),
                        "root_actions": (["agent", "mean_utility"], [[name, ""], ["beta", 0.5]]),
                    }
                    for table in (*module._REQUIRED_TABLES, "root_actions"):
                        columns, rows = fixtures.get(table, (["agent"], []))
                        filename = f"{table}.csv"
                        with (root / filename).open("w", newline="") as output:
                            writer = csv.writer(output)
                            writer.writerow(columns)
                            writer.writerows(rows)
                        manifest["tables"][table] = filename
                        manifest["row_counts"][table] = len(rows)
                    tables = module._load_tables(root, manifest)
                    self.assertEqual(tables["agents"]["agent_name"].tolist(), [name, "beta"])
                    self.assertTrue(pd.isna(tables["agents"].iloc[0]["iterations"]))
                    matches = tables["matches"]
                    self.assertEqual(matches["winner_agent"].iloc[0], name)
                    self.assertTrue(pd.isna(matches["winner_agent"].iloc[2]))
                    self.assertEqual(matches["winner_player"].iloc[0], 0)
                    matches["self_play"] = module._boolean(matches["self_play"])
                    agents = tables["agents"]["agent_name"].tolist()
                    performance = module._agent_performance(matches, agents)
                    for row in performance.to_dict(orient="records"):
                        self.assertEqual([row[key] for key in ("games", "wins", "losses", "draws")], [3, 1, 1, 1])
                        self.assertAlmostEqual(row["win_rate"], 1 / 3)
                    pairwise = module._pairwise_performance(matches, agents)
                    self.assertEqual(set(pairwise["agent"]), {name, "beta"})
                    self.assertTrue((pairwise["wins"] == 1).all())
                    if module is spotf:
                        self.assertEqual(tables["root_actions"]["agent"].iloc[0], name)
                        self.assertTrue(pd.isna(tables["root_actions"]["mean_utility"].iloc[0]))

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

    def test_common_statistics_preserve_specialized_views_and_empty_pairs(self):
        import pandas as pd
        from meeple_bots.reporting.common import competitive_performance, first_player_advantage
        from meeple_bots.games.boop import reporting as boop
        from meeple_bots.games.spirits_of_the_forest import reporting as spotf

        matches = pd.DataFrame([
            ("NA", "beta", 0, False),
            ("beta", "NA", 0, False),
            ("NA", "beta", None, False),
            ("NA", "NA", 0, True),
        ], columns=["player_0_agent", "player_1_agent", "winner_player", "self_play"])
        matches["agent_a"] = matches["player_0_agent"]
        matches["agent_b"] = matches["player_1_agent"]
        agents = ["beta", "NA", "unplayed"]
        performance = competitive_performance(matches, agents).set_index("agent")
        self.assertEqual(performance.index.tolist(), agents)
        for name in agents[:2]:
            self.assertEqual(performance.loc[name, ["games", "wins", "draws", "losses"]].tolist(), [3, 1, 1, 1])
            self.assertEqual(performance.loc[name, "score"], 0.5)
            self.assertAlmostEqual(performance.loc[name, "win_rate"], 1 / 3)
        self.assertEqual(performance.loc["unplayed", "games"], 0)
        self.assertEqual(performance.loc["unplayed", "ci_high"], 0)
        for module in (boop, spotf):
            pairs = module._pairwise_performance(matches, agents)
            self.assertEqual(pairs["agent"].tolist(), ["beta", "NA"])
            empty = module._pairwise_performance(matches.iloc[-1:], agents)
            self.assertTrue(empty.empty)
            self.assertEqual(empty.columns.tolist(), pairs.columns.tolist())
            self.assertTrue((module._agent_performance(matches.iloc[-1:], agents)["games"] == 0).all())
        self.assertEqual(spotf._pairwise_performance(matches, agents)["score_rate"].tolist(), [0.5, 0.5])
        first = first_player_advantage(matches).set_index("pairing")
        # Historical overall seat advantage includes self-play and excludes draws.
        self.assertEqual(first.loc["overall", "games"], 4)
        self.assertEqual(first.loc["overall", "decisive_games"], 3)
        self.assertEqual(first.loc["NA vs beta", "decisive_games"], 2)
        self.assertEqual(first.loc["overall", "player_0_win_rate"], 1)

    def test_common_loader_validates_required_and_optional_row_counts(self):
        from meeple_bots.reporting.common import load_tables

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "matches.csv").write_text("winner_player\n0\n")
            (root / "roots.csv").write_text("agent\nNA\n")
            manifest = {"tables": {"matches": "matches.csv"}, "row_counts": {"matches": 1}}
            tables = load_tables(root, manifest, ("matches",), empty_as_missing=True, optional=("root_actions",))
            self.assertTrue(tables["root_actions"].empty)
            manifest["tables"]["root_actions"] = "roots.csv"
            manifest["row_counts"]["root_actions"] = 2
            with self.assertRaisesRegex(ValueError, "root_actions has 1 rows; expected 2"):
                load_tables(root, manifest, ("matches",), empty_as_missing=True, optional=("root_actions",))
            manifest["row_counts"]["matches"] = 0
            with self.assertRaisesRegex(ValueError, "matches has 1 rows; expected 0"):
                load_tables(root, manifest, ("matches",), empty_as_missing=False)

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
        # Even when selection is cheap, throughput must use the total agent cost.
        moves["selection_seconds"] = 0.001
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
