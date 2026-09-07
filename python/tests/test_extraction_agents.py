"""Protect complete agent identity when extracting and combining studies."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from meeple_bots.extraction import extract_tournament


EVALUATORS = (
    "cutoff_evaluator", "rollout_evaluator",
    "rollout_fallback_evaluator", "progressive_bias_evaluator",
)


class AgentExtractionTests(unittest.TestCase):
    def agent(self):
        return {
            "name": "search", "type": "mcts", "iterations": 50,
            "rollout_depth": 8, "exploration": 1.4,
            "rollout_policy": "conditional",
            "rollout_condition": {"kind": "turn_phase", "phase": "collect"},
            "rollout_primary_policy": "epsilon_greedy", "rollout_epsilon": 0.25,
            "rollout_fallback_policy": "greedy", "progressive_bias_weight": 0.1,
            "root_diagnostics": False, "self_play": False,
            **{
                field: {
                    "kind": "game_heuristic", "index": 0,
                    "params": {"gemstone_early_bonus": 0},
                }
                for field in EVALUATORS
            },
        }

    def trace(self, path, *agents):
        # No matches are needed to reproduce corruption of configuration identity.
        path.write_text(json.dumps({
            "record_type": "tournament", "schema_version": 1,
            "game": "spotf", "total_matches": 0, "agents": agents,
        }) + "\n", encoding="utf-8")
        return path

    def rows(self, directory):
        with (directory / "agents.csv").open(encoding="utf-8", newline="") as source:
            return list(csv.DictReader(source))

    def test_same_name_with_different_parameters_or_diagnostics_is_rejected(self):
        for field in (*EVALUATORS, "root_diagnostics"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                first = self.agent()
                second = copy.deepcopy(first)
                if field == "root_diagnostics":
                    second[field] = True
                else:
                    second[field]["params"]["gemstone_early_bonus"] = 10
                traces = [
                    self.trace(root / "first.jsonl", first),
                    self.trace(root / "second.jsonl", second),
                ]
                output = root / "data"
                with self.assertRaisesRegex(ValueError, 'agent "search" has conflicting configurations') as error:
                    extract_tournament(traces, output)
                self.assertIn(str(traces[0]), str(error.exception))
                self.assertIn(str(traces[1]), str(error.exception))
                self.assertFalse(output.exists())

                # Even --overwrite must not replace valid prior data on conflict.
                output.mkdir()
                existing = output / "agents.csv"
                existing.write_bytes(b"existing results\n")
                with self.assertRaises(ValueError):
                    extract_tournament(traces, output, overwrite=True)
                self.assertEqual(existing.read_bytes(), b"existing results\n")

    def test_equivalent_configurations_merge_without_losing_nested_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.agent()
            second = dict(reversed(copy.deepcopy(first).items()))
            second["self_play"] = True
            for field in EVALUATORS:
                second[field] = dict(reversed(second[field].items()))
                second[field]["params"]["gemstone_early_bonus"] = 0.0
            traces = [
                self.trace(root / "first.jsonl", first),
                self.trace(root / "second.jsonl", second),
            ]
            extract_tournament(traces, root / "data")
            rows = self.rows(root / "data")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["self_play"], "True")
            config = json.loads(rows[0]["config_json"])
            self.assertEqual(config, {k: v for k, v in first.items() if k not in {"name", "self_play"}})
            self.assertEqual(rows[0]["config_json"], json.dumps(config, sort_keys=True, separators=(",", ":")))

    def test_distinct_names_preserve_distinct_configurations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.agent()
            second = copy.deepcopy(first)
            second["name"] = "search-bonus"
            second["cutoff_evaluator"]["params"]["gemstone_early_bonus"] = 10
            trace = self.trace(root / "study.jsonl", first, second)
            extract_tournament(trace, root / "data")
            rows = self.rows(root / "data")
            self.assertEqual([row["agent_name"] for row in rows], ["search", "search-bonus"])
            self.assertEqual([
                json.loads(row["config_json"])["cutoff_evaluator"]["params"]["gemstone_early_bonus"]
                for row in rows
            ], [0, 10])

    def test_legacy_and_random_configurations_preserve_only_recorded_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            agents = (
                {"name": "legacy", "type": "mcts", "iterations": 10, "heuristic": 0},
                {"name": "random", "type": "random"},
            )
            trace = self.trace(root / "legacy.jsonl", *agents)
            extract_tournament(trace, root / "data")
            for row, original in zip(self.rows(root / "data"), agents, strict=True):
                self.assertEqual(
                    json.loads(row["config_json"]),
                    {k: v for k, v in original.items() if k != "name"},
                )
