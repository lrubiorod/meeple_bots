"""Total agent timing survives native conversion, traces and extraction."""

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from meeple_bots import (
    Boop, ConnectFour, Match, MctsAgent, RandomAgent, SpiritsOfTheForest, TicTacToe,
)
from meeple_bots.extraction import extract_tournament
from meeple_bots.serialization import match_result_dict
from meeple_bots.tournaments import (
    TournamentAgent, TournamentConfig, TournamentTrace, run_tournament, tournament_header,
)


class TimingTests(unittest.TestCase):
    def test_all_games_return_finalized_component_times_with_and_without_observer(self):
        for game in (TicTacToe(), ConnectFour(), Boop(), SpiritsOfTheForest()):
            with self.subTest(game=type(game).__name__):
                observed = []
                match = Match(game=game, first=RandomAgent(), second=RandomAgent(), seed=42)
                plain = match.run()
                live = replace(match, observe_move=observed.append).run()
                self.assertEqual(plain, live)
                self.assertEqual(len(observed), len(live.moves))
                self.assertEqual(live.unassigned_maintenance_seconds, (0.0, 0.0))
                for result in (plain, live):
                    serialized = match_result_dict(result)
                    for move, raw in zip(result.moves, serialized["moves"], strict=True):
                        self.assertGreater(move.selection_seconds, 0)
                        self.assertGreater(move.maintenance_seconds, 0)
                        self.assertAlmostEqual(
                            move.decision_seconds,
                            move.selection_seconds + move.maintenance_seconds,
                        )
                        self.assertEqual(raw["selection_seconds"], move.selection_seconds)
                        self.assertEqual(raw["maintenance_seconds"], move.maintenance_seconds)
                for observation, move in zip(observed, live.moves, strict=True):
                    self.assertLessEqual(observation.decision_seconds, move.decision_seconds)

    def test_actual_iterations_are_preserved_with_tree_and_graph_reuse(self):
        for transpositions in (False, True):
            for budget in ({"iterations": 32}, {"time_budget": 0.001}):
                with self.subTest(transpositions=transpositions, budget=budget):
                    result = Match(
                        game=ConnectFour(),
                        first=MctsAgent(
                            **budget, rollout_depth=42, tree_reuse=True,
                            transpositions=transpositions,
                        ),
                        second=RandomAgent(), seed=23,
                    ).run()
                    for move in result.moves:
                        self.assertAlmostEqual(
                            move.decision_seconds,
                            move.selection_seconds + move.maintenance_seconds,
                        )
                        if move.player == 0:
                            self.assertGreaterEqual(move.search_iterations, 1)
                            if "iterations" in budget:
                                self.assertEqual(move.search_iterations, 32)

    def config(self, path):
        return TournamentConfig(
            game=TicTacToe(), output=path, pairing_mode="round_robin",
            seat_mode="paired", matches_per_pair=2, seed=42,
            max_plies=9, workers=1,
            agents=(
                TournamentAgent("control", RandomAgent()),
                TournamentAgent("search", MctsAgent(iterations=8, rollout_depth=9)),
            ),
        )

    def test_extraction_preserves_total_and_components_and_rejects_mixed_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config(root / "current.jsonl")
            run_tournament(config)
            records = [json.loads(line) for line in config.output.read_text().splitlines()]
            self.assertEqual(records[0]["decision_timing_scope"], "agent_total_v1")
            manifest = extract_tournament(config.output, root / "current-data")
            self.assertEqual(manifest["decision_timing_scope"], "agent_total_v1")
            with (root / "current-data/moves.csv").open() as source:
                rows = list(csv.DictReader(source))
            moves = [move for record in records[1:] for move in record["result"]["moves"]]
            for row, move in zip(rows, moves, strict=True):
                for field in ("decision_seconds", "selection_seconds", "maintenance_seconds"):
                    self.assertEqual(float(row[field]), move[field])

            # Simulate a historical selection-only trace. Its unknown maintenance
            # must remain unknown; it cannot be reconstructed from old records.
            del records[0]["decision_timing_scope"]
            for move in moves:
                move["decision_seconds"] = move.pop("selection_seconds")
                del move["maintenance_seconds"]
            legacy = root / "legacy.jsonl"
            legacy.write_text("".join(json.dumps(record) + "\n" for record in records))
            legacy_manifest = extract_tournament(legacy, root / "legacy-data")
            self.assertEqual(legacy_manifest["decision_timing_scope"], "selection_only_legacy")
            with (root / "legacy-data/moves.csv").open() as source:
                self.assertTrue(all(row["maintenance_seconds"] == "" for row in csv.DictReader(source)))
            with self.assertRaisesRegex(ValueError, "timing studies"):
                extract_tournament([config.output, legacy], root / "mixed")
            self.assertFalse((root / "mixed").exists())

            original = legacy.read_bytes()
            with self.assertRaisesRegex(ValueError, "header differs"):
                with TournamentTrace(legacy, tournament_header(config, config.output, 1), resume=True):
                    pass
            self.assertEqual(legacy.read_bytes(), original)
