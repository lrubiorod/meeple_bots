"""Generic extraction must replay games, not merely trust trace structure."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from meeple_bots.extraction import extract_tournament


class GenericReplayTests(unittest.TestCase):
    def records(self, game, sequence, winner=0):
        actions = [
            {"type": "connect_four", "column": value}
            if game == "connect-four" else
            {"type": "tic_tac_toe", "row": value // 3, "column": value % 3}
            for value in sequence
        ]
        utilities = [0, 0] if winner is None else [1, -1] if winner == 0 else [-1, 1]
        return [
            {
                "record_type": "tournament", "schema_version": 1,
                "game": game, "total_matches": 1,
                "agents": [{"name": name, "type": "random"} for name in ("alpha", "beta")],
            },
            {
                "record_type": "match", "match_number": 1,
                "pairing_number": 1, "pairing_match_number": 1,
                "agent_a": "alpha", "agent_b": "beta", "agent_a_player": 0,
                "players": ["alpha", "beta"], "self_play": False,
                "winner": None if winner is None else "agent_a" if winner == 0 else "agent_b",
                "result": {
                    "seed": 0, "plies": len(actions), "winner": winner,
                    "utilities": utilities,
                    "moves": [
                        {"ply": index + 1, "player": index % 2, "action": action}
                        for index, action in enumerate(actions)
                    ],
                },
            },
        ]

    def write(self, path, records):
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    def test_completed_wins_and_draws_are_accepted(self):
        cases = (
            ("tic-tac-toe", [0, 3, 1, 4, 2], 0),
            ("tic-tac-toe", [0, 3, 1, 4, 8, 5], 1),
            ("tic-tac-toe", [0, 1, 2, 4, 3, 5, 7, 6, 8], None),
            ("connect-four", [0, 0, 1, 1, 2, 2, 3], 0),
            ("connect-four", [0, 1, 0, 1, 2, 1, 2, 1], 1),
        )
        for game, sequence, winner in cases:
            with self.subTest(game=game, winner=winner), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                trace = root / "study.jsonl"
                self.write(trace, self.records(game, sequence, winner))
                summary = extract_tournament(trace, root / "data")
                self.assertTrue(summary["complete"])
                with (root / "data/moves.csv").open() as source:
                    moves = list(csv.DictReader(source))
                self.assertEqual([row["terminal_after"] for row in moves], ["False"] * (len(sequence) - 1) + ["True"])

    def test_structurally_valid_but_illegal_games_are_rejected(self):
        cases = (
            ("tic-tac-toe", [0, 0, 1, 4, 2]),  # Reuse an occupied cell.
            ("connect-four", [0] * 7),  # Overflow a full column.
            ("tic-tac-toe", [0, 3, 1, 4]),  # Stop before terminal.
            ("connect-four", [0, 0, 1, 1, 2, 2]),
            ("tic-tac-toe", [0, 3, 1, 4, 2, 8]),  # Action after a win.
            ("connect-four", [0, 0, 1, 1, 2, 2, 3, 6]),
            ("tic-tac-toe", []),
            ("connect-four", []),
        )
        for game, sequence in cases:
            with self.subTest(game=game, sequence=sequence):
                self.assert_rejected(self.records(game, sequence))

    def assert_rejected(self, records):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "broken.jsonl"
            self.write(trace, records)
            data = root / "data"
            data.mkdir()
            existing = data / "matches.csv"
            existing.write_bytes(b"previous extraction\n")
            with self.assertRaisesRegex(ValueError, "match 1"):
                extract_tournament(trace, data, overwrite=True)
            self.assertEqual(existing.read_bytes(), b"previous extraction\n")
            self.assertEqual(sorted(path.name for path in data.iterdir()), ["matches.csv"])

    def test_invalid_action_types_players_coordinates_and_results_are_rejected(self):
        for game, sequence in (
            ("tic-tac-toe", [0, 3, 1, 4, 2]),
            ("connect-four", [0, 0, 1, 1, 2, 2, 3]),
        ):
            for corruption in ("type", "coordinate", "boolean", "player", "winner", "utilities", "boolean_utility"):
                with self.subTest(game=game, corruption=corruption):
                    records = copy.deepcopy(self.records(game, sequence))
                    result = records[1]["result"]
                    first = result["moves"][0]
                    if corruption == "type":
                        first["action"] = {"type": "not-a-game", "column": 999}
                    elif corruption == "coordinate":
                        first["action"]["column"] = 999
                    elif corruption == "boolean":
                        first["action"]["column"] = True
                    elif corruption == "player":
                        first["player"] = 1
                    elif corruption == "winner":
                        result["winner"] = 1
                        records[1]["winner"] = "agent_b"  # Internally consistent forgery.
                        result["utilities"] = [-1, 1]
                    elif corruption == "utilities":
                        result["utilities"] = [0, 0]
                    else:
                        result["utilities"] = [True, -1]
                    self.assert_rejected(records)
