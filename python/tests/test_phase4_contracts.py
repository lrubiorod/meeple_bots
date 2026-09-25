"""Public identities and wire contracts kept during the match boundary refactor."""

import subprocess
import sys
import unittest

import meeple_bots
from meeple_bots import api, serialization
from meeple_bots.matches import execution, models, trace
from meeple_bots.serialization import action_dict
from meeple_bots import tournaments
from meeple_bots.connect6 import Connect6Action
from meeple_bots.lost_cities import LostCitiesAction
from meeple_bots.splendor import SplendorAction, SplendorMoveKind


class Phase4Contracts(unittest.TestCase):
    def test_public_types_are_the_authoritative_match_types(self):
        for name in (
            "Move", "MatchResult", "BatchResult", "BatchMatchResult",
            "BatchProgress", "BatchProgressStatus", "RootActionDiagnostic",
            "TreeReuseDiagnostic",
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(api, name), getattr(models, name))
                self.assertIs(getattr(meeple_bots, name), getattr(models, name))
        for name in ("Match", "Batch"):
            with self.subTest(name=name):
                self.assertIs(getattr(api, name), getattr(execution, name))
                self.assertIs(getattr(meeple_bots, name), getattr(execution, name))

    def test_wire_codecs_keep_the_serialization_import_path(self):
        for name in (
            "action_dict", "agent_dict", "game_name", "match_result_dict",
            "trace_match_dict", "write_jsonl",
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(serialization, name), getattr(trace, name))
        self.assertIs(tournaments.TournamentTrace, trace.TournamentTrace)
        self.assertIs(tournaments.match_record, trace.match_record)

    def test_representative_action_encodings(self):
        self.assertEqual(
            action_dict(api.TicTacToeAction(0, 1)),
            {"type": "tic_tac_toe", "row": 0, "column": 1},
        )
        self.assertEqual(
            action_dict(api.ConnectFourAction(2)),
            {"type": "connect_four", "column": 2},
        )
        self.assertEqual(
            action_dict(api.BoopAction(api.BoopPieceKind.KITTEN, 1, 2)),
            {"type": "boop", "piece": "kitten", "row": 1, "column": 2,
             "resolution": {"type": "none"}},
        )
        self.assertEqual(
            action_dict(api.EndSpiritCollection()),
            {"type": "spotf", "kind": "end_collection"},
        )
        self.assertEqual(action_dict(Connect6Action(3)), {"type": "connect6", "position": 3})
        self.assertEqual(
            action_dict(LostCitiesAction("play", card=(2, 5))),
            {"type": "lost_cities", "kind": "play", "card": [2, 5]},
        )
        self.assertEqual(
            action_dict(SplendorAction(SplendorMoveKind.PASS)),
            {"type": "splendor", "kind": "pass", "returned": [0] * 6,
             "payment": [0] * 6, "noble": None},
        )

    def test_non_report_imports_do_not_require_plotting_extras(self):
        code = """
import importlib.abc
import sys
class BlockReports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pandas', 'matplotlib', 'seaborn', 'numpy'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockReports())
import meeple_bots
import meeple_bots.cli
import meeple_bots.studies
import meeple_bots.probes
import meeple_bots.reporting
from meeple_bots import Match, Batch
assert not {'pandas', 'matplotlib', 'seaborn', 'numpy'} & sys.modules.keys()
"""
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
