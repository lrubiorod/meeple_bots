"""Native Splendor integration, public chance replay and tournament persistence."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from meeple_bots import (
    Batch, Match, MctsAgent, RandomAgent, Splendor, SplendorAction,
    SplendorMoveKind, replay_splendor,
)
from meeple_bots.serialization import match_result_dict
from meeple_bots.cli import _load_tournament_config
from meeple_bots.tournaments import run_tournament, TournamentTrace


class SplendorTests(unittest.TestCase):
    def test_native_public_setup_and_chance(self):
        game = Splendor()
        state = game.initial_state(42)
        self.assertEqual(state, game.initial_state(42))
        self.assertEqual(state.bank, (4, 4, 4, 4, 4, 5))
        self.assertEqual(tuple(map(len, state.remaining)), (36, 26, 16))
        self.assertEqual(len(state.nobles), 3)
        self.assertEqual(state.consecutive_passes, 0)
        action = next(action for action in game.legal_actions(state)
                      if action.kind == SplendorMoveKind.RESERVE_VISIBLE)
        pending = game.apply_action(state, action)
        self.assertEqual(pending.status, "chance")
        self.assertEqual(game.legal_actions(pending), ())
        outcomes = game.chance_outcomes(pending)
        self.assertAlmostEqual(sum(probability for _, probability in outcomes), 1.0)
        outcome = outcomes[0][0]
        after = game.apply_chance_outcome(pending, outcome)
        self.assertEqual(after.status, "player_turn")
        self.assertEqual(sum(map(len, after.remaining)), 77)
        with self.assertRaises(TypeError):
            game.apply_action(pending, outcome)
        with self.assertRaises(ValueError):
            game.apply_chance_outcome(state, outcome)
        with self.assertRaises(ValueError):
            game.apply_action(state, SplendorAction(SplendorMoveKind.TAKE_SAME, color=5))
        with self.assertRaises(ValueError):
            game.apply_action(state, SplendorAction(SplendorMoveKind.PASS))

    def test_match_replay_reproducibility_and_batch(self):
        game = Splendor()
        first = Match(game, RandomAgent(), RandomAgent(), seed=42).run()
        second = Match(game, RandomAgent(), RandomAgent(), seed=42).run()
        self.assertEqual(first, second)
        self.assertTrue(first.chance_events)
        self.assertEqual(replay_splendor(42, first.moves, first.chance_events), first.splendor_state)
        with self.assertRaises(ValueError):
            replay_splendor(42, first.moves, first.chance_events[:-1])
        raw = match_result_dict(first)
        self.assertEqual(len(raw["chance_events"]), len(first.chance_events))
        self.assertTrue(raw["splendor_state"]["finished"])
        batch = Batch(game, RandomAgent(), RandomAgent(), matches=2, seed=42, workers=2).run()
        self.assertEqual(batch.matches, 2)

    def test_mcts_selectors_finish_reproducibly_with_chance_and_diagnostics(self):
        for selector in ("uct", "ucb1_tuned"):
            with self.subTest(selector=selector):
                agent = MctsAgent(
                    iterations=8, rollout_depth=20, selection_policy=selector, heuristic=0,
                    tree_reuse=True, transpositions=True, root_diagnostics=True,
                )
                match = Match(Splendor(), agent, RandomAgent(), seed=42)
                result = match.run()
                self.assertEqual(result, match.run())
                self.assertTrue(result.splendor_state.finished)
                self.assertTrue(result.chance_events)
                self.assertEqual(replay_splendor(42, result.moves, result.chance_events),
                                 result.splendor_state)
                decisions = [move for move in result.moves if move.player == 0]
                self.assertTrue(all(move.root_actions for move in decisions))
                self.assertTrue(all(move.tree_reuse is not None for move in decisions))
                self.assertTrue(all(move.search_iterations == 8 for move in decisions))
                raw = match_result_dict(result)
                self.assertEqual(raw["splendor_state"]["consecutive_passes"],
                                 result.splendor_state.consecutive_passes)

    def test_tournament_trace_resume_and_reject_missing_chance(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "study.toml"
            config.write_text('''game = "splendor"
output = "trace.jsonl"
matches_per_pair = 2
seed = 42
max_plies = 10000
workers = 2
seat_mode = "paired"
[[agents]]
name = "a"
kind = "mcts"
iterations = 8
rollout_depth = 20
[[agents]]
name = "b"
kind = "random"
''')
            parsed = _load_tournament_config(config)
            run_tournament(parsed)
            trace = Path(tmp) / "trace.jsonl"
            original = trace.read_bytes()
            header = json.loads(original.splitlines()[0])
            with TournamentTrace(trace, header, resume=True):
                pass
            self.assertEqual(trace.read_bytes(), original)
            rows = [json.loads(line) for line in original.splitlines()]
            altered = json.loads(json.dumps(rows))
            altered[1]["result"]["scores"][0] += 1
            trace.write_text("".join(json.dumps(row) + "\n" for row in altered))
            with self.assertRaisesRegex(ValueError, "scores differ"):
                with TournamentTrace(trace, header, resume=True):
                    pass
            rows[1]["result"]["chance_events"] = []
            trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaises(ValueError):
                with TournamentTrace(trace, header, resume=True):
                    pass
