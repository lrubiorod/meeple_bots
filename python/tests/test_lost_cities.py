"""Hidden-information boundaries, seeded chance, registry and administrative traces."""
import copy
import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from meeple_bots import LostCities, LostCitiesAction, LostCitiesObservation, LostCitiesState, Match, Batch, RandomAgent, MctsAgent, _native
from meeple_bots.api import _native_agent
from meeple_bots.cli import main
from meeple_bots.game_config import create_game
from meeple_bots._capabilities import game_search_capabilities
from meeple_bots.serialization import match_result_dict
from meeple_bots.studies import StudyRunner
from meeple_bots.tournaments import TournamentConfig, TournamentAgent, TournamentTrace, run_tournament, tournament_header


class LostCitiesTests(unittest.TestCase):
    def test_registry_and_setup(self):
        game = create_game('lost_cities')
        self.assertEqual(game, LostCities())
        state = game.initial_state(42)
        self.assertEqual(tuple(map(len, state.hands)), (8, 8))
        self.assertEqual(len(state.deck), 44)
        self.assertEqual(state, game.initial_state(42))
        caps = game_search_capabilities('lost_cities')
        self.assertEqual((caps['players'], caps['stochastic'], caps['imperfect_information']), (2, True, True))
        self.assertEqual(caps['selection_policies'], [])
        with self.assertRaises(ValueError):
            create_game('lost_cities', {'rounds': 3})

    def test_observation_has_no_hidden_handle_and_roundtrips_serialization(self):
        game = LostCities()
        state = game.initial_state(42)
        o = game.observation(state, 0)
        self.assertEqual(o.hand, state.hands[0])
        self.assertEqual(o.opponent_hand_size, 8)
        self.assertEqual(o.deck_size, 44)
        self.assertEqual(set(o.to_dict()), {'observer', 'current_player', 'phase', 'dealt', 'hand', 'opponent_hand_size', 'deck_size', 'expeditions', 'discards', 'blocked_discard'})
        self.assertFalse(hasattr(o, '_position'))
        self.assertFalse(hasattr(o, 'deck'))
        self.assertFalse(hasattr(o, 'hands'))
        self.assertNotEqual(o, game.observation(state, 1))
        self.assertEqual(o, LostCitiesObservation.from_dict(json.loads(json.dumps(o.to_dict()))))
        worlds = [game.sample_determinization(o, 0, seed) for seed in range(20)]
        self.assertGreater(len({w.state.hands[1] for w in worlds}), 1)
        self.assertGreater(len({w.deck_order for w in worlds}), 1)
        self.assertTrue(all(len(w.deck_order) == 44 for w in worlds))
        expected = json.dumps(o.to_dict(), sort_keys=True)
        for world in worlds:
            world.validate()
            observed = world.observation(0)
            self.assertEqual(observed, o)
            self.assertEqual(hash(observed), hash(o))
            self.assertEqual(json.dumps(observed.to_dict(), sort_keys=True), expected)
            self.assertEqual(world.legal_actions(), game.legal_actions(state))
        self.assertEqual(worlds[0], game.sample_determinization(o, 0, 0))
        with self.assertRaises(ValueError):
            game.sample_determinization(o, 1, 0)
        with self.assertRaises(ValueError):
            game.sample_determinization(replace(o, deck_size=45), 0)

    def test_draw_observation_chance_and_pool_independence(self):
        game = LostCities()
        state = game.initial_state(19)
        card = state.hands[0][0]
        state = game.apply_action(state, LostCitiesAction('discard', card))
        self.assertEqual(state.current_player, 0)
        self.assertEqual(state.phase, 'draw')
        self.assertNotIn(LostCitiesAction('draw_discard', color=card[0]), game.legal_actions(state))
        for observer in (0, 1):
            o = game.observation(state, observer)
            self.assertEqual(len(o.hand), 7 if observer == 0 else 8)
            self.assertEqual(o.opponent_hand_size, 8 if observer == 0 else 7)
            world = game.sample_determinization(o, observer, 33)
            self.assertEqual(world.observation(observer), o)
        state = game.apply_action(state, LostCitiesAction('draw_deck'))
        self.assertEqual(state.status, 'chance')
        self.assertEqual(game.legal_actions(state), ())
        o = game.observation(state, 0)
        world = game.sample_determinization(o, 0, 77)
        self.assertEqual(world.observation(0), o)
        next_card = world.deck_order[0]
        resolved = world.resolve_pending_draws()
        self.assertIn(next_card, resolved.state.hands[0])
        self.assertEqual(resolved.deck_order, world.deck_order[1:])
        self.assertEqual(resolved.state.status, 'player')
        self.assertFalse(hasattr(world, 'sample_chance'))
        with self.assertRaises(ValueError):
            game.sample_chance(world.state)  # Inspection snapshot is not a real-game position.
        outcomes = game.chance_outcomes(state)
        self.assertAlmostEqual(sum(p for _, p in outcomes), 1.)
        for action, probability in outcomes:
            self.assertAlmostEqual(probability, state.deck.count(action.card) / len(state.deck))
        draws = {game.sample_chance(state, seed) for seed in range(20)}
        self.assertGreater(len(draws), 1)
        for draw in draws:
            after = game.apply_chance_outcome(state, draw)
            after._native_position().validate()
            self.assertEqual(len(after.deck), 43)
            self.assertEqual(after.current_player, 1)
            self.assertEqual(after.phase, 'play')
        with self.assertRaises(ValueError):
            game.apply_action(state, next(iter(draws)))

    def test_mcts_rejected_at_python_and_native_boundaries_and_study(self):
        game = LostCities()
        with self.assertRaisesRegex(ValueError, 'imperfect information'):
            Match(game=game, first=MctsAgent(), second=RandomAgent())
        with self.assertRaisesRegex(ValueError, 'imperfect information'):
            Batch(game=game, agent_a=RandomAgent(), agent_b=MctsAgent())
        # Construct native search config through another game to bypass Python compatibility.
        from meeple_bots import TicTacToe
        native_search = _native_agent(MctsAgent(iterations=4), TicTacToe())
        for first, second in [(native_search, _native.AgentConfig.random()), (_native.AgentConfig.random(), native_search)]:
            with self.assertRaisesRegex(RuntimeError, 'imperfect information'):
                _native.run_match('lost_cities', first, second, 42, 10000)
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'SO-ISMCTS tuning is not supported'):
                StudyRunner('lost_cities', output=Path(tmp))
            out = io.StringIO()
            with redirect_stderr(out), redirect_stdout(out):
                result = main(['study', '--game', 'lost_cities', '--output', tmp])
            self.assertNotEqual(result, 0)
            self.assertIn('SO-ISMCTS tuning is not supported', out.getvalue())

    def test_random_match_trace_replay_batch_and_cli(self):
        game = LostCities()
        result = Match(game=game, first=RandomAgent(), second=RandomAgent(), seed=42).run()
        self.assertEqual(result.lost_cities_state.status, 'terminal')
        self.assertEqual(result.lost_cities_state.deck, ())
        self.assertEqual(result.scores, result.lost_cities_state.scores)
        payload = json.loads(json.dumps(match_result_dict(result)))
        replay = _native.LostCitiesPosition.replay(payload['moves'], payload['chance_events'])
        self.assertEqual(LostCitiesState.from_dict(replay.snapshot()), result.lost_cities_state)
        self.assertEqual(sum(e['after_ply'] == 0 for e in payload['chance_events']), 16)
        corrupted = copy.deepcopy(payload)
        corrupted['chance_events'].pop(0)
        with self.assertRaises(ValueError):
            _native.LostCitiesPosition.replay(corrupted['moves'], corrupted['chance_events'])
        batch = Batch(game=game, agent_a=RandomAgent(), agent_b=RandomAgent(), matches=2, workers=1).run()
        self.assertEqual(batch.matches, 2)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(['match', '--game', 'lost_cities', '--first', 'random', '--second', 'random', '--seed', '42']), 0)

    def test_small_tournament_and_resume_validate_hidden_chance_trace(self):
        with TemporaryDirectory() as tmp:
            config = TournamentConfig(game=LostCities(), agents=(TournamentAgent('a', RandomAgent()), TournamentAgent('b', RandomAgent())), output=Path(tmp)/'random.jsonl', matches_per_pair=2, workers=1, pairing_mode='round_robin', seat_mode='paired', seed=42, max_plies=10000)
            summary = run_tournament(config)
            self.assertEqual(summary['matches'], 2)
            header = tournament_header(config, config.output, config.workers)
            with TournamentTrace(config.output, header, resume=True) as trace:
                self.assertEqual(len(trace.completed_match_numbers), 2)
            rows = [json.loads(line) for line in config.output.read_text().splitlines()]
            rows[1]['result']['chance_events'].pop(0)
            config.output.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
            with self.assertRaisesRegex(ValueError, 'Lost Cities'):
                with TournamentTrace(config.output, header, resume=True):
                    pass


    def test_simulation_draw_consumes_order_and_opponent_draw_stays_private(self):
        game = LostCities()
        state = game.initial_state(42)
        # Advance player 0 using real rules to obtain player 1's decision.
        state = game.apply_action(state, game.legal_actions(state)[0])
        state = game.apply_action(state, LostCitiesAction('draw_deck'))
        state = game.apply_chance_outcome(state, game.sample_chance(state, 4))
        observation = game.observation(state, 0)
        after_observations = []
        for seed in range(20):
            world = game.sample_determinization(observation, 0, seed)
            # Same public discard if that card is in the sampled hand.
            card = state.hands[1][0]
            action = LostCitiesAction('discard', card)
            if action not in world.legal_actions():
                continue
            world = world.apply_action(action)
            next_card = world.deck_order[0]
            after = world.apply_action(LostCitiesAction('draw_deck'))
            after.validate()
            self.assertEqual(after.deck_order, world.deck_order[1:])
            self.assertIn(next_card, after.state.hands[1])
            self.assertEqual(after.state.status, 'player')
            after_observations.append(after.observation(0))
        self.assertGreater(len(after_observations), 1)
        self.assertTrue(all(o == after_observations[0] for o in after_observations))
        self.assertEqual(len({json.dumps(o.to_dict(), sort_keys=True) for o in after_observations}), 1)
