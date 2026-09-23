"""Probe correctness only: never assert strategic preferences or win rates."""
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from meeple_bots import LostCities, LostCitiesAction, LostCitiesObservation, MctsAgent, SoIsmctsAgent
from meeple_bots.cli import main
from meeple_bots.probes import select_cases, run_probes
from meeple_bots.probes.core import action_dict
from meeple_bots.probes.games.lost_cities import CASES, build_state
from meeple_bots.probes.report import aggregate, render
from meeple_bots.probes.runner import observation_search

GAME = LostCities()
FULL_DECK = Counter((c, v) for c in range(5) for v in [0, 0, 0, *range(2, 11)])


def cards(state):
    result = Counter(state.deck)
    for hand in state.hands:
        result.update(hand)
    for player in state.expeditions:
        for pile in player:
            result.update(pile)
    for pile in state.discards:
        result.update(pile)
    return result


class ProbeFixtureTests(unittest.TestCase):
    def test_all_fixtures_reachable_conserving_and_candidates_legal(self):
        for case in CASES:
            with self.subTest(case=case.id):
                position = case.builder()
                state, fixture = build_state(case.id.split('.', 1)[1])
                replay = GAME.initial_state(fixture['initial_seed'])
                for transition in fixture['transitions']:
                    action = LostCitiesAction.from_dict(transition['action'])
                    if transition['chance']:
                        self.assertIn(action, [a for a, _ in GAME.chance_outcomes(replay)])
                        replay = GAME.apply_chance_outcome(replay, action)
                    else:
                        self.assertIn(action, GAME.legal_actions(replay))
                        replay = GAME.apply_action(replay, action)
                self.assertEqual(replay, state)
                self.assertEqual(cards(state), FULL_DECK)
                self.assertEqual(position.observation, GAME.observation(state, position.root_player))
                self.assertIsInstance(position.observation, LostCitiesObservation)
                self.assertFalse(hasattr(position.observation, '_position'))
                self.assertFalse(hasattr(position.observation, 'hands'))
                self.assertTrue(all(a in position.legal_actions for a in case.candidate_actions))
                self.assertEqual(len(state.hands[0]), 7 if state.phase == 'draw' else 8)
                self.assertEqual(position, case.builder())

    def test_authoritative_hidden_worlds_have_equal_observations(self):
        state, _ = build_state('preserve_low_sequence_early')
        # Complete our turn, then branch only on the opponent's hidden deck draw.
        for player in (0, 1):
            action = next(a for a in GAME.legal_actions(state) if a.kind == 'discard')
            state = GAME.apply_action(state, action)
            state = GAME.apply_action(state, LostCitiesAction('draw_deck'))
            outcomes = GAME.chance_outcomes(state)
            if player == 0:
                state = GAME.apply_chance_outcome(state, outcomes[0][0])
        a = GAME.apply_chance_outcome(state, outcomes[0][0])
        b = GAME.apply_chance_outcome(state, outcomes[-1][0])
        self.assertNotEqual(a.hands[1], b.hands[1])
        self.assertNotEqual(a.deck, b.deck)
        self.assertEqual(GAME.observation(a, 0), GAME.observation(b, 0))
        self.assertEqual(cards(a), FULL_DECK)
        self.assertEqual(cards(b), FULL_DECK)

    def test_determinizations_preserve_information_and_cards(self):
        for case in CASES:
            p = case.builder()
            observation = p.observation
            worlds = []
            for seed in (0, 1, 2):
                world = GAME.sample_determinization(observation, p.root_player, seed)
                world.validate()
                self.assertEqual(world.observation(p.root_player), observation)
                self.assertEqual(world.state.hands[p.root_player], observation.hand)
                self.assertEqual(world.state.expeditions, observation.expeditions)
                self.assertEqual(world.state.discards, observation.discards)
                self.assertEqual(world.state.phase, observation.phase)
                self.assertEqual(world.state.current_player, observation.current_player)
                self.assertEqual(len(world.state.hands[1-p.root_player]), observation.opponent_hand_size)
                self.assertEqual(len(world.deck_order), observation.deck_size)
                self.assertEqual(cards(world.state), FULL_DECK)
                self.assertEqual(Counter(world.deck_order), Counter(world.state.deck))
                worlds.append(world)
            self.assertNotEqual(worlds[0].deck_order, worlds[1].deck_order)

    def test_fixture_contrasts_have_intended_public_features(self):
        early = CASES[0].builder().observation
        late = CASES[1].builder().observation
        self.assertEqual(early.deck_size, 44)
        self.assertEqual(late.deck_size, 4)
        for obs in (early, late):
            self.assertEqual(sorted(v for c, v in obs.hand if c == 0), [4, 6, 8])
        self.assertEqual(CASES[-1].builder().observation.phase, 'draw')
        self.assertEqual(CASES[7].builder().observation.expeditions[1][0][-1], (0, 3))
        self.assertEqual(CASES[8].builder().observation.expeditions[1][0][-1], (0, 10))

    def test_hidden_deck_order_is_not_observable(self):
        # Authoritative gameplay uses a deck multiset; simulation worlds carry
        # the hidden order. These seeds keep the same hands but permute that order.
        p = CASES[1].builder()
        a, b = [GAME.sample_determinization(p.observation, 0, seed) for seed in (21, 34)]
        self.assertEqual(a.state.hands, b.state.hands)
        self.assertEqual(Counter(a.deck_order), Counter(b.deck_order))
        self.assertNotEqual(a.deck_order, b.deck_order)
        self.assertEqual(a.observation(0), b.observation(0))


class ProbeFrameworkTests(unittest.TestCase):
    def run_case(self, output, **kwargs):
        return run_probes((CASES[0],), SoIsmctsAgent(iterations=2), output=output,
                          progress=lambda _: None, **kwargs)

    def test_registration_and_filters(self):
        self.assertEqual(len(select_cases(game='lost_cities')), 10)
        self.assertEqual(len(select_cases(suite='ordering')), 3)
        self.assertEqual(select_cases(probe='preserve-low-sequence-early'), (CASES[0],))
        self.assertEqual(select_cases(probe=CASES[0].id), (CASES[0],))
        for kwargs in ({'game': 'spotf'}, {'suite': 'missing'}, {'probe': 'missing'}, {'cases': (CASES[0], CASES[0])}):
            with self.assertRaises(ValueError):
                select_cases(**kwargs)

    def test_observation_only_input_all_actions_and_independent_runs(self):
        calls = []
        real = SoIsmctsAgent.search
        def spy(agent, observation, legal, *, seed):
            self.assertIsInstance(observation, LostCitiesObservation)
            self.assertEqual(legal, CASES[0].builder().legal_actions)
            calls.append((agent.iterations, seed, observation))
            return real(agent, observation, legal, seed=seed)
        with TemporaryDirectory() as tmp, patch.object(SoIsmctsAgent, 'search', spy):
            output = Path(tmp)/'capture'
            self.run_case(output, iterations=(4, 8), seeds=3, seed=9)
            rows = [json.loads(line) for line in (output/'runs.jsonl').read_text().splitlines()]
            self.assertEqual([(b, s) for b, s, _ in calls], [(b, s) for b in (4, 8) for s in (9, 10, 11)])
            self.assertTrue(all(o == calls[0][2] for _, _, o in calls))
            p = CASES[0].builder()
            for row in rows:
                self.assertEqual(len(row['root_actions']), len(p.legal_actions))
                self.assertEqual(sum(e['focused'] for e in row['root_actions']), 3)
                self.assertEqual(row['root_visits'], row['iterations'])
                self.assertEqual(row['diagnostics']['determinizations_sampled'], row['iterations'])
                self.assertEqual(sum(e['visits'] for e in row['root_actions']), row['iterations'])
                for e in row['root_actions']:
                    self.assertEqual(e['availability'], row['iterations'])
                    self.assertLessEqual(e['visits'], e['availability'])
                    if e['q'] is not None:
                        self.assertTrue(-1 <= e['q'] <= 1)
                expected = real(SoIsmctsAgent(iterations=row['iterations']), p.observation, p.legal_actions, seed=row['search_seed'])
                self.assertEqual(row['selected_action'], expected['action'].to_dict())
            self.assertEqual(len(rows), 6)

    def test_reproducibility_and_focus_does_not_change_search(self):
        with TemporaryDirectory() as tmp:
            def capture(name, case):
                out = Path(tmp)/name
                run_probes((case,), SoIsmctsAgent(iterations=4, tree_reuse=True), iterations=(8,),
                           seeds=2, output=out, progress=lambda _: None)
                rows = [json.loads(line) for line in (out/'runs.jsonl').read_text().splitlines()]
                for row in rows:
                    row.pop('elapsed_seconds')
                    for edge in row['root_actions']:
                        edge.pop('focused')
                return rows
            first = capture('a', CASES[0])
            self.assertEqual(first, capture('b', CASES[0]))
            self.assertEqual(first, capture('unfocused', replace(CASES[0], candidate_actions=())))

    def test_aggregation_missing_availability_and_unvisited_q(self):
        # Adapter contract also handles perfect-information diagnostics, without AMAF availability.
        def search(agent, observation, legal, *, seed):
            return {'action': legal[seed % 2], 'root_visits': agent.iterations,
                    'root_actions': [{'action': a, 'visits': agent.iterations if i == seed % 2 else 0,
                                     'q': .5 if seed == 0 else -.25} for i, a in enumerate(legal)]}
        with TemporaryDirectory() as tmp:
            summary = run_probes((replace(CASES[0], game='adapter_test', candidate_actions=()),),
                                 MctsAgent(), iterations=(8,), seeds=2, output=Path(tmp)/'run',
                                 search=search, progress=lambda _: None)
            a, b, *rest = summary[0]['actions']
            self.assertEqual(a['selected_count'], 1)
            self.assertEqual(a['selected_percentage'], 50)
            self.assertEqual(a['median_visits'], 4)
            self.assertEqual(a['median_q'], .5)
            self.assertEqual(b['median_q'], -.25)
            self.assertTrue(all(e['median_availability'] is None for e in summary[0]['actions']))
            self.assertTrue(all(e['median_q'] is None for e in rest))
            text = render(summary)
            self.assertIn('Dominant selection (tie)', text)
            self.assertIn('root-player', text)
            self.assertIn('—', text)

    def test_persistence_reports_progress_and_path_protection(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)/'run'
            progress = []
            summaries = run_probes((CASES[0],), SoIsmctsAgent(), iterations=(4, 8), seeds=2,
                                   output=out, progress=progress.append)
            metadata = json.loads((out/'metadata.json').read_text())
            self.assertEqual(metadata['search_seeds'], [0, 1])
            self.assertEqual(metadata['q_orientation'], 'root_player')
            self.assertEqual(len(metadata['native_sha256']), 64)
            self.assertEqual(metadata['probes'][0]['id'], CASES[0].id)
            self.assertEqual(json.loads((out/'summary.json').read_text()), summaries)
            rows = [json.loads(line) for line in (out/'runs.jsonl').read_text().splitlines()]
            self.assertEqual(aggregate(rows), summaries)
            report = (out/'report.txt').read_text()
            self.assertIn('Selection share across budgets', report)
            self.assertIn('Median avail', report)
            self.assertIn('2/2 complete', '\n'.join(progress))
            before = {p.name: p.read_bytes() for p in out.iterdir()}
            with self.assertRaises(FileExistsError):
                self.run_case(out, seeds=1)
            self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir()})

    def test_invalid_inputs_do_not_create_output(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)/'run'
            for kwargs in ({'iterations': ()}, {'iterations': (0,)}, {'iterations': (4, 4)},
                           {'seeds': 0}, {'seed': -1}, {'seed': 2**64-1, 'seeds': 2}):
                with self.assertRaises(ValueError):
                    self.run_case(out, **kwargs)
                self.assertFalse(out.exists())

    def test_cli_listing_single_probe_and_config(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp)/'run'
            profile = Path(tmp)/'agent.toml'
            profile.write_text('agent = "so_ismcts"\niterations = 7\nselection_policy = "ucb1_tuned"\n')
            with redirect_stdout(io.StringIO()) as listing:
                self.assertEqual(main(['probe', '--game', 'lost_cities', '--suite', 'ordering', '--list']), 0)
            self.assertIn('preserve_low_sequence_early', listing.getvalue())
            with redirect_stdout(io.StringIO()) as human, redirect_stderr(io.StringIO()):
                self.assertEqual(main(['probe', '--game', 'lost_cities', '--agent', 'so_ismcts',
                    '--agent-config', str(profile), '--probe', 'discard-pile-vs-deck',
                    '--iterations', '4,8', '--seeds', '2', '--output', str(output)]), 0)
            self.assertIn('Draw deck', human.getvalue())
            rows = [json.loads(line) for line in (output/'runs.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['agent']['selection_policy'] == 'ucb1_tuned' for r in rows))
            self.assertEqual(len(rows), 4)

    def test_all_cases_native_smoke(self):
        with TemporaryDirectory() as tmp:
            summary = run_probes(CASES, SoIsmctsAgent(), iterations=(2,), seeds=1,
                                 output=Path(tmp)/'run', progress=lambda _: None)
            self.assertEqual(len(summary), 10)
            self.assertTrue(all(sum(a['selected_count'] for a in s['actions']) == 1 for s in summary))
