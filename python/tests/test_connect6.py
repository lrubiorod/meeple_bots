"""Connect6 rules stay native; configurations survive every orchestration path."""
import io
import json
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from meeple_bots import Connect6, Connect6Action, Match, MctsAgent, RandomAgent, create_game
from meeple_bots.api import _analyze_trace
from meeple_bots.cli import main, build_parser
from meeple_bots.extraction import extract_tournament
from meeple_bots.studies import StudyRunner, generic_baseline
from meeple_bots.tournaments import TournamentAgent, TournamentConfig, run_tournament


class Connect6Tests(unittest.TestCase):
    def test_configuration_and_turns(self):
        self.assertEqual(Connect6().board_size, 19)
        for n in (9, 11):
            game = create_game('connect6', {'board_size': n})
            self.assertEqual(pickle.loads(pickle.dumps(game)), game)
            state = game.initial_state()
            self.assertEqual(len(state.legal_actions()), n*n)
            for cell, player, remaining in ((0, 1, 2), (1, 1, 1), (2, 0, 2)):
                state = state.apply_action(Connect6Action(cell))
                self.assertEqual((state.current_player, state.placements_remaining), (player, remaining))
                self.assertEqual(len(state.legal_actions()), n*n-cell-1)
            with self.assertRaises(ValueError): state.apply_action(Connect6Action(0))
        for params in ({'board_size':5}, {'board_size': True}, {'board_size':9.5}, {'oops':9}):
            with self.assertRaises((ValueError, TypeError)): create_game('connect6', params)
        for name in ('boop','connect-four','tic-tac-toe','spotf','splendor'):
            self.assertEqual(create_game(name), create_game(name, {}))
            with self.assertRaises(ValueError): create_game(name, {'board_size':9})

    def test_commutative_turn(self):
        state = Connect6(9).initial_state().apply_action(Connect6Action(0))
        a, b = Connect6Action(1), Connect6Action(2)
        self.assertEqual(state.apply_action(a).apply_action(b), state.apply_action(b).apply_action(a))

    def test_mcts_all_existing_mechanisms_and_native_replay(self):
        game = Connect6(6)
        for selector in ('uct', 'ucb1_tuned'):
            for reuse in (False, True):
                for transpositions in (False, True):
                    agent = MctsAgent(iterations=8, rollout_depth=36, selection_policy=selector,
                                      tree_reuse=reuse, transpositions=transpositions)
                    first = Match(game, agent, RandomAgent(), seed=42).run()
                    second = Match(game, agent, RandomAgent(), seed=42).run()
                    self.assertEqual([m.action for m in first.moves], [m.action for m in second.moves])
                    self.assertEqual(first.game_params, {'board_size':6})
                    state = game.initial_state()
                    for move in first.moves:
                        self.assertEqual(state.current_player, move.player)
                        state = state.apply_action(move.action)
                    self.assertTrue(state.terminal)
                    self.assertEqual(first.winner, state.winner)
                    _analyze_trace(game, first.moves, seed=42)

    def test_cli_default_custom_and_analyze(self):
        args = build_parser().parse_args(['match','--game','connect6'])
        self.assertEqual(create_game(args.game, args.game_params).board_size,19)
        for command in ('match', 'analyze'):
            options = ['--first','random','--second','random'] if command == 'match' else ['--samples','2','--max-depth','8','--target-time','0.001']
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(io.StringIO()):
                self.assertEqual(main([command,'--game','connect6','--game-param','board_size=9','--json',*options]), 0)
            data = json.loads(output.getvalue())
            self.assertEqual(data['game_params'], {'board_size':9})
            if command == 'analyze': self.assertEqual(data['initial_legal_actions'],81)

    def test_worker_tournament_extract_and_study_resume_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = TournamentConfig(Connect6(6), root/'games.jsonl', 'round_robin', 'paired', 2, 42, 36, 2,
                                      (TournamentAgent('a',RandomAgent()),TournamentAgent('b',RandomAgent())))
            run_tournament(config)
            rows = [json.loads(line) for line in config.output.read_text().splitlines()]
            self.assertEqual(rows[0]['game_params'], {'board_size':6})
            for row in rows[1:]: self.assertEqual(row['result']['game_params'], {'board_size':6})
            summary = extract_tournament(config.output, root/'data')
            self.assertTrue(summary['complete'])
            toml = root/'tournament.toml'
            toml.write_text('game = "connect6"\ngame_params = {board_size = 6}\n'
                            'output = "cli.jsonl"\nmatches_per_pair = 2\nworkers = 2\n'
                            '[[agents]]\nname = "a"\nkind = "random"\n'
                            '[[agents]]\nname = "b"\nkind = "random"\n')
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(['tournament', '--config', str(toml)]), 0)
            self.assertEqual(json.loads((root/'cli.jsonl').read_text().splitlines()[0])['game_params'], {'board_size':6})
            base = generic_baseline('connect6')
            options = dict(output=root/'study', budget=1, game_params={'board_size':9}, progress=lambda _:None)
            runner = StudyRunner('connect6', base, **options)
            self.assertEqual(runner.game.board_size,9)
            self.assertEqual(runner.state['request']['game_params'], {'board_size':9})
            resumed = StudyRunner('connect6', base, resume=True, **options)
            self.assertEqual(resumed.game.board_size,9)
            with self.assertRaises(ValueError):
                StudyRunner('connect6', base, resume=True, **{**options,'game_params':{'board_size':11}})


if __name__ == '__main__': unittest.main()
