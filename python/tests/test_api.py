import csv
import io
import importlib.util
import json
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from meeple_bots import (
    Batch,
    BatchProgressStatus,
    Boop,
    BoopAction,
    BoopPiece,
    BoopPieceKind,
    ConnectFour,
    ConnectFourAction,
    ConditionalRollout,
    EpsilonGreedy,
    GameHeuristic,
    Greedy,
    HumanAgent,
    HumanMoveObservation,
    Match,
    MatchMoveObservation,
    MctsAgent,
    NeutralEvaluator,
    ProgressiveBias,
    RandomAgent,
    SpiritTile,
    SpiritsOfTheForest,
    TakeSpiritTile,
    TicTacToe,
    TicTacToeAction,
    TurnPhaseIs,
    UniformRandom,
    benchmark_mcts_agent,
    evaluate_game,
)
from meeple_bots.cli import (
    _batch_agent_dict,
    _load_tournament_config,
    _parse_inline_mcts_profile,
    _serialized_rollout_policy_description,
    _tournament_pairings,
    build_parser,
    main,
)
from meeple_bots.extraction import _agent_row
from meeple_bots._concurrency import ordered_parallel_map, resolve_workers
from meeple_bots.games.boop.gui import BoopGui
from meeple_bots.games.connect_four.gui import ConnectFourGui
from meeple_bots.games.spirits_of_the_forest.gui import SpiritsOfTheForestGui
from meeple_bots.games.tic_tac_toe.gui import GuiPlayer, TicTacToeGui
from meeple_bots.gui.server import run_gui
from meeple_bots.reporting import wilson_interval


REPORT_DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(module) is not None
    for module in ("matplotlib", "pandas", "seaborn")
)


class MatchApiTests(unittest.TestCase):
    @staticmethod
    def wait_for_gui(gui, predicate, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = gui.snapshot()
            if predicate(state):
                return state
            time.sleep(0.005)
        raise AssertionError(f"GUI state did not arrive: {gui.snapshot()}")

    def test_wilson_interval_handles_known_and_invalid_counts(self) -> None:
        low, high = wilson_interval(5, 10)

        self.assertAlmostEqual(low, 0.236593, places=6)
        self.assertAlmostEqual(high, 0.763407, places=6)
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))
        with self.assertRaises(ValueError):
            wilson_interval(2, 1)

    def test_batch_alternates_sides_and_aggregates_results(self) -> None:
        events = []
        callback_threads = []

        def record_progress(event) -> None:
            events.append(event)
            callback_threads.append(threading.get_ident())

        result = Batch(
            game=TicTacToe(),
            agent_a=RandomAgent(),
            agent_b=RandomAgent(),
            matches=4,
            seed=42,
            workers=2,
        ).run(record_progress)

        self.assertEqual(result.matches, 4)
        self.assertEqual(result.workers, 2)
        self.assertEqual(
            result.agent_a_wins + result.agent_b_wins + result.draws,
            result.matches,
        )
        self.assertEqual([game.seed for game in result.games], [42, 43, 44, 45])
        self.assertEqual(
            [game.agent_a_player for game in result.games],
            [0, 1, 0, 1],
        )
        self.assertEqual(len(events), 8)
        started = [event for event in events if event.status is BatchProgressStatus.STARTED]
        completed = [
            event for event in events if event.status is BatchProgressStatus.COMPLETED
        ]
        self.assertEqual([event.match_number for event in started], [1, 2, 3, 4])
        self.assertEqual([event.match_number for event in completed], [1, 2, 3, 4])
        self.assertTrue(all(event.result is None for event in started))
        self.assertTrue(all(event.match_result is None for event in started))
        self.assertTrue(all(event.match_result is not None for event in completed))
        self.assertTrue(
            all(
                event.match_result is not None
                and len(event.match_result.moves) == event.match_result.plies
                for event in completed
            )
        )
        self.assertEqual(
            [event.result for event in completed],
            list(result.games),
        )
        self.assertEqual(set(callback_threads), {threading.get_ident()})

    def test_batch_results_are_reproducible_except_for_timing(self) -> None:
        common = {
            "game": TicTacToe(),
            "agent_a": RandomAgent(),
            "agent_b": MctsAgent(iterations=4, rollout_depth=9),
            "matches": 4,
            "seed": 7,
        }
        serial = Batch(
            **common,
            workers=1,
        ).run()
        parallel = Batch(
            **common,
            workers=3,
        ).run()
        serial_games = [
            (game.seed, game.agent_a_player, game.winner, game.plies, game.utilities)
            for game in serial.games
        ]
        parallel_games = [
            (game.seed, game.agent_a_player, game.winner, game.plies, game.utilities)
            for game in parallel.games
        ]

        self.assertEqual(serial.workers, 1)
        self.assertEqual(parallel.workers, 3)
        self.assertEqual(serial_games, parallel_games)

    def test_auto_workers_leave_one_physical_core_free(self) -> None:
        with patch(
            "meeple_bots._concurrency._physical_core_count",
            return_value=8,
        ):
            self.assertEqual(resolve_workers("auto"), 7)

    def test_worker_pool_runs_jobs_on_distinct_threads_and_preserves_order(self) -> None:
        barrier = threading.Barrier(2)
        worker_threads = set()
        lock = threading.Lock()

        def run_job(value: int) -> int:
            with lock:
                worker_threads.add(threading.get_ident())
            barrier.wait(timeout=2)
            return value * 2

        results = list(ordered_parallel_map(run_job, [1, 2], workers=2))

        self.assertEqual(results, [(1, 2), (2, 4)])
        self.assertEqual(len(worker_threads), 2)

    def test_batch_rejects_invalid_workers(self) -> None:
        with self.assertRaisesRegex(ValueError, "workers"):
            Batch(workers=0)
        with self.assertRaisesRegex(TypeError, "workers"):
            Batch(workers=True)
        with self.assertRaisesRegex(TypeError, "workers"):
            Batch(workers="all")

    def test_batch_results_are_reproducible_with_the_same_parallelism(self) -> None:
        batch = Batch(
            game=TicTacToe(),
            agent_a=RandomAgent(),
            agent_b=MctsAgent(iterations=4, rollout_depth=9),
            matches=4,
            seed=7,
            workers=2,
        )

        first = batch.run()
        repeated = batch.run()
        first_games = [
            (game.seed, game.agent_a_player, game.winner, game.plies, game.utilities)
            for game in first.games
        ]
        repeated_games = [
            (game.seed, game.agent_a_player, game.winner, game.plies, game.utilities)
            for game in repeated.games
        ]

        self.assertEqual(first_games, repeated_games)

    def test_same_seed_produces_same_report(self) -> None:
        match = Match(
            game=TicTacToe(),
            first=MctsAgent(iterations=100),
            second=RandomAgent(),
            seed=42,
        )

        self.assertEqual(match.run(), match.run())

    def test_report_contains_every_move(self) -> None:
        result = Match(
            game=TicTacToe(),
            first=RandomAgent(),
            second=RandomAgent(),
            seed=7,
        ).run()

        self.assertEqual(len(result.moves), result.plies)
        for ply, move in enumerate(result.moves):
            self.assertEqual(move.player, ply % 2)
            self.assertIn(move.action.row, range(3))
            self.assertIn(move.action.column, range(3))
        if result.winner is not None:
            other = 1 - result.winner
            self.assertGreater(result.utilities[result.winner], result.utilities[other])

    def test_match_observer_receives_every_tic_tac_toe_move(self) -> None:
        observations = []
        result = Match(
            game=TicTacToe(),
            first=RandomAgent(),
            second=RandomAgent(),
            seed=7,
            observe_move=observations.append,
        ).run()

        self.assertEqual(len(observations), result.plies)
        self.assertTrue(all(isinstance(item, MatchMoveObservation) for item in observations))
        self.assertTrue(all(item.decision_seconds >= 0 for item in observations))
        self.assertEqual(observations[-1].board, result.final_board)
        self.assertEqual(
            [item.action for item in observations],
            [move.action for move in result.moves],
        )

    def test_match_observer_receives_every_connect_four_move(self) -> None:
        observations = []
        result = Match(
            game=ConnectFour(),
            first=RandomAgent(),
            second=RandomAgent(),
            seed=11,
            observe_move=observations.append,
        ).run()

        self.assertEqual(len(observations), result.plies)
        self.assertTrue(
            all(isinstance(item.action, ConnectFourAction) for item in observations)
        )
        self.assertTrue(all(item.decision_seconds >= 0 for item in observations))
        self.assertEqual(observations[-1].board, result.final_board)
        self.assertEqual(
            [item.action for item in observations],
            [move.action for move in result.moves],
        )

    def test_match_observer_receives_every_boop_move_and_pool(self) -> None:
        observations = []
        result = Match(
            game=Boop(),
            first=RandomAgent(),
            second=RandomAgent(),
            seed=9,
            observe_move=observations.append,
        ).run()

        self.assertEqual(len(observations), result.plies)
        self.assertTrue(all(isinstance(item.action, BoopAction) for item in observations))
        self.assertTrue(all(item.pools is not None for item in observations))
        self.assertEqual(observations[-1].board, result.final_board)
        self.assertEqual(observations[-1].pools, result.pools)

    def test_tic_tac_toe_gui_runs_agents_and_accepts_human_moves(self) -> None:
        watched = TicTacToeGui()
        watched.start(
            GuiPlayer("random"),
            GuiPlayer("random"),
            seed=4,
            minimum_move_seconds=0,
        )
        finished = self.wait_for_gui(watched, lambda state: state["status"] == "finished")
        self.assertGreaterEqual(len(finished["moves"]), 5)
        self.assertEqual(
            sum(cell is not None for cell in finished["board"]),
            len(finished["moves"]),
        )

        played = TicTacToeGui()
        played.start(
            GuiPlayer("human"),
            GuiPlayer("human"),
            minimum_move_seconds=0,
        )
        waiting = self.wait_for_gui(
            played,
            lambda state: state["status"] == "waiting_human",
        )
        self.assertEqual(waiting["active_player"], 0)
        played.submit_move(1, 1)
        waiting = self.wait_for_gui(
            played,
            lambda state: state["status"] == "waiting_human"
            and state["active_player"] == 1,
        )
        self.assertEqual(waiting["board"][4], 0)
        played.cancel()

    def test_connect_four_gui_runs_agents_and_accepts_columns(self) -> None:
        watched = ConnectFourGui()
        watched.start(
            GuiPlayer("random"),
            GuiPlayer("random"),
            seed=6,
            minimum_move_seconds=0,
        )
        finished = self.wait_for_gui(watched, lambda state: state["status"] == "finished")
        self.assertGreaterEqual(len(finished["moves"]), 7)
        self.assertEqual(sum(cell is not None for cell in finished["board"]), len(finished["moves"]))

        played = ConnectFourGui()
        played.start(
            GuiPlayer("human"),
            GuiPlayer("human"),
            minimum_move_seconds=0,
        )
        waiting = self.wait_for_gui(
            played,
            lambda state: state["status"] == "waiting_human",
        )
        self.assertEqual(waiting["legal_actions"], list(range(7)))
        played.submit_move(3)
        waiting = self.wait_for_gui(
            played,
            lambda state: state["status"] == "waiting_human"
            and state["active_player"] == 1,
        )
        self.assertEqual(waiting["board"][5 * 7 + 3], 0)
        self.assertEqual(waiting["last_move"], [5, 3])
        played.cancel()

    def test_boop_gui_runs_agents_and_accepts_action_indices(self) -> None:
        watched = BoopGui()
        watched.start(
            GuiPlayer("random"),
            GuiPlayer("random"),
            seed=9,
            minimum_move_seconds=0,
        )
        finished = self.wait_for_gui(
            watched,
            lambda state: state["status"] == "finished",
            timeout=5,
        )
        self.assertGreater(len(finished["moves"]), 0)
        self.assertEqual(finished["moves"][-1]["board"], finished["board"])
        self.assertEqual(finished["moves"][-1]["pools"], finished["pools"])
        pieces_in_pools = sum(
            pool["kittens"] + pool["cats"] for pool in finished["pools"]
        )
        pieces_on_board = sum(cell is not None for cell in finished["board"])
        self.assertEqual(pieces_in_pools, 16 - pieces_on_board)

        played = BoopGui()
        played.start(
            GuiPlayer("human"),
            GuiPlayer("human"),
            minimum_move_seconds=0,
        )
        waiting = self.wait_for_gui(
            played,
            lambda state: state["status"] == "waiting_human",
        )
        self.assertEqual(len(waiting["legal_actions"]), 36)
        played.submit_move(waiting["legal_actions"][0]["index"])
        waiting = self.wait_for_gui(
            played,
            lambda state: state["status"] == "waiting_human"
            and state["active_player"] == 1,
        )
        self.assertEqual(waiting["pools"][0]["kittens"], 7)
        self.assertIsNotNone(waiting["board"][0])
        played.cancel()

    def test_gui_dispatches_connect_four_without_changing_the_server(self) -> None:
        with patch("meeple_bots.gui.server.serve_gui") as serve:
            run_gui(game="connect-four", open_browser=False)

        application, page = serve.call_args.args
        self.assertEqual(application.snapshot()["game"], "connect-four")
        self.assertIn("Connect Four", page)

    def test_gui_dispatches_boop_without_changing_the_server(self) -> None:
        with patch("meeple_bots.gui.server.serve_gui") as serve:
            run_gui(game="boop", open_browser=False)

        application, page = serve.call_args.args
        self.assertEqual(application.snapshot()["game"], "boop")
        self.assertIn("Meeple Bots · Boop", page)

    def test_invalid_mcts_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MctsAgent(iterations=0)
        with self.assertRaises(TypeError):
            MctsAgent(heuristic=True)
        with self.assertRaises(ValueError):
            MctsAgent(heuristic=-1)
        with self.assertRaises(ValueError):
            EpsilonGreedy(epsilon=1.1, evaluator=GameHeuristic(0))
        with self.assertRaises(TypeError):
            MctsAgent(rollout_policy="uniform_random")
        with self.assertRaises(ValueError):
            MctsAgent(iterations=10, time_budget=0.1)
        with self.assertRaises(ValueError):
            MctsAgent(time_budget=0)
        with self.assertRaises(TypeError):
            HumanAgent(observe_action="not callable")

    def test_epsilon_greedy_rollout_does_not_require_a_cutoff_heuristic(self) -> None:
        result = Match(
            game=Boop(),
            first=MctsAgent(
                iterations=2,
                rollout_depth=2,
                cutoff_evaluator=NeutralEvaluator(),
                rollout_policy=EpsilonGreedy(0.1, GameHeuristic(0)),
            ),
            second=RandomAgent(),
            seed=19,
        ).run()

        self.assertGreater(result.plies, 0)

    def test_uniform_random_is_the_default_rollout_policy(self) -> None:
        self.assertIsInstance(MctsAgent().rollout_policy, UniformRandom)

    def test_epsilon_greedy_rollout_runs_with_a_supported_heuristic(self) -> None:
        result = Match(
            game=Boop(),
            first=MctsAgent(
                iterations=2,
                rollout_depth=2,
                cutoff_evaluator=NeutralEvaluator(),
                rollout_policy=EpsilonGreedy(0.1, GameHeuristic(0)),
            ),
            second=RandomAgent(),
            seed=19,
        ).run()

        self.assertGreater(result.plies, 0)

    def test_conditional_rollout_runs_on_spotf_and_is_rejected_elsewhere(self) -> None:
        policy = ConditionalRollout(
            condition=TurnPhaseIs("collect"),
            primary=EpsilonGreedy(0.25, GameHeuristic(2)),
            fallback=UniformRandom(),
        )
        result = Match(
            game=SpiritsOfTheForest(),
            first=MctsAgent(iterations=2, rollout_depth=2, rollout_policy=policy),
            second=RandomAgent(),
            seed=19,
        ).run()

        self.assertGreater(result.plies, 0)
        with self.assertRaisesRegex(ValueError, "only supported by spotf"):
            Match(game=Boop(), first=MctsAgent(rollout_policy=policy))

    def test_conditional_rollout_is_preserved_in_extracted_agent_metadata(self) -> None:
        policy = ConditionalRollout(
            TurnPhaseIs("collect"),
            EpsilonGreedy(0.25, GameHeuristic(2)),
            UniformRandom(),
        )

        row = _agent_row(
            _batch_agent_dict(
                "collect-h2",
                MctsAgent(iterations=5, rollout_depth=2, rollout_policy=policy),
            )
        )

        self.assertEqual(row["rollout_policy"], "conditional")
        self.assertEqual(row["rollout_condition"], "turn_phase")
        self.assertEqual(row["rollout_condition_phase"], "collect")
        self.assertEqual(row["rollout_primary_policy"], "epsilon_greedy")
        self.assertEqual(row["rollout_epsilon"], 0.25)
        self.assertEqual(row["rollout_fallback_policy"], "uniform_random")
        description = _serialized_rollout_policy_description(
            _batch_agent_dict(
                "collect-h2",
                MctsAgent(iterations=5, rollout_depth=2, rollout_policy=policy),
            )
        )
        self.assertEqual(
            description,
            "conditional(collect ? epsilon_greedy(epsilon=0.25), "
            "evaluator=game_heuristic(2) : uniform_random)",
        )

    def test_progressive_bias_is_preserved_in_extracted_agent_metadata(self) -> None:
        agent = MctsAgent(
            iterations=5,
            rollout_depth=2,
            progressive_bias=ProgressiveBias(
                0.25,
                GameHeuristic(2),
                TurnPhaseIs("collect"),
            ),
            root_diagnostics=True,
        )

        row = _agent_row(_batch_agent_dict("progressive", agent))

        self.assertEqual(row["progressive_bias_weight"], 0.25)
        self.assertEqual(row["progressive_bias_evaluator"], "game_heuristic")
        self.assertEqual(row["progressive_bias_heuristic"], 2)
        self.assertEqual(row["progressive_bias_condition"], "turn_phase")
        self.assertEqual(row["progressive_bias_condition_phase"], "collect")
        self.assertTrue(row["root_diagnostics"])

    def test_conditional_rollout_can_be_benchmarked_by_analyze(self) -> None:
        agent = MctsAgent(
            iterations=1,
            rollout_depth=2,
            rollout_policy=ConditionalRollout(
                TurnPhaseIs("collect"),
                EpsilonGreedy(0.25, GameHeuristic(2)),
                UniformRandom(),
            ),
        )

        benchmark = benchmark_mcts_agent(
            SpiritsOfTheForest(), agent, median_depth=3, seed=7
        )

        self.assertEqual(benchmark.agent, agent)
        self.assertGreater(benchmark.milliseconds_per_iteration, 0.0)

    def test_conditional_progressive_bias_records_root_diagnostics(self) -> None:
        bias = ProgressiveBias(0.25, GameHeuristic(2), TurnPhaseIs("collect"))
        result = Match(
            game=SpiritsOfTheForest(),
            first=MctsAgent(
                iterations=16,
                rollout_depth=2,
                cutoff_evaluator=GameHeuristic(2),
                rollout_policy=UniformRandom(),
                progressive_bias=bias,
                root_diagnostics=True,
            ),
            second=RandomAgent(),
            seed=19,
        ).run()

        first_search = next(move for move in result.moves if move.player == 0)
        self.assertTrue(first_search.root_actions)
        self.assertEqual(
            sum(root.selected for root in first_search.root_actions),
            1,
        )
        self.assertTrue(
            all(root.heuristic_value is not None for root in first_search.root_actions)
        )
        with self.assertRaisesRegex(ValueError, "only supported by spotf"):
            Match(
                game=Boop(),
                first=MctsAgent(progressive_bias=bias),
            )

    def test_boop_match_accepts_both_heuristics(self) -> None:
        for heuristic in (0, 1):
            result = Match(
                game=Boop(),
                first=MctsAgent(iterations=4, rollout_depth=1, heuristic=heuristic),
                second=RandomAgent(),
                seed=19,
            ).run()

            self.assertGreater(result.plies, 0)
            self.assertEqual(len(result.moves), result.plies)

        with self.assertRaisesRegex(ValueError, "available indices: 0..1"):
            Match(game=Boop(), first=MctsAgent(heuristic=2))

    def test_games_without_heuristics_reject_an_index(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "tic-tac-toe does not provide MCTS heuristics",
        ):
            Match(game=TicTacToe(), first=MctsAgent(heuristic=0))

        with self.assertRaisesRegex(
            ValueError,
            "connect-four does not provide MCTS heuristics",
        ):
            Match(
                game=ConnectFour(),
                first=MctsAgent(iterations=1, heuristic=0),
            )

    def test_game_evaluation_has_reproducible_structural_metrics(self) -> None:
        first = evaluate_game(TicTacToe(), samples=16, seed=42)
        repeated = evaluate_game(TicTacToe(), samples=16, seed=42)

        self.assertEqual(first.initial_legal_actions, 9)
        self.assertEqual(first.terminal_rate, 1.0)
        self.assertLessEqual(first.estimated_depth, 9)
        self.assertFalse(first.depth_is_lower_bound)
        self.assertEqual(
            first.effective_branching_factor,
            repeated.effective_branching_factor,
        )
        self.assertEqual(first.estimated_depth, repeated.estimated_depth)
        self.assertEqual(first.player_turn_depth_p95, repeated.player_turn_depth_p95)
        self.assertEqual(first.actions_per_player_turn_mean, 1.0)
        self.assertEqual(first.actions_per_player_turn_p95, 1)
        self.assertGreaterEqual(len(first.rollout_costs), 1)
        self.assertEqual(
            [budget.seconds for budget in first.rollout_costs[0].iteration_budgets],
            [1, 2, 5, 10, 20],
        )
        self.assertEqual(
            [experiment.label for experiment in first.suggested_experiments[:3]],
            ["Fast", "Balanced", "Wide"],
        )
        self.assertGreater(first.milliseconds_per_iteration, 0.0)
        self.assertAlmostEqual(
            first.estimated_decision_time_ms,
            first.recommended_iterations * first.milliseconds_per_iteration,
        )

    def test_game_evaluation_validates_target_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_time"):
            evaluate_game(TicTacToe(), samples=8, target_time=0)

        with self.assertRaisesRegex(TypeError, "target_time"):
            evaluate_game(TicTacToe(), samples=8, target_time=True)

    def test_game_evaluation_marks_a_truncated_depth(self) -> None:
        report = evaluate_game(TicTacToe(), samples=8, max_depth=1)

        self.assertEqual(report.terminal_rate, 0.0)
        self.assertEqual(report.estimated_depth, 1)
        self.assertTrue(report.depth_is_lower_bound)

    def test_configured_mcts_benchmark_uses_exact_agent_and_shared_positions(self) -> None:
        agent = MctsAgent(iterations=4, rollout_depth=4)

        benchmark = benchmark_mcts_agent(
            TicTacToe(),
            agent,
            median_depth=6,
            seed=42,
        )

        self.assertEqual(benchmark.agent, agent)
        self.assertEqual(benchmark.sampled_positions, 3)
        self.assertEqual(
            [timing.sampled_ply for timing in benchmark.position_timings],
            [0, 2, 4],
        )
        self.assertGreater(benchmark.decision_time_mean_ms, 0.0)
        self.assertLessEqual(
            benchmark.decision_time_p50_ms,
            benchmark.decision_time_p95_ms,
        )
        self.assertAlmostEqual(
            benchmark.milliseconds_per_iteration,
            benchmark.decision_time_mean_ms / agent.iterations,
        )
        self.assertEqual(
            [timing.iterations for timing in benchmark.position_timings],
            [4, 4, 4],
        )
        self.assertTrue(all(timing.nodes >= 2 for timing in benchmark.position_timings))

    def test_time_budget_records_actual_iterations_in_match_trace(self) -> None:
        result = Match(
            first=MctsAgent(time_budget=0.001, rollout_depth=2),
            second=RandomAgent(),
            seed=42,
        ).run()

        mcts_moves = [move for move in result.moves if move.player == 0]
        self.assertTrue(mcts_moves)
        self.assertTrue(all(move.decision_seconds > 0 for move in mcts_moves))
        self.assertTrue(all((move.search_iterations or 0) >= 1 for move in mcts_moves))
        self.assertTrue(all((move.search_nodes or 0) >= 2 for move in mcts_moves))

    def test_configured_mcts_benchmark_supports_a_game_heuristic(self) -> None:
        agent = MctsAgent(
            iterations=1,
            rollout_depth=1,
            cutoff_evaluator=NeutralEvaluator(),
            rollout_policy=EpsilonGreedy(0.2, GameHeuristic(0)),
        )
        benchmark = benchmark_mcts_agent(
            Boop(),
            agent,
            median_depth=3,
            seed=7,
        )

        self.assertEqual(benchmark.agent, agent)
        self.assertEqual(benchmark.sampled_positions, 3)

    def test_inline_mcts_profile_accepts_short_rollout_policy_fields(self) -> None:
        profile = _parse_inline_mcts_profile(
            "name=informed,i=100,d=32,c=0.8,ce=neutral,p=epsilon,rh=1,e=0.25"
        )

        self.assertEqual(profile.name, "informed")
        self.assertEqual(profile.agent.iterations, 100)
        self.assertEqual(profile.agent.rollout_depth, 32)
        self.assertEqual(profile.agent.exploration, 0.8)
        self.assertIsInstance(profile.agent.cutoff_evaluator, NeutralEvaluator)
        self.assertEqual(
            profile.agent.rollout_policy,
            EpsilonGreedy(0.25, GameHeuristic(1)),
        )

    def test_inline_mcts_profile_accepts_collect_scoped_rollout(self) -> None:
        profile = _parse_inline_mcts_profile(
            "i=5000,d=32,h=2,p=epsilon,rh=2,e=0.25,phase=collect"
        )

        self.assertEqual(
            profile.agent.rollout_policy,
            ConditionalRollout(
                TurnPhaseIs("collect"),
                EpsilonGreedy(0.25, GameHeuristic(2)),
                UniformRandom(),
            ),
        )
        self.assertIn("phase-collect", profile.name)

    def test_inline_mcts_profile_accepts_time_budget(self) -> None:
        profile = _parse_inline_mcts_profile("t=0.25,d=32")

        self.assertEqual(profile.name, "mcts-t0.25-d32")
        self.assertIsNone(profile.agent.iterations)
        self.assertEqual(profile.agent.time_budget, 0.25)

        with self.assertRaisesRegex(ValueError, "cannot combine iterations"):
            _parse_inline_mcts_profile("i=100,t=0.25")

    def test_inline_mcts_profile_accepts_progressive_bias(self) -> None:
        profile = _parse_inline_mcts_profile(
            "i=100,d=130,h=2,pb=0.25,pbh=2,pbphase=collect,rd=true"
        )

        self.assertEqual(
            profile.agent.progressive_bias,
            ProgressiveBias(0.25, GameHeuristic(2), TurnPhaseIs("collect")),
        )
        self.assertTrue(profile.agent.root_diagnostics)
        self.assertIn("pb0.25", profile.name)

    def test_scripted_humans_receive_positions_and_finish_a_match(self) -> None:
        first_moves = iter([(0, 0), (0, 1), (0, 2)])
        second_moves = iter([(1, 0), (1, 1)])
        observed_turns = []

        def select_first(turn):
            observed_turns.append(turn)
            return TicTacToeAction(*next(first_moves))

        def select_second(turn):
            observed_turns.append(turn)
            return TicTacToeAction(*next(second_moves))

        result = Match(
            first=HumanAgent(select_first),
            second=HumanAgent(select_second),
        ).run()

        self.assertEqual(result.winner, 0)
        self.assertEqual(result.plies, 5)
        self.assertEqual(observed_turns[0].player, 0)
        self.assertIsInstance(observed_turns[0].game, TicTacToe)
        self.assertEqual(observed_turns[0].board, ((None,) * 3,) * 3)
        self.assertEqual(len(observed_turns[0].legal_actions), 9)

    def test_scripted_humans_can_finish_connect_four(self) -> None:
        first_moves = iter([0, 1, 2, 3])
        second_moves = iter([0, 1, 2])
        observed_turns = []

        def select_first(turn):
            observed_turns.append(turn)
            return ConnectFourAction(next(first_moves))

        def select_second(turn):
            observed_turns.append(turn)
            return ConnectFourAction(next(second_moves))

        result = Match(
            game=ConnectFour(),
            first=HumanAgent(select_first),
            second=HumanAgent(select_second),
        ).run()

        self.assertEqual(result.winner, 0)
        self.assertEqual(result.plies, 7)
        self.assertIsInstance(observed_turns[0].game, ConnectFour)
        self.assertEqual(observed_turns[0].board, ((None,) * 7,) * 6)
        self.assertEqual(len(observed_turns[0].legal_actions), 7)
        self.assertTrue(
            all(isinstance(move.action, ConnectFourAction) for move in result.moves)
        )

    def test_random_agents_can_finish_boop(self) -> None:
        result = Match(
            game=Boop(),
            first=RandomAgent(),
            second=RandomAgent(),
            seed=9,
            max_plies=1_000,
        ).run()

        self.assertIsNotNone(result.winner)
        self.assertTrue(all(isinstance(move.action, BoopAction) for move in result.moves))
        self.assertEqual(len(result.final_board), 6)
        self.assertTrue(
            all(
                cell is None or isinstance(cell, BoopPiece)
                for row in result.final_board
                for cell in row
            )
        )
        self.assertIsNotNone(result.pools)

    def test_random_agents_finish_spirits_with_reproducible_scores(self) -> None:
        first = Match(
            game=SpiritsOfTheForest(),
            first=RandomAgent(),
            second=RandomAgent(),
            seed=42,
            max_plies=256,
        ).run()
        repeated = Match(
            game=SpiritsOfTheForest(),
            first=RandomAgent(),
            second=RandomAgent(),
            seed=42,
            max_plies=256,
        ).run()

        self.assertEqual(first.moves, repeated.moves)
        self.assertEqual(first.scores, repeated.scores)
        self.assertEqual(sum(item.tiles for item in first.spirit_collections), 48)
        self.assertEqual(len(first.final_board), 4)
        self.assertTrue(all(tile is None for row in first.final_board for tile in row))

    def test_spirits_human_receives_the_seeded_forest_and_typed_actions(self) -> None:
        turns = []

        def inspect(turn):
            turns.append(turn)
            raise RuntimeError("inspection complete")

        with self.assertRaisesRegex(RuntimeError, "inspection complete"):
            Match(
                game=SpiritsOfTheForest(),
                first=HumanAgent(inspect),
                second=RandomAgent(),
                seed=5,
            ).run()

        turn = turns[0]
        self.assertEqual(len(turn.board), 4)
        self.assertEqual(len(turn.board[0]), 12)
        self.assertTrue(all(isinstance(tile, SpiritTile) for row in turn.board for tile in row))
        self.assertEqual(len(turn.legal_actions), 8)
        self.assertTrue(all(isinstance(action, TakeSpiritTile) for action in turn.legal_actions))
        self.assertEqual(turn.gemstone_pools[0].available, 3)

    def test_spirits_mcts_heuristics_and_gui_are_available(self) -> None:
        for heuristic in (0, 1, 2):
            result = Match(
                game=SpiritsOfTheForest(),
                first=MctsAgent(iterations=2, rollout_depth=2, heuristic=heuristic),
                second=RandomAgent(),
                seed=9,
                max_plies=256,
            ).run()
            self.assertIsNotNone(result.scores)

        with self.assertRaisesRegex(ValueError, "available indices: 0..2"):
            Match(game=SpiritsOfTheForest(), first=MctsAgent(heuristic=3))

        gui = SpiritsOfTheForestGui()
        gui.start(
            GuiPlayer("human"),
            GuiPlayer("random"),
            seed=9,
            minimum_move_seconds=0,
        )
        waiting = self.wait_for_gui(gui, lambda state: state["status"] == "waiting_human")
        self.assertEqual(len(waiting["forest"]), 48)
        self.assertEqual(len(waiting["legal_actions"]), 8)
        gui.submit_move(waiting["legal_actions"][0]["index"])
        waiting = self.wait_for_gui(
            gui,
            lambda state: state["status"] == "waiting_human"
            and state["phase"] == "place_gemstone",
        )
        self.assertGreater(len(waiting["legal_actions"]), 1)
        gui.cancel()

    def test_cli_and_gui_dispatch_spotf(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "match",
                    "--game",
                    "spotf",
                    "--first",
                    "random",
                    "--second",
                    "random",
                    "--seed",
                    "9",
                    "--json",
                ]
            )
        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(payload["scores"]), 2)
        self.assertTrue(
            all(
                move["action"]["type"] == "spotf"
                for move in payload["moves"]
            )
        )

        with patch("meeple_bots.gui.server.serve_gui") as serve:
            run_gui(game="spotf", open_browser=False)
        application, page = serve.call_args.args
        self.assertEqual(application.snapshot()["game"], "spotf")
        self.assertIn("Spirits of the Forest", page)

        parsed = build_parser().parse_args(
            ["gui", "--game", "spirits-of-the-forest", "--no-browser"]
        )
        self.assertEqual(parsed.game, "spotf")

    def test_human_boop_selector_receives_pools_and_typed_actions(self) -> None:
        observed_turns = []

        def inspect_turn(turn):
            observed_turns.append(turn)
            raise RuntimeError("inspection complete")

        with self.assertRaisesRegex(RuntimeError, "inspection complete"):
            Match(
                game=Boop(),
                first=HumanAgent(inspect_turn),
                second=RandomAgent(),
            ).run()

        turn = observed_turns[0]
        self.assertIsInstance(turn.game, Boop)
        self.assertEqual(len(turn.legal_actions), 36)
        self.assertTrue(
            all(
                action.piece is BoopPieceKind.KITTEN
                for action in turn.legal_actions
            )
        )
        self.assertEqual(turn.pools[0].kittens, 8)
        self.assertEqual(turn.pools[0].cats, 0)

    def test_human_boop_observer_receives_the_state_after_boops(self) -> None:
        first_targets = iter([(2, 2), (2, 3)])
        observations = []

        def select_target(turn):
            row, column = next(first_targets)
            return next(
                action
                for action in turn.legal_actions
                if action.piece is BoopPieceKind.KITTEN
                and action.row == row
                and action.column == column
            )

        def select_safe_move(turn):
            return next(
                action
                for action in turn.legal_actions
                if action.piece is BoopPieceKind.KITTEN
                and action.row == 0
                and action.column == 0
            )

        def observe(move):
            observations.append(move)
            if len(observations) == 2:
                raise RuntimeError("observation complete")

        with self.assertRaisesRegex(RuntimeError, "observation complete"):
            Match(
                game=Boop(),
                first=HumanAgent(select_target, observe_action=observe),
                second=HumanAgent(select_safe_move),
            ).run()

        result = observations[1]
        self.assertIsInstance(result, HumanMoveObservation)
        self.assertEqual(result.player, 0)
        self.assertIsNone(result.board[2][2])
        self.assertEqual(
            result.board[2][1],
            BoopPiece(player=0, kind=BoopPieceKind.KITTEN),
        )
        self.assertEqual(
            result.board[2][3],
            BoopPiece(player=0, kind=BoopPieceKind.KITTEN),
        )

    def test_cli_prompts_for_human_moves(self) -> None:
        output = io.StringIO()
        prompts = io.StringIO()
        moves = ["0 0", "1 0", "0 1", "1 1", "0 2"]

        with (
            patch("builtins.input", side_effect=moves),
            redirect_stdout(output),
            redirect_stderr(prompts),
        ):
            exit_code = main(["match", "--first", "human", "--second", "human"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Winner: player 0", output.getvalue())
        self.assertIn("Final board:", output.getvalue())
        self.assertIn("0 | X X X", output.getvalue())
        self.assertIn("enter row and column", prompts.getvalue())
        self.assertIn("Board after player 0's move:", prompts.getvalue())

    def test_cli_shows_final_board_for_automated_match(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "match",
                    "--first",
                    "random",
                    "--second",
                    "random",
                    "--seed",
                    "9",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Final board:", output.getvalue())
        self.assertIn("    0 1 2", output.getvalue())

    def test_cli_shows_final_connect_four_board_with_gravity(self) -> None:
        output = io.StringIO()
        prompts = io.StringIO()
        moves = ["0", "0", "1", "1", "2", "2", "3"]

        with (
            patch("builtins.input", side_effect=moves),
            redirect_stdout(output),
            redirect_stderr(prompts),
        ):
            exit_code = main(
                [
                    "match",
                    "--game",
                    "connect-four",
                    "--first",
                    "human",
                    "--second",
                    "human",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("4 | O O O . . . .", output.getvalue())
        self.assertIn("5 | X X X X . . .", output.getvalue())

    def test_cli_can_return_json(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "match",
                    "--first",
                    "random",
                    "--second",
                    "random",
                    "--seed",
                    "9",
                    "--json",
                ]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["seed"], 9)
        self.assertEqual(len(payload["moves"]), payload["plies"])

    def test_cli_can_run_connect_four(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "match",
                    "--game",
                    "connect-four",
                    "--first",
                    "random",
                    "--second",
                    "random",
                    "--seed",
                    "9",
                    "--json",
                ]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertGreaterEqual(payload["plies"], 7)
        self.assertTrue(
            all(move["action"]["type"] == "connect_four" for move in payload["moves"])
        )

    def test_cli_can_run_boop(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "match",
                    "--game",
                    "boop",
                    "--first",
                    "random",
                    "--second",
                    "random",
                    "--seed",
                    "9",
                    "--max-plies",
                    "1000",
                    "--json",
                ]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(
            all(move["action"]["type"] == "boop" for move in payload["moves"])
        )

    def test_cli_heuristic_flag_defaults_to_zero(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "match",
                    "--game",
                    "boop",
                    "--first",
                    "mcts",
                    "--second",
                    "random",
                    "--mcts-iterations",
                    "1",
                    "--mcts-rollout-depth",
                    "1",
                    "--first-mcts-heuristic",
                    "--seed",
                    "9",
                    "--json",
                ]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["players"][0]["heuristic"], 0)
        self.assertIsNone(payload["players"][1]["heuristic"])

    def test_cli_rejects_heuristics_for_non_mcts_players(self) -> None:
        errors = io.StringIO()
        with redirect_stderr(errors):
            exit_code = main(
                [
                    "match",
                    "--game",
                    "boop",
                    "--first",
                    "random",
                    "--first-mcts-heuristic",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("requires the corresponding player to be MCTS", errors.getvalue())

    def test_cli_match_loads_an_mcts_profile(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory, "match-profile.toml")
            profile.write_text(
                '\n'.join(
                    [
                        'name = "match-mcts"',
                        "iterations = 1",
                        "rollout_depth = 1",
                        "use_heuristic = true",
                        "heuristic_index = 1",
                        'rollout_policy = "epsilon_greedy_heuristic"',
                        "rollout_epsilon = 0.2",
                    ]
                ),
                encoding="utf-8",
            )
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "match",
                        "--game",
                        "boop",
                        "--first",
                        "mcts",
                        "--second",
                        "random",
                        "--first-mcts-config",
                        str(profile),
                        "--seed",
                        "9",
                        "--json",
                    ]
                )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["players"][0]["heuristic"], 1)
        self.assertEqual(
            payload["players"][0]["rollout_policy"],
            "epsilon_greedy",
        )
        self.assertEqual(
            payload["players"][0]["cutoff_evaluator"],
            {"kind": "game_heuristic", "index": 1},
        )
        self.assertEqual(
            payload["players"][0]["rollout_evaluator"],
            {"kind": "game_heuristic", "index": 1},
        )
        self.assertEqual(payload["players"][0]["rollout_epsilon"], 0.2)

    def test_cli_match_rejects_a_profile_for_a_non_mcts_player(self) -> None:
        errors = io.StringIO()
        with redirect_stderr(errors):
            exit_code = main(
                [
                    "match",
                    "--first",
                    "random",
                    "--first-mcts-config",
                    "profile.toml",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn(
            "--first-mcts-config requires the corresponding player to be MCTS",
            errors.getvalue(),
        )

    def test_cli_match_rejects_a_profile_and_heuristic_for_the_same_player(
        self,
    ) -> None:
        errors = io.StringIO()
        with redirect_stderr(errors):
            exit_code = main(
                [
                    "match",
                    "--first-mcts-config",
                    "profile.toml",
                    "--first-mcts-heuristic",
                    "0",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn(
            "--first-mcts-config cannot be combined with --first-mcts-heuristic",
            errors.getvalue(),
        )

    def test_cli_analyze_returns_simple_evaluation_json(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "analyze",
                    "--game",
                    "tic-tac-toe",
                    "--samples",
                    "8",
                    "--target-time",
                    "1",
                    "--json",
                ]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["game"], "tic-tac-toe")
        self.assertEqual(payload["initial_legal_actions"], 9)
        self.assertLessEqual(payload["estimated_depth"], 9)
        self.assertEqual(payload["target_time_seconds"], 1.0)
        self.assertIn("player_turn_depth_p95", payload)
        self.assertGreaterEqual(len(payload["rollout_costs"]), 1)
        self.assertEqual(
            [budget["seconds"] for budget in payload["rollout_costs"][0]["iteration_budgets"]],
            [1, 2, 5, 10, 20],
        )
        self.assertEqual(payload["suggested_experiments"][1]["label"], "Balanced")
        self.assertIn("recommended_iterations", payload)
        self.assertIn("milliseconds_per_iteration", payload)
        self.assertIn("estimated_decision_time_ms", payload)
        self.assertEqual(payload["configured_agent_benchmarks"], [])

    def test_cli_analyze_compares_more_than_two_exact_agent_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = []
            for name, iterations, depth in (
                ("short", 1, 2),
                ("balanced", 2, 4),
                ("deep", 3, 8),
            ):
                profile = root / f"{name}.toml"
                profile.write_text(
                    "\n".join(
                        [
                            f'name = "{name}"',
                            f"iterations = {iterations}",
                            f"rollout_depth = {depth}",
                            "use_heuristic = false",
                        ]
                    ),
                    encoding="utf-8",
                )
                profiles.append(profile)

            arguments = [
                "analyze",
                "--game",
                "tic-tac-toe",
                "--samples",
                "8",
                "--target-time",
                "0.1",
                "--json",
            ]
            for profile in profiles:
                arguments.extend(("--agent-config", str(profile)))
            arguments.extend(("--agent", "name=inline,iterations=1,depth=3"))
            output = io.StringIO()
            progress = io.StringIO()
            with redirect_stdout(output), redirect_stderr(progress):
                exit_code = main(arguments)

        payload = json.loads(output.getvalue())
        benchmarks = payload["configured_agent_benchmarks"]
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(benchmarks), 4)
        self.assertEqual([row["rank"] for row in benchmarks], [1, 2, 3, 4])
        self.assertEqual(
            {row["name"] for row in benchmarks},
            {"short", "balanced", "deep", "inline"},
        )
        self.assertEqual(benchmarks[0]["relative_to_fastest"], 1.0)
        self.assertEqual(
            [row["decision_time_mean_ms"] for row in benchmarks],
            sorted(row["decision_time_mean_ms"] for row in benchmarks),
        )
        self.assertTrue(all(row["sampled_positions"] == 3 for row in benchmarks))
        self.assertTrue(all(row["target_time_ratio"] > 0.0 for row in benchmarks))
        self.assertIn("Benchmarking short", progress.getvalue())
        self.assertIn("Benchmarking balanced", progress.getvalue())
        self.assertIn("Benchmarking deep", progress.getvalue())
        self.assertIn("Benchmarking inline", progress.getvalue())

    def test_cli_analyze_accepts_repeated_inline_agents_with_defaults(self) -> None:
        output = io.StringIO()
        progress = io.StringIO()
        with redirect_stdout(output), redirect_stderr(progress):
            exit_code = main(
                [
                    "analyze",
                    "--game",
                    "tic-tac-toe",
                    "--samples",
                    "8",
                    "--target-time",
                    "0.1",
                    "--agent",
                    "--agent",
                    "i=5,d=8",
                    "--agent",
                    "name=custom,iterations=2,exploration=0.5",
                    "--json",
                ]
            )

        payload = json.loads(output.getvalue())
        benchmarks = {
            benchmark["name"]: benchmark
            for benchmark in payload["configured_agent_benchmarks"]
        }
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            set(benchmarks),
            {"mcts-i1000-d16", "mcts-i5-d8", "custom"},
        )
        self.assertEqual(benchmarks["mcts-i1000-d16"]["iterations"], 1000)
        self.assertEqual(benchmarks["mcts-i1000-d16"]["rollout_depth"], 16)
        self.assertIsNone(benchmarks["mcts-i1000-d16"]["heuristic"])
        self.assertEqual(benchmarks["mcts-i5-d8"]["iterations"], 5)
        self.assertEqual(benchmarks["mcts-i5-d8"]["rollout_depth"], 8)
        self.assertEqual(benchmarks["custom"]["exploration"], 0.5)
        self.assertIn("Benchmarking mcts-i1000-d16", progress.getvalue())

    def test_cli_analyze_inline_agent_supports_a_heuristic(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(io.StringIO()):
            exit_code = main(
                [
                    "analyze",
                    "--game",
                    "boop",
                    "--samples",
                    "1",
                    "--max-depth",
                    "1",
                    "--target-time",
                    "0.1",
                    "--agent",
                    "i=1,d=1,h=0",
                    "--json",
                ]
            )

        payload = json.loads(output.getvalue())
        benchmark = payload["configured_agent_benchmarks"][0]
        self.assertEqual(exit_code, 0)
        self.assertEqual(benchmark["name"], "mcts-h0-i1-d1")
        self.assertEqual(benchmark["heuristic"], 0)

    def test_cli_analyze_rejects_duplicate_inline_agent_aliases(self) -> None:
        errors = io.StringIO()
        with redirect_stderr(errors):
            exit_code = main(
                [
                    "analyze",
                    "--game",
                    "boop",
                    "--agent",
                    "i=1,iterations=2",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn(
            "duplicate inline analyze agent field: iterations",
            errors.getvalue(),
        )

    def test_cli_analyze_rejects_invalid_inline_agent_fields_before_sampling(self) -> None:
        errors = io.StringIO()
        with redirect_stderr(errors):
            exit_code = main(
                [
                    "analyze",
                    "--game",
                    "boop",
                    "--agent",
                    "iterations=oops,unknown=1",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("unknown inline analyze agent field: unknown", errors.getvalue())

    def test_cli_batch_loads_an_mcts_profile_and_reports_progress(self) -> None:
        output = io.StringIO()
        progress = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory, "test-profile.toml")
            profile.write_text(
                '\n'.join(
                    [
                        'name = "test-mcts"',
                        "iterations = 4",
                        "rollout_depth = 9",
                        "exploration = 1.4142135623730951",
                        "use_heuristic = false",
                        "heuristic_index = 0",
                    ]
                ),
                encoding="utf-8",
            )
            with redirect_stdout(output), redirect_stderr(progress):
                exit_code = main(
                    [
                        "batch",
                        "--game",
                        "tic-tac-toe",
                    "--matches",
                    "4",
                    "--workers",
                    "2",
                    "--agent-b-config",
                        str(profile),
                        "--seed",
                        "42",
                        "--json",
                    ]
                )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["matches"], 4)
        self.assertEqual(payload["workers"], 2)
        self.assertEqual(payload["agents"]["a"]["type"], "random")
        self.assertEqual(payload["agents"]["b"]["name"], "test-mcts")
        self.assertEqual(payload["agents"]["b"]["iterations"], 4)
        self.assertEqual(
            payload["summary"]["agent_a_wins"]
            + payload["summary"]["agent_b_wins"]
            + payload["summary"]["draws"],
            4,
        )
        self.assertEqual(
            [game["agent_a_player"] for game in payload["games"]],
            [0, 1, 0, 1],
        )
        self.assertIn("[1/4] starting", progress.getvalue())
        self.assertIn("[4/4] completed", progress.getvalue())

    def test_cli_batch_requires_a_profile_for_each_mcts_agent(self) -> None:
        errors = io.StringIO()
        with redirect_stderr(errors):
            exit_code = main(
                [
                    "batch",
                    "--game",
                    "tic-tac-toe",
                    "--matches",
                    "2",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("--agent-b-config is required", errors.getvalue())

    def test_cli_batch_profile_can_enable_the_boop_heuristic(self) -> None:
        output = io.StringIO()
        progress = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory, "heuristic.toml")
            profile.write_text(
                '\n'.join(
                    [
                        'name = "strategic"',
                        "iterations = 1",
                        "rollout_depth = 1",
                        "use_heuristic = true",
                        "heuristic_index = 1",
                    ]
                ),
                encoding="utf-8",
            )
            with redirect_stdout(output), redirect_stderr(progress):
                exit_code = main(
                    [
                        "batch",
                        "--game",
                        "boop",
                        "--matches",
                        "1",
                        "--agent-b-config",
                        str(profile),
                        "--json",
                    ]
                )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["agents"]["b"]["name"], "strategic")
        self.assertEqual(payload["agents"]["b"]["heuristic"], 1)

    def test_cli_batch_writes_an_extractable_jsonl_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "batches" / "boop-random.jsonl"
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(io.StringIO()):
                exit_code = main(
                    [
                        "batch",
                        "--game",
                        "boop",
                        "--matches",
                        "2",
                        "--agent-b",
                        "random",
                        "--workers",
                        "2",
                        "--seed",
                        "91",
                        "--max-plies",
                        "1000",
                        "--output",
                        str(trace),
                        "--json",
                    ]
                )

            summary = json.loads(output.getvalue())
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            tournament_trace = self._create_small_boop_tournament(root)
            extraction_dir = root / "combined" / "data"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                extraction_exit = main(
                    [
                        "extract",
                        "--input",
                        str(trace),
                        str(tournament_trace),
                        "--output-dir",
                        str(extraction_dir),
                    ]
                )
            studies = self._read_csv(extraction_dir / "studies.csv")
            matches = self._read_csv(extraction_dir / "matches.csv")

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["output"], str(trace))
        self.assertEqual(records[0]["record_type"], "tournament")
        self.assertEqual(records[0]["study_type"], "batch")
        self.assertEqual(records[0]["total_pairings"], 1)
        self.assertEqual(records[0]["total_matches"], 2)
        self.assertEqual(
            records[0]["agents"],
            [
                {
                    "name": "random",
                    "type": "random",
                    "self_play": True,
                }
            ],
        )
        self.assertEqual([record["match_number"] for record in records[1:]], [1, 2])
        self.assertEqual(
            [record["pairing_match_number"] for record in records[1:]],
            [1, 2],
        )
        self.assertTrue(all(record["result"]["moves"] for record in records[1:]))
        self.assertEqual(extraction_exit, 0)
        self.assertEqual(len(matches), 3)
        self.assertEqual(
            {study["study_type"] for study in studies},
            {"batch", "tournament"},
        )

    def test_cli_batch_protects_an_existing_trace_unless_overwrite_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory, "batch.jsonl")
            trace.write_text("keep me\n", encoding="utf-8")
            arguments = [
                "batch",
                "--game",
                "tic-tac-toe",
                "--matches",
                "1",
                "--agent-b",
                "random",
                "--workers",
                "1",
                "--output",
                str(trace),
            ]
            errors = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(errors):
                protected_exit = main(arguments)

            self.assertEqual(protected_exit, 1)
            self.assertIn("--overwrite", errors.getvalue())
            self.assertEqual(trace.read_text(encoding="utf-8"), "keep me\n")

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                overwritten_exit = main([*arguments, "--overwrite"])
            records = [json.loads(line) for line in trace.read_text().splitlines()]

        self.assertEqual(overwritten_exit, 0)
        self.assertEqual(records[0]["study_type"], "batch")
        self.assertEqual(len(records), 2)

    def test_cli_tournament_runs_round_robin_and_selected_self_play(self) -> None:
        output = io.StringIO()
        progress = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "tournament.toml")
            trace = Path(directory, "generated", "matches.jsonl")
            config.write_text(
                '\n'.join(
                    [
                        'game = "tic-tac-toe"',
                        'output = "generated/matches.jsonl"',
                        "matches_per_pair = 2",
                        "seed = 17",
                        "max_plies = 9",
                        "workers = 2",
                        "",
                        "[[agents]]",
                        'name = "alpha"',
                        'kind = "random"',
                        "self_play = true",
                        "",
                        "[[agents]]",
                        'name = "beta"',
                        'kind = "random"',
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(_load_tournament_config(config).workers, 2)

            with redirect_stdout(output), redirect_stderr(progress):
                exit_code = main(
                    [
                        "tournament",
                        "--config",
                        str(config),
                        "--workers",
                        "1",
                        "--json",
                    ]
                )

            records = [json.loads(line) for line in trace.read_text().splitlines()]

        summary = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["pairings"], 2)
        self.assertEqual(summary["pairing_mode"], "round_robin")
        self.assertEqual(summary["matches"], 4)
        self.assertEqual(summary["workers"], 1)
        self.assertEqual(summary["output"], str(trace))
        self.assertEqual(summary["standings"]["alpha"]["games"], 2)
        self.assertEqual(summary["standings"]["alpha"]["self_play_games"], 2)
        self.assertEqual(records[0]["record_type"], "tournament")
        self.assertEqual(records[0]["schema_version"], 1)
        self.assertEqual(records[0]["study_type"], "tournament")
        self.assertEqual(records[0]["pairing_mode"], "round_robin")
        self.assertEqual(records[0]["workers"], 1)
        self.assertEqual(len(records), 5)
        self.assertEqual(
            [record["result"]["seed"] for record in records[1:]],
            [17, 18, 19, 20],
        )
        self.assertEqual(
            [record["agent_a_player"] for record in records[1:3]],
            [0, 1],
        )
        self.assertTrue(all(record["result"]["moves"] for record in records[1:]))
        self.assertIn("[4/4]", progress.getvalue())

    def test_cli_tournament_rejects_duplicate_agent_names(self) -> None:
        errors = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "tournament.toml")
            config.write_text(
                '\n'.join(
                    [
                        'game = "tic-tac-toe"',
                        "matches_per_pair = 1",
                        "[[agents]]",
                        'name = "same"',
                        'kind = "random"',
                        "[[agents]]",
                        'name = "same"',
                        'kind = "random"',
                    ]
                ),
                encoding="utf-8",
            )
            with redirect_stderr(errors):
                exit_code = main(
                    [
                        "tournament",
                        "--config",
                        str(config),
                        "--output",
                        str(Path(directory, "matches.jsonl")),
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("agent names must be unique", errors.getvalue())

    def test_tournament_agent_grid_expands_cartesian_product(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory, "grid.toml")
            config_path.write_text(
                '\n'.join(
                    [
                        'game = "spotf"',
                        "matches_per_pair = 1",
                        "[[agents]]",
                        'name = "mcts-h0"',
                        'kind = "mcts"',
                        "iterations = [100, 1000, 10000]",
                        "rollout_depth = [8, 16, 32]",
                        "use_heuristic = true",
                        "heuristic_index = 0",
                    ]
                ),
                encoding="utf-8",
            )

            config = _load_tournament_config(config_path)

        self.assertIsInstance(config.game, SpiritsOfTheForest)
        self.assertEqual(len(config.agents), 9)
        self.assertEqual(
            [agent.name for agent in config.agents],
            [
                "mcts-h0-i100-d8",
                "mcts-h0-i100-d16",
                "mcts-h0-i100-d32",
                "mcts-h0-i1000-d8",
                "mcts-h0-i1000-d16",
                "mcts-h0-i1000-d32",
                "mcts-h0-i10000-d8",
                "mcts-h0-i10000-d16",
                "mcts-h0-i10000-d32",
            ],
        )
        self.assertEqual(
            [
                (agent.agent.iterations, agent.agent.rollout_depth)
                for agent in config.agents
                if isinstance(agent.agent, MctsAgent)
            ],
            [
                (100, 8),
                (100, 16),
                (100, 32),
                (1000, 8),
                (1000, 16),
                (1000, 32),
                (10000, 8),
                (10000, 16),
                (10000, 32),
            ],
        )

    def test_adjacent_pairing_mode_only_pairs_neighbouring_sweep_values(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory, "adjacent.toml")
            config_path.write_text(
                "\n".join(
                    [
                        'game = "boop"',
                        'pairing_mode = "adjacent"',
                        "matches_per_pair = 1",
                        "[[agents]]",
                        'name = "uct"',
                        'kind = "mcts"',
                        "iterations = 1",
                        "rollout_depth = 1",
                        "exploration = [0.25, 0.5, 1.0, 1.4]",
                    ]
                ),
                encoding="utf-8",
            )

            config = _load_tournament_config(config_path)
            pairs = [
                (agent_a.name, agent_b.name)
                for agent_a, agent_b in _tournament_pairings(
                    config.agents, config.pairing_mode
                )
            ]

        self.assertEqual(
            pairs,
            [
                ("uct-c0.25", "uct-c0.5"),
                ("uct-c0.5", "uct-c1.0"),
                ("uct-c1.0", "uct-c1.4"),
            ],
        )

    def test_adjacent_pairing_mode_keeps_cross_template_round_robin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory, "cross-template.toml")
            config_path.write_text(
                "\n".join(
                    [
                        'game = "tic-tac-toe"',
                        'pairing_mode = "adjacent"',
                        "matches_per_pair = 1",
                        "[[agents]]",
                        'name = "uct"',
                        'kind = "mcts"',
                        "iterations = 1",
                        "rollout_depth = 1",
                        "exploration = [0.5, 1.0, 1.5]",
                        "[[agents]]",
                        'name = "random"',
                        'kind = "random"',
                        "[[agents]]",
                        'name = "other"',
                        'kind = "mcts"',
                        "iterations = 1",
                        "rollout_depth = 1",
                    ]
                ),
                encoding="utf-8",
            )

            config = _load_tournament_config(config_path)
            pairs = {
                (agent_a.name, agent_b.name)
                for agent_a, agent_b in _tournament_pairings(
                    config.agents, config.pairing_mode
                )
            }

        self.assertEqual(len(pairs), 9)
        for sweep_name in ("uct-c0.5", "uct-c1.0", "uct-c1.5"):
            self.assertIn((sweep_name, "random"), pairs)
            self.assertIn((sweep_name, "other"), pairs)
        self.assertIn(("random", "other"), pairs)
        self.assertNotIn(("uct-c0.5", "uct-c1.5"), pairs)

    def test_adjacent_pairing_mode_connects_each_grid_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory, "adjacent-grid.toml")
            config_path.write_text(
                "\n".join(
                    [
                        'game = "boop"',
                        'pairing_mode = "adjacent"',
                        "matches_per_pair = 1",
                        "[[agents]]",
                        'name = "grid"',
                        'kind = "mcts"',
                        "iterations = 1",
                        "rollout_depth = [32, 64]",
                        "exploration = [0.5, 1.0, 1.5]",
                    ]
                ),
                encoding="utf-8",
            )

            config = _load_tournament_config(config_path)
            pairs = [
                (agent_a.name, agent_b.name)
                for agent_a, agent_b in _tournament_pairings(
                    config.agents, config.pairing_mode
                )
            ]

        expected = {
            ("grid-d32-c0.5", "grid-d32-c1.0"),
            ("grid-d32-c1.0", "grid-d32-c1.5"),
            ("grid-d64-c0.5", "grid-d64-c1.0"),
            ("grid-d64-c1.0", "grid-d64-c1.5"),
            ("grid-d32-c0.5", "grid-d64-c0.5"),
            ("grid-d32-c1.0", "grid-d64-c1.0"),
            ("grid-d32-c1.5", "grid-d64-c1.5"),
        }
        self.assertEqual(set(pairs), expected)
        self.assertEqual(len(pairs), len(set(pairs)))
        for agent_a, agent_b in _tournament_pairings(
            config.agents, config.pairing_mode
        ):
            differing_parameters = sum(
                (
                    agent_a.agent.rollout_depth != agent_b.agent.rollout_depth,
                    agent_a.agent.exploration != agent_b.agent.exploration,
                )
            )
            self.assertEqual(differing_parameters, 1)

    def test_round_robin_pairing_mode_preserves_all_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory, "round-robin.toml")
            config_path.write_text(
                "\n".join(
                    [
                        'game = "boop"',
                        "matches_per_pair = 1",
                        "[[agents]]",
                        'name = "uct"',
                        'kind = "mcts"',
                        "iterations = 1",
                        "rollout_depth = 1",
                        "exploration = [0.5, 1.0, 1.5, 2.0]",
                    ]
                ),
                encoding="utf-8",
            )

            config = _load_tournament_config(config_path)
            default_pairs = _tournament_pairings(config.agents)
            explicit_pairs = _tournament_pairings(config.agents, "round_robin")

        self.assertEqual(config.pairing_mode, "round_robin")
        self.assertEqual(default_pairs, explicit_pairs)
        self.assertEqual(len(default_pairs), 6)

    def test_adjacent_pairing_mode_supports_scalar_templates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory, "scalar.toml")
            config_path.write_text(
                "\n".join(
                    [
                        'game = "tic-tac-toe"',
                        'pairing_mode = "adjacent"',
                        "matches_per_pair = 1",
                        "[[agents]]",
                        'name = "first"',
                        'kind = "random"',
                        "[[agents]]",
                        'name = "second"',
                        'kind = "random"',
                    ]
                ),
                encoding="utf-8",
            )

            config = _load_tournament_config(config_path)
            pairs = _tournament_pairings(config.agents, config.pairing_mode)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(
            (pairs[0][0].name, pairs[0][1].name),
            ("first", "second"),
        )

    def test_tournament_rejects_unknown_pairing_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory, "invalid-pairing.toml")
            config_path.write_text(
                "\n".join(
                    [
                        'game = "tic-tac-toe"',
                        'pairing_mode = "nearest"',
                        "matches_per_pair = 1",
                        "[[agents]]",
                        'name = "first"',
                        'kind = "random"',
                        "[[agents]]",
                        'name = "second"',
                        'kind = "random"',
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "round_robin or adjacent"):
                _load_tournament_config(config_path)

    def test_tournament_agent_grid_accepts_time_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory, "time-grid.toml")
            config_path.write_text(
                "\n".join(
                    [
                        'game = "tic-tac-toe"',
                        "matches_per_pair = 1",
                        "[[agents]]",
                        'name = "timed"',
                        'kind = "mcts"',
                        "time_budget = [0.01, 0.02]",
                        "rollout_depth = 4",
                    ]
                ),
                encoding="utf-8",
            )

            config = _load_tournament_config(config_path)

        self.assertEqual([agent.name for agent in config.agents], ["timed-t0.01", "timed-t0.02"])
        self.assertEqual(
            [agent.agent.time_budget for agent in config.agents],
            [0.01, 0.02],
        )
        self.assertTrue(all(agent.agent.iterations is None for agent in config.agents))

    def test_tournament_supports_independent_cutoff_and_rollout_evaluators(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory, "evaluators.toml")
            config_path.write_text(
                "\n".join(
                    [
                        'game = "spotf"',
                        "matches_per_pair = 1",
                        "[[agents]]",
                        'name = "neutral"',
                        'kind = "mcts"',
                        "iterations = 1",
                        "rollout_depth = 1",
                        'cutoff_evaluator = { kind = "neutral" }',
                        'rollout_policy = { kind = "uniform_random" }',
                        "[[agents]]",
                        'name = "cutoff-only"',
                        'kind = "mcts"',
                        "iterations = 1",
                        "rollout_depth = 1",
                        'cutoff_evaluator = { kind = "game_heuristic", index = 1 }',
                        'rollout_policy = { kind = "uniform_random" }',
                        "[[agents]]",
                        'name = "rollout-only"',
                        'kind = "mcts"',
                        "iterations = 1",
                        "rollout_depth = 1",
                        'cutoff_evaluator = { kind = "neutral" }',
                        'rollout_policy = { kind = "epsilon_greedy", epsilon = 0.2, '
                        'evaluator = { kind = "game_heuristic", index = 0 } }',
                        "[[agents]]",
                        'name = "both"',
                        'kind = "mcts"',
                        "iterations = 1",
                        "rollout_depth = 1",
                        'cutoff_evaluator = { kind = "game_heuristic", index = 0 }',
                        'rollout_policy = { kind = "greedy", '
                        'evaluator = { kind = "game_heuristic", index = 1 } }',
                    ]
                ),
                encoding="utf-8",
            )

            config = _load_tournament_config(config_path)

        neutral, cutoff_only, rollout_only, both = [
            entry.agent for entry in config.agents
        ]
        self.assertIsInstance(neutral.cutoff_evaluator, NeutralEvaluator)
        self.assertIsInstance(neutral.rollout_policy, UniformRandom)
        self.assertEqual(cutoff_only.cutoff_evaluator, GameHeuristic(1))
        self.assertIsInstance(cutoff_only.rollout_policy, UniformRandom)
        self.assertIsInstance(rollout_only.cutoff_evaluator, NeutralEvaluator)
        self.assertEqual(
            rollout_only.rollout_policy,
            EpsilonGreedy(0.2, GameHeuristic(0)),
        )
        self.assertEqual(both.cutoff_evaluator, GameHeuristic(0))
        self.assertEqual(both.rollout_policy, Greedy(GameHeuristic(1)))

    def test_tournament_agent_grid_can_vary_rollout_epsilon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory, "rollout-grid.toml")
            config_path.write_text(
                "\n".join(
                    [
                        'game = "spotf"',
                        "matches_per_pair = 1",
                        "[[agents]]",
                        'name = "random"',
                        'kind = "random"',
                        "[[agents]]",
                        'name = "informed"',
                        'kind = "mcts"',
                        "iterations = 1",
                        "rollout_depth = 1",
                        "use_heuristic = true",
                        "heuristic_index = 0",
                        'rollout_policy = "epsilon_greedy_heuristic"',
                        "rollout_epsilon = [0.0, 0.2]",
                    ]
                ),
                encoding="utf-8",
            )

            config = _load_tournament_config(config_path)

        self.assertEqual(
            [agent.name for agent in config.agents],
            ["random", "informed-e0.0", "informed-e0.2"],
        )
        policies = [agent.agent.rollout_policy for agent in config.agents[1:]]
        self.assertTrue(
            all(isinstance(policy, EpsilonGreedy) for policy in policies)
        )
        self.assertEqual([policy.epsilon for policy in policies], [0.0, 0.2])

    def test_tournament_structured_grid_expands_nested_evaluator_and_epsilon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory, "structured-grid.toml")
            config_path.write_text(
                "\n".join(
                    [
                        'game = "spotf"',
                        "matches_per_pair = 1",
                        "[[agents]]",
                        'name = "random"',
                        'kind = "random"',
                        "[[agents]]",
                        'name = "structured"',
                        'kind = "mcts"',
                        "iterations = 1",
                        "rollout_depth = 1",
                        'cutoff_evaluator = { kind = "game_heuristic", index = [0, 1] }',
                        'rollout_policy = { kind = "epsilon_greedy", '
                        'epsilon = [0.0, 0.2], evaluator = { '
                        'kind = "game_heuristic", index = [0, 1] } }',
                    ]
                ),
                encoding="utf-8",
            )

            config = _load_tournament_config(config_path)

        self.assertEqual(
            [entry.name for entry in config.agents],
            [
                "random",
                "structured-h0-rh0-e0.0",
                "structured-h0-rh0-e0.2",
                "structured-h0-rh1-e0.0",
                "structured-h0-rh1-e0.2",
                "structured-h1-rh0-e0.0",
                "structured-h1-rh0-e0.2",
                "structured-h1-rh1-e0.0",
                "structured-h1-rh1-e0.2",
            ],
        )
        agents = [entry.agent for entry in config.agents[1:]]
        self.assertEqual(
            [agent.cutoff_evaluator.index for agent in agents],
            [0, 0, 0, 0, 1, 1, 1, 1],
        )
        self.assertEqual(
            [agent.rollout_policy.evaluator.index for agent in agents],
            [0, 0, 1, 1, 0, 0, 1, 1],
        )
        self.assertEqual(
            [agent.rollout_policy.epsilon for agent in agents],
            [0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2],
        )

    def test_tournament_conditional_grid_expands_primary_epsilon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory, "conditional-grid.toml")
            config_path.write_text(
                "\n".join(
                    [
                        'game = "spotf"',
                        "matches_per_pair = 1",
                        "[[agents]]",
                        'name = "random"',
                        'kind = "random"',
                        "[[agents]]",
                        'name = "collect"',
                        'kind = "mcts"',
                        "iterations = 1",
                        "rollout_depth = 1",
                        'cutoff_evaluator = { kind = "game_heuristic", index = 2 }',
                        'rollout_policy = { kind = "conditional", condition = { '
                        'kind = "turn_phase", phase = "collect" }, primary = { '
                        'kind = "epsilon_greedy", epsilon = [0.0, 0.25], evaluator = { '
                        'kind = "game_heuristic", index = 2 } }, fallback = { '
                        'kind = "uniform_random" } }',
                    ]
                ),
                encoding="utf-8",
            )

            config = _load_tournament_config(config_path)

        self.assertEqual(
            [entry.name for entry in config.agents],
            ["random", "collect-e0.0", "collect-e0.25"],
        )
        policies = [entry.agent.rollout_policy for entry in config.agents[1:]]
        self.assertTrue(
            all(isinstance(policy, ConditionalRollout) for policy in policies)
        )
        self.assertEqual(
            [policy.primary.epsilon for policy in policies],
            [0.0, 0.25],
        )
        self.assertTrue(
            all(isinstance(policy.fallback, UniformRandom) for policy in policies)
        )

    def test_tournament_grid_expands_progressive_bias_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory, "progressive-grid.toml")
            config_path.write_text(
                "\n".join(
                    [
                        'game = "spotf"',
                        "matches_per_pair = 1",
                        "[[agents]]",
                        'name = "random"',
                        'kind = "random"',
                        "[[agents]]",
                        'name = "progressive"',
                        'kind = "mcts"',
                        "iterations = 4",
                        "rollout_depth = 2",
                        'cutoff_evaluator = { kind = "game_heuristic", index = 2 }',
                        'rollout_policy = { kind = "uniform_random" }',
                        'progressive_bias = { weight = [0.0, 0.25], evaluator = { '
                        'kind = "game_heuristic", index = 2 }, condition = { '
                        'kind = "turn_phase", phase = "collect" } }',
                        "root_diagnostics = true",
                    ]
                ),
                encoding="utf-8",
            )

            config = _load_tournament_config(config_path)

        self.assertEqual(
            [entry.name for entry in config.agents],
            ["random", "progressive-pb0.0", "progressive-pb0.25"],
        )
        agents = [entry.agent for entry in config.agents[1:]]
        self.assertEqual(
            [agent.progressive_bias.weight for agent in agents],
            [0.0, 0.25],
        )
        self.assertTrue(all(agent.root_diagnostics for agent in agents))

    def test_tournament_agent_grid_rejects_duplicate_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory, "grid.toml")
            config_path.write_text(
                '\n'.join(
                    [
                        'game = "tic-tac-toe"',
                        "matches_per_pair = 1",
                        "[[agents]]",
                        'name = "mcts"',
                        'kind = "mcts"',
                        "iterations = [10, 10]",
                        "rollout_depth = 4",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "contains duplicate values"):
                _load_tournament_config(config_path)

    def test_tournament_agent_grid_supports_exploration_and_heuristics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory, "grid.toml")
            config_path.write_text(
                '\n'.join(
                    [
                        'game = "boop"',
                        "matches_per_pair = 1",
                        "[[agents]]",
                        'name = "mcts"',
                        'kind = "mcts"',
                        "iterations = 1",
                        "rollout_depth = 1",
                        "exploration = [0.5, 1.5]",
                        "use_heuristic = true",
                        "heuristic_index = [0, 1]",
                    ]
                ),
                encoding="utf-8",
            )

            config = _load_tournament_config(config_path)

        self.assertEqual(
            [agent.name for agent in config.agents],
            [
                "mcts-c0.5-h0",
                "mcts-c0.5-h1",
                "mcts-c1.5-h0",
                "mcts-c1.5-h1",
            ],
        )

    def test_cli_extracts_analysis_tables_from_a_boop_tournament(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = self._create_small_boop_tournament(root)
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["extract", "--input", str(trace), "--json"])

            summary = json.loads(output.getvalue())
            output_dir = Path(summary["output_dir"])
            manifest = json.loads((output_dir / "manifest.json").read_text())
            agents = self._read_csv(output_dir / "agents.csv")
            matches = self._read_csv(output_dir / "matches.csv")
            boop_matches = self._read_csv(output_dir / "boop_matches.csv")
            turns = self._read_csv(output_dir / "turns.csv")
            boops = self._read_csv(output_dir / "boops.csv")

            self.assertEqual(exit_code, 0)
            self.assertEqual(output_dir, root / "boop-study" / "data")
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["processed_matches"], 1)
            self.assertEqual(len(matches), 1)
            self.assertEqual(len(boop_matches), 1)
            self.assertEqual(boop_matches[0]["match_number"], matches[0]["match_number"])
            self.assertNotIn("win_by_cat_line", matches[0])
            self.assertIn("win_by_cat_line", boop_matches[0])
            self.assertEqual(len(agents), 2)
            self.assertEqual(len(turns), int(matches[0]["plies"]))
            self.assertEqual(manifest["row_counts"]["turns"], len(turns))
            self.assertEqual(manifest["row_counts"]["boops"], len(boops))
            self.assertTrue({turn["zone"] for turn in turns} <= {"center", "middle", "outer"})
            self.assertEqual(turns[0]["strategic_phase"], "all_kittens")
            self.assertEqual(turns[-1]["terminal_after"], "True")
            for filename in (
                "resolutions.csv",
                "winning_lines.csv",
            ):
                self.assertTrue((output_dir / filename).is_file())
            for filename in (
                "agents.csv",
                "matches.csv",
                "boop_matches.csv",
                "turns.csv",
                "boops.csv",
                "resolutions.csv",
                "winning_lines.csv",
            ):
                with Path(output_dir, filename).open(encoding="utf-8", newline="") as source:
                    header = next(csv.reader(source))
                self.assertEqual(len(header), len(set(header)), filename)

            errors = io.StringIO()
            with redirect_stderr(errors):
                repeated_exit = main(["extract", "--input", str(trace)])
            self.assertEqual(repeated_exit, 1)
            self.assertIn("--overwrite", errors.getvalue())
            with redirect_stdout(io.StringIO()):
                overwritten_exit = main(
                    ["extract", "--input", str(trace), "--overwrite"]
                )
            self.assertEqual(overwritten_exit, 0)

    def test_cli_extracts_analysis_tables_from_a_spotf_tournament(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = self._create_small_spotf_tournament(root)
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["extract", "--input", str(trace), "--json"])

            summary = json.loads(output.getvalue())
            output_dir = Path(summary["output_dir"])
            manifest = json.loads((output_dir / "manifest.json").read_text())
            agents = self._read_csv(output_dir / "agents.csv")
            matches = self._read_csv(output_dir / "matches.csv")
            spotf_matches = self._read_csv(output_dir / "spotf_matches.csv")
            actions = self._read_csv(output_dir / "actions.csv")
            player_turns = self._read_csv(output_dir / "player_turns.csv")
            tile_takes = self._read_csv(output_dir / "tile_takes.csv")
            gemstone_actions = self._read_csv(output_dir / "gemstone_actions.csv")
            categories = self._read_csv(output_dir / "categories.csv")

            self.assertEqual(exit_code, 0)
            self.assertTrue(summary["complete"])
            self.assertEqual(manifest["game"], "spotf")
            self.assertEqual(len(matches), 1)
            self.assertEqual(len(spotf_matches), 1)
            self.assertEqual(len(actions), int(matches[0]["plies"]))
            self.assertEqual(len(tile_takes), 48)
            self.assertEqual(len(categories), 24)
            self.assertLess(len(player_turns), len(actions))
            self.assertLessEqual(len(gemstone_actions), len(player_turns))
            self.assertEqual(actions[-1]["terminal_after"], "True")
            self.assertIn("time_budget", agents[0])
            self.assertIn("decision_seconds", actions[0])
            self.assertIn("search_iterations", actions[0])
            self.assertIn("search_nodes", actions[0])
            self.assertIn("p0_reachable_score_after", actions[0])
            self.assertIn("p0_categories_reachable_after", actions[0])
            self.assertIn("tile_progress_fraction", actions[0])
            self.assertIn("tile_quarter", actions[0])
            self.assertGreater(float(actions[0]["decision_seconds"]), 0.0)
            self.assertIn("search_iterations", player_turns[0])
            self.assertIn("reachable_score_delta", player_turns[0])
            self.assertIn("gemstones_removed_delta", player_turns[0])
            self.assertIn("game_quarter", tile_takes[0])
            self.assertIn("categories_reachable_delta", tile_takes[0])
            self.assertIn("absent_penalty", categories[0])
            self.assertIn("lost_majority", categories[0])
            self.assertEqual(
                int(actions[-1]["physical_turn"]),
                int(spotf_matches[0]["physical_turns"]),
            )
            self.assertEqual(
                {category["category_type"] for category in categories},
                {"spirit", "power_source"},
            )
            for filename in (
                "agents.csv",
                "matches.csv",
                "spotf_matches.csv",
                "actions.csv",
                "player_turns.csv",
                "tile_takes.csv",
                "gemstone_actions.csv",
                "categories.csv",
            ):
                with Path(output_dir, filename).open(encoding="utf-8", newline="") as source:
                    header = next(csv.reader(source))
                self.assertEqual(len(header), len(set(header)), filename)

    def test_cli_extract_combines_compatible_studies_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_trace = self._create_small_spotf_tournament(root)
            second_trace = root / "spotf-replica.jsonl"
            second_trace.write_text(first_trace.read_text(encoding="utf-8"), encoding="utf-8")
            output_dir = root / "combined" / "data"
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "extract",
                        "--input",
                        str(first_trace),
                        str(second_trace),
                        "--output-dir",
                        str(output_dir),
                        "--json",
                    ]
                )

            summary = json.loads(output.getvalue())
            manifest = json.loads((output_dir / "manifest.json").read_text())
            studies = self._read_csv(output_dir / "studies.csv")
            agents = self._read_csv(output_dir / "agents.csv")
            matches = self._read_csv(output_dir / "matches.csv")
            categories = self._read_csv(output_dir / "categories.csv")

            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["processed_matches"], 2)
            self.assertEqual(len(summary["inputs"]), 2)
            self.assertEqual(len(studies), 2)
            self.assertEqual(len(agents), 2)
            self.assertEqual([row["match_number"] for row in matches], ["1", "2"])
            self.assertEqual(
                [row["source_match_number"] for row in matches],
                ["1", "1"],
            )
            self.assertEqual(
                [row["pairing_number"] for row in matches],
                ["1", "2"],
            )
            self.assertEqual(
                {row["study_id"] for row in matches},
                {"spotf-study", "spotf-replica"},
            )
            self.assertEqual(
                {row["study_id"] for row in categories},
                {"spotf-study", "spotf-replica"},
            )
            self.assertIsNone(manifest["source"])
            self.assertEqual(len(manifest["sources"]), 2)
            self.assertEqual(manifest["analysis_schema_version"], 5)
            self.assertEqual(manifest["row_counts"]["studies"], 2)
            self.assertTrue((output_dir / "root_actions.csv").is_file())

    def test_cli_extract_rejects_conflicting_agent_names_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_trace = self._create_small_spotf_tournament(root)
            records = first_trace.read_text(encoding="utf-8").splitlines()
            header = json.loads(records[0])
            header["agents"][0] = {
                "name": "alpha",
                "type": "mcts",
                "iterations": 5000,
                "rollout_depth": 32,
                "exploration": 2**0.5,
                "heuristic": 0,
                "self_play": False,
            }
            conflicting_trace = root / "spotf-conflict.jsonl"
            conflicting_trace.write_text(
                "\n".join([json.dumps(header), *records[1:]]) + "\n",
                encoding="utf-8",
            )
            output_dir = root / "combined" / "data"
            errors = io.StringIO()
            with redirect_stderr(errors):
                exit_code = main(
                    [
                        "extract",
                        "--input",
                        str(first_trace),
                        str(conflicting_trace),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn('agent "alpha" has conflicting configurations', errors.getvalue())
            self.assertIn("use distinct agent names", errors.getvalue())
            self.assertFalse(output_dir.exists())

    def test_cli_extract_accepts_a_partial_trace_and_ignores_a_truncated_tail(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete_trace = self._create_small_boop_tournament(root)
            records = complete_trace.read_text(encoding="utf-8").splitlines()
            header = json.loads(records[0])
            header["total_matches"] = 2
            partial_trace = root / "partial.jsonl"
            partial_trace.write_text(
                json.dumps(header) + "\n" + records[1] + "\n" + '{"record_type"',
                encoding="utf-8",
            )

            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    ["extract", "--input", str(partial_trace), "--json"]
                )
            summary = json.loads(output.getvalue())
            manifest = json.loads(
                Path(summary["output_dir"], "manifest.json").read_text()
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["processed_matches"], 1)
            self.assertFalse(summary["complete"])
            self.assertTrue(summary["truncated_last_line"])
            self.assertFalse(manifest["complete"])

    def test_cli_extract_reports_games_without_an_analyzer_before_creating_output(
        self,
    ) -> None:
        for game in ("connect-four", "tic-tac-toe"):
            with self.subTest(game=game), tempfile.TemporaryDirectory() as directory:
                trace = Path(directory, f"{game}.jsonl")
                trace.write_text(
                    json.dumps(
                        {
                            "record_type": "tournament",
                            "schema_version": 1,
                            "game": game,
                            "total_matches": 0,
                            "agents": [],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                errors = io.StringIO()
                with redirect_stderr(errors):
                    exit_code = main(["extract", "--input", str(trace)])

                self.assertEqual(exit_code, 1)
                self.assertIn(
                    f"tournament analysis is not available for {game}",
                    errors.getvalue(),
                )
                self.assertFalse(Path(directory, game).exists())

    def test_cli_report_rejects_an_unimplemented_game_before_creating_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = root / "connect-four"
            data = study / "data"
            data.mkdir(parents=True)
            (data / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "game": "connect-four",
                    }
                ),
                encoding="utf-8",
            )
            errors = io.StringIO()
            with redirect_stderr(errors):
                exit_code = main(["report", "--input", str(data)])

            self.assertEqual(exit_code, 1)
            self.assertIn(
                "tournament report is not available for connect-four",
                errors.getvalue(),
            )
            self.assertFalse((study / "report").exists())

    @unittest.skipUnless(
        REPORT_DEPENDENCIES_AVAILABLE,
        "optional report dependencies are not installed",
    )
    def test_cli_generates_a_partial_boop_report_and_protects_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = self._create_small_boop_tournament(root)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["extract", "--input", str(trace)]), 0)
            study = root / "boop-study"
            data = study / "data"
            manifest_path = data / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["complete"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["report", "--input", str(data), "--json"])
            summary = json.loads(output.getvalue())
            report = study / "report"

            self.assertEqual(exit_code, 0)
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["figures"], 11)
            self.assertEqual(summary["tables"], 16)
            self.assertTrue((report / "index.html").is_file())
            self.assertTrue((report / "summary.json").is_file())
            self.assertEqual(len(list((report / "figures").glob("*.png"))), 11)
            self.assertEqual(len(list((report / "tables").glob("*.csv"))), 16)
            self.assertIn("Preliminary", (report / "index.html").read_text())

            errors = io.StringIO()
            with redirect_stderr(errors):
                repeated_exit = main(["report", "--input", str(data)])
            self.assertEqual(repeated_exit, 1)
            self.assertIn("--overwrite", errors.getvalue())

    @unittest.skipUnless(
        REPORT_DEPENDENCIES_AVAILABLE,
        "optional report dependencies are not installed",
    )
    def test_cli_generates_a_spotf_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = self._create_small_spotf_tournament(root)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["extract", "--input", str(trace)]), 0)
            study = root / "spotf-study"
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    ["report", "--input", str(study / "data"), "--json"]
                )

            summary = json.loads(output.getvalue())
            report = study / "report"
            report_summary = json.loads((report / "summary.json").read_text())
            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["game"], "spotf")
            self.assertFalse(report_summary["search_data_available"])
            self.assertEqual(summary["figures"], 12)
            self.assertEqual(summary["tables"], 15)
            self.assertEqual(len(list((report / "figures").glob("*.png"))), 12)
            self.assertEqual(len(list((report / "tables").glob("*.csv"))), 15)
            self.assertTrue((report / "tables" / "search_performance.csv").is_file())
            self.assertTrue((report / "tables" / "strategic_progress.csv").is_file())
            self.assertTrue((report / "tables" / "root_selection.csv").is_file())
            self.assertIn(
                "Spirits of the Forest tournament report",
                (report / "index.html").read_text(),
            )

    @unittest.skipUnless(
        REPORT_DEPENDENCIES_AVAILABLE,
        "optional report dependencies are not installed",
    )
    def test_spotf_search_analytics_compare_actual_compute_cost(self) -> None:
        import pandas as pd

        from meeple_bots.games.spirits_of_the_forest.reporting import (
            _search_by_phase,
            _search_performance,
        )

        actions = pd.DataFrame(
            [
                {
                    "agent": "timed",
                    "decision_seconds": 0.5,
                    "search_iterations": 100,
                    "search_nodes": 40,
                    "legal_actions_before": 5,
                    "game_quarter": "q1",
                    "tile_quarter": "q1",
                    "phase_before": "collect",
                },
                {
                    "agent": "timed",
                    "decision_seconds": 0.5,
                    "search_iterations": 200,
                    "search_nodes": 80,
                    "legal_actions_before": 3,
                    "game_quarter": "q4",
                    "tile_quarter": "q4",
                    "phase_before": "collect",
                },
            ]
        )
        agents = pd.DataFrame(
            [{"agent_name": "timed", "time_budget": 0.5}]
        )

        performance = _search_performance(actions, agents).iloc[0]
        by_phase = _search_by_phase(actions)
        self.assertEqual(performance["measured_decisions"], 2)
        self.assertAlmostEqual(performance["iterations_per_second"], 300.0)
        self.assertAlmostEqual(
            performance["milliseconds_per_iteration"], 10.0 / 3.0
        )
        self.assertAlmostEqual(performance["nodes_per_iteration"], 0.4)
        self.assertAlmostEqual(performance["mean_budget_utilization"], 1.0)
        self.assertEqual(set(by_phase["game_quarter"]), {"q1", "q4"})
        self.assertTrue((by_phase["milliseconds_per_iteration"] > 0).all())

    @unittest.skipUnless(
        REPORT_DEPENDENCIES_AVAILABLE,
        "optional report dependencies are not installed",
    )
    def test_zone_density_accounts_for_different_zone_sizes(self) -> None:
        import pandas as pd

        from meeple_bots.games.boop.reporting import _placement_cells, _zone_rates

        turns = pd.DataFrame(
            {
                "match_number": [1, 1],
                "agent": ["alpha", "alpha"],
                "zone": ["center", "outer"],
            }
        )
        rates = _zone_rates(turns, ["match_number", "agent"]).set_index("zone")

        self.assertEqual(rates.loc["center", "move_share"], 0.5)
        self.assertEqual(rates.loc["outer", "move_share"], 0.5)
        self.assertEqual(rates.loc["center", "density_per_cell"], 0.125)
        self.assertEqual(rates.loc["outer", "density_per_cell"], 0.025)

        placements = _placement_cells(
            pd.DataFrame(
                {
                    "match_number": [1, 2, 2, 2],
                    "row": [2, 0, 0, 0],
                    "column": [2, 0, 0, 0],
                }
            )
        ).set_index(["row", "column"])
        self.assertEqual(placements.loc[(2, 2), "placement_share"], 0.5)
        self.assertEqual(placements.loc[(0, 0), "placement_share"], 0.5)

    @staticmethod
    def _create_small_boop_tournament(root: Path) -> Path:
        config = root / "boop-tournament.toml"
        trace = root / "boop-study.jsonl"
        config.write_text(
            "\n".join(
                [
                    'game = "boop"',
                    'output = "boop-study.jsonl"',
                    "matches_per_pair = 1",
                    "seed = 91",
                    "max_plies = 1000",
                    "",
                    "[[agents]]",
                    'name = "alpha"',
                    'kind = "random"',
                    "",
                    "[[agents]]",
                    'name = "beta"',
                    'kind = "random"',
                ]
            ),
            encoding="utf-8",
        )
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            exit_code = main(["tournament", "--config", str(config)])
        if exit_code != 0:
            raise AssertionError("failed to create the boop extraction fixture")
        return trace

    @staticmethod
    def _create_small_spotf_tournament(root: Path) -> Path:
        config = root / "spotf-tournament.toml"
        trace = root / "spotf-study.jsonl"
        config.write_text(
            "\n".join(
                [
                    'game = "spotf"',
                    'output = "spotf-study.jsonl"',
                    "matches_per_pair = 1",
                    "seed = 91",
                    "max_plies = 1000",
                    "workers = 1",
                    "",
                    "[[agents]]",
                    'name = "alpha"',
                    'kind = "random"',
                    "",
                    "[[agents]]",
                    'name = "beta"',
                    'kind = "random"',
                ]
            ),
            encoding="utf-8",
        )
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            exit_code = main(["tournament", "--config", str(config)])
        if exit_code != 0:
            raise AssertionError("failed to create the spotf extraction fixture")
        return trace

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as source:
            return list(csv.DictReader(source))


if __name__ == "__main__":
    unittest.main()
