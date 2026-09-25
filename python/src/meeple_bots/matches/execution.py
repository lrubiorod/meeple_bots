"""Match and batch orchestration over the native match engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from .. import _native
from .._agent_config import MctsAgent, RandomAgent, SoIsmctsAgent, _positive_u32
from .._concurrency import WorkerSetting, ordered_parallel_map, resolve_workers
from ..connect6 import Connect6, Connect6Action
from ..game_config import game_parameters
from ..game_types import (
    Boop, BoopPiece, BoopPieceKind, BoopPool, ConnectFour, ConnectFourAction,
    Game, GameAction, MatchMoveObservation, MatchMoveObserver,
    SpiritsOfTheForest, TicTacToe, TicTacToeAction,
)
from ..lost_cities import LostCities, LostCitiesAction, LostCitiesChanceEvent, LostCitiesState
from ..native_bridge import (
    _action_from_native, _board_rows, _boop_action_from_selector,
    _final_board_from_native, _gemstone_pools_from_native, _native_game,
    _pools_from_native, _spirit_collections_from_native,
    _spirits_action_from_native, _spirits_state_from_native,
    _validate_agent_evaluators,
)
from ..splendor import Splendor, SplendorChanceOutcome, ChanceEvent, SplendorState
from .human import Agent, HumanAgent, _native_agent
from .models import (
    BatchMatchResult, BatchProgress, BatchProgressCallback, BatchProgressStatus,
    BatchResult, MatchResult, Move, RootActionDiagnostic, TreeReuseDiagnostic,
    _BatchJob, _BatchMatchOutcome,
)

_MAX_U64 = 2**64 - 1

@dataclass(frozen=True, slots=True)
class Match:
    """Configuration for one match executed by the Rust engine."""

    game: Game = field(default_factory=TicTacToe)
    first: Agent = field(default_factory=MctsAgent)
    second: Agent = field(default_factory=RandomAgent)
    seed: int = 0
    max_plies: int = 10_000
    observe_move: MatchMoveObserver | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.game, (TicTacToe, ConnectFour, Boop, SpiritsOfTheForest, Splendor, LostCities, Connect6)):
            raise TypeError(
                "game must be TicTacToe, ConnectFour, Connect6, Boop, SpiritsOfTheForest, Splendor, or LostCities"
            )
        if not isinstance(self.first, (RandomAgent, SoIsmctsAgent, MctsAgent, HumanAgent)):
            raise TypeError("first must be SoIsmctsAgent, RandomAgent, MctsAgent, or HumanAgent")
        if not isinstance(self.second, (RandomAgent, SoIsmctsAgent, MctsAgent, HumanAgent)):
            raise TypeError("second must be SoIsmctsAgent, RandomAgent, MctsAgent, or HumanAgent")
        _validate_agent_evaluators(self.game, self.first)
        _validate_agent_evaluators(self.game, self.second)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= self.seed <= _MAX_U64:
            raise ValueError(f"seed must be between 0 and {_MAX_U64}")
        _positive_u32("max_plies", self.max_plies)
        if self.observe_move is not None:
            if not callable(self.observe_move):
                raise TypeError("observe_move must be callable")

    def run(self) -> MatchResult:
        """Execute the match and return its complete immutable report."""

        raw = _native.run_match(
            _native_game(self.game),
            _native_agent(self.first, self.game),
            _native_agent(self.second, self.game),
            self.seed,
            self.max_plies,
            _match_move_observer(self.observe_move, self.game),
            game_params=game_parameters(self.game),
        )
        moves = tuple(
            Move(
                player=item["player"],
                action=_action_from_native(item["action"]),
                decision_seconds=item.get("decision_seconds", 0.0),
                selection_seconds=item.get("selection_seconds"),
                maintenance_seconds=item.get("maintenance_seconds"),
                search_iterations=item.get("search_iterations"),
                search_nodes=item.get("search_nodes"),
                terminal_simulations=item.get("terminal_simulations"),
                cutoff_simulations=item.get("cutoff_simulations"),
                root_actions=tuple(
                    RootActionDiagnostic(
                        action_index=root["action_index"],
                        visits=root["visits"],
                        mean_utility=root["mean_utility"],
                        heuristic_value=root.get("heuristic_value"),
                        progressive_bias=root.get("progressive_bias"),
                        selected=root["selected"],
                    )
                    for root in item.get("root_actions", ())
                ),
                tree_reuse=(
                    None
                    if item.get("tree_reuse") is None
                    else TreeReuseDiagnostic(**item["tree_reuse"])
                ),
            )
            for item in raw["moves"]
        )
        return MatchResult(
            unassigned_maintenance_seconds=tuple(
                raw.get("unassigned_maintenance_seconds", (0.0, 0.0))
            ),
            game_params=game_parameters(self.game),
            seed=raw["seed"],
            plies=raw["plies"],
            utilities=tuple(raw["utilities"]),
            winner=raw["winner"],
            moves=moves,
            final_board=_final_board_from_native(
                raw["spirit_forest"]
                if isinstance(self.game, SpiritsOfTheForest)
                else raw["final_board"],
                self.game,
            ),
            pools=_pools_from_native(raw["pools"]),
            spirit_collections=_spirit_collections_from_native(
                raw["spirit_collections"]
            ),
            gemstone_pools=_gemstone_pools_from_native(raw["gemstone_pools"]),
            scores=None if raw["scores"] is None else tuple(raw["scores"]),
            chance_events=tuple(
                (LostCitiesChanceEvent(event["after_ply"], LostCitiesAction.from_dict(event["outcome"]))
                 if isinstance(self.game, LostCities) else ChanceEvent(event["after_ply"], SplendorChanceOutcome(event["outcome"]["card"])))
                for event in raw.get("chance_events", ())
            ),
            lost_cities_state=(LostCitiesState.from_dict(raw["lost_cities_state"]) if "lost_cities_state" in raw else None),
            splendor_state=(SplendorState.from_dict(raw["splendor_state"])
                            if "splendor_state" in raw else None),
        )


@dataclass(frozen=True, slots=True)
class Batch:
    """A reproducible series of automated matches between two participants."""

    game: Game = field(default_factory=TicTacToe)
    agent_a: RandomAgent | SoIsmctsAgent | MctsAgent = field(default_factory=RandomAgent)
    agent_b: RandomAgent | SoIsmctsAgent | MctsAgent = field(default_factory=MctsAgent)
    matches: int = 20
    seed: int = 0
    max_plies: int = 10_000
    alternate_sides: bool = True
    workers: WorkerSetting = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.game, (TicTacToe, ConnectFour, Boop, SpiritsOfTheForest, Splendor, LostCities, Connect6)):
            raise TypeError(
                "game must be TicTacToe, ConnectFour, Connect6, Boop, SpiritsOfTheForest, Splendor, or LostCities"
            )
        for name, agent in (("agent_a", self.agent_a), ("agent_b", self.agent_b)):
            if not isinstance(agent, (RandomAgent, SoIsmctsAgent, MctsAgent)):
                raise TypeError(f"{name} must be SoIsmctsAgent, RandomAgent or MctsAgent")
            _validate_agent_evaluators(self.game, agent)
        _positive_u32("matches", self.matches)
        _positive_u32("max_plies", self.max_plies)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= self.seed <= _MAX_U64:
            raise ValueError(f"seed must be between 0 and {_MAX_U64}")
        if not isinstance(self.alternate_sides, bool):
            raise TypeError("alternate_sides must be a boolean")
        resolve_workers(self.workers)

    def run(
        self,
        progress: BatchProgressCallback | None = None,
    ) -> BatchResult:
        """Run matches concurrently and optionally report submission and completion."""

        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable")

        batch_started = perf_counter()
        games = []
        agent_a_wins = 0
        agent_b_wins = 0
        draws = 0
        worker_count = min(resolve_workers(self.workers), self.matches)
        jobs = (
            _BatchJob(
                match_number=offset + 1,
                seed=(self.seed + offset) & _MAX_U64,
                agent_a_player=offset % 2 if self.alternate_sides else 0,
            )
            for offset in range(self.matches)
        )

        def notify_started(job: _BatchJob) -> None:
            if progress is not None:
                progress(
                    BatchProgress(
                        status=BatchProgressStatus.STARTED,
                        match_number=job.match_number,
                        total_matches=self.matches,
                        seed=job.seed,
                        agent_a_player=job.agent_a_player,
                        elapsed_seconds=perf_counter() - batch_started,
                    )
                )

        for job, outcome in ordered_parallel_map(
            self._run_job,
            jobs,
            worker_count,
            notify_started,
        ):
            game_result = outcome.summary
            if game_result.winner is None:
                draws += 1
            elif game_result.winner == 0:
                agent_a_wins += 1
            else:
                agent_b_wins += 1
            games.append(game_result)
            if progress is not None:
                progress(
                    BatchProgress(
                        status=BatchProgressStatus.COMPLETED,
                        match_number=job.match_number,
                        total_matches=self.matches,
                        seed=job.seed,
                        agent_a_player=job.agent_a_player,
                        elapsed_seconds=perf_counter() - batch_started,
                        result=game_result,
                        match_result=outcome.match_result,
                    )
                )

        elapsed_seconds = perf_counter() - batch_started
        total_plies = sum(game.plies for game in games)
        return BatchResult(
            seed=self.seed,
            matches=self.matches,
            workers=worker_count,
            alternate_sides=self.alternate_sides,
            agent_a_wins=agent_a_wins,
            agent_b_wins=agent_b_wins,
            draws=draws,
            total_plies=total_plies,
            average_plies=total_plies / self.matches,
            elapsed_seconds=elapsed_seconds,
            games=tuple(games),
        )

    def _run_job(self, job: _BatchJob) -> _BatchMatchOutcome:
        match_started = perf_counter()
        first, second = (
            (self.agent_a, self.agent_b)
            if job.agent_a_player == 0
            else (self.agent_b, self.agent_a)
        )
        match = Match(
            game=self.game,
            first=first,
            second=second,
            seed=job.seed,
            max_plies=self.max_plies,
        ).run()
        if match.winner is None:
            winner = None
        elif match.winner == job.agent_a_player:
            winner = 0
        else:
            winner = 1
        return _BatchMatchOutcome(
            summary=BatchMatchResult(
                match_number=job.match_number,
                seed=job.seed,
                agent_a_player=job.agent_a_player,
                winner=winner,
                plies=match.plies,
                utilities=(
                    match.utilities[job.agent_a_player],
                    match.utilities[1 - job.agent_a_player],
                ),
                duration_seconds=perf_counter() - match_started,
            ),
            match_result=match,
        )


def _match_move_observer(observer: MatchMoveObserver | None, game: Game):
    if observer is None:
        return None

    if isinstance(game, Boop):
        def observe_boop(
            player: int,
            flat_board,
            native_pools,
            native_action,
            decision_seconds: float,
            search_iterations: int | None,
            search_nodes: int | None,
        ) -> None:
            board = _board_rows(
                [
                    None
                    if piece is None
                    else BoopPiece(player=piece[0], kind=BoopPieceKind(piece[1]))
                    for piece in flat_board
                ],
                columns=6,
            )
            pools = tuple(
                BoopPool(kittens=kittens, cats=cats)
                for kittens, cats in native_pools
            )
            observer(
                MatchMoveObservation(
                    game=game,
                    player=player,
                    action=_boop_action_from_selector(native_action),
                    board=board,
                    decision_seconds=decision_seconds,
                    search_iterations=search_iterations,
                    search_nodes=search_nodes,
                    pools=pools,
                )
            )

        return observe_boop

    if isinstance(game, SpiritsOfTheForest):
        def observe_spirits(
            player: int,
            native_state,
            native_action,
            decision_seconds: float,
            search_iterations: int | None,
            search_nodes: int | None,
        ) -> None:
            board, collections, gems, phase, active, scores = _spirits_state_from_native(
                native_state
            )
            observer(
                MatchMoveObservation(
                    game=game,
                    player=player,
                    action=_spirits_action_from_native(native_action),
                    board=board,
                    decision_seconds=decision_seconds,
                    search_iterations=search_iterations,
                    search_nodes=search_nodes,
                    spirit_collections=collections,
                    gemstone_pools=gems,
                    scores=scores,
                    phase=phase,
                    active_player=active,
                )
            )

        return observe_spirits

    def observe(
        player: int,
        flat_board,
        native_action,
        decision_seconds: float,
        search_iterations: int | None,
        search_nodes: int | None,
    ) -> None:
        if isinstance(game, TicTacToe):
            action: GameAction = TicTacToeAction(
                row=native_action[0],
                column=native_action[1],
            )
            board = _board_rows(flat_board, columns=3)
        elif isinstance(game, Connect6):
            action = Connect6Action(native_action)
            board = _board_rows(flat_board, columns=game.board_size)
        elif isinstance(game, ConnectFour):
            action = ConnectFourAction(column=native_action)
            board = _board_rows(flat_board, columns=7)
        observer(
            MatchMoveObservation(
                game=game,
                player=player,
                action=action,
                board=board,
                decision_seconds=decision_seconds,
                search_iterations=search_iterations,
                search_nodes=search_nodes,
            )
        )

    return observe
