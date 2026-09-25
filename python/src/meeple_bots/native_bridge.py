"""Convert Python configuration and traces to Rust binding inputs."""

from __future__ import annotations

from . import _native
from .native_config import (
    _native_evaluator, _native_progressive_bias, _native_rollout_policy,
    _native_base_rollout_policy, native_agent_config,
)
from ._capabilities import game_search_capabilities
from ._agent_config import _non_negative_u32
from ._agent_config import (
    ConditionalRollout, GameHeuristic, Mast, MctsAgent, RandomAgent,
    RolloutPolicy, SoIsmctsAgent, StateEvaluator, UniformRandom,
)
from .connect6 import Connect6, Connect6Action
from .game_config import game_parameters
from .game_types import (
    BoopPieceKind, BoopPiece, BoopPool, BoopPosition, ForestPosition,
    Spirit, PowerSource, SpiritsTurnPhase, SpiritTile, SpiritCollection,
    SpiritGemstonePool, SpiritGemstoneSacrifice, SpiritsOfTheForestAction,
    BoopResolution, GameAction, GameBoard, BoardCell,
    Boop, BoopAction, BoopGraduateLine, BoopRecoverPiece, ConnectFour,
    ConnectFourAction, EndSpiritCollection, Game, MoveSpiritGemstone,
    PlaceSpiritGemstone, SkipSpiritGemstone, SpiritsOfTheForest,
    TakeSpiritTile, TicTacToe, TicTacToeAction,
)
from .lost_cities import LostCities, LostCitiesAction
from .matches.models import Move
from .splendor import Splendor, SplendorAction

def _native_game(game: Game) -> str:
    if isinstance(game, Connect6):
        return "connect6"
    if isinstance(game, LostCities):
        return "lost_cities"
    if isinstance(game, Splendor):
        return "splendor"
    if isinstance(game, TicTacToe):
        return "tic_tac_toe"
    if isinstance(game, ConnectFour):
        return "connect_four"
    if isinstance(game, SpiritsOfTheForest):
        return "spotf"
    return "boop"

def _analyze_trace(game: Game, moves: tuple[Move, ...], *, seed: int = 0):
    """Dispatch a completed trace to the selected game's native analyzer."""

    native_moves = []
    if isinstance(game, Connect6):
        native_moves = [(m.player, m.action.position) for m in moves]
    elif isinstance(game, Boop):
        for move in moves:
            if not isinstance(move.action, BoopAction):
                raise TypeError("boop trace contains a non-boop action")
            resolution = move.action.resolution
            if isinstance(resolution, BoopGraduateLine):
                native_resolution = (
                    "graduate",
                    [(position.row, position.column) for position in resolution.positions],
                )
            elif isinstance(resolution, BoopRecoverPiece):
                native_resolution = (
                    "recover",
                    [(resolution.position.row, resolution.position.column)],
                )
            else:
                native_resolution = ("none", [])
            native_moves.append(
                (
                    move.player,
                    (
                        move.action.piece.value,
                        move.action.row,
                        move.action.column,
                        native_resolution,
                    ),
                )
            )
    elif isinstance(game, SpiritsOfTheForest):
        for move in moves:
            action = move.action
            if isinstance(action, TakeSpiritTile):
                sacrifice = None
                if action.sacrifice is not None:
                    source = action.sacrifice.source
                    sacrifice = (
                        "available" if source is None else "forest",
                        None if source is None else (source.row, source.column),
                    )
                native_action = (
                    "take_tile",
                    [(action.position.row, action.position.column)],
                    sacrifice,
                )
            elif isinstance(action, EndSpiritCollection):
                native_action = ("end_collection", [], None)
            elif isinstance(action, PlaceSpiritGemstone):
                native_action = (
                    "place_gemstone",
                    [(action.target.row, action.target.column)],
                    None,
                )
            elif isinstance(action, MoveSpiritGemstone):
                native_action = (
                    "move_gemstone",
                    [
                        (action.source.row, action.source.column),
                        (action.target.row, action.target.column),
                    ],
                    None,
                )
            elif isinstance(action, SkipSpiritGemstone):
                native_action = ("skip_gemstone", [], None)
            else:
                raise TypeError("spotf trace contains a non-spotf action")
            native_moves.append((move.player, native_action))
    elif isinstance(game, ConnectFour):
        for move in moves:
            if not isinstance(move.action, ConnectFourAction):
                raise TypeError("connect-four trace contains a non-connect-four action")
            native_moves.append((move.player, move.action.column))
    elif isinstance(game, TicTacToe):
        for move in moves:
            if not isinstance(move.action, TicTacToeAction):
                raise TypeError("tic-tac-toe trace contains a non-tic-tac-toe action")
            native_moves.append((move.player, (move.action.row, move.action.column)))
    return _native.analyze_trace(_native_game(game), native_moves, seed, game_params=game_parameters(game))


def _action_from_native(raw: dict[str, object]) -> GameAction:
    if raw["type"] == "connect6":
        return Connect6Action(raw["position"])
    if raw["type"] == "lost_cities":
        return LostCitiesAction.from_dict(raw)
    if raw["type"] == "splendor":
        return SplendorAction.from_dict(raw)
    if raw["type"] == "tic_tac_toe":
        return TicTacToeAction(row=raw["row"], column=raw["column"])
    if raw["type"] == "connect_four":
        return ConnectFourAction(column=raw["column"])
    if raw["type"] in ("spotf", "spirits_of_the_forest"):
        return _spirits_action_from_mapping(raw)
    return BoopAction(
        piece=BoopPieceKind(raw["piece"]),
        row=raw["row"],
        column=raw["column"],
        resolution=_boop_resolution_from_native(raw["resolution"]),
    )


def _validate_agent_evaluators(game: Game, agent: object) -> None:
    if isinstance(agent, SoIsmctsAgent) and not isinstance(game, LostCities):
        raise ValueError("SO-ISMCTS is only supported by Lost Cities")
    if isinstance(game, LostCities) and not isinstance(agent, (RandomAgent, SoIsmctsAgent)):
        raise ValueError("Lost Cities has imperfect information: standard MCTS is not compatible; use SoIsmctsAgent or RandomAgent")
    if isinstance(agent, MctsAgent):
        if isinstance(game, Splendor) and agent.selection_policy == "uct_rave":
            raise ValueError("UCT-RAVE is only supported by deterministic MCTS")
        if isinstance(agent.rollout_policy, ConditionalRollout):
            if not game_search_capabilities(_game_display_name(game))["turn_phase_conditions"]:
                raise ValueError(
                    "turn-phase rollout conditions are only supported by spotf"
                )
        _validate_game_evaluator(game, agent.cutoff_evaluator)
        for rollout_evaluator in _rollout_evaluators(agent.rollout_policy):
            _validate_game_evaluator(game, rollout_evaluator)
        if agent.progressive_bias is not None:
            if (
                agent.progressive_bias.condition is not None
                and not game_search_capabilities(_game_display_name(game))["turn_phase_conditions"]
            ):
                raise ValueError(
                    "turn-phase progressive bias conditions are only supported by spotf"
                )
            _validate_game_evaluator(game, agent.progressive_bias.evaluator)


def _rollout_evaluator(policy: RolloutPolicy) -> StateEvaluator | None:
    if isinstance(policy, ConditionalRollout):
        return _rollout_evaluator(policy.primary)
    if isinstance(policy, (UniformRandom, Mast)):
        return None
    return policy.evaluator


def _rollout_evaluators(policy: RolloutPolicy) -> tuple[StateEvaluator, ...]:
    if isinstance(policy, ConditionalRollout):
        return tuple(
            evaluator
            for branch in (policy.primary, policy.fallback)
            if (evaluator := _rollout_evaluator(branch)) is not None
        )
    evaluator = _rollout_evaluator(policy)
    return () if evaluator is None else (evaluator,)


def _validate_game_evaluator(game: Game, evaluator: StateEvaluator | None) -> None:
    if isinstance(evaluator, GameHeuristic):
        _validate_game_heuristic(game, evaluator.index)
        _validate_game_heuristic_params(game, evaluator)


def _validate_game_heuristic_params(game: Game, evaluator: GameHeuristic) -> None:
    allowed = game_search_capabilities(_game_display_name(game))["heuristics"][evaluator.index]
    for name, value in evaluator.params.items():
        if name not in allowed:
            raise ValueError(
                f"{_game_display_name(game)} heuristic {evaluator.index} "
                f"does not accept parameter {name!r}"
            )
        minimum, maximum = allowed[name]["minimum"], allowed[name]["maximum"]
        if minimum is not None and value < minimum:
            raise ValueError(
                f"{_game_display_name(game)} heuristic {evaluator.index} "
                f"parameter {name!r} must be at least {minimum:g}"
            )
        if maximum is not None and value > maximum:
            raise ValueError(
                f"{_game_display_name(game)} heuristic {evaluator.index} "
                f"parameter {name!r} must be at most {maximum:g}"
            )


def _validate_game_heuristic(game: Game, heuristic: int | None) -> None:
    if heuristic is None:
        return
    _non_negative_u32("heuristic", heuristic)
    indices = tuple(game_search_capabilities(_game_display_name(game))["heuristics"])
    if not indices:
        raise ValueError(f"{_game_display_name(game)} does not provide MCTS heuristics")
    if heuristic not in indices:
        available = (f"available index: {indices[0]}" if len(indices) == 1
                     else f"available indices: {indices[0]}..{indices[-1]}")
        raise ValueError(
            f"{_game_display_name(game)} does not provide MCTS heuristic {heuristic}; {available}"
        )


def _game_display_name(game: Game) -> str:
    if isinstance(game, Connect6):
        return "connect6"
    if isinstance(game, LostCities):
        return "lost_cities"
    if isinstance(game, Splendor):
        return "splendor"
    if isinstance(game, TicTacToe):
        return "tic-tac-toe"
    if isinstance(game, ConnectFour):
        return "connect-four"
    if isinstance(game, SpiritsOfTheForest):
        return "spotf"
    return "boop"


def _board_rows(flat_board: list[BoardCell], columns: int) -> GameBoard:
    return tuple(
        tuple(flat_board[start : start + columns])
        for start in range(0, len(flat_board), columns)
    )


def _boop_action_from_selector(raw) -> BoopAction:
    piece, row, column, raw_resolution = raw
    resolution_type, positions = raw_resolution
    if resolution_type == "graduate":
        resolution: BoopResolution = BoopGraduateLine(
            tuple(BoopPosition(row, column) for row, column in positions)
        )
    elif resolution_type == "recover":
        resolution = BoopRecoverPiece(BoopPosition(*positions[0]))
    else:
        resolution = None
    return BoopAction(BoopPieceKind(piece), row, column, resolution)


def _boop_resolution_from_native(raw) -> BoopResolution:
    if raw["type"] == "graduate":
        return BoopGraduateLine(
            tuple(BoopPosition(row, column) for row, column in raw["positions"])
        )
    if raw["type"] == "recover":
        return BoopRecoverPiece(BoopPosition(raw["row"], raw["column"]))
    return None


def _spirits_action_from_native(raw) -> SpiritsOfTheForestAction:
    kind, positions, sacrifice = raw
    if kind == "take_tile":
        row, column = positions[0]
        parsed_sacrifice = None
        if sacrifice is not None:
            sacrifice_kind, source = sacrifice
            parsed_sacrifice = SpiritGemstoneSacrifice(
                None if sacrifice_kind == "available" else ForestPosition(*source)
            )
        return TakeSpiritTile(ForestPosition(row, column), parsed_sacrifice)
    if kind == "end_collection":
        return EndSpiritCollection()
    if kind == "place_gemstone":
        return PlaceSpiritGemstone(ForestPosition(*positions[0]))
    if kind == "move_gemstone":
        return MoveSpiritGemstone(
            ForestPosition(*positions[0]), ForestPosition(*positions[1])
        )
    if kind == "skip_gemstone":
        return SkipSpiritGemstone()
    raise ValueError(f"unknown Spirits of the Forest action: {kind}")


def _spirits_action_from_mapping(raw: dict[str, object]) -> SpiritsOfTheForestAction:
    kind = raw["kind"]
    if kind == "take_tile":
        sacrifice = raw["sacrifice"]
        parsed = None
        if sacrifice is not None:
            parsed = SpiritGemstoneSacrifice(
                None
                if sacrifice["kind"] == "available"
                else ForestPosition(sacrifice["row"], sacrifice["column"])
            )
        return TakeSpiritTile(
            ForestPosition(raw["row"], raw["column"]),
            parsed,
        )
    if kind == "end_collection":
        return EndSpiritCollection()
    if kind == "place_gemstone":
        return PlaceSpiritGemstone(ForestPosition(raw["row"], raw["column"]))
    if kind == "move_gemstone":
        return MoveSpiritGemstone(
            ForestPosition(raw["source_row"], raw["source_column"]),
            ForestPosition(raw["target_row"], raw["target_column"]),
        )
    return SkipSpiritGemstone()


def _spirits_state_from_native(raw):
    flat_forest, raw_collections, raw_gems, raw_phase, active, raw_scores = raw
    board = _board_rows(
        [
            None
            if tile is None
            else SpiritTile(
                spirit=Spirit(tile[0]),
                spirit_symbols=tile[1],
                power_source=None if tile[2] is None else PowerSource(tile[2]),
                gemstone=tile[3],
            )
            for tile in flat_forest
        ],
        columns=12,
    )
    collections = tuple(
        SpiritCollection(tuple(spirits), tuple(sources), tiles)
        for spirits, sources, tiles in raw_collections
    )
    gems = tuple(
        SpiritGemstonePool(available, placed, removed)
        for available, placed, removed in raw_gems
    )
    return (
        board,
        collections,
        gems,
        SpiritsTurnPhase(raw_phase),
        active,
        tuple(raw_scores),
    )


def _initial_spirits_state(seed: int):
    return _spirits_state_from_native(_native.spirits_initial_state(seed))


def _final_board_from_native(flat_board, game: Game) -> GameBoard:
    if isinstance(game, Connect6):
        return _board_rows([None if p is None else p["player"] for p in flat_board], columns=game.board_size)
    if isinstance(game, (Splendor, LostCities)):
        return ()
    if isinstance(game, SpiritsOfTheForest):
        return _board_rows(
            [
                None
                if tile is None
                else SpiritTile(
                    spirit=Spirit(tile["spirit"]),
                    spirit_symbols=tile["spirit_symbols"],
                    power_source=(
                        None
                        if tile["power_source"] is None
                        else PowerSource(tile["power_source"])
                    ),
                    gemstone=tile["gemstone"],
                )
                for tile in flat_board
            ],
            columns=12,
        )
    cells: list[BoardCell] = []
    for piece in flat_board:
        if piece is None:
            cells.append(None)
        elif piece["kind"] == "token":
            cells.append(piece["player"])
        else:
            cells.append(
                BoopPiece(
                    player=piece["player"],
                    kind=BoopPieceKind(piece["kind"]),
                )
            )
    columns = 3 if isinstance(game, TicTacToe) else 7 if isinstance(game, ConnectFour) else 6
    return _board_rows(cells, columns)


def _pools_from_native(raw) -> tuple[BoopPool, BoopPool] | None:
    if raw is None:
        return None
    pools = tuple(BoopPool(kittens=pool["kittens"], cats=pool["cats"]) for pool in raw)
    return pools


def _spirit_collections_from_native(raw):
    if raw is None:
        return None
    return tuple(
        SpiritCollection(
            tuple(collection["spirit_symbols"]),
            tuple(collection["power_sources"]),
            collection["tiles"],
        )
        for collection in raw
    )


def _gemstone_pools_from_native(raw):
    if raw is None:
        return None
    return tuple(
        SpiritGemstonePool(pool["available"], pool["placed"], pool["removed"])
        for pool in raw
    )
