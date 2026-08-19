"""Statistical tables and figures for extracted boop tournaments."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import wilson_interval


_REQUIRED_TABLES = (
    "agents",
    "matches",
    "boop_matches",
    "turns",
    "boops",
    "resolutions",
    "winning_lines",
)
_ZONE_AREAS = {"center": 4, "middle": 12, "outer": 20}
_ZONE_ORDER = list(_ZONE_AREAS)
_BOOTSTRAP_SAMPLES = 2_000
_BOOTSTRAP_SEED = 0


def generate_boop_report(
    input_dir: Path,
    output_dir: Path,
    manifest: dict[str, object],
) -> dict[str, object]:
    """Build all Boop-specific report artifacts in an empty output directory."""

    tables = _load_tables(input_dir, manifest)
    derived = _derive_tables(tables)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True)
    tables_dir.mkdir(parents=True)

    for name, table in derived.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)

    sns.set_theme(style="whitegrid", context="notebook")
    figure_descriptions = _generate_figures(tables, derived, figures_dir)
    summary = _summary(manifest, tables, derived, figure_descriptions)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "index.html").write_text(
        _render_html(input_dir, manifest, derived, summary, figure_descriptions),
        encoding="utf-8",
    )
    return summary


def _load_tables(input_dir: Path, manifest: dict[str, object]) -> dict[str, pd.DataFrame]:
    manifest_tables = manifest.get("tables")
    if not isinstance(manifest_tables, dict):
        raise TypeError("extraction manifest tables must be an object")
    row_counts = manifest.get("row_counts")
    if not isinstance(row_counts, dict):
        raise TypeError("extraction manifest row_counts must be an object")

    tables: dict[str, pd.DataFrame] = {}
    for name in _REQUIRED_TABLES:
        filename = manifest_tables.get(name)
        if not isinstance(filename, str):
            raise ValueError(f"extraction manifest does not define table {name}")
        path = input_dir / filename
        try:
            table = pd.read_csv(path)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"extraction table not found: {path}") from error
        expected_rows = row_counts.get(name)
        if not isinstance(expected_rows, int):
            raise TypeError(f"extraction row count for {name} must be an integer")
        if len(table) != expected_rows:
            raise ValueError(
                f"extraction table {name} has {len(table)} rows; expected {expected_rows}"
            )
        tables[name] = table
    if tables["matches"].empty:
        raise ValueError("cannot generate a report without completed matches")
    return tables


def _derive_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    matches = tables["matches"].copy()
    agents = tables["agents"]["agent_name"].astype(str).tolist()
    matches["self_play"] = _boolean(matches["self_play"])
    boop_matches = tables["boop_matches"].copy()
    boop_matches["win_by_cat_line"] = _boolean(boop_matches["win_by_cat_line"])
    boop_matches["win_by_eight_cats"] = _boolean(boop_matches["win_by_eight_cats"])

    derived = {
        "agent_performance": _agent_performance(matches, agents),
        "pairwise_performance": _pairwise_performance(matches, agents),
        "first_player_advantage": _first_player_advantage(matches),
        "game_lengths": _game_lengths(matches),
        "placement_cells": _placement_cells(tables["turns"]),
        "placement_cells_by_agent": _placement_cells(tables["turns"], by_agent=True),
        "zone_strategy": _zone_strategy(tables["turns"]),
        "zone_by_outcome": _zone_by_outcome(tables["turns"]),
        "zone_progress": _zone_progress(tables["turns"]),
        "cat_progress": _cat_progress(tables["turns"], matches),
        "first_graduation": _first_graduation(boop_matches, matches),
        "resolution_summary": _resolution_summary(tables["resolutions"]),
        "boop_effectiveness": _boop_effectiveness(tables["boops"], tables["turns"]),
        "winning_methods": _winning_methods(boop_matches),
        "winning_orientations": _winning_orientations(tables["winning_lines"]),
        "winning_cells": _winning_cells(tables["winning_lines"]),
    }
    return derived


def _boolean(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.lower()
    result = normalized.map({"true": True, "false": False})
    if result.isna().any():
        raise ValueError("expected boolean CSV values")
    return result.astype(bool)


def _agent_performance(matches: pd.DataFrame, agents: list[str]) -> pd.DataFrame:
    competitive = matches.loc[~matches["self_play"]]
    rows = []
    for agent in agents:
        games = competitive.loc[
            (competitive["player_0_agent"] == agent)
            | (competitive["player_1_agent"] == agent)
        ]
        wins = int((games["winner_agent"] == agent).sum())
        draws = int(games["winner_agent"].isna().sum())
        total = len(games)
        low, high = wilson_interval(wins, total)
        rows.append(
            {
                "agent": agent,
                "games": total,
                "wins": wins,
                "losses": total - wins - draws,
                "draws": draws,
                "win_rate": wins / total if total else 0.0,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "agent",
            "games",
            "wins",
            "losses",
            "draws",
            "win_rate",
            "ci_low",
            "ci_high",
        ],
    )


def _pairwise_performance(matches: pd.DataFrame, agents: list[str]) -> pd.DataFrame:
    competitive = matches.loc[~matches["self_play"]]
    rows = []
    for agent in agents:
        for opponent in agents:
            if agent == opponent:
                continue
            games = competitive.loc[
                ((competitive["agent_a"] == agent) & (competitive["agent_b"] == opponent))
                | ((competitive["agent_a"] == opponent) & (competitive["agent_b"] == agent))
            ]
            if games.empty:
                continue
            wins = int((games["winner_agent"] == agent).sum())
            low, high = wilson_interval(wins, len(games))
            rows.append(
                {
                    "agent": agent,
                    "opponent": opponent,
                    "games": len(games),
                    "wins": wins,
                    "win_rate": wins / len(games),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["agent", "opponent", "games", "wins", "win_rate", "ci_low", "ci_high"],
    )


def _first_player_advantage(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups: list[tuple[str, pd.DataFrame]] = [("overall", matches)]
    competitive = matches.loc[~matches["self_play"]].copy()
    if not competitive.empty:
        competitive["pairing"] = competitive.apply(
            lambda row: " vs ".join(sorted((str(row["agent_a"]), str(row["agent_b"])))),
            axis=1,
        )
        groups.extend((name, group) for name, group in competitive.groupby("pairing", sort=True))
    for pairing, games in groups:
        decisive = games.loc[games["winner_player"].notna()]
        wins = int((decisive["winner_player"] == 0).sum())
        low, high = wilson_interval(wins, len(decisive))
        rows.append(
            {
                "pairing": pairing,
                "games": len(games),
                "decisive_games": len(decisive),
                "player_0_wins": wins,
                "player_0_win_rate": wins / len(decisive) if len(decisive) else 0.0,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def _game_lengths(matches: pd.DataFrame) -> pd.DataFrame:
    result = matches[
        ["match_number", "agent_a", "agent_b", "winner_agent", "plies", "duration_seconds"]
    ].copy()
    result["pairing"] = result.apply(
        lambda row: " vs ".join(sorted((str(row["agent_a"]), str(row["agent_b"])))), axis=1
    )
    result["seconds_per_ply"] = result["duration_seconds"] / result["plies"]
    return result


def _placement_cells(turns: pd.DataFrame, *, by_agent: bool = False) -> pd.DataFrame:
    identity = ["match_number", "agent"] if by_agent else ["match_number"]
    bases = turns[identity].drop_duplicates()
    cells = pd.DataFrame(
        [(row, column) for row in range(6) for column in range(6)],
        columns=["row", "column"],
    )
    bases["_join"] = 1
    cells["_join"] = 1
    complete = bases.merge(cells, on="_join").drop(columns="_join")
    counts = (
        turns.groupby(identity + ["row", "column"], sort=True)
        .size()
        .rename("placements")
        .reset_index()
    )
    complete = complete.merge(counts, on=identity + ["row", "column"], how="left")
    complete["placements"] = complete["placements"].fillna(0).astype(int)
    complete["match_share"] = complete["placements"] / complete.groupby(identity)[
        "placements"
    ].transform("sum")
    output_keys = ["agent", "row", "column"] if by_agent else ["row", "column"]
    return (
        complete.groupby(output_keys, sort=True)
        .agg(
            matches=("match_number", "nunique"),
            placements=("placements", "sum"),
            placement_share=("match_share", "mean"),
        )
        .reset_index()
    )


def _zone_strategy(turns: pd.DataFrame) -> pd.DataFrame:
    per_match = _zone_rates(turns, ["match_number", "agent"])
    rows = []
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    for (agent, zone), group in per_match.groupby(["agent", "zone"], sort=True):
        values = group["density_per_cell"].to_numpy(dtype=float)
        low, high = _bootstrap_mean_interval(values, rng)
        rows.append(
            {
                "agent": agent,
                "zone": zone,
                "matches": len(values),
                "mean_move_share": group["move_share"].mean(),
                "mean_density_per_cell": values.mean(),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def _zone_by_outcome(turns: pd.DataFrame) -> pd.DataFrame:
    per_match = _zone_rates(turns, ["match_number", "agent", "outcome"])
    result = (
        per_match.groupby(["outcome", "zone"], sort=True)
        .agg(
            matches=("match_number", "nunique"),
            mean_move_share=("move_share", "mean"),
            mean_density_per_cell=("density_per_cell", "mean"),
        )
        .reset_index()
    )
    return result


def _zone_progress(turns: pd.DataFrame) -> pd.DataFrame:
    working = turns[["match_number", "progress_fraction", "zone"]].copy()
    working["game_decile"] = np.ceil(working["progress_fraction"] * 10).clip(1, 10).astype(int)
    per_match = _zone_rates(working, ["match_number", "game_decile"])
    return (
        per_match.groupby(["game_decile", "zone"], sort=True)
        .agg(matches=("match_number", "nunique"), mean_move_share=("move_share", "mean"))
        .reset_index()
    )


def _zone_rates(turns: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    bases = turns[keys].drop_duplicates()
    zones = pd.DataFrame({"zone": _ZONE_ORDER})
    bases["_join"] = 1
    zones["_join"] = 1
    complete = bases.merge(zones, on="_join").drop(columns="_join")
    counts = turns.groupby(keys + ["zone"], sort=True).size().rename("moves").reset_index()
    complete = complete.merge(counts, on=keys + ["zone"], how="left")
    complete["moves"] = complete["moves"].fillna(0).astype(int)
    complete["total_moves"] = complete.groupby(keys)["moves"].transform("sum")
    complete["move_share"] = complete["moves"] / complete["total_moves"]
    complete["density_per_cell"] = complete["move_share"] / complete["zone"].map(_ZONE_AREAS)
    return complete


def _bootstrap_mean_interval(
    values: np.ndarray[Any, np.dtype[np.float64]],
    rng: np.random.Generator,
) -> tuple[float, float]:
    if len(values) == 0:
        return (0.0, 0.0)
    if len(values) == 1:
        value = float(values[0])
        return (value, value)
    indices = rng.integers(0, len(values), size=(_BOOTSTRAP_SAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return (float(low), float(high))


def _cat_progress(turns: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    working = turns.copy()
    working["game_decile"] = np.ceil(working["progress_fraction"] * 10).clip(1, 10).astype(int)
    winner_by_match = matches.set_index("match_number")["winner_player"]
    perspectives = []
    for player in (0, 1):
        perspective = working[
            [
                "match_number",
                "ply",
                "game_decile",
                f"p{player}_total_cats_after",
                f"p{player}_board_cats_after",
                f"p{player}_pool_cats_after",
            ]
        ].copy()
        perspective.columns = [
            "match_number",
            "ply",
            "game_decile",
            "total_cats",
            "board_cats",
            "pool_cats",
        ]
        perspective["player"] = player
        perspective["outcome"] = perspective["match_number"].map(winner_by_match).map(
            lambda winner: (
                "draw"
                if pd.isna(winner)
                else ("win" if int(winner) == player else "loss")
            )
        )
        perspectives.append(perspective)
    combined = pd.concat(perspectives, ignore_index=True)
    snapshots = (
        combined.sort_values("ply")
        .groupby(["match_number", "player", "outcome", "game_decile"], as_index=False)
        .tail(1)
    )
    return (
        snapshots.groupby(["outcome", "game_decile"], sort=True)
        .agg(
            player_matches=("match_number", "size"),
            mean_total_cats=("total_cats", "mean"),
            mean_board_cats=("board_cats", "mean"),
            mean_pool_cats=("pool_cats", "mean"),
        )
        .reset_index()
    )


def _first_graduation(boop_matches: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    result = boop_matches.merge(
        matches[["match_number", "winner_player"]],
        on="match_number",
        how="left",
        validate="one_to_one",
    )
    result["has_graduation"] = result["first_graduation_player"].notna()
    result["first_to_graduate_won"] = (
        result["has_graduation"]
        & (result["first_graduation_player"] == result["winner_player"])
    )
    return result[
        [
            "match_number",
            "first_graduation_ply",
            "first_graduation_player",
            "first_graduation_agent",
            "winner_player",
            "has_graduation",
            "first_to_graduate_won",
        ]
    ]


def _resolution_summary(resolutions: pd.DataFrame) -> pd.DataFrame:
    if resolutions.empty:
        return pd.DataFrame(
            columns=["type", "orientation", "events", "kittens_promoted", "cats_recycled"]
        )
    result = (
        resolutions.assign(orientation=resolutions["orientation"].fillna("not_applicable"))
        .groupby(["type", "orientation"], sort=True)
        .agg(
            events=("match_number", "size"),
            kittens_promoted=("kittens_promoted", "sum"),
            cats_recycled=("cats_recycled", "sum"),
        )
        .reset_index()
    )
    return result


def _boop_effectiveness(boops: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "agent",
        "target_relation",
        "placed_piece",
        "target_piece",
        "outcome",
        "interactions",
        "interaction_share",
        "interactions_per_100_placements",
    ]
    if boops.empty:
        return pd.DataFrame(columns=columns)
    placements = turns.groupby("agent").size()
    result = (
        boops.groupby(
            [
                "actor_agent",
                "target_relation",
                "placed_piece",
                "target_piece",
                "outcome",
            ],
            sort=True,
        )
        .size()
        .rename("interactions")
        .reset_index()
        .rename(columns={"actor_agent": "agent"})
    )
    result["interaction_share"] = result["interactions"] / result.groupby(
        ["agent", "target_relation"]
    )["interactions"].transform("sum")
    result["interactions_per_100_placements"] = (
        100 * result["interactions"] / result["agent"].map(placements)
    )
    return result[columns]


def _winning_methods(boop_matches: pd.DataFrame) -> pd.DataFrame:
    def method(row: pd.Series) -> str:
        if row["win_by_cat_line"] and row["win_by_eight_cats"]:
            return "cat_line_and_eight_cats"
        if row["win_by_cat_line"]:
            return "cat_line"
        if row["win_by_eight_cats"]:
            return "eight_cats"
        return "unknown"

    methods = boop_matches.apply(method, axis=1).value_counts(sort=False)
    total = int(methods.sum())
    return pd.DataFrame(
        {
            "method": methods.index,
            "matches": methods.values,
            "share": methods.values / total if total else methods.values,
        }
    )


def _winning_orientations(lines: pd.DataFrame) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame(columns=["orientation", "lines", "share"])
    winner_lines = lines.loc[_boolean(lines["is_declared_winner"])]
    counts = winner_lines["orientation"].value_counts(sort=False)
    return pd.DataFrame(
        {
            "orientation": counts.index,
            "lines": counts.values,
            "share": counts.values / counts.sum(),
        }
    )


def _winning_cells(lines: pd.DataFrame) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame(columns=["row", "column", "appearances"])
    winner_lines = lines.loc[_boolean(lines["is_declared_winner"])]
    parts = [
        winner_lines[[f"row_{index}", f"column_{index}"]].rename(
            columns={f"row_{index}": "row", f"column_{index}": "column"}
        )
        for index in (1, 2, 3)
    ]
    cells = pd.concat(parts, ignore_index=True)
    return cells.groupby(["row", "column"], sort=True).size().rename("appearances").reset_index()


def _generate_figures(
    tables: dict[str, pd.DataFrame],
    derived: dict[str, pd.DataFrame],
    output_dir: Path,
) -> list[dict[str, str]]:
    figures = [
        ("competition.png", "Competitive strength", _plot_competition),
        ("first_player.png", "First-player advantage", _plot_first_player),
        ("game_length.png", "Game length and runtime", _plot_game_length),
        ("placement_heatmap.png", "Overall placement heatmap", _plot_placement_heatmap),
        ("agent_heatmaps.png", "Placement heatmaps by agent", _plot_agent_heatmaps),
        ("zone_strategy.png", "Board-zone preferences", _plot_zone_strategy),
        ("zone_progress.png", "Board zones over game progress", _plot_zone_progress),
        ("cat_progress.png", "Cat development over game progress", _plot_cat_progress),
        ("graduations.png", "Graduations and recoveries", _plot_graduations),
        ("boop_effectiveness.png", "Boop interaction outcomes", _plot_boop_effectiveness),
        ("endgames.png", "Winning mechanisms and lines", _plot_endgames),
    ]
    descriptions = []
    for filename, title, plotter in figures:
        plotter(tables, derived, output_dir / filename)
        descriptions.append({"file": f"figures/{filename}", "title": title})
    return descriptions


def _plot_competition(
    tables: dict[str, pd.DataFrame], derived: dict[str, pd.DataFrame], path: Path
) -> None:
    del tables
    performance = derived["agent_performance"].sort_values("win_rate")
    pairwise = derived["pairwise_performance"]
    order = performance["agent"].tolist()
    figure, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.55 * len(order))))
    matrix = pairwise.pivot(index="agent", columns="opponent", values="win_rate").reindex(
        index=order, columns=order
    )
    sns.heatmap(matrix, annot=True, fmt=".0%", vmin=0, vmax=1, center=0.5, cmap="vlag", ax=axes[0])
    axes[0].set_title("Head-to-head win rate")
    axes[0].set_xlabel("Opponent")
    axes[0].set_ylabel("Agent")

    positions = np.arange(len(performance))
    axes[1].errorbar(
        performance["win_rate"],
        positions,
        xerr=np.vstack(
            (
                performance["win_rate"] - performance["ci_low"],
                performance["ci_high"] - performance["win_rate"],
            )
        ),
        fmt="o",
        capsize=4,
    )
    axes[1].set_yticks(positions, performance["agent"])
    axes[1].axvline(0.5, color="gray", linestyle="--")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Win rate (95% Wilson interval)")
    axes[1].set_title("Overall competitive performance")
    _save(figure, path)


def _plot_first_player(
    tables: dict[str, pd.DataFrame], derived: dict[str, pd.DataFrame], path: Path
) -> None:
    del tables
    advantage = derived["first_player_advantage"].sort_values(
        ["pairing"],
        key=lambda values: values.map(
            lambda value: "" if value == "overall" else value
        ),
    )
    figure, axis = plt.subplots(figsize=(11, max(5, 0.38 * len(advantage))))
    positions = np.arange(len(advantage))
    axis.errorbar(
        advantage["player_0_win_rate"],
        positions,
        xerr=np.vstack(
            (
                advantage["player_0_win_rate"] - advantage["ci_low"],
                advantage["ci_high"] - advantage["player_0_win_rate"],
            )
        ),
        fmt="o",
        capsize=3,
    )
    axis.set_yticks(positions, advantage["pairing"])
    axis.axvline(0.5, color="gray", linestyle="--")
    axis.set_xlim(0, 1)
    axis.set_xlabel("Player 0 win rate (95% Wilson interval)")
    axis.set_ylabel("Pairing")
    axis.set_title("First-player advantage overall and by matchup")
    _save(figure, path)


def _plot_game_length(
    tables: dict[str, pd.DataFrame], derived: dict[str, pd.DataFrame], path: Path
) -> None:
    del tables
    lengths = derived["game_lengths"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(lengths, x="plies", bins="auto", ax=axes[0])
    axes[0].axvline(lengths["plies"].median(), color="black", linestyle="--", label="median")
    axes[0].legend()
    axes[0].set_title("Game length")
    sns.scatterplot(lengths, x="plies", y="duration_seconds", alpha=0.65, ax=axes[1])
    if (lengths["duration_seconds"] > 0).all():
        axes[1].set_yscale("log")
    axes[1].set_title("Runtime versus game length")
    axes[1].set_ylabel("Duration in seconds (log scale)")
    _save(figure, path)


def _plot_placement_heatmap(
    tables: dict[str, pd.DataFrame], derived: dict[str, pd.DataFrame], path: Path
) -> None:
    del tables
    matrix = _board_matrix(derived["placement_cells"], "placement_share")
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt=".1%", cmap="mako", square=True, ax=axis)
    axis.set_title("Share of all placements")
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    _save(figure, path)


def _plot_agent_heatmaps(
    tables: dict[str, pd.DataFrame], derived: dict[str, pd.DataFrame], path: Path
) -> None:
    del tables
    placements = derived["placement_cells_by_agent"]
    agents = placements["agent"].drop_duplicates().tolist()
    columns = 3
    rows = max(1, math.ceil(len(agents) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.5 * rows), squeeze=False)
    for axis, agent in zip(axes.flat, agents, strict=False):
        matrix = _board_matrix(placements.loc[placements["agent"] == agent], "placement_share")
        sns.heatmap(matrix, cmap="mako", vmin=0, square=True, cbar=False, ax=axis)
        axis.set_title(agent)
        axis.set_xlabel("Column")
        axis.set_ylabel("Row")
    for axis in list(axes.flat)[len(agents) :]:
        axis.set_visible(False)
    figure.suptitle("Placement share by agent", y=1.01)
    _save(figure, path)


def _plot_zone_strategy(
    tables: dict[str, pd.DataFrame], derived: dict[str, pd.DataFrame], path: Path
) -> None:
    del tables
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(
        derived["zone_strategy"],
        x="agent",
        y="mean_density_per_cell",
        hue="zone",
        hue_order=_ZONE_ORDER,
        ax=axes[0],
    )
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_title("Zone density per cell and match")
    axes[0].set_ylabel("Mean share per cell")
    sns.barplot(
        derived["zone_by_outcome"],
        x="zone",
        y="mean_density_per_cell",
        hue="outcome",
        order=_ZONE_ORDER,
        ax=axes[1],
    )
    axes[1].set_title("Zone density in wins and losses")
    axes[1].set_ylabel("Mean share per cell")
    _save(figure, path)


def _plot_zone_progress(
    tables: dict[str, pd.DataFrame], derived: dict[str, pd.DataFrame], path: Path
) -> None:
    del tables
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        derived["zone_progress"],
        x="game_decile",
        y="mean_move_share",
        hue="zone",
        hue_order=_ZONE_ORDER,
        marker="o",
        ax=axis,
    )
    axis.set_xticks(range(1, 11))
    axis.set_ylim(0, 1)
    axis.set_title("Placement zones over normalized game progress")
    axis.set_xlabel("Game decile")
    axis.set_ylabel("Mean placement share")
    _save(figure, path)


def _plot_cat_progress(
    tables: dict[str, pd.DataFrame], derived: dict[str, pd.DataFrame], path: Path
) -> None:
    del tables
    progress = derived["cat_progress"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    metrics = (
        ("mean_total_cats", "Total cats acquired"),
        ("mean_board_cats", "Cats on board"),
        ("mean_pool_cats", "Cats in pool"),
    )
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        sns.lineplot(progress, x="game_decile", y=metric, hue="outcome", marker="o", ax=axis)
        axis.set_title(title)
        axis.set_xlabel("Game decile")
        axis.set_ylabel("Mean cats")
        axis.set_xticks(range(1, 11))
    _save(figure, path)


def _plot_graduations(
    tables: dict[str, pd.DataFrame], derived: dict[str, pd.DataFrame], path: Path
) -> None:
    first = derived["first_graduation"]
    resolutions = derived["resolution_summary"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.histplot(first.loc[first["has_graduation"]], x="first_graduation_ply", ax=axes[0])
    axes[0].set_title("First graduation ply")

    graduated = first.loc[first["has_graduation"]]
    wins = int(graduated["first_to_graduate_won"].sum())
    total = len(graduated)
    rate = wins / total if total else 0.0
    low, high = wilson_interval(wins, total)
    axes[1].bar(["First graduate wins"], [rate], color=sns.color_palette()[0])
    axes[1].errorbar([0], [rate], yerr=[[rate - low], [high - rate]], fmt="none", color="black")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Share of matches")
    axes[1].set_title(f"First-graduation advantage (n={total})")

    event_counts = resolutions.groupby("type", sort=True)["events"].sum().reset_index()
    sns.barplot(event_counts, x="type", y="events", ax=axes[2])
    axes[2].set_title("Graduations and eight-piece recoveries")
    axes[2].set_xlabel("Resolution type")
    _save(figure, path)


def _plot_boop_effectiveness(
    tables: dict[str, pd.DataFrame], derived: dict[str, pd.DataFrame], path: Path
) -> None:
    del tables
    effectiveness = derived["boop_effectiveness"]
    figure, axes = plt.subplots(1, 3, figsize=(20, 6))
    if effectiveness.empty:
        for axis in axes:
            axis.text(0.5, 0.5, "No boop interactions", ha="center", va="center")
            axis.set_axis_off()
    else:
        overall = (
            effectiveness.groupby(["target_relation", "outcome"], sort=True)["interactions"]
            .sum()
            .reset_index()
        )
        overall["share"] = overall["interactions"] / overall.groupby("target_relation")[
            "interactions"
        ].transform("sum")
        sns.barplot(overall, x="target_relation", y="share", hue="outcome", ax=axes[0])
        axes[0].set_title("Interaction outcome by target relation")
        axes[0].set_ylabel("Share of interactions")
        agent_rates = (
            effectiveness.groupby(["agent", "outcome"], sort=True)[
                "interactions_per_100_placements"
            ]
            .sum()
            .reset_index()
        )
        sns.barplot(
            agent_rates,
            x="agent",
            y="interactions_per_100_placements",
            hue="outcome",
            ax=axes[1],
        )
        axes[1].tick_params(axis="x", rotation=35)
        axes[1].set_title("Boop outcomes per 100 placements")
        piece_rates = effectiveness.copy()
        piece_rates["piece_pair"] = (
            piece_rates["placed_piece"] + " → " + piece_rates["target_piece"]
        )
        piece_rates = (
            piece_rates.groupby(["piece_pair", "outcome"], sort=True)["interactions"]
            .sum()
            .reset_index()
        )
        piece_rates["share"] = piece_rates["interactions"] / piece_rates.groupby(
            "piece_pair"
        )["interactions"].transform("sum")
        sns.barplot(piece_rates, x="piece_pair", y="share", hue="outcome", ax=axes[2])
        axes[2].tick_params(axis="x", rotation=25)
        axes[2].set_title("Outcomes by placed and target piece")
        axes[2].set_xlabel("Placed piece → target piece")
        axes[2].set_ylabel("Share of interactions")
    _save(figure, path)


def _plot_endgames(
    tables: dict[str, pd.DataFrame], derived: dict[str, pd.DataFrame], path: Path
) -> None:
    del tables
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    sns.barplot(derived["winning_methods"], x="method", y="share", ax=axes[0])
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Winning mechanisms")
    sns.barplot(
        derived["winning_orientations"], x="orientation", y="share", ax=axes[1]
    )
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_title("Winning-line orientations")
    matrix = _board_matrix(derived["winning_cells"], "appearances")
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="rocket_r", square=True, ax=axes[2])
    axes[2].set_title("Cells used by winning lines")
    axes[2].set_xlabel("Column")
    axes[2].set_ylabel("Row")
    _save(figure, path)


def _board_matrix(table: pd.DataFrame, value: str) -> pd.DataFrame:
    matrix = table.pivot(index="row", columns="column", values=value)
    return matrix.reindex(index=range(6), columns=range(6), fill_value=0).fillna(0)


def _save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _summary(
    manifest: dict[str, object],
    tables: dict[str, pd.DataFrame],
    derived: dict[str, pd.DataFrame],
    figures: list[dict[str, str]],
) -> dict[str, object]:
    matches = tables["matches"]
    first = derived["first_player_advantage"].iloc[0]
    graduations = derived["first_graduation"]
    graduated = graduations.loc[graduations["has_graduation"]]
    first_wins = int(graduated["first_to_graduate_won"].sum())
    return {
        "schema_version": 1,
        "game": "boop",
        "complete": bool(manifest.get("complete", False)),
        "matches": len(matches),
        "agents": len(tables["agents"]),
        "turns": len(tables["turns"]),
        "median_plies": float(matches["plies"].median()),
        "median_duration_seconds": float(matches["duration_seconds"].median()),
        "player_0_win_rate": float(first["player_0_win_rate"]),
        "player_0_win_rate_ci": [float(first["ci_low"]), float(first["ci_high"])],
        "first_graduation_win_rate": first_wins / len(graduated) if len(graduated) else None,
        "first_graduation_matches": len(graduated),
        "winning_methods": {
            str(row.method): int(row.matches)
            for row in derived["winning_methods"].itertuples(index=False)
        },
        "figures": len(figures),
        "tables": len(derived),
    }


def _render_html(
    input_dir: Path,
    manifest: dict[str, object],
    derived: dict[str, pd.DataFrame],
    summary: dict[str, object],
    figures: list[dict[str, str]],
) -> str:
    completeness = "Complete study" if summary["complete"] else "Preliminary: partial study"
    warning = (
        ""
        if summary["complete"]
        else '<p class="warning">This report uses a partial extraction. Results may change.</p>'
    )
    cards = "".join(
        '<div class="card">'
        f"<strong>{html.escape(label)}</strong>"
        f"<span>{html.escape(value)}</span>"
        "</div>"
        for label, value in (
            ("Matches", f"{summary['matches']:,}"),
            ("Turns", f"{summary['turns']:,}"),
            ("Agents", str(summary["agents"])),
            ("Median plies", f"{summary['median_plies']:.1f}"),
            ("Player 0 win rate", f"{summary['player_0_win_rate']:.1%}"),
            (
                "First graduate wins",
                "n/a"
                if summary["first_graduation_win_rate"] is None
                else f"{summary['first_graduation_win_rate']:.1%}",
            ),
        )
    )
    images = "".join(
        f'<section><h2>{html.escape(figure["title"])}</h2>'
        f'<img src="{html.escape(figure["file"])}" alt="{html.escape(figure["title"])}"></section>'
        for figure in figures
    )
    performance = derived["agent_performance"].copy()
    for column in ("win_rate", "ci_low", "ci_high"):
        performance[column] = performance[column].map(lambda value: f"{value:.1%}")
    table_html = performance.to_html(index=False, classes="data", border=0)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Boop tournament report</title>
  <style>
    body {{
      font-family: system-ui, sans-serif; max-width: 1200px; margin: auto;
      padding: 2rem; color: #20242a;
    }}
    h1, h2 {{ color: #172b4d; }}
    .meta {{ color: #5f6b7a; }}
    .warning {{ padding: 1rem; background: #fff2cc; border-left: 4px solid #d6a400; }}
    .cards {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;
    }}
    .card {{
      display: flex; flex-direction: column; padding: 1rem;
      background: #f4f6f8; border-radius: .5rem;
    }}
    .card span {{ font-size: 1.5rem; margin-top: .35rem; }}
    section {{ margin-top: 2.5rem; }}
    img {{ width: 100%; height: auto; border: 1px solid #d9dee5; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ padding: .55rem; text-align: right; border-bottom: 1px solid #d9dee5; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Boop tournament report</h1>
  <p class="meta">
    {html.escape(completeness)} · Source: {html.escape(str(input_dir))} ·
    Analysis schema {manifest.get('analysis_schema_version')}
  </p>
  {warning}
  <div class="cards">{cards}</div>
  <section><h2>Agent performance</h2>{table_html}</section>
  {images}
  <section>
    <h2>Interpretation notes</h2>
    <p>
      Self-play is excluded from competitive win rates but retained for strategic behavior.
      Zone density corrects for the center, middle, and outer zones containing 4, 12, and
      20 cells. Turn-level strategy is summarized by match so long games do not dominate
      the averages. Winner/loser differences are associations, not evidence of causation.
    </p>
  </section>
</body>
</html>
"""
