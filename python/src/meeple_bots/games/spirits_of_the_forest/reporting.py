"""Statistical tables and figures for extracted Spirits of the Forest tournaments."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ...reporting import wilson_interval


_REQUIRED_TABLES = (
    "agents",
    "matches",
    "spotf_matches",
    "actions",
    "player_turns",
    "tile_takes",
    "gemstone_actions",
    "categories",
)
_SPIRIT_ORDER = [
    "moss",
    "flowers",
    "fruits",
    "mushrooms",
    "water",
    "vines",
    "branches",
    "leaves",
    "webs",
]
_POWER_ORDER = ["fire", "moon", "sun"]
_GEMSTONE_ACTION_ORDER = ["place_gemstone", "move_gemstone", "skip_gemstone"]


def generate_spotf_report(
    input_dir: Path,
    output_dir: Path,
    manifest: dict[str, object],
) -> dict[str, object]:
    """Build all SPOTF-specific report artifacts in an empty output directory."""

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
    row_counts = manifest.get("row_counts")
    if not isinstance(manifest_tables, dict):
        raise TypeError("extraction manifest tables must be an object")
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
    matches["self_play"] = _boolean(matches["self_play"])
    agents = tables["agents"]["agent_name"].astype(str).tolist()
    return {
        "agent_performance": _agent_performance(matches, agents),
        "pairwise_performance": _pairwise_performance(matches, agents),
        "first_player_advantage": _first_player_advantage(matches),
        "game_lengths": _game_lengths(matches, tables["spotf_matches"]),
        "score_performance": _score_performance(matches, tables["spotf_matches"], agents),
        "turn_structure": _turn_structure(tables["player_turns"], tables["actions"], agents),
        "search_performance": _search_performance(tables["actions"], tables["agents"]),
        "search_by_phase": _search_by_phase(tables["actions"]),
        "strategic_progress": _strategic_progress(tables["player_turns"]),
        "spirit_performance": _category_performance(
            tables["categories"], "spirit", _SPIRIT_ORDER
        ),
        "power_source_performance": _category_performance(
            tables["categories"], "power_source", _POWER_ORDER
        ),
        "gemstone_strategy": _gemstone_strategy(tables["gemstone_actions"], agents),
        "sacrifice_strategy": _sacrifice_strategy(tables["tile_takes"], agents),
        "sacrifice_timing": _sacrifice_timing(tables["tile_takes"], agents),
    }


def _boolean(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    result = values.astype(str).str.lower().map({"true": True, "false": False})
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
        low, high = wilson_interval(wins, len(games))
        rows.append(
            {
                "agent": agent,
                "games": len(games),
                "wins": wins,
                "losses": len(games) - wins - draws,
                "draws": draws,
                "win_rate": wins / len(games) if len(games) else 0.0,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def _pairwise_performance(matches: pd.DataFrame, agents: list[str]) -> pd.DataFrame:
    competitive = matches.loc[~matches["self_play"]]
    rows = []
    for agent in agents:
        for opponent in agents:
            if agent == opponent:
                continue
            games = competitive.loc[
                ((competitive["agent_a"] == agent) & (competitive["agent_b"] == opponent))
                | (
                    (competitive["agent_a"] == opponent)
                    & (competitive["agent_b"] == agent)
                )
            ]
            if games.empty:
                continue
            wins = int((games["winner_agent"] == agent).sum())
            draws = int(games["winner_agent"].isna().sum())
            low, high = wilson_interval(wins, len(games))
            rows.append(
                {
                    "agent": agent,
                    "opponent": opponent,
                    "games": len(games),
                    "wins": wins,
                    "draws": draws,
                    "win_rate": wins / len(games),
                    "score_rate": (wins + 0.5 * draws) / len(games),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


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
        first_wins = int((decisive["winner_player"] == 0).sum())
        low, high = wilson_interval(first_wins, len(decisive))
        rows.append(
            {
                "pairing": pairing,
                "games": len(games),
                "decisive_games": len(decisive),
                "player_0_wins": first_wins,
                "player_0_win_rate": first_wins / len(decisive) if len(decisive) else 0.0,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def _game_lengths(matches: pd.DataFrame, spotf_matches: pd.DataFrame) -> pd.DataFrame:
    columns = ["match_number", "physical_turns", "actions_per_turn", "score_margin"]
    return matches[
        ["match_number", "agent_a", "agent_b", "winner_agent", "plies", "duration_seconds"]
    ].merge(spotf_matches[columns], on="match_number", validate="one_to_one")


def _score_performance(
    matches: pd.DataFrame,
    spotf_matches: pd.DataFrame,
    agents: list[str],
) -> pd.DataFrame:
    merged = matches.merge(spotf_matches, on="match_number", validate="one_to_one")
    rows = []
    for agent in agents:
        observations = []
        for match in merged.itertuples(index=False):
            for player in (0, 1):
                if getattr(match, f"player_{player}_agent") != agent:
                    continue
                score = getattr(match, f"score_{player}")
                opponent_score = getattr(match, f"score_{1 - player}")
                observations.append(
                    (
                        score,
                        score - opponent_score,
                        getattr(match, f"tiles_{player}"),
                    )
                )
        values = np.asarray(observations, dtype=float)
        rows.append(
            {
                "agent": agent,
                "games": len(observations),
                "mean_score": float(values[:, 0].mean()) if len(values) else 0.0,
                "median_score": float(np.median(values[:, 0])) if len(values) else 0.0,
                "mean_score_margin": float(values[:, 1].mean()) if len(values) else 0.0,
                "mean_tiles": float(values[:, 2].mean()) if len(values) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _turn_structure(
    player_turns: pd.DataFrame,
    actions: pd.DataFrame,
    agents: list[str],
) -> pd.DataFrame:
    rows = []
    for agent in agents:
        turns = player_turns.loc[player_turns["agent"] == agent]
        agent_actions = actions.loc[actions["agent"] == agent]
        rows.append(
            {
                "agent": agent,
                "turns": len(turns),
                "mean_actions_per_turn": turns["actions"].mean() if len(turns) else 0.0,
                "mean_tiles_per_turn": turns["tiles_collected"].mean() if len(turns) else 0.0,
                "two_tile_turn_rate": (
                    (turns["tiles_collected"] == 2).mean() if len(turns) else 0.0
                ),
                "mean_symbols_per_turn": (
                    turns["symbols_collected"].mean() if len(turns) else 0.0
                ),
                "end_collection_rate": (
                    _boolean(turns["used_end_collection"]).mean() if len(turns) else 0.0
                ),
                "mean_legal_actions": (
                    agent_actions["legal_actions_before"].mean()
                    if len(agent_actions)
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _numeric(table: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = table.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _search_performance(actions: pd.DataFrame, agents: pd.DataFrame) -> pd.DataFrame:
    measured = _numeric(
        actions,
        ("decision_seconds", "search_iterations", "search_nodes", "legal_actions_before"),
    )
    measured = measured.loc[
        measured["search_iterations"].notna() & (measured["decision_seconds"] > 0)
    ]
    agent_configs = agents.copy()
    agent_configs["time_budget"] = pd.to_numeric(
        agent_configs["time_budget"], errors="coerce"
    )
    rows = []
    for config in agent_configs.itertuples(index=False):
        selected = measured.loc[measured["agent"] == config.agent_name]
        seconds = selected["decision_seconds"]
        iterations = selected["search_iterations"]
        node_rows = selected.loc[selected["search_nodes"].notna()]
        nodes = node_rows["search_nodes"]
        total_seconds = float(seconds.sum())
        total_iterations = float(iterations.sum())
        total_nodes = float(nodes.sum())
        node_seconds = float(node_rows["decision_seconds"].sum())
        node_iterations = float(node_rows["search_iterations"].sum())
        budget = float(config.time_budget) if pd.notna(config.time_budget) else np.nan
        utilization = seconds / budget if budget > 0 else pd.Series(dtype=float)
        rows.append(
            {
                "agent": config.agent_name,
                "measured_decisions": len(selected),
                "configured_time_budget": budget,
                "mean_decision_seconds": seconds.mean() if len(selected) else np.nan,
                "p50_decision_seconds": seconds.median() if len(selected) else np.nan,
                "p95_decision_seconds": (
                    seconds.quantile(0.95) if len(selected) else np.nan
                ),
                "mean_iterations": iterations.mean() if len(selected) else np.nan,
                "p50_iterations": iterations.median() if len(selected) else np.nan,
                "p95_iterations": (
                    iterations.quantile(0.95) if len(selected) else np.nan
                ),
                "iterations_per_second": (
                    total_iterations / total_seconds if total_seconds else np.nan
                ),
                "milliseconds_per_iteration": (
                    1_000.0 * total_seconds / total_iterations
                    if total_iterations
                    else np.nan
                ),
                "mean_nodes": nodes.mean() if len(nodes) else np.nan,
                "nodes_per_second": (
                    total_nodes / node_seconds if node_seconds else np.nan
                ),
                "nodes_per_iteration": (
                    total_nodes / node_iterations if node_iterations else np.nan
                ),
                "mean_legal_actions": (
                    selected["legal_actions_before"].mean() if len(selected) else np.nan
                ),
                "mean_budget_utilization": (
                    utilization.mean() if len(utilization) else np.nan
                ),
                "p95_budget_utilization": (
                    utilization.quantile(0.95) if len(utilization) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _search_by_phase(actions: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "agent",
        "game_quarter",
        "phase",
        "decisions",
        "mean_decision_seconds",
        "p95_decision_seconds",
        "mean_iterations",
        "iterations_per_second",
        "milliseconds_per_iteration",
        "mean_nodes",
        "nodes_per_second",
        "mean_legal_actions",
    )
    measured = _numeric(
        actions,
        ("decision_seconds", "search_iterations", "search_nodes", "legal_actions_before"),
    )
    measured = measured.loc[
        measured["search_iterations"].notna() & (measured["decision_seconds"] > 0)
    ]
    rows = []
    for (agent, quarter, phase), selected in measured.groupby(
        ["agent", "tile_quarter", "phase_before"], sort=True
    ):
        total_seconds = float(selected["decision_seconds"].sum())
        total_iterations = float(selected["search_iterations"].sum())
        node_rows = selected.loc[selected["search_nodes"].notna()]
        nodes = node_rows["search_nodes"]
        node_seconds = float(node_rows["decision_seconds"].sum())
        rows.append(
            {
                "agent": agent,
                "game_quarter": quarter,
                "phase": phase,
                "decisions": len(selected),
                "mean_decision_seconds": selected["decision_seconds"].mean(),
                "p95_decision_seconds": selected["decision_seconds"].quantile(0.95),
                "mean_iterations": selected["search_iterations"].mean(),
                "iterations_per_second": (
                    total_iterations / total_seconds if total_seconds else np.nan
                ),
                "milliseconds_per_iteration": (
                    1_000.0 * total_seconds / total_iterations
                    if total_iterations
                    else np.nan
                ),
                "mean_nodes": nodes.mean() if len(nodes) else np.nan,
                "nodes_per_second": (
                    float(nodes.sum()) / node_seconds if node_seconds else np.nan
                ),
                "mean_legal_actions": selected["legal_actions_before"].mean(),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _strategic_progress(player_turns: pd.DataFrame) -> pd.DataFrame:
    turns = _numeric(
        player_turns,
        (
            "score_delta",
            "reachable_score_after",
            "reachable_score_delta",
            "categories_present_after",
            "categories_reachable_after",
            "categories_leading_after",
            "gemstones_removed_delta",
            "gemstones_removed_after",
            "gemstones_usable_after",
            "tiles_collected",
        ),
    )
    turns["removed_gemstone"] = turns["gemstones_removed_delta"] > 0
    turns["two_tile_turn"] = turns["tiles_collected"] == 2
    return (
        turns.groupby(["agent", "tile_quarter"], sort=True)
        .agg(
            turns=("match_number", "size"),
            mean_score_delta=("score_delta", "mean"),
            mean_reachable_score=("reachable_score_after", "mean"),
            mean_reachable_score_delta=("reachable_score_delta", "mean"),
            mean_categories_present=("categories_present_after", "mean"),
            mean_categories_reachable=("categories_reachable_after", "mean"),
            mean_categories_leading=("categories_leading_after", "mean"),
            mean_gemstones_usable=("gemstones_usable_after", "mean"),
            mean_gemstones_removed=("gemstones_removed_after", "mean"),
            gemstone_removal_rate=("removed_gemstone", "mean"),
            two_tile_turn_rate=("two_tile_turn", "mean"),
        )
        .reset_index()
        .rename(columns={"tile_quarter": "game_quarter"})
    )


def _category_performance(
    categories: pd.DataFrame,
    category_type: str,
    order: list[str],
) -> pd.DataFrame:
    selected = categories.loc[categories["category_type"] == category_type].copy()
    selected["won_or_tied_majority"] = _boolean(selected["won_or_tied_majority"])
    selected["absent_penalty"] = _boolean(selected["absent_penalty"])
    selected["lost_majority"] = _boolean(selected["lost_majority"])
    result = (
        selected.groupby(["agent", "category"], sort=False)
        .agg(
            games=("match_number", "size"),
            mean_count=("count", "mean"),
            mean_points=("points", "mean"),
            mean_count_gap=("count_gap", "mean"),
            majority_rate=("won_or_tied_majority", "mean"),
            absence_rate=("absent_penalty", "mean"),
            lost_majority_rate=("lost_majority", "mean"),
        )
        .reset_index()
    )
    result["category"] = pd.Categorical(result["category"], categories=order, ordered=True)
    return result.sort_values(["agent", "category"]).reset_index(drop=True)


def _gemstone_strategy(gemstones: pd.DataFrame, agents: list[str]) -> pd.DataFrame:
    rows = []
    for agent in agents:
        selected = gemstones.loc[gemstones["agent"] == agent]
        total = len(selected)
        for action in _GEMSTONE_ACTION_ORDER:
            count = int((selected["action"] == action).sum())
            rows.append(
                {
                    "agent": agent,
                    "action": action,
                    "count": count,
                    "rate": count / total if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _sacrifice_strategy(tile_takes: pd.DataFrame, agents: list[str]) -> pd.DataFrame:
    rows = []
    for agent in agents:
        takes = tile_takes.loc[tile_takes["agent"] == agent]
        total = len(takes)
        opponent = takes["reservation_relation"] == "opponent"
        rows.append(
            {
                "agent": agent,
                "tiles_taken": total,
                "own_reservations_collected": int(
                    (takes["reservation_relation"] == "own").sum()
                ),
                "opponent_reservations_taken": int(opponent.sum()),
                "opponent_reservation_rate": opponent.mean() if total else 0.0,
                "available_sacrifices": int((takes["sacrifice"] == "available").sum()),
                "forest_sacrifices": int((takes["sacrifice"] == "forest").sum()),
            }
        )
    return pd.DataFrame(rows)


def _sacrifice_timing(tile_takes: pd.DataFrame, agents: list[str]) -> pd.DataFrame:
    rows = []
    for agent in agents:
        agent_takes = tile_takes.loc[tile_takes["agent"] == agent]
        for quarter in ("q1", "q2", "q3", "q4"):
            takes = agent_takes.loc[agent_takes["tile_quarter"] == quarter]
            reservations = takes.loc[takes["reservation_relation"] == "opponent"]
            sacrifices = takes["sacrifice"] != "none"
            rows.append(
                {
                    "agent": agent,
                    "game_quarter": quarter,
                    "tiles_taken": len(takes),
                    "opponent_reservations_taken": len(reservations),
                    "sacrifices": int(sacrifices.sum()),
                    "sacrifice_rate": sacrifices.mean() if len(takes) else 0.0,
                    "available_sacrifices": int((takes["sacrifice"] == "available").sum()),
                    "forest_sacrifices": int((takes["sacrifice"] == "forest").sum()),
                    "second_tile_rate": (
                        _boolean(takes["second_tile"]).mean() if len(takes) else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _generate_figures(
    tables: dict[str, pd.DataFrame],
    derived: dict[str, pd.DataFrame],
    figures_dir: Path,
) -> dict[str, str]:
    descriptions: dict[str, str] = {}

    performance = derived["agent_performance"].sort_values("win_rate")
    figure, axis = plt.subplots(figsize=(10, max(4, 0.55 * len(performance))))
    errors = np.vstack(
        [
            performance["win_rate"] - performance["ci_low"],
            performance["ci_high"] - performance["win_rate"],
        ]
    )
    axis.barh(performance["agent"], performance["win_rate"], xerr=errors, capsize=3)
    axis.axvline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Win rate (95% Wilson interval)", ylabel="Agent", xlim=(0, 1))
    descriptions["competition.png"] = "Overall competitive win rate by agent."
    _save_figure(figure, figures_dir / "competition.png")

    pairwise = derived["pairwise_performance"]
    matrix = pairwise.pivot(index="agent", columns="opponent", values="score_rate")
    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="RdYlGn", ax=axis)
    axis.set(title="Head-to-head score rate", xlabel="Opponent", ylabel="Agent")
    descriptions["pairwise.png"] = "Head-to-head score rate, counting a draw as half a win."
    _save_figure(figure, figures_dir / "pairwise.png")

    first = derived["first_player_advantage"].iloc[0]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(["Player 0", "Player 1"], [first["player_0_win_rate"], 1 - first["player_0_win_rate"]])
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set(ylabel="Win rate among decisive games", ylim=(0, 1))
    descriptions["first_player.png"] = "Observed result advantage for the starting player."
    _save_figure(figure, figures_dir / "first_player.png")

    lengths = derived["game_lengths"].copy()
    lengths["pairing"] = lengths.apply(
        lambda row: " vs ".join(sorted((str(row["agent_a"]), str(row["agent_b"])))),
        axis=1,
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(lengths["plies"], bins="auto", ax=axes[0])
    axes[0].set(title="Tree plies per game", xlabel="Plies")
    sns.histplot(lengths["physical_turns"], bins="auto", ax=axes[1])
    axes[1].set(title="Physical turns per game", xlabel="Player turns")
    descriptions["game_length.png"] = "Structural plies compared with actual player turns."
    _save_figure(figure, figures_dir / "game_length.png")

    scores = derived["score_performance"].sort_values("mean_score")
    figure, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.5 * len(scores))))
    axes[0].barh(scores["agent"], scores["mean_score"])
    axes[0].set(title="Mean final score", xlabel="Points", ylabel="Agent")
    axes[1].barh(scores["agent"], scores["mean_score_margin"])
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set(title="Mean score margin", xlabel="Own score - opponent score")
    descriptions["scores.png"] = "Final score and signed score margin by agent."
    _save_figure(figure, figures_dir / "scores.png")

    turn_structure = derived["turn_structure"].set_index("agent")
    figure, axis = plt.subplots(figsize=(10, 5))
    turn_metrics = [
        "mean_actions_per_turn",
        "mean_tiles_per_turn",
        "mean_symbols_per_turn",
    ]
    turn_structure[turn_metrics].plot.bar(ax=axis)
    axis.set(title="Turn structure by agent", xlabel="Agent", ylabel="Mean per physical turn")
    axis.tick_params(axis="x", rotation=35)
    descriptions["turn_structure.png"] = "Actions, collected tiles, and symbols per physical turn."
    _save_figure(figure, figures_dir / "turn_structure.png")

    _category_heatmap(
        derived["spirit_performance"],
        _SPIRIT_ORDER,
        figures_dir / "spirits.png",
        "Mean points by spirit",
    )
    descriptions["spirits.png"] = "Mean scoring contribution for each spirit category."

    _category_heatmap(
        derived["power_source_performance"],
        _POWER_ORDER,
        figures_dir / "power_sources.png",
        "Mean points by power source",
    )
    descriptions["power_sources.png"] = "Mean scoring contribution for fire, moon, and sun."

    gemstone = derived["gemstone_strategy"].pivot(index="agent", columns="action", values="rate")
    gemstone = gemstone.reindex(columns=_GEMSTONE_ACTION_ORDER, fill_value=0)
    figure, axis = plt.subplots(figsize=(10, 5))
    gemstone.plot.bar(stacked=True, ax=axis)
    axis.set(title="Gemstone decisions", xlabel="Agent", ylabel="Share", ylim=(0, 1))
    axis.tick_params(axis="x", rotation=35)
    descriptions["gemstones.png"] = "Relative use of placing, moving, and skipping gemstones."
    _save_figure(figure, figures_dir / "gemstones.png")

    sacrifice = derived["sacrifice_strategy"].set_index("agent")
    figure, axis = plt.subplots(figsize=(10, 5))
    sacrifice[["available_sacrifices", "forest_sacrifices"]].plot.bar(ax=axis)
    axis.set(title="Sacrifices used to take opponent reservations", xlabel="Agent", ylabel="Count")
    axis.tick_params(axis="x", rotation=35)
    descriptions["sacrifices.png"] = "Supply and forest gems sacrificed by each agent."
    _save_figure(figure, figures_dir / "sacrifices.png")

    search = derived["search_performance"].loc[
        derived["search_performance"]["measured_decisions"] > 0
    ].sort_values("iterations_per_second")
    if not search.empty:
        figure, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.5 * len(search))))
        axes[0].barh(search["agent"], search["iterations_per_second"])
        axes[0].set(title="Search throughput", xlabel="Iterations / second", ylabel="Agent")
        axes[1].barh(search["agent"], search["nodes_per_second"])
        axes[1].set(title="Tree growth throughput", xlabel="Nodes / second")
        descriptions["search_efficiency.png"] = (
            "Measured MCTS iteration and node throughput for each agent configuration."
        )
        _save_figure(figure, figures_dir / "search_efficiency.png")

        measured_actions = _numeric(
            tables["actions"],
            ("decision_seconds", "search_iterations"),
        )
        measured_actions = measured_actions.loc[
            measured_actions["search_iterations"].notna()
            & (measured_actions["decision_seconds"] > 0)
        ]
        figure, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.lineplot(
            data=measured_actions,
            x="tile_quarter",
            y="search_iterations",
            hue="agent",
            marker="o",
            ax=axes[0],
        )
        axes[0].set(title="Iterations reached by game quarter", xlabel="Game quarter")
        sns.lineplot(
            data=measured_actions,
            x="tile_quarter",
            y="decision_seconds",
            hue="agent",
            marker="o",
            legend=False,
            ax=axes[1],
        )
        axes[1].set(title="Decision time by game quarter", xlabel="Game quarter")
        descriptions["search_timing.png"] = (
            "Mean iterations and decision time as the game state becomes cheaper to search."
        )
        _save_figure(figure, figures_dir / "search_timing.png")

    progress = derived["strategic_progress"]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.lineplot(
        data=progress,
        x="game_quarter",
        y="mean_categories_reachable",
        hue="agent",
        marker="o",
        ax=axes[0],
    )
    axes[0].set(title="Still-reachable scoring categories", xlabel="Game quarter")
    sns.lineplot(
        data=progress,
        x="game_quarter",
        y="mean_gemstones_usable",
        hue="agent",
        marker="o",
        legend=False,
        ax=axes[1],
    )
    axes[1].set(title="Usable gemstones retained", xlabel="Game quarter")
    descriptions["strategic_progress.png"] = (
        "Evolution of scoring-category viability and gemstone conservation by agent."
    )
    _save_figure(figure, figures_dir / "strategic_progress.png")

    sacrifice_timing = derived["sacrifice_timing"].pivot(
        index="agent", columns="game_quarter", values="sacrifice_rate"
    )
    figure, axis = plt.subplots(
        figsize=(8, max(4, 0.55 * len(sacrifice_timing)))
    )
    sns.heatmap(
        sacrifice_timing.reindex(columns=["q1", "q2", "q3", "q4"]),
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
        ax=axis,
    )
    axis.set(title="Gemstone sacrifice timing", xlabel="Game quarter", ylabel="Agent")
    descriptions["sacrifice_timing.png"] = (
        "Share of collected tiles that required a gemstone sacrifice in each game quarter."
    )
    _save_figure(figure, figures_dir / "sacrifice_timing.png")

    return descriptions


def _category_heatmap(
    table: pd.DataFrame,
    order: list[str],
    path: Path,
    title: str,
) -> None:
    matrix = table.pivot(index="agent", columns="category", values="mean_points")
    matrix = matrix.reindex(columns=order)
    figure, axis = plt.subplots(figsize=(max(7, len(order)), max(4, 0.55 * len(matrix))))
    sns.heatmap(matrix, annot=True, fmt=".2f", center=0, cmap="RdYlGn", ax=axis)
    axis.set(title=title, xlabel="Category", ylabel="Agent")
    _save_figure(figure, path)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _summary(
    manifest: dict[str, object],
    tables: dict[str, pd.DataFrame],
    derived: dict[str, pd.DataFrame],
    figures: dict[str, str],
) -> dict[str, object]:
    first = derived["first_player_advantage"].iloc[0]
    lengths = derived["game_lengths"]
    measured_search = derived["search_performance"].loc[
        derived["search_performance"]["measured_decisions"] > 0
    ]
    return {
        "game": "spotf",
        "complete": bool(manifest.get("complete", False)),
        "matches": int(len(tables["matches"])),
        "agents": int(len(tables["agents"])),
        "figures": len(figures),
        "tables": len(derived),
        "mean_plies": float(lengths["plies"].mean()),
        "mean_physical_turns": float(lengths["physical_turns"].mean()),
        "mean_actions_per_turn": float(lengths["actions_per_turn"].mean()),
        "mean_score_margin": float(lengths["score_margin"].mean()),
        "player_0_win_rate": float(first["player_0_win_rate"]),
        "search_data_available": not measured_search.empty,
        "search_agents": int(len(measured_search)),
        "agent_performance": derived["agent_performance"].to_dict(orient="records"),
    }


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _render_html(
    input_dir: Path,
    manifest: dict[str, object],
    derived: dict[str, pd.DataFrame],
    summary: dict[str, object],
    figures: dict[str, str],
) -> str:
    status = "Complete" if manifest.get("complete") else "Preliminary / partial trace"
    cards = "".join(
        "<div class='card'>"
        f"<strong>{html.escape(label)}</strong>"
        f"<span>{html.escape(value)}</span>"
        "</div>"
        for label, value in (
            ("Status", status),
            ("Matches", str(summary["matches"])),
            ("Mean plies", f"{summary['mean_plies']:.1f}"),
            ("Mean physical turns", f"{summary['mean_physical_turns']:.1f}"),
            ("Actions / turn", f"{summary['mean_actions_per_turn']:.2f}"),
            ("Mean score margin", f"{summary['mean_score_margin']:.2f}"),
        )
    )
    figure_sections = "".join(
        "<article class='figure'>"
        f"<h2>{html.escape(filename.removesuffix('.png').replace('_', ' ').title())}</h2>"
        f"<p>{html.escape(description)}</p>"
        f"<img src='figures/{html.escape(filename)}' alt='{html.escape(description)}'>"
        "</article>"
        for filename, description in figures.items()
    )
    table_sections = "".join(
        "<details>"
        f"<summary>{html.escape(name.replace('_', ' ').title())}</summary>"
        f"{table.head(30).to_html(index=False, border=0, float_format=_format_float)}"
        f"<p><a href='tables/{html.escape(name)}.csv'>Download complete CSV</a></p>"
        "</details>"
        for name, table in derived.items()
    )
    source = html.escape(str(input_dir))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Spirits of the Forest tournament report</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; background: #f4f6f1; color: #1d2a20; }}
    header, main {{ max-width: 1200px; margin: auto; padding: 1.5rem; }}
    header {{ padding-top: 2.5rem; }}
    h1 {{ margin-bottom: .25rem; }}
    .source {{ color: #536257; overflow-wrap: anywhere; }}
    .cards {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: .8rem;
    }}
    .card, .figure, details {{
      background: white; border-radius: 10px; padding: 1rem; box-shadow: 0 2px 8px #20302018;
    }}
    .card span {{ display: block; margin-top: .4rem; font-size: 1.25rem; }}
    .figures {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
      gap: 1rem; margin-top: 1rem;
    }}
    .figure img {{ width: 100%; height: auto; }}
    details {{ margin: .8rem 0; overflow-x: auto; }}
    summary {{ cursor: pointer; font-weight: 700; }}
    table {{ border-collapse: collapse; margin-top: 1rem; width: 100%; font-size: .85rem; }}
    th, td {{ border-bottom: 1px solid #dfe5dd; padding: .4rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    @media (max-width: 600px) {{ .figures {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Spirits of the Forest tournament report</h1>
    <p class="source">Extracted data: {source}</p>
    <div class="cards">{cards}</div>
  </header>
  <main>
    <section class="figures">{figure_sections}</section>
    <section><h2>Analysis tables</h2>{table_sections}</section>
  </main>
</body>
</html>
"""
