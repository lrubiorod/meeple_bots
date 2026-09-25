"""Shared extraction loading and competitive statistics for all game reports."""

from pathlib import Path

import pandas as pd

from .base import read_analysis_csv, wilson_interval


def load_tables(
    input_dir: Path, manifest: dict[str, object], required: tuple[str, ...],
    *, empty_as_missing: bool, optional: tuple[str, ...] = (),
) -> dict[str, pd.DataFrame]:
    manifest_tables = manifest.get("tables")
    if not isinstance(manifest_tables, dict):
        raise TypeError("extraction manifest tables must be an object")
    row_counts = manifest.get("row_counts")
    if not isinstance(row_counts, dict):
        raise TypeError("extraction manifest row_counts must be an object")

    tables: dict[str, pd.DataFrame] = {}
    for name in (*required, *optional):
        if name in optional and not isinstance(manifest_tables.get(name), str):
            tables[name] = pd.DataFrame()
            continue
        filename = manifest_tables.get(name)
        if not isinstance(filename, str):
            raise ValueError(f"extraction manifest does not define table {name}")
        path = input_dir / filename
        try:
            table = read_analysis_csv(path, empty_as_missing=empty_as_missing)
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


def boolean(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.lower()
    result = normalized.map({"true": True, "false": False})
    if result.isna().any():
        raise ValueError("expected boolean CSV values")
    return result.astype(bool)


def competitive_results(matches: pd.DataFrame) -> pd.DataFrame:
    self_play = boolean(matches["self_play"])
    rows = []
    for match in matches.loc[~self_play].itertuples(index=False):
        winner = (
            None if pd.isna(match.winner_player) or str(match.winner_player) == ""
            else int(match.winner_player)
        )
        players = (match.player_0_agent, match.player_1_agent)
        for seat, agent in enumerate(players):
            rows.append({
                "agent": agent,
                "opponent": players[1 - seat],
                "seat": seat,
                "wins": int(winner == seat),
                "draws": int(winner is None),
                "losses": int(winner is not None and winner != seat),
            })
    return pd.DataFrame(
        rows, columns=["agent", "opponent", "seat", "wins", "draws", "losses"]
    )


def performance(results: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    columns = [*keys, "games", "wins", "draws", "losses", "score"]
    if results.empty:
        return pd.DataFrame(columns=columns)
    grouped = results.groupby(keys, as_index=False)[["wins", "draws", "losses"]].sum()
    grouped["games"] = grouped[["wins", "draws", "losses"]].sum(axis=1)
    grouped["score"] = (grouped["wins"] + 0.5 * grouped["draws"]) / grouped["games"]
    return grouped[columns]


def competitive_performance(
    matches: pd.DataFrame, agents: list[str], *, pairwise: bool = False
) -> pd.DataFrame:
    """Include unplayed agents in totals; omit unobserved opponent pairs."""
    keys = ["agent", "opponent"] if pairwise else ["agent"]
    table = performance(competitive_results(matches), keys)
    if pairwise:
        order = pd.MultiIndex.from_tuples(
            [(agent, opponent) for agent in agents for opponent in agents if agent != opponent],
            names=keys,
        )
        table = table.set_index(keys).reindex(order).dropna(subset=["games"]).reset_index()
    else:
        table = table.set_index("agent").reindex(agents).fillna(0).rename_axis("agent").reset_index()
    for column in ("games", "wins", "draws", "losses"):
        table[column] = table[column].astype(int)
    table["win_rate"] = table["wins"].div(table["games"].replace(0, float("nan"))).fillna(0)
    intervals = [wilson_interval(row.wins, row.games) for row in table.itertuples()]
    table["ci_low"] = [low for low, _ in intervals]
    table["ci_high"] = [high for _, high in intervals]
    return table


def first_player_advantage(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    matches["self_play"] = boolean(matches["self_play"])
    matches["winner_player"] = pd.to_numeric(matches["winner_player"], errors="raise")
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
