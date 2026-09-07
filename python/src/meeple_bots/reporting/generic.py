"""Game-independent reports built exclusively from common extraction tables."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _load_tables(input_dir: Path, manifest: dict) -> dict[str, pd.DataFrame]:
    filenames = manifest.get("tables")
    counts = manifest.get("row_counts")
    if not isinstance(filenames, dict) or not isinstance(counts, dict):
        raise TypeError("extraction manifest tables and row_counts must be objects")
    tables = {}
    for name in ("agents", "matches", "moves"):
        filename = filenames.get(name)
        if not isinstance(filename, str):
            raise ValueError(f"extraction manifest does not define table {name}")
        # Preserve names such as 'NA'; convert only known numeric columns below.
        table = pd.read_csv(input_dir / filename, keep_default_na=False)
        expected = counts.get(name)
        if not isinstance(expected, int):
            raise TypeError(f"extraction row count for {name} must be an integer")
        if len(table) != expected:
            raise ValueError(
                f"extraction table {name} has {len(table)} rows; expected {expected}"
            )
        tables[name] = table
    if tables["matches"].empty:
        raise ValueError("cannot generate a report without completed matches")
    return tables


def _competitive_results(matches: pd.DataFrame) -> pd.DataFrame:
    self_play = matches["self_play"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if self_play.isna().any():
        raise ValueError("expected boolean CSV values for self_play")
    rows = []
    for match in matches.loc[~self_play].itertuples(index=False):
        winner = None if str(match.winner_player) == "" else int(match.winner_player)
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


def _performance(results: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    columns = [*keys, "games", "wins", "draws", "losses", "score"]
    if results.empty:
        return pd.DataFrame(columns=columns)
    grouped = results.groupby(keys, as_index=False)[["wins", "draws", "losses"]].sum()
    grouped["games"] = grouped[["wins", "draws", "losses"]].sum(axis=1)
    grouped["score"] = (grouped["wins"] + 0.5 * grouped["draws"]) / grouped["games"]
    return grouped[columns]


def _decision_performance(moves: pd.DataFrame, agents: pd.DataFrame) -> pd.DataFrame:
    measured = moves.copy()
    for column in ("decision_seconds", "search_iterations"):
        if column not in measured:
            measured[column] = float("nan")
        measured[column] = pd.to_numeric(measured[column], errors="coerce")
    rows = []
    for config in agents.to_dict(orient="records"):
        selected = measured.loc[measured["agent"] == config["agent_name"]]
        seconds = selected["decision_seconds"].dropna()
        iterations = selected["search_iterations"].dropna()
        # Use the same decisions in numerator and denominator for throughput.
        paired = selected.loc[
            selected["search_iterations"].notna() & (selected["decision_seconds"] > 0)
        ]
        total_seconds = paired["decision_seconds"].sum()
        rows.append({
            "agent": config["agent_name"],
            "configured_iterations": config.get("iterations", ""),
            "configured_time_budget": config.get("time_budget", ""),
            "decisions": len(selected),
            "timed_decisions": len(seconds),
            "search_decisions": len(iterations),
            "throughput_decisions": len(paired),
            "mean_decision_seconds": seconds.mean(),
            "p50_decision_seconds": seconds.median() if len(seconds) else float("nan"),
            "mean_iterations": iterations.mean(),
            "p50_iterations": iterations.median() if len(iterations) else float("nan"),
            "min_iterations": iterations.min(),
            "max_iterations": iterations.max(),
            "iterations_per_second": (
                paired["search_iterations"].sum() / total_seconds
                if total_seconds > 0 else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def _figures(derived: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, str]:
    figures = {}
    performance = derived["agent_performance"]
    decisions = derived["decision_performance"]
    timed = decisions.loc[decisions["timed_decisions"] > 0]
    for filename, table, column, label, scale in (
        ("score.png", performance, "score", "Competitive score", 1),
        ("decision_time.png", timed, "mean_decision_seconds", "Mean decision time (ms)", 1000),
    ):
        if table.empty:
            continue
        figure, axis = plt.subplots(figsize=(8, max(3, 0.35 * len(table))))
        try:
            axis.barh(table["agent"].astype(str), table[column] * scale)
            axis.set_xlabel(label)
            if column == "score":
                axis.set_xlim(0, 1)
                axis.axvline(0.5, color="gray", linestyle="--", linewidth=1)
            figure.tight_layout()
            figure.savefig(output_dir / filename, dpi=140)
        finally:
            plt.close(figure)
        figures[filename] = label
    return figures


def _render_html(manifest: dict, summary: dict, derived: dict, figures: dict) -> str:
    game = html.escape(str(manifest["game"]))
    status = "Complete study" if summary["complete"] else "Preliminary: partial study"
    sections = []
    descriptions = {
        "agent_performance": "Competitive results",
        "seat_performance": "Results by agent and seat (0 starts)",
        "pairwise_performance": "Results by opponent",
        "decision_performance": "Decision time and completed search iterations",
    }
    for name, table in derived.items():
        content = (
            "<p>No observations.</p>" if table.empty
            else table.to_html(index=False, border=0, na_rep="—", float_format=lambda x: f"{x:.6g}")
        )
        sections.append(
            f'<h2>{descriptions[name]}</h2><p><a href="tables/{name}.csv">Download CSV</a></p>'
            f'<div class="table">{content}</div>'
        )
    images = "".join(
        f'<figure><img src="figures/{name}" alt="{label}"><figcaption>{label}</figcaption></figure>'
        for name, label in figures.items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{game} tournament report</title><style>
body {{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#202630}}
table {{border-collapse:collapse}} th,td {{padding:.5rem;border-bottom:1px solid #ddd;text-align:right}}
.table {{overflow-x:auto}} img {{max-width:100%}} figure {{margin:1rem 0}}
</style></head><body><h1>{game} tournament report</h1>
<p>{status} · {summary['matches']} matches · {summary['competitive_matches']} competitive matches</p>
<p>Score = (wins + 0.5 × draws) / games. Competitive results exclude self-play.
Overall scores depend on the opponents faced; use results by opponent for direct comparisons.
These are descriptive results, without confidence intervals.</p>
<p>Decision metrics include self-play. Missing measurements are left blank or shown as —.
Iteration counts are actual completed iterations, including under time budgets.
Throughput is total iterations divided by total time on decisions with both measurements
and positive elapsed time.</p>
{images}{''.join(sections)}</body></html>
"""


def generate_generic_report(input_dir: Path, output_dir: Path, manifest: dict) -> dict:
    """Generate the standard report artifacts without game-specific analysis."""
    tables = _load_tables(input_dir, manifest)
    results = _competitive_results(tables["matches"])
    derived = {
        "agent_performance": _performance(results, ["agent"]),
        "seat_performance": _performance(results, ["agent", "seat"]),
        "pairwise_performance": _performance(results, ["agent", "opponent"]),
        "decision_performance": _decision_performance(tables["moves"], tables["agents"]),
    }
    for directory in ("tables", "figures"):
        (output_dir / directory).mkdir(parents=True)
    for name, table in derived.items():
        table.to_csv(output_dir / "tables" / f"{name}.csv", index=False)
    figures = _figures(derived, output_dir / "figures")
    summary = {
        "schema_version": 1,
        "game": manifest["game"],
        "complete": bool(manifest.get("complete", False)),
        "matches": len(tables["matches"]),
        "competitive_matches": len(results) // 2,
        "agents": len(tables["agents"]),
        "figures": len(figures),
        "tables": len(derived),
        "search_data_available": bool((derived["decision_performance"]["search_decisions"] > 0).any()),
        "agent_performance": derived["agent_performance"].to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "index.html").write_text(
        _render_html(manifest, summary, derived, figures), encoding="utf-8"
    )
    return summary
