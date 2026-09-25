"""Dependency-light report CSV loading, timing labels, and Wilson intervals."""

from __future__ import annotations

from pathlib import Path

def read_analysis_csv(path: Path, *, empty_as_missing: bool):
    """Preserve literal identifiers; only empty cells may represent missing data."""
    import pandas as pd

    identifiers = (
        "agent_name", "agent", "agent_a", "agent_b", "player_0_agent", "player_1_agent",
        "winner_agent", "actor_agent", "target_agent", "reservation_agent",
        "first_graduation_agent", "study_id", "source",
    )
    return pd.read_csv(
        path,
        keep_default_na=False,
        na_values=[""] if empty_as_missing else [],
        dtype={column: str for column in identifiers},
    )


def decision_timing_description(manifest: dict) -> str:
    """Explain the measurement boundary, including historical extractions."""
    if manifest.get("decision_timing_scope") == "agent_total_v1":
        return (
            "Decision time is total agent cost: selection plus lifecycle maintenance, "
            "including updates after opponent actions and final cleanup. "
            "Game transitions and observer/UI work are excluded."
        )
    return "Legacy decision timing covers selection only; lifecycle maintenance was not measured."


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if successes < 0 or total < 0 or successes > total:
        raise ValueError("successes must be between zero and total")
    if total == 0:
        return (0.0, 0.0)
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    margin = (
        z
        * (
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total * total)
        )
        ** 0.5
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))

