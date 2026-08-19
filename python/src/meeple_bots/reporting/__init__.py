"""Generic report dispatch for extracted tournament data."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path


_REPORT_TARGETS = ("index.html", "summary.json", "figures", "tables")


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


def generate_study_report(
    input_dir: Path,
    output_dir: Path | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Generate the report registered for an extracted tournament's game."""

    input_dir = input_dir.resolve()
    manifest_path = input_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"extraction manifest not found: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid extraction manifest: {error.msg}") from error
    if not isinstance(manifest, dict):
        raise TypeError("extraction manifest must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise ValueError("report supports extraction schema_version 1")

    game = manifest.get("game")
    if not isinstance(game, str):
        raise TypeError("extraction manifest game must be a string")
    if game != "boop":
        raise ValueError(f"tournament report is not available for {game}")

    if output_dir is None:
        output_dir = (
            input_dir.parent / "report"
            if input_dir.name == "data"
            else input_dir / "report"
        )
    else:
        output_dir = output_dir.resolve()

    existing = [output_dir / target for target in _REPORT_TARGETS if (output_dir / target).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "report output already exists; use --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )

    try:
        from .boop import generate_boop_report
    except ModuleNotFoundError as error:
        if error.name in {"matplotlib", "numpy", "pandas", "seaborn"}:
            raise RuntimeError(
                "report dependencies are not installed; install meeple-bots[report]"
            ) from error
        raise

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".report-", dir=output_dir.parent) as temporary:
        temporary_dir = Path(temporary)
        summary = generate_boop_report(input_dir, temporary_dir, manifest)
        output_dir.mkdir(parents=True, exist_ok=True)
        for target_name in _REPORT_TARGETS:
            source = temporary_dir / target_name
            target = output_dir / target_name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            source.replace(target)

    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "game": game,
        "complete": bool(manifest.get("complete", False)),
        "matches": summary["matches"],
        "figures": summary["figures"],
        "tables": summary["tables"],
    }
