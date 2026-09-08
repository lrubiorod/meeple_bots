"""Shared graphical player configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..api import MctsAgent

PlayerKind = Literal["human", "random", "mcts"]


@dataclass(frozen=True, slots=True)
class GuiPlayer:
    """Configuration for one player shown in a graphical interface."""

    kind: PlayerKind
    iterations: int | None = 1_000
    time_budget: float | None = None
    exploration: float = 2.0**0.5
    rollout_depth: int = 256
    heuristic: int | None = None
    tree_reuse: bool = False
    transpositions: bool = False
    selection_policy: str = "uct"

    def __post_init__(self) -> None:
        if self.kind not in ("human", "random", "mcts"):
            raise ValueError("player kind must be human, random, or mcts")
        if self.kind == "mcts":
            MctsAgent(
                iterations=self.iterations,
                time_budget=self.time_budget,
                exploration=self.exploration,
                rollout_depth=self.rollout_depth,
                heuristic=self.heuristic,
                tree_reuse=self.tree_reuse,
                transpositions=self.transpositions,
                selection_policy=self.selection_policy,
            )

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-compatible representation used by GUI state."""

        return {
            "kind": self.kind,
            "iterations": self.iterations,
            "time_budget": self.time_budget,
            "exploration": self.exploration,
            "rollout_depth": self.rollout_depth,
            "heuristic": self.heuristic,
            "tree_reuse": self.tree_reuse,
            "transpositions": self.transpositions,
            "selection_policy": self.selection_policy,
        }


def parse_gui_player(
    raw: Any,
    name: str,
    *,
    default_rollout_depth: int,
    available_heuristics: tuple[int, ...] = (),
    default_mcts: GuiPlayer | None = None,
) -> GuiPlayer:
    """Validate one player configuration received from a browser."""

    if not isinstance(raw, dict):
        raise ValueError(f"{name} player configuration must be an object")
    if raw.get("kind") == "mcts" and default_mcts is not None:
        defaults = default_mcts.as_dict()
        # Supplying either budget selects that mode; never inherit a second budget.
        if "iterations" in raw or "time_budget" in raw:
            defaults["iterations"] = None
            defaults["time_budget"] = None
        raw = {**defaults, **raw}
    heuristic = raw.get("heuristic")
    if heuristic is not None:
        if isinstance(heuristic, bool) or not isinstance(heuristic, int):
            raise ValueError(f"{name} heuristic must be an integer or null")
        if heuristic not in available_heuristics:
            available = ", ".join(str(index) for index in available_heuristics) or "none"
            raise ValueError(f"{name} heuristic must be one of: {available}")
    return GuiPlayer(
        kind=raw.get("kind"),
        iterations=raw.get("iterations", 1_000),
        time_budget=raw.get("time_budget"),
        exploration=raw.get("exploration", 2.0**0.5),
        rollout_depth=raw.get("rollout_depth", default_rollout_depth),
        heuristic=heuristic,
        tree_reuse=raw.get("tree_reuse", False),
        transpositions=raw.get("transpositions", False),
        selection_policy=raw.get("selection_policy", "uct"),
    )
