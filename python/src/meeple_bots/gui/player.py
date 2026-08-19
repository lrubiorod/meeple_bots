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
    iterations: int = 1_000
    exploration: float = 2.0**0.5
    rollout_depth: int = 256

    def __post_init__(self) -> None:
        if self.kind not in ("human", "random", "mcts"):
            raise ValueError("player kind must be human, random, or mcts")
        if self.kind == "mcts":
            MctsAgent(
                iterations=self.iterations,
                exploration=self.exploration,
                rollout_depth=self.rollout_depth,
            )

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-compatible representation used by GUI state."""

        return {
            "kind": self.kind,
            "iterations": self.iterations,
            "exploration": self.exploration,
            "rollout_depth": self.rollout_depth,
        }


def parse_gui_player(
    raw: Any,
    name: str,
    *,
    default_rollout_depth: int,
) -> GuiPlayer:
    """Validate one player configuration received from a browser."""

    if not isinstance(raw, dict):
        raise ValueError(f"{name} player configuration must be an object")
    return GuiPlayer(
        kind=raw.get("kind"),
        iterations=raw.get("iterations", 1_000),
        exploration=raw.get("exploration", 2.0**0.5),
        rollout_depth=raw.get("rollout_depth", default_rollout_depth),
    )
