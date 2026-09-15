"""Load GUI startup settings from the shared reference TOML profiles."""

from importlib.metadata import distribution
from pathlib import Path

from .._mcts_profiles import _load_mcts_profile
from .._agent_config import MctsAgent
from .player import GuiPlayer, ConfiguredGuiPlayer


def _baseline_path(filename: str) -> Path:
    # Resolve from this source tree, never from the caller's working directory.
    repository = Path(__file__).resolve().parents[4]
    if (repository / "pyproject.toml").is_file():
        return repository / "configs" / "mcts" / filename
    return Path(distribution("meeple-bots").locate_file(f"configs/mcts/{filename}"))


def _load_gui_baseline(filename: str) -> GuiPlayer:
    path = _baseline_path(filename)
    agent = _load_mcts_profile(path).agent
    player_type = ConfiguredGuiPlayer if filename in ("cant-stop-baseline.toml", "splendor-baseline.toml") else GuiPlayer
    extra = {} if player_type is GuiPlayer else {
        "rollout_policy": agent.rollout_policy, "progressive_bias": agent.progressive_bias,
        "root_diagnostics": agent.root_diagnostics,
    }
    player = player_type(
        "mcts", **extra, iterations=agent.iterations, time_budget=agent.time_budget,
        exploration=agent.exploration, rollout_depth=agent.rollout_depth,
        heuristic=agent.heuristic, tree_reuse=agent.tree_reuse,
        transpositions=agent.transpositions,
        selection_policy=agent.selection_policy,
        rave_equivalence=agent.rave_equivalence,
        progressive_widening=agent.progressive_widening,
        progressive_widening_k=agent.progressive_widening_k,
        progressive_widening_alpha=agent.progressive_widening_alpha,

    )
    # Do not silently discard policies or evaluator parameters absent from the GUI.
    represented = player.to_agent() if isinstance(player, ConfiguredGuiPlayer) else MctsAgent(**{k: v for k, v in player.as_dict().items() if k != "kind"})
    if represented != agent:
        raise ValueError(f"GUI baseline {path} uses settings not supported by the GUI")
    return player


TIC_TAC_TOE_BASELINE = _load_gui_baseline("tic-tac-toe-baseline.toml")
CONNECT_FOUR_BASELINE = _load_gui_baseline("connect-four-baseline.toml")
BOOP_BASELINE = _load_gui_baseline("boop-baseline.toml")
SPOTF_BASELINE = _load_gui_baseline("spotf-baseline.toml")

CANT_STOP_BASELINE = _load_gui_baseline("cant-stop-baseline.toml")

SPLENDOR_BASELINE = _load_gui_baseline("splendor-baseline.toml")

CONNECT6_BASELINE = _load_gui_baseline("connect6-baseline.toml")
