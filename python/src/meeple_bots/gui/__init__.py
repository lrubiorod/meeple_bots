"""Local graphical interfaces for playing and watching Meeple Bots games."""

from .player import GuiPlayer
from .server import run_gui, serve_gui

__all__ = ["GuiPlayer", "run_gui", "serve_gui"]
