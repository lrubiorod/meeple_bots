"""Splendor adapter for the native step-session GUI lifecycle."""

from pathlib import Path
from uuid import uuid4

from ....splendor import SplendorSession
from ....gui.baselines import SPLENDOR_BASELINE
from ....gui.step_session import NativeStepGui


class SplendorGui(NativeStepGui):
    def __init__(self, trace_dir=Path("results/gui/splendor")):
        super().__init__(
            trace_dir, session_type=SplendorSession, baseline=SPLENDOR_BASELINE,
            finished=lambda state: state["finished"],
            trace_slug="splendor", trace_format="splendor_session_v1",
            session_id=uuid4().hex,
        )

    def submit_move(self, action, turn, session_id):
        self._submit_move(action, turn, session_id)
