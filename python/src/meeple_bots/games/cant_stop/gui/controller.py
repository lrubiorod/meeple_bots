"""Can't Stop adapter for the native step-session GUI lifecycle."""

from pathlib import Path

from .. import CantStopSession
from ....gui.baselines import CANT_STOP_BASELINE
from ....gui.step_session import NativeStepGui


class CantStopGui(NativeStepGui):
    def __init__(self, trace_dir=Path("results/gui/cant-stop")):
        super().__init__(
            trace_dir, session_type=CantStopSession, baseline=CANT_STOP_BASELINE,
            finished=lambda state: state["winner"] is not None,
            trace_slug="cant-stop", trace_format="cant_stop_session_v1",
        )

    def submit_move(self, action, turn):
        self._submit_move(action, turn)
