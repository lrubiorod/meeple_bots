"""Stable tournament-report facade; optional renderers load on demand."""

from .base import read_analysis_csv, decision_timing_description, wilson_interval
from .dispatch import _REPORT_TARGETS, generate_study_report
