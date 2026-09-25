"""Compatibility surface for Study evidence/reporting and shared work metrics."""
from importlib import import_module
from .search_metrics import quantile, search_adequacy


_REPORT_NAMES = ('mechanism_effects', '_fmt', '_table', '_curve', 'cutoff_screening', 'study_diagnostics', 'write_study_report', 'family_study_diagnostics', 'write_family_study_report')
_EVIDENCE_NAMES = ('_timings', 'paired_interval', 'summarize_contrast')


def __getattr__(name):
    if name in _EVIDENCE_NAMES:
        return getattr(import_module('.studies.race', __package__), name)
    if name in _REPORT_NAMES:
        return getattr(import_module('.studies.report', __package__), name)
    raise AttributeError(name)


def __dir__():
    return sorted(set(globals()) | set(_EVIDENCE_NAMES) | set(_REPORT_NAMES))
