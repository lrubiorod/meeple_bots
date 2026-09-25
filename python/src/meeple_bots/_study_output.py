"""Private compatibility wrapper for Study terminal output."""
from .studies.report import compute_budget, extension_notice, announce_plan as render_plan


def announce_plan(runner):
    render_plan(runner.state, runner.family, runner.family_profile, runner.phase_names,
                runner.base, runner.budget, runner.progress)


def announce_extension(runner, phase, completed=False):
    message, limit_reached = extension_notice(runner.state, phase, completed)
    if limit_reached:
        phase['extension_limit_reached'] = True
    if message:
        runner.progress(message)
