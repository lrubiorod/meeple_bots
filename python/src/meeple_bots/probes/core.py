"""Small observation-only contracts for behavioral measurements, not verdicts."""
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ProbePosition:
    observation: Any
    legal_actions: tuple
    root_player: int
    # Administrative provenance only; never passed to search.
    fixture: dict


@dataclass(frozen=True)
class ProbeCase:
    id: str
    game: str
    description: str
    tags: tuple[str, ...]
    builder: Callable[[], ProbePosition]
    action_label: Callable[[Any], str]
    candidate_actions: tuple = ()
    notes: str = ''
    search: Callable | None = None


def action_dict(action):
    """Use the game's existing action serialization."""
    if hasattr(action, 'to_dict'):
        action = action.to_dict()
    elif not isinstance(action, dict):
        from ..serialization import action_dict as serialize
        action = serialize(action)
    return json_value(action)


def json_value(value):
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, bytes)):
        return [json_value(v) for v in value]
    return value
