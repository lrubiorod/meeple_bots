"""Version-1 probe capture and comparison artifact I/O, without aggregation."""
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from .core import action_dict, json_value


def capture_metadata(cases, positions, agent, iterations, decision_seconds, variant_name, seed, seeds):
    from .. import _native
    from ..serialization import agent_dict

    native_path = Path(_native.__file__)
    return {'version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'native_sha256': sha256(native_path.read_bytes()).hexdigest(),
            'iterations': list(iterations) if decision_seconds is None else [], 'decision_seconds': decision_seconds, 'variant': variant_name, 'search_seeds': list(range(seed, seed+seeds)),
            'agent': agent_dict('probe', agent), 'q_orientation': 'root_player',
            'fresh_search_per_run': True,
            'probes': [{'id': c.id, 'game': c.game, 'description': c.description, 'tags': c.tags,
                        'notes': c.notes, 'root_player': p.root_player,
                        'candidate_actions': [action_dict(a) for a in c.candidate_actions],
                        'observation': p.observation.to_dict() if hasattr(p.observation, 'to_dict') else asdict(p.observation) if is_dataclass(p.observation) else p.observation,
                        'fixture': p.fixture} for c, p in zip(cases, positions)]}


def write_capture_json(output, name, data):
    (output/name).write_text(json.dumps(json_value(data), indent=2, allow_nan=False) + '\n')


def open_runs(output):
    return (output/'runs.jsonl').open('x')


def append_run(raw, row):
    raw.write(json.dumps(row, allow_nan=False) + '\n')
    raw.flush()


def write_capture_report(output, text):
    (output/'report.txt').write_text(text)


def write_variant_artifacts(output, summaries, comparison):
    (output/'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False) + '\n')
    (output/'comparison.txt').write_text(comparison)


def load_capture(path, name):
    path = Path(path)
    meta = json.loads((path / 'metadata.json').read_text())
    if meta.get('version', 1) != 1:
        raise ValueError(f'{name}: unsupported capture schema version')
    if meta.get('q_orientation', 'root_player') != 'root_player':
        raise ValueError(f'{name}: incompatible Q orientation')
    warnings = [f'{name}: missing {field}; assuming legacy version-1 conventions'
                for field in ('q_orientation', 'version') if field not in meta]
    rows = [json.loads(line) for line in (path / 'runs.jsonl').read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'{name}: no saved runs')
    return meta, rows, warnings


def write_comparison_artifacts(output, data, report):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'comparison.json').write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    (output / 'comparison.txt').write_text(report)
