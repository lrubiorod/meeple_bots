"""Phase 1 entrypoint, configuration, and import-boundary characterization."""

import ast
import importlib.metadata
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import tomllib
import unittest
from unittest.mock import patch

from meeple_bots import MctsAgent
from meeple_bots._search_profiles import load_named_search_profile, load_search_profile
from meeple_bots.cli import build_parser
from meeple_bots.tournament_config import _load_tournament_config, tournament_config_from_values


class CliOwnershipTests(unittest.TestCase):
    def test_module_and_installed_console_entrypoints_have_same_help(self):
        target = next(entry.value for entry in importlib.metadata.distribution('meeple-bots').entry_points
                      if entry.name == 'meeple-bots')
        self.assertEqual(target, 'meeple_bots.cli:main')
        module = subprocess.run([sys.executable, '-m', 'meeple_bots', '--help'],
                                capture_output=True, text=True, check=True)
        console = subprocess.run([str(Path(sys.executable).with_name('meeple-bots')), '--help'],
                                 capture_output=True, text=True, check=True)
        self.assertEqual(module.stdout, console.stdout)
        self.assertIn('tournament', module.stdout)

    def test_argparse_aliases_and_removed_option_remain_visible(self):
        parser = build_parser()
        match = parser.parse_args(['match', '--first-agent-config', 'a.toml',
                                   '--second-mcts-config', 'b.toml'])
        self.assertEqual(match.first_mcts_config, Path('a.toml'))
        self.assertEqual(match.second_mcts_config, Path('b.toml'))
        study = parser.parse_args(['study', '--game', 'spotf', '--max-pairs', '4',
                                   '--screening-time', '0.1', '--game-param', 'x=2'])
        self.assertEqual(study.max_pairs, 4)
        self.assertEqual(study.screening_time, .1)
        self.assertEqual(study.game_params, {'x': 2})

    def test_configuration_layers_do_not_import_cli(self):
        source = Path(__file__).parents[1] / 'src' / 'meeple_bots'
        for name in ('tournament_config.py', '_mcts_profiles.py', '_search_profiles.py',
                     'game_config.py'):
            tree = ast.parse((source / name).read_text())
            imports = [node.module for node in ast.walk(tree)
                       if isinstance(node, ast.ImportFrom)]
            self.assertFalse(any(module and module.endswith('cli') for module in imports), name)
        self.assertLessEqual(len((source / '__main__.py').read_text().splitlines()), 6)

    def test_normal_cli_import_does_not_require_report_extras(self):
        script = '''import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pandas', 'matplotlib', 'seaborn'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
import meeple_bots.cli
assert not {'pandas', 'matplotlib', 'seaborn'} & set(sys.modules)
'''
        subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=True)

    def test_search_profile_decoding_reads_toml_once(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'profile.toml'
            path.write_text('name = "named"\niterations = 8\nrollout_depth = 9\n')
            original = Path.read_text
            reads = []
            def tracked(file, *args, **kwargs):
                reads.append(file)
                return original(file, *args, **kwargs)
            with patch.object(Path, 'read_text', tracked):
                profile = load_named_search_profile(path)
            self.assertEqual(reads, [path])
            self.assertEqual(profile.name, 'named')
            self.assertEqual(profile.agent, MctsAgent(iterations=8, rollout_depth=9))
            self.assertEqual(load_search_profile(path), profile.agent)

    def test_tournament_file_and_mapping_decode_with_relative_output(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'plan.toml'
            path.write_text('''game = "boop"
output = "trace.jsonl"
seat_mode = "paired"
matches_per_pair = 2
[[agents]]
name = "random"
kind = "random"
[[agents]]
name = "search"
kind = "mcts"
iterations = [4, 8]
rollout_depth = 9
''')
            from_file = _load_tournament_config(path)
            from_values = tournament_config_from_values(tomllib.loads(path.read_text()), path)
            self.assertEqual(from_file.output, Path(tmp) / 'trace.jsonl')
            self.assertEqual(from_file.output, from_values.output)
            self.assertEqual([a.name for a in from_file.agents],
                             ['random', 'search-i4', 'search-i8'])
            self.assertEqual([(a.name, a.grid_position, a.agent) for a in from_file.agents],
                             [(a.name, a.grid_position, a.agent) for a in from_values.agents])


if __name__ == '__main__':
    unittest.main()
