"""CLI integration for fixed-position measurements."""
from pathlib import Path
import sys


def add_parser(subparsers):
    parser = subparsers.add_parser('probe', help='measure search behavior at fixed positions')
    parser.add_argument('--game', required=True)
    parser.add_argument('--list', action='store_true', help='list matching diagnostic cases')
    parser.add_argument('--suite', help='filter by suite/tag')
    parser.add_argument('--probe', help='full probe ID or short name (hyphens accepted)')
    parser.add_argument('--agent', choices=('mcts', 'so_ismcts'))
    parser.add_argument('--agent-config', type=Path, help='normal search-agent TOML profile')
    parser.add_argument('--iterations', default='1000', help='comma-separated independent iteration budgets')
    parser.add_argument('--seeds', type=int, default=32, help='number of search seeds')
    parser.add_argument('--seed', type=int, default=0, help='first search seed; never changes fixtures')
    parser.add_argument('--output', type=Path, help='new output directory; never overwritten')


def run(args):
    from .._agent_config import MctsAgent, SoIsmctsAgent
    from .._search_profiles import load_search_profile, resolve_family
    from .registry import select_cases
    from .runner import run_probes
    from .report import render
    cases = select_cases(game=args.game, suite=args.suite, probe=args.probe)
    if args.list:
        for case in cases:
            print(f'{case.id} [{", ".join(case.tags)}]\n  {case.description}')
        return 0
    if args.output is None:
        raise ValueError('--output is required for probe measurements')
    baseline = load_search_profile(args.agent_config) if args.agent_config else None
    family = resolve_family(args.game, args.agent, baseline)
    agent = baseline or (SoIsmctsAgent() if family == 'so_ismcts' else MctsAgent())
    summary = run_probes(cases, agent, iterations=tuple(int(n.strip()) for n in args.iterations.split(',')),
                         seeds=args.seeds, seed=args.seed, output=args.output,
                         progress=lambda message: print(message, file=sys.stderr, flush=True))
    print(render(summary), end='')
    print(f'Results: {args.output}')
    return 0
