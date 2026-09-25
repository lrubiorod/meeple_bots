"""CLI integration for fixed-position measurements."""
from pathlib import Path
import sys


def add_parser(subparsers):
    parser = subparsers.add_parser('probe', help='measure search behavior at fixed positions')
    parser.add_argument('operation', nargs='?', choices=('compare',))
    parser.add_argument('--game')
    parser.add_argument('--input', action='append', default=[], metavar='NAME=DIRECTORY', help='saved capture for offline comparison')
    parser.add_argument('--baseline', help='comparison control variant')
    parser.add_argument('--top-k', type=int, default=3, help='selected actions per variant shown in comparison')
    parser.add_argument('--list', action='store_true', help='list matching diagnostic cases')
    parser.add_argument('--suite', help='filter by suite/tag')
    parser.add_argument('--probe', help='full probe ID or short name (hyphens accepted)')
    parser.add_argument('--agent', choices=('mcts', 'so_ismcts'))
    parser.add_argument('--agent-config', type=Path, help='normal search-agent TOML profile')
    parser.add_argument('--variant', action='append', default=[], metavar='NAME=PATH', help='repeat for named agent TOML profiles')
    budgets = parser.add_mutually_exclusive_group()
    budgets.add_argument('--decision-time', type=float, help='fixed seconds/decision for throughput measurements')
    budgets.add_argument('--iterations', default='1000', help='comma-separated independent iteration budgets')
    parser.add_argument('--seeds', type=int, default=32, help='number of search seeds')
    parser.add_argument('--seed', type=int, default=0, help='first search seed; never changes fixtures')
    parser.add_argument('--output', type=Path, help='new output directory; never overwritten')


def run(args):
    if args.operation == 'compare':
        from .compare import write_comparison
        from .report import render_comparison
        inputs = {}
        for value in args.input:
            name, separator, path = value.partition('=')
            if not separator or not name.strip() or not path or name in inputs:
                raise ValueError('--input requires unique NAME=DIRECTORY values')
            inputs[name] = Path(path)
        if args.output is None:
            raise ValueError('--output is required for comparison')
        data = write_comparison(inputs, output=args.output, baseline=args.baseline, top_k=args.top_k)
        print(render_comparison(data), end='')
        print(f'Results: {args.output}')
        return 0
    if not args.game:
        raise ValueError('--game is required for probe measurements')
    if args.input or args.baseline:
        raise ValueError('--input/--baseline require probe compare')
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
    if args.variant:
        if args.agent_config:
            raise ValueError('--variant and --agent-config are mutually exclusive')
        from .runner import run_variants
        from .report import render_variants
        variants = {}
        for specification in args.variant:
            name, separator, path = specification.partition('=')
            if not separator or name in variants:
                raise ValueError('variants require unique NAME=PATH values')
            agent = load_search_profile(Path(path))
            resolve_family(args.game, args.agent, agent)
            variants[name] = agent
        summaries = run_variants(cases, variants, output=args.output,
            iterations=tuple(int(n.strip()) for n in args.iterations.split(',')),
            decision_seconds=args.decision_time, seeds=args.seeds, seed=args.seed,
            progress=lambda message: print(message, file=sys.stderr, flush=True))
        print(render_variants(summaries), end='')
        print(f'Results: {args.output}')
        return 0
    baseline = load_search_profile(args.agent_config) if args.agent_config else None
    family = resolve_family(args.game, args.agent, baseline)
    agent = baseline or (SoIsmctsAgent() if family == 'so_ismcts' else MctsAgent())
    summary = run_probes(cases, agent, iterations=tuple(int(n.strip()) for n in args.iterations.split(',')),
                         seeds=args.seeds, seed=args.seed, output=args.output, decision_seconds=args.decision_time,
                         progress=lambda message: print(message, file=sys.stderr, flush=True))
    print(render(summary), end='')
    print(f'Results: {args.output}')
    return 0
