# Python compatibility baseline (Phase 0)

Study resume checks four independent contracts: the Study protocol and frozen
request, the persisted phase plan and tournament traces, the Python/native
engine fingerprint, and any explicit mixed-engine continuation. Protocol 23
and the frozen request fields (game, family, baseline, budgets, stages, seeds,
workers, and related options) are unchanged by Phase 0. A saved plan is resumed,
not regenerated. Tournament trace headers and completed job identities are
validated before remaining jobs run.

Resume is strict by default. New checkpoints record
`request.engine.fingerprint_algorithm = "python-tree-v2"`. This algorithm hashes
every production `.py` file recursively under the installed `meeple_bots`
package, sorted by POSIX-style relative path. Each path and its content are
length-delimited before SHA-256 hashing. Checkout location, traversal order,
timestamps, tests, docs, caches, `local/`, and `results/` do not contribute.
Moving or renaming source therefore changes the fingerprint intentionally. The
existing native-extension SHA-256 remains a separate required match. Git
revision is recorded for provenance but is not itself a resume gate.

Unversioned checkpoints retain the historical root-only Python algorithm:
SHA-256 over sorted root `*.py` files, each filename immediately followed by
its bytes. A legacy checkpoint can resume while its historical source and
native hashes match; it is not silently rewritten to the new algorithm. Future
source moves will normally reject such checkpoints. An unknown fingerprint
algorithm is rejected rather than guessed. `--resume --allow-engine-change`
explicitly accepts a known Python/native mismatch, records the prior and new
engines and completed games in `engine_changes`, and retains prior results as a
mixed-engine study. It does not waive protocol, request, plan, or trace checks,
and it is never enabled automatically for a source refactor.

The installed wheel contains the Python source tree and the native extension,
so the same runtime traversal works in editable and wheel installs. A
distribution mode that omits `.py` files is outside this runtime contract and
would need a packaged source manifest before support. Non-Python GUI baseline
resources and probe fixtures are not consumed by Study. External TOML agent
profiles are decoded into the frozen Study request; edits that change their
resolved values fail request compatibility independently of the source hash.

## Contracts for subsequent moves

| Contract | Classification | Phase 0 guard |
| --- | --- | --- |
| `meeple_bots`, `meeple_bots.api`, `meeple_bots.cli:main`, `python -m meeple_bots` | Public entrypoints | Import/CLI and existing command tests |
| CLI profile helpers, `_mcts_profiles`, `gui.controller.GuiController`, `gui.application.GuiApplication`, `reporting.common` documented in `development.md` | Documented compatibility | Import/profile and existing GUI/report tests; use temporary re-exports if moved |
| `_study_profiles` and other non-public paths used by tests | Private test seam | Preserve a deliberate seam during extraction; no indefinite public promise |
| Tournament trace schema 1, paired seed/seat/job identity | Persistence compatibility | Trace resume and pairing tests |
| Study protocol 23, request, phase IDs, 100,000 seed stride, completed contrasts, engine metadata | Persistence compatibility | Versioned plan fixture, resume and interrupted-run tests |
| Generic analyze JSON and legacy deterministic-MCTS projection | Persistence compatibility | Key-set fixture and analysis tests |
| Probe capture schema 1, all root actions, nullable Q/availability | Persistence compatibility | Minimal capture fixture and compare test |
| `studies.run_matches`, `studies._fingerprint` | Private test seams | `test_studies.py` monkeypatches them; a future move needs an explicit injection or stable forwarding seam |
| Private tuner/report helpers and internal module locations not documented above | Internal movable | No permanent import-path promise |

The fixtures characterize formats and identities, not exact native timings,
environment-specific hashes, strategic outcomes, or every private function.
Study comparison timing and statistical semantics, analyze estimates, and
probe measurements remain separate contracts; similar field names do not make
their implementations interchangeable.

## Phase 1 import locations

`meeple_bots.cli` is now a package. Its `main` and `build_parser` entrypoints
remain stable. The package initializer re-exports previously imported profile
helpers, tournament loaders, `_match_agent`, `_evaluation_dict`, and the test
seams `_batch_agent_dict`/`_tournament_pairings`. Their implementations live in
`_mcts_profiles`, `tournament_config`, `cli.commands`, `cli.analyze_compat`,
`serialization`, and `tournaments`, respectively. These private re-exports
serve existing tests and documented CLI helper paths; assess their consumers
in Phase 6 rather than treating them as permanent public APIs. No repository
test patches `meeple_bots.cli` globals; Study's separate monkeypatch seams
remain unchanged. Source relocation intentionally changes the strict Study
fingerprint, so historical resumes still require a matching engine or the
explicit mixed-engine override.
