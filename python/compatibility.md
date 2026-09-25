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

## Phase 2 Study ownership

`meeple_bots.studies` is a lazy compatibility facade for `StudyRunner`,
`run_study`, and the existing Study helper imports. The implementation now
lives in `studies/coordinator.py`, `planning.py`, `profiles.py`, `tuners.py`,
`calibration.py`, `race.py`, `budget.py`, `persistence.py`, and `report.py`.
The protocol remains 23; stage IDs, extension limit 3, seed stride 100,000,
paired evidence, and frozen request are unchanged. Source relocation changes
the strict `python-tree-v2` hash intentionally. Old checkpoints require their
original source tree or an explicit mixed-engine continuation.

`_study_profiles`, `_study_tuners`, and `_study_output` remain private
compatibility paths with current CLI, test, or documented consumers. Their classes
and generator functions re-export the authoritative Study definitions. The
old output wrapper also preserves its historical `announce_extension(runner,
phase)` state update; Study itself uses the pure report calculation and commits
the flag in the coordinator. `study_analysis` re-exports paired evidence,
Study report functions, and the neutral `search_metrics` functions. The public
Study facade remains stable. Tests patch actual dependencies at
`studies.coordinator.run_matches`, `studies.coordinator._fingerprint`, and
`studies.planning.proposals` after relocation.

## Phase 1 import locations

`meeple_bots.cli` is now a package. Its `main` and `build_parser` entrypoints
remain stable. The package initializer re-exports previously imported profile
helpers, tournament loaders, `_match_agent`, `_evaluation_dict`, and the test
seams `_batch_agent_dict`/`_tournament_pairings`. Their implementations live in
`_mcts_profiles`, `tournament_config`, `cli.commands`, `analysis.report`,
`serialization`, and `tournaments`, respectively. These private re-exports
serve existing tests and documented CLI helper paths. The intermediate private
`cli.analyze_compat` module was retired in Phase 6; `_evaluation_dict` remains
available from `meeple_bots.cli`. No repository
test patches `meeple_bots.cli` globals; Study's separate monkeypatch seams
remain unchanged. Source relocation intentionally changes the strict Study
fingerprint, so historical resumes still require a matching engine or the
explicit mixed-engine override.

## Phase 3 Analyze and Probe ownership

`meeple_bots.analysis` remains the public facade. `analysis.measurement` owns
sampling, horizon/calibration and search summaries; `analysis.report` owns text,
generic JSON and the supported legacy deterministic-MCTS projection. The former
does not import the latter. `meeple_bots.cli._evaluation_dict` directly re-exports
the legacy projection from `analysis.report`. The Analyze
sampled-full horizon remains distinct from Study's practical terminal horizon.

`search_metrics` now defines only `quantile` and `search_adequacy`. Old
`study_analysis` imports re-export those same functions; Study imports the
neutral owner directly. The existing workflow-specific result keys and units
remain unchanged.

`probes.metrics` now defines canonical action identity, aggregation and
important-action selection. `probes.report` re-exports those names for current
tests and local users. `probes.artifacts` owns version-1 capture reads/writes and
comparison artifact writes. `probes.compare` computes descriptive differences;
`probes.report` renders them. The historical `probes.compare.render_comparison`
import lazily re-exports the same function for current test consumers.
Existing captures are read without migration or regeneration, and
comparison remains search-free. Native-free package import remains outside the
current contract.

## Phase 4 API, trace, extraction and reporting ownership

`meeple_bots` and `meeple_bots.api` keep their public class/function import
paths. The authoritative match/batch result classes live in `matches.models`,
`Match`/`Batch` in `matches.execution`, and human callbacks in `matches.human`;
old paths re-export the identical objects. `native_config` and `native_bridge`
own Rust configuration and game-aware conversion. `game_config` still exports
`normalize_game_parameters` from the lower-level `_game_parameters` owner.

`matches.trace` owns version-1 action, agent and result codecs plus tournament
trace validation and resume I/O. `serialization` and `tournaments` retain their
historical imports as re-exports of the same functions/classes. The schema,
`agent_total_v1` timing meaning, append/flush behavior and strict resume
validation are unchanged. `extraction` keeps `extract_tournament` as a facade;
the pipeline, schema and game adapters preserve the CSV and manifest contracts.
`reporting` keeps `generate_study_report` and its helper imports as a facade;
renderers now depend on `reporting.base`/`reporting.common`, below dispatch.
These compatibility paths have live tests and callers; retain the public facades.
Pure source relocation changes Study's strict fingerprint as documented above.
