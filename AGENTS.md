# Meeple Bots

## Purpose and architecture

Build a maintainable framework for comparing reusable agents across board games.
Rust owns authoritative rules and search; Python owns orchestration, interfaces and analysis.
Games and agents depend on core contracts, not on each other. Register concrete combinations
in the catalog/bindings. Keep game-specific rules out of generic agents and Python extractors.

## Scope and execution

- Complete the requested behavior with the smallest coherent change. Preserve unrelated work.
- Before editing, identify the affected path, acceptance criteria and relevant checks.
  Local fixes need no plan document or architecture review.
- Use `rg` on relevant files/symbols, then their callers and tests. Expand only to resolve
  concrete dependencies or uncertainty. Reuse established facts.
- Batch independent reads. Read relevant excerpts and failure summaries; avoid repeated full
  files, searches or successful test logs.
- Follow existing patterns. New abstractions, dependencies, options or compatibility machinery
  must serve the requested behavior or an existing public contract.
- Update affected callers and documentation. Mention unrelated findings without fixing them.
  Do not create audits or reports unless requested.
- Resolve routine choices autonomously; ask about material ambiguity in behavior, compatibility
  or scope. No subagents unless requested.

## Validation proportional to risk

Choose checks for what could break, not merely for the number of edited files:

| Change | Default validation |
| --- | --- |
| Documentation/comments | Review the diff and affected references; no builds or tests. |
| Python behavior | A focused unittest module or test-name filter using `.venv/bin/python`. |
| Rust behavior | `cargo fmt --all --check` and relevant `cargo test -p <package>` tests. |
| Rust consumed by Python | Rust checks above, rebuild the native extension, then relevant Python integration tests. |
| Shared contracts or broad refactor | Affected dependents; workspace suites when impact cannot be bounded reliably. |

- Add focused regressions for bugs and important boundaries. Prefer the existing suite;
  avoid tests that merely repeat implementation details.
- `cargo test` compiles its targets. Add `cargo check` only for targets/features not covered
  by the selected build. Do not automatically stack check, test, Clippy and full suites.
- Reuse the environment and build cache. Group Rust edits before rebuilding the extension;
  rebuild again only after further Rust changes needed by Python tests. Python-only edits
  need no native rebuild. Timing claims require release builds.
- Stop after relevant checks pass. Repeat or broaden only for new changes, failures or a
  specific unresolved risk; explain the risk before an expensive additional check.
- Report pre-existing failures separately; do not repair unrelated code to obtain a green run.
- Run benchmarks, tournaments or large datasets only when requested or necessary evidence.

## Repository boundaries

- Follow `.gitignore`: personal studies/scripts in `local/`, generated results in `results/`.
  Keep reusable regression tests versioned.
- Retain the shared Boop/SPOTF baselines and template-study reference. Do not change experimental
  parameters or other search techniques while implementing an isolated improvement.
- No commits, pushes, staging or history rewriting unless requested. Never discard user changes.
- Respect workspace permissions. Explain significant side effects when approval is required.

## Communication and completion

- Communicate in Spanish; keep code, identifiers and technical documentation in English.
- Give brief updates for findings, decisions or blockers; avoid narrating routine commands.
- Finish with the outcome, validation and material limitations; explain Rust concepts when useful.
  Give commit names or report updates when requested.
- Stop when the requested behavior is implemented and verified, or explain a concrete blocker.
  Do not continue with optional polishing or improvements.
