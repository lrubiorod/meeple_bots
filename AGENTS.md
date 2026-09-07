# Meeple Bots - Codex Instructions

## Communication

- Always communicate with me in Spanish, including plans, explanations, approval requests, summaries, and errors.
- Keep code, identifiers, commit messages, and technical documentation in English unless I explicitly ask otherwise.
- Keep explanations concise and focused on information useful for understanding or reviewing the change.
- Explain new or non-obvious Rust concepts in simple Spanish when they are relevant.
- Do not restate the task or describe obvious code line by line.

## Working style

- Prefer the smallest change that fully solves the requested task.
- Keep changes focused on the current task and do not modify unrelated files.
- Prefer simple and explicit solutions over premature abstractions.
- Preserve the existing architecture unless the task genuinely requires changing it.
- Do not implement speculative improvements or adjacent features that were not requested. Mention them instead if they are worth considering.
- For straightforward localized tasks, implement directly.
- Before a non-trivial or architectural change, briefly explain the intended approach, then proceed unless a material ambiguity or required approval blocks the work.
- Infer routine details from the repository and existing patterns instead of asking unnecessary clarification questions.

## Repository exploration

- Treat the repository root as the normal workspace.
- Stay inside the repository unless access outside it is genuinely necessary.
- Prefer targeted searches with `rg` and specific files or symbols over broad repository scans.
- Once the relevant implementation has been located, do not continue exploring unrelated directories.
- Do not repeatedly reread unchanged files.
- Read project documentation only when it is relevant to the current task.

## Scope and safety

- Do not access or modify files outside the repository unless necessary for the requested task.
- If outside access, network access, software installation, system configuration, or another significant side effect requires approval, explain briefly why it is needed.
- When approval is required, keep the explanation short: state what the command does, its relevant side effect or risk, and whether you recommend approving it.
- Be especially careful with destructive commands such as `rm`, `git reset`, `git clean`, or commands that overwrite existing work.
- Never use destructive Git operations on user changes unless explicitly requested.

## Git

- Do not create commits unless I explicitly request it.
- Do not push to any remote unless I explicitly request it.
- Do not rewrite Git history unless I explicitly request it.
- Preserve existing uncommitted user changes.
- Use `git status` and `git diff` when they are useful for reviewing the work, but do not repeatedly run them without a reason.

## Validation

- Run the smallest relevant validation for the change first.
- Prefer targeted package or test commands over full-workspace checks for localized changes.
- Run broader workspace checks when shared/core behavior changed, several crates are affected, targeted checks fail, or the task explicitly requires full validation.
- Do not repeat a successful test or check unless code changed afterward or a new concern justifies it.
- Do not run benchmarks, tournaments, long simulations, or large data analyses unless requested or necessary for the task.
- Do not add tests that merely duplicate implementation details; add tests when they meaningfully protect behavior or reproduce a bug.

Typical Rust validation, chosen according to scope:

- Localized crate change: `cargo check -p <package>` and relevant `cargo test -p <package>`.
- Formatting when Rust files changed: `cargo fmt`.
- Broad/shared change when justified: `cargo check --workspace` and `cargo test --workspace`.

If validation fails:

- Identify the cause before making substantial additional changes.
- Prefer the smallest reasonable fix.
- Do not start an unrelated refactor just to make a failing check pass.
- Report unresolved failures clearly.

## Repository architecture

The Rust workspace contains shared infrastructure, games, agents, and Python bindings. Keep performance-sensitive game simulation and search logic in Rust; keep orchestration, experiment configuration, tournament analysis, and reporting in Python unless the existing architecture indicates otherwise.

When changing generic MCTS infrastructure, avoid game-specific branching in the generic search code. Prefer traits, policies, evaluators, or existing extension points when game-specific behavior is genuinely required.

## Completion

When the requested work is complete:

- Stop once the relevant implementation and validation are complete.
- Give a concise summary of what changed.
- State which tests or checks were run and whether they passed.
- Mention important trade-offs or unresolved issues only when relevant.
- Do not propose or implement additional work unless it materially affects the requested task.