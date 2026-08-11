# Meeple Bots - Codex Instructions

## General working style

- Always communicate with me in Spanish, including explanations, plans, approval requests, summaries, and error descriptions, even if these instructions or the project code are written in English.
- Keep code, identifiers, commit messages, and technical documentation in English unless I explicitly ask otherwise.
- Before implementing a non-trivial change, explain the proposed approach first.
- Keep changes small, focused, and limited to the current task.
- Do not modify unrelated files.
- Prefer simple and explicit solutions over premature abstractions.
- Explain new Rust concepts in simple Spanish when they appear.
- Do not make Git commits or push changes to GitHub unless I explicitly request it.

## Repository boundaries

- Treat the repository root as the normal writable workspace.
- Do not access or modify files outside the repository unless it is genuinely necessary.
- If access outside the repository is required, explain why and request approval first.
- Do not use unrestricted or full-access permissions unless I explicitly request it.

## Approval requests

Whenever you need my approval to run a command or perform an action, explain it briefly in Spanish before requesting approval.

Include:

1. What the command or action does.
2. Why it is needed.
3. Whether it only reads information or whether it writes, deletes, renames, installs software, accesses the network, or affects anything outside the repository.
4. The main risk, if any.
5. Whether you recommend approving it.

Keep the explanation short and understandable for someone learning Rust and development tooling.

### Commands that modify the repository

Before requesting approval for a command that writes to the repository, also state:

- Which files will be affected.
- Whether each file will be created, modified, renamed, or deleted.
- A short summary of the intended change.
- Whether the operation is easily reversible with Git.

For patching commands such as `apply_patch` or `git apply`, summarize the patch before requesting approval.

Do not assume that a write command should be permanently approved just because it is commonly used.

## Command safety

Treat commands approximately according to the following risk levels:

- Read-only commands such as `rg`, `cat`, `sed`, `grep`, `git status`, and `git diff` are normally low risk.
- Validation commands such as `cargo check`, `cargo test`, and `cargo fmt` are normally low risk, but explain them when approval is required.
- Commands that modify files, such as `apply_patch`, `git apply`, file creation, or code generation, require a clear explanation of the intended changes.
- Destructive commands such as `rm`, `git reset`, `git clean`, or operations that overwrite files require special care and explicit approval.
- Commands using `sudo`, modifying system configuration, installing software, accessing the network, or operating outside the repository always require an explanation and explicit approval.

## Git

- Do not create commits unless I explicitly request it.
- Do not push to any remote unless I explicitly request it.
- Do not rewrite Git history.
- Do not run destructive Git commands such as `git reset --hard` or `git clean` without explicit approval.
- Prefer using `git diff` and `git status` to show me what changed.

## Validation

After Rust changes, run the relevant checks when appropriate:

- `cargo fmt`
- `cargo check --workspace`
- `cargo test --workspace`

If a command fails:

- Explain the failure before making significant additional changes.
- Do not start a broad refactor just to make a failing command pass.
- Prefer the smallest reasonable fix.

## Learning-oriented explanations

When presenting implemented code:

- Explain the purpose of each changed file.
- Explain important Rust concepts introduced by the change.
- Point out relevant trade-offs or assumptions.
- Distinguish clearly between necessary design choices and choices that could reasonably be made differently.