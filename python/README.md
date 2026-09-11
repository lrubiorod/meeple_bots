# Python interface

[Back to the project overview](../README.md)

The public `meeple_bots` package provides typed games, agents, matches and results.
Rust owns rules and search; Python provides orchestration, interfaces and analysis.

| Task | Guide |
| --- | --- |
| Install or rebuild the extension | [Development](development.md#install-for-development) |
| Use the Python API | [API usage](usage.md#python-api) |
| Play or watch in the browser | [GUI](usage.md#play-or-watch) |
| Run CLI commands | [Command-line workflows](usage.md#command-line-workflows) |
| Generate an automatic MCTS study | [Automatic diagnosis](studies.md#automatic-mcts-diagnosis) |
| Configure profiles and tournaments | [Study reference](studies.md#run-a-study) |
| Extract and validate traces | [Extraction](studies.md#2-extract-analysis-tables) |
| Generate generic, Boop or SPOTF reports | [Reports](studies.md#3-generate-a-tournament-report) |
| Understand timing and artifacts | [Study reference](studies.md#decision-timing) |
| Understand module boundaries or troubleshoot | [Development](development.md#package-boundary) |

Commands assume the repository root and an activated `.venv`. Run
`meeple-bots COMMAND --help` for options in the installed version.

After changing any Rust crate used by Python, rebuild with `maturin develop --release`.
An editable Python installation does not rebuild the native extension automatically.
