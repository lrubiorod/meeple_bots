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
| Calibrate a compatible search-agent family | [Automatic diagnosis](studies.md#incremental-mcts-study) |
| Measure structure and search cost | [Analysis](../crates/evaluation/README.md) |
| Diagnose search choices at fixed positions | [Behavioral probes](probes.md) |
| Configure profiles and tournaments | [Study reference](studies.md#run-a-study) |
| Extract and validate traces | [Extraction](studies.md#2-extract-analysis-tables) |
| Generate generic, Boop or SPOTF reports | [Reports](studies.md#3-generate-a-tournament-report) |
| Understand timing and artifacts | [Study reference](studies.md#decision-timing) |
| Understand module boundaries or troubleshoot | [Development](development.md#package-boundary) |

Commands assume the repository root and an activated `.venv`. Run
`meeple-bots COMMAND --help` for options in the installed version.

After changing any Rust crate used by Python, rebuild with `maturin develop --release`.
An editable Python installation does not rebuild the native extension automatically.

Lost Cities is available as `LostCities` / CLI `lost_cities`, with Random and
SO-ISMCTS agents. `analyze` and `study` automatically resolve its compatible SO-ISMCTS
family; ordinary MCTS is rejected. The debugging GUI shows both hands, while search
receives only its player observation. See the [game guide](../games/lost-cities/README.md)
for the single-round variant and the supported match, batch and tournament interfaces.
