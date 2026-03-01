# Architecture and Dependency Notes

## High-Level Components

- `src/redteam_llm/cli.py`: CLI entrypoints for standard red-team runs and G2PIA runs.
- `src/redteam_llm/orchestrator.py`: Coordinates attacker, target model calls, and judge evaluation.
- `src/redteam_llm/attacker.py`: Creates adversarial prompt candidates for iterative testing.
- `src/redteam_llm/judge.py`: Parses and normalizes safety assessment output.
- `src/redteam/modules/g2pia_defensive.py`: Defensive G2PIA simulation and report generation.
- `dashboard/app.py`: Flask dashboard that renders `results.json`.

## Data Flow

1. User starts CLI command (`redteam ...`).
2. Config is loaded from `config.yaml`.
3. Orchestrator runs attack/evaluation loop and produces result records.
4. Results are written to JSON output files.
5. Dashboard reads JSON output and renders aggregate statistics.

## Dependency Strategy

Canonical package metadata lives in `pyproject.toml`.

- Core runtime dependencies are declared in `[project.dependencies]`.
- Heavy optional semantic dependencies for G2PIA are grouped under `[project.optional-dependencies.g2pia]`.
- Developer tooling dependencies are grouped under `[project.optional-dependencies.dev]`.

`requirements.txt` mirrors these groups for pip-oriented environments.

## Quality Tooling

- Lint: `ruff`
- Tests: `pytest` + `pytest-cov`
- Type-check (informational baseline): `mypy`
- CI workflow: `.github/workflows/ci.yml`
