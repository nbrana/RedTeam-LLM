# RedTeam-LLM

Local red-team evaluation framework for LLM safety testing with an Ollama-first workflow.

## What This Repository Contains

- A CLI-driven red-team loop with attacker, orchestrator, and judge modules.
- A defensive G2PIA simulation harness for robustness benchmarking.
- A lightweight Flask dashboard for browsing `results.json`.
- Example scripts, notebooks, and test fixtures for local experimentation.

## Quickstart (Recommended)

1. Install and start [Ollama](https://ollama.com), then pull a local model:
   ```bash
   ollama pull llama3
   ```
2. Create an environment and install this project:
   ```bash
   uv sync --extra dev
   ```
3. Run quality checks:
   ```bash
   uv run make ci
   ```
4. Run a red-team session:
   ```bash
   uv run python main.py run -c config.yaml -o results.json
   ```
5. Launch the dashboard:
   ```bash
   uv run python dashboard/app.py
   ```
   Open `http://localhost:5000`.

## Advanced Setup (pip workflow)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Optional heavy dependencies for richer G2PIA semantics are included in `requirements.txt`.

## CLI Usage

Run default session:
```bash
redteam run --config config.yaml --output results.json
```

Run defensive G2PIA simulation:
```bash
redteam run-g2pia \
  --dataset tests/data/g2pia_synthetic.jsonl \
  --victim-model phi3:mini \
  --candidates 5 \
  --allow-adversarial-sim \
  --out reports/g2pia_lab_run.json
```

## Developer Commands

`Makefile` commands:

- `make install-dev` installs editable package + dev tooling.
- `make lint` runs `ruff`.
- `make test` runs `pytest` with coverage.
- `make typecheck` runs `mypy` (informational baseline).
- `make ci` runs lint + tests + type-check locally.

## Safety Notes

- The G2PIA path is defensive and measurement-focused; adversarial candidate generation is off by default.
- Enable adversarial simulation only with `--allow-adversarial-sim` in controlled environments.

## Documentation

- Methodology and safety checklist: `docs/g2pia_readme.md`
- Architecture and dependency guide: `docs/architecture.md`
- Windows setup: `docs/windows_setup.md`
- Contribution guide: `CONTRIBUTING.md`
