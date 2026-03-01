# Contributing

## Development Setup

Preferred workflow:

```bash
uv sync --extra dev
```

Alternative pip workflow:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Local Quality Gates

Run the same core checks used in CI:

```bash
uv run make lint
uv run make test
uv run make typecheck
```

Or run all three:

```bash
uv run make ci
```

## Testing Guidelines

- Add unit tests under `tests/` for any behavior change.
- Prefer deterministic tests with local fixtures from `tests/data/`.
- Avoid tests that require network access or external hosted model APIs.

## Safety and Security Guidelines

- Keep adversarial simulation usage lab-only.
- Never commit secrets (`.env`, private keys, tokens).
- Use defensive wording and avoid adding operationally harmful content.

## Generated Artifacts Policy

Generated runtime outputs should not be committed:

- `results.json`
- `consolidated_results.json`
- `final_report.md`
- `reports/*.json`
- `reports/*.html`
- coverage and cache artifacts

If you need to share outputs, attach them to an issue/PR discussion instead of versioning them.
