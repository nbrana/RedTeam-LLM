PYTHON ?= python
PIP ?= pip

.PHONY: install-dev lint test typecheck ci

install-dev:
	$(PIP) install -e ".[dev]"

lint:
	ruff check .

test:
	pytest

typecheck:
	mypy

ci: lint test typecheck
