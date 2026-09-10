# Invoke with make -C code from the repository root, or make from this directory.
# Run Python from the repository root so code resolves to the local package.
RUN = cd .. && uv run --project code --no-sync

.PHONY: install install-dev type format check test test-all example

install:
	uv sync --locked

install-dev:
	uv sync --locked --extra dev
	$(RUN) pre-commit install --config code/.pre-commit-config.yaml --install-hooks

type:
	$(RUN) pyright --project code

format:
	$(RUN) ruff check --config code/pyproject.toml --cache-dir code/.ruff_cache --fix code
	$(RUN) ruff format --config code/pyproject.toml --cache-dir code/.ruff_cache code

check:
	$(RUN) ruff check --config code/pyproject.toml --cache-dir code/.ruff_cache code
	$(RUN) ruff format --config code/pyproject.toml --cache-dir code/.ruff_cache --check code
	$(RUN) pyright --project code

test:
	$(RUN) python -m pytest -c code/pyproject.toml code/tests

test-all:
	$(RUN) python -m pytest -c code/pyproject.toml code/tests --runslow

example:
	$(RUN) python -m code.experiments.archived.mnist.run_train_mnist code/experiments/archived/mnist/mnist.yaml
