# Research code

This directory is both the `code` Python package and the Python project root. The distribution
is named `nd-rl`. Documentation, proposals, and research records live in the parent repository.

## Layout

| Path | Purpose |
| --- | --- |
| `pyproject.toml`, `uv.lock`, `.python-version` | Package configuration, locked dependencies, Python version |
| `shared_code/` | Reusable implementation shared across experiments |
| `experiments/current/`, `experiments/archived/` | Runnable experiment-specific code |
| `ndbench/` | Capability benchmarks and interp measurements; [`ndbench/README.md`](ndbench/README.md) says which need a GPU |
| `models/`, `log.py`, `utils.py` | Shared library code |
| `tests/`, `conftest.py` | Tests and pytest configuration |
| `Makefile`, `.pre-commit-config.yaml` | Development commands and Git hooks |
| `.vscode/` | Optional editor settings for opening this directory in VS Code |
| `.venv/`, `logs/` | Ignored local environment and generated logs |

## Install and check

From the repository root:

```bash
make -C code install-dev
make -C code check
make -C code test-all
```

Or run `cd code` and then `make install-dev`, `make check`, and `make test-all`.
Installation uses Python 3.11 and the lockfile, creates `code/.venv`, and installs the commit
and push hooks. Use `make install` for runtime dependencies only.

Commands from inside this directory:

```bash
make test       # Fast tests; skips slow tests
make type       # Pyright
make format     # Apply Ruff fixes and formatting
make check      # Read-only lint, formatting, and type checks
```

After intentionally changing dependencies, run `uv lock` and `uv sync --extra dev` here, and
commit `pyproject.toml` and `uv.lock` together.

The Make targets run Python from the repository root so `code` resolves to this package rather
than Python's standard-library module of the same name. For custom Python commands, use this
pattern from the repository root:

```bash
uv run --project code --no-sync python -c 'from code.models import MLP; print(MLP)'
```

To run hooks manually from the repository root, use
`uv run --project code --no-sync pre-commit run --config code/.pre-commit-config.yaml --all-files`.
GitHub workflow definitions remain in the required repository-root `.github/workflows/` directory.

## Large-file guard

The standard [check-added-large-files hook](https://github.com/pre-commit/pre-commit-hooks#check-added-large-files)
runs at commit and push time across the repository. It uses `--maxkb=1000 --enforce-all`, so
both added and modified files selected by pre-commit are checked. The upstream hook measures
in 1,024-byte units: the limit is **1,024,000 bytes**, approximately 1 MB; exactly that size is allowed.
Files managed by Git LFS are exempt according to the upstream hook's behavior.

Install both hooks with `make -C code install-dev`. To intentionally bypass this check for one
command, use the standard hook ID:

```bash
SKIP=check-added-large-files git commit -m "Add intentional large artifact"
SKIP=check-added-large-files git push
```

Other checks still run. This is a local check of selected files, not a scan of every historical
blob in the outgoing commits. Removing a large file in a later commit does not remove it from
Git history, and the standard hook may not catch that earlier version. Each clone needs hook
installation. No custom checker script is maintained in this repository.

## Apollo example

The inherited MLP, configuration, logging, checkpoint helper, and tests demonstrate Apollo's
engineering pattern. They are example code, not natural-deduction research results.

From this directory:

```bash
make example
```

This downloads MNIST and trains a small model. W&B is disabled in
`experiments/archived/mnist/mnist.yaml`; replace `wandb: null` with your own `project` and `entity`
mapping to enable it. Data, checkpoints, and logs are ignored by Git. Save durable artifact
locations and reproduction commands in [experiment summaries](../experiment-summaries/README.md).

See [template provenance](../docs/apollo-template.md) and the [workspace guide](../README.md).
