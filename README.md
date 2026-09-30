# Natural deduction research workspace

This repository is a shared working environment, a durable context store for agents, and a
knowledge base for people. It is initialized from the
[Apollo Research engineering sample](https://github.com/ApolloResearch/sample).

## Start here

- **People:** read the [project state](docs/STATE.md), then the [knowledge index](docs/README.md).
- **Agents:** read [AGENTS.md](AGENTS.md), then [docs/STATE.md](docs/STATE.md), and follow links
  relevant to the task. Save useful context back into the repository.
- **Finding people's work:** [where things are](docs/WHERE_THINGS_ARE.md) lists each person's
  weights and datasets on Hugging Face, best models, branches, and write-ups.
- **Papers:** use the [literature workflow](docs/literature/README.md) and the generated
  [project bibliography](references.bib). Open this folder as an Obsidian vault to browse notes
  and insert citations.
- **Project direction:** the original [repo design principles](planning/repo-design-principles.md)
  and [group culture](group_docs/culture.md) describe the intended working style.

## Repository map

| Path | Purpose |
| --- | --- |
| `docs/` | Curated findings, current state, and navigation |
| `experiment-summaries/` | Dated research records, including speculative and negative results |
| `planning/` | Project plans and open research directions |
| `group_docs/` | Existing onboarding, culture, and research advice |
| `code/` | Shared Python code, runnable experiments, dependencies, tests, and tooling |
| `scratch/` | Ignored, disposable local exploration; create when needed |
| `artifacts/` | Ignored, generated data and model outputs; create when needed |

The Python library is `code` (for example, `from code.models import MLP`); `nd-rl` is the
repository and distribution name. Python project files and the virtual environment live inside
`code/`; see the [code guide](code/README.md).

The knowledge store uses linked Markdown files in Git. No database service is required.
Read [the research workflow](docs/workflow.md) for record conventions and how to publish a finding.
The original files in `planning/` and `group_docs/` were preserved without edits.

## Python setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from the repo root:

```bash
make -C code install-dev
make -C code check
make -C code test-all
```

Or enter `code/` and use the same Make targets without `-C code`.
[The code guide](code/README.md) covers installation, development, and the MNIST example.
GitHub Actions runs these checks once the repository is hosted on GitHub.

See [template provenance](docs/apollo-template.md) for the source revision and adaptations,
and [ACCESS.md](ACCESS.md) for the workspace's access-status record.
