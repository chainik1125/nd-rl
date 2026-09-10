# Working in this research workspace


<!-- My adds -->

## Introduction


<!-- End of my adds -->

## Find context

1. Read `README.md` and `docs/STATE.md`.
2. Use `docs/README.md` and `experiment-summaries/README.md` to locate task-relevant context.
3. Read the relevant plans, evidence, and code before editing. Follow links selectively instead
   of loading the whole repository. Group guidance lives in `group_docs/`.

## Store context

- Keep reusable code in `code/` and meaningful regression tests in `code/tests/`.
- Use `experiment-summaries/YYYY-MM-DD-short-slug/README.md` for each research effort. Copy
  `experiment-summaries/TEMPLATE.md` and include the question, status, commands, configuration, seed,
  code revision, observations, limitations, and artifact locations.
- Keep tentative claims in experiment records. Promote supported conclusions to `docs/findings/`
  using its template and link the supporting evidence. Preserve negative and inconclusive results.
- Add an index link when adding a durable record. Update `docs/STATE.md` when the project state
  changes, including the next concrete step. Do not treat a template or planned run as a result.
- Put one-off work in ignored `scratch/`, and large generated outputs in ignored `artifacts/`
  or external storage. Important findings must not exist only in ignored files or chat history.
- Record secrets by variable name only. Use stable artifact locations and hashes where useful.

## Work safely and quickly

### Research note dates and history

- New research notes start with YAML frontmatter containing `written_on: YYYY-MM-DD`.
  Preserve that original date on subsequent edits; do not infer unknown dates for existing notes.
- Use `docs/NOTE_TEMPLATE.md`. The full committed update history belongs in Git, not in an
  update-history table or a manually maintained last-updated field inside each note.
- Before relying on a note's history, inspect `git log --follow -p -- path/to/note.md`.
  Check `git diff -- path/to/note.md` and `git diff --cached -- path/to/note.md` for uncommitted edits.
- When committing note changes, explain the substantive change and add an `Edited-by:` trailer
  identifying the actual writer, such as `agent:claude`, `agent:codex`, or `human:Name`.
  Do not infer human/agent authorship from the Git account or claim a human authored an agent draft.
- See `docs/workflow.md#research-note-history` for attribution and history conventions.

### Development

- Preserve the user's drafts and unrelated changes. Prefer links over duplicating guidance.
- Use separate branches/worktrees for independent work when useful. Record the relevant branch
  and commit in research notes; not every exploratory branch needs to be merged.
- Python configuration, lockfile, environment, tests, and tooling live in `code/`.
- From the repository root, use `make -C code check` and relevant tests after
  `make -C code install-dev`. `make -C code test-all` includes slow tests. Use
  `make -C code format` for intentional formatting. See `code/README.md` for custom commands.
- Documentation-only changes need link and factual checks, not a model-training run.
- Distinguish observed results, interpretations, and open hypotheses. Report which checks ran
  and any unresolved failures. Never invent experiment results or promote sample code as evidence.

### Literature

- Read `docs/literature/README.md` before adding papers or citations.
- `references.bib` is generated from Zotero; do not edit it manually. Use
  `uv run --script code/tools/literature.py add <arxiv-id, DOI, or .bib path>`.
- Cite papers as `[@citekey]`. Learning notes in `docs/literature/<citekey>.md` are optional;
  link them with ordinary relative Markdown links and follow the research-note date convention.
- Adding a paper is not evidence of having read or validated it. Keep claims and interpretations
  distinct, and preserve citation keys used by existing notes.
