# Working in this research workspace

## Find context

1. Read `README.md`.
2. `docs/STATE.md` and `experiment-summaries/README.md` are indexes of about 150 KB each. Do not read them whole.
   Search them (`grep`, or the last 30 lines for recent work) for the topic at hand, then follow the links.
3. `docs/README.md` maps the rest of `docs/`. Read the relevant plans, evidence and code before editing. Follow links
   selectively instead of loading the whole repository.
4. Agent runs on pods follow `docs/proposals/orchestration/AGENT_POLICY.md` and the role briefs beside it.

## Store context

- Keep reusable code in `code/` and meaningful regression tests in `code/tests/`.
- An experiment that has a hypothesis to test (roughly, more than 5 hours of human-equivalent effort) leaves two
  things. One-off literature reviews and quick tests do not need them.
  - **A working folder** with its code, pre-registration, logs and write-ups. For agent runs this is the run's branch
    `dan_<run-id>` on the fork `dan-pandori/nd-takehome`, with large files in the bucket
    `hf://buckets/dan-pandori/nd-rl/<run-id>/`. For work done in this repository it is a folder under
    `code/experiments/current/`.
  - **A summary folder** `experiment-summaries/<date>-<slug>/`, following `experiment-summaries/TEMPLATE.md`: a clear,
    human-readable summary, the key figures, and a link to the working folder at a named commit. Add its line to
    `experiment-summaries/README.md`.

## Work safely and quickly

### Research note dates and history

- New research notes start with YAML frontmatter containing `written_on: YYYY-MM-DD` and `written_by:`
  (`agent:claude`, `agent:codex` or `human:Name`). Preserve the original date on later edits; do not infer unknown
  dates for existing notes.
- Use `docs/project_strategy/NOTE_TEMPLATE.md`. The full committed update history belongs in Git, not in an
  update-history table or a manually maintained last-updated field inside each note.
- Before relying on a note's history, inspect `git log --follow -p -- path/to/note.md`.
  Check `git diff -- path/to/note.md` and `git diff --cached -- path/to/note.md` for uncommitted edits.
- When committing note changes, explain the substantive change and add an `Edited-by:` trailer
  identifying the actual writer, such as `agent:claude`, `agent:codex`, or `human:Name`.
  Do not infer human/agent authorship from the Git account or claim a human authored an agent draft.
  Role sessions started by `runrole` also add `Agent-role:` and `Run-id:`.

### Development

- **Preserve the user's drafts and unrelated changes.** Uncommitted edits are someone's work and exist nowhere else.
  - Run `git status` on a folder before running anything in it that writes files.
  - If a file there has uncommitted changes, copy it aside before running a script that could touch it.
  - Read a script before running it, even to see its options: not every script handles `--help`.
  - A script that regenerates a document must not write without an explicit flag, and backs up what it replaces.
- Prefer links over duplicating guidance.
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
- Cite papers as `[@citekey]`. Research-note filenames and titles are free-form; associate a note
  with one or more papers using a `papers:` YAML list of citekeys. Optional learning notes may live
  in `docs/literature/`; link notes with ordinary relative Markdown links and follow the
  research-note date convention.
- Adding a paper is not evidence of having read or validated it. Keep claims and interpretations
  distinct, and preserve citation keys used by existing notes.
