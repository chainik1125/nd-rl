# Literature



## For humans

This folder is meant to store information about literature related to this project.

1. Relevant papers should be added to references.bib - (we can then just import this into the paper, and you can use it for any .tex notes your write).

2. The repo is set up to use a Zotero library (natural-deduction-rl) as the primary reference. Add the paper in Zotero/Zotero extension and it will appear here.

3. I also sometime find it helpful to add papers manually. You can do this by running:

```
uv run --script code/tools/literature.py add arxiv:2501.12948
```

 or similar. Note that this still routes via Zotero so you will need to have it open. If this gets annoying we can add a fully manual add.

4. It may be helpful to store research notes about key papers. You can do this by adding a .md file formatted similarly to /literature/TEMPLATE.md. The metadata at the top is linked to Obsidian, which will allow both humans and Claude's to more effectively find which papers are relevant for what.




## For Claudes
[The project bibliography](../../references.bib) contains papers we can cite. A paper does **not**
need a learning note. Optional notes may use any descriptive filename and title; use
[the template](TEMPLATE.md). Associate notes with papers through citekeys in YAML metadata, not
through filenames.

Zotero is the source of truth. Better BibTeX automatically exports the `natural-deduction-rl`
collection to `references.bib` in BibLaTeX format (the standard `.bib` format read by Pandoc and
Obsidian Citations). **Do not edit that generated file by hand.** Edit metadata in Zotero, or add
new papers through the command below. Commit bibliography changes with the relevant notes.

From the repository root, with Zotero running:

```bash
uv run --script code/tools/literature.py status
uv run --script code/tools/literature.py add arxiv:2601.04728
uv run --script code/tools/literature.py add '10.48550/arXiv.2601.04728'
uv run --script code/tools/literature.py add scratch/paper.bib
```

The command accepts one paper at a time and checks the library for matches by title, DOI and
arXiv identifier when available. Existing project papers are skipped. If a matching paper is
already elsewhere in the same library, add the existing item to this collection in Zotero;
the command refuses to create another copy. This is a conservative check, not a general
bibliographic deduplication system: materially different metadata can still need manual merging.

Zotero's local import API uses the selected collection. Leave `natural-deduction-rl` selected
when adding from the terminal, and avoid switching collections during import. The command checks
the destination before importing and verifies collection membership afterward. It never retries
an import automatically. PDFs and attachments remain in Zotero, outside Git.

## Associate notes with papers

A research note anywhere in the repository may declare the papers it is substantially about:

```yaml
---
written_on: 2026-09-13
written_by: "human:Dmitry"
papers:
  - guoDeepSeekR1IncentivizesReasoning2025
  - anotherCitationKey
---
```

The filename and Markdown title are free-form. Use the exact citekeys from `references.bib`, with
one list item per paper. Search a citekey in Obsidian to find associated notes and inline citations.
To find notes that declare the paper specifically in `papers:`, search from the repository root:

```bash
rg -l '^  - guoDeepSeekR1IncentivizesReasoning2025$' --glob '*.md'
```

The `papers:` property is a note-level association for discovery. An inline `[@citekey]` is a
citation supporting a particular statement. Use both when a note is about a paper and makes claims
that rely on it.

## Cite papers and link notes

Cite a paper in any Markdown note with a Pandoc citation:

```markdown
See [@donoway2026excessdescriptionlengthlearning].
```

Link a learning note with an ordinary relative Markdown link, for example from another file in
this directory: `[our learnings](why-outcome-rewards-work.md)` once that note exists. Cite the paper
for its results and link our note for our interpretation. Creating a bibliography entry does not
imply that we read or validated the paper.

Keep citation keys stable. Imported BibTeX keys are retained by Better BibTeX's default import
setting. For browser-added papers, pin the key in Zotero (Better BibTeX → Pin BibTeX key) before
using it in durable notes. Renaming a key requires updating references throughout the repo.

## Browser and Obsidian

Save papers with the [Zotero Connector](https://www.zotero.org/download/connectors), choosing the
project collection in its save popup. The same automatic export handles browser saves. Set or
inspect the Connector hotkey in your browser's extension-shortcut settings. Zotero Connector
5.0.212 was found installed in Chrome; a live browser save and its shortcut have not been tested.

Open the repository root as an Obsidian vault. The repo includes configuration for the
[Citations plugin](https://github.com/hans/obsidian-citation-plugin), pointing at `references.bib`.
On this machine the plugin distribution is installed locally; collaborators install Citations
from Obsidian's Community Plugins, then enable it for this vault. If Obsidian prompts about
community plugins when opening a new vault, review and enable Citations.

- `Cmd+Shift+C` on macOS (`Ctrl+Shift+C` elsewhere): search and insert a paper citation.
- Command palette → **Citations: Open literature note**: create/open an optional canonical note.
  For reliable reopening, this plugin command names that one note `<citekey>.md`; its generated
  `papers:` property is the durable association. Create descriptive or multi-paper notes from the
  repository template instead, with any filename and title.
- Fill in the original `written_on` date and `written_by` when creating a note; preserve the date.
- Follow ordinary Markdown links between research notes. Full history and editor attribution live
  in Git; see [the note-history convention](../workflow.md#research-note-history).

Citations inserts `[@key]`; it does not itself turn every citation into a formatted reference in
reading mode. Pandoc can render references when exporting, using `--citeproc --bibliography
references.bib`. Pandoc is not installed by this setup.

## Another checkout or collaborator

Install Zotero and [Better BibTeX](https://retorque.re/zotero-better-bibtex/installation/), open
Zotero, and register the export for this checkout:

```bash
uv run --script code/tools/literature.py setup --collection '/1/natural-deduction-rl'
```

The first path component is the local Zotero library ID or name. The command records local IDs
in ignored `code/.zotero.local.json`; no token or password is needed. The script's dependencies
and lockfile live under `code/tools/`, independently of the training environment.

The current collection belongs to Dmitry's **personal** Zotero library. Git shares exported
citations and notes, but does not grant collaborators write access to that Zotero collection.
For shared editing, move to a Zotero group library and configure one agreed export owner.
Avoid several independent libraries overwriting the same bibliography. A collaborator should
not register an empty collection to overwrite the shared file.

Implementation uses Zotero's local Connector import endpoint and Better BibTeX's
[JSON-RPC API](https://retorque.re/zotero-better-bibtex/exporting/json-rpc/); it never writes the
Zotero database directly. Local APIs can change across Zotero versions. The integration was
checked with Zotero 9.0.6 and Better BibTeX 9.0.63 on 2026-09-08.

## Reading guides

- [Open-source RL training scripts](opensource-rl-scripts.md): concrete OLMo, Nemotron, INTELLECT-3.1, and DAPO recipes, with recommendations and reproduction limitations for natural-deduction research.
- [Techniques from the literature we have not tried yet (lit-review, 2026-09-29)](2026-09-29-lit-review/REVIEW.md): a reviewed, ranked shortlist for the 3.2 M state-env / `lean_seq` setup (search as the EI expert, frontier target supply, support-expanding GRPO advantages, transductive EI on textbook theorems, a steps-to-go value, least-likely proof selection), rejections with reasons, create-vs-elicit measurements, a 58-paper screened table, [23 paper notes](2026-09-29-lit-review/notes/) and [papers to add to Zotero](2026-09-29-lit-review/to_add_to_zotero.md). [Summary and review](../../experiment-summaries/2026-09-29-lit-review-untried-techniques-search-expert-and-frontier-supply/README.md).
- [Create vs elicit since 2025, exploration outside support, and model organisms (lit-review-2, 2026-10-02)](2026-10-02-lit-review-2/REVIEW.md): a reviewed extension of lit-review (51 new papers, 11 in depth): the equal-k pass@k standard and where our `trajectory` / `rl-from-ckpt` designs fall short, which exploration methods show out-of-support evidence, model-organism domains with exact coverage control plus RL, and ranked implications. [11 paper notes](2026-10-02-lit-review-2/notes/), [claim ledger](2026-10-02-lit-review-2/claims.md), [screened table](2026-10-02-lit-review-2/screened.md), [papers to add to Zotero](2026-10-02-lit-review-2/to_add_to_zotero.md). [Summary and review](../../experiment-summaries/2026-10-02-lit-review-2-create-vs-elicit-exploration-model-organisms/README.md).
- [Learning beyond training data: recommendations for natural-deduction RL](../../astra_suggestions/learning-beyond-training-data.md): five priority readings, five additional readings, suggested experiments, and limitations of natural deduction as a research setting.
