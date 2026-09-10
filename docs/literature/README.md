# Literature

[The project bibliography](../../references.bib) contains papers we can cite. A paper does **not**
need a learning note. Optional notes live here as `<citekey>.md`; use [the template](TEMPLATE.md).

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

## Cite papers and learning notes

Cite a paper in any Markdown note with a Pandoc citation:

```markdown
See [@donoway2026excessdescriptionlengthlearning].
```

Link a learning note with an ordinary relative Markdown link, for example from another file in
this directory: `[our learnings](donoway2026excessdescriptionlengthlearning.md)` once that note
exists. Cite the paper for its results and link our note for our interpretation. Creating a
bibliography entry does not imply that we read or validated the paper.

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
- Command palette → **Citations: Open literature note**: create/open an optional learning note.
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
