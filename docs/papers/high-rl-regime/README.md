---
written_on: 2026-10-28
written_by: agent:claude
---

# High-RL regime paper: skeleton

The group paper from the latest group-meeting notes (pasted by Leon on 2026-10-28): how to study high-RL regimes with a from-scratch toy prover. This
directory has the one-page introduction only, as a skeleton for Leon to write by hand. The rest of the paper is
commented out in `main.tex`.

Branch `leon/high-rl-paper`, started from `leon/main` at `d8bbd496`. Nothing here has been shared or submitted.

| file | what |
|---|---|
| `main.tex` | NeurIPS 2025 style (`[preprint]`), biblatex on the repo's `references.bib`, the writing-note macros |
| `sections/01_intro.tex` | the introduction: four paragraphs, a teaser-figure placeholder, a "talk about" box per paragraph with its points and sources |
| `neurips_2025.sty` | the NeurIPS 2025 style, as downloaded from neurips.cc; 2026's was not posted at the usual URL on 2026-10-28 |
| `build.sh` | builds `main.pdf` with tectonic |
| `main.pdf` | the built skeleton, notes on |

## Build

```bash
bash docs/papers/high-rl-regime/build.sh
```

`build.sh` uses tectonic 0.17.0 at `/scratch/cowifornia/tools/tectonic/tectonic` on serrano. Set `TECTONIC=` to use
another copy. The first build downloads its TeX bundle into `/scratch/cowifornia/tools/tectonic-cache`. The script copies
the repo-root `references.bib` next to `main.tex`; that copy is gitignored, because Zotero generates the root file
(see `docs/literature/README.md` for adding papers).

## Writing notes

- `\talk{...}` prints an orange "Talk about" box, `\todo{...}` a red marker, and `\figplaceholder{...}` a grey
  figure box.
- `\notesfalse` in `main.tex` hides all of them, which shows the page as a reader would see it.
- The compute numbers in the boxes come from Dan's rl-compute-share review: branch `origin/dan`, file
  `docs/literature/2026-10-19-rl-compute-share/REVIEW.md`. It is not on `leon/main`. That review covers the EI runs
  only; the same ratio for the GRPO runs has not been computed.
- Papers named in the boxes but missing from `references.bib`: DeepSeek-V3.2, the Grok 4 post, the Epoch estimates,
  MacDiarmid et al. 2025, Baker et al. 2025, Interplay (arXiv 2512.07783), Echo Chamber (arXiv 2504.07912), and the
  persona selection model (PSM) source. Add them through Zotero before citing them.
