# Instructions for literature readers (run capability-defs, Part 1)

The run asks: **what should "capability" mean for this project, and how do we measure it?** The project
trains small transformers from scratch on propositional natural-deduction proofs in Lean 4 syntax, then
runs RL (expert iteration, GRPO) against a Lean verifier. Its question: does RL against a verifier
**create** a capability, or **elicit** one the pretrained model already had? Dan's objection to pass@k:
"There is some k at which even randomly initialized weights will solve any proof in our dataset. Is there
a more principled way to say that a capability has emerged than just picking a k?" He also suggested the
teacher-forced log-likelihood of reference proofs.

Your job: for your thread, find how the field (ML, RL, and the fields they borrow from) **defines and
measures** a capability / ability / skill / competence, and above all **how it tells a new capability apart
from better access to an old one**. We need definitions that can be made **quantitative** on our models.

## Rules (hard)

1. **Screen only papers new to the two earlier reviews.** The list is
   `capability_defs/lit/_prior_screened.tsv` (columns: review, arXiv id, title). Check it (grep the id and a
   title word) before screening a paper. A prior paper may be cited for context but does not count.
2. **Verify every claim you write down against the source text.** Fetch the text with
   `python3 capability_defs/lit/fetch.py <arXiv id>` (or `--url <pdf/html url> --name <slug>` for non-arXiv
   sources), then check the exact words with `python3 capability_defs/lit/quote.py <id-or-slug> "<phrase>"`.
   It prints the page / section / line. Record the location next to the claim (section, equation, table,
   figure or page). A claim you could not find in the text is marked **UNVERIFIED** (say why: no text
   access, paraphrase only, etc.). Never invent a quote, a number or a location.
3. **Downloads one at a time, never in bulk** (`fetch.py` takes a lock; do not run several fetches in
   parallel, do not script loops over many ids). The VPS has 2 vCPUs and ~3 GB free RAM, shared with other
   work; at most **one** process of yours at a time. Full texts stay in `~/cd_sources/` (outside the repo;
   copyright). Do not commit them.
4. Write only these files (all under `/home/dan/work/capability-defs/capability_defs/lit/`):
   - `notes/<slug>.md` — one per **in-depth** paper, in the TEMPLATE format below.
   - `_screen_<YOUR-TAG>.md` — your screened-paper table (format below), one row per screened paper.
   - `_claims_<YOUR-TAG>.md` — your claim ledger: `| # | paper id | claim (short verbatim quote or exact number) | location | status V/UNVERIFIED |`.
   Do not edit any other file, do not commit, do not touch `references.bib`, do not run
   `code/tools/literature.py`, do not contact anyone.
5. Use WebSearch (load it with ToolSearch `select:WebSearch` if needed) to find papers and arXiv ids; use
   `fetch.py` to get text (prefer arXiv). WebFetch is acceptable for a non-arXiv page if `fetch.py --url`
   fails; quote-check what you can.

## What "in depth" means

Read the abstract, introduction, the definitions / method section, the main results and the limitations
(more if needed). In the note, separate **what the paper claims** from **our interpretation**. For every
in-depth paper the note must answer, in its "Connections and questions" section:

- **Definition offered:** what quantity the paper treats as a capability (or ability / skill / knowledge /
  competence / coverage / support / usable information …), computed from what, under what protocol.
- **New vs better access:** does the paper distinguish creating a capability from improving access to it?
  How exactly (decision rule, threshold, null model, budget)? If not, how could its quantity be turned into
  such a rule?
- **Null / floor:** how it handles the "any k solves it eventually" (random-weights) objection, if at all.
- **Transfer to our setting:** how the quantity would be computed on our checkpoints (a base model "pend",
  RL models r8 / r16; per-theorem sample counts at k = 256; teacher-forced per-step log p of proofs; a
  random-init checkpoint exists), its cost, and its main failure mode here.

## TEMPLATE for notes/<slug>.md

```
---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - <citekey like yue2025doesrl>
---

# <short title>

Paper: [@<citekey>]
Source: <arXiv id with version / URL; which version was read>

## Learnings

<what the paper claims — with verified quotes and locations>

## Evidence and limitations

<sections / equations / figures; caveats; what was not checked>

## Connections and questions

<the four bullets above, plus links to related notes in this folder>
```

## Screened-table row format (`_screen_<TAG>.md`)

`| title (first author) | year | id (arXiv id+version, DOI or URL) | thread | definition / measure offered (≤ 20 words) | relevance 0–3 | depth (abstract / sections read / full) | claims verified (n V / n total) |`

## Your final reply to the executor

≤ 600 words: for each in-depth paper, one line "definition offered → how it separates new vs elicited →
usable here? (cost)", then the 3–5 most useful ideas for a capability definition in our project, each with
its verified source location, and any paper you think the executor must read personally.
